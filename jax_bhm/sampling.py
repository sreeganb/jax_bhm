"""
MCMC sampling for protein structures using BlackJax.

This module provides functions for sampling protein structure conformations
using Bayesian inference with BlackJax's MCMC algorithms.
"""

import jax
import jax.numpy as jnp
import blackjax
from typing import Callable, Tuple, Dict, Optional
from jax_bhm.structure import ProteinStructure


def log_probability_fn(
    positions_flat: jax.Array,
    energy_fn: Callable,
    temperature: float = 1.0
) -> float:
    """
    Convert energy function to log probability for Bayesian sampling.
    
    Uses Boltzmann distribution: P(x) ∝ exp(-E(x) / T)
    
    Args:
        positions_flat: Flattened array of positions (n_atoms * 3,)
        energy_fn: Function that takes positions and returns energy
        temperature: Temperature parameter for Boltzmann distribution
        
    Returns:
        Log probability
    """
    # Reshape to (n_atoms, 3)
    n_coords = positions_flat.shape[0]
    positions = positions_flat.reshape(n_coords // 3, 3)
    
    # Calculate energy
    energy = energy_fn(positions)
    
    # Convert to log probability using Boltzmann distribution
    log_prob = -energy / temperature
    
    return log_prob


def sample_structure(
    initial_structure: ProteinStructure,
    energy_fn: Callable[[jax.Array], float],
    n_samples: int = 1000,
    n_warmup: int = 500,
    step_size: float = 0.01,
    temperature: float = 1.0,
    algorithm: str = "nuts",
    key: Optional[jax.Array] = None
) -> Tuple[jax.Array, Dict]:
    """
    Sample protein structure conformations using MCMC.
    
    This function performs Bayesian sampling of protein structures using
    BlackJax's MCMC algorithms (NUTS or HMC).
    
    Args:
        initial_structure: Starting structure for sampling
        energy_fn: Energy function that takes positions array
        n_samples: Number of samples to generate
        n_warmup: Number of warmup/burn-in samples
        step_size: Step size for MCMC algorithm
        temperature: Temperature for Boltzmann distribution
        algorithm: MCMC algorithm to use ("nuts" or "hmc")
        key: JAX random key for reproducibility
        
    Returns:
        Tuple of (samples, info) where:
            - samples: Array of shape (n_samples, n_atoms, 3)
            - info: Dictionary with sampling statistics
            
    Example:
        >>> from jax_bhm.structure import create_linear_chain
        >>> from jax_bhm.scoring import create_energy_fn
        >>> structure = create_linear_chain(5)
        >>> restraints = jnp.array([[0, 4, 15.0, 1.0]])
        >>> energy_fn = create_energy_fn(distance_restraints=restraints)
        >>> samples, info = sample_structure(structure, energy_fn, n_samples=100)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Flatten initial positions
    initial_position = initial_structure.positions.flatten()
    
    # Create log probability function
    def logprob_fn(pos):
        return log_probability_fn(pos, energy_fn, temperature)
    
    # Initialize sampling algorithm
    if algorithm.lower() == "nuts":
        sampler = blackjax.nuts(logprob_fn, step_size=step_size)
    elif algorithm.lower() == "hmc":
        sampler = blackjax.hmc(logprob_fn, step_size=step_size, num_integration_steps=10)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'nuts' or 'hmc'.")
    
    # Initialize state
    initial_state = sampler.init(initial_position)
    
    # Warmup phase
    def warmup_step(state, key):
        new_state, info = sampler.step(key, state)
        return new_state, info
    
    warmup_keys = jax.random.split(key, n_warmup)
    final_warmup_state, warmup_info = jax.lax.scan(
        warmup_step, initial_state, warmup_keys
    )
    
    # Sampling phase
    def sampling_step(state, key):
        new_state, info = sampler.step(key, state)
        return new_state, (new_state.position, info)
    
    key, sampling_key = jax.random.split(key)
    sampling_keys = jax.random.split(sampling_key, n_samples)
    
    final_state, (samples_flat, sampling_info) = jax.lax.scan(
        sampling_step, final_warmup_state, sampling_keys
    )
    
    # Reshape samples
    n_atoms = initial_structure.n_atoms
    samples = samples_flat.reshape(n_samples, n_atoms, 3)
    
    # Compile statistics
    info = {
        'acceptance_rate': jnp.mean(sampling_info.is_accepted) if hasattr(sampling_info, 'is_accepted') else None,
        'n_samples': n_samples,
        'n_warmup': n_warmup,
        'algorithm': algorithm,
    }
    
    return samples, info


def sample_structure_gradient_based(
    initial_structure: ProteinStructure,
    energy_fn: Callable[[jax.Array], float],
    n_samples: int = 1000,
    n_warmup: int = 500,
    temperature: float = 1.0,
    key: Optional[jax.Array] = None,
    inverse_mass_matrix: Optional[jax.Array] = None
) -> Tuple[jax.Array, Dict]:
    """
    Sample using NUTS with adaptive step size and mass matrix.
    
    This is a higher-level interface that uses BlackJax's window adaptation
    for automatic tuning of step size and mass matrix.
    
    Args:
        initial_structure: Starting structure
        energy_fn: Energy function
        n_samples: Number of samples
        n_warmup: Number of warmup samples
        temperature: Temperature parameter
        key: Random key
        inverse_mass_matrix: Optional inverse mass matrix
        
    Returns:
        Tuple of (samples, info)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    initial_position = initial_structure.positions.flatten()
    
    def logprob_fn(pos):
        return log_probability_fn(pos, energy_fn, temperature)
    
    # Use window adaptation for better sampling
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logprob_fn,
        n_warmup,
        initial_step_size=0.01,
    )
    
    # Run warmup
    key, warmup_key = jax.random.split(key)
    (state, parameters), _ = warmup.run(warmup_key, initial_position)
    
    # Create kernel with adapted parameters
    kernel = blackjax.nuts(logprob_fn, **parameters).step
    
    # Sampling
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state.position, info)
        
        keys = jax.random.split(rng_key, num_samples)
        _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)
        return positions, infos
    
    key, sample_key = jax.random.split(key)
    samples_flat, sampling_info = inference_loop(sample_key, kernel, state, n_samples)
    
    # Reshape
    n_atoms = initial_structure.n_atoms
    samples = samples_flat.reshape(n_samples, n_atoms, 3)
    
    info = {
        'acceptance_rate': jnp.mean(sampling_info.is_accepted),
        'n_samples': n_samples,
        'n_warmup': n_warmup,
        'algorithm': 'nuts_adaptive',
        'step_size': parameters.get('step_size', None),
    }
    
    return samples, info


def calculate_sampling_statistics(
    samples: jax.Array,
    energy_fn: Optional[Callable] = None
) -> Dict:
    """
    Calculate statistics from sampling results.
    
    Args:
        samples: Array of shape (n_samples, n_atoms, 3)
        energy_fn: Optional energy function to evaluate energies
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean_structure': jnp.mean(samples, axis=0),
        'std_structure': jnp.std(samples, axis=0),
        'mean_std_per_atom': jnp.mean(jnp.std(samples, axis=0)),
    }
    
    if energy_fn is not None:
        energies = jax.vmap(energy_fn)(samples)
        stats['mean_energy'] = jnp.mean(energies)
        stats['std_energy'] = jnp.std(energies)
        stats['min_energy'] = jnp.min(energies)
        stats['max_energy'] = jnp.max(energies)
    
    return stats


def get_best_structure(
    samples: jax.Array,
    energy_fn: Callable[[jax.Array], float]
) -> Tuple[jax.Array, float]:
    """
    Find the sample with the lowest energy.
    
    Args:
        samples: Array of shape (n_samples, n_atoms, 3)
        energy_fn: Energy function
        
    Returns:
        Tuple of (best_structure, best_energy)
    """
    energies = jax.vmap(energy_fn)(samples)
    best_idx = jnp.argmin(energies)
    best_structure = samples[best_idx]
    best_energy = energies[best_idx]
    
    return best_structure, best_energy
