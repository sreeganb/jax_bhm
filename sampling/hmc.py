#---------------------------------------------------
# Run Hamiltonian Monte Carlo sampling using blackjax for the 
# toy model problem. 
#---------------------------------------------------

import blackjax
import blackjax.mcmc.random_walk as random_walk
import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Callable, Tuple, Union, Optional


def create_hmc_kernel(log_prob_fn: Callable, 
                      inverse_mass_matrix: Union[float, jnp.ndarray] = 1.0, 
                      step_size: Union[float, jnp.ndarray] = 0.01,
                      num_integration_steps: int = 10):
    """
    Create a BlackJAX HMC kernel.
    
    Args:
        log_prob_fn: Log probability function (flat_coords -> scalar)
        inverse_mass_matrix: Inverse mass matrix. Can be:
            - scalar: isotropic (same for all dimensions)
            - 1D array: diagonal mass matrix (per-dimension scaling)
            - 2D array: full mass matrix (for correlated parameters)
            For multi-scale problems (e.g., coordinates + nuisance params),
            use different values to account for different parameter scales.
            Larger values = larger effective step size for that dimension.
        step_size: Step size for leapfrog integration (scalar or array).
                   Should be tuned based on gradient magnitude - typically 
                   step_size * gradient_norm ≈ 1 for stable integration.
        num_integration_steps: Number of leapfrog steps per proposal
        
    Returns:
        BlackJAX HMC kernel 
    """
    # HMC kernel definition 
    hmc_kernel = blackjax.hmc(
        log_prob_fn, 
        inverse_mass_matrix=inverse_mass_matrix, 
        step_size=step_size, 
        num_integration_steps=num_integration_steps
    )
    
    return hmc_kernel


def create_nuts_kernel(log_prob_fn: Callable,
                       inverse_mass_matrix: Union[float, jnp.ndarray] = 1.0,
                       step_size: float = 0.01):
    """
    Create a BlackJAX NUTS (No-U-Turn Sampler) kernel.
    
    NUTS automatically adapts the number of leapfrog steps, making it
    more robust than HMC for multi-scale problems.
    
    Args:
        log_prob_fn: Log probability function
        inverse_mass_matrix: Inverse mass matrix (scalar, 1D, or 2D array)
        step_size: Step size for leapfrog integration
        
    Returns:
        BlackJAX NUTS kernel
    """
    nuts_kernel = blackjax.nuts(
        log_prob_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix
    )
    return nuts_kernel


def estimate_step_size(log_prob_fn: Callable, 
                       position: jnp.ndarray,
                       target_acceptance: float = 0.65,
                       inverse_mass_matrix: Optional[Union[float, jnp.ndarray]] = None) -> float:
    """
    Estimate a good step size based on gradient magnitude, accounting for mass matrix.
    
    Rule of thumb: step_size * sqrt(inverse_mass) * gradient_norm ≈ 1
    
    Args:
        log_prob_fn: Log probability function
        position: Current position
        target_acceptance: Target acceptance rate (0.65 is optimal for HMC)
        inverse_mass_matrix: If provided, accounts for mass scaling
        
    Returns:
        Estimated step size
    """
    grad_fn = jax.grad(log_prob_fn)
    grad = grad_fn(position)
    
    # Account for mass matrix scaling
    if inverse_mass_matrix is not None:
        inv_mass = jnp.asarray(inverse_mass_matrix)
        if inv_mass.ndim == 0:
            # Scalar mass
            scaled_grad = grad * jnp.sqrt(inv_mass)
        else:
            # Per-dimension mass
            scaled_grad = grad * jnp.sqrt(inv_mass)
    else:
        scaled_grad = grad
    
    grad_norm = float(jnp.linalg.norm(scaled_grad))
    
    if grad_norm < 1e-8:
        print("Warning: Gradient is nearly zero. Using default step size 0.01")
        return 0.01
    
    # Base step size from scaled gradient norm
    base_step = 1.0 / grad_norm
    
    # Scale for target acceptance (smaller step = higher acceptance)
    # Empirically, 0.65 acceptance needs step_size ≈ 0.5 / grad_norm
    step_size = base_step * (1.0 - target_acceptance + 0.15)
    
    # Clamp to reasonable range
    step_size = float(jnp.clip(step_size, 1e-6, 1.0))
    
    print(f"Gradient norm (mass-scaled): {grad_norm:.2f}")
    print(f"Estimated step size: {step_size:.6f}")
    
    return step_size


def run_hmc_sampling(
    rng_key: jax.Array,
    log_prob_fn: Callable,
    initial_position: jax.Array,
    n_steps: int = 1000,
    step_size: Union[float, jnp.ndarray, str] = "auto",
    inverse_mass_matrix: Union[float, jnp.ndarray] = 1.0,
    num_integration_steps: int = 10,
    burnin: int = 0,
    thin: int = 1,
    save_interval: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Hamiltonian Monte Carlo sampling using BlackJAX.
    
    Args:
        rng_key: JAX random key
        log_prob_fn: Log probability function to sample from
        initial_position: Starting position (flat array)
        n_steps: Total number of MCMC steps
        step_size: Step size for leapfrog integration. Can be:
               - "auto": automatically estimate from gradient norm
               - float: same step size for all dimensions
               - jnp.ndarray: per-dimension step size
        inverse_mass_matrix: Inverse mass matrix. Can be:
               - scalar: isotropic
               - 1D array: diagonal (per-dimension scaling)
               For multi-scale problems, use different values for different
               parameter groups (e.g., coords vs nuisance params)
        num_integration_steps: Number of leapfrog steps per proposal
        burnin: Number of initial steps to discard
        thin: Save every `thin` samples after burnin
        save_interval: How often to print progress (0 = no printing)
        verbose: Whether to print progress
        
    Returns:
        positions: Array of saved positions (n_saved, n_dims)
        log_probs: Log probabilities at saved positions
        acceptance_rate: Overall acceptance rate
    """
    initial_position = jnp.asarray(initial_position)
    
    # Auto-estimate step size if requested
    if isinstance(step_size, str) and step_size == "auto":
        if verbose:
            print("Auto-estimating step size...")
        step_size = estimate_step_size(log_prob_fn, initial_position, 
                                       inverse_mass_matrix=inverse_mass_matrix)
    
    # HMC kernel creation
    kernel = create_hmc_kernel(log_prob_fn, inverse_mass_matrix=inverse_mass_matrix, 
                               step_size=step_size, num_integration_steps=num_integration_steps)
    
    # Initialize state
    state = kernel.init(initial_position)
    
    # JIT compile step function
    @jax.jit
    def step_fn(rng_key, state):
        return kernel.step(rng_key, state)
    
    # Pre-split keys
    keys = jax.random.split(rng_key, n_steps)
    
    # Storage
    positions = []
    log_probs = []
    accepts = []
    
    # Format step_size for display
    step_size_arr = jnp.asarray(step_size)
    if step_size_arr.ndim == 0:
        step_size_str = f"step_size={float(step_size)}"
    else:
        step_size_str = f"step_size=[{float(step_size_arr.min()):.3f}, {float(step_size_arr.max()):.3f}] (per-dim)"
    
    if verbose:
        print(f"Running HMC sampling: {n_steps} steps, {step_size_str}")
        print(f"  Burnin: {burnin}, Thin: {thin}")
    
    t0 = time.time()
    curr_state = state
    
    # Print interval
    print_every = max(1, n_steps // 10) if save_interval > 0 else n_steps + 1
    
    for i in range(n_steps):
        curr_state, info = step_fn(keys[i], curr_state)
        accepts.append(float(info.is_accepted))
        
        # After burnin, save every `thin` steps
        if i >= burnin and (i - burnin) % thin == 0:
            positions.append(np.array(curr_state.position))
            log_probs.append(float(curr_state.logdensity))
        
        # Progress printing
        if verbose and (i + 1) % print_every == 0:
            recent_acc = np.mean(accepts[-min(1000, len(accepts)):])
            print(f"  Step {i+1:6d}/{n_steps} | LogProb: {curr_state.logdensity:10.2f} | Accept: {recent_acc:.1%}")
    
    dt = time.time() - t0
    overall_acc = np.mean(accepts)
    
    if verbose:
        print(f"Completed in {dt:.2f}s ({n_steps/dt:.0f} steps/s)")
        print(f"Overall acceptance rate: {overall_acc:.1%}")
        print(f"Saved {len(positions)} samples")
    
    return np.array(positions), np.array(log_probs), overall_acc


def run_nuts_sampling(
    rng_key: jax.Array,
    log_prob_fn: Callable,
    initial_position: jax.Array,
    n_steps: int = 1000,
    step_size: Union[float, str] = "auto",
    inverse_mass_matrix: Union[float, jnp.ndarray] = 1.0,
    burnin: int = 0,
    thin: int = 1,
    save_interval: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run NUTS (No-U-Turn Sampler) using BlackJAX.
    
    NUTS automatically adapts the number of leapfrog steps, making it more
    robust for multi-scale problems (e.g., sampling coordinates + nuisance params).
    
    Args:
        rng_key: JAX random key
        log_prob_fn: Log probability function to sample from
        initial_position: Starting position (flat array)
        n_steps: Total number of MCMC steps
        step_size: Step size for leapfrog integration.
               - "auto": automatically estimate from gradient norm
               - float: fixed step size
        inverse_mass_matrix: Inverse mass matrix. Use different values for
               different parameter scales (e.g., coords vs nuisance params)
        burnin: Number of initial steps to discard
        thin: Save every `thin` samples after burnin
        save_interval: How often to print progress (0 = no printing)
        verbose: Whether to print progress
        
    Returns:
        positions: Array of saved positions (n_saved, n_dims)
        log_probs: Log probabilities at saved positions
        acceptance_rate: Overall acceptance rate
    """
    initial_position = jnp.asarray(initial_position)
    
    # Auto-estimate step size if requested
    if isinstance(step_size, str) and step_size == "auto":
        if verbose:
            print("Auto-estimating step size for NUTS...")
        step_size = estimate_step_size(log_prob_fn, initial_position,
                                       inverse_mass_matrix=inverse_mass_matrix)
    
    # NUTS kernel creation
    kernel = create_nuts_kernel(log_prob_fn, inverse_mass_matrix=inverse_mass_matrix,
                                step_size=step_size)
    
    # Initialize state
    state = kernel.init(initial_position)
    
    # JIT compile step function
    @jax.jit
    def step_fn(rng_key, state):
        return kernel.step(rng_key, state)
    
    # Pre-split keys
    keys = jax.random.split(rng_key, n_steps)
    
    # Storage
    positions = []
    log_probs = []
    accepts = []
    
    if verbose:
        print(f"Running NUTS sampling: {n_steps} steps, step_size={step_size:.6f}")
        print(f"  Burnin: {burnin}, Thin: {thin}")
    
    t0 = time.time()
    curr_state = state
    
    # Print interval
    print_every = max(1, n_steps // 10) if save_interval > 0 else n_steps + 1
    
    for i in range(n_steps):
        curr_state, info = step_fn(keys[i], curr_state)
        accepts.append(float(info.is_accepted))
        
        # After burnin, save every `thin` steps
        if i >= burnin and (i - burnin) % thin == 0:
            positions.append(np.array(curr_state.position))
            log_probs.append(float(curr_state.logdensity))
        
        # Progress printing (NUTS also shows tree depth info)
        if verbose and (i + 1) % print_every == 0:
            recent_acc = np.mean(accepts[-min(1000, len(accepts)):])
            print(f"  Step {i+1:6d}/{n_steps} | LogProb: {curr_state.logdensity:10.2f} | Accept: {recent_acc:.1%}")
    
    dt = time.time() - t0
    overall_acc = np.mean(accepts)
    
    if verbose:
        print(f"Completed in {dt:.2f}s ({n_steps/dt:.0f} steps/s)")
        print(f"Overall acceptance rate: {overall_acc:.1%}")
        print(f"Saved {len(positions)} samples")
    
    return np.array(positions), np.array(log_probs), overall_acc