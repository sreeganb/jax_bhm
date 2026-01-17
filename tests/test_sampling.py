"""Tests for MCMC sampling functions."""

import jax
import jax.numpy as jnp
import pytest
from jax_bhm.structure import create_linear_chain
from jax_bhm.scoring import create_energy_fn
from jax_bhm.sampling import (
    log_probability_fn,
    sample_structure,
    calculate_sampling_statistics,
    get_best_structure,
)


def test_log_probability_fn():
    """Test log probability function."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    restraints = jnp.array([[0, 1, 1.0, 1.0]])
    energy_fn = create_energy_fn(distance_restraints=restraints, use_excluded_volume=False)
    
    positions_flat = positions.flatten()
    log_prob = log_probability_fn(positions_flat, energy_fn, temperature=1.0)
    
    # Should return a scalar
    assert isinstance(log_prob, jnp.ndarray) or isinstance(log_prob, float)
    # Energy is 0 for perfect restraint, so log_prob should be 0
    assert jnp.isclose(log_prob, 0.0, atol=1e-5)


def test_sample_structure_basic():
    """Test basic structure sampling."""
    # Create a simple structure
    structure = create_linear_chain(5)
    
    # Simple restraints
    restraints = jnp.array([[0, 4, 10.0, 1.0]])
    energy_fn = create_energy_fn(
        distance_restraints=restraints,
        use_excluded_volume=False
    )
    
    # Run sampling with small number of samples
    key = jax.random.PRNGKey(42)
    samples, info = sample_structure(
        initial_structure=structure,
        energy_fn=energy_fn,
        n_samples=10,
        n_warmup=10,
        step_size=0.01,
        algorithm="nuts",
        key=key
    )
    
    # Check output shape
    assert samples.shape == (10, 5, 3)
    assert info['n_samples'] == 10
    assert info['n_warmup'] == 10
    assert info['algorithm'] == 'nuts'


def test_sample_structure_hmc():
    """Test sampling with HMC algorithm."""
    structure = create_linear_chain(3)
    restraints = jnp.array([[0, 2, 5.0, 1.0]])
    energy_fn = create_energy_fn(distance_restraints=restraints, use_excluded_volume=False)
    
    key = jax.random.PRNGKey(123)
    samples, info = sample_structure(
        initial_structure=structure,
        energy_fn=energy_fn,
        n_samples=5,
        n_warmup=5,
        algorithm="hmc",
        key=key
    )
    
    assert samples.shape == (5, 3, 3)
    assert info['algorithm'] == 'hmc'


def test_calculate_sampling_statistics():
    """Test sampling statistics calculation."""
    # Create some dummy samples
    samples = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]],
        [[0.0, -0.1, 0.0], [1.0, -0.1, 0.0]],
    ])
    
    stats = calculate_sampling_statistics(samples)
    
    assert 'mean_structure' in stats
    assert 'std_structure' in stats
    assert 'mean_std_per_atom' in stats
    
    assert stats['mean_structure'].shape == (2, 3)


def test_calculate_sampling_statistics_with_energy():
    """Test sampling statistics with energy function."""
    samples = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
    ])
    
    restraints = jnp.array([[0, 1, 1.0, 1.0]])
    energy_fn = create_energy_fn(distance_restraints=restraints, use_excluded_volume=False)
    
    stats = calculate_sampling_statistics(samples, energy_fn)
    
    assert 'mean_energy' in stats
    assert 'std_energy' in stats
    assert 'min_energy' in stats
    assert 'max_energy' in stats


def test_get_best_structure():
    """Test getting best structure from samples."""
    # Create samples with known energies
    samples = jnp.array([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # Distance 2.0
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Distance 1.0 (best)
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],  # Distance 3.0
    ])
    
    # Target distance 1.0
    restraints = jnp.array([[0, 1, 1.0, 1.0]])
    energy_fn = create_energy_fn(distance_restraints=restraints, use_excluded_volume=False)
    
    best_structure, best_energy = get_best_structure(samples, energy_fn)
    
    # Best should be the second sample (index 1)
    assert jnp.allclose(best_structure, samples[1])
    assert jnp.isclose(best_energy, 0.0, atol=1e-5)


def test_sample_structure_invalid_algorithm():
    """Test that invalid algorithm raises error."""
    structure = create_linear_chain(3)
    restraints = jnp.array([[0, 2, 5.0, 1.0]])
    energy_fn = create_energy_fn(distance_restraints=restraints)
    
    with pytest.raises(ValueError):
        sample_structure(
            initial_structure=structure,
            energy_fn=energy_fn,
            n_samples=5,
            n_warmup=5,
            algorithm="invalid_algorithm",
        )
