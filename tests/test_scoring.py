"""Tests for scoring functions."""

import jax.numpy as jnp
import pytest
from jax_bhm.scoring import (
    distance_restraint_score,
    angle_restraint_score,
    excluded_volume_score,
    energy_function,
    create_energy_fn,
)


def test_distance_restraint_score():
    """Test distance restraint scoring."""
    # Create simple positions
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    
    # Perfect restraint (actual distance = target)
    restraints = jnp.array([[0, 1, 1.0, 1.0]])
    score = distance_restraint_score(positions, restraints, k=10.0)
    assert jnp.isclose(score, 0.0, atol=1e-6)
    
    # Violated restraint
    restraints = jnp.array([[0, 2, 1.0, 1.0]])  # actual is 2.0, target is 1.0
    score = distance_restraint_score(positions, restraints, k=10.0)
    assert score > 0.0
    expected = 1.0 * 10.0 * (2.0 - 1.0) ** 2  # weight * k * deviation^2
    assert jnp.isclose(score, expected, atol=1e-6)


def test_angle_restraint_score():
    """Test angle restraint scoring."""
    # Create positions forming a right angle
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    
    # Target angle of 90 degrees (pi/2 radians)
    restraints = jnp.array([[0, 1, 2, jnp.pi / 2, 1.0]])
    score = angle_restraint_score(positions, restraints, k=5.0)
    assert jnp.isclose(score, 0.0, atol=1e-5)
    
    # Different target angle
    restraints = jnp.array([[0, 1, 2, jnp.pi, 1.0]])  # Target 180 degrees
    score = angle_restraint_score(positions, restraints, k=5.0)
    assert score > 0.0


def test_excluded_volume_score():
    """Test excluded volume scoring."""
    # No overlap
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    score = excluded_volume_score(positions, radius=2.0, k=100.0)
    assert jnp.isclose(score, 0.0, atol=1e-6)
    
    # Overlap
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # Distance 1.0 < radius 2.0
    ])
    score = excluded_volume_score(positions, radius=2.0, k=100.0)
    assert score > 0.0


def test_energy_function():
    """Test combined energy function."""
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    
    distance_restraints = jnp.array([[0, 2, 2.0, 1.0]])
    
    # Test with distance restraints only
    energy = energy_function(
        positions,
        distance_restraints=distance_restraints,
        use_excluded_volume=False
    )
    assert jnp.isclose(energy, 0.0, atol=1e-6)
    
    # Test with excluded volume
    energy = energy_function(
        positions,
        distance_restraints=distance_restraints,
        use_excluded_volume=True
    )
    assert energy > 0.0  # Should have some excluded volume penalty


def test_create_energy_fn():
    """Test energy function factory."""
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    restraints = jnp.array([[0, 1, 1.0, 1.0]])
    energy_fn = create_energy_fn(distance_restraints=restraints)
    
    # Should be callable
    energy = energy_fn(positions)
    assert isinstance(energy, jnp.ndarray) or isinstance(energy, float)
    assert jnp.isclose(energy, 0.0, atol=1e-5) or energy >= 0.0


def test_multiple_restraints():
    """Test with multiple restraints."""
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    
    # Multiple distance restraints
    restraints = jnp.array([
        [0, 1, 1.0, 1.0],
        [1, 2, 1.0, 1.0],
        [2, 3, 1.0, 1.0],
    ])
    
    score = distance_restraint_score(positions, restraints, k=10.0)
    assert jnp.isclose(score, 0.0, atol=1e-6)
