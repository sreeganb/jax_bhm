"""
Scoring functions for protein structure evaluation.

This module provides JAX-based scoring functions for evaluating protein
structures based on various restraints and energy terms.
"""

import jax
import jax.numpy as jnp
from typing import Callable


def distance_restraint_score(
    positions: jax.Array,
    restraints: jax.Array,
    k: float = 10.0
) -> float:
    """
    Calculate distance restraint score using harmonic potential.
    
    This implements a simple harmonic restraint on distances between atoms,
    similar to distance restraints used in IMP.
    
    Args:
        positions: Array of shape (n_atoms, 3) containing atomic coordinates
        restraints: Array of shape (n_restraints, 4) where each row is
                   [atom_i, atom_j, target_distance, weight]
        k: Force constant for harmonic potential
    
    Returns:
        Total distance restraint score (lower is better)
    
    Example:
        >>> positions = jnp.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        >>> restraints = jnp.array([[0, 1, 1.0, 1.0]])  # atoms 0-1 should be 1.0 apart
        >>> score = distance_restraint_score(positions, restraints)
    """
    def restraint_energy(restraint):
        i, j, target_dist, weight = restraint[0].astype(int), restraint[1].astype(int), restraint[2], restraint[3]
        actual_dist = jnp.linalg.norm(positions[i] - positions[j])
        deviation = actual_dist - target_dist
        return weight * k * deviation ** 2
    
    return jnp.sum(jax.vmap(restraint_energy)(restraints))


def angle_restraint_score(
    positions: jax.Array,
    angle_restraints: jax.Array,
    k: float = 5.0
) -> float:
    """
    Calculate angle restraint score using harmonic potential.
    
    Evaluates restraints on angles between three atoms.
    
    Args:
        positions: Array of shape (n_atoms, 3) containing atomic coordinates
        angle_restraints: Array of shape (n_restraints, 5) where each row is
                         [atom_i, atom_j, atom_k, target_angle, weight]
                         Target angle in radians
        k: Force constant for harmonic potential
    
    Returns:
        Total angle restraint score (lower is better)
    
    Example:
        >>> positions = jnp.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
        >>> restraints = jnp.array([[0, 1, 2, jnp.pi/2, 1.0]])  # 90 degree angle
        >>> score = angle_restraint_score(positions, restraints)
    """
    def angle_energy(restraint):
        i, j, k_idx = restraint[0].astype(int), restraint[1].astype(int), restraint[2].astype(int)
        target_angle, weight = restraint[3], restraint[4]
        
        # Calculate vectors
        v1 = positions[i] - positions[j]
        v2 = positions[k_idx] - positions[j]
        
        # Calculate angle
        cos_angle = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2) + 1e-8)
        cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
        actual_angle = jnp.arccos(cos_angle)
        
        deviation = actual_angle - target_angle
        return weight * k * deviation ** 2
    
    return jnp.sum(jax.vmap(angle_energy)(angle_restraints))


def excluded_volume_score(
    positions: jax.Array,
    radius: float = 2.0,
    k: float = 100.0
) -> float:
    """
    Calculate excluded volume score to prevent atom overlap.
    
    Uses a soft-sphere repulsive potential.
    
    Args:
        positions: Array of shape (n_atoms, 3) containing atomic coordinates
        radius: Minimum allowed distance between atoms
        k: Force constant for repulsive potential
    
    Returns:
        Total excluded volume score (lower is better)
    """
    n_atoms = positions.shape[0]
    
    # Calculate pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    
    # Exclude self-interactions (diagonal)
    mask = jnp.eye(n_atoms) == 0
    
    # Repulsive potential when distance < radius
    overlap = jnp.maximum(0, radius - distances)
    energy = k * overlap ** 2
    
    # Sum over upper triangle to avoid double counting
    return jnp.sum(jnp.triu(energy * mask, k=1))


def energy_function(
    positions: jax.Array,
    distance_restraints: jax.Array = None,
    angle_restraints: jax.Array = None,
    use_excluded_volume: bool = True,
    distance_k: float = 10.0,
    angle_k: float = 5.0,
    excluded_volume_k: float = 100.0,
    excluded_volume_radius: float = 2.0
) -> float:
    """
    Combined energy function for protein structure scoring.
    
    This function combines multiple scoring terms to evaluate the overall
    quality of a protein structure, similar to IMP's scoring function.
    
    Args:
        positions: Array of shape (n_atoms, 3) containing atomic coordinates
        distance_restraints: Optional distance restraints
        angle_restraints: Optional angle restraints
        use_excluded_volume: Whether to include excluded volume term
        distance_k: Force constant for distance restraints
        angle_k: Force constant for angle restraints
        excluded_volume_k: Force constant for excluded volume
        excluded_volume_radius: Radius for excluded volume
    
    Returns:
        Total energy score (lower is better)
    """
    total_energy = 0.0
    
    if distance_restraints is not None and distance_restraints.shape[0] > 0:
        total_energy += distance_restraint_score(positions, distance_restraints, k=distance_k)
    
    if angle_restraints is not None and angle_restraints.shape[0] > 0:
        total_energy += angle_restraint_score(positions, angle_restraints, k=angle_k)
    
    if use_excluded_volume:
        total_energy += excluded_volume_score(positions, radius=excluded_volume_radius, k=excluded_volume_k)
    
    return total_energy


def create_energy_fn(
    distance_restraints: jax.Array = None,
    angle_restraints: jax.Array = None,
    **kwargs
) -> Callable[[jax.Array], float]:
    """
    Create a callable energy function with fixed restraints.
    
    This is useful for passing to sampling algorithms.
    
    Args:
        distance_restraints: Distance restraints to use
        angle_restraints: Angle restraints to use
        **kwargs: Additional arguments to pass to energy_function
    
    Returns:
        A function that takes positions and returns energy
    """
    def energy_fn(positions):
        return energy_function(
            positions,
            distance_restraints=distance_restraints,
            angle_restraints=angle_restraints,
            **kwargs
        )
    
    return energy_fn
