"""
Rigid Body System using IMP with JAX-compatible scoring.

Creates a multi-protein system with rigid bodies and provides
JAX-differentiable distance restraints for sampling.

The state space is the rigid body transformations (quaternion + translation)
for each rigid body, which we sample using BlackJAX.
"""

import numpy as np
from functools import partial
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import IMP
import IMP.core
import IMP.atom
import IMP.algebra
import IMP.container


@dataclass
class RigidBodyConfig:
    """Configuration for a rigid body (protein subunit)."""
    name: str
    n_copies: int
    radius: float
    mass: float = 1.0


def create_rigid_body_system(
    configs: List[RigidBodyConfig],
    box_size: float = 100.0,
    seed: int = 42,
) -> Tuple[IMP.Model, List[IMP.core.RigidBody], Dict]:
    """
    Create an IMP system with rigid bodies.
    
    Each protein type becomes a rigid body with its center as the 
    reference frame origin.
    
    Args:
        configs: List of rigid body configurations
        box_size: Size of initial random placement box
        seed: Random seed for initial positions
        
    Returns:
        model: IMP Model
        rigid_bodies: List of IMP RigidBody objects
        info: Dictionary with particle info for JAX interface
    """
    np.random.seed(seed)
    
    model = IMP.Model()
    rigid_bodies = []
    all_particles = []
    rb_to_particles = {}  # Maps RB index to particle indices
    
    particle_idx = 0
    
    for config in configs:
        for copy_idx in range(config.n_copies):
            # Create the main particle for this rigid body
            name = f"{config.name}_{copy_idx}"
            p = IMP.Particle(model, name)
            
            # Random initial position
            pos = IMP.algebra.Vector3D(
                np.random.uniform(-box_size/2, box_size/2),
                np.random.uniform(-box_size/2, box_size/2),
                np.random.uniform(-box_size/2, box_size/2),
            )
            
            # Setup as XYZR
            xyzr = IMP.core.XYZR.setup_particle(p, IMP.algebra.Sphere3D(pos, config.radius))
            xyzr.set_coordinates_are_optimized(True)
            
            # Setup mass
            IMP.atom.Mass.setup_particle(p, config.mass)
            
            # Create rigid body with this particle as the only member
            # The rigid body reference frame is centered on this particle
            rb = IMP.core.RigidBody.setup_particle(
                p,
                IMP.algebra.ReferenceFrame3D(
                    IMP.algebra.Transformation3D(pos)
                )
            )
            
            rigid_bodies.append(rb)
            all_particles.append(p)
            rb_to_particles[len(rigid_bodies) - 1] = [particle_idx]
            particle_idx += 1
    
    # Store metadata
    info = {
        'n_bodies': len(rigid_bodies),
        'n_particles': len(all_particles),
        'configs': configs,
        'particles': all_particles,
        'rb_to_particles': rb_to_particles,
        'radii': jnp.array([p.get_radius() for p in 
                           [IMP.core.XYZR(p) for p in all_particles]]),
    }
    
    return model, rigid_bodies, info


def get_coordinates_from_model(model: IMP.Model, particles: List) -> jnp.ndarray:
    """Extract current XYZ coordinates from IMP model as JAX array."""
    coords = []
    for p in particles:
        xyz = IMP.core.XYZ(p)
        coords.append([xyz.get_x(), xyz.get_y(), xyz.get_z()])
    return jnp.array(coords)


def set_coordinates_to_model(model: IMP.Model, particles: List, coords: jnp.ndarray):
    """Set XYZ coordinates in IMP model from JAX array."""
    for i, p in enumerate(particles):
        xyz = IMP.core.XYZ(p)
        xyz.set_coordinates(IMP.algebra.Vector3D(
            float(coords[i, 0]),
            float(coords[i, 1]),
            float(coords[i, 2]),
        ))


# =============================================================================
# JAX-compatible scoring functions
# =============================================================================

def jax_harmonic_distance_score(
    coords: jnp.ndarray,
    pairs: jnp.ndarray,
    d0: float,
    k: float,
) -> jnp.ndarray:
    """
    JAX implementation of harmonic distance pair score.
    
    Score = 0.5 * k * (d - d0)^2 for each pair
    
    Args:
        coords: (N, 3) array of particle coordinates
        pairs: (M, 2) array of particle index pairs
        d0: Equilibrium distance
        k: Force constant
        
    Returns:
        Total score (scalar)
    """
    # Get coordinates for each pair
    xyz1 = coords[pairs[:, 0]]  # (M, 3)
    xyz2 = coords[pairs[:, 1]]  # (M, 3)
    
    # Compute distances
    diff = xyz1 - xyz2
    distances = jnp.linalg.norm(diff, axis=1)
    
    # Harmonic score
    scores = 0.5 * k * (distances - d0) ** 2
    
    return jnp.sum(scores)


def jax_upper_bound_distance_score(
    coords: jnp.ndarray,
    pairs: jnp.ndarray,
    d_max: float,
    k: float,
) -> jnp.ndarray:
    """
    JAX implementation of upper bound distance score (like crosslinks).
    
    Score = 0.5 * k * max(0, d - d_max)^2
    
    Args:
        coords: (N, 3) array of particle coordinates
        pairs: (M, 2) array of particle index pairs
        d_max: Maximum allowed distance
        k: Force constant
        
    Returns:
        Total score (scalar)
    """
    xyz1 = coords[pairs[:, 0]]
    xyz2 = coords[pairs[:, 1]]
    
    diff = xyz1 - xyz2
    distances = jnp.linalg.norm(diff, axis=1)
    
    # Upper bound: penalize only when d > d_max
    violations = jnp.maximum(0.0, distances - d_max)
    scores = 0.5 * k * violations ** 2
    
    return jnp.sum(scores)


def jax_excluded_volume_score(
    coords: jnp.ndarray,
    radii: jnp.ndarray,
    k: float = 1.0,
) -> jnp.ndarray:
    """
    JAX implementation of soft sphere excluded volume.
    
    Penalizes overlap between spheres.
    
    Args:
        coords: (N, 3) array of particle coordinates
        radii: (N,) array of particle radii
        k: Force constant
        
    Returns:
        Total penalty (scalar, negative log-probability)
    """
    n = coords.shape[0]
    
    # All pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    distances = jnp.linalg.norm(diff, axis=2)  # (N, N)
    
    # Sum of radii for each pair
    sum_radii = radii[:, None] + radii[None, :]  # (N, N)
    
    # Overlap (positive when spheres overlap)
    overlap = jnp.maximum(0.0, sum_radii - distances)
    
    # Mask diagonal (self-interaction)
    mask = 1.0 - jnp.eye(n)
    overlap = overlap * mask
    
    # Harmonic penalty for overlap
    penalty = 0.5 * k * jnp.sum(overlap ** 2)
    
    return penalty


def jax_soft_box_prior(
    coords: jnp.ndarray,
    box_min: jnp.ndarray,
    box_max: jnp.ndarray,
    steepness: float = 10.0,
) -> jnp.ndarray:
    """
    Soft bounding box prior (log probability).
    
    Returns 0 inside box, negative outside.
    
    Args:
        coords: (N, 3) array of coordinates
        box_min: (3,) minimum corner
        box_max: (3,) maximum corner
        steepness: How sharp the boundary is
        
    Returns:
        Log probability (scalar, <= 0)
    """
    # Distance outside box for each dimension
    below = jnp.maximum(0.0, box_min - coords)
    above = jnp.maximum(0.0, coords - box_max)
    
    # Total violation
    violation = jnp.sum(below ** 2 + above ** 2)
    
    return -steepness * violation


# =============================================================================
# Combined scoring for sampling
# =============================================================================

def create_scoring_functions(
    n_particles: int,
    radii: jnp.ndarray,
    distance_pairs: jnp.ndarray,
    target_distance: float = 30.0,
    distance_k: float = 0.1,
    exvol_k: float = 1.0,
    box_size: float = 150.0,
    box_steepness: float = 5.0,
):
    """
    Create log_prior_fn, log_likelihood_fn, log_prob_fn for sampling.
    
    The state is a flat array of shape (n_particles * 3,) representing
    the XYZ coordinates of all rigid body centers.
    
    Args:
        n_particles: Number of particles/rigid bodies
        radii: Array of particle radii
        distance_pairs: (M, 2) array of particle pairs for distance restraints
        target_distance: Target distance for harmonic restraints
        distance_k: Force constant for distance restraints
        exvol_k: Force constant for excluded volume
        box_size: Bounding box size (centered at origin)
        box_steepness: Soft box penalty strength
        
    Returns:
        log_prior_fn, log_likelihood_fn, log_prob_fn
    """
    box_half = box_size / 2.0
    box_min = jnp.array([-box_half, -box_half, -box_half])
    box_max = jnp.array([box_half, box_half, box_half])
    
    @jax.jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Log prior: soft box constraint."""
        coords = flat_coords.reshape(-1, 3)
        return jax_soft_box_prior(coords, box_min, box_max, box_steepness)
    
    @jax.jit
    def log_likelihood_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Log likelihood: distance restraints + excluded volume."""
        coords = flat_coords.reshape(-1, 3)
        
        # Distance restraints (negative score = log likelihood)
        dist_score = jax_harmonic_distance_score(
            coords, distance_pairs, target_distance, distance_k
        )
        
        # Excluded volume penalty
        exvol_score = jax_excluded_volume_score(coords, radii, exvol_k)
        
        # Convert to log likelihood (negative energy)
        return -(dist_score + exvol_score)
    
    @jax.jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Full log posterior."""
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)
    
    return log_prior_fn, log_likelihood_fn, log_prob_fn


def generate_distance_pairs(
    configs: List[RigidBodyConfig],
    pair_type: str = 'inter',
) -> jnp.ndarray:
    """
    Generate particle index pairs for distance restraints.
    
    Args:
        configs: List of rigid body configs
        pair_type: 'inter' for between different types, 
                   'intra' for within same type,
                   'all' for all pairs
                   
    Returns:
        (M, 2) array of particle index pairs
    """
    # Build particle type assignments
    particle_types = []
    for config in configs:
        particle_types.extend([config.name] * config.n_copies)
    
    n = len(particle_types)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if pair_type == 'all':
                pairs.append([i, j])
            elif pair_type == 'inter' and particle_types[i] != particle_types[j]:
                pairs.append([i, j])
            elif pair_type == 'intra' and particle_types[i] == particle_types[j]:
                pairs.append([i, j])
    
    return jnp.array(pairs) if pairs else jnp.zeros((0, 2), dtype=jnp.int32)


# =============================================================================
# Verification against IMP scoring
# =============================================================================

def verify_against_imp(
    model: IMP.Model,
    particles: List,
    pairs: jnp.ndarray,
    d0: float,
    k: float,
):
    """
    Verify that JAX scoring matches IMP scoring.
    
    Creates an IMP scoring function and compares to JAX.
    """
    # Create IMP scoring function
    pairscore = IMP.core.HarmonicDistancePairScore(d0, k)
    lpc = IMP.container.ListPairContainer(model)
    
    for i, j in pairs:
        lpc.add((particles[int(i)], particles[int(j)]))
    
    sf = IMP.core.RestraintsScoringFunction([
        IMP.container.PairsRestraint(pairscore, lpc)
    ])
    
    # Get coordinates
    coords = get_coordinates_from_model(model, particles)
    
    # IMP score
    imp_score = sf.evaluate(False)
    
    # JAX score
    jax_score = float(jax_harmonic_distance_score(coords, pairs, d0, k))
    
    print(f"IMP score:  {imp_score:.6f}")
    print(f"JAX score:  {jax_score:.6f}")
    print(f"Difference: {abs(imp_score - jax_score):.2e}")
    
    return abs(imp_score - jax_score) < 1e-5


if __name__ == "__main__":
    # Test the system
    configs = [
        RigidBodyConfig("A", n_copies=8, radius=24.0, mass=50000.0),
        RigidBodyConfig("B", n_copies=8, radius=14.0, mass=25000.0),
        RigidBodyConfig("C", n_copies=16, radius=16.0, mass=30000.0),
    ]
    
    model, rigid_bodies, info = create_rigid_body_system(configs, box_size=100.0)
    
    print(f"Created system with {info['n_bodies']} rigid bodies")
    print(f"Total particles: {info['n_particles']}")
    
    # Generate pairs for testing
    pairs = generate_distance_pairs(configs, pair_type='inter')
    print(f"Generated {len(pairs)} inter-type distance pairs")
    
    # Verify scoring
    print("\nVerifying JAX vs IMP scoring:")
    verify_against_imp(model, info['particles'], pairs[:10], d0=50.0, k=0.1)