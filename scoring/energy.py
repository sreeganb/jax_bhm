"""
Scoring functions (Energy/Log-Probability) for JAX.
"""
import jax
import jax.numpy as jnp
from typing import Dict

@jax.jit
def compute_distance_matrix(coords1: jnp.ndarray, coords2: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise euclidean distances."""
    diff = coords1[:, None, :] - coords2[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-6)

@jax.jit
def log_excluded_volume_kernel(flat_coords: jnp.ndarray, flat_radii: jnp.ndarray, k_stiffness: float = 100.0) -> float:
    """
    Computes soft-sphere excluded volume penalty.
    """
    dists = compute_distance_matrix(flat_coords, flat_coords)
    r_sum_matrix = flat_radii[:, None] + flat_radii[None, :]
    
    overlaps = r_sum_matrix - dists
    active_overlaps = jax.nn.relu(overlaps)
    
    # Mask diagonal/lower triangle to avoid self-interaction and double counting
    N = flat_coords.shape[0]
    mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    
    final_overlaps = jnp.where(mask, active_overlaps, 0.0)
    penalty = jnp.sum(k_stiffness * final_overlaps**2)
    return -penalty

@jax.jit
def compute_pair_nll(coords_A: jnp.ndarray, coords_B: jnp.ndarray, target_dist: float, sigma: float) -> float:
    """
    Computes NLL for distance restraints between two sets of particles.
    (Sum of squared errors / 2sigma^2)
    This is a simplification: assuming we want ALL pairs to match, or centroid? 
    Usually restraints are specific. For this toy model, let's assume 
    we want the closest pair or average distance to match. 
    
    User prompt implies a global restraint like 'AA': 48.22. 
    Let's implement mean distance deviation for simplicity in this toy model.
    """
    dists = compute_distance_matrix(coords_A, coords_B)
    
    # If same set, exclude diagonal
    # For A-A, simplistic approach: take mean of all valid pairs
    # For A-B, take mean of all pairs
    # Note: This determines the "shape" of the potential. 
    
    current_dist = jnp.mean(dists)
    # If they are the same set, current_dist includes 0s (diagonal). 
    # Proper dealing with self-set distance is tricky without mask info passed in.
    # For this toy model, assuming pairs are distinct sets or handled by caller would be best.
    # But let's stick to a robust simple NLL: (mean_dist - target)^2
    
    return 0.5 * ((current_dist - target_dist) / sigma) ** 2

def log_probability(
    flat_coords: jnp.ndarray, 
    system_template,  # ParticleSystem instance for structure
    flat_radii: jnp.ndarray, 
    target_dists: Dict[str, float], 
    nuisance_params: Dict[str, float]
) -> float:
    """
    Total log probability = Excluded Volume + Restraints
    """
    # 1. Excluded Volume
    ev_score = log_excluded_volume_kernel(flat_coords, flat_radii)
    
    # 2. Restraints NLL
    coords = system_template.unflatten(flat_coords)
    nll_score = 0.0
    
    # Parse target_dists keys like 'AA', 'AB'
    for pair_key, dist_val in target_dists.items():
        type1, type2 = pair_key[0], pair_key[1]
        sigma = nuisance_params.get(pair_key, 1.0)
        
        c1 = coords[type1]
        c2 = coords[type2]
        
        # Calculate NLL
        # Handling the self-case (AA) is important to avoid 0s in mean if using diagonal
        dists = compute_distance_matrix(c1, c2)
        if type1 == type2:
            N = c1.shape[0]
            mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
            valid_dists = dists[mask]
            mean_dist = jnp.mean(valid_dists)
        else:
            mean_dist = jnp.mean(dists)
            
        nll_score += 0.5 * ((mean_dist - dist_val) / sigma) ** 2

    return ev_score - nll_score
