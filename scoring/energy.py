"""
Scoring functions (Energy/Log-Probability) for JAX.
Simplified implementation matching the union-of-argmin pairing strategy.

For BlackJAX samplers: returns LOG-PROBABILITY (higher = better).
"""
import jax
import jax.numpy as jnp
from typing import Dict

@jax.jit
def compute_pairwise_distances(coords: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise distance matrix for N particles (N x N)."""
    diff = coords[:, None, :] - coords[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)


@jax.jit
def compute_distances_between(coords_a: jnp.ndarray, coords_b: jnp.ndarray) -> jnp.ndarray:
    """Compute distance matrix between two sets of coordinates (Na x Nb)."""
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)


@jax.jit
def excluded_volume_score(coords: jnp.ndarray,
                          radii: jnp.ndarray,
                          exvol_sigma: float = 0.1) -> float:
    """
    Excluded volume penalty (returns positive value = penalty).
    
    Formula: sum((overlap / exvol_sigma)^2) for all i < j where overlap > 0.
    """
    dists = compute_pairwise_distances(coords)
    r_sum = radii[:, None] + radii[None, :]
    overlaps = jax.nn.relu(r_sum - dists)  # Only positive overlaps
    
    n = coords.shape[0]
    mask = jnp.triu(jnp.ones((n, n)), k=1)  # Upper triangle (i < j)
    return jnp.sum(((overlaps * mask) / exvol_sigma) ** 2)


@jax.jit
def log_excluded_volume_kernel(
    flat_coords: jnp.ndarray,
    radii: jnp.ndarray,
    k_stiffness: float = 10.0
) -> float:
    """
    Log probability for excluded volume (negative penalty).
    
    Returns a log-probability (higher = better, no overlaps).
    Use as part of MCMC log_prob_fn.
    
    Args:
        flat_coords: Flattened coordinates (N*3,)
        radii: Particle radii (N,)
        k_stiffness: Stiffness of overlap penalty
    
    Returns:
        Log probability (0 if no overlap, negative if overlapping)
    """
    coords = flat_coords.reshape(-1, 3)
    dists = compute_pairwise_distances(coords)
    r_sum = radii[:, None] + radii[None, :]
    overlaps = jax.nn.relu(r_sum - dists)
    
    n = coords.shape[0]
    mask = jnp.triu(jnp.ones((n, n)), k=1)
    penalty = k_stiffness * jnp.sum((overlaps * mask) ** 2)
    
    return -penalty


@jax.jit
def gaussian_nll_matrix(dists: jnp.ndarray, target_dist: float, sigma: float) -> jnp.ndarray:
    """
    Compute Gaussian negative log-likelihood matrix.
    
    NLL = 0.5 * ((d - target) / sigma)^2 + 0.5 * log(2 * pi * sigma^2)
    """
    return 0.5 * ((dists - target_dist) / sigma) ** 2 + 0.5 * jnp.log(2.0 * jnp.pi * sigma**2)


@jax.jit
def union_argmin_score(score_matrix: jnp.ndarray, same_type: bool) -> float:
    """
    Union-of-argmin pairing strategy.
    
    For each row, find the best column (argmin).
    For each column, find the best row (argmin).
    Take the union of these pairs and sum their scores.
    
    If same_type=True, only consider upper triangle (i < j) to avoid self-pairs.
    
    Uses jax.lax.cond to handle the boolean in a JIT-compatible way.
    """
    return jax.lax.cond(
        same_type,
        _union_argmin_same_type,
        _union_argmin_diff_type,
        score_matrix
    )


@jax.jit
def _union_argmin_same_type(score_matrix: jnp.ndarray) -> float:
    """Union-of-argmin for same-type pairs (upper triangle only)."""
    n_a, n_b = score_matrix.shape
    INF = 1e12
    
    # Mask lower triangle and diagonal
    tri_mask = jnp.triu(jnp.ones((n_a, n_b)), k=1)
    masked = jnp.where(tri_mask > 0, score_matrix, INF)
    
    # Row-wise: each particle finds its best partner (j > i)
    row_best_j = jnp.argmin(masked, axis=1)
    row_best_vals = jnp.min(masked, axis=1)
    row_valid = row_best_vals < INF
    
    # Column-wise: each particle finds its best partner (i < j)
    col_best_i = jnp.argmin(masked, axis=0)
    col_best_vals = jnp.min(masked, axis=0)
    col_valid = col_best_vals < INF
    
    # Build selection masks
    row_sel = jnp.zeros((n_a, n_b), dtype=bool)
    row_sel = row_sel.at[jnp.arange(n_a), row_best_j].set(row_valid)
    
    col_sel = jnp.zeros((n_a, n_b), dtype=bool)
    col_sel = col_sel.at[col_best_i, jnp.arange(n_b)].set(col_valid)
    
    # Union of both selections
    union_mask = row_sel | col_sel
    return jnp.sum(jnp.where(union_mask, score_matrix, 0.0))


@jax.jit
def _union_argmin_diff_type(score_matrix: jnp.ndarray) -> float:
    """Union-of-argmin for different-type pairs (full matrix)."""
    n_a, n_b = score_matrix.shape
    
    # Row-wise: each A particle finds its best B partner
    row_best_j = jnp.argmin(score_matrix, axis=1)
    row_valid = jnp.isfinite(jnp.min(score_matrix, axis=1))
    
    # Column-wise: each B particle finds its best A partner
    col_best_i = jnp.argmin(score_matrix, axis=0)
    col_valid = jnp.isfinite(jnp.min(score_matrix, axis=0))
    
    # Build selection masks
    row_sel = jnp.zeros((n_a, n_b), dtype=bool)
    row_sel = row_sel.at[jnp.arange(n_a), row_best_j].set(row_valid)
    
    col_sel = jnp.zeros((n_a, n_b), dtype=bool)
    col_sel = col_sel.at[col_best_i, jnp.arange(n_b)].set(col_valid)
    
    # Union of both selections
    union_mask = row_sel | col_sel
    return jnp.sum(jnp.where(union_mask, score_matrix, 0.0))


def log_probability(
    flat_coords: jnp.ndarray,
    system_template,
    flat_radii: jnp.ndarray,
    target_dists: Dict[str, float],
    nuisance_params: Dict[str, float],
    exclusion_weight: float = 1.0,
    pair_weight: float = 1.0,
    exvol_sigma: float = 0.1
) -> float:
    """
    Compute total log-probability for BlackJAX samplers.
    
    Returns: log p(coords) = -exclusion_penalty - pairwise_nll
    
    Higher values = better configurations (MCMC maximizes this).
    
    Args:
        flat_coords: Flattened coordinates (n_particles * 3,)
        system_template: ParticleSystem with unflatten() method
        flat_radii: Particle radii (n_particles,)
        target_dists: Target distances, e.g. {'AA': 48.2, 'AB': 38.5}
        nuisance_params: Sigma values, e.g. {'AA': 5.0, 'AB': 5.0}
        exclusion_weight: Weight for excluded volume term
        pair_weight: Weight for pairwise distance term
        exvol_sigma: Sigma for excluded volume penalty
    
    Returns:
        Log-probability (scalar). Higher = better.
    """
    # Reshape flat coords to (N, 3)
    coords_3d = flat_coords.reshape(-1, 3)
    
    # Get coords by type using system template
    coords_by_type = system_template.unflatten(flat_coords)
    
    # 1. Excluded volume penalty (positive value)
    ev_penalty = exclusion_weight * excluded_volume_score(coords_3d, flat_radii, exvol_sigma)
    
    # 2. Pairwise distance NLL using union-of-argmin
    pairwise_nll = 0.0
    for pair_key, target_dist in target_dists.items():
        type1, type2 = pair_key[0], pair_key[1]
        sigma = nuisance_params.get(pair_key, 1.0)
        
        coords_a = coords_by_type[type1]
        coords_b = coords_by_type[type2]
        
        # Compute distance matrix
        dists = compute_distances_between(coords_a, coords_b)
        
        # Compute Gaussian NLL matrix (with proper normalization)
        nll_matrix = gaussian_nll_matrix(dists, target_dist, sigma)
        
        # Apply union-of-argmin pairing
        same_type = (type1 == type2)
        pairwise_nll += pair_weight * union_argmin_score(nll_matrix, same_type)
    
    # Return log-probability (negative of total penalty)
    return -(ev_penalty + pairwise_nll)