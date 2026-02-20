#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test CCC computation and log-likelihood functions.

This script:
1. Generates a density map for ideal coordinates and saves to MRC
2. Reads the MRC back and computes CCC with regenerated density (sanity check)
3. Perturbs coordinates with increasing magnitudes
4. Computes CCC and log-likelihood (Gaussian, Laplace, Cauchy) for each perturbation
5. Plots log(likelihood) vs RMSD using seaborn

Usage:
    python test_ccc_scoring.py
    python test_ccc_scoring.py --n_perturbations 50 --max_rmsd 100.0
"""

import numpy as np
import sys
import os
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
import mrcfile

import jax.numpy as jnp
import jax

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.compute_ccc import (
    setup_grid,
    calc_cg_density_from_dict,
    write_map,
    compare_data_jax_full,
)
from scoring.em_log_likelihood import gaussian_ll, laplace_ll, cauchy_ll, differentiable_ccc


# =============================================================================
# Configuration
# =============================================================================

def get_types_config() -> Dict[str, Dict[str, float]]:
    """Get particle type configuration."""
    return {
        'A': {'radius': 24.0, 'copy': 8, 'mass': 50000.0},
        'B': {'radius': 14.0, 'copy': 8, 'mass': 25000.0},
        'C': {'radius': 16.0, 'copy': 16, 'mass': 30000.0},
    }


# =============================================================================
# Density generation utilities
# =============================================================================

def coords_dict_to_flat(coords: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Convert coordinates dict to flat array."""
    coords_list = []
    for k in sorted(coords.keys()):
        coords_list.append(coords[k].reshape(-1))
    return jnp.concatenate(coords_list)


def flat_to_coords_dict(flat: jnp.ndarray, types_config: Dict) -> Dict[str, jnp.ndarray]:
    """Convert flat array back to coordinates dict."""
    coords = {}
    idx = 0
    for k in sorted(types_config.keys()):
        n = int(types_config[k]['copy'])
        coords[k] = flat[idx:idx + n*3].reshape(n, 3)
        idx += n * 3
    return coords


def compute_rmsd(coords1: Dict[str, jnp.ndarray], coords2: Dict[str, jnp.ndarray]) -> float:
    """Compute RMSD between two coordinate sets."""
    flat1 = coords_dict_to_flat(coords1)
    flat2 = coords_dict_to_flat(coords2)
    n_atoms = len(flat1) // 3
    diff = flat1 - flat2
    msd = jnp.sum(diff ** 2) / n_atoms
    return float(jnp.sqrt(msd))


def generate_density(
    coords: Dict[str, jnp.ndarray],
    types_config: Dict,
    bins: tuple,
    resolution: float,
) -> jnp.ndarray:
    """Generate density map from coordinates."""
    mass_dict = {k: types_config[k]['mass'] for k in types_config}
    radius_dict = {k: types_config[k]['radius'] for k in types_config}
    
    return calc_cg_density_from_dict(
        coords_dict=coords,
        mass_dict=mass_dict,
        radius_dict=radius_dict,
        bins=bins,
        resolution=resolution,
    )


def perturb_coords(
    coords: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    perturbation_scale: float,
) -> Dict[str, jnp.ndarray]:
    """Apply random Gaussian perturbation to coordinates."""
    perturbed = {}
    for k in sorted(coords.keys()):
        rng_key, subkey = jax.random.split(rng_key)
        noise = jax.random.normal(subkey, coords[k].shape) * perturbation_scale
        perturbed[k] = coords[k] + noise
    return perturbed


# =============================================================================
# Main test function
# =============================================================================

def run_ccc_test(
    resolution: float = 50.0,
    voxel_size: float = 5.0,
    box_size: float = 300.0,
    n_perturbations: int = 30,
    max_rmsd: float = 80.0,
    sigma_ccc: float = 0.1,
    output_dir: str = "output",
    random_seed: int = 42,
    verbose: bool = True,
):
    """
    Run comprehensive CCC scoring test.
    
    Args:
        resolution: Target resolution in Angstroms
        voxel_size: Voxel size in Angstroms
        box_size: Cubic box size in Angstroms
        n_perturbations: Number of perturbation levels to test
        max_rmsd: Maximum RMSD perturbation to test
        sigma_ccc: Width parameter for likelihood functions
        output_dir: Output directory for plots and data
        random_seed: Random seed for reproducibility
        verbose: Print progress information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng_key = jax.random.PRNGKey(random_seed)
    
    # Get ideal coordinates and config
    ideal_coords = get_ideal_coords()
    types_config = get_types_config()
    
    if verbose:
        print("=" * 70)
        print("CCC Scoring Function Test")
        print("=" * 70)
        print(f"Resolution:       {resolution:.1f} Å")
        print(f"Voxel size:       {voxel_size:.1f} Å")
        print(f"Box size:         {box_size:.1f} Å")
        print(f"N perturbations:  {n_perturbations}")
        print(f"Max RMSD:         {max_rmsd:.1f} Å")
        print(f"sigma_ccc:        {sigma_ccc}")
        print("")
    
    # Compute center of mass for box centering
    all_coords_list = []
    for k in sorted(ideal_coords.keys()):
        all_coords_list.append(np.array(ideal_coords[k]))
    all_coords_np = np.vstack(all_coords_list)
    com = np.mean(all_coords_np, axis=0)
    
    if verbose:
        print(f"Ideal coordinates COM: [{com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}]")
    
    # Setup grid
    bins, grid_shape = setup_grid(
        box_size=(box_size, box_size, box_size),
        voxel_size=voxel_size,
        center=(float(com[0]), float(com[1]), float(com[2])),
    )
    
    if verbose:
        print(f"Grid shape: {grid_shape}")
        print("")
    
    # =========================================================================
    # Step 1: Generate reference density from ideal coordinates
    # =========================================================================
    if verbose:
        print("Step 1: Generating reference density from ideal coordinates...")
    
    reference_density = generate_density(
        ideal_coords, types_config, bins, resolution
    )
    
    # Save to MRC
    reference_mrc_path = output_dir / "test_ideal_density.mrc"
    write_map(
        density=reference_density,
        voxel_size=voxel_size,
        output_path=str(reference_mrc_path),
        bins=bins,
    )
    
    if verbose:
        print(f"  Reference density saved to: {reference_mrc_path}")
        print(f"  Density shape: {reference_density.shape}")
        print(f"  Density range: [{float(jnp.min(reference_density)):.6f}, {float(jnp.max(reference_density)):.6f}]")
    
    # =========================================================================
    # Step 2: Read MRC back and verify CCC = 1.0
    # =========================================================================
    if verbose:
        print("\nStep 2: Reading MRC back and verifying CCC...")
    
    with mrcfile.open(str(reference_mrc_path), mode='r') as mrc:
        loaded_density = jnp.array(mrc.data, dtype=jnp.float32)
    
    # Regenerate density and compute CCC
    regenerated_density = generate_density(
        ideal_coords, types_config, bins, resolution
    )
    
    ccc_self = float(compare_data_jax_full(regenerated_density, loaded_density))
    ccc_diff = float(differentiable_ccc(regenerated_density, loaded_density))
    
    if verbose:
        print(f"  CCC (regenerated vs loaded): {ccc_self:.6f}")
        print(f"  CCC (differentiable):        {ccc_diff:.6f}")
        print(f"  Expected: ~1.0 (sanity check)")
    
    # =========================================================================
    # Step 3: Perturb coordinates and compute CCC + log-likelihoods
    # =========================================================================
    if verbose:
        print(f"\nStep 3: Testing {n_perturbations} perturbation levels...")
        print("-" * 70)
        print(f"{'Pert #':<8} {'Scale':<10} {'RMSD':<12} {'CCC':<10} {'Gauss LL':<12} {'Laplace LL':<12} {'Cauchy LL':<12}")
        print("-" * 70)
    
    # Storage for results
    results = {
        'perturbation_scale': [],
        'rmsd': [],
        'ccc': [],
        'gaussian_ll': [],
        'laplace_ll': [],
        'cauchy_ll': [],
    }
    
    # Perturbation scales (roughly corresponding to target RMSDs)
    # Scale by sqrt(3) to account for 3D
    perturbation_scales = np.linspace(0, max_rmsd / np.sqrt(3), n_perturbations)
    
    sigma_param = jnp.array(sigma_ccc)
    
    for i, scale in enumerate(perturbation_scales):
        rng_key, pert_key = jax.random.split(rng_key)
        
        # Perturb coordinates
        if scale == 0:
            perturbed_coords = ideal_coords
        else:
            perturbed_coords = perturb_coords(ideal_coords, pert_key, scale)
        
        # Compute RMSD
        rmsd = compute_rmsd(ideal_coords, perturbed_coords)
        
        # Generate perturbed density
        perturbed_density = generate_density(
            perturbed_coords, types_config, bins, resolution
        )
        
        # Compute CCC
        ccc = float(differentiable_ccc(perturbed_density, reference_density))
        
        # Compute log-likelihoods
        ccc_jax = jnp.array(ccc)
        ll_gauss = float(gaussian_ll(ccc_jax, sigma_param))
        ll_laplace = float(laplace_ll(ccc_jax, sigma_param))
        ll_cauchy = float(cauchy_ll(ccc_jax, sigma_param))
        
        # Store results
        results['perturbation_scale'].append(scale)
        results['rmsd'].append(rmsd)
        results['ccc'].append(ccc)
        results['gaussian_ll'].append(ll_gauss)
        results['laplace_ll'].append(ll_laplace)
        results['cauchy_ll'].append(ll_cauchy)
        
        if verbose:
            print(f"{i:<8} {scale:<10.2f} {rmsd:<12.2f} {ccc:<10.4f} {ll_gauss:<12.4f} {ll_laplace:<12.4f} {ll_cauchy:<12.4f}")
    
    if verbose:
        print("-" * 70)
    
    # =========================================================================
    # Step 4: Create plots
    # =========================================================================
    if verbose:
        print("\nStep 4: Creating plots...")
    
    # Set seaborn style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Convert to numpy for plotting
    rmsd_arr = np.array(results['rmsd'])
    ccc_arr = np.array(results['ccc'])
    gauss_arr = np.array(results['gaussian_ll'])
    laplace_arr = np.array(results['laplace_ll'])
    cauchy_arr = np.array(results['cauchy_ll'])
    
    # -------------------------------------------------------------------------
    # Plot 1: CCC vs RMSD
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(x=rmsd_arr, y=ccc_arr, ax=ax1, marker='o', markersize=6, linewidth=2)
    ax1.set_xlabel('RMSD from Ideal Coordinates (Å)', fontsize=14)
    ax1.set_ylabel('Cross-Correlation Coefficient (CCC)', fontsize=14)
    ax1.set_title('CCC vs RMSD Perturbation', fontsize=16)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect CCC = 1.0')
    ax1.legend()
    ax1.set_ylim(-0.1, 1.1)
    
    fig1.tight_layout()
    ccc_plot_path = output_dir / "test_ccc_vs_rmsd.png"
    fig1.savefig(ccc_plot_path, dpi=150)
    if verbose:
        print(f"  CCC plot saved to: {ccc_plot_path}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Log-Likelihood vs RMSD (all three on one plot)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    palette = sns.color_palette("husl", 3)
    
    sns.lineplot(x=rmsd_arr, y=gauss_arr, ax=ax2, marker='o', markersize=5, 
                 linewidth=2, label=f'Gaussian (σ={sigma_ccc})', color=palette[0])
    sns.lineplot(x=rmsd_arr, y=laplace_arr, ax=ax2, marker='s', markersize=5, 
                 linewidth=2, label=f'Laplace (b={sigma_ccc})', color=palette[1])
    sns.lineplot(x=rmsd_arr, y=cauchy_arr, ax=ax2, marker='^', markersize=5, 
                 linewidth=2, label=f'Cauchy (γ={sigma_ccc})', color=palette[2])
    
    ax2.set_xlabel('RMSD from Ideal Coordinates (Å)', fontsize=14)
    ax2.set_ylabel('Log-Likelihood', fontsize=14)
    ax2.set_title(f'Log-Likelihood vs RMSD (σ/b/γ = {sigma_ccc})', fontsize=16)
    ax2.legend(fontsize=12)
    
    fig2.tight_layout()
    ll_plot_path = output_dir / "test_loglikelihood_vs_rmsd.png"
    fig2.savefig(ll_plot_path, dpi=150)
    if verbose:
        print(f"  Log-likelihood plot saved to: {ll_plot_path}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Separate subplots for each likelihood
    # -------------------------------------------------------------------------
    fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Gaussian
    sns.lineplot(x=rmsd_arr, y=gauss_arr, ax=axes[0], marker='o', markersize=5, 
                 linewidth=2, color=palette[0])
    axes[0].set_xlabel('RMSD (Å)', fontsize=12)
    axes[0].set_ylabel('Log-Likelihood', fontsize=12)
    axes[0].set_title(f'Gaussian (σ={sigma_ccc})', fontsize=14)
    
    # Laplace
    sns.lineplot(x=rmsd_arr, y=laplace_arr, ax=axes[1], marker='s', markersize=5, 
                 linewidth=2, color=palette[1])
    axes[1].set_xlabel('RMSD (Å)', fontsize=12)
    axes[1].set_ylabel('Log-Likelihood', fontsize=12)
    axes[1].set_title(f'Laplace (b={sigma_ccc})', fontsize=14)
    
    # Cauchy
    sns.lineplot(x=rmsd_arr, y=cauchy_arr, ax=axes[2], marker='^', markersize=5, 
                 linewidth=2, color=palette[2])
    axes[2].set_xlabel('RMSD (Å)', fontsize=12)
    axes[2].set_ylabel('Log-Likelihood', fontsize=12)
    axes[2].set_title(f'Cauchy (γ={sigma_ccc})', fontsize=14)
    
    fig3.suptitle('Log-Likelihood vs RMSD by Distribution Type', fontsize=16, y=1.02)
    fig3.tight_layout()
    separate_plot_path = output_dir / "test_loglikelihood_separate.png"
    fig3.savefig(separate_plot_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"  Separate plots saved to: {separate_plot_path}")
    
    # -------------------------------------------------------------------------
    # Plot 4: Log-likelihood vs CCC
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    
    sns.lineplot(x=ccc_arr, y=gauss_arr, ax=ax4, marker='o', markersize=5, 
                 linewidth=2, label=f'Gaussian (σ={sigma_ccc})', color=palette[0])
    sns.lineplot(x=ccc_arr, y=laplace_arr, ax=ax4, marker='s', markersize=5, 
                 linewidth=2, label=f'Laplace (b={sigma_ccc})', color=palette[1])
    sns.lineplot(x=ccc_arr, y=cauchy_arr, ax=ax4, marker='^', markersize=5, 
                 linewidth=2, label=f'Cauchy (γ={sigma_ccc})', color=palette[2])
    
    ax4.set_xlabel('Cross-Correlation Coefficient (CCC)', fontsize=14)
    ax4.set_ylabel('Log-Likelihood', fontsize=14)
    ax4.set_title(f'Log-Likelihood vs CCC (σ/b/γ = {sigma_ccc})', fontsize=16)
    ax4.legend(fontsize=12)
    ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect CCC')
    
    fig4.tight_layout()
    ll_ccc_plot_path = output_dir / "test_loglikelihood_vs_ccc.png"
    fig4.savefig(ll_ccc_plot_path, dpi=150)
    if verbose:
        print(f"  LL vs CCC plot saved to: {ll_ccc_plot_path}")
    
    # -------------------------------------------------------------------------
    # Plot 5: Comparison of likelihood "tails" (normalized)
    # -------------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    
    # Normalize to max=0 for comparison
    gauss_norm = gauss_arr - np.max(gauss_arr)
    laplace_norm = laplace_arr - np.max(laplace_arr)
    cauchy_norm = cauchy_arr - np.max(cauchy_arr)
    
    sns.lineplot(x=rmsd_arr, y=gauss_norm, ax=ax5, marker='o', markersize=5, 
                 linewidth=2, label='Gaussian (normalized)', color=palette[0])
    sns.lineplot(x=rmsd_arr, y=laplace_norm, ax=ax5, marker='s', markersize=5, 
                 linewidth=2, label='Laplace (normalized)', color=palette[1])
    sns.lineplot(x=rmsd_arr, y=cauchy_norm, ax=ax5, marker='^', markersize=5, 
                 linewidth=2, label='Cauchy (normalized)', color=palette[2])
    
    ax5.set_xlabel('RMSD from Ideal Coordinates (Å)', fontsize=14)
    ax5.set_ylabel('Normalized Log-Likelihood (max=0)', fontsize=14)
    ax5.set_title('Tail Behavior Comparison (Normalized)', fontsize=16)
    ax5.legend(fontsize=12)
    
    fig5.tight_layout()
    norm_plot_path = output_dir / "test_loglikelihood_normalized.png"
    fig5.savefig(norm_plot_path, dpi=150)
    if verbose:
        print(f"  Normalized plot saved to: {norm_plot_path}")
    
    plt.close('all')
    
    # =========================================================================
    # Save numerical results
    # =========================================================================
    results_path = output_dir / "test_ccc_results.npz"
    np.savez(
        results_path,
        perturbation_scale=np.array(results['perturbation_scale']),
        rmsd=rmsd_arr,
        ccc=ccc_arr,
        gaussian_ll=gauss_arr,
        laplace_ll=laplace_arr,
        cauchy_ll=cauchy_arr,
        sigma_ccc=sigma_ccc,
        resolution=resolution,
        voxel_size=voxel_size,
    )
    if verbose:
        print(f"  Numerical results saved to: {results_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"  Self-CCC (should be ~1.0):      {ccc_self:.6f}")
        print(f"  CCC range:                      [{min(ccc_arr):.4f}, {max(ccc_arr):.4f}]")
        print(f"  RMSD range:                     [{min(rmsd_arr):.2f}, {max(rmsd_arr):.2f}] Å")
        print(f"  Gaussian LL range:              [{min(gauss_arr):.2f}, {max(gauss_arr):.2f}]")
        print(f"  Laplace LL range:               [{min(laplace_arr):.2f}, {max(laplace_arr):.2f}]")
        print(f"  Cauchy LL range:                [{min(cauchy_arr):.2f}, {max(cauchy_arr):.2f}]")
        print("")
        print("Key observations:")
        print("  - Gaussian: Fastest decay (quadratic penalty)")
        print("  - Laplace:  Linear decay (robust to outliers)")
        print("  - Cauchy:   Slowest decay (heavy tails, most robust)")
        print("=" * 70)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test CCC computation and log-likelihood functions."
    )
    parser.add_argument('--resolution', type=float, default=50.0,
                       help='Target resolution in Angstroms (default: 50.0)')
    parser.add_argument('--voxel_size', type=float, default=5.0,
                       help='Voxel size in Angstroms (default: 5.0)')
    parser.add_argument('--box_size', type=float, default=300.0,
                       help='Cubic box size in Angstroms (default: 300.0)')
    parser.add_argument('--n_perturbations', type=int, default=30,
                       help='Number of perturbation levels (default: 30)')
    parser.add_argument('--max_rmsd', type=float, default=80.0,
                       help='Maximum RMSD to test (default: 80.0)')
    parser.add_argument('--sigma_ccc', type=float, default=0.1,
                       help='Width parameter for likelihood functions (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for plots and data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"JAX backend: {jax.default_backend()}")
    
    run_ccc_test(
        resolution=args.resolution,
        voxel_size=args.voxel_size,
        box_size=args.box_size,
        n_perturbations=args.n_perturbations,
        max_rmsd=args.max_rmsd,
        sigma_ccc=args.sigma_ccc,
        output_dir=args.output_dir,
        random_seed=args.seed,
        verbose=not args.quiet,
    )
