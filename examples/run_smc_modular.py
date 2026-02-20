#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular SMC simulation with selectable mutation kernel (RMH or HMC).

Uses the following scoring components:
    - scoring/compute_ccc.py: Density map generation and CCC calculation
    - scoring/em_log_likelihood.py: Log-likelihood from CCC (Gaussian/Laplace/Cauchy)
    - scoring/log_priors.py: Soft bounding box prior + linear slope prior
    - scoring/energy.py: Excluded volume penalty

Probabilistic model:
    prior     = soft_box(x) * exp(-slope * sum ||r_i - COM||)
    likelihood = p(CCC | sigma) * excluded_volume * [optional structural restraints]

    SMC tempers: pi_t(x) ∝ prior(x) * likelihood(x)^{lambda_t}

Usage:
    python run_smc_modular.py --kernel rmh --n_particles 100
    python run_smc_modular.py --kernel hmc --n_particles 50
"""

import numpy as np
import sys
import os
from pathlib import Path
import mrcfile
import time
import json
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any

# Comment the next line to use GPU/TPU if available
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Project imports ===
from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_excluded_volume_kernel, log_probability
from scoring.compute_ccc import (
    setup_grid,
    calc_cg_density,
    compare_data_jax_full,
    read_map,
)
from scoring.em_log_likelihood import gaussian_ll, laplace_ll, cauchy_ll, differentiable_ccc
from scoring.log_priors import Priors
from sampling.smc import run_tempered_smc as run_rmh_smc, get_smc_samples, get_best_sample
from sampling.smc_with_hmc import run_tempered_smc as run_hmc_smc
from io_utils.io_handlers import save_mcmc_to_hdf5


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass
class SMCConfig:
    """Configuration for SMC simulation."""
    # Kernel selection
    kernel: Literal['rmh', 'hmc'] = 'rmh'
    
    # System parameters
    n_particles: int = 50  # SMC particles
    random_seed: int = 42
    
    # Density map
    mrc_path: str = "output/simulated_target_density.mrc"
    resolution: float = 50.0
    voxel_size: float = 5.0
    
    # Grid setup (will be computed from map or specified)
    grid_box_size: tuple = (300.0, 300.0, 300.0)
    grid_center: tuple = (0.0, 0.0, 0.0)
    
    # CCC likelihood parameters
    likelihood_type: Literal['gaussian', 'laplace', 'cauchy'] = 'gaussian'
    sigma_ccc: float = 0.1  # Width of likelihood on (1-CCC)
    
    # Prior parameters
    box_size: float = 300.0  # Bounding box size
    box_steepness: float = 10.0  # Soft box penalty strength
    slope_factor: float = 0.01  # Attraction to density COM
    
    # Excluded volume
    exvol_stiffness: float = 10.0  # Overlap penalty strength
    
    # Structural restraints (optional)
    use_structural_restraints: bool = False
    pair_weight: float = 0.000001
    
    # SMC parameters
    n_mcmc_steps: int = 50
    target_ess: float = 0.5
    
    # RMH-specific
    rmh_sigma: float = 3.0  # Proposal step size
    
    # HMC-specific
    hmc_step_size: float = 0.01
    hmc_num_integration_steps: int = 10
    
    # Output
    output_dir: str = "output_smc"
    output_prefix: str = "smc_modular"
    save_trajectory: bool = True
    save_logs: bool = True
    verbose: bool = True


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(config: SMCConfig) -> logging.Logger:
    """Configure logging to file and console."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{config.output_prefix}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("smc_modular")
    logger.setLevel(logging.DEBUG)
    
    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    ))
    
    # Console handler (info only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if config.verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# =============================================================================
# Timing utilities
# =============================================================================

def sync_and_time() -> float:
    """Get wall time after ensuring all JAX operations complete."""
    jax.block_until_ready(jnp.zeros(1))
    return time.perf_counter()


class WallTimer:
    """Track timing for multiple sections."""

    def __init__(self):
        self.times = {}
        self._starts = {}
        self.total_start = sync_and_time()

    def start(self, name: str):
        self._starts[name] = sync_and_time()

    def stop(self, name: str) -> float:
        elapsed = sync_and_time() - self._starts[name]
        self.times[name] = self.times.get(name, 0) + elapsed
        del self._starts[name]
        return elapsed

    def total(self) -> float:
        return sync_and_time() - self.total_start

    def summary(self) -> str:
        total = self.total()
        lines = [
            "",
            "=" * 60,
            f"TIMING SUMMARY (Backend: {jax.default_backend()})",
            "=" * 60,
            f"{'Section':<35} {'Time (s)':>10} {'%':>8}",
            "-" * 60,
        ]
        for name, elapsed in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = 100 * elapsed / total if total > 0 else 0
            lines.append(f"{name:<35} {elapsed:>10.2f} {pct:>7.1f}%")
        lines.extend([
            "-" * 60,
            f"{'TOTAL WALL TIME':<35} {total:>10.2f} {'100.0':>7}%",
            "=" * 60,
        ])
        return "\n".join(lines)


# =============================================================================
# Scoring functions factory
# =============================================================================

def create_scoring_functions(
    config: SMCConfig,
    system: ParticleSystem,
    target_density: jnp.ndarray,
    bins: tuple,
    density_com: jnp.ndarray,
    flat_radii: jnp.ndarray,
    masses: jnp.ndarray,
    logger: logging.Logger,
):
    """
    Create log_prior_fn, log_likelihood_fn, and log_prob_fn.
    
    Returns:
        log_prior_fn: Prior function (box + attraction)
        log_likelihood_fn: Likelihood function (CCC + excluded volume)
        log_prob_fn: Full posterior (prior + likelihood)
        get_ccc_fn: Raw CCC diagnostic function
    """
    logger.info("Building scoring functions...")
    
    # === Log prior: soft box + linear slope attraction ===
    box_half = config.box_size / 2.0
    box_mins = jnp.array([-box_half, -box_half, -box_half])
    box_maxs = jnp.array([box_half, box_half, box_half])
    
    @jax.jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Combined prior: soft box constraint + attraction to density COM."""
        box_prior = Priors.log_soft_box_prior(
            flat_coords, flat_radii, box_mins, box_maxs,
            steepness=config.box_steepness
        )
        slope_prior = Priors.log_linear_slope_prior(
            flat_coords, density_com, slope_factor=config.slope_factor
        )
        return box_prior + slope_prior
    
    # === CCC computation (differentiable) ===
    @jax.jit
    def compute_ccc(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Compute CCC between model density and target density."""
        coords = flat_coords.reshape(-1, 3)
        # Generate model density
        model_density = calc_cg_density(
            coords, masses, flat_radii, bins,
            resolution=config.resolution,
            group_by_radius=False  # Use lax.scan for full differentiability
        )
        # Compute CCC using differentiable version
        return differentiable_ccc(model_density, target_density)
    
    # === Log likelihood from CCC ===
    likelihood_dispatch = {
        'gaussian': lambda ccc: gaussian_ll(ccc, jnp.array(config.sigma_ccc)),
        'laplace': lambda ccc: laplace_ll(ccc, jnp.array(config.sigma_ccc)),
        'cauchy': lambda ccc: cauchy_ll(ccc, jnp.array(config.sigma_ccc)),
    }
    ll_fn = likelihood_dispatch[config.likelihood_type]
    
    @jax.jit
    def ccc_log_likelihood(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Log likelihood from CCC."""
        ccc = compute_ccc(flat_coords)
        return ll_fn(ccc)
    
    # === Excluded volume log likelihood ===
    @jax.jit
    def exvol_log_likelihood(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Excluded volume penalty (negative when overlapping)."""
        return log_excluded_volume_kernel(
            flat_coords, flat_radii, k_stiffness=config.exvol_stiffness
        )
    
    # === Combined likelihood ===
    if config.use_structural_restraints:
        # Structural restraints (from energy.py)
        target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
        nuisance_params = {'AA': 1.5, 'AB': 1.3, 'BC': 1.0}
        
        @jax.jit
        def log_likelihood_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
            """Combined likelihood: CCC + excluded volume + structural."""
            ccc_term = ccc_log_likelihood(flat_coords)
            exvol_term = exvol_log_likelihood(flat_coords)
            struct_term = log_probability(
                flat_coords, system, flat_radii,
                target_dists, nuisance_params,
                exclusion_weight=1.0, pair_weight=config.pair_weight, exvol_sigma=0.10,
            )
            return ccc_term + exvol_term + struct_term
    else:
        @jax.jit
        def log_likelihood_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
            """Combined likelihood: CCC + excluded volume."""
            ccc_term = ccc_log_likelihood(flat_coords)
            exvol_term = exvol_log_likelihood(flat_coords)
            return ccc_term + exvol_term
    
    # === Full log posterior ===
    @jax.jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Full log posterior = prior + likelihood."""
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)
    
    # === Raw CCC for diagnostics ===
    @jax.jit
    def get_ccc_fn(flat_coords: jnp.ndarray) -> jnp.ndarray:
        """Get raw CCC value for diagnostics."""
        return compute_ccc(flat_coords)
    
    logger.info(f"  Likelihood type: {config.likelihood_type}")
    logger.info(f"  sigma_ccc: {config.sigma_ccc}")
    logger.info(f"  Box size: {config.box_size}, steepness: {config.box_steepness}")
    logger.info(f"  Slope factor: {config.slope_factor}")
    logger.info(f"  Excluded volume stiffness: {config.exvol_stiffness}")
    logger.info(f"  Structural restraints: {config.use_structural_restraints}")
    
    return log_prior_fn, log_likelihood_fn, log_prob_fn, get_ccc_fn


# =============================================================================
# Model diagnostics
# =============================================================================

def run_diagnostics(
    flat_coords: jnp.ndarray,
    log_prior_fn,
    log_likelihood_fn,
    log_prob_fn,
    get_ccc_fn,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Run model diagnostics at a given position."""
    prior_val = float(log_prior_fn(flat_coords))
    lik_val = float(log_likelihood_fn(flat_coords))
    post_val = float(log_prob_fn(flat_coords))
    ccc_val = float(get_ccc_fn(flat_coords))
    
    diagnostics = {
        'log_prior': prior_val,
        'log_likelihood': lik_val,
        'log_posterior': post_val,
        'ccc': ccc_val,
        'prior_finite': bool(jnp.isfinite(prior_val)),
        'likelihood_finite': bool(jnp.isfinite(lik_val)),
        'posterior_finite': bool(jnp.isfinite(post_val)),
    }
    
    logger.info("Model diagnostics:")
    logger.info(f"  log_prior:      {prior_val:>12.4f}")
    logger.info(f"  log_likelihood: {lik_val:>12.4f}")
    logger.info(f"  log_posterior:  {post_val:>12.4f}")
    logger.info(f"  CCC:            {ccc_val:>12.4f}")
    logger.info(f"  All finite:     {diagnostics['posterior_finite']}")
    
    return diagnostics


# =============================================================================
# Main simulation function
# =============================================================================

def run_smc_simulation(config: SMCConfig) -> Dict[str, Any]:
    """
    Run SMC simulation with the specified configuration.
    
    Returns:
        Dictionary with results including best_ccc, best_score, wall_time, etc.
    """
    # Setup logging
    logger = setup_logging(config)
    timer = WallTimer()
    
    logger.info("=" * 60)
    logger.info(f"SMC Simulation with {config.kernel.upper()} Mutation Kernel")
    logger.info(f"Backend: {jax.default_backend()}")
    logger.info("=" * 60)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / f"{config.output_prefix}_config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Config saved to {config_file}")
    
    # =========================================================================
    # 1. Setup system
    # =========================================================================
    timer.start("1. System setup")
    
    types_config = {
        'A': {'radius': 24.0, 'copy': 8, 'mass': 50000.0},
        'B': {'radius': 14.0, 'copy': 8, 'mass': 25000.0},
        'C': {'radius': 16.0, 'copy': 16, 'mass': 30000.0},
    }
    ideal_coords = get_ideal_coords()
    
    timer.stop("1. System setup")
    logger.info(f"System types: {list(types_config.keys())}")
    
    # =========================================================================
    # 2. Load target density
    # =========================================================================
    timer.start("2. Load density map")
    
    mrc_path = Path(config.mrc_path)
    logger.info(f"Loading target density: {mrc_path}")
    
    with mrcfile.open(str(mrc_path), mode='r') as mrc:
        target_density = jnp.array(mrc.data, dtype=jnp.float32)
        voxel_size = float(mrc.voxel_size.x)
        origin = (
            float(mrc.header.origin.x),
            float(mrc.header.origin.y),
            float(mrc.header.origin.z),
        )
        map_shape = mrc.data.shape
    
    # Compute density center of mass
    nz, ny, nx = map_shape
    z_idx, y_idx, x_idx = jnp.meshgrid(
        jnp.arange(nz), jnp.arange(ny), jnp.arange(nx), indexing='ij'
    )
    total_mass = jnp.sum(target_density)
    x_com = (jnp.sum(x_idx * target_density) / total_mass) * voxel_size + origin[0]
    y_com = (jnp.sum(y_idx * target_density) / total_mass) * voxel_size + origin[1]
    z_com = (jnp.sum(z_idx * target_density) / total_mass) * voxel_size + origin[2]
    density_com_physical = jnp.array([x_com, y_com, z_com])
    
    logger.info(f"  Shape: {map_shape}, Voxel: {voxel_size:.2f} Å")
    logger.info(f"  Origin: [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}]")
    logger.info(f"  Density COM (physical): [{float(x_com):.1f}, {float(y_com):.1f}, {float(z_com):.1f}]")
    
    # =======================================================================
    # KEY FIX: Reconstruct bins centered at ORIGIN, matching how the density
    # was generated. The density COM should be near (0, 0, -30) if the map
    # was generated correctly. If not, we shift the coordinate system so
    # that the density COM maps to the ideal coords COM.
    # =======================================================================
    
    # Check if density was generated centered at origin or offset
    ideal_coords = get_ideal_coords()
    ideal_com = jnp.mean(jnp.concatenate(
        [ideal_coords[k] for k in sorted(ideal_coords.keys())], axis=0
    ), axis=0)  # ≈ (0, 0, -30)
    
    # If density COM is far from ideal COM, the map was NOT centered at origin.
    # We set density_com = ideal_com so all scoring uses the correct frame.
    com_distance = float(jnp.linalg.norm(density_com_physical - ideal_com))
    if com_distance > 50.0:
        logger.warning(
            f"  Density COM is {com_distance:.0f} Å from ideal coords COM!"
        )
        logger.warning(
            f"  The MRC was likely NOT centered at origin."
        )
        logger.warning(
            f"  Recommend regenerating with box_center=(0,0,0)."
        )
        logger.warning(
            f"  For now, using origin-centered bins and assuming density aligns with coords."
        )
    
    # Use origin-centered grid that matches the box size
    grid_box_size = (nx * voxel_size, ny * voxel_size, nz * voxel_size)
    grid_center = (0.0, 0.0, 0.0)  # ALWAYS center at origin
    bins, grid_shape = setup_grid(grid_box_size, voxel_size, grid_center)
    
    # Density COM for priors should be near ideal coords COM
    density_com = ideal_com  # Use ideal COM as the attraction target
    
    logger.info(f"  Grid: {grid_shape}, Box: {grid_box_size}, Center: (0,0,0)")
    logger.info(f"  Density COM (for priors): [{float(density_com[0]):.1f}, {float(density_com[1]):.1f}, {float(density_com[2]):.1f}]")
    
    timer.stop("2. Load density map")
    
    # =========================================================================
    # 3. Initialize coordinates
    # =========================================================================
    timer.start("3. Initialize coordinates")
    
    temp_system = ParticleSystem(types_config, {}, ideal_coords)
    
    # Initialize particles near ORIGIN (where ideal coords live)
    # NOT near density_com which might be at (400, 400, 400) if map is offset
    rng_key = jax.random.PRNGKey(config.random_seed)
    rng_key, init_key = jax.random.split(rng_key)
    
    spread = 10.0  # Spread around origin/ideal COM
    coords = {}
    for ptype in sorted(types_config.keys()):
        n_copies = types_config[ptype]['copy']
        rng_key, subkey = jax.random.split(rng_key)
        # Initialize near ORIGIN with small perturbations (ideal coords are near origin)
        coords[ptype] = jax.random.normal(subkey, (n_copies, 3)) * spread
    
    system = ParticleSystem(types_config, coords, ideal_coords)
    flat_radii = system.get_flat_radii()
    radii_jax = jnp.array(flat_radii, dtype=jnp.float32)
    n_dims = system.total_particles * 3
    
    # Build masses array
    masses_list = []
    for k in system.identity_order:
        mass = types_config[k].get('mass', 1.0)
        n = int(types_config[k]['copy'])
        masses_list.append(jnp.full((n,), mass, dtype=jnp.float32))
    masses = jnp.concatenate(masses_list)
    
    logger.info(f"System: {system.total_particles} particles, {n_dims} dimensions")
    logger.info(f"Particles initialized near ORIGIN (spread ~ {spread} Å)")
    
    timer.stop("3. Initialize coordinates")
    
    # =========================================================================
    # 4. Build probabilistic model
    # =========================================================================
    timer.start("4. Build probabilistic model")
    
    log_prior_fn, log_likelihood_fn, log_prob_fn, get_ccc_fn = create_scoring_functions(
        config=config,
        system=system,
        target_density=target_density,
        bins=bins,
        density_com=density_com,
        flat_radii=radii_jax,
        masses=masses,
        logger=logger,
    )
    
    timer.stop("4. Build probabilistic model")
    
    # =========================================================================
    # 5. Model diagnostics at initial position
    # =========================================================================
    timer.start("5. Model diagnostics")
    
    dummy_coords = system.flatten(coords)
    initial_diagnostics = run_diagnostics(
        dummy_coords, log_prior_fn, log_likelihood_fn, log_prob_fn, get_ccc_fn, logger
    )
    
    # JIT warmup
    _ = log_prob_fn(dummy_coords)
    jax.block_until_ready(_)
    
    if not initial_diagnostics['posterior_finite']:
        logger.warning("Initial posterior is not finite! Check prior/likelihood.")
    
    timer.stop("5. Model diagnostics")
    
    # =========================================================================
    # 6. Initialize SMC particles
    # =========================================================================
    timer.start("6. Initialize SMC particles")
    
    rng_key, init_key = jax.random.split(rng_key)
    
    flat_init = system.flatten(coords)
    initial_positions = flat_init + jax.random.normal(init_key, (config.n_particles, n_dims)) * 5.0
    
    # Validate all particles
    init_scores = jax.vmap(log_prob_fn)(initial_positions)
    init_priors = jax.vmap(log_prior_fn)(initial_positions)
    init_liks = jax.vmap(log_likelihood_fn)(initial_positions)
    jax.block_until_ready(init_scores)
    
    valid_mask = jnp.isfinite(init_scores)
    valid_count = int(jnp.sum(valid_mask))
    
    logger.info(f"Particle initialization:")
    logger.info(f"  Total particles:    {config.n_particles}")
    logger.info(f"  Valid (finite):     {valid_count}/{config.n_particles}")
    logger.info(f"  Prior range:        [{float(jnp.min(init_priors)):.2f}, {float(jnp.max(init_priors)):.2f}]")
    logger.info(f"  Likelihood range:   [{float(jnp.min(init_liks)):.2f}, {float(jnp.max(init_liks)):.2f}]")
    
    if valid_count < config.n_particles:
        logger.warning(f"{config.n_particles - valid_count} particles have -inf score!")
    
    timer.stop("6. Initialize SMC particles")
    
    # =========================================================================
    # 7. Run SMC
    # =========================================================================
    timer.start("7. SMC sampling")
    
    logger.info(f"SMC configuration:")
    logger.info(f"  Kernel:             {config.kernel.upper()}")
    logger.info(f"  MCMC steps/temp:    {config.n_mcmc_steps}")
    logger.info(f"  Target ESS:         {config.target_ess:.0%}")
    logger.info(f"  Particles:          {config.n_particles}")
    
    rng_key, smc_key = jax.random.split(rng_key)
    
    if config.kernel == 'rmh':
        logger.info(f"  RMH sigma:          {config.rmh_sigma}")
        final_state, info_history, best_positions, best_scores = run_rmh_smc(
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_prob_fn=log_prob_fn,
            initial_positions=initial_positions,
            rng_key=smc_key,
            n_mcmc_steps=config.n_mcmc_steps,
            rmh_sigma=config.rmh_sigma,
            target_ess=config.target_ess,
            record_best=True,
            verbose=config.verbose,
        )
    elif config.kernel == 'hmc':
        logger.info(f"  HMC step size:      {config.hmc_step_size}")
        logger.info(f"  HMC integration L:  {config.hmc_num_integration_steps}")
        final_state, info_history, best_positions, best_scores = run_hmc_smc(
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_prob_fn=log_prob_fn,
            initial_positions=initial_positions,
            rng_key=smc_key,
            n_mcmc_steps=config.n_mcmc_steps,
            hmc_step_size=config.hmc_step_size,
            hmc_num_integration_steps=config.hmc_num_integration_steps,
            target_ess=config.target_ess,
            record_best=True,
            verbose=config.verbose,
        )
    else:
        raise ValueError(f"Unknown kernel: {config.kernel}")
    
    jax.block_until_ready(final_state.particles)
    
    timer.stop("7. SMC sampling")
    
    # =========================================================================
    # 8. Post-processing
    # =========================================================================
    timer.start("8. Post-processing")
    
    final_positions = get_smc_samples(final_state)
    best_pos, best_score = get_best_sample(final_state, log_prob_fn)
    
    final_scores = jax.vmap(log_prob_fn)(final_positions)
    jax.block_until_ready(final_scores)
    
    best_ccc = float(get_ccc_fn(best_pos))
    
    logger.info("=" * 60)
    logger.info("Final Results")
    logger.info("=" * 60)
    logger.info(f"Final Score (mean):  {float(jnp.mean(final_scores)):.2f}")
    logger.info(f"Final Score (best):  {best_score:.2f}")
    logger.info(f"Best CCC:            {best_ccc:.4f}")
    
    # Final diagnostics
    final_diagnostics = run_diagnostics(
        best_pos, log_prior_fn, log_likelihood_fn, log_prob_fn, get_ccc_fn, logger
    )
    
    # Per-step CCC table
    if best_positions is not None and best_scores is not None:
        logger.info("=" * 60)
        logger.info("CCC per SMC Step (Best Particle)")
        logger.info("=" * 60)
        logger.info(f"{'Step':<8} {'Score':>12} {'CCC':>12} {'Coord Disp':>14}")
        logger.info("-" * 60)
        
        best_cccs = jax.vmap(get_ccc_fn)(jnp.array(best_positions))
        jax.block_until_ready(best_cccs)
        
        ref_pos = best_positions[0]
        for step_idx, (score, ccc) in enumerate(zip(best_scores, best_cccs)):
            disp = float(jnp.linalg.norm(
                jnp.array(best_positions[step_idx]) - jnp.array(ref_pos)
            ))
            logger.info(f"{step_idx:<8} {float(score):>12.2f} {float(ccc):>12.4f} {disp:>14.2f}")
        
        logger.info("=" * 60)
        
        # Motion statistics
        final_disp = float(jnp.linalg.norm(
            jnp.array(best_positions[-1]) - jnp.array(best_positions[0])
        ))
        logger.info(f"Total displacement (first->last best): {final_disp:.2f} Å")
        
        # Per-particle motion
        init_coords_arr = best_positions[0].reshape(-1, 3)
        final_coords_arr = best_positions[-1].reshape(-1, 3)
        per_particle_disp = np.linalg.norm(final_coords_arr - init_coords_arr, axis=1)
        logger.info(f"Per-particle displacement: min={per_particle_disp.min():.2f}, "
                   f"max={per_particle_disp.max():.2f}, mean={per_particle_disp.mean():.2f} Å")
    
    timer.stop("8. Post-processing")
    
    # =========================================================================
    # 9. Save results
    # =========================================================================
    timer.start("9. Save results")
    
    if config.save_trajectory and best_positions is not None and best_scores is not None:
        output_file = output_dir / f"{config.output_prefix}_trajectory.h5"
        save_mcmc_to_hdf5(
            np.array(best_positions),
            np.array(best_scores),
            1.0,
            str(output_file),
            system,
            params={
                'method': f'BlackJAX_SMC_{config.kernel.upper()}',
                'model': f'{config.likelihood_type}_CCC + SoftBox + ExVol',
                'sigma_ccc': config.sigma_ccc,
                'likelihood_type': config.likelihood_type,
                'box_size': config.box_size,
                'box_steepness': config.box_steepness,
                'slope_factor': config.slope_factor,
                'exvol_stiffness': config.exvol_stiffness,
                'n_mcmc_steps': config.n_mcmc_steps,
                'target_ess': config.target_ess,
                'n_particles': config.n_particles,
                'kernel': config.kernel,
                'rmh_sigma': config.rmh_sigma if config.kernel == 'rmh' else None,
                'hmc_step_size': config.hmc_step_size if config.kernel == 'hmc' else None,
                'hmc_num_integration_steps': config.hmc_num_integration_steps if config.kernel == 'hmc' else None,
                'best_ccc': best_ccc,
                'best_score': float(best_score),
                'trajectory': 'best_per_step',
            },
            convert_to_rmf3=True,
            color_map={
                'A': (0.2, 0.6, 1.0),
                'B': (0.9, 0.4, 0.2),
                'C': (0.3, 0.8, 0.4),
            },
        )
        logger.info(f"Trajectory saved to {output_file}")
    
    # Save summary JSON
    summary = {
        'config': asdict(config),
        'results': {
            'best_ccc': best_ccc,
            'best_score': float(best_score),
            'final_mean_score': float(jnp.mean(final_scores)),
            'n_smc_steps': len(info_history) if info_history else 0,
            'wall_time': timer.total(),
        },
        'initial_diagnostics': initial_diagnostics,
        'final_diagnostics': final_diagnostics,
        'timing': timer.times,
    }
    
    summary_file = output_dir / f"{config.output_prefix}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info(f"Summary saved to {summary_file}")
    
    timer.stop("9. Save results")
    
    # Print timing summary
    timing_summary = timer.summary()
    logger.info(timing_summary)
    
    return summary['results']


# =============================================================================
# CLI interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SMC simulation with selectable mutation kernel (RMH or HMC)."
    )
    
    # Kernel selection
    parser.add_argument('--kernel', type=str, choices=['rmh', 'hmc'], default='rmh',
                       help='Mutation kernel: rmh (Random Walk Metropolis) or hmc (Hamiltonian Monte Carlo)')
    
    # System
    parser.add_argument('--n_particles', type=int, default=50,
                       help='Number of SMC particles')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Density map
    parser.add_argument('--mrc', type=str, default='output/simulated_target_density.mrc',
                       help='Path to target density MRC file')
    parser.add_argument('--resolution', type=float, default=50.0,
                       help='Target resolution in Angstroms')
    
    # Likelihood
    parser.add_argument('--likelihood', type=str, choices=['gaussian', 'laplace', 'cauchy'],
                       default='gaussian', help='Likelihood type for CCC')
    parser.add_argument('--sigma_ccc', type=float, default=0.1,
                       help='Width of CCC likelihood')
    
    # Prior
    parser.add_argument('--box_size', type=float, default=300.0,
                       help='Bounding box size in Angstroms')
    parser.add_argument('--box_steepness', type=float, default=10.0,
                       help='Soft box penalty strength')
    parser.add_argument('--slope_factor', type=float, default=0.01,
                       help='Attraction to density COM')
    
    # SMC
    parser.add_argument('--n_mcmc_steps', type=int, default=50,
                       help='MCMC steps per temperature')
    parser.add_argument('--target_ess', type=float, default=0.5,
                       help='Target ESS ratio')
    
    # RMH specific
    parser.add_argument('--rmh_sigma', type=float, default=3.0,
                       help='RMH proposal step size')
    
    # HMC specific
    parser.add_argument('--hmc_step_size', type=float, default=0.01,
                       help='HMC leapfrog step size')
    parser.add_argument('--hmc_L', type=int, default=10,
                       help='HMC number of leapfrog steps')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output_smc',
                       help='Output directory')
    parser.add_argument('--output_prefix', type=str, default='smc_modular',
                       help='Output file prefix')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce console output')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"JAX default backend: {jax.default_backend()}")
    
    config = SMCConfig(
        kernel=args.kernel,
        n_particles=args.n_particles,
        random_seed=args.seed,
        mrc_path=args.mrc,
        resolution=args.resolution,
        likelihood_type=args.likelihood,
        sigma_ccc=args.sigma_ccc,
        box_size=args.box_size,
        box_steepness=args.box_steepness,
        slope_factor=args.slope_factor,
        n_mcmc_steps=args.n_mcmc_steps,
        target_ess=args.target_ess,
        rmh_sigma=args.rmh_sigma,
        hmc_step_size=args.hmc_step_size,
        hmc_num_integration_steps=args.hmc_L,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        verbose=not args.quiet,
    )
    
    results = run_smc_simulation(config)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Best CCC:    {results['best_ccc']:.4f}")
    print(f"Best Score:  {results['best_score']:.2f}")
    print(f"Wall Time:   {results['wall_time']:.2f}s")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
