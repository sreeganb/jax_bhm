"""
Example script to run SMC simulation using BlackJAX with wall time measurement.
"""
import numpy as np
import sys
import os
from pathlib import Path
import mrcfile
import time

# comment the next line to use GPU/TPU if available
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax

print("JAX default backend:", jax.default_backend())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.smc import run_tempered_smc, get_smc_samples, get_best_sample
from io_utils.io_handlers import save_mcmc_to_hdf5
from scoring.em_score import (
    create_em_config_from_mrcfile,
    create_em_config_from_arrays,
    calc_projection_jax,
    calculate_ccc_score,
    create_em_log_prob_fn,
    pairwise_correlation_jax,
)


# =============================================================================
# Timing utilities for JAX code
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
    
    def summary(self):
        total = self.total()
        print("\n" + "=" * 60)
        print(f"TIMING SUMMARY (Backend: {jax.default_backend()})")
        print("=" * 60)
        print(f"{'Section':<35} {'Time (s)':>10} {'%':>8}")
        print("-" * 60)
        for name, elapsed in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = 100 * elapsed / total if total > 0 else 0
            print(f"{name:<35} {elapsed:>10.2f} {pct:>7.1f}%")
        print("-" * 60)
        print(f"{'TOTAL WALL TIME':<35} {total:>10.2f} {'100.0':>7}%")
        print("=" * 60)


def main():
    # Initialize timer
    timer = WallTimer()
    
    print("=" * 60)
    print("SMC Sampling with EM Density Scoring")
    print(f"Backend: {jax.default_backend()}")
    print("=" * 60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # 1. Setup system
    # =========================================================================
    timer.start("1. System setup")
    
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }
    
    ideal_coords = get_ideal_coords()
    
    timer.stop("1. System setup")
    
    # =========================================================================
    # 2. Load target density
    # =========================================================================
    timer.start("2. Load density map")
    
    mrc_path = output_dir / "simulated_target_density.mrc"
    resolution = 50.0
    
    print(f"\nLoading target density: {mrc_path}")
    with mrcfile.open(str(mrc_path), mode='r') as mrc:
        em_config = create_em_config_from_mrcfile(mrc, resolution)
        density_voxel_size = float(mrc.voxel_size.x)
        print(f"  Shape: {mrc.data.shape}, Voxel: {density_voxel_size:.2f} Å")
    
    timer.stop("2. Load density map")
    
    # =========================================================================
    # 3. Initialize coordinates
    # =========================================================================
    timer.start("3. Initialize coordinates")
    
    temp_system = ParticleSystem(types_config, {}, ideal_coords)
    
    # Initialize particles centered at origin, within a smaller box
    # to ensure they stay inside the ±500 box constraint after noise is added
    init_box_size = 200.0  # Smaller initial box
    coords = temp_system.get_random_coords(
        jax.random.PRNGKey(9090), box_size=[init_box_size, init_box_size, init_box_size]
    )
    
    system = ParticleSystem(types_config, coords, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    timer.stop("3. Initialize coordinates")
    
    # =========================================================================
    # 4. Setup scoring functions
    # =========================================================================
    timer.start("4. Setup scoring functions")

    # Slope for EM score to keep particles close to the map
    slope = 0.1
    
    # EM score: returns scale * CCC (log-probability, higher = better)
    em_scale = 500.0
    em_log_prob = create_em_log_prob_fn(em_config, flat_radii, scale=em_scale, slope=slope)
    radii_jax = jnp.array(flat_radii, dtype=jnp.float32)
    
    target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
    nuisance_params = {'AA': 1.6, 'AB': 1.4, 'BC': 1.0}
    box_size = 500.0
    
    def log_prior_fn(flat_coords):
        """Log prior: uniform in box, -inf outside."""
        coords = flat_coords.reshape(-1, 3)
        in_box = jnp.all((coords >= -box_size) & (coords <= box_size))
        return jnp.where(in_box, 0.0, -jnp.inf)
    
    @jax.jit
    def log_likelihood_fn(flat_coords):
        """
        Log likelihood = EM log-prob + pair/exvol log-prob
        
        Both terms return higher values for better configurations.
        """
        # EM term: scale * CCC (higher CCC = higher log-prob)
        em_log_prob_value = em_log_prob(flat_coords)
        
        # Pair + excluded volume term: -(ev_penalty + pair_nll)
        # Already returns log-probability (negative of penalties)
        structure_log_prob = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0, pair_weight=1.0, exvol_sigma=1.0
        )
        
        return em_log_prob_value + structure_log_prob
    
    def log_prob_fn(flat_coords):
        """Total log probability = log_prior + log_likelihood."""
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)
    
    # Function to get CCC without recomputing projection
    get_ccc = jax.jit(lambda x: em_log_prob.with_ccc(x)[1])
    
    timer.stop("4. Setup scoring functions")
    
    # =========================================================================
    # 5. JIT compilation warmup
    # =========================================================================
    timer.start("5. JIT compilation (warmup)")
    
    # Force JIT compilation by running once
    dummy_coords = system.flatten(coords)
    _ = log_prob_fn(dummy_coords)
    _ = get_ccc(dummy_coords)
    jax.block_until_ready(_)
    
    timer.stop("5. JIT compilation (warmup)")
    
    # =========================================================================
    # 6. Initialize SMC particles
    # =========================================================================
    timer.start("6. Initialize SMC particles")
    
    n_particles = 100
    rng_key = jax.random.PRNGKey(9090)
    rng_key, init_key = jax.random.split(rng_key)
    
    flat_init = system.flatten(coords)
    
    # Smaller noise to keep particles valid
    initial_positions = flat_init + jax.random.normal(init_key, (n_particles, n_dims)) * 5.0
    
    # Check how many particles are valid
    init_scores = jax.vmap(log_prob_fn)(initial_positions)
    valid_count = jnp.sum(jnp.isfinite(init_scores))
    jax.block_until_ready(init_scores)
    
    print(f"\nValid particles: {int(valid_count)}/{n_particles}")
    print(f"Initial Score (mean of valid): {jnp.nanmean(jnp.where(jnp.isfinite(init_scores), init_scores, jnp.nan)):.2f}")
    
    # Debug: check coordinate ranges
    coords_3d = flat_init.reshape(-1, 3)
    print(f"Coord ranges: X[{float(coords_3d[:,0].min()):.1f}, {float(coords_3d[:,0].max()):.1f}], "
          f"Y[{float(coords_3d[:,1].min()):.1f}, {float(coords_3d[:,1].max()):.1f}], "
          f"Z[{float(coords_3d[:,2].min()):.1f}, {float(coords_3d[:,2].max()):.1f}]")
    
    timer.stop("6. Initialize SMC particles")
    
    # =========================================================================
    # 7. Run SMC (main computation)
    # =========================================================================
    timer.start("7. SMC sampling")
    
    rng_key, smc_key = jax.random.split(rng_key)
    final_state, info_history, best_positions, best_scores = run_tempered_smc(
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_positions,
        rng_key=smc_key,
        n_mcmc_steps=50,
        rmh_sigma=4.0,
        target_ess=0.7,
        record_best=True,
    )
    
    # IMPORTANT: Block until SMC is complete before stopping timer
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
    
    print(f"\nFinal Score (mean): {jnp.mean(final_scores):.2f}")
    print(f"Best Score: {best_score:.2f}")
    
    # Print CCC for each step's best particle
    if best_positions is not None and best_scores is not None:
        print("\n" + "=" * 60)
        print("CCC Score per SMC Step (Best Particle)")
        print("=" * 60)
        print(f"{'Step':<8} {'Score':>12} {'CCC':>12}")
        print("-" * 60)
        
        for step_idx, (pos, score) in enumerate(zip(best_positions, best_scores)):
            ccc_value = get_ccc(jnp.array(pos))
            print(f"{step_idx:<8} {score:>12.2f} {float(ccc_value):>12.4f}")
        
        print("=" * 60)
    
    timer.stop("8. Post-processing")
    
    # =========================================================================
    # 9. Save results
    # =========================================================================
    timer.start("9. Save results")
    
    if best_positions is not None and best_scores is not None:
        output_file = output_dir / "smc_trajectory.h5"
        save_mcmc_to_hdf5(
            np.array(best_positions),
            np.array(best_scores),
            1.0,
            str(output_file),
            system,
            params={'method': 'BlackJAX_SMC', 'trajectory': 'best_per_step'},
            convert_to_rmf3=True,
            color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
        )
        print(f"\nSaved to {output_file}")
    
    timer.stop("9. Save results")
    
    # =========================================================================
    # Print timing summary
    # =========================================================================
    timer.summary()


if __name__ == "__main__":
    main()