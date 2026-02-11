"""
Run SMC simulation with HMC mutation kernel.

Usage:
    # CPU testing:
    python run_smc_hmc_simulation.py

    # GPU (comment out the JAX_PLATFORM_NAME line below):
    python run_smc_hmc_simulation.py

GPU memory notes:
    All particle-parallel operations use batched_vmap instead of jax.vmap
    to avoid OOM with large density maps. The batch_size parameter controls
    how many particles are evaluated simultaneously. Default is 16; reduce
    further if your GPU has limited memory (<8GB).
"""
import numpy as np
import sys
import os
from pathlib import Path
import mrcfile
import time

# ============================================================================
# Backend selection: comment this line to use GPU/TPU
# ============================================================================
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.smc_with_hmc import (
    run_tempered_smc,
    get_smc_samples,
    get_best_sample,
    batched_vmap,
)
from io_utils.io_handlers import save_mcmc_to_hdf5
from scoring.em_score import (
    create_em_config_from_mrcfile,
    create_em_log_prob_fn,
    calculate_ccc_jax,
)


# =============================================================================
# GPU memory config
# =============================================================================
# How many particles to evaluate in parallel.
# Tune based on your GPU: 160^3 map × 32 particles × float32 ≈ ~800MB per particle
#   8 GB GPU  -> SCORE_BATCH_SIZE = 8
#   16 GB GPU -> SCORE_BATCH_SIZE = 16
#   24 GB GPU -> SCORE_BATCH_SIZE = 32
#   CPU       -> SCORE_BATCH_SIZE = 200 (no issue, set to n_particles)
SCORE_BATCH_SIZE = 16


# =============================================================================
# Timing utilities
# =============================================================================

def sync_and_time() -> float:
    jax.block_until_ready(jnp.zeros(1))
    return time.perf_counter()


class WallTimer:
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


# =============================================================================
# Differentiable prior for HMC
# =============================================================================

def make_soft_box_prior(box_size: float, steepness: float = 1.0):
    """
    Differentiable soft-wall box prior for HMC compatibility.

    Uses quadratic penalty beyond the boundary instead of -inf.
    Gradient is well-defined everywhere.
    """
    @jax.jit
    def log_prior_fn(flat_coords):
        coords = flat_coords.reshape(-1, 3)
        excess = jnp.maximum(jnp.abs(coords) - box_size, 0.0)
        penalty = -jnp.sum(excess ** 2) / (2.0 * steepness ** 2)
        return penalty

    return log_prior_fn


def main():
    timer = WallTimer()

    print("=" * 60)
    print("SMC Sampling with HMC Kernel + EM Density Scoring")
    print(f"Backend: {jax.default_backend()}")
    print(f"Score batch size: {SCORE_BATCH_SIZE}")
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

    init_box_size = 500.0
    coords = temp_system.get_random_coords(
        jax.random.PRNGKey(128907189),
        box_size=[init_box_size, init_box_size, init_box_size],
    )

    system = ParticleSystem(types_config, coords, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3

    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")

    timer.stop("3. Initialize coordinates")

    # =========================================================================
    # 4. Scoring functions (all JAX-differentiable)
    # =========================================================================
    timer.start("4. Setup scoring functions")

    slope = 0.05
    em_scale = 500.0
    em_log_prob = create_em_log_prob_fn(em_config, flat_radii, scale=em_scale, slope=slope)
    radii_jax = jnp.array(flat_radii, dtype=jnp.float32)

    target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
    nuisance_params = {'AA': 1.3, 'AB': 1.1, 'BC': 1.0}
    box_size = 500.0

    # Soft prior (HMC-compatible, no -inf)
    log_prior_fn = make_soft_box_prior(box_size, steepness=1.0)

    @jax.jit
    def log_likelihood_fn(flat_coords):
        em_val = em_log_prob(flat_coords)
        struct_val = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0, pair_weight=2.0, exvol_sigma=0.10,
        )
        return em_val + struct_val

    @jax.jit
    def log_prob_fn(flat_coords):
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)

    @jax.jit
    def get_ccc(flat_coords):
        coords = flat_coords.reshape(-1, 3)
        return calculate_ccc_jax(coords, radii_jax, em_config, slope=0.0)

    timer.stop("4. Setup scoring functions")

    # =========================================================================
    # 5. Gradient check + JIT warmup
    # =========================================================================
    timer.start("5. Gradient check + JIT warmup")

    dummy_coords = system.flatten(coords)

    print("\nGradient check (HMC requirement):")
    grad_prior = jax.grad(log_prior_fn)(dummy_coords)
    grad_likelihood = jax.grad(log_likelihood_fn)(dummy_coords)
    jax.block_until_ready(grad_prior)
    jax.block_until_ready(grad_likelihood)

    prior_grad_ok = bool(jnp.all(jnp.isfinite(grad_prior)))
    like_grad_ok = bool(jnp.all(jnp.isfinite(grad_likelihood)))
    print(f"  Prior gradient finite:      {prior_grad_ok}")
    print(f"  Likelihood gradient finite:  {like_grad_ok}")
    print(f"  Prior grad norm:            {float(jnp.linalg.norm(grad_prior)):.4f}")
    print(f"  Likelihood grad norm:       {float(jnp.linalg.norm(grad_likelihood)):.4f}")

    if not (prior_grad_ok and like_grad_ok):
        print("\n  WARNING: Non-finite gradients detected! HMC may produce NaN.")

    # Warmup scoring JIT
    _ = log_prob_fn(dummy_coords)
    _ = get_ccc(dummy_coords)
    jax.block_until_ready(_)

    timer.stop("5. Gradient check + JIT warmup")

    # =========================================================================
    # 6. Initialize SMC particles (batched scoring)
    # =========================================================================
    timer.start("6. Initialize SMC particles")

    n_particles = 200
    rng_key = jax.random.PRNGKey(90998210)
    rng_key, init_key = jax.random.split(rng_key)

    flat_init = system.flatten(coords)
    initial_positions = flat_init + jax.random.normal(init_key, (n_particles, n_dims)) * 5.0

    # --- BATCHED scoring to avoid OOM ---
    score_particles = batched_vmap(log_prob_fn, batch_size=SCORE_BATCH_SIZE)
    init_scores = score_particles(initial_positions)
    valid_count = jnp.sum(jnp.isfinite(init_scores))
    jax.block_until_ready(init_scores)

    print(f"\nValid particles: {int(valid_count)}/{n_particles}")
    print(f"Initial Score (mean): "
          f"{float(jnp.nanmean(jnp.where(jnp.isfinite(init_scores), init_scores, jnp.nan))):.2f}")

    coords_3d = flat_init.reshape(-1, 3)
    print(f"Coord ranges: "
          f"X[{float(coords_3d[:,0].min()):.1f}, {float(coords_3d[:,0].max()):.1f}], "
          f"Y[{float(coords_3d[:,1].min()):.1f}, {float(coords_3d[:,1].max()):.1f}], "
          f"Z[{float(coords_3d[:,2].min()):.1f}, {float(coords_3d[:,2].max()):.1f}]")

    timer.stop("6. Initialize SMC particles")

    # =========================================================================
    # 7. HMC tuning parameters
    # =========================================================================
    hmc_step_size = 0.01
    hmc_num_integration_steps = 20
    n_mcmc_steps = 50

    hmc_inverse_mass_matrix = None  # identity

    # =========================================================================
    # 8. Run SMC with HMC
    # =========================================================================
    timer.start("7. SMC-HMC sampling")

    rng_key, smc_key = jax.random.split(rng_key)
    final_state, info_history, best_positions, best_scores = run_tempered_smc(
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_positions,
        rng_key=smc_key,
        n_mcmc_steps=n_mcmc_steps,
        hmc_step_size=hmc_step_size,
        hmc_inverse_mass_matrix=hmc_inverse_mass_matrix,
        hmc_num_integration_steps=hmc_num_integration_steps,
        target_ess=0.75,
        record_best=True,
        score_batch_size=SCORE_BATCH_SIZE,
    )

    jax.block_until_ready(final_state.particles)
    timer.stop("7. SMC-HMC sampling")

    # =========================================================================
    # 9. Post-processing (batched scoring)
    # =========================================================================
    timer.start("8. Post-processing")

    final_positions = get_smc_samples(final_state)
    best_pos, best_score = get_best_sample(
        final_state, log_prob_fn, batch_size=SCORE_BATCH_SIZE
    )

    # --- BATCHED ---
    final_scores = score_particles(final_positions)
    jax.block_until_ready(final_scores)

    best_ccc = get_ccc(best_pos)

    print(f"\nFinal Score (mean): {float(jnp.mean(final_scores)):.2f}")
    print(f"Best Score: {best_score:.2f}")
    print(f"Best CCC: {float(best_ccc):.4f}")

    if best_positions is not None and best_scores is not None:
        print("\n" + "=" * 60)
        print("CCC Score per SMC Step (Best Particle)")
        print("=" * 60)
        print(f"{'Step':<8} {'Score':>12} {'CCC':>12}")
        print("-" * 60)

        # --- BATCHED CCC computation ---
        ccc_batched = batched_vmap(get_ccc, batch_size=SCORE_BATCH_SIZE)
        best_cccs = ccc_batched(jnp.array(best_positions))
        jax.block_until_ready(best_cccs)

        for step_idx, (score, ccc) in enumerate(zip(best_scores, best_cccs)):
            print(f"{step_idx:<8} {float(score):>12.2f} {float(ccc):>12.4f}")

        print("=" * 60)

    timer.stop("8. Post-processing")

    # =========================================================================
    # 10. Save results
    # =========================================================================
    timer.start("9. Save results")

    if best_positions is not None and best_scores is not None:
        output_file = output_dir / "smc_hmc_trajectory.h5"
        save_mcmc_to_hdf5(
            np.array(best_positions),
            np.array(best_scores),
            1.0,
            str(output_file),
            system,
            params={
                'method': 'BlackJAX_SMC_HMC',
                'trajectory': 'best_per_step',
                'best_ccc': float(best_ccc),
                'hmc_step_size': float(hmc_step_size),
                'hmc_L': hmc_num_integration_steps,
                'n_mcmc_steps': n_mcmc_steps,
                'score_batch_size': SCORE_BATCH_SIZE,
            },
            convert_to_rmf3=True,
            color_map={
                'A': (0.2, 0.6, 1.0),
                'B': (0.9, 0.4, 0.2),
                'C': (0.3, 0.8, 0.4),
            },
        )
        print(f"\nSaved to {output_file}")

    timer.stop("9. Save results")
    timer.summary()


if __name__ == "__main__":
    main()