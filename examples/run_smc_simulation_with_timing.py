"""
SMC simulation using BlackJAX with RMH kernel + proper probabilistic EM scoring.

Probabilistic model (from em_score.py):
    prior     = soft_box(x) * exp(-lambda * sum_i ||r_i - r_COM||)
    likelihood = exp(-(1 - CCC(x))^2 / (2 * sigma_ccc^2))  + structural restraints

    SMC tempers: pi_t(x) ∝ prior(x) * likelihood(x)^{lambda_t}

The prior is always "on" (guides particles toward density region).
The likelihood is gradually turned on via tempering.
"""
import numpy as np
import sys
import os
from pathlib import Path
import mrcfile
import time

# comment the next line to use GPU/TPU if available
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

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
    create_em_scoring_model,
    calculate_ccc_jax,
    diagnose_model,
)


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
    timer = WallTimer()

    print("=" * 60)
    print("SMC Sampling with Proper Probabilistic EM Scoring (RMH kernel)")
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
        print(f"  Density COM: [{float(em_config.density_com[0]):.1f}, "
              f"{float(em_config.density_com[1]):.1f}, "
              f"{float(em_config.density_com[2]):.1f}]")

    timer.stop("2. Load density map")

    # =========================================================================
    # 3. Initialize coordinates
    # =========================================================================
    timer.start("3. Initialize coordinates")

    temp_system = ParticleSystem(types_config, {}, ideal_coords)

    init_box_size = 250.0
    coords = temp_system.get_random_coords(
        jax.random.PRNGKey(128907189),
        box_size=[init_box_size, init_box_size, init_box_size],
    )

    system = ParticleSystem(types_config, coords, ideal_coords)
    flat_radii = system.get_flat_radii()
    radii_jax = jnp.array(flat_radii, dtype=jnp.float32)
    n_dims = system.total_particles * 3

    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")

    timer.stop("3. Initialize coordinates")

    # =========================================================================
    # 4. Build probabilistic model
    # =========================================================================
    timer.start("4. Build probabilistic model")

    # --- Hyperparameters ---
    # sigma_ccc: width of Gaussian on (1-CCC). Smaller = sharper CCC discrimination.
    #   sigma=0.1: only CCC > 0.9 gets significant probability
    #   sigma=0.3: moderate (good default)
    #   sigma=1.0: very flat
    #
    # lambda_attract: exponential attraction to density COM.
    #   With 32 particles at ~200Å mean distance:
    #     0.001 -> log_prior ~ -6    (gentle)
    #     0.005 -> log_prior ~ -32   (moderate)
    #     0.01  -> log_prior ~ -64   (strong)

    sigma_ccc = 0.0025
    lambda_attract = 0.03
    box_size = 250.0
    box_steepness = 1.0

    print(f"\nProbabilistic model:")
    print(f"  Likelihood: Gaussian on (1-CCC), sigma_ccc = {sigma_ccc}")
    print(f"  Prior:      exp(-lambda * sum ||r_i - COM||), lambda = {lambda_attract}")
    print(f"  Box:        soft quadratic, size = {box_size}, steepness = {box_steepness}")

    # Create proper probabilistic model
    log_prior_fn, ccc_log_likelihood_fn, _ = create_em_scoring_model(
        config=em_config,
        radii=flat_radii,
        sigma_ccc=sigma_ccc,
        lambda_attract=lambda_attract,
        box_size=box_size,
        box_steepness=box_steepness,
    )

    # Structural restraints
    target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
    nuisance_params = {'AA': 1.5, 'AB': 1.3, 'BC': 1.0}

    # Combined likelihood = CCC Gaussian likelihood + structural restraints
    @jax.jit
    def log_likelihood_fn(flat_coords):
        """Combined likelihood: Gaussian CCC + structural restraints."""
        ccc_term = ccc_log_likelihood_fn(flat_coords)
        struct_term = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0, pair_weight=0.0001, exvol_sigma=0.10,
        )
        return ccc_term + struct_term

    # Full log-posterior (for scoring/diagnostics only)
    @jax.jit
    def log_prob_fn(flat_coords):
        """Full log-posterior = prior + likelihood."""
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)

    # Raw CCC for diagnostics (no model transformation)
    @jax.jit
    def get_ccc(flat_coords):
        """Get raw CCC for diagnostics."""
        coords = flat_coords.reshape(-1, 3)
        return calculate_ccc_jax(coords, radii_jax, em_config)

    timer.stop("4. Build probabilistic model")

    # =========================================================================
    # 5. Model diagnostics at initial position
    # =========================================================================
    timer.start("5. Model diagnostics")

    dummy_coords = system.flatten(coords)

    print("\nModel diagnostics at initial position:")
    diag = diagnose_model(
        dummy_coords, em_config, flat_radii,
        sigma_ccc=sigma_ccc, lambda_attract=lambda_attract, box_size=box_size,
    )
    for key, val in diag.items():
        print(f"  {key:<25s} = {val:>12.4f}")

    # Check individual components
    print("\nComponent values:")
    prior_val = log_prior_fn(dummy_coords)
    lik_val = log_likelihood_fn(dummy_coords)
    ccc_val = get_ccc(dummy_coords)
    struct_val = log_probability(
        dummy_coords, system, flat_radii,
        target_dists, nuisance_params,
        exclusion_weight=1.0, pair_weight=0.00075, exvol_sigma=0.10,
    )

    print(f"  log_prior:                 {float(prior_val):>12.4f}")
    print(f"  log_likelihood (combined): {float(lik_val):>12.4f}")
    print(f"    - CCC likelihood:        {float(ccc_log_likelihood_fn(dummy_coords)):>12.4f}")
    print(f"    - Structural restraints: {float(struct_val):>12.4f}")
    print(f"  log_posterior:             {float(prior_val + lik_val):>12.4f}")
    print(f"  Raw CCC:                   {float(ccc_val):>12.4f}")

    # Verify no -inf values (critical for SMC!)
    print("\nSanity checks:")
    print(f"  Prior is finite:      {bool(jnp.isfinite(prior_val))}")
    print(f"  Likelihood is finite: {bool(jnp.isfinite(lik_val))}")
    print(f"  Posterior is finite:  {bool(jnp.isfinite(prior_val + lik_val))}")

    if not jnp.isfinite(prior_val):
        print("\n  ⚠️  WARNING: Prior returns -inf! This will break SMC.")
        print("  The soft box prior should never return -inf.")

    # JIT warmup
    _ = log_prob_fn(dummy_coords)
    jax.block_until_ready(_)

    timer.stop("5. Model diagnostics")

    # =========================================================================
    # 6. Initialize SMC particles
    # =========================================================================
    timer.start("6. Initialize SMC particles")

    n_particles = 250
    rng_key = jax.random.PRNGKey(90998210)
    rng_key, init_key = jax.random.split(rng_key)

    flat_init = system.flatten(coords)
    initial_positions = flat_init + jax.random.normal(init_key, (n_particles, n_dims)) * 5.0

    # Validate all particles
    init_scores = jax.vmap(log_prob_fn)(initial_positions)
    init_priors = jax.vmap(log_prior_fn)(initial_positions)
    init_liks = jax.vmap(log_likelihood_fn)(initial_positions)
    jax.block_until_ready(init_scores)

    valid_mask = jnp.isfinite(init_scores)
    valid_count = int(jnp.sum(valid_mask))

    print(f"\nParticle initialization:")
    print(f"  Total particles:    {n_particles}")
    print(f"  Valid (finite):     {valid_count}/{n_particles}")
    print(f"  Prior range:        [{float(jnp.min(init_priors)):.2f}, {float(jnp.max(init_priors)):.2f}]")
    print(f"  Likelihood range:   [{float(jnp.min(init_liks)):.2f}, {float(jnp.max(init_liks)):.2f}]")
    print(f"  Score range:        [{float(jnp.nanmin(jnp.where(valid_mask, init_scores, jnp.nan))):.2f}, "
          f"{float(jnp.nanmax(jnp.where(valid_mask, init_scores, jnp.nan))):.2f}]")
    print(f"  Score mean (valid): {float(jnp.nanmean(jnp.where(valid_mask, init_scores, jnp.nan))):.2f}")

    if valid_count < n_particles:
        print(f"\n  ⚠️  {n_particles - valid_count} particles have -inf score!")
        n_inf_prior = int(jnp.sum(~jnp.isfinite(init_priors)))
        n_inf_lik = int(jnp.sum(~jnp.isfinite(init_liks)))
        print(f"  Due to prior:      {n_inf_prior}")
        print(f"  Due to likelihood: {n_inf_lik}")

    timer.stop("6. Initialize SMC particles")

    # =========================================================================
    # 7. Run SMC
    # =========================================================================
    timer.start("7. SMC sampling")

    # --- SMC hyperparameters ---
    # n_mcmc_steps: RMH mutations per temperature step.
    #   More = better mixing but slower. 100-500 is typical.
    # rmh_sigma: proposal step size.
    #   Too large = low acceptance, too small = slow mixing.
    #   Should be ~fraction of typical coordinate scale.
    # target_ess: controls how aggressively tempering advances.
    #   0.5 = default (moderate steps)
    #   0.75 = conservative (smaller λ increments, more steps)
    #   0.9 = very conservative

    n_mcmc_steps = 100
    rmh_sigma = 5.0
    target_ess = 0.75

    print(f"\nSMC configuration:")
    print(f"  MCMC steps per temp:  {n_mcmc_steps}")
    print(f"  RMH sigma:            {rmh_sigma}")
    print(f"  Target ESS:           {target_ess:.0%}")
    print(f"  Particles:            {n_particles}")

    rng_key, smc_key = jax.random.split(rng_key)
    final_state, info_history, best_positions, best_scores = run_tempered_smc(
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_positions,
        rng_key=smc_key,
        n_mcmc_steps=n_mcmc_steps,
        rmh_sigma=rmh_sigma,
        target_ess=target_ess,
        record_best=True,
    )

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

    best_ccc = get_ccc(best_pos)

    print(f"\n{'='*60}")
    print(f"Final Results")
    print(f"{'='*60}")
    print(f"Final Score (mean):  {float(jnp.mean(final_scores)):.2f}")
    print(f"Final Score (best):  {best_score:.2f}")
    print(f"Best CCC:            {float(best_ccc):.4f}")

    # Model diagnostics for best particle
    print(f"\nModel diagnostics for best particle:")
    diag_best = diagnose_model(
        best_pos, em_config, flat_radii,
        sigma_ccc=sigma_ccc, lambda_attract=lambda_attract, box_size=box_size,
    )
    for key, val in diag_best.items():
        print(f"  {key:<25s} = {val:>12.4f}")

    # Per-step CCC table
    if best_positions is not None and best_scores is not None:
        print(f"\n{'='*60}")
        print("CCC per SMC Step (Best Particle)")
        print(f"{'='*60}")
        print(f"{'Step':<8} {'Score':>12} {'CCC':>12} {'Coord Disp':>14}")
        print("-" * 60)

        best_cccs = jax.vmap(get_ccc)(jnp.array(best_positions))
        jax.block_until_ready(best_cccs)

        ref_pos = best_positions[0]
        for step_idx, (score, ccc) in enumerate(zip(best_scores, best_cccs)):
            disp = float(jnp.linalg.norm(
                jnp.array(best_positions[step_idx]) - jnp.array(ref_pos)
            ))
            print(f"{step_idx:<8} {float(score):>12.2f} {float(ccc):>12.4f} {disp:>14.2f}")

        print("=" * 60)

        # Motion statistics
        final_disp = float(jnp.linalg.norm(
            jnp.array(best_positions[-1]) - jnp.array(best_positions[0])
        ))
        print(f"Total displacement (first→last best): {final_disp:.2f} Å")

        # Per-particle motion
        init_coords = best_positions[0].reshape(-1, 3)
        final_coords = best_positions[-1].reshape(-1, 3)
        per_particle_disp = np.linalg.norm(final_coords - init_coords, axis=1)
        print(f"Per-particle displacement: min={per_particle_disp.min():.2f}, "
              f"max={per_particle_disp.max():.2f}, mean={per_particle_disp.mean():.2f} Å")

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
            params={
                'method': 'BlackJAX_SMC_RMH',
                'model': 'Gaussian_CCC + Exp_Distance + Structural',
                'sigma_ccc': sigma_ccc,
                'lambda_attract': lambda_attract,
                'n_mcmc_steps': n_mcmc_steps,
                'rmh_sigma': rmh_sigma,
                'target_ess': target_ess,
                'n_particles': n_particles,
                'best_ccc': float(best_ccc),
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
        print(f"\nSaved to {output_file}")

    timer.stop("9. Save results")

    timer.summary()


if __name__ == "__main__":
    main()
