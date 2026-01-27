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


# =============================================================================
# Helper functions
# =============================================================================

def compute_density_center_of_mass(density: np.ndarray, voxel_size: float) -> np.ndarray:
    """Compute center of mass of density map."""
    nz, ny, nx = density.shape
    half_x, half_y, half_z = nx * voxel_size / 2, ny * voxel_size / 2, nz * voxel_size / 2
    
    x = np.linspace(-half_x + voxel_size/2, half_x - voxel_size/2, nx)
    y = np.linspace(-half_y + voxel_size/2, half_y - voxel_size/2, ny)
    z = np.linspace(-half_z + voxel_size/2, half_z - voxel_size/2, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    density_pos = np.maximum(density, 0)
    total_mass = np.sum(density_pos)
    
    if total_mass < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    
    return np.array([
        np.sum(X * density_pos) / total_mass,
        np.sum(Y * density_pos) / total_mass,
        np.sum(Z * density_pos) / total_mass
    ])


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
        density_data = mrc.data.copy()
        density_voxel_size = float(mrc.voxel_size.x)
        print(f"  Shape: {mrc.data.shape}, Voxel: {density_voxel_size:.2f} Å")
    
    density_com = compute_density_center_of_mass(density_data, density_voxel_size)
    print(f"  Density COM: [{density_com[0]:.2f}, {density_com[1]:.2f}, {density_com[2]:.2f}] Å")
    
    timer.stop("2. Load density map")
    
    # =========================================================================
    # 3. Initialize coordinates
    # =========================================================================
    timer.start("3. Initialize coordinates")
    
    temp_system = ParticleSystem(types_config, {}, ideal_coords)
    coords = temp_system.get_random_coords(
        jax.random.PRNGKey(9090), box_size=[500.0, 500.0, 500.0]
    )
    
    # Shift to density COM
    identity_order = sorted(types_config.keys())
    coords_array = jnp.concatenate([coords[k] for k in identity_order], axis=0)
    coords_com = jnp.mean(coords_array, axis=0)
    shift_vector = density_com - np.array(coords_com)
    
    shifted_coords = {k: coords[k] + jnp.array(shift_vector) for k in coords}
    
    system = ParticleSystem(types_config, shifted_coords, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    timer.stop("3. Initialize coordinates")
    
    # =========================================================================
    # 4. Setup scoring functions
    # =========================================================================
    timer.start("4. Setup scoring functions")
    
    em_log_prob = create_em_log_prob_fn(em_config, flat_radii, scale=1.0)
    radii_jax = jnp.array(flat_radii, dtype=jnp.float32)
    
    target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
    box_size = 500.0

    # ---- NEW: nuisance parameter prior config ----
    sigma_keys = ['AA', 'AB', 'BC']

    # Per-parameter mean/std for truncated normal
    sigma_mu = {'AA': 2.0, 'AB': 1.6, 'BC': 1.1}
    sigma_sd = {'AA': 1.1, 'AB': 1.0, 'BC': 0.9}
    trunc_a, trunc_b = 0.0, 5.0  # for truncnorm in standardized units

    # Per-parameter alpha/beta for inverse-gamma (shape/scale)
    invgamma_alpha = {'AA': 3.0, 'AB': 2.5, 'BC': 4.0}
    invgamma_beta  = {'AA': 6.0, 'AB': 5.0, 'BC': 7.0}

    # Per-parameter scale for half-Cauchy
    halfcauchy_scale = {'AA': 2.0, 'AB': 1.5, 'BC': 3.0}

    # Select which prior to use: "truncnorm", "invgamma", or "halfcauchy"
    sigma_prior_type = "truncnorm"

    def unpack_state(state):
        coords = state[:-3]
        sigma = state[-3:]  # [sigma_aa, sigma_ab, sigma_bc]
        return coords, sigma

    def log_prior_sigma(sigma):
        # sigma is a length-3 vector in order AA, AB, BC
        logp = 0.0
        for i, key in enumerate(sigma_keys):
            x = sigma[i]
            if sigma_prior_type == "truncnorm":
                loc = sigma_mu[key]
                scale = sigma_sd[key]
                logp += jax.scipy.stats.truncnorm.logpdf(
                    x, a=trunc_a, b=trunc_b, loc=loc, scale=scale
                )
            elif sigma_prior_type == "invgamma":
                # logpdf for Inverse-Gamma(alpha, beta)
                a = invgamma_alpha[key]
                b = invgamma_beta[key]
                # log p(x) = a*log(b) - lgamma(a) - (a+1)log(x) - b/x
                logp += a * jnp.log(b) - jax.scipy.special.gammaln(a) - (a + 1.0) * jnp.log(x) - b / x
            elif sigma_prior_type == "halfcauchy":
                # logpdf for Half-Cauchy(scale)
                s = halfcauchy_scale[key]
                # p(x) = 2 / (pi*s*(1 + (x/s)^2)), x>0
                logp += jnp.log(2.0) - jnp.log(jnp.pi * s) - jnp.log1p((x / s) ** 2)
            else:
                raise ValueError(f"Unknown sigma prior: {sigma_prior_type}")
        return logp

    def log_prior_fn(state):
        coords, sigma = unpack_state(state)

        coords_reshaped = coords.reshape(-1, 3)
        in_box = jnp.all((coords_reshaped >= -box_size) & (coords_reshaped <= box_size))
        logp_coords = jnp.where(in_box, 0.0, -jnp.inf)

        logp_sigma = log_prior_sigma(sigma)
        return logp_coords + logp_sigma
    
    @jax.jit
    def log_likelihood_fn(state):
        coords, sigma = unpack_state(state)

        nuisance_params = {'AA': sigma[0], 'AB': sigma[1], 'BC': sigma[2]}

        ccc = em_log_prob(coords)
        em_score = -(1.0 - ccc)

        log_lik = log_probability(
            coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0, pair_weight=1.0, exvol_sigma=0.1
        )
        return em_score + log_lik
    
    def log_prob_fn(state):
        return log_prior_fn(state) + log_likelihood_fn(state)
    
    timer.stop("4. Setup scoring functions")
    
    # =========================================================================
    # 5. JIT compilation warmup
    # =========================================================================
    timer.start("5. JIT compilation (warmup)")
    
    # Force JIT compilation by running once with full state (coords + sigma)
    dummy_coords = system.flatten(shifted_coords)

    if sigma_prior_type == "truncnorm":
        z = jax.random.truncated_normal(jax.random.PRNGKey(0), lower=trunc_a, upper=trunc_b, shape=(3,))
        dummy_sigma = jnp.array([
            sigma_mu['AA'] + sigma_sd['AA'] * z[0],
            sigma_mu['AB'] + sigma_sd['AB'] * z[1],
            sigma_mu['BC'] + sigma_sd['BC'] * z[2],
        ])
    elif sigma_prior_type == "invgamma":
        key = jax.random.PRNGKey(0)
        g_aa = jax.random.gamma(key, invgamma_alpha['AA'])
        g_ab = jax.random.gamma(key, invgamma_alpha['AB'])
        g_bc = jax.random.gamma(key, invgamma_alpha['BC'])
        dummy_sigma = jnp.array([
            invgamma_beta['AA'] / g_aa,
            invgamma_beta['AB'] / g_ab,
            invgamma_beta['BC'] / g_bc,
        ])
    elif sigma_prior_type == "halfcauchy":
        key = jax.random.PRNGKey(0)
        c = jax.random.cauchy(key, (3,))
        dummy_sigma = jnp.array([
            jnp.abs(c[0]) * halfcauchy_scale['AA'],
            jnp.abs(c[1]) * halfcauchy_scale['AB'],
            jnp.abs(c[2]) * halfcauchy_scale['BC'],
        ])
    else:
        raise ValueError(f"Unknown sigma prior: {sigma_prior_type}")

    dummy_state = jnp.concatenate([dummy_coords, dummy_sigma], axis=0)
    _ = log_prob_fn(dummy_state)
    jax.block_until_ready(_)
    
    timer.stop("5. JIT compilation (warmup)")
    
    # =========================================================================
    # 6. Initialize SMC particles
    # =========================================================================
    timer.start("6. Initialize SMC particles")
    
    n_particles = 50
    rng_key = jax.random.PRNGKey(9090)
    rng_key, init_key = jax.random.split(rng_key)
    
    flat_shifted = system.flatten(shifted_coords)

    # ---- NEW: sample sigma from chosen prior ----
    rng_key, sigma_key = jax.random.split(rng_key)

    if sigma_prior_type == "truncnorm":
        # sample standard truncnorm, then scale/shift per parameter
        z = jax.random.truncated_normal(sigma_key, lower=trunc_a, upper=trunc_b, shape=(n_particles, 3))
        sigma_samples = []
        for i, key in enumerate(sigma_keys):
            sigma_samples.append(sigma_mu[key] + sigma_sd[key] * z[:, i])
        sigma_samples = jnp.stack(sigma_samples, axis=1)

    elif sigma_prior_type == "invgamma":
        # sample x ~ InvGamma(alpha, beta) by x = beta / Gamma(alpha, 1)
        sigma_samples = []
        for i, key in enumerate(sigma_keys):
            a = invgamma_alpha[key]
            b = invgamma_beta[key]
            g = jax.random.gamma(sigma_key, a, shape=(n_particles,))
            sigma_samples.append(b / g)
        sigma_samples = jnp.stack(sigma_samples, axis=1)

    elif sigma_prior_type == "halfcauchy":
        # sample x = |Cauchy(0, s)|
        sigma_samples = []
        for i, key in enumerate(sigma_keys):
            s = halfcauchy_scale[key]
            c = jax.random.cauchy(sigma_key, shape=(n_particles,))
            sigma_samples.append(jnp.abs(c) * s)
        sigma_samples = jnp.stack(sigma_samples, axis=1)
    else:
        raise ValueError(f"Unknown sigma prior: {sigma_prior_type}")

    coord_samples = flat_shifted + jax.random.normal(init_key, (n_particles, n_dims)) * 10.0
    initial_positions = jnp.concatenate([coord_samples, sigma_samples], axis=1)

    init_scores = jax.vmap(log_prob_fn)(initial_positions)
    jax.block_until_ready(init_scores)
    print(f"\nInitial Score (mean): {jnp.mean(init_scores):.2f}")
    
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
        n_mcmc_steps=60,
        rmh_sigma=2.0,
        target_ess=0.5,
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
    
    timer.stop("8. Post-processing")
    
    # =========================================================================
    # 9. Save results
    # =========================================================================
    timer.start("9. Save results")
    
    if best_positions is not None and best_scores is not None:
        diffs = np.linalg.norm(np.diff(np.array(best_positions), axis=0), axis=1)
        print(f"Best-step motion: min={diffs.min():.4f}, max={diffs.max():.4f}, mean={diffs.mean():.4f}")

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
