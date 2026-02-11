"""
Run script for HMC sampler 
1) Define particle system
2) Define scoring function / log probability
3) Initialize starting position
4) Run HMC sampling
5) Report results
6) Save trajectory

NOTE: In this code, the sigmas are going to be sampled and will be initialized 
accordingly, we will provide a prior for them and they will be sampled as part 
of the HMC sampling.
"""
import numpy as np
import sys
import os
from pathlib import Path

#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax

print("JAX default backend:", jax.default_backend())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.hmc import run_hmc_sampling, run_nuts_sampling, estimate_step_size
from io_utils.io_handlers import save_mcmc_to_hdf5
from scoring.log_priors import Priors

def main():
    print("=" * 60)
    print("HMC Sampling with BlackJAX")
    print("=" * 60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # 1. Define particle system
    # =========================================================================
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }
    
    ideal_coords = get_ideal_coords()
    
    box_size = 500.0
    
    coords = ParticleSystem(types_config, {}, ideal_coords).get_random_coords(
        jax.random.PRNGKey(2387), box_size=[box_size, box_size, box_size], center_at_origin=True
    )
    
    system = ParticleSystem(types_config, {}, coords)
    flat_radii = system.get_flat_radii()
    n_coord_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_coord_dims} coordinate dimensions")
    
    # =========================================================================
    # 2. Define nuisance parameters
    # =========================================================================
    # Nuisance parameter names and their order
    nuisance_keys = ['AA', 'AB', 'BC']
    n_nuisance = len(nuisance_keys)
    
    # Bounds for nuisance parameters (used for prior)
    nuisance_bounds = {
        'AA': (0.1, 4.0),
        'AB': (0.1, 4.0),
        'BC': (0.1, 4.0),
    }
    
    # Target distances
    target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
    
    # Prior hyperparameters for inverse gamma
    nuisance_prior_alpha = 3.0  # shape
    nuisance_prior_beta = 0.5   # scale
    
    # Total dimensions = coords + nuisance
    n_total_dims = n_coord_dims + n_nuisance
    
    print(f"\nSystem: {system.total_particles} particles")
    print(f"Coordinate dimensions: {n_coord_dims}")
    print(f"Nuisance parameters: {n_nuisance} ({nuisance_keys})")
    print(f"Total dimensions: {n_total_dims}")
    
    # =========================================================================
    # 3. Define scoring function / log probability
    # =========================================================================
    # NOTE: Use a bounded transform for nuisance parameters to avoid hard rejections.
    # State layout: [coords (n_coord_dims), nuisance_u (n_nuisance)]
    # Transform: sigma = low + (high-low) * sigmoid(u)
    
    @jax.jit
    def log_prob_fn(state):
        """
        Joint log probability over coordinates AND nuisance parameters.
        
        State layout: [coords (n_coord_dims), nuisance_u (n_nuisance)]
        where nuisance_u is unconstrained and mapped to bounds with sigmoid.
        """
        # Split state into coords and nuisance (unconstrained)
        flat_coords = state[:n_coord_dims]
        nuisance_u = state[n_coord_dims:]
        
        # --- Prior on coordinates: Uniform in box ---
        log_prior = jnp.sum(Priors.log_uniform_prior(
            flat_coords, lower_bound=-box_size, upper_bound=box_size
        ))
        
        # --- Prior on nuisance params: Inverse Gamma with bounded transform ---
        nuisance_vals = []
        for i, key in enumerate(nuisance_keys):
            u = nuisance_u[i]
            low, high = nuisance_bounds[key]
            scale = high - low
            sig = jax.nn.sigmoid(u)
            sigma_val = low + scale * sig
            nuisance_vals.append(sigma_val)
            
            # Inverse Gamma prior on sigma
            ig_logp = Priors.log_inverse_gamma_prior(
                sigma_val, nuisance_prior_alpha, nuisance_prior_beta
            )
            
            # Jacobian for bounded transform: log |d sigma / d u|
            # d sigma / d u = scale * sigmoid(u) * (1 - sigmoid(u))
            log_sig = -jax.nn.softplus(-u)
            log_one_minus_sig = -jax.nn.softplus(u)
            log_jac = jnp.log(scale) + log_sig + log_one_minus_sig
            
            log_prior += ig_logp + log_jac
        
        # --- Likelihood ---
        nuisance_dict = {key: nuisance_vals[i] for i, key in enumerate(nuisance_keys)}
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_dict,
            exclusion_weight=1.0,
            pair_weight=2.0,
            exvol_sigma=0.1
        )
        
        return log_prior + log_lik
    
    # =========================================================================
    # 4. Initialize state (coords + nuisance_u)
    # =========================================================================
    rng_key = jax.random.PRNGKey(123)
    
    # Initial coordinates
    initial_coords = system.flatten(coords)
    
    # Initial nuisance parameters (bounded transform)
    rng_key, init_key = jax.random.split(rng_key)
    initial_nuisance_values = jnp.array([
        np.random.uniform(*nuisance_bounds[key]) for key in nuisance_keys
    ])
    
    # Invert transform: u = logit((sigma - low) / (high - low))
    initial_nuisance_u = []
    for i, key in enumerate(nuisance_keys):
        low, high = nuisance_bounds[key]
        scale = high - low
        sigma = float(initial_nuisance_values[i])
        p = (sigma - low) / scale
        p = np.clip(p, 1e-6, 1 - 1e-6)
        u = np.log(p / (1 - p))
        initial_nuisance_u.append(u)
    initial_nuisance_u = jnp.array(initial_nuisance_u)
    
    # Concatenate into single state vector (coords + nuisance_u)
    initial_state = jnp.concatenate([initial_coords, initial_nuisance_u])
    
    print(f"\nInitial nuisance values (bounded sigmoid transform):")
    for i, key in enumerate(nuisance_keys):
        print(f"  σ_{key}: {float(initial_nuisance_values[i]):.3f} (u: {float(initial_nuisance_u[i]):.3f})")
    
    # =========================================================================
    # 4. Run HMC Sampling
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running HMC Sampling...")
    print("-" * 60)
    
    rng_key, sample_key = jax.random.split(rng_key)
    
    # HMC parameters for multi-scale problem
    n_steps = 100000
    
    # Inverse mass matrix (diagonal)
    # Coordinates were previously over-damped (inverse mass too small), leading to tiny moves.
    # Set inverse mass back to 1.0 to allow reasonable momentum for coordinates.
    coord_inv_mass_val = 1.0
    coord_mass = jnp.full(n_coord_dims, coord_inv_mass_val)
    nuisance_mass = jnp.ones(n_nuisance) * 0.05
    inv_mass = jnp.concatenate([coord_mass, nuisance_mass])
    
    # Fewer integration steps to keep trajectory length moderate when step size is larger.
    num_int_steps = 20
    
    # ------------------------------------------------------------
    # Phase 1: Warmup / Relaxation
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1: Warmup / Relaxation (600000 steps)")
    print("Goal: Relax coordinates and adapt step size")
    print("=" * 60)
    
    warmup_steps = 300000
    
    positions_warmup, log_probs_warmup, acc_warmup = run_hmc_sampling(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_position=initial_state,
        n_steps=warmup_steps,
        step_size="auto",
        inverse_mass_matrix=inv_mass,
        num_integration_steps=num_int_steps,
        burnin=0,
        thin=100,
        save_interval=200,
        verbose=True,
    )
    
    print(f"Warmup complete. Final LogProb: {log_probs_warmup[-1]:.2f}")
    
    # Use the last position from warmup as the start for production
    start_pos_prod = positions_warmup[-1]
    
    # Re-estimate step size given the relaxed state
    print("\nRe-estimating step size for production run...")
    step_size_prod = estimate_step_size(log_prob_fn, start_pos_prod, inverse_mass_matrix=inv_mass)
    
    # Aggressively increase step size to improve exploration
    # The auto-estimator is conservative (aims for ~99% stability on difficult start).
    # Since we are now relaxed and acceptance was >95%, we can increase step size.
    scaling_factor = 80.0  # boost more to get acceptance into 60-80% range
    step_size_prod = step_size_prod * scaling_factor
    print(f"New estimated step size (scaled by {scaling_factor}x): {step_size_prod}")
    
    # ------------------------------------------------------------
    # Phase 2: Production Sampling
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Production Sampling")
    print("=" * 60)
    
    rng_key, sample_key = jax.random.split(rng_key)
    n_steps_prod = 600000
    
    positions_prod, log_probs_prod, acc_prod = run_hmc_sampling(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_position=start_pos_prod,
        n_steps=n_steps_prod,
        step_size=step_size_prod,
        inverse_mass_matrix=inv_mass,
        num_integration_steps=num_int_steps,
        burnin=0,       # Already burned in
        thin=500,
        save_interval=1000,
        verbose=True,
    )
    
# Combined results
    positions = positions_prod
    log_probs = log_probs_prod
    acceptance_rate = acc_prod
    step_size = step_size_prod

    # =========================================================================
    # 5. Report results
    # =========================================================================
    print("\n" + "-" * 60)
    print("Results Summary")
    print("-" * 60)
    
    best_idx = np.argmax(log_probs)
    print(f"Saved samples: {len(log_probs)}")
    print(f"Acceptance rate: {acceptance_rate:.1%}")
    print(f"Best log probability: {log_probs[best_idx]:.2f}")
    print(f"Final log probability: {log_probs[-1]:.2f}")
    print(f"Mean log probability: {np.mean(log_probs):.2f}")
    
    # Report nuisance parameter estimates (convert from bounded sigmoid transform)
    print(f"\nNuisance parameter estimates (last 50% of samples, bounded transform):")
    n_samples = len(positions)
    posterior_samples = positions[n_samples//2:]  # Use second half
    for i, key in enumerate(nuisance_keys):
        u_samples = posterior_samples[:, n_coord_dims + i]
        low, high = nuisance_bounds[key]
        scale = high - low
        nuisance_samples = low + scale * (1.0 / (1.0 + np.exp(-u_samples)))
        mean_val = np.mean(nuisance_samples)
        std_val = np.std(nuisance_samples)
        print(f"  σ_{key}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Motion between frames
    if len(positions) > 1:
        diffs = np.linalg.norm(np.diff(positions[:, :n_coord_dims], axis=0), axis=1)
        print(f"\nCoordinate motion: min={diffs.min():.6f}, max={diffs.max():.6f}, mean={diffs.mean():.6f}")
        
    # Check if we moved at all from start
    total_disp = np.linalg.norm(positions[-1, :n_coord_dims] - positions[0, :n_coord_dims])
    print(f"Total displacement from start of production: {total_disp:.6f}")
    
    # =========================================================================
    # 6. Save trajectory
    # =========================================================================
    output_file = output_dir / "hmc_trajectory.h5"
    
    save_mcmc_to_hdf5(
        positions=positions,
        log_probs=log_probs,
        acceptance_rate=acceptance_rate,
        filename=str(output_file),
        system_template=system,
        params={
            'method': 'BlackJAX_HMC',
            'n_steps': n_steps,
            'step_size': step_size,
            'num_integration_steps': num_int_steps,
        },
        convert_to_rmf3=True,
        color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
    )
    
    print(f"\nTrajectory saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HMC sampling with BlackJAX")
    args = parser.parse_args()
    
    main()
