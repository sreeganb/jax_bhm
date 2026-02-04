"""
Example script to run RMH (Random-walk Metropolis-Hastings) sampling using BlackJAX.

This script demonstrates:
1. Setting up a particle system
2. Defining a log probability function with nuisance parameters
3. Running RMH MCMC sampling over BOTH coordinates AND nuisance params
4. Saving the trajectory to HDF5 (and optionally RMF3)
"""
import numpy as np
import sys
import os
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.rmh import run_rmh_sampling
from io_utils.io_handlers import save_mcmc_to_hdf5
from scoring.log_priors import Priors


def main():
    print("=" * 60)
    print("RMH Sampling with BlackJAX (Joint Coords + Nuisance)")
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
    
    # =========================================================================
    # 2. Define nuisance parameters
    # =========================================================================
    # Nuisance parameter names and their order
    nuisance_keys = ['AA', 'AB', 'BC']
    n_nuisance = len(nuisance_keys)
    
    # Bounds for nuisance parameters (used for prior)
    nuisance_bounds = {
        'AA': (0.1, 5.0),
        'AB': (0.1, 5.0),
        'BC': (0.1, 5.0),
    }
    
    # Target distances
    target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
    
    # Prior hyperparameters for inverse gamma
    nuisance_prior_alpha = 3.0  # shape
    nuisance_prior_beta = 1.0   # scale
    
    # Total dimensions = coords + nuisance
    n_total_dims = n_coord_dims + n_nuisance
    
    print(f"\nSystem: {system.total_particles} particles")
    print(f"Coordinate dimensions: {n_coord_dims}")
    print(f"Nuisance parameters: {n_nuisance} ({nuisance_keys})")
    print(f"Total dimensions: {n_total_dims}")
    
    # =========================================================================
    # 3. Define joint log probability
    # =========================================================================
    @jax.jit
    def log_prob_fn(state):
        """
        Joint log probability over coordinates AND nuisance parameters.
        
        State layout: [coords (n_coord_dims), nuisance (n_nuisance)]
        """
        # Split state into coords and nuisance
        flat_coords = state[:n_coord_dims]
        nuisance_raw = state[n_coord_dims:]
        
        # --- Prior on coordinates: Uniform in box ---
        log_prior = jnp.sum(Priors.log_uniform_prior(
            flat_coords, lower_bound=-box_size, upper_bound=box_size
        ))
        
        # --- Prior on nuisance params: Inverse Gamma + bounds ---
        for i, key in enumerate(nuisance_keys):
            sigma_val = nuisance_raw[i]
            low, high = nuisance_bounds[key]
            
            # Enforce bounds (return -inf if outside)
            in_bounds = (sigma_val >= low) & (sigma_val <= high)
            
            # Inverse Gamma prior
            ig_logp = Priors.log_inverse_gamma_prior(
                sigma_val, nuisance_prior_alpha, nuisance_prior_beta
            )
            
            log_prior += jnp.where(in_bounds, ig_logp, -jnp.inf)
        
        # --- Likelihood ---
        # Convert nuisance array to dict for log_probability
        nuisance_dict = {key: nuisance_raw[i] for i, key in enumerate(nuisance_keys)}
        
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_dict,
            exclusion_weight=1.0,
            pair_weight=2.0,
            exvol_sigma=0.1
        )
        
        return log_prior + log_lik
    
    # =========================================================================
    # 4. Initialize state (coords + nuisance)
    # =========================================================================
    rng_key = jax.random.PRNGKey(123)
    
    # Initial coordinates
    initial_coords = system.flatten(coords)
    
    # Initial nuisance parameters (sample from prior or set manually)
    rng_key, init_key = jax.random.split(rng_key)
    initial_nuisance = jnp.array([
        np.random.uniform(*nuisance_bounds[key]) for key in nuisance_keys
    ])
    
    # Concatenate into single state vector
    initial_state = jnp.concatenate([initial_coords, initial_nuisance])
    
    print(f"\nInitial nuisance values:")
    for i, key in enumerate(nuisance_keys):
        print(f"  σ_{key}: {float(initial_nuisance[i]):.3f}")
    
    # =========================================================================
    # 5. Run RMH Sampling
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running RMH Sampling (Joint)...")
    print("-" * 60)
    
    rng_key, sample_key = jax.random.split(rng_key)
    
    n_steps = 500000
    # =========================================================================
    # 5. Run RMH Sampling with per-dimension sigma
    # =========================================================================
    
    # Create per-dimension sigma vector
    sigma_vec = jnp.concatenate([
        jnp.ones(n_coord_dims) * 0.4,    # Coords: σ=0.4
        jnp.ones(n_nuisance) * 0.05        # Nuisance: σ=0.05
    ])
    
    positions, log_probs, acceptance_rate = run_rmh_sampling(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_position=initial_state,
        n_steps=n_steps,
        sigma=sigma_vec,  # Pass vector instead of scalar
        burnin=10000,
        thin=100,
        save_interval=1000,
        verbose=True,
    )    
    # =========================================================================
    # 6. Extract and report results
    # =========================================================================
    print("\n" + "-" * 60)
    print("Results Summary")
    print("-" * 60)
    
    # Split saved positions into coords and nuisance
    saved_coords = positions[:, :n_coord_dims]
    saved_nuisance = positions[:, n_coord_dims:]
    
    best_idx = np.argmax(log_probs)
    print(f"Saved samples: {len(log_probs)}")
    print(f"Acceptance rate: {acceptance_rate:.1%}")
    print(f"Best log probability: {log_probs[best_idx]:.2f}")
    print(f"Final log probability: {log_probs[-1]:.2f}")
    print(f"Mean log probability: {np.mean(log_probs):.2f}")
    
    # Nuisance parameter statistics
    print(f"\nNuisance parameter posteriors:")
    print(f"{'Param':<10} {'Mean':>10} {'Std':>10} {'Best':>10}")
    print("-" * 42)
    for i, key in enumerate(nuisance_keys):
        vals = saved_nuisance[:, i]
        print(f"σ_{key:<7} {np.mean(vals):>10.3f} {np.std(vals):>10.3f} {saved_nuisance[best_idx, i]:>10.3f}")
    
    # Coordinate motion
    if len(saved_coords) > 1:
        diffs = np.linalg.norm(np.diff(saved_coords, axis=0), axis=1)
        print(f"\nCoord motion: min={diffs.min():.2f}, max={diffs.max():.2f}, mean={diffs.mean():.2f}")
    
    # =========================================================================
    # 7. Save trajectory
    # =========================================================================
    output_file = output_dir / "rmh_trajectory.h5"
    
    # Save only coordinates for visualization
    save_mcmc_to_hdf5(
        positions=saved_coords,  # Only coords for RMF3
        log_probs=log_probs,
        acceptance_rate=acceptance_rate,
        filename=str(output_file),
        system_template=system,
        params={
            'method': 'BlackJAX_RMH_Joint',
            'n_steps': n_steps,
            'sigma_coords': float(sigma_vec[0]),      # Fixed
            'sigma_nuisance': float(sigma_vec[-1]),   # Fixed
            'nuisance_keys': nuisance_keys,
            'nuisance_posterior_mean': {key: float(np.mean(saved_nuisance[:, i])) 
                                        for i, key in enumerate(nuisance_keys)},
            'nuisance_posterior_std': {key: float(np.std(saved_nuisance[:, i])) 
                                       for i, key in enumerate(nuisance_keys)},
        },
        convert_to_rmf3=True,
        color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
    )    
    # Also save nuisance samples separately
    nuisance_file = output_dir / "nuisance_samples.npz"
    np.savez(
        nuisance_file,
        samples=saved_nuisance,
        keys=nuisance_keys,
        log_probs=log_probs,
    )
    
    print(f"\nTrajectory saved to: {output_file}")
    print(f"Nuisance samples saved to: {nuisance_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()