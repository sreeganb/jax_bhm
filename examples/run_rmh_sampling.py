"""
Example script to run RMH (Random-walk Metropolis-Hastings) sampling using BlackJAX.

This script demonstrates:
1. Setting up a particle system
2. Defining a log probability function
3. Running RMH MCMC sampling
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
from sampling.rmh import run_rmh_sampling, run_parallel_rmh, run_annealed_rmh
from io_utils.io_handlers import save_mcmc_to_hdf5


def main():
    print("=" * 60)
    print("RMH Sampling with BlackJAX")
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
    
    coords = ParticleSystem(types_config, {}, ideal_coords).get_random_coords(
        jax.random.PRNGKey(42), box_size=[500.0, 500.0, 500.0]
    )
    
    system = ParticleSystem(types_config, {}, coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    # =========================================================================
    # 2. Define scoring function / log probability
    # =========================================================================
    target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
    nuisance_params = {'AA': 1.5, 'AB': 1.2, 'BC': 1.0}  # Softer constraints
    
    box_size = 500.0
    
    def log_prob_fn(flat_coords):
        """Combined prior + likelihood with softer penalties."""
        coords = flat_coords.reshape(-1, 3)
        
        # Soft box prior (Gaussian penalty for out-of-box)
        out_low = jnp.sum(jnp.minimum(coords, 0) ** 2)
        out_high = jnp.sum(jnp.maximum(coords - box_size, 0) ** 2)
        log_prior = -0.01 * (out_low + out_high)
        
        # Likelihood from scoring with softer weights
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0,  # Softer exclusion
            pair_weight=1.0, 
            exvol_sigma=0.1  # Larger sigma = softer
        )
        
        return log_prior + log_lik
    
    # =========================================================================
    # 3. Initialize starting position
    # =========================================================================
    initial_position = system.flatten(coords)  
    rng_key = jax.random.PRNGKey(123)
      
    # =========================================================================
    # 4. Run Annealed RMH Sampling (uses temperature schedule for exploration)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running Annealed RMH Sampling...")
    print("-" * 60)
    
    rng_key, sample_key = jax.random.split(rng_key)
    
    # Annealing parameters
    n_steps = 50000
    sigma = 1.0          # Larger moves
    temp_start = 50.0     # Start hot (flat landscape)
    temp_end = 1.0        # Cool to true distribution
    save_every = 10       # Save every 10 steps
    
    positions, log_probs, acceptance_rate = run_annealed_rmh(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_position=initial_position,
        n_steps=n_steps,
        sigma=sigma,
        temp_start=temp_start,
        temp_end=temp_end,
        save_every=save_every,
        verbose=True,
    )
    
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
    
    # Motion between frames
    if len(positions) > 1:
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        print(f"\nFrame motion: min={diffs.min():.2f}, max={diffs.max():.2f}, mean={diffs.mean():.2f}")
    
    # =========================================================================
    # 6. Save trajectory
    # =========================================================================
    output_file = output_dir / "rmh_trajectory.h5"
    
    save_mcmc_to_hdf5(
        positions=positions,
        log_probs=log_probs,
        acceptance_rate=acceptance_rate,
        filename=str(output_file),
        system_template=system,
        params={
            'method': 'BlackJAX_Annealed_RMH',
            'n_steps': n_steps,
            'sigma': sigma,
            'temp_start': temp_start,
            'temp_end': temp_end,
        },
        convert_to_rmf3=True,
        color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
    )
    
    print(f"\nTrajectory saved to: {output_file}")
    print("=" * 60)


def run_parallel_example():
    """
    Alternative: Run multiple parallel chains and save final states.
    Useful for exploring the landscape with multiple starting points.
    """
    print("=" * 60)
    print("Parallel RMH Chains with BlackJAX")
    print("=" * 60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Setup system (same as main)
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }
    
    ideal_coords = get_ideal_coords()
    system = ParticleSystem(types_config, {}, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    # Scoring
    target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
    nuisance_params = {'AA': 2.0, 'AB': 1.5, 'BC': 1.0}
    box_size = 500.0
    
    def log_prob_fn(flat_coords):
        coords = flat_coords.reshape(-1, 3)
        in_box = jnp.all((coords >= 0) & (coords <= box_size))
        log_prior = jnp.where(in_box, 0.0, -jnp.inf)
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=0.1, pair_weight=0.1, exvol_sigma=0.1
        )
        return log_prior + log_lik
    
    # Initialize multiple chains
    rng_key = jax.random.PRNGKey(123)
    rng_key, init_key = jax.random.split(rng_key)
    
    n_chains = 50
    initial_positions = jax.random.uniform(init_key, (n_chains, n_dims)) * box_size
    
    print(f"\nRunning {n_chains} parallel chains...")
    
    rng_key, sample_key = jax.random.split(rng_key)
    
    final_positions, final_log_probs, mean_acc = run_parallel_rmh(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_positions,
        n_steps=20000,
        sigma=2.0,
        verbose=True,
    )
    
    # Results
    print("\n" + "-" * 60)
    print("Parallel Chains Results")
    print("-" * 60)
    
    best_idx = np.argmax(final_log_probs)
    print(f"Number of chains: {n_chains}")
    print(f"Mean acceptance: {mean_acc:.1%}")
    print(f"Best final log prob: {final_log_probs[best_idx]:.2f}")
    print(f"Mean final log prob: {np.mean(final_log_probs):.2f}")
    print(f"Std final log prob: {np.std(final_log_probs):.2f}")
    
    # Save final states as a "trajectory" (each chain = one frame)
    output_file = output_dir / "rmh_parallel_final.h5"
    
    save_mcmc_to_hdf5(
        positions=final_positions,
        log_probs=final_log_probs,
        acceptance_rate=mean_acc,
        filename=str(output_file),
        system_template=system,
        params={
            'method': 'BlackJAX_RMH_Parallel',
            'n_chains': n_chains,
            'n_steps': 2000,
            'sigma': 2.0,
        },
        convert_to_rmf3=True,
        color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
    )
    
    print(f"\nSaved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RMH sampling with BlackJAX")
    parser.add_argument("--parallel", action="store_true", 
                        help="Run parallel chains instead of single chain")
    args = parser.parse_args()
    
    if args.parallel:
        run_parallel_example()
    else:
        main()
