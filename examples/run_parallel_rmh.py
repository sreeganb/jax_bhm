"""
Run multiple independent RMH chains in parallel and save each trajectory.
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
from sampling.rmh import create_rmh_kernel
from io_utils.io_handlers import save_mcmc_to_hdf5
from scoring.log_priors import Priors
import time


def run_parallel_rmh_with_trajectories(
    rng_key: jax.Array,
    log_prob_fn,
    initial_positions: jax.Array,
    n_steps: int = 1000,
    sigma: float = 1.0,
    save_every: int = 100,
    verbose: bool = True,
):
    """
    Run multiple independent RMH chains in parallel, saving full trajectories.
    
    Returns:
        trajectories: (n_chains, n_saved, n_dims) - positions for each chain
        log_probs: (n_chains, n_saved) - log probs for each chain
        acceptance_rates: (n_chains,) - acceptance rate per chain
    """
    n_chains, n_dims = initial_positions.shape
    n_saved = n_steps // save_every
    
    # Create kernel
    kernel = create_rmh_kernel(log_prob_fn, sigma)
    
    # Initialize all chains
    initial_positions = jnp.asarray(initial_positions)
    states = jax.vmap(kernel.init)(initial_positions)
    
    # JIT step function
    @jax.jit
    def parallel_step(states, keys):
        def step_fn(state, key):
            return kernel.step(key, state)
        return jax.vmap(step_fn)(states, keys)
    
    if verbose:
        print(f"Running {n_chains} parallel RMH chains: {n_steps} steps each, σ={sigma}")
        print(f"Saving every {save_every} steps → {n_saved} samples per chain")
    
    # Storage
    trajectories = np.zeros((n_chains, n_saved, n_dims), dtype=np.float32)
    log_probs = np.zeros((n_chains, n_saved), dtype=np.float32)
    accepts = np.zeros(n_chains, dtype=np.float32)
    
    t0 = time.time()
    print_every = max(1, n_steps // 10)
    save_idx = 0
    
    for i in range(n_steps):
        rng_key, step_key = jax.random.split(rng_key)
        chain_keys = jax.random.split(step_key, n_chains)
        states, infos = parallel_step(states, chain_keys)
        accepts += np.array(infos.is_accepted)
        
        # Save at intervals
        if (i + 1) % save_every == 0:
            trajectories[:, save_idx, :] = np.array(states.position)
            log_probs[:, save_idx] = np.array(states.logdensity)
            save_idx += 1
        
        if verbose and (i + 1) % print_every == 0:
            mean_logp = float(jnp.mean(states.logdensity))
            mean_acc = float(np.mean(accepts)) / (i + 1)
            print(f"  Step {i+1:6d}/{n_steps} | Mean LogProb: {mean_logp:10.2f} | Accept: {mean_acc:.2%}")
    
    dt = time.time() - t0
    acceptance_rates = accepts / n_steps
    
    if verbose:
        print(f"Completed in {dt:.2f}s ({n_steps * n_chains / dt:.0f} total steps/s)")
        print(f"Mean acceptance rate: {float(np.mean(acceptance_rates)):.2%}")
    
    return trajectories, log_probs, acceptance_rates


def main():
    print("=" * 60)
    print("Parallel RMH Sampling with BlackJAX")
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
    
    system = ParticleSystem(types_config, {}, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    # =========================================================================
    # 2. Define scoring function
    # =========================================================================
    target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
    nuisance_params = {'AA': 1.5, 'AB': 1.2, 'BC': 1.0}
    
    @jax.jit
    def log_prob_fn(flat_coords):
        log_prior = jnp.sum(Priors.log_uniform_prior(
            flat_coords, lower_bound=-box_size, upper_bound=box_size
        ))
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=0.5,
            pair_weight=1.0,
            exvol_sigma=0.5
        )
        return log_prior + log_lik
    
    # =========================================================================
    # 3. Initialize multiple chains with different random starts
    # =========================================================================
    n_chains = 8
    rng_key = jax.random.PRNGKey(42)
    
    # Generate different initial positions for each chain
    initial_positions = []
    for i in range(n_chains):
        rng_key, init_key = jax.random.split(rng_key)
        coords = system.get_random_coords(
            init_key, box_size=[box_size, box_size, box_size], center_at_origin=True
        )
        temp_system = ParticleSystem(types_config, {}, coords)
        initial_positions.append(temp_system.flatten(coords))
    
    initial_positions = jnp.stack(initial_positions)
    print(f"Initialized {n_chains} chains")
    
    # =========================================================================
    # 4. Run parallel sampling
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running Parallel RMH Sampling...")
    print("-" * 60)
    
    rng_key, sample_key = jax.random.split(rng_key)
    
    n_steps = 100000
    sigma = 0.5
    save_every = 100
    
    trajectories, log_probs, acceptance_rates = run_parallel_rmh_with_trajectories(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_positions,
        n_steps=n_steps,
        sigma=sigma,
        save_every=save_every,
        verbose=True,
    )
    
    # =========================================================================
    # 5. Report results
    # =========================================================================
    print("\n" + "-" * 60)
    print("Results Summary")
    print("-" * 60)
    
    for i in range(n_chains):
        best_idx = np.argmax(log_probs[i])
        print(f"Chain {i}: Accept={acceptance_rates[i]:.2%}, "
              f"Best={log_probs[i, best_idx]:.2f}, "
              f"Final={log_probs[i, -1]:.2f}")
    
    # Find overall best
    best_chain = np.unravel_index(np.argmax(log_probs), log_probs.shape)
    print(f"\nOverall best: Chain {best_chain[0]}, Sample {best_chain[1]}, "
          f"LogProb={log_probs[best_chain]:.2f}")
    
    # =========================================================================
    # 6. Save each chain's trajectory
    # =========================================================================
    print("\n" + "-" * 60)
    print("Saving Trajectories...")
    print("-" * 60)
    
    for i in range(n_chains):
        output_file = output_dir / f"rmh_chain_{i:02d}.h5"
        
        save_mcmc_to_hdf5(
            positions=trajectories[i],
            log_probs=log_probs[i],
            acceptance_rate=float(acceptance_rates[i]),
            filename=str(output_file),
            system_template=system,
            params={
                'method': 'BlackJAX_RMH_Parallel',
                'chain_id': i,
                'n_chains': n_chains,
                'n_steps': n_steps,
                'sigma': sigma,
            },
            convert_to_rmf3=True,
            color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
        )
        print(f"  Saved chain {i} to {output_file}")
    
    # Also save combined best samples from all chains
    # Take best sample from each chain
    best_positions = np.array([trajectories[i, np.argmax(log_probs[i])] for i in range(n_chains)])
    best_log_probs = np.array([np.max(log_probs[i]) for i in range(n_chains)])
    
    combined_file = output_dir / "rmh_best_from_all_chains.h5"
    save_mcmc_to_hdf5(
        positions=best_positions,
        log_probs=best_log_probs,
        acceptance_rate=float(np.mean(acceptance_rates)),
        filename=str(combined_file),
        system_template=system,
        params={
            'method': 'BlackJAX_RMH_Parallel_Best',
            'n_chains': n_chains,
            'description': 'Best sample from each chain',
        },
        convert_to_rmf3=True,
        color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
    )
    print(f"\nSaved best samples to {combined_file}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()