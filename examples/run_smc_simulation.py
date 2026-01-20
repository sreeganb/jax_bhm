"""
Example script to run SMC simulation using BlackJAX.
"""
import jax.numpy as jnp
import jax
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.smc import run_tempered_smc, get_smc_samples, get_best_sample
from io_utils.io_handlers import save_mcmc_to_hdf5


def main():
    print("Setting up system...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Define particle system
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }
    
    ideal_coords = get_ideal_coords()
    
    # 2. Create system template
    system = ParticleSystem(types_config, {}, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    print(f"System: {system.total_particles} particles, {n_dims} dimensions")
    
    # 3. Restraints
    target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
    nuisance_params = {'AA': 5.0, 'AB': 5.0, 'BC': 5.0}
    
    # 4. Define prior and likelihood separately (required by SMC)
    box_size = 300.0
    
    def log_prior_fn(flat_coords):
        """Uniform prior in box."""
        coords = flat_coords.reshape(-1, 3)
        in_box = jnp.all((coords >= 0) & (coords <= box_size))
        return jnp.where(in_box, 0.0, -jnp.inf)
    
    def log_likelihood_fn(flat_coords):
        """Likelihood from scoring function."""
        return log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=0.1, pair_weight=0.1, exvol_sigma=0.1
        )
    
    def log_prob_fn(flat_coords):
        """Combined log probability."""
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)
    
    # 5. Initialize particles (n_particles, n_dims)
    n_particles = 100
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    initial_positions = jax.random.uniform(init_key, (n_particles, n_dims)) * box_size
    
    # Check initial scores
    init_scores = jax.vmap(log_prob_fn)(initial_positions)
    print(f"Initial Score (mean): {jnp.mean(init_scores):.2f}")
    
    # 6. Run SMC
    rng_key, smc_key = jax.random.split(rng_key)
    # Use smaller RMH sigma and more MCMC steps to encourage movement
    final_state, info_history, best_positions, best_scores = run_tempered_smc(
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_positions,
        rng_key=smc_key,
        n_mcmc_steps=50,
        rmh_sigma=0.5,
        target_ess=0.7,
        record_best=True,
    )
    
    # 7. Get results
    final_positions = get_smc_samples(final_state)
    best_pos, best_score = get_best_sample(final_state, log_prob_fn)
    
    final_scores = jax.vmap(log_prob_fn)(final_positions)
    print(f"\nFinal Score (mean): {jnp.mean(final_scores):.2f}")
    print(f"Best Score: {best_score:.2f}")
    
    # 8. Save
    # Save the best sample at each tempering step to visualize a true trajectory.
    if best_positions is not None and best_scores is not None:
        # Report motion between steps
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
    else:
        output_file = output_dir / "smc_samples.h5"
        save_mcmc_to_hdf5(
            np.array(final_positions),
            np.array(final_scores),
            1.0,
            str(output_file),
            system,
            params={'method': 'BlackJAX_SMC', 'trajectory': 'final_population'},
            convert_to_rmf3=True,
            color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
        )
        print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
