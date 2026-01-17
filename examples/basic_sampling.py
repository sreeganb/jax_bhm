"""
Basic example demonstrating protein structure sampling with JAX-BHM.

This example shows a simpler case without excluded volume to better
demonstrate the MCMC sampling behavior.
"""

import jax
import jax.numpy as jnp
from jax_bhm.structure import create_linear_chain
from jax_bhm.scoring import create_energy_fn
from jax_bhm.sampling import sample_structure, calculate_sampling_statistics, get_best_structure


def main():
    """Run a basic protein structure sampling demonstration."""
    
    print("=" * 70)
    print("JAX-BHM: Basic Protein Structure Sampling Example")
    print("=" * 70)
    
    # Set random seed
    key = jax.random.PRNGKey(42)
    
    # 1. Create a small linear chain
    print("\n1. Creating initial structure...")
    n_residues = 5
    initial_structure = create_linear_chain(n_residues, spacing=3.8)
    print(f"   Created linear chain with {n_residues} CA atoms")
    print(f"   Initial positions:")
    for i, pos in enumerate(initial_structure.positions):
        print(f"      Atom {i}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # 2. Define a simple distance restraint
    print("\n2. Defining distance restraints...")
    # Initial distance between atoms 0 and 4 is about 15.2 Å
    # Let's constrain it to be 10.0 Å
    distance_restraints = jnp.array([
        [0, 4, 10.0, 1.0],   # Bring ends closer together
    ])
    print(f"   Restraint: atoms 0-4 should be 10.0 Å apart")
    initial_dist = jnp.linalg.norm(
        initial_structure.positions[0] - initial_structure.positions[4]
    )
    print(f"   Initial distance: {initial_dist:.2f} Å")
    
    # 3. Create energy function (no excluded volume for simplicity)
    print("\n3. Creating energy function...")
    energy_fn = create_energy_fn(
        distance_restraints=distance_restraints,
        use_excluded_volume=False,  # Disable for simpler sampling
        distance_k=1.0  # Weaker force constant
    )
    
    initial_energy = energy_fn(initial_structure.positions)
    print(f"   Initial energy: {initial_energy:.4f}")
    
    # 4. Run MCMC sampling
    print("\n4. Running MCMC sampling...")
    print("   Algorithm: NUTS")
    print("   Samples: 500 (with 200 warmup)")
    print("   Temperature: 10.0")
    
    samples, sampling_info = sample_structure(
        initial_structure=initial_structure,
        energy_fn=energy_fn,
        n_samples=500,
        n_warmup=200,
        step_size=0.5,
        temperature=10.0,
        algorithm="nuts",
        key=key
    )
    
    print(f"\n   Sampling complete!")
    print(f"   Generated {samples.shape[0]} samples")
    
    # 5. Analyze results
    print("\n5. Analyzing results...")
    stats = calculate_sampling_statistics(samples, energy_fn)
    
    print(f"   Mean energy: {stats['mean_energy']:.4f}")
    print(f"   Std energy: {stats['std_energy']:.4f}")
    print(f"   Min energy: {stats['min_energy']:.4f}")
    print(f"   Energy improved: {initial_energy > stats['min_energy']}")
    
    # 6. Best structure
    print("\n6. Checking best structure...")
    best_positions, best_energy = get_best_structure(samples, energy_fn)
    best_dist = jnp.linalg.norm(best_positions[0] - best_positions[4])
    
    print(f"   Best energy: {best_energy:.4f}")
    print(f"   Best distance (atoms 0-4): {best_dist:.2f} Å")
    print(f"   Target distance: 10.0 Å")
    print(f"   Error: {abs(best_dist - 10.0):.2f} Å")
    
    # 7. Show sampling statistics
    print("\n7. Sampling diversity...")
    mean_structure = stats['mean_structure']
    mean_dist = jnp.linalg.norm(mean_structure[0] - mean_structure[4])
    print(f"   Mean distance (atoms 0-4): {mean_dist:.2f} Å")
    print(f"   Std per atom: {stats['mean_std_per_atom']:.3f} Å")
    print(f"   (Higher std indicates more exploration)")
    
    # 8. Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Initial structure had atoms 0-4 at {initial_dist:.2f} Å")
    print(f"Target distance: 10.0 Å")
    print(f"Best sampled distance: {best_dist:.2f} Å")
    print(f"Energy reduction: {initial_energy - best_energy:.4f}")
    print("\nThe MCMC sampler successfully explored conformations that")
    print("better satisfy the distance restraint!")
    print("=" * 70)


if __name__ == "__main__":
    main()
