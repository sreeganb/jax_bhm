"""
Simple example demonstrating protein structure sampling with JAX-BHM.

This example shows how to:
1. Create a simple protein structure
2. Define distance restraints
3. Sample conformations using BlackJax
4. Analyze the results
"""

import jax
import jax.numpy as jnp
from jax_bhm.structure import create_linear_chain, ProteinStructure
from jax_bhm.scoring import create_energy_fn
from jax_bhm.sampling import sample_structure, calculate_sampling_statistics, get_best_structure


def main():
    """Run a simple protein structure sampling demonstration."""
    
    print("=" * 70)
    print("JAX-BHM: Protein Structure Sampling Demonstration")
    print("=" * 70)
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # 1. Create a linear chain of 10 residues
    print("\n1. Creating initial structure...")
    n_residues = 10
    initial_structure = create_linear_chain(n_residues, spacing=3.8)
    print(f"   Created linear chain with {n_residues} CA atoms")
    print(f"   Initial structure shape: {initial_structure.positions.shape}")
    
    # 2. Define distance restraints
    # Let's add restraints to encourage the chain to fold
    print("\n2. Defining distance restraints...")
    distance_restraints = jnp.array([
        # [atom_i, atom_j, target_distance, weight]
        [0, 9, 10.0, 1.0],   # Bring ends closer together
        [2, 7, 8.0, 1.0],    # Create a kink in the middle
        [1, 5, 12.0, 0.5],   # Weaker restraint
    ])
    print(f"   Defined {distance_restraints.shape[0]} distance restraints")
    for i, restraint in enumerate(distance_restraints):
        print(f"      Restraint {i+1}: atoms {int(restraint[0])}-{int(restraint[1])} -> {restraint[2]:.1f} Å (weight={restraint[3]:.1f})")
    
    # 3. Create energy function
    print("\n3. Creating energy function...")
    energy_fn = create_energy_fn(
        distance_restraints=distance_restraints,
        use_excluded_volume=True,
        excluded_volume_radius=2.0,
        excluded_volume_k=50.0
    )
    
    # Calculate initial energy
    initial_energy = energy_fn(initial_structure.positions)
    print(f"   Initial energy: {initial_energy:.2f}")
    
    # 4. Run MCMC sampling
    print("\n4. Running MCMC sampling...")
    print("   Algorithm: NUTS (No-U-Turn Sampler)")
    print("   Warmup samples: 500")
    print("   Production samples: 1000")
    print("   This may take a minute...")
    
    samples, sampling_info = sample_structure(
        initial_structure=initial_structure,
        energy_fn=energy_fn,
        n_samples=1000,
        n_warmup=500,
        step_size=0.01,
        temperature=1.0,
        algorithm="nuts",
        key=key
    )
    
    print(f"\n   Sampling complete!")
    print(f"   Generated {samples.shape[0]} samples")
    if sampling_info['acceptance_rate'] is not None:
        print(f"   Acceptance rate: {sampling_info['acceptance_rate']:.2%}")
    
    # 5. Analyze results
    print("\n5. Analyzing sampling results...")
    stats = calculate_sampling_statistics(samples, energy_fn)
    
    print(f"   Mean energy: {stats['mean_energy']:.2f}")
    print(f"   Std energy: {stats['std_energy']:.2f}")
    print(f"   Min energy: {stats['min_energy']:.2f}")
    print(f"   Max energy: {stats['max_energy']:.2f}")
    print(f"   Mean positional std per atom: {stats['mean_std_per_atom']:.3f} Å")
    
    # 6. Get best structure
    print("\n6. Finding best structure...")
    best_positions, best_energy = get_best_structure(samples, energy_fn)
    print(f"   Best energy found: {best_energy:.2f}")
    print(f"   Improvement from initial: {initial_energy - best_energy:.2f}")
    
    # 7. Calculate final distances for restraints
    print("\n7. Checking restraint satisfaction in best structure...")
    for i, restraint in enumerate(distance_restraints):
        atom_i, atom_j = int(restraint[0]), int(restraint[1])
        target_dist = restraint[2]
        actual_dist = jnp.linalg.norm(best_positions[atom_i] - best_positions[atom_j])
        diff = abs(actual_dist - target_dist)
        print(f"   Restraint {i+1}: target={target_dist:.1f} Å, actual={actual_dist:.2f} Å, diff={diff:.2f} Å")
    
    # 8. Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Successfully sampled {n_residues}-residue protein structure")
    print(f"Initial energy: {initial_energy:.2f}")
    print(f"Final best energy: {best_energy:.2f}")
    print(f"Energy reduction: {initial_energy - best_energy:.2f}")
    print("\nThis demonstrates Bayesian inference for protein structure")
    print("sampling using JAX-based scoring and BlackJax MCMC.")
    print("=" * 70)


if __name__ == "__main__":
    main()
