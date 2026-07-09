import os
import numpy as np

import IMP
import IMP.atom
import IMP.core
import IMP.algebra
import IMP.pmi.macros
import IMP.pmi.output
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.tools
import IMP.pmi.topology

import jax
import jax.numpy as jnp

from sampling.wrapper_imp_blackjax import (
    build_flexible_bead_rmh_wrapper,
    run_rmh_on_imp_system,
)


def get_all_leaves(list_of_hs):
    """Get leaves from one hierarchy or a list of hierarchies."""
    if not hasattr(list_of_hs, "__iter__"):
        list_of_hs = [list_of_hs]
    leaves = []
    for h in list_of_hs:
        leaves.extend(IMP.atom.get_leaves(h))
    return leaves


def get_rbs_and_beads(hiers):
    """Return (rigid_bodies, beads) preserving first-seen RB order."""
    rbs = set()
    beads = []
    rbs_ordered = []
    for p in get_all_leaves(hiers):
        if IMP.core.RigidMember.get_is_setup(p):
            rb = IMP.core.RigidMember(p).get_rigid_body()
            if rb not in rbs:
                rbs.add(rb)
                rbs_ordered.append(rb)
        elif IMP.core.NonRigidMember.get_is_setup(p):
            rb = IMP.core.NonRigidMember(p).get_rigid_body()
            if rb not in rbs:
                rbs.add(rb)
                rbs_ordered.append(rb)
            beads.append(p)
        else:
            beads.append(p)
    return rbs_ordered, beads


def build_imp_system():
    """Build your example string-of-beads system."""
    data_dir = os.path.join(os.getcwd(), "data")
    pdb_dir = os.path.join(data_dir, "pdb")
    fasta_dir = os.path.join(data_dir, "fasta")
    topology_file = os.path.join(data_dir, "topology.txt")

    m = IMP.Model()

    topology = IMP.pmi.topology.TopologyReader(
        topology_file,
        pdb_dir=pdb_dir,
        fasta_dir=fasta_dir,
    )

    print("Parsed topology components:")
    for c in topology.get_components():
        print(
            f"  mol={c.molname} pdb={c.pdb_file} chain={c.chain} "
            f"res={c.residue_range} rb={c.rigid_body}"
        )

    bs = IMP.pmi.macros.BuildSystem(m, name="ToyModel", resolutions=[1])
    bs.add_state(topology)
    root_hier, dof = bs.execute_macro()

    out = IMP.pmi.output.Output()
    out.init_rmf("ini_all.rmf3", [root_hier])
    out.write_rmf("ini_all.rmf3")
    out.close_rmf("ini_all.rmf3")

    print("\nBuild complete.")

    for state in IMP.atom.get_by_type(root_hier, IMP.atom.STATE_TYPE):
        for mol in IMP.atom.get_by_type(state, IMP.atom.MOLECULE_TYPE):
            leaves = IMP.core.get_leaves(mol)
            print(f"  {mol.get_name()} : {len(leaves)} bead(s)")

    molecules = bs.get_molecules()[0]
    for mol_names in molecules:
        for mol in molecules[mol_names]:
            cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
            cr.set_label(mol.get_name())
            cr.add_to_model()
            print(f"Added connectivity restraint for molecule: {mol.get_name()}")

    # Add matched restraints for residue pairs: 1-1, 2-2, ..., 11-11.
    distance_max = 10.0
    kappa = 40.0
    n_extra_restraints = 10
    # adding distance restraints only between flexible beads and not the 
    # rigid bodies
    n_pairs = 1 + n_extra_restraints
    distance_restraints = []

    for residue_index in range(1, n_pairs + 1):
        ts = IMP.core.HarmonicUpperBound(distance_max, kappa)

        sel1 = IMP.atom.Selection(
            root_hier,
            resolution=1,
            molecule="KCOIL",
            residue_index=residue_index,
            copy_index=0,
        )
        sel2 = IMP.atom.Selection(
            root_hier,
            resolution=1,
            molecule="ECOIL",
            residue_index=residue_index,
            copy_index=0,
        )

        particle_1 = sel1.get_selected_particles()
        particle_2 = sel2.get_selected_particles()
        if not particle_1 or not particle_2:
            raise ValueError(
                f"Could not find selection for residue pair {residue_index}-{residue_index}."
            )

        dr = IMP.core.DistanceRestraint(m, ts, particle_1[0], particle_2[0])
        print(f"Added distance restraint for residue pair {residue_index}-{residue_index}.")
        distance_restraints.append(dr)

    print(f"Added {len(distance_restraints)} matched distance restraints (1-1 to {n_pairs}-{n_pairs}).")

    # Need to add distance restraints that will act as connectivity restraints. 
    # TODO: 

    IMP.pmi.tools.shuffle_configuration(
        root_hier,
        max_translation=500,
        avoidcollision_rb=False,
        bounding_box=((-100, -100, 0), (100, 100, 100)),
    )

    out = IMP.pmi.output.Output()
    out.init_rmf("shuffled_particles.rmf3", [root_hier])
    out.write_rmf("shuffled_particles.rmf3")
    out.close_rmf("shuffled_particles.rmf3")

    sf_imp = IMP.core.RestraintsScoringFunction(distance_restraints)
    print(f"Initial shuffled IMP score: {sf_imp.evaluate(False):.4f}")

    # Keep rigid-body bookkeeping, but we only sample beads.
    rbs, beads = get_rbs_and_beads(root_hier)
    bead_indices = [int(b.get_particle_index()) for b in beads]

    print(f"Rigid bodies found: {len(rbs)}")
    print(f"Flexible beads sampled: {len(bead_indices)}")

    return m, root_hier, sf_imp, bead_indices


def make_rmf_step_callback(root_hier, rmf_path, write_stride=10):
    """Create callback that dumps RMF frames every write_stride steps."""
    output = IMP.pmi.output.Output()
    output.init_rmf(rmf_path, [root_hier])

    def _callback(step, position, log_prob, is_accepted):
        if step % write_stride == 0:
            output.write_rmf(rmf_path)

    return output, _callback


def main():
    m, root_hier, sf_imp, flexible_particle_indices = build_imp_system()

    # Optional weak box prior to avoid unconstrained drift.
    box_half_width = 300.0
    box_sigma = 20.0

    def log_prior_fn(flat):
        excess = jnp.maximum(jnp.abs(flat) - box_half_width, 0.0)
        return -0.5 * jnp.sum((excess / box_sigma) ** 2)

    parameter_space, log_posterior = build_flexible_bead_rmh_wrapper(
        model=m,
        scoring_function=sf_imp,
        flexible_particle_indices=flexible_particle_indices,
        temperature=1.0,
        log_prior_fn=log_prior_fn,
    )

    print(f"Sampling dimension: {parameter_space.dim}")

    rmf_output, step_callback = make_rmf_step_callback(
        root_hier,
        rmf_path="rmh_flexible_trajectory.rmf3",
        write_stride=10,
    )

    rng_key = jax.random.PRNGKey(0)

    result = run_rmh_on_imp_system(
        parameter_space=parameter_space,
        log_prob_fn=log_posterior,
        rng_key=rng_key,
        n_steps=1000,
        sigma=2.0,
        step_callback=step_callback,
        verbose=True,
    )

    rmf_output.close_rmf("rmh_flexible_trajectory.rmf3")

    final_score = sf_imp.evaluate(False)
    print("\nRMH completed.")
    print(f"  Acceptance rate: {result.acceptance_rate:.2%}")
    print(f"  Best log posterior: {np.max(result.log_probs):.6f}")
    print(f"  Final IMP score: {final_score:.6f}")

    # Save final coordinates snapshot.
    out = IMP.pmi.output.Output()
    out.init_rmf("rmh_flexible_final.rmf3", [root_hier])
    out.write_rmf("rmh_flexible_final.rmf3")
    out.close_rmf("rmh_flexible_final.rmf3")


if __name__ == "__main__":
    main()
