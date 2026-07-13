import os
import numpy as np
from pathlib import Path

# Some GPU/XLA builds report noisy GEMM autotuner mismatches for this workload.
# Lowering autotune level avoids those warnings and favors stable kernels.
_xla_flag = "--xla_gpu_autotune_level=0"
_xla_flags = os.environ.get("XLA_FLAGS", "")
if _xla_flag not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} {_xla_flag}".strip()

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
    run_smc_on_imp_system,
    run_adaptive_smc_on_imp_system,
)
from sampling.imp_blackjax_adapter import IMPDOFSpace, IMPSMCAdapter


def prepare_sampler_output_dir(sampler_name):
    """Create fresh '<sampler_name>_output' dir with old_* rollover policy."""
    target_dir = Path.cwd() / f"{sampler_name}_output"
    old_dir = target_dir.parent / f"old_{target_dir.name}"

    if target_dir.exists():
        if old_dir.exists():
            timestamp = str(int(old_dir.stat().st_mtime_ns))
            rolled_old_dir = target_dir.parent / f"{old_dir.name}_{timestamp}"
            suffix = 1
            while rolled_old_dir.exists():
                rolled_old_dir = target_dir.parent / f"{old_dir.name}_{timestamp}_{suffix}"
                suffix += 1
            old_dir.rename(rolled_old_dir)
        target_dir.rename(old_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


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
    connectivity_wrappers = []
    for mol_names in molecules:
        for mol in molecules[mol_names]:
            cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
            cr.set_label(mol.get_name())
            cr.add_to_model()
            connectivity_wrappers.append(cr)
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

    # PMI restraints are wrappers; RestraintsScoringFunction needs raw IMP restraints.
    connectivity_restraints = []
    for cr in connectivity_wrappers:
        if hasattr(cr, "get_restraint"):
            connectivity_restraints.append(cr.get_restraint())
        elif hasattr(cr, "get_restraint_set"):
            connectivity_restraints.append(cr.get_restraint_set())
        else:
            raise TypeError(
                "ConnectivityRestraint object has no get_restraint()/get_restraint_set() method. "
                "Cannot extract underlying IMP restraint."
            )

    all_restraints = [*connectivity_restraints, *distance_restraints]
    sf_imp = IMP.core.RestraintsScoringFunction(all_restraints)
    print(
        "Scoring function contains "
        f"{len(connectivity_restraints)} connectivity restraint(s) and "
        f"{len(distance_restraints)} explicit distance restraint(s)."
    )
    print(f"Initial shuffled IMP score: {sf_imp.evaluate(False):.4f}")

    # Keep rigid-body bookkeeping, but we only sample beads.
    rbs, beads = get_rbs_and_beads(root_hier)
    bead_indices = [int(b.get_particle_index()) for b in beads]

    print(f"Rigid bodies found: {len(rbs)}")
    print(f"Flexible beads sampled: {len(bead_indices)}")

    return m, root_hier, dof, sf_imp, bead_indices


def make_rmf_step_callback(root_hier, rmf_path, write_stride=10):
    """Create callback that dumps RMF frames every write_stride steps."""
    output = IMP.pmi.output.Output()
    output.init_rmf(rmf_path, [root_hier])

    def _callback(step, position, log_prob, is_accepted):
        if step % write_stride == 0:
            output.write_rmf(rmf_path)

    return output, _callback


def get_ji_particle_indices(ji, jm_initial):
    """Return IMP particle index for each row in jm_initial['xyz']."""
    for method_name in ("get_particle_indices", "get_particle_indexes"):
        method = getattr(ji, method_name, None)
        if callable(method):
            values = [int(v) for v in method()]
            if len(values) == int(np.asarray(jm_initial["xyz"]).shape[0]):
                return values

    for attr_name in ("particle_indices", "particle_indexes", "particle_index"):
        values = getattr(ji, attr_name, None)
        if values is not None:
            values = [int(v) for v in values]
            if len(values) == int(np.asarray(jm_initial["xyz"]).shape[0]):
                return values

    return None


def build_leaf_rows_from_coordinates(root_hier, jm_initial_xyz, atol=1e-3):
    """Map hierarchy leaves to JAX xyz rows by nearest-coordinate matching."""
    leaves = IMP.atom.get_leaves(root_hier)
    rows = np.asarray(jm_initial_xyz, dtype=np.float64)
    used_rows = set()
    leaf_rows = []
    atol2 = float(atol) * float(atol)

    for leaf_idx, particle in enumerate(leaves):
        c = IMP.core.XYZ(particle).get_coordinates()
        leaf_xyz = np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float64)

        d2 = np.sum((rows - leaf_xyz[None, :]) ** 2, axis=1)
        order = np.argsort(d2)

        chosen = None
        for row_idx in order:
            if int(row_idx) in used_rows:
                continue
            chosen = int(row_idx)
            break

        if chosen is None or float(d2[chosen]) > atol2:
            raise RuntimeError(
                "Could not map hierarchy leaf to JAX xyz row by coordinates. "
                f"leaf_idx={leaf_idx}, min_dist={float(np.sqrt(np.min(d2))):.4e}, atol={atol:.1e}"
            )

        used_rows.add(chosen)
        leaf_rows.append(chosen)

    return np.asarray(leaf_rows, dtype=np.int32)


def write_best_positions_to_rmf(root_hier, smc_adapter, best_positions, rmf_path, leaf_rows, n_jax_rows):
    """Write SMC best-per-step trajectory by mapping decoded rows to hierarchy leaves."""
    leaves = IMP.atom.get_leaves(root_hier)

    out = IMP.pmi.output.Output()
    out.init_rmf(rmf_path, [root_hier])
    n_written = 0

    for pos in best_positions:
        # Skip non-finite states if SMC produced unstable particles late in tempering.
        if not np.all(np.isfinite(np.asarray(pos))):
            continue

        xyz = smc_adapter.decode_xyz(jnp.asarray(pos))
        if xyz.shape[0] != n_jax_rows:
            raise ValueError(
                f"SMC decode size mismatch: decoded {xyz.shape[0]} rows, "
                f"but JAX mapping has {n_jax_rows} rows."
            )

        leaf_xyz = xyz[leaf_rows]

        for particle, coord in zip(leaves, leaf_xyz):
            IMP.core.XYZ(particle).set_coordinates(
                IMP.algebra.Vector3D(float(coord[0]), float(coord[1]), float(coord[2]))
            )
        out.write_rmf(rmf_path)
        n_written += 1

    out.close_rmf(rmf_path)
    if n_written == 0:
        raise RuntimeError(
            "No finite SMC best positions were available to write into RMF."
        )


def build_smc_adapter_context(root_hier, dof, sf_imp, box_half_width):
    """Build SMC adapter plus hierarchy-to-JAX row mapping."""
    ji = sf_imp._get_jax()
    jm_initial = ji.get_jax_model()
    ji_particle_indices = get_ji_particle_indices(ji, jm_initial)
    n_jax_rows = int(np.asarray(jm_initial["xyz"]).shape[0])

    if ji_particle_indices is not None:
        leaves = IMP.atom.get_leaves(root_hier)
        leaf_particle_indices = [int(p.get_index()) for p in leaves]
        pid_to_jax_row = {int(pid): i for i, pid in enumerate(ji_particle_indices)}
        missing = [pid for pid in leaf_particle_indices if pid not in pid_to_jax_row]
        if missing:
            raise RuntimeError(
                "JAX mapping exists but does not cover all hierarchy leaves. "
                f"First missing particle indices: {missing[:10]}"
            )
        leaf_rows = np.asarray([pid_to_jax_row[pid] for pid in leaf_particle_indices], dtype=np.int32)
    else:
        print("ji does not expose particle indices; using coordinate-based leaf mapping.")
        leaf_rows = build_leaf_rows_from_coordinates(root_hier, jm_initial["xyz"], atol=1e-3)

    dof_mode = "flex"
    smc_adapter = IMPSMCAdapter(
        IMPDOFSpace.from_imp(dof, ji, jm_initial, mode=dof_mode),
        ji.score_func,
        kT=1.0,
        box_half_width=box_half_width,
    )

    rbs, beads = get_rbs_and_beads(root_hier)
    print(
        "SMC adapter context: "
        f"mode={dof_mode}, sampled_flexible_beads={len(beads)}, "
        f"rigid_bodies_fixed_in_score={len(rbs)}, "
        f"jax_rows_scored={n_jax_rows}"
    )
    return smc_adapter, leaf_rows, n_jax_rows


def run_rmh_case(parameter_space, log_posterior, sf_imp, root_hier, rmh_trajectory_rmf, rmh_final_rmf):
    """Run RMH sampling and write trajectory/final snapshots."""
    rmf_output, step_callback = make_rmf_step_callback(
        root_hier,
        rmf_path=str(rmh_trajectory_rmf),
        write_stride=10,
    )

    result = run_rmh_on_imp_system(
        parameter_space=parameter_space,
        log_prob_fn=log_posterior,
        rng_key=jax.random.PRNGKey(0),
        n_steps=1000,
        sigma=2.0,
        step_callback=step_callback,
        verbose=True,
        debug=True,
        debug_stride=50,
    )

    rmf_output.close_rmf(str(rmh_trajectory_rmf))

    final_score = sf_imp.evaluate(False)
    final_eval = log_posterior.evaluate(parameter_space.pack())
    print("\nRMH completed.")
    print(f"  Acceptance rate: {result.acceptance_rate:.2%}")
    print(f"  Best log posterior: {np.max(result.log_probs):.6f}")
    print(f"  Final IMP score: {final_score:.6f}")
    print(
        "  Final posterior components: "
        f"score={final_eval.score:.6f}, "
        f"log_prior={final_eval.log_prior:.6f}, "
        f"log_posterior={final_eval.log_posterior:.6f}"
    )

    out = IMP.pmi.output.Output()
    out.init_rmf(str(rmh_final_rmf), [root_hier])
    out.write_rmf(str(rmh_final_rmf))
    out.close_rmf(str(rmh_final_rmf))


def run_fixed_smc_case(
    root_hier,
    dof,
    sf_imp,
    box_half_width,
    smc_best_trajectory_rmf,
    smc_final_rmf,
):
    """Run fixed-temperature-step SMC and write outputs."""
    IMP.pmi.tools.shuffle_configuration(
        root_hier,
        max_translation=500,
        avoidcollision_rb=False,
        bounding_box=((-100, -100, 0), (100, 100, 100)),
    )
    print(f"Randomized IMP score before SMC: {sf_imp.evaluate(False):.4f}")

    # Rebuild adapter from the current IMP coordinates so fixed rigid-body rows
    # in flex-only mode are synchronized with the exact shuffled conformation.
    smc_adapter, leaf_rows, n_jax_rows = build_smc_adapter_context(
        root_hier=root_hier,
        dof=dof,
        sf_imp=sf_imp,
        box_half_width=box_half_width,
    )

    smc_state, smc_info, smc_best_pos, smc_best_scores, smc_lambdas = run_smc_on_imp_system(
        adapter=smc_adapter,
        rng_key=jax.random.PRNGKey(1),
        n_particles=100,
        n_temperature_steps=100,
        schedule="geometric",
        kernel="rmh",
        rmh_sigma=2.0,
        n_mcmc_steps=50,
        score_batch_size=16,
        save_rmf3_path=None,
        verbose=True,
    )

    write_best_positions_to_rmf(
        root_hier=root_hier,
        smc_adapter=smc_adapter,
        best_positions=smc_best_pos,
        rmf_path=str(smc_best_trajectory_rmf),
        leaf_rows=leaf_rows,
        n_jax_rows=n_jax_rows,
    )

    smc_final_score = sf_imp.evaluate(False)
    finite_best = np.asarray(smc_best_scores, dtype=float)
    finite_best = finite_best[np.isfinite(finite_best)]
    print("\nSMC completed.")
    if finite_best.size > 0:
        print(f"  Best log posterior (finite): {float(np.max(finite_best)):.6f}")
    else:
        print("  Best log posterior (finite): none")
    print(f"  Final IMP score: {smc_final_score:.6f}")

    out = IMP.pmi.output.Output()
    out.init_rmf(str(smc_final_rmf), [root_hier])
    out.write_rmf(str(smc_final_rmf))
    out.close_rmf(str(smc_final_rmf))


def run_adaptive_smc_case(
    root_hier,
    dof,
    sf_imp,
    box_half_width,
    adaptive_smc_best_trajectory_rmf,
    adaptive_smc_final_rmf,
):
    """Run adaptive-step SMC and write outputs."""
    IMP.pmi.tools.shuffle_configuration(
        root_hier,
        max_translation=500,
        avoidcollision_rb=False,
        bounding_box=((-100, -100, 0), (100, 100, 100)),
    )
    print(f"Randomized IMP score before SMC: {sf_imp.evaluate(False):.4f}")

    # Rebuild adapter from the current IMP coordinates so fixed rigid-body rows
    # in flex-only mode are synchronized with the exact shuffled conformation.
    smc_adapter, leaf_rows, n_jax_rows = build_smc_adapter_context(
        root_hier=root_hier,
        dof=dof,
        sf_imp=sf_imp,
        box_half_width=box_half_width,
    )

    state, info, best_pos, best_scores, lambdas = run_adaptive_smc_on_imp_system(
        adapter=smc_adapter,
        rng_key=jax.random.PRNGKey(2),
        n_particles=100,
        max_temperature_steps=200,
        target_ess=0.5,
        rmh_sigma=2.0,
        n_mcmc_steps=50,
        score_batch_size=16,
        save_rmf3_path=None,
        verbose=True,
    )

    write_best_positions_to_rmf(
        root_hier=root_hier,
        smc_adapter=smc_adapter,
        best_positions=best_pos,
        rmf_path=str(adaptive_smc_best_trajectory_rmf),
        leaf_rows=leaf_rows,
        n_jax_rows=n_jax_rows,
    )

    smc_final_score = sf_imp.evaluate(False)
    finite_best = np.asarray(best_scores, dtype=float)
    finite_best = finite_best[np.isfinite(finite_best)]
    print("\nAdaptive SMC completed.")
    if finite_best.size > 0:
        print(f"  Best log posterior (finite): {float(np.max(finite_best)):.6f}")
    else:
        print("  Best log posterior (finite): none")
    print(f"  Final IMP score: {smc_final_score:.6f}")

    out = IMP.pmi.output.Output()
    out.init_rmf(str(adaptive_smc_final_rmf), [root_hier])
    out.write_rmf(str(adaptive_smc_final_rmf))
    out.close_rmf(str(adaptive_smc_final_rmf))


def run_replica_exchange_case(m, root_hier, dof, sf_imp, rex_final_rmf):
    """Run IMP ReplicaExchange with flexible-only movers and save snapshot."""
    IMP.pmi.tools.shuffle_configuration(
        root_hier,
        max_translation=500,
        avoidcollision_rb=False,
        bounding_box=((-100, -100, 0), (100, 100, 100)),
    )
    print(f"Randomized IMP score before ReplicaExchange: {sf_imp.evaluate(False):.4f}")

    all_movers = list(dof.get_movers())
    flexible_movers = [m for m in all_movers if isinstance(m, IMP.core.BallMover)]
    print(
        f"Running IMP ReplicaExchange with flexible-only movers: "
        f"{len(flexible_movers)} of {len(all_movers)} total movers"
    )

    rex = IMP.pmi.macros.ReplicaExchange(
        m,
        root_hier=root_hier,
        monte_carlo_sample_objects=flexible_movers,
        output_objects=[],
        number_of_frames=500,
    )
    rex.execute_macro()

    out = IMP.pmi.output.Output()
    out.init_rmf(str(rex_final_rmf), [root_hier])
    out.write_rmf(str(rex_final_rmf))
    out.close_rmf(str(rex_final_rmf))
    print(f"Final IMP score after ReplicaExchange: {sf_imp.evaluate(False):.4f}")


def main():
    m, root_hier, dof, sf_imp, flexible_particle_indices = build_imp_system()

    rmh_output_dir = prepare_sampler_output_dir("rmh")
    smc_output_dir = prepare_sampler_output_dir("smc")
    adaptive_smc_output_dir = prepare_sampler_output_dir("adaptive_smc")
    rex_output_dir = prepare_sampler_output_dir("rex")

    rmh_trajectory_rmf = rmh_output_dir / "rmh_flexible_trajectory.rmf3"
    rmh_final_rmf = rmh_output_dir / "rmh_flexible_final.rmf3"
    smc_best_trajectory_rmf = smc_output_dir / "smc_flexible_best_trajectory.rmf3"
    smc_final_rmf = smc_output_dir / "smc_flexible_final.rmf3"
    adaptive_smc_best_trajectory_rmf = (
        adaptive_smc_output_dir / "adaptive_smc_flexible_best_trajectory.rmf3"
    )
    adaptive_smc_final_rmf = adaptive_smc_output_dir / "adaptive_smc_flexible_final.rmf3"
    rex_final_rmf = rex_output_dir / "replica_exchange_flexible_final.rmf3"

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

    initial_eval = log_posterior.evaluate(parameter_space.pack())
    print(
        "Initial posterior components: "
        f"score={initial_eval.score:.6f}, "
        f"log_prior={initial_eval.log_prior:.6f}, "
        f"log_posterior={initial_eval.log_posterior:.6f}"
    )

    print(f"Sampling dimension: {parameter_space.dim}")

    run_rmh_case(
        parameter_space=parameter_space,
        log_posterior=log_posterior,
        sf_imp=sf_imp,
        root_hier=root_hier,
        rmh_trajectory_rmf=rmh_trajectory_rmf,
        rmh_final_rmf=rmh_final_rmf,
    )

    run_fixed_smc_case(
        root_hier=root_hier,
        dof=dof,
        sf_imp=sf_imp,
        box_half_width=box_half_width,
        smc_best_trajectory_rmf=smc_best_trajectory_rmf,
        smc_final_rmf=smc_final_rmf,
    )

    run_adaptive_smc_case(
        root_hier=root_hier,
        dof=dof,
        sf_imp=sf_imp,
        box_half_width=box_half_width,
        adaptive_smc_best_trajectory_rmf=adaptive_smc_best_trajectory_rmf,
        adaptive_smc_final_rmf=adaptive_smc_final_rmf,
    )

    run_replica_exchange_case(
        m=m,
        root_hier=root_hier,
        dof=dof,
        sf_imp=sf_imp,
        rex_final_rmf=rex_final_rmf,
    )
    

if __name__ == "__main__":
    main()
