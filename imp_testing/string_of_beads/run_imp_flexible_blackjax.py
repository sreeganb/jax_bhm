import os
import sys
import contextlib
import time
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
    run_rmh_on_imp_system,
    run_smc_on_imp_system,
    run_adaptive_smc_on_imp_system,
)
from sampling.imp_blackjax_adapter import (
    IMPDOFSpace,
    IMPSMCAdapter,
    assert_imp_roundtrip,
    write_flat_to_imp,
)


def get_runtime_environment_info():
    """Collect backend/device info for timing reports."""
    jax_devices = jax.devices()
    jax_platforms = sorted({d.platform for d in jax_devices})
    jax_device_names = [str(d) for d in jax_devices]
    default_backend = jax.default_backend()
    is_jax_cpu_only = bool(jax_platforms) and all(p == "cpu" for p in jax_platforms)

    return {
        "jax_default_backend": default_backend,
        "jax_platforms": jax_platforms,
        "jax_device_count": len(jax_devices),
        "jax_device_names": jax_device_names,
        "jax_cpu_only": is_jax_cpu_only,
        # IMP sampling is C++ scoring path unless explicitly routed through _get_jax().
        "imp_replica_exchange_uses_jax_score_path": False,
    }


def write_timing_report_txt(report_path, benchmark_config, env_info, timing_results):
    """Write a detailed plain-text benchmarking report."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rmh = timing_results.get("rmh", {})
    rex = timing_results.get("rex", {})

    rmh_steps = int(rmh.get("n_steps", 0))
    rex_frames = int(rex.get("number_of_frames", 0))
    rmh_seconds = float(rmh.get("elapsed_seconds", float("nan")))
    rex_seconds = float(rex.get("elapsed_seconds", float("nan")))

    rmh_rate = (rmh_steps / rmh_seconds) if (rmh_steps > 0 and rmh_seconds > 0.0) else float("nan")
    rex_rate = (rex_frames / rex_seconds) if (rex_frames > 0 and rex_seconds > 0.0) else float("nan")

    lines = []
    lines.append("IMP vs BlackJAX Sampling Timing Report")
    lines.append("=" * 80)
    lines.append(f"Generated at unix_time: {time.time():.6f}")
    lines.append("")

    lines.append("Benchmark configuration")
    lines.append("-" * 80)
    for k in sorted(benchmark_config.keys()):
        lines.append(f"{k}: {benchmark_config[k]}")
    lines.append("")

    lines.append("Runtime environment")
    lines.append("-" * 80)
    lines.append(f"jax_default_backend: {env_info['jax_default_backend']}")
    lines.append(f"jax_platforms: {env_info['jax_platforms']}")
    lines.append(f"jax_device_count: {env_info['jax_device_count']}")
    lines.append("jax_device_names:")
    for name in env_info["jax_device_names"]:
        lines.append(f"  - {name}")
    lines.append(f"jax_cpu_only: {env_info['jax_cpu_only']}")
    lines.append("")

    lines.append("Sampler timing")
    lines.append("-" * 80)
    lines.append("BlackJAX RMH")
    lines.append(f"  n_steps: {rmh_steps}")
    lines.append(f"  elapsed_seconds: {rmh_seconds:.6f}")
    lines.append(f"  steps_per_second: {rmh_rate:.6f}")
    lines.append(f"  acceptance_rate: {rmh.get('acceptance_rate', 'n/a')}")
    lines.append(f"  final_imp_score: {rmh.get('final_imp_score', 'n/a')}")
    lines.append("")
    lines.append("IMP ReplicaExchange")
    lines.append(f"  number_of_frames: {rex_frames}")
    lines.append(f"  monte_carlo_steps: {rex.get('monte_carlo_steps', 'n/a')}")
    lines.append(f"  elapsed_seconds: {rex_seconds:.6f}")
    lines.append(f"  frames_per_second: {rex_rate:.6f}")
    lines.append(f"  final_imp_score: {rex.get('final_imp_score', 'n/a')}")
    lines.append("")

    lines.append("Score-path interpretation")
    lines.append("-" * 80)
    lines.append(
        "BlackJAX RMH path: uses IMP->JAX score path via sf_imp._get_jax(), "
        "adapter log_prob(), and JAX-jitted score wrappers."
    )
    lines.append(
        "IMP ReplicaExchange path: uses IMP C++ scoring path inside PMI ReplicaExchange "
        "(no sf_imp._get_jax() call in the REX loop)."
    )
    lines.append(
        "Therefore REX in this script is not using the JAX-jitted scoring pipeline for sampling."
    )
    lines.append("")

    lines.append("CPU/GPU interpretation")
    lines.append("-" * 80)
    if env_info["jax_cpu_only"]:
        lines.append(
            "JAX reports CPU-only devices in this run. BlackJAX RMH is running on CPU backend."
        )
    else:
        lines.append(
            "JAX reports non-CPU devices; BlackJAX RMH can run on accelerator backend for JAX kernels."
        )
    lines.append(
        "IMP ReplicaExchange sampling uses IMP's native C++ path and is generally CPU-side in this workflow."
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TeeStream:
    """Write text to multiple streams at once."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def tee_to_log(log_path):
    """Mirror stdout/stderr into a log file while preserving terminal output."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            yield


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


def _select_one_particle(root_hier, *, state_index, molecule, residue_index, copy_index=0):
    """Return a single selected particle or raise with useful context."""
    sel = IMP.atom.Selection(
        root_hier,
        state_index=int(state_index),
        resolution=1,
        molecule=str(molecule),
        residue_index=int(residue_index),
        copy_index=int(copy_index),
    )
    particles = sel.get_selected_particles()
    if not particles:
        raise ValueError(
            "Selection failed for "
            f"state={state_index}, molecule={molecule}, residue={residue_index}, copy_index={copy_index}."
        )
    return particles[0]


def _add_inter_copy_assembly_restraints(
    model,
    root_hier,
    n_copies,
    residue_pairs,
    mean_distance,
    kappa,
    left_molecule="KCOIL",
    right_molecule="KCOIL",
    close_ring=False,
):
    """
    Add assembly restraints between adjacent copies/states.

    Example default behavior:
    state i: KCOIL[first/last]  <->  state i+1: KCOIL[first/last]
    """
    if int(n_copies) <= 1:
        return []

    pair_states = [(i, i + 1) for i in range(int(n_copies) - 1)]
    if bool(close_ring) and int(n_copies) > 2:
        pair_states.append((int(n_copies) - 1, 0))

    restraints = []
    for left_state, right_state in pair_states:
        for left_residue, right_residue in residue_pairs:
            ts = IMP.core.Harmonic(float(mean_distance), float(kappa))
            p_left = _select_one_particle(
                root_hier,
                state_index=left_state,
                molecule=left_molecule,
                residue_index=int(left_residue),
                copy_index=0,
            )
            p_right = _select_one_particle(
                root_hier,
                state_index=right_state,
                molecule=right_molecule,
                residue_index=int(right_residue),
                copy_index=0,
            )
            dr = IMP.core.DistanceRestraint(model, ts, p_left, p_right)
            restraints.append(dr)
            print(
                "Added inter-copy assembly restraint: "
                f"state={left_state}:{left_molecule}[{left_residue}] "
                f"<-> state={right_state}:{right_molecule}[{right_residue}]"
            )
    return restraints


def build_imp_system(
    copy_count=1,
    inter_copy_residue_pairs=((1, 1), (52, 52)),
    inter_mean_distance=5.0,
    inter_kappa=15.0,
    inter_left_molecule="KCOIL",
    inter_right_molecule="KCOIL",
    close_ring=False,
):
    """Build string-of-beads system with copy-count assembly restraints.

    Restraints included:
    1) Connectivity restraints (within each copy/state)
    2) Harmonic inter-copy assembly restraints (between adjacent copies/states)

    No other restraints are added here.
    """
    copy_count = int(copy_count)
    if copy_count < 1:
        raise ValueError(f"copy_count must be >= 1, got {copy_count}.")

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
    for _ in range(copy_count):
        # Each added state is a structural copy of the topology.
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

    state_molecules = bs.get_molecules()
    connectivity_wrappers = []
    for state_idx, molecules in enumerate(state_molecules):
        for mol_names in molecules:
            for mol in molecules[mol_names]:
                cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
                cr.set_label(f"{mol.get_name()}_state{state_idx}")
                cr.add_to_model()
                connectivity_wrappers.append(cr)
                print(
                    "Added connectivity restraint for molecule: "
                    f"{mol.get_name()} (state={state_idx})"
                )

    distance_restraints = []
    assembly_restraints = _add_inter_copy_assembly_restraints(
        model=m,
        root_hier=root_hier,
        n_copies=copy_count,
        residue_pairs=[(int(a), int(b)) for a, b in inter_copy_residue_pairs],
        mean_distance=float(inter_mean_distance),
        kappa=float(inter_kappa),
        left_molecule=str(inter_left_molecule),
        right_molecule=str(inter_right_molecule),
        close_ring=bool(close_ring),
    )
    distance_restraints.extend(assembly_restraints)
    print(
        "Added inter-copy harmonic assembly restraints: "
        f"{len(assembly_restraints)}"
    )

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
    
    # Add excluded volume restraint to the scoring function
    ev = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
        root_hier, resolution=1
    )
    excluded_volume_restraint = ev.get_restraint()

    all_restraints = [*connectivity_restraints, *distance_restraints, excluded_volume_restraint]
    sf_imp = IMP.core.RestraintsScoringFunction(all_restraints)
    print(
        "Scoring function contains "
        f"{len(connectivity_restraints)} connectivity restraint(s) and "
        f"{len(distance_restraints)} inter-copy harmonic restraint(s)."
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


def make_imp_sync_fn(model, smc_adapter):
    """Build a callback that writes a flat sampled state into IMP coordinates."""

    def _sync(flat_position):
        write_flat_to_imp(model, smc_adapter.dof_space, flat_position)

    return _sync


def build_smc_adapter_context(root_hier, dof, sf_imp, box_half_width, dof_mode="flex"):
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

    dof_mode = str(dof_mode).lower()
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


def run_rmh_case(
    model,
    root_hier,
    dof,
    sf_imp,
    box_half_width,
    dof_mode,
    rmh_trajectory_rmf,
    rmh_final_rmf,
    rmh_n_steps,
):
    """Run RMH sampling and write trajectory/final snapshots."""
    rmh_adapter, leaf_rows, n_jax_rows = build_smc_adapter_context(
        root_hier=root_hier,
        dof=dof,
        sf_imp=sf_imp,
        box_half_width=box_half_width,
        dof_mode=dof_mode,
    )

    proposal_fn = None
    if hasattr(rmh_adapter, "make_rmh_proposal_fn"):
        proposal_cfg = (
            rmh_adapter.suggested_rmh_proposal()
            if hasattr(rmh_adapter, "suggested_rmh_proposal")
            else {}
        )
        proposal_fn = rmh_adapter.make_rmh_proposal_fn(**proposal_cfg)

    x0 = rmh_adapter.encode()
    ji = sf_imp._get_jax()
    roundtrip_tol = float(os.environ.get("IMP_ROUNDTRIP_ATOL", "1e-2"))
    roundtrip_strict = os.environ.get("IMP_ROUNDTRIP_STRICT", "0") == "1"
    roundtrip_err = assert_imp_roundtrip(
        model,
        ji,
        rmh_adapter,
        flat=np.asarray(x0),
        atol=roundtrip_tol,
        warn_only=(not roundtrip_strict),
    )
    print(
        "IMP/JAX roundtrip check completed "
        f"(max_abs_err={roundtrip_err:.3e}, atol={roundtrip_tol:.1e}, "
        f"strict={roundtrip_strict})."
    )

    initial_log_prob = float(rmh_adapter.log_prob(x0))
    initial_log_prior = float(rmh_adapter.log_prior(x0))
    initial_score = rmh_adapter.imp_score(x0)
    print(
        "Initial RMH posterior components: "
        f"score={initial_score:.6f}, "
        f"log_prior={initial_log_prior:.6f}, "
        f"log_posterior={initial_log_prob:.6f}"
    )
    print(f"Sampling dimension: {x0.shape[0]}")

    sync_fn = make_imp_sync_fn(model=model, smc_adapter=rmh_adapter)

    rmf_output, step_callback = make_rmf_step_callback(
        root_hier,
        rmf_path=str(rmh_trajectory_rmf),
        write_stride=10,
    )

    t0 = time.perf_counter()
    result = run_rmh_on_imp_system(
        log_prob_fn=rmh_adapter.log_prob,
        initial_position=x0,
        rng_key=jax.random.PRNGKey(0),
        n_steps=int(rmh_n_steps),
        sigma=2.0,
        proposal_fn=proposal_fn,
        sync_fn=sync_fn,
        sync_stride=10,
        step_callback=step_callback,
        verbose=True,
    )
    rmh_elapsed = time.perf_counter() - t0

    rmf_output.close_rmf(str(rmh_trajectory_rmf))

    # Ensure final frame reflects the terminal chain state.
    sync_fn(result.positions[-1])

    final_score = sf_imp.evaluate(False)
    final_log_prior = float(rmh_adapter.log_prior(result.positions[-1]))
    final_log_prob = float(rmh_adapter.log_prob(result.positions[-1]))
    print("\nRMH completed.")
    print(f"  Acceptance rate: {result.acceptance_rate:.2%}")
    print(f"  Best log posterior: {np.max(result.log_probs):.6f}")
    print(f"  Final IMP score: {final_score:.6f}")
    print(
        "  Final posterior components: "
        f"score={final_score:.6f}, "
        f"log_prior={final_log_prior:.6f}, "
        f"log_posterior={final_log_prob:.6f}"
    )

    out = IMP.pmi.output.Output()
    out.init_rmf(str(rmh_final_rmf), [root_hier])
    out.write_rmf(str(rmh_final_rmf))
    out.close_rmf(str(rmh_final_rmf))

    return {
        "n_steps": int(rmh_n_steps),
        "elapsed_seconds": float(rmh_elapsed),
        "acceptance_rate": float(result.acceptance_rate),
        "final_imp_score": float(final_score),
        "best_log_posterior": float(np.max(result.log_probs)),
    }


def run_fixed_smc_case(
    root_hier,
    dof,
    sf_imp,
    box_half_width,
    dof_mode,
    smc_debug,
    smc_debug_stride,
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
        dof_mode=dof_mode,
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
        debug=smc_debug,
        debug_stride=smc_debug_stride,
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
    dof_mode,
    smc_debug,
    smc_debug_stride,
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
        dof_mode=dof_mode,
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
        debug=smc_debug,
        debug_stride=smc_debug_stride,
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


def run_replica_exchange_case(
    m,
    root_hier,
    dof,
    sf_imp,
    rex_final_rmf,
    rex_number_of_frames,
    rex_monte_carlo_steps,
):
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
        monte_carlo_sample_objects=all_movers,
        output_objects=[],
        monte_carlo_steps=int(rex_monte_carlo_steps),
        number_of_frames=int(rex_number_of_frames),
    )
    t0 = time.perf_counter()
    rex.execute_macro()
    rex_elapsed = time.perf_counter() - t0

    out = IMP.pmi.output.Output()
    out.init_rmf(str(rex_final_rmf), [root_hier])
    out.write_rmf(str(rex_final_rmf))
    out.close_rmf(str(rex_final_rmf))
    final_score = float(sf_imp.evaluate(False))
    print(f"Final IMP score after ReplicaExchange: {final_score:.4f}")

    return {
        "number_of_frames": int(rex_number_of_frames),
        "monte_carlo_steps": int(rex_monte_carlo_steps),
        "elapsed_seconds": float(rex_elapsed),
        "final_imp_score": float(final_score),
    }


def main():
    copy_count = 10
    # Example multi-copy setup:
    # copy_count = 2

    m, root_hier, dof, sf_imp, _ = build_imp_system(
        copy_count=copy_count,
        inter_copy_residue_pairs=((1, 1), (52, 52)),
        inter_mean_distance=15.0,
        inter_kappa=5.0,
        inter_left_molecule="KCOIL",
        inter_right_molecule="KCOIL",
        close_ring=False,
    )

    print(f"Configured copy_count={copy_count}")

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

    # Optional weak box width for adapter prior.
    box_half_width = 300.0

    # Sampling mode: choose from {'flex', 'rigid', 'all'}.
    smc_dof_mode = "all"
    smc_debug = True
    smc_debug_stride = 10

    # Benchmark controls for direct RMH vs REX comparison.
    rmh_n_steps = 10000
    rex_number_of_frames = 1000
    rex_monte_carlo_steps = 2

    env_info = get_runtime_environment_info()
    print("Runtime environment summary:")
    print(f"  JAX default backend: {env_info['jax_default_backend']}")
    print(f"  JAX platforms: {env_info['jax_platforms']}")
    print(f"  JAX CPU-only: {env_info['jax_cpu_only']}")

    benchmark_config = {
        "copy_count": int(copy_count),
        "dof_mode": smc_dof_mode,
        "rmh_n_steps": int(rmh_n_steps),
        "rex_number_of_frames": int(rex_number_of_frames),
        "rex_monte_carlo_steps": int(rex_monte_carlo_steps),
        "box_half_width": float(box_half_width),
    }

    timing_results = {}

    with tee_to_log(rmh_output_dir / "rmh_run.log"):
        timing_results["rmh"] = run_rmh_case(
            model=m,
            root_hier=root_hier,
            dof=dof,
            sf_imp=sf_imp,
            box_half_width=box_half_width,
            dof_mode=smc_dof_mode,
            rmh_trajectory_rmf=rmh_trajectory_rmf,
            rmh_final_rmf=rmh_final_rmf,
            rmh_n_steps=rmh_n_steps,
        )

#    with tee_to_log(smc_output_dir / "smc_run.log"):
#        run_fixed_smc_case(
#            root_hier=root_hier,
#            dof=dof,
#            sf_imp=sf_imp,
#            box_half_width=box_half_width,
#            dof_mode=smc_dof_mode,
#            smc_debug=smc_debug,
#            smc_debug_stride=smc_debug_stride,
#            smc_best_trajectory_rmf=smc_best_trajectory_rmf,
#            smc_final_rmf=smc_final_rmf,
#        )

#    with tee_to_log(adaptive_smc_output_dir / "adaptive_smc_run.log"):
#        run_adaptive_smc_case(
#            root_hier=root_hier,
#            dof=dof,
#            sf_imp=sf_imp,
#            box_half_width=box_half_width,
#            dof_mode=smc_dof_mode,
#            smc_debug=smc_debug,
#            smc_debug_stride=smc_debug_stride,
#            adaptive_smc_best_trajectory_rmf=adaptive_smc_best_trajectory_rmf,
#            adaptive_smc_final_rmf=adaptive_smc_final_rmf,
#        )

    with tee_to_log(rex_output_dir / "rex_run.log"):
        timing_results["rex"] = run_replica_exchange_case(
            m=m,
            root_hier=root_hier,
            dof=dof,
            sf_imp=sf_imp,
            rex_final_rmf=rex_final_rmf,
            rex_number_of_frames=rex_number_of_frames,
            rex_monte_carlo_steps=rex_monte_carlo_steps,
        )

    timing_report_path = Path.cwd() / "sampling_timing_report.txt"
    write_timing_report_txt(
        report_path=timing_report_path,
        benchmark_config=benchmark_config,
        env_info=env_info,
        timing_results=timing_results,
    )
    print(f"Saved timing report: {timing_report_path}")
    

if __name__ == "__main__":
    main()
