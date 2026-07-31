"""
IMP <-> JAX/BlackJAX bridge.

Responsibility: everything needed to turn a BuiltSystem + RestraintBundle into
something BlackJAX can sample, and to get sampled states back into IMP
coordinates and RMF files.

This exists as one module because the previous script contained three copies of
the same setup sequence (one each for RMH, SMC and adaptive SMC). Every
JAX-backed sampler in samplers.py now shares this single implementation.

The imports from `sampling.*` are the only dependencies on the surrounding
jax_bhm repo; nothing here modifies those modules.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

import IMP
import IMP.atom
import IMP.core
import IMP.algebra
import IMP.pmi.output

import jax
import jax.numpy as jnp

from .runtime import ensure_repo_on_path

# Locate the jax_bhm repo root if the package is not pip-installed.
ensure_repo_on_path()

from sampling.imp_blackjax_adapter import (  # noqa: E402
    IMPDOFSpace,
    IMPSMCAdapter,
    assert_imp_roundtrip,
    write_flat_to_imp,
)


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------

@dataclass
class JaxContext:
    """
    Everything a JAX sampler needs, built once from the current IMP state.

    adapter      IMPSMCAdapter exposing log_prob / log_prior / imp_score /
                 encode / decode_xyz
    dof_space    the IMPDOFSpace behind the adapter
    ji           the JAX interface object returned by ScoringFunction._get_jax()
    leaf_rows    for each hierarchy leaf, the row index in the JAX xyz array
    n_jax_rows   number of rows in the JAX xyz array
    dof_mode     'flex' | 'rigid' | 'all'

    IMPORTANT: the context snapshots the *current* IMP coordinates. Any sampler
    that shuffles the configuration must rebuild the context afterwards,
    otherwise rows that are held fixed (e.g. rigid bodies in flex-only mode)
    will correspond to the pre-shuffle conformation.
    """
    adapter: object
    dof_space: object
    ji: object
    leaf_rows: np.ndarray
    n_jax_rows: int
    dof_mode: str

    def sync_fn(self, model):
        """Callback that writes a flat sampled vector back into IMP."""
        def _sync(flat_position):
            write_flat_to_imp(model, self.dof_space, flat_position)
        return _sync


# --------------------------------------------------------------------------
# leaf -> JAX row mapping
# --------------------------------------------------------------------------

def _ji_particle_indices(ji, n_rows):
    """
    Ask the JAX interface for the IMP particle index behind each xyz row.

    IMP builds differ in how (and whether) they expose this, so several names
    are probed. Returns None if no usable mapping of the right length is found.
    """
    for method_name in ("get_particle_indices", "get_particle_indexes"):
        method = getattr(ji, method_name, None)
        if callable(method):
            values = [int(v) for v in method()]
            if len(values) == n_rows:
                return values

    for attr_name in ("particle_indices", "particle_indexes", "particle_index"):
        values = getattr(ji, attr_name, None)
        if values is not None:
            values = [int(v) for v in values]
            if len(values) == n_rows:
                return values

    return None


def _leaf_rows_by_coordinates(root_hier, xyz, atol=1e-3):
    """
    Fallback mapping: match each hierarchy leaf to its JAX row by coordinates.

    Only used when the JAX interface does not expose particle indices. Greedy
    nearest-unused-row matching; raises if any leaf is further than `atol` from
    its best available row, since a silent mismatch here would corrupt every
    trajectory written afterwards.
    """
    leaves = IMP.atom.get_leaves(root_hier)
    rows = np.asarray(xyz, dtype=np.float64)
    used = set()
    mapping = []
    atol2 = float(atol) ** 2

    for leaf_index, particle in enumerate(leaves):
        c = IMP.core.XYZ(particle).get_coordinates()
        leaf_xyz = np.array([float(c[0]), float(c[1]), float(c[2])])
        d2 = np.sum((rows - leaf_xyz[None, :]) ** 2, axis=1)

        chosen = None
        for row_index in np.argsort(d2):
            if int(row_index) not in used:
                chosen = int(row_index)
                break

        if chosen is None or float(d2[chosen]) > atol2:
            raise RuntimeError(
                "Coordinate-based leaf/JAX-row mapping failed at leaf "
                f"{leaf_index}: nearest free row is "
                f"{float(np.sqrt(np.min(d2))):.3e} away (atol={atol:.1e}). "
                "Duplicate bead coordinates are the usual cause; shuffle the "
                "configuration before building the context."
            )

        used.add(chosen)
        mapping.append(chosen)

    return np.asarray(mapping, dtype=np.int32)


def build_leaf_row_map(root_hier, ji, jax_model, atol=1e-3, verbose=True):
    """Return (leaf_rows, n_jax_rows) mapping hierarchy leaves to JAX rows."""
    n_rows = int(np.asarray(jax_model["xyz"]).shape[0])
    particle_indices = _ji_particle_indices(ji, n_rows)

    if particle_indices is None:
        if verbose:
            print("  JAX interface exposes no particle indices; "
                  "falling back to coordinate matching.")
        return _leaf_rows_by_coordinates(root_hier, jax_model["xyz"], atol), n_rows

    leaves = IMP.atom.get_leaves(root_hier)
    row_of_particle = {int(pid): i for i, pid in enumerate(particle_indices)}
    leaf_pids = [int(p.get_index()) for p in leaves]

    missing = [pid for pid in leaf_pids if pid not in row_of_particle]
    if missing:
        raise RuntimeError(
            "The JAX particle mapping does not cover every hierarchy leaf. "
            f"{len(missing)} missing, first few: {missing[:10]}"
        )

    return np.asarray([row_of_particle[pid] for pid in leaf_pids], dtype=np.int32), n_rows


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def build_jax_context(built, bundle, box_half_width=300.0, dof_mode="all",
                      kT=1.0, verbose=True):
    """
    Build a JaxContext from the current IMP coordinates.

    dof_mode selects which degrees of freedom are sampled:
        'flex'  -- flexible beads only, rigid bodies frozen
        'rigid' -- rigid-body translations/rotations only
        'all'   -- both
    """
    dof_mode = str(dof_mode).lower()
    if dof_mode not in ("flex", "rigid", "all"):
        raise ValueError(f"dof_mode must be flex|rigid|all, got '{dof_mode}'")

    # Preflight: the adapter's flat layout is
    # [rb_translations | rb_quaternions | flex_xyz] and has no block for a
    # non-rigid member's body-frame offset. Catch that here rather than letting
    # IMPDOFSpace.from_imp raise NotImplementedError several frames down.
    nonrigid = [p for p in IMP.atom.get_leaves(built.root_hier)
                if IMP.core.NonRigidMember.get_is_setup(p)]
    if nonrigid:
        raise NotImplementedError(
            f"{len(nonrigid)} particle(s) are IMP.core.NonRigidMember, which "
            "sampling.imp_blackjax_adapter does not support (they need a "
            "body-frame 3-DOF block the flat layout has no room for).\n"
            "Fix: build with SystemSpec(allow_nonrigid_members=False), the "
            "default, so unstructured residues become independent flexible "
            "beads. Non-rigid members are usable with the replica_exchange "
            "sampler, which scores through IMP's C++ path."
        )

    scoring_function = bundle.scoring_function
    ji = scoring_function._get_jax()
    jax_model = ji.get_jax_model()

    leaf_rows, n_rows = build_leaf_row_map(built.root_hier, ji, jax_model,
                                           verbose=verbose)

    # IMPDOFSpace.from_imp asserts that the xyz table is particle-indexed, i.e.
    # that it has exactly one row per particle in the model. Any IMP machinery
    # that allocates particles after the scoring function was built breaks that
    # -- most notably ReplicaExchange.execute_macro(), which adds movers, MC
    # bookkeeping and an RMF output hierarchy to the same model. Check here so
    # the failure names its cause instead of surfacing as a bare AssertionError
    # several frames down.
    try:
        n_model_particles = len(list(built.model.get_particle_indexes()))
    except Exception:
        n_model_particles = None

    if n_model_particles is not None and n_rows != n_model_particles:
        raise RuntimeError(
            f"The JAX xyz table has {n_rows} rows but the model now holds "
            f"{n_model_particles} particles, so the adapter's particle-indexed "
            "assumption no longer holds.\n"
            "The usual cause is having already run an IMP-native sampler "
            "(replica_exchange) in this process: execute_macro() allocates "
            "extra particles on the same model.\n"
            "Fix: run the JAX samplers (rmh, smc, adaptive_smc) before "
            "replica_exchange, or run each sampler in its own process."
        )

    dof_space = IMPDOFSpace.from_imp(built.dof, ji, jax_model, mode=dof_mode)
    adapter = IMPSMCAdapter(
        dof_space,
        ji.score_func,
        kT=kT,
        box_half_width=box_half_width,
    )

    if verbose:
        rigid_bodies, beads = built.rigid_bodies_and_beads()
        print(f"  JAX context: mode={dof_mode}, rigid bodies={len(rigid_bodies)}, "
              f"flexible beads={len(beads)}, scored rows={n_rows}, "
              f"sampled dimension={int(np.asarray(adapter.encode()).shape[0])}")

    return JaxContext(
        adapter=adapter,
        dof_space=dof_space,
        ji=ji,
        leaf_rows=leaf_rows,
        n_jax_rows=n_rows,
        dof_mode=dof_mode,
    )


def check_roundtrip(built, context, flat=None, atol=None, strict=None, verbose=True):
    """
    Verify that encode -> write_flat_to_imp -> re-score is self-consistent.

    Controlled by IMP_ROUNDTRIP_ATOL and IMP_ROUNDTRIP_STRICT for parity with
    the previous script. Returns the max absolute error.
    """
    if atol is None:
        atol = float(os.environ.get("IMP_ROUNDTRIP_ATOL", "1e-2"))
    if strict is None:
        strict = os.environ.get("IMP_ROUNDTRIP_STRICT", "0") == "1"

    flat = context.adapter.encode() if flat is None else flat
    error = assert_imp_roundtrip(
        built.model,
        context.ji,
        context.adapter,
        flat=np.asarray(flat),
        atol=atol,
        warn_only=(not strict),
    )
    if verbose:
        print(f"  IMP/JAX roundtrip: max_abs_err={error:.3e} "
              f"(atol={atol:.1e}, strict={strict})")
    return error


# --------------------------------------------------------------------------
# writing sampled states back out
# --------------------------------------------------------------------------

def apply_flat_position(built, context, flat_position):
    """Write one flat sampled vector into IMP coordinates."""
    write_flat_to_imp(built.model, context.dof_space, flat_position)


def write_snapshot_rmf(built, rmf_path):
    """Write the current IMP conformation as a single-frame RMF3."""
    rmf_path = str(rmf_path)
    output = IMP.pmi.output.Output()
    output.init_rmf(rmf_path, [built.root_hier])
    output.write_rmf(rmf_path)
    output.close_rmf(rmf_path)
    return rmf_path


def write_positions_rmf(built, context, positions, rmf_path, skip_nonfinite=True):
    """
    Write a trajectory from a sequence of flat sampled vectors.

    Decodes each vector to xyz, reorders rows into hierarchy-leaf order via
    context.leaf_rows, and writes one RMF frame per position. Non-finite states
    (which SMC can produce late in tempering) are skipped rather than written as
    NaNs that break every downstream analysis.
    """
    rmf_path = str(rmf_path)
    leaves = IMP.atom.get_leaves(built.root_hier)

    output = IMP.pmi.output.Output()
    output.init_rmf(rmf_path, [built.root_hier])
    n_written = 0

    for position in positions:
        position = np.asarray(position)
        if skip_nonfinite and not np.all(np.isfinite(position)):
            continue

        xyz = np.asarray(context.adapter.decode_xyz(jnp.asarray(position)))
        if xyz.shape[0] != context.n_jax_rows:
            raise ValueError(
                f"Decoded {xyz.shape[0]} rows but the leaf mapping expects "
                f"{context.n_jax_rows}."
            )

        for particle, coord in zip(leaves, xyz[context.leaf_rows]):
            IMP.core.XYZ(particle).set_coordinates(
                IMP.algebra.Vector3D(float(coord[0]), float(coord[1]), float(coord[2]))
            )
        output.write_rmf(rmf_path)
        n_written += 1

    output.close_rmf(rmf_path)
    if n_written == 0:
        raise RuntimeError(
            f"No finite positions were available to write into {rmf_path}."
        )
    return n_written


def make_rmf_stride_writer(built, rmf_path, stride=10):
    """
    Return (output, callback) writing an RMF frame every `stride` steps.

    The callback matches the signature the BlackJAX wrappers expect:
    (step, position, log_prob, is_accepted). It writes whatever is currently in
    the IMP hierarchy, so it should be paired with a sync callback at the same
    or a finer stride.
    """
    rmf_path = str(rmf_path)
    output = IMP.pmi.output.Output()
    output.init_rmf(rmf_path, [built.root_hier])

    def _callback(step, position, log_prob, is_accepted):
        if step % int(stride) == 0:
            output.write_rmf(rmf_path)

    return output, _callback
