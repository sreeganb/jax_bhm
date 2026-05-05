"""
IMP-BlackJAX SMC adapter for protein structure sampling.

Bridges IMP's JAX scoring function with the BlackJAX SMC sampler in
smc_base_sampler.py. Handles the two key challenges:

  1. DOF parameterization: maps flat BlackJAX position vectors onto
     IMP's particle coordinate array (xyz).

  2. Score convention: IMP scores are energies (lower = better), so
     the log-likelihood exposed to BlackJAX is -score / kT.

Degrees of freedom supported
-----------------------------
IMP has two kinds of DOFs:

  * Rigid bodies (RBs) — a set of particles that move as one unit.
    Each RB has a centre-of-mass (COM) translation (3 floats) and a
    rotation represented as a unit quaternion (4 floats, w first).

  * Flexible beads — individual particles whose xyz are free (3 floats).

This module parameterises the system as:

    position = [rb0_tx, rb0_ty, rb0_tz, rb0_qw, rb0_qx, rb0_qy, rb0_qz,
                rb1_tx, ...,
                fb0_x, fb0_y, fb0_z,
                fb1_x, ...]

and builds the full particle-coordinate array by applying each RB
transform to its constituent particles and directly placing the
flexible beads.

Usage
-----
    # ---- build IMP system ----
    model, system, hier, molecules = build_system(cfg)
    dof = build_degrees_of_freedom(model, hier, molecules, repo_dir)
    output_objs, sf_imp = build_restraints(...)

    # ---- build JAX score ----
    ji, jax_score_func, jm_initial = build_jax_score(sf_imp)

    # ---- build SMC adapter ----
    adapter = IMPSMCAdapter(dof, hier, ji, jax_score_func, jm_initial,
                            kT=1.0, box_half_width=500.0)

    # ---- sample ----
    initial_positions = adapter.sample_prior(n_particles=200, rng_key)
    state, info, best_pos, best_scores, lambdas = run_base_smc_rmh(
        log_prior_fn=adapter.log_prior,
        log_likelihood_fn=adapter.log_likelihood,
        log_prob_fn=adapter.log_prob,
        initial_positions=initial_positions,
        rng_key=rng_key,
        n_temperature_steps=30,
        schedule="geometric",
        rmh_sigma=adapter.suggested_rmh_sigma(),
        n_mcmc_steps=5,
    )

    # ---- decode best sample back to xyz ----
    best_xyz = adapter.decode_xyz(best_pos[−1])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.dof
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def _normalise_quat(q: jnp.ndarray) -> jnp.ndarray:
    """Project quaternion onto the unit sphere."""
    return q / (jnp.linalg.norm(q) + 1e-12)


def _get_rb_from_mover(mover) -> IMP.core.RigidBody:
    """Return an IMP.core.RigidBody from old/new RigidBodyMover APIs."""
    get_rb = getattr(mover, "get_rigid_body", None)
    if callable(get_rb):
        return IMP.core.RigidBody(get_rb())

    get_index = getattr(mover, "get_index", None)
    if callable(get_index):
        pi = int(get_index())
        model = mover.get_model()
        return IMP.core.RigidBody(model, pi)

    raise AttributeError(
        "RigidBodyMover does not expose get_rigid_body() or get_index(); "
        "cannot recover rigid body."
    )


def _extract_particle_indices_from_ji(ji, jm_initial: dict) -> Optional[List[int]]:
    """Try IMP/JAX API variants to get xyz-row -> IMP-particle index mapping."""
    candidate_methods = [
        "get_particle_indices",
        "get_particle_indexes",
    ]
    for method_name in candidate_methods:
        method = getattr(ji, method_name, None)
        if callable(method):
            values = list(method())
            if values:
                return [int(v) for v in values]

    candidate_attrs = [
        "particle_indices",
        "particle_indexes",
        "particle_index",
    ]
    for attr_name in candidate_attrs:
        if hasattr(ji, attr_name):
            values = getattr(ji, attr_name)
            try:
                values = list(values)
            except TypeError:
                continue
            if values:
                return [int(v) for v in values]

    candidate_jm_keys = [
        "particle_indices",
        "particle_indexes",
        "particle_index",
        "indexes",
        "indices",
    ]
    n_xyz = int(np.asarray(jm_initial["xyz"]).shape[0])
    for key in candidate_jm_keys:
        if key not in jm_initial:
            continue
        values = np.asarray(jm_initial[key]).reshape(-1)
        if values.shape[0] != n_xyz:
            continue
        if not np.issubdtype(values.dtype, np.integer):
            continue
        return [int(v) for v in values.tolist()]

    return None


def _build_particle_index_map_by_coordinates(
    dof: IMP.pmi.dof.DegreesOfFreedom,
    base_xyz: np.ndarray,
) -> Dict[int, int]:
    """
    Coordinate-based fallback when IMP/JAX does not expose particle indices.

    Matches only particles controlled by the DOF movers (RB + Ball). This is
    sufficient for mapping all parameters in the adapter.
    """
    pid_to_particle = {}

    for mover in dof.get_movers():
        if isinstance(mover, IMP.core.RigidBodyMover):
            rb = _get_rb_from_mover(mover)
            for leaf in IMP.core.get_leaves(rb.get_rigid_body_as_hierarchy()):
                pid_to_particle[int(leaf.get_index())] = leaf
        elif isinstance(mover, IMP.core.BallMover):
            for p in mover.get_particles():
                pid_to_particle[int(p.get_index())] = p

    if not pid_to_particle:
        return {}

    rounded_to_indices: Dict[Tuple[float, float, float], List[int]] = {}
    for jax_idx, xyz in enumerate(base_xyz):
        key = (round(float(xyz[0]), 6), round(float(xyz[1]), 6), round(float(xyz[2]), 6))
        rounded_to_indices.setdefault(key, []).append(jax_idx)

    pid_to_jax: Dict[int, int] = {}
    used_indices: set = set()

    for pid, particle in pid_to_particle.items():
        xyz = np.asarray(IMP.core.XYZ(particle).get_coordinates(), dtype=float)
        key = (round(float(xyz[0]), 6), round(float(xyz[1]), 6), round(float(xyz[2]), 6))

        candidates = rounded_to_indices.get(key, [])
        assigned = None
        while candidates:
            idx = candidates.pop()
            if idx not in used_indices:
                assigned = idx
                break

        if assigned is None:
            available = np.array(
                [i for i in range(base_xyz.shape[0]) if i not in used_indices],
                dtype=np.int32,
            )
            if available.size == 0:
                raise RuntimeError(
                    "Failed to map IMP particle indices to JAX xyz rows: no rows left."
                )
            deltas = base_xyz[available] - xyz[None, :]
            d2 = np.sum(deltas * deltas, axis=1)
            best_local = int(np.argmin(d2))
            best_d2 = float(d2[best_local])
            assigned = int(available[best_local])
            if best_d2 > 1e-6:
                raise RuntimeError(
                    "Failed to infer particle-index mapping from coordinates; "
                    f"min squared distance={best_d2:.3e}. "
                    "Please use an IMP build exposing get_particle_indices() or "
                    "provide particle indices in ji/jm."
                )

        pid_to_jax[pid] = assigned
        used_indices.add(assigned)

    return pid_to_jax


def _extract_rb_info(dof: IMP.pmi.dof.DegreesOfFreedom,
                     base_xyz: np.ndarray,
                     particle_index_map: Dict[int, int],
                     ) -> Tuple[List, np.ndarray]:
    """
    Walk the rigid bodies in `dof` and return:
      - A list of RB descriptors, each being a dict with keys:
          dof_offset  : int  – first index in the flat DOF vector for this RB
          n_dof       : int  – always 7 (3 translation + 4 quaternion)
          com         : np.ndarray shape (3,) – initial centre of mass
          quat_init   : np.ndarray shape (4,) – initial quaternion [w,x,y,z]
          member_jax_indices : list[int] – indices into jax xyz array
          member_local_xyz   : np.ndarray shape (M,3) – positions relative to COM
      - The residual xyz for non-RB particles (flexible beads etc.), shape (K, 3).
    """
    rb_descriptors = []
    dof_offset = 0

    handled_particle_ids: set = set()

    rb_movers = [m for m in dof.get_movers()
                 if isinstance(m, IMP.core.RigidBodyMover)]

    for mover in rb_movers:
        rb = _get_rb_from_mover(mover)
        members = IMP.core.get_leaves(rb.get_rigid_body_as_hierarchy())

        member_jax_idxs = []
        member_local = []
        com = np.array(rb.get_coordinates())

        for leaf in members:
            pid = leaf.get_index()
            if pid in particle_index_map:
                jax_idx = particle_index_map[pid]
                member_jax_idxs.append(jax_idx)
                handled_particle_ids.add(jax_idx)
                local = base_xyz[jax_idx] - com
                member_local.append(local)

        if not member_jax_idxs:
            continue

        rot = rb.get_reference_frame().get_transformation_to().get_rotation()
        qv = rot.get_quaternion()
        quat_init = np.array([qv[0], qv[1], qv[2], qv[3]])

        rb_descriptors.append({
            "dof_offset": dof_offset,
            "n_dof": 7,
            "com_init": com.copy(),
            "quat_init": quat_init.copy(),
            "member_jax_indices": member_jax_idxs,
            "member_local_xyz": np.array(member_local),
        })
        dof_offset += 7

    return rb_descriptors, handled_particle_ids, dof_offset


# ---------------------------------------------------------------------------
# Public DOF descriptor
# ---------------------------------------------------------------------------

@dataclass
class IMPDOFSpace:
    """
    Lightweight descriptor of the IMP system's degrees of freedom.

    Attributes
    ----------
    n_dof : int
        Total dimension of the flat position vector seen by BlackJAX.
    rb_descriptors : list
        One dict per rigid body (see _extract_rb_info).
    fb_jax_indices : list[int]
        JAX xyz-array indices of flexible-bead particles.
    fb_dof_offset : int
        Where flexible-bead coords start in the flat DOF vector.
    base_jm : dict
        Initial IMP JAX model dict (used as the constant template).
    """
    n_dof: int
    rb_descriptors: List[dict]
    fb_jax_indices: List[int]
    fb_dof_offset: int
    base_jm: dict

    @staticmethod
    def from_imp(dof: IMP.pmi.dof.DegreesOfFreedom,
                 ji,
                 jm_initial: dict) -> "IMPDOFSpace":
        """
        Build a DOF descriptor from an already-constructed IMP DOF object
        and the corresponding IMP JAX interface `ji`.

        Parameters
        ----------
        dof : IMP.pmi.dof.DegreesOfFreedom
        ji  : result of sf_imp._get_jax()
        jm_initial : dict returned by ji.get_jax_model()
        """
        base_xyz = np.array(jm_initial["xyz"])  # (N, 3)

        # Build particle-id -> jax-index map with IMP API compatibility.
        imp_particle_indices = _extract_particle_indices_from_ji(ji, jm_initial)
        if imp_particle_indices is not None:
            pid_to_jax = {
                int(pid): int(jax_idx)
                for jax_idx, pid in enumerate(imp_particle_indices)
            }
        else:
            pid_to_jax = _build_particle_index_map_by_coordinates(dof, base_xyz)

        rb_descriptors, handled, dof_offset = _extract_rb_info(
            dof, base_xyz, pid_to_jax
        )

        # Flexible bead movers
        fb_jax_indices = []
        fb_movers = [m for m in dof.get_movers()
                     if isinstance(m, IMP.core.BallMover)]
        for mover in fb_movers:
            for p in mover.get_particles():
                pid = p.get_index()
                if pid in pid_to_jax:
                    jax_idx = pid_to_jax[pid]
                    if jax_idx not in handled:
                        fb_jax_indices.append(jax_idx)

        n_fb = len(fb_jax_indices)
        n_dof = dof_offset + 3 * n_fb

        return IMPDOFSpace(
            n_dof=n_dof,
            rb_descriptors=rb_descriptors,
            fb_jax_indices=fb_jax_indices,
            fb_dof_offset=dof_offset,
            base_jm=jm_initial,
        )

    def encode(self) -> np.ndarray:
        """
        Read the current IMP particle positions and return the flat
        DOF vector (numpy, shape (n_dof,)).
        """
        vec = np.zeros(self.n_dof, dtype=np.float32)
        for rb in self.rb_descriptors:
            o = rb["dof_offset"]
            vec[o:o+3] = rb["com_init"]
            vec[o+3:o+7] = rb["quat_init"]

        base_xyz = np.array(self.base_jm["xyz"])
        for i, jax_idx in enumerate(self.fb_jax_indices):
            o = self.fb_dof_offset + 3 * i
            vec[o:o+3] = base_xyz[jax_idx]

        return vec


# ---------------------------------------------------------------------------
# Core adapter: encode / decode / score functions
# ---------------------------------------------------------------------------

class IMPSMCAdapter:
    """
    Wraps an IMP JAX scoring function for use with the BlackJAX SMC
    sampler in smc_base_sampler.py.

    Parameters
    ----------
    dof_space : IMPDOFSpace
        DOF descriptor built via IMPDOFSpace.from_imp().
    jax_score_func : callable
        Function  jm_dict -> scalar   (IMP JAX score; lower = better).
    kT : float
        Thermal energy for the Boltzmann weight.  log_likelihood = -score/kT.
    box_half_width : float
        Soft Gaussian prior half-width for translations (Å).
    rotation_prior_kappa : float
        Concentration for the vMF-like rotation prior on quaternion.
        0.0 = uniform on SO(3) approximated as isotropic Gaussian.
    """

    def __init__(
        self,
        dof_space: IMPDOFSpace,
        jax_score_func,
        kT: float = 1.0,
        box_half_width: float = 500.0,
        rotation_prior_kappa: float = 0.0,
    ):
        self.dof_space = dof_space
        self.kT = float(kT)
        self.box_half_width = float(box_half_width)
        self.rotation_prior_kappa = float(rotation_prior_kappa)

        # Freeze the base jax-model as a JAX constant dict.
        self._base_jm_jax = {
            k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
            for k, v in dof_space.base_jm.items()
        }
        self._base_xyz = jnp.asarray(np.array(dof_space.base_jm["xyz"]))  # (N, 3)

        # Pre-compute static arrays for JAX (no Python loops at score time).
        # Rigid-body member data
        self._rb_offsets = jnp.array(
            [rb["dof_offset"] for rb in dof_space.rb_descriptors], dtype=jnp.int32
        )
        self._rb_member_indices = [
            jnp.array(rb["member_jax_indices"], dtype=jnp.int32)
            for rb in dof_space.rb_descriptors
        ]
        self._rb_local_xyz = [
            jnp.array(rb["member_local_xyz"], dtype=jnp.float32)
            for rb in dof_space.rb_descriptors
        ]

        # Flexible bead indices
        self._fb_indices = jnp.array(dof_space.fb_jax_indices, dtype=jnp.int32)
        self._fb_dof_offset = dof_space.fb_dof_offset

        self._jax_score_func = jax_score_func
        self._build_jax_fns()

    # ------------------------------------------------------------------
    # Build JIT-compiled functions once at construction
    # ------------------------------------------------------------------

    def _build_jax_fns(self) -> None:
        base_xyz = self._base_xyz
        base_jm = self._base_jm_jax
        rb_offsets = self._rb_offsets
        rb_member_indices = self._rb_member_indices
        rb_local_xyz = self._rb_local_xyz
        fb_indices = self._fb_indices
        fb_dof_offset = self._fb_dof_offset
        jax_score_func = self._jax_score_func
        kT = self.kT
        box_hw = self.box_half_width

        def _flat_to_xyz(flat: jnp.ndarray) -> jnp.ndarray:
            """Decode flat DOF vector -> full particle xyz array."""
            xyz = base_xyz  # start from template

            # --- rigid bodies ---
            for i, (rb_idxs, local) in enumerate(
                zip(rb_member_indices, rb_local_xyz)
            ):
                o = rb_offsets[i]
                translation = flat[o : o + 3]          # (3,)
                raw_quat    = flat[o + 3 : o + 7]      # (4,)
                quat        = _normalise_quat(raw_quat)
                R           = _quat_to_rotation_matrix(quat)   # (3,3)
                rotated     = (R @ local.T).T           # (M, 3)
                world_xyz   = rotated + translation[None, :]    # (M, 3)
                xyz = xyz.at[rb_idxs].set(world_xyz)

            # --- flexible beads ---
            if fb_indices.shape[0] > 0:
                n_fb = fb_indices.shape[0]
                fb_coords = flat[fb_dof_offset : fb_dof_offset + 3 * n_fb]
                fb_coords  = fb_coords.reshape(n_fb, 3)
                xyz = xyz.at[fb_indices].set(fb_coords)

            return xyz

        def _score(flat: jnp.ndarray) -> jnp.ndarray:
            xyz = _flat_to_xyz(flat)
            jm = {**base_jm, "xyz": xyz}
            return jax_score_func(jm)

        def _log_likelihood(flat: jnp.ndarray) -> jnp.ndarray:
            """BlackJAX log_likelihood_fn: -IMP_score / kT."""
            return -_score(flat) / kT

        def _log_prior(flat: jnp.ndarray) -> jnp.ndarray:
            """
            Independent Gaussian prior on translations, uniform on
            quaternion orientations (approximated as isotropic Gaussian
            on the raw 4-vector before normalisation).
            """
            log_p = jnp.float32(0.0)
            for i in range(len(rb_offsets)):
                o = rb_offsets[i]
                t = flat[o : o + 3]
                # Soft harmonic wall at ±box_hw Å
                log_p = log_p - jnp.sum(jnp.where(
                    jnp.abs(t) > box_hw,
                    0.5 * ((jnp.abs(t) - box_hw) / (0.1 * box_hw)) ** 2,
                    0.0,
                ))
            if fb_indices.shape[0] > 0:
                n_fb = fb_indices.shape[0]
                fb = flat[fb_dof_offset : fb_dof_offset + 3 * n_fb]
                log_p = log_p - jnp.sum(jnp.where(
                    jnp.abs(fb) > box_hw,
                    0.5 * ((jnp.abs(fb) - box_hw) / (0.1 * box_hw)) ** 2,
                    0.0,
                ))
            return log_p

        def _log_prob(flat: jnp.ndarray) -> jnp.ndarray:
            return _log_prior(flat) + _log_likelihood(flat)

        self._flat_to_xyz = jax.jit(_flat_to_xyz)
        self._score_jit   = jax.jit(_score)
        self.log_likelihood = jax.jit(_log_likelihood)
        self.log_prior      = jax.jit(_log_prior)
        self.log_prob       = jax.jit(_log_prob)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self) -> jnp.ndarray:
        """
        Read the current IMP DOF state and return a flat JAX array
        of shape (n_dof,).
        """
        return jnp.asarray(self.dof_space.encode())

    def decode_xyz(self, flat: jnp.ndarray) -> np.ndarray:
        """
        Given a flat DOF vector, return the full particle xyz array
        as a numpy array of shape (N, 3).
        """
        return np.array(self._flat_to_xyz(flat))

    def imp_score(self, flat: jnp.ndarray) -> float:
        """Return the raw IMP energy for a flat DOF vector."""
        return float(self._score_jit(flat))

    def sample_prior(
        self,
        n_particles: int,
        rng_key: jax.Array,
        translation_sigma: float = 200.0,
        rotation_jitter: float = 0.1,
    ) -> jnp.ndarray:
        """
        Draw `n_particles` initial positions from the prior.

        Translations: sampled from N(current_COM, translation_sigma²).
        Quaternions:  current quaternion + small Gaussian noise, then
                      normalised (approximates uniform SO(3) for large noise).
        Flexible beads: sampled from N(current_xyz, translation_sigma²).

        Returns
        -------
        positions : jnp.ndarray  shape (n_particles, n_dof)
        """
        base = self.dof_space.encode()  # (n_dof,)
        n_dof = self.dof_space.n_dof

        key, subkey = jax.random.split(rng_key)
        noise = jax.random.normal(subkey, shape=(n_particles, n_dof)) * translation_sigma

        # Start from the encoded initial configuration, broadcast to particles.
        positions = jnp.broadcast_to(jnp.asarray(base), (n_particles, n_dof)) + noise

        # Re-normalise the quaternion slice for each rigid body.
        for rb in self.dof_space.rb_descriptors:
            o = rb["dof_offset"]
            q_slice = positions[:, o + 3 : o + 7]
            norms = jnp.linalg.norm(q_slice, axis=-1, keepdims=True) + 1e-12
            positions = positions.at[:, o + 3 : o + 7].set(q_slice / norms)

        return positions

    def suggested_rmh_sigma(self) -> float:
        """
        A heuristic RMH step size: ~2 Å for coarse proteins.
        Rotate by ~0.05 rad per step.
        """
        return 5.0

    @property
    def n_dof(self) -> int:
        return self.dof_space.n_dof

    def dof_summary(self) -> str:
        lines = [
            f"IMPSMCAdapter DOF summary",
            f"  Total DOF    : {self.dof_space.n_dof}",
            f"  Rigid bodies : {len(self.dof_space.rb_descriptors)} "
            f"(each 7 DOF: 3 translation + 4 quaternion)",
            f"  Flexible bead particles: {len(self.dof_space.fb_jax_indices)} "
            f"(each 3 DOF)",
            f"  kT           : {self.kT}",
            f"  Box half-width: {self.box_half_width} Å",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience function: run SMC on an IMP system
# ---------------------------------------------------------------------------

def run_smc_on_imp_system(
    adapter: "IMPSMCAdapter",
    rng_key: jax.Array,
    n_particles: int = 500,
    n_temperature_steps: int = 30,
    schedule: str = "geometric",
    kernel: str = "rmh",
    rmh_sigma: Optional[float] = None,
    hmc_step_size: float = 0.05,
    hmc_num_integration_steps: int = 5,
    n_mcmc_steps: int = 10,
    score_batch_size: int = 32,
    verbose: bool = True,
):
    """
    High-level wrapper: sample an IMP system with BlackJAX base SMC.

    Parameters
    ----------
    adapter : IMPSMCAdapter
    rng_key : jax.Array
    n_particles : int
        Number of SMC particles (population size).
    n_temperature_steps : int
        Number of tempering steps (lambda 0 → 1).
    schedule : str
        'linear', 'geometric', or 'sigmoid'.
    kernel : str
        'rmh' or 'hmc'.
    rmh_sigma : float | None
        RMH step size; if None uses adapter.suggested_rmh_sigma().
    hmc_step_size : float
    hmc_num_integration_steps : int
    n_mcmc_steps : int
        MCMC sweeps per SMC step.
    score_batch_size : int
        Batch size for particle scoring (memory control).
    verbose : bool

    Returns
    -------
    state, info_history, best_positions, best_scores, lambdas
        Same as smc_base_sampler.run_base_smc_rmh / run_base_smc_hmc.
    """
    # Deferred import to avoid circular dependency.
    from .smc_base_sampler import run_base_smc_rmh, run_base_smc_hmc

    if verbose:
        print(adapter.dof_summary())

    key_init, key_smc = jax.random.split(rng_key)
    initial_positions = adapter.sample_prior(
        n_particles=n_particles,
        rng_key=key_init,
        translation_sigma=150.0,
    )

    if verbose:
        print(f"\nInitial positions shape: {initial_positions.shape}")
        example_score = adapter.imp_score(initial_positions[0])
        print(f"Example IMP score (particle 0): {example_score:.2f}")

    common_kwargs = dict(
        log_prior_fn=adapter.log_prior,
        log_likelihood_fn=adapter.log_likelihood,
        log_prob_fn=adapter.log_prob,
        initial_positions=initial_positions,
        rng_key=key_smc,
        n_temperature_steps=n_temperature_steps,
        schedule=schedule,
        n_mcmc_steps=n_mcmc_steps,
        record_best=True,
        verbose=verbose,
        score_batch_size=score_batch_size,
    )

    if kernel == "rmh":
        sigma = rmh_sigma if rmh_sigma is not None else adapter.suggested_rmh_sigma()
        return run_base_smc_rmh(rmh_sigma=sigma, **common_kwargs)
    elif kernel == "hmc":
        return run_base_smc_hmc(
            hmc_step_size=hmc_step_size,
            hmc_num_integration_steps=hmc_num_integration_steps,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'rmh' or 'hmc'.")
