"""
sampling/wrapper_imp_blackjax.py
================================

Adapter that lets IMP's *structured* JAX model (a dict with ``'xyz'``,
``'r'``, ``'rigid_bodies'``, ...) be sampled with BlackJAX SMC kernels
that operate on flat ``(n_dims,)`` particle vectors.

Two layers
----------
* :class:`IMPDOFSpace` -- pure pytree-level mapping between the flat
  vector and the IMP JAX model.  No scoring, no sampling.
* :class:`IMPSMCAdapter` -- holds the JAX score function and exposes
  the ``log_prior`` / ``log_likelihood`` / ``log_prob`` callables that
  the BlackJAX SMC runner expects.

A convenience function :func:`run_smc_on_imp_system` wires the adapter
into :func:`sampling.smc_base_sampler.run_base_smc_rmh` (or the HMC
variant).

The intended usage in a downstream PSD-modeling repo is::

    from sampling.imp_blackjax_adapter import (
        IMPDOFSpace, IMPSMCAdapter, run_smc_on_imp_system,
    )

    # ... build your IMP system, restraints, JAX score function ...
    ji, jax_score_func, jm_initial = build_jax_score(sf_imp)

    dof_space = IMPDOFSpace.from_imp(dof, ji, jm_initial)
    adapter   = IMPSMCAdapter(dof_space, jax_score_func, kT=1.0,
                              box_half_width=500.0)

    state, info, best_pos, best_scores, lambdas = run_smc_on_imp_system(
        adapter, jax.random.PRNGKey(0),
        n_particles=200, n_temperature_steps=30,
        kernel="rmh", rmh_sigma=5.0, n_mcmc_steps=5,
    )
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import tempfile
import os

import jax
import jax.numpy as jnp
import numpy as np

# Reuse the BlackJAX runners that already live in this package.
# (Relative import so the adapter can be installed as part of jax_bhm.)
from .smc_base_sampler import (
    run_base_smc_rmh,
    run_base_smc_hmc,
)

from .rmh import (
    run_rmh_sampling
)

# =============================================================================
# DOF space: flat-vector <-> structured IMP JAX model
# =============================================================================

@dataclass
class IMPDOFSpace:
    """
    Bidirectional mapping between IMP's structured JAX model dict and a
    flat 1-D parameter vector that BlackJAX SMC kernels can sample over.

    Layout of the flat vector
    -------------------------
    ::

        [ rb_translations  (n_rb, 3)  flatten ]   # 3 * n_rb entries
        [ rb_quaternions   (n_rb, 4)  flatten ]   # 4 * n_rb entries
        [ flex_xyz         (n_fb, 3)  flatten ]   # 3 * n_fb entries

        n_dims = 3 * n_rb + 4 * n_rb + 3 * n_fb
               = 7 * n_rb + 3 * n_fb

    The static parts of ``jm`` -- particle radii, internal coordinates of
    rigid-body members, the ``_AllRigidBodies`` topology, etc. -- are kept
    in :attr:`template_jm` and reused on every :meth:`decode` call.

    Notes
    -----
    * Quaternions are renormalised inside :meth:`decode` so that Gaussian
      RMH/HMC proposals still produce valid unit rotations.  The Jacobian
      term that would correct for the projection from R^4 to S^3 is
      omitted; this is harmless for SMC tempering aimed at mode-finding,
      but it does introduce a small bias in the rotational density.
    * Rigid-body members ("slaved" particles) are *not* given their own
      DOFs.  Their positions are recomputed from the parent's
      (translation, quaternion) on every :meth:`decode` call.
    * Nested rigid bodies are supported as long as the topology stored
      in ``jm['rigid_bodies'].bodies`` is in topological order
      (parents before children), which is the IMP default.
    """

    # ---- topology ----------------------------------------------------------
    rb_particle_indexes: np.ndarray   # (n_rb,)  IMP particle index of each RB
    n_rb: int                          # number of rigid bodies
    flex_particle_indexes: np.ndarray  # (n_fb,)  IMP particle indices of free xyz
    n_fb: int                          # number of free / flexible beads

    # ---- static part of the JAX model --------------------------------------
    template_jm: dict                  # {'xyz', 'r', 'rigid_bodies', ...}

    # ---- cached layout of the flat vector ----------------------------------
    rb_trans_slice: slice
    rb_quat_slice: slice
    flex_slice: slice
    n_dims: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_imp(
        cls,
        dof: Optional[Any],
        ji: Any,
        jm_initial: dict,
        flex_particle_indexes: Optional[Sequence[int]] = None,
    ) -> "IMPDOFSpace":
        """
        Build an :class:`IMPDOFSpace` from an IMP system.

        Parameters
        ----------
        dof
            A PMI ``DegreesOfFreedom`` object, or anything else.  Currently
            unused -- kept in the signature so a future implementation can
            cross-check the inferred layout against PMI's own DOF tracking
            without breaking callers.  Pass ``None`` if you don't have one.
        ji
            The :class:`IMP._jax_util.JAXRestraintInfo` returned by
            ``RestraintsScoringFunction._get_jax()``.  Currently unused
            here (the score function is held by :class:`IMPSMCAdapter`),
            but accepted to keep the call site self-documenting.
        jm_initial
            The IMP JAX model dict, typically produced via
            ``IMP._jax_util._get_jax_model(model, ['rigid_bodies'])``
            or your local ``materialize_jax_model_arrays()``.  Must
            contain at least ``'xyz'`` and -- if rigid bodies exist --
            ``'rigid_bodies'``.
        flex_particle_indexes
            Optional explicit list of particle indices in ``jm['xyz']``
            that should get their own (3,) DOF block.  If ``None``,
            every particle that is *not* a member of any rigid body is
            treated as a flexible bead.
        """
        del dof, ji  # currently unused, see docstring

        # ---- rigid-body indices --------------------------------------------
        if 'rigid_bodies' in jm_initial:
            allrbs = jm_initial['rigid_bodies']
            rb_particle_indexes = np.array(
                [int(b.particle_index) for b in allrbs.bodies],
                dtype=np.int64,
            )
        else:
            rb_particle_indexes = np.zeros((0,), dtype=np.int64)

        n_rb = int(rb_particle_indexes.shape[0])

        # ---- flexible beads -------------------------------------------------
        if flex_particle_indexes is None:
            # Every particle that participates in any rigid body
            # (root or member) is considered "slaved".
            slaved: set = set()
            if 'rigid_bodies' in jm_initial:
                for b in jm_initial['rigid_bodies'].bodies:
                    slaved.add(int(b.particle_index))
                    slaved.update(int(pi) for pi in
                                  np.asarray(b.member_particle_indexes))
            n_total_xyz = int(jm_initial['xyz'].shape[0])
            flex_idx = np.array(
                [pi for pi in range(n_total_xyz) if pi not in slaved],
                dtype=np.int64,
            )
        else:
            flex_idx = np.asarray(flex_particle_indexes, dtype=np.int64)

        n_fb = int(flex_idx.shape[0])

        # ---- flat-vector layout --------------------------------------------
        a = 0
        rb_trans_slice = slice(a, a + 3 * n_rb); a += 3 * n_rb
        rb_quat_slice  = slice(a, a + 4 * n_rb); a += 4 * n_rb
        flex_slice     = slice(a, a + 3 * n_fb); a += 3 * n_fb
        n_dims = a

        return cls(
            rb_particle_indexes=rb_particle_indexes,
            n_rb=n_rb,
            flex_particle_indexes=flex_idx,
            n_fb=n_fb,
            template_jm=jm_initial,
            rb_trans_slice=rb_trans_slice,
            rb_quat_slice=rb_quat_slice,
            flex_slice=flex_slice,
            n_dims=n_dims,
        )

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def _slice(self, flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Split a flat (n_dims,) vector into (rb_trans, rb_quat, flex_xyz)."""
        rb_t = flat[self.rb_trans_slice].reshape(self.n_rb, 3)
        rb_q = flat[self.rb_quat_slice].reshape(self.n_rb, 4)
        flex = flat[self.flex_slice].reshape(self.n_fb, 3)
        return rb_t, rb_q, flex

    def encode(self, jm: Optional[dict] = None) -> jnp.ndarray:
        """
        Pack the variable parts of an IMP JAX model into a flat
        ``(n_dims,)`` vector.  If ``jm`` is omitted, the template is used.
        """
        if jm is None:
            jm = self.template_jm

        if self.n_rb > 0:
            rb_idx = jnp.asarray(self.rb_particle_indexes)
            rb_t = jm['xyz'][rb_idx]                        # (n_rb, 3)
            rb_q = jm['rigid_bodies'].quaternion            # (n_rb, 4)
        else:
            rb_t = jnp.zeros((0, 3))
            rb_q = jnp.zeros((0, 4))

        if self.n_fb > 0:
            flex_idx = jnp.asarray(self.flex_particle_indexes)
            flex = jm['xyz'][flex_idx]                       # (n_fb, 3)
        else:
            flex = jnp.zeros((0, 3))

        return jnp.concatenate([
            rb_t.reshape(-1),
            rb_q.reshape(-1),
            flex.reshape(-1),
        ])

    def decode(self, flat: jnp.ndarray) -> dict:
        """
        Decode a flat ``(n_dims,)`` vector back into a fully-consistent
        IMP JAX model dict.  This:

        1. Renormalises every quaternion to unit length.
        2. Writes the new translations and quaternions into a *copy* of
           the template (the template itself is never mutated).
        3. Calls ``body.update_members(jm)`` for each rigid body so the
           global xyz coordinates of slaved particles are consistent
           with the new (translation, quaternion) of the parent.

        The returned dict can be passed directly to the JAX score
        function returned by ``RestraintsScoringFunction._get_jax()``.
        """
        rb_t, rb_q, flex = self._slice(flat)

        # ---- renormalise quaternions  (keeps them on S^3) ------------------
        if self.n_rb > 0:
            qnorm = jnp.linalg.norm(rb_q, axis=1, keepdims=True)
            qnorm = jnp.where(qnorm > 1e-12, qnorm, 1.0)
            rb_q_unit = rb_q / qnorm
        else:
            rb_q_unit = rb_q

        # ---- shallow-copy the template, then patch the variable slots -----
        # Shallow copy is safe because we replace whole arrays / dataclass
        # instances rather than mutating in place.
        jm = dict(self.template_jm)

        # Replace 'rigid_bodies' with a new dataclass instance whose
        # quaternion field is the new (unit) quaternions.  bodies / intcoord
        # / rb_index_from_particle are shared with the template.
        if self.n_rb > 0:
            jm['rigid_bodies'] = dataclasses.replace(
                jm['rigid_bodies'],
                quaternion=rb_q_unit,
            )

        # Patch xyz: rigid-body roots first, then flexible beads.
        # Some IMP JAX model builders return NumPy arrays; convert once so
        # `.at[...]` indexed updates are always available.
        xyz = jnp.asarray(jm['xyz'])
        if self.n_rb > 0:
            xyz = xyz.at[jnp.asarray(self.rb_particle_indexes)].set(rb_t)
        if self.n_fb > 0:
            xyz = xyz.at[jnp.asarray(self.flex_particle_indexes)].set(flex)
        jm['xyz'] = xyz

        # Propagate every rigid body onto its members.  Iterating in storage
        # order works for nested bodies because IMP stores parents before
        # children.
        if self.n_rb > 0:
            for body in jm['rigid_bodies'].bodies:
                jm = body.update_members(jm)

        return jm

    def decode_xyz(self, flat: jnp.ndarray) -> jnp.ndarray:
        """Return the ``(n_total_xyz, 3)`` global xyz array for ``flat``."""
        return self.decode(flat)['xyz']


# =============================================================================
# Adapter: provides log_prior, log_likelihood, log_prob over flat vectors
# =============================================================================

class IMPSMCAdapter:
    """
    Wrap an :class:`IMPDOFSpace` together with an IMP JAX score function
    so that BlackJAX SMC kernels can sample the IMP posterior.

    Parameters
    ----------
    dof_space
        :class:`IMPDOFSpace` describing how to map flat vectors onto
        the IMP JAX model.
    jax_score_func
        Callable ``jm -> scalar``.  Typically
        ``ji.score_func`` where ``ji = sf._get_jax()``.  IMP score
        conventions are *minimised*, so we treat
        ``log_likelihood = -score / kT``.
    kT
        Thermal energy in IMP score units.  ``1.0`` is a reasonable
        default for IMP-PMI restraints; use ``~0.6`` if your scores
        are calibrated in kcal/mol at 300 K.
    box_half_width
        Soft uniform-cube prior half-width (Angstrom).  Translations
        and flexible-bead positions outside the box are penalised
        with a quadratic term -- this keeps the chain from drifting
        to infinity at low ``lambda`` values.
    box_prior_sigma
        Standard deviation of the soft-wall penalty.  Default 50 A.
    quat_prior_sigma
        Standard deviation of the soft prior on ``||q|| - 1``.  This
        is purely a regulariser to keep raw quaternion magnitudes
        well-scaled relative to ``rmh_sigma``; the rotation itself
        is unaffected because :meth:`IMPDOFSpace.decode` renormalises.
    """

    def __init__(
        self,
        dof_space: IMPDOFSpace,
        jax_score_func: Callable,
        kT: float = 1.0,
        box_half_width: float = 500.0,
        box_prior_sigma: float = 50.0,
        quat_prior_sigma: float = 0.5,
    ):
        self.dof_space = dof_space
        self.jax_score_func = jax_score_func
        self.kT = float(kT)
        self.box_half_width = float(box_half_width)
        self.box_prior_sigma = float(box_prior_sigma)
        self.quat_prior_sigma = float(quat_prior_sigma)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def dof_summary(self) -> str:
        s = self.dof_space
        lines = [
            "IMPSMCAdapter DOF summary",
            f"  Total DOF              : {s.n_dims}",
            f"  Rigid bodies           : {s.n_rb}  (each 7 DOF: 3 translation + 4 quaternion)",
            f"  Flexible bead particles: {s.n_fb}  (each 3 DOF)",
            f"  kT                     : {self.kT}",
            f"  Box half-width         : {self.box_half_width} A",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Encoding / decoding (delegates to IMPDOFSpace)
    # ------------------------------------------------------------------
    def encode(self, jm: Optional[dict] = None) -> jnp.ndarray:
        return self.dof_space.encode(jm)

    def decode(self, flat: jnp.ndarray) -> dict:
        return self.dof_space.decode(flat)

    def decode_xyz(self, flat: jnp.ndarray) -> jnp.ndarray:
        return self.dof_space.decode_xyz(flat)

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------
    def imp_score(self, flat: jnp.ndarray) -> jnp.ndarray:
        """Raw IMP score (sum of restraint penalties).  Lower is better."""
        jm = self.decode(flat)
        return self.jax_score_func(jm)

    def log_likelihood(self, flat: jnp.ndarray) -> jnp.ndarray:
        """Boltzmann log-likelihood:  ``-score(flat) / kT``."""
        return -self.imp_score(flat) / self.kT

    def log_prior(self, flat: jnp.ndarray) -> jnp.ndarray:
        """
        Soft uniform-box prior on translations + flexible beads, plus
        a soft prior on ``||q|| ~ 1`` for each rigid body's quaternion.
        Returns a *log* density (already negative outside the box).
        """
        rb_t, rb_q, flex = self.dof_space._slice(flat)

        # ---- soft cube prior on positions -------------------------------
        pos = jnp.concatenate([rb_t.reshape(-1), flex.reshape(-1)])
        if pos.size > 0:
            excess = jnp.maximum(jnp.abs(pos) - self.box_half_width, 0.0)
            box_term = -0.5 * jnp.sum(excess ** 2) / (self.box_prior_sigma ** 2)
        else:
            box_term = jnp.array(0.0)

        # ---- soft norm prior on raw quaternions -------------------------
        if self.dof_space.n_rb > 0:
            qnorm = jnp.linalg.norm(rb_q, axis=1)
            quat_term = -0.5 * jnp.sum((qnorm - 1.0) ** 2) / (self.quat_prior_sigma ** 2)
        else:
            quat_term = jnp.array(0.0)

        return box_term + quat_term

    def log_prob(self, flat: jnp.ndarray) -> jnp.ndarray:
        """Full log-posterior: ``log_prior(flat) + log_likelihood(flat)``."""
        return self.log_prior(flat) + self.log_likelihood(flat)

    # ------------------------------------------------------------------
    # Initial particle population
    # ------------------------------------------------------------------
    def init_particles(
        self,
        rng_key: jax.Array,
        n_particles: int,
        trans_jitter: float = 5.0,
        quat_jitter: float = 0.1,
        flex_jitter: float = 2.0,
        from_prior: bool = False,
    ) -> jnp.ndarray:
        """
        Build an ``(n_particles, n_dims)`` array of starting particles.

        Parameters
        ----------
        rng_key
            JAX PRNGKey.
        n_particles
            Number of SMC particles to draw.
        trans_jitter
            Std-dev (Angstrom) of Gaussian noise added to rigid-body
            translations around the encoded current state.
        quat_jitter
            Std-dev of Gaussian noise added to raw quaternion
            components.  ~0.1 corresponds to a few degrees of rotation.
        flex_jitter
            Std-dev (Angstrom) of Gaussian noise added to flexible
            beads.
        from_prior
            If ``True``, draw translations and flexible beads
            uniformly inside the box and quaternions uniformly on
            S^3 (true prior samples).  Otherwise, jitter around the
            encoded current state (recommended for refinement runs).
        """
        s = self.dof_space
        if from_prior:
            keys = jax.random.split(rng_key, 3)
            rb_t = jax.random.uniform(
                keys[0], (n_particles, s.n_rb, 3),
                minval=-self.box_half_width, maxval=self.box_half_width,
            ).reshape(n_particles, -1)
            # Uniform on S^3: sample a Gaussian and normalise.
            rb_q_raw = jax.random.normal(keys[1], (n_particles, s.n_rb, 4))
            qnorm = jnp.linalg.norm(rb_q_raw, axis=-1, keepdims=True)
            rb_q_raw = rb_q_raw / jnp.where(qnorm > 1e-12, qnorm, 1.0)
            rb_q = rb_q_raw.reshape(n_particles, -1)
            flex = jax.random.uniform(
                keys[2], (n_particles, s.n_fb, 3),
                minval=-self.box_half_width, maxval=self.box_half_width,
            ).reshape(n_particles, -1)
            return jnp.concatenate([rb_t, rb_q, flex], axis=1)

        # Jitter around current encoded state.
        flat0 = self.encode()
        keys = jax.random.split(rng_key, 3)
        rb_t_jit = jax.random.normal(keys[0], (n_particles, 3 * s.n_rb)) * trans_jitter
        rb_q_jit = jax.random.normal(keys[1], (n_particles, 4 * s.n_rb)) * quat_jitter
        flex_jit = jax.random.normal(keys[2], (n_particles, 3 * s.n_fb)) * flex_jitter
        jitter = jnp.concatenate([rb_t_jit, rb_q_jit, flex_jit], axis=1)
        return flat0[None, :] + jitter


# =============================================================================
# Convenience runner
# =============================================================================

def run_smc_on_imp_system(
    adapter: IMPSMCAdapter,
    rng_key: jax.Array,
    n_particles: int = 200,
    n_temperature_steps: int = 30,
    schedule: str = "geometric",
    kernel: str = "rmh",
    # ---- RMH knobs (used if kernel == 'rmh')
    rmh_sigma: float = 5.0,
    # ---- HMC knobs (used if kernel == 'hmc')
    hmc_step_size: float = 0.01,
    hmc_inverse_mass_matrix: Optional[jnp.ndarray] = None,
    hmc_num_integration_steps: int = 10,
    # ---- Common
    n_mcmc_steps: int = 5,
    score_batch_size: int = 50,
    # ---- Initial particles
    init_from_prior: bool = False,
    init_trans_jitter: float = 5.0,
    init_quat_jitter: float = 0.1,
    init_flex_jitter: float = 2.0,
    # ---- Misc
    verbose: bool = True,
) -> Tuple[Any, List[Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end SMC sampling of an IMP system.

    This:

    1. Builds an ``(n_particles, n_dims)`` initial population from the
       adapter (either by jittering around the current state or by
       drawing from the soft prior).
    2. Wraps :meth:`IMPSMCAdapter.log_prior`, :meth:`log_likelihood` and
       :meth:`log_prob` with ``jax.jit``.
    3. Calls :func:`sampling.smc_base_sampler.run_base_smc_rmh` (or the
       HMC variant) to do the actual sampling.

    Returns
    -------
    state
        Final BlackJAX ``SMCState`` -- ``state.particles`` is shape
        ``(n_particles, n_dims)``.
    info_history
        List of per-step ``SMCInfo`` objects from BlackJAX.
    best_positions
        ``(n_temperature_steps + 1, n_dims)`` -- best particle at each
        temperature step (incl. initial).  Decode any row with
        ``adapter.decode_xyz(...)`` to get full xyz.
    best_scores
        ``(n_temperature_steps + 1,)``  matching ``log_prob`` values.
    lambdas
        Temperature schedule actually used.
    """
    init_key, sample_key = jax.random.split(rng_key)

    # ---- initial particles -------------------------------------------------
    initial_positions = adapter.init_particles(
        init_key,
        n_particles=n_particles,
        trans_jitter=init_trans_jitter,
        quat_jitter=init_quat_jitter,
        flex_jitter=init_flex_jitter,
        from_prior=init_from_prior,
    )

    # ---- jitted scalar-particle scoring functions --------------------------
    log_prior_fn      = jax.jit(adapter.log_prior)
    log_likelihood_fn = jax.jit(adapter.log_likelihood)
    log_prob_fn       = jax.jit(adapter.log_prob)

    if verbose:
        print(adapter.dof_summary())
        print(f"  Initial population shape: {initial_positions.shape}")

    # ---- dispatch to the BlackJAX-base-SMC runner --------------------------
    if kernel == "rmh":
        return run_base_smc_rmh(
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_prob_fn=log_prob_fn,
            initial_positions=initial_positions,
            rng_key=sample_key,
            n_temperature_steps=n_temperature_steps,
            schedule=schedule,
            rmh_sigma=rmh_sigma,
            n_mcmc_steps=n_mcmc_steps,
            score_batch_size=score_batch_size,
            verbose=verbose,
        )
    elif kernel == "hmc":
        return run_base_smc_hmc(
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_prob_fn=log_prob_fn,
            initial_positions=initial_positions,
            rng_key=sample_key,
            n_temperature_steps=n_temperature_steps,
            schedule=schedule,
            hmc_step_size=hmc_step_size,
            hmc_inverse_mass_matrix=hmc_inverse_mass_matrix,
            hmc_num_integration_steps=hmc_num_integration_steps,
            n_mcmc_steps=n_mcmc_steps,
            score_batch_size=score_batch_size,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Use 'rmh' or 'hmc'.")
    
def run_rmh_on_imp_system(adapter, sample_key, rmh_sigma, n_mcmc_steps, imp_model, save_rmf3_path=None, verbose=False):
    """
    1) Run the Random-walk Metropolis-Hastings (RMH) algorithm on an IMP system.
    2) Return the final state, best positions, and best scores.
    3) Optionally save the trajectory to an RMF3 file if a path is provided.
    4) To save the RMF3 file, a helper function that just saves a h5py file which
    can later be converted to an RMF3 file, the h5py file should store information
    similar to what IMP does with its model hierarchy
    """
    
    