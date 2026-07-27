"""
Minimal modular IMP <-> BlackJAX wrapper.

Current scope
-------------
- Flexible bead coordinates only.
- Random-walk Metropolis-Hastings (RMH) using BlackJAX.
- Basic fixed-schedule SMC for comparing convergence against RMH.
- Adaptive tempered SMC (BlackJAX adaptive schedule).

Design goal
-----------
Keep the code small and easy to extend:
- Add rigid-body sampling by creating another parameter block.
- Add nuisance parameter sampling by creating another parameter block.

The sampler only sees a flat vector. Parameter blocks are responsible for
packing/unpacking between that flat vector and IMP model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple, Union

import time

import blackjax
import blackjax.mcmc.random_walk as random_walk
import jax
import jax.numpy as jnp
import numpy as np

import IMP
import IMP.algebra
import IMP.core


ArrayLike = Union[float, jnp.ndarray, np.ndarray]


class ParameterBlock(Protocol):
    """Protocol for a modular parameter block."""

    @property
    def size(self) -> int:
        """Number of scalar DOFs contributed by this block."""

    def pack(self) -> jnp.ndarray:
        """Read parameters from IMP and return a flat vector."""

    def unpack(self, flat: jnp.ndarray) -> None:
        """Write a block-specific flat vector back into IMP."""


@dataclass
class FlexibleBeadBlock:
    """Coordinate block for flexible beads (3 DOF per bead)."""

    model: Any
    particle_indices: Sequence[int]

    def __post_init__(self) -> None:
        self.particle_indices = [int(i) for i in self.particle_indices]
        if len(self.particle_indices) == 0:
            raise ValueError("FlexibleBeadBlock needs at least one particle index.")
        pidx = [IMP.ParticleIndex(i) for i in self.particle_indices]
        self._particles = IMP.get_particles(self.model, pidx)

    @property
    def size(self) -> int:
        return 3 * len(self._particles)

    def pack(self) -> jnp.ndarray:
        coords = []
        for p in self._particles:
            xyz = IMP.core.XYZ(p)
            c = xyz.get_coordinates()
            coords.extend([float(c[0]), float(c[1]), float(c[2])])
        return jnp.asarray(coords, dtype=jnp.float32)

    def unpack(self, flat: jnp.ndarray) -> None:
        arr = np.asarray(flat, dtype=np.float64).reshape(len(self._particles), 3)
        for particle, xyz_new in zip(self._particles, arr):
            xyz = IMP.core.XYZ(particle)
            xyz.set_coordinates(IMP.algebra.Vector3D(*xyz_new.tolist()))


@dataclass
class IMPParameterSpace:
    """Composable parameter space built from parameter blocks."""

    blocks: Sequence[ParameterBlock]

    def __post_init__(self) -> None:
        self.blocks = list(self.blocks)
        if len(self.blocks) == 0:
            raise ValueError("IMPParameterSpace requires at least one parameter block.")
        self._sizes = [int(b.size) for b in self.blocks]
        self._offsets = np.cumsum([0] + self._sizes)

    @property
    def dim(self) -> int:
        return int(sum(self._sizes))

    def pack(self) -> jnp.ndarray:
        return jnp.concatenate([b.pack().reshape(-1) for b in self.blocks], axis=0)

    def unpack(self, flat: jnp.ndarray) -> None:
        flat = jnp.asarray(flat).reshape(-1)
        if flat.shape[0] != self.dim:
            raise ValueError(f"Expected flat vector of size {self.dim}, got {flat.shape[0]}.")

        for i, block in enumerate(self.blocks):
            start = int(self._offsets[i])
            stop = int(self._offsets[i + 1])
            block.unpack(flat[start:stop])


@dataclass
class RMHResult:
    positions: np.ndarray
    log_probs: np.ndarray
    accepted: np.ndarray
    acceptance_rate: float


@dataclass
class DebugEvaluation:
    score: float
    log_prior: float
    log_posterior: float


class IMPLogPosterior:
    """
    Log-posterior adapter for BlackJAX.

    The IMP score function is minimized, so log-posterior is:
      log p(theta) = -score(theta) / temperature + log_prior(theta)
    """

    def __init__(
        self,
        parameter_space: IMPParameterSpace,
        score_fn: Callable[[], float],
        temperature: float = 1.0,
        log_prior_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ):
        self.parameter_space = parameter_space
        self.score_fn = score_fn
        self.temperature = float(temperature)
        self.log_prior_fn = log_prior_fn

    def __call__(self, flat: jnp.ndarray) -> jnp.ndarray:
        evaluation = self.evaluate(flat)
        return jnp.asarray(evaluation.log_posterior, dtype=jnp.float32)

    def evaluate(self, flat: jnp.ndarray) -> DebugEvaluation:
        self.parameter_space.unpack(flat)
        score = float(self.score_fn())
        log_prior = 0.0 if self.log_prior_fn is None else float(self.log_prior_fn(flat))
        log_posterior = log_prior - (score / self.temperature)
        return DebugEvaluation(
            score=score,
            log_prior=log_prior,
            log_posterior=log_posterior,
        )


def create_rmh_kernel(log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray], sigma: ArrayLike):
    """Create RMH kernel with scalar or per-dimension Gaussian sigma."""
    sigma = jnp.asarray(sigma)
    if sigma.ndim == 0:
        return random_walk.normal_random_walk(log_prob_fn, float(sigma))

    def random_step(key, position):
        return jax.random.normal(key, position.shape) * sigma

    return random_walk.additive_step_random_walk(log_prob_fn, random_step)


def run_rmh_on_imp_system(
    log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_position: jnp.ndarray,
    rng_key: jax.Array,
    n_steps: int = 1000,
    sigma: ArrayLike = 1.0,
    proposal_fn: Optional[Callable[[jax.Array, jnp.ndarray], jnp.ndarray]] = None,
    sync_fn: Optional[Callable[[np.ndarray], None]] = None,
    sync_stride: int = 10,
    step_callback: Optional[Callable[[int, np.ndarray, float, bool], None]] = None,
    save_rmf3_path: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
    debug_stride: int = 1,
    debug_tolerance: float = 1e-5,
) -> RMHResult:
    """
    Run RMH over a flat position vector using a pure log-probability callable.

    Sampling is fully JAX-native (``lax.scan``). IMP writes are optional and
    happen only through ``sync_fn`` after the chain has been generated.

    Parameters
    ----------
    log_prob_fn:
        Pure callable over flat vectors.
    initial_position:
        Starting point in flat coordinates.
    rng_key:
        JAX RNG key.
    n_steps:
        Number of RMH iterations.
    sigma:
        Proposal width (scalar or per-dimension vector).
    proposal_fn:
        Optional proposal function for BlackJAX RMH kernel. If omitted, uses
        isotropic/per-dimension Gaussian random walk controlled by ``sigma``.
    sync_fn:
        Optional callback ``sync_fn(flat_position)`` used to write sampled
        positions into IMP for trajectory/final output.
    sync_stride:
        Call ``sync_fn`` every this many steps (and always on the final step).
    step_callback:
        Optional callback(step, position, log_prob, is_accepted).
    debug:
        Retained for backward compatibility. This function now assumes a pure
        ``log_prob_fn`` and does not perform IMP-backed recomputation checks.
    debug_stride:
        Print debug output every this many steps when debug=True.
    debug_tolerance:
        Warn if stored BlackJAX logdensity and fresh recomputation differ by
        more than this amount.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if sync_stride <= 0:
        raise ValueError("sync_stride must be positive.")

    x0 = jnp.asarray(initial_position).reshape(-1)

    if proposal_fn is None:
        kernel = create_rmh_kernel(log_prob_fn, sigma)
        state = kernel.init(x0)

        def _step_fn(carry, key):
            new_state, info = kernel.step(key, carry)
            return new_state, (new_state.position, new_state.logdensity, info.is_accepted)
    else:
        base_kernel = blackjax.rmh.build_kernel()
        state = blackjax.rmh.init(x0, log_prob_fn)

        def _step_fn(carry, key):
            new_state, info = base_kernel(key, carry, log_prob_fn, proposal_fn)
            return new_state, (new_state.position, new_state.logdensity, info.is_accepted)

    if verbose:
        print(f"Running RMH for {n_steps} steps in {x0.shape[0]} dimensions.")
        if debug:
            print(
                "[RMH debug] IMP-backed debug recomputation is disabled in the "
                "decoupled RMH path."
            )

    keys = jax.random.split(rng_key, n_steps)
    final_state, (positions_jax, log_probs_jax, accepted_jax) = jax.lax.scan(
        _step_fn, state, keys
    )

    _ = final_state
    positions = np.asarray(jax.device_get(positions_jax))
    log_probs = np.asarray(jax.device_get(log_probs_jax))
    accepted = np.asarray(jax.device_get(accepted_jax), dtype=bool)

    if sync_fn is not None or step_callback is not None:
        for i in range(n_steps):
            pos_np = np.asarray(positions[i])
            lp = float(log_probs[i])
            is_acc = bool(accepted[i])

            if sync_fn is not None and ((i % sync_stride == 0) or (i == n_steps - 1)):
                sync_fn(pos_np)

            if step_callback is not None:
                step_callback(i, pos_np, lp, is_acc)

    acceptance_rate = float(np.mean(accepted.astype(np.float64)))
    if verbose:
        print(f"Acceptance rate: {acceptance_rate:.2%}")

    if save_rmf3_path is not None:
        raise ValueError(
            "save_rmf3_path is not supported in the decoupled RMH API. "
            "Use sync_fn + step_callback for RMF output."
        )

    return RMHResult(
        positions=positions,
        log_probs=log_probs,
        accepted=accepted,
        acceptance_rate=acceptance_rate,
    )


def make_imp_score_function(scoring_function: Any) -> Callable[[], float]:
    """Wrap IMP scoring function object into a no-arg callable."""

    def _score() -> float:
        return float(scoring_function.evaluate(False))

    return _score


def build_flexible_bead_rmh_wrapper(
    model: Any,
    scoring_function: Any,
    flexible_particle_indices: Sequence[int],
    temperature: float = 1.0,
    log_prior_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> Tuple[IMPParameterSpace, IMPLogPosterior]:
    """
    Convenience constructor for the current use-case (flexible beads only).

    Returns
    -------
    parameter_space:
        Contains a single FlexibleBeadBlock.
    log_posterior:
        Callable to pass to BlackJAX RMH.
    """
    flex_block = FlexibleBeadBlock(model=model, particle_indices=flexible_particle_indices)
    parameter_space = IMPParameterSpace(blocks=[flex_block])
    score_fn = make_imp_score_function(scoring_function)
    log_posterior = IMPLogPosterior(
        parameter_space=parameter_space,
        score_fn=score_fn,
        temperature=temperature,
        log_prior_fn=log_prior_fn,
    )
    return parameter_space, log_posterior


# -----------------------------------------------------------------------------
# Backward-compatible aliases so existing imports do not break.
# -----------------------------------------------------------------------------
IMPDOFSpace = IMPParameterSpace
IMPSMCAdapter = IMPLogPosterior


def run_smc_on_imp_system(
    adapter: Any,
    rng_key: jax.Array,
    n_particles: int = 500,
    n_temperature_steps: int = 30,
    schedule: str = "geometric",
    kernel: str = "rmh",
    rmh_sigma: Optional[ArrayLike] = None,
    rmh_proposal_fn: Optional[Callable[[jax.Array, jnp.ndarray], jnp.ndarray]] = None,
    hmc_step_size: float = 0.05,
    hmc_num_integration_steps: int = 5,
    n_mcmc_steps: int = 10,
    score_batch_size: int = 32,
    save_rmf3_path: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
    debug_stride: int = 1,
):
    """
    Run a stand-alone fixed-schedule SMC sampler on an IMP/JAX adapter.

    This is the JIT/vmap-friendly path: the adapter provides JAX log-prior,
    log-likelihood, and log-posterior functions, and the actual SMC loop is
    delegated to :func:`sampling.smc_base_sampler.run_base_smc_rmh` or
    :func:`sampling.smc_base_sampler.run_base_smc_hmc`.

    The returned trajectory is the best particle at each temperature step.
    """
    # Import here to keep the wrapper lightweight and avoid circular imports.
    from .smc_base_sampler import run_base_smc_hmc, run_base_smc_rmh

    if verbose:
        print(adapter.dof_summary())

    # Draw an initial particle population from the adapter's prior.
    key_init, key_smc = jax.random.split(rng_key)
    initial_positions = adapter.sample_prior(
        n_particles=n_particles,
        rng_key=key_init,
        translation_sigma=150.0,
    )

    if verbose:
        print(f"\nInitial positions shape: {initial_positions.shape}")
        print(f"Example IMP score (particle 0): {adapter.imp_score(initial_positions[0]):.2f}")
        print(
            "Running fixed-schedule SMC with "
            f"kernel={kernel}, schedule={schedule}, steps={n_temperature_steps}"
        )

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
        debug=debug,
        debug_stride=debug_stride,
    )

    if kernel == "rmh":
        sigma = rmh_sigma if rmh_sigma is not None else adapter.suggested_rmh_sigma()

        proposal_fn = rmh_proposal_fn
        if proposal_fn is None and hasattr(adapter, "make_rmh_proposal_fn"):
            proposal_cfg = adapter.suggested_rmh_proposal() if hasattr(adapter, "suggested_rmh_proposal") else {}
            proposal_fn = adapter.make_rmh_proposal_fn(**proposal_cfg)

        state, info_history, best_positions, best_scores, lambdas = run_base_smc_rmh(
            rmh_sigma=sigma,
            rmh_proposal_fn=proposal_fn,
            **common_kwargs,
        )
    elif kernel == "hmc":
        state, info_history, best_positions, best_scores, lambdas = run_base_smc_hmc(
            hmc_step_size=hmc_step_size,
            hmc_num_integration_steps=hmc_num_integration_steps,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'rmh' or 'hmc'.")

    # Save only the best configuration from each temperature step.
    if save_rmf3_path is not None and best_positions is not None:
        from io_utils.rmf3_converter import write_xyz_trajectory_rmf3

        best_xyz = np.stack([adapter.decode_xyz(pos) for pos in best_positions], axis=0)
        write_xyz_trajectory_rmf3(save_rmf3_path, best_xyz, verbose=verbose)

    return state, info_history, best_positions, best_scores, lambdas


def run_adaptive_smc_on_imp_system(
    adapter: Any,
    rng_key: jax.Array,
    n_particles: int = 500,
    max_temperature_steps: Optional[int] = None,
    n_mcmc_steps: int = 10,
    score_batch_size: Optional[int] = None,
    rmh_sigma: Optional[ArrayLike] = None,
    rmh_proposal_fn: Optional[Callable[[jax.Array, jnp.ndarray], jnp.ndarray]] = None,
    target_ess: float = 0.5,
    save_rmf3_path: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
    debug_stride: int = 1,
):
    """
    Run BlackJAX adaptive tempered SMC on an IMP/JAX adapter.

    This uses the adaptive tempering path from :mod:`sampling.smc` and keeps
    the same wrapper style as :func:`run_smc_on_imp_system`.
    """
    from .smc import run_tempered_smc

    if verbose:
        print(adapter.dof_summary())

    key_init, key_smc = jax.random.split(rng_key)
    initial_positions = adapter.sample_prior(
        n_particles=n_particles,
        rng_key=key_init,
        translation_sigma=150.0,
    )

    sigma = rmh_sigma if rmh_sigma is not None else adapter.suggested_rmh_sigma()

    proposal_fn = rmh_proposal_fn
    if proposal_fn is None and hasattr(adapter, "make_rmh_proposal_fn"):
        proposal_cfg = adapter.suggested_rmh_proposal() if hasattr(adapter, "suggested_rmh_proposal") else {}
        proposal_fn = adapter.make_rmh_proposal_fn(**proposal_cfg)

    if verbose:
        print(f"\nInitial positions shape: {initial_positions.shape}")
        print(f"Example IMP score (particle 0): {adapter.imp_score(initial_positions[0]):.2f}")
        print(
            "Running adaptive tempered SMC with "
            f"target_ess={target_ess:.0%}, n_mcmc_steps={n_mcmc_steps}, rmh_sigma={sigma}, "
            f"max_temperature_steps={max_temperature_steps}"
        )
        if score_batch_size is not None:
            print("  Note: score_batch_size is ignored for adaptive SMC in this wrapper.")

    state, info_history, best_positions, best_scores, lambdas = run_tempered_smc(
        log_prior_fn=adapter.log_prior,
        log_likelihood_fn=adapter.log_likelihood,
        log_prob_fn=adapter.log_prob,
        initial_positions=initial_positions,
        rng_key=key_smc,
        n_mcmc_steps=n_mcmc_steps,
        rmh_sigma=sigma,
        rmh_proposal_fn=proposal_fn,
        target_ess=target_ess,
        max_temperature_steps=max_temperature_steps,
        record_best=True,
        verbose=verbose,
        debug=debug,
        debug_stride=debug_stride,
    )

    if save_rmf3_path is not None and best_positions is not None:
        from io_utils.rmf3_converter import write_xyz_trajectory_rmf3

        best_xyz = np.stack([adapter.decode_xyz(pos) for pos in best_positions], axis=0)
        write_xyz_trajectory_rmf3(save_rmf3_path, best_xyz, verbose=verbose)

    return state, info_history, best_positions, best_scores, lambdas
