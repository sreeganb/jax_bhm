"""
Minimal modular IMP <-> BlackJAX wrapper.

Current scope
-------------
- Flexible bead coordinates only.
- Random-walk Metropolis-Hastings (RMH) using BlackJAX.

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
        self.parameter_space.unpack(flat)
        score = float(self.score_fn())
        lp = jnp.asarray(0.0) if self.log_prior_fn is None else self.log_prior_fn(flat)
        return lp - jnp.asarray(score / self.temperature)


def create_rmh_kernel(log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray], sigma: ArrayLike):
    """Create RMH kernel with scalar or per-dimension Gaussian sigma."""
    sigma = jnp.asarray(sigma)
    if sigma.ndim == 0:
        return random_walk.normal_random_walk(log_prob_fn, float(sigma))

    def random_step(key, position):
        return jax.random.normal(key, position.shape) * sigma

    return random_walk.additive_step_random_walk(log_prob_fn, random_step)


def run_rmh_on_imp_system(
    parameter_space: IMPParameterSpace,
    log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
    rng_key: jax.Array,
    n_steps: int = 1000,
    sigma: ArrayLike = 1.0,
    initial_position: Optional[jnp.ndarray] = None,
    step_callback: Optional[Callable[[int, np.ndarray, float, bool], None]] = None,
    verbose: bool = True,
) -> RMHResult:
    """
    Run step-by-step RMH and keep IMP in sync with accepted states.

    Parameters
    ----------
    parameter_space:
        Knows how to unpack sampled vectors into IMP.
    log_prob_fn:
        Callable over flat vectors, usually IMPLogPosterior(...).
    rng_key:
        JAX RNG key.
    n_steps:
        Number of RMH iterations.
    sigma:
        Proposal width (scalar or per-dimension vector).
    initial_position:
        Starting point. If None, reads current IMP coordinates.
    step_callback:
        Optional callback(step, position, log_prob, is_accepted).
    """
    kernel = create_rmh_kernel(log_prob_fn, sigma)
    x0 = parameter_space.pack() if initial_position is None else jnp.asarray(initial_position)
    state = kernel.init(x0)

    keys = jax.random.split(rng_key, n_steps)
    positions: List[np.ndarray] = []
    log_probs: List[float] = []
    accepted: List[bool] = []

    if verbose:
        print(f"Running RMH for {n_steps} steps in {parameter_space.dim} dimensions.")

    for i in range(n_steps):
        state, info = kernel.step(keys[i], state)

        # Keep IMP model aligned with the chain state after accept/reject.
        parameter_space.unpack(state.position)

        pos_np = np.asarray(state.position)
        lp = float(state.logdensity)
        is_acc = bool(info.is_accepted)

        positions.append(pos_np)
        log_probs.append(lp)
        accepted.append(is_acc)

        if step_callback is not None:
            step_callback(i, pos_np, lp, is_acc)

    acceptance_rate = float(np.mean(np.asarray(accepted, dtype=np.float64)))
    if verbose:
        print(f"Acceptance rate: {acceptance_rate:.2%}")

    return RMHResult(
        positions=np.asarray(positions),
        log_probs=np.asarray(log_probs),
        accepted=np.asarray(accepted),
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


def run_smc_on_imp_system(*args, **kwargs):
    raise NotImplementedError(
        "SMC was intentionally removed from this minimal wrapper. "
        "Use run_rmh_on_imp_system(...) for flexible-bead RMH."
    )
