"""
SMC sampler using BlackJAX's base SMC (blackjax.smc.base).

Unlike adaptive_tempered_smc, this uses a FIXED temperature schedule
and does NOT adaptively choose step sizes.  The user specifies:
  - a temperature ladder  (lambda_0 = 0 ... lambda_T = 1)
  - a mutation kernel     (RMH or HMC with fixed parameters)
  - a resampling scheme   (systematic by default)

At each step the algorithm:
  1. Re-weights particles:  w_t(x) = (lambda_t - lambda_{t-1}) * log_likelihood(x)
  2. Resamples according to those weights.
  3. Mutates every particle with the MCMC kernel targeting the
     tempered distribution  pi_t(x) = prior(x) * likelihood(x)^{lambda_t}.

This is the "textbook" SMC sampler (Del Moral et al., 2006) and
provides a baseline against which adaptive variants can be compared.
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.base as smc_base
import blackjax.smc.resampling as resampling
from typing import Callable, Tuple, List, Any, Optional, Literal
import time


def _safe_float(value):
    """Convert scalar-like objects (JAX/NumPy/Python) to float or return None."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_smc_info_fields(info) -> dict:
    """Best-effort extraction of useful fields from BlackJAX SMC info objects."""
    fields = {
        "ess": None,
        "log_normalizer_increment": None,
        "mean_acceptance_rate": None,
    }

    # Generic attributes that may exist on info objects.
    for key in ("ess", "effective_sample_size", "log_normalizer_increment"):
        if hasattr(info, key):
            val = _safe_float(getattr(info, key))
            if key in ("ess", "effective_sample_size") and val is not None:
                fields["ess"] = val
            elif key == "log_normalizer_increment":
                fields["log_normalizer_increment"] = val

    # update_info is typically where per-particle acceptance rates live.
    update_info = getattr(info, "update_info", None)
    if update_info is not None:
        if isinstance(update_info, dict):
            if "acceptance_rate" in update_info:
                acc = np.asarray(update_info["acceptance_rate"], dtype=float)
                if acc.size > 0:
                    fields["mean_acceptance_rate"] = float(np.mean(acc))
        else:
            if hasattr(update_info, "acceptance_rate"):
                acc = np.asarray(getattr(update_info, "acceptance_rate"), dtype=float)
                if acc.size > 0:
                    fields["mean_acceptance_rate"] = float(np.mean(acc))

    return fields


# =============================================================================
# Temperature schedules
# =============================================================================

def linear_schedule(n_steps: int) -> np.ndarray:
    """Linearly spaced temperatures from 0 to 1 (inclusive)."""
    return np.linspace(0.0, 1.0, n_steps + 1)


def geometric_schedule(n_steps: int, min_beta: float = 1e-4) -> np.ndarray:
    """Geometrically spaced temperatures from ~0 to 1."""
    betas = np.geomspace(min_beta, 1.0, n_steps)
    return np.concatenate([[0.0], betas])


def sigmoid_schedule(n_steps: int, steepness: float = 6.0) -> np.ndarray:
    """Sigmoid-shaped schedule (slow start, fast middle, slow end)."""
    t = np.linspace(-1, 1, n_steps + 1)
    betas = 1.0 / (1.0 + np.exp(-steepness * t))
    betas = (betas - betas[0]) / (betas[-1] - betas[0])
    return betas


SCHEDULE_REGISTRY = {
    "linear": linear_schedule,
    "geometric": geometric_schedule,
    "sigmoid": sigmoid_schedule,
}


# =============================================================================
# Memory-safe batched vmap
# =============================================================================

def batched_vmap(fn: Callable, batch_size: int = 16):
    """
    Apply *fn* to an array of inputs in batches to control GPU memory.

    Like ``jax.vmap(fn)(inputs)`` but processes *batch_size* inputs at a
    time so peak memory is O(batch_size) instead of O(n_total).
    """
    @jax.jit
    def _score_batch(batch):
        return jax.vmap(fn)(batch)

    def batched_fn(inputs):
        n = inputs.shape[0]
        if n <= batch_size:
            return _score_batch(inputs)

        results = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_result = _score_batch(inputs[start:end])
            jax.block_until_ready(batch_result)
            results.append(batch_result)
        return jnp.concatenate(results, axis=0)

    return batched_fn


# =============================================================================
# Base SMC runner – RMH kernel
# =============================================================================

def run_base_smc_rmh(
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    log_prob_fn: Callable,
    initial_positions: jnp.ndarray,
    rng_key: jax.Array,
    # Temperature schedule
    n_temperature_steps: int = 20,
    schedule: str = "linear",
    # RMH parameters (fixed)
    rmh_sigma: float | jnp.ndarray = 1.0,
    rmh_proposal_fn: Optional[Callable[[jax.Array, jnp.ndarray], jnp.ndarray]] = None,
    n_mcmc_steps: int = 10,
    # Resampling
    resampling_fn: Callable = resampling.systematic,
    # Tracking
    record_best: bool = True,
    verbose: bool = True,
    score_batch_size: int = 16,
    debug: bool = False,
    debug_stride: int = 1,
) -> Tuple[Any, List, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run base SMC with a fixed temperature schedule and RMH mutations.

    Parameters
    ----------
    log_prior_fn : Callable
        Log-prior: position (n_dims,) -> scalar.
    log_likelihood_fn : Callable
        Log-likelihood: position (n_dims,) -> scalar.
    log_prob_fn : Callable
        Full log-posterior for diagnostic scoring.
    initial_positions : jnp.ndarray
        Shape (n_particles, n_dims).
    rng_key : jax.Array
    n_temperature_steps : int
        Number of tempering steps (lambdas between 0 and 1).
    schedule : str
        Temperature schedule type: 'linear', 'geometric', or 'sigmoid'.
    rmh_sigma : float
        Fixed RMH proposal standard deviation.
    n_mcmc_steps : int
        Number of MCMC mutation sweeps per SMC step.
    resampling_fn : Callable
        Resampling algorithm (default: systematic).
    record_best : bool
        Track best particle per step.
    verbose : bool
        Print progress.
    score_batch_size : int
        Batch size when scoring particles for tracking.

    Returns
    -------
    state : SMCState
        Final SMC state (particles + weights).
    info_history : list[SMCInfo]
        Per-step diagnostics.
    best_positions : ndarray | None
        (n_steps+1, n_dims)  best particle per step.
    best_scores : ndarray | None
        (n_steps+1,)  corresponding scores.
    lambdas : ndarray
        The temperature schedule actually used.
    """
    n_particles, n_dims = initial_positions.shape

    # ----- temperature schedule ------------------------------------------
    schedule_fn = SCHEDULE_REGISTRY.get(schedule)
    if schedule_fn is None:
        raise ValueError(
            f"Unknown schedule '{schedule}'. Choose from: {list(SCHEDULE_REGISTRY.keys())}"
        )
    lambdas = schedule_fn(n_temperature_steps)
    assert lambdas[0] == 0.0 and lambdas[-1] == 1.0

    if verbose:
        print(f"Temperature schedule ({schedule}): {len(lambdas)} points")
        print(f"  First 5 lambdas: {lambdas[:5]}")
        print(f"  Last  5 lambdas: {lambdas[-5:]}")

    # ----- RMH kernel ---------------------------------------------------
    rmh_kernel = blackjax.rmh.build_kernel()

    # Numerical safety floor used when the model returns non-finite values.
    bad_logp = jnp.array(-1e30, dtype=jnp.float32)

    @jax.jit
    def _safe_logp(x):
        v = log_prob_fn(x)
        return jnp.where(jnp.isfinite(v), v, bad_logp)

    @jax.jit
    def _safe_loglikelihood(x):
        v = log_likelihood_fn(x)
        return jnp.where(jnp.isfinite(v), v, bad_logp)

    def _rmh_proposal(rng_key, position):
        return position + jax.random.normal(rng_key, shape=position.shape) * rmh_sigma

    proposal_fn = rmh_proposal_fn if rmh_proposal_fn is not None else _rmh_proposal

    # ----- batched scoring for tracking ---------------------------------
    score_batched = batched_vmap(_safe_logp, batch_size=score_batch_size)

    def get_best_stats(particles):
        scores = score_batched(particles)
        idx = jnp.argmax(scores)
        return particles[idx], scores[idx], jnp.mean(scores), jnp.std(scores)

    # ----- initialise base SMC state ------------------------------------
    # smc_base.init requires init_update_params; we don't use them so pass {}.
    state = smc_base.init(initial_positions, {})

    if verbose:
        print(f"\nRunning BlackJAX Base SMC (RMH kernel)")
        print(f"  Particles: {n_particles}, Dims: {n_dims}")
        print(f"  RMH sigma: {rmh_sigma}")
        print(
            "  RMH proposal: "
            f"{'custom (adapter-provided)' if rmh_proposal_fn is not None else 'isotropic Gaussian'}"
        )
        print(f"  MCMC steps/temp: {n_mcmc_steps}")
        print(f"  Temperature steps: {n_temperature_steps}")
        print(f"  Score batch size: {score_batch_size}")

    # ----- tracking lists -----------------------------------------------
    info_history: List[Any] = []
    best_positions: List[np.ndarray] = []
    best_scores: List[float] = []

    if record_best:
        pos, score, mean_score, std_score = get_best_stats(state.particles)
        jax.block_until_ready(score)
        best_positions.append(np.array(pos))
        best_scores.append(float(score))
        if verbose:
            print(
                f"Initial | Best: {score:.2f}, Mean: {mean_score:.2f}, "
                f"Std: {std_score:.2f}"
            )

    # =====================================================================
    # Main SMC loop
    # =====================================================================
    t0 = time.perf_counter()

    for step_idx in range(1, len(lambdas)):
        rng_key, step_key = jax.random.split(rng_key)

        lam_prev = lambdas[step_idx - 1]
        lam_curr = lambdas[step_idx]
        delta_lam = lam_curr - lam_prev

        # --- tempered log-density at current lambda ----------------------
        def _tempered_logdensity(position, _lam=lam_curr):
            lp = log_prior_fn(position)
            ll = _safe_loglikelihood(position)
            val = lp + _lam * ll
            return jnp.where(jnp.isfinite(val), val, bad_logp)

        # --- update_fn (batched MCMC mutations) --------------------------
        # smc_base.step passes (keys, particles, update_params).
        # Must return (new_particles, update_info).
        @jax.jit
        def update_fn(keys, particles, update_params):
            def _mutate_one(key, particle):
                st = blackjax.rmh.init(particle, _tempered_logdensity)
                def _body(carry, _):
                    k, s = carry
                    k, subk = jax.random.split(k)
                    s, info = rmh_kernel(subk, s, _tempered_logdensity, proposal_fn)
                    return (k, s), info.is_accepted
                (_, final_st), accepted = jax.lax.scan(_body, (key, st), jnp.arange(n_mcmc_steps))
                return final_st.position, jnp.mean(accepted)
            new_particles, accept_rates = jax.vmap(_mutate_one)(keys, particles)
            return new_particles, {"acceptance_rate": accept_rates}

        # --- weight_fn (batched importance weights) ----------------------
        # Incremental weight = (lambda_t - lambda_{t-1}) * log_likelihood(x)
        @jax.jit
        def weight_fn(particles):
            ll = jax.vmap(_safe_loglikelihood)(particles)
            w = delta_lam * ll
            return jnp.where(jnp.isfinite(w), w, bad_logp)

        # --- take one base SMC step -------------------------------------
        state, info = smc_base.step(
            step_key,
            state,
            update_fn,
            weight_fn,
            resampling_fn,
        )
        jax.block_until_ready(state.particles)

        info_history.append(info)

        # --- tracking ----------------------------------------------------
        if record_best:
            pos, score, mean_score, std_score = get_best_stats(state.particles)
            jax.block_until_ready(score)
            best_positions.append(np.array(pos))
            best_scores.append(float(score))

        if debug and (step_idx % max(debug_stride, 1) == 0):
            # Report non-finite diagnostics only in debug mode to avoid noisy output.
            finite_mask = jnp.isfinite(jax.vmap(log_prob_fn)(state.particles))
            n_bad = int(state.particles.shape[0] - jnp.sum(finite_mask))
            fields = _extract_smc_info_fields(info)
            msg = (
                f"  [debug] step={step_idx:3d} lambda={lam_curr:.4f} "
                f"non-finite logp particles: {n_bad}/{state.particles.shape[0]}"
            )
            if fields["mean_acceptance_rate"] is not None:
                msg += f" | mean_accept={fields['mean_acceptance_rate']:.3f}"
            if fields["ess"] is not None:
                msg += f" | ess={fields['ess']:.2f}"
            if fields["log_normalizer_increment"] is not None:
                msg += f" | dlogZ={fields['log_normalizer_increment']:.4f}"
            print(
                msg
            )

        if verbose:
            print(
                f"Step {step_idx:3d}/{n_temperature_steps} | "
                f"λ = {lam_curr:.4f} | "
                f"Best: {best_scores[-1]:.2f} | "
                f"Mean: {float(mean_score):.2f}"
            )

    dt = time.perf_counter() - t0
    if verbose:
        print(f"\nBase SMC completed in {dt:.2f}s ({n_temperature_steps} temperature steps)")
        print(f"  Average: {dt / max(n_temperature_steps, 1):.2f}s per step")

    # ----- stack results ------------------------------------------------
    if record_best and best_positions:
        best_positions_arr = np.stack(best_positions, axis=0)
        best_scores_arr = np.array(best_scores)
    else:
        best_positions_arr = None
        best_scores_arr = None

    return state, info_history, best_positions_arr, best_scores_arr, lambdas


# =============================================================================
# Base SMC runner – HMC kernel
# =============================================================================

def run_base_smc_hmc(
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    log_prob_fn: Callable,
    initial_positions: jnp.ndarray,
    rng_key: jax.Array,
    # Temperature schedule
    n_temperature_steps: int = 20,
    schedule: str = "linear",
    # HMC parameters (fixed)
    hmc_step_size: float = 0.01,
    hmc_inverse_mass_matrix: Optional[jnp.ndarray] = None,
    hmc_num_integration_steps: int = 10,
    n_mcmc_steps: int = 10,
    # Resampling
    resampling_fn: Callable = resampling.systematic,
    # Tracking
    record_best: bool = True,
    verbose: bool = True,
    score_batch_size: int = 16,
    debug: bool = False,
    debug_stride: int = 1,
) -> Tuple[Any, List, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run base SMC with a fixed temperature schedule and HMC mutations.

    Parameters are analogous to ``run_base_smc_rmh`` except for the
    HMC-specific knobs.

    Returns
    -------
    state, info_history, best_positions, best_scores, lambdas
    """
    n_particles, n_dims = initial_positions.shape

    # ----- temperature schedule ------------------------------------------
    schedule_fn = SCHEDULE_REGISTRY.get(schedule)
    if schedule_fn is None:
        raise ValueError(
            f"Unknown schedule '{schedule}'. Choose from: {list(SCHEDULE_REGISTRY.keys())}"
        )
    lambdas = schedule_fn(n_temperature_steps)
    assert lambdas[0] == 0.0 and lambdas[-1] == 1.0

    if verbose:
        print(f"Temperature schedule ({schedule}): {len(lambdas)} points")
        print(f"  First 5 lambdas: {lambdas[:5]}")
        print(f"  Last  5 lambdas: {lambdas[-5:]}")

    # ----- HMC kernel ---------------------------------------------------
    if hmc_inverse_mass_matrix is None:
        hmc_inverse_mass_matrix = jnp.ones(n_dims)
    else:
        hmc_inverse_mass_matrix = jnp.asarray(hmc_inverse_mass_matrix)

    hmc_step_size_jnp = jnp.float32(hmc_step_size)
    hmc_kernel = blackjax.hmc.build_kernel()

    # ----- batched scoring for tracking ---------------------------------
    score_batched = batched_vmap(log_prob_fn, batch_size=score_batch_size)

    def get_best_stats(particles):
        scores = score_batched(particles)
        idx = jnp.argmax(scores)
        return particles[idx], scores[idx], jnp.mean(scores), jnp.std(scores)

    # ----- initialise base SMC state ------------------------------------
    # smc_base.init requires init_update_params; we don't use them so pass {}.
    state = smc_base.init(initial_positions, {})

    if verbose:
        print(f"\nRunning BlackJAX Base SMC (HMC kernel)")
        print(f"  Particles: {n_particles}, Dims: {n_dims}")
        print(f"  HMC: step_size={hmc_step_size:.4f}, "
              f"L={hmc_num_integration_steps}, "
              f"mass_matrix={'diagonal' if hmc_inverse_mass_matrix.ndim == 1 else 'dense'}")
        print(f"  MCMC steps/temp: {n_mcmc_steps}")
        print(f"  Temperature steps: {n_temperature_steps}")
        print(f"  Score batch size: {score_batch_size}")

    # ----- tracking lists -----------------------------------------------
    info_history: List[Any] = []
    best_positions: List[np.ndarray] = []
    best_scores: List[float] = []

    if record_best:
        pos, score, mean_score, std_score = get_best_stats(state.particles)
        jax.block_until_ready(score)
        best_positions.append(np.array(pos))
        best_scores.append(float(score))
        if verbose:
            print(
                f"Initial | Best: {score:.2f}, Mean: {mean_score:.2f}, "
                f"Std: {std_score:.2f}"
            )

    # =====================================================================
    # Main SMC loop
    # =====================================================================
    t0 = time.perf_counter()

    for step_idx in range(1, len(lambdas)):
        rng_key, step_key = jax.random.split(rng_key)

        lam_prev = lambdas[step_idx - 1]
        lam_curr = lambdas[step_idx]
        delta_lam = lam_curr - lam_prev

        # --- tempered log-density at current lambda ----------------------
        def _tempered_logdensity(position, _lam=lam_curr):
            return log_prior_fn(position) + _lam * log_likelihood_fn(position)

        # --- update_fn (batched HMC mutations) ---------------------------
        # smc_base.step passes (keys, particles, update_params).
        # Must return (new_particles, update_info).
        @jax.jit
        def update_fn(keys, particles, update_params):
            def _mutate_one(key, particle):
                st = blackjax.hmc.init(particle, _tempered_logdensity)
                def _body(carry, _):
                    k, s = carry
                    k, subk = jax.random.split(k)
                    s, info = hmc_kernel(
                        subk, s, _tempered_logdensity,
                        hmc_step_size_jnp,
                        hmc_inverse_mass_matrix,
                        hmc_num_integration_steps,
                    )
                    return (k, s), info.is_accepted
                (_, final_st), accepted = jax.lax.scan(_body, (key, st), jnp.arange(n_mcmc_steps))
                return final_st.position, jnp.mean(accepted)
            new_particles, accept_rates = jax.vmap(_mutate_one)(keys, particles)
            return new_particles, {"acceptance_rate": accept_rates}

        # --- weight_fn ---------------------------------------------------
        @jax.jit
        def weight_fn(particles):
            ll = jax.vmap(log_likelihood_fn)(particles)
            return delta_lam * ll

        # --- take one base SMC step -------------------------------------
        state, info = smc_base.step(
            step_key,
            state,
            update_fn,
            weight_fn,
            resampling_fn,
        )
        jax.block_until_ready(state.particles)

        info_history.append(info)

        # --- tracking ----------------------------------------------------
        if record_best:
            pos, score, mean_score, std_score = get_best_stats(state.particles)
            jax.block_until_ready(score)
            best_positions.append(np.array(pos))
            best_scores.append(float(score))

        if verbose:
            print(
                f"Step {step_idx:3d}/{n_temperature_steps} | "
                f"λ = {lam_curr:.4f} | "
                f"Best: {best_scores[-1]:.2f} | "
                f"Mean: {float(mean_score):.2f}"
            )

        if debug and (step_idx % max(debug_stride, 1) == 0):
            finite_mask = jnp.isfinite(jax.vmap(log_prob_fn)(state.particles))
            n_bad = int(state.particles.shape[0] - jnp.sum(finite_mask))
            fields = _extract_smc_info_fields(info)
            msg = (
                f"  [debug] step={step_idx:3d} lambda={lam_curr:.4f} "
                f"non-finite logp particles: {n_bad}/{state.particles.shape[0]}"
            )
            if fields["mean_acceptance_rate"] is not None:
                msg += f" | mean_accept={fields['mean_acceptance_rate']:.3f}"
            if fields["ess"] is not None:
                msg += f" | ess={fields['ess']:.2f}"
            if fields["log_normalizer_increment"] is not None:
                msg += f" | dlogZ={fields['log_normalizer_increment']:.4f}"
            print(
                msg
            )

    dt = time.perf_counter() - t0
    if verbose:
        print(f"\nBase SMC completed in {dt:.2f}s ({n_temperature_steps} temperature steps)")
        print(f"  Average: {dt / max(n_temperature_steps, 1):.2f}s per step")

    # ----- stack results ------------------------------------------------
    if record_best and best_positions:
        best_positions_arr = np.stack(best_positions, axis=0)
        best_scores_arr = np.array(best_scores)
    else:
        best_positions_arr = None
        best_scores_arr = None

    return state, info_history, best_positions_arr, best_scores_arr, lambdas


# =============================================================================
# Utility helpers (same interface as smc.py / smc_with_hmc.py)
# =============================================================================

def get_smc_samples(state) -> jnp.ndarray:
    """Extract particles from an SMCState."""
    return state.particles


def get_best_sample(
    state, log_prob_fn: Callable, batch_size: int = 16
) -> Tuple[jnp.ndarray, float]:
    """Identify the single best particle from the final population."""
    particles = state.particles
    scores = batched_vmap(log_prob_fn, batch_size=batch_size)(particles)
    best_idx = jnp.argmax(scores)
    return particles[best_idx], float(scores[best_idx])
