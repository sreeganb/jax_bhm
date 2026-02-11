"""
SMC sampler using BlackJAX's adaptive tempered SMC with HMC mutation kernel.

Key differences from RMH version:
- HMC requires differentiable log-densities (no hard -inf boundaries!)
- HMC kernel needs step_size, inverse_mass_matrix, num_integration_steps
- HMC state tracks logdensity_grad in addition to logdensity
- Better exploration per step than RMH, but more expensive per step

GPU memory notes:
- Uses batched_vmap for scoring to avoid OOM when evaluating many particles.
  The EM density scoring with large maps (e.g. 160^3) uses significant
  memory per particle, so vmapping all particles at once can exceed GPU RAM.
- The SMC step itself (BlackJAX internals) also vmaps over particles.
  If that OOMs, reduce n_particles or map resolution.
"""
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
from typing import Callable, Tuple, List, Any, Optional
import time


# =============================================================================
# Memory-safe batched vmap
# =============================================================================

def batched_vmap(fn: Callable, batch_size: int = 16):
    """
    Apply fn to an array of inputs in batches to control GPU memory.

    Like jax.vmap(fn)(inputs) but processes batch_size inputs at a time,
    so peak memory is O(batch_size) instead of O(n_total).

    Each batch is JIT-compiled; results are concatenated at the end.
    block_until_ready() is called between batches to free intermediates.

    Parameters
    ----------
    fn : Callable
        Scalar function: single input -> scalar output.
    batch_size : int
        Max inputs to process simultaneously. Tune to GPU memory.

    Returns
    -------
    batched_fn : Callable
        Takes (n, ...) array, returns (n,) results.
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


def run_tempered_smc(
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    log_prob_fn: Callable,
    initial_positions: jnp.ndarray,
    rng_key: jax.Array,
    n_mcmc_steps: int = 10,
    # HMC-specific parameters
    hmc_step_size: float = 0.01,
    hmc_inverse_mass_matrix: Optional[jnp.ndarray] = None,
    hmc_num_integration_steps: int = 10,
    # SMC parameters
    target_ess: float = 0.5,
    record_best: bool = True,
    verbose: bool = True,
    # GPU memory management
    score_batch_size: int = 16,
) -> Tuple[Any, List, np.ndarray, np.ndarray]:
    """
    Run BlackJAX adaptive tempered SMC with HMC mutation kernel.

    Parameters
    ----------
    log_prior_fn : Callable
        Differentiable log-prior: position (n_dims,) -> scalar.
    log_likelihood_fn : Callable
        Differentiable log-likelihood: position (n_dims,) -> scalar.
    log_prob_fn : Callable
        Full log-posterior for scoring (can be non-differentiable).
    initial_positions : jnp.ndarray
        Shape (n_particles, n_dims).
    rng_key : jax.Array
    n_mcmc_steps : int
        HMC mutations per SMC temperature step.
    hmc_step_size : float
        Leapfrog step size.
    hmc_inverse_mass_matrix : jnp.ndarray or None
        Diagonal inverse mass matrix (n_dims,). None = identity.
    hmc_num_integration_steps : int
        Leapfrog steps per HMC proposal (L).
    target_ess : float
        Target ESS ratio for adaptive tempering.
    record_best : bool
        Track best particle per step.
    verbose : bool
        Print progress.
    score_batch_size : int
        Particles scored simultaneously for tracking. Reduce if OOM
        during scoring. Does NOT affect the SMC step internals.
    """
    n_particles, n_dims = initial_positions.shape

    # =========================================================================
    # 1. HMC parameters
    # =========================================================================
    if hmc_inverse_mass_matrix is None:
        hmc_inverse_mass_matrix = jnp.ones(n_dims)
    else:
        hmc_inverse_mass_matrix = jnp.asarray(hmc_inverse_mass_matrix)

    hmc_step_size = jnp.float32(hmc_step_size)

    # =========================================================================
    # 2. Build HMC kernel
    # =========================================================================
    hmc_kernel = blackjax.hmc.build_kernel()

    def mcmc_step_fn(rng_key, state, logdensity_fn):
        return hmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            hmc_step_size,
            hmc_inverse_mass_matrix,
            hmc_num_integration_steps,
        )

    mcmc_init_fn = blackjax.hmc.init

    # =========================================================================
    # 3. Build adaptive tempered SMC
    # =========================================================================
    tempered_smc = blackjax.adaptive_tempered_smc(
        logprior_fn=log_prior_fn,
        loglikelihood_fn=log_likelihood_fn,
        mcmc_step_fn=mcmc_step_fn,
        mcmc_init_fn=mcmc_init_fn,
        mcmc_parameters={},
        resampling_fn=resampling.systematic,
        target_ess=target_ess,
        num_mcmc_steps=n_mcmc_steps,
    )

    # =========================================================================
    # 4. JIT-compile step + batched scoring
    # =========================================================================
    jit_step = jax.jit(tempered_smc.step)
    score_batched = batched_vmap(log_prob_fn, batch_size=score_batch_size)

    def get_best_stats(particles):
        scores = score_batched(particles)
        idx = jnp.argmax(scores)
        return particles[idx], scores[idx], jnp.mean(scores), jnp.std(scores)

    # =========================================================================
    # 5. Initialize
    # =========================================================================
    if verbose:
        print("Initializing SMC...")

    state = tempered_smc.init(initial_positions)

    if verbose:
        print(f"Running BlackJAX Adaptive Tempered SMC (HMC kernel)")
        print(f"  Particles: {n_particles}, Dims: {n_dims}")
        print(f"  HMC: step_size={float(hmc_step_size):.4f}, "
              f"L={hmc_num_integration_steps}, "
              f"mass_matrix={'diagonal' if hmc_inverse_mass_matrix.ndim == 1 else 'dense'}")
        print(f"  MCMC steps/iter: {n_mcmc_steps}, Target ESS: {target_ess:.0%}")
        print(f"  Score batch size: {score_batch_size}")

    # =========================================================================
    # 6. Warmup JIT (compile before timing)
    # =========================================================================
    if verbose:
        print("  JIT compiling SMC step (first call)...")

    warmup_key, rng_key = jax.random.split(rng_key)
    _warmup_state, _warmup_info = jit_step(warmup_key, state)
    jax.block_until_ready(_warmup_state.particles)

    if verbose:
        print("  JIT compilation done.")

    # Reset state (don't use warmup result)
    state = tempered_smc.init(initial_positions)

    # =========================================================================
    # 7. Main SMC loop
    # =========================================================================
    t0 = time.perf_counter()

    info_history = []
    best_positions = []
    best_scores = []
    step_count = 0

    if record_best:
        pos, score, mean_score, std_score = get_best_stats(state.particles)
        jax.block_until_ready(score)
        best_positions.append(np.array(pos))
        best_scores.append(float(score))
        if verbose:
            print(f"Initial | Best: {score:.2f}, Mean: {mean_score:.2f}, "
                  f"Std: {std_score:.2f}")

    while state.tempering_param < 1.0:
        rng_key, step_key = jax.random.split(rng_key)

        state, info = jit_step(step_key, state)
        jax.block_until_ready(state.particles)

        info_history.append(info)
        step_count += 1

        if record_best:
            pos, score, mean_score, std_score = get_best_stats(state.particles)
            jax.block_until_ready(score)
            best_positions.append(np.array(pos))
            best_scores.append(float(score))

        if verbose:
            acc_str = ""
            try:
                update_info = info.update_info
                if hasattr(update_info, 'acceptance_rate'):
                    acc_rate = float(jnp.mean(update_info.acceptance_rate))
                    acc_str = f" | Accept: {acc_rate:.2%}"
            except (AttributeError, TypeError):
                pass

            print(
                f"Step {step_count:3d} | λ = {float(state.tempering_param):.4f} | "
                f"Best: {best_scores[-1]:.2f} | Mean: {float(mean_score):.2f}"
                f"{acc_str}"
            )

    dt = time.perf_counter() - t0
    if verbose:
        print(f"\nSMC completed in {dt:.2f}s ({step_count} temperature steps)")
        print(f"  Average: {dt/max(step_count,1):.2f}s per step")

    if record_best and best_positions:
        best_positions = np.stack(best_positions, axis=0)
        best_scores = np.array(best_scores)
    else:
        best_positions = None
        best_scores = None

    return state, info_history, best_positions, best_scores


def get_smc_samples(state) -> jnp.ndarray:
    """Extract particles from SMC state."""
    return state.particles


def get_best_sample(state, log_prob_fn, batch_size: int = 16) -> Tuple[jnp.ndarray, float]:
    """Identify the single best particle (memory-safe)."""
    particles = state.particles
    scores = batched_vmap(log_prob_fn, batch_size=batch_size)(particles)
    best_idx = jnp.argmax(scores)
    return particles[best_idx], float(scores[best_idx])