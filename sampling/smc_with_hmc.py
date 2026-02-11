"""
SMC sampler using BlackJAX's adaptive tempered SMC with HMC mutation kernel.

Key differences from RMH version:
- HMC requires differentiable log-densities (no hard -inf boundaries!)
- HMC kernel needs step_size, inverse_mass_matrix, num_integration_steps
- HMC state tracks logdensity_grad in addition to logdensity
- Better exploration per step than RMH, but more expensive per step

GPU/JIT notes:
- All scoring functions must use jax.numpy (not numpy) for GPU compatibility
- The SMC loop itself is Python-level (not jit-compiled) because the number
  of tempering steps is unknown ahead of time. However, each step internally
  is fully JIT-compiled by BlackJAX.
- For GPU: remove JAX_PLATFORM_NAME="cpu" and ensure CUDA is available.
"""
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
from typing import Callable, Tuple, List, Any, Optional
import time


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
) -> Tuple[Any, List, np.ndarray, np.ndarray]:
    """
    Run BlackJAX adaptive tempered SMC with HMC mutation kernel.

    IMPORTANT: log_prior_fn and log_likelihood_fn must be JAX-differentiable.
    Avoid hard boundaries (-inf); use soft penalties instead (see example script).

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
        JAX PRNG key.
    n_mcmc_steps : int
        Number of HMC mutation steps per SMC temperature step.
    hmc_step_size : float
        Leapfrog integrator step size. Start small (0.001-0.01).
    hmc_inverse_mass_matrix : jnp.ndarray or None
        Diagonal inverse mass matrix, shape (n_dims,). None = identity.
    hmc_num_integration_steps : int
        Number of leapfrog steps per HMC proposal (L). Typical: 5-50.
    target_ess : float
        Target ESS ratio for adaptive tempering (0-1).
    record_best : bool
        Track best particle per step.
    verbose : bool
        Print progress.

    Returns
    -------
    state : SMCState
        Final SMC state with particles at lambda=1.
    info_history : list
        SMCInfo from each tempering step.
    best_positions : np.ndarray or None
        Shape (n_steps+1, n_dims) - best particle at each step.
    best_scores : np.ndarray or None
        Shape (n_steps+1,) - best score at each step.
    """
    n_particles, n_dims = initial_positions.shape

    # =========================================================================
    # 1. HMC parameters
    # =========================================================================
    if hmc_inverse_mass_matrix is None:
        hmc_inverse_mass_matrix = jnp.ones(n_dims)
    else:
        hmc_inverse_mass_matrix = jnp.asarray(hmc_inverse_mass_matrix)

    # Ensure step size is a JAX scalar for tracing
    hmc_step_size = jnp.float32(hmc_step_size)

    # =========================================================================
    # 2. Build HMC kernel
    # =========================================================================
    hmc_kernel = blackjax.hmc.build_kernel()

    # Close over HMC parameters so the signature matches what SMC expects:
    #   mcmc_step_fn(rng_key, state, logdensity_fn) -> (state, info)
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
    # 4. JIT-compile the step function
    # =========================================================================
    # This is the key optimization: the step function is compiled once and
    # reused for every tempering step. On GPU, this avoids repeated dispatch.
    jit_step = jax.jit(tempered_smc.step)

    # Also JIT the scoring for best-particle tracking
    jit_score_all = jax.jit(jax.vmap(log_prob_fn))

    def get_best_stats(particles):
        scores = jit_score_all(particles)
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

    # =========================================================================
    # 6. Warmup: trigger JIT compilation before timing
    # =========================================================================
    if verbose:
        print("  JIT compiling SMC step (first call)...")

    warmup_key, rng_key = jax.random.split(rng_key)
    _warmup_state, _warmup_info = jit_step(warmup_key, state)
    jax.block_until_ready(_warmup_state.particles)

    if verbose:
        print("  JIT compilation done.")

    # Reset state (don't use the warmup result)
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

        # JIT-compiled SMC step (resample + HMC mutations + adapt temperature)
        state, info = jit_step(step_key, state)

        # Block for timing accuracy (no-op on CPU)
        jax.block_until_ready(state.particles)

        info_history.append(info)
        step_count += 1

        if record_best:
            pos, score, mean_score, std_score = get_best_stats(state.particles)
            jax.block_until_ready(score)
            best_positions.append(np.array(pos))
            best_scores.append(float(score))

        if verbose:
            # Extract HMC acceptance rate from info if available
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

    # Stack results
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


def get_best_sample(state, log_prob_fn) -> Tuple[jnp.ndarray, float]:
    """Identify the single best particle from the final population."""
    particles = state.particles
    scores = jax.jit(jax.vmap(log_prob_fn))(particles)
    best_idx = jnp.argmax(scores)
    return particles[best_idx], float(scores[best_idx])