"""
SMC sampler using BlackJAX's tempered SMC with RMH kernel.
"""
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from typing import Callable
import time


def run_tempered_smc(
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    log_prob_fn: Callable,
    initial_positions: jnp.ndarray,
    rng_key,
    n_mcmc_steps: int = 10,
    rmh_sigma: float = 1.0,
    target_ess: float = 0.5,
    record_best: bool = True,
):
    """
    Run BlackJAX adaptive tempered SMC with RMH mutation kernel.
    
    Args:
        log_prior_fn: Log prior probability function
        log_likelihood_fn: Log likelihood function  
        initial_positions: (n_particles, n_dims) initial particle positions
        rng_key: JAX PRNGKey
        n_mcmc_steps: MCMC steps per temperature level
        rmh_sigma: RMH proposal standard deviation
        target_ess: Target ESS ratio for adaptive tempering (0.5 = 50%)
        
    Returns:
        final_state: TemperedSMCState with final particles
        info_history: List of step info
        best_positions: (n_steps, n_dims) best sample per tempering step (if record_best)
        best_scores: (n_steps,) best scores per step (if record_best)
    """
    n_particles, n_dims = initial_positions.shape
    
    # Create RMH kernel functions
    def mcmc_init_fn(position, logdensity_fn):
        """Initialize RMH state."""
        return blackjax.rmh.init(position, logdensity_fn)
    
    def mcmc_step_fn(rng_key, state, logdensity_fn, **mcmc_params):
        """Single RMH step."""
        sigma = mcmc_params.get("sigma", rmh_sigma)
        proposal = blackjax.mcmc.random_walk.normal(sigma)
        kernel = blackjax.rmh.build_kernel()
        return kernel(rng_key, state, logdensity_fn, proposal)
    
    # Build adaptive tempered SMC kernel
    smc_kernel = blackjax.smc.adaptive_tempered.build_kernel(
        logprior_fn=log_prior_fn,
        loglikelihood_fn=log_likelihood_fn,
        mcmc_step_fn=mcmc_step_fn,
        mcmc_init_fn=mcmc_init_fn,
        resampling_fn=resampling.systematic,
        target_ess=target_ess,
    )
    
    # Initialize state
    state = blackjax.smc.tempered.init(initial_positions)
    
    print(f"Running BlackJAX Adaptive Tempered SMC")
    print(f"  Particles: {n_particles}, MCMC steps: {n_mcmc_steps}, σ: {rmh_sigma}")
    t0 = time.time()
    
    # JIT compile kernel
    @jax.jit
    def step(key, state):
        return smc_kernel(
            key, state, 
            num_mcmc_steps=n_mcmc_steps,
            mcmc_parameters={"sigma": jnp.array([rmh_sigma])}  # Array with single value
        )
    
    info_history = []
    best_positions = []
    best_scores = []
    step_count = 0
    
    # Run until tempering_param reaches 1.0
    while state.tempering_param < 1.0:
        rng_key, step_key = jax.random.split(rng_key)
        state, info = step(step_key, state)
        info_history.append(info)
        step_count += 1
        
        if record_best and log_prob_fn is not None:
            step_log_probs = jax.vmap(log_prob_fn)(state.particles)
            best_idx = jnp.argmax(step_log_probs)
            best_positions.append(state.particles[best_idx])
            best_scores.append(step_log_probs[best_idx])

        if step_count % 5 == 0 or state.tempering_param >= 1.0:
            print(f"Step {step_count:3d} | λ = {state.tempering_param:.4f}")
    
    dt = time.time() - t0
    print(f"SMC completed in {dt:.2f}s ({step_count} temperature steps)")
    
    if record_best and best_positions:
        best_positions = jnp.stack(best_positions, axis=0)
        best_scores = jnp.array(best_scores)
    else:
        best_positions = None
        best_scores = None

    return state, info_history, best_positions, best_scores


def get_smc_samples(state):
    """Extract particle positions from SMC state."""
    return state.particles


def get_best_sample(state, log_prob_fn):
    """Get the highest probability sample from SMC output."""
    positions = state.particles
    log_probs = jax.vmap(log_prob_fn)(positions)
    best_idx = jnp.argmax(log_probs)
    return positions[best_idx], float(log_probs[best_idx])
