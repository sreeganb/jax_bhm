"""
SMC sampler using BlackJAX's tempered SMC with RMH kernel.

Key insight from BlackJAX docs:
- Use `blackjax.adaptive_tempered_smc()` constructor (not build_kernel)
- Pass `blackjax.rmh.build_kernel()` directly as the mcmc_kernel
- Pass `blackjax.rmh.init` as the mcmc_init function
- Use `extend_params()` to wrap MCMC parameters
"""
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params  # IMPORTANT: This wraps params correctly
from typing import Callable, Tuple, List, Any
import time


def run_tempered_smc(
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    log_prob_fn: Callable,
    initial_positions: jnp.ndarray,
    rng_key: jax.Array,
    n_mcmc_steps: int = 10,
    rmh_sigma: float = 1.0,
    target_ess: float = 0.5,
    record_best: bool = True,
    verbose: bool = True,
) -> Tuple[Any, List, np.ndarray, np.ndarray]:
    """
    Run BlackJAX adaptive tempered SMC with RMH mutation kernel.
    
    This follows the BlackJAX documentation pattern exactly.
    """
    n_particles, n_dims = initial_positions.shape
    
    # =========================================================================
    # STEP 1: Define the RMH proposal distribution
    # =========================================================================
    # For RMH, we need a proposal function that generates new positions
    # blackjax.mcmc.random_walk.normal(sigma) returns such a function
    
    # The RMH kernel needs a "proposal_distribution" which is a callable
    # that takes (rng_key, position) and returns a new proposed position
    
    def rmh_proposal_distribution(rng_key, position):
        """
        Normal random walk proposal.
        position has shape (n_dims,) for a single particle.
        Returns new position with same shape.
        """
        return position + jax.random.normal(rng_key, shape=position.shape) * rmh_sigma
    
    # =========================================================================
    # STEP 2: Create the RMH kernel using blackjax
    # =========================================================================
    # blackjax.rmh.build_kernel() returns a kernel function with signature:
    #   kernel(rng_key, state, logdensity_fn, proposal_distribution) -> (new_state, info)
    #
    # For SMC, we need to wrap this so it matches what adaptive_tempered_smc expects
    
    rmh_kernel = blackjax.rmh.build_kernel()
    
    # =========================================================================
    # STEP 3: Define mcmc_step_fn that SMC will call
    # =========================================================================
    # SMC calls: mcmc_step_fn(rng_key, state, logdensity_fn) -> (new_state, info)
    # Note: The logdensity_fn passed by SMC is the TEMPERED density at current lambda
    
    def mcmc_step_fn(rng_key, state, logdensity_fn):
        """
        One step of RMH mutation.
        
        Args:
            rng_key: JAX random key
            state: RMHState with .position and .logdensity
            logdensity_fn: The tempered log density (prior + lambda * likelihood)
        
        Returns:
            new_state, info
        """
        return rmh_kernel(rng_key, state, logdensity_fn, rmh_proposal_distribution)
    
    # =========================================================================
    # STEP 4: Define mcmc_init_fn
    # =========================================================================
    # This initializes the MCMC state for each particle
    # blackjax.rmh.init(position, logdensity_fn) -> RMHState
    
    mcmc_init_fn = blackjax.rmh.init
    
    # =========================================================================
    # STEP 5: Build the adaptive tempered SMC sampler
    # =========================================================================
    # Use the HIGH-LEVEL constructor, NOT build_kernel
    # This matches the documentation pattern exactly
    
    tempered_smc = blackjax.adaptive_tempered_smc(
        logprior_fn=log_prior_fn,
        loglikelihood_fn=log_likelihood_fn,
        mcmc_step_fn=mcmc_step_fn,
        mcmc_init_fn=mcmc_init_fn,
        mcmc_parameters={},  # RMH doesn't need extra params (sigma is in proposal)
        resampling_fn=resampling.systematic,
        target_ess=target_ess,
        num_mcmc_steps=n_mcmc_steps,
    )
    
    # =========================================================================
    # STEP 6: Initialize SMC state
    # =========================================================================
    # tempered_smc.init() takes the initial positions array (n_particles, n_dims)
    
    if verbose:
        print("Initializing SMC...")
    
    state = tempered_smc.init(initial_positions)
    
    if verbose:
        print(f"Running BlackJAX Adaptive Tempered SMC")
        print(f"  Particles: {n_particles}, MCMC steps: {n_mcmc_steps}, σ: {rmh_sigma}")
        print(f"  Target ESS: {target_ess:.0%}")
    
    # =========================================================================
    # STEP 7: Helper to compute statistics
    # =========================================================================
    
    def get_best_stats(particles):
        """Find best particle and score in current population."""
        scores = jax.vmap(log_prob_fn)(particles)
        idx = jnp.argmax(scores)
        return particles[idx], scores[idx], jnp.mean(scores), jnp.std(scores)
    
    # =========================================================================
    # STEP 8: Run the SMC loop
    # =========================================================================
    # The loop runs until state.tempering_param reaches 1.0
    
    t0 = time.time()
    
    info_history = []
    best_positions = []
    best_scores = []
    step_count = 0
    
    # Record initial state
    if record_best:
        pos, score, mean_score, std_score = get_best_stats(state.particles)
        best_positions.append(np.array(pos))
        best_scores.append(float(score))
        if verbose:
            print(f"Initial | Best: {score:.2f}, Mean: {mean_score:.2f}, Std: {std_score:.2f}")

    # Main SMC loop - run until lambda = 1
    while state.tempering_param < 1.0:
        # Split random key
        rng_key, step_key = jax.random.split(rng_key)
        
        # Store old particles to measure movement
        old_particles = np.array(state.particles)
        
        # =====================================================================
        # STEP 8a: Take one SMC step using tempered_smc.step()
        # =====================================================================
        # This does: 
        #   1. Compute weights based on likelihood change
        #   2. Resample particles if ESS too low
        #   3. Run n_mcmc_steps of RMH mutation on each particle
        #   4. Update tempering parameter adaptively
        
        state, info = tempered_smc.step(step_key, state)
        
        # Block until computation is done (for accurate timing)
        _ = state.tempering_param.block_until_ready()
        
        info_history.append(info)
        step_count += 1
        
        # Compute movement statistics
        new_particles = np.array(state.particles)
        movement = np.linalg.norm(new_particles - old_particles, axis=1)
        mean_movement = float(np.mean(movement))
        
        # Record best
        if record_best:
            pos, score, mean_score, std_score = get_best_stats(state.particles)
            best_positions.append(np.array(pos))
            best_scores.append(float(score))

        if verbose:
            print(
                f"Step {step_count:3d} | λ = {float(state.tempering_param):.4f} | "
                f"Best: {best_scores[-1]:.2f} | Mean: {mean_score:.2f} | "
                f"Move: {mean_movement:.2f}"
            )
    
    dt = time.time() - t0
    if verbose:
        print(f"\nSMC completed in {dt:.2f}s ({step_count} temperature steps)")
    
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
    scores = jax.vmap(log_prob_fn)(particles)
    best_idx = jnp.argmax(scores)
    return particles[best_idx], float(scores[best_idx])