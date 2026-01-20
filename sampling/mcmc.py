"""
MCMC Sampling wrappers using BlackJAX.
"""
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import time

def setup_rmh(log_prob_fn, step_size: float, inverse_mass_matrix=None):
    """
    Setup Random Walk Metropolis algorithm.
    """
    # Simple isotropic normal proposal
    proposal = blackjax.mcmc.random_walk.normal(step_size)
    
    # Or if inverse_mass_matrix is provided (typically for HMC/NUTS, but RMH can use it for scaling of proposal)
    # For simple RMH with normal proposal, sigma is scalar or vector.
    
    rmh = blackjax.rmh(log_prob_fn, proposal)
    return rmh

def run_chain(
    rng_key,
    kernel,
    initial_state_or_position,
    n_steps: int,
    log_prob_fn=None, # needed if initializing state from position
):
    """
    Run MCMC chain.
    """
    # Initialize state if needed
    if isinstance(initial_state_or_position, (np.ndarray, jnp.ndarray)):
        if log_prob_fn is None:
            raise ValueError("log_prob_fn required to initialize state")
        state = kernel.init(initial_state_or_position)
    else:
        state = initial_state_or_position

    # JIT the step function
    @jax.jit
    def one_step(state, key):
        new_state, info = kernel.step(key, state)
        return new_state, info

    # Run loop
    keys = jax.random.split(rng_key, n_steps)
    
    # We'll use a python loop for progress tracking, 
    # but for max speed jax.lax.scan is better. 
    # Given the user wants "simple" and "logging", Python loop is okay for reasonable n_steps.
    # To mitigate Python overhead, we can scan in chunks. Let's do pure python for simplicity/debuggability first.
    
    positions = []
    log_probs = []
    accepts = []
    
    curr_state = state
    
    print(f"Starting sampling for {n_steps} steps...")
    t0 = time.time()
    
    # Simple Python loop
    for i, k in enumerate(keys):
        curr_state, info = one_step(curr_state, k)
        
        positions.append(curr_state.position)
        log_probs.append(curr_state.logdensity)
        accepts.append(info.is_accepted)
        
        if (i+1) % (n_steps // 10) == 0:
            acc = np.mean(accepts[-1000:])
            print(f"Step {i+1}/{n_steps} | LogProb: {curr_state.logdensity:.2f} | Accept: {acc:.2%}")
            
    dt = time.time() - t0
    print(f"Finished in {dt:.2f}s ({n_steps/dt:.1f} steps/s)")
    
    return np.array(positions), np.array(log_probs), np.mean(accepts)
