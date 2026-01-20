"""
MCMC Sampling wrappers using BlackJAX RMH (Random-walk Metropolis-Hastings).
"""
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Callable, Optional, Tuple, Union


def create_rmh_kernel(log_prob_fn: Callable, sigma: float = 1.0):
    """
    Create a BlackJAX RMH kernel with Gaussian proposals.
    
    Args:
        log_prob_fn: Log probability function (flat_coords -> scalar)
        sigma: Standard deviation for isotropic Gaussian proposal
        
    Returns:
        BlackJAX RMH kernel object with .init() and .step() methods
    """
    proposal = blackjax.mcmc.random_walk.normal(sigma)
    kernel = blackjax.rmh(log_prob_fn, proposal)
    return kernel


def run_rmh_sampling(
    rng_key: jax.Array,
    log_prob_fn: Callable,
    initial_position: jax.Array,
    n_steps: int = 1000,
    sigma: float = 1.0,
    burnin: int = 0,
    thin: int = 1,
    save_interval: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Random-walk Metropolis-Hastings sampling using BlackJAX.
    
    Args:
        rng_key: JAX random key
        log_prob_fn: Log probability function to sample from
        initial_position: Starting position (flat array)
        n_steps: Total number of MCMC steps
        sigma: Proposal standard deviation
        burnin: Number of initial steps to discard
        thin: Save every `thin` samples after burnin
        save_interval: How often to print progress (0 = no printing)
        verbose: Whether to print progress
        
    Returns:
        positions: Array of saved positions (n_saved, n_dims)
        log_probs: Log probabilities at saved positions
        acceptance_rate: Overall acceptance rate
    """
    # Create kernel
    kernel = create_rmh_kernel(log_prob_fn, sigma)
    
    # Initialize state
    initial_position = jnp.asarray(initial_position)
    state = kernel.init(initial_position)
    
    # JIT compile step function
    @jax.jit
    def step_fn(state, key):
        new_state, info = kernel.step(key, state)
        return new_state, info
    
    # Pre-split keys
    keys = jax.random.split(rng_key, n_steps)
    
    # Storage
    positions = []
    log_probs = []
    accepts = []
    
    if verbose:
        print(f"Running RMH sampling: {n_steps} steps, σ={sigma}")
        print(f"  Burnin: {burnin}, Thin: {thin}")
    
    t0 = time.time()
    curr_state = state
    
    # Print interval
    print_every = max(1, n_steps // 10) if save_interval > 0 else n_steps + 1
    
    for i in range(n_steps):
        curr_state, info = step_fn(curr_state, keys[i])
        accepts.append(float(info.is_accepted))
        
        # After burnin, save every `thin` steps
        if i >= burnin and (i - burnin) % thin == 0:
            positions.append(np.array(curr_state.position))
            log_probs.append(float(curr_state.logdensity))
        
        # Progress printing
        if verbose and (i + 1) % print_every == 0:
            recent_acc = np.mean(accepts[-min(1000, len(accepts)):])
            print(f"  Step {i+1:6d}/{n_steps} | LogProb: {curr_state.logdensity:10.2f} | Accept: {recent_acc:.1%}")
    
    dt = time.time() - t0
    overall_acc = np.mean(accepts)
    
    if verbose:
        print(f"Completed in {dt:.2f}s ({n_steps/dt:.0f} steps/s)")
        print(f"Overall acceptance rate: {overall_acc:.1%}")
        print(f"Saved {len(positions)} samples")
    
    return np.array(positions), np.array(log_probs), overall_acc


def run_annealed_rmh(
    rng_key: jax.Array,
    log_prob_fn: Callable,
    initial_position: jax.Array,
    n_steps: int = 1000,
    sigma: float = 1.0,
    temp_start: float = 10.0,
    temp_end: float = 1.0,
    save_every: int = 10,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run RMH with simulated annealing (temperature schedule).
    
    Samples from p(x)^(1/T) where T decreases from temp_start to temp_end.
    This helps escape local minima during early exploration.
    
    Args:
        rng_key: JAX random key
        log_prob_fn: Log probability function (at T=1)
        initial_position: Starting position
        n_steps: Total MCMC steps
        sigma: Proposal standard deviation
        temp_start: Starting temperature (higher = flatter landscape)
        temp_end: Final temperature (1.0 = true distribution)
        save_every: Save position every N steps
        verbose: Print progress
        
    Returns:
        positions: Saved positions
        log_probs: Log probs at T=1 (not annealed)
        acceptance_rate: Overall acceptance rate
    """
    initial_position = jnp.asarray(initial_position)
    proposal = blackjax.mcmc.random_walk.normal(sigma)
    
    # Temperature schedule (geometric)
    temps = np.geomspace(temp_start, temp_end, n_steps)
    
    positions = []
    log_probs = []
    accepts = []
    
    if verbose:
        print(f"Running Annealed RMH: {n_steps} steps, σ={sigma}")
        print(f"  Temperature: {temp_start:.1f} → {temp_end:.1f}")
    
    t0 = time.time()
    curr_pos = initial_position
    curr_logp = log_prob_fn(curr_pos)
    
    keys = jax.random.split(rng_key, n_steps)
    print_every = max(1, n_steps // 10)
    
    for i in range(n_steps):
        T = temps[i]
        
        # Propose
        prop_key, accept_key = jax.random.split(keys[i])
        proposal_delta = jax.random.normal(prop_key, curr_pos.shape) * sigma
        prop_pos = curr_pos + proposal_delta
        prop_logp = log_prob_fn(prop_pos)
        
        # Accept/reject at temperature T
        log_alpha = (prop_logp - curr_logp) / T
        accept = jnp.log(jax.random.uniform(accept_key)) < log_alpha
        
        curr_pos = jnp.where(accept, prop_pos, curr_pos)
        curr_logp = jnp.where(accept, prop_logp, curr_logp)
        accepts.append(float(accept))
        
        # Save
        if (i + 1) % save_every == 0:
            positions.append(np.array(curr_pos))
            log_probs.append(float(curr_logp))
        
        if verbose and (i + 1) % print_every == 0:
            recent_acc = np.mean(accepts[-min(1000, len(accepts)):])
            print(f"  Step {i+1:6d}/{n_steps} | T={T:.3f} | LogProb: {curr_logp:10.2f} | Accept: {recent_acc:.1%}")
    
    dt = time.time() - t0
    overall_acc = np.mean(accepts)
    
    if verbose:
        print(f"Completed in {dt:.2f}s ({n_steps/dt:.0f} steps/s)")
        print(f"Overall acceptance rate: {overall_acc:.1%}")
        print(f"Saved {len(positions)} samples")
    
    return np.array(positions), np.array(log_probs), overall_acc


def run_parallel_rmh(
    rng_key: jax.Array,
    log_prob_fn: Callable,
    initial_positions: jax.Array,
    n_steps: int = 1000,
    sigma: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run multiple independent RMH chains in parallel using vmap.
    
    Args:
        rng_key: JAX random key
        log_prob_fn: Log probability function
        initial_positions: Array of starting positions (n_chains, n_dims)
        n_steps: Steps per chain
        sigma: Proposal standard deviation
        verbose: Print progress
        
    Returns:
        final_positions: Final position of each chain (n_chains, n_dims)
        final_log_probs: Final log prob of each chain (n_chains,)
        mean_acceptance: Mean acceptance rate across chains
    """
    n_chains = initial_positions.shape[0]
    
    # Create kernel
    kernel = create_rmh_kernel(log_prob_fn, sigma)
    
    # Initialize all chains
    initial_positions = jnp.asarray(initial_positions)
    init_states = jax.vmap(kernel.init)(initial_positions)
    
    # JIT step function
    @jax.jit
    def step_fn(state, key):
        new_state, info = kernel.step(key, state)
        return new_state, info
    
    # Vectorized step
    @jax.jit
    def parallel_step(states, keys):
        return jax.vmap(step_fn)(states, keys)
    
    if verbose:
        print(f"Running {n_chains} parallel RMH chains: {n_steps} steps each, σ={sigma}")
    
    t0 = time.time()
    states = init_states
    total_accepts = 0
    
    print_every = max(1, n_steps // 10)
    
    for i in range(n_steps):
        rng_key, step_key = jax.random.split(rng_key)
        chain_keys = jax.random.split(step_key, n_chains)
        states, infos = parallel_step(states, chain_keys)
        total_accepts += jnp.sum(infos.is_accepted)
        
        if verbose and (i + 1) % print_every == 0:
            mean_logp = jnp.mean(states.logdensity)
            print(f"  Step {i+1:6d}/{n_steps} | Mean LogProb: {mean_logp:10.2f}")
    
    dt = time.time() - t0
    mean_acc = float(total_accepts) / (n_chains * n_steps)
    
    if verbose:
        print(f"Completed in {dt:.2f}s")
        print(f"Mean acceptance rate: {mean_acc:.1%}")
    
    return np.array(states.position), np.array(states.logdensity), mean_acc


# Legacy API for backward compatibility
def setup_rmh(log_prob_fn, step_size: float, inverse_mass_matrix=None):
    """
    Setup Random Walk Metropolis-Hastings algorithm.
    
    DEPRECATED: Use create_rmh_kernel() instead.
    """
    if inverse_mass_matrix is not None:
        proposal = blackjax.mcmc.random_walk.normal(jnp.sqrt(jnp.diag(inverse_mass_matrix)))
    else:
        proposal = blackjax.mcmc.random_walk.normal(step_size)
    
    rmh = blackjax.rmh(log_prob_fn, proposal)
    return rmh


def run_chain(
    rng_key,
    kernel,
    initial_state_or_position,
    n_steps: int,
    log_prob_fn=None,
):
    """
    Run MCMC chain.
    
    DEPRECATED: Use run_rmh_sampling() instead.
    """
    if isinstance(initial_state_or_position, (np.ndarray, jnp.ndarray)):
        if log_prob_fn is None:
            raise ValueError("log_prob_fn required to initialize state")
        state = kernel.init(initial_state_or_position)
    else:
        state = initial_state_or_position

    @jax.jit
    def one_step(state, key):
        new_state, info = kernel.step(key, state)
        return new_state, info

    keys = jax.random.split(rng_key, n_steps)
    
    positions = []
    log_probs = []
    accepts = []
    
    curr_state = state
    
    print(f"Starting sampling for {n_steps} steps...")
    t0 = time.time()
    
    for i, k in enumerate(keys):
        curr_state, info = one_step(curr_state, k)
        
        positions.append(curr_state.position)
        log_probs.append(curr_state.logdensity)
        accepts.append(info.is_accepted)
        
        if n_steps >= 10 and (i+1) % (n_steps // 10) == 0:
            acc = np.mean(accepts[-1000:])
            print(f"Step {i+1}/{n_steps} | LogProb: {curr_state.logdensity:.2f} | Accept: {acc:.2%}")
            
    dt = time.time() - t0
    print(f"Finished in {dt:.2f}s ({n_steps/dt:.1f} steps/s)")
    
    return np.array(positions), np.array(log_probs), np.mean(accepts)
