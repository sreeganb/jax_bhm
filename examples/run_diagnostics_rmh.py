"""
Understanding the BlackJAX random_walk API correctly.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax
import blackjax
import blackjax.mcmc.random_walk as rw
import inspect

position = jnp.array([100.0, 200.0, 300.0])
sigma = 2.0
key = jax.random.PRNGKey(42)

print("=" * 60)
print("Understanding BlackJAX random_walk")
print("=" * 60)

# Look at additive_step_random_walk source
print("\n1. additive_step_random_walk source:")
src = inspect.getsource(rw.additive_step_random_walk)
print(src)

# So additive_step_random_walk takes a random_step function
# The random_step function should return a STEP (delta), not new position
# And additive_step_random_walk adds it to position internally

# Let's check normal() - it should return the STEP
print("\n2. normal() source:")
src = inspect.getsource(rw.normal)
print(src)

# AH HA! Let's see what normal() actually does
# It might be that normal() returns a step, but WE were using it wrong
# with blackjax.rmh() which expects a different kind of proposal

print("\n" + "=" * 60)
print("Testing the CORRECT usage")
print("=" * 60)

# The correct way is to use normal_random_walk which wraps everything correctly
def dummy_logprob(x):
    return -0.5 * jnp.sum(x**2)  # Simple Gaussian

# This should work correctly because it uses additive_step_random_walk internally
kernel = rw.normal_random_walk(dummy_logprob, sigma)

state = kernel.init(position)
print(f"Initial position: {state.position}")
print(f"Initial logdensity: {state.logdensity:.4f}")
print(f"Expected logdensity: {dummy_logprob(position):.4f}")

# Run a few steps
key = jax.random.PRNGKey(42)
accepts = []
for i in range(20):
    key, step_key = jax.random.split(key)
    new_state, info = kernel.step(step_key, state)
    accepts.append(float(info.is_accepted))
    if i < 5:
        diff = jnp.linalg.norm(new_state.position - state.position)
        print(f"Step {i}: accepted={info.is_accepted}, pos_diff={diff:.4f}, logp={new_state.logdensity:.4f}")
    state = new_state

print(f"\nAcceptance rate: {sum(accepts)/len(accepts):.1%}")

# Now let's check what blackjax.rmh expects
print("\n" + "=" * 60)
print("Checking blackjax.rmh")
print("=" * 60)

# blackjax.rmh is different from normal_random_walk
# Let's see its source
print("\nblackjax.rmh is an alias for:")
print(f"  {blackjax.rmh}")

# Check rmh_as_top_level_api
print("\nrmh_as_top_level_api source:")
try:
    src = inspect.getsource(rw.rmh_as_top_level_api)
    print(src[:1500])
except Exception as e:
    print(f"Error: {e}")

# The key insight: blackjax.rmh expects a TRANSITION GENERATOR
# that returns the NEW POSITION, not a step
# But normal_random_walk/additive_step_random_walk expect a STEP generator

print("\n" + "=" * 60)
print("SOLUTION: Use normal_random_walk, NOT blackjax.rmh + normal()")
print("=" * 60)

# Let's test with our actual problem
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
import numpy as np

types_config = {
    'A': {'radius': 24.0, 'copy': 8},
    'B': {'radius': 14.0, 'copy': 8},
    'C': {'radius': 16.0, 'copy': 16},
}

ideal_coords = get_ideal_coords()
box_size = 500.0

coords = ParticleSystem(types_config, {}, ideal_coords).get_random_coords(
    jax.random.PRNGKey(2387), box_size=[box_size, box_size, box_size], center_at_origin=True
)

system = ParticleSystem(types_config, coords, ideal_coords)
flat_radii = jnp.array(system.get_flat_radii())

target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
nuisance_params = {'AA': 1.5, 'AB': 1.2, 'BC': 1.0}

def log_prob_fn(flat_coords):
    in_box = jnp.all(jnp.abs(flat_coords) <= box_size)
    log_prior = jnp.where(in_box, 0.0, -jnp.inf)
    log_lik = log_probability(
        flat_coords, system, flat_radii,
        target_dists, nuisance_params,
        exclusion_weight=1.0, pair_weight=1.0, exvol_sigma=0.1
    )
    return log_prior + log_lik

initial_position = system.flatten(coords)
sigma = 2.0
n_steps = 100

print(f"\nInitial log prob: {log_prob_fn(initial_position):.2f}")

# USE normal_random_walk - this is the CORRECT API!
kernel = rw.normal_random_walk(log_prob_fn, sigma)
state = kernel.init(initial_position)

print(f"State initial logdensity: {state.logdensity:.2f}")

key = jax.random.PRNGKey(123)
accepts = []
for i in range(n_steps):
    key, step_key = jax.random.split(key)
    new_state, info = kernel.step(step_key, state)
    accepts.append(float(info.is_accepted))
    if i < 10 or i % 20 == 0:
        print(f"Step {i:3d}: logp={new_state.logdensity:.2f}, accepted={info.is_accepted}")
    state = new_state

print(f"\n{'='*60}")
print(f"FINAL RESULTS with normal_random_walk:")
print(f"  Acceptance rate: {np.mean(accepts):.1%}")
print(f"  Initial log prob: {log_prob_fn(initial_position):.2f}")
print(f"  Final log prob: {state.logdensity:.2f}")
print(f"{'='*60}")