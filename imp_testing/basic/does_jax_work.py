import jax
import jax.numpy as jnp
import time
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Verify device
print(f"JAX Devices: {jax.devices()}")

# Simple Benchmark Function
def benchmark_m2():
    size = 3000
    x = jnp.ones((size, size))
    
    # Warm up (JIT compilation)
    jnp.dot(x, x).block_until_ready()
    
    start = time.time()
    result = jnp.dot(x, x).block_until_ready()
    end = time.time()
    
    print(f"M2 CPU Matrix Multi ({size}x{size}): {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark_m2()
