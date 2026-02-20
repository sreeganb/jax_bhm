"""
Normalized differentiable log-likelihood functions for CCC-based EM density scoring.

Each function is the proper log-pdf of the corresponding distribution
evaluated at x = (1 - CCC), so the scale parameters (sigma, b, gamma)
can be sampled as nuisance parameters without diverging.

    gaussian_ll:  -log(sigma) - (1-CCC)^2 / (2*sigma^2)         + const
    laplace_ll:   -log(2b)    - |1-CCC| / b                     + const
    cauchy_ll:    -log(pi*gamma) - log(1 + ((1-CCC)/gamma)^2)   + const

All are JAX-differentiable with no control flow or hard boundaries.
"""

import jax.numpy as jnp

_EPS = 1e-8


def gaussian_ll(ccc: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Normalized Gaussian log-likelihood on mismatch x = 1 - CCC.

    log p(x | sigma) = -0.5*log(2*pi) - log(sigma) - x^2 / (2*sigma^2)
    """
    x = 1.0 - ccc
    return -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - x ** 2 / (2.0 * sigma ** 2)


def laplace_ll(ccc: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Normalized Laplace log-likelihood on mismatch x = 1 - CCC.

    log p(x | b) = -log(2b) - |x| / b

    Uses smooth abs (sqrt(x^2 + eps)) for gradient stability.
    """
    x = 1.0 - ccc
    return -jnp.log(2.0 * b) - jnp.sqrt(x ** 2 + _EPS) / b


def cauchy_ll(ccc: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """Normalized Cauchy log-likelihood on mismatch x = 1 - CCC.

    log p(x | gamma) = -log(pi * gamma) - log(1 + (x/gamma)^2)
    """
    x = 1.0 - ccc
    return -jnp.log(jnp.pi * gamma) - jnp.log1p((x / gamma) ** 2)


def differentiable_ccc(density_sim: jnp.ndarray, density_ref: jnp.ndarray) -> jnp.ndarray:
    """Pearson CCC with epsilon-guarded denominator for gradient stability."""
    a = density_sim.flatten().astype(jnp.float32)
    b = density_ref.flatten().astype(jnp.float32)
    a_c = a - jnp.mean(a)
    b_c = b - jnp.mean(b)
    num = jnp.sum(a_c * b_c)
    den = jnp.sqrt(jnp.sum(a_c ** 2) * jnp.sum(b_c ** 2) + _EPS)
    return num / den