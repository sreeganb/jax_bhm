import jax
import jax.numpy as jnp

class Priors:
    @staticmethod
    def log_uniform_prior(x, lower_bound, upper_bound):
        """Uniform prior between lower_bound and upper_bound (vectorized)."""
        lb = jnp.asarray(lower_bound)
        ub = jnp.asarray(upper_bound)
        in_bounds = (x >= lb) & (x <= ub)
        logp = -jnp.log(ub - lb)
        return jnp.where(in_bounds, logp, -jnp.inf)

    @staticmethod
    def log_jeffreys_prior(x, lower_bound, upper_bound):
        """Jeffreys prior between lower_bound and upper_bound (vectorized)."""
        lb = jnp.asarray(lower_bound)
        ub = jnp.asarray(upper_bound)
        in_bounds = (x >= lb) & (x <= ub)
        norm = -jnp.log(jnp.log(ub / lb))
        logp = -jnp.log(x) + norm
        return jnp.where(in_bounds, logp, -jnp.inf)
    
    @staticmethod
    def log_inverse_gamma_prior(x, alpha, beta):
        """Log of inverse gamma prior."""
        coeff = alpha * jnp.log(beta) - jax.lax.lgamma(alpha)
        logp = coeff - (alpha + 1) * jnp.log(x) - beta / x
        return logp
    
    @staticmethod
    def log_half_cauchy_prior(x, scale):
        """Log of half-Cauchy prior."""
        logp = jnp.log(2 / (jnp.pi * scale * (1 + (x / scale) ** 2)))
        return logp