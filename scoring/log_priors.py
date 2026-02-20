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
    
    @staticmethod
    def log_soft_box_prior(flat_coords, radii, box_mins, box_maxs, steepness=10.0):
        """
        Soft bounding box prior that penalizes particles near/outside box boundaries.
        
        For HMC compatibility, uses smooth differentiable penalty instead of hard walls.
        
        Args:
            flat_coords: (n_particles * 3,) flattened coordinates
            radii: (n_particles,) particle radii
            box_mins: (3,) minimum box boundaries [x_min, y_min, z_min]
            box_maxs: (3,) maximum box boundaries [x_max, y_max, z_max]
            steepness: controls how sharply the penalty increases near boundaries
            
        Returns:
            log_prior: scalar, 0 when all particles inside box, increasingly negative as
                    particles approach/cross boundaries
        
        Implementation:
            Uses smooth quadratic penalty that grows as particle surfaces approach boundaries.
            For particle i at position r_i with radius R_i:
                - Inner box: [box_min + R_i, box_max - R_i]
                - Penalty = -steepness * sum of squared violations
        """
        coords = flat_coords.reshape(-1, 3)  # (n_particles, 3)
        n_particles = coords.shape[0]
        
        # Convert to JAX arrays
        box_mins = jnp.array(box_mins)
        box_maxs = jnp.array(box_maxs)
        radii = jnp.array(radii)
        
        # For each particle, effective boundaries account for radius
        # Shape: (n_particles, 3)
        effective_mins = box_mins[None, :] + radii[:, None]  # particle surface shouldn't go below this
        effective_maxs = box_maxs[None, :] - radii[:, None]  # particle surface shouldn't go above this
        
        # Calculate violations (how far particle extends beyond boundaries)
        # Positive values mean violation
        lower_violations = effective_mins - coords  # positive if too low
        upper_violations = coords - effective_maxs  # positive if too high
        
        # Smooth penalty using ReLU-like function (only penalize violations)
        # Using jnp.maximum(0, x)^2 for smooth gradients
        lower_penalty = jnp.sum(jnp.maximum(0.0, lower_violations)**2)
        upper_penalty = jnp.sum(jnp.maximum(0.0, upper_violations)**2)
        
        total_penalty = lower_penalty + upper_penalty
        
        # Return log prior (negative of penalty)
        log_prior = -steepness * total_penalty
        
        return log_prior
    
    @staticmethod
    def log_linear_slope_prior(flat_coords, map_com, slope_factor=0.01):
        """Compute a prior penalty based on the fact that for the EM score 
            the function is exp(-lambda d) where d is the distance of the particles 
            from the center of mass of the density map.
            This encourages particles to be closer to the center of mass, which is 
            where the density map is strongest, and thus can help guide the SMC 
            sampling towards more promising regions of the parameter space.
        """
        coords = flat_coords.reshape(-1, 3)  # (n_particles, 3)
        map_com = jnp.array(map_com)  # (3,)
        distances = jnp.linalg.norm(coords - map_com, axis=1)  # (n_particles,)
        log_prior = -slope_factor * jnp.sum(distances)
        return log_prior