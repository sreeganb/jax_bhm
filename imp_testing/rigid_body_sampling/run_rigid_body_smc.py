#!/usr/bin/env python3
"""
Run SMC sampling on IMP rigid body system using BlackJAX.

This script:
1. Creates a multi-protein rigid body system using IMP
2. Defines JAX-differentiable distance restraints
3. Runs BlackJAX tempered SMC with RMH mutation kernel
4. Saves best trajectory as RMF3 file
5. Optionally compares with simple RMH sampling

Usage:
    python run_rigid_body_smc.py --n_particles 50 --n_mcmc_steps 20
    python run_rigid_body_smc.py --method rmh --n_steps 5000
    python run_rigid_body_smc.py --output trajectory.rmf3
"""

import argparse
import time
import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Any, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling

# IMP imports
import IMP
import IMP.atom
import IMP.core
import IMP.display
import IMP.rmf
import RMF

# Local imports
from rigid_body_imp_system import (
    RigidBodyConfig,
    create_rigid_body_system,
    create_scoring_functions,
    generate_distance_pairs,
    get_coordinates_from_model,
    set_coordinates_to_model,
)


# =============================================================================
# RMF3 Trajectory Writer
# =============================================================================

class RMF3TrajectoryWriter:
    """
    Write IMP particle trajectories to RMF3 format.
    
    Usage:
        writer = RMF3TrajectoryWriter("output.rmf3", model, particles)
        writer.add_frame(coords, score=score, lambda_val=0.5)
        writer.add_frame(coords2, score=score2, lambda_val=1.0)
        writer.close()
    """
    
    def __init__(
        self,
        filename: str,
        model: IMP.Model,
        particles: List,
        particle_names: Optional[List[str]] = None,
        color_by_type: Optional[dict] = None,
    ):
        """
        Initialize RMF3 writer.
        
        Args:
            filename: Output RMF3 file path
            model: IMP Model
            particles: List of IMP particles to save
            particle_names: Optional names for particles
            color_by_type: Dict mapping particle type prefix to RGB tuple
        """
        self.filename = filename
        self.model = model
        self.particles = particles
        self.frame_count = 0
        
        # Create output directory if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Create RMF file
        self.rmf_file = RMF.create_rmf_file(filename)
        
        # Create a simple hierarchy for the particles
        self.root = IMP.atom.Hierarchy.setup_particle(
            IMP.Particle(model, "System")
        )
        
        # Add particles to hierarchy with proper decorators
        self.hierarchies = []
        for i, p in enumerate(particles):
            # Get or create name
            if particle_names:
                name = particle_names[i]
            else:
                name = p.get_name()
            
            # Create hierarchy node for this particle
            h = IMP.atom.Hierarchy(p)
            if not IMP.atom.Hierarchy.get_is_setup(p):
                h = IMP.atom.Hierarchy.setup_particle(p)
            
            # Ensure XYZR is setup
            if not IMP.core.XYZR.get_is_setup(p):
                IMP.core.XYZR.setup_particle(p, IMP.algebra.Sphere3D(
                    IMP.algebra.Vector3D(0, 0, 0), 1.0
                ))
            
            # Set color if provided
            if color_by_type:
                for prefix, color in color_by_type.items():
                    if name.startswith(prefix):
                        if not IMP.display.Colored.get_is_setup(p):
                            IMP.display.Colored.setup_particle(
                                p, IMP.display.Color(*color)
                            )
                        break
            
            self.root.add_child(h)
            self.hierarchies.append(h)
        
        # Add hierarchy to RMF
        IMP.rmf.add_hierarchy(self.rmf_file, self.root)
        
        # Add custom keys for score and lambda
        self.score_key = self.rmf_file.get_root_node().get_file().get_key(
            RMF.FLOAT, "score", True
        )
        self.lambda_key = self.rmf_file.get_root_node().get_file().get_key(
            RMF.FLOAT, "lambda", True
        )
        
        print(f"RMF3 writer initialized: {filename}")
    
    def add_frame(
        self,
        coords: np.ndarray,
        score: Optional[float] = None,
        lambda_val: Optional[float] = None,
        frame_name: Optional[str] = None,
    ):
        """
        Add a frame to the trajectory.
        
        Args:
            coords: (N, 3) array of particle coordinates
            score: Optional log probability score
            lambda_val: Optional tempering parameter (for SMC)
            frame_name: Optional name for this frame
        """
        # Update particle coordinates in IMP model
        for i, p in enumerate(self.particles):
            xyz = IMP.core.XYZ(p)
            xyz.set_coordinates(IMP.algebra.Vector3D(
                float(coords[i, 0]),
                float(coords[i, 1]),
                float(coords[i, 2]),
            ))
        
        # Create new frame
        if frame_name:
            frame_id = self.rmf_file.add_frame(frame_name)
        else:
            frame_id = self.rmf_file.add_frame(f"frame_{self.frame_count}")
        
        # Save coordinates to RMF
        IMP.rmf.save_frame(self.rmf_file)
        
        # Add score and lambda as frame attributes if provided
        root_node = self.rmf_file.get_root_node()
        if score is not None:
            try:
                root_node.set_value(self.score_key, float(score))
            except:
                pass  # Some RMF versions handle this differently
        
        if lambda_val is not None:
            try:
                root_node.set_value(self.lambda_key, float(lambda_val))
            except:
                pass
        
        self.frame_count += 1
    
    def close(self):
        """Close the RMF file."""
        del self.rmf_file
        print(f"RMF3 trajectory saved: {self.filename} ({self.frame_count} frames)")


def save_trajectory_to_rmf3(
    filename: str,
    model: IMP.Model,
    particles: List,
    trajectory: np.ndarray,
    scores: Optional[np.ndarray] = None,
    lambdas: Optional[np.ndarray] = None,
    particle_names: Optional[List[str]] = None,
    color_by_type: Optional[dict] = None,
):
    """
    Save a complete trajectory to RMF3 file.
    
    Args:
        filename: Output RMF3 file path
        model: IMP Model
        particles: List of IMP particles
        trajectory: (n_frames, n_particles, 3) coordinate array
        scores: Optional (n_frames,) array of scores
        lambdas: Optional (n_frames,) array of lambda values
        particle_names: Optional list of particle names
        color_by_type: Dict mapping type prefix to RGB color
    """
    writer = RMF3TrajectoryWriter(
        filename, model, particles, particle_names, color_by_type
    )
    
    n_frames = trajectory.shape[0]
    
    for i in range(n_frames):
        coords = trajectory[i]
        score = float(scores[i]) if scores is not None else None
        lambda_val = float(lambdas[i]) if lambdas is not None else None
        
        writer.add_frame(coords, score=score, lambda_val=lambda_val)
    
    writer.close()


# =============================================================================
# Simple RMH Sampler (for comparison)
# =============================================================================

def run_simple_rmh(
    log_prob_fn,
    initial_position: jnp.ndarray,
    rng_key: jax.Array,
    n_steps: int = 1000,
    sigma: float = 1.0,
    verbose: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simple Random Walk Metropolis-Hastings sampler.
    
    Args:
        log_prob_fn: Log probability function
        initial_position: Starting position
        rng_key: JAX random key
        n_steps: Number of MCMC steps
        sigma: Proposal standard deviation
        verbose: Print progress
        
    Returns:
        samples: (n_steps, n_dims) array of samples
        scores: (n_steps,) array of log probabilities
        acceptance_rate: Scalar acceptance rate
    """
    n_dims = initial_position.shape[0]
    
    # Initialize
    current_pos = initial_position
    current_score = log_prob_fn(current_pos)
    
    samples = []
    scores = []
    n_accepted = 0
    
    t0 = time.time()
    
    for step in range(n_steps):
        rng_key, prop_key, accept_key = jax.random.split(rng_key, 3)
        
        # Propose new position
        proposal = current_pos + jax.random.normal(prop_key, (n_dims,)) * sigma
        proposal_score = log_prob_fn(proposal)
        
        # Accept/reject
        log_alpha = proposal_score - current_score
        u = jax.random.uniform(accept_key)
        
        if jnp.log(u) < log_alpha:
            current_pos = proposal
            current_score = proposal_score
            n_accepted += 1
        
        samples.append(np.array(current_pos))
        scores.append(float(current_score))
        
        if verbose and (step + 1) % 500 == 0:
            acc_rate = n_accepted / (step + 1)
            print(f"Step {step+1:5d} | Score: {current_score:.2f} | "
                  f"Acc: {acc_rate:.1%}")
    
    dt = time.time() - t0
    acceptance_rate = n_accepted / n_steps
    
    if verbose:
        print(f"\nRMH completed in {dt:.2f}s")
        print(f"Acceptance rate: {acceptance_rate:.1%}")
        print(f"Final score: {scores[-1]:.2f}")
    
    return (
        jnp.array(samples),
        jnp.array(scores),
        acceptance_rate,
    )


# =============================================================================
# BlackJAX Tempered SMC with RMH
# =============================================================================

def run_tempered_smc(
    log_prior_fn,
    log_likelihood_fn,
    log_prob_fn,
    initial_positions: jnp.ndarray,
    rng_key: jax.Array,
    n_mcmc_steps: int = 10,
    rmh_sigma: float = 1.0,
    target_ess: float = 0.5,
    verbose: bool = True,
) -> Tuple[Any, List, jnp.ndarray, jnp.ndarray]:
    """
    Run BlackJAX adaptive tempered SMC with RMH mutation kernel.
    
    Args:
        log_prior_fn: Log prior function
        log_likelihood_fn: Log likelihood function
        log_prob_fn: Full log posterior (for diagnostics)
        initial_positions: (n_particles, n_dims) initial positions
        rng_key: JAX random key
        n_mcmc_steps: MCMC steps per temperature
        rmh_sigma: RMH proposal step size
        target_ess: Target effective sample size ratio
        verbose: Print progress
        
    Returns:
        final_state: SMC state at lambda=1
        info_history: List of SMC step info
        best_positions: Best particle position at each step
        best_scores: Best score at each step
    """
    n_particles, n_dims = initial_positions.shape
    
    # RMH proposal distribution
    def rmh_proposal(rng_key, position):
        return position + jax.random.normal(rng_key, shape=position.shape) * rmh_sigma
    
    # Build RMH kernel
    rmh_kernel = blackjax.rmh.build_kernel()
    
    def mcmc_step_fn(rng_key, state, logdensity_fn):
        return rmh_kernel(rng_key, state, logdensity_fn, rmh_proposal)
    
    mcmc_init_fn = blackjax.rmh.init
    
    # Build adaptive tempered SMC
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
    
    # Initialize
    state = tempered_smc.init(initial_positions)
    
    if verbose:
        print(f"Running BlackJAX Tempered SMC")
        print(f"  Particles: {n_particles}, Dims: {n_dims}")
        print(f"  MCMC steps: {n_mcmc_steps}, σ: {rmh_sigma}")
        print(f"  Target ESS: {target_ess:.0%}")
    
    # Helper to find best particle
    def get_best(particles):
        scores = jax.vmap(log_prob_fn)(particles)
        idx = jnp.argmax(scores)
        return particles[idx], scores[idx], jnp.mean(scores)
    
    t0 = time.time()
    
    info_history = []
    best_positions = []
    best_scores = []
    step_count = 0
    
    # Initial stats
    pos, score, mean_score = get_best(state.particles)
    best_positions.append(np.array(pos))
    best_scores.append(float(score))
    
    if verbose:
        print(f"Initial | Best: {score:.2f}, Mean: {mean_score:.2f}")
    
    # SMC loop until lambda = 1
    while state.tempering_param < 1.0:
        rng_key, step_key = jax.random.split(rng_key)
        
        state, info = tempered_smc.step(step_key, state)
        _ = state.tempering_param.block_until_ready()
        
        info_history.append(info)
        step_count += 1
        
        # Record best
        pos, score, mean_score = get_best(state.particles)
        best_positions.append(np.array(pos))
        best_scores.append(float(score))
        
        if verbose:
            print(f"Step {step_count:3d} | λ={float(state.tempering_param):.4f} | "
                  f"Best: {score:.2f} | Mean: {mean_score:.2f}")
    
    dt = time.time() - t0
    
    if verbose:
        print(f"\nSMC completed in {dt:.2f}s ({step_count} temperature steps)")
    
    return (
        state,
        info_history,
        jnp.stack(best_positions),
        jnp.array(best_scores),
    )


# =============================================================================
# Main run function
# =============================================================================

@dataclass
class RunConfig:
    """Configuration for the run."""
    # System
    n_copies_A: int = 8
    n_copies_B: int = 8
    n_copies_C: int = 16
    radius_A: float = 24.0
    radius_B: float = 14.0
    radius_C: float = 16.0
    
    # Scoring
    target_distance: float = 40.0  # Target inter-particle distance
    distance_k: float = 0.1  # Distance restraint strength
    exvol_k: float = 1.0  # Excluded volume strength
    box_size: float = 150.0  # Bounding box
    box_steepness: float = 5.0  # Soft box penalty
    
    # Sampling method
    method: str = 'smc'  # 'smc' or 'rmh'
    
    # SMC parameters
    n_smc_particles: int = 50
    n_mcmc_steps: int = 20
    target_ess: float = 0.5
    rmh_sigma: float = 3.0
    
    # RMH parameters (if method='rmh')
    n_rmh_steps: int = 5000
    
    # Output
    output_dir: str = "output"
    output_prefix: str = "rigid_body_smc"
    save_rmf3: bool = True
    save_interval: int = 1  # Save every N steps (for RMH)
    
    # General
    seed: int = 42
    verbose: bool = True


def run_simulation(config: RunConfig):
    """
    Run the full simulation.
    
    Args:
        config: Run configuration
        
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Rigid Body SMC Sampling with IMP + BlackJAX")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Method: {config.method.upper()}")
    
    # =========================================================================
    # 1. Create system
    # =========================================================================
    print("\n[1] Creating rigid body system...")
    
    rb_configs = [
        RigidBodyConfig("A", config.n_copies_A, config.radius_A, 50000.0),
        RigidBodyConfig("B", config.n_copies_B, config.radius_B, 25000.0),
        RigidBodyConfig("C", config.n_copies_C, config.radius_C, 30000.0),
    ]
    
    model, rigid_bodies, info = create_rigid_body_system(
        rb_configs, box_size=config.box_size * 0.8, seed=config.seed
    )
    
    n_particles = info['n_particles']
    n_dims = n_particles * 3
    radii = info['radii']
    
    print(f"  Created {n_particles} particles ({n_dims} dimensions)")
    
    # =========================================================================
    # 2. Generate distance pairs
    # =========================================================================
    print("\n[2] Generating distance restraints...")
    
    # Inter-type pairs (A-B, A-C, B-C)
    pairs = generate_distance_pairs(rb_configs, pair_type='inter')
    print(f"  Generated {len(pairs)} inter-type distance pairs")
    
    # =========================================================================
    # 3. Create scoring functions
    # =========================================================================
    print("\n[3] Building scoring functions...")
    
    log_prior_fn, log_likelihood_fn, log_prob_fn = create_scoring_functions(
        n_particles=n_particles,
        radii=radii,
        distance_pairs=pairs,
        target_distance=config.target_distance,
        distance_k=config.distance_k,
        exvol_k=config.exvol_k,
        box_size=config.box_size,
        box_steepness=config.box_steepness,
    )
    
    # Test scoring
    initial_coords = get_coordinates_from_model(model, info['particles'])
    initial_flat = initial_coords.flatten()
    
    initial_prior = float(log_prior_fn(initial_flat))
    initial_lik = float(log_likelihood_fn(initial_flat))
    initial_score = float(log_prob_fn(initial_flat))
    
    print(f"  Initial log_prior:      {initial_prior:.2f}")
    print(f"  Initial log_likelihood: {initial_lik:.2f}")
    print(f"  Initial log_posterior:  {initial_score:.2f}")
    
    # =========================================================================
    # 4. Run sampling
    # =========================================================================
    rng_key = jax.random.PRNGKey(config.seed)
    
    if config.method == 'smc':
        print(f"\n[4] Running SMC sampling...")
        print(f"  Particles: {config.n_smc_particles}")
        print(f"  MCMC steps: {config.n_mcmc_steps}")
        print(f"  Target ESS: {config.target_ess:.0%}")
        print(f"  RMH sigma: {config.rmh_sigma}")
        
        # Initialize SMC particles around initial position
        rng_key, init_key = jax.random.split(rng_key)
        initial_positions = (
            initial_flat[None, :] + 
            jax.random.normal(init_key, (config.n_smc_particles, n_dims)) * 5.0
        )
        
        # Run SMC
        final_state, info_history, best_positions, best_scores = run_tempered_smc(
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_prob_fn=log_prob_fn,
            initial_positions=initial_positions,
            rng_key=rng_key,
            n_mcmc_steps=config.n_mcmc_steps,
            rmh_sigma=config.rmh_sigma,
            target_ess=config.target_ess,
            verbose=config.verbose,
        )
        
        # Extract best result
        final_scores = jax.vmap(log_prob_fn)(final_state.particles)
        best_idx = jnp.argmax(final_scores)
        best_position = final_state.particles[best_idx]
        best_score = float(final_scores[best_idx])
        
        results = {
            'method': 'smc',
            'n_temp_steps': len(info_history),
            'best_score': best_score,
            'best_position': np.array(best_position),
            'best_trajectory': np.array(best_positions),
            'score_trajectory': np.array(best_scores),
            'final_mean_score': float(jnp.mean(final_scores)),
        }
        
    elif config.method == 'rmh':
        print(f"\n[4] Running simple RMH sampling...")
        print(f"  Steps: {config.n_rmh_steps}")
        print(f"  Sigma: {config.rmh_sigma}")
        
        samples, scores, acc_rate = run_simple_rmh(
            log_prob_fn=log_prob_fn,
            initial_position=initial_flat,
            rng_key=rng_key,
            n_steps=config.n_rmh_steps,
            sigma=config.rmh_sigma,
            verbose=config.verbose,
        )
        
        best_idx = jnp.argmax(scores)
        
        results = {
            'method': 'rmh',
            'n_steps': config.n_rmh_steps,
            'best_score': float(scores[best_idx]),
            'best_position': np.array(samples[best_idx]),
            'acceptance_rate': float(acc_rate),
            'final_score': float(scores[-1]),
            'score_trajectory': np.array(scores),
            'samples': np.array(samples),  # Keep all samples for RMF3
        }
    
    else:
        raise ValueError(f"Unknown method: {config.method}")
    
    # =========================================================================
    # 5. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Method:          {config.method.upper()}")
    print(f"Initial score:   {initial_score:.2f}")
    print(f"Best score:      {results['best_score']:.2f}")
    print(f"Improvement:     {results['best_score'] - initial_score:.2f}")
    
    if config.method == 'smc':
        print(f"Temp steps:      {results['n_temp_steps']}")
        print(f"Final mean:      {results['final_mean_score']:.2f}")
    else:
        print(f"Acceptance:      {results['acceptance_rate']:.1%}")
    
    # Update IMP model with best coordinates
    best_coords = results['best_position'].reshape(-1, 3)
    set_coordinates_to_model(model, info['particles'], best_coords)
    
    results['model'] = model
    results['particles'] = info['particles']
    results['rigid_bodies'] = rigid_bodies
    
    # =========================================================================
    # 6. Save RMF3 trajectory
    # =========================================================================
    if config.save_rmf3:
        print(f"\n[5] Saving RMF3 trajectory...")
        
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        rmf3_filename = output_dir / f"{config.output_prefix}_{config.method}.rmf3"
        
        # Get particle names for coloring
        particle_names = [p.get_name() for p in info['particles']]
        
        # Color scheme by particle type
        color_by_type = {
            'A': (0.2, 0.6, 1.0),   # Blue
            'B': (0.9, 0.4, 0.2),   # Orange
            'C': (0.3, 0.8, 0.4),   # Green
        }
        
        if config.method == 'smc':
            # For SMC: save best particle at each temperature step
            trajectory = results['best_trajectory'].reshape(-1, n_particles, 3)
            scores = results['score_trajectory']
            
            # Compute lambda values for each frame
            # Frame 0 is initial (lambda=0), last frame is lambda=1
            n_frames = len(scores)
            lambdas = np.linspace(0, 1, n_frames)
            
            save_trajectory_to_rmf3(
                str(rmf3_filename),
                model,
                info['particles'],
                trajectory,
                scores=scores,
                lambdas=lambdas,
                particle_names=particle_names,
                color_by_type=color_by_type,
            )
            
        else:  # RMH
            # For RMH: save samples at intervals
            all_samples = results.get('samples', None)
            all_scores = results['score_trajectory']
            
            if all_samples is None:
                # If we don't have all samples, just save final state
                trajectory = results['best_position'].reshape(1, n_particles, 3)
                scores = np.array([results['best_score']])
            else:
                # Save at intervals
                interval = config.save_interval
                indices = list(range(0, len(all_scores), interval))
                trajectory = all_samples[indices].reshape(-1, n_particles, 3)
                scores = all_scores[indices]
            
            save_trajectory_to_rmf3(
                str(rmf3_filename),
                model,
                info['particles'],
                trajectory,
                scores=scores,
                particle_names=particle_names,
                color_by_type=color_by_type,
            )
        
        results['rmf3_file'] = str(rmf3_filename)
        print(f"  Saved: {rmf3_filename}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SMC/RMH sampling on IMP rigid body system"
    )
    
    # Method
    parser.add_argument('--method', type=str, choices=['smc', 'rmh'],
                       default='smc', help='Sampling method')
    
    # System size
    parser.add_argument('--n_A', type=int, default=8, help='Copies of protein A')
    parser.add_argument('--n_B', type=int, default=8, help='Copies of protein B')
    parser.add_argument('--n_C', type=int, default=16, help='Copies of protein C')
    
    # Scoring
    parser.add_argument('--target_dist', type=float, default=40.0,
                       help='Target inter-particle distance')
    parser.add_argument('--distance_k', type=float, default=0.1,
                       help='Distance restraint strength')
    parser.add_argument('--box_size', type=float, default=150.0,
                       help='Bounding box size')
    
    # SMC parameters
    parser.add_argument('--n_particles', type=int, default=50,
                       help='Number of SMC particles')
    parser.add_argument('--n_mcmc_steps', type=int, default=20,
                       help='MCMC steps per temperature')
    parser.add_argument('--target_ess', type=float, default=0.5,
                       help='Target ESS ratio')
    parser.add_argument('--sigma', type=float, default=3.0,
                       help='RMH proposal sigma')
    
    # RMH parameters
    parser.add_argument('--n_steps', type=int, default=5000,
                       help='RMH steps (if method=rmh)')
    
    # Output
    parser.add_argument('--output_dir', '-o', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--output_prefix', type=str, default='rigid_body_smc',
                       help='Output file prefix')
    parser.add_argument('--no_rmf3', action='store_true',
                       help='Disable RMF3 output')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save interval for RMH trajectory')
    
    # General
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Less output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = RunConfig(
        n_copies_A=args.n_A,
        n_copies_B=args.n_B,
        n_copies_C=args.n_C,
        target_distance=args.target_dist,
        distance_k=args.distance_k,
        box_size=args.box_size,
        method=args.method,
        n_smc_particles=args.n_particles,
        n_mcmc_steps=args.n_mcmc_steps,
        target_ess=args.target_ess,
        rmh_sigma=args.sigma,
        n_rmh_steps=args.n_steps,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        save_rmf3=not args.no_rmf3,
        save_interval=args.save_interval,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    results = run_simulation(config)
    
    print("\n✓ Simulation complete!")
    if 'rmf3_file' in results:
        print(f"  Trajectory: {results['rmf3_file']}")
    
    return results


if __name__ == "__main__":
    main()