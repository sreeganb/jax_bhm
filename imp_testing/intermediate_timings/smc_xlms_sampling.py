#!/usr/bin/env python
"""
SMC Sampling with IMP CrossLink Restraint using BlackJAX.

This script:
1. Loads protein structures (Rpt1, Rpt2) via IMP
2. Loads synthetic crosslinks
3. Uses excluded volume as prior
4. Uses IMP CrossLinkMSRestraint as tempered likelihood
5. Runs BlackJAX SMC sampling
6. Saves trajectory to RMF3 format

Usage:
    python smc_xlms_sampling.py \
        --pdb data/pdb/chopped_base_proteasome.pdb \
        --fasta data/fasta/mod_5gjr.fasta \
        --crosslinks synthetic_data/rpt1_rpt2_crosslinks_imp_format.csv \
        --proteins Rpt1 Rpt2 \
        --chains v w \
        --n-particles 50 \
        --n-mcmc-steps 20
"""

import argparse
import time
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from contextlib import contextmanager
from functools import partial

# JAX imports
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# IMP imports
try:
    import IMP
    import IMP.core
    import IMP.atom
    import IMP.algebra
    import IMP.isd
    import IMP.rmf
    import RMF
    import IMP.pmi.topology
    import IMP.pmi.tools
    import IMP.pmi.dof
    import IMP.pmi.io.crosslink
    import IMP.pmi.restraints.crosslinking
    import IMP.pmi.restraints.stereochemistry
    import ihm.cross_linkers
    print(f"IMP version: {IMP.get_module_version()}")
except ImportError as e:
    print(f"IMP not available: {e}")
    sys.exit(1)

import pandas as pd

# BlackJAX imports
try:
    import blackjax
    print(f"BlackJAX available")
except ImportError:
    print("BlackJAX not available. Install with: pip install blackjax")
    sys.exit(1)


# =============================================================================
# Timing utilities
# =============================================================================

def sync_and_time() -> float:
    """Get wall time after ensuring all JAX operations complete."""
    jax.block_until_ready(jnp.zeros(1))
    return time.perf_counter()


class WallTimer:
    """Track timing for multiple sections."""
    
    def __init__(self):
        self.times = {}
        self._starts = {}
        self.total_start = sync_and_time()
    
    def start(self, name: str):
        self._starts[name] = sync_and_time()
    
    def stop(self, name: str) -> float:
        elapsed = sync_and_time() - self._starts[name]
        self.times[name] = self.times.get(name, 0) + elapsed
        del self._starts[name]
        return elapsed
    
    def total(self) -> float:
        return sync_and_time() - self.total_start
    
    def summary(self):
        total = self.total()
        print("\n" + "=" * 60)
        print(f"TIMING SUMMARY (Backend: {jax.default_backend()})")
        print("=" * 60)
        print(f"{'Section':<40} {'Time (s)':>10} {'%':>8}")
        print("-" * 60)
        for name, elapsed in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = 100 * elapsed / total if total > 0 else 0
            print(f"{name:<40} {elapsed:>10.2f} {pct:>7.1f}%")
        print("-" * 60)
        print(f"{'TOTAL WALL TIME':<40} {total:>10.2f} {'100.0':>7}%")
        print("=" * 60)


# =============================================================================
# IMP System Setup
# =============================================================================

class IMPSystemWrapper:
    """
    Wrapper for IMP system that provides scoring interface for JAX sampling.
    
    This class maintains the IMP model and provides methods to:
    1. Get/set particle coordinates
    2. Evaluate crosslink restraint score
    3. Evaluate excluded volume score
    """
    
    def __init__(
        self,
        pdb_file: str,
        fasta_file: str,
        crosslink_file: str,
        proteins: List[str],
        chain_ids: List[str],
        xl_length: float = 21.0,
        xl_slope: float = 0.01,
    ):
        self.pdb_file = pdb_file
        self.fasta_file = fasta_file
        self.crosslink_file = crosslink_file
        self.proteins = proteins
        self.chain_ids = chain_ids
        
        # Create IMP system
        self._setup_system()
        self._setup_crosslink_restraint(xl_length, xl_slope)
        self._setup_excluded_volume()
        self._cache_particle_info()
        
    def _setup_system(self):
        """Create IMP model and load structures."""
        print("\n  Setting up IMP system...")
        
        self.mdl = IMP.Model()
        self.sys = IMP.pmi.topology.System(self.mdl, name='SMC_XLMS')
        self.state = self.sys.create_state()
        
        # Read sequences
        print(f"    Reading sequences from: {self.fasta_file}")
        sequences = IMP.pmi.topology.Sequences(self.fasta_file)
        
        self.molecules = []
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
        
        for i, (protein, chain_id) in enumerate(zip(self.proteins, self.chain_ids)):
            print(f"    Creating molecule: {protein} (chain {chain_id})")
            
            seq = sequences[protein]
            print(f"      Sequence length: {len(seq)}")
            
            molecule = self.state.create_molecule(
                protein,
                sequence=seq,
                chain_id=chain_id
            )
            
            structure = molecule.add_structure(self.pdb_file, chain_id=chain_id)
            molecule.add_representation(
                structure,
                resolutions=[1],
                color=colors[i % len(colors)]
            )
            
            self.molecules.append(molecule)
        
        # Build system
        self.root_hier = self.sys.build()
        
        # Setup degrees of freedom
        self.dof = IMP.pmi.dof.DegreesOfFreedom(self.mdl)
        for mol in self.molecules:
            self.dof.create_rigid_body(mol)
        
        print(f"    Built system with {len(self.molecules)} molecules")
        
    def _setup_crosslink_restraint(self, length: float, slope: float):
        """Setup crosslink restraint."""
        print(f"\n  Setting up crosslink restraint...")
        
        df = pd.read_csv(self.crosslink_file)
        print(f"    Loaded {len(df)} crosslinks")
        
        xldbkc = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
        xldbkc.set_standard_keys()
        
        xldb = IMP.pmi.io.crosslink.CrossLinkDataBase()
        xldb.create_set_from_file(file_name=self.crosslink_file, converter=xldbkc)
        
        self.xlr = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(
            root_hier=self.root_hier,
            database=xldb,
            length=length,
            resolution=1.0,
            slope=slope,
            weight=1.0,
            linker=ihm.cross_linkers.dss,
        )
        self.xlr.add_to_model()
        
        self.xl_sf = IMP.core.RestraintsScoringFunction([self.xlr.get_restraint()])
        
        print(f"    Crosslink restraint ready")
        
    def _setup_excluded_volume(self):
        """Setup excluded volume restraint."""
        print(f"\n  Setting up excluded volume restraint...")
        
        self.evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
            included_objects=self.molecules,
            resolution=1,
        )
        self.evr.add_to_model()
        
        self.ev_sf = IMP.core.RestraintsScoringFunction([self.evr.get_restraint()])
        
        print(f"    Excluded volume restraint ready")
        
    def _cache_particle_info(self):
        """Cache particle information for fast coordinate access."""
        # Get all XYZ particles (rigid body members)
        self.xyz_particles = []
        self.rigid_bodies = []
        
        for rb in self.dof.get_rigid_bodies():
            self.rigid_bodies.append(rb)
        
        # For coordinate manipulation, we work with rigid body reference frames
        self.n_rigid_bodies = len(self.rigid_bodies)
        # Each rigid body has 6 DOF: 3 translation + 3 rotation (quaternion -> 4, but 3 independent)
        # For simplicity, we'll use 7 parameters: 3 translation + 4 quaternion
        self.n_params_per_rb = 7
        self.n_dims = self.n_rigid_bodies * self.n_params_per_rb
        
        print(f"    Cached {self.n_rigid_bodies} rigid bodies")
        print(f"    Total parameters: {self.n_dims} (7 per rigid body)")
        
    def get_flat_coords(self) -> np.ndarray:
        """Get current rigid body parameters as flat array."""
        params = []
        for rb in self.rigid_bodies:
            rf = rb.get_reference_frame()
            trans = rf.get_transformation_to()
            
            # Translation
            t = trans.get_translation()
            params.extend([t[0], t[1], t[2]])
            
            # Rotation (quaternion)
            rot = trans.get_rotation()
            q = rot.get_quaternion()
            params.extend([q[0], q[1], q[2], q[3]])
            
        return np.array(params, dtype=np.float64)
    
    def set_flat_coords(self, params: np.ndarray):
        """Set rigid body parameters from flat array."""
        params = np.asarray(params, dtype=np.float64)
        
        for i, rb in enumerate(self.rigid_bodies):
            offset = i * self.n_params_per_rb
            
            # Translation
            t = IMP.algebra.Vector3D(
                params[offset], params[offset + 1], params[offset + 2]
            )
            
            # Rotation (quaternion) - normalize to ensure valid rotation
            q = params[offset + 3:offset + 7]
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-10:
                q = q / q_norm
            else:
                q = [1, 0, 0, 0]  # Identity rotation
            
            rot = IMP.algebra.Rotation3D(q[0], q[1], q[2], q[3])
            trans = IMP.algebra.Transformation3D(rot, t)
            rf = IMP.algebra.ReferenceFrame3D(trans)
            
            rb.set_reference_frame(rf)
    
    def evaluate_crosslink_score(self) -> float:
        """Evaluate crosslink restraint score."""
        return self.xl_sf.evaluate(False)
    
    def evaluate_excluded_volume_score(self) -> float:
        """Evaluate excluded volume score."""
        return self.ev_sf.evaluate(False)
    
    def evaluate_total_score(self) -> float:
        """Evaluate total score (XL + EV)."""
        return self.evaluate_crosslink_score() + self.evaluate_excluded_volume_score()
    
    def shuffle(self, max_translation: float = 100.0):
        """Shuffle configuration randomly."""
        sel = IMP.atom.Selection(self.root_hier).get_selected_particles()
        IMP.pmi.tools.shuffle_configuration(
            sel,
            max_translation=max_translation,
            bounding_box=((-300, -300, -300), (300, 300, 300)),
            avoidcollision_rb=False,
        )
    
    def save_to_rmf3(self, filename: str, positions_list: List[np.ndarray], scores: List[float]):
        """Save trajectory to RMF3 format."""
        print(f"\n  Saving trajectory to: {filename}")
        
        rmf_file = RMF.create_rmf_file(filename)
        IMP.rmf.add_hierarchy(rmf_file, self.root_hier)
        IMP.rmf.add_restraints(rmf_file, [self.xlr.get_restraint()])
        
        for i, (params, score) in enumerate(zip(positions_list, scores)):
            self.set_flat_coords(params)
            IMP.rmf.save_frame(rmf_file, str(i))
            
            if (i + 1) % 50 == 0:
                print(f"    Saved frame {i + 1}/{len(positions_list)}")
        
        print(f"    Saved {len(positions_list)} frames")


# =============================================================================
# SMC Sampling with IMP Scoring
# =============================================================================

def create_imp_log_prob_fns(imp_system: IMPSystemWrapper):
    """
    Create log probability functions that use IMP for scoring.
    
    Since IMP scoring is not JAX-compatible, we use a callback approach.
    """
    
    def log_prior_fn(params: jnp.ndarray) -> float:
        """
        Log prior: negative excluded volume score.
        
        We use excluded volume as a soft prior to prevent clashes.
        """
        # Convert to numpy and set coordinates
        params_np = np.asarray(params)
        imp_system.set_flat_coords(params_np)
        
        # Evaluate excluded volume (IMP returns positive scores for violations)
        ev_score = imp_system.evaluate_excluded_volume_score()
        
        # Convert to log probability (negative score = higher probability)
        return -ev_score
    
    def log_likelihood_fn(params: jnp.ndarray) -> float:
        """
        Log likelihood: negative crosslink score.
        
        This is what gets tempered in SMC.
        """
        params_np = np.asarray(params)
        imp_system.set_flat_coords(params_np)
        
        xl_score = imp_system.evaluate_crosslink_score()
        
        return -xl_score
    
    def log_prob_fn(params: jnp.ndarray) -> float:
        """Combined log probability."""
        params_np = np.asarray(params)
        imp_system.set_flat_coords(params_np)
        
        total_score = imp_system.evaluate_total_score()
        
        return -total_score
    
    return log_prior_fn, log_likelihood_fn, log_prob_fn


def run_smc_with_imp(
    imp_system: IMPSystemWrapper,
    n_particles: int = 50,
    n_mcmc_steps: int = 20,
    rmh_sigma: float = 5.0,
    target_ess: float = 0.5,
    max_steps: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run SMC sampling using BlackJAX with IMP scoring.
    
    Since IMP scoring is not JAX-differentiable, we use a custom loop
    that evaluates IMP scores outside of JAX's JIT compilation.
    """
    
    rng_key = jax.random.PRNGKey(seed)
    n_dims = imp_system.n_dims
    
    print(f"\n  SMC Configuration:")
    print(f"    Particles: {n_particles}")
    print(f"    Dimensions: {n_dims}")
    print(f"    MCMC steps per temperature: {n_mcmc_steps}")
    print(f"    RMH sigma: {rmh_sigma}")
    print(f"    Target ESS: {target_ess}")
    
    # Initialize particles from shuffled configurations
    print(f"\n  Initializing {n_particles} particles...")
    initial_positions = []
    initial_scores = []
    
    for i in range(n_particles):
        imp_system.shuffle(max_translation=150.0)
        params = imp_system.get_flat_coords()
        initial_positions.append(params)
        
        score = imp_system.evaluate_total_score()
        initial_scores.append(-score)  # Log probability
        
    initial_positions = np.array(initial_positions)
    initial_scores = np.array(initial_scores)
    
    print(f"    Initial scores: mean={np.mean(initial_scores):.2f}, "
          f"min={np.min(initial_scores):.2f}, max={np.max(initial_scores):.2f}")
    
    # SMC state
    positions = initial_positions.copy()
    log_weights = np.zeros(n_particles)
    lmbda = 0.0  # Temperature parameter
    
    # Track best samples
    best_positions = [positions[np.argmax(initial_scores)].copy()]
    best_scores = [np.max(initial_scores)]
    
    step = 0
    
    while lmbda < 1.0 and step < max_steps:
        step += 1
        
        # Compute current likelihoods
        log_likelihoods = np.zeros(n_particles)
        log_priors = np.zeros(n_particles)
        
        for i in range(n_particles):
            imp_system.set_flat_coords(positions[i])
            log_likelihoods[i] = -imp_system.evaluate_crosslink_score()
            log_priors[i] = -imp_system.evaluate_excluded_volume_score()
        
        # Find next temperature using bisection
        def compute_ess(delta_lmbda):
            new_weights = log_weights + delta_lmbda * log_likelihoods
            new_weights = new_weights - np.max(new_weights)  # Stabilize
            weights = np.exp(new_weights)
            weights = weights / np.sum(weights)
            return 1.0 / np.sum(weights ** 2) / n_particles
        
        # Binary search for delta_lmbda
        delta_low, delta_high = 0.0, 1.0 - lmbda
        
        if compute_ess(delta_high) >= target_ess:
            delta_lmbda = delta_high
        else:
            for _ in range(20):  # Binary search iterations
                delta_mid = (delta_low + delta_high) / 2
                if compute_ess(delta_mid) > target_ess:
                    delta_low = delta_mid
                else:
                    delta_high = delta_mid
            delta_lmbda = delta_low
        
        # Update temperature and weights
        lmbda_new = min(lmbda + delta_lmbda, 1.0)
        log_weights = log_weights + (lmbda_new - lmbda) * log_likelihoods
        lmbda = lmbda_new
        
        # Normalize weights
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        ess = 1.0 / np.sum(weights ** 2)
        
        # Resample if ESS is low
        if ess < n_particles * target_ess * 0.5:
            rng_key, resample_key = jax.random.split(rng_key)
            indices = np.random.choice(n_particles, size=n_particles, p=weights)
            positions = positions[indices].copy()
            log_weights = np.zeros(n_particles)
        
        # MCMC moves (Random Walk Metropolis)
        rng_key, mcmc_key = jax.random.split(rng_key)
        n_accepted = 0
        
        for mcmc_step in range(n_mcmc_steps):
            for i in range(n_particles):
                # Propose new position
                rng_key, prop_key = jax.random.split(rng_key)
                proposal = positions[i] + np.array(jax.random.normal(prop_key, (n_dims,))) * rmh_sigma
                
                # Normalize quaternions in proposal
                for rb_idx in range(imp_system.n_rigid_bodies):
                    offset = rb_idx * 7 + 3
                    q = proposal[offset:offset + 4]
                    q_norm = np.linalg.norm(q)
                    if q_norm > 1e-10:
                        proposal[offset:offset + 4] = q / q_norm
                
                # Compute acceptance probability
                imp_system.set_flat_coords(proposal)
                prop_log_prior = -imp_system.evaluate_excluded_volume_score()
                prop_log_lik = -imp_system.evaluate_crosslink_score()
                prop_log_prob = prop_log_prior + lmbda * prop_log_lik
                
                imp_system.set_flat_coords(positions[i])
                curr_log_prior = -imp_system.evaluate_excluded_volume_score()
                curr_log_lik = -imp_system.evaluate_crosslink_score()
                curr_log_prob = curr_log_prior + lmbda * curr_log_lik
                
                log_alpha = prop_log_prob - curr_log_prob
                
                # Accept/reject
                rng_key, accept_key = jax.random.split(rng_key)
                if np.log(jax.random.uniform(accept_key)) < log_alpha:
                    positions[i] = proposal.copy()
                    n_accepted += 1
        
        acceptance_rate = n_accepted / (n_mcmc_steps * n_particles)
        
        # Track best sample
        current_scores = []
        for i in range(n_particles):
            imp_system.set_flat_coords(positions[i])
            current_scores.append(-imp_system.evaluate_total_score())
        current_scores = np.array(current_scores)
        
        best_idx = np.argmax(current_scores)
        best_positions.append(positions[best_idx].copy())
        best_scores.append(current_scores[best_idx])
        
        if verbose:
            print(f"  Step {step:3d} | λ={lmbda:.4f} | ESS={ess:.1f} | "
                  f"Accept={acceptance_rate:.1%} | Best={best_scores[-1]:.2f} | "
                  f"Mean={np.mean(current_scores):.2f}")
        
        if lmbda >= 1.0:
            break
    
    print(f"\n  SMC completed in {step} steps")
    print(f"  Final λ = {lmbda:.4f}")
    print(f"  Best score: {max(best_scores):.2f}")
    
    return best_positions, best_scores


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SMC Sampling with IMP CrossLink Restraint")
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--crosslinks", type=str, required=True, help="Crosslink CSV file")
    parser.add_argument("--proteins", type=str, nargs="+", default=["Rpt1", "Rpt2"])
    parser.add_argument("--chains", type=str, nargs="+", default=["v", "w"])
    parser.add_argument("--n-particles", type=int, default=50, help="Number of SMC particles")
    parser.add_argument("--n-mcmc-steps", type=int, default=20, help="MCMC steps per temperature")
    parser.add_argument("--rmh-sigma", type=float, default=5.0, help="RMH proposal sigma")
    parser.add_argument("--output", type=str, default="smc_xlms_trajectory.rmf3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    timer = WallTimer()
    
    print("=" * 60)
    print("SMC Sampling with IMP CrossLink Restraint")
    print("=" * 60)
    print(f"  PDB: {args.pdb}")
    print(f"  FASTA: {args.fasta}")
    print(f"  Crosslinks: {args.crosslinks}")
    print(f"  Proteins: {args.proteins}")
    print(f"  Chains: {args.chains}")
    
    # Setup IMP system
    timer.start("1. IMP System Setup")
    
    imp_system = IMPSystemWrapper(
        pdb_file=args.pdb,
        fasta_file=args.fasta,
        crosslink_file=args.crosslinks,
        proteins=args.proteins,
        chain_ids=args.chains,
    )
    
    timer.stop("1. IMP System Setup")
    
    # Initial score
    print(f"\n  Initial XL score: {imp_system.evaluate_crosslink_score():.2f}")
    print(f"  Initial EV score: {imp_system.evaluate_excluded_volume_score():.2f}")
    
    # Run SMC
    timer.start("2. SMC Sampling")
    
    best_positions, best_scores = run_smc_with_imp(
        imp_system=imp_system,
        n_particles=args.n_particles,
        n_mcmc_steps=args.n_mcmc_steps,
        rmh_sigma=args.rmh_sigma,
        seed=args.seed,
        verbose=True,
    )
    
    timer.stop("2. SMC Sampling")
    
    # Save results
    timer.start("3. Save Results")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    imp_system.save_to_rmf3(str(output_path), best_positions, best_scores)
    
    # Also save scores to CSV
    scores_file = str(output_path).replace('.rmf3', '_scores.csv')
    pd.DataFrame({
        'step': range(len(best_scores)),
        'score': best_scores,
    }).to_csv(scores_file, index=False)
    print(f"  Saved scores to: {scores_file}")
    
    timer.stop("3. Save Results")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total frames: {len(best_positions)}")
    print(f"  Initial score: {best_scores[0]:.2f}")
    print(f"  Final score: {best_scores[-1]:.2f}")
    print(f"  Best score: {max(best_scores):.2f}")
    print(f"  Output: {args.output}")
    
    timer.summary()


if __name__ == "__main__":
    main()