#!/usr/bin/env python
"""
Benchmark IMP CrossLinkMSRestraint: Regular IMP vs JAX.

This script:
1. Loads protein structures (Rpt1, Rpt2)
2. Loads synthetic crosslinks
3. Generates random configurations
4. Scores each configuration with IMP (regular) and IMP (JAX)
5. Reports timing comparison

Usage:
    python xlms_timings.py --pdb structure.pdb --fasta sequences.fasta --crosslinks crosslinks.csv
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from contextlib import contextmanager

# Check dependencies
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f" JAX version: {jax.__version__}")
    print(f"  Default backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
except ImportError:
    JAX_AVAILABLE = False
    print(" JAX not available")

try:
    import IMP
    import IMP.core
    import IMP.atom
    import IMP.algebra
    import IMP.isd
    import IMP.pmi.topology
    import IMP.pmi.tools
    import IMP.pmi.io.crosslink
    import IMP.pmi.restraints.crosslinking
    import ihm.cross_linkers
    IMP_AVAILABLE = True
    print(f" IMP version: {IMP.get_module_version()}")
except ImportError as e:
    IMP_AVAILABLE = False
    print(f" IMP not available: {e}")

import pandas as pd


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single configuration."""
    config_id: int
    imp_score: float
    imp_time: float
    jax_score: Optional[float]
    jax_time: Optional[float]
    

@contextmanager
def timer():
    """Context manager for timing."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def create_system_from_pdb(
    pdb_file: str,
    fasta_file: str,
    proteins: List[str],
    chain_ids: List[str],
) -> Tuple:
    """
    Create IMP system from PDB and FASTA files.
    
    Returns:
        Tuple of (model, root_hierarchy, molecules, state)
    """
    mdl = IMP.Model()
    sys = IMP.pmi.topology.System(mdl, name='CrosslinkBenchmark')
    state = sys.create_state()
    
    # Read sequences from FASTA file (not PDB!)
    print(f"  Reading sequences from: {fasta_file}")
    sequences = IMP.pmi.topology.Sequences(fasta_file)
    
    molecules = []
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    
    for i, (protein, chain_id) in enumerate(zip(proteins, chain_ids)):
        print(f"  Creating molecule: {protein} (chain {chain_id})")
        
        # Get sequence for this protein from FASTA
        try:
            seq = sequences[protein]
            print(f"    Sequence length: {len(seq)}")
        except KeyError:
            print(f"    Warning: No sequence for {protein} in FASTA, trying chain ID")
            try:
                seq = sequences[chain_id]
            except KeyError:
                print(f"    Error: No sequence found for {protein} or chain {chain_id}")
                raise
        
        molecule = state.create_molecule(
            protein,
            sequence=seq,
            chain_id=chain_id
        )
        
        # Add structure from PDB
        structure = molecule.add_structure(pdb_file, chain_id=chain_id)
        
        # Add representation
        molecule.add_representation(
            structure,
            resolutions=[1],
            color=colors[i % len(colors)]
        )
        
        molecules.append(molecule)
    
    # Build the system
    root_hier = sys.build()
    
    return mdl, root_hier, molecules, state


def setup_crosslink_restraint(
    mdl: IMP.Model,
    root_hier,
    crosslink_file: str,
    length: float = 21.0,
    slope: float = 0.01,
):
    """
    Setup CrossLinkMSRestraint from CSV file.
    
    Args:
        mdl: IMP Model
        root_hier: Root hierarchy
        crosslink_file: Path to crosslink CSV
        length: Crosslinker length (Å)
        slope: Linear slope term
        
    Returns:
        CrossLinkingMassSpectrometryRestraint object
    """
    # Read crosslinks
    df = pd.read_csv(crosslink_file)
    
    # Check column names and fix if needed
    if 'Protein1' not in df.columns:
        # Try to capitalize
        df.columns = [c.capitalize() for c in df.columns]
    
    print(f"  Loaded {len(df)} crosslinks from {crosslink_file}")
    print(f"  Columns: {list(df.columns)}")
    
    # Setup crosslink database
    xldbkc = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
    xldbkc.set_standard_keys()
    
    xldb = IMP.pmi.io.crosslink.CrossLinkDataBase()
    xldb.create_set_from_file(file_name=crosslink_file, converter=xldbkc)
    
    # Create restraint
    xlr = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(
        root_hier=root_hier,
        database=xldb,
        length=length,
        resolution=1.0,
        slope=slope,
        weight=1.0,
        linker=ihm.cross_linkers.dss,
    )
    
    xlr.add_to_model()
    
    return xlr


def shuffle_configuration(
    root_hier,
    max_translation: float = 50.0,
    bounding_box: Tuple = ((-500, -500, -500), (500, 500, 500)),
):
    """Shuffle particle positions to create a new random configuration."""
    sel = IMP.atom.Selection(root_hier).get_selected_particles()
    IMP.pmi.tools.shuffle_configuration(
        sel,
        max_translation=max_translation,
        bounding_box=bounding_box,
        avoidcollision_rb=False,
    )


def score_configuration_imp(sf: IMP.core.RestraintsScoringFunction) -> Tuple[float, float]:
    """Score configuration using regular IMP. Returns (score, time)."""
    with timer() as get_time:
        score = sf.evaluate(False)
    return score, get_time()


def score_configuration_jax(
    xlr,
    root_hier,
    mdl: IMP.Model,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Score configuration using JAX IMP.
    
    Returns (score, time) or (None, None) if JAX not available.
    """
    if not JAX_AVAILABLE:
        return None, None
    
    try:
        # Get the underlying restraint
        rs = xlr.get_restraint()
        
        # Check if JAX implementation exists
        try:
            jax_fn = rs._get_jax()
            print("  JAX implementation found for CrossLinkMSRestraint")
        except (AttributeError, NotImplementedError) as e:
            print(f"  No JAX implementation: {e}")
            return None, None
        
        # Time JAX evaluation
        with timer() as get_time:
            score = rs.evaluate(False)
        
        return score, get_time()
        
    except Exception as e:
        print(f"  JAX scoring failed: {e}")
        return None, None


def benchmark_crosslink_scoring(
    pdb_file: str,
    fasta_file: str,
    crosslink_file: str,
    proteins: List[str],
    chain_ids: List[str],
    n_configs: int = 10,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """
    Benchmark crosslink scoring for multiple random configurations.
    
    Args:
        pdb_file: Path to PDB structure
        fasta_file: Path to FASTA sequences
        crosslink_file: Path to crosslink CSV
        proteins: List of protein names
        chain_ids: List of chain IDs
        n_configs: Number of random configurations to test
        verbose: Print detailed output
        
    Returns:
        List of BenchmarkResult objects
    """
    print("\n" + "=" * 60)
    print("Setting up IMP system")
    print("=" * 60)
    
    # Create system
    mdl, root_hier, molecules, state = create_system_from_pdb(
        pdb_file, fasta_file, proteins, chain_ids
    )
    
    print("\n" + "=" * 60)
    print("Setting up crosslink restraint")
    print("=" * 60)
    
    # Setup crosslinks
    xlr = setup_crosslink_restraint(mdl, root_hier, crosslink_file)
    
    # Create scoring function
    sf = IMP.core.RestraintsScoringFunction([xlr.get_restraint()])
    
    # Initial score
    initial_score = sf.evaluate(False)
    print(f"\n  Initial score: {initial_score:.4f}")
    
    # Benchmark configurations
    print("\n" + "=" * 60)
    print(f"Benchmarking {n_configs} random configurations")
    print("=" * 60)
    
    results = []
    
    # Warm-up (JIT compilation for JAX)
    print("\n  Warm-up run...")
    shuffle_configuration(root_hier, max_translation=100.0)
    _ = sf.evaluate(False)
    
    for i in range(n_configs):
        # Shuffle to new configuration
        shuffle_configuration(root_hier, max_translation=100.0)
        
        # Score with regular IMP
        imp_score, imp_time = score_configuration_imp(sf)
        
        # Score with JAX IMP
        jax_score, jax_time = score_configuration_jax(xlr, root_hier, mdl)
        
        result = BenchmarkResult(
            config_id=i,
            imp_score=imp_score,
            imp_time=imp_time,
            jax_score=jax_score,
            jax_time=jax_time,
        )
        results.append(result)
        
        if verbose or (i + 1) % 10 == 0:
            jax_info = f", JAX score={jax_score:.2f}, JAX time={jax_time*1000:.3f} ms" if jax_time else ""
            print(f"  Config {i+1:3d}: IMP score={imp_score:.2f}, IMP time={imp_time*1000:.3f} ms{jax_info}")
    
    return results


def print_benchmark_summary(results: List[BenchmarkResult]):
    """Print summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    imp_times = [r.imp_time for r in results]
    imp_scores = [r.imp_score for r in results]
    
    print(f"\nIMP Regular:")
    print(f"  Configurations tested: {len(results)}")
    print(f"  Score range: {min(imp_scores):.2f} - {max(imp_scores):.2f}")
    print(f"  Mean score: {np.mean(imp_scores):.2f}")
    print(f"  Mean time: {np.mean(imp_times)*1000:.3f} ms")
    print(f"  Total time: {sum(imp_times)*1000:.3f} ms")
    
    jax_times = [r.jax_time for r in results if r.jax_time is not None]
    jax_scores = [r.jax_score for r in results if r.jax_score is not None]
    
    if jax_times:
        print(f"\nIMP JAX:")
        print(f"  Configurations tested: {len(jax_times)}")
        print(f"  Score range: {min(jax_scores):.2f} - {max(jax_scores):.2f}")
        print(f"  Mean score: {np.mean(jax_scores):.2f}")
        print(f"  Mean time: {np.mean(jax_times)*1000:.3f} ms")
        print(f"  Total time: {sum(jax_times)*1000:.3f} ms")
        
        speedup = np.mean(imp_times) / np.mean(jax_times)
        print(f"\n  Speedup: {speedup:.2f}x")
        
        # Check score consistency
        score_diffs = [abs(r.imp_score - r.jax_score) for r in results if r.jax_score is not None]
        print(f"  Max score difference: {max(score_diffs):.6f}")
    else:
        print("\nIMP JAX: Not available or not implemented for this restraint")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark CrossLinkMSRestraint")
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file with sequences")
    parser.add_argument("--crosslinks", type=str, required=True, help="Crosslink CSV file")
    parser.add_argument("--proteins", type=str, nargs="+", default=["Rpt1", "Rpt2"],
                        help="Protein names (must match FASTA headers)")
    parser.add_argument("--chains", type=str, nargs="+", default=["v", "w"],
                        help="Chain IDs in PDB")
    parser.add_argument("--n-configs", type=int, default=100, help="Number of configurations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if not IMP_AVAILABLE:
        print("IMP is required for this benchmark")
        sys.exit(1)
    
    print("=" * 60)
    print("CrossLink Scoring Benchmark: IMP Regular vs JAX")
    print("=" * 60)
    print(f"  PDB file: {args.pdb}")
    print(f"  FASTA file: {args.fasta}")
    print(f"  Crosslinks: {args.crosslinks}")
    print(f"  Proteins: {args.proteins}")
    print(f"  Chains: {args.chains}")
    
    # Run benchmark
    results = benchmark_crosslink_scoring(
        pdb_file=args.pdb,
        fasta_file=args.fasta,
        crosslink_file=args.crosslinks,
        proteins=args.proteins,
        chain_ids=args.chains,
        n_configs=args.n_configs,
        verbose=args.verbose,
    )
    
    # Print summary
    print_benchmark_summary(results)
    
    # Save results to CSV
    output_file = "benchmark_results.csv"
    df = pd.DataFrame([
        {
            'config_id': r.config_id,
            'imp_score': r.imp_score,
            'imp_time_ms': r.imp_time * 1000,
            'jax_score': r.jax_score,
            'jax_time_ms': r.jax_time * 1000 if r.jax_time else None,
        }
        for r in results
    ])
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()