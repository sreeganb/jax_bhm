"""
Parameter sweep for SMC simulation with RMH kernel.

Two-phase systematic search:

Phase 1 - SMC hyperparameter grid search:
    Varies (n_particles, n_mcmc_steps, rmh_sigma) with pair_weight fixed at ~0
    to find the combination that maximizes CCC from EM density fitting alone.

Phase 2 - Pair score sweep:
    Fixes SMC hyperparameters to the best from Phase 1, then varies pair_weight
    through [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0] to assess the effect of
    structural restraints on CCC.

All runs are serial (JAX already parallelizes internally on the available device).
Results are saved as individual .h5 files with descriptive names + a summary CSV.

Usage:
    python run_parameter_sweep.py                 # Run both phases
    python run_parameter_sweep.py --phase 1       # Phase 1 only
    python run_parameter_sweep.py --phase 2       # Phase 2 only (uses Phase 1 results)
"""
import sys
import os
import csv
import json
import argparse
import itertools
import io
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_smc_simulation_with_timing import main as run_smc


# =============================================================================
# Tee-ing stdout to both terminal and log file
# =============================================================================

class TeeOutput:
    """Context manager that duplicates stdout to both terminal and a file."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.original_stdout = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.file = open(self.filepath, 'w')
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.file.close()
        return False

    def write(self, data):
        self.original_stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.original_stdout.flush()
        self.file.flush()


# =============================================================================
# Configuration: edit these grids to control the sweep
# =============================================================================

# Phase 1: SMC hyperparameter grid
PARTICLE_COUNTS = [50, 100, 250, 500, 1000]
MCMC_STEPS = [20, 30, 40, 50]
RMH_SIGMAS = [1.0, 2.0, 3.0, 5.0, 10.0]

# Phase 2: pair_weight sweep (with best Phase 1 params)
PAIR_WEIGHTS = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]

# Fixed parameters (not swept)
FIXED_PARAMS = {
    'target_ess': 0.65,
    'sigma_ccc': 0.005,
    'lambda_attract': 0.1,
    'box_size': 300.0,
    'random_seed': 90998210,
}


def make_output_dir(base="output_sweep"):
    """Create timestamped output directory."""
    sweep_dir = Path(base)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    return sweep_dir


def run_single(params, sweep_dir, run_label):
    """
    Run a single SMC simulation with the given parameters.

    Terminal output is also saved to <run_label>.out alongside the .h5 and .rmf3 files.
    Returns the result dict or None if the run failed.
    """
    print("\n" + "#" * 70)
    print(f"  RUN: {run_label}")
    print(f"  Params: np={params['n_particles']}, ms={params['n_mcmc_steps']}, "
          f"sig={params['rmh_sigma']}, pw={params['pair_weight']}")
    print("#" * 70 + "\n")

    filename = f"{run_label}.h5"
    log_file = sweep_dir / f"{run_label}.out"

    try:
        with TeeOutput(str(log_file)):
            result = run_smc(
                output_dir=str(sweep_dir),
                output_filename=filename,
                **params,
            )
        result['run_label'] = run_label
        result['output_file'] = str(sweep_dir / filename)
        result['log_file'] = str(log_file)
        result['status'] = 'success'
        print(f"  Log saved to {log_file}")
        return result
    except Exception as e:
        print(f"\n  FAILED: {e}\n")
        return {
            'run_label': run_label,
            'status': 'failed',
            'error': str(e),
            **params,
            'best_ccc': float('nan'),
            'best_score': float('nan'),
            'wall_time': float('nan'),
        }


def save_results_csv(results, filepath):
    """Append/create a CSV summary of all runs."""
    if not results:
        return

    fieldnames = [
        'run_label', 'status', 'best_ccc', 'best_score', 'wall_time',
        'n_particles', 'n_mcmc_steps', 'rmh_sigma', 'target_ess',
        'pair_weight', 'sigma_ccc', 'lambda_attract',
    ]

    file_exists = Path(filepath).exists()
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults saved to {filepath}")


def print_summary_table(results, title):
    """Print a formatted summary of results sorted by CCC."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    # Sort by CCC descending
    sorted_results = sorted(
        [r for r in results if r['status'] == 'success'],
        key=lambda r: r['best_ccc'],
        reverse=True,
    )

    header = (f"{'Rank':<6} {'Label':<40} {'CCC':>8} {'Score':>12} "
              f"{'Time(s)':>10}")
    print(header)
    print("-" * 90)

    for i, r in enumerate(sorted_results, 1):
        print(f"{i:<6} {r['run_label']:<40} {r['best_ccc']:>8.4f} "
              f"{r['best_score']:>12.2f} {r['wall_time']:>10.1f}")

    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print(f"\n  ({len(failed)} runs failed)")

    print("=" * 90)

    if sorted_results:
        best = sorted_results[0]
        print(f"\n  BEST: {best['run_label']}")
        print(f"    CCC   = {best['best_ccc']:.4f}")
        print(f"    Score = {best['best_score']:.2f}")
        print(f"    np={best['n_particles']}, ms={best['n_mcmc_steps']}, "
              f"sig={best['rmh_sigma']}, pw={best['pair_weight']}")

    return sorted_results


def phase1_grid_search(sweep_dir):
    """
    Phase 1: Grid search over (n_particles, n_mcmc_steps, rmh_sigma).

    pair_weight is fixed at 0.000001 (essentially off) to isolate the effect
    of SMC hyperparameters on CCC.
    """
    print("\n" + "=" * 90)
    print("  PHASE 1: SMC Hyperparameter Grid Search")
    print(f"  Grid: {len(PARTICLE_COUNTS)} x {len(MCMC_STEPS)} x {len(RMH_SIGMAS)} "
          f"= {len(PARTICLE_COUNTS) * len(MCMC_STEPS) * len(RMH_SIGMAS)} runs")
    print("=" * 90)

    results = []
    combos = list(itertools.product(PARTICLE_COUNTS, MCMC_STEPS, RMH_SIGMAS))
    total = len(combos)

    for i, (np_, ms, sig) in enumerate(combos, 1):
        print(f"\n>>> Phase 1: Run {i}/{total}")

        params = {
            'n_particles': np_,
            'n_mcmc_steps': ms,
            'rmh_sigma': sig,
            'pair_weight': 0.000001,
            **FIXED_PARAMS,
        }

        label = f"phase1_np{np_}_ms{ms}_sig{sig}"
        result = run_single(params, sweep_dir, label)
        results.append(result)

        # Save incrementally so we don't lose progress on crashes
        save_results_csv([result], sweep_dir / "phase1_results.csv")

    sorted_results = print_summary_table(results, "PHASE 1 RESULTS")

    # Save best params for Phase 2
    if sorted_results:
        best = sorted_results[0]
        best_params = {
            'n_particles': best['n_particles'],
            'n_mcmc_steps': best['n_mcmc_steps'],
            'rmh_sigma': best['rmh_sigma'],
        }
        with open(sweep_dir / "best_phase1_params.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\nBest Phase 1 params saved to {sweep_dir / 'best_phase1_params.json'}")
        return best_params

    return None


def phase2_pair_weight_sweep(sweep_dir, best_smc_params=None):
    """
    Phase 2: Sweep pair_weight with fixed best SMC hyperparameters from Phase 1.

    If best_smc_params is None, tries to load from best_phase1_params.json.
    """
    if best_smc_params is None:
        params_file = sweep_dir / "best_phase1_params.json"
        if not params_file.exists():
            print(f"ERROR: {params_file} not found. Run Phase 1 first.")
            return
        with open(params_file) as f:
            best_smc_params = json.load(f)

    print("\n" + "=" * 90)
    print("  PHASE 2: Pair Weight Sweep")
    print(f"  Fixed SMC params: np={best_smc_params['n_particles']}, "
          f"ms={best_smc_params['n_mcmc_steps']}, "
          f"sig={best_smc_params['rmh_sigma']}")
    print(f"  pair_weight values: {PAIR_WEIGHTS}")
    print("=" * 90)

    results = []
    total = len(PAIR_WEIGHTS)

    for i, pw in enumerate(PAIR_WEIGHTS, 1):
        print(f"\n>>> Phase 2: Run {i}/{total}")

        params = {
            **best_smc_params,
            'pair_weight': pw,
            **FIXED_PARAMS,
        }

        # Format pair_weight for filename: 0.0 -> "0", 0.001 -> "1e-3"
        if pw == 0.0:
            pw_str = "0"
        elif pw < 0.01:
            pw_str = f"{pw:.0e}".replace("+", "").replace("-0", "-")
        else:
            pw_str = str(pw)

        label = f"phase2_pw{pw_str}"
        result = run_single(params, sweep_dir, label)
        results.append(result)

        save_results_csv([result], sweep_dir / "phase2_results.csv")

    print_summary_table(results, "PHASE 2 RESULTS: pair_weight effect on CCC")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for SMC simulation with RMH kernel."
    )
    parser.add_argument(
        '--phase', type=int, choices=[1, 2], default=None,
        help="Run only Phase 1 or Phase 2. Default: run both sequentially."
    )
    parser.add_argument(
        '--output-dir', type=str, default='output_sweep',
        help="Base output directory for sweep results (default: output_sweep)."
    )
    args = parser.parse_args()

    sweep_dir = make_output_dir(args.output_dir)

    # Copy the density map to sweep dir so the simulation can find it
    src_mrc = Path("output") / "simulated_target_density.mrc"
    dst_mrc = sweep_dir / "simulated_target_density.mrc"
    if src_mrc.exists() and not dst_mrc.exists():
        import shutil
        shutil.copy2(str(src_mrc), str(dst_mrc))
        print(f"Copied density map to {dst_mrc}")

    start_time = datetime.now()
    print(f"\nSweep started at {start_time.isoformat()}")

    if args.phase is None or args.phase == 1:
        best_params = phase1_grid_search(sweep_dir)
    else:
        best_params = None

    if args.phase is None or args.phase == 2:
        phase2_pair_weight_sweep(sweep_dir, best_params)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"  Total sweep time: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"  Results in: {sweep_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
