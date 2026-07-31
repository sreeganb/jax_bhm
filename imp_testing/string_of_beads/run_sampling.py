#!/usr/bin/env python
"""
User-facing driver.

The whole point of this file is that it reads top to bottom as three
declarations and one call:

    1. representation  -- what the system is
    2. restraints      -- what constrains it
    3. sampler         -- how it is explored

Everything mechanical lives in the impbuild package. Edit the CONFIG block
below, or override from the command line:

    python run_sampling.py                                # default samplers
    python run_sampling.py --copies 20 --samplers rmh
    python run_sampling.py --samplers replica_exchange rmh --frames 200
    python run_sampling.py --dof-mode flex --rmh-steps 50000

Run from the directory that contains data/topology.txt.
"""

import argparse
from pathlib import Path

# XLA flags must be set before jax is imported anywhere, so this comes first.
from impbuild.runtime import set_xla_flags

# Some GPU/XLA builds emit noisy GEMM autotuner mismatches for this workload;
# level 0 favours stable kernels over autotuned ones.
set_xla_flags("--xla_gpu_autotune_level=0")

from impbuild import (                                     # noqa: E402
    DistanceRestraintSpec,
    RestraintBuilder,
    SystemSpec,
    build_system,
    molecule_specs_from_topology,
    run_sampler,
)
from impbuild.runtime import (                             # noqa: E402
    print_environment,
    runtime_environment,
    write_timing_report,
)
from impbuild.samplers import ShuffleConfig                # noqa: E402


# ==========================================================================
# CONFIG
# ==========================================================================

DATA_DIR = Path("data")
TOPOLOGY_FILE = DATA_DIR / "topology.txt"
PDB_DIR = DATA_DIR / "pdb"
FASTA_DIR = DATA_DIR / "fasta"

# How many copies of the whole molecule set to build.
N_COPIES = 10

# Harmonic distance restraints, declared once and expanded over copies.
#
# The first one is the test case asked for: residue 1 of KCOIL to residue 52 of
# ECOIL, applied inside every copy. The second ties consecutive copies together
# so a multi-copy system does not simply drift apart; drop it if you want the
# copies independent.
DISTANCE_RESTRAINTS = [
    DistanceRestraintSpec(
        molecule1="KCOIL", residue1=1,
        molecule2="ECOIL", residue2=52,
        distance=10.0, kappa=1.0,
        scope="intra_copy",
        label="KCOIL1_ECOIL52",
    ),
    DistanceRestraintSpec(
        molecule1="KCOIL", residue1=52,
        molecule2="KCOIL", residue2=1,
        distance=15.0, kappa=1.0,
        scope="inter_copy_chain",
        label="copy_linker",
    ),
]

# JAX sampling degrees of freedom: 'flex' | 'rigid' | 'all'
DOF_MODE = "all"
BOX_HALF_WIDTH = 300.0


# ==========================================================================
# pipeline
# ==========================================================================

def build(n_copies, verbose=True):
    """Step 1 and 2: representation, then restraints."""
    if not TOPOLOGY_FILE.exists():
        raise FileNotFoundError(
            f"{TOPOLOGY_FILE} not found. Run this script from the directory "
            "that contains data/topology.txt."
        )

    print("--- representation ---")
    molecules = molecule_specs_from_topology(
        str(TOPOLOGY_FILE), pdb_dir=str(PDB_DIR), fasta_dir=str(FASTA_DIR),
        resolutions=(1,), verbose=verbose,
    )
    built = build_system(
        SystemSpec(molecules=molecules, n_copies=n_copies, name="ToyModel"),
        verbose=verbose,
    )

    print("--- restraints ---")
    bundle = (
        RestraintBuilder(built, verbose=verbose)
        .connectivity()                       # keeps each chain intact
        .excluded_volume(resolution=1)        # keeps beads from overlapping
        .distances(DISTANCE_RESTRAINTS)       # the harmonic test restraints
        .finalize()
    )
    return built, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copies", type=int, default=N_COPIES,
                        help="number of copies of the molecule set")
    parser.add_argument("--samplers", nargs="+",
                        default=["rmh", "replica_exchange"],
                        help="samplers to run, in order: "
                             "replica_exchange rmh smc adaptive_smc")
    parser.add_argument("--dof-mode", default=DOF_MODE,
                        choices=["flex", "rigid", "all"])
    parser.add_argument("--rmh-steps", type=int, default=500000)
    parser.add_argument("--frames", type=int, default=1000,
                        help="ReplicaExchange frames")
    parser.add_argument("--mc-steps", type=int, default=10,
                        help="ReplicaExchange MC steps per frame")
    parser.add_argument("--smc-particles", type=int, default=100)
    parser.add_argument("--no-shuffle", action="store_true",
                        help="skip the pre-sampling randomisation")
    parser.add_argument("--report", default="sampling_report.txt")
    args = parser.parse_args()

    env = print_environment(runtime_environment())

    built, bundle = build(args.copies)

    shuffle = ShuffleConfig(enabled=not args.no_shuffle)

    # Per-sampler keyword arguments. Adding a sampler means adding an entry
    # here; the call below does not change.
    sampler_kwargs = {
        "replica_exchange": dict(
            number_of_frames=args.frames,
            monte_carlo_steps=args.mc_steps,
        ),
        "rmh": dict(
            n_steps=args.rmh_steps,
            dof_mode=args.dof_mode,
            box_half_width=BOX_HALF_WIDTH,
        ),
        "smc": dict(
            n_particles=args.smc_particles,
            dof_mode=args.dof_mode,
            box_half_width=BOX_HALF_WIDTH,
        ),
        "adaptive_smc": dict(
            n_particles=args.smc_particles,
            dof_mode=args.dof_mode,
            box_half_width=BOX_HALF_WIDTH,
        ),
    }

    print("--- sampling ---")

    # JAX samplers must run before any IMP-native sampler in the same process:
    # ReplicaExchange.execute_macro() allocates particles on the shared model,
    # which breaks the adapter's one-row-per-particle assumption. Reorder rather
    # than fail, and say so, since the order affects nothing else.
    IMP_NATIVE = {"replica_exchange"}
    ordered = ([n for n in args.samplers if n not in IMP_NATIVE]
               + [n for n in args.samplers if n in IMP_NATIVE])
    if ordered != list(args.samplers):
        print(f"  reordered to {ordered}: JAX samplers must precede "
              "replica_exchange within one process")

    results = {}
    for name in ordered:
        results[name] = run_sampler(
            name, built, bundle,
            shuffle=shuffle,
            **sampler_kwargs.get(name, {}),
        )

    report = write_timing_report(
        args.report,
        config={
            "n_copies": args.copies,
            "dof_mode": args.dof_mode,
            "samplers": ",".join(args.samplers),
            "restraints": bundle.summary,
        },
        env_info=env,
        results=results,
    )
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
