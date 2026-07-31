"""
impbuild -- a decoupled IMP/PMI modelling front-end.

The package separates the three things that were previously tangled together in
a single 1000-line script:

    representation.py   what the system *is*      -> BuiltSystem
    restraints.py       what the system *must*    -> RestraintBundle
    samplers.py         how the system is *sampled* -> SamplerResult

Supporting modules:

    jax_bridge.py       IMP <-> JAX/BlackJAX plumbing (one implementation,
                        shared by every JAX-backed sampler)
    runtime.py          output directories, log tee-ing, environment/timing
                        reports

Nothing in this package writes to, or imports from, your existing repo modules
other than `sampling.wrapper_imp_blackjax` and `sampling.imp_blackjax_adapter`,
which are imported read-only inside jax_bridge/samplers.

Typical use (see run_sampling.py):

    from impbuild import (
        SystemSpec, molecule_specs_from_topology, build_system,
        RestraintBuilder, DistanceRestraintSpec, run_sampler,
    )

    built  = build_system(SystemSpec(molecules=..., n_copies=10))
    bundle = (RestraintBuilder(built)
              .connectivity()
              .excluded_volume()
              .distances([...])
              .finalize())
    result = run_sampler("replica_exchange", built, bundle)
"""

from .representation import (
    FragmentSpec,
    MoleculeSpec,
    SystemSpec,
    BuiltSystem,
    build_system,
    molecule_specs_from_topology,
)
from .restraints import (
    DistanceRestraintSpec,
    HarmonicDistanceRestraint,
    RestraintBuilder,
    RestraintBundle,
)
from .samplers import (
    SamplerResult,
    available_samplers,
    run_sampler,
)
from .runtime import (
    prepare_output_dir,
    runtime_environment,
    tee_to_log,
    write_timing_report,
)

__all__ = [
    "FragmentSpec",
    "MoleculeSpec",
    "SystemSpec",
    "BuiltSystem",
    "build_system",
    "molecule_specs_from_topology",
    "DistanceRestraintSpec",
    "HarmonicDistanceRestraint",
    "RestraintBuilder",
    "RestraintBundle",
    "SamplerResult",
    "available_samplers",
    "run_sampler",
    "prepare_output_dir",
    "runtime_environment",
    "tee_to_log",
    "write_timing_report",
]
