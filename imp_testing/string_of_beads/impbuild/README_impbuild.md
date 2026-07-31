# impbuild

A decoupled replacement for `run_imp_flexible_blackjax.py`.

## Install

Drop these into `imp_testing/string_of_beads/` (next to `data/`):

```
impbuild/
    __init__.py
    representation.py
    restraints.py
    samplers.py
    jax_bridge.py
    runtime.py
run_sampling.py
```

Nothing in `sampling/`, `scoring/`, `io_utils/`, `representation/` or any other
existing repo directory is touched. `impbuild` imports exactly two modules from
the repo, read-only:

- `sampling.imp_blackjax_adapter` (`IMPDOFSpace`, `IMPSMCAdapter`,
  `assert_imp_roundtrip`, `write_flat_to_imp`)
- `sampling.wrapper_imp_blackjax` (`run_rmh_on_imp_system`,
  `run_smc_on_imp_system`, `run_adaptive_smc_on_imp_system`)

If `jax_bhm` is not pip-installed, `runtime.ensure_repo_on_path()` walks up the
directory tree until it finds a folder containing `sampling/` and `scoring/` and
prepends it to `sys.path`. No `PYTHONPATH` fiddling needed.

## Run

```bash
cd imp_testing/string_of_beads
python run_sampling.py                                   # REX + RMH, 10 copies
python run_sampling.py --copies 20 --samplers rmh
python run_sampling.py --samplers replica_exchange --frames 500 --mc-steps 10
python run_sampling.py --samplers smc adaptive_smc --dof-mode flex
```

Outputs land in `<sampler>_output/` with rollover to `old_<sampler>_output/`:
a `.log`, a trajectory RMF3, a final RMF3, and (for ReplicaExchange) the usual
PMI `stat.*.out` files. A combined `sampling_report.txt` is written at the end.

## Structure

| Module | Owns |
|---|---|
| `representation.py` | `MoleculeSpec`, `SystemSpec` → `BuiltSystem` (hierarchy + DOF) |
| `restraints.py` | `RestraintBuilder` → `RestraintBundle` (wrappers + raw restraints + scoring function) |
| `samplers.py` | `run_sampler(name, built, bundle, ...)` → `SamplerResult` |
| `jax_bridge.py` | IMP↔JAX adapter, leaf→row mapping, RMF writing (shared by all JAX samplers) |
| `runtime.py` | output dirs, log tee, environment probe, timing report |

The user-facing script is three declarations and one call:

```python
built  = build_system(SystemSpec(molecules=..., n_copies=10))
bundle = (RestraintBuilder(built)
          .connectivity()
          .excluded_volume(resolution=1)
          .distances(DISTANCE_RESTRAINTS)
          .finalize())
result = run_sampler("rmh", built, bundle, n_steps=10000)
```

## Corrections to the previous script

**Copies, not states.** `bs.add_state(topology)` in a loop created *N states*.
A state is an alternative conformation of the same system (multi-state
modelling); copies are additional molecules in one assembly. The old code's
`copy_index` was 0 everywhere and its "inter-copy" restraints connected
different states, which is not physically meaningful. `build_system` now creates
copy 0 with `state.create_molecule()` and copies 1..N-1 with `create_copy()`,
each with its own chain ID and identical representation, all inside one state.
Selections are now `IMP.atom.Selection(root_hier, molecule="KCOIL",
copy_index=i, residue_index=r)`.

`create_copy` rather than `create_clone`: clones are intended for
symmetry-constrained duplicates and cannot take independent representation.
Copies get identical representation added explicitly and are independently
sampleable.

**Three duplicated sampler bodies → one.** RMH, SMC and adaptive SMC each
repeated shuffle → `build_smc_adapter_context` → run → write RMF → print score.
That is now `_prepare_jax_context` plus `_run_smc_family`; adding a fourth
sampler is one function and one `@register` line.

**Shuffle-then-build ordering.** `build_smc_adapter_context` snapshots current
IMP coordinates into the rows held fixed during sampling. RMH built the context
from post-`build_imp_system` coordinates while SMC rebuilt after its own
shuffle — an inconsistency. All JAX samplers now shuffle first, then build.

**Dead code removed.** `run_replica_exchange_case` computed `flexible_movers`,
logged that it was using them, then passed `all_movers`. The mover set is now an
explicit parameter that defaults to all of them.

**Excluded volume is global.** One `ExcludedVolumeSphere` over the root
hierarchy covers intra-copy and inter-copy overlap; per-copy restraints would
miss the latter, which is precisely what matters when scaling up.

**Distance restraints are declarative.** `DistanceRestraintSpec` names two
residues and a `scope` (`intra_copy`, `inter_copy_chain`, `inter_copy_ring`,
`explicit`) and is expanded over copies at build time, so changing `n_copies`
from 1 to 100 requires no restraint-list edits. The default config implements
the requested test case: KCOIL residue 1 ↔ ECOIL residue 52, inside each copy.

**Selection errors are loud.** `select_one_particle` raises with molecule,
residue, copy and resolution in the message instead of indexing `particles[0]`
from a possibly-empty or ambiguous selection, and rejects specs whose two
endpoints collapse onto the same bead.

## Things to check on first run

These depend on your IMP build and on `data/topology.txt`, and I could not
execute them here:

1. **`Molecule.create_copy(chain_id)`** — present in current PMI. If your IMP is
   older and it is missing, that is the one API to swap.
2. **`residue_range`** from `TopologyReader` — handled as `'all'` → `None`, but
   confirm your topology's format parses as expected from the printed summary.
3. **Rigid vs. flexible** — `molecule_specs_from_topology` sets `rigid=True` when
   the topology's rigid-body column is non-empty. For a pure string-of-beads
   toy you probably want everything flexible; if so, set `spec.rigid = False`
   on the returned specs before calling `build_system`.
4. **`ExcludedVolumeSphere` cost at high copy counts** — it is O(N) with a
   neighbour list but the constant matters. If 10+ copies crawl, raise the
   resolution or restrict `included_objects`.
5. **Chain ID overflow** — single-character IDs run out at 62 chains; the
   generator then emits two-character IDs. Some RMF viewers dislike those.

## GPU

`runtime_environment()` reports the JAX backend and devices, and the banner
prints at startup. Only the JAX samplers (`rmh`, `smc`, `adaptive_smc`) run
through `ScoringFunction._get_jax()`; `replica_exchange` uses IMP's C++ path and
is CPU-side regardless. A GPU comparison is therefore meaningful *between JAX
runs on different backends* — force CPU with `JAX_PLATFORMS=cpu` and compare
against the default — not between a JAX sampler and ReplicaExchange, which the
old timing report conflated.
