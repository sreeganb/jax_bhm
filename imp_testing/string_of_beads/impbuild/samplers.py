"""
Samplers.

Responsibility: run a chosen sampling algorithm on a BuiltSystem +
RestraintBundle and produce log files, RMF3 trajectories and a SamplerResult.

Every sampler is registered under a name and shares one entry point:

    result = run_sampler("rmh", built, bundle, n_steps=10000)
    result = run_sampler("replica_exchange", built, bundle, number_of_frames=1000)

Adding a sampler means writing one function and adding one @register line;
nothing else in the package changes.

The setup shared by every JAX-backed sampler (shuffle -> build JAX context ->
run -> write trajectory -> report) lives in `_run_jax_sampler` and
`_run_smc_family`, which is where the three duplicated blocks of the previous
script collapsed to.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import IMP
import IMP.core
import IMP.pmi.macros
import IMP.pmi.tools

from . import jax_bridge
from .runtime import ensure_repo_on_path, prepare_output_dir, tee_to_log

ensure_repo_on_path()


# --------------------------------------------------------------------------
# results and shared configuration
# --------------------------------------------------------------------------

@dataclass
class SamplerResult:
    """Uniform result object, whatever the sampler."""
    sampler: str
    elapsed_seconds: float
    n_units: int = 0                 # steps or frames, sampler dependent
    unit_name: str = "steps"
    final_score: float = float("nan")
    best_log_posterior: float = float("nan")
    acceptance_rate: Optional[float] = None
    output_dir: Optional[str] = None
    files: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        record = {
            "sampler": self.sampler,
            "elapsed_seconds": float(self.elapsed_seconds),
            "n_units": int(self.n_units),
            "unit_name": self.unit_name,
            "final_score": float(self.final_score),
            "best_log_posterior": float(self.best_log_posterior),
            "output_dir": self.output_dir,
        }
        if self.acceptance_rate is not None:
            record["acceptance_rate"] = float(self.acceptance_rate)
        record.update({f"file_{k}": v for k, v in self.files.items()})
        record.update(self.extra)
        return record


@dataclass
class ShuffleConfig:
    """
    Randomisation applied before sampling.

    The default bounding box grows with the number of copies so that a 100-copy
    system is not shuffled into the same volume as a 1-copy system, which would
    start every run deep inside an excluded-volume clash.
    """
    enabled: bool = True
    max_translation: float = 300.0
    bounding_box: Optional[tuple] = None
    avoid_collision_rb: bool = False

    def box_for(self, built):
        if self.bounding_box is not None:
            return self.bounding_box
        half = 60.0 * max(1.0, float(built.n_copies) ** (1.0 / 3.0))
        return ((-half, -half, -half), (half, half, half))

    def apply(self, built, bundle=None, verbose=True):
        if not self.enabled:
            return
        IMP.pmi.tools.shuffle_configuration(
            built.root_hier,
            max_translation=self.max_translation,
            avoidcollision_rb=self.avoid_collision_rb,
            bounding_box=self.box_for(built),
        )
        if verbose and bundle is not None:
            print(f"  shuffled; score = {bundle.evaluate():.4f}")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_SAMPLERS = {}


def register(name):
    def decorator(fn):
        _SAMPLERS[name] = fn
        return fn
    return decorator


def available_samplers():
    return sorted(_SAMPLERS)


def run_sampler(name, built, bundle, output_dir=None, log=True,
                shuffle=None, verbose=True, **kwargs):
    """
    Run a registered sampler.

    name        one of available_samplers()
    built       BuiltSystem
    bundle      RestraintBundle
    output_dir  where logs/RMFs go; defaults to ./<name>_output with rollover
    log         mirror stdout/stderr into <output_dir>/<name>.log
    shuffle     ShuffleConfig, or None for the default
    **kwargs    forwarded to the sampler
    """
    if name not in _SAMPLERS:
        raise KeyError(f"Unknown sampler '{name}'. Available: {available_samplers()}")

    output_dir = Path(output_dir) if output_dir else prepare_output_dir(name)
    shuffle = shuffle if shuffle is not None else ShuffleConfig()

    def _run():
        print(f"=== sampler: {name} ===")
        started = time.perf_counter()
        result = _SAMPLERS[name](
            built=built, bundle=bundle, output_dir=output_dir,
            shuffle=shuffle, verbose=verbose, **kwargs
        )
        result.output_dir = str(output_dir)
        print(f"=== {name} finished in {result.elapsed_seconds:.2f} s "
              f"(wall {time.perf_counter() - started:.2f} s) ===")
        return result

    if log:
        with tee_to_log(output_dir / f"{name}.log"):
            return _run()
    return _run()


# --------------------------------------------------------------------------
# IMP native sampling
# --------------------------------------------------------------------------

@register("replica_exchange")
def _replica_exchange(built, bundle, output_dir, shuffle, verbose,
                      number_of_frames=1000,
                      monte_carlo_steps=10,
                      number_of_best_scoring_models=5,
                      min_temperature=1.0,
                      max_temperature=2.5,
                      movers=None,
                      **kwargs):
    """
    IMP PMI ReplicaExchange, i.e. IMP's native C++ Monte Carlo.

    This is the reference path: it does *not* go through the JAX scoring
    pipeline, so it is CPU-side no matter what JAX reports as its backend. Use
    it as the correctness/performance baseline for the JAX samplers.
    """
    shuffle.apply(built, bundle, verbose)

    all_movers = list(built.dof.get_movers())
    selected = list(movers) if movers is not None else all_movers
    if verbose:
        print(f"  movers: {len(selected)} of {len(all_movers)}")
        print(f"  frames={number_of_frames}, mc_steps={monte_carlo_steps}")

    # PMI resolves global_output_directory relative to the working directory.
    try:
        rel_output = str(Path(output_dir).relative_to(Path.cwd()))
    except ValueError:
        rel_output = str(output_dir)

    macro = IMP.pmi.macros.ReplicaExchange(
        built.model,
        root_hier=built.root_hier,
        monte_carlo_sample_objects=selected,
        output_objects=list(bundle.wrappers),
        monte_carlo_steps=int(monte_carlo_steps),
        number_of_frames=int(number_of_frames),
        number_of_best_scoring_models=int(number_of_best_scoring_models),
        replica_exchange_minimum_temperature=float(min_temperature),
        replica_exchange_maximum_temperature=float(max_temperature),
        global_output_directory=rel_output,
        **kwargs,
    )

    started = time.perf_counter()
    macro.execute_macro()
    elapsed = time.perf_counter() - started

    final_rmf = jax_bridge.write_snapshot_rmf(
        built, Path(output_dir) / "replica_exchange_final.rmf3"
    )
    final_score = bundle.evaluate()
    print(f"  final IMP score: {final_score:.4f}")

    return SamplerResult(
        sampler="replica_exchange",
        elapsed_seconds=elapsed,
        n_units=int(number_of_frames),
        unit_name="frames",
        final_score=final_score,
        files={"final_rmf": final_rmf},
        extra={"monte_carlo_steps": int(monte_carlo_steps)},
    )


# --------------------------------------------------------------------------
# shared JAX sampler scaffolding
# --------------------------------------------------------------------------

def _prepare_jax_context(built, bundle, shuffle, box_half_width, dof_mode,
                         verbose, roundtrip=True):
    """
    Shuffle, then build the JAX context, in that order.

    The order matters: the context snapshots current IMP coordinates into the
    rows that are held fixed during sampling. Building it before shuffling
    leaves those rows describing a conformation the sampler never sees.
    """
    shuffle.apply(built, bundle, verbose)
    context = jax_bridge.build_jax_context(
        built, bundle, box_half_width=box_half_width,
        dof_mode=dof_mode, verbose=verbose,
    )
    if roundtrip:
        jax_bridge.check_roundtrip(built, context, verbose=verbose)
    return context


def _report_posterior(context, flat, prefix, verbose=True):
    """Print score / log-prior / log-posterior for one state."""
    if not verbose:
        return
    print(f"  {prefix}: score={float(context.adapter.imp_score(flat)):.6f}, "
          f"log_prior={float(context.adapter.log_prior(flat)):.6f}, "
          f"log_posterior={float(context.adapter.log_prob(flat)):.6f}")


# --------------------------------------------------------------------------
# BlackJAX random-walk Metropolis-Hastings
# --------------------------------------------------------------------------

@register("rmh")
def _rmh(built, bundle, output_dir, shuffle, verbose,
         n_steps=10000,
         sigma=2.0,
         seed=0,
         box_half_width=300.0,
         dof_mode="all",
         sync_stride=10,
         rmf_stride=10,
         **kwargs):
    """Random-walk Metropolis-Hastings through the JAX scoring path."""
    import jax
    from sampling.wrapper_imp_blackjax import run_rmh_on_imp_system

    context = _prepare_jax_context(built, bundle, shuffle, box_half_width,
                                   dof_mode, verbose)

    x0 = context.adapter.encode()
    _report_posterior(context, x0, "initial", verbose)

    # Use the adapter's own proposal if it provides one (it understands the
    # manifold structure of rigid-body rotations; an isotropic Gaussian on
    # quaternion components does not).
    proposal_fn = None
    if hasattr(context.adapter, "make_rmh_proposal_fn"):
        cfg = (context.adapter.suggested_rmh_proposal()
               if hasattr(context.adapter, "suggested_rmh_proposal") else {})
        proposal_fn = context.adapter.make_rmh_proposal_fn(**cfg)
        if verbose:
            print(f"  using adapter proposal: {cfg}")

    trajectory_rmf = Path(output_dir) / "rmh_trajectory.rmf3"
    rmf_output, step_callback = jax_bridge.make_rmf_stride_writer(
        built, trajectory_rmf, stride=rmf_stride
    )

    started = time.perf_counter()
    result = run_rmh_on_imp_system(
        log_prob_fn=context.adapter.log_prob,
        initial_position=x0,
        rng_key=jax.random.PRNGKey(int(seed)),
        n_steps=int(n_steps),
        sigma=float(sigma),
        proposal_fn=proposal_fn,
        sync_fn=context.sync_fn(built.model),
        sync_stride=int(sync_stride),
        step_callback=step_callback,
        verbose=verbose,
        **kwargs,
    )
    elapsed = time.perf_counter() - started

    rmf_output.close_rmf(str(trajectory_rmf))

    # Make sure the saved final frame is the terminal chain state, not whatever
    # the last strided sync happened to leave behind.
    final_position = result.positions[-1]
    jax_bridge.apply_flat_position(built, context, final_position)
    _report_posterior(context, final_position, "final", verbose)

    final_rmf = jax_bridge.write_snapshot_rmf(
        built, Path(output_dir) / "rmh_final.rmf3"
    )
    final_score = bundle.evaluate()
    print(f"  acceptance rate: {result.acceptance_rate:.2%}")
    print(f"  final IMP score: {final_score:.4f}")

    return SamplerResult(
        sampler="rmh",
        elapsed_seconds=elapsed,
        n_units=int(n_steps),
        unit_name="steps",
        final_score=final_score,
        best_log_posterior=float(np.max(result.log_probs)),
        acceptance_rate=float(result.acceptance_rate),
        files={"trajectory_rmf": str(trajectory_rmf), "final_rmf": final_rmf},
        extra={"dof_mode": dof_mode, "sigma": float(sigma)},
    )


# --------------------------------------------------------------------------
# sequential Monte Carlo (fixed schedule and adaptive share one implementation)
# --------------------------------------------------------------------------

def _run_smc_family(name, runner, runner_kwargs, built, bundle, output_dir,
                    shuffle, verbose, box_half_width, dof_mode, seed):
    """
    Common body for every particle-based SMC variant.

    The fixed-schedule and adaptive runners have the same call signature apart
    from their schedule arguments and return the same 5-tuple, so everything
    except `runner` and `runner_kwargs` is shared.
    """
    import jax

    context = _prepare_jax_context(built, bundle, shuffle, box_half_width,
                                   dof_mode, verbose, roundtrip=False)
    _report_posterior(context, context.adapter.encode(), "initial", verbose)

    started = time.perf_counter()
    state, info, best_positions, best_scores, lambdas = runner(
        adapter=context.adapter,
        rng_key=jax.random.PRNGKey(int(seed)),
        save_rmf3_path=None,
        verbose=verbose,
        **runner_kwargs,
    )
    elapsed = time.perf_counter() - started

    trajectory_rmf = Path(output_dir) / f"{name}_best_trajectory.rmf3"
    n_frames = jax_bridge.write_positions_rmf(
        built, context, best_positions, trajectory_rmf
    )

    # write_positions_rmf leaves the hierarchy at the last written frame, so the
    # final snapshot and the reported score are consistent with each other.
    final_rmf = jax_bridge.write_snapshot_rmf(
        built, Path(output_dir) / f"{name}_final.rmf3"
    )
    final_score = bundle.evaluate()

    finite = np.asarray(best_scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    best = float(np.max(finite)) if finite.size else float("nan")

    print(f"  frames written: {n_frames} of {len(best_positions)} "
          f"({len(best_positions) - n_frames} non-finite skipped)")
    print(f"  best log posterior: {best:.6f}")
    print(f"  final IMP score: {final_score:.4f}")

    return SamplerResult(
        sampler=name,
        elapsed_seconds=elapsed,
        n_units=int(len(lambdas)) if lambdas is not None else 0,
        unit_name="temperature_steps",
        final_score=final_score,
        best_log_posterior=best,
        files={"trajectory_rmf": str(trajectory_rmf), "final_rmf": final_rmf},
        extra={"dof_mode": dof_mode,
               "n_particles": runner_kwargs.get("n_particles"),
               "n_mcmc_steps": runner_kwargs.get("n_mcmc_steps")},
    )


@register("smc")
def _smc(built, bundle, output_dir, shuffle, verbose,
         n_particles=100,
         n_temperature_steps=100,
         schedule="geometric",
         kernel="rmh",
         rmh_sigma=2.0,
         n_mcmc_steps=50,
         score_batch_size=16,
         seed=1,
         box_half_width=300.0,
         dof_mode="all",
         debug=False,
         debug_stride=10,
         **kwargs):
    """Sequential Monte Carlo on a fixed tempering schedule."""
    from sampling.wrapper_imp_blackjax import run_smc_on_imp_system

    runner_kwargs = dict(
        n_particles=int(n_particles),
        n_temperature_steps=int(n_temperature_steps),
        schedule=schedule,
        kernel=kernel,
        rmh_sigma=float(rmh_sigma),
        n_mcmc_steps=int(n_mcmc_steps),
        score_batch_size=int(score_batch_size),
        debug=bool(debug),
        debug_stride=int(debug_stride),
        **kwargs,
    )
    return _run_smc_family("smc", run_smc_on_imp_system, runner_kwargs,
                           built, bundle, output_dir, shuffle, verbose,
                           box_half_width, dof_mode, seed)


@register("adaptive_smc")
def _adaptive_smc(built, bundle, output_dir, shuffle, verbose,
                  n_particles=100,
                  max_temperature_steps=200,
                  target_ess=0.5,
                  rmh_sigma=2.0,
                  n_mcmc_steps=50,
                  score_batch_size=16,
                  seed=2,
                  box_half_width=300.0,
                  dof_mode="all",
                  debug=False,
                  debug_stride=10,
                  **kwargs):
    """Sequential Monte Carlo with an ESS-adaptive tempering schedule."""
    from sampling.wrapper_imp_blackjax import run_adaptive_smc_on_imp_system

    runner_kwargs = dict(
        n_particles=int(n_particles),
        max_temperature_steps=int(max_temperature_steps),
        target_ess=float(target_ess),
        rmh_sigma=float(rmh_sigma),
        n_mcmc_steps=int(n_mcmc_steps),
        score_batch_size=int(score_batch_size),
        debug=bool(debug),
        debug_stride=int(debug_stride),
        **kwargs,
    )
    return _run_smc_family("adaptive_smc", run_adaptive_smc_on_imp_system,
                           runner_kwargs, built, bundle, output_dir, shuffle,
                           verbose, box_half_width, dof_mode, seed)
