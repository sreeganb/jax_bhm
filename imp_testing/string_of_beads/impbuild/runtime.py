"""
Runtime plumbing: output directories, log capture, environment probing and
timing reports.

None of this knows anything about IMP or about a particular sampler; it is
deliberately import-light so it can be used before IMP is loaded.
"""

import contextlib
import os
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------
# repo import bootstrap
# --------------------------------------------------------------------------

def ensure_repo_on_path(marker_dirs=("sampling", "scoring")):
    """
    Make the jax_bhm repo root importable without installing it.

    Walks up from this file until it finds a directory containing all of
    `marker_dirs`, and prepends it to sys.path. Safe to call repeatedly.
    Returns the repo root as a Path, or None if not found (in which case the
    package is presumably pip-installed and normal imports will work).
    """
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if all((candidate / d).is_dir() for d in marker_dirs):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    return None


# --------------------------------------------------------------------------
# output directories
# --------------------------------------------------------------------------

def prepare_output_dir(name, parent=None, rollover=True):
    """
    Create a fresh `<name>_output` directory.

    If `rollover` is True and the directory already exists it is renamed to
    `old_<name>_output`; a pre-existing `old_` directory is timestamped rather
    than destroyed, so no run is ever silently overwritten.
    """
    parent = Path(parent) if parent is not None else Path.cwd()
    target = parent / f"{name}_output"
    old = target.parent / f"old_{target.name}"

    if target.exists() and rollover:
        if old.exists():
            stamp = str(int(old.stat().st_mtime_ns))
            rolled = target.parent / f"{old.name}_{stamp}"
            suffix = 1
            while rolled.exists():
                rolled = target.parent / f"{old.name}_{stamp}_{suffix}"
                suffix += 1
            old.rename(rolled)
        target.rename(old)

    target.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------------------------------
# log capture
# --------------------------------------------------------------------------

class _TeeStream:
    """Minimal file-like object that fans writes out to several streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        # Some libraries probe this before emitting progress bars.
        return False


@contextlib.contextmanager
def tee_to_log(log_path):
    """Mirror stdout/stderr into `log_path` while keeping terminal output."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        tee = _TeeStream(sys.stdout, handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            yield log_path


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def set_xla_flags(*flags):
    """
    Append XLA flags to the environment if not already present.

    Must be called *before* jax is imported to have any effect, which is why it
    lives here rather than in jax_bridge.
    """
    current = os.environ.get("XLA_FLAGS", "")
    additions = [f for f in flags if f not in current]
    if additions:
        os.environ["XLA_FLAGS"] = " ".join([current, *additions]).strip()
    return os.environ.get("XLA_FLAGS", "")


def runtime_environment():
    """
    Report which backend JAX will actually use.

    This is the answer to "am I really on the GPU?". Note that it describes the
    JAX side only: IMP's own Monte Carlo (ReplicaExchange) runs through IMP's
    C++ scoring path and is CPU-side regardless of what JAX reports.
    """
    info = {
        "jax_available": False,
        "jax_default_backend": None,
        "jax_platforms": [],
        "jax_device_count": 0,
        "jax_device_names": [],
        "jax_cpu_only": True,
    }
    try:
        import jax
    except ImportError:
        return info

    devices = jax.devices()
    platforms = sorted({d.platform for d in devices})
    info.update(
        jax_available=True,
        jax_default_backend=jax.default_backend(),
        jax_platforms=platforms,
        jax_device_count=len(devices),
        jax_device_names=[str(d) for d in devices],
        jax_cpu_only=bool(platforms) and all(p == "cpu" for p in platforms),
    )
    return info


def print_environment(info=None):
    """Short human-readable environment banner."""
    info = info if info is not None else runtime_environment()
    print("Runtime environment")
    print(f"  JAX available     : {info['jax_available']}")
    print(f"  JAX backend       : {info['jax_default_backend']}")
    print(f"  JAX platforms     : {info['jax_platforms']}")
    print(f"  JAX devices       : {info['jax_device_names']}")
    print(f"  JAX CPU-only      : {info['jax_cpu_only']}")
    return info


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def write_timing_report(report_path, config, env_info, results):
    """
    Write a plain-text benchmark report.

    `results` maps a sampler name to a SamplerResult (or to any object exposing
    `.as_dict()`, or to a plain dict). Rate columns are derived from whichever
    of n_steps / n_frames the sampler reported, so adding a new sampler needs no
    change here.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(value):
        if hasattr(value, "as_dict"):
            return value.as_dict()
        return dict(value)

    lines = ["IMP sampling report", "=" * 78,
             f"Generated (unix time): {time.time():.3f}", ""]

    lines += ["Configuration", "-" * 78]
    for key in sorted(config):
        lines.append(f"{key}: {config[key]}")
    lines.append("")

    lines += ["Runtime environment", "-" * 78]
    for key in sorted(env_info):
        lines.append(f"{key}: {env_info[key]}")
    lines.append("")

    lines += ["Sampler timing", "-" * 78]
    for name, value in results.items():
        record = _to_dict(value)
        elapsed = float(record.get("elapsed_seconds", float("nan")))
        units = record.get("n_units", 0) or 0
        unit_name = record.get("unit_name", "steps")
        rate = (units / elapsed) if (units > 0 and elapsed > 0) else float("nan")

        lines.append(name)
        lines.append(f"  {unit_name}: {units}")
        lines.append(f"  elapsed_seconds: {elapsed:.6f}")
        lines.append(f"  {unit_name}_per_second: {rate:.6f}")
        for key in sorted(record):
            if key in ("elapsed_seconds", "n_units", "unit_name"):
                continue
            lines.append(f"  {key}: {record[key]}")
        lines.append("")

    lines += [
        "Interpretation",
        "-" * 78,
        "JAX-backed samplers (rmh, smc, adaptive_smc) score through",
        "  ScoringFunction._get_jax() and run on the JAX backend named above.",
        "IMP ReplicaExchange scores through IMP's native C++ path and does not",
        "  route through the JAX pipeline, so it is CPU-side in this workflow.",
        "A GPU speed comparison is therefore only meaningful *between JAX runs*",
        "  on different backends, not between a JAX sampler and ReplicaExchange.",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
