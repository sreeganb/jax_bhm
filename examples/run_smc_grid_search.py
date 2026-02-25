#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid-search launcher for examples/run_smc_modular.py with multi-GPU scheduling.

Features
--------
- Runs RMH and HMC parameter sweeps requested by user.
- Uses up to N GPUs in parallel (one job per GPU) via CUDA_VISIBLE_DEVICES.
- Creates per-run output folders under output_grid_search/.
- Collects per-run summary JSON -> consolidated CSV.
- Generates PDF report with quick visual summaries.

Default sweep specification
---------------------------
RMH:
- rmh_sigma: 1.0 .. 10.0 (step 2.0)
- n_mcmc_steps: 5 .. 55 (step 10)
- n_particles: 40 .. 100 (step 10)

HMC:
- hmc_step_size: 0.1 .. 1.0 (step 0.2)
- hmc_num_integration_steps: 5, 10, 15, 20
- n_mcmc_steps: 5 .. 55 (step 10)
- n_particles: 40 .. 100 (step 10)

Shared settings (as requested)
------------------------------
- structural restraints OFF
- exvol_stiffness = 0.1
- slope_factor = 0.001
- likelihood = gaussian
- sigma_ccc = 0.01
- box_size = 300.0
- box_steepness = 10.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class JobSpec:
    kernel: str
    n_particles: int
    n_mcmc_steps: int
    rmh_sigma: Optional[float] = None
    hmc_step_size: Optional[float] = None
    hmc_L: Optional[int] = None
    seed: int = 42
    phase: str = "grid"
    run_name: str = ""
    run_dir: Optional[Path] = None


def _float_token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _rmh_jobs() -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for n_mcmc_steps in range(5, 56, 10):
        for n_particles in range(40, 101, 10):
            for rmh_sigma in range(1, 11, 2):
                run_name = (
                    f"rmh_np{n_particles:03d}_nm{n_mcmc_steps:03d}_"
                    f"sig{_float_token(float(rmh_sigma))}"
                )
                jobs.append(
                    JobSpec(
                        kernel="rmh",
                        n_particles=n_particles,
                        n_mcmc_steps=n_mcmc_steps,
                        rmh_sigma=float(rmh_sigma),
                        run_name=run_name,
                    )
                )
    return jobs


def _hmc_jobs() -> List[JobSpec]:
    jobs: List[JobSpec] = []
    hmc_step_sizes = [0.1 * (i + 1) for i in range(8)]  # 0.1, 0.2, ..., 0.8
    for n_mcmc_steps in range(5, 56, 10):
        for n_particles in range(40, 101, 10):
            for hmc_L in range(5, 21, 5):
                for hmc_step_size in hmc_step_sizes:
                    run_name = (
                        f"hmc_np{n_particles:03d}_nm{n_mcmc_steps:03d}_"
                        f"hs{_float_token(hmc_step_size)}_L{hmc_L:02d}"
                    )
                    jobs.append(
                        JobSpec(
                            kernel="hmc",
                            n_particles=n_particles,
                            n_mcmc_steps=n_mcmc_steps,
                            hmc_step_size=hmc_step_size,
                            hmc_L=hmc_L,
                            run_name=run_name,
                        )
                    )
    return jobs


def build_jobs(include_rmh: bool, include_hmc: bool) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    if include_rmh:
        jobs.extend(_rmh_jobs())
    if include_hmc:
        jobs.extend(_hmc_jobs())
    return jobs


def command_for_job(
    job: JobSpec,
    run_script: Path,
    mrc_path: str,
    target_ess: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(run_script),
        "--kernel", job.kernel,
        "--n_particles", str(job.n_particles),
        "--n_mcmc_steps", str(job.n_mcmc_steps),
        "--mrc", mrc_path,
        "--likelihood", "gaussian",
        "--sigma_ccc", "0.01",
        "--box_size", "300.0",
        "--box_steepness", "10.0",
        "--slope_factor", "0.001",
        "--exvol_stiffness", "0.1",
        "--disable_structural_restraints",
        "--pair_weight", "0.0",
        "--target_ess", str(target_ess),
        "--seed", str(job.seed),
        "--output_dir", str(job.run_dir),
        "--output_prefix", "run",
        "--quiet",
    ]

    if job.kernel == "rmh":
        cmd.extend(["--rmh_sigma", str(job.rmh_sigma)])
    else:
        cmd.extend([
            "--hmc_step_size", str(job.hmc_step_size),
            "--hmc_L", str(job.hmc_L),
        ])

    return cmd


def summary_path(job: JobSpec) -> Path:
    return job.run_dir / "run_summary.json"


def read_run_summary(job: JobSpec) -> Dict:
    summary_file = summary_path(job)
    if not summary_file.exists():
        return {}
    with open(summary_file, "r") as handle:
        return json.load(handle)


def row_from_job(job: JobSpec, status: str, runtime_sec: float = 0.0) -> Dict:
    row = {
        "run_name": job.run_name,
        "phase": job.phase,
        "kernel": job.kernel,
        "seed": job.seed,
        "n_particles": job.n_particles,
        "n_mcmc_steps": job.n_mcmc_steps,
        "rmh_sigma": job.rmh_sigma,
        "hmc_step_size": job.hmc_step_size,
        "hmc_L": job.hmc_L,
        "status": status,
        "launcher_runtime_sec": runtime_sec,
        "best_ccc": "",
        "best_score": "",
        "final_mean_score": "",
        "initial_log_likelihood": "",
        "final_log_likelihood": "",
        "final_log_posterior": "",
        "n_smc_steps": "",
        "wall_time": "",
        "summary_json": str(summary_path(job)),
        "run_dir": str(job.run_dir),
    }

    summary = read_run_summary(job)
    if summary:
        results = summary.get("results", {})
        initial_diag = summary.get("initial_diagnostics", {})
        final_diag = summary.get("final_diagnostics", {})
        row["best_ccc"] = results.get("best_ccc", "")
        row["best_score"] = results.get("best_score", "")
        row["final_mean_score"] = results.get("final_mean_score", "")
        row["initial_log_likelihood"] = initial_diag.get("log_likelihood", "")
        row["final_log_likelihood"] = final_diag.get("log_likelihood", "")
        row["final_log_posterior"] = final_diag.get("log_posterior", "")
        row["n_smc_steps"] = results.get("n_smc_steps", "")
        row["wall_time"] = results.get("wall_time", "")

    return row


def write_csv(rows: List[Dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "phase",
        "kernel",
        "seed",
        "n_particles",
        "n_mcmc_steps",
        "rmh_sigma",
        "hmc_step_size",
        "hmc_L",
        "status",
        "launcher_runtime_sec",
        "best_ccc",
        "best_score",
        "final_mean_score",
        "initial_log_likelihood",
        "final_log_likelihood",
        "final_log_posterior",
        "n_smc_steps",
        "wall_time",
        "summary_json",
        "run_dir",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def write_pdf_report(rows: List[Dict], pdf_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np
    except Exception as exc:
        print(f"[WARN] Could not generate PDF report (matplotlib missing?): {exc}")
        return

    completed = [r for r in rows if r["status"] == "completed"]
    failed = [r for r in rows if r["status"] != "completed"]

    grid_completed = [r for r in completed if r.get("phase") == "grid"]
    rerun_completed = [r for r in completed if r.get("phase") == "rerun"]

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.axis("off")

        lines = [
            "SMC Grid Search Summary",
            "",
            f"Total runs indexed: {len(rows)}",
            f"Completed: {len(completed)}",
            f"Completed (grid): {len(grid_completed)}",
            f"Completed (rerun): {len(rerun_completed)}",
            f"Failed/other: {len(failed)}",
            "",
        ]

        for kernel in ("rmh", "hmc"):
            krows = [r for r in completed if r["kernel"] == kernel and r["best_ccc"] != ""]
            if not krows:
                lines.append(f"{kernel.upper()}: no completed runs")
                continue
            best = max(krows, key=lambda r: _safe_float(r["best_ccc"], -1e9))
            lines.append(
                f"{kernel.upper()} best CCC = {best['best_ccc']} | best_score={best['best_score']} | final_loglik={best['final_log_likelihood']} | run={best['run_name']}"
            )

        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if completed:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            for kernel, marker in (("rmh", "o"), ("hmc", "x")):
                krows = [r for r in completed if r["kernel"] == kernel]
                if not krows:
                    continue
                x = np.array([_safe_float(r["wall_time"]) for r in krows])
                y = np.array([_safe_float(r["best_ccc"]) for r in krows])
                ax.scatter(x, y, s=20, alpha=0.6, marker=marker, label=kernel.upper())
            ax.set_xlabel("Wall Time (s)")
            ax.set_ylabel("Best CCC")
            ax.set_title("Best CCC vs Wall Time")
            ax.legend()
            ax.grid(True, alpha=0.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            for kernel, marker in (("rmh", "o"), ("hmc", "x")):
                krows = [r for r in completed if r["kernel"] == kernel]
                if not krows:
                    continue
                x = np.array([_safe_float(r["best_ccc"]) for r in krows])
                y = np.array([_safe_float(r["final_log_likelihood"]) for r in krows])
                ax.scatter(x, y, s=20, alpha=0.6, marker=marker, label=kernel.upper())
            ax.set_xlabel("Best CCC")
            ax.set_ylabel("Final log-likelihood")
            ax.set_title("Final log-likelihood vs Best CCC")
            ax.legend()
            ax.grid(True, alpha=0.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for kernel in ("rmh", "hmc"):
            krows = [r for r in completed if r["kernel"] == kernel]
            if not krows:
                continue
            top = sorted(krows, key=lambda r: _safe_float(r["best_ccc"], -1e9), reverse=True)[:20]
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            ax.axis("off")
            title = f"Top {len(top)} {kernel.upper()} runs by Best CCC"
            table_lines = [title, ""]
            for idx, row in enumerate(top, start=1):
                if kernel == "rmh":
                    param_text = (
                        f"np={row['n_particles']}, nm={row['n_mcmc_steps']}, "
                        f"sigma={row['rmh_sigma']}"
                    )
                else:
                    param_text = (
                        f"np={row['n_particles']}, nm={row['n_mcmc_steps']}, "
                        f"step={row['hmc_step_size']}, L={row['hmc_L']}"
                    )
                table_lines.append(
                    f"{idx:2d}. CCC={row['best_ccc']} | score={row['best_score']} | loglik={row['final_log_likelihood']} | {param_text} | seed={row['seed']} | run={row['run_name']}"
                )
            ax.text(0.02, 0.98, "\n".join(table_lines), va="top", ha="left", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if rerun_completed:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            ax.axis("off")
            top_rerun = sorted(
                rerun_completed,
                key=lambda r: _safe_float(r["best_ccc"], -1e9),
                reverse=True,
            )[:20]
            table_lines = ["Top rerun results (phase=rerun)", ""]
            for idx, row in enumerate(top_rerun, start=1):
                if row["kernel"] == "rmh":
                    param_text = f"np={row['n_particles']}, nm={row['n_mcmc_steps']}, sigma={row['rmh_sigma']}"
                else:
                    param_text = f"np={row['n_particles']}, nm={row['n_mcmc_steps']}, step={row['hmc_step_size']}, L={row['hmc_L']}"
                table_lines.append(
                    f"{idx:2d}. CCC={row['best_ccc']} | score={row['best_score']} | loglik={row['final_log_likelihood']} | {param_text} | seed={row['seed']}"
                )
            ax.text(0.02, 0.98, "\n".join(table_lines), va="top", ha="left", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search launcher for run_smc_modular.py")
    parser.add_argument("--output_root", type=str, default="output_grid_search",
                        help="Root folder for all grid-search outputs")
    parser.add_argument("--mrc", type=str, default="output/simulated_30ang_3vox.mrc",
                        help="Target MRC path passed to run_smc_modular.py")
    parser.add_argument("--target_ess", type=float, default=0.5,
                        help="Target ESS shared across runs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed used for all runs")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--max_parallel", type=int, default=4,
                        help="Max concurrent jobs (typically number of GPUs)")
    parser.add_argument("--include_rmh", action="store_true", help="Include RMH grid")
    parser.add_argument("--include_hmc", action="store_true", help="Include HMC grid")
    parser.add_argument("--report_only", action="store_true",
                        help="Do not launch runs; only aggregate existing outputs")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun jobs even if run_summary.json already exists")
    parser.add_argument("--limit_jobs", type=int, default=0,
                        help="Optional limit for quick tests (0 = no limit)")
    parser.add_argument("--rerun_top_k", type=int, default=10,
                        help="After grid phase, rerun top-K parameter combinations by best CCC (0 disables)")
    parser.add_argument("--rerun_n_seeds", type=int, default=4,
                        help="Number of seeds to use for each top-K rerun")
    parser.add_argument("--rerun_seed_start", type=int, default=1001,
                        help="Starting seed for rerun phase")
    parser.add_argument("--poll_seconds", type=float, default=3.0,
                        help="Polling interval for scheduler loop")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned jobs/commands without executing")
    return parser.parse_args()


def collect_existing_rows(jobs: List[JobSpec]) -> List[Dict]:
    rows: List[Dict] = []
    for job in jobs:
        if summary_path(job).exists():
            rows.append(row_from_job(job, status="completed", runtime_sec=0.0))
        else:
            rows.append(row_from_job(job, status="missing", runtime_sec=0.0))
    return rows


def _schedule_jobs(
    queue: List[JobSpec],
    gpu_ids: List[str],
    run_script: Path,
    mrc_path: str,
    target_ess: float,
    poll_seconds: float,
    rows: List[Dict],
    csv_path: Path,
) -> None:
    running: Dict[str, Dict] = {}
    launched = 0
    completed_now = 0
    total_to_run = len(queue)

    while queue or running:
        for gpu_id in gpu_ids:
            if gpu_id in running:
                continue
            if not queue:
                continue

            job = queue.pop(0)
            job.run_dir.mkdir(parents=True, exist_ok=True)
            cmd = command_for_job(
                job=job,
                run_script=run_script,
                mrc_path=mrc_path,
                target_ess=target_ess,
            )

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

            process = subprocess.Popen(
                cmd,
                cwd=str((Path(__file__).resolve().parent.parent)),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            running[gpu_id] = {
                "job": job,
                "proc": process,
                "start": time.time(),
            }
            launched += 1
            print(
                f"[LAUNCH] gpu={gpu_id} phase={job.phase} job={job.run_name} "
                f"({launched}/{total_to_run})"
            )

        finished_gpus: List[str] = []
        for gpu_id, info in running.items():
            proc: subprocess.Popen = info["proc"]
            ret = proc.poll()
            if ret is None:
                continue

            job: JobSpec = info["job"]
            runtime_sec = time.time() - info["start"]
            status = "completed" if (ret == 0 and summary_path(job).exists()) else "failed"
            rows.append(row_from_job(job, status=status, runtime_sec=runtime_sec))
            completed_now += 1
            print(
                f"[DONE] gpu={gpu_id} phase={job.phase} status={status} job={job.run_name} "
                f"runtime={runtime_sec:.1f}s ({completed_now}/{total_to_run})"
            )
            finished_gpus.append(gpu_id)

            write_csv(rows, csv_path)

        for gpu_id in finished_gpus:
            del running[gpu_id]

        if queue or running:
            time.sleep(poll_seconds)


def _rerun_jobs_from_top(
    rows: List[Dict],
    output_root: Path,
    top_k: int,
    n_seeds: int,
    seed_start: int,
) -> List[JobSpec]:
    completed_grid = [
        r for r in rows
        if r.get("status") == "completed" and r.get("phase") == "grid" and r.get("best_ccc") != ""
    ]
    if not completed_grid or top_k <= 0 or n_seeds <= 0:
        return []

    top = sorted(
        completed_grid,
        key=lambda r: _safe_float(r.get("best_ccc"), -1e9),
        reverse=True,
    )[:top_k]

    rerun_root = output_root / "rerun_top"
    rerun_root.mkdir(parents=True, exist_ok=True)

    jobs: List[JobSpec] = []
    for rank, row in enumerate(top, start=1):
        kernel = row["kernel"]
        for seed_idx in range(n_seeds):
            seed = seed_start + seed_idx
            if kernel == "rmh":
                base = JobSpec(
                    kernel="rmh",
                    n_particles=int(float(row["n_particles"])),
                    n_mcmc_steps=int(float(row["n_mcmc_steps"])),
                    rmh_sigma=float(row["rmh_sigma"]),
                    seed=seed,
                    phase="rerun",
                )
            else:
                base = JobSpec(
                    kernel="hmc",
                    n_particles=int(float(row["n_particles"])),
                    n_mcmc_steps=int(float(row["n_mcmc_steps"])),
                    hmc_step_size=float(row["hmc_step_size"]),
                    hmc_L=int(float(row["hmc_L"])),
                    seed=seed,
                    phase="rerun",
                )

            if base.kernel == "rmh":
                base.run_name = (
                    f"rerun_rank{rank:02d}_{base.kernel}_np{base.n_particles:03d}_"
                    f"nm{base.n_mcmc_steps:03d}_sig{_float_token(base.rmh_sigma)}_seed{seed}"
                )
            else:
                base.run_name = (
                    f"rerun_rank{rank:02d}_{base.kernel}_np{base.n_particles:03d}_"
                    f"nm{base.n_mcmc_steps:03d}_hs{_float_token(base.hmc_step_size)}_"
                    f"L{base.hmc_L:02d}_seed{seed}"
                )

            base.run_dir = rerun_root / base.run_name
            jobs.append(base)

    return jobs


def launch_grid(args: argparse.Namespace) -> int:
    include_rmh = args.include_rmh
    include_hmc = args.include_hmc
    if not include_rmh and not include_hmc:
        include_rmh = True
        include_hmc = True

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rmh_root = output_root / "rmh"
    hmc_root = output_root / "hmc"
    rmh_root.mkdir(parents=True, exist_ok=True)
    hmc_root.mkdir(parents=True, exist_ok=True)

    run_script = (Path(__file__).resolve().parent / "run_smc_modular.py").resolve()

    jobs = build_jobs(include_rmh=include_rmh, include_hmc=include_hmc)
    for job in jobs:
        root = rmh_root if job.kernel == "rmh" else hmc_root
        job.seed = args.seed
        job.run_dir = root / job.run_name

    if args.limit_jobs and args.limit_jobs > 0:
        jobs = jobs[: args.limit_jobs]

    print(f"[INFO] Planned jobs: {len(jobs)}")
    print(f"[INFO] Output root: {output_root}")

    csv_path = output_root / "grid_search_summary.csv"
    pdf_path = output_root / "grid_search_report.pdf"

    if args.report_only:
        rows = collect_existing_rows(jobs)
        write_csv(rows, csv_path)
        write_pdf_report(rows, pdf_path)
        print(f"[INFO] Wrote CSV: {csv_path}")
        print(f"[INFO] Wrote PDF: {pdf_path}")
        return 0

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided via --gpus")

    max_parallel = min(args.max_parallel, len(gpu_ids))
    gpu_ids = gpu_ids[:max_parallel]

    rows: List[Dict] = []
    queue: List[JobSpec] = []

    for job in jobs:
        already_done = summary_path(job).exists()
        if already_done and not args.rerun:
            rows.append(row_from_job(job, status="completed", runtime_sec=0.0))
            continue
        queue.append(job)

    print(f"[INFO] Already complete/skipped: {len(rows)}")
    print(f"[INFO] To launch now: {len(queue)}")

    if args.dry_run:
        for idx, job in enumerate(queue[:20], start=1):
            cmd = command_for_job(
                job=job,
                run_script=run_script,
                mrc_path=args.mrc,
                target_ess=args.target_ess,
            )
            print(f"[DRY-RUN {idx:04d}] {job.run_name}")
            print("  " + " ".join(cmd))
        if len(queue) > 20:
            print(f"[DRY-RUN] ... {len(queue) - 20} more jobs not shown")
        write_csv(rows, csv_path)
        print(f"[INFO] Wrote partial CSV: {csv_path}")
        return 0

    _schedule_jobs(
        queue=queue,
        gpu_ids=gpu_ids,
        run_script=run_script,
        mrc_path=args.mrc,
        target_ess=args.target_ess,
        poll_seconds=args.poll_seconds,
        rows=rows,
        csv_path=csv_path,
    )

    if args.rerun_top_k > 0 and args.rerun_n_seeds > 0:
        rerun_jobs = _rerun_jobs_from_top(
            rows=rows,
            output_root=output_root,
            top_k=args.rerun_top_k,
            n_seeds=args.rerun_n_seeds,
            seed_start=args.rerun_seed_start,
        )
        if rerun_jobs:
            rerun_queue: List[JobSpec] = []
            for job in rerun_jobs:
                already_done = summary_path(job).exists()
                if already_done and not args.rerun:
                    rows.append(row_from_job(job, status="completed", runtime_sec=0.0))
                    continue
                rerun_queue.append(job)

            print(f"[INFO] Rerun phase queued jobs: {len(rerun_queue)}")
            _schedule_jobs(
                queue=rerun_queue,
                gpu_ids=gpu_ids,
                run_script=run_script,
                mrc_path=args.mrc,
                target_ess=args.target_ess,
                poll_seconds=args.poll_seconds,
                rows=rows,
                csv_path=csv_path,
            )

    # Add rows for any jobs that remained skipped in queue logic (none expected), then finalize report
    write_csv(rows, csv_path)
    write_pdf_report(rows, pdf_path)

    n_failed = sum(1 for r in rows if r["status"] != "completed")
    print(f"[INFO] Grid search finished. Rows: {len(rows)} | Failed: {n_failed}")
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] PDF: {pdf_path}")

    return 0 if n_failed == 0 else 2


def main() -> int:
    args = parse_args()
    return launch_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())
