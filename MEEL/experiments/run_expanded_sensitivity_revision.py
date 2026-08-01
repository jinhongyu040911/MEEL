from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "cache" / "runs" / "meel_expanded_sensitivity_revision"
DEFAULT_LOG_DIR = EXPERIMENTS / "expanded_sensitivity_revision_logs"
DEFAULT_MANIFEST = EXPERIMENTS / "expanded_sensitivity_revision_manifest.json"

PYTHON = sys.executable

BASE_CONFIGS: dict[str, dict[str, Any]] = {
    "MR2_Chinese": {
        "lr": 0.0003,
        "batch_size": 256,
        "lambda_eq": 0.5,
        "lambda_sep": 0.02,
        "lambda_sparse": 0.02,
        "lambda_unc": 0.08,
        "lambda_real_anchor": 0.02,
        "uncertainty_scale": 3.0,
        "direction_focal_gamma": 1.5,
    },
    "MR2_English": {
        "lr": 0.0005,
        "batch_size": 32,
        "lambda_eq": 0.75,
        "lambda_sep": 0.02,
        "lambda_sparse": 0.02,
        "lambda_unc": 0.08,
        "lambda_real_anchor": 0.0025,
        "uncertainty_scale": 5.0,
        "direction_focal_gamma": 1.5,
    },
    "weibo": {
        "lr": 0.0005,
        "batch_size": 128,
        "lambda_eq": 1.0,
        "lambda_sep": 0.02,
        "lambda_sparse": 0.02,
        "lambda_unc": 0.05,
        "lambda_real_anchor": 0.0025,
        "uncertainty_scale": 5.0,
        "direction_focal_gamma": 1.5,
    },
}

SWEEPS: dict[str, list[Any]] = {
    "direction_focal_gamma": [0.0, 0.5, 1.0, 1.5, 2.0],
    "uncertainty_scale": [1.0, 3.0, 5.0, 7.0, 10.0],
    "lambda_eq": [0.0, 0.25, 0.5, 0.75, 1.0],
    "lambda_sep": [0.0, 0.01, 0.02, 0.05, 0.1],
    "lambda_sparse": [0.0, 0.01, 0.02, 0.05, 0.1],
    "lambda_unc": [0.0, 0.03, 0.05, 0.08, 0.1],
    "lambda_real_anchor": [0.0, 0.0025, 0.005, 0.01, 0.02],
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def value_tag(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def build_command(dataset: str, config: dict[str, Any], output_root: Path, run_name: str, seed: int, epochs: tuple[int, int, int]) -> list[str]:
    return [
        PYTHON,
        "scripts/train_mel_net.py",
        "--dataset",
        dataset,
        "--variant",
        "baseline",
        "--device",
        "cuda",
        "--num-workers",
        "0",
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--lr",
        str(config["lr"]),
        "--batch-size",
        str(config["batch_size"]),
        "--dropout",
        "0.1",
        "--lambda-eq",
        str(config["lambda_eq"]),
        "--lambda-sep",
        str(config["lambda_sep"]),
        "--lambda-sparse",
        str(config["lambda_sparse"]),
        "--lambda-unc",
        str(config["lambda_unc"]),
        "--lambda-real-anchor",
        str(config["lambda_real_anchor"]),
        "--uncertainty-scale",
        str(config["uncertainty_scale"]),
        "--direction-focal-gamma",
        str(config["direction_focal_gamma"]),
        "--epochs-pretrain",
        str(epochs[0]),
        "--epochs-joint",
        str(epochs[1]),
        "--epochs-finetune",
        str(epochs[2]),
        "--seed",
        str(seed),
        "--resume",
    ]


def build_jobs(output_root: Path, datasets: list[str], sweeps: list[str], seed: int, epochs: tuple[int, int, int]) -> list[dict[str, Any]]:
    jobs = []
    for dataset in datasets:
        base = BASE_CONFIGS[dataset]
        for sweep_name in sweeps:
            for value in SWEEPS[sweep_name]:
                config = dict(base)
                config[sweep_name] = value
                run_name = f"expanded_sensitivity_revision_{dataset}_{sweep_name}_{value_tag(value)}_seed{seed}"
                jobs.append(
                    {
                        "dataset": dataset,
                        "sweep": sweep_name,
                        "value": value,
                        "seed": seed,
                        "run_name": run_name,
                        "run_dir": str(output_root / dataset / run_name),
                        "base_config": base,
                        "config": config,
                        "command": build_command(dataset, config, output_root, run_name, seed, epochs),
                    }
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-variable sensitivity sweeps for revision-requested MEEL hyperparameters.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--datasets", default="MR2_Chinese,MR2_English,weibo")
    parser.add_argument("--sweeps", default="direction_focal_gamma,uncertainty_scale,lambda_eq,lambda_sep,lambda_sparse,lambda_unc,lambda_real_anchor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-pretrain", type=int, default=8)
    parser.add_argument("--epochs-joint", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    log_dir = Path(args.log_dir)
    manifest_path = Path(args.manifest)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    sweeps = [item.strip() for item in args.sweeps.split(",") if item.strip()]
    unknown_datasets = sorted(set(datasets) - set(BASE_CONFIGS))
    unknown_sweeps = sorted(set(sweeps) - set(SWEEPS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    if unknown_sweeps:
        raise ValueError(f"Unknown sweeps: {unknown_sweeps}")

    output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = (args.epochs_pretrain, args.epochs_joint, args.epochs_finetune)
    jobs = build_jobs(output_root, datasets, sweeps, args.seed, epochs)
    manifest: dict[str, Any] = {
        "suite": "expanded_sensitivity_revision",
        "created_at": now(),
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
        "purpose": "revision one-variable sweeps for focal exponent, activation sharpness, and objective coefficients",
        "base_configs": BASE_CONFIGS,
        "sweeps": {key: SWEEPS[key] for key in sweeps},
        "datasets": datasets,
        "seed": args.seed,
        "epochs": {"pretrain": args.epochs_pretrain, "joint": args.epochs_joint, "finetune": args.epochs_finetune},
        "job_count": len(jobs),
        "jobs": jobs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"planned_jobs": len(jobs), "manifest": str(manifest_path)}, indent=2))
        return

    failures: list[dict[str, Any]] = []
    for i, job in enumerate(jobs, start=1):
        summary = Path(job["run_dir"]) / "summary.json"
        stdout_path = log_dir / f"{job['run_name']}.stdout.log"
        stderr_path = log_dir / f"{job['run_name']}.stderr.log"
        if summary.exists():
            job["status"] = "SKIPPED_COMPLETED"
            continue
        print(f"[{i}/{len(jobs)}] running {job['run_name']}", flush=True)
        job["status"] = "RUNNING"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            stdout.write(f"\n===== {now()} START {' '.join(job['command'])} =====\n")
            result = subprocess.run(job["command"], cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
        job["returncode"] = result.returncode
        job["status"] = "DONE" if result.returncode == 0 and summary.exists() else "FAILED"
        if job["status"] == "FAILED":
            failures.append(job)
            break
    manifest["finished_at"] = now()
    manifest["status"] = "FAILED" if failures else "DONE"
    manifest["failures"] = failures
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
