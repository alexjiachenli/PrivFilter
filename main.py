"""Experiment entry point and run metadata."""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import platform
import random
import statistics
import time
import numpy as np
import torch
import yaml
import algorithm
from handle import Handle
import differential_privacy as dp
from mpc_dp_release import DP_RELEASE_CLIENT_LDP, canonical_dp_release_source

logger = logging.getLogger("logger")


def seed(seeds):
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    random.seed(seeds)


def write_csv_result(metrics, params, total_time=0.0, filepath="outputs/results.csv"):
    import csv
    import os

    fieldnames = [
        "dataset",
        "epoch",
        "defence_method",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "accuracy",
        "f1",
        "clean_accuracy",
        "attack_success_rate",
        "runtime",
        "note",
        "malicious_accept_rate",
        "benign_reject_rate",
        "benign_acceptance_rate",
        "mean_accepted_client_count",
        "mean_malicious_score",
        "mean_benign_score",
    ]
    row = {
        "dataset": params.get("type", "Unknown"),
        "epoch": params.get("epochs", 0),
        "defence_method": params.get("defence_method", "None"),
        "tp": metrics.get("tp", 0),
        "fp": metrics.get("fp", 0),
        "tn": metrics.get("tn", 0),
        "fn": metrics.get("fn", 0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "f1": metrics.get("f1", 0.0),
        "clean_accuracy": metrics.get("clean_accuracy", 0.0),
        "attack_success_rate": metrics.get("attack_success_rate", 0.0),
        "runtime": total_time,
        "note": params.get("note", ""),
        "malicious_accept_rate": metrics.get("malicious_accept_rate", ""),
        "benign_reject_rate": metrics.get("benign_reject_rate", ""),
        "benign_acceptance_rate": metrics.get("benign_acceptance_rate", ""),
        "mean_accepted_client_count": metrics.get("mean_accepted_client_count", ""),
        "mean_malicious_score": metrics.get("mean_malicious_score", ""),
        "mean_benign_score": metrics.get("mean_benign_score", ""),
    }
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()
            writer.writerow(row)
        logger.info(f"Results written to {filepath}")
    except Exception as e:
        logger.error(f"Failed to write results to csv: {e}")


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_privacy_accounting(params):
    """Record the Gaussian reference budget for the selected releases and schedule."""
    if not params.get("enable_ldp", False):
        return None
    dp.require_balanced_schedule(params)
    sigma = params.get("ldp_calibrated_sigma")
    if sigma is None and params.get("ldp_noise_multiplier") is not None:
        sigma = float(params["ldp_noise_multiplier"]) * float(
            params.get("ldp_max_grad_norm", params.get("ldp_sensitivity", 1.0))
        )
    if sigma is None:
        return None
    calibration = dp.evaluate_accountant_for_sigma(params, sigma=float(sigma))
    dp._record_release_accounting(params, calibration)
    target = params.get("ldp_target_epsilon")
    params["ldp_budget_met"] = bool(
        target is None or calibration["epsilon"] <= float(target) * (1 + 1e-06)
    )
    logger.info(
        "[LDP][accounting] conversion=%s, %d release(s)/round (%s), worst-case client %d releases, epsilon=%.4f (exact Gaussian %.4f) against target %s; release cap enforced=%s",
        calibration["rdp_conversion"],
        calibration["releases_per_round"],
        calibration["release_basis"],
        calibration["worst_case_releases"],
        calibration["epsilon"],
        calibration["epsilon_exact_gaussian"],
        target,
        params.get("ldp_release_cap_enforced"),
    )
    if not params["ldp_budget_met"]:
        logger.warning(
            "[LDP][accounting] this configuration does not meet its declared epsilon target; the label is wrong, not merely loose."
        )
    return calibration


def _realized_release_counts(release_metrics):
    """Per-client counts of coordinator-visible DP releases, from the release log.

    Returns ``None`` when the run opened none, so the caller falls back to round
    counts. Each metric lists the clients that contributed to that one release,
    which is exactly the granularity composition needs.
    """
    if not release_metrics:
        return None
    partitioned = any(
        (
            str(metric.get("release_pairing", "")) == "partition_correlated"
            for metric in release_metrics
        )
    )
    counts = {}
    seen = set()
    for metric in release_metrics:
        round_id = metric.get("round_id")
        for client_id in metric.get("client_ids") or []:
            key = int(client_id)
            if partitioned:
                if (key, round_id) in seen:
                    continue
                seen.add((key, round_id))
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return [counts[key] for key in sorted(counts)]


def _participation_summary(counts_dict):
    counts = [int(v) for _, v in sorted((counts_dict or {}).items())]
    if not counts:
        return {"counts": [], "min": 0, "mean": 0.0, "median": 0.0, "max": 0, "std": 0.0}
    return {
        "counts": counts,
        "min": int(min(counts)),
        "mean": float(statistics.mean(counts)),
        "median": float(statistics.median(counts)),
        "max": int(max(counts)),
        "std": float(statistics.pstdev(counts) if len(counts) > 1 else 0.0),
    }


def write_run_artifacts(handle, metrics, total_time):
    """Write metrics, environment details, schedules, and accounting metadata."""
    params = handle.params
    counts = getattr(handle, "participation_counts", {}) or {}
    release_metrics = getattr(handle, "mpc_dp_release_metrics", []) or []
    calibration = {
        key: value
        for key, value in params.items()
        if key.startswith(("ldp_", "mpc_dp_")) and "seed" not in key
    }
    summary = {
        "schema": "privfilter-run-v1",
        "config": params,
        "seed": params.get("seed"),
        "metrics": metrics,
        "clean_accuracy": metrics.get("clean_accuracy"),
        "attack_success_rate": metrics.get("attack_success_rate"),
        "wall_clock_runtime_s": float(total_time),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(next(handle.model.parameters()).device),
        },
        "participation": _participation_summary(counts),
        "privacy_calibration": calibration,
        "mpc_dp_release_metrics": release_metrics,
        "output_folder": handle.folder_path,
    }
    output = Path(handle.folder_path)
    records = {
        params.get("experiment_summary_path") or str(output / "run_summary.json"): summary,
        params.get("participation_schedule_path") or str(output / "schedule.json"): {
            "seed": params.get("seed"),
            "round_client_schedule": getattr(handle, "round_client_schedule", []),
        },
        params.get("participation_counts_path")
        or str(output / "participation_counts.json"): counts,
        params.get("privacy_calibration_path")
        or str(output / "privacy_calibration.json"): calibration,
    }
    for path, record in records.items():
        _ensure_parent(path)
        Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    service = getattr(handle, "mpc_dp_release_service", None)
    if service is not None:
        service.write_artifacts(
            handle.folder_path,
            participation_counts=counts,
            delta=float(params.get("ldp_target_delta", params.get("ldp_delta", 1e-5))),
        )
    return summary


def run():
    parser = argparse.ArgumentParser(description="Run a curated PrivFilter experiment")
    parser.add_argument("--params", required=True, help="JSON or YAML configuration")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "GTSRB"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--poison-client-num", type=int)
    parser.add_argument("--note")
    args = parser.parse_args()
    params_path = Path(os.path.expandvars(args.params)).expanduser().resolve()
    with params_path.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    if not isinstance(params, dict):
        raise ValueError("Configuration must be a mapping")
    for argument, key in [
        ("dataset", "type"),
        ("epochs", "epochs"),
        ("dirichlet_alpha", "dirichlet_alpha"),
        ("poison_client_num", "poison_client_num"),
        ("note", "note"),
    ]:
        value = getattr(args, argument)
        if value is not None:
            params[key] = value
    if params.get("type") not in {"CIFAR10", "CIFAR100", "GTSRB"}:
        raise ValueError("Supported datasets: CIFAR10, CIFAR100, GTSRB")
    if params.get("algorithm", "FedAvg") != "FedAvg":
        raise ValueError("Supported algorithm: FedAvg")
    allowed = {
        "",
        "none",
        "ours",
        "filterfl",
        "ours_cryptgpu",
        "priv-filterfl",
        "ours_plaintext_cryptgpu",
        "krum",
        "trmean",
        "trimmed_mean",
        "flame",
        "shieldfl",
        "crowdguard",
        "fltrust",
        "fedredefense",
        "fedre",
        "rofl",
        "elsa",
        "lockdown",
    }
    if str(params.get("defence_method", "")).lower() not in allowed:
        raise ValueError("Unsupported defence_method in curated release")
    # Root-relative runtime assets work even when the launcher is called elsewhere.
    os.chdir(Path(__file__).resolve().parent)
    params.setdefault("output_root", "outputs")
    params.setdefault("result_csv_path", str(Path(params["output_root"]) / "results.csv"))
    seed(int(params["seed"]))
    started = time.time()
    resolve_privacy_accounting(params)
    handle = Handle(
        current_time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        params=params,
        name=params.get("name", params["type"]),
    )
    handle.load_data()
    handle.create_model()
    with (Path(handle.folder_path) / "params.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(params, file)
    metrics = algorithm.FedAvg(handle)
    total = time.time() - started
    write_csv_result(metrics, params, total_time=total, filepath=params["result_csv_path"])
    write_run_artifacts(handle, metrics, total_time=total)
    logger.info("Completed in %.2f seconds; outputs: %s", total, handle.folder_path)


if __name__ == "__main__":
    run()
