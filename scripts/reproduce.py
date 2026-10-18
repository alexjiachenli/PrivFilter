#!/usr/bin/env python3
"""List, materialize and execute experiment configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def catalog() -> dict:
    return json.loads((ROOT / "reproduction/manifest.json").read_text())


def parameters(recipe: dict) -> dict:
    result = json.loads((ROOT / recipe["config"]).read_text())
    for key in recipe.get("remove_keys", []):
        result.pop(key, None)
    result.update(copy.deepcopy(recipe["overrides"]))
    digest = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != recipe["normalized_source_sha256"]:
        raise ValueError(f"Configuration differs from normalized source: {recipe['id']}")
    return result


def runtime_parameters(recipe: dict) -> dict:
    result = parameters(recipe)
    result.update(copy.deepcopy(recipe.get("runtime_overrides", {})))
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git_state() -> dict:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "-z"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "commit": result.stdout.strip() if result.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
        "status_entries": [entry for entry in status.stdout.split("\0") if entry]
        if status.returncode == 0
        else None,
    }


def local_test_seeds(seed_dir: Path) -> None:
    for party in range(3):
        directory = seed_dir / f"party_{party}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "master_seed.bin"
        with path.open("xb") as handle:
            handle.write(os.urandom(32))
        path.chmod(0o600)


def validate_recipe(recipe: dict) -> None:
    """Run the runtime's configuration guards without loading data or models."""
    sys.path.insert(0, str(ROOT))
    import algorithm
    import main as runtime_main

    p = runtime_parameters(recipe)
    if p.get("type") not in {"CIFAR10", "CIFAR100", "GTSRB"}:
        raise ValueError("Unsupported dataset")
    algorithm._read_ldp_config(SimpleNamespace(params=p))
    runtime_main.resolve_privacy_accounting(p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--run", help="Exact recipe ID from --list")
    selector.add_argument("--suite", help="Select a suite from the catalog")
    parser.add_argument("--list", action="store_true", help="Show matching recipe IDs")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration guards without data loading or training",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute the selected recipes sequentially"
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Write runnable params.json files without training",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed-root", type=Path, default=ROOT / ".private/noise_seeds")
    parser.add_argument(
        "--local-test-seeds",
        action="store_true",
        help="Generate all parties' seeds locally for simulator diagnostics only",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override global rounds for a smoke test; changes the paper recipe",
    )
    parser.add_argument(
        "--data-root", type=Path, help="Dataset cache (if omitted, use the runtime default)"
    )
    args = parser.parse_args()
    if args.execute and args.materialize_only:
        parser.error("choose --execute or --materialize-only")
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds must be positive")
    recipes = catalog()["runs"]
    if args.run:
        recipes = [run for run in recipes if run["id"] == args.run]
    elif args.suite:
        recipes = [run for run in recipes if args.suite in run["suites"]]
    if not recipes:
        parser.error("no recipe matches the selector")
    if args.validate:
        failures = []
        for recipe in recipes:
            try:
                validate_recipe(recipe)
            except Exception as error:
                failures.append((recipe["id"], str(error)))
        for run_id, error in failures:
            print(f"FAIL {run_id}: {error}", file=sys.stderr)
        print(f"Configuration validation: {len(recipes) - len(failures)}/{len(recipes)} passed")
        return 1 if failures else 0
    if args.list:
        for recipe in recipes:
            d = parameters(recipe)
            print(
                f"{recipe['id']}\t{','.join(recipe['suites'])}\t{d['type']}\t{d.get('defence_method')}\t{recipe['execution']['detector']}"
            )
        return 0
    if not args.run and not args.suite:
        parser.error("select --run or --suite, or use --list")
    if not args.execute and not args.materialize_only:
        for recipe in recipes:
            print(
                json.dumps(
                    {
                        "id": recipe["id"],
                        "execution": recipe["execution"],
                        "source": recipe["source"],
                    }
                )
            )
        print(
            "Plan only. Add --execute to train or --materialize-only to write configuration files."
        )
        return 0

    # Check the entire selection before starting a potentially long suite.
    for recipe in recipes:
        if args.execute:
            validate_recipe(recipe)
        p = runtime_parameters(recipe)
        real = (
            p.get("priv_filterfl_mpc_mode") == "cryptgpu" or p.get("mpc_dp_backend") == "cryptgpu"
        )
        if args.local_test_seeds and real:
            parser.error(
                "--local-test-seeds applies only to simulator diagnostics; provision real-backend party seeds separately"
            )
        destination = args.output_dir.expanduser().resolve() / recipe["id"]
        if destination.exists():
            parser.error(
                f"output directory already exists: {destination}; choose another --output-dir"
            )
        if args.execute and p.get("mpc_dp_noise_generation") and not args.local_test_seeds:
            seed_dir = args.seed_root.expanduser().resolve() / recipe["id"]
            for party in range(3):
                if not (seed_dir / f"party_{party}/master_seed.bin").is_file():
                    parser.error(
                        f"missing party_{party} seed for {recipe['id']}; use scripts/provision_noise_seed.py or --local-test-seeds for simulator diagnostics"
                    )
        if args.execute and p.get("mpc_dp_noise_generation") and args.local_test_seeds:
            seed_dir = args.seed_root.expanduser().resolve() / recipe["id"]
            if any((seed_dir / f"party_{party}/master_seed.bin").exists() for party in range(3)):
                parser.error(
                    f"test seeds already exist for {recipe['id']}; select a new --seed-root"
                )

    for recipe in recipes:
        p = runtime_parameters(recipe)
        run_dir = args.output_dir.expanduser().resolve() / recipe["id"]
        seed_dir = args.seed_root.expanduser().resolve() / recipe["id"]
        p.update(
            {
                "name": recipe["id"],
                "run_id": recipe["id"],
                "experiment_name": recipe["id"],
                "environment_name": p["type"],
                "output_root": str(run_dir / "artifacts"),
                "experiment_summary_path": str(run_dir / "summary.json"),
                "result_csv_path": str(run_dir / "final_metrics.csv"),
                "participation_schedule_path": str(run_dir / "participation_schedule.json"),
                "participation_counts_path": str(run_dir / "participation_counts.json"),
                "privacy_calibration_path": str(run_dir / "privacy_calibration.json"),
            }
        )
        if args.rounds is not None:
            p["epochs"] = args.rounds
        if args.data_root is not None:
            p["data_root"] = str(args.data_root.expanduser().resolve())
        noise = p.get("mpc_dp_noise_generation")
        if noise:
            noise["seed_dir"] = str(seed_dir / "party_{party}")
            noise["domain_context"] = recipe["id"]
            if args.execute and args.local_test_seeds:
                local_test_seeds(seed_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        params_path = run_dir / "params.json"
        write_json(params_path, p)
        command = [sys.executable, str(ROOT / "main.py"), "--params", str(params_path)]
        receipt = {
            "recipe": recipe["id"],
            "configuration_source": recipe["source"],
            "repository_state": git_state(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": command,
            "params_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
            "smoke_test_round_override": args.rounds,
            "all_party_seeds_generated_locally": bool(args.local_test_seeds),
            "execution": recipe["execution"],
        }
        write_json(run_dir / "run_receipt.json", receipt)
        print(f"{recipe['id']}: {params_path}", flush=True)
        if args.execute:
            started = time.time()
            result = subprocess.run(command, cwd=ROOT, check=False)
            receipt.update(
                {"return_code": result.returncode, "elapsed_seconds": time.time() - started}
            )
            write_json(run_dir / "run_receipt.json", receipt)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
