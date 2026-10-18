#!/usr/bin/env python3
"""Provision one party's private noise seed using OS randomness.

The file is created with user-only permissions.
"""

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", help="Directory that will contain master_seed.bin")
    parser.add_argument("--run-name", help="Run name for default private seed layout")
    parser.add_argument("--party", type=int, help="Party id for default private seed layout")
    parser.add_argument(
        "--root",
        default=".private/noise_seeds",
        help="Root used with --run-name/--party",
    )
    parser.add_argument("--bytes", type=int, default=32, help="Seed length in bytes")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing seed")
    args = parser.parse_args()

    if args.seed_dir:
        seed_dir = Path(args.seed_dir)
    else:
        if args.run_name is None or args.party is None:
            parser.error("provide either --seed-dir or both --run-name and --party")
        seed_dir = Path(args.root) / args.run_name / f"party_{int(args.party)}"

    if args.bytes < 32:
        raise SystemExit("--bytes must be at least 32")

    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_path = seed_dir / "master_seed.bin"
    if seed_path.exists() and not args.force:
        raise SystemExit(f"seed already exists, refusing to overwrite without --force: {seed_path}")

    data = os.urandom(int(args.bytes))
    with seed_path.open("wb") as handle:
        handle.write(data)
    os.chmod(seed_path, 0o600)
    print(f"created private seed file: {seed_path}")
    print("seed bytes were not printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
