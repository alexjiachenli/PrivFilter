#!/usr/bin/env python3
"""Create one party's private noise seed; seed bytes are never printed."""

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Recipe ID from scripts/reproduce.py --list")
    parser.add_argument("--party", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / ".private/noise_seeds"
    )
    args = parser.parse_args()
    if not args.run or Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("--run must be a simple recipe ID")
    directory = args.root.expanduser() / args.run / f"party_{args.party}"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / "master_seed.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(os.urandom(32))
    print(f"Created party {args.party} seed at {path}. Keep it private.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
