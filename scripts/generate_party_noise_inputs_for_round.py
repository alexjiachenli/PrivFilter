#!/usr/bin/env python3
"""Generate one party's bounded discrete-Gaussian integer noise for one round."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpc_dp_release import BoundedDiscreteGaussianSampler, PartyLocalDiscreteGaussianNoiseProvider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--party-id", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--sigma-int-party", type=int, required=True)
    parser.add_argument("--tail-sigma", type=float, default=8.0)
    parser.add_argument("--cdf-precision-bits", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--domain-context", default="")
    parser.add_argument(
        "--stream-label",
        action="append",
        default=None,
        help=(
            "domain separator for releases that share a round. Repeat it to "
            "write the SUM of several independent streams, which is how the "
            "pre-detection release carries Z_a + Z_b while the post-detection "
            "release carries Z_a alone: their difference stays masked by Z_b, "
            "and each client is charged one composition per round instead of "
            "two. Two releases must never draw the identical single stream -- "
            "subtracting them would then cancel the noise exactly."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "binary"],
        default="text",
        help=(
            "text writes one decimal per line for Input-P<i>-0; binary writes "
            "raw little-endian int64 for Input-Binary-P<i>-0"
        ),
    )
    args = parser.parse_args()

    with open(args.seed_file, "rb") as handle:
        seed = handle.read()
    if len(seed) < 32:
        raise SystemExit(f"seed file is too short: {args.seed_file}")

    sampler = BoundedDiscreteGaussianSampler(
        sigma=args.sigma_int_party,
        tail_sigma=args.tail_sigma,
        cdf_precision_bits=args.cdf_precision_bits,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written = 0
    binary = args.format == "binary"
    stream_labels = args.stream_label if args.stream_label is not None else [""]
    with output.open(
        "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})
    ) as handle:
        while written < args.count:
            n = min(args.chunk_size, args.count - written)
            samples = np.zeros(n, dtype=np.int64)
            for label in stream_labels:
                uniforms = PartyLocalDiscreteGaussianNoiseProvider._uniform_uint64(
                    seed=seed,
                    round_id=args.round_id,
                    party_id=args.party_id,
                    offset=written,
                    count=n,
                    domain_context=args.domain_context.encode("utf-8"),
                    stream_label=label.encode("utf-8"),
                )
                samples = samples + sampler.sample_from_uint64(uniforms).astype(np.int64)
            if binary:
                # Same seeded stream, same values; only the encoding differs,
                # so a text and a binary file for one round are interchangeable.
                handle.write(np.ascontiguousarray(samples, dtype="<i8").tobytes())
            else:
                np.savetxt(handle, samples, fmt="%d")
            written += n

    meta = {
        "party_id": args.party_id,
        "round_id": args.round_id,
        "count": args.count,
        "output": str(output),
        "bytes": os.path.getsize(output),
        "noise_generation_s": time.time() - started,
        "domain_context": args.domain_context,
        "stream_labels": list(stream_labels),
        "format": args.format,
        **sampler.metadata(),
    }
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
