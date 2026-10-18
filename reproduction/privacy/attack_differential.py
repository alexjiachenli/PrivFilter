"""Evaluate rejected-update reconstruction and distinguishing from release pairs.

Arms compare plaintext access, shared noise, independent noise and partition-
correlated noise, with an optional colluding party. The coordinator residual
is converted to sum coordinates before comparison with the rejected update.
"""

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from mpc_dp_release import MpcDpReleaseService  # noqa: E402

SCALE = 65536
SIGMA_NOMINAL = 2.4023136318435663
CLIP = 1.0
B_DP = 20  # Per-client composition budget.

ARMS = ["plaintext", "shared_noise", "independent", "partition", "partition_collude"]


def seeds(directory, salt=0):
    """Deterministic simulator audit streams, never production party secrets."""
    for pid in range(3):
        party = os.path.join(directory, f"party_{pid}")
        os.makedirs(party, exist_ok=True)
        with open(os.path.join(party, "master_seed.bin"), "wb") as fh:
            fh.write(bytes([(pid * 37 + salt + 11) % 251]) * 64)


def params(seed_dir, pairing, run_name):
    return {
        "name": run_name,
        "dp_release_source": "mpc_aggregate_ref_dp_fast",
        "mpc_dp_backend": "simulated_mpc",
        "mpc_dp_protocol": "replicated-field",
        "mpc_dp_ring_bits": 64,
        "mpc_dp_output_mode": "aggregate_reference",
        "mpc_dp_clip_mode": "local_reference_clip_then_mpc_sum",
        "mpc_dp_noise_mode": "party_local_distributed_dgauss",
        "mpc_dp_noise_share_mode": "party_local_seeded_stream",
        "mpc_dp_fixed_point_scale": SCALE,
        "mpc_dp_clip_norm": CLIP,
        "mpc_dp_adjacency": "add_remove",
        "mpc_dp_noise_multiplier": SIGMA_NOMINAL,
        "mpc_dp_release_pairing": pairing,
        "mpc_dp_noise_generation": {"seed_dir": os.path.join(seed_dir, "party_{party}")},
        "allow_zero_noise_mpc_dp_release": False,
    }


def flat(state, keys):
    return torch.cat([state[k].reshape(-1).double() for k in keys])


def as_client_models(deltas, gcpu, keys):
    """Add update vectors to the base model for the model-based release API."""
    out = []
    for delta in deltas:
        model = {k: v.clone() for k, v in gcpu.items()}
        for k in keys:
            model[k] = gcpu[k].float() + delta[k].float()
        out.append(model)
    return out


def coordinator_difference(arm, deltas, gcpu, keys, mask, seed_dir, run_name):
    """The residual the coordinator holds after subtracting its two releases.

    Returns (recovered, sigma_eff) where `recovered` is its estimate of the
    rejected clients' summed update.
    """
    n = len(deltas)
    n_acc = int(sum(mask))
    truth = torch.stack([flat(deltas[i], keys) for i in range(n) if not mask[i]]).sum(0)

    if arm == "plaintext":
        # No release at all: the coordinator is handed the updates.
        return truth.clone(), 0.0

    pairing = "partition_correlated" if arm.startswith("partition") else "independent"
    # Reusing the release role selects the same noise stream.
    post_role = "unfiltered_pre_detection" if arm == "shared_noise" else "filtered_post_detection"
    models = as_client_models(deltas, gcpu, keys)
    svc = MpcDpReleaseService(params(seed_dir, pairing, run_name))
    try:
        pre = svc.make_fast_aggregate_reference_release(
            [{k: v.clone() for k, v in m.items()} for m in models],
            {k: v.clone() for k, v in gcpu.items()},
            list(range(n)),
            round_id=1,
        )
        post = svc.make_fast_aggregate_reference_release(
            [{k: v.clone() for k, v in m.items()} for m in models],
            {k: v.clone() for k, v in gcpu.items()},
            list(range(n)),
            round_id=1,
            accept_mask=mask,
            release_role=post_role,
        )
    finally:
        svc.cleanup()

    # Remove the base model and averaging denominator before differencing.
    base = flat(gcpu, keys)
    s_all = (flat(pre, keys) - base) * n
    s_acc = (flat(post, keys) - base) * n_acc

    recovered = s_all - s_acc
    if arm == "shared_noise":
        # A shared noise stream cancels when the released sums are subtracted.
        return recovered, 0.0
    # Three honest shares yield sigma*sqrt(3/2); any two yield sigma.
    sigma_all_honest = SIGMA_NOMINAL * math.sqrt(1.5)
    if arm == "partition":
        return recovered, sigma_all_honest
    if arm == "partition_collude":
        # Model one known party share by reducing residual noise to sigma.
        resid = recovered - truth
        return truth + resid * (SIGMA_NOMINAL / sigma_all_honest), SIGMA_NOMINAL
    if arm == "independent":
        return recovered, SIGMA_NOMINAL * math.sqrt(2.0) * math.sqrt(1.5)
    raise ValueError(arm)


def gdp_auc(mu):
    """AUC of the optimal test between N(0,1) and N(mu,1)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(mu / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    with open(os.path.join(args.updates, "index.json")) as fh:
        index = json.load(fh)
    keys = index["float_keys"]
    d = index["d"]

    tmp = tempfile.mkdtemp(prefix="attackA_")
    rows = []

    for rnd in range(1, args.rounds + 1):
        path = os.path.join(args.updates, f"round_{rnd:03d}.pt")
        if not os.path.exists(path):
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        deltas = blob["clipped_deltas"]
        gcpu = blob["global_state"]
        n = len(deltas)

        # Rotate the single rejected client across rounds.
        victim = (rnd - 1) % n
        mask = [1] * n
        mask[victim] = 0
        truth = flat(deltas[victim], keys)

        seeds(tmp, salt=rnd)
        for arm in ARMS:
            rec, sigma_eff = coordinator_difference(
                arm, deltas, gcpu, keys, mask, tmp, f"attackA_r{rnd}_{arm}"
            )
            err = float((rec - truth).norm())
            rel = err / float(truth.norm())
            cos = float(torch.dot(rec, truth) / (rec.norm() * truth.norm() + 1e-30))

            # Correlate the residual with a candidate update at each budget.
            mu_round = float(truth.norm()) / sigma_eff if sigma_eff > 0 else float("inf")
            mu_budget = mu_round * math.sqrt(B_DP)
            rows.append(
                {
                    "round": rnd,
                    "arm": arm,
                    "victim_index": victim,
                    "victim_client": blob["sampled"][victim],
                    "victim_malicious": blob["malicious"][victim],
                    "d": d,
                    "truth_norm": float(truth.norm()),
                    "residual_norm": err,
                    "relative_error": rel,
                    "cosine": cos,
                    "sigma_effective": sigma_eff,
                    "mu_one_round": mu_round,
                    "mu_full_budget": mu_budget,
                    "auc_one_round": gdp_auc(mu_round) if sigma_eff > 0 else 1.0,
                    "auc_full_budget": gdp_auc(mu_budget) if sigma_eff > 0 else 1.0,
                }
            )
            print(
                f"[A] r{rnd:02d} {arm:18s} rel_err {rel:10.2f}  cos {cos:+.4f}  "
                f"sigma {sigma_eff:.4f}  AUC(1) {rows[-1]['auc_one_round']:.4f}  "
                f"AUC({B_DP}) {rows[-1]['auc_full_budget']:.4f}",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "experiment": "A_differential_reconstruction",
                "sigma_nominal": SIGMA_NOMINAL,
                "sigma_all_honest": SIGMA_NOMINAL * math.sqrt(1.5),
                "clip_norm": CLIP,
                "B_DP": B_DP,
                "d": d,
                "rows": rows,
            },
            fh,
            indent=2,
        )
    print(f"[A] wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
