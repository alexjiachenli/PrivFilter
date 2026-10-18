"""Audit rejected-client membership from paired-release residuals.

The score is the residual's inner product with a candidate direction, divided
by that direction's norm. Oracle, cohort-mean and random direction models
define the attacker knowledge. Report single-round AUC and budget-level AUC
from resampled independent round statistics.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attack_differential import (  # noqa: E402
    B_DP,
    CLIP,
    SIGMA_NOMINAL,
    as_client_models,
    flat,
    params,
    seeds,
)
from mpc_dp_release import MpcDpReleaseService  # noqa: E402

ATTACKERS = ["oracle", "cohort_mean", "random"]


def auc(pos, neg):
    """Rank-based AUC: P(a positive scores above a negative)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks over ties
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        tie = vals == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float(
        (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    )


def gdp_auc(mu):
    return 0.5 * (1.0 + math.erf(mu / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument(
        "--arm", default="partition", choices=["partition", "partition_collude", "independent"]
    )
    ap.add_argument("--bootstrap", type=int, default=4000)
    args = ap.parse_args()

    with open(os.path.join(args.updates, "index.json")) as fh:
        index = json.load(fh)
    keys, d = index["float_keys"], index["d"]

    sigma_all_honest = SIGMA_NOMINAL * math.sqrt(1.5)
    sigma_eff = {
        "partition": sigma_all_honest,
        "partition_collude": SIGMA_NOMINAL,
        "independent": SIGMA_NOMINAL * math.sqrt(2.0) * math.sqrt(1.5),
    }[args.arm]
    pairing = "independent" if args.arm == "independent" else "partition_correlated"

    tmp = tempfile.mkdtemp(prefix="attackA2_")
    trials = []  # one per (round, victim, candidate)
    residual_sigmas = []
    cohort_cos = []  # how alike the cohort's updates are

    for rnd in range(1, args.rounds + 1):
        path = os.path.join(args.updates, f"round_{rnd:03d}.pt")
        if not os.path.exists(path):
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        deltas, gcpu = blob["clipped_deltas"], blob["global_state"]
        n = len(deltas)
        flat_deltas = [flat(dd, keys) for dd in deltas]
        models = as_client_models(deltas, gcpu, keys)
        base = flat(gcpu, keys)

        seeds(tmp, salt=rnd)
        svc = MpcDpReleaseService(params(tmp, pairing, f"a2_r{rnd}"))
        try:
            pre = svc.make_fast_aggregate_reference_release(
                [{k: v.clone() for k, v in m.items()} for m in models],
                {k: v.clone() for k, v in gcpu.items()},
                list(range(n)),
                round_id=1,
            )
            s_all = (flat(pre, keys) - base) * n

            # Accepting every client produces a noise-only residual for the absent-update control.
            post_none = svc.make_fast_aggregate_reference_release(
                [{k: v.clone() for k, v in m.items()} for m in models],
                {k: v.clone() for k, v in gcpu.items()},
                list(range(n)),
                round_id=1,
                accept_mask=[1] * n,
                release_role="filtered_post_detection",
            )
            noise_only = s_all - (flat(post_none, keys) - base) * n
            if args.arm == "partition_collude":
                # Apply the same collusion noise reduction to control and victim residuals.
                noise_only = noise_only * (SIGMA_NOMINAL / sigma_all_honest)
            for cand in range(n):
                probe = flat_deltas[cand]
                trials.append(
                    {
                        "round": rnd,
                        "victim": -1,
                        "candidate": cand,
                        "is_rejected": 0,
                        "absent": 1,
                        "attacker": "oracle",
                        "statistic": float(torch.dot(noise_only, probe) / probe.norm()),
                    }
                )

            # Measure pairwise update similarity within the cohort.
            for i in range(n):
                for j in range(i + 1, n):
                    cohort_cos.append(
                        float(
                            torch.dot(flat_deltas[i], flat_deltas[j])
                            / (flat_deltas[i].norm() * flat_deltas[j].norm())
                        )
                    )

            # Evaluate every client as the rejected victim.
            for victim in range(n):
                mask = [1] * n
                mask[victim] = 0
                post = svc.make_fast_aggregate_reference_release(
                    [{k: v.clone() for k, v in m.items()} for m in models],
                    {k: v.clone() for k, v in gcpu.items()},
                    list(range(n)),
                    round_id=1,
                    accept_mask=mask,
                    release_role="filtered_post_detection",
                )
                s_acc = (flat(post, keys) - base) * int(sum(mask))
                residual = s_all - s_acc
                if args.arm == "partition_collude":
                    # One party's noise share is known to the colluding
                    # coordinator; removing it leaves the calibrated sigma.
                    residual = flat_deltas[victim] + (residual - flat_deltas[victim]) * (
                        SIGMA_NOMINAL / sigma_all_honest
                    )
                residual_sigmas.append(float((residual - flat_deltas[victim]).std()))

                cohort_mean = torch.stack(flat_deltas).mean(0)
                gen = torch.Generator().manual_seed(1000 * rnd + victim)
                rand_dir = torch.randn(d, generator=gen, dtype=torch.float64)

                for cand in range(n):
                    probes = {
                        "oracle": flat_deltas[cand],
                        "cohort_mean": cohort_mean,
                        "random": rand_dir,
                    }
                    for name, probe in probes.items():
                        t = float(torch.dot(residual, probe) / probe.norm())
                        trials.append(
                            {
                                "round": rnd,
                                "victim": victim,
                                "candidate": cand,
                                "is_rejected": int(cand == victim),
                                "attacker": name,
                                "statistic": t,
                            }
                        )
                print(f"[A2] r{rnd:02d} victim {victim} done", flush=True)
        finally:
            svc.cleanup()

    # Single-round AUC.
    results = {}
    rng = np.random.RandomState(0)
    for name in ATTACKERS:
        sub = [t for t in trials if t["attacker"] == name and not t.get("absent")]
        pos = [t["statistic"] for t in sub if t["is_rejected"]]
        neg = [t["statistic"] for t in sub if not t["is_rejected"]]
        one = auc(pos, neg)

        # Budget-level AUC from resampled independent rounds.
        boot = []
        for _ in range(args.bootstrap):
            p = rng.choice(pos, size=B_DP, replace=True).sum()
            q = rng.choice(neg, size=B_DP, replace=True).sum()
            boot.append((p, q))
        bp = [x for x, _ in boot]
        bq = [y for _, y in boot]
        results[name] = {
            "n_positive": len(pos),
            "n_negative": len(neg),
            "auc_one_round_empirical": one,
            "auc_full_budget_empirical": auc(bp, bq),
            "mean_statistic_positive": float(np.mean(pos)),
            "mean_statistic_negative": float(np.mean(neg)),
            "sd_statistic_negative": float(np.std(neg)),
        }
        if name == "oracle":
            # Score victim probes against the absent-update control residual.
            absent = [t["statistic"] for t in trials if t.get("absent")]
            ab = []
            for _ in range(args.bootstrap):
                ab.append(
                    (
                        rng.choice(pos, size=B_DP, replace=True).sum(),
                        rng.choice(absent, size=B_DP, replace=True).sum(),
                    )
                )
            results[name]["adjacency"] = {
                "n_absent": len(absent),
                "auc_one_round_empirical": auc(pos, absent),
                "auc_full_budget_empirical": auc([x for x, _ in ab], [y for _, y in ab]),
                "mean_statistic_absent": float(np.mean(absent)),
                "sd_statistic_absent": float(np.std(absent)),
                "why": "positives are 'this client's update is in the residual'; "
                "negatives are a residual with no rejected update, so the "
                "test is presence vs absence -- what DP bounds",
            }

    mu_round = CLIP / sigma_eff
    summary = {
        "experiment": "A2_membership_audit",
        "arm": args.arm,
        "sigma_effective": sigma_eff,
        "sigma_nominal": SIGMA_NOMINAL,
        "residual_noise_sd_measured": float(np.mean(residual_sigmas)),
        "cohort_update_cosine_mean": float(np.mean(cohort_cos)) if cohort_cos else None,
        "cohort_update_cosine_sd": float(np.std(cohort_cos)) if cohort_cos else None,
        "clip_norm": CLIP,
        "B_DP": B_DP,
        "d": d,
        "analytic_bound": {
            "mu_one_round": mu_round,
            "mu_full_budget": mu_round * math.sqrt(B_DP),
            "auc_one_round": gdp_auc(mu_round),
            "auc_full_budget": gdp_auc(mu_round * math.sqrt(B_DP)),
        },
        "attackers": results,
        "trials": len(trials),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== A2 membership audit ({args.arm}) ===")
    print(
        f"residual noise sd  : measured {summary['residual_noise_sd_measured']:.4f}  "
        f"expected {sigma_eff:.4f}"
    )
    print(
        f"cohort update cos  : {summary['cohort_update_cosine_mean']:.4f} +/- "
        f"{summary['cohort_update_cosine_sd']:.4f}  "
        f"(how alike the candidates are)"
    )
    print(
        f"analytic bound     : AUC(1) {gdp_auc(mu_round):.4f}   "
        f"AUC({B_DP}) {gdp_auc(mu_round * math.sqrt(B_DP)):.4f}"
    )
    adj = results["oracle"].get("adjacency", {})
    print(f"\npresence vs absence -- what epsilon bounds:")
    print(
        f"  {'oracle':12s} empirical AUC(1) {adj.get('auc_one_round_empirical', float('nan')):.4f}   "
        f"AUC({B_DP}) {adj.get('auc_full_budget_empirical', float('nan')):.4f}"
    )
    print(f"\nwhich cohort member was rejected -- the practical question:")
    for name in ATTACKERS:
        r = results[name]
        print(
            f"  {name:12s} empirical AUC(1) {r['auc_one_round_empirical']:.4f}   "
            f"AUC({B_DP}) {r['auc_full_budget_empirical']:.4f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
