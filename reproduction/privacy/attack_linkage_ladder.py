"""Evaluate cross-round client linkage and rejection disclosure.

Positive linkage pairs share a client across different rounds; negative pairs
use different clients. Similarity uses each disclosure mode's visible fields.
The count-only linkage AUC is derived as 0.5 because it has no per-client
observation to pair.
"""

import argparse
import glob
import itertools
import json
import os
import random

import numpy as np

# What the coordinator holds per client per round, at each rung of the ladder.
RUNGS = {
    "scores_opened": ["p_target", "logit_margin", "target_logit_mean", "max_other_logit_mean"],
    "scalar_score": ["score_margin"],
    "bits_opened": ["predicted_reject"],
}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    for v in np.unique(vals):
        tie = vals == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float(
        (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    )


def load_records(paths):
    out = []
    for path in paths:
        with open(path) as fh:
            blob = json.load(fh)
        recs = blob.get("metrics", {}).get("detection_score_records", [])
        for r in recs:
            out.append({**r, "_source_run": os.path.realpath(path)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-glob", required=True, help="glob over completed runs' results/raw/*.json"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.raw_glob))
    if not paths:
        raise SystemExit(f"no runs matched {args.raw_glob}")
    records = load_records(paths)
    print(f"[C] {len(records)} client-round records from {len(paths)} runs")

    # Construct linkage pairs within each training run.
    by_run = {}
    for r in records:
        key = (r["_source_run"], int(r.get("seed", 0)))
        by_run.setdefault(key, []).append(r)

    rng = random.Random(args.seed)
    results = {}

    for rung, fields in RUNGS.items():
        pos_scores, neg_scores = [], []
        for recs in by_run.values():
            # Normalise each field so no single one dominates the distance.
            arr = np.array([[float(r[f]) for f in fields] for r in recs], float)
            sd = arr.std(0)
            sd[sd == 0] = 1.0
            arr = (arr - arr.mean(0)) / sd
            ids = [int(r["client_id"]) for r in recs]
            rounds = [int(r["round"]) for r in recs]

            by_client = {}
            for i, cid in enumerate(ids):
                by_client.setdefault(cid, []).append(i)

            # positives: same client, different rounds
            same = [c for c, idx in by_client.items() if len(idx) > 1]
            n_target = args.pairs // (2 * len(by_run))
            for _ in range(n_target):
                if not same:
                    break
                c = rng.choice(same)
                i, j = rng.sample(by_client[c], 2)
                if rounds[i] == rounds[j]:
                    continue
                pos_scores.append(-float(np.linalg.norm(arr[i] - arr[j])))
            # negatives: different clients, different rounds
            for _ in range(n_target):
                i, j = rng.randrange(len(recs)), rng.randrange(len(recs))
                if ids[i] == ids[j] or rounds[i] == rounds[j]:
                    continue
                neg_scores.append(-float(np.linalg.norm(arr[i] - arr[j])))

        results[rung] = {
            "fields": fields,
            "n_positive": len(pos_scores),
            "n_negative": len(neg_scores),
            "linkage_auc": auc(pos_scores, neg_scores),
        }
        print(
            f"[C] {rung:16s} linkage AUC {results[rung]['linkage_auc']:.4f}   "
            f"({len(pos_scores)} pos / {len(neg_scores)} neg)"
        )

    # Count-only observations give a derived linkage AUC of 0.5.
    results["count_only_ours"] = {
        "fields": [],
        "n_positive": 0,
        "n_negative": 0,
        "linkage_auc": 0.5,
        "derived": True,
        "why": "no per-client object is opened, so no two observations can be "
        "paired; the adversary is reduced to guessing",
    }
    print(
        f"[C] {'count_only_ours':16s} linkage AUC 0.5000   (derived: nothing per-client is opened)"
    )

    # Evaluate rejection disclosure separately from cross-round linkage.
    truth = [int(bool(r["predicted_reject"])) for r in records]
    n_rej, n_acc_ = sum(truth), len(truth) - sum(truth)
    disclosure = {}
    if n_rej and n_acc_:
        # Use scores increasing with rejection: score_margin > 0.
        channels = {
            "scores_opened": [float(r["score_margin"]) for r in records],
            "scalar_score": [float(r["score_margin"]) for r in records],
            "bits_opened": [float(bool(r["predicted_reject"])) for r in records],
            # Use the rejected fraction as the count-only per-client score.
            "count_only_ours": None,
        }
        keyof = [
            (r["_source_run"], int(r.get("seed", 0)), int(r["round"]))
            for r in records
        ]
        per_round = {}
        for k, t in zip(keyof, truth):
            per_round.setdefault(k, []).append(t)
        rates = {k: float(np.mean(v)) for k, v in per_round.items()}
        channels["count_only_ours"] = [rates[k] for k in keyof]

        for name, scores in channels.items():
            pos = [s for s, t in zip(scores, truth) if t]
            neg = [s for s, t in zip(scores, truth) if not t]
            # Within-round AUC controls for changing rejection rates across rounds.
            num = den = 0.0
            for k in per_round:
                p = [s for s, t, kk in zip(scores, truth, keyof) if kk == k and t]
                q = [s for s, t, kk in zip(scores, truth, keyof) if kk == k and not t]
                if p and q:
                    num += auc(p, q) * len(p) * len(q)
                    den += len(p) * len(q)
            disclosure[name] = {
                "rejection_disclosure_auc_pooled": auc(pos, neg),
                "rejection_disclosure_auc_within_round": (num / den) if den else float("nan"),
                "n_rejected": len(pos),
                "n_accepted": len(neg),
            }
            print(
                f"[C] {name:16s} rejection-disclosure AUC  pooled "
                f"{disclosure[name]['rejection_disclosure_auc_pooled']:.4f}   "
                f"within-round "
                f"{disclosure[name]['rejection_disclosure_auc_within_round']:.4f}"
            )
    for name, val in disclosure.items():
        results.setdefault(name, {}).update(val)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "experiment": "C_linkage_ladder",
                "source_runs": paths,
                "records": len(records),
                "rungs": results,
            },
            fh,
            indent=2,
        )
    print(f"[C] wrote {args.out}")


if __name__ == "__main__":
    main()
