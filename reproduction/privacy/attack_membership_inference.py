"""Loss-threshold membership inference on complete model views.

The local model precedes clipping; the clean aggregate includes clipping and
averaging; the DP release also includes quantization and noise. Nonfinite
losses trigger BN re-estimation on selected nonmembers before rescoring.
Members belong to the victim client; nonmembers come from other CIFAR-10
training samples. Comparisons measure the combined view transformations.
"""

import argparse
import copy
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attack_differential import as_client_models, params, seeds  # noqa: E402
from mpc_dp_release import MpcDpReleaseService  # noqa: E402

RUNGS = ["local_model", "clean_aggregate", "dp_release"]


def auc(pos, neg):
    """AUC with members as the positive class."""
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


@torch.no_grad()
def refresh_bn(model, dataset, indices, device, batch=256, passes=2):
    """Re-estimate BatchNorm running statistics on attacker-held data."""
    model.to(device).train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None  # cumulative average over the passes
    for _ in range(passes):
        for start in range(0, len(indices), batch):
            chunk = indices[start : start + batch]
            model(torch.stack([dataset[i][0] for i in chunk]).to(device))
    model.eval()
    return model


@torch.no_grad()
def losses(model, dataset, indices, device, batch=256):
    """Per-sample cross-entropy: the membership signal itself."""
    model.eval().to(device)
    out = []
    for start in range(0, len(indices), batch):
        chunk = indices[start : start + batch]
        xs = torch.stack([dataset[i][0] for i in chunk]).to(device)
        ys = torch.tensor([dataset[i][1] for i in chunk], device=device)
        logits = model(xs)
        out.extend(F.cross_entropy(logits, ys, reduction="none").cpu().tolist())
    return out


@torch.no_grad()
def accuracy(model, dataset, indices, device, batch=256):
    """Compute classification accuracy on the selected samples."""
    model.eval().to(device)
    right = 0
    for start in range(0, len(indices), batch):
        chunk = indices[start : start + batch]
        xs = torch.stack([dataset[i][0] for i in chunk]).to(device)
        ys = torch.tensor([dataset[i][1] for i in chunk], device=device)
        pred = model(xs).argmax(1)
        right += int((pred == ys).sum())
    return right / max(1, len(indices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", required=True)
    ap.add_argument("--params", default=os.path.join(REPO, "configs", "privfilter_cifar10.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", default="4,8,12", help="rounds whose per-client models were kept")
    ap.add_argument("--n-nonmembers", type=int, default=2000)
    args = ap.parse_args()
    args.out = os.path.abspath(os.path.expanduser(args.out))
    args.params = os.path.abspath(os.path.expanduser(args.params))
    args.updates = os.path.abspath(os.path.expanduser(args.updates))
    os.chdir(REPO)

    with open(os.path.join(args.updates, "index.json")) as fh:
        index = json.load(fh)
    keys = index["float_keys"]
    partitions = {int(k): v for k, v in index["partitions"].items()}

    with open(args.params) as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(
        {
            "epochs": 1,
            "seed": index["seed"],
            "dirichlet_alpha": index["alpha"],
            "enable_ldp": False,
            "name": "privacy_attack_mia",
            "environment_name": "privacy_attack_mia",
            "output_root": os.path.join(os.path.dirname(args.out), "mia_artifacts"),
        }
    )

    import main as fl_main  # noqa: E402
    from handle import Handle  # noqa: E402

    fl_main.logger.setLevel("ERROR")
    fl_main.seed(index["seed"])
    handle = Handle(current_time="mia", params=cfg, name="privacy_attack_mia")
    handle.load_data()
    handle.create_model()
    device = next(handle.model.parameters()).device
    dataset = handle.train_dataset

    tmp = tempfile.mkdtemp(prefix="attackB_")
    rows = []

    for rnd in [int(x) for x in args.rounds.split(",")]:
        path = os.path.join(args.updates, f"round_{rnd:03d}.pt")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "local_states" not in blob:
            print(f"[B] round {rnd} has no per-client models, skipping")
            continue
        deltas = blob["clipped_deltas"]
        gcpu = blob["global_state"]
        sampled = blob["sampled"]
        n = len(deltas)

        # Exclude every cohort member when selecting nonmember samples.
        held = set()
        for cid in sampled:
            held.update(partitions.get(int(cid), []))
        pool = [i for i in range(len(dataset)) if i not in held]
        rs = np.random.RandomState(1000 + rnd)
        nonmembers = rs.choice(pool, size=min(args.n_nonmembers, len(pool)), replace=False).tolist()

        # The two aggregates, built once for the round.
        models = as_client_models(deltas, gcpu, keys)
        clean_agg = copy.deepcopy(gcpu)
        for k in keys:
            clean_agg[k] = gcpu[k].float() + torch.stack([d[k] for d in deltas]).mean(0)

        seeds(tmp, salt=rnd)
        svc = MpcDpReleaseService(params(tmp, "partition_correlated", f"b_r{rnd}"))
        try:
            dp_release = svc.make_fast_aggregate_reference_release(
                [{k: v.clone() for k, v in m.items()} for m in models],
                {k: v.clone() for k, v in gcpu.items()},
                list(range(n)),
                round_id=1,
            )
        finally:
            svc.cleanup()

        net = copy.deepcopy(handle.model)
        for vi in range(n):
            victim = int(sampled[vi])
            members = partitions.get(victim, [])
            if len(members) < 50:
                continue
            observed = {
                "local_model": blob["local_states"][vi],
                "clean_aggregate": clean_agg,
                "dp_release": dp_release,
            }
            for rung in RUNGS:
                state = observed[rung]
                net.load_state_dict({k: v.to(device) for k, v in state.items()})
                lm = losses(net, dataset, members, device)
                ln = losses(net, dataset, nonmembers, device)
                repaired = False
                if not (np.isfinite(lm).all() and np.isfinite(ln).all()):
                    # Refresh nonfinite-loss views using attacker-held nonmember data.
                    refresh_bn(net, dataset, nonmembers, device)
                    lm = losses(net, dataset, members, device)
                    ln = losses(net, dataset, nonmembers, device)
                    repaired = True
                finite = np.isfinite(lm).all() and np.isfinite(ln).all()
                # Negate loss so higher scores indicate membership; nonfinite views use AUC 0.5.
                a = auc([-x for x in lm], [-x for x in ln]) if finite else 0.5
                acc = accuracy(net, dataset, nonmembers, device) if finite else float("nan")
                rows.append(
                    {
                        "round": rnd,
                        "victim_client": victim,
                        "victim_malicious": int(blob["malicious"][vi]),
                        "rung": rung,
                        "mia_auc": a,
                        "bn_reestimated": repaired,
                        "usable_object": bool(finite),
                        "accuracy_on_heldout": acc,
                        "n_members": len(members),
                        "n_nonmembers": len(ln),
                        "mean_member_loss": float(np.mean(lm)) if finite else float("nan"),
                        "mean_nonmember_loss": float(np.mean(ln)) if finite else float("nan"),
                    }
                )
                print(
                    f"[B] r{rnd:02d} client {victim:3d} {rung:16s} "
                    f"MIA AUC {a:.4f}  acc {acc:.4f}  "
                    f"loss mem {np.mean(lm):.4f} non {np.mean(ln):.4f}"
                    f"{'  [BN re-estimated]' if repaired else ''}",
                    flush=True,
                )

    summary = {}
    for rung in RUNGS:
        sub = [r for r in rows if r["rung"] == rung]
        vals = [r["mia_auc"] for r in sub]
        accs = [r["accuracy_on_heldout"] for r in sub if np.isfinite(r["accuracy_on_heldout"])]
        summary[rung] = {
            "n": len(vals),
            "mia_auc_mean": float(np.mean(vals)) if vals else float("nan"),
            "mia_auc_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "mia_auc_max": float(np.max(vals)) if vals else float("nan"),
            "accuracy_mean": float(np.mean(accs)) if accs else float("nan"),
            "usable_objects": int(sum(r["usable_object"] for r in sub)),
            "bn_reestimated": int(sum(r["bn_reestimated"] for r in sub)),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {"experiment": "B_membership_inference", "rungs": summary, "rows": rows}, fh, indent=2
        )
    print("\n=== B membership inference ===")
    for rung in RUNGS:
        s = summary[rung]
        print(
            f"  {rung:16s} AUC {s['mia_auc_mean']:.4f} +/- {s['mia_auc_sd']:.4f} "
            f"(max {s['mia_auc_max']:.4f}, n={s['n']})   "
            f"heldout acc {s['accuracy_mean']:.4f}   "
            f"usable {s['usable_objects']}/{s['n']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
