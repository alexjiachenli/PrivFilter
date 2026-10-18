"""Collect client updates and sample memberships for privacy evaluation.

Use the configured data partition and local training, retain clipping metadata
and optionally save local models. These outputs provide attack ground truth.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import main as fl_main  # noqa: E402  (sets up the module-level logger train.py uses)
import train  # noqa: E402
from handle import Handle  # noqa: E402


def clip_delta(delta, clip_norm):
    """Floating-point clipping used to build the audit control aggregate."""
    flat = torch.cat([v.reshape(-1).double() for v in delta.values()])
    norm = float(flat.norm())
    factor = min(1.0, clip_norm / max(norm, 1e-12))
    return {k: (v.double() * factor).float() for k, v in delta.items()}, norm, factor


def float_keys(state):
    return [k for k, v in state.items() if torch.is_tensor(v) and v.dtype.is_floating_point]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(REPO, "configs", "privfilter_cifar10.json"))
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--keep-local-models",
        default="",
        help="comma-separated rounds whose per-client models are kept "
        "for the plaintext rung of the membership ladder; the "
        "models are 20 MB each, so they are not kept by default",
    )
    args = ap.parse_args()
    args.out = os.path.abspath(os.path.expanduser(args.out))
    args.params = os.path.abspath(os.path.expanduser(args.params))
    os.chdir(REPO)
    keep_models = {int(x) for x in args.keep_local_models.split(",") if x.strip()}

    os.makedirs(args.out, exist_ok=True)
    with open(args.params) as fh:
        params = yaml.safe_load(fh)

    # Collect local updates; apply the selected release mechanism in the audit.
    params.update(
        {
            "epochs": args.rounds,
            "seed": args.seed,
            "dirichlet_alpha": args.alpha,
            "enable_ldp": False,
            "is_poison": True,
            "name": "privacy_attack_collect",
            "save_model": False,
            "environment_name": "privacy_attack_collect",
            "output_root": os.path.join(args.out, "collector_artifacts"),
        }
    )
    fl_main.logger.setLevel("ERROR")

    fl_main.seed(args.seed)

    handle = Handle(
        current_time=time.strftime("%b.%d_%H.%M.%S"), params=params, name="privacy_attack_collect"
    )
    handle.load_data()
    handle.create_model()

    clip_norm = float(params.get("mpc_dp_clip_norm", 1.0))
    K = int(params["num_clients"])
    N = int(params["number_of_total_participants"])
    adversarial = set(int(x) for x in handle.adversarial_namelist)

    # Record per-client sample indices as membership ground truth.
    partitions = (
        {int(cid): [int(i) for i in idx] for cid, idx in enumerate(handle.clients_data_num_indices)}
        if hasattr(handle, "clients_data_num_indices")
        else None
    )
    if partitions is None:
        partitions = {}
        for cid, (_pos, loader) in enumerate(handle.train_data):
            sampler = getattr(loader, "sampler", None)
            idx = getattr(sampler, "indices", None)
            partitions[int(cid)] = [int(i) for i in idx] if idx is not None else []

    rng = np.random.RandomState(args.seed)
    rounds = []
    device = next(handle.model.parameters()).device

    for rnd in range(1, args.rounds + 1):
        sampled = sorted(rng.choice(N, size=K, replace=False).tolist())
        global_state = copy.deepcopy(handle.model.state_dict())
        gcpu = {k: v.detach().cpu().clone() for k, v in global_state.items()}
        fkeys = float_keys(gcpu)

        deltas, raw_norms, factors, local_states = [], [], [], []
        for cid in sampled:
            w = train.standard_train(
                rnd,
                handle.clients_data_num[cid],
                cid,
                handle.params,
                copy.deepcopy(handle.model).to(device),
                handle.train_data[cid],
                handle,
                poison=(cid in adversarial),
                enable_ldp=False,
                force_cpu_data=True,
            )
            local = w["clean"] if isinstance(w, dict) and "clean" in w else w
            local = {k: v.detach().cpu().clone() for k, v in local.items()}
            delta = {k: (local[k].float() - gcpu[k].float()) for k in fkeys}
            clipped, norm, factor = clip_delta(delta, clip_norm)
            deltas.append(clipped)
            raw_norms.append(norm)
            factors.append(factor)
            local_states.append(local)

        payload = {
            "round": rnd,
            "sampled": sampled,
            "malicious": [int(c in adversarial) for c in sampled],
            "global_state": gcpu,
            "clipped_deltas": deltas,
            "raw_update_norms": raw_norms,
            "clip_factors": factors,
            "float_keys": fkeys,
        }
        if rnd in keep_models:
            payload["local_states"] = local_states
        torch.save(payload, os.path.join(args.out, f"round_{rnd:03d}.pt"))

        # Advance the global model between collection rounds.
        with torch.no_grad():
            new_state = copy.deepcopy(gcpu)
            for k in fkeys:
                new_state[k] = gcpu[k].float() + torch.stack([d[k] for d in deltas]).mean(0)
            handle.model.load_state_dict({k: v.to(device) for k, v in new_state.items()})

        rounds.append(
            {
                "round": rnd,
                "sampled": sampled,
                "malicious": [int(c in adversarial) for c in sampled],
                "raw_update_norms": raw_norms,
                "clip_factors": factors,
            }
        )
        print(
            f"[collect] round {rnd:3d}/{args.rounds}  "
            f"raw||delta|| {min(raw_norms):.2f}-{max(raw_norms):.2f}  "
            f"clip factor {min(factors):.4f}-{max(factors):.4f}",
            flush=True,
        )

    d = int(sum(int(np.prod(gcpu[k].shape)) for k in fkeys))
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(
            {
                "rounds": rounds,
                "partitions": partitions,
                "adversarial": sorted(adversarial),
                "clip_norm": clip_norm,
                "d": d,
                "K": K,
                "N": N,
                "seed": args.seed,
                "alpha": args.alpha,
                "float_keys": fkeys,
            },
            fh,
            indent=2,
        )
    print(f"[collect] done: {args.rounds} rounds, d={d}, out={args.out}")


if __name__ == "__main__":
    main()
