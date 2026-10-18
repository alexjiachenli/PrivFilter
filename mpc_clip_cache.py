"""Party-local clipping-share cache with input and round bindings.

The local harness uses in-memory input hashes to detect stale cache entries.
These hashes provide cache consistency, not cryptographic input authentication.
Each worker reads and writes only its own shares.
"""

import hashlib
import json
from pathlib import Path
import secrets
import shutil
import tempfile

import numpy as np


class ClipShareCache:
    def __init__(self, root):
        self.root = root
        self.pending = None

    def clear(self):
        if self.pending is not None:
            shutil.rmtree(self.pending["descriptor"]["path"], ignore_errors=True)
            self.pending = None

    def prepare(self, role, round_id, mode, radius, client_ids, deltas, base, names, shapes):
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("duplicate client IDs cannot be bound to clipping shares")
        digest = hashlib.sha256()
        digest.update(json.dumps([(n, shapes[n]) for n in names]).encode())
        for name in names:
            value = base[name].detach().cpu().float().contiguous().numpy()
            digest.update(memoryview(value).cast("B"))
        binding = (int(round_id), mode, int(radius), digest.digest())
        inputs = {
            cid: hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).digest()
            for cid, x in zip(client_ids, deltas)
        }
        if role == "unfiltered_pre_detection":
            self.clear()
            descriptor = dict(
                path=tempfile.mkdtemp(prefix="clip_cache_", dir=self.root),
                cache_id=secrets.token_hex(16),
                round_id=int(round_id),
                mode=mode,
                radius=int(radius),
                shape=[len(deltas), int(deltas[0].size)],
            )
            self.pending = dict(
                descriptor=descriptor,
                binding=binding,
                inputs=inputs,
                client_ids=list(client_ids),
                ready=False,
            )
            return dict(descriptor, action="write", indices=list(range(len(deltas))))
        if role != "filtered_post_detection" or self.pending is None:
            return None
        pending = self.pending
        if (
            not pending["ready"]
            or binding != pending["binding"]
            or any(pending["inputs"].get(cid) != value for cid, value in inputs.items())
        ):
            self.clear()
            raise RuntimeError(
                "clipping cache binding mismatch: round, model, inputs or clipping rule changed"
            )
        return dict(
            pending["descriptor"],
            action="read",
            indices=[pending["client_ids"].index(cid) for cid in client_ids],
        )


def _metadata(descriptor):
    return {key: descriptor[key] for key in ("cache_id", "round_id", "mode", "radius", "shape")}


def save_party_clip(descriptor, rank, share):
    """Save a worker's clipping shares before aggregation."""
    root = Path(descriptor["path"])
    np.save(root / f"party_{rank}.npy", share)
    (root / f"party_{rank}.json").write_text(json.dumps(_metadata(descriptor)))


def load_party_clip(descriptor, rank):
    root = Path(descriptor["path"])
    if json.loads((root / f"party_{rank}.json").read_text()) != _metadata(descriptor):
        raise RuntimeError("cached clipping party metadata mismatch")
    share = np.load(root / f"party_{rank}.npy", mmap_mode="r", allow_pickle=False)
    if share.dtype != np.int64 or list(share.shape) != descriptor["shape"]:
        raise RuntimeError("cached clipping party share shape/dtype mismatch")
    return np.ascontiguousarray(share[descriptor["indices"]])
