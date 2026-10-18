"""Clipping-cache binding and cleanup tests."""

import math
import sys


import multiprocessing


import tempfile


from pathlib import Path


import unittest


from unittest.mock import patch


import numpy as np


import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpc_l2_clip import (
    integer_l2_clip_reference,
    integer_l2_nearest_reference,
    nearest_inner_radius,
    MPC_NEAREST_CLIP,
)


class ClipCacheBindingTests(unittest.TestCase):
    def test_service_finalizer_removes_unconsumed_cache(self):
        import gc
        import weakref
        from mpc_dp_release import MpcDpReleaseService

        service = MpcDpReleaseService(
            dict(
                mpc_dp_backend="cryptgpu",
                mpc_dp_output_mode="aggregate_reference",
                mpc_dp_clip_mode="mpc_l2_integer",
                mpc_dp_noise_mode="zero_noise_debug",
                allow_zero_noise_mpc_dp_release=True,
                debug_zero_noise_run=True,
            )
        )
        root = Path(service.temp_dir)
        service.clip_cache.prepare(
            "unfiltered_pre_detection",
            1,
            "mpc_l2_integer",
            65536,
            [1],
            [np.array([1, 2], dtype=np.int64)],
            {"w": torch.zeros(2)},
            ["w"],
            {"w": (2,)},
        )
        ref = weakref.ref(service)
        del service
        gc.collect()
        self.assertIsNone(ref())
        self.assertFalse(root.exists())

    def test_binding_reorder_mismatch_and_cleanup(self):
        from mpc_clip_cache import ClipShareCache

        with tempfile.TemporaryDirectory() as directory:
            cache = ClipShareCache(directory)
            base = {"w": torch.zeros(2)}
            xs = [np.array([3, 4], dtype=np.int64), np.array([5, 6], dtype=np.int64)]

            def prepare(role, **kw):
                args = dict(
                    round_id=1,
                    mode="mpc_l2_integer",
                    radius=65536,
                    client_ids=[1, 2],
                    deltas=xs,
                    base=base,
                    names=["w"],
                    shapes={"w": (2,)},
                )
                args.update(kw)
                return cache.prepare(role, **args)

            for change in (
                {"round_id": 2},
                {"mode": MPC_NEAREST_CLIP},
                {"radius": 65535},
                {"base": {"w": torch.ones(2)}},
                {"deltas": [xs[0] + 1, xs[1]]},
                {"client_ids": [1, 3]},
            ):
                desc = prepare("unfiltered_pre_detection")
                cache.pending["ready"] = True
                with self.assertRaises(RuntimeError):
                    prepare("filtered_post_detection", **change)
                self.assertFalse(Path(desc["path"]).exists())
            prepare("unfiltered_pre_detection")
            cache.pending["ready"] = True
            desc = prepare("filtered_post_detection", client_ids=[2, 1], deltas=xs[::-1])
            self.assertEqual(desc["indices"], [1, 0])
            cache.clear()


if __name__ == "__main__":
    unittest.main()
