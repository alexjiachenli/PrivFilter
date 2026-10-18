"""Fixed-state release pairing and residual-noise regression tests."""

import math


import os


import sys


import tempfile


import unittest


from pathlib import Path


import numpy as np


import torch


from mpc_dp_release import (
    MPC_DP_BACKEND_SIMULATED,
    MPC_DP_PAIRING_INDEPENDENT,
    MPC_DP_PAIRING_PARTITION,
    MpcDpReleaseService,
    canonical_mpc_dp_release_pairing,
)


SCALE = 65536


SIGMA = 0.7123263374786376


def _seeds(directory):
    for pid in range(3):
        party = os.path.join(directory, f"party_{pid}")
        os.makedirs(party, exist_ok=True)
        with open(os.path.join(party, "master_seed.bin"), "wb") as handle:
            handle.write(bytes([pid + 3]) * 64)


def _params(seed_dir, pairing):
    return {
        "name": "release_pairing_unittest",
        "dp_release_source": "mpc_aggregate_ref_dp_fast",
        "mpc_dp_backend": "simulated_mpc",
        "mpc_dp_protocol": "replicated-field",
        "mpc_dp_ring_bits": 64,
        "mpc_dp_output_mode": "aggregate_reference",
        "mpc_dp_clip_mode": "local_reference_clip_then_mpc_sum",
        "mpc_dp_noise_mode": "party_local_distributed_dgauss",
        "mpc_dp_noise_share_mode": "party_local_seeded_stream",
        "mpc_dp_fixed_point_scale": SCALE,
        "mpc_dp_clip_norm": 1.0,
        "mpc_dp_noise_multiplier": SIGMA,
        "mpc_dp_release_pairing": pairing,
        "mpc_dp_noise_generation": {"seed_dir": os.path.join(seed_dir, "party_{party}")},
        "allow_zero_noise_mpc_dp_release": False,
    }


class PairingResolutionTests(unittest.TestCase):
    def test_default_is_independent(self):
        self.assertEqual(canonical_mpc_dp_release_pairing(None), MPC_DP_PAIRING_INDEPENDENT)

    def test_aliases(self):
        for alias in ("partition", "correlated", "one-release", "PARTITION_CORRELATED"):
            self.assertEqual(canonical_mpc_dp_release_pairing(alias), MPC_DP_PAIRING_PARTITION)

    def test_unknown_is_rejected(self):
        with self.assertRaises(ValueError):
            canonical_mpc_dp_release_pairing("share_the_noise")


class ReleasePairingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        _seeds(self.tmp.name)
        torch.manual_seed(3)
        self.ori = {"w": torch.zeros(4096)}
        # Use a distinct rejected update to expose accidental noise cancellation.
        self.clients = [
            {"w": torch.full((4096,), 0.001)},
            {"w": torch.full((4096,), 0.002)},
            {"w": torch.full((4096,), 0.050)},
            {"w": torch.full((4096,), 0.003)},
        ]
        self.mask = [1, 1, 0, 1]

    def _round(self, pairing):
        """One round: the pre-detection release, then the filtered one."""
        service = MpcDpReleaseService(_params(self.tmp.name, pairing))
        try:
            pre = service.make_fast_aggregate_reference_release(
                [{k: v.clone() for k, v in c.items()} for c in self.clients],
                {k: v.clone() for k, v in self.ori.items()},
                list(range(len(self.clients))),
                round_id=1,
            )
            post = service.make_fast_aggregate_reference_release(
                [{k: v.clone() for k, v in c.items()} for c in self.clients],
                {k: v.clone() for k, v in self.ori.items()},
                list(range(len(self.clients))),
                round_id=1,
                accept_mask=self.mask,
                release_role="filtered_post_detection",
            )
            return pre, post, list(service.metrics)
        finally:
            service.cleanup()

    def test_the_difference_is_never_exact(self):
        """Check that independent noise remains after subtracting released sums."""
        for pairing, expected_sigma in (
            (MPC_DP_PAIRING_INDEPENDENT, SIGMA * math.sqrt(2)),
            (MPC_DP_PAIRING_PARTITION, SIGMA),
        ):
            with self.subTest(pairing=pairing):
                pre, post, _ = self._round(pairing)
                # Undo the base and averaging denominator to compare sum-coordinate noise.
                gap = (pre["w"] * 4.0 - post["w"] * 3.0).std().item()
                self.assertGreater(gap, 0.5 * expected_sigma)
                self.assertLess(
                    abs(gap / expected_sigma - 1.0),
                    0.25,
                    f"residual noise {gap:.4f}, expected ~{expected_sigma:.4f}",
                )

    def test_partition_leaves_exactly_one_sigma_less_noise_in_the_difference(self):
        """Check the pre-release noise increase under partition-correlated pairing."""
        pre_i, _, _ = self._round(MPC_DP_PAIRING_INDEPENDENT)
        pre_p, _, _ = self._round(MPC_DP_PAIRING_PARTITION)
        ratio = (pre_p["w"].std() / pre_i["w"].std()).item()
        self.assertLess(
            abs(ratio - math.sqrt(2)),
            0.25,
            f"pre-detection noise ratio {ratio:.3f}, expected ~1.414",
        )

    def test_the_metrics_record_which_pairing_ran(self):
        _pre, _post, metrics = self._round(MPC_DP_PAIRING_PARTITION)
        self.assertEqual(metrics[0]["release_pairing"], MPC_DP_PAIRING_PARTITION)
        self.assertEqual(metrics[0]["noise_stream_labels"], ["", "filtered_post_detection"])
        self.assertEqual(metrics[1]["noise_stream_labels"], ["filtered_post_detection"])

    def test_independent_pairing_keeps_the_streams_disjoint(self):
        _pre, _post, metrics = self._round(MPC_DP_PAIRING_INDEPENDENT)
        self.assertEqual(metrics[0]["noise_stream_labels"], [""])
        self.assertEqual(metrics[1]["noise_stream_labels"], ["filtered_post_detection"])


if __name__ == "__main__":
    unittest.main()
