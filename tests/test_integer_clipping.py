"""Integer-radius bounds and optional three-party agreement tests."""

import math


import multiprocessing


import tempfile


from pathlib import Path


import unittest


from unittest.mock import patch


import numpy as np


import torch


from mpc_l2_clip import (
    integer_l2_clip_reference,
    integer_l2_nearest_reference,
    nearest_inner_radius,
    MPC_NEAREST_CLIP,
)


def _fixtures():
    rng = np.random.default_rng(1729)
    values = rng.integers(-(1 << 20), 1 << 20, (20, 127), dtype=np.int64)
    values[:6] = 0
    values[1, :2] = [3, 4]
    values[2, :2] = [39321, 52428]
    values[3, :2] = [39322, -52429]
    values[4, 0] = -65536
    values[5, 0] = 65537
    return values


def _integer_clip_worker():
    from cryptgpu_runtime import activate_runtime

    activate_runtime()
    from crypten import comm
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
    from mpc_l2_clip import cryptgpu_l2_clip, _floor_nonnegative_power_two, _negative, _l2_factor

    rank = comm.get().get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def share(x):
        rng = np.random.default_rng(991)
        a = rng.integers(-(1 << 63), (1 << 63) - 1, x.shape, dtype=np.int64)
        b = rng.integers(-(1 << 63), (1 << 63) - 1, x.shape, dtype=np.int64)
        s = (a, b, x - a - b)[rank]
        return ArithmeticSharedTensor.from_shares(torch.from_numpy(s).to(device), precision=0)

    values = _fixtures()
    probe = share(values)
    np.testing.assert_array_equal(
        _negative(probe).reveal().cpu().numpy(), (values < 0).astype(np.int64), err_msg="sign"
    )
    np.testing.assert_array_equal(
        (probe * probe).reveal().cpu().numpy(), values * values, err_msg="square"
    )
    ns = (probe * probe).sum(dim=1, keepdim=True)
    expected_ns = (values * values).sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(ns.reveal().cpu().numpy(), expected_ns, err_msg="norm")
    ns_chunk = probe[:, :1] * 0
    for i in range(0, 127, 37):
        p = ArithmeticSharedTensor.from_shares(probe.share[:, i : i + 37].contiguous(), precision=0)
        ns_chunk = ns_chunk + (p * p).sum(dim=1, keepdim=True)
    np.testing.assert_array_equal(
        ns_chunk.reveal().cpu().numpy(), expected_ns, err_msg="chunk norm"
    )
    expected_factors = np.array(
        [
            [(1 << 30) * 65536 // max(65536, math.isqrt(int(n)) + int(math.isqrt(int(n)) ** 2 < n))]
            for n in expected_ns[:, 0]
        ]
    )
    factor = _l2_factor(ns, 65536)
    np.testing.assert_array_equal(factor.reveal().cpu().numpy(), expected_factors, err_msg="factor")
    sign = 1 - _negative(probe) * 2
    mag = probe * sign
    np.testing.assert_array_equal(mag.reveal().cpu().numpy(), np.abs(values), err_msg="magnitude")
    prod = mag * factor
    np.testing.assert_array_equal(
        prod.reveal().cpu().numpy(), np.abs(values) * expected_factors, err_msg="product"
    )
    quot = _floor_nonnegative_power_two(prod, 30)
    np.testing.assert_array_equal(
        quot.reveal().cpu().numpy(),
        np.abs(values) * expected_factors // (1 << 30),
        err_msg="quotient",
    )
    clipped, timing = cryptgpu_l2_clip(share(values), 65536, chunk_size=37)
    actual = clipped.reveal().cpu().numpy()
    expected = np.stack([integer_l2_clip_reference(x, 65536) for x in values])
    np.testing.assert_array_equal(actual, expected)
    # Large masked signed shares exercise both positive and negative wraps.
    q = 1 << 30
    products = np.tile(
        np.array([0, 1, q - 1, q, q + 1, 65536 * q, (1 << 57) - 1], dtype=np.int64), 200
    )
    divided = _floor_nonnegative_power_two(share(products), 30).reveal().cpu().numpy()
    np.testing.assert_array_equal(divided, products // q)
    # Nearest uses a public offset and a single wrap-sign correction. Exercise
    # the expanded nonnegative domain as well as ties and mixed signs.
    products = np.tile(
        np.array([0, q // 2, q - 1, q, (1 << 58) - 1, 1 << 58, (1 << 59) + q // 2], dtype=np.int64),
        200,
    )
    divided = (
        _floor_nonnegative_power_two(share(products), 30, one_sided=True).reveal().cpu().numpy()
    )
    np.testing.assert_array_equal(divided, products // q)
    clipped, _ = cryptgpu_l2_clip(share(values), 65536, chunk_size=37, mode=MPC_NEAREST_CLIP)
    actual = clipped.reveal().cpu().numpy()
    expected = np.stack([integer_l2_nearest_reference(x, 65536) for x in values])
    np.testing.assert_array_equal(actual, expected)
    assert all(sum(int(v) ** 2 for v in row) <= 65536**2 for row in actual)
    return {"rank": rank, "timing": timing, "cases": len(products)}


class IntegerL2ClipTests(unittest.TestCase):
    def test_nearest_radius_and_rejection_of_insufficient_headroom(self):
        self.assertEqual(nearest_inner_radius(65536, 4901770), 64429)
        with self.assertRaises(ValueError):
            nearest_inner_radius(1, 5)
        rng = np.random.default_rng(881)
        for dimension in (1, 2, 3, 127, 4096, 4901770):
            for radius in (65536,):
                x = rng.integers(-1000, 1001, dimension, dtype=np.int64)
                y = integer_l2_nearest_reference(x, radius)
                self.assertLessEqual(int(np.sum(y * y)), radius**2)
        for x in _fixtures():
            y = integer_l2_nearest_reference(x, 65536)
            self.assertLessEqual(sum(int(v) ** 2 for v in y), 65536**2)

    def test_radius_zero_negative_and_boundary(self):
        for x in _fixtures():
            result = integer_l2_clip_reference(x, 65536)
            self.assertLessEqual(sum(int(v) ** 2 for v in result), 65536**2)
            self.assertTrue(np.all(np.abs(result) <= np.abs(x)))
            if sum(int(v) ** 2 for v in x) <= 65536**2:
                np.testing.assert_array_equal(result, x)

    def test_quantization_really_can_exceed_old_bound(self):
        result = np.rint(np.array([1.0, 1.0]) / math.sqrt(2) * 65536).astype(np.int64)
        self.assertGreater(sum(int(v) ** 2 for v in result), 65536**2)

    def test_input_domain_is_checked(self):
        with self.assertRaises(OverflowError):
            integer_l2_clip_reference(np.array([1 << 28], dtype=np.int64), 65536)

    @unittest.skipUnless(torch.cuda.is_available(), "real CryptGPU GPU test")
    def test_real_three_party_exact_agreement_and_truncation(self):
        from cryptgpu_runtime import activate_runtime

        activate_runtime()
        from crypten import mpc

        multiprocessing.set_start_method("spawn", force=True)
        results = mpc.run_multiprocess(world_size=3)(_integer_clip_worker)()
        self.assertEqual(len(results), 3)
        self.assertEqual({r["rank"] for r in results}, {0, 1, 2})

    @unittest.skipUnless(torch.cuda.is_available(), "real CryptGPU GPU test")
    def test_release_uses_raw_inputs_and_clips_before_opening(self):
        from mpc_dp_release import MpcDpReleaseService

        params = dict(
            mpc_dp_backend="cryptgpu",
            mpc_dp_output_mode="aggregate_reference",
            mpc_dp_clip_mode="mpc_l2_integer",
            mpc_dp_clip_norm=1.0,
            mpc_dp_noise_mode="zero_noise_debug",
            allow_zero_noise_mpc_dp_release=True,
            debug_zero_noise_run=True,
            mpc_dp_clip_chunk_size=37,
        )
        service = MpcDpReleaseService(params)
        try:
            values = _fixtures()[2:5]
            clients = [{"w": torch.from_numpy(x.astype(np.float32) / 65536)} for x in values]
            base = {"w": torch.zeros(127)}
            with patch.object(
                service,
                "_serialize_locally_clipped_deltas",
                side_effect=AssertionError("harness clipping called"),
            ):
                actual = service.make_fast_aggregate_reference_release(
                    clients,
                    base,
                    [1, 2, 3],
                    1,
                    accept_mask=[1, 0, 1],
                    release_role="filtered_post_detection",
                )
            expected = np.stack([integer_l2_clip_reference(values[i], 65536) for i in (0, 2)]).sum(
                0
            ) / (2 * 65536)
            np.testing.assert_array_equal(actual["w"].numpy(), expected.astype(np.float32))
            self.assertGreater(service.metrics[-1]["mpc_clip_s"], 0)
            self.assertIsNone(service.metrics[-1]["local_clip_factors_min"])
            self.assertFalse(service.metrics[-1]["privacy_result_valid"])
            # Exact binary-mask multiplication must not reintroduce the
            # upstream probabilistic fixed-point truncation after clipping.
            with tempfile.TemporaryDirectory() as directory:
                mask = np.array([1, 0, 1], dtype=np.int64)
                first = np.array([0, 1, 1], dtype=np.int64)
                second = np.array([1, 1, 0], dtype=np.int64)
                for rank, bits in enumerate((first, second, mask ^ first ^ second)):
                    np.save(Path(directory) / f"mask_share_{rank}.npy", bits)
                actual = service.make_fast_aggregate_reference_release(
                    clients,
                    base,
                    [1, 2, 3],
                    2,
                    mask_share_dir=directory,
                    release_role="filtered_post_detection",
                )
            np.testing.assert_array_equal(actual["w"].numpy(), expected.astype(np.float32))
            self.assertEqual(service.metrics[-1]["clients_contributing"], 2)
        finally:
            service.cleanup()

    @unittest.skipUnless(torch.cuda.is_available(), "real CryptGPU GPU test")
    def test_cache_reuses_party_shares_for_open_and_secret_masks(self):
        from mpc_dp_release import MpcDpReleaseService

        for mode, secret_mask in [("mpc_l2_integer", False), (MPC_NEAREST_CLIP, True)]:
            params = dict(
                mpc_dp_backend="cryptgpu",
                mpc_dp_output_mode="aggregate_reference",
                mpc_dp_clip_mode=mode,
                mpc_dp_clip_norm=1.0,
                mpc_dp_noise_mode="zero_noise_debug",
                allow_zero_noise_mpc_dp_release=True,
                debug_zero_noise_run=True,
                mpc_dp_clip_chunk_size=37,
            )
            service = MpcDpReleaseService(params)
            try:
                values = _fixtures()[2:5]
                clients = [{"w": torch.from_numpy(x.astype(np.float32) / 65536)} for x in values]
                base = {"w": torch.zeros(127)}
                service.make_fast_aggregate_reference_release(clients, base, [1, 2, 3], 1)
                cache_path = Path(service.clip_cache.pending["descriptor"]["path"])
                self.assertTrue(cache_path.exists())
                with tempfile.TemporaryDirectory() as directory:
                    kwargs = dict(accept_mask=[1, 0, 1], release_role="filtered_post_detection")
                    if secret_mask:
                        for rank, bits in enumerate(([1, 0, 1], [0, 1, 1], [0, 1, 1])):
                            np.save(
                                Path(directory) / f"mask_share_{rank}.npy",
                                np.array(bits, dtype=np.int64),
                            )
                        kwargs["mask_share_dir"] = directory
                    actual = service.make_fast_aggregate_reference_release(
                        clients, base, [1, 2, 3], 1, **kwargs
                    )
                reference = (
                    integer_l2_nearest_reference
                    if mode == MPC_NEAREST_CLIP
                    else integer_l2_clip_reference
                )
                expected = (reference(values[0], 65536) + reference(values[2], 65536)) / (2 * 65536)
                np.testing.assert_array_equal(actual["w"].numpy(), expected.astype(np.float32))
                self.assertTrue(service.metrics[-1]["mpc_clip_cache_hit"])
                self.assertEqual(service.metrics[-1]["mpc_clip_s"], 0)
                self.assertFalse(cache_path.exists())
                self.assertIsNone(service.clip_cache.pending)
            finally:
                service.cleanup()


if __name__ == "__main__":
    unittest.main()
