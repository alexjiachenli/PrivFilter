"""MP-SPDZ aggregate serialization tests with no external runtime required."""

import os
import sys
from pathlib import Path


import tempfile


import unittest


from unittest import mock


import numpy as np


import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from secure_aggregation import (
    MPC_VECTOR_CHUNK,
    RING_BITS,
    SCALE,
    SecureAggregation,
    _strict_binary_output,
)


class StrictBinaryOutputTests(unittest.TestCase):
    def test_reads_little_endian_int64(self):
        values = np.array([0, 1, -1, 2**40, -(2**40)], dtype="<i8")
        parsed = _strict_binary_output(values.tobytes(), context="ctx")
        self.assertTrue(np.array_equal(parsed, values))

    def test_partial_trailing_record_fails_closed(self):
        blob = np.array([1, 2], dtype="<i8").tobytes()[:-3]
        with self.assertRaisesRegex(RuntimeError, "multiple of 8"):
            _strict_binary_output(blob, context="ctx")

    def test_text_is_not_accepted_as_binary(self):
        with self.assertRaisesRegex(RuntimeError, "must be binary output"):
            _strict_binary_output("123\n", context="ctx")


class BinaryParseTests(unittest.TestCase):
    def setUp(self):
        self.service = SecureAggregation.__new__(SecureAggregation)

    def test_binary_round_trip_matches_text_path(self):
        shapes = {"w": (2, 2), "b": (3,)}
        summed = np.array([10000, -20000, 30000, 5, 15000, -5, 0], dtype="<i8")

        from_binary = self.service._parse_mpc_output(
            summed.tobytes(), ["w", "b"], shapes, num_clients=2
        )
        from_text = self.service._parse_mpc_output(
            "\n".join(str(int(v)) for v in summed) + "\n",
            ["w", "b"],
            shapes,
            num_clients=2,
        )

        for name in shapes:
            self.assertEqual(tuple(from_binary[name].shape), shapes[name])
            self.assertTrue(torch.allclose(from_binary[name].cpu(), from_text[name].cpu()))
        self.assertTrue(
            torch.allclose(
                from_binary["w"].cpu().reshape(-1),
                torch.tensor([0.5, -1.0, 1.5, 0.00025]),
            )
        )

    def test_truncated_binary_output_fails_closed(self):
        short = np.array([1, 2], dtype="<i8").tobytes()
        with self.assertRaisesRegex(RuntimeError, "output count mismatch"):
            self.service._parse_mpc_output(short, ["w"], {"w": (3,)}, num_clients=1)


class WriteClientInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = os.path.join(self.temp.name, "backend")
        os.makedirs(os.path.join(self.backend, "Player-Data"))

        self.service = SecureAggregation.__new__(SecureAggregation)
        self.service.mp_spdz_path = self.backend
        self.service.player_data_dir = os.path.join(self.temp.name, "staged")
        self.service.enable_share_cache = False
        os.makedirs(self.service.player_data_dir, exist_ok=True)

    def _write(self, clients, names):
        with mock.patch("secure_aggregation.require_mp_spdz_runtime_lock"):
            return self.service._write_client_inputs(clients, names)

    def _party_shares(self, count):
        out = []
        for pid in range(3):
            path = os.path.join(self.backend, "Player-Data", f"Input-Binary-P{pid}-0")
            with open(path, "rb") as handle:
                blob = handle.read()
            self.assertEqual(len(blob), count * 8)
            out.append(np.frombuffer(blob, dtype="<i8").astype(np.uint64))
        return out

    def test_shares_reconstruct_the_fixed_point_sum(self):
        names = ["w", "b"]
        clients = [
            {"w": torch.tensor([[0.5, -1.0], [0.25, 0.0]]), "b": torch.tensor([1.5, -2.5])},
            {"w": torch.tensor([[1.0, 1.0], [-0.5, 2.0]]), "b": torch.tensor([0.5, 0.5])},
        ]
        sizes = self._write(clients, names)
        self.assertEqual(sizes, {"w": 4, "b": 2})

        s0, s1, s2 = self._party_shares(6)
        recovered = (s0 + s1 + s2).view(np.int64)

        expected = np.concatenate(
            [
                np.round(sum(c[name].numpy().reshape(-1) for c in clients) * SCALE).astype(np.int64)
                for name in names
            ]
        )
        self.assertTrue(np.array_equal(recovered, expected))

    def test_sharing_randomness_is_not_reproducible_from_client_index(self):
        # Sharing randomness must be independent of public client and parameter identifiers.
        clients = [{"w": torch.zeros(64)}]
        self._write(clients, ["w"])
        first = self._party_shares(64)[0].copy()
        self._write(clients, ["w"])
        second = self._party_shares(64)[0]
        self.assertFalse(np.array_equal(first, second))

    def test_shares_cover_the_whole_ring(self):
        clients = [{"w": torch.zeros(4096)}]
        self._write(clients, ["w"])
        share = self._party_shares(4096)[0]
        # A half-ring draw would leave the top bit clear on every coordinate.
        self.assertTrue(bool((share >> np.uint64(RING_BITS - 1)).any()))

    def test_non_finite_weights_are_rejected(self):
        clients = [{"w": torch.tensor([1.0, float("nan")])}]
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            self._write(clients, ["w"])


class AggregateStrictParserTests(unittest.TestCase):
    def setUp(self):
        self.service = SecureAggregation.__new__(SecureAggregation)

    def _assert_rejects_common_invalid_outputs(self, parser, valid_lines):
        invalid_outputs = {
            "junk": [*valid_lines[:-1], "unexpected", valid_lines[-1]],
            "extra": [*valid_lines, "0"],
            "fractional": ["1.5", *valid_lines[1:]],
            "nan": ["nan", *valid_lines[1:]],
            "infinity": ["inf", *valid_lines[1:]],
        }
        for label, lines in invalid_outputs.items():
            with self.subTest(label=label):
                with self.assertRaises(RuntimeError):
                    parser("\n".join(lines) + "\n")

    def test_aggregate_parser_accepts_exact_integer_count_only(self):
        parser = lambda output: self.service._parse_mpc_output(
            output,
            ["weight"],
            {"weight": (2,)},
            num_clients=2,
        )
        parsed = parser("10000\n-20000\n")
        self.assertTrue(torch.allclose(parsed["weight"].cpu(), torch.tensor([0.5, -1.0])))
        self._assert_rejects_common_invalid_outputs(parser, ["10000", "-20000"])


if __name__ == "__main__":
    unittest.main()
