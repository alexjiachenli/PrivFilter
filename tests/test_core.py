"""CPU checks for release accounting, detector decisions and BN folding."""

import copy
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import differential_privacy as dp
from detector_rules import (
    calibrated_centered_reject_bits,
    secure_calibrated_centered_decision_values,
    secure_mean_target_logit_margin,
    secure_mean_target_probability,
)
from tools.fuse_bn import fuse_conv_bn_with_stats


class CoreTests(unittest.TestCase):
    def test_response_probability_matches_batch_mean(self):
        logits = torch.tensor([[1.2, 3.0, -0.4], [0.7, -1.0, 2.2]], dtype=torch.float64)
        probabilities = torch.softmax(logits, dim=1)
        response = secure_mean_target_probability(probabilities, num_classes=3)
        torch.testing.assert_close(response, probabilities[:, 0].mean())

    def test_response_margin_matches_other_class_maximum(self):
        logits = torch.tensor([[1.2, 3.0, -0.4], [0.7, -1.0, 2.2]], dtype=torch.float64)
        response = secure_mean_target_logit_margin(logits, num_classes=3)
        expected = (logits[:, 0] - logits[:, 1:].max(dim=1).values).mean()
        torch.testing.assert_close(response, expected)

    def test_loo_rule_matches_explicit_peer_reference_and_secure_sign(self):
        scores = [0.01, 0.03, 0.02, 0.07, 0.9, 0.04]
        options = dict(threshold_q=0.15, center_weight=1.0, center_mode="loo_mean", min_clients=5)
        expected = [
            s - sum(scores[:i] + scores[i + 1 :]) / (len(scores) - 1) > 0.15
            for i, s in enumerate(scores)
        ]
        self.assertEqual(calibrated_centered_reject_bits(scores, **options), expected)
        signed = secure_calibrated_centered_decision_values(scores, **options)
        self.assertEqual([x > 0 for x in signed], expected)
        shifted = [s + 17 for s in scores]
        self.assertEqual(calibrated_centered_reject_bits(shifted, **options), expected)

    def test_nonfinite_scores_fail(self):
        with self.assertRaises(ValueError):
            calibrated_centered_reject_bits(
                [0.0, 0.1, 0.2, 0.3, float("nan")],
                threshold_q=0.15,
                center_weight=1.0,
                center_mode="loo_mean",
            )

    def test_reference_accounting_counts_partition_once(self):
        p = dict(
            epochs=200,
            num_clients=10,
            number_of_total_participants=100,
            ldp_target_delta=1e-5,
            ldp_max_grad_norm=1.0,
            enable_ldp=True,
            ldp_rdp_conversion="paper_tight",
            client_sampling_strategy="balanced_block",
            ldp_accountant_variant="balanced_direct_composition",
            dp_release_source="mpc_aggregate_ref_dp_fast",
            ldp_ori_weight_source="aggregate_ref",
            mpc_dp_release_pairing="partition_correlated",
        )
        dp.require_balanced_schedule(p)
        result = dp.evaluate_accountant_for_sigma(p, sigma=2.4023136318435663)
        self.assertEqual(result["worst_case_releases"], 20)
        self.assertAlmostEqual(result["epsilon"], 10.0, places=3)
        self.assertGreaterEqual(result["epsilon"], result["epsilon_exact_gaussian"])
        p["mpc_dp_release_pairing"] = "independent"
        self.assertEqual(
            dp.evaluate_accountant_for_sigma(p, sigma=2.4023136318435663)["worst_case_releases"], 40
        )
        p["client_sampling_strategy"] = "uniform_random"
        with self.assertRaises(ValueError):
            dp.require_balanced_schedule(p)

    def test_batch_stat_folding_matches_train_mode_bn(self):
        torch.manual_seed(19)
        conv = torch.nn.Conv2d(3, 5, 3, padding=1, bias=False).double()
        bn = torch.nn.BatchNorm2d(5).double().train()
        with torch.no_grad():
            bn.weight.copy_(torch.linspace(0.4, 1.3, 5))
            bn.bias.copy_(torch.linspace(-0.1, 0.2, 5))
            x = torch.randn(7, 3, 8, 8, dtype=torch.float64)
            z = conv(x)
            reference = bn(z)
            stats = dict(mean=z.mean((0, 2, 3)), var=z.var((0, 2, 3), unbiased=False))
            folded = fuse_conv_bn_with_stats(copy.deepcopy(conv), copy.deepcopy(bn), **stats)
            torch.testing.assert_close(folded(x), reference, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
