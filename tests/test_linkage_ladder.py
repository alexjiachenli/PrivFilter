"""Keep linkage pairs and disclosure comparisons within each training run."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def record(client_id, round_id, score, reject):
    return {
        "seed": 1,
        "client_id": client_id,
        "round": round_id,
        "p_target": score,
        "logit_margin": score,
        "target_logit_mean": score,
        "max_other_logit_mean": score,
        "score_margin": score,
        "predicted_reject": reject,
    }


class LinkageLadderTests(unittest.TestCase):
    def evaluate(self, runs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, records in enumerate(runs):
                path = root / f"run_{index}" / "summary.json"
                path.parent.mkdir()
                path.write_text(
                    json.dumps({"metrics": {"detection_score_records": records}})
                )
            output = root / "audit" / "linkage.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "reproduction/privacy/attack_linkage_ladder.py"),
                    "--raw-glob",
                    str(root / "run_*" / "summary.json"),
                    "--out",
                    str(output),
                    "--pairs",
                    "200",
                    "--seed",
                    "0",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(output.read_text())["rungs"]

    def test_same_seed_does_not_create_cross_run_pairs(self):
        rungs = self.evaluate(
            [
                [record(0, 1, -1, False), record(1, 1, 1, True)],
                [record(0, 2, -1, False), record(1, 2, 1, True)],
            ]
        )
        for name in ("scores_opened", "scalar_score", "bits_opened"):
            with self.subTest(rung=name):
                self.assertEqual(rungs[name]["n_positive"], 0)
                self.assertEqual(rungs[name]["n_negative"], 0)
                self.assertTrue(math.isnan(rungs[name]["linkage_auc"]))

    def test_within_round_disclosure_does_not_compare_distinct_runs(self):
        rungs = self.evaluate(
            [
                [record(0, 1, -2, True), record(1, 1, -1, True)],
                [record(0, 1, 1, False), record(1, 1, 2, False)],
            ]
        )
        for name, result in rungs.items():
            with self.subTest(rung=name):
                self.assertTrue(
                    math.isnan(result["rejection_disclosure_auc_within_round"])
                )
        self.assertEqual(rungs["scalar_score"]["rejection_disclosure_auc_pooled"], 0.0)

    def test_cross_round_pairs_remain_valid_with_distinct_run_scales(self):
        rungs = self.evaluate(
            [
                [
                    record(client, round_id, offset + scale * client, bool(client))
                    for round_id in (1, 2)
                    for client in (0, 1)
                ]
                for offset, scale in ((0, 1), (10000, 1000))
            ]
        )
        for name in ("scores_opened", "scalar_score", "bits_opened"):
            with self.subTest(rung=name):
                self.assertGreater(rungs[name]["n_positive"], 0)
                self.assertGreater(rungs[name]["n_negative"], 0)
                self.assertEqual(rungs[name]["linkage_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
