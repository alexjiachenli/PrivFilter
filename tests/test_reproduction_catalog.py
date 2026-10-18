"""Verify configuration integrity and experiment-profile consistency."""

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "reproduction/manifest.json").read_text())
        cls.rows = {row["id"]: row for row in cls.catalog["runs"]}

    def resolve(self, identifier):
        row = self.rows[identifier]
        config = json.loads((ROOT / row["config"]).read_text())
        for key in row["remove_keys"]:
            config.pop(key, None)
        config.update(row["overrides"])
        return config

    def test_every_disk_recipe_matches_its_source_digest(self):
        for row in self.catalog["runs"]:
            with self.subTest(recipe=row["id"]):
                config = self.resolve(row["id"])
                serialized = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                self.assertEqual(
                    hashlib.sha256(serialized).hexdigest(), row["normalized_source_sha256"]
                )

    def test_identifiers_and_profile_paths_are_portable(self):
        ids = [row["id"] for row in self.catalog["runs"]]
        self.assertEqual(len(ids), len(set(ids)))
        files = [file.name for file in (ROOT / "configs").glob("*.json")]
        self.assertEqual(len(files), len({name.casefold() for name in files}))
        for row in self.catalog["runs"]:
            self.assertEqual(Path(row["id"]).name, row["id"])
            self.assertFalse(Path(row["config"]).is_absolute())
            self.assertTrue((ROOT / row["config"]).is_file())

    def test_single_patch_control_does_not_turn_into_dba(self):
        plain = self.resolve("main_no_defense_cifar10_seed1")
        distributed = self.resolve("mmdbac_nodef_cifar10_seed1")
        self.assertEqual(plain["trigger_num"], 1)
        self.assertNotEqual(plain.get("attack_mode"), "dba")
        self.assertEqual(distributed["trigger_num"], 4)
        self.assertEqual(distributed["attack_mode"], "dba")

    def test_main_and_integer_routes_remain_distinct(self):
        for dataset in ("cifar10", "cifar100", "gtsrb"):
            for seed in (1, 2, 3):
                p = self.resolve(f"mmtable_sim_{dataset}_seed{seed}")
                self.assertEqual(p["seed"], seed)
                self.assertEqual(p["robust_detector_threshold_q"], 0.15)
                self.assertEqual(p["mpc_dp_backend"], "simulated_mpc")
                self.assertEqual(p["mpc_dp_clip_mode"], "local_reference_clip_then_mpc_sum")
        integer = self.resolve("privfilter_mpc_integer_cifar10_seed1")
        self.assertEqual(integer["mpc_dp_backend"], "cryptgpu")
        self.assertEqual(integer["mpc_dp_clip_mode"], "mpc_l2_integer_nearest")
        self.assertIs(integer["mpc_dp_clip_cache"], True)
        self.assertEqual(integer["secure_mask_mode"], "secret_shared")

    def test_evasion_and_historical_sampling_guards_are_explicit(self):
        for identifier, row in self.rows.items():
            p = self.resolve(identifier)
            if p.get("attack_mode") == "priv_filterfl_detector_evasion_adaptive":
                self.assertIs(p["adaptive_evasion_fail_closed"], True)
            if "privacy-control" in row["suites"]:
                self.assertNotEqual(
                    p.get("client_sampling_strategy", "uniform_random"), "balanced_block"
                )
                self.assertIs(row["runtime_overrides"]["allow_unbalanced_dp_schedule"], True)


if __name__ == "__main__":
    unittest.main()
