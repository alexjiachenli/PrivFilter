"""Check reproducibility of privacy-audit client partitions and identities."""

import contextlib
import io
import json
import logging
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.privacy import collect_updates as collector


class TinyDataset:
    def __len__(self):
        return 400

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        return torch.zeros(3, 32, 32), index % 10


class CollectionSeedTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(os.chdir, os.getcwd())

    def collect_setup(self, seed, ambient_seed):
        random.seed(ambient_seed)
        result = {}

        class SetupComplete(Exception):
            pass

        def capture_setup(handle):
            result["partitions"] = [
                list(loader.sampler.indices) for _, loader in handle.train_data
            ]
            result["adversarial"] = sorted(handle.adversarial_namelist)
            raise SetupComplete

        logger = logging.getLogger("logger")
        previous_handlers, previous_level = list(logger.handlers), logger.level
        with tempfile.TemporaryDirectory() as temporary:
            params = json.loads((ROOT / "configs/privfilter_cifar10.json").read_text())
            params.update(
                number_of_total_participants=20, num_clients=5, poison_client_num=5
            )
            params_path = Path(temporary) / "params.json"
            params_path.write_text(json.dumps(params))
            argv = [
                "collect_updates.py",
                "--params",
                str(params_path),
                "--rounds",
                "1",
                "--seed",
                str(seed),
                "--out",
                str(Path(temporary) / "updates"),
            ]
            try:
                with (
                    patch.object(sys, "argv", argv),
                    patch("handle.datasets.CIFAR10", return_value=TinyDataset()),
                    patch.object(collector.Handle, "create_model", capture_setup),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SetupComplete),
                ):
                    collector.main()
            finally:
                for handler in list(logger.handlers):
                    if handler not in previous_handlers:
                        logger.removeHandler(handler)
                        handler.close()
                logger.setLevel(previous_level)
        return result

    def test_seed_controls_partitions_and_malicious_clients(self):
        first = self.collect_setup(seed=1, ambient_seed=111)
        repeated = self.collect_setup(seed=1, ambient_seed=222)
        changed = self.collect_setup(seed=2, ambient_seed=111)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first["partitions"], changed["partitions"])
        self.assertNotEqual(first["adversarial"], changed["adversarial"])


if __name__ == "__main__":
    unittest.main()
