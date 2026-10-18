"""A small synthetic CPU round exercises the retained training/detection route."""

import copy
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import algorithm
from handle import Handle
import main

ROOT = Path(__file__).resolve().parents[1]


class RoundReferenceTests(unittest.TestCase):
    def test_testing_does_not_change_the_round_start_reference(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        handle = SimpleNamespace(
            model=model, params={}, test_data=None, test_un_target_label_data=None
        )
        with (
            patch.object(algorithm, "get_clients", return_value=([0], [])),
            patch.object(algorithm, "device", torch.device("cpu")),
            patch.object(algorithm.test, "normal_test", return_value=0.0),
        ):
            _, _, _, reference = algorithm._client_selection(1, handle)
            algorithm._run_testing(handle, {"weight": torch.tensor([[1.5]])})
        self.assertEqual(reference["weight"].item(), 0.0)
        self.assertEqual(model.weight.item(), 1.5)
        self.assertNotEqual(reference["weight"].data_ptr(), model.weight.data_ptr())


class ReferenceFailureTests(unittest.TestCase):
    def run_mock_round(self, release_source):
        state = {"weight": torch.zeros(1, 1)}
        handle = SimpleNamespace(
            model=torch.nn.Linear(1, 1, bias=False),
            start_epoch=1,
            mpc_dp_aggregate_reference=state,
            params={
                "epochs": 1,
                "defence_method": None,
                "dp_release_source": release_source,
                "ldp_ori_weight_source": "aggregate_ref"
                if release_source == algorithm.DP_RELEASE_MPC_AGGREGATE_FAST
                else "diff",
                "log_memory_each_epoch": False,
            },
        )
        with (
            patch.object(algorithm, "_read_ldp_config", return_value=True),
            patch.object(algorithm, "get_clients", return_value=([0], [])),
            patch.object(
                algorithm, "_train_clients", return_value=([state], [state], [state], [1])
            ),
            patch.object(algorithm, "_aggregate_for_trigger", return_value=state),
            patch.object(algorithm, "_generate_trigger", return_value=([], None)),
            patch.object(algorithm, "_masked_aggregate_final", return_value=state),
            patch.object(algorithm, "_run_testing", return_value=(0.0, 0.0)),
            patch.object(
                algorithm,
                "_make_filtered_dp_reference",
                side_effect=RuntimeError("reference backend failed"),
            ),
            patch.object(
                algorithm, "create_ldp_instance", side_effect=RuntimeError("client noise failed")
            ),
        ):
            return algorithm.FedAvg(handle)

    def test_mpc_filtered_reference_failure_stops_training(self):
        with self.assertRaisesRegex(RuntimeError, "reference backend failed"):
            self.run_mock_round(algorithm.DP_RELEASE_MPC_AGGREGATE_FAST)

    def test_client_noise_snapshot_failure_retains_warning_behavior(self):
        with self.assertLogs(algorithm.logger, level="WARNING") as logs:
            metrics = self.run_mock_round(algorithm.DP_RELEASE_CLIENT_LDP)
        self.assertEqual(len(metrics["round_metrics"]), 1)
        self.assertTrue(any("client noise failed" in message for message in logs.output))


class SyntheticRoundTest(unittest.TestCase):
    def test_training_synthesis_folding_and_two_noisy_releases(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        main.seed(12)
        with tempfile.TemporaryDirectory() as temporary:
            params = json.loads((ROOT / "configs/privfilter_cifar10.json").read_text())
            params.update(
                epochs=1,
                num_clients=5,
                number_of_total_participants=5,
                local_epochs=1,
                poison_epochs=1,
                poison_client_num=1,
                batch_size=2,
                trigger_batch_size=2,
                cgan_e1=1,
                cgan_e2=1,
                gpu_preload=False,
                output_root=temporary,
                log_memory_each_epoch=False,
                log_memory_cryptgpu=False,
            )
            noise_root = Path(temporary) / "seeds"
            for party in range(3):
                directory = noise_root / f"party_{party}"
                directory.mkdir(parents=True)
                (directory / "master_seed.bin").write_bytes(os.urandom(32))
            params["mpc_dp_noise_generation"]["seed_dir"] = str(noise_root / "party_{party}")
            handle = Handle("synthetic", params, "unit-test")
            handlers = list(main.logger.handlers)
            self.addCleanup(self._close_handlers, handlers)
            handle.create_model()
            handle.participants_list = list(range(5))
            handle.adversarial_namelist = [0]
            handle.benign_namelist = list(range(1, 5))
            handle.clients_data_num = [2] * 5
            handle.train_data = []
            for client in range(5):
                loader = DataLoader(
                    TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([1, 2])), batch_size=2
                )
                handle.train_data.append((client, loader))
            loader = DataLoader(
                TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([1, 2])), batch_size=2
            )
            handle.test_data = loader
            handle.test_un_target_label_data = loader
            initial_reference = copy.deepcopy(handle.model.state_dict())
            release_references = []
            release_method = algorithm.MpcDpReleaseService.make_fast_aggregate_reference_release

            def capture_reference(service, *args, **kwargs):
                release_references.append(
                    (
                        kwargs["release_role"],
                        {
                            key: value.detach().cpu().clone()
                            for key, value in kwargs["global_model"].items()
                        },
                    )
                )
                return release_method(service, *args, **kwargs)

            try:
                with patch.object(
                    algorithm.MpcDpReleaseService,
                    "make_fast_aggregate_reference_release",
                    new=capture_reference,
                ):
                    metrics = algorithm.FedAvg(handle)
                self.assertEqual(len(metrics["round_metrics"]), 1)
                self.assertTrue(math.isfinite(metrics["clean_accuracy"]))
                self.assertTrue(0 <= metrics["attack_success_rate"] <= 100)
                releases = handle.mpc_dp_release_service.metrics
                self.assertEqual(
                    [r["release_role"] for r in releases],
                    ["unfiltered_pre_detection", "filtered_post_detection"],
                )
                self.assertEqual(len(release_references), 2)
                for role, reference in release_references:
                    for key, value in initial_reference.items():
                        with self.subTest(release=role, parameter=key):
                            self.assertTrue(torch.equal(reference[key], value.cpu()))
                self.assertEqual(sum(handle.participation_counts.values()), 5)
                self.assertTrue(
                    all(torch.isfinite(v).all() for v in handle.model.state_dict().values())
                )
                main.write_run_artifacts(handle, metrics, 0.0)
                self.assertTrue((Path(handle.folder_path) / "run_summary.json").is_file())
            finally:
                service = getattr(handle, "mpc_dp_release_service", None)
                if service is not None:
                    service.cleanup()

    @staticmethod
    def _close_handlers(handlers):
        for handler in handlers:
            main.logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    unittest.main()
