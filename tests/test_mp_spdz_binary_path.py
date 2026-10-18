"""MP-SPDZ executable discovery and unchanged party command contract."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from secure_aggregation import SecureAggregation, _resolve_mp_spdz_party_binary


class PartyBinaryPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def binary(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        return path.resolve()

    def test_root_build_wins_over_legacy_layout(self):
        default = self.binary("ps-rep-ring-party.x")
        self.binary("bin/Linux-amd64/ps-rep-ring-party.x")
        self.assertEqual(_resolve_mp_spdz_party_binary(str(self.root)), str(default))

    def test_legacy_layout_is_supported(self):
        legacy = self.binary("bin/Linux-amd64/ps-rep-ring-party.x")
        self.assertEqual(_resolve_mp_spdz_party_binary(str(self.root)), str(legacy))

    def test_explicit_executable_wins_and_invalid_override_fails(self):
        self.binary("ps-rep-ring-party.x")
        explicit = self.binary("custom/ps-rep-ring-party.x")
        os.environ["MP_SPDZ_PARTY_BINARY"] = str(explicit)
        self.assertEqual(_resolve_mp_spdz_party_binary(str(self.root)), str(explicit))
        os.environ["MP_SPDZ_PARTY_BINARY"] = "missing/ps-rep-ring-party.x"
        with self.assertRaises(FileNotFoundError):
            _resolve_mp_spdz_party_binary(str(self.root))

    def test_other_protocol_and_non_executable_file_are_rejected(self):
        self.binary("replicated-ring-party.x")
        target = self.binary("ps-rep-ring-party.x")
        target.chmod(0o644)
        with self.assertRaises(FileNotFoundError):
            _resolve_mp_spdz_party_binary(str(self.root))

    def test_selected_path_is_used_for_every_party(self):
        selected = self.binary("custom/ps-rep-ring-party.x")
        os.environ["MP_SPDZ_PARTY_BINARY"] = str(selected)
        (self.root / "compile.py").write_text("# mock compiler\n")
        (self.root / "Programs/Bytecode").mkdir(parents=True)
        (self.root / "Programs/Schedules").mkdir(parents=True)
        service = SecureAggregation(mp_spdz_path=str(self.root))
        self.addCleanup(service.cleanup)

        def compile_stub(command, **kwargs):
            name = Path(command[-1]).stem
            (self.root / f"Programs/Bytecode/{name}-0.bc").write_bytes(b"bytecode")
            (self.root / f"Programs/Schedules/{name}.sch").write_text("schedule")
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch("secure_aggregation.require_mp_spdz_runtime_lock"),
            mock.patch("secure_aggregation.subprocess.run", side_effect=compile_stub),
            mock.patch.object(SecureAggregation, "_ssl_setup_done", True),
            mock.patch(
                "secure_aggregation.run_mp_spdz_parties", return_value=("7\n", {})
            ) as parties,
        ):
            self.assertEqual(service._run_mpc_computation("test program"), "7\n")
        commands = parties.call_args.args[0]
        self.assertEqual(len(commands), 3)
        for rank, command in enumerate(commands):
            self.assertEqual(command[0], str(selected))
            self.assertEqual(command[1], str(rank))
            self.assertEqual(command[-2:], ["-R", "64"])
        self.assertEqual(service.last_execution_receipt["protocol"], "ps-rep-ring")


if __name__ == "__main__":
    unittest.main()
