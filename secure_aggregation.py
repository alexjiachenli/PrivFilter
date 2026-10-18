"""MP-SPDZ final aggregation for the RoFL and ELSA baseline configurations."""

import os
import sys
import torch
import numpy as np
import subprocess
import tempfile
import json
import shutil
import time
import hashlib
import re
from typing import List, Dict, Any
import logging
from collections import OrderedDict
from pathlib import Path
from device import device
from mp_spdz_runtime import (
    MpSpdzPartyExecutionError,
    mp_spdz_runtime_lock,
    require_mp_spdz_runtime_lock,
    run_mp_spdz_parties,
)

logger = logging.getLogger("logger")

RING_BITS = 64
SCALE = 10**4
MPC_VECTOR_CHUNK = 1 << 20
_STRICT_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_MIN_RING_INTEGER = -(1 << (RING_BITS - 1))
_MAX_RING_INTEGER = (1 << (RING_BITS - 1)) - 1


def _strict_integer_output(output: str, *, context: str) -> List[int]:
    """Parse decimal integer output within the signed-ring range."""

    if not isinstance(output, str):
        raise RuntimeError(f"{context} must be text")
    values: List[int] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        token = raw_line.strip()
        if not token:
            continue
        if _STRICT_DECIMAL_INTEGER.fullmatch(token) is None:
            raise RuntimeError(
                f"{context} contains a non-integer token: line={line_number}, token={token!r}"
            )
        value = int(token, 10)
        if value < _MIN_RING_INTEGER or value > _MAX_RING_INTEGER:
            raise RuntimeError(
                f"{context} exceeds the {RING_BITS}-bit signed ring range: line={line_number}"
            )
        values.append(value)
    return values


def _strict_binary_output(blob: bytes, *, context: str) -> np.ndarray:
    """Parse little-endian int64 output.

    INTOUTPUT writes eight-byte records; partial records are rejected.
    """

    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise RuntimeError(f"{context} must be binary output")
    blob = bytes(blob)
    if len(blob) % 8 != 0:
        raise RuntimeError(f"{context} byte length must be a multiple of 8: bytes={len(blob)}")
    return np.frombuffer(blob, dtype="<i8")


def _validated_nonnegative_count(value: int, *, name: str, allow_zero: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _program_identity(program_code: str, compiler_flags: List[str]) -> tuple[str, str]:
    source_hash = hashlib.sha256(program_code.encode("utf-8")).hexdigest()
    identity_payload = json.dumps(
        {
            "compiler_flags": list(compiler_flags),
            "program_code": program_code,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return source_hash, hashlib.sha256(identity_payload).hexdigest()


def _resolve_mp_spdz_party_binary(mp_spdz_path: str) -> str:
    """Locate the ps-rep-ring executable."""
    root = Path(mp_spdz_path).expanduser().resolve()
    explicit = os.environ.get("MP_SPDZ_PARTY_BINARY")
    if explicit:
        candidate = Path(explicit).expanduser()
        candidates = [candidate if candidate.is_absolute() else root / candidate]
    else:
        candidates = [
            root / "ps-rep-ring-party.x",
            root / "bin" / "Linux-amd64" / "ps-rep-ring-party.x",
        ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise FileNotFoundError(
        "Executable ps-rep-ring-party.x not found. Build the MP-SPDZ 0.4.1 "
        "ps-rep-ring-party.x target or set MP_SPDZ_PARTY_BINARY to that executable. "
        "Checked: " + ", ".join(str(path) for path in candidates)
    )


class SecureAggregation:
    _global_compiled_programs = {}
    _global_share_cache = OrderedDict()
    _ssl_setup_done = False

    def __init__(
        self,
        mp_spdz_path: str | None = None,
        max_transient_retries: int = 1,
        retry_backoff_seconds: float = 1.0,
    ):
        self.mp_spdz_path = str(
            Path(
                mp_spdz_path
                or os.environ.get("MP_SPDZ_ROOT")
                or Path(__file__).resolve().parent / "mp-spdz-0.4.1"
            )
            .expanduser()
            .resolve()
        )
        if not Path(self.mp_spdz_path).is_dir():
            raise FileNotFoundError(
                "MP-SPDZ 0.4.1 is required for the corrected RoFL/ELSA aggregation. "
                "Set MP_SPDZ_ROOT or mp_spdz_root to its installation directory."
            )
        self.max_transient_retries = min(1, max(0, int(max_transient_retries)))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.last_execution_receipt = None
        self.last_runtime_lock_receipt = None
        self.temp_dir = tempfile.mkdtemp(prefix="secure_agg_")
        self.player_data_dir = os.path.join(self.temp_dir, "Player-Data")
        os.makedirs(self.player_data_dir, exist_ok=True)

        try:
            self.enable_share_cache = bool(int(os.environ.get("SPDZ_ENABLE_SHARE_CACHE", "0")))
        except Exception:
            self.enable_share_cache = False
        try:
            self.share_cache_max = int(os.environ.get("SPDZ_SHARE_CACHE_MAX", "64"))
        except Exception:
            self.share_cache_max = 64

        self.mp_spdz_available = os.path.exists(self.mp_spdz_path)

        self.persistent_share_dir = os.path.join(self.mp_spdz_path, "Player-Data-Persistent")
        os.makedirs(self.persistent_share_dir, exist_ok=True)

        logger.info(f"SecureAggregation temporary directory: {self.temp_dir}")
        logger.info(f"MP-SPDZ runtime: {self.mp_spdz_path}")
        logger.info(f"Player-Data directory: {self.player_data_dir}")
        logger.info(f"Persistent share directory: {self.persistent_share_dir}")

    def cleanup(self):
        try:
            if hasattr(self, "temp_dir") and self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"Removed SecureAggregation temporary directory: {self.temp_dir}")

            try:
                SecureAggregation._global_share_cache.clear()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"SecureAggregation cleanup failed: {e}")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _generate_client_hash(self, client_weight: Dict[str, torch.Tensor]) -> str:
        """Build an in-memory cache key from tensor shapes, statistics and sampled values."""
        hash_parts = []
        for name in sorted(client_weight.keys()):
            tensor = client_weight[name]
            shape_str = str(tuple(tensor.shape))

            t_float = tensor.detach().to(torch.float32)

            if t_float.numel() > 0:
                mean_val = float(t_float.mean().item())
                std_val = float(t_float.std(unbiased=False).item())
            else:
                mean_val = 0.0
                std_val = 0.0

            flat = t_float.flatten().cpu()
            total = int(flat.numel())
            sample_size = min(10, total)
            if total > sample_size:
                half = sample_size // 2
                sample_vals = torch.cat([flat[:half], flat[-(sample_size - half) :]])
            else:
                sample_vals = flat

            sample_list = sample_vals[:sample_size].tolist()
            sample_str = ",".join([f"{float(v):.6f}" for v in sample_list])

            tensor_signature = f"{shape_str}|{mean_val:.6f}|{std_val:.6f}|{sample_str}"
            hash_parts.append(f"{name}:{hashlib.md5(tensor_signature.encode()).hexdigest()[:8]}")

        client_str = "|".join(hash_parts)
        return hashlib.md5(client_str.encode()).hexdigest()[:12]

    def _get_cached_shares(
        self, client_hash: str, param_names: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """Retrieve an in-memory share-cache entry and update its LRU position."""
        if not self.enable_share_cache:
            return None
        if client_hash in SecureAggregation._global_share_cache:
            SecureAggregation._global_share_cache.move_to_end(client_hash)
            cached_shares = SecureAggregation._global_share_cache.get(client_hash)
            if cached_shares and all(name in cached_shares for name in param_names):
                return cached_shares
        return None

    def _cache_client_shares(
        self, client_hash: str, param_names: List[str], shares: Dict[str, List[np.ndarray]]
    ):
        """Store an in-memory share-cache entry within the LRU capacity."""
        if not self.enable_share_cache:
            return

        SecureAggregation._global_share_cache[client_hash] = shares
        SecureAggregation._global_share_cache.move_to_end(client_hash)
        try:
            while len(SecureAggregation._global_share_cache) > max(1, int(self.share_cache_max)):
                SecureAggregation._global_share_cache.popitem(last=False)
        except Exception:
            pass

    def _tensor_to_fixed_point(self, tensor: torch.Tensor, scale: int = SCALE) -> List[int]:
        """Quantize a tensor to signed fixed-point integers."""
        flattened = tensor.flatten().cpu().numpy()

        fixed_point = (flattened * scale).astype(np.int64)
        return fixed_point.tolist()

    def _fixed_point_to_tensor(
        self, fixed_point: List[int], shape: tuple, scale: int = SCALE
    ) -> torch.Tensor:
        """Decode fixed-point values into a tensor with the supplied shape."""

        float_array = np.array(fixed_point, dtype=np.float32) / scale

        tensor = torch.from_numpy(float_array.reshape(shape)).to(device)
        return tensor

    def _prepare_mpc_program(self, param_sizes: Dict[str, int]) -> str:
        """Generate the binary-I/O secure-sum program.

        Each party supplies a little-endian int64 share vector. Fixed-size SIMD
        chunks bound register use. Averaging is applied after reconstruction.
        """
        total = int(sum(int(size) for size in param_sizes.values()))
        if total <= 0:
            raise ValueError("MP-SPDZ aggregation requires at least one parameter")
        lines = ["from Compiler.types import sint", ""]
        for base in range(0, total, MPC_VECTOR_CHUNK):
            size = min(MPC_VECTOR_CHUNK, total - base)
            lines.append(f"_a = sint.get_input_from(0, size={size}, binary=True)")
            lines.append(f"_b = sint.get_input_from(1, size={size}, binary=True)")
            lines.append(f"_c = sint.get_input_from(2, size={size}, binary=True)")
            lines.append("(_a + _b + _c).reveal().binary_output(0)")
        return "\n".join(lines) + "\n"

    def _write_client_inputs(
        self, client_weights: List[Dict[str, torch.Tensor]], param_names: List[str]
    ) -> Dict[str, int]:
        """Split client weights into additive shares over Z_2^RING_BITS.

        Materialize one per-party sum in param_names order as little-endian int64
        input for sint.get_input_from(..., binary=True).
        """
        require_mp_spdz_runtime_lock(self.mp_spdz_path)
        os.makedirs(self.player_data_dir, exist_ok=True)

        param_sizes: Dict[str, int] = {}
        for name in param_names:
            t = client_weights[0][name]
            param_sizes[name] = int(np.prod(t.shape))

        zero_vecs = {
            name: [np.zeros(param_sizes[name], dtype=np.uint64) for _ in range(3)]
            for name in param_names
        }

        # Sample full-ring additive masks from a CSPRNG-seeded stream.
        rng = np.random.default_rng()

        cache_hits = 0
        cache_misses = 0
        max_acc_est = 0

        for cid, weights in enumerate(client_weights):
            client_hash = self._generate_client_hash(weights)
            cached_shares = self._get_cached_shares(client_hash, param_names)

            if cached_shares is not None:
                cache_hits += 1
                for name in param_names:
                    for party in range(3):
                        zero_vecs[name][party] += cached_shares[name][party]
                continue

            cache_misses += 1
            client_shares = {}
            for name in param_names:
                flat = weights[name].detach().cpu().numpy().astype(np.float64).reshape(-1)
                if not np.isfinite(flat).all():
                    raise ValueError(f"Parameter {name} contains NaN/Inf and cannot be quantized")
                fixed = np.round(flat * SCALE).astype(np.int64)

                max_acc_est = max(max_acc_est, int(np.max(np.abs(fixed))) * len(client_weights))

                r0 = rng.integers(0, 1 << RING_BITS, size=fixed.size, dtype=np.uint64)
                r1 = rng.integers(0, 1 << RING_BITS, size=fixed.size, dtype=np.uint64)
                r2 = fixed.view(np.uint64) - r0 - r1

                client_shares[name] = [r0.copy(), r1.copy(), r2.copy()]
                zero_vecs[name][0] += r0
                zero_vecs[name][1] += r1
                zero_vecs[name][2] += r2

            self._cache_client_shares(client_hash, param_names, client_shares)

        if max_acc_est > (1 << 62):
            logger.warning(
                "Aggregate approaches the signed 64-bit range; reduce SCALE or bound the input values."
            )

        spdz_pd = os.path.join(self.mp_spdz_path, "Player-Data")
        os.makedirs(spdz_pd, exist_ok=True)
        for pid in range(3):
            payload = (
                np.concatenate([zero_vecs[name][pid] for name in param_names])
                .view(np.int64)
                .astype("<i8", copy=False)
                .tobytes()
            )
            staged = os.path.join(self.player_data_dir, f"Input-Binary-P{pid}-0")
            with open(staged, "wb") as handle:
                handle.write(payload)
            shutil.copyfile(staged, os.path.join(spdz_pd, f"Input-Binary-P{pid}-0"))

        return param_sizes

    def _run_mpc_computation(
        self,
        program_code: str,
        program_name: str = "secure_aggregation",
        binary_output: bool = False,
    ):
        """Compile and execute ps-rep-ring aggregation.

        binary_output returns the raw int64 Binary-Output-P0-0 file as bytes;
        text output returns decimal stdout.
        """

        require_mp_spdz_runtime_lock(self.mp_spdz_path)

        compile_script = os.path.join(self.mp_spdz_path, "compile.py")
        if not os.path.exists(compile_script):
            raise FileNotFoundError(f"MP-SPDZ compiler not found: {compile_script}")
        executable_path = _resolve_mp_spdz_party_binary(self.mp_spdz_path)

        # Bind program-cache identity to source and compiler flags.
        compiler_flags = ["-R", str(RING_BITS)]
        program_hash, program_identity_hash = _program_identity(program_code, compiler_flags)
        runtime_program_name = f"{program_name}_{program_identity_hash}"
        canonical_backend = os.path.realpath(self.mp_spdz_path)
        cache_key = (canonical_backend, runtime_program_name)

        bytecode_file = os.path.join(
            self.mp_spdz_path, "Programs", "Bytecode", f"{runtime_program_name}-0.bc"
        )
        schedule_file = os.path.join(
            self.mp_spdz_path, "Programs", "Schedules", f"{runtime_program_name}.sch"
        )

        program_file = os.path.join(self.mp_spdz_path, f"{runtime_program_name}.mpc")

        with open(program_file, "w", encoding="utf-8", newline="") as f:
            f.write(program_code)

        need_compile = True
        cache_entry = SecureAggregation._global_compiled_programs.get(cache_key)
        if isinstance(cache_entry, dict):
            expected_metadata = {
                "program_identity_sha256": program_identity_hash,
                "program_source_sha256": program_hash,
                "compiler_flags": compiler_flags,
                "program_file": program_file,
                "bytecode_file": bytecode_file,
                "schedule_file": schedule_file,
            }
            metadata_matches = all(
                cache_entry.get(key) == value for key, value in expected_metadata.items()
            )
            artifact_hashes_match = False
            if metadata_matches:
                try:
                    artifact_hashes_match = (
                        _sha256_file(program_file) == program_hash
                        and _sha256_file(bytecode_file) == cache_entry.get("bytecode_sha256")
                        and _sha256_file(schedule_file) == cache_entry.get("schedule_sha256")
                    )
                except (FileNotFoundError, OSError):
                    artifact_hashes_match = False
            if metadata_matches and artifact_hashes_match:
                need_compile = False
            else:
                # Recompile invalid cache entries while holding the runtime lock.
                SecureAggregation._global_compiled_programs.pop(cache_key, None)

        if need_compile:
            compile_cmd = [
                "python3",
                "compile.py",
                *compiler_flags,
                os.path.basename(program_file),
            ]

            # Require fresh bytecode and schedule artifacts after compilation.
            for artifact_path in (bytecode_file, schedule_file):
                try:
                    os.remove(artifact_path)
                except FileNotFoundError:
                    pass

            start_time = time.time()
            try:
                subprocess.run(
                    compile_cmd,
                    cwd=self.mp_spdz_path,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                compile_time = time.time() - start_time

                if not os.path.isfile(bytecode_file) or not os.path.isfile(schedule_file):
                    raise RuntimeError(
                        "MP-SPDZ compilation did not produce the expected schedule and bytecode"
                    )

                # Record the source identity and compiled-artifact hashes.
                SecureAggregation._global_compiled_programs[cache_key] = {
                    "timestamp": time.time(),
                    "program_identity_sha256": program_identity_hash,
                    "program_source_sha256": program_hash,
                    "compiler_flags": list(compiler_flags),
                    "program_file": program_file,
                    "bytecode_file": bytecode_file,
                    "bytecode_sha256": _sha256_file(bytecode_file),
                    "schedule_file": schedule_file,
                    "schedule_sha256": _sha256_file(schedule_file),
                }

            except subprocess.CalledProcessError as e:
                logger.error(f"MP-SPDZ compilation failed: {e.stderr}")
                raise
            except subprocess.TimeoutExpired:
                logger.error("MP-SPDZ compilation timed out after 30 minutes")
                raise

        if not SecureAggregation._ssl_setup_done:
            try:
                ssl_setup_script = os.path.join(self.mp_spdz_path, "Scripts", "setup-ssl.sh")
                player_data_dir = os.path.join(self.mp_spdz_path, "Player-Data")

                ssl_files_exist = all(
                    os.path.exists(os.path.join(player_data_dir, f"P{i}.pem")) for i in range(3)
                )

                if not ssl_files_exist and os.path.exists(ssl_setup_script):
                    subprocess.run(
                        ["bash", ssl_setup_script, "3"], cwd=self.mp_spdz_path, check=True
                    )

                SecureAggregation._ssl_setup_done = True
            except Exception as ssl_e:
                logger.warning(f"SSL setup failed: {ssl_e}")

        base_port = 37000

        commands = [
            [
                executable_path,
                str(party_id),
                runtime_program_name,
                "-pn",
                str(base_port),
                "-h",
                "localhost",
                "-R",
                "64",
            ]
            for party_id in range(3)
        ]
        binary_output_path = os.path.join(self.mp_spdz_path, "Player-Data", "Binary-Output-P0-0")
        if binary_output:
            # Remove stale output before launching parties.
            for pid in range(3):
                try:
                    os.remove(
                        os.path.join(
                            self.mp_spdz_path,
                            "Player-Data",
                            f"Binary-Output-P{pid}-0",
                        )
                    )
                except FileNotFoundError:
                    pass

        try:
            output, receipt = run_mp_spdz_parties(
                commands,
                mp_spdz_path=self.mp_spdz_path,
                base_port=base_port,
                timeout_seconds=300,
                max_transient_retries=self.max_transient_retries,
                retry_backoff_seconds=self.retry_backoff_seconds,
            )
            self.last_execution_receipt = {
                **receipt,
                "protocol": "ps-rep-ring",
                "program_name": runtime_program_name,
                "program_sha256": program_hash,
                "program_identity_sha256": program_identity_hash,
                "compiler_flags": list(compiler_flags),
            }
            if receipt.get("retry_count", 0):
                logger.warning(
                    "[secure_aggregation][MP-SPDZ][retry_receipt] %s",
                    json.dumps(self.last_execution_receipt, sort_keys=True),
                )
            if binary_output:
                try:
                    with open(binary_output_path, "rb") as handle:
                        return handle.read()
                except FileNotFoundError as error:
                    raise RuntimeError(
                        "MP-SPDZ binary output missing: " + binary_output_path
                    ) from error
            return output
        except MpSpdzPartyExecutionError as exc:
            self.last_execution_receipt = {
                **exc.receipt,
                "protocol": "ps-rep-ring",
                "program_name": runtime_program_name,
                "program_sha256": program_hash,
                "program_identity_sha256": program_identity_hash,
                "compiler_flags": list(compiler_flags),
            }
            logger.error("ps-rep-ring execution failed: %s", exc)
            raise RuntimeError(f"ps-rep-ring execution failed: {exc}") from exc

    def _parse_mpc_output(
        self, output, param_names: List[str], original_shapes: Dict[str, tuple], num_clients: int
    ) -> Dict[str, torch.Tensor]:
        """Decode binary or decimal MPC output and reconstruct parameter tensors."""
        client_count = _validated_nonnegative_count(
            num_clients, name="num_clients", allow_zero=False
        )
        result_weights = {}
        scale = SCALE

        missing_shapes = [name for name in param_names if name not in original_shapes]
        if missing_shapes:
            raise RuntimeError(f"MP-SPDZ parameter shapes missing: {missing_shapes}")
        expected_values = sum(int(np.prod(original_shapes[name])) for name in param_names)
        if isinstance(output, str):
            numeric_values = np.asarray(
                _strict_integer_output(output, context="MP-SPDZ aggregation output"),
                dtype=np.int64,
            )
        else:
            numeric_values = _strict_binary_output(output, context="MP-SPDZ aggregation output")
        if len(numeric_values) != expected_values:
            raise RuntimeError(
                "MP-SPDZ aggregation output count mismatch: "
                f"expected={expected_values}, observed={len(numeric_values)}"
            )

        divisor = float(scale) * float(client_count)
        line_idx = 0
        for param_name in param_names:
            if param_name not in original_shapes:
                continue
            shape = original_shapes[param_name]
            size = int(np.prod(shape))
            if size == 0:
                continue
            chunk = numeric_values[line_idx : line_idx + size]
            line_idx += size
            tensor = (
                torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float64))
                .div_(divisor)
                .to(torch.float32)
                .reshape(shape)
                .to(device)
            )
            result_weights[param_name] = tensor

        return result_weights

    def secure_aggregate(
        self, client_weights: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Return the aggregate of the supplied client weight dictionaries."""
        if not client_weights:
            return {}

        mpc_start_time = time.time()

        optimized_weights = list(client_weights)

        param_names = list(optimized_weights[0].keys())
        original_shapes = {name: optimized_weights[0][name].shape for name in param_names}

        try:
            with mp_spdz_runtime_lock(
                self.mp_spdz_path, operation="secure_aggregate"
            ) as lock_receipt:
                self.last_runtime_lock_receipt = lock_receipt

                input_start_time = time.time()
                param_sizes = self._write_client_inputs(optimized_weights, param_names)
                input_time = time.time() - input_start_time

                program_start_time = time.time()
                program_code = self._prepare_mpc_program(param_sizes)
                program_time = time.time() - program_start_time

                computation_start_time = time.time()
                mpc_output = self._run_mpc_computation(program_code, binary_output=True)
                computation_time = time.time() - computation_start_time

            parse_start_time = time.time()
            aggregated_weights = self._parse_mpc_output(
                mpc_output, param_names, original_shapes, len(optimized_weights)
            )
            parse_time = time.time() - parse_start_time

            total_mpc_time = time.time() - mpc_start_time
            logger.info(f"MPC total time: {total_mpc_time:.4f}s")

            if not aggregated_weights:
                raise RuntimeError("Secure aggregation returned no parameters")

            return aggregated_weights

        except Exception as e:
            total_mpc_time = time.time() - mpc_start_time
            logger.error(f"Secure aggregation failed after {total_mpc_time:.4f}s: {str(e)}")
            raise
