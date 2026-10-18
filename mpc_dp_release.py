"""Paired noisy aggregate releases for PrivFilter."""

import copy
import hashlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cryptgpu_runtime import activate_runtime as activate_cryptgpu_runtime
from differential_privacy import evaluate_accountant_for_sigma
from memory_utils import state_dict_to_cpu
from mpc_l2_clip import (
    LOCAL_INTEGER_CLIP,
    MPC_INTEGER_CLIP,
    MPC_NEAREST_CLIP,
    MPC_CLIP_MODES,
    INTEGER_CLIP_MODES,
    nearest_inner_radius,
    cryptgpu_l2_clip,
    integer_l2_clip_reference,
)
from mpc_clip_cache import ClipShareCache, load_party_clip, save_party_clip


logger = logging.getLogger("logger")

DP_RELEASE_CLIENT_LDP = "client_ldp"
DP_RELEASE_MPC_ASSISTED = "mpc_per_client_dp_debug"
DP_RELEASE_MPC_AGGREGATE_REF = "mpc_aggregate_ref_dp_secure_clip_debug"
DP_RELEASE_MPC_AGGREGATE_FAST = "mpc_aggregate_ref_dp_fast"

# Independent pairing uses separate streams for the two releases.
# Partition pairing uses (Z_a + Z_b, Z_a), equivalent to disjoint noisy sums
# for a fixed accepted/rejected partition.
MPC_DP_PAIRING_INDEPENDENT = "independent"
MPC_DP_PAIRING_PARTITION = "partition_correlated"
VALID_MPC_DP_PAIRINGS = (MPC_DP_PAIRING_INDEPENDENT, MPC_DP_PAIRING_PARTITION)

PRE_DETECTION_STREAM = ""
POST_DETECTION_STREAM = "filtered_post_detection"


def canonical_mpc_dp_release_pairing(value: Any) -> str:
    mode = str(value or MPC_DP_PAIRING_INDEPENDENT).strip().lower().replace("-", "_")
    aliases = {
        "independent": MPC_DP_PAIRING_INDEPENDENT,
        "disjoint": MPC_DP_PAIRING_INDEPENDENT,
        "two_releases": MPC_DP_PAIRING_INDEPENDENT,
        "partition": MPC_DP_PAIRING_PARTITION,
        "partition_correlated": MPC_DP_PAIRING_PARTITION,
        "correlated": MPC_DP_PAIRING_PARTITION,
        "one_release": MPC_DP_PAIRING_PARTITION,
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_MPC_DP_PAIRINGS:
        raise ValueError(
            f"unsupported mpc_dp_release_pairing={value!r}; expected one of {VALID_MPC_DP_PAIRINGS}"
        )
    return mode


MPC_DP_NOISE_ZERO = "zero_noise_clipping_test"
MPC_DP_NOISE_ZERO_DEBUG = "zero_noise_debug"
MPC_DP_NOISE_INSECURE = "insecure_test_noise"
MPC_DP_NOISE_BOUNDED_DG = "bounded_discrete_gaussian"
MPC_DP_NOISE_DISTRIBUTED_SHARES = "distributed_discrete_gaussian_shares"
MPC_DP_NOISE_PARTY_LOCAL_DGAUSS = "party_local_distributed_dgauss"
MPC_DP_NOISE_PRIVATE_FILE_DGAUSS = "private_file_distributed_dgauss"

MPC_DP_OUTPUT_PER_CLIENT = "per_client"
MPC_DP_OUTPUT_AGGREGATE_REFERENCE = "aggregate_reference"

MPC_DP_BACKEND_SIMULATED = "simulated_mpc"
MPC_DP_BACKEND_CRYPTGPU = "cryptgpu"
_MPC_DP_BACKEND_ALIASES = {
    "cryptgpu": MPC_DP_BACKEND_CRYPTGPU,
    "crypt-gpu": MPC_DP_BACKEND_CRYPTGPU,
    "cgpu": MPC_DP_BACKEND_CRYPTGPU,
    "simulated_mpc": MPC_DP_BACKEND_SIMULATED,
    "simulated": MPC_DP_BACKEND_SIMULATED,
    "simulate": MPC_DP_BACKEND_SIMULATED,
    "simulation": MPC_DP_BACKEND_SIMULATED,
    "mock_mpc": MPC_DP_BACKEND_SIMULATED,
    "plaintext": MPC_DP_BACKEND_SIMULATED,
}


def canonical_mpc_dp_backend(value: Any) -> str:
    """Select CryptGPU 3PC or the local numerical reference."""
    mode = str(value or MPC_DP_BACKEND_CRYPTGPU).strip().lower()
    mode = _MPC_DP_BACKEND_ALIASES.get(mode, mode)
    if mode not in (MPC_DP_BACKEND_CRYPTGPU, MPC_DP_BACKEND_SIMULATED):
        raise NotImplementedError("mpc_dp_backend must be cryptgpu or simulated_mpc")
    return mode


DEFAULT_FIXED_POINT_SCALE = 2**16
DEFAULT_RING_BITS = 64
DEFAULT_SFIX_K = 31


def canonical_dp_release_source(value: Any) -> str:
    """Resolve aggregate release or client-LDP configuration."""
    source = str(value or DP_RELEASE_CLIENT_LDP).strip().lower().replace("-", "_")
    if source in ("client", "local", "ldp", "client_ldp", "client_noise"):
        return DP_RELEASE_CLIENT_LDP
    if source == DP_RELEASE_MPC_AGGREGATE_FAST:
        return source
    raise ValueError("dp_release_source must be client_ldp or mpc_aggregate_ref_dp_fast")


def is_mpc_dp_source(source: Any) -> bool:
    return canonical_dp_release_source(source) in [
        DP_RELEASE_MPC_ASSISTED,
        DP_RELEASE_MPC_AGGREGATE_REF,
        DP_RELEASE_MPC_AGGREGATE_FAST,
    ]


def _is_float_release_tensor(_: str, value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.is_floating_point()


def _tensor_nbytes(state_dict: Dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
    return total


def _secure_random_int64_array(shape: Tuple[int, ...]) -> np.ndarray:
    """Generate additive input-sharing masks independently of DP noise."""
    count = int(np.prod(shape))
    data = os.urandom(count * 8)
    raw = np.frombuffer(data, dtype=np.uint64).copy()
    # Floating-point input transport uses bounded masks; integer clipping uses full-ring shares.
    span = np.uint64(1 << 30)
    signed = (raw % (2 * span)).astype(np.int64) - int(span)
    return signed.reshape(shape)


def quantize_tensor(tensor: torch.Tensor, scale: int) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Cannot quantize tensor containing NaN/Inf")
    return np.round(arr * float(scale)).astype(np.int64)


def audit_dp_release_state_dict(
    model_or_state_dict: Any, num_clients: int = 1, output_path: str = None
) -> Dict[str, Any]:
    if hasattr(model_or_state_dict, "state_dict"):
        model = model_or_state_dict
        state_dict = model.state_dict()
        model_name = model.__class__.__name__
        param_names = set(name for name, _ in model.named_parameters())
        buffer_names = set(name for name, _ in model.named_buffers())
        trainable_names = set(
            name for name, p in model.named_parameters() if getattr(p, "requires_grad", False)
        )
    else:
        model = None
        state_dict = model_or_state_dict
        model_name = type(model_or_state_dict).__name__
        param_names = set()
        buffer_names = set()
        trainable_names = set()

    records = []
    storage_seen: Dict[Tuple[int, int, str], str] = {}
    duplicate_storage = []
    included_total = 0
    trainable_float = 0
    floating_buffer = 0
    nonfloating_buffer = 0
    excluded_nonfloat = 0
    suspicious_keys = []

    for key, value in state_dict.items():
        is_tensor = isinstance(value, torch.Tensor)
        dtype = str(value.dtype) if is_tensor else type(value).__name__
        shape = list(value.shape) if is_tensor else None
        numel = int(value.numel()) if is_tensor else 0
        is_float = bool(is_tensor and value.is_floating_point())
        included = bool(is_float)
        is_parameter = key in param_names if param_names else None
        is_buffer = key in buffer_names if buffer_names else None
        is_trainable = key in trainable_names if trainable_names else None
        kind = "unknown"
        if is_parameter:
            kind = "trainable_parameter" if is_trainable else "frozen_parameter"
        elif is_buffer:
            kind = "floating_buffer" if is_float else "nonfloating_buffer"
        elif is_tensor:
            kind = "floating_tensor_unclassified" if is_float else "nonfloating_tensor_unclassified"

        if included:
            included_total += numel
            if is_parameter or (param_names and key in param_names):
                trainable_float += numel if is_trainable is not False else 0
            elif is_buffer or ("running_mean" in key or "running_var" in key):
                floating_buffer += numel
        elif is_tensor and not is_float:
            excluded_nonfloat += numel
            if is_buffer or "num_batches_tracked" in key:
                nonfloating_buffer += numel

        if is_tensor:
            try:
                storage_key = (
                    int(value.untyped_storage().data_ptr()),
                    int(value.storage_offset()),
                    str(value.device),
                )
                if storage_key in storage_seen:
                    duplicate_storage.append({"key": key, "duplicates": storage_seen[storage_key]})
                else:
                    storage_seen[storage_key] = key
            except Exception:
                pass

        lowered = key.lower()
        if any(
            token in lowered
            for token in ["optimizer", "momentum_buffer", "grad", "exp_avg", "exp_avg_sq"]
        ):
            suspicious_keys.append(key)

        records.append(
            {
                "key": key,
                "shape": shape,
                "numel": numel,
                "dtype": dtype,
                "kind": kind,
                "is_parameter": is_parameter,
                "is_buffer": is_buffer,
                "is_trainable": is_trainable,
                "included": included,
                "excluded_reason": None
                if included
                else ("non_floating" if is_tensor else "non_tensor"),
            }
        )

    total_float_state = sum(r["numel"] for r in records if r["included"])
    expected_total = total_float_state
    inflated = included_total > max(expected_total, 1) * 1.05
    audit = {
        "model_name": model_name,
        "total_serialized_float_coordinates": int(included_total),
        "per_client_included_coordinate_count_d": int(included_total),
        "num_clients_K": int(num_clients),
        "per_round_coordinate_count_K_times_d": int(num_clients) * int(included_total),
        "is_4901770_per_client": int(included_total) == 4_901_770,
        "is_4901770_per_round": int(num_clients) * int(included_total) == 4_901_770,
        "trainable_parameter_float_count": int(trainable_float),
        "floating_buffer_count": int(floating_buffer),
        "nonfloating_buffer_count": int(nonfloating_buffer),
        "excluded_nonfloating_tensor_count": int(excluded_nonfloat),
        "per_key": records,
        "top20_largest_tensors": sorted(records, key=lambda x: x["numel"], reverse=True)[:20],
        "duplicate_tensor_storage": duplicate_storage,
        "suspicious_optimizer_or_gradient_keys": suspicious_keys,
        "optimizer_state_included": any("optimizer" in x.lower() for x in suspicious_keys),
        "gradients_included": any("grad" in x.lower() for x in suspicious_keys),
        "unexpected_large_count_error": bool(inflated or suspicious_keys),
        "notes": [
            "Included tensors are floating state_dict entries only.",
            "Non-floating buffers such as num_batches_tracked are excluded from MPC delta release and copied from the public/global model.",
        ],
    }
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2)
    if audit["unexpected_large_count_error"]:
        raise ValueError(
            f"Unexpected DP release serialization contents: {audit['suspicious_optimizer_or_gradient_keys']}"
        )
    return audit


def _integer_norm_sq_bound(flat_deltas: Sequence[np.ndarray]) -> Tuple[int, int]:
    if not flat_deltas:
        return 0, 0
    max_abs = 0
    max_bound = 0
    for delta in flat_deltas:
        if delta.size == 0:
            continue
        local_max = int(np.max(np.abs(delta.astype(object))))
        bound = int(delta.size) * local_max * local_max
        max_abs = max(max_abs, local_max)
        max_bound = max(max_bound, bound)
    return max_abs, max_bound


@dataclass
class MpcDpReleaseResult:
    client_id: int
    round_id: int
    weights: Dict[str, Any]
    metrics: Dict[str, Any]


def _noise_generation_config(params: Dict[str, Any]) -> Dict[str, Any]:
    nested = params.get("mpc_dp_noise_generation") or {}
    if not isinstance(nested, dict):
        raise ValueError("mpc_dp_noise_generation must be a mapping when provided")
    return nested


def _get_noise_config(params: Dict[str, Any], key: str, default: Any) -> Any:
    nested = _noise_generation_config(params)
    return nested.get(key, params.get(f"mpc_dp_{key}", default))


class BoundedDiscreteGaussianSampler:
    """Finite-precision bounded integer discrete-Gaussian inverse-CDF sampler."""

    def __init__(self, sigma: int, tail_sigma: float = 8.0, cdf_precision_bits: int = 64):
        self.sigma = int(math.ceil(float(sigma)))
        if self.sigma <= 0:
            raise ValueError("Discrete-Gaussian sigma must be positive")
        self.tail_sigma = float(tail_sigma)
        if self.tail_sigma <= 0:
            raise ValueError("tail_sigma must be positive")
        self.cdf_precision_bits = int(cdf_precision_bits)
        if not (32 <= self.cdf_precision_bits <= 64):
            raise ValueError("cdf_precision_bits must be in [32, 64]")
        self.tail_bound = int(math.ceil(self.tail_sigma * float(self.sigma)))
        self.support = np.arange(-self.tail_bound, self.tail_bound + 1, dtype=np.int64)
        self.cdf_table_build_s = 0.0
        started = time.time()
        z = self.support.astype(np.float64)
        weights = np.exp(-(z * z) / (2.0 * float(self.sigma) * float(self.sigma)))
        probs = weights / float(np.sum(weights))
        cumulative = np.cumsum(probs, dtype=np.float64)
        max_u = (1 << self.cdf_precision_bits) - 1
        thresholds = np.rint(cumulative * float(max_u)).astype(object)
        cdf = np.empty(len(thresholds), dtype=np.uint64)
        prev = 0
        for idx, value in enumerate(thresholds):
            intval = int(value)
            if idx == len(thresholds) - 1:
                intval = max_u
            else:
                intval = max(prev + 1, min(max_u, intval))
            cdf[idx] = np.uint64(intval)
            prev = intval
        self.cdf = cdf
        self.cdf_table_build_s = time.time() - started
        self.tail_probability_bound = float(
            math.erfc(self.tail_bound / (math.sqrt(2.0) * float(self.sigma)))
        )
        self.cdf_quantization_error_bound = float(
            len(self.support) / float(1 << self.cdf_precision_bits)
        )
        self.statistical_distance_bound = float(
            self.tail_probability_bound + self.cdf_quantization_error_bound
        )

    def sample_from_uint64(self, uniforms: np.ndarray) -> np.ndarray:
        if uniforms.dtype != np.uint64:
            uniforms = uniforms.astype(np.uint64, copy=False)
        indices = np.searchsorted(self.cdf, uniforms, side="left")
        return self.support[indices].astype(np.int64, copy=False)

    def metadata(self) -> Dict[str, Any]:
        return {
            "distribution": "bounded_discrete_gaussian",
            "sampler": "bounded_inverse_cdf",
            "sigma_int_party": int(self.sigma),
            "tail_sigma": float(self.tail_sigma),
            "tail_bound_int": int(self.tail_bound),
            "cdf_precision_bits": int(self.cdf_precision_bits),
            "cdf_support_size": int(len(self.support)),
            "cdf_table_build_s": float(self.cdf_table_build_s),
            "sampler_tail_probability_bound": float(self.tail_probability_bound),
            "sampler_cdf_quantization_error_bound": float(self.cdf_quantization_error_bound),
            "sampler_statistical_distance_bound": float(self.statistical_distance_bound),
        }


class PartyLocalDiscreteGaussianNoiseProvider:
    """Generate distributed discrete-Gaussian noise from party-local seed files."""

    def __init__(self, params: Dict[str, Any], sigma_int_target: int, sensitivity_int: int):
        self.params = params
        cfg = _noise_generation_config(params)
        self.provider = str(cfg.get("provider", "party_local_seeded_stream")).strip().lower()
        if self.provider != "party_local_seeded_stream":
            raise NotImplementedError(
                "Only party_local_seeded_stream is implemented for party-local dgauss"
            )
        self.num_parties = int(cfg.get("num_parties", 3))
        self.corruption_threshold = int(cfg.get("corruption_threshold", 1))
        default_hmin = self.num_parties - self.corruption_threshold
        self.min_honest_noise_parties = int(cfg.get("min_honest_noise_parties", default_hmin))
        if self.num_parties <= 0:
            raise ValueError("num_parties must be positive")
        if self.num_parties != 3:
            raise NotImplementedError("CryptGPU aggregate release expects exactly 3 parties")
        if not (1 <= self.min_honest_noise_parties <= self.num_parties):
            raise ValueError("min_honest_noise_parties must be in [1, num_parties]")
        self.calibration = str(cfg.get("calibration", "honest_majority_hmin"))
        self.distribution = str(cfg.get("distribution", "bounded_discrete_gaussian"))
        self.sampler_name = str(cfg.get("sampler", "bounded_inverse_cdf"))
        if (
            self.distribution != "bounded_discrete_gaussian"
            or self.sampler_name != "bounded_inverse_cdf"
        ):
            raise NotImplementedError(
                "Only bounded_discrete_gaussian/bounded_inverse_cdf is implemented"
            )
        configured_target_sigma = cfg.get("sigma_int_target", "auto")
        if configured_target_sigma in [None, "auto"]:
            self.sigma_int_target = int(math.ceil(float(sigma_int_target)))
        else:
            self.sigma_int_target = int(configured_target_sigma)
        if self.sigma_int_target <= 0:
            raise RuntimeError("party_local_distributed_dgauss requires positive sigma_int_target")
        configured_party_sigma = cfg.get("sigma_int_party", "auto")
        if configured_party_sigma in [None, "auto"]:
            self.sigma_int_party = int(
                math.ceil(self.sigma_int_target / math.sqrt(self.min_honest_noise_parties))
            )
        else:
            self.sigma_int_party = int(configured_party_sigma)
        self.sensitivity_int = int(sensitivity_int)
        self.tail_sigma = float(cfg.get("tail_sigma", 8.0))
        self.cdf_precision_bits = int(cfg.get("cdf_precision_bits", 64))
        self.fail_if_seed_missing = bool(cfg.get("fail_if_seed_missing", True))
        self.allow_coordinator_seed_generation = bool(
            cfg.get("allow_coordinator_seed_generation", False)
        )
        if self.allow_coordinator_seed_generation:
            raise RuntimeError(
                "Coordinator-side seed generation is rejected in paper-level MPC-DP mode"
            )
        self.run_name = str(
            params.get("name")
            or params.get("experiment_name")
            or cfg.get("run_name")
            or "unnamed_run"
        )
        self.domain_context = str(cfg.get("domain_context", self.run_name))
        default_seed_dir = os.path.join(
            ".private",
            "noise_seeds",
            self.run_name,
            "party_{party}",
        )
        self.seed_dir_template = str(
            cfg.get("seed_dir", params.get("mpc_dp_noise_seed_dir", default_seed_dir))
        )
        self.chunk_size = int(
            cfg.get("chunk_size", params.get("mpc_dp_noise_generation_chunk_size", 65536))
        )
        self.delete_temp_noise_inputs_after_round = bool(
            cfg.get("delete_temp_noise_inputs_after_round", True)
        )
        self.sampler = BoundedDiscreteGaussianSampler(
            sigma=self.sigma_int_party,
            tail_sigma=self.tail_sigma,
            cdf_precision_bits=self.cdf_precision_bits,
        )

    def seed_file_for_party(self, party_id: int) -> str:
        seed_dir = self.seed_dir_template.format(
            run_name=self.run_name,
            party=int(party_id),
            pid=int(party_id),
        )
        return os.path.join(seed_dir, "master_seed.bin")

    def load_master_seed(self, party_id: int) -> bytes:
        path = self.seed_file_for_party(party_id)
        if not os.path.exists(path):
            if self.fail_if_seed_missing:
                raise RuntimeError(f"Missing party-local noise seed for party {party_id}: {path}")
            raise RuntimeError("Missing party seed and fail_if_seed_missing=false is not supported")
        with open(path, "rb") as handle:
            seed = handle.read()
        if len(seed) < 32:
            raise RuntimeError(f"Party seed file is too short for party {party_id}: {path}")
        return seed

    @staticmethod
    def _uniform_uint64(
        seed: bytes,
        round_id: int,
        party_id: int,
        offset: int,
        count: int,
        domain_context: bytes = b"",
        stream_label: bytes = b"",
    ) -> np.ndarray:
        """Expand a party seed into domain-separated uniform draws.

        The stream label and round identify independent release streams.
        """
        if isinstance(domain_context, str):
            domain_context = domain_context.encode("utf-8")
        if isinstance(stream_label, str):
            stream_label = stream_label.encode("utf-8")
        context_digest = hashlib.sha256(domain_context or b"default").digest()
        label = (
            b"Priv-FilterFL|mpc-aggregate-ref-dp|dgauss-v1|"
            + context_digest
            + int(round_id).to_bytes(8, "little", signed=False)
            + int(party_id).to_bytes(4, "little", signed=False)
            + int(offset).to_bytes(8, "little", signed=False)
            + int(count).to_bytes(8, "little", signed=False)
        )
        if stream_label:
            label += b"|stream|" + hashlib.sha256(stream_label).digest()
        data = hashlib.shake_256(seed + label).digest(int(count) * 8)
        return np.frombuffer(data, dtype=np.uint64).copy()

    def generate_party_noise_file(
        self, party_id: int, round_id: int, count: int, output_path: str
    ) -> Dict[str, Any]:
        started = time.time()
        seed = self.load_master_seed(party_id)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        written = 0
        with open(output_path, "w", encoding="utf-8") as handle:
            while written < int(count):
                n = min(self.chunk_size, int(count) - written)
                uniforms = self._uniform_uint64(
                    seed,
                    round_id,
                    party_id,
                    written,
                    n,
                    domain_context=self.domain_context,
                )
                samples = self.sampler.sample_from_uint64(uniforms)
                np.savetxt(handle, samples, fmt="%d")
                written += n
        return {
            "party_id": int(party_id),
            "round_id": int(round_id),
            "count": int(count),
            "output_path": os.path.abspath(output_path),
            "noise_generation_s": time.time() - started,
            "temp_noise_input_bytes": int(os.path.getsize(output_path)),
            "seed_file": os.path.abspath(self.seed_file_for_party(party_id)),
            "coordinator_generated_noise": False,
            "party_local_noise": True,
            "client_generated_noise": False,
            **self.metadata(),
        }

    def generate_party_noise_summed(
        self,
        party_id: int,
        round_id: int,
        count: int,
        stream_labels: Sequence[str],
    ) -> np.ndarray:
        """Sum independent noise streams for one party.

        The pre-filtering release uses Z_a + Z_b and the filtered release uses Z_a.
        Their difference retains Z_b. For a fixed partition, this pair is an
        invertible transform of disjoint accepted and rejected releases.
        """

        labels = list(stream_labels)
        if not labels:
            raise ValueError("at least one noise stream label is required")
        total = np.zeros(int(count), dtype=np.int64)
        for label in labels:
            total += self.generate_party_noise_array(
                party_id=party_id, round_id=round_id, count=count, stream_label=label
            )
        return total

    def generate_party_noise_array(
        self, party_id: int, round_id: int, count: int, stream_label: str = ""
    ) -> np.ndarray:
        """Evaluate the party sampler locally with identical stream and chunk offsets."""

        seed = self.load_master_seed(party_id)
        pieces = []
        written = 0
        while written < int(count):
            n = min(self.chunk_size, int(count) - written)
            uniforms = self._uniform_uint64(
                seed,
                round_id,
                party_id,
                written,
                n,
                domain_context=self.domain_context,
                stream_label=stream_label,
            )
            pieces.append(self.sampler.sample_from_uint64(uniforms))
            written += n
        if not pieces:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(pieces).astype(np.int64, copy=False)

    def generate_party_noise_file_via_utility(
        self,
        party_id: int,
        round_id: int,
        count: int,
        output_path: str,
        output_format: str = "text",
        stream_labels: Sequence[str] = (PRE_DETECTION_STREAM,),
    ) -> Dict[str, Any]:
        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scripts",
            "generate_party_noise_inputs_for_round.py",
        )
        cmd = [
            sys.executable,
            script,
            "--seed-file",
            self.seed_file_for_party(party_id),
            "--output",
            output_path,
            "--round-id",
            str(int(round_id)),
            "--party-id",
            str(int(party_id)),
            "--count",
            str(int(count)),
            "--sigma-int-party",
            str(int(self.sigma_int_party)),
            "--tail-sigma",
            str(float(self.tail_sigma)),
            "--cdf-precision-bits",
            str(int(self.cdf_precision_bits)),
            "--chunk-size",
            str(int(self.chunk_size)),
            "--domain-context",
            self.domain_context,
            "--format",
            str(output_format),
        ]
        # Each party sums its streams locally.
        for label in stream_labels:
            cmd += ["--stream-label", str(label)]
        started = time.time()
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Party-local noise input generator failed\n"
                f"cmd={' '.join(cmd)}\nstdout={exc.stdout}\nstderr={exc.stderr}"
            ) from exc
        try:
            meta = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Party-local generator returned non-JSON output: {proc.stdout[:500]}"
            ) from exc
        meta.update(
            {
                "party_id": int(party_id),
                "round_id": int(round_id),
                "count": int(count),
                "output_path": os.path.abspath(output_path),
                "noise_generation_s": float(meta.get("noise_generation_s", 0.0)),
                "party_utility_wall_s": time.time() - started,
                "temp_noise_input_bytes": int(
                    meta.get(
                        "bytes", os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    )
                ),
                "seed_file": os.path.abspath(self.seed_file_for_party(party_id)),
                "domain_context": self.domain_context,
                "coordinator_generated_noise": False,
                "party_local_noise": True,
                "client_generated_noise": False,
                **self.metadata(),
            }
        )
        return meta

    def metadata(self) -> Dict[str, Any]:
        effective_min = math.sqrt(float(self.min_honest_noise_parties)) * float(
            self.sigma_int_party
        )
        all_honest = math.sqrt(float(self.num_parties)) * float(self.sigma_int_party)
        return {
            "noise_seed_mode": "party_local_seeded_stream",
            "noise_domain_context": self.domain_context,
            "num_noise_parties": int(self.num_parties),
            "corruption_threshold": int(self.corruption_threshold),
            "min_honest_noise_parties": int(self.min_honest_noise_parties),
            "noise_calibration": self.calibration,
            "sigma_int_target": int(self.sigma_int_target),
            "sigma_int_party": int(self.sigma_int_party),
            "effective_min_honest_sigma_int": float(effective_min),
            "all_honest_sigma_int": float(all_honest),
            "distributed_noise_assumption": (
                "The release is MPC-assisted under honest-majority correctness for clipping and aggregation, "
                "with distributed finite-precision discrete-Gaussian noise generation. Privacy of the opened "
                "aggregate reference holds against the coordinator if at least H_min noise-contributing MPC "
                "parties honestly and independently sample calibrated noise shares and do not reveal their "
                "noise seeds/shares to the coordinator."
            ),
            **self.sampler.metadata(),
        }


CRYPTGPU_DP_JOB_SCHEMA = "fldd-cryptgpu-dp-aggregate-v1"


def _picklable_noise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Select JSON-serializable configuration fields for spawned noise providers."""

    keep: Dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str):
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        keep[key] = value
    return keep


def _cryptgpu_dp_aggregate_worker(job_dir: str) -> Any:
    """Evaluate one aggregate release inside a CryptGPU party.

    The precision-zero circuit computes sum(client deltas) + sum(party noise)
    over Z_2^64 and reveals only the noisy total. from_shares preserves the
    already-quantized integers; reveal returns the raw ring sum without encoder
    decoding. Each party draws its noise from its own seed file.

    This module-level worker is importable by multiprocessing spawn.
    """

    import json as _json
    import os as _os

    import numpy as _np

    from cryptgpu_runtime import activate_runtime as _activate

    _activate()
    import torch as _torch
    from crypten import comm as _comm
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

    with open(_os.path.join(job_dir, "job.json"), "r", encoding="utf-8") as handle:
        job = _json.load(handle)
    if job.get("schema") != CRYPTGPU_DP_JOB_SCHEMA:
        raise RuntimeError(f"Unexpected CryptGPU DP job schema: {job.get('schema')!r}")

    rank = int(_comm.get().get_rank())
    world_size = int(_comm.get().get_world_size())
    if world_size != 3:
        raise RuntimeError(f"CryptGPU MPC-DP aggregate expects a 3-party session, got {world_size}")

    total_values = int(job["total_values"])

    cache = job.get("clip_cache")
    cache_hit = bool(cache and cache["action"] == "read")
    cache_started = time.perf_counter()
    delta_share = (
        load_party_clip(cache, rank)
        if cache_hit
        else _np.load(_os.path.join(job_dir, f"delta_share_{rank}.npy"))
    )
    cache_read_s = time.perf_counter() - cache_started if cache_hit else 0.0
    if delta_share.dtype != _np.int64:
        raise RuntimeError(f"delta share dtype must be int64, got {delta_share.dtype}")

    mask_share_dir = job.get("mask_share_dir")
    clip_timing = {
        "mpc_clip_s": 0.0,
        "mpc_clip_norm_s": 0.0,
        "mpc_clip_factor_s": 0.0,
        "mpc_clip_rescale_s": 0.0,
        "mpc_clip_cache_read_s": cache_read_s,
        "mpc_clip_cache_write_s": 0.0,
    }
    if job.get("clip_mode") in MPC_CLIP_MODES:
        num_clients = int(job["num_clients"])
        if delta_share.size != num_clients * total_values:
            raise RuntimeError("secure clipping requires every individual client share")
        device = _torch.device("cuda", 0) if _torch.cuda.is_available() else _torch.device("cpu")
        block = _torch.from_numpy(delta_share.reshape(num_clients, total_values)).to(device)
        clipped = ArithmeticSharedTensor.from_shares(block, precision=0)
        if not cache_hit:
            clipped, measured = cryptgpu_l2_clip(
                clipped,
                int(job["clip_radius_int"]),
                int(job["clip_chunk_size"]),
                mode=job["clip_mode"],
            )
            clip_timing.update(measured)
        if cache and cache["action"] == "write":
            cache_started = time.perf_counter()
            own_share = clipped.share
            if hasattr(own_share, "tensor"):
                own_share = own_share.tensor()
            save_party_clip(cache, rank, own_share.detach().cpu().numpy())
            clip_timing["mpc_clip_cache_write_s"] = time.perf_counter() - cache_started
            del own_share
        raw_share = clipped.share if mask_share_dir else clipped.sum(dim=0).share
        if hasattr(raw_share, "tensor"):
            raw_share = raw_share.tensor()
        delta_share = raw_share.detach().cpu().numpy().astype(_np.int64, copy=False)
        del clipped, block, raw_share
    masked_sum_ct = None
    accepted_count = None
    if not mask_share_dir:
        # Open-mask inputs are pre-summed over accepted clients.
        if delta_share.size != total_values:
            raise RuntimeError(f"delta share size mismatch: {delta_share.size} != {total_values}")
        local = delta_share.reshape(-1)
    else:
        # Apply secret XOR-shared acceptance bits to the full cohort.
        # mixed_mul operates on precision-zero ring integers.
        from crypten.mpc.primitives import resharing as _resharing
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        from crypten.mpc.primitives.binary import BinarySharedTensor

        num_clients = int(job["num_clients"])
        if delta_share.size != num_clients * total_values:
            raise RuntimeError(
                f"delta share must be {num_clients} x {total_values} int64 values, "
                f"got {delta_share.size}"
            )
        mask_raw = _np.load(_os.path.join(mask_share_dir, f"mask_share_{rank}.npy"))
        if mask_raw.size != num_clients:
            raise RuntimeError(
                f"accept-mask share has {mask_raw.size} bits for {num_clients} clients"
            )
        device = _torch.device("cuda", 0) if _torch.cuda.is_available() else None
        block = _torch.from_numpy(
            _np.ascontiguousarray(delta_share.reshape(num_clients, total_values))
        )
        mask_t = _torch.from_numpy(_np.ascontiguousarray(mask_raw))
        if device is not None:
            block = block.to(device)
            mask_t = mask_t.to(device)
        deltas = ArithmeticSharedTensor.from_shares(block, precision=0)
        mask = BinarySharedTensor.from_shares(mask_t, src=rank).reshape([num_clients, 1])
        exact_integer_mask = job.get("clip_mode") in INTEGER_CLIP_MODES
        if exact_integer_mask:
            from mpc_l2_clip import _bit_to_arithmetic

            mask_arith = _bit_to_arithmetic(mask)
            masked_sum_ct = (deltas * mask_arith).sum(dim=0)
        else:
            masked_sum_ct = _resharing.mixed_mul(deltas, mask).sum(dim=0)
        ones = ArithmeticSharedTensor.from_shares(
            _torch.ones(num_clients, 1, dtype=_torch.int64, device=block.device)
            if rank == 0
            else _torch.zeros(num_clients, 1, dtype=_torch.int64, device=block.device),
            precision=0,
        )
        count_ct = (
            mask_arith.sum() if exact_integer_mask else _resharing.mixed_mul(ones, mask).sum()
        )

        # Keep the masked aggregate shared until each party adds its noise.
        # mixed_mul introduces an encoder scale; remove it with secure truncation
        # before reading the raw additive shares.
        masked_arith = masked_sum_ct if exact_integer_mask else masked_sum_ct._tensor
        product_scale = int(masked_arith.encoder._scale)
        if product_scale > 1:
            masked_arith = _resharing.truncation(masked_arith, product_scale)
        masked_share = masked_arith.share
        if hasattr(masked_share, "tensor"):
            masked_share = masked_share.tensor()
        local = masked_share.detach().reshape(-1).cpu().numpy().astype(_np.int64, copy=False)

        # Open the accepted count for the averaging denominator.
        accepted_count = float(count_ct.get_plain_text().item())

    noise_source = str(job["noise_source"])
    if noise_source == "party_local_seeded_stream":
        provider = PartyLocalDiscreteGaussianNoiseProvider(
            params=job["noise_params"],
            sigma_int_target=int(job["sigma_int_target"]),
            sensitivity_int=int(job["sensitivity_int"]),
        )
        noise = provider.generate_party_noise_summed(
            party_id=rank,
            round_id=int(job["round_id"]),
            count=total_values,
            stream_labels=[str(label) for label in job.get("stream_labels", [""])],
        )
    elif noise_source == "coordinator_share_file":
        noise = _np.load(_os.path.join(job_dir, f"noise_share_{rank}.npy"))
    else:
        raise NotImplementedError(f"Unsupported CryptGPU DP noise source: {noise_source!r}")
    if noise.dtype != _np.int64 or noise.size != total_values:
        raise RuntimeError(
            f"party noise must be {total_values} int64 values, got {noise.size} of {noise.dtype}"
        )
    local = local + noise

    # Signed int64 overflow implements reduction modulo 2**64.
    share = _torch.from_numpy(_np.ascontiguousarray(local))
    shared = ArithmeticSharedTensor.from_shares(share, precision=0)
    opened = shared.reveal()

    with open(_os.path.join(job_dir, f"clip_timing_{rank}.json"), "w") as handle:
        _json.dump(clip_timing, handle)

    if rank != 0:
        return None

    # File output avoids the result queue's pipe-capacity limit while workers join.
    out_path = _os.path.join(job_dir, "opened.npy")
    _np.save(out_path, opened.cpu().numpy().astype(_np.int64, copy=False))
    return {
        "opened_path": out_path,
        "count": int(opened.numel()),
        "accepted_count": accepted_count,
    }


class MpcDpReleaseService:
    """Create paired aggregate releases with CryptGPU or a local numerical reference.

    Clipping supports bounded integer circuits and floating-point reference inputs.
    """

    def __init__(self, params: Dict[str, Any], mp_spdz_path: str | None = None):
        self.params = params
        self.backend = canonical_mpc_dp_backend(
            params.get("mpc_dp_backend", MPC_DP_BACKEND_CRYPTGPU)
        )
        self.simulated_execution = self.backend == MPC_DP_BACKEND_SIMULATED
        self.cryptgpu_execution = self.backend == MPC_DP_BACKEND_CRYPTGPU
        self.protocol = str(params.get("mpc_dp_protocol", "replicated-field")).strip().lower()
        self.clip_mode = str(params.get("mpc_dp_clip_mode", MPC_NEAREST_CLIP)).strip().lower()
        self.noise_mode = (
            str(params.get("mpc_dp_noise_mode", MPC_DP_NOISE_PARTY_LOCAL_DGAUSS)).strip().lower()
        )
        if self.noise_mode == "zero_noise":
            self.noise_mode = MPC_DP_NOISE_ZERO
        if self.noise_mode == "zero_noise_debug":
            self.noise_mode = MPC_DP_NOISE_ZERO_DEBUG
        if self.noise_mode == "distributed_discrete_gaussian_shares":
            self.noise_mode = MPC_DP_NOISE_PRIVATE_FILE_DGAUSS
        if self.noise_mode not in (
            MPC_DP_NOISE_PARTY_LOCAL_DGAUSS,
            MPC_DP_NOISE_ZERO,
            MPC_DP_NOISE_ZERO_DEBUG,
        ):
            raise NotImplementedError(
                "Only party-local seeded noise and gated zero-noise arithmetic tests are included"
            )
        self.output_mode = (
            str(params.get("mpc_dp_output_mode", MPC_DP_OUTPUT_AGGREGATE_REFERENCE)).strip().lower()
        )
        default_share_mode = (
            "party_local_seeded_stream"
            if self.noise_mode == MPC_DP_NOISE_PARTY_LOCAL_DGAUSS
            else "simulation_debug"
        )
        self.noise_share_mode = (
            str(params.get("mpc_dp_noise_share_mode", default_share_mode)).strip().lower()
        )
        self.adjacency = str(params.get("mpc_dp_adjacency", "add_remove")).strip().lower()
        self.release_pairing = canonical_mpc_dp_release_pairing(
            params.get("mpc_dp_release_pairing")
        )
        self.scale = int(params.get("mpc_dp_fixed_point_scale", DEFAULT_FIXED_POINT_SCALE))
        self.clip_norm = float(params.get("mpc_dp_clip_norm", params.get("ldp_max_grad_norm", 1.0)))
        self.noise_multiplier = float(
            params.get("mpc_dp_noise_multiplier", params.get("ldp_noise_multiplier", 0.0))
        )
        self.statistical_security = int(params.get("mpc_dp_statistical_security", 128))
        self.allow_zero_noise = bool(params.get("allow_zero_noise_mpc_dp_release", False))
        self.debug_zero_noise_run = bool(params.get("debug_zero_noise_run", False))
        self.ring_bits = int(params.get("mpc_dp_ring_bits", DEFAULT_RING_BITS))
        self.sfix_k = int(params.get("mpc_dp_sfix_k", DEFAULT_SFIX_K))
        self.metrics: List[Dict[str, Any]] = []
        self.released_client_ids: List[int] = []
        # Round IDs align with client IDs for paired-release accounting.
        self.released_round_ids: List[int] = []
        self.last_execution_receipt = None

        if self.protocol not in ["replicated-field", "ps-rep-ring"]:
            raise NotImplementedError("mpc_dp_protocol must be replicated-field or ps-rep-ring")
        if self.output_mode != MPC_DP_OUTPUT_AGGREGATE_REFERENCE:
            raise NotImplementedError("This release includes aggregate_reference output only")
        if self.clip_mode not in ["local_reference_clip_then_mpc_sum", *INTEGER_CLIP_MODES]:
            raise NotImplementedError("unknown mpc_dp_clip_mode")
        if self.clip_mode in INTEGER_CLIP_MODES:
            if not self.cryptgpu_execution:
                raise NotImplementedError("integer clipping currently requires cryptgpu")
            if not math.isfinite(self.clip_norm) or not 1 <= math.floor(
                self.clip_norm * self.scale
            ) < (1 << 28):
                raise ValueError("integer clipping radius must be in [1, 2**28)")
        if self.adjacency not in ["add_remove", "replace_one"]:
            raise ValueError("mpc_dp_adjacency must be add_remove or replace_one")
        if self.noise_share_mode not in [
            "party_local_inputs",
            "simulation_debug",
            "party_local_seeded_stream",
        ]:
            raise ValueError(
                "mpc_dp_noise_share_mode must be party_local_inputs, simulation_debug, or party_local_seeded_stream"
            )
        if self.scale != DEFAULT_FIXED_POINT_SCALE:
            raise ValueError("Aggregate releases require mpc_dp_fixed_point_scale=65536")
        if self.ring_bits < 64:
            raise ValueError("Aggregate releases require mpc_dp_ring_bits >= 64")
        if self.sfix_k < 31:
            raise ValueError("mpc_dp_sfix_k must be at least 31 for Q16 clipping")
        if (
            params.get("mpc_dp_public_noise_seed") is not None
            or self.noise_share_mode == "public_seed"
        ):
            raise RuntimeError(
                "Public-seed noise is rejected because the coordinator can subtract it"
            )
        if self.noise_mode in [MPC_DP_NOISE_ZERO, MPC_DP_NOISE_ZERO_DEBUG]:
            if not (self.allow_zero_noise and self.debug_zero_noise_run):
                raise RuntimeError(
                    "zero-noise mpc_dp_noise_mode requires "
                    "allow_zero_noise_mpc_dp_release=true and debug_zero_noise_run=true because it is not a DP mechanism"
                )
        elif self.noise_mode == MPC_DP_NOISE_PARTY_LOCAL_DGAUSS:
            if self.noise_share_mode != "party_local_seeded_stream":
                raise RuntimeError(
                    "party_local_distributed_dgauss requires mpc_dp_noise_share_mode=party_local_seeded_stream"
                )
        else:
            raise NotImplementedError(f"Unsupported mpc_dp_noise_mode={self.noise_mode!r}")
        if self.simulated_execution:
            # Local numerical evaluation has no MPC confidentiality.
            if self.output_mode != MPC_DP_OUTPUT_AGGREGATE_REFERENCE:
                raise NotImplementedError(
                    "mpc_dp_backend=simulated_mpc implements only the aggregate "
                    "reference route; set mpc_dp_output_mode=aggregate_reference"
                )
            if self.clip_mode != "local_reference_clip_then_mpc_sum":
                raise NotImplementedError(
                    "mpc_dp_backend=simulated_mpc implements only "
                    "mpc_dp_clip_mode=local_reference_clip_then_mpc_sum"
                )
            logger.warning(
                "[MPC-DP][simulated] Local numerical-reference execution: "
                "privacy_valid=false, timing_scope=local_reference."
            )
        elif self.cryptgpu_execution:
            # Validate the aggregate output and clipping interfaces.
            if self.output_mode != MPC_DP_OUTPUT_AGGREGATE_REFERENCE:
                raise NotImplementedError(
                    "mpc_dp_backend=cryptgpu implements only the aggregate "
                    "reference route; set mpc_dp_output_mode=aggregate_reference"
                )
            if self.clip_mode not in ("local_reference_clip_then_mpc_sum", *INTEGER_CLIP_MODES):
                raise NotImplementedError(
                    "mpc_dp_backend=cryptgpu implements only "
                    "local_reference_clip_then_mpc_sum or certified integer L2 clipping"
                )
            if int(self.ring_bits) != 64:
                raise NotImplementedError(
                    "mpc_dp_backend=cryptgpu evaluates in Z_2^64; set "
                    f"mpc_dp_ring_bits=64, got {self.ring_bits}"
                )

        self.sensitivity_int = self._sensitivity_int()
        self.sigma_int = int(math.ceil(self.noise_multiplier * self.sensitivity_int))
        self.tail_bound = self._resolve_tail_bound()
        self.temp_dir = tempfile.mkdtemp(prefix="mpc_dp_release_")
        # Clean unconsumed private caches when the service is finalized.
        self._temp_finalizer = weakref.finalize(
            self,
            shutil.rmtree,
            self.temp_dir,
            ignore_errors=True,
        )
        self.clip_cache = ClipShareCache(self.temp_dir)
        self._active_clip_cache = None
        self.player_data_dir = os.path.join(self.temp_dir, "Player-Data")
        os.makedirs(self.player_data_dir, exist_ok=True)
        self.noise_provider = None
        if self.noise_mode == MPC_DP_NOISE_PARTY_LOCAL_DGAUSS:
            self.noise_provider = PartyLocalDiscreteGaussianNoiseProvider(
                params=self.params,
                sigma_int_target=self.sigma_int,
                sensitivity_int=self.sensitivity_int,
            )

    def cleanup(self):
        if hasattr(self, "clip_cache"):
            self.clip_cache.clear()
        if getattr(self, "temp_dir", None):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _resolve_tail_bound(self) -> int:
        raw = self.params.get("mpc_dp_tail_bound", None)
        if raw is None:
            return 0
        return int(raw)

    def _sensitivity_int(self) -> int:
        scaled = float(self.clip_norm) * float(self.scale)
        c_hat = math.floor(scaled) if self.clip_mode in INTEGER_CLIP_MODES else int(round(scaled))
        return c_hat if self.adjacency == "add_remove" else 2 * c_hat

    def _release_keys(self, global_model: Dict[str, Any]) -> List[str]:
        return [key for key, value in global_model.items() if _is_float_release_tensor(key, value)]

    def _serialize_deltas(
        self,
        clean_client_weights: Sequence[Dict[str, Any]],
        global_model: Dict[str, Any],
        client_ids: Sequence[int],
    ) -> Tuple[List[str], Dict[str, Tuple[int, ...]], List[int], List[np.ndarray]]:
        global_cpu = state_dict_to_cpu(global_model)
        clean_cpu = [state_dict_to_cpu(sd) for sd in clean_client_weights]
        param_names = self._release_keys(global_cpu)
        shapes = {name: tuple(global_cpu[name].shape) for name in param_names}
        sizes = [int(np.prod(shapes[name])) for name in param_names]
        flat_deltas = []
        for client_idx, weights in enumerate(clean_cpu):
            pieces = []
            for name in param_names:
                if name not in weights:
                    raise KeyError(f"Client {client_ids[client_idx]} missing key {name}")
                delta = (
                    weights[name].detach().cpu().float() - global_cpu[name].detach().cpu().float()
                )
                pieces.append(quantize_tensor(delta, self.scale).reshape(-1))
            flat_deltas.append(
                np.concatenate(pieces).astype(np.int64) if pieces else np.array([], dtype=np.int64)
            )
        return param_names, shapes, sizes, flat_deltas

    def _serialize_locally_clipped_deltas(
        self,
        clean_client_weights: Sequence[Dict[str, Any]],
        global_model: Dict[str, Any],
        client_ids: Sequence[int],
    ) -> Tuple[List[str], Dict[str, Tuple[int, ...]], List[int], List[np.ndarray], List[float]]:
        global_cpu = state_dict_to_cpu(global_model)
        clean_cpu = [state_dict_to_cpu(sd) for sd in clean_client_weights]
        param_names = self._release_keys(global_cpu)
        shapes = {name: tuple(global_cpu[name].shape) for name in param_names}
        sizes = [int(np.prod(shapes[name])) for name in param_names]
        flat_deltas = []
        clip_factors = []
        for client_idx, weights in enumerate(clean_cpu):
            pieces = []
            for name in param_names:
                if name not in weights:
                    raise KeyError(f"Client {client_ids[client_idx]} missing key {name}")
                delta = (
                    weights[name].detach().cpu().float() - global_cpu[name].detach().cpu().float()
                )
                pieces.append(delta.reshape(-1))
            if not pieces:
                flat_deltas.append(np.array([], dtype=np.int64))
                clip_factors.append(1.0)
                continue
            flat = torch.cat(pieces).float()
            norm = float(torch.linalg.vector_norm(flat, ord=2).item())
            factor = 1.0 if norm <= 0.0 else min(1.0, float(self.clip_norm) / norm)
            flat_deltas.append(quantize_tensor(flat * factor, self.scale).reshape(-1))
            clip_factors.append(float(factor))
        return param_names, shapes, sizes, flat_deltas, clip_factors

    def _check_ring_safety(self, flat_deltas: Sequence[np.ndarray]) -> Tuple[int, int]:
        max_abs_delta_int, norm_sq_bound = _integer_norm_sq_bound(flat_deltas)
        # Reserve eight bits of signed-ring headroom for norm-square arithmetic.
        safe_bound = 1 << max(1, self.ring_bits - 8)
        if norm_sq_bound >= safe_bound:
            raise OverflowError(
                "Q16 norm-square bound is too large for configured 64-bit ring: "
                f"norm_sq_bound={norm_sq_bound}, safe_bound={safe_bound}, ring_bits={self.ring_bits}"
            )
        return max_abs_delta_int, norm_sq_bound

    def _noise_share_arrays_for_fast_route(
        self, total_values: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if self.noise_mode not in (MPC_DP_NOISE_ZERO, MPC_DP_NOISE_ZERO_DEBUG):
            raise NotImplementedError("Only gated zero-noise tests use host noise arrays")
        return [np.zeros(total_values, dtype=np.int64) for _ in range(3)], {
            "noise_share_mode": "zero_noise_debug",
            "coordinator_generated_noise_shares": False,
            "privacy_result_valid": False,
            "debug_zero_noise": True,
        }

    def _simulated_fast_aggregate_sum(
        self,
        flat_clipped_deltas: Sequence[np.ndarray],
        round_id: int,
        stream_labels: Sequence[str] = (PRE_DETECTION_STREAM,),
    ) -> Tuple[np.ndarray, int, Dict[str, Any], Dict[str, float]]:
        """Evaluate the clipped integer sum and party noise locally over Z_2^64.

        Ring reconstruction and summation match the aggregate circuit. All inputs
        remain visible to the local process.
        """

        total_values = int(flat_clipped_deltas[0].size) if flat_clipped_deltas else 0
        t_noise = time.time()
        if self.noise_mode in [MPC_DP_NOISE_ZERO, MPC_DP_NOISE_ZERO_DEBUG]:
            noise_party_values = [np.zeros(total_values, dtype=np.int64) for _ in range(3)]
            noise_meta = {
                "noise_share_mode": "zero_noise_debug",
                "coordinator_generated_noise_shares": False,
                "debug_zero_noise": True,
            }
        elif self.noise_mode == MPC_DP_NOISE_PARTY_LOCAL_DGAUSS:
            if self.noise_provider is None:
                raise RuntimeError(
                    "party_local_distributed_dgauss noise provider was not initialized"
                )
            noise_party_values = [
                self.noise_provider.generate_party_noise_summed(
                    party_id=pid,
                    round_id=round_id,
                    count=total_values,
                    stream_labels=stream_labels,
                )
                for pid in range(self.noise_provider.num_parties)
            ]
            noise_meta = {
                **self.noise_provider.metadata(),
                "noise_mode": MPC_DP_NOISE_PARTY_LOCAL_DGAUSS,
                "noise_share_mode": "party_local_seeded_stream",
                "party_local_noise_stream_reproduced_in_process": True,
                "debug_zero_noise": False,
            }
        else:
            raise NotImplementedError(
                "mpc_dp_backend=simulated_mpc supports zero-noise debug, "
                "party_local_distributed_dgauss and simulation_debug noise "
                f"shares; got noise_mode={self.noise_mode!r} with "
                f"noise_share_mode={self.noise_share_mode!r}"
            )

        noise_s = time.time() - t_noise

        t_sum = time.time()
        # Open masks have already selected inputs; secret masks are applied in-circuit.
        simulated_mask = getattr(self, "_active_simulated_mask", None)
        if simulated_mask is not None:
            if len(simulated_mask) != len(flat_clipped_deltas):
                raise RuntimeError(
                    f"simulated mask has {len(simulated_mask)} bits for "
                    f"{len(flat_clipped_deltas)} clients"
                )
            contributions = [
                delta for delta, bit in zip(flat_clipped_deltas, simulated_mask) if bit
            ]
        else:
            contributions = list(flat_clipped_deltas)
        noisy_sum = np.zeros(total_values, dtype=np.int64)
        for client_delta in contributions:
            noisy_sum += client_delta.reshape(-1).astype(np.int64, copy=False)
        for noise_share in noise_party_values:
            noisy_sum += noise_share.reshape(-1).astype(np.int64, copy=False)
        sum_s = time.time() - t_sum

        # Record local-reference visibility independently of the selected noise mode.
        noise_meta.update(
            {
                "contributing_client_count": float(len(contributions)),
                "coordinator_generated_noise_shares": False,
                "coordinator_side_python_parsed_noise_shares": True,
                "privacy_result_valid": False,
                "simulated_mpc_execution": True,
                "privacy_invalid_reason": (
                    "mpc_dp_backend=simulated_mpc evaluates the aggregate-sum "
                    "circuit in the coordinator process, so no MPC party boundary "
                    "protects the clipped client deltas or the noise shares"
                ),
            }
        )
        return (
            noisy_sum,
            total_values,
            noise_meta,
            {
                "noise_s": float(noise_s),
                "sum_s": float(sum_s),
            },
        )

    def _cryptgpu_fast_aggregate_sum(
        self,
        flat_clipped_deltas: Sequence[np.ndarray],
        round_id: int,
        stream_labels: Sequence[str] = (PRE_DETECTION_STREAM,),
    ):
        cache = self._active_clip_cache
        try:
            result = self._cryptgpu_fast_aggregate_sum_impl(
                flat_clipped_deltas, round_id, stream_labels
            )
            if cache and cache["action"] == "write":
                self.clip_cache.pending["ready"] = True
            return result
        except BaseException:
            self.clip_cache.clear()
            raise
        finally:
            if cache and cache["action"] == "read":
                self.clip_cache.clear()
            self._active_clip_cache = None
            job_dir = getattr(self, "_active_dp_job_dir", None)
            if job_dir:
                shutil.rmtree(job_dir, ignore_errors=True)
                self._active_dp_job_dir = None

    def _cryptgpu_fast_aggregate_sum_impl(
        self,
        flat_clipped_deltas: Sequence[np.ndarray],
        round_id: int,
        stream_labels: Sequence[str] = (PRE_DETECTION_STREAM,),
    ) -> Tuple[np.ndarray, int, Dict[str, Any], Dict[str, float]]:
        """Evaluate an aggregate sum on three CryptGPU parties.

        Each party contributes its input and noise shares; only the noisy total is opened.
        """

        total_values = int(flat_clipped_deltas[0].size) if flat_clipped_deltas else 0
        num_clients = len(flat_clipped_deltas)
        if total_values <= 0 or num_clients <= 0:
            raise RuntimeError("CryptGPU MPC-DP aggregate requires at least one client value")

        t_share = time.time()
        cache = self._active_clip_cache
        cache_hit = bool(cache and cache["action"] == "read")
        mask_share_dir = getattr(self, "_active_mask_share_dir", None)
        if cache_hit:
            # Workers load their own cached shares.
            delta_party_values = []
        elif mask_share_dir or self.clip_mode in MPC_CLIP_MODES:
            # Preserve client blocks for multiplication by secret acceptance shares.
            blocks = [np.zeros((num_clients, total_values), dtype=np.int64) for _ in range(3)]
            for index, client_delta in enumerate(flat_clipped_deltas):
                values = client_delta.reshape(-1).astype(np.int64, copy=False)
                if self.clip_mode in MPC_CLIP_MODES:
                    # CryptGPU input masks span the full Z_2^64 ring.
                    s0 = np.frombuffer(os.urandom(values.size * 8), dtype=np.int64).copy()
                    s1 = np.frombuffer(os.urandom(values.size * 8), dtype=np.int64).copy()
                else:
                    s0 = _secure_random_int64_array(values.shape)
                    s1 = _secure_random_int64_array(values.shape)
                blocks[0][index] = s0
                blocks[1][index] = s1
                blocks[2][index] = (values - s0 - s1).astype(np.int64)
            delta_party_values = [block.reshape(-1) for block in blocks]
        else:
            # Open-mask aggregation needs one pre-summed share vector per party.
            delta_party_values = [np.zeros(total_values, dtype=np.int64) for _ in range(3)]
            for client_delta in flat_clipped_deltas:
                values = client_delta.reshape(-1).astype(np.int64, copy=False)
                s0 = _secure_random_int64_array(values.shape)
                s1 = _secure_random_int64_array(values.shape)
                delta_party_values[0] += s0
                delta_party_values[1] += s1
                delta_party_values[2] += (values - s0 - s1).astype(np.int64)

        job_dir = tempfile.mkdtemp(prefix="cryptgpu_dp_", dir=self.temp_dir)
        self._active_dp_job_dir = job_dir
        for pid in range(len(delta_party_values)):
            np.save(
                os.path.join(job_dir, f"delta_share_{pid}.npy"),
                delta_party_values[pid],
            )

        if self.noise_mode == MPC_DP_NOISE_PARTY_LOCAL_DGAUSS:
            if self.noise_provider is None:
                raise RuntimeError(
                    "party_local_distributed_dgauss noise provider was not initialized"
                )
            noise_source = "party_local_seeded_stream"
            noise_meta = {
                **self.noise_provider.metadata(),
                "noise_mode": MPC_DP_NOISE_PARTY_LOCAL_DGAUSS,
                "noise_share_mode": "party_local_seeded_stream",
                "coordinator_generated_noise_shares": False,
                "coordinator_generated_noise": False,
                "client_generated_noise": False,
                "party_local_noise": True,
                "coordinator_side_python_parsed_noise_shares": False,
                "privacy_result_valid": True,
                "debug_zero_noise": False,
                "requires_external_party_noise_seeds": True,
                "party_noise_seed_files": [
                    os.path.abspath(self.noise_provider.seed_file_for_party(pid))
                    for pid in range(self.noise_provider.num_parties)
                ],
            }
        else:
            noise_source = "coordinator_share_file"
            noise_party_values, noise_meta = self._noise_share_arrays_for_fast_route(total_values)
            noise_meta = dict(noise_meta)
            noise_meta["coordinator_side_python_parsed_noise_shares"] = True
            for pid in range(3):
                np.save(
                    os.path.join(job_dir, f"noise_share_{pid}.npy"),
                    np.asarray(noise_party_values[pid], dtype=np.int64).reshape(-1),
                )

        job = {
            "schema": CRYPTGPU_DP_JOB_SCHEMA,
            "round_id": int(round_id),
            "num_clients": int(num_clients),
            "total_values": int(total_values),
            "noise_source": noise_source,
            "noise_params": _picklable_noise_params(self.params),
            "stream_labels": [str(label) for label in stream_labels],
            "mask_share_dir": mask_share_dir,
            "clip_mode": self.clip_mode,
            "clip_radius_int": math.floor(self.clip_norm * self.scale),
            "clip_cache": cache,
            "clip_chunk_size": int(self.params.get("mpc_dp_clip_chunk_size", 32768)),
            "sigma_int_target": int(self.sigma_int),
            "sensitivity_int": int(self.sensitivity_int),
        }
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as handle:
            json.dump(job, handle, sort_keys=True)
        share_s = time.time() - t_share

        t_sum = time.time()
        activate_cryptgpu_runtime()
        try:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":
                multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        from crypten import mpc as mpc_launcher

        results = mpc_launcher.run_multiprocess(world_size=3)(_cryptgpu_dp_aggregate_worker)(
            job_dir
        )
        sum_s = time.time() - t_sum

        receipt = next((value for value in (results or []) if isinstance(value, dict)), None)
        if receipt is None:
            raise RuntimeError("CryptGPU MPC-DP aggregate returned no opened vector")
        opened = np.load(receipt["opened_path"]).reshape(-1)
        if opened.size != total_values:
            raise RuntimeError(
                f"CryptGPU MPC-DP opened vector size mismatch: {opened.size} != {total_values}"
            )

        accepted_count = receipt.get("accepted_count")
        party_clip_timings = []
        for pid in range(3):
            with open(os.path.join(job_dir, f"clip_timing_{pid}.json")) as handle:
                party_clip_timings.append(json.load(handle))
        clip_timings = {
            key: max(item[key] for item in party_clip_timings) for key in party_clip_timings[0]
        }
        shutil.rmtree(job_dir, ignore_errors=True)

        if accepted_count is not None:
            noise_meta["contributing_client_count"] = float(accepted_count)
        noise_meta.update(
            {
                "simulated_mpc_execution": False,
                "mpc_engine": MPC_DP_BACKEND_CRYPTGPU,
                "mpc_sharing": "replicated_secret_sharing_ring_2_64",
                "mpc_corruption_model": "semi_honest_honest_majority_1_of_3",
                "fixed_point_encoder_bypassed": True,
                **clip_timings,
                "mpc_clip_cache_hit": cache_hit,
                "mpc_clip_cache_enabled": bool(cache),
                "mpc_clip_party_timings": party_clip_timings,
            }
        )
        return (
            opened,
            total_values,
            noise_meta,
            {
                "share_s": float(share_s),
                "sum_s": float(sum_s),
            },
        )

    def _record_output_validation(self, status: str) -> None:
        receipt = dict(self.last_execution_receipt or {})
        receipt["output_validation"] = str(status)
        if status != "passed":
            receipt["party_execution_status"] = receipt.get("status")
            receipt["status"] = "failed"
            receipt["failure_class"] = "invalid_or_rejected_output"
        self.last_execution_receipt = receipt

    def make_fast_aggregate_reference_release(
        self,
        clean_client_weights: Iterable[Dict[str, Any]],
        global_model: Dict[str, Any],
        client_ids: Iterable[int],
        round_id: int,
        accept_mask: Optional[Sequence[int]] = None,
        release_role: str = "unfiltered_pre_detection",
        mask_share_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a noisy aggregate reference model.

        accept_mask=None includes the full cohort for synthesis and BN calibration.
        A binary mask selects the filtered reference used in the next round.
        release_role identifies the output in recorded metrics.
        """
        started = time.time()
        clean_list = list(clean_client_weights)
        client_id_list = [int(x) for x in client_ids]
        if len(clean_list) != len(client_id_list):
            raise ValueError("clean_client_weights and client_ids length mismatch")
        if not clean_list:
            return {}

        if mask_share_dir:
            # Keep the full cohort when acceptance bits are secret-shared.
            mask = None
        elif accept_mask is not None:
            mask = [int(bool(int(m))) for m in accept_mask]
            if len(mask) != len(clean_list):
                raise ValueError(
                    f"accept_mask has {len(mask)} entries for {len(clean_list)} clients"
                )
            kept = [index for index, bit in enumerate(mask) if bit]
            if not kept:
                raise RuntimeError(
                    "[MPC-DP] every client was rejected; a filtered DP reference "
                    "cannot be formed this round"
                )
            clean_list = [clean_list[index] for index in kept]
            client_id_list = [client_id_list[index] for index in kept]
        else:
            mask = None

        # Domain-separated streams preserve residual noise between paired releases.
        is_pre_detection = release_role == "unfiltered_pre_detection"
        noise_stream_label = PRE_DETECTION_STREAM if is_pre_detection else str(release_role)
        if self.release_pairing == MPC_DP_PAIRING_PARTITION and is_pre_detection:
            # Pair the pre-filtering Z_a + Z_b stream with filtered Z_a.
            noise_stream_labels = (PRE_DETECTION_STREAM, POST_DETECTION_STREAM)
        else:
            noise_stream_labels = (noise_stream_label,)
        self._active_mask_share_dir = mask_share_dir
        # The local reference applies secret-mask inputs arithmetically without
        # preselecting clean_list.
        self._active_simulated_mask = (
            [int(bool(int(m))) for m in accept_mask]
            if (mask_share_dir and accept_mask is not None)
            else None
        )
        if self._active_simulated_mask is not None:
            if len(self._active_simulated_mask) != len(clean_list):
                raise ValueError(
                    f"accept_mask has {len(self._active_simulated_mask)} entries for "
                    f"{len(clean_list)} clients"
                )
            kept_ids = [cid for cid, bit in zip(client_id_list, self._active_simulated_mask) if bit]
            if not kept_ids:
                raise RuntimeError(
                    "[MPC-DP] every client was rejected; a filtered DP reference "
                    "cannot be formed this round"
                )

        global_cpu = state_dict_to_cpu(global_model)
        t0 = time.time()
        local_clip_s = 0.0
        if self.clip_mode in INTEGER_CLIP_MODES:
            _param_names, shapes, _sizes, flat_deltas = self._serialize_deltas(
                clean_list, global_cpu, client_id_list
            )
            # Validate ring-domain bounds before clipping.
            max_abs_delta_int, norm_sq_bound = self._check_ring_safety(flat_deltas)
            if self.clip_mode == MPC_NEAREST_CLIP:
                nearest_inner_radius(math.floor(self.clip_norm * self.scale), flat_deltas[0].size)
            clip_factors = []
            if self.clip_mode == LOCAL_INTEGER_CLIP:
                clip_start = time.perf_counter()
                flat_deltas = [
                    integer_l2_clip_reference(x, math.floor(self.clip_norm * self.scale))
                    for x in flat_deltas
                ]
                local_clip_s = time.perf_counter() - clip_start
        else:
            clip_start = time.perf_counter()
            _param_names, shapes, _sizes, flat_deltas, clip_factors = (
                self._serialize_locally_clipped_deltas(clean_list, global_cpu, client_id_list)
            )
            local_clip_s = time.perf_counter() - clip_start
            max_abs_delta_int, norm_sq_bound = self._check_ring_safety(flat_deltas)
        self._active_clip_cache = None
        if self.clip_mode in MPC_CLIP_MODES and self.params.get("mpc_dp_clip_cache", True):
            self._active_clip_cache = self.clip_cache.prepare(
                release_role,
                round_id,
                self.clip_mode,
                math.floor(self.clip_norm * self.scale),
                client_id_list,
                flat_deltas,
                global_cpu,
                _param_names,
                shapes,
            )
        serialize_s = time.time() - t0

        if self.simulated_execution:
            t1 = time.time()
            opened_sum_array, total_values, noise_meta, sim_timing = (
                self._simulated_fast_aggregate_sum(
                    flat_deltas, round_id=round_id, stream_labels=noise_stream_labels
                )
            )
            input_s = float(sim_timing["noise_s"])
            mpc_s = float(sim_timing["sum_s"])
            mpc_timing = {
                "compile_s": 0.0,
                "online_runtime_s": float(sim_timing["sum_s"]),
                "compile_cache_hit": None,
                "compile_cache_validation": "not_applicable_simulated",
                "program_sha256": None,
                "compile_identity_sha256": None,
                "compiled_artifact_manifest_sha256": None,
                "attempt_count": 0,
                "retry_count": 0,
                "retry_reasons": [],
                "cleanup_complete": True,
                "base_port": None,
                "run_counter": None,
                "dynamic_ports": False,
            }
            self.last_execution_receipt = {
                "status": "completed",
                "party_topology": "simulated-single-process",
                "protocol": self.protocol,
                "output_validation": "not_applicable_simulated",
                "simulated_mpc_execution": True,
            }
            opened_sum_values = opened_sum_array
            parse_s = 0.0
            _ = time.time() - t1
        elif self.cryptgpu_execution:
            t1 = time.time()
            opened_sum_array, total_values, noise_meta, cg_timing = (
                self._cryptgpu_fast_aggregate_sum(
                    flat_deltas, round_id=round_id, stream_labels=noise_stream_labels
                )
            )
            input_s = float(cg_timing["share_s"])
            mpc_s = float(cg_timing["sum_s"])
            mpc_timing = {
                "compile_s": 0.0,
                "online_runtime_s": float(cg_timing["sum_s"]),
                "compile_cache_hit": None,
                "compile_cache_validation": "not_applicable_cryptgpu",
                "program_sha256": None,
                "compile_identity_sha256": None,
                "compiled_artifact_manifest_sha256": None,
                "attempt_count": 1,
                "retry_count": 0,
                "retry_reasons": [],
                "cleanup_complete": True,
                "base_port": None,
                "run_counter": None,
                "dynamic_ports": False,
            }
            self.last_execution_receipt = {
                "status": "completed",
                "party_topology": "cryptgpu-3pc-local-processes",
                "protocol": "replicated-ring-2-64",
                "output_validation": "binary_all_reduce_exact",
                "simulated_mpc_execution": False,
            }
            opened_sum_values = opened_sum_array
            parse_s = 0.0
            self._record_output_validation("passed")
            _ = time.time() - t1
        else:
            raise NotImplementedError(
                "Only CryptGPU and the historical numerical reference are included"
            )

        pos = 0
        out = {}
        # Secret-mask aggregation divides by the opened accepted count.
        contributing = noise_meta.get("contributing_client_count")
        if contributing is None:
            # Open-mask inputs already contain only contributing clients.
            contributing = len(clean_list)
        contributing = int(round(float(contributing)))
        if contributing <= 0:
            raise RuntimeError(
                "[MPC-DP] the release reported no contributing clients; refusing "
                "to divide by the offered count instead"
            )
        denom = float(contributing * self.scale)
        for key, base_value in global_cpu.items():
            if key in shapes:
                size = int(np.prod(shapes[key]))
                delta_arr = (
                    np.asarray(opened_sum_values[pos : pos + size], dtype=np.float64).reshape(
                        shapes[key]
                    )
                    / denom
                )
                delta_tensor = torch.from_numpy(delta_arr.astype(np.float32))
                out[key] = base_value.detach().cpu().float() + delta_tensor
                pos += size
            else:
                out[key] = copy.deepcopy(base_value)

        privacy_valid = bool(noise_meta.get("privacy_result_valid", False))
        # Charge all offered clients when individual acceptance bits remain secret.
        self.released_client_ids.extend(client_id_list)
        self.released_round_ids.extend([int(round_id)] * len(client_id_list))
        metric = {
            "round_id": int(round_id),
            "client_ids": client_id_list,
            "release_type": "mpc_aggregate_reference_client_level_dp",
            "release_role": str(release_role),
            "noise_stream_label": noise_stream_label,
            "noise_stream_labels": list(noise_stream_labels),
            "release_pairing": self.release_pairing,
            "accept_mask_applied": mask is not None or bool(mask_share_dir),
            "accept_mask_secret_shared": bool(mask_share_dir),
            "clients_contributing": contributing,
            "clients_charged_for_composition": len(client_id_list),
            "clients_offered": len(mask) if mask is not None else len(clean_list),
            "backend": self.backend,
            "simulated_mpc_execution": bool(self.simulated_execution),
            "mpc_protocol": ("replicated-ring-2-64" if self.cryptgpu_execution else self.protocol),
            "mpc_engine": self.backend,
            "clip_mode": self.clip_mode,
            "clip_location": "mpc" if self.clip_mode in MPC_CLIP_MODES else "harness",
            "clip_integer_l2_bound_enforced": self.clip_mode in INTEGER_CLIP_MODES,
            "clip_radius_int": math.floor(self.clip_norm * self.scale),
            "clip_inner_radius_int": (
                nearest_inner_radius(math.floor(self.clip_norm * self.scale), total_values)
                if self.clip_mode == MPC_NEAREST_CLIP
                else math.floor(self.clip_norm * self.scale)
            ),
            "local_clip_s": local_clip_s,
            "local_clip_timing_scope": "serialize_and_clip"
            if self.clip_mode == "local_reference_clip_then_mpc_sum"
            else "clip_only",
            "noise_mode": self.noise_mode,
            "noise_share_mode": self.noise_share_mode,
            "output_mode": "aggregate_reference_fast",
            "fixed_point_scale": self.scale,
            "clip_norm": self.clip_norm,
            "adjacency": self.adjacency,
            "sensitivity_int": str(self.sensitivity_int),
            # The pre-filtering sum has add/remove sensitivity C.
            # The filtered reference uses a fixed-partition sensitivity calculation;
            # data-dependent acceptance can alter other clients' contributions.
            "sensitivity_conditional_on": ("opened_accept_mask" if mask is not None else None),
            "noise_multiplier": self.noise_multiplier,
            "sigma_int": str(self.sigma_int),
            "sigma_int_target": int(noise_meta.get("sigma_int_target", self.sigma_int)),
            "sigma_int_party": noise_meta.get("sigma_int_party"),
            "tail_bound": self.tail_bound,
            "tail_error_bound": noise_meta.get("sampler_tail_probability_bound"),
            "sampler_tail_probability_bound": noise_meta.get("sampler_tail_probability_bound"),
            "sampler_cdf_quantization_error_bound": noise_meta.get(
                "sampler_cdf_quantization_error_bound"
            ),
            "sampler_statistical_distance_bound": noise_meta.get(
                "sampler_statistical_distance_bound"
            ),
            "distributed_noise_assumption": (
                noise_meta.get("distributed_noise_assumption")
                if self.noise_mode
                in [MPC_DP_NOISE_PRIVATE_FILE_DGAUSS, MPC_DP_NOISE_PARTY_LOCAL_DGAUSS]
                else None
            ),
            "ring_bits": self.ring_bits,
            "sfix_k": self.sfix_k,
            "max_abs_delta_int": None
            if self.clip_mode in MPC_CLIP_MODES
            else int(max_abs_delta_int),
            "norm_sq_bound": "public_input_domain_lt_2**56"
            if self.clip_mode in MPC_CLIP_MODES
            else str(norm_sq_bound),
            "local_clip_factors_min": float(min(clip_factors)) if clip_factors else None,
            "local_clip_factors_max": float(max(clip_factors)) if clip_factors else None,
            "num_clients": len(clean_list),
            "K": len(clean_list),
            "values_per_client": total_values,
            "d": total_values,
            "per_round_coordinate_count": len(clean_list) * total_values,
            "opened_coordinate_count": total_values,
            "opened_coordinates": total_values,
            "opened_release_bytes": _tensor_nbytes(out),
            "serialize_s": serialize_s,
            "input_write_s": input_s,
            "mpc_compile_s": mpc_timing["compile_s"],
            "mpc_online_runtime_s": mpc_timing["online_runtime_s"],
            "mpc_compile_cache_hit": mpc_timing.get("compile_cache_hit", False),
            "mpc_compile_cache_validation": mpc_timing.get("compile_cache_validation"),
            "mpc_program_sha256": mpc_timing.get("program_sha256"),
            "mpc_compile_identity_sha256": mpc_timing.get("compile_identity_sha256"),
            "mpc_compiled_artifact_manifest_sha256": mpc_timing.get(
                "compiled_artifact_manifest_sha256"
            ),
            "mpc_attempt_count": mpc_timing.get("attempt_count"),
            "mpc_retry_count": mpc_timing.get("retry_count"),
            "mpc_retry_reasons": mpc_timing.get("retry_reasons", []),
            "mpc_cleanup_complete": mpc_timing.get("cleanup_complete"),
            "mpc_base_port": mpc_timing.get("base_port"),
            "mpc_run_counter": mpc_timing.get("run_counter"),
            "mpc_dynamic_ports": mpc_timing.get("dynamic_ports"),
            "opening_output_mode": (
                "cryptgpu_all_reduce_int64"
                if self.cryptgpu_execution
                else "in_process_numpy_sum"
                if self.simulated_execution
                else "binary_int64_output_file"
            ),
            "mpc_compile_and_run_s": mpc_s,
            "parse_s": parse_s,
            "total_s": time.time() - started,
            "opened_values": "one_noisy_aggregate_reference_delta",
            "noise_added_to": "aggregate_sum_before_division",
            "per_client_noisy_models_opened": False,
            "privacy_result_valid": privacy_valid,
            "debug_zero_noise": self.noise_mode in [MPC_DP_NOISE_ZERO, MPC_DP_NOISE_ZERO_DEBUG],
            **noise_meta,
        }
        self.metrics.append(metric)
        return out

    def write_artifacts(
        self, output_dir: str, participation_counts: Dict[Any, int], delta: float
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "mpc_dp_release_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump({"records": self.metrics}, handle, indent=2)

        released_ids = sorted(set(int(x) for x in self.released_client_ids))

        # Independent pairing counts contributing releases. Partition-correlated
        # pairing counts distinct rounds per client for the fixed-partition reference.
        partitioned = self.release_pairing == MPC_DP_PAIRING_PARTITION
        release_counts: Dict[int, int] = {}
        seen_round_client = set()
        round_ids = list(self.released_round_ids)
        for index, cid in enumerate(self.released_client_ids):
            key = int(cid)
            if partitioned:
                round_id = round_ids[index] if index < len(round_ids) else None
                if (key, round_id) in seen_round_client:
                    continue
                seen_round_client.add((key, round_id))
            release_counts[key] = release_counts.get(key, 0) + 1
        counts = {str(cid): int(release_counts.get(cid, 0)) for cid in released_ids}

        rounds_selected = {
            str(cid): int((participation_counts or {}).get(cid, 0)) for cid in released_ids
        }
        inflated = {
            cid: (counts[cid], rounds_selected[cid])
            for cid in counts
            if counts[cid] != rounds_selected[cid]
        }
        if inflated:
            sample = dict(list(inflated.items())[:3])
            logger.info(
                "[MPC-DP][accounting] composing over release counts, not round "
                "counts: %d/%d clients contributed to more releases than rounds "
                "selected (client: releases, rounds) %s",
                len(inflated),
                len(counts),
                sample,
            )
        eps_per_client = {}
        worst = 0.0
        privacy_valid = bool(self.metrics) and all(
            bool(m.get("privacy_result_valid")) for m in self.metrics
        )
        fast_metrics = [
            m for m in self.metrics if m.get("output_mode") == "aggregate_reference_fast"
        ]
        latest_fast = fast_metrics[-1] if fast_metrics else {}
        effective_sigma_int = float(
            latest_fast.get("effective_min_honest_sigma_int", self.sigma_int)
        )
        sensitivity_sum_int = int(self.sensitivity_int)
        latest_k = int(latest_fast.get("num_clients", latest_fast.get("K", 1)) or 1)
        if (
            self.noise_mode in [MPC_DP_NOISE_ZERO, MPC_DP_NOISE_ZERO_DEBUG, MPC_DP_NOISE_INSECURE]
            or not privacy_valid
        ):
            eps_per_client = {cid: None for cid in counts}
            worst_value = None
        else:
            accountant_params = dict(self.params)
            accountant_params["ldp_accountant_variant"] = "balanced_direct_composition"
            accountant_params["ldp_max_grad_norm"] = float(sensitivity_sum_int) / float(self.scale)
            sigma_float = float(effective_sigma_int) / float(self.scale)
            for cid, count in counts.items():
                eps = evaluate_accountant_for_sigma(
                    accountant_params,
                    sigma=max(sigma_float, 1e-12),
                    actual_participations=[count],
                    # These counts already include the service's release multiplicity.
                    participations_are_releases=True,
                )["epsilon"]
                eps_per_client[cid] = float(eps)
                worst = max(worst, float(eps))
            worst_value = str(worst)

        fast_seen = any(m.get("output_mode") == "aggregate_reference_fast" for m in self.metrics)
        party_local_seen = any(
            m.get("noise_mode") == MPC_DP_NOISE_PARTY_LOCAL_DGAUSS for m in self.metrics
        )
        distributed_seen = party_local_seen or any(
            m.get("noise_mode")
            in [MPC_DP_NOISE_PRIVATE_FILE_DGAUSS, MPC_DP_NOISE_DISTRIBUTED_SHARES]
            for m in self.metrics
        )
        accounting = {
            "composition_basis": (
                "rounds_per_client_under_partition_correlated_pairing"
                if partitioned
                else "releases_per_client"
            ),
            "release_pairing": self.release_pairing,
            "releases_per_client": counts,
            "rounds_selected_per_client": rounds_selected,
            "release_type": "mpc_aggregate_reference_dp",
            "mechanism": (
                "distributed_bounded_discrete_gaussian" if distributed_seen else self.noise_mode
            ),
            "adjacency": self.adjacency,
            "clip_norm": self.clip_norm,
            "fixed_point_scale": self.scale,
            "sensitivity_int": str(self.sensitivity_int),
            "sensitivity_sum_int": sensitivity_sum_int,
            "sensitivity_average_int": float(sensitivity_sum_int) / float(max(1, latest_k)),
            "noise_multiplier": self.noise_multiplier,
            "sigma_int": str(self.sigma_int),
            "sigma_int_target": latest_fast.get("sigma_int_target", self.sigma_int),
            "sigma_int_party": latest_fast.get("sigma_int_party"),
            "num_noise_parties": latest_fast.get("num_noise_parties"),
            "corruption_threshold": latest_fast.get("corruption_threshold"),
            "min_honest_noise_parties": latest_fast.get("min_honest_noise_parties"),
            "effective_min_honest_sigma_int": latest_fast.get("effective_min_honest_sigma_int"),
            "all_honest_sigma_int": latest_fast.get("all_honest_sigma_int"),
            "delta": float(delta),
            "realized_participations": counts,
            "accounting_client_ids": released_ids,
            "epsilon_per_client": eps_per_client,
            "worst_case_epsilon": worst_value,
            "subsampling_amplification_claimed": False,
            "accounting_domain": "aggregate_sum_then_divide" if fast_seen else None,
            "noise_added_to": "aggregate_sum_before_division" if fast_seen else None,
            "opened_object": "one_noisy_aggregate_reference_model",
            "per_client_noisy_models_opened": False,
            "tail_error_bound": latest_fast.get("sampler_tail_probability_bound"),
            "tail_probability_bound": latest_fast.get("sampler_tail_probability_bound"),
            "sampler_tail_probability_bound": latest_fast.get("sampler_tail_probability_bound"),
            "sampler_cdf_quantization_error_bound": latest_fast.get(
                "sampler_cdf_quantization_error_bound"
            ),
            "sampler_statistical_distance": latest_fast.get("sampler_statistical_distance_bound"),
            "sampler_statistical_distance_bound": latest_fast.get(
                "sampler_statistical_distance_bound"
            ),
            "sampler": latest_fast.get("sampler"),
            "distributed_noise_assumption": (
                latest_fast.get("distributed_noise_assumption") if distributed_seen else None
            ),
            "noise_share_mode": self.noise_share_mode,
            "privacy_result_valid": privacy_valid,
            "debug_zero_noise": self.noise_mode in [MPC_DP_NOISE_ZERO, MPC_DP_NOISE_ZERO_DEBUG],
            "coordinator_generated_noise": bool(
                latest_fast.get(
                    "coordinator_generated_noise",
                    latest_fast.get("coordinator_generated_noise_shares", False),
                )
            ),
            "client_generated_noise": bool(latest_fast.get("client_generated_noise", False)),
            "party_local_noise": bool(latest_fast.get("party_local_noise", False)),
            "accounting_approximation": (
                "Conservative continuous-Gaussian RDP calibration for integer distributed noise shares; "
                "exact discrete-Gaussian RDP/statistical-distance accounting is not implemented."
                if privacy_valid
                else None
            ),
            "notes": (
                "Distributed finite-precision bounded discrete-Gaussian party noise shares are added to the "
                "aggregate sum before division. This is not local DP and not full in-circuit MPC noise sampling."
                if distributed_seen
                else None
            ),
            "stage_warning": (
                None
                if privacy_valid
                else "This run is debug/simulation only and is not a DP privacy guarantee."
            ),
        }
        accounting_path = os.path.join(output_dir, "mpc_dp_accounting.json")
        with open(accounting_path, "w", encoding="utf-8") as handle:
            json.dump(accounting, handle, indent=2)
