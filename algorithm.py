import copy
import logging
import random
import time
import math
import json
import collections
import statistics
from pathlib import Path
import torch
import train
import test
from gan import gan_defense
from models.ResNet8 import ResNet8
from device import device
from differential_privacy import create_ldp_instance, apply_ldp_to_weights
from mpc_dp_release import (
    DP_RELEASE_CLIENT_LDP,
    DP_RELEASE_MPC_AGGREGATE_FAST,
    MpcDpReleaseService,
    canonical_dp_release_source,
    is_mpc_dp_source,
)
from memory_utils import memory_snapshot, clear_cuda_memory, state_dict_to_cpu
from backdoor_attack import is_dba_attack
import os
import shutil
import tempfile
import baseline_shieldfl
import baseline_input_validation
import baseline_crowdguard
import baseline_fltrust
from cryptgpu_runtime import activate_runtime as _activate_cryptgpu_runtime
from detector_rules import MASK_SECRET_SHARED, canonical_secure_mask_mode

CRYPTGPU_AGG_SCALE = 10**4
from detector_rules import (
    ACCEPT_BITS_ONLY,
    CALIBRATED_CENTERED_SCORE,
    DEBUG_LOGITS,
    DIAGNOSTIC_SCORES,
    FIXED_TAU,
    calibrated_centered_reject_bits,
    canonical_calibrated_score_type,
    canonical_center_mode,
    canonical_detector_output_mode,
    canonical_final_detector,
    secure_calibrated_centered_decision_values,
    secure_mean_target_logit_margin,
    secure_mean_target_probability,
)

logger = logging.getLogger("logger")
PRIV_FILTERFL_DETECTOR_EVASION_ATTACK_MODES = {"priv_filterfl_detector_evasion_adaptive"}
EVASION_TRIGGER_SOURCE_COORDINATOR = "previous_public_detector_trigger"
EVASION_TRIGGER_SOURCE_SELF_SYNTH = "self_synth_recent_published_globals"
SUPPORTED_EVASION_TRIGGER_SOURCES = {
    EVASION_TRIGGER_SOURCE_COORDINATOR,
    EVASION_TRIGGER_SOURCE_SELF_SYNTH,
}
SELFSYNTH_GLOBAL_SOURCE_COORDINATOR_RING = "coordinator_ring"
SELFSYNTH_GLOBAL_SOURCE_COALITION_HELD = "coalition_held"
SUPPORTED_SELFSYNTH_GLOBAL_SOURCES = {
    SELFSYNTH_GLOBAL_SOURCE_COORDINATOR_RING,
    SELFSYNTH_GLOBAL_SOURCE_COALITION_HELD,
}
SECURE_AGGREGATION_RECEIPT_SCHEMA = "fldd-secure-aggregation-receipt-v2"


def _record_detection_score(
    handle,
    method,
    client_id,
    is_malicious,
    score,
    top_idx,
    threshold,
    predicted_reject,
    max_other=None,
    logit_margin=None,
    target_logit_mean=None,
    max_other_logit_mean=None,
    require_target_top=None,
):
    if not bool(handle.params.get("record_detection_diagnostics", False)):
        return
    score = float(score)
    threshold = float(threshold)
    top_margin = None if max_other is None else score - float(max_other)
    effective_require_target_top = True if require_target_top is None else bool(require_target_top)
    record = {
        "round": int(getattr(handle, "current_epoch", 0)),
        "seed": int(handle.params.get("seed", 0)),
        "method": str(method),
        "client_id": int(client_id),
        "malicious_gt": bool(is_malicious),
        "p_target": score,
        "top_idx": int(top_idx),
        "top_is_target": int(top_idx) == 0,
        "target_idx": 0,
        "tau": threshold,
        "score_margin": score - threshold,
        "top_margin": top_margin,
        "decision_margin": min(score - threshold, top_margin)
        if effective_require_target_top and top_margin is not None
        else score - threshold,
        "logit_margin": float(logit_margin) if logit_margin is not None else None,
        "target_logit_mean": float(target_logit_mean) if target_logit_mean is not None else None,
        "max_other_logit_mean": float(max_other_logit_mean)
        if max_other_logit_mean is not None
        else None,
        "predicted_reject": bool(predicted_reject),
        "bn_source": "noisy_reference"
        if bool(handle.params.get("cryptgpu_bn_use_noisy", False))
        else "clean_trigger_batch",
    }
    if require_target_top is not None:
        record["require_target_top"] = bool(require_target_top)
    records = getattr(handle, "detection_score_records", None)
    if records is None:
        records = []
        handle.detection_score_records = records
    records.append(record)


def _detector_canary_evaluate(handle, epoch):
    """Compare the cohort median response with its trailing median baseline.

    Readings below the configured ratio trigger the canary and are excluded from
    the baseline history.
    """
    params = handle.params
    reading = {
        "round": int(epoch),
        "enabled": False,
        "cohort_median": None,
        "baseline": None,
        "ratio": None,
        "fired": False,
    }
    if not bool(params.get("detector_canary_enabled", False)):
        return reading
    reading["enabled"] = True
    records = getattr(handle, "detection_score_records", None) or []
    current = [
        float(r["p_target"])
        for r in records
        if int(r.get("round", -1)) == int(epoch) and r.get("p_target") is not None
    ]
    if not current:
        return reading
    median = float(statistics.median(current))
    reading["cohort_median"] = median
    history = getattr(handle, "detector_canary_history", None)
    if history is None:
        history = collections.deque(maxlen=int(params.get("detector_canary_window", 20)))
        handle.detector_canary_history = history
    min_prior = int(params.get("detector_canary_min_baseline_rounds", 10))
    if len(history) >= min_prior:
        baseline = float(statistics.median(history))
        reading["baseline"] = baseline
        if baseline > 0.0:
            ratio = median / baseline
            reading["ratio"] = ratio
            if ratio < float(params.get("detector_canary_ratio", 0.01)):
                reading["fired"] = True
    if not reading["fired"]:
        history.append(median)
    return reading


def _record_bn_buffer_discrepancy(handle, client_index, clean_state, noisy_state):
    """Record BN-buffer differences between clean and noisy models."""
    if not bool(handle.params.get("record_bn_diagnostics", False)):
        return
    values = []
    per_key = []
    for key, clean_value in clean_state.items():
        if "running_mean" not in key and "running_var" not in key:
            continue
        noisy_value = noisy_state.get(key)
        if not isinstance(clean_value, torch.Tensor) or not isinstance(noisy_value, torch.Tensor):
            continue
        clean_cpu = clean_value.detach().to("cpu", dtype=torch.float32)
        noisy_cpu = noisy_value.detach().to("cpu", dtype=torch.float32)
        if clean_cpu.shape != noisy_cpu.shape:
            continue
        diff = noisy_cpu - clean_cpu
        relative_l2 = float(diff.norm().item() / max(clean_cpu.norm().item(), 1e-12))
        mean_abs = float(diff.abs().mean().item()) if diff.numel() else 0.0
        values.append(relative_l2)
        per_key.append(
            {
                "key": key,
                "relative_l2": relative_l2,
                "mean_abs": mean_abs,
                "numel": int(diff.numel()),
            }
        )
    if not per_key:
        return
    records = getattr(handle, "bn_discrepancy_records", None)
    if records is None:
        records = []
        handle.bn_discrepancy_records = records
    records.append(
        {
            "round": int(getattr(handle, "current_epoch", 0)),
            "client_index": int(client_index),
            "diagnostic": "state_dict_bn_buffer_proxy",
            "mean_relative_l2": float(sum(values) / len(values)),
            "max_relative_l2": float(max(values)),
            "per_key": per_key,
        }
    )


def _bn_stat_discrepancy(clean_stats, noisy_stats):
    """Compare named BN batch statistics using aggregate discrepancy metrics."""
    per_key = []
    relative_values = []
    for layer_name in sorted(set(clean_stats) & set(noisy_stats)):
        clean_layer = clean_stats.get(layer_name)
        noisy_layer = noisy_stats.get(layer_name)
        if not isinstance(clean_layer, dict) or not isinstance(noisy_layer, dict):
            continue
        for statistic in ("mean", "var"):
            clean_value = clean_layer.get(statistic)
            noisy_value = noisy_layer.get(statistic)
            if not isinstance(clean_value, torch.Tensor) or not isinstance(
                noisy_value, torch.Tensor
            ):
                continue
            clean_cpu = clean_value.detach().to("cpu", dtype=torch.float32)
            noisy_cpu = noisy_value.detach().to("cpu", dtype=torch.float32)
            if clean_cpu.shape != noisy_cpu.shape:
                continue
            if not torch.isfinite(clean_cpu).all() or not torch.isfinite(noisy_cpu).all():
                raise RuntimeError(f"non-finite BN diagnostic values for {layer_name}.{statistic}")
            difference = noisy_cpu - clean_cpu
            relative_l2 = float(difference.norm().item() / max(clean_cpu.norm().item(), 1e-12))
            relative_values.append(relative_l2)
            per_key.append(
                {
                    "key": f"{layer_name}.{statistic}",
                    "relative_l2": relative_l2,
                    "mean_abs": float(difference.abs().mean().item())
                    if difference.numel()
                    else 0.0,
                    "numel": int(difference.numel()),
                }
            )
    if not per_key:
        return None
    return {
        "mean_relative_l2": float(sum(relative_values) / len(relative_values)),
        "max_relative_l2": float(max(relative_values)),
        "per_key": per_key,
    }


def _record_bn_batch_stat_discrepancy(
    handle, active_client_index, client_id, clean_state, noisy_state, trigger_tensor_cpu, model_spec
):
    """Measure clean and noisy trigger-batch BN statistics used for folding."""
    if not bool(handle.params.get("record_bn_diagnostics", False)):
        return
    from tools.fuse_bn import extract_bn_stats_from_model

    diagnostic_device = torch.device("cpu")
    clean_model = noisy_model = None
    clean_stats = noisy_stats = None
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            clean_model = _build_detection_model(model_spec, diagnostic_device)
            noisy_model = _build_detection_model(model_spec, diagnostic_device)
            clean_model.load_state_dict(
                {
                    key: value.detach().to(device=diagnostic_device, dtype=torch.float32)
                    for key, value in clean_state.items()
                    if isinstance(value, torch.Tensor)
                }
            )
            noisy_model.load_state_dict(
                {
                    key: value.detach().to(device=diagnostic_device, dtype=torch.float32)
                    for key, value in noisy_state.items()
                    if isinstance(value, torch.Tensor)
                }
            )
            trigger = trigger_tensor_cpu.to(device=diagnostic_device, dtype=torch.float32)
            clean_stats = extract_bn_stats_from_model(clean_model, trigger)
            noisy_stats = extract_bn_stats_from_model(noisy_model, trigger)
        discrepancy = _bn_stat_discrepancy(clean_stats, noisy_stats)
        if discrepancy is None:
            raise RuntimeError("no aligned BN batch statistics were produced")
        records = getattr(handle, "bn_discrepancy_records", None)
        if records is None:
            records = []
            handle.bn_discrepancy_records = records
        records.append(
            {
                "round": int(getattr(handle, "current_epoch", 0)),
                "client_index": int(active_client_index),
                "client_id": int(client_id),
                "diagnostic": "trigger_batch_bn_stat_proxy",
                "reference_source": str(
                    handle.params.get("dp_cbf_stats_source", "per_client_noisy_model")
                ),
                **discrepancy,
            }
        )
    finally:
        del clean_stats, noisy_stats, clean_model, noisy_model


def _record_cryptgpu_runtime(handle, event, **extra):
    """Record parent-process runtime metrics for CryptGPU."""
    import os as _os
    import resource as _resource
    import torch.multiprocessing as _torch_mp

    soft_limit, hard_limit = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    fd_count = None
    try:
        fd_count = len(list(_os.scandir("/proc/self/fd")))
    except Exception:
        pass
    record = {
        "round": int(getattr(handle, "current_epoch", 0)),
        "event": str(event),
        "sharing_strategy": str(_torch_mp.get_sharing_strategy()),
        "rlimit_nofile_soft": int(soft_limit),
        "rlimit_nofile_hard": int(hard_limit),
        "parent_fd_count": fd_count,
    }
    record.update(extra)
    records = getattr(handle, "cryptgpu_runtime_records", None)
    if records is None:
        records = []
        handle.cryptgpu_runtime_records = records
    records.append(record)
    logger.info(f"[ours_cryptgpu][runtime] {record}")


def _record_secure_aggregation(
    handle, backend, status, active_clients, fallback, execution_receipt=None
):
    records = getattr(handle, "secure_aggregation_records", None)
    if records is None:
        records = []
        handle.secure_aggregation_records = records
    receipt = dict(execution_receipt or {})
    records.append(
        {
            "round": int(getattr(handle, "current_epoch", 0)),
            "backend": str(backend),
            "status": str(status),
            "active_clients": int(active_clients),
            "fallback": bool(fallback),
            **receipt,
        }
    )


def _record_detector_output(handle, client_ids, accept_bits, detector, output_mode):
    """Record acceptance bits returned by the detector."""
    if len(client_ids) != len(accept_bits):
        raise RuntimeError("detector output count does not match active client count")
    records = getattr(handle, "detector_output_records", None)
    if records is None:
        records = []
        handle.detector_output_records = records
    for client_id, accept in zip(client_ids, accept_bits):
        bit = int(accept)
        if bit not in (0, 1):
            raise RuntimeError(f"detector returned a non-bit value: {accept!r}")
        records.append(
            {
                "round": int(getattr(handle, "current_epoch", 0)),
                "client_id": int(client_id),
                "detector": str(detector),
                "output_mode": str(output_mode),
                "accept": bit,
                "revealed_fields": ["accept"],
            }
        )


def _record_plaintext_detector_screen_output(
    handle,
    client_ids,
    scores,
    reject_bits,
    detector,
    output_mode,
    score_type="logit_margin",
    extra_diagnostics=None,
):
    """Record plaintext response scores and rejection decisions for utility runs."""
    if output_mode != DIAGNOSTIC_SCORES:
        raise RuntimeError("plaintext detector-screen outputs require diagnostic_scores mode")
    if not len(client_ids) == len(scores) == len(reject_bits):
        raise RuntimeError("plaintext detector-screen output count does not match active clients")
    if extra_diagnostics is not None and len(extra_diagnostics) != len(client_ids):
        raise RuntimeError(
            "plaintext detector-screen diagnostic count does not match active clients"
        )
    records = getattr(handle, "detector_output_records", None)
    if records is None:
        records = []
        handle.detector_output_records = records
    score_field = "p_target" if score_type == "target_probability" else "logit_margin"
    for position, (client_id, score, reject) in enumerate(zip(client_ids, scores, reject_bits)):
        score = float(score)
        if not math.isfinite(score):
            raise RuntimeError("plaintext detector screen produced a non-finite score")
        bit = int(bool(reject))
        record = {
            "round": int(getattr(handle, "current_epoch", 0)),
            "client_id": int(client_id),
            "detector": str(detector),
            "output_mode": str(output_mode),
            score_field: score,
            "reject": bit,
            "revealed_fields": [score_field, "reject"],
            "diagnostic_only": True,
            "privacy_valid": False,
        }
        if extra_diagnostics is not None:
            extra = extra_diagnostics[position]
            if not isinstance(extra, dict):
                raise RuntimeError("plaintext detector-screen extra diagnostic must be a mapping")
            collisions = set(record).intersection(extra)
            if collisions:
                raise RuntimeError(
                    "plaintext detector-screen diagnostic field collision: "
                    + ", ".join(sorted(collisions))
                )
            record.update(extra)
            record["revealed_fields"] = [score_field, *sorted(extra), "reject"]
        records.append(record)


def get_clients(epoch, handle):
    total = int(handle.params["number_of_total_participants"])
    k = int(handle.params["num_clients"])
    sampling_strategy = (
        str(
            handle.params.get(
                "client_sampling_strategy",
                "legacy_balanced"
                if handle.params.get("balanced_schedule", False)
                else "uniform_random",
            )
        )
        .strip()
        .lower()
    )
    if sampling_strategy in ["frozen_manifest", "frozen_schedule"]:
        if getattr(handle, "_frozen_schedule", None) is None:
            schedule_path = handle.params.get("frozen_participation_schedule_path")
            if not isinstance(schedule_path, str) or not schedule_path.strip():
                raise ValueError(
                    "frozen_manifest sampling requires frozen_participation_schedule_path"
                )
            document = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
            schedule = document.get("round_client_schedule")
            if not isinstance(schedule, list) or len(schedule) != int(handle.params["epochs"]):
                raise ValueError("frozen participation schedule round count drift")
            participant_ids = set(handle.participants_list)
            normalized = []
            for round_index, row in enumerate(schedule, start=1):
                if (
                    not isinstance(row, list)
                    or len(row) != k
                    or len(set(row)) != k
                    or any((type(client_id) is not int for client_id in row))
                    or any((client_id not in participant_ids for client_id in row))
                ):
                    raise ValueError(f"invalid frozen participation cohort at round {round_index}")
                normalized.append(list(row))
            handle._frozen_schedule = normalized
            handle.schedule_type = "frozen_manifest"
        agent_name_keys = list(handle._frozen_schedule[int(epoch) - 1])
        adversarial_name_keys = (
            [
                client_id
                for client_id in agent_name_keys
                if client_id in getattr(handle, "adversarial_namelist", [])
            ]
            if handle.params["is_poison"]
            else []
        )
        logger.info(f"[frozen_manifest] epoch={epoch} clients={agent_name_keys}")
    elif sampling_strategy in ["balanced_block", "balanced_block_randomized", "block_balanced"]:
        if not hasattr(handle, "_balanced_schedule") or handle._balanced_schedule is None:
            handle._init_block_balanced_schedule()
        agent_name_keys = list(handle._balanced_schedule[int(epoch) - 1])
        resample = int(getattr(handle, "canary_resample_salt", 0) or 0)
        if resample > 0:
            rng = random.Random(
                int(epoch) * 1000003 ^ resample * 31337 ^ int(handle.params.get("seed", 0))
            )
            agent_name_keys = rng.sample(
                list(handle.participants_list), int(handle.params["num_clients"])
            )
            logger.info(f"[canary][resample] epoch={epoch} retry={resample} drew a fresh cohort")
        adversarial_name_keys = (
            [c for c in agent_name_keys if c in getattr(handle, "adversarial_namelist", [])]
            if handle.params["is_poison"]
            else []
        )
        logger.info(f"[balanced_block] epoch={epoch} clients={agent_name_keys}")
    elif sampling_strategy in ["legacy_balanced", "balanced_legacy"]:
        if not hasattr(handle, "_balanced_schedule") or handle._balanced_schedule is None:
            handle._init_balanced_schedule()
        agent_name_keys = list(handle._balanced_schedule[int(epoch) - 1])
        adversarial_name_keys = (
            [c for c in agent_name_keys if c in getattr(handle, "adversarial_namelist", [])]
            if handle.params["is_poison"]
            else []
        )
        logger.info(f"[balanced_schedule] epoch={epoch} clients={agent_name_keys}")
    elif sampling_strategy in ["dba_complete_shards", "complete_shards"]:
        if not is_dba_attack(handle.params):
            raise ValueError("dba_complete_shards sampling requires attack_mode=dba")
        if getattr(handle, "_frozen_schedule", None) is None:
            handle._frozen_schedule = handle.build_dba_complete_shards_schedule()
            handle.schedule_type = "dba_complete_shards"
        epoch_index = int(epoch) - 1
        if epoch_index < 0 or epoch_index >= len(handle._frozen_schedule):
            raise IndexError(
                f"dba_complete_shards epoch {epoch} is outside 1..{len(handle._frozen_schedule)}"
            )
        agent_name_keys = list(handle._frozen_schedule[epoch_index])
        adversarial_name_keys = (
            [
                client_id
                for client_id in agent_name_keys
                if client_id in getattr(handle, "adversarial_namelist", [])
            ]
            if handle.params["is_poison"]
            else []
        )
        logger.info(
            f"[dba_complete_shards] epoch={epoch} attackers={adversarial_name_keys} clients={agent_name_keys}"
        )
    else:
        adversarial_name_keys = []
        if handle.params["is_poison"] is True:
            agent_name_keys = random.sample(handle.participants_list, handle.params["num_clients"])
            for client in agent_name_keys:
                if (
                    hasattr(handle, "adversarial_namelist")
                    and client in handle.adversarial_namelist
                ):
                    adversarial_name_keys.append(client)
        else:
            agent_name_keys = random.sample(handle.participants_list, handle.params["num_clients"])
    if getattr(handle, "participation_counts", None) is None:
        handle.participation_counts = {cid: 0 for cid in handle.participants_list}
    for cid in agent_name_keys:
        handle.participation_counts[cid] = handle.participation_counts.get(cid, 0) + 1
    if not hasattr(handle, "round_client_schedule") or handle.round_client_schedule is None:
        handle.round_client_schedule = []
    handle.round_client_schedule.append(list(agent_name_keys))
    logger.info(f"Server Epoch:{epoch} choose agents : {agent_name_keys}.")
    return (agent_name_keys, adversarial_name_keys)


def Aggregation(w_ori, w, lens, beta, eta, defence_method, params):
    method = str(defence_method).lower() if defence_method is not None else ""

    def _is_float_tensor(v):
        return isinstance(v, torch.Tensor) and v.is_floating_point()

    if defence_method == "Trmean" or method in ["trmean", "trimmed_mean", "tm"]:
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            w_ori[k] = torch.zeros_like(w_ori[k])
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            ws = []
            for i in range(len(w)):
                if _is_float_tensor(w[i].get(k)):
                    ws.append(w[i][k])
            if not ws:
                continue
            ws = torch.stack(ws)
            ws = torch.sort(ws, dim=0)[0]
            a = int(len(w) * eta)
            ws = ws[a : len(w) - a]
            w_ori[k] = torch.mean(ws, dim=0)
    elif defence_method == "Krum" or method == "krum":
        dist = torch.zeros(lens, lens).to(device)
        for i in range(lens):
            for j in range(lens):
                for k in w_ori.keys():
                    if not _is_float_tensor(w_ori[k]):
                        continue
                    if not _is_float_tensor(w[i].get(k)) or not _is_float_tensor(w[j].get(k)):
                        continue
                    dist[i][j] += torch.norm(w[i][k] - w[j][k]) ** 2
        for i in range(lens):
            dist[i][i] = 10000000000.0
        min_score = 1e20
        min_idx = 0
        for i in range(lens):
            score = 0
            for j in range(lens - int(eta * lens) - 2):
                score += torch.sort(dist[i])[0][j]
            if score < min_score:
                min_score = score
                min_idx = i
        for k in w_ori.keys():
            w_ori[k] = w[min_idx][k]
    elif defence_method == "Krum-avg" or method in ["krum-avg", "krum_avg", "krumavg"]:
        dist = torch.zeros(lens, lens).to(device)
        for i in range(lens):
            for j in range(lens):
                for k in w_ori.keys():
                    if not _is_float_tensor(w_ori[k]):
                        continue
                    if not _is_float_tensor(w[i].get(k)) or not _is_float_tensor(w[j].get(k)):
                        continue
                    dist[i][j] += torch.norm(w[i][k] - w[j][k]) ** 2
        for i in range(lens):
            dist[i][i] = 10000000000.0
        scores = []
        for i in range(lens):
            score = 0
            for j in range(lens - int(eta * lens) - 2):
                score += torch.sort(dist[i])[0][j]
            scores.append(score)
        ws = []
        for i in range(lens):
            ws.append((w[i], scores[i]))
        ws = sorted(ws, key=lambda x: x[1])
        ws = ws[0 : int(lens - int(eta * lens) - 2)]
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            w_ori[k] = torch.zeros_like(w_ori[k])
        for i in range(len(ws)):
            for k in w_ori.keys():
                if not _is_float_tensor(w_ori[k]):
                    continue
                w_ori[k] = w_ori[k] + ws[i][0][k]
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            w_ori[k] = w_ori[k] / len(ws)
    elif method == "flame":
        active_indices = [i for i in range(len(w)) if i < len(beta) and beta[i] != 0]
        if not active_indices:
            return w_ori
        norms = []
        updates = []
        for i in active_indices:
            update = {}
            norm_sq = 0.0
            for k in w_ori.keys():
                if not _is_float_tensor(w_ori[k]):
                    continue
                if not _is_float_tensor(w[i].get(k)):
                    continue
                delta = w[i][k] - w_ori[k]
                update[k] = delta
                norm_sq += float(torch.sum(delta.float() ** 2).item())
            norms.append(math.sqrt(norm_sq))
            updates.append(update)
        clip_value = params.get("flame_clip", None)
        if clip_value is None:
            clip_value = float(torch.median(torch.tensor(norms)).item() if norms else 0.0)
            clip_value *= float(params.get("flame_clip_scale", 1.0))
        clip_value = max(float(clip_value), 1e-12)
        eps = 1e-08
        for idx, update in enumerate(updates):
            norm = norms[idx]
            scale = min(1.0, clip_value / (norm + eps))
            for k in update.keys():
                update[k] = update[k] * scale
        avg_update = {}
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            avg_update[k] = torch.zeros_like(w_ori[k], dtype=torch.float32)
        for update in updates:
            for k in avg_update.keys():
                avg_update[k] = avg_update[k] + update[k].float()
        for k in avg_update.keys():
            avg_update[k] = avg_update[k] / max(1, len(updates))
        noise_multiplier = float(params.get("flame_noise_multiplier", 0.0))
        if noise_multiplier > 0:
            std = noise_multiplier * clip_value / math.sqrt(max(1, len(updates)))
            for k in avg_update.keys():
                avg_update[k] = avg_update[k] + torch.normal(
                    mean=0.0, std=std, size=avg_update[k].shape
                )
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            w_ori[k] = w_ori[k] + eta * avg_update[k].to(w_ori[k].device)
        return w_ori
    elif method in ["fltrust", "safefl", "fltrust_mpc"]:
        weights = params.get("_fltrust_weight_runtime", None)
        active_indices = [i for i in range(len(w)) if i < len(beta) and beta[i] != 0]
        if not active_indices:
            return w_ori
        if not weights:
            weights = {i: 1.0 / len(active_indices) for i in active_indices}
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            update = torch.zeros_like(w_ori[k], dtype=torch.float32)
            for i in active_indices:
                if not _is_float_tensor(w[i].get(k)):
                    continue
                weight = float(weights.get(i, 0.0))
                if weight == 0.0:
                    continue
                update = update + (w[i][k].float() - w_ori[k].float()) * weight
            w_ori[k] = w_ori[k] + eta * update.to(w_ori[k].dtype).to(w_ori[k].device)
        return w_ori
    elif method == "shieldfl":
        runtime = params.get("_shieldfl_runtime", None)
        expected_round = params.get("_fedavg_round", None)
        active_indices = [i for i in range(len(w)) if i < len(beta) and beta[i] != 0]
        if not active_indices:
            return w_ori
        if (
            runtime is not None
            and expected_round is not None
            and (int(runtime.get("round", -1)) != int(expected_round))
        ):
            raise RuntimeError(
                f"ShieldFL runtime is from round {runtime.get('round')} but this is round {expected_round}"
            )
        if runtime is None:
            weights = {i: 1.0 / len(active_indices) for i in active_indices}
            for k in w_ori.keys():
                if not _is_float_tensor(w_ori[k]):
                    continue
                avg_update = torch.zeros_like(w_ori[k], dtype=torch.float32)
                for i in active_indices:
                    if not _is_float_tensor(w[i].get(k)):
                        continue
                    avg_update = avg_update + (w[i][k].float() - w_ori[k].float()) * weights[i]
                w_ori[k] = w_ori[k] + eta * avg_update.to(w_ori[k].dtype).to(w_ori[k].device)
            return w_ori
        if set(active_indices) != set((int(i) for i in runtime["active_indices"])):
            raise RuntimeError(
                "ShieldFL runtime active set does not match the aggregation's active set"
            )
        confidence = {int(i): float(v) for i, v in runtime["weights"].items()}
        handled = set()
        if bool(runtime.get("aggregate_normalized", True)):
            direction = runtime["direction"]
            scale = float(runtime["scale"])
            offset = 0
            for name in runtime["param_names"]:
                if name not in w_ori or not _is_float_tensor(w_ori[name]):
                    continue
                numel = w_ori[name].numel()
                piece = direction[offset : offset + numel].view(w_ori[name].shape)
                offset += numel
                step = float(eta) * scale * piece.to(w_ori[name].device, torch.float32)
                w_ori[name] = (w_ori[name].float() - step).to(w_ori[name].dtype)
                handled.add(name)
            if offset != direction.numel():
                raise RuntimeError("ShieldFL direction length does not match the parameter vector")
        for k in w_ori.keys():
            if k in handled or not _is_float_tensor(w_ori[k]):
                continue
            avg_update = torch.zeros_like(w_ori[k], dtype=torch.float32)
            for i in active_indices:
                if not _is_float_tensor(w[i].get(k)):
                    continue
                weight = confidence.get(i, 0.0)
                if weight == 0.0:
                    continue
                avg_update = avg_update + (w[i][k].float() - w_ori[k].float()) * weight
            w_ori[k] = w_ori[k] + eta * avg_update.to(w_ori[k].dtype).to(w_ori[k].device)
        return w_ori
    elif method in ["rofl", "elsa"]:
        runtime = params.get("_input_validation_runtime", None)
        expected_round = params.get("_fedavg_round", None)
        active_indices = [i for i in range(len(w)) if i < len(beta) and beta[i] != 0]
        if not active_indices:
            return w_ori
        if runtime is not None:
            if expected_round is not None and int(runtime.get("round", -1)) != int(expected_round):
                raise RuntimeError(
                    f"[{method}] input-validation runtime is from round {runtime.get('round')} but this is round {expected_round}"
                )
            if set(active_indices) != set((int(i) for i in runtime["accepted_indices"])):
                raise RuntimeError(
                    f"[{method}] aggregation active set {active_indices} differs from the validated set {runtime['accepted_indices']}"
                )
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            avg_update = torch.zeros_like(w_ori[k], dtype=torch.float32)
            for i in active_indices:
                if not _is_float_tensor(w[i].get(k)):
                    continue
                avg_update = avg_update + (w[i][k].float() - w_ori[k].float())
            avg_update = avg_update / float(len(active_indices))
            w_ori[k] = w_ori[k] + eta * avg_update.to(w_ori[k].dtype).to(w_ori[k].device)
        return w_ori
    else:
        for i in range(len(w)):
            for k in w_ori.keys():
                if not _is_float_tensor(w_ori[k]):
                    continue
                if not _is_float_tensor(w[i].get(k)):
                    continue
                w[i][k] = (w[i][k] - w_ori[k]) * beta[i]
        w_avg = None
        for i in range(len(w)):
            if i == 0:
                w_avg = copy.deepcopy(w[0])
                for k in w_avg.keys():
                    if not _is_float_tensor(w_avg[k]):
                        continue
                    w_avg[k] = w[i][k]
            else:
                for k in w_avg.keys():
                    if not _is_float_tensor(w_avg[k]):
                        continue
                    w_avg[k] += w[i][k]
        for k in w_ori.keys():
            if not _is_float_tensor(w_ori[k]):
                continue
            w_avg[k] = torch.div(w_avg[k], lens)
            w_ori[k] += eta * w_avg[k]
    return w_ori


def _fixed_tau_require_target_top(params):
    value = params.get("fixed_tau_require_target_top", True)
    if type(value) is not bool:
        raise ValueError("fixed_tau_require_target_top must be a boolean")
    return value


DP_CBF_STATS_PER_CLIENT = "per_client_noisy_model"
DP_CBF_STATS_AGGREGATE_REF = "aggregate_noisy_reference"


def canonical_dp_cbf_stats_source(value):
    """Normalize the BN statistics source identifier."""
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return DP_CBF_STATS_PER_CLIENT
    return text


def validate_dp_cbf_stats_source_pairing(params):
    """Check that the release route provides the selected BN statistics source."""
    stats_source = canonical_dp_cbf_stats_source(params.get("dp_cbf_stats_source"))
    if stats_source != DP_CBF_STATS_AGGREGATE_REF:
        return stats_source
    release_source = canonical_dp_release_source(
        params.get("dp_release_source", DP_RELEASE_CLIENT_LDP)
    )
    if release_source not in [DP_RELEASE_MPC_AGGREGATE_FAST]:
        raise ValueError(
            f"dp_cbf_stats_source=aggregate_noisy_reference requires dp_release_source=mpc_aggregate_ref_dp_fast or mpc_aggregate_ref_dp_secure_clip_debug because only those routes open an aggregate noisy reference model; received dp_release_source={release_source}"
        )
    return stats_source


def _read_ldp_config(handle):
    params = handle.params
    detector = canonical_final_detector(params.get("final_detector", FIXED_TAU))
    mode = canonical_detector_output_mode(params.get("detector_output_mode", DEBUG_LOGITS))
    params["final_detector"] = detector
    params["detector_output_mode"] = mode
    enable_ldp = bool(params.get("enable_ldp", False))
    if params.get("consistency_mechanism", "none") not in ("none", None, ""):
        raise ValueError("This release has no legacy split-upload consistency path")
    if detector == CALIBRATED_CENTERED_SCORE:
        if not enable_ldp:
            raise ValueError("calibrated_centered_score requires enable_ldp=true")
        if mode == DIAGNOSTIC_SCORES:
            _validate_plaintext_detector_screen_config(params)
        elif mode == ACCEPT_BITS_ONLY:
            _cryptgpu_detector_output_boundary(params)
            if _is_simulated_mpc_mode(params):
                raise ValueError("accept_bits_only requires real CryptGPU")
        else:
            raise ValueError(
                "calibrated_centered_score requires diagnostic_scores or accept_bits_only"
            )
    elif mode != DEBUG_LOGITS:
        raise ValueError("fixed_tau requires debug_logits")
    if enable_ldp:
        source = canonical_dp_release_source(params.get("dp_release_source", DP_RELEASE_CLIENT_LDP))
        params["dp_release_source"] = source
        validate_dp_cbf_stats_source_pairing(params)
        if source != DP_RELEASE_CLIENT_LDP:
            if source != DP_RELEASE_MPC_AGGREGATE_FAST:
                raise ValueError("Only aggregate-reference MPC DP is supported")
            params["ldp_accountant_variant"] = "balanced_direct_composition"
            params["mpc_dp_output_mode"] = "aggregate_reference"
            params.setdefault("mpc_dp_clip_mode", "local_reference_clip_then_mpc_sum")
    attack_mode = str(params.get("attack_mode", "")).strip().lower()
    if attack_mode in PRIV_FILTERFL_DETECTOR_EVASION_ATTACK_MODES:
        if int(params.get("adaptive_evasion_start_round", 2)) < 2:
            raise ValueError("adaptive_evasion_start_round must be >=2")
        source = params.get("adaptive_evasion_trigger_source", EVASION_TRIGGER_SOURCE_COORDINATOR)
        if source not in SUPPORTED_EVASION_TRIGGER_SOURCES:
            raise ValueError("Unsupported adaptive_evasion_trigger_source")
        staleness = int(params.get("adaptive_evasion_trigger_staleness", 1))
        if staleness < 1 or (
            source == EVASION_TRIGGER_SOURCE_COORDINATOR
            and int(params.get("adaptive_evasion_start_round", 2)) <= staleness
        ):
            raise ValueError("Invalid adaptive probe staleness")
    return enable_ldp


def _robust_detector_runtime_config(params, final_detector, detector_output_mode):
    """Parse the public constants used by the selected detector."""
    config = {"final_detector": final_detector, "detector_output_mode": detector_output_mode}
    if final_detector not in {CALIBRATED_CENTERED_SCORE}:
        return config

    def finite_number(name, *, positive=False):
        value = params.get(name)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"{final_detector} requires finite numeric {name}")
        parsed = float(value)
        if positive and parsed <= 0.0:
            raise ValueError(f"{final_detector} requires positive {name}")
        return parsed

    def minimum_clients():
        value = params.get("robust_detector_min_clients")
        if type(value) is not int or value < 2:
            raise ValueError(f"{final_detector} requires integer robust_detector_min_clients>=2")
        return value

    if final_detector == CALIBRATED_CENTERED_SCORE:
        center_weight = finite_number("robust_detector_center_weight")
        if center_weight < 0.0 or center_weight > 1.0:
            raise ValueError(f"{final_detector} requires robust_detector_center_weight in [0, 1]")
        center_mode = params.get("robust_detector_center_mode")
        try:
            center_mode = canonical_center_mode(center_mode)
        except ValueError as exc:
            raise ValueError(
                f"{final_detector} requires valid robust_detector_center_mode"
            ) from exc
        params["robust_detector_center_mode"] = center_mode
        score_type = canonical_calibrated_score_type(params.get("robust_detector_score_type"))
        params["robust_detector_score_type"] = score_type
        config.update(
            {
                "threshold_q": finite_number("robust_detector_threshold_q"),
                "center_weight": center_weight,
                "center_mode": center_mode,
                "score_type": score_type,
                "min_clients": minimum_clients(),
            }
        )
        return config
    require_guard = params.get("robust_detector_require_guard")
    if type(require_guard) is not bool:
        raise ValueError(f"{final_detector} requires boolean robust_detector_require_guard")
    config.update(
        {
            "z_threshold": finite_number("robust_detector_z_threshold", positive=True),
            "variance_floor": finite_number("robust_detector_variance_floor", positive=True),
            "min_clients": minimum_clients(),
            "guard_margin": finite_number("robust_detector_guard_margin"),
            "require_guard": require_guard,
        }
    )
    return config


def _cryptgpu_detector_output_boundary(params):
    """Validate output and privacy settings before preparing CryptGPU inputs."""
    fail_closed = bool(params.get("cryptgpu_fail_closed", True))
    final_detector = canonical_final_detector(params.get("final_detector", FIXED_TAU))
    detector_output_mode = canonical_detector_output_mode(
        params.get("detector_output_mode", DEBUG_LOGITS)
    )
    robust_detector = final_detector in {CALIBRATED_CENTERED_SCORE}
    robust_bit_only = robust_detector and detector_output_mode == ACCEPT_BITS_ONLY
    if robust_detector != (detector_output_mode == ACCEPT_BITS_ONLY):
        raise ValueError(
            "secure cohort-score detectors and accept_bits_only must be enabled together; the legacy fixed_tau detector remains debug_logits for exact backward compatibility"
        )
    cryptgpu_compare_debug = bool(params.get("cryptgpu_compare_debug", False))
    if robust_bit_only:
        incompatible_diagnostics = [
            name
            for name in (
                "record_detection_diagnostics",
                "record_bn_diagnostics",
                "cryptgpu_compare_debug",
            )
            if bool(params.get(name, False))
        ]
        if incompatible_diagnostics:
            raise ValueError(
                "accept_bits_only forbids plaintext diagnostics: "
                + ", ".join(incompatible_diagnostics)
            )
        if not fail_closed:
            raise ValueError("accept_bits_only requires cryptgpu_fail_closed=true")
    return {
        "fail_closed": fail_closed,
        "final_detector": final_detector,
        "detector_output_mode": detector_output_mode,
        "robust_bit_only": robust_bit_only,
        "cryptgpu_compare_debug": cryptgpu_compare_debug,
    }


def _client_selection(epoch, handle):
    agent_name_keys, adversarial_name_keys = get_clients(epoch, handle)
    lens = len(agent_name_keys)
    # Testing loads the new global model; both releases retain the round-start base.
    ori_weight = copy.deepcopy(handle.model.state_dict())
    return (agent_name_keys, adversarial_name_keys, lens, ori_weight)


def _record_coalition_held_global(handle, epoch, adversarial_name_keys):
    """Retain published models received by sampled members of the attacker coalition."""
    params = handle.params
    attack_mode = str(params.get("attack_mode", params.get("attack.mode", ""))).strip().lower()
    if attack_mode not in PRIV_FILTERFL_DETECTOR_EVASION_ATTACK_MODES:
        return
    if (
        str(params.get("adaptive_evasion_selfsynth_global_source", "")).strip().lower()
        != SELFSYNTH_GLOBAL_SOURCE_COALITION_HELD
    ):
        return
    if not adversarial_name_keys:
        return
    published = getattr(handle, "prev_noisy_w_glob", None)
    if published is None:
        return
    held = getattr(handle, "attacker_held_globals", None)
    if held is None:
        held = collections.deque(maxlen=4)
        handle.attacker_held_globals = held
    held.append({"round": int(epoch), "state": copy.deepcopy(published)})
    logger.info(
        f"[adaptive_attack][self_synth] round={epoch}: coalition received the published global (members sampled: {len(adversarial_name_keys)}); coalition now holds {len(held)} global(s)"
    )


def _maybe_prepare_self_synth_attacker_trigger(handle, epoch):
    """Synthesize the attacker probe from its two most recent published models.

    The result is stored on ``handle`` for the current round of local training.
    """
    handle.attacker_self_synth_trigger = None
    handle.attacker_self_synth_trigger_round = None
    handle.attacker_self_synth_trigger_digest = None
    handle.attacker_self_synth_trigger_digest_schema = None
    handle.attacker_self_synth_trigger_status = "inactive"
    params = handle.params
    attack_mode = str(params.get("attack_mode", params.get("attack.mode", ""))).strip().lower()
    if attack_mode not in PRIV_FILTERFL_DETECTOR_EVASION_ATTACK_MODES:
        return
    if not bool(params.get("is_poison", False)):
        return
    trigger_source = (
        str(params.get("adaptive_evasion_trigger_source", EVASION_TRIGGER_SOURCE_COORDINATOR))
        .strip()
        .lower()
    )
    if trigger_source != EVASION_TRIGGER_SOURCE_SELF_SYNTH:
        return
    if int(epoch) < int(params.get("adaptive_evasion_start_round", 2)):
        handle.attacker_self_synth_trigger_status = "before_start_round"
        return
    global_source = (
        str(
            params.get(
                "adaptive_evasion_selfsynth_global_source", SELFSYNTH_GLOBAL_SOURCE_COORDINATOR_RING
            )
        )
        .strip()
        .lower()
    )
    if global_source not in SUPPORTED_SELFSYNTH_GLOBAL_SOURCES:
        raise ValueError(
            f"adaptive_evasion_selfsynth_global_source must be one of {sorted(SUPPORTED_SELFSYNTH_GLOBAL_SOURCES)}"
        )
    pair_gap = None
    if global_source == SELFSYNTH_GLOBAL_SOURCE_COALITION_HELD:
        held = getattr(handle, "attacker_held_globals", None)
        if held is None or len(held) < 2:
            handle.attacker_self_synth_trigger_status = "insufficient_coalition_held_globals"
            logger.warning(
                f"[adaptive_attack][self_synth] round={epoch}: coalition holds fewer than two published globals; probe not synthesized"
            )
            return
        newer, older = (held[-1], held[-2])
        g_recent, g_older = (newer["state"], older["state"])
        pair_gap = int(newer["round"]) - int(older["round"])
    else:
        g_recent = getattr(handle, "prev_noisy_w_glob", None)
        g_older = getattr(handle, "prev_noisy_w_glob_prev", None)
        pair_gap = 1
    if g_recent is None or g_older is None:
        handle.attacker_self_synth_trigger_status = "missing_self_synth_globals"
        logger.warning(
            f"[adaptive_attack][self_synth] round={epoch}: two published globals not yet available; probe not synthesized"
        )
        return
    try:
        g_imgs, _ = _generate_trigger(
            copy.deepcopy(g_older), copy.deepcopy(g_recent), params, epoch
        )
    except Exception as exc:
        handle.attacker_self_synth_trigger_status = f"self_synth_failed:{exc}"
        logger.warning(
            f"[adaptive_attack][self_synth] round={epoch}: synthesis failed ({exc}); probe not synthesized"
        )
        return
    if not g_imgs:
        handle.attacker_self_synth_trigger_status = "self_synth_empty"
        return
    probe = g_imgs[0].detach().to("cpu", dtype=torch.float32).contiguous().clone()
    digest = train._detector_probe_digest(probe)
    coordinator_digest = getattr(handle, "previous_detection_trigger_digest", None)
    handle.attacker_self_synth_trigger = probe
    handle.attacker_self_synth_trigger_round = int(epoch)
    handle.attacker_self_synth_trigger_digest = digest
    handle.attacker_self_synth_trigger_digest_schema = train._DETECTOR_PROBE_DIGEST_SCHEMA
    handle.attacker_self_synth_trigger_status = "ready"
    handle.attacker_self_synth_pair_gap = pair_gap
    logger.info(
        f"[adaptive_attack][self_synth] round={epoch}: synthesized attacker probe digest={digest[:12]} from {global_source} globals (pair_gap={pair_gap}); coordinator_prev_digest={('none' if coordinator_digest is None else coordinator_digest[:12])}"
    )


def _train_clients(epoch, handle, agent_name_keys, adversarial_name_keys, enable_ldp):
    w_locals = []
    w_locals_clean = []
    w_locals_noisy = []
    beta = []
    defence_method = str(handle.params.get("defence_method", "")).lower()
    lockdown_enabled = defence_method == "lockdown"
    if lockdown_enabled and (not hasattr(handle, "lockdown_states")):
        handle.lockdown_states = {}
    dp_release_source = canonical_dp_release_source(
        handle.params.get("dp_release_source", DP_RELEASE_CLIENT_LDP)
    )
    client_generates_noisy = enable_ldp and dp_release_source == DP_RELEASE_CLIENT_LDP
    _record_coalition_held_global(handle, epoch, adversarial_name_keys)
    _maybe_prepare_self_synth_attacker_trigger(handle, epoch)
    for client in agent_name_keys:
        try:
            from memory_utils import clear_cuda_memory

            clear_cuda_memory("before_client_train")
        except Exception:
            pass
        lockdown_state = None
        if lockdown_enabled:
            lockdown_state = handle.lockdown_states.get(client)
            if lockdown_state is None:
                lockdown_state = {}
                handle.lockdown_states[client] = lockdown_state
        if client in adversarial_name_keys:
            w = train.standard_train(
                epoch,
                handle.clients_data_num[client],
                client,
                handle.params,
                copy.deepcopy(handle.model).to(device),
                handle.train_data[client],
                handle,
                poison=True,
                enable_ldp=client_generates_noisy,
                lockdown_state=lockdown_state,
                lockdown_config=handle.params,
            )
        else:
            w = train.standard_train(
                epoch,
                handle.clients_data_num[client],
                client,
                handle.params,
                copy.deepcopy(handle.model).to(device),
                handle.train_data[client],
                handle,
                poison=False,
                enable_ldp=client_generates_noisy,
                lockdown_state=lockdown_state,
                lockdown_config=handle.params,
            )
        beta.append(1)
        if enable_ldp and isinstance(w, dict) and ("clean" in w) and ("noisy" in w):
            w_locals_clean.append(w["clean"])
            w_locals_noisy.append(w["noisy"])
            w_locals.append(w["noisy"])
            logger.debug(f"[LDP] Client {client} submitted clean and noisy weights")
        else:
            w_locals.append(w)
            if enable_ldp:
                w_locals_clean.append(w)
                w_locals_noisy.append(w)
        try:
            from memory_utils import clear_cuda_memory

            clear_cuda_memory("after_client_train")
        except Exception:
            pass
    if enable_ldp and is_mpc_dp_source(dp_release_source):
        logger.info(
            "[MPC-DP] Clients did not upload local noisy releases; backend opens coordinator-visible release"
        )
        if not hasattr(handle, "mpc_dp_release_service") or handle.mpc_dp_release_service is None:
            handle.mpc_dp_release_service = MpcDpReleaseService(handle.params)
        ref_release = handle.mpc_dp_release_service.make_fast_aggregate_reference_release(
            clean_client_weights=w_locals_clean,
            global_model=handle.model.to("cpu").state_dict(),
            client_ids=agent_name_keys,
            round_id=epoch,
            accept_mask=None,
            release_role="unfiltered_pre_detection",
        )
        handle.mpc_dp_aggregate_reference = ref_release
        w_locals_noisy = [ref_release for _ in agent_name_keys]
        w_locals = list(w_locals_noisy)
        if not hasattr(handle, "mpc_dp_release_metrics") or handle.mpc_dp_release_metrics is None:
            handle.mpc_dp_release_metrics = []
        handle.mpc_dp_release_metrics = list(handle.mpc_dp_release_service.metrics)
    return (w_locals, w_locals_clean, w_locals_noisy, beta)


def _aggregate_for_trigger(handle, w_locals, w_locals_noisy, lens, beta, enable_ldp, epoch):
    source = canonical_dp_release_source(
        handle.params.get("dp_release_source", DP_RELEASE_CLIENT_LDP)
    )
    aggregate_ref = getattr(handle, "mpc_dp_aggregate_reference", None)
    if enable_ldp and source in [DP_RELEASE_MPC_AGGREGATE_FAST] and (aggregate_ref is not None):
        logger.info(
            "[MPC-DP][trigger] Use single MPC-opened aggregate noisy reference model directly"
        )
        return copy.deepcopy(aggregate_ref)
    weights_for_aggregation = w_locals_noisy if enable_ldp and w_locals_noisy else w_locals
    if enable_ldp and w_locals_noisy:
        weights_for_aggregation = _prepare_coordinator_noisy_view(
            handle, weights_for_aggregation, epoch=epoch
        )
    beta_for_trigger = beta
    lens_for_trigger = lens
    if enable_ldp:
        logger.info("[LDP] Aggregate with noisy weights (for trigger image generation)")
    else:
        logger.info("[LDP] Aggregate with clean weights (for trigger image generation)")
    w_glob = Aggregation(
        copy.deepcopy(handle.model.to("cpu").state_dict()),
        copy.deepcopy(weights_for_aggregation),
        lens_for_trigger,
        beta_for_trigger,
        handle.params["eta"],
        handle.params["defence_method"],
        handle.params,
    )
    return w_glob


def _prepare_coordinator_noisy_view(handle, noisy_weights, epoch):
    mode = str(handle.params.get("coordinator_noisy_transcript", "linked")).strip().lower()
    if not noisy_weights or mode in ["", "linked", "client_linked"]:
        return noisy_weights
    if mode not in ["anonymous_shuffled", "anonymous", "shuffled"]:
        raise ValueError(f"Unsupported coordinator_noisy_transcript mode: {mode}")
    perm = list(range(len(noisy_weights)))
    seed = int(handle.params.get("seed", 1))
    rng = random.Random(seed * 1000003 + int(epoch) * 9176 + len(noisy_weights))
    rng.shuffle(perm)
    shuffled = [copy.deepcopy(noisy_weights[idx]) for idx in perm]
    logger.info(
        f"[LDP][anonymous_view] epoch={epoch} applied shuffled unlabeled noisy transcript for coordinator-visible DP branch"
    )
    return shuffled


def _generate_trigger(ori_weight, w_glob, params, epoch):

    def _sanitize_state_dict(sd, tag):
        out = {}
        bad_keys = []
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                t = v.detach().to("cpu")
                if t.is_floating_point() and (not torch.isfinite(t).all()):
                    bad_keys.append(k)
                    t = torch.nan_to_num(t, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
                out[k] = t
            else:
                out[k] = v
        if bad_keys:
            logger.warning(
                f"[trigger_sanitize] Non-finite values detected in {tag}: count={len(bad_keys)}, examples={bad_keys[:5]}"
            )
        return out

    ori_weight = _sanitize_state_dict(ori_weight, "ori_weight")
    w_glob = _sanitize_state_dict(w_glob, "w_glob")
    g_imgs = gan_defense(ori_weight, w_glob, params, the_epoch=epoch)
    try:
        g_imgs = [img.detach().to("cpu") if hasattr(img, "detach") else img for img in g_imgs]
    except Exception:
        pass
    if not g_imgs:
        trigger_tensor = torch.tensor([], device="cpu", dtype=torch.float32)
        return (g_imgs, trigger_tensor)
    sanitized_imgs = []
    nonfinite_imgs = 0
    for img in g_imgs:
        if isinstance(img, torch.Tensor):
            t = img.detach().to("cpu").float()
            if not torch.isfinite(t).all():
                nonfinite_imgs += 1
                t = torch.nan_to_num(t, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
            sanitized_imgs.append(t)
        else:
            sanitized_imgs.append(img)
    if nonfinite_imgs:
        logger.warning(
            f"[trigger_sanitize] Non-finite trigger images detected and sanitized: count={nonfinite_imgs}"
        )
    g_imgs = sanitized_imgs
    trigger_tensor = g_imgs[0].detach().to("cpu").float().view(-1)
    if not torch.isfinite(trigger_tensor).all():
        logger.warning(
            "[trigger_sanitize] Trigger tensor still non-finite after image sanitation; applying nan_to_num again"
        )
        trigger_tensor = torch.nan_to_num(
            trigger_tensor, nan=0.0, posinf=1000000.0, neginf=-1000000.0
        )
    try:
        _tt_sz = int(trigger_tensor.numel())
        _tt_head = trigger_tensor[:5].tolist() if _tt_sz >= 5 else trigger_tensor.tolist()
        _tt_tail = trigger_tensor[-5:].tolist() if _tt_sz >= 5 else trigger_tensor.tolist()
        logger.info(f"[align_debug] trigger_size={_tt_sz}, head={_tt_head}, tail={_tt_tail}")
    except Exception as _e:
        logger.debug(f"[align_debug] trigger summary failed: {_e}")
    return (g_imgs, trigger_tensor)


def _extract_running_bn_stats(model):
    bn_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_stats[name] = {
                "mean": m.running_mean.detach().clone(),
                "var": m.running_var.detach().clone(),
            }
    return bn_stats


def _flatten_state_dict_params(state_dict, param_names):
    vecs = []
    for name in param_names:
        if name not in state_dict:
            continue
        v = state_dict[name]
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            vecs.append(v.detach().to("cpu").float().view(-1))
    if not vecs:
        return torch.tensor([], device="cpu", dtype=torch.float32)
    return torch.cat(vecs, dim=0)


def _flatten_model_grads(model, param_names):
    vecs = []
    for name, param in model.named_parameters():
        if name not in param_names:
            continue
        if param.grad is None:
            continue
        vecs.append(param.grad.detach().to("cpu").float().view(-1))
    if not vecs:
        return torch.tensor([], device="cpu", dtype=torch.float32)
    return torch.cat(vecs, dim=0)


def _kmeans_cosine(vectors, k=2, max_iter=10, eps=1e-08):
    if not vectors or len(vectors) < k:
        return [0 for _ in vectors]
    vecs = []
    for v in vectors:
        n = torch.norm(v)
        vecs.append(v / (n + eps) if n > 0 else v.clone())
    centroids = []
    norms = [torch.norm(v).item() for v in vecs]
    first_idx = int(torch.tensor(norms).argmax().item())
    centroids.append(vecs[first_idx])
    while len(centroids) < k:
        sims = []
        for v in vecs:
            sim = max([torch.dot(v, c).item() for c in centroids])
            sims.append(sim)
        next_idx = int(torch.tensor(sims).argmin().item())
        centroids.append(vecs[next_idx])
    labels = [0 for _ in vecs]
    for _ in range(max_iter):
        changed = False
        for i, v in enumerate(vecs):
            sims = [torch.dot(v, c).item() for c in centroids]
            new_label = int(torch.tensor(sims).argmax().item())
            if new_label != labels[i]:
                labels[i] = new_label
                changed = True
        for c in range(k):
            members = [vecs[i] for i, lab in enumerate(labels) if lab == c]
            if not members:
                centroids[c] = vecs[random.randrange(len(vecs))]
            else:
                mean = torch.mean(torch.stack(members, dim=0), dim=0)
                n = torch.norm(mean)
                centroids[c] = mean / (n + eps) if n > 0 else mean
        if not changed:
            break
    return labels


def _robust_threshold(values, z=2.5):
    if not values:
        return (0.0, 0.0, 0.0)
    vals = torch.tensor(values, dtype=torch.float32)
    med = float(torch.median(vals).item())
    mad = float(torch.median(torch.abs(vals - med)).item())
    scale = 1.4826 * mad if mad > 0 else 1.0
    thr = med + z * scale
    return (med, scale, thr)


def _prepare_active_client_weights_for_detection(
    handle, enable_ldp, w_locals, w_locals_clean, w_locals_noisy, ori_weight, beta
):
    from memory_utils import state_dict_to_cpu

    if enable_ldp:
        use_noisy_detect = bool(handle.params.get("ldp_detection_use_noisy", False))
        noisy_mode = (
            str(handle.params.get("coordinator_noisy_transcript", "linked")).strip().lower()
        )
        if use_noisy_detect and noisy_mode in ["anonymous_shuffled", "anonymous", "shuffled"]:
            raise RuntimeError(
                "ldp_detection_use_noisy is incompatible with coordinator_noisy_transcript=anonymous_shuffled because the noisy DP branch cannot retain client linkage."
            )
        if use_noisy_detect and w_locals_noisy:
            weights_for_detection = w_locals_noisy
        elif w_locals_clean:
            weights_for_detection = w_locals_clean
        else:
            weights_for_detection = w_locals
    else:
        weights_for_detection = w_locals
    ori_weight_cpu = state_dict_to_cpu(ori_weight)
    active_indices = []
    client_weights_active = []
    client_weights_noisy_for_bn = []
    for i in range(handle.params["num_clients"]):
        if beta[i] == 0:
            continue
        src_sd = weights_for_detection[i]
        parameter = {}
        for k, v in src_sd.items():
            if isinstance(v, torch.Tensor):
                try:
                    v_cpu = v.detach().to("cpu")
                except Exception:
                    v_cpu = (
                        torch.as_tensor(v).detach().to("cpu")
                        if not isinstance(v, torch.Tensor)
                        else v.detach().to("cpu")
                    )
                base = ori_weight_cpu[k]
                parameter[k] = (v_cpu - base) * beta[i] + base
            else:
                parameter[k] = v
        active_indices.append(i)
        client_weights_active.append(parameter)
        stats_source = canonical_dp_cbf_stats_source(handle.params.get("dp_cbf_stats_source"))
        aggregate_ref = getattr(handle, "mpc_dp_aggregate_reference", None)
        if enable_ldp and stats_source == DP_CBF_STATS_AGGREGATE_REF:
            if aggregate_ref is None:
                raise RuntimeError(
                    "dp_cbf_stats_source=aggregate_noisy_reference requires an MPC-opened aggregate noisy reference, but none was produced this round; check dp_release_source"
                )
            src_noisy_sd = aggregate_ref
        elif enable_ldp and w_locals_noisy and (len(w_locals_noisy) > i):
            src_noisy_sd = w_locals_noisy[i]
        else:
            src_noisy_sd = None
        if src_noisy_sd is not None:
            noisy_param = {}
            for k, v in src_noisy_sd.items():
                if isinstance(v, torch.Tensor):
                    try:
                        v_cpu = v.detach().to("cpu")
                    except Exception:
                        v_cpu = (
                            torch.as_tensor(v).detach().to("cpu")
                            if not isinstance(v, torch.Tensor)
                            else v.detach().to("cpu")
                        )
                    base = ori_weight_cpu[k]
                    noisy_param[k] = (v_cpu - base) * beta[i] + base
                else:
                    noisy_param[k] = v
            client_weights_noisy_for_bn.append(noisy_param)
            if bool(handle.params.get("bn_folding_use_running_stats", False)):
                _record_bn_buffer_discrepancy(handle, i, parameter, noisy_param)
        else:
            if bool(handle.params.get("cryptgpu_fail_closed", False)) and (
                bool(handle.params.get("cryptgpu_bn_use_noisy", True))
                or bool(handle.params.get("record_bn_diagnostics", False))
            ):
                raise RuntimeError(
                    f"[ours_cryptgpu] noisy/reference BN weights required by folding or diagnostics are unavailable for active client index {i}"
                )
            client_weights_noisy_for_bn.append(dict(parameter))
    return (active_indices, client_weights_active, client_weights_noisy_for_bn)


def _infer_detection_model_spec(params):
    t = str(params.get("type", "CIFAR10")).lower()
    arch_override = str(params.get("arch", "")).lower().strip()
    if t in ["mnist"]:
        return {"arch": "cnnmnist", "num_classes": 10, "in_channels": 1}
    if t in ["cifar10", "cifar-10"] and arch_override == "resnet50_cifar":
        return {"arch": "resnet50_cifar", "num_classes": 10, "in_channels": 3}
    if t in ["cifar10", "cifar-10"] and arch_override in ["resnet18", "resnet18_cifar"]:
        return {"arch": "resnet18", "num_classes": 10, "in_channels": 3}
    if t in ["cifar100", "cifar-100"]:
        return {"arch": "resnet18", "num_classes": 100, "in_channels": 3}
    if t in ["tinyimagenet", "tiny-imagenet"]:
        return {"arch": "resnet18", "num_classes": 200, "in_channels": 3}
    if t in ["gtsrb"]:
        return {"arch": "resnet8", "num_classes": 43, "in_channels": 3}
    return {"arch": "resnet8", "num_classes": 10, "in_channels": 3}


def _is_simulated_mpc_mode(params):
    mode = (
        str(
            params.get(
                "priv_filterfl_mpc_mode",
                params.get("cryptgpu_mpc_mode", params.get("mpc_mode", "cryptgpu")),
            )
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    if mode in ["simulated", "simulate", "simulation", "simulated_mpc", "mock_mpc", "plaintext"]:
        return True
    return bool(
        params.get("priv_filterfl_simulate_mpc", False)
        or params.get("cryptgpu_simulate_mpc", False)
        or params.get("simulate_mpc", False)
    )


PLAINTEXT_SCREEN_DP_PROFILE_CLIENT_LDP = "client_ldp_per_client_noisy_model"
PLAINTEXT_SCREEN_DP_PROFILE_MPC_HARDENING = "mpc_aggregate_ref_aggregate_noisy_reference"


def plaintext_screen_dp_profile(params):
    """Identify the matched release and BN-statistics profile for plaintext evaluation.

    Supported profiles use either client-noisy models or one noisy aggregate
    reference. Return ``None`` for an unsupported pairing.
    """
    stats_source = canonical_dp_cbf_stats_source(params.get("dp_cbf_stats_source"))
    release_source = canonical_dp_release_source(
        params.get("dp_release_source", DP_RELEASE_CLIENT_LDP)
    )
    if release_source == DP_RELEASE_CLIENT_LDP and stats_source == DP_CBF_STATS_PER_CLIENT:
        return PLAINTEXT_SCREEN_DP_PROFILE_CLIENT_LDP
    if (
        release_source in [DP_RELEASE_MPC_AGGREGATE_FAST]
        and stats_source == DP_CBF_STATS_AGGREGATE_REF
    ):
        return PLAINTEXT_SCREEN_DP_PROFILE_MPC_HARDENING
    return None


def _validate_plaintext_detector_screen_config(params):
    """Validate the backend, output mode and BN source for plaintext evaluation.

    Stored BN statistics require ``bn_source_ablation=true``; the main path
    estimates statistics on the synthesized batch.
    """
    requirements = (
        (params.get("enable_ldp") is True, "enable_ldp=true"),
        (
            str(params.get("defence_method", "")).strip().lower().replace("_", "-")
            in {"priv-filterfl", "ours-cryptgpu"},
            "defence_method=priv-filterfl",
        ),
        (
            params.get("allow_plaintext_detector_screen") is True,
            "allow_plaintext_detector_screen=true",
        ),
        (params.get("diagnostic_only") is True, "diagnostic_only=true"),
        (
            str(params.get("revision_validity", "")).strip().lower() == "debug",
            "revision_validity=debug",
        ),
        (params.get("revision_privacy_valid") is False, "revision_privacy_valid=false (boolean)"),
        (
            str(params.get("priv_filterfl_mpc_mode", "")).strip().lower().replace("-", "_")
            == "simulated",
            "priv_filterfl_mpc_mode=simulated",
        ),
        (params.get("record_detection_diagnostics") is True, "record_detection_diagnostics=true"),
        (params.get("cryptgpu_bn_use_noisy") is True, "cryptgpu_bn_use_noisy=true"),
        (
            params.get("bn_folding_use_running_stats") is False
            or params.get("bn_source_ablation") is True,
            "bn_folding_use_running_stats=false, unless the run declares bn_source_ablation=true (the DP-CBF ablation, which exists to fold the training-time statistics instead)",
        ),
        (params.get("ldp_detection_use_noisy") is False, "ldp_detection_use_noisy=false"),
        (
            plaintext_screen_dp_profile(params) is not None,
            "one of the two declared DP profiles: (dp_release_source=client_ldp, dp_cbf_stats_source=per_client_noisy_model) or (dp_release_source=mpc_aggregate_ref_dp_fast|mpc_aggregate_ref_dp_secure_clip_debug, dp_cbf_stats_source=aggregate_noisy_reference)",
        ),
        (params.get("cryptgpu_fail_closed") is True, "cryptgpu_fail_closed=true"),
    )
    missing = [description for satisfied, description in requirements if not satisfied]
    if missing:
        raise ValueError(
            "diagnostic_scores plaintext detector screen requires: " + ", ".join(missing)
        )


def _build_detection_model(model_spec, target_device):
    arch = str(model_spec.get("arch", "resnet8")).lower()
    num_classes = int(model_spec.get("num_classes", 10))
    in_channels = int(model_spec.get("in_channels", 3))
    if arch == "resnet8":
        from models.ResNet8 import ResNet8 as _PTModel

        pt_model = _PTModel(num_classes=num_classes)
    elif arch == "resnet18":
        from models.ResNet18 import ResNet18 as _PTModel

        pt_model = _PTModel(num_classes=num_classes, in_channels=in_channels)
    elif arch == "resnet50_cifar":
        from models.ResNet50Cifar import ResNet50Cifar as _PTModel

        pt_model = _PTModel(num_classes=num_classes, in_channels=in_channels)
    else:
        from models.CNNMnist import CNNMnist as _PTModel

        pt_model = _PTModel(num_classes=num_classes)
    return pt_model.to(target_device).float().eval()


def _strip_bn_layers_in_place(model):
    try:
        import torch.nn as _nn

        if hasattr(model, "bn1") and isinstance(model.bn1, _nn.BatchNorm2d):
            model.bn1 = _nn.Identity()
        for _, module in model.named_modules():
            if hasattr(module, "bn1") and isinstance(module.bn1, _nn.BatchNorm2d):
                module.bn1 = _nn.Identity()
            if hasattr(module, "bn2") and isinstance(module.bn2, _nn.BatchNorm2d):
                module.bn2 = _nn.Identity()
    except Exception:
        pass
    return model


def _prepare_plaintext_trigger_model(
    client_sd,
    trigger_tensor_cpu,
    model_spec,
    noisy_bn_sd=None,
    use_noisy_for_bn=True,
    use_running_stats=False,
    log_prefix="ours_plaintext_cryptgpu",
    target_device=None,
    client_idx=None,
    fail_closed=False,
    precomputed_bn_stats=None,
):
    from tools.fuse_bn import extract_bn_stats_from_model, fuse_model_arch

    if target_device is None:
        target_device = (
            torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        )
    arch = str(model_spec.get("arch", "resnet8")).lower()
    pt_model = _build_detection_model(model_spec, target_device)
    sd_fp32 = {
        k: v.detach().to(device=target_device, dtype=torch.float32)
        for k, v in client_sd.items()
        if hasattr(v, "detach")
    }
    has_bn_keys = any(
        ("bn" in k or "running_mean" in k or "running_var" in k for k in client_sd.keys())
    )
    trigger_dev = trigger_tensor_cpu.to(device=target_device, dtype=torch.float32)
    if has_bn_keys:
        if precomputed_bn_stats is not None:
            try:
                pt_model.load_state_dict(sd_fp32)
                pt_model = fuse_model_arch(pt_model, arch, bn_stats=precomputed_bn_stats)
                pt_model.to(target_device).float().eval()
            except Exception as exc:
                if fail_closed:
                    raise RuntimeError(
                        f"[{log_prefix}] client{client_idx} precomputed noisy BN fusion failed"
                    ) from exc
                raise
        else:
            should_use_noisy = use_noisy_for_bn and noisy_bn_sd is not None
            if should_use_noisy:
                try:
                    pt_model_noisy = _build_detection_model(model_spec, target_device)
                    sd_noisy_fp32 = {
                        k: v.detach().to(device=target_device, dtype=torch.float32)
                        for k, v in noisy_bn_sd.items()
                        if hasattr(v, "detach")
                    }
                    pt_model_noisy.load_state_dict(sd_noisy_fp32)
                    if use_running_stats:
                        bn_stats_from_noisy = _extract_running_bn_stats(pt_model_noisy)
                    else:
                        bn_stats_from_noisy = extract_bn_stats_from_model(
                            pt_model_noisy, trigger_dev
                        )
                    del pt_model_noisy
                    pt_model.load_state_dict(sd_fp32)
                    pt_model = fuse_model_arch(pt_model, arch, bn_stats=bn_stats_from_noisy)
                    pt_model.to(target_device).float().eval()
                    try:
                        logger.debug(
                            f"[{log_prefix}] client{client_idx} successfully fused BN using noisy weights"
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    if fail_closed:
                        raise RuntimeError(
                            f"[{log_prefix}] client{client_idx} noisy/reference BN fusion failed"
                        ) from exc
                    try:
                        logger.warning(
                            f"[{log_prefix}] client{client_idx} BN fusion with noisy weights failed, fallback to clean weights: {exc}"
                        )
                    except Exception:
                        pass
                    pt_model.load_state_dict(sd_fp32)
                    if use_running_stats:
                        pt_model = fuse_model_arch(pt_model, arch)
                    else:
                        pt_model = fuse_model_arch(pt_model, arch, x=trigger_dev)
                    pt_model.to(target_device).float().eval()
            else:
                pt_model.load_state_dict(sd_fp32)
                try:
                    if not use_noisy_for_bn:
                        logger.debug(
                            f"[{log_prefix}] client{client_idx} configured to use clean weights to get BN params"
                        )
                    elif noisy_bn_sd is None:
                        logger.debug(
                            f"[{log_prefix}] client{client_idx} no noisy weights available; use clean weights to get BN params"
                        )
                except Exception:
                    pass
                if use_running_stats:
                    pt_model = fuse_model_arch(pt_model, arch)
                else:
                    pt_model = fuse_model_arch(pt_model, arch, x=trigger_dev)
                pt_model.to(target_device).float().eval()
    else:
        pt_model.load_state_dict(sd_fp32, strict=False)
        pt_model = _strip_bn_layers_in_place(pt_model)
    if arch in ["resnet8", "resnet18", "resnet50_cifar"]:
        try:
            import torch.nn.functional as _F
            import torch.nn as _nn

            with torch.no_grad():
                dummy = torch.zeros_like(trigger_dev, device=target_device, dtype=torch.float32)
                z = pt_model.conv1(dummy)
                if hasattr(pt_model, "bn1") and isinstance(pt_model.bn1, torch.nn.BatchNorm2d):
                    z = _F.relu(pt_model.bn1(z))
                else:
                    z = _F.relu(z)
                z = pt_model.layer1(z)
                z = pt_model.layer2(z)
                z = pt_model.layer3(z)
                if hasattr(pt_model, "layer4"):
                    z = pt_model.layer4(z)
                h, w = (int(z.shape[-2]), int(z.shape[-1]))
            pt_model.avgpool = _nn.AvgPool2d(kernel_size=(h, w))
            pt_model.to(target_device).float().eval()
        except Exception:
            try:
                import torch.nn as _nn

                kernel = (8, 8) if arch in ["resnet8", "resnet50_cifar"] else (1, 1)
                pt_model.avgpool = _nn.AvgPool2d(kernel_size=kernel)
                pt_model.to(target_device).float().eval()
            except Exception:
                pass
    elif arch == "cnnmnist":
        try:
            import torch.nn as _nn

            if hasattr(pt_model, "pool") and isinstance(pt_model.pool, _nn.MaxPool2d):
                pt_model.pool = _nn.AvgPool2d(
                    kernel_size=pt_model.pool.kernel_size,
                    stride=pt_model.pool.stride,
                    padding=pt_model.pool.padding,
                )
                pt_model.to(target_device).float().eval()
        except Exception:
            pass
    return pt_model


def _plaintext_logits_all_fn(
    client_weights_list,
    trigger_tensor_cpu,
    client_weights_noisy_for_bn=None,
    use_noisy_for_bn=True,
    model_spec=None,
    use_running_stats=False,
    log_prefix="ours_plaintext_cryptgpu",
    fail_closed=False,
):
    import numpy as np

    if model_spec is None:
        model_spec = {"arch": "resnet8", "num_classes": 10, "in_channels": 3}
    num_classes = int(model_spec.get("num_classes", 10))
    local_device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    rng_devices = []
    if local_device.type == "cuda":
        device_index = local_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        rng_devices.append(int(device_index))
    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        full_input = trigger_tensor_cpu.to(device=local_device, dtype=torch.float32)
        results = []
        for idx, client_sd in enumerate(client_weights_list):
            try:
                noisy_sd = None
                if client_weights_noisy_for_bn is not None and idx < len(
                    client_weights_noisy_for_bn
                ):
                    noisy_sd = client_weights_noisy_for_bn[idx]
                pt_model = _prepare_plaintext_trigger_model(
                    client_sd=client_sd,
                    trigger_tensor_cpu=trigger_tensor_cpu,
                    model_spec=model_spec,
                    noisy_bn_sd=noisy_sd,
                    use_noisy_for_bn=use_noisy_for_bn,
                    use_running_stats=use_running_stats,
                    log_prefix=log_prefix,
                    target_device=local_device,
                    client_idx=idx,
                    fail_closed=fail_closed,
                )
                with torch.no_grad():
                    logits = pt_model(full_input)
                results.append(logits.detach().cpu().numpy())
                try:
                    del pt_model, logits
                    if local_device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception as exc:
                try:
                    logger.error(f"[{log_prefix}] client{idx} forward failed: {repr(exc)}")
                except Exception:
                    pass
                if fail_closed:
                    raise RuntimeError(
                        f"[{log_prefix}] client{idx} plaintext forward failed"
                    ) from exc
                results.append(
                    np.zeros((trigger_tensor_cpu.shape[0], num_classes), dtype=np.float32)
                )
        return results


def _detect_backdoor_flame(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    beta,
):
    logger.info("[flame] Use cosine KMeans to filter abnormal updates")
    if (
        enable_ldp
        and bool(handle.params.get("flame_detection_use_noisy", False))
        and w_locals_noisy
    ):
        weights_for_detection = w_locals_noisy
    elif enable_ldp and w_locals_clean:
        weights_for_detection = w_locals_clean
    else:
        weights_for_detection = w_locals
    param_names = [name for name, _ in handle.model.named_parameters()]
    ori_vec = _flatten_state_dict_params(ori_weight, param_names)
    updates = []
    active_indices = []
    for i in range(len(weights_for_detection)):
        if beta[i] == 0:
            continue
        client_vec = _flatten_state_dict_params(weights_for_detection[i], param_names)
        if client_vec.numel() == 0 or ori_vec.numel() == 0:
            updates.append(torch.zeros(1, dtype=torch.float32))
        else:
            updates.append(client_vec - ori_vec)
        active_indices.append(i)
    if len(updates) <= 1:
        logger.info("[flame] Not enough clients, skip clustering")
        return (beta, 0, 0, 0, 0)
    k = int(handle.params.get("flame_cluster_k", 2))
    labels = _kmeans_cosine(updates, k=k, max_iter=int(handle.params.get("flame_kmeans_iter", 10)))
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    benign_label = max(counts, key=counts.get)
    logger.info(f"[flame] cluster_sizes={counts}, benign_label={benign_label}")
    tp = tn = fp = fn = 0
    for idx, client_idx in enumerate(active_indices):
        pred_bd = labels[idx] != benign_label
        gt_backdoor = agent_name_keys[client_idx] in adversarial_name_keys
        if pred_bd:
            beta[client_idx] = 0
        logger.info(
            f"[flame] client_index={client_idx}, ID={agent_name_keys[client_idx]}, GT={('malicious' if gt_backdoor else 'benign')}, pred_backdoor={pred_bd}"
        )
        if pred_bd and gt_backdoor:
            tp += 1
        elif pred_bd and (not gt_backdoor):
            fp += 1
        elif not pred_bd and gt_backdoor:
            fn += 1
        else:
            tn += 1
    return (beta, tp, fp, tn, fn)


def _detect_backdoor_shieldfl(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    beta,
    epoch,
):
    """Apply ShieldFL's rejection rule and publish its continuous aggregation weights.

    The returned confusion matrix uses the reporting threshold on those weights.
    See ``baseline_shieldfl.py`` for the method definition.
    """
    logger.info("[shieldfl] Use cosine distance to an elected poison-baseline gradient")
    if (
        enable_ldp
        and bool(handle.params.get("shieldfl_detection_use_noisy", False))
        and w_locals_noisy
    ):
        weights_for_detection = w_locals_noisy
    elif enable_ldp and w_locals_clean:
        weights_for_detection = w_locals_clean
    else:
        weights_for_detection = w_locals
    param_names = [name for name, _ in handle.model.named_parameters()]
    ori_vec = _flatten_state_dict_params(ori_weight, param_names)
    gradients = []
    active_indices = []
    for i in range(len(weights_for_detection)):
        if beta[i] == 0:
            continue
        client_vec = _flatten_state_dict_params(weights_for_detection[i], param_names)
        if client_vec.numel() == 0 or ori_vec.numel() == 0:
            continue
        gradients.append(ori_vec - client_vec)
        active_indices.append(i)
    handle.params.pop("_shieldfl_runtime", None)
    if not gradients:
        logger.info("[shieldfl] No active client; nothing to weight")
        return (beta, 0, 0, 0, 0)
    prev_grad_by_client = getattr(handle, "shieldfl_prev_gradient", None)
    if prev_grad_by_client is None:
        prev_grad_by_client = {}
        handle.shieldfl_prev_gradient = prev_grad_by_client
    previous_gradients = [prev_grad_by_client.get(agent_name_keys[i]) for i in active_indices]
    prev_global = getattr(handle, "shieldfl_prev_global_update", None)
    kappa = float(handle.params.get("shieldfl_kappa", baseline_shieldfl.DEFAULT_KAPPA))
    reject_frac = float(
        handle.params.get(
            "shieldfl_reject_threshold_frac", baseline_shieldfl.DEFAULT_REJECT_THRESHOLD_FRAC
        )
    )
    aggregate_normalized = bool(handle.params.get("shieldfl_aggregate_normalized", True))
    normalization = (
        str(handle.params.get("shieldfl_normalization", baseline_shieldfl.DEFAULT_NORMALIZATION))
        .strip()
        .lower()
    )
    result = baseline_shieldfl.shieldfl_round(
        gradients,
        prev_global,
        previous_gradients,
        kappa=kappa,
        is_first_round=prev_global is None,
        reject_threshold_frac=reject_frac,
        normalization=normalization,
    )
    for slot, i in enumerate(active_indices):
        prev_grad_by_client[agent_name_keys[i]] = gradients[slot].clone()
    if result.aggregate is not None:
        handle.shieldfl_prev_global_update = result.aggregate.clone()
    tp = tn = fp = fn = 0
    secjudge_rejected = []
    derived_rejected = []
    for slot, client_idx in enumerate(active_indices):
        gt_backdoor = agent_name_keys[client_idx] in adversarial_name_keys
        if not result.checked[slot]:
            beta[client_idx] = 0
            secjudge_rejected.append(str(agent_name_keys[client_idx]))
        pred_bd = bool(result.rejected[slot])
        if pred_bd:
            derived_rejected.append(str(agent_name_keys[client_idx]))
        logger.info(
            f"[shieldfl] client_index={client_idx}, ID={agent_name_keys[client_idx]}, GT={('malicious' if gt_backdoor else 'benign')}, checked={result.checked[slot]}, derived_pred_backdoor={pred_bd}, conf={result.confidence[slot]:.6f}, cos_poison={result.cos_to_poison_baseline[slot]}, selected_norm={result.selected_norms[slot]:.6g}"
        )
        if pred_bd and gt_backdoor:
            tp += 1
        elif pred_bd and (not gt_backdoor):
            fp += 1
        elif not pred_bd and gt_backdoor:
            fn += 1
        else:
            tn += 1
    admitted = [i for slot, i in enumerate(active_indices) if result.checked[slot]]
    if result.aggregate is None:
        logger.warning(
            "[shieldfl] no submission passed secure judgment; there is no aggregate this round (the harness keeps the previous global model)"
        )
    else:
        handle.params["_shieldfl_runtime"] = {
            "round": int(epoch),
            "param_names": list(param_names),
            "direction": result.aggregate.detach().to("cpu", torch.float32),
            "scale": float(result.scale),
            "weights": {
                int(i): float(result.confidence[slot])
                for slot, i in enumerate(active_indices)
                if result.checked[slot]
            },
            "active_indices": admitted,
            "aggregate_normalized": aggregate_normalized,
        }
    handle.shieldfl_confidence = {
        i: result.confidence[slot] for slot, i in enumerate(active_indices)
    }
    handle.shieldfl_poison_baseline_client = (
        agent_name_keys[active_indices[result.poison_baseline_index]]
        if result.poison_baseline_index is not None
        else None
    )
    record = {
        "round": int(epoch),
        "rule": "shieldfl-tifs22-alg3",
        "aggregate_object": "normalized_gradients"
        if aggregate_normalized
        else "raw_deltas_deviation",
        "kappa": kappa,
        "normalization": normalization,
        "n_active": len(active_indices),
        "n_admitted": len(admitted),
        "secjudge_rejected_client_ids": secjudge_rejected,
        "election_performed": bool(result.election_performed),
        "poison_baseline_client": None
        if handle.shieldfl_poison_baseline_client is None
        else str(handle.shieldfl_poison_baseline_client),
        "scale": None if result.scale is None else float(result.scale),
        "weight_sum_raw": float(result.weight_sum),
        "degenerate_zero_weights": bool(result.aggregate is not None and result.weight_sum == 0.0),
        "aggregate_norm": None
        if result.aggregate is None
        else float(torch.norm(result.aggregate).item()),
        "confidence_by_client": {
            str(agent_name_keys[i]): round(float(result.confidence[slot]), 6)
            for slot, i in enumerate(active_indices)
        },
        "derived_reject_threshold_frac": reject_frac,
        "derived_rejected_client_ids": derived_rejected,
    }
    records = getattr(handle, "shieldfl_round_records", None)
    if records is None:
        records = []
        handle.shieldfl_round_records = records
    records.append(record)
    logger.info(
        f"[shieldfl] poison_baseline_client={handle.shieldfl_poison_baseline_client}, checked={len(admitted)}/{len(active_indices)}, election={result.election_performed}, scale={result.scale}, weight_sum={result.weight_sum:.6g}, derived_rejected={len(derived_rejected)}"
    )
    return (beta, tp, fp, tn, fn)


def _apply_input_validation_rofl_elsa(
    handle, method, w_locals, ori_weight, beta, agent_name_keys, adversarial_name_keys, epoch
):
    """Validate submitted models with RoFL or ELSA and return filtering statistics.

    Aggregation consumes these validated objects; their identities are stored in
    ``_input_validation_runtime`` for the final binding check.
    """
    method = str(method).strip().lower()
    spec = baseline_input_validation.spec_from_params(handle.params, method)
    handle.params.pop("_input_validation_runtime", None)
    active = [i for i in range(len(w_locals)) if i < len(beta) and beta[i] != 0]
    if not active:
        logger.info(f"[{method}] no active submission this round")
        return (w_locals, beta, 0, 0, 0, 0)
    global_cpu = {
        k: v.detach().to("cpu") if isinstance(v, torch.Tensor) else v for k, v in ori_weight.items()
    }
    honest_mask = [agent_name_keys[i] not in adversarial_name_keys for i in active]
    result = baseline_input_validation.validate_submissions(
        [w_locals[i] for i in active], global_cpu, spec, honest_mask=honest_mask
    )
    new_w = list(w_locals)
    tp = fp = tn = fn = 0
    accepted_positions = []
    invalid_positions = []
    for slot, i in enumerate(active):
        gt_backdoor = agent_name_keys[i] in adversarial_name_keys
        if result.valid[slot]:
            new_w[i] = result.validated[slot]
            accepted_positions.append(i)
            if gt_backdoor:
                fn += 1
            else:
                tn += 1
        else:
            beta[i] = 0
            invalid_positions.append(i)
            if gt_backdoor:
                tp += 1
            else:
                fp += 1
        logger.info(
            f"[{method}] client_index={i}, ID={agent_name_keys[i]}, GT={('malicious' if gt_backdoor else 'benign')}, norm_submitted={result.norms_submitted[slot]:.6g}, client_rule_applied={result.client_rule_applied[slot]}, norm_validated={result.norms_after_client_rule[slot]:.6g}, bound={result.l2_bound:.6g}, valid={result.valid[slot]}, reason={result.invalid_reasons[slot] or 'none'}"
        )
    receipt = result.receipt(round_id=epoch, client_ids=[agent_name_keys[i] for i in active])
    receipt["accepted_indices"] = accepted_positions
    receipt["invalid_indices"] = invalid_positions
    receipt["client_clip_scope"] = spec.client_clip_scope
    records = getattr(handle, "input_validation_records", None)
    if records is None:
        records = []
        handle.input_validation_records = records
    records.append(receipt)
    handle.params["_input_validation_runtime"] = {
        "round": int(epoch),
        "method": method,
        "accepted_indices": list(accepted_positions),
        "object_ids": {int(i): id(new_w[i]) for i in accepted_positions},
        "l2_bound": float(result.l2_bound),
    }
    logger.info(
        f"[{method}] rule={result.rule} bound_rule={result.bound_rule} l2_bound={result.l2_bound:.6g} (median {result.median_norm:.6g} x {result.median_multiplier}) n_valid={len(accepted_positions)}/{len(active)} clipped_by_client_rule={sum(result.client_rule_applied)} validation_backend={baseline_input_validation.VALIDATION_BACKEND}"
    )
    return (new_w, beta, tp, fp, tn, fn)


def _crowdguard_validation_batches(handle, agent_id, max_samples):
    """Select a deterministic prefix of a client's training data."""
    loader = None
    try:
        loader = handle.train_data[agent_id][1]
    except Exception:
        loader = None
    if loader is None:
        return []
    batches = []
    taken = 0
    for data, target in loader:
        if taken >= max_samples:
            break
        room = max_samples - taken
        if int(target.shape[0]) > room:
            data = data[:room]
            target = target[:room]
        batches.append((data.detach().clone(), target.detach().clone()))
        taken += int(target.shape[0])
    return batches


def _detect_backdoor_crowdguard(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    beta,
):
    """Apply the CrowdGuard baseline defined in ``baseline_crowdguard.py``."""
    logger.info("[crowdguard] Clients validate each other's hidden-layer behaviour")
    if (
        enable_ldp
        and bool(handle.params.get("crowdguard_detection_use_noisy", False))
        and w_locals_noisy
    ):
        weights_for_detection = w_locals_noisy
    elif enable_ldp and w_locals_clean:
        weights_for_detection = w_locals_clean
    else:
        weights_for_detection = w_locals
    active_indices = [i for i in range(len(weights_for_detection)) if beta[i] != 0]
    if len(active_indices) <= 2:
        logger.info("[crowdguard] Not enough clients to vote, skip")
        return (beta, 0, 0, 0, 0)
    max_samples = int(
        handle.params.get(
            "crowdguard_max_validation_samples", baseline_crowdguard.DEFAULT_MAX_VALIDATION_SAMPLES
        )
    )
    spatial_average = handle.params.get("crowdguard_spatial_average", True)
    if type(spatial_average) is not bool:
        raise ValueError("crowdguard_spatial_average must be a boolean")
    global_model = copy.deepcopy(handle.model).to("cpu")
    global_model.load_state_dict(ori_weight)
    models = []
    for i in active_indices:
        m = copy.deepcopy(handle.model).to("cpu")
        m.load_state_dict(weights_for_detection[i])
        models.append(m)
    device_for_validation = handle.params.get("crowdguard_device", None) or device
    if bool(handle.params.get("crowdguard_debug_candidate_asr", False)):
        try:
            for slot, i in enumerate(active_indices):
                gt = "malicious" if agent_name_keys[i] in adversarial_name_keys else "benign"
                probe = models[slot].to(device)
                acc = test.normal_test(probe, None, handle.params, handle, poison=False)
                bd = test.normal_test(probe, None, handle.params, handle, poison=True)
                models[slot].to("cpu")
                logger.info(
                    f"[crowdguard][debug] slot={slot} ID={agent_name_keys[i]} GT={gt} candidate_model_ASR={bd:.2f} CA={acc:.2f}"
                )
        except Exception as exc:
            logger.warning(f"[crowdguard][debug] candidate ASR probe failed: {exc}")
    votes = []
    voters = []
    for own_slot, i in enumerate(active_indices):
        batches = _crowdguard_validation_batches(handle, agent_name_keys[i], max_samples)
        if not batches:
            logger.info(
                f"[crowdguard] client_index={i}, ID={agent_name_keys[i]} has no local validation data; it abstains"
            )
            continue
        detected = baseline_crowdguard.validate_models(
            global_model,
            models,
            own_slot,
            batches,
            device_for_validation,
            spatial_average=spatial_average,
        )
        row = [
            baseline_crowdguard.VOTE_FOR_POISONED
            if slot in detected
            else baseline_crowdguard.VOTE_FOR_BENIGN
            for slot in range(len(models))
        ]
        votes.append(row)
        voters.append(i)
        logger.info(
            f"[crowdguard] voter_index={i}, ID={agent_name_keys[i]}, flags_slots={detected}"
        )
    if not votes:
        logger.info("[crowdguard] No client produced a vote; accept all")
        return (beta, 0, 0, 0, 0)
    benign_slots = set(baseline_crowdguard.fuse_votes(votes))
    logger.info(
        f"[crowdguard] fused verdict: benign_slots={sorted(benign_slots)} of {len(models)} (voters={len(votes)})"
    )
    tp = tn = fp = fn = 0
    for slot, client_idx in enumerate(active_indices):
        pred_bd = slot not in benign_slots
        gt_backdoor = agent_name_keys[client_idx] in adversarial_name_keys
        if pred_bd:
            beta[client_idx] = 0
        logger.info(
            f"[crowdguard] client_index={client_idx}, ID={agent_name_keys[client_idx]}, GT={('malicious' if gt_backdoor else 'benign')}, pred_backdoor={pred_bd}"
        )
        if pred_bd and gt_backdoor:
            tp += 1
        elif pred_bd and (not gt_backdoor):
            fp += 1
        elif not pred_bd and gt_backdoor:
            fn += 1
        else:
            tn += 1
    return (beta, tp, fp, tn, fn)


def _fltrust_root_loader(handle):
    """Select the clean root dataset used by FLTrust with a fixed sampling seed."""
    cached = getattr(handle, "fltrust_root_loader", None)
    if cached is not None:
        return cached
    size = int(handle.params.get("fltrust_root_size", baseline_fltrust.DEFAULT_ROOT_SIZE))
    seed = int(handle.params.get("fltrust_root_seed", handle.params.get("seed", 0)))
    dataset = handle.train_dataset
    total = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:size].tolist()
    batch_size = int(handle.params.get("batch_size", 64))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=min(batch_size, max(1, size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    handle.fltrust_root_loader = loader
    handle.fltrust_root_indices = indices
    logger.info(
        f"[fltrust] Root dataset: {len(indices)} clean samples (seed={seed}); this is the coordinator-side assumption FLTrust needs"
    )
    return loader


def _fltrust_server_update(handle, ori_weight, param_names):
    """Train the shared model on the root dataset and return its update."""
    loader = _fltrust_root_loader(handle)
    model = copy.deepcopy(handle.model)
    model.load_state_dict(ori_weight)
    model = model.to(device)
    model.train()
    lr = float(handle.params["lr"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(handle.params.get("momentum", 0.9)),
        weight_decay=float(handle.params.get("decay", 0.0)),
    )
    loss_func = torch.nn.CrossEntropyLoss()
    local_epochs = int(handle.params.get("local_epochs", 1))
    for _ in range(local_epochs):
        for data, target in loader:
            data, target = (data.to(device), target.to(device))
            optimizer.zero_grad()
            loss = loss_func(model(data), target)
            loss.backward()
            optimizer.step()
    server_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    model.to("cpu")
    server_vec = _flatten_state_dict_params(server_state, param_names)
    ori_vec = _flatten_state_dict_params(ori_weight, param_names)
    return server_vec - ori_vec


def _detect_backdoor_fltrust(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    beta,
):
    """Apply the SAFEFL-derived FLTrust baseline in ``baseline_fltrust.py``."""
    logger.info("[fltrust] Trust-bootstrap against a coordinator-held root dataset")
    if (
        enable_ldp
        and bool(handle.params.get("fltrust_detection_use_noisy", False))
        and w_locals_noisy
    ):
        weights_for_detection = w_locals_noisy
    elif enable_ldp and w_locals_clean:
        weights_for_detection = w_locals_clean
    else:
        weights_for_detection = w_locals
    param_names = [name for name, _ in handle.model.named_parameters()]
    ori_vec = _flatten_state_dict_params(ori_weight, param_names)
    updates = []
    active_indices = []
    for i in range(len(weights_for_detection)):
        if beta[i] == 0:
            continue
        client_vec = _flatten_state_dict_params(weights_for_detection[i], param_names)
        if client_vec.numel() == 0 or ori_vec.numel() == 0:
            continue
        updates.append(client_vec - ori_vec)
        active_indices.append(i)
    if not updates:
        logger.info("[fltrust] No active clients, skip")
        return (beta, 0, 0, 0, 0)
    server_update = _fltrust_server_update(handle, ori_weight, param_names)
    result = baseline_fltrust.fltrust_round(updates, server_update)
    handle.fltrust_weights = {
        i: result.normalized_weights[slot] * result.scale_factors[slot]
        for slot, i in enumerate(active_indices)
    }
    handle.params["_fltrust_weight_runtime"] = dict(handle.fltrust_weights)
    tp = tn = fp = fn = 0
    for slot, client_idx in enumerate(active_indices):
        pred_bd = bool(result.rejected[slot])
        gt_backdoor = agent_name_keys[client_idx] in adversarial_name_keys
        if pred_bd:
            beta[client_idx] = 0
        logger.info(
            f"[fltrust] client_index={client_idx}, ID={agent_name_keys[client_idx]}, GT={('malicious' if gt_backdoor else 'benign')}, pred_backdoor={pred_bd}, trust={result.trust_scores[slot]:.6f}, weight={result.normalized_weights[slot]:.6f}"
        )
        if pred_bd and gt_backdoor:
            tp += 1
        elif pred_bd and (not gt_backdoor):
            fp += 1
        elif not pred_bd and gt_backdoor:
            fn += 1
        else:
            tn += 1
    return (beta, tp, fp, tn, fn)


def _detect_backdoor_fedredefense(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    g_img,
    beta,
):
    logger.info("[fedredefense] Use update reconstruction error to filter abnormal updates")
    if g_img is None:
        logger.info("[fedredefense] No synthetic samples provided, skip detection")
        return (beta, 0, 0, 0, 0)
    if (
        enable_ldp
        and bool(handle.params.get("fedre_detection_use_noisy", False))
        and w_locals_noisy
    ):
        weights_for_detection = w_locals_noisy
    elif enable_ldp and w_locals_clean:
        weights_for_detection = w_locals_clean
    else:
        weights_for_detection = w_locals
    param_names = [name for name, _ in handle.model.named_parameters()]
    ori_vec = _flatten_state_dict_params(ori_weight, param_names)
    model = copy.deepcopy(handle.model)
    try:
        model.load_state_dict(ori_weight)
    except Exception:
        model.load_state_dict(ori_weight, strict=False)
    model.to(device)
    model.train()
    with torch.no_grad():
        logits = model(g_img.to(device))
        pseudo = torch.argmax(logits, dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    grads = []
    max_batches = int(handle.params.get("fedre_grad_batches", 1))
    batch_size = int(handle.params.get("fedre_grad_batch_size", max(1, g_img.size(0))))
    num = g_img.size(0)
    for b in range(min(max_batches, max(1, math.ceil(num / batch_size)))):
        start = b * batch_size
        end = min(num, (b + 1) * batch_size)
        if start >= end:
            break
        model.zero_grad()
        inputs = g_img[start:end].to(device)
        labels = pseudo[start:end].to(device)
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        grad_vec = _flatten_model_grads(model, param_names)
        if grad_vec.numel() > 0:
            grads.append(grad_vec)
    if not grads or ori_vec.numel() == 0:
        logger.info("[fedredefense] Gradient vector is empty, skip detection")
        return (beta, 0, 0, 0, 0)
    if len(grads) == 1:
        G = grads[0].view(1, -1)
    else:
        G = torch.stack(grads, dim=0)
    errors = []
    active_indices = []
    eps = 1e-08
    ridge = float(handle.params.get("fedre_ridge", 0.001))
    for i in range(len(weights_for_detection)):
        if beta[i] == 0:
            continue
        client_vec = _flatten_state_dict_params(weights_for_detection[i], param_names)
        if client_vec.numel() == 0:
            errors.append(0.0)
            active_indices.append(i)
            continue
        delta = client_vec - ori_vec
        if G.shape[0] == 1:
            g = G[0]
            denom = float(torch.dot(g, g).item()) + eps
            lr_hat = float(torch.dot(delta, g).item()) / denom
            recon = lr_hat * g
        else:
            A = (G @ G.t()).float()
            A += ridge * torch.eye(A.size(0))
            b = (G @ delta).float()
            try:
                coeff = torch.linalg.solve(A, b)
            except Exception:
                coeff = torch.linalg.lstsq(A, b).solution
            recon = torch.sum(coeff.view(-1, 1) * G, dim=0)
        err = float(torch.norm(delta - recon).item() / (torch.norm(delta).item() + eps))
        errors.append(err)
        active_indices.append(i)
    tp = tn = fp = fn = 0
    topk = float(handle.params.get("fedre_topk", 0.0))
    if topk > 0:
        k = max(1, int(round(len(errors) * topk)))
        order = sorted(range(len(errors)), key=lambda idx: errors[idx], reverse=True)
        flagged = set(order[:k])
        thr = None
    else:
        _, _, thr = _robust_threshold(errors, z=float(handle.params.get("fedre_z", 2.5)))
        flagged = set([idx for idx, e in enumerate(errors) if e > thr])
    for local_idx, client_idx in enumerate(active_indices):
        pred_bd = local_idx in flagged
        gt_backdoor = agent_name_keys[client_idx] in adversarial_name_keys
        if pred_bd:
            beta[client_idx] = 0
        logger.info(
            f"[fedredefense] client_index={client_idx}, ID={agent_name_keys[client_idx]}, GT={('malicious' if gt_backdoor else 'benign')}, err={errors[local_idx]:.6f}, thr={thr}, pred_backdoor={pred_bd}"
        )
        if pred_bd and gt_backdoor:
            tp += 1
        elif pred_bd and (not gt_backdoor):
            fp += 1
        elif not pred_bd and gt_backdoor:
            fn += 1
        else:
            tn += 1
    return (beta, tp, fp, tn, fn)


def _lockdown_consensus_mask(masks, threshold):
    if not masks:
        return None
    consensus = {}
    for name in masks[0].keys():
        stacked = torch.stack([m[name].float() for m in masks], dim=0)
        consensus[name] = (stacked.mean(dim=0) >= threshold).float()
    return consensus


def _apply_lockdown_consensus(w_glob, ori_weight, consensus_mask):
    if consensus_mask is None:
        return w_glob
    for k, v in w_glob.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k not in consensus_mask:
            continue
        mask = consensus_mask[k].to(v.device)
        base = ori_weight[k].to(v.device) if isinstance(ori_weight.get(k), torch.Tensor) else v
        w_glob[k] = v * mask + base * (1.0 - mask)
    return w_glob


def _detect_backdoor_ours(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    g_img,
    beta,
    threshold,
):
    """Evaluate client responses on the synthesized probe."""
    tp = tn = fp = fn = 0

    def _infer_model_spec(params):
        t = str(params.get("type", "CIFAR10")).lower()
        arch_override = str(params.get("arch", "")).lower().strip()
        if t in ["mnist"]:
            return {"arch": "cnnmnist", "num_classes": 10, "in_channels": 1}
        if t in ["cifar10", "cifar-10"] and arch_override == "resnet50_cifar":
            return {"arch": "resnet50_cifar", "num_classes": 10, "in_channels": 3}
        if t in ["cifar10", "cifar-10"] and arch_override in ["resnet18", "resnet18_cifar"]:
            return {"arch": "resnet18", "num_classes": 10, "in_channels": 3}
        if t in ["cifar100", "cifar-100"]:
            return {"arch": "resnet18", "num_classes": 100, "in_channels": 3}
        if t in ["tinyimagenet", "tiny-imagenet"]:
            return {"arch": "resnet18", "num_classes": 200, "in_channels": 3}
        if t in ["gtsrb"]:
            return {"arch": "resnet8", "num_classes": 43, "in_channels": 3}
        return {"arch": "resnet8", "num_classes": 10, "in_channels": 3}

    model_spec = _infer_model_spec(handle.params)
    arch = model_spec["arch"]
    num_classes = model_spec["num_classes"]
    in_channels = model_spec["in_channels"]
    if isinstance(g_img, (list, tuple)):
        g_img_list = [batch for batch in g_img if batch is not None]
    else:
        g_img_list = [g_img] if g_img is not None else []
    if not g_img_list:
        logger.info("[filterfl] No trigger images provided")
        return (beta, tp, fp, tn, fn)
    trigger_batch = g_img_list[0]
    for i in range(handle.params["num_clients"]):
        if beta[i] == 0:
            continue
        if enable_ldp:
            use_noisy_detect = bool(handle.params.get("ldp_detection_use_noisy", False))
            if use_noisy_detect and w_locals_noisy:
                parameter = copy.deepcopy(w_locals_noisy[i])
            elif w_locals_clean:
                parameter = copy.deepcopy(w_locals_clean[i])
            else:
                parameter = copy.deepcopy(w_locals[i])
        else:
            parameter = copy.deepcopy(w_locals[i])
        param_device = None
        try:
            for _k, _v in parameter.items():
                if isinstance(_v, torch.Tensor):
                    param_device = _v.device
                    break
        except Exception:
            param_device = device
        if param_device is None:
            param_device = device
        ori_weight_dev = {}
        for k in ori_weight.keys():
            v = ori_weight[k]
            if isinstance(v, torch.Tensor):
                try:
                    ori_weight_dev[k] = v.detach().to(param_device)
                except Exception:
                    ori_weight_dev[k] = v
            else:
                ori_weight_dev[k] = v
        for k in parameter.keys():
            base = ori_weight_dev[k]
            parameter[k] = (parameter[k] - base) * beta[i] + base
        if arch == "resnet8":
            from models.ResNet8 import ResNet8

            new_model = ResNet8(num_classes=num_classes)
        elif arch == "resnet18":
            from models.ResNet18 import ResNet18

            new_model = ResNet18(num_classes=num_classes, in_channels=in_channels)
        elif arch == "cnnmnist":
            from models.CNNMnist import CNNMnist

            new_model = CNNMnist(num_classes=num_classes)
        else:
            from models.ResNet8 import ResNet8

            new_model = ResNet8(num_classes=num_classes)
        try:
            new_model.load_state_dict(parameter)
        except Exception:
            try:
                parameter_cpu = {
                    kk: vv.detach().to("cpu") if isinstance(vv, torch.Tensor) else vv
                    for kk, vv in parameter.items()
                }
                new_model.load_state_dict(parameter_cpu)
            except Exception:
                new_model.load_state_dict(parameter)
        new_model.to(param_device)
        new_model.train()
        with torch.no_grad():
            g_img_dev = trigger_batch.detach().to(param_device)
            probs_mean = torch.mean(torch.softmax(new_model(g_img_dev), dim=1), dim=0)
            top_probability, out_n = torch.max(probs_mean, dim=0)
            other_mask = torch.ones_like(probs_mean, dtype=torch.bool)
            other_mask[0] = False
            decided = {
                "p_target": float(probs_mean[0].item()),
                "top_probability": float(top_probability.item()),
                "top_idx": int(out_n.item()),
                "max_other": float(probs_mean[other_mask].max().item())
                if bool(other_mask.any())
                else 0.0,
            }
            logger.info(
                "{}, {}, {}".format(
                    agent_name_keys[i], decided["top_idx"], decided["top_probability"]
                )
            )
            if float(threshold) < decided["top_probability"] and decided["top_idx"] == 0:
                beta[i] = 0
        gt_backdoor = agent_name_keys[i] in adversarial_name_keys
        pred_backdoor = beta[i] == 0
        _record_detection_score(
            handle,
            "filterfl",
            agent_name_keys[i],
            gt_backdoor,
            decided["p_target"],
            decided["top_idx"],
            threshold,
            pred_backdoor,
            max_other=decided["max_other"],
        )
        if pred_backdoor and gt_backdoor:
            tp += 1
        elif pred_backdoor and (not gt_backdoor):
            fp += 1
        elif not pred_backdoor and gt_backdoor:
            fn += 1
        else:
            tn += 1
    return (beta, tp, fp, tn, fn)


def _detect_backdoor_ours_plaintext_cryptgpu(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    g_img,
    beta,
    threshold,
):
    import torch

    logger.info(
        "[ours_plaintext_cryptgpu] Use plaintext forward with CryptGPU-aligned preprocessing"
    )
    final_detector = canonical_final_detector(handle.params.get("final_detector", FIXED_TAU))
    detector_output_mode = canonical_detector_output_mode(
        handle.params.get("detector_output_mode", DEBUG_LOGITS)
    )
    robust_diagnostic_screen = (
        final_detector in {CALIBRATED_CENTERED_SCORE} and detector_output_mode == DIAGNOSTIC_SCORES
    )
    plaintext_detector_screen = robust_diagnostic_screen
    if final_detector in {CALIBRATED_CENTERED_SCORE}:
        if not robust_diagnostic_screen:
            raise ValueError(
                "plaintext robust detection is available only as the explicitly authorized diagnostic_scores screen"
            )
        _validate_plaintext_detector_screen_config(handle.params)
        if not enable_ldp:
            raise ValueError("diagnostic_scores plaintext detector screen requires enable_ldp=true")
    elif detector_output_mode != DEBUG_LOGITS:
        raise ValueError("fixed_tau plaintext detection requires debug_logits")
    detector_config = _robust_detector_runtime_config(
        handle.params, final_detector, detector_output_mode
    )
    active_indices, client_weights_active, client_weights_noisy_for_bn = (
        _prepare_active_client_weights_for_detection(
            handle, enable_ldp, w_locals, w_locals_clean, w_locals_noisy, ori_weight, beta
        )
    )
    if enable_ldp:
        if bool(handle.params.get("ldp_detection_use_noisy", False)):
            logger.info("[LDP][ours_plaintext_cryptgpu] Use noisy weights for backdoor detection")
        else:
            logger.info("[LDP][ours_plaintext_cryptgpu] Use clean weights for backdoor detection")
    logger.info(f"[ours_plaintext_cryptgpu] active_clients={len(active_indices)}")
    tp = tn = fp = fn = 0
    if len(client_weights_active) == 0:
        if plaintext_detector_screen:
            raise RuntimeError("[ours_plaintext_cryptgpu] detector screen has no active clients")
        logger.info("[ours_plaintext_cryptgpu] No active clients to detect")
        return (beta, tp, fp, tn, fn)
    model_spec = _infer_detection_model_spec(handle.params)
    if isinstance(g_img, (list, tuple)):
        g_img_list = [gi for gi in g_img if gi is not None]
    else:
        g_img_list = [g_img] if g_img is not None else []
    if len(g_img_list) == 0:
        if plaintext_detector_screen:
            raise RuntimeError("[ours_plaintext_cryptgpu] detector screen has no trigger")
        logger.info("[ours_plaintext_cryptgpu] No trigger images provided")
        return (beta, tp, fp, tn, fn)
    use_running_stats = bool(handle.params.get("bn_folding_use_running_stats", False))
    use_noisy_for_bn = bool(handle.params.get("cryptgpu_bn_use_noisy", True))
    plaintext_fail_closed = bool(handle.params.get("cryptgpu_fail_closed", True))

    def _run_trigger(g_img_single):
        g_img_cpu = g_img_single.detach().to("cpu").contiguous()
        threshold_float = float(threshold)
        logits_infos = _plaintext_logits_all_fn(
            client_weights_list=client_weights_active,
            trigger_tensor_cpu=g_img_cpu,
            client_weights_noisy_for_bn=client_weights_noisy_for_bn,
            use_noisy_for_bn=use_noisy_for_bn,
            model_spec=model_spec,
            use_running_stats=use_running_stats,
            log_prefix="ours_plaintext_cryptgpu",
            fail_closed=plaintext_fail_closed,
        )
        bsz = int(g_img_cpu.shape[0])
        num_classes = int(model_spec.get("num_classes", 10))
        if plaintext_fail_closed and (
            not isinstance(logits_infos, list) or len(logits_infos) != len(active_indices)
        ):
            raise RuntimeError("[ours_plaintext_cryptgpu] plaintext forward result count mismatch")
        logits_list = []
        for idx_local in range(len(active_indices)):
            try:
                info = logits_infos[idx_local]
                arr = info
                global_arr = None
                provenance = None
            except Exception:
                arr = None
                global_arr = None
                provenance = None
            if arr is None:
                if plaintext_fail_closed:
                    raise RuntimeError(
                        f"[ours_plaintext_cryptgpu] plaintext forward result is missing for active client index {active_indices[idx_local]}"
                    )
                logits_list.append(torch.zeros((bsz, num_classes), dtype=torch.float32))
                continue
            try:
                t = torch.as_tensor(arr, dtype=torch.float32)
                if t.ndim != 2:
                    raise RuntimeError("ndim!=2")
                if plaintext_fail_closed and tuple(t.shape) != (bsz, num_classes):
                    raise RuntimeError(f"shape={tuple(t.shape)} expected=({bsz}, {num_classes})")
                if plaintext_fail_closed and (not bool(torch.isfinite(t).all())):
                    raise RuntimeError("non-finite plaintext logits")
                if t.shape[0] != bsz:
                    if t.shape[0] > bsz:
                        t = t[:bsz]
                    else:
                        pad_rows = bsz - t.shape[0]
                        t = torch.cat([t, t[-1:].repeat(pad_rows, 1)], dim=0)
                if t.shape[1] != num_classes:
                    if t.shape[1] > num_classes:
                        t = t[:, :num_classes]
                    else:
                        pad_cols = num_classes - t.shape[1]
                        t = torch.cat([t, torch.zeros((bsz, pad_cols), dtype=t.dtype)], dim=1)
                logits_list.append(t)
            except Exception as exc:
                if plaintext_fail_closed:
                    raise RuntimeError(
                        f"[ours_plaintext_cryptgpu] invalid plaintext forward result for active client index {active_indices[idx_local]}"
                    ) from exc
                logits_list.append(torch.zeros((bsz, num_classes), dtype=torch.float32))
        client_metrics = []
        for local_idx, i in enumerate(active_indices):
            logits = logits_list[local_idx]
            probs = torch.softmax(logits, dim=1)
            probs_mean = probs.mean(dim=0)
            p_target = probs_mean[0].item() if probs_mean.numel() > 0 else 0.0
            top_idx = int(torch.argmax(probs_mean).item()) if probs_mean.numel() > 0 else -1
            other_mask = torch.ones_like(probs_mean, dtype=torch.bool)
            if probs_mean.numel() > 0:
                other_mask[0] = False
            max_other = float(probs_mean[other_mask].max().item()) if other_mask.any() else 0.0
            other_logits = logits[:, other_mask]
            per_sample_max_other_logits = other_logits.max(dim=1).values
            logit_margin = float((logits[:, 0] - per_sample_max_other_logits).mean().item())
            target_logit_mean = float(logits[:, 0].mean().item())
            max_other_logit_mean = float(per_sample_max_other_logits.mean().item())
            metric = {
                "active_index": i,
                "p_target": float(p_target),
                "top_idx": top_idx,
                "max_other": max_other,
                "logit_margin": logit_margin,
                "target_logit_mean": target_logit_mean,
                "max_other_logit_mean": max_other_logit_mean,
            }
            client_metrics.append(metric)
        if robust_diagnostic_screen:
            margins = [row["logit_margin"] for row in client_metrics]
            scores = margins
            score_type = "logit_margin"
            score_type = detector_config["score_type"]
            if score_type == "target_probability":
                scores = [row["p_target"] for row in client_metrics]
            pred_list = calibrated_centered_reject_bits(
                scores,
                threshold_q=detector_config["threshold_q"],
                center_weight=detector_config["center_weight"],
                center_mode=detector_config["center_mode"],
                min_clients=detector_config["min_clients"],
            )
            _record_plaintext_detector_screen_output(
                handle,
                [agent_name_keys[row["active_index"]] for row in client_metrics],
                scores,
                pred_list,
                final_detector,
                detector_output_mode,
                score_type=score_type,
                extra_diagnostics=None,
            )
        else:
            require_target_top = _fixed_tau_require_target_top(handle.params)
            pred_list = [
                bool(
                    row["p_target"] > threshold_float
                    and (row["top_idx"] == 0 if require_target_top else True)
                )
                for row in client_metrics
            ]
        for row, pred_bd in zip(client_metrics, pred_list):
            i = row["active_index"]
            _record_detection_score(
                handle,
                "ours_plaintext_cryptgpu",
                agent_name_keys[i],
                agent_name_keys[i] in adversarial_name_keys,
                row["p_target"],
                row["top_idx"],
                threshold_float,
                pred_bd,
                max_other=row["max_other"],
                logit_margin=row["logit_margin"],
                target_logit_mean=row["target_logit_mean"],
                max_other_logit_mean=row["max_other_logit_mean"],
                require_target_top=_fixed_tau_require_target_top(handle.params)
                if final_detector == FIXED_TAU
                else None,
            )
            logger.info(
                f"[ours_plaintext_cryptgpu] client_index={i}, ID={agent_name_keys[i]}, p_target={row['p_target']:.4f}, top={row['top_idx']}, logit_margin={row['logit_margin']:.6f}, detector={final_detector}, predict_backdoor={pred_bd}"
            )
        return pred_list

    predictions = _run_trigger(g_img_list[0])
    for local_idx, i in enumerate(active_indices):
        pred_bd = bool(predictions[local_idx])
        if pred_bd:
            if agent_name_keys[i] in adversarial_name_keys:
                tp += 1
            else:
                fp += 1
            beta[i] = 0
        elif agent_name_keys[i] in adversarial_name_keys:
            fn += 1
        else:
            tn += 1
    return (beta, tp, fp, tn, fn)


def _secure_mask_mode(params):
    return canonical_secure_mask_mode(params.get("secure_mask_mode"))


def _prepare_mask_share_dir(handle, epoch):
    """Allocate the round-local mask-share directory for secret-mask execution."""
    if _secure_mask_mode(handle.params) != MASK_SECRET_SHARED:
        return None
    root = getattr(handle, "mask_share_root", None)
    if root is None:
        root = tempfile.mkdtemp(prefix="fldd_accept_mask_")
        handle.mask_share_root = root
    path = os.path.join(root, f"round_{int(epoch)}")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    handle.current_mask_share_dir = path
    return path


def _mask_application(handle, world_size=3):
    """Return the mask mode and its party-share directory.

    CryptGPU uses persisted secret shares. Numerical utility runs apply the
    mask directly and return no share directory.
    """
    if _secure_mask_mode(handle.params) != MASK_SECRET_SHARED:
        return (False, None)
    return (True, getattr(handle, "current_mask_share_dir", None))


def _require_mask_shares(mask_share_dir, world_size=3):
    """Check that every MPC party has a mask-share file."""
    if not mask_share_dir:
        raise RuntimeError(
            "[secure_mask] secret_shared mode requires a mask-share directory; the detector did not produce one"
        )
    missing = [
        rank
        for rank in range(world_size)
        if not os.path.isfile(os.path.join(mask_share_dir, f"mask_share_{rank}.npy"))
    ]
    if missing:
        raise RuntimeError(
            f"[secure_mask] missing accept-mask shares for rank(s) {missing} in {mask_share_dir}; refusing to fall back to the opened mask"
        )
    return mask_share_dir


def _persist_accept_mask_shares(mask_share_dir, accept_binary):
    """Persist this party's XOR mask share across inference and aggregation sessions."""
    import os as _os
    import numpy as _np
    from crypten import comm as _comm

    rank = int(_comm.get().get_rank())
    share = accept_binary.share
    if hasattr(share, "tensor"):
        share = share.tensor()
    _os.makedirs(mask_share_dir, exist_ok=True)
    _np.save(
        _os.path.join(mask_share_dir, f"mask_share_{rank}.npy"),
        share.detach().reshape(-1).cpu().numpy().astype(_np.int64),
    )


def _cryptgpu_logits_all_fn(
    client_weights_list,
    trigger_tensor_cpu,
    client_weights_noisy_for_bn=None,
    use_noisy_for_bn=True,
    micro_batch_size=None,
    model_spec=None,
    use_running_stats=False,
    fail_closed=True,
    detector_config=None,
    mask_share_dir=None,
):
    import os
    import sys
    import torch
    import numpy as np

    _activate_cryptgpu_runtime()
    import crypten as crypten_gpu

    try:
        torch.set_default_dtype(torch.float32)
    except Exception:
        pass
    dev_index = 0
    try:
        dev_index = int(os.getenv("CRYPTGPU_DEVICE_INDEX", "0"))
    except Exception:
        dev_index = 0
    torch.cuda.set_device(dev_index)
    local_device = torch.device("cuda", dev_index)
    full_input = trigger_tensor_cpu.to(device=local_device, dtype=torch.float32)
    do_micro_batch = False
    try:
        if micro_batch_size is not None:
            micro_batch_size = int(micro_batch_size)
            do_micro_batch = micro_batch_size > 0 and micro_batch_size < int(full_input.shape[0])
    except Exception:
        do_micro_batch = False
    if not do_micro_batch:
        x_batches = [crypten_gpu.cryptensor(full_input, src=0)]
        mb_slices = [(0, int(full_input.shape[0]))]
    else:
        x_batches = []
        mb_slices = []
        B = int(full_input.shape[0])
        for s in range(0, B, micro_batch_size):
            e = min(s + micro_batch_size, B)
            x_mb = crypten_gpu.cryptensor(full_input[s:e], src=0)
            x_batches.append(x_mb)
            mb_slices.append((s, e))
    detector_config = dict(detector_config or {})
    detector = canonical_final_detector(detector_config.get("final_detector", FIXED_TAU))
    output_mode = canonical_detector_output_mode(
        detector_config.get("detector_output_mode", DEBUG_LOGITS)
    )
    bit_only_robust = detector in {CALIBRATED_CENTERED_SCORE} and output_mode == ACCEPT_BITS_ONLY
    bit_only_accept = bit_only_robust
    if output_mode == ACCEPT_BITS_ONLY and (not bit_only_accept):
        raise ValueError(
            "detector_output_mode=accept_bits_only currently requires a secure cohort-score detector or the strictly qualified client clean-BN pre-fusion route"
        )
    encrypted_scores = []
    results = []
    arch = "resnet8"
    num_classes = 10
    in_channels = 3
    try:
        if isinstance(model_spec, dict):
            arch = str(model_spec.get("arch", arch)).lower()
            num_classes = int(model_spec.get("num_classes", num_classes))
            in_channels = int(model_spec.get("in_channels", in_channels))
    except Exception:
        pass
    for idx in range(len(client_weights_list)):
        try:
            sd = client_weights_list[idx]
            has_bn_keys = any(
                ("bn" in k or "running_mean" in k or "running_var" in k for k in sd.keys())
            )
            if arch == "resnet8":
                from models.ResNet8 import ResNet8 as _PTModel

                pt_model = _PTModel(num_classes=num_classes).to(local_device)
            elif arch == "resnet18":
                from models.ResNet18 import ResNet18 as _PTModel

                pt_model = _PTModel(num_classes=num_classes, in_channels=in_channels).to(
                    local_device
                )
            elif arch == "resnet50_cifar":
                from models.ResNet50Cifar import ResNet50Cifar as _PTModel

                pt_model = _PTModel(num_classes=num_classes, in_channels=in_channels).to(
                    local_device
                )
            else:
                from models.CNNMnist import CNNMnist as _PTModel

                pt_model = _PTModel(num_classes=num_classes).to(local_device)
            pt_model.eval()
            sd_fp32 = {
                k: v.detach().to(device=local_device, dtype=torch.float32)
                for k, v in sd.items()
                if hasattr(v, "detach")
            }
            pt_model.float()
            if has_bn_keys:
                should_use_noisy = (
                    use_noisy_for_bn
                    and client_weights_noisy_for_bn is not None
                    and (idx < len(client_weights_noisy_for_bn))
                )
                if should_use_noisy:
                    try:
                        import logging as _lg

                        _lg.getLogger("logger").debug(
                            f"[ours_cryptgpu] client{idx} using noisy weights to get BN params"
                        )
                        from tools.fuse_bn import extract_bn_stats_from_model, fuse_model_arch

                        sd_noisy = client_weights_noisy_for_bn[idx]
                        if arch == "resnet8":
                            from models.ResNet8 import ResNet8 as _PTModelNoisy

                            pt_model_noisy = _PTModelNoisy(num_classes=num_classes).to(local_device)
                        elif arch == "resnet18":
                            from models.ResNet18 import ResNet18 as _PTModelNoisy

                            pt_model_noisy = _PTModelNoisy(
                                num_classes=num_classes, in_channels=in_channels
                            ).to(local_device)
                        elif arch == "resnet50_cifar":
                            from models.ResNet50Cifar import ResNet50Cifar as _PTModelNoisy

                            pt_model_noisy = _PTModelNoisy(
                                num_classes=num_classes, in_channels=in_channels
                            ).to(local_device)
                        else:
                            from models.CNNMnist import CNNMnist as _PTModelNoisy

                            pt_model_noisy = _PTModelNoisy(num_classes=num_classes).to(local_device)
                        pt_model_noisy.eval()
                        sd_noisy_fp32 = {
                            k: v.detach().to(device=local_device, dtype=torch.float32)
                            for k, v in sd_noisy.items()
                            if hasattr(v, "detach")
                        }
                        pt_model_noisy.float()
                        pt_model_noisy.load_state_dict(sd_noisy_fp32)
                        if use_running_stats:
                            bn_stats_from_noisy = _extract_running_bn_stats(pt_model_noisy)
                        else:
                            bn_stats_from_noisy = extract_bn_stats_from_model(
                                pt_model_noisy, trigger_tensor_cpu.to(local_device)
                            )
                        del pt_model_noisy
                        pt_model.load_state_dict(sd_fp32)
                        pt_model = fuse_model_arch(pt_model, arch, bn_stats=bn_stats_from_noisy)
                        pt_model.to(local_device).float().eval()
                        try:
                            import logging as _lg

                            _lg.getLogger("logger").debug(
                                f"[ours_cryptgpu] client{idx} successfully fused BN using noisy weights"
                            )
                        except Exception:
                            pass
                    except Exception as _fuse_batch_e:
                        try:
                            import logging as _lg, traceback as _tb

                            _lg.getLogger("logger").warning(
                                f"[ours_cryptgpu] client{idx} BN fusion with noisy weights failed, fallback to clean weights: {_fuse_batch_e}"
                            )
                        except Exception:
                            pass
                        if fail_closed:
                            raise RuntimeError(
                                f"[ours_cryptgpu] noisy-BN fusion failed for client offset {idx}"
                            ) from _fuse_batch_e
                        pt_model.load_state_dict(sd_fp32)
                        from tools.fuse_bn import fuse_model_arch

                        if use_running_stats:
                            pt_model = fuse_model_arch(pt_model, arch)
                        else:
                            pt_model = fuse_model_arch(
                                pt_model, arch, x=trigger_tensor_cpu.to(local_device)
                            )
                        pt_model.to(local_device).float().eval()
                else:
                    pt_model.load_state_dict(sd_fp32)
                    try:
                        import logging as _lg

                        if not use_noisy_for_bn:
                            _lg.getLogger("logger").debug(
                                f"[ours_cryptgpu] client{idx} configured to use clean weights to get BN params"
                            )
                        elif client_weights_noisy_for_bn is None:
                            _lg.getLogger("logger").debug(
                                f"[ours_cryptgpu] client{idx} no noisy weights available; use clean weights to get BN params"
                            )
                        else:
                            _lg.getLogger("logger").debug(
                                f"[ours_cryptgpu] client{idx} index out of range; use clean weights to get BN params"
                            )
                    except Exception:
                        pass
                    try:
                        from tools.fuse_bn import fuse_model_arch

                        if use_running_stats:
                            pt_model = fuse_model_arch(pt_model, arch)
                        else:
                            pt_model = fuse_model_arch(
                                pt_model, arch, x=trigger_tensor_cpu.to(local_device)
                            )
                        pt_model.to(local_device).float().eval()
                    except Exception as _fuse_batch_e:
                        try:
                            import logging as _lg, traceback as _tb

                            _lg.getLogger("logger").error(
                                f"[ours_cryptgpu] client{idx} batch-aware BN fusion failed (requires recomputing mean/var with trigger): {_fuse_batch_e}\n"
                                + _tb.format_exc()
                            )
                        except Exception:
                            pass
                        raise
            else:
                pt_model.load_state_dict(sd_fp32, strict=False)
                try:
                    import torch.nn as _nn

                    if hasattr(pt_model, "bn1") and isinstance(pt_model.bn1, _nn.BatchNorm2d):
                        pt_model.bn1 = _nn.Identity()
                    for name, module in pt_model.named_modules():
                        if hasattr(module, "bn1") and isinstance(module.bn1, _nn.BatchNorm2d):
                            module.bn1 = _nn.Identity()
                        if hasattr(module, "bn2") and isinstance(module.bn2, _nn.BatchNorm2d):
                            module.bn2 = _nn.Identity()
                except Exception:
                    pass
            if arch in ["resnet8", "resnet18", "resnet50_cifar"]:
                try:
                    import torch.nn.functional as _F

                    with torch.no_grad():
                        dummy = torch.zeros_like(
                            trigger_tensor_cpu, device=local_device, dtype=torch.float32
                        )
                        z = pt_model.conv1(dummy)
                        if hasattr(pt_model, "bn1") and isinstance(
                            pt_model.bn1, torch.nn.BatchNorm2d
                        ):
                            z = _F.relu(pt_model.bn1(z))
                        else:
                            z = _F.relu(z)
                        z = pt_model.layer1(z)
                        z = pt_model.layer2(z)
                        z = pt_model.layer3(z)
                        if hasattr(pt_model, "layer4"):
                            z = pt_model.layer4(z)
                        h, w = (int(z.shape[-2]), int(z.shape[-1]))
                    import torch.nn as _nn

                    pt_model.avgpool = _nn.AvgPool2d(kernel_size=(h, w))
                    pt_model.to(local_device).float().eval()
                except Exception:
                    try:
                        import torch.nn as _nn

                        kernel = (8, 8) if arch in ["resnet8", "resnet50_cifar"] else (1, 1)
                        pt_model.avgpool = _nn.AvgPool2d(kernel_size=kernel)
                        pt_model.to(local_device).float().eval()
                    except Exception:
                        pass
            elif arch == "cnnmnist":
                try:
                    import torch.nn as _nn

                    if hasattr(pt_model, "pool") and isinstance(pt_model.pool, _nn.MaxPool2d):
                        pt_model.pool = _nn.AvgPool2d(
                            kernel_size=pt_model.pool.kernel_size,
                            stride=pt_model.pool.stride,
                            padding=pt_model.pool.padding,
                        )
                        pt_model.to(local_device).float().eval()
                except Exception:
                    pass
            dummy = torch.zeros_like(trigger_tensor_cpu, device=local_device, dtype=torch.float32)
            cmodel = crypten_gpu.nn.from_pytorch(pt_model, dummy)
            try:
                cmodel.cuda(device=dev_index)
            except Exception:
                pass
            try:
                cmodel.encrypt()
            except Exception:
                cmodel.encrypt(src=0)
            import numpy as _np

            if bit_only_accept:
                score_sum = None
                sample_count = 0
                for x_mb, (slice_start, slice_end) in zip(x_batches, mb_slices):
                    logits_ct = cmodel(x_mb)
                    if (
                        detector in {CALIBRATED_CENTERED_SCORE}
                        and detector_config.get("score_type") == "target_probability"
                    ):
                        encrypted_probabilities = logits_ct.softmax(dim=1)
                        batch_mean_score = secure_mean_target_probability(
                            encrypted_probabilities, num_classes=num_classes
                        )
                    else:
                        batch_mean_score = secure_mean_target_logit_margin(
                            logits_ct, num_classes=num_classes
                        )
                    batch_size = int(slice_end - slice_start)
                    batch_score_sum = batch_mean_score * float(batch_size)
                    score_sum = (
                        batch_score_sum if score_sum is None else score_sum + batch_score_sum
                    )
                    sample_count += batch_size
                if score_sum is None or sample_count <= 0:
                    raise RuntimeError("robust detector received an empty trigger batch")
                encrypted_scores.append(score_sum / float(sample_count))
            elif len(x_batches) == 1:
                logits_ct = cmodel(x_batches[0])
                logits = logits_ct.get_plain_text()
                results.append(logits.detach().cpu().numpy())
            else:
                logits_all = []
                for _mb_idx, x_mb in enumerate(x_batches):
                    logits_ct = cmodel(x_mb)
                    logits = logits_ct.get_plain_text()
                    logits_all.append(logits.detach().cpu().numpy())
                results.append(_np.concatenate(logits_all, axis=0))
            try:
                from memory_utils import clear_cuda_memory

                del pt_model, cmodel, logits_ct, sd_fp32, dummy
                if not bit_only_accept:
                    del logits
                clear_cuda_memory("cryptgpu_per_client_done")
            except Exception:
                pass
        except Exception as e:
            try:
                import logging as _lg, traceback as _tb

                _lg.getLogger("logger").error(
                    f"[ours_cryptgpu] client{idx}forwardfailed: {repr(e)}\n" + _tb.format_exc()
                )
            except Exception:
                pass
            if fail_closed:
                raise RuntimeError(
                    f"[ours_cryptgpu] encrypted forward failed for client offset {idx}"
                ) from e
            if bit_only_accept:
                raise RuntimeError(
                    f"[ours_cryptgpu] bit-only detector failed for client offset {idx}"
                ) from e
            results.append(np.zeros((trigger_tensor_cpu.shape[0], 10), dtype=np.float32))
    if bit_only_accept:
        decision_vector_ct = None
        decision_values_ct = secure_calibrated_centered_decision_values(
            encrypted_scores,
            threshold_q=float(detector_config["threshold_q"]),
            center_weight=float(detector_config["center_weight"]),
            center_mode=detector_config["center_mode"],
            min_clients=int(detector_config.get("min_clients", 5)),
        )
        results = []
        from crypten.mpc.primitives.converters import get_msb as _get_msb

        if decision_vector_ct is None:
            if any((int(value.nelement()) != 1 for value in decision_values_ct)):
                shapes = [tuple(value.size()) for value in decision_values_ct]
                raise RuntimeError(
                    f"[ours_cryptgpu] secure detector decisions are not scalar; shapes={shapes}"
                )
            decision_vector_ct = crypten_gpu.stack(decision_values_ct, dim=0)
        if int(decision_vector_ct.nelement()) != len(encrypted_scores):
            raise RuntimeError(
                "[ours_cryptgpu] secure detector decision vector has an unexpected shape"
            )
        reject_binary = _get_msb((-decision_vector_ct)._tensor)
        accept_binary = reject_binary ^ 1
        if mask_share_dir:
            _persist_accept_mask_shares(mask_share_dir, accept_binary)
        opened = accept_binary.reveal()
        if hasattr(opened, "tensor"):
            opened = opened.tensor()
        results = [int(value) for value in opened.detach().reshape(-1).cpu().tolist()]
        if len(results) != len(encrypted_scores) or any((bit not in (0, 1) for bit in results)):
            raise RuntimeError(
                f"[ours_cryptgpu] secure detector opened an invalid accept-bit vector: {results!r}"
            )
    try:
        from memory_utils import clear_cuda_memory

        try:
            for _xb in x_batches:
                del _xb
        except Exception:
            pass
        clear_cuda_memory("cryptgpu_loop_end")
    except Exception:
        pass
    return results


def _cryptgpu_logits_all_worker(
    client_weights_list,
    trigger_tensor_cpu,
    client_weights_noisy_for_bn=None,
    use_noisy_for_bn=True,
    micro_batch_size=None,
    model_spec=None,
    use_running_stats=False,
    fail_closed=True,
    detector_config=None,
    mask_share_dir=None,
):
    """Run inference in one MPC rank.

    ``run_multiprocess`` manages communicator initialization and teardown.
    """
    import os
    import sys

    _activate_cryptgpu_runtime()
    import crypten as crypten_gpu

    os.environ["CRYPTEN_BACKEND"] = "gloo"
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            frac_env = os.getenv("CRYPTGPU_MEM_FRACTION", "")
            frac = float(frac_env) if frac_env else 0.33
            try:
                dev_index = _torch.cuda.current_device()
            except Exception:
                dev_index = 0
            _torch.cuda.set_per_process_memory_fraction(frac, device=dev_index)
    except Exception:
        pass
    try:
        results = _cryptgpu_logits_all_fn(
            client_weights_list,
            trigger_tensor_cpu,
            client_weights_noisy_for_bn,
            use_noisy_for_bn,
            micro_batch_size,
            model_spec,
            use_running_stats,
            fail_closed,
            detector_config,
            mask_share_dir,
        )
    finally:
        try:
            from memory_utils import clear_cuda_memory

            clear_cuda_memory("cryptgpu_worker_shutdown")
        except Exception:
            pass
    return results


def _detect_backdoor_ours_cryptgpu(
    handle,
    enable_ldp,
    w_locals,
    w_locals_clean,
    w_locals_noisy,
    ori_weight,
    agent_name_keys,
    adversarial_name_keys,
    g_img,
    beta,
    threshold,
):
    """Evaluate BN-folded client models with the three-party CryptGPU runtime.

    The output mode determines whether logits or acceptance bits are opened.
    """
    boundary = _cryptgpu_detector_output_boundary(handle.params)
    fail_closed = boundary["fail_closed"]
    final_detector = boundary["final_detector"]
    detector_output_mode = boundary["detector_output_mode"]
    robust_bit_only = boundary["robust_bit_only"]
    bit_only_accept = robust_bit_only
    cryptgpu_compare_debug = boundary["cryptgpu_compare_debug"]
    import time
    import torch
    import numpy as np

    logger.info("[ours_cryptgpu] Use CryptGPU to run secure forward on GPU")
    try:
        if bool(handle.params.get("log_memory_cryptgpu", True)):
            logger.info(memory_snapshot(tag="ours_cryptgpu/before_mpc"))
    except Exception:
        pass
    active_indices, client_weights_active, client_weights_noisy_for_bn = (
        _prepare_active_client_weights_for_detection(
            handle, enable_ldp, w_locals, w_locals_clean, w_locals_noisy, ori_weight, beta
        )
    )
    if enable_ldp:
        if bool(handle.params.get("ldp_detection_use_noisy", False)):
            logger.info("[LDP][ours_cryptgpu] Use noisy weights for backdoor detection")
        else:
            logger.info("[LDP][ours_cryptgpu] Use clean weights for backdoor detection")
    logger.info(f"[ours_cryptgpu] active_clients={len(active_indices)}")
    tp = tn = fp = fn = 0
    if len(client_weights_active) == 0:
        logger.info("[ours_cryptgpu] No active clients to detect")
        return (beta, tp, fp, tn, fn)

    def _infer_model_spec(params):
        t = str(params.get("type", "CIFAR10")).lower()
        arch_override = str(params.get("arch", "")).lower().strip()
        if t in ["mnist"]:
            return {"arch": "cnnmnist", "num_classes": 10, "in_channels": 1}
        if t in ["cifar10", "cifar-10"] and arch_override == "resnet50_cifar":
            return {"arch": "resnet50_cifar", "num_classes": 10, "in_channels": 3}
        if t in ["cifar10", "cifar-10"] and arch_override in ["resnet18", "resnet18_cifar"]:
            return {"arch": "resnet18", "num_classes": 10, "in_channels": 3}
        if t in ["cifar100", "cifar-100"]:
            return {"arch": "resnet18", "num_classes": 100, "in_channels": 3}
        if t in ["tinyimagenet", "tiny-imagenet"]:
            return {"arch": "resnet18", "num_classes": 200, "in_channels": 3}
        if t in ["gtsrb"]:
            return {"arch": "resnet8", "num_classes": 43, "in_channels": 3}
        return {"arch": "resnet8", "num_classes": 10, "in_channels": 3}

    model_spec = _infer_model_spec(handle.params)
    if isinstance(g_img, (list, tuple)):
        g_img_list = [gi for gi in g_img if gi is not None]
    else:
        g_img_list = [g_img] if g_img is not None else []
    if len(g_img_list) == 0:
        if bit_only_accept:
            raise RuntimeError(
                "[ours_cryptgpu] accept_bits_only detector has no trigger; fail closed instead of silently accepting active clients"
            )
        logger.info("[ours_cryptgpu] No trigger images provided")
        return (beta, tp, fp, tn, fn)
    if bool(handle.params.get("record_bn_diagnostics", False)) and (
        not bool(handle.params.get("bn_folding_use_running_stats", False))
    ):
        diagnostic_trigger_cpu = g_img_list[0].detach().to("cpu").contiguous()
        for local_idx, active_client_index in enumerate(active_indices):
            _record_bn_batch_stat_discrepancy(
                handle,
                active_client_index,
                agent_name_keys[active_client_index],
                client_weights_active[local_idx],
                client_weights_noisy_for_bn[local_idx],
                diagnostic_trigger_cpu,
                model_spec,
            )
    world_size_cfg = 3
    use_gpu_pref = True
    use_gpu_actual = bool(
        use_gpu_pref and torch.cuda.is_available() and (torch.cuda.device_count() > 0)
    )
    if not use_gpu_actual:
        raise RuntimeError("[ours_cryptgpu] No available GPU detected")
    try:
        import multiprocessing as _mp

        sm = _mp.get_start_method(allow_none=True)
        if sm != "spawn":
            _mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    try:
        import torch.multiprocessing as _torch_mp

        sharing_strategy = (
            str(handle.params.get("cryptgpu_sharing_strategy", "file_system")).strip().lower()
        )
        if sharing_strategy:
            _torch_mp.set_sharing_strategy(sharing_strategy)
        logger.info(
            f"[ours_cryptgpu] torch_multiprocessing_sharing_strategy={_torch_mp.get_sharing_strategy()}"
        )
    except Exception as _sharing_e:
        raise RuntimeError(
            f"[ours_cryptgpu] failed to configure tensor sharing strategy: {_sharing_e}"
        ) from _sharing_e
    detector_config = _robust_detector_runtime_config(
        handle.params, final_detector, detector_output_mode
    )
    if robust_bit_only:
        public_constants = f"threshold_q={detector_config['threshold_q']}, center_weight={detector_config['center_weight']}, center_mode={detector_config['center_mode']}, score_type={detector_config['score_type']}"
        logger.info(
            f"[ours_cryptgpu][robust_detector] secure configured score + {final_detector}; designated output is one accept bit per active client; "
            + public_constants
        )
    _record_cryptgpu_runtime(
        handle,
        "detector_start",
        fail_closed=fail_closed,
        final_detector=final_detector,
        detector_output_mode=detector_output_mode,
    )

    def _run_trigger(g_img_single):
        g_img_cpu = g_img_single.detach().to("cpu").contiguous()
        threshold_float = float(threshold)
        detector_config_for_trigger = dict(detector_config)
        logits_infos = None
        t_mpc_total_start = time.time()
        try:
            import os as _os_path_fix

            _activate_cryptgpu_runtime()
            import importlib
            import crypten as crypten_launcher

            if not hasattr(crypten_launcher, "mpc"):
                importlib.invalidate_caches()
                crypten_launcher = importlib.reload(crypten_launcher)
            from crypten import mpc as mpc_launcher

            prefused_client_weights = client_weights_active
            use_running_stats = bool(handle.params.get("bn_folding_use_running_stats", False))
            if bool(handle.params.get("allow_server_prefuse", False)):
                try:
                    from tools.fuse_bn import extract_bn_stats_from_model, fuse_model_arch

                    prefused_client_weights = []
                    for idx, sd in enumerate(client_weights_active):
                        import torch as _torch

                        arch = model_spec["arch"]
                        num_classes = int(model_spec["num_classes"])
                        in_channels = int(model_spec["in_channels"])
                        if arch == "resnet8":
                            from models.ResNet8 import ResNet8 as _PTModel

                            pt_model = _PTModel(num_classes=num_classes).to("cpu").eval().float()
                        elif arch == "resnet18":
                            from models.ResNet18 import ResNet18 as _PTModel

                            pt_model = (
                                _PTModel(num_classes=num_classes, in_channels=in_channels)
                                .to("cpu")
                                .eval()
                                .float()
                            )
                        else:
                            from models.CNNMnist import CNNMnist as _PTModel

                            pt_model = _PTModel(num_classes=num_classes).to("cpu").eval().float()
                        sd_cpu_fp32 = {
                            k: v.detach().to("cpu", dtype=_torch.float32)
                            for k, v in sd.items()
                            if hasattr(v, "detach")
                        }
                        pt_model.load_state_dict(sd_cpu_fp32)
                        use_noisy_for_bn = bool(handle.params.get("cryptgpu_bn_use_noisy", True))
                        has_bn_keys = any(
                            (
                                "bn" in k or "running_mean" in k or "running_var" in k
                                for k in sd.keys()
                            )
                        )
                        should_use_noisy = (
                            use_noisy_for_bn
                            and has_bn_keys
                            and (client_weights_noisy_for_bn is not None)
                            and (idx < len(client_weights_noisy_for_bn))
                        )
                        if should_use_noisy:
                            try:
                                logger.debug(
                                    f"[ours_cryptgpu] prefusion client {idx} using noisy weights to get BN params"
                                )
                                sd_noisy = client_weights_noisy_for_bn[idx]
                                if arch == "resnet8":
                                    from models.ResNet8 import ResNet8 as _PTModelNoisy

                                    pt_model_noisy = (
                                        _PTModelNoisy(num_classes=num_classes)
                                        .to("cpu")
                                        .eval()
                                        .float()
                                    )
                                elif arch == "resnet18":
                                    from models.ResNet18 import ResNet18 as _PTModelNoisy

                                    pt_model_noisy = (
                                        _PTModelNoisy(
                                            num_classes=num_classes, in_channels=in_channels
                                        )
                                        .to("cpu")
                                        .eval()
                                        .float()
                                    )
                                else:
                                    from models.CNNMnist import CNNMnist as _PTModelNoisy
                                pt_model_noisy = (
                                    _PTModelNoisy(num_classes=num_classes).to("cpu").eval().float()
                                )
                                sd_noisy_cpu_fp32 = {
                                    k: v.detach().to("cpu", dtype=_torch.float32)
                                    for k, v in sd_noisy.items()
                                    if hasattr(v, "detach")
                                }
                                pt_model_noisy.load_state_dict(sd_noisy_cpu_fp32)
                                if use_running_stats:
                                    bn_stats_from_noisy = _extract_running_bn_stats(pt_model_noisy)
                                else:
                                    bn_stats_from_noisy = extract_bn_stats_from_model(
                                        pt_model_noisy, g_img_cpu
                                    )
                                del pt_model_noisy
                                fused_model = fuse_model_arch(
                                    pt_model, arch, bn_stats=bn_stats_from_noisy
                                )
                                logger.debug(
                                    f"[ours_cryptgpu] prefusion client {idx} successfully fused BN using noisy weights"
                                )
                            except Exception as _prefuse_noisy_e:
                                logger.warning(
                                    f"[ours_cryptgpu] prefusion client {idx} use noisy weights failed; fallback to clean weights: {_prefuse_noisy_e}"
                                )
                                if use_running_stats:
                                    fused_model = fuse_model_arch(pt_model, arch)
                                else:
                                    fused_model = fuse_model_arch(pt_model, arch, x=g_img_cpu)
                        else:
                            if not use_noisy_for_bn:
                                logger.debug(
                                    f"[ours_cryptgpu] prefusion client {idx} configured to use clean weights to get BN params"
                                )
                            if use_running_stats:
                                fused_model = fuse_model_arch(pt_model, arch)
                            else:
                                fused_model = fuse_model_arch(pt_model, arch, x=g_img_cpu)
                        fused_model.eval().float()
                        fused_sd = {
                            k: v.detach().to("cpu", dtype=_torch.float32)
                            for k, v in fused_model.state_dict().items()
                        }
                        prefused_client_weights.append(fused_sd)
                    logger.info(
                        f"[ours_cryptgpu] prefusion complete: {len(prefused_client_weights)} clients"
                    )
                except Exception as _prefuse_e:
                    import traceback as _tb

                    logger.warning(
                        f"[ours_cryptgpu] prefusion failed; fallback to subprocess fusion: {repr(_prefuse_e)}\n"
                        + _tb.format_exc()
                    )
                    prefused_client_weights = client_weights_active
            use_noisy_for_bn = bool(handle.params.get("cryptgpu_bn_use_noisy", True))
            use_running_stats = bool(handle.params.get("bn_folding_use_running_stats", False))
            logger.info(
                f"[ours_cryptgpu] BN foldingconfig: cryptgpu_bn_use_noisy={use_noisy_for_bn}, bn_folding_use_running_stats={use_running_stats}"
            )
            chunk_size = int(handle.params.get("cryptgpu_chunk_size", 4))
            if chunk_size <= 0:
                chunk_size = 1
            if robust_bit_only:
                if len(prefused_client_weights) < detector_config_for_trigger["min_clients"]:
                    raise RuntimeError(
                        f"robust bit-only detector has too few active clients: {len(prefused_client_weights)} < {detector_config_for_trigger['min_clients']}"
                    )
                if chunk_size < len(prefused_client_weights):
                    raise ValueError(
                        f"accept_bits_only requires one MPC session for the whole cohort; cryptgpu_chunk_size={chunk_size}, active_clients={len(prefused_client_weights)}"
                    )
                chunk_size = len(prefused_client_weights)
            all_logits_infos = []
            round_mask_share_dir = _prepare_mask_share_dir(
                handle, int(getattr(handle, "current_epoch", 0))
            )
            if round_mask_share_dir and chunk_size < len(prefused_client_weights):
                raise ValueError(
                    f"secure_mask_mode=secret_shared requires one MPC session for the whole cohort; cryptgpu_chunk_size={chunk_size}, active_clients={len(prefused_client_weights)}"
                )
            micro_batch_size = int(handle.params.get("cryptgpu_micro_batch", 0))
            for _start in range(0, len(prefused_client_weights), chunk_size):
                _end = min(_start + chunk_size, len(prefused_client_weights))
                _record_cryptgpu_runtime(
                    handle, "chunk_before", chunk_start=int(_start), chunk_end=int(_end)
                )
                sub_weights = prefused_client_weights[_start:_end]
                sub_noisy_for_bn = (
                    client_weights_noisy_for_bn[_start:_end]
                    if client_weights_noisy_for_bn is not None
                    else None
                )
                try:
                    _sub_res = mpc_launcher.run_multiprocess(world_size=world_size_cfg)(
                        _cryptgpu_logits_all_worker
                    )(
                        sub_weights,
                        g_img_cpu,
                        sub_noisy_for_bn,
                        use_noisy_for_bn,
                        micro_batch_size,
                        model_spec,
                        use_running_stats,
                        fail_closed,
                        detector_config_for_trigger,
                        round_mask_share_dir,
                    )
                except Exception as _chunk_e:
                    logger.error(
                        f"[ours_cryptgpu] MPC sub-batch({_start}:{_end}) forwardfailed: {_chunk_e}"
                    )
                    if fail_closed:
                        raise RuntimeError(
                            f"[ours_cryptgpu] MPC sub-batch({_start}:{_end}) failed"
                        ) from _chunk_e
                    _sub_res = None
                if bit_only_accept:
                    if not isinstance(_sub_res, list) or len(_sub_res) != world_size_cfg:
                        raise RuntimeError(
                            f"secure detector did not return one accept-bit vector per MPC rank: {_sub_res!r}"
                        )
                    if not all((isinstance(rank_bits, list) for rank_bits in _sub_res)):
                        raise RuntimeError(
                            "secure detector rank result is not an accept-bit vector"
                        )
                    canonical_bits = _sub_res[0]
                    if any((rank_bits != canonical_bits for rank_bits in _sub_res[1:])):
                        raise RuntimeError("secure detector ranks disagree on public accept bits")
                    _sub_res = canonical_bits
                elif (
                    isinstance(_sub_res, list)
                    and len(_sub_res) > 0
                    and isinstance(_sub_res[0], list)
                ):
                    _sub_res = _sub_res[0]
                if _sub_res is None:
                    if fail_closed:
                        raise RuntimeError(
                            f"[ours_cryptgpu] MPC sub-batch({_start}:{_end}) returned no result"
                        )
                    import numpy as _np

                    bsz = int(g_img_cpu.shape[0])
                    num_classes = int(model_spec.get("num_classes", 10))
                    for _ in range(_end - _start):
                        all_logits_infos.append(_np.zeros((bsz, num_classes), dtype=_np.float32))
                else:
                    all_logits_infos.extend(_sub_res)
                try:
                    if bool(handle.params.get("log_memory_cryptgpu", True)):
                        logger.info(
                            memory_snapshot(tag=f"ours_cryptgpu/chunk({_start}:{_end})/after_mpc")
                        )
                except Exception:
                    pass
                try:
                    from memory_utils import clear_cuda_memory

                    clear_cuda_memory(tag=f"ours_cryptgpu/chunk({_start}:{_end})/cleanup")
                except Exception:
                    pass
                _record_cryptgpu_runtime(
                    handle, "chunk_after", chunk_start=int(_start), chunk_end=int(_end)
                )
            logits_infos = all_logits_infos
        except Exception as e:
            logger.error(f"[ours_cryptgpu] MPC(GPU) forwardfailed: {e}")
            if fail_closed:
                raise RuntimeError("[ours_cryptgpu] secure forward failed") from e
            logits_infos = None
        t_mpc_total = time.time() - t_mpc_total_start
        try:
            logger.info(f"[ours_cryptgpu] mpc_total_s={t_mpc_total:.3f}")
        except Exception:
            pass
        try:
            if bool(handle.params.get("log_memory_cryptgpu", True)):
                logger.info(memory_snapshot(tag="ours_cryptgpu/after_mpc"))
        except Exception:
            pass
        if bit_only_accept:
            if (
                isinstance(logits_infos, list)
                and logits_infos
                and isinstance(logits_infos[0], list)
            ):
                logits_infos = logits_infos[0]
            if not isinstance(logits_infos, list):
                raise RuntimeError("secure detector did not return an accept-bit list")
            if len(logits_infos) != len(active_indices):
                raise RuntimeError(
                    f"secure detector accept-bit count mismatch: expected {len(active_indices)}, got {len(logits_infos)}"
                )
            accept_bits = []
            for value in logits_infos:
                if isinstance(value, bool):
                    bit = int(value)
                elif isinstance(value, int) and value in (0, 1):
                    bit = value
                else:
                    raise RuntimeError(f"secure detector returned non-bit output {value!r}")
                accept_bits.append(bit)
            _record_detector_output(
                handle,
                [agent_name_keys[index] for index in active_indices],
                accept_bits,
                final_detector,
                detector_output_mode,
            )
            return [bit == 0 for bit in accept_bits]
        import torch as _torch

        bsz = int(g_img_cpu.shape[0])
        num_classes_spec = int(model_spec.get("num_classes", 10))
        num_classes = num_classes_spec
        if (
            isinstance(logits_infos, list)
            and len(logits_infos) > 0
            and isinstance(logits_infos[0], list)
        ):
            logits_infos = logits_infos[0]
        if (
            isinstance(logits_infos, list)
            and len(logits_infos) > 0
            and isinstance(logits_infos[0], list)
        ):
            logits_infos = logits_infos[0]
        if isinstance(logits_infos, list) and (not fail_closed):
            for _probe in logits_infos:
                try:
                    if _probe is None:
                        continue
                    _tprobe = _torch.as_tensor(_probe)
                    if _tprobe.ndim == 2:
                        if _tprobe.shape[0] == bsz and _tprobe.shape[1] > 1:
                            num_classes = int(_tprobe.shape[1])
                            break
                except Exception:
                    pass
        if logits_infos is None or not isinstance(logits_infos, list):
            if fail_closed:
                raise RuntimeError("[ours_cryptgpu] MPC result is missing or invalid")
            try:
                logger.warning(
                    f"[ours_cryptgpu] MPC result missing or invalid type; fallback to {len(active_indices)} clients' zero logits"
                )
            except Exception:
                pass
            logits_list = [
                _torch.zeros((bsz, num_classes), dtype=_torch.float32)
                for _ in range(len(active_indices))
            ]
        else:
            logits_list = []
            if fail_closed and len(logits_infos) != len(active_indices):
                raise RuntimeError(
                    f"[ours_cryptgpu] MPC result count mismatch: expected {len(active_indices)}, got {len(logits_infos)}"
                )
            for idx_local in range(len(active_indices)):
                try:
                    arr = logits_infos[idx_local]
                except Exception:
                    arr = None
                if arr is None:
                    if fail_closed:
                        raise RuntimeError(
                            f"[ours_cryptgpu] missing MPC result for active client index {active_indices[idx_local]}"
                        )
                    try:
                        logger.warning(
                            f"[ours_cryptgpu] client_index={active_indices[idx_local]}  MPC result missing; fallback to zero logits"
                        )
                    except Exception:
                        if fail_closed:
                            raise RuntimeError(
                                f"[ours_cryptgpu] invalid MPC result shape for active client index {active_indices[idx_local]}"
                            )
                        pass
                    logits_list.append(_torch.zeros((bsz, num_classes), dtype=_torch.float32))
                else:
                    try:
                        t = _torch.as_tensor(arr, dtype=_torch.float32)
                        if t.ndim != 2:
                            raise RuntimeError("ndim!=2")
                        if fail_closed and tuple(t.shape) != (bsz, num_classes):
                            raise RuntimeError(
                                f"shape={tuple(t.shape)} expected=({bsz}, {num_classes})"
                            )
                        if fail_closed and (not bool(_torch.isfinite(t).all())):
                            raise RuntimeError("non-finite MPC logits")
                        if t.shape[0] != bsz:
                            if t.shape[0] > bsz:
                                t = t[:bsz]
                            else:
                                pad_rows = bsz - t.shape[0]
                                t = _torch.cat([t, t[-1:].repeat(pad_rows, 1)], dim=0)
                        if t.shape[1] != num_classes:
                            if t.shape[1] > num_classes:
                                t = t[:, :num_classes]
                            else:
                                pad_cols = num_classes - t.shape[1]
                                t = _torch.cat(
                                    [t, _torch.zeros((bsz, pad_cols), dtype=t.dtype)], dim=1
                                )
                        logits_list.append(t)
                    except Exception as _shape_e:
                        if fail_closed:
                            raise RuntimeError(
                                f"[ours_cryptgpu] invalid MPC result shape for active client index {active_indices[idx_local]}"
                            ) from _shape_e
                        try:
                            logger.warning(
                                f"[ours_cryptgpu] client_index={active_indices[idx_local]} MPC result shape mismatch,expected≈({bsz}, {num_classes}), calibration failed; fallback to zero logits"
                            )
                        except Exception:
                            pass
                        logits_list.append(_torch.zeros((bsz, num_classes), dtype=_torch.float32))
            if len(logits_list) < len(active_indices):
                pad = len(active_indices) - len(logits_list)
                if fail_closed:
                    raise RuntimeError(
                        f"[ours_cryptgpu] MPC result count mismatch: expected {len(active_indices)}, got {len(logits_list)}"
                    )
                try:
                    logger.warning(
                        f"[ours_cryptgpu] MPC result count insufficient; pad {pad}  zero logits to match active clients"
                    )
                except Exception:
                    pass
                logits_list += [
                    _torch.zeros((bsz, num_classes), dtype=_torch.float32) for _ in range(pad)
                ]
            elif len(logits_list) > len(active_indices):
                if fail_closed:
                    raise RuntimeError(
                        f"[ours_cryptgpu] MPC result count mismatch: expected {len(active_indices)}, got {len(logits_list)}"
                    )
                try:
                    logger.warning(
                        f"[ours_cryptgpu] MPC result count exceeded; truncate to {len(active_indices)}  to match active clients"
                    )
                except Exception:
                    pass
                logits_list = logits_list[: len(active_indices)]
        pred_list = []
        for local_idx, i in enumerate(active_indices):
            logits = logits_list[local_idx]
            probs = _torch.softmax(logits, dim=1)
            probs_mean = probs.mean(dim=0)
            p_target = probs_mean[0].item() if probs_mean.numel() > 0 else 0.0
            top_idx = int(_torch.argmax(probs_mean).item()) if probs_mean.numel() > 0 else -1
            other_mask = _torch.ones_like(probs_mean, dtype=_torch.bool)
            if probs_mean.numel() > 0:
                other_mask[0] = False
            max_other = float(probs_mean[other_mask].max().item()) if other_mask.any() else 0.0
            other_logits = logits[:, other_mask]
            per_sample_max_other_logits = other_logits.max(dim=1).values
            logit_margin = float((logits[:, 0] - per_sample_max_other_logits).mean().item())
            target_logit_mean = float(logits[:, 0].mean().item())
            max_other_logit_mean = float(per_sample_max_other_logits.mean().item())
            pred_bd = bool(top_idx == 0 and p_target > threshold_float)
            pred_list.append(pred_bd)
            _record_detection_score(
                handle,
                "priv-filterfl",
                agent_name_keys[i],
                agent_name_keys[i] in adversarial_name_keys,
                p_target,
                top_idx,
                threshold_float,
                pred_bd,
                max_other=max_other,
                logit_margin=logit_margin,
                target_logit_mean=target_logit_mean,
                max_other_logit_mean=max_other_logit_mean,
            )
            if cryptgpu_compare_debug:
                try:
                    sd_pt = client_weights_active[local_idx]
                    arch = model_spec["arch"]
                    num_classes = int(model_spec["num_classes"])
                    in_channels = int(model_spec["in_channels"])
                    if arch == "resnet8":
                        from models.ResNet8 import ResNet8 as _PTModel

                        _pt_model = _PTModel(num_classes=num_classes)
                    elif arch == "resnet18":
                        from models.ResNet18 import ResNet18 as _PTModel

                        _pt_model = _PTModel(num_classes=num_classes, in_channels=in_channels)
                    else:
                        from models.CNNMnist import CNNMnist as _PTModel

                        _pt_model = _PTModel(num_classes=num_classes)
                    _pt_model.eval()
                    _pt_model.to("cpu")
                    _pt_model.load_state_dict(sd_pt)
                    with _torch.no_grad():
                        pt_logits = _pt_model(g_img_cpu)
                    pt_probs = _torch.softmax(pt_logits, dim=1)
                    pt_probs_mean = pt_probs.mean(dim=0)
                    pt_p0 = pt_probs_mean[0].item() if pt_probs_mean.numel() > 0 else 0.0
                    pt_top = (
                        int(_torch.argmax(pt_probs_mean).item())
                        if pt_probs_mean.numel() > 0
                        else -1
                    )
                    logits_mean_mpc = logits.mean(dim=0)
                    logits_mean_pt = pt_logits.mean(dim=0)
                    diff_vec = logits_mean_mpc - logits_mean_pt
                    l2 = float(diff_vec.norm().item())
                    max_abs = float(_torch.max(_torch.abs(diff_vec)).item())
                    logger.info(
                        f"[ours_cryptgpu_compare] client_index={i}, ID={agent_name_keys[i]}, mpc_p0={p_target:.4f}, mpc_top={top_idx}, pt_p0={pt_p0:.4f}, pt_top={pt_top}, logits_l2={l2:.6f}, logits_maxabs={max_abs:.6f}"
                    )
                except Exception as _cmp_e:
                    logger.warning(f"[ours_cryptgpu_compare] Plaintext comparison failed: {_cmp_e}")
        return pred_list

    predictions = _run_trigger(g_img_list[0])
    for local_idx, i in enumerate(active_indices):
        pred_bd = bool(predictions[local_idx])
        if pred_bd:
            if agent_name_keys[i] in adversarial_name_keys:
                tp += 1
            else:
                fp += 1
            beta[i] = 0
        elif agent_name_keys[i] in adversarial_name_keys:
            fn += 1
        else:
            tn += 1
    return (beta, tp, fp, tn, fn)


def _cryptgpu_aggregate_worker(job_dir):
    """Open the masked update sum in one CryptGPU party.

    Party inputs are integer shares over Z_2^64. ``from_shares(precision=0)``
    preserves their existing fixed-point encoding. The module-level worker
    supports multiprocessing with ``spawn``.
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
    rank = int(_comm.get().get_rank())
    if int(_comm.get().get_world_size()) != 3:
        raise RuntimeError("CryptGPU masked aggregation expects a 3-party session")
    share = _np.load(_os.path.join(job_dir, f"delta_share_{rank}.npy"))
    mask_share_dir = job.get("mask_share_dir")
    total_values = int(job["total_values"])
    if not mask_share_dir:
        if share.dtype != _np.int64 or share.size != total_values:
            raise RuntimeError(
                f"delta share must be {total_values} int64 values, got {share.size} of {share.dtype}"
            )
        shared = ArithmeticSharedTensor.from_shares(
            _torch.from_numpy(_np.ascontiguousarray(share)), precision=0
        )
        opened = shared.reveal()
        accepted_count = float(job["active_clients"])
    else:
        from crypten.mpc.primitives import resharing as _resharing
        from crypten.mpc.primitives.binary import BinarySharedTensor

        num_clients = int(job["num_clients"])
        if share.dtype != _np.int64 or share.size != num_clients * total_values:
            raise RuntimeError(
                f"delta share must be {num_clients} x {total_values} int64 values, got {share.size} of {share.dtype}"
            )
        mask_raw = _np.load(_os.path.join(mask_share_dir, f"mask_share_{rank}.npy"))
        if mask_raw.size != num_clients:
            raise RuntimeError(
                f"accept-mask share has {mask_raw.size} bits for {num_clients} clients"
            )
        device = _torch.device("cuda", 0) if _torch.cuda.is_available() else None
        delta_t = _torch.from_numpy(_np.ascontiguousarray(share.reshape(num_clients, total_values)))
        mask_t = _torch.from_numpy(_np.ascontiguousarray(mask_raw))
        if device is not None:
            delta_t = delta_t.to(device)
            mask_t = mask_t.to(device)
        deltas = ArithmeticSharedTensor.from_shares(delta_t, precision=0)
        mask = BinarySharedTensor.from_shares(mask_t, src=rank).reshape([num_clients, 1])
        masked_sum = _resharing.mixed_mul(deltas, mask).sum(dim=0)
        ones = ArithmeticSharedTensor.from_shares(
            _torch.ones(num_clients, 1, dtype=_torch.int64, device=delta_t.device)
            if rank == 0
            else _torch.zeros(num_clients, 1, dtype=_torch.int64, device=delta_t.device),
            precision=0,
        )
        count_ct = _resharing.mixed_mul(ones, mask).sum()
        opened = masked_sum.get_plain_text()
        accepted_count = float(count_ct.get_plain_text().item())
    if rank != 0:
        return None
    out_path = _os.path.join(job_dir, "opened.npy")
    _np.save(out_path, opened.cpu().numpy().astype(_np.int64, copy=False))
    return {"opened_path": out_path, "count": int(opened.numel()), "accepted_count": accepted_count}


def _cryptgpu_secure_aggregate(
    client_weights_list, lens, beta, w_ori, eta, world_size=3, mask_share_dir=None
):
    """Aggregate masked updates with three-party replicated sharing.

    Return ``w_ori + eta * sum_i(beta_i * (w_i - w_ori)) / lens``.
    """
    import json
    import multiprocessing as mp
    import os
    import shutil
    import tempfile
    import numpy as np
    import torch

    if int(world_size) != 3:
        raise ValueError(f"CryptGPU masked aggregation requires world_size=3, got {world_size}")
    param_names = list(w_ori.keys())
    shapes = {name: tuple(w_ori[name].shape) for name in param_names}

    def vectorize(state):
        pieces = []
        for name in param_names:
            value = state[name]
            tensor = (
                value.detach()
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(np.array(value))
            )
            pieces.append(tensor.to(torch.float64).reshape(-1).cpu())
        return torch.cat(pieces, dim=0)

    w_ori_vec = vectorize(w_ori)
    total_values = int(w_ori_vec.numel())
    rng = np.random.default_rng()
    secret_mask = bool(mask_share_dir)
    if secret_mask:
        num_clients = len(client_weights_list)
        if num_clients == 0:
            raise ValueError(
                "[cryptgpu_secure_aggregate] secret_shared mode needs the full client list"
            )
        blocks = [np.zeros((num_clients, total_values), dtype=np.int64) for _ in range(3)]
        for index, client_state in enumerate(client_weights_list):
            delta = vectorize(client_state) - w_ori_vec
            fixed = torch.round(delta * CRYPTGPU_AGG_SCALE).to(torch.int64).numpy()
            s0 = rng.integers(0, 1 << 64, size=fixed.size, dtype=np.uint64).view(np.int64)
            s1 = rng.integers(0, 1 << 64, size=fixed.size, dtype=np.uint64).view(np.int64)
            blocks[0][index] = s0
            blocks[1][index] = s1
            blocks[2][index] = fixed - s0 - s1
        share_sums = [block.reshape(-1) for block in blocks]
        active = num_clients
    else:
        weights = [float(w) for w in beta if float(w) != 0.0]
        if len(weights) != len(client_weights_list):
            raise ValueError(
                f"[cryptgpu_secure_aggregate] weight/client mismatch: {len(weights)} non-zero beta entries for {len(client_weights_list)} clients"
            )
        num_clients = len(client_weights_list)
        active = 0
        share_sums = [np.zeros(total_values, dtype=np.int64) for _ in range(3)]
        for index, client_state in enumerate(client_weights_list):
            weight = weights[index]
            delta = (vectorize(client_state) - w_ori_vec) * weight
            fixed = torch.round(delta * CRYPTGPU_AGG_SCALE).to(torch.int64).numpy()
            s0 = rng.integers(0, 1 << 64, size=fixed.size, dtype=np.uint64).view(np.int64)
            s1 = rng.integers(0, 1 << 64, size=fixed.size, dtype=np.uint64).view(np.int64)
            share_sums[0] += s0
            share_sums[1] += s1
            share_sums[2] += fixed - s0 - s1
            active += 1
    if active == 0:
        return {
            name: w_ori[name].detach().cpu().clone()
            if isinstance(w_ori[name], torch.Tensor)
            else torch.as_tensor(np.array(w_ori[name]))
            for name in param_names
        }
    job_dir = tempfile.mkdtemp(prefix="cryptgpu_agg_")
    try:
        for pid in range(3):
            np.save(os.path.join(job_dir, f"delta_share_{pid}.npy"), share_sums[pid])
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "total_values": total_values,
                    "active_clients": active,
                    "num_clients": num_clients,
                    "mask_share_dir": _require_mask_shares(mask_share_dir, world_size)
                    if secret_mask
                    else None,
                },
                handle,
            )
        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        _activate_cryptgpu_runtime()
        from crypten import mpc as mpc_launcher

        results = mpc_launcher.run_multiprocess(world_size=3)(_cryptgpu_aggregate_worker)(job_dir)
        receipt = next((value for value in results or [] if isinstance(value, dict)), None)
        if receipt is None:
            raise RuntimeError("[cryptgpu_secure_aggregate] no opened vector returned")
        opened = np.load(receipt["opened_path"]).reshape(-1)
        if secret_mask:
            lens = float(receipt.get("accepted_count", lens))
            if lens <= 0:
                raise RuntimeError("[cryptgpu_secure_aggregate] MPC reported no accepted clients")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    if opened.size != total_values:
        raise RuntimeError(
            f"[cryptgpu_secure_aggregate] opened {opened.size} values, expected {total_values}"
        )
    final_vec = w_ori_vec + float(eta) * (
        torch.from_numpy(opened).to(torch.float64) / CRYPTGPU_AGG_SCALE / float(lens)
    )
    aggregated = {}
    offset = 0
    for name in param_names:
        numel = int(np.prod(shapes[name])) if shapes[name] else 1
        aggregated[name] = (
            final_vec.narrow(0, offset, numel).reshape(shapes[name]).to(torch.float32)
        )
        offset += numel
    return aggregated


def _make_filtered_dp_reference(handle, w_locals_clean, agent_name_keys, beta, epoch, ori_weight):
    """Release the filtered noisy reference used for next-round synthesis.

    Return ``None`` when the release service is unavailable or every client
    is rejected; the caller then retains the previous filtered reference.
    """
    service = getattr(handle, "mpc_dp_release_service", None)
    if service is None:
        return None
    mask = [1 if float(b) != 0.0 else 0 for b in beta[: len(w_locals_clean)]]
    if not any(mask):
        logger.warning(
            "[MPC-DP][filtered_reference] every client rejected in round %s; keeping the previous round's filtered reference",
            epoch,
        )
        return None
    secret_mask, mask_share_dir = _mask_application(handle)
    if secret_mask and mask_share_dir:
        _require_mask_shares(mask_share_dir)
    started = time.time()
    release = service.make_fast_aggregate_reference_release(
        clean_client_weights=w_locals_clean,
        global_model=ori_weight,
        client_ids=agent_name_keys,
        round_id=epoch,
        accept_mask=mask,
        release_role="filtered_post_detection",
        mask_share_dir=mask_share_dir,
    )
    handle.mpc_dp_release_metrics = list(service.metrics)
    logger.info(
        "[MPC-DP][filtered_reference] opened filtered DP reference (%s) from %d/%d clients in %.2fs",
        "mask in MPC" if secret_mask and mask_share_dir else "mask opened",
        sum(mask),
        len(mask),
        time.time() - started,
    )
    return release


def _guard_shieldfl_soft_weight_backend(handle):
    """Require an aggregation backend that preserves continuous confidence weights."""
    method = str(handle.params.get("defence_method", "")).strip().lower()
    if method not in ["shieldfl", "fltrust", "safefl", "fltrust_mpc"]:
        return
    backend = str(handle.params.get("secure_agg_backend", "none")).strip().lower()
    backend_norm = backend.replace("-", "_")
    plaintext = backend in ["none", "plain", "plaintext", "disable", "disabled"]
    simulated = backend_norm in [
        "simulated",
        "simulate",
        "simulation",
        "simulated_mpc",
        "mock_mpc",
        "sim_mpc",
    ]
    if plaintext or simulated:
        return
    raise ValueError(
        f"defence_method={method} is a soft-weighting rule and cannot run under secure_agg_backend={backend}: masked secure aggregation would drop its confidence weights. Use secure_agg_backend=none or simulated_mpc; its secure-backend cost is reported separately."
    )


def _require_input_validation_binding(handle, final_weights, beta):
    """Check that final aggregation uses the objects validated in this round."""
    method = str(handle.params.get("defence_method", "")).strip().lower()
    if method not in ["rofl", "elsa"]:
        return
    runtime = handle.params.get("_input_validation_runtime", None)
    round_id = int(getattr(handle, "current_epoch", -1))
    if runtime is None or int(runtime.get("round", -2)) != round_id:
        raise RuntimeError(f"[{method}] final aggregation without this round's input validation")
    accepted = [int(i) for i in runtime["accepted_indices"]]
    active = [i for i in range(len(beta)) if beta[i] != 0]
    if active != accepted:
        raise RuntimeError(
            f"[{method}] final aggregation active set {active} != validated set {accepted}"
        )
    for i in accepted:
        if id(final_weights[i]) != int(runtime["object_ids"][i]):
            raise RuntimeError(
                f"[{method}] final aggregation is not using the validated object for client index {i}"
            )


def _masked_aggregate_final(handle, final_weights_for_aggregation, lens, beta):
    _guard_shieldfl_soft_weight_backend(handle)
    _require_input_validation_binding(handle, final_weights_for_aggregation, beta)
    if sum(beta) == 0:
        _record_secure_aggregation(
            handle,
            handle.params.get("secure_agg_backend", "none"),
            "skipped_all_rejected",
            0,
            False,
        )
        raise RuntimeError("[secure_aggregation] all clients filtered; cannot aggregate")
    backend = str(handle.params.get("secure_agg_backend", "none")).lower().replace("-", "_")
    if backend in {"none", "plain", "plaintext", "disabled", "simulated_mpc", "simulated"}:
        return Aggregation(
            copy.deepcopy(handle.model.state_dict()),
            copy.deepcopy(final_weights_for_aggregation),
            lens,
            beta,
            handle.params["eta"],
            handle.params["defence_method"],
            handle.params,
        )
    if backend in {"mp_spdz", "mpspdz"}:
        if _secure_mask_mode(handle.params) == MASK_SECRET_SHARED:
            raise ValueError("MP-SPDZ final aggregation does not support a secret detector mask")
        from secure_aggregation import SecureAggregation

        root = (
            handle.params.get("mp_spdz_root")
            or os.environ.get("MP_SPDZ_ROOT")
            or str(Path(__file__).resolve().parent / "mp-spdz-0.4.1")
        )
        if not Path(root).is_dir():
            raise FileNotFoundError(f"MP-SPDZ runtime not found: {root}")
        active = [final_weights_for_aggregation[i] for i in range(len(beta)) if beta[i] != 0]
        service = SecureAggregation(mp_spdz_path=str(root))
        try:
            result = service.secure_aggregate(active)
            execution = service.last_execution_receipt or {}
            receipt = {
                "receipt_schema": SECURE_AGGREGATION_RECEIPT_SCHEMA,
                "attempt_count": execution.get("attempt_count", 1),
                "retry_count": execution.get("retry_count", 0),
                "retry_reasons": execution.get("retry_reasons", []),
            }
            _record_secure_aggregation(handle, "mp-spdz", "success", len(active), False, receipt)
            return result
        except Exception:
            _record_secure_aggregation(handle, "mp-spdz", "failed", len(active), False)
            raise
        finally:
            service.cleanup()
    if backend not in {"cryptgpu", "cgpu"}:
        raise ValueError(f"Unsupported secure_agg_backend={backend!r}")
    secret_mask, mask_share_dir = _mask_application(handle)
    active = (
        list(final_weights_for_aggregation)
        if secret_mask and mask_share_dir
        else [final_weights_for_aggregation[i] for i in range(len(beta)) if beta[i] != 0]
    )
    start = time.time()
    try:
        result = _cryptgpu_secure_aggregate(
            active,
            lens,
            beta,
            copy.deepcopy(handle.model.state_dict()),
            handle.params["eta"],
            world_size=int(handle.params.get("cryptgpu_world_size", 3)),
            mask_share_dir=mask_share_dir,
        )
    except Exception:
        _record_secure_aggregation(handle, backend, "failed", len(active), False)
        raise
    _record_secure_aggregation(
        handle,
        "cryptgpu-secret-mask" if secret_mask and mask_share_dir else "cryptgpu",
        "success",
        len(active),
        False,
        {
            "receipt_schema": SECURE_AGGREGATION_RECEIPT_SCHEMA,
            "runtime_lock_acquired": False,
            "runtime_lock_released": False,
            "attempt_count": 1,
            "retry_count": 0,
            "retry_reasons": [],
            "cleanup_complete": True,
            "runtime_seconds": time.time() - start,
        },
    )
    return result


def _run_testing(handle, w_glob):
    try:
        handle.model.load_state_dict(w_glob)
    except Exception:
        try:
            w_glob_cpu = {
                k: v.detach().to("cpu") if isinstance(v, torch.Tensor) else v
                for k, v in w_glob.items()
            }
            handle.model.load_state_dict(w_glob_cpu)
        except Exception:
            handle.model.load_state_dict(w_glob)
    handle.model.to(device)
    use_gpu_preload = handle.params.get("gpu_preload", False) and hasattr(handle, "gpu_test_data")
    if use_gpu_preload:
        acc = test.normal_test(handle.model, None, handle.params, handle, poison=False)
        asr = test.normal_test(handle.model, None, handle.params, handle, poison=True)
    else:
        acc = test.normal_test(handle.model, handle.test_data, handle.params, handle, poison=False)
        asr = test.normal_test(
            handle.model, handle.test_un_target_label_data, handle.params, handle, poison=True
        )
    return (acc, asr)


def FedAvg(handle):
    count_all_attack = 0
    count_all_defense = 0
    count_correct_defense = 0
    enable_ldp = _read_ldp_config(handle)
    global_tp = 0
    global_fp = 0
    global_tn = 0
    global_fn = 0
    acc = 0.0
    asr = 0.0
    final_precision = 0.0
    final_recall = 0.0
    final_accuracy = 0.0
    final_f1_score = 0.0
    accepted_client_counts = []
    round_metrics = []
    canary_readings = []
    canary_fired_round = None
    canary_restored_round = None
    canary_restore_state = None
    canary_fire_log = []
    canary_attempts = 0
    canary_attempt_cap_hit = False
    canary_rollback_pending = None
    canary_first_fired_round = None
    canary_halted = False
    canary_ckpt_ring = collections.deque(maxlen=12)
    timing_stats = {
        "client_selection": 0.0,
        "client_training": 0.0,
        "aggregation": 0.0,
        "defense": 0.0,
        "testing": 0.0,
        "other": 0.0,
    }
    epoch = handle.start_epoch - 1
    canary_max_attempts = int(
        handle.params.get("detector_canary_max_attempts", 3 * handle.params["epochs"])
    )
    while True:
        epoch += 1
        if epoch > handle.params["epochs"]:
            break
        canary_attempts += 1
        if canary_attempts > canary_max_attempts:
            canary_attempt_cap_hit = True
            logger.warning(
                f"[canary] attempt cap {canary_max_attempts} reached at effective round {epoch}; the attacker has denied completion of training. Stopping."
            )
            break
        handle.current_epoch = int(epoch)
        handle.params["_fedavg_round"] = int(epoch)
        handle.params.pop("_shieldfl_runtime", None)
        handle.params.pop("_input_validation_runtime", None)
        if bool(handle.params.get("lr_decay", False)):
            if epoch % handle.params["lr_decay_epoch"] == 0:
                handle.params["lr"] = handle.params["lr"] * handle.params["lr_decay_gamma"]
                handle.params["poison_lr"] = (
                    handle.params["poison_lr"] * handle.params["lr_decay_gamma"]
                )
        epoch_start_time = time.time()
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                try:
                    _torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if bool(handle.params.get("log_memory_each_epoch", True)):
                logger.info(memory_snapshot(tag=f"epoch={epoch}/start"))
        except Exception:
            pass
        client_selection_start = time.time()
        agent_name_keys, adversarial_name_keys, lens, ori_weight = _client_selection(epoch, handle)
        count_all_attack += len(adversarial_name_keys)
        client_selection_time = time.time() - client_selection_start
        timing_stats["client_selection"] += client_selection_time
        client_training_start = time.time()
        w_locals, w_locals_clean, w_locals_noisy, beta = _train_clients(
            epoch, handle, agent_name_keys, adversarial_name_keys, enable_ldp
        )
        beta = beta
        client_training_time = time.time() - client_training_start
        timing_stats["client_training"] += client_training_time
        aggregation_start = time.time()
        w_glob = _aggregate_for_trigger(
            handle, w_locals, w_locals_noisy, lens, beta, enable_ldp, epoch
        )
        aggregation_time = time.time() - aggregation_start
        timing_stats["aggregation"] += aggregation_time
        defense_start = time.time()
        defence_method = handle.params["defence_method"]
        defence_method_norm = str(defence_method).lower() if defence_method is not None else ""
        ori_for_trigger = ori_weight
        try:
            source_mode_trigger = str(handle.params.get("ldp_ori_weight_source", "clean")).lower()
            if enable_ldp and source_mode_trigger in [
                "diff",
                "noisy_agg",
                "aggregate_ref",
                "aggregate_noisy_reference",
            ]:
                if hasattr(handle, "prev_noisy_w_glob") and handle.prev_noisy_w_glob is not None:
                    ori_for_trigger = copy.deepcopy(handle.prev_noisy_w_glob)
                    logger.info(
                        f"[LDP][trigger] Use previous-round noisy global model as ori_weight (source={source_mode_trigger})"
                    )
                else:
                    logger.info(
                        "[LDP][trigger] Previous-round noisy global weights not found; use current model weights as ori_weight"
                    )
        except Exception as _e:
            logger.warning(
                f"[LDP][trigger] Failed to read previous-round noisy global weights: {_e}; fallback to current model weights for trigger generation"
            )
        g_imgs, trigger_tensor = _generate_trigger(ori_for_trigger, w_glob, handle.params, epoch)
        tp = fp = tn = fn = 0
        if defence_method_norm == "ours":
            defence_method_norm = "filterfl"
        elif defence_method_norm == "ours_cryptgpu":
            defence_method_norm = "priv-filterfl"
        if defence_method_norm in ["filterfl", "priv-filterfl", "ours_plaintext_cryptgpu"]:
            standard = handle.params["ours_standard"][(epoch - 1) // 500]
            if defence_method_norm == "filterfl":
                beta, tp, fp, tn, fn = _detect_backdoor_ours(
                    handle,
                    enable_ldp,
                    w_locals,
                    w_locals_clean,
                    w_locals_noisy,
                    ori_weight,
                    agent_name_keys,
                    adversarial_name_keys,
                    g_imgs,
                    beta,
                    threshold=standard,
                )
            elif defence_method_norm == "priv-filterfl":
                if _is_simulated_mpc_mode(handle.params):
                    logger.info(
                        "[priv-filterfl] priv_filterfl_mpc_mode=simulated, using local CryptGPU-aligned forward as simulated MPC"
                    )
                    beta, tp, fp, tn, fn = _detect_backdoor_ours_plaintext_cryptgpu(
                        handle,
                        enable_ldp,
                        w_locals,
                        w_locals_clean,
                        w_locals_noisy,
                        ori_weight,
                        agent_name_keys,
                        adversarial_name_keys,
                        g_imgs,
                        beta,
                        threshold=standard,
                    )
                else:
                    beta, tp, fp, tn, fn = _detect_backdoor_ours_cryptgpu(
                        handle,
                        enable_ldp,
                        w_locals,
                        w_locals_clean,
                        w_locals_noisy,
                        ori_weight,
                        agent_name_keys,
                        adversarial_name_keys,
                        g_imgs,
                        beta,
                        threshold=standard,
                    )
            elif defence_method_norm == "ours_plaintext_cryptgpu":
                beta, tp, fp, tn, fn = _detect_backdoor_ours_plaintext_cryptgpu(
                    handle,
                    enable_ldp,
                    w_locals,
                    w_locals_clean,
                    w_locals_noisy,
                    ori_weight,
                    agent_name_keys,
                    adversarial_name_keys,
                    g_imgs,
                    beta,
                    threshold=standard,
                )
        elif defence_method_norm in ["flame"]:
            beta, tp, fp, tn, fn = _detect_backdoor_flame(
                handle,
                enable_ldp,
                w_locals,
                w_locals_clean,
                w_locals_noisy,
                ori_weight,
                agent_name_keys,
                adversarial_name_keys,
                beta,
            )
        elif defence_method_norm in ["shieldfl"]:
            beta, tp, fp, tn, fn = _detect_backdoor_shieldfl(
                handle,
                enable_ldp,
                w_locals,
                w_locals_clean,
                w_locals_noisy,
                ori_weight,
                agent_name_keys,
                adversarial_name_keys,
                beta,
                epoch,
            )
        elif defence_method_norm in ["crowdguard"]:
            beta, tp, fp, tn, fn = _detect_backdoor_crowdguard(
                handle,
                enable_ldp,
                w_locals,
                w_locals_clean,
                w_locals_noisy,
                ori_weight,
                agent_name_keys,
                adversarial_name_keys,
                beta,
            )
        elif defence_method_norm in ["fltrust", "safefl", "fltrust_mpc"]:
            beta, tp, fp, tn, fn = _detect_backdoor_fltrust(
                handle,
                enable_ldp,
                w_locals,
                w_locals_clean,
                w_locals_noisy,
                ori_weight,
                agent_name_keys,
                adversarial_name_keys,
                beta,
            )
        elif defence_method_norm in ["fedredefense", "fedre"]:
            beta, tp, fp, tn, fn = _detect_backdoor_fedredefense(
                handle,
                enable_ldp,
                w_locals,
                w_locals_clean,
                w_locals_noisy,
                ori_weight,
                agent_name_keys,
                adversarial_name_keys,
                g_imgs[0],
                beta,
            )
        elif defence_method_norm in ["elsa", "rofl"]:
            if enable_ldp:
                raise NotImplementedError(
                    f"[{defence_method_norm}] input validation is defined on the submitted client model; the LDP clean/noisy split is not supported"
                )
            w_locals, beta, tp, fp, tn, fn = _apply_input_validation_rofl_elsa(
                handle,
                defence_method_norm,
                w_locals,
                ori_weight,
                beta,
                agent_name_keys,
                adversarial_name_keys,
                epoch,
            )
        elif defence_method_norm in ["lockdown"]:
            logger.info(
                "[lockdown] Use isolated subspace training without explicit client filtering"
            )
        if g_imgs:
            cached_probe = g_imgs[0].detach().to("cpu", dtype=torch.float32).contiguous().clone()
            handle.previous_detection_trigger = cached_probe
            handle.previous_detection_trigger_round = int(epoch)
            attack_mode = (
                str(handle.params.get("attack_mode", handle.params.get("attack.mode", "")))
                .strip()
                .lower()
            )
            if attack_mode in PRIV_FILTERFL_DETECTOR_EVASION_ATTACK_MODES:
                cached_digest = train._detector_probe_digest(cached_probe)
                handle.previous_detection_trigger_digest_schema = (
                    train._DETECTOR_PROBE_DIGEST_SCHEMA
                )
                handle.previous_detection_trigger_digest = cached_digest
                ring = getattr(handle, "detector_trigger_ring", None)
                if ring is None:
                    ring = collections.deque(maxlen=64)
                    handle.detector_trigger_ring = ring
                ring.append(
                    {
                        "round": int(epoch),
                        "trigger": cached_probe,
                        "digest_schema": train._DETECTOR_PROBE_DIGEST_SCHEMA,
                        "digest": cached_digest,
                    }
                )
        inc_defense = 0
        inc_correct = 0
        for i in range(len(agent_name_keys)):
            if beta[i] == 0:
                lens -= 1
                inc_defense += 1
                logger.info(agent_name_keys[i])
                if agent_name_keys[i] in adversarial_name_keys:
                    inc_correct += 1
        accepted_client_counts.append(int(sum((1 for flag in beta if flag != 0))))
        canary_reading = _detector_canary_evaluate(handle, epoch)
        if canary_reading["enabled"]:
            canary_readings.append(canary_reading)
            if canary_reading["ratio"] is not None:
                logger.info(
                    f"[canary] round={epoch} cohort_median={canary_reading['cohort_median']:.3e} baseline={canary_reading['baseline']:.3e} ratio={canary_reading['ratio']:.4f}"
                )
            if canary_reading["fired"] and canary_fired_round is None:
                canary_fired_round = int(epoch)
                lag = int(handle.params.get("detector_canary_rollback_lag", 5))
                restored = None
                for entry in reversed(canary_ckpt_ring):
                    if entry["round"] <= int(epoch) - lag:
                        restored = entry
                        break
                if restored is None and canary_ckpt_ring:
                    restored = canary_ckpt_ring[0]
                logger.warning(
                    f"[canary] FIRED at round {epoch} (ratio={canary_reading['ratio']:.2e} < {handle.params.get('detector_canary_ratio', 0.01)}): the cohort-wide trigger response has collapsed, so per-client scores can no longer separate anyone. Rolling back to round {(restored['round'] if restored else 'n/a')} and halting."
                )
                response = (
                    str(handle.params.get("detector_canary_response", "halt")).strip().lower()
                )
                if response == "halt":
                    if restored is not None:
                        canary_restore_state = {k: v.clone() for k, v in restored["state"].items()}
                        canary_restored_round = int(restored["round"])
                    canary_halted = True
                elif response == "rollback_continue" and restored is not None:
                    handle.model.load_state_dict(
                        {k: v.clone() for k, v in restored["state"].items()}
                    )
                    for attr, key in (
                        ("prev_clean_w_glob", "prev_clean"),
                        ("prev_noisy_w_glob", "prev_noisy"),
                        ("prev_noisy_w_glob_prev", "prev_noisy_prev"),
                    ):
                        if restored.get(key) is not None:
                            setattr(
                                handle,
                                attr,
                                {
                                    k: v.clone() if isinstance(v, torch.Tensor) else v
                                    for k, v in restored[key].items()
                                },
                            )
                    handle.detector_canary_history = collections.deque(
                        restored.get("canary_history") or [],
                        maxlen=int(handle.params.get("detector_canary_window", 20)),
                    )
                    while canary_ckpt_ring and canary_ckpt_ring[-1]["round"] > int(
                        restored["round"]
                    ):
                        canary_ckpt_ring.pop()
                    escalate = float(handle.params.get("detector_canary_escalate", 1.0))
                    if escalate != 1.0:
                        old_q = float(handle.params["robust_detector_threshold_q"])
                        handle.params["robust_detector_threshold_q"] = old_q * escalate
                        logger.warning(
                            f"[canary] escalating operating point q {old_q:.4f} -> {old_q * escalate:.4f} after fire #{len(canary_fire_log) + 1}"
                        )
                    handle.canary_resample_salt = (
                        int(getattr(handle, "canary_resample_salt", 0) or 0) + 1
                    )
                    canary_fire_log.append(
                        {
                            "fired_round": int(epoch),
                            "restored_round": int(restored["round"]),
                            "attempt": int(canary_attempts),
                            "q_after": float(handle.params["robust_detector_threshold_q"]),
                            "resample_salt": int(handle.canary_resample_salt),
                        }
                    )
                    if canary_first_fired_round is None:
                        canary_first_fired_round = int(epoch)
                    canary_fired_round = None
                    canary_rollback_pending = int(restored["round"])
        if canary_reading["enabled"] and (not canary_halted):

            def _snap(obj):
                if obj is None:
                    return None
                return {
                    k: v.detach().to("cpu").clone() if isinstance(v, torch.Tensor) else v
                    for k, v in obj.items()
                }

            canary_ckpt_ring.append(
                {
                    "round": int(epoch),
                    "state": _snap(handle.model.state_dict()),
                    "prev_clean": _snap(getattr(handle, "prev_clean_w_glob", None)),
                    "prev_noisy": _snap(getattr(handle, "prev_noisy_w_glob", None)),
                    "prev_noisy_prev": _snap(getattr(handle, "prev_noisy_w_glob_prev", None)),
                    "canary_history": list(getattr(handle, "detector_canary_history", []) or []),
                }
            )
        if canary_rollback_pending is not None:
            epoch = int(canary_rollback_pending)
            canary_rollback_pending = None
            try:
                clear_cuda_memory(tag=f"epoch={epoch}/canary_rollback")
            except Exception:
                pass
            continue
        final_weights_for_aggregation = w_locals
        if enable_ldp:
            use_noisy_final = bool(handle.params.get("ldp_final_agg_use_noisy", False))
            if use_noisy_final and w_locals_noisy is not None:
                logger.info(
                    "[LDP][final_aggregation] Use detected noisy weights for final aggregation"
                )
                final_weights_for_aggregation = w_locals_noisy
            elif w_locals_clean is not None:
                logger.info(
                    "[LDP][final_aggregation] Use detected clean weights for final aggregation"
                )
                final_weights_for_aggregation = w_locals_clean
        try:
            w_masked = _masked_aggregate_final(handle, final_weights_for_aggregation, lens, beta)
            w_glob = w_masked if w_masked is not None else ori_weight
        except RuntimeError as e:
            msg = str(e)
            if "all clients filtered" in msg:
                logger.warning(
                    "[secure_aggregation] All clients filtered; fallback to previous clean global model"
                )
                if hasattr(handle, "prev_clean_w_glob") and handle.prev_clean_w_glob is not None:
                    w_glob = handle.prev_clean_w_glob
                else:
                    logger.warning(
                        "[secure_aggregation] Previous clean global model cache not found; fallback to current model weights"
                    )
                    w_glob = ori_weight
            else:
                raise
        if defence_method_norm == "lockdown" and bool(
            handle.params.get("lockdown_consensus", True)
        ):
            try:
                masks = []
                if hasattr(handle, "lockdown_states"):
                    for client_id in agent_name_keys:
                        st = handle.lockdown_states.get(client_id)
                        if st and st.get("mask") is not None:
                            masks.append(st["mask"])
                if masks:
                    threshold = float(handle.params.get("lockdown_consensus_threshold", 0.5))
                    consensus = _lockdown_consensus_mask(masks, threshold)
                    w_glob = _apply_lockdown_consensus(w_glob, ori_weight, consensus)
                    handle.lockdown_consensus_mask = consensus
                    logger.info(f"[lockdown] Consensus mask applied (threshold={threshold})")
            except Exception as _e:
                logger.warning(f"[lockdown] Consensus fusion failed; skip: {_e}")
        global_tp += tp
        global_fp += fp
        global_tn += tn
        global_fn += fn
        count_all_defense += inc_defense
        count_correct_defense += inc_correct
        logger.info(
            "count_all_attack: {}, count_all_defense: {}, count_correct_defense: {}".format(
                count_all_attack, count_all_defense, count_correct_defense
            )
        )
        defense_time = time.time() - defense_start
        timing_stats["defense"] += defense_time
        try:
            from memory_utils import state_dict_to_cpu

            handle.prev_clean_w_glob = state_dict_to_cpu(w_glob)
        except Exception as _e:
            try:
                handle.prev_clean_w_glob = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in w_glob.items()
                }
            except Exception:
                handle.prev_clean_w_glob = w_glob
        testing_start = time.time()
        acc, asr = _run_testing(handle, w_glob)
        testing_time = time.time() - testing_start
        timing_stats["testing"] += testing_time
        round_metrics.append(
            {
                "round": int(epoch),
                "clean_accuracy": float(acc),
                "attack_success_rate": float(asr),
                "accepted_client_count": int(accepted_client_counts[-1]),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
        if canary_halted:
            logger.warning(
                f"[canary] halting after round {epoch}: training stopped at the rolled-back round-{canary_restored_round} global model. The attacker denied completion of training; it did not get a backdoor accepted without the collapse being visible."
            )
            if canary_restore_state is not None:
                handle.model.load_state_dict(canary_restore_state)
                w_glob = canary_restore_state
                handle.prev_clean_w_glob = state_dict_to_cpu(canary_restore_state)
            acc, asr = _run_testing(handle, w_glob)
            round_metrics.append(
                {
                    "round": int(epoch),
                    "clean_accuracy": float(acc),
                    "attack_success_rate": float(asr),
                    "accepted_client_count": int(accepted_client_counts[-1]),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "canary_halt": True,
                }
            )
            break
        mpc_reference_required = False
        try:
            source_mode = str(handle.params.get("ldp_ori_weight_source", "clean")).lower()
            if enable_ldp and source_mode in [
                "diff",
                "noisy_agg",
                "aggregate_ref",
                "aggregate_noisy_reference",
            ]:
                dp_release_source_for_snapshot = canonical_dp_release_source(
                    handle.params.get("dp_release_source", DP_RELEASE_CLIENT_LDP)
                )
                if (
                    source_mode in ["noisy_agg", "aggregate_ref", "aggregate_noisy_reference"]
                    and dp_release_source_for_snapshot in [DP_RELEASE_MPC_AGGREGATE_FAST]
                    and (getattr(handle, "mpc_dp_aggregate_reference", None) is not None)
                ):
                    mpc_reference_required = True
                    filtered_reference = _make_filtered_dp_reference(
                        handle,
                        w_locals_clean,
                        agent_name_keys,
                        beta,
                        epoch,
                        state_dict_to_cpu(ori_weight),
                    )
                    if filtered_reference:
                        prior_reference = getattr(handle, "prev_noisy_w_glob", None)
                        if prior_reference is not None:
                            handle.prev_noisy_w_glob_prev = prior_reference
                        handle.prev_noisy_w_glob = filtered_reference
                        logger.info(
                            f"[MPC-DP] Saved previous-round DP reference from the post-detection filtered aggregate (source={source_mode})"
                        )
                    elif getattr(handle, "prev_noisy_w_glob", None) is None:
                        raise RuntimeError(
                            "[MPC-DP] no filtered DP reference could be formed and no previous one exists; refusing to substitute the unfiltered pre-detection aggregate, which contains the rejected clients"
                        )
                    source_mode = "aggregate_ref_saved"
                if source_mode == "noisy_agg":
                    if w_locals_noisy is None:
                        logger.warning(
                            "[LDP][ori_weight_source=noisy_agg] w_locals_noisy unavailable; fallback to diff+noise generation path"
                        )
                        source_mode = "diff"
                    else:
                        logger.info(
                            "[LDP][ori_weight_source=noisy_agg] Use noisy weights with plaintext masked aggregation to produce previous noisy global weights"
                        )
                        noisy_mode = (
                            str(handle.params.get("coordinator_noisy_transcript", "linked"))
                            .strip()
                            .lower()
                        )
                        if noisy_mode in ["anonymous_shuffled", "anonymous", "shuffled"]:
                            raise RuntimeError(
                                "ldp_ori_weight_source=noisy_agg is incompatible with coordinator_noisy_transcript=anonymous_shuffled because masked selection would relink noisy reports."
                            )
                        base_sd_cpu = copy.deepcopy(handle.model.to("cpu").state_dict())
                        prev_noisy_w_glob = Aggregation(
                            base_sd_cpu,
                            copy.deepcopy(w_locals_noisy),
                            lens,
                            beta,
                            handle.params["eta"],
                            handle.params["defence_method"],
                            handle.params,
                        )
                        handle.prev_noisy_w_glob = prev_noisy_w_glob
                        logger.info(
                            "[LDP] Saved previous-round noisy global weights (source: noisy_agg) for next-round ori_weight"
                        )
                if source_mode == "diff":
                    if not hasattr(handle, "ldp_for_ori") or handle.ldp_for_ori is None:
                        handle.ldp_for_ori = create_ldp_instance(handle.params)
                    ldp_inst = handle.ldp_for_ori
                    from memory_utils import state_dict_to_cpu

                    w_glob_cpu = state_dict_to_cpu(w_glob)
                    ori_weight_cpu = state_dict_to_cpu(ori_weight)

                    def _is_float_tensor(t: torch.Tensor) -> bool:
                        return t.dtype in (
                            torch.float32,
                            torch.float64,
                            torch.float16,
                            torch.bfloat16,
                        )

                    def _is_excluded_key(k: str) -> bool:
                        return (
                            "running_mean" in k or "running_var" in k or "num_batches_tracked" in k
                        )

                    update_clean = {}
                    float_update_keys = []
                    excluded_keys = []
                    for k in w_glob_cpu.keys():
                        if _is_excluded_key(k):
                            excluded_keys.append(k)
                            continue
                        if k in ori_weight_cpu and _is_float_tensor(ori_weight_cpu[k]):
                            update_clean[k] = w_glob_cpu[k] - ori_weight_cpu[k]
                            float_update_keys.append(k)
                        else:
                            excluded_keys.append(k)
                    if len(excluded_keys) > 0:
                        logger.debug(
                            f"[LDP] [ori_weight_noisy_prev] excluded non-noise keys count={len(excluded_keys)}, examples={excluded_keys[:5]}"
                        )
                    _, noisy_update = apply_ldp_to_weights(update_clean, ldp_inst)

                    def _sanitize(t: torch.Tensor) -> torch.Tensor:
                        if not isinstance(t, torch.Tensor):
                            t = torch.as_tensor(t)
                        t = t.detach().cpu()
                        if not t.is_floating_point():
                            t = t.float()
                        if not torch.isfinite(t).all():
                            t = torch.nan_to_num(t, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
                        return t

                    invalid_keys = []
                    for k in list(noisy_update.keys()):
                        t = noisy_update[k]
                        if isinstance(t, torch.Tensor) and (not torch.isfinite(t).all()):
                            invalid_keys.append(k)
                        noisy_update[k] = _sanitize(t)
                    if len(invalid_keys) > 0:
                        logger.warning(
                            f"[LDP] Found non-finite keys in previous noisy update count={len(invalid_keys)}, examples={invalid_keys[:5]}.cleaned and replaced."
                        )
                    prev_noisy_w_glob = {}
                    for k in ori_weight_cpu.keys():
                        if k in float_update_keys:
                            prev_noisy_w_glob[k] = _sanitize(ori_weight_cpu[k] + noisy_update[k])
                        else:
                            prev_noisy_w_glob[k] = _sanitize(w_glob_cpu[k])
                    handle.prev_noisy_w_glob = prev_noisy_w_glob
                    logger.info(
                        "[LDP] Saved previous-round noisy global weights (source: diff) for next-round ori_weight"
                    )
        except Exception as e:
            if mpc_reference_required:
                raise
            logger.warning(f"[LDP] Failed to save previous-round noisy global weights: {e}")
        epoch_total_time = time.time() - epoch_start_time
        if round_metrics and round_metrics[-1].get("round") == int(epoch):
            round_metrics[-1]["round_wall_s"] = float(epoch_total_time)
            round_metrics[-1]["client_training_s"] = float(client_training_time)
            round_metrics[-1]["defense_s"] = float(defense_time)
            round_metrics[-1]["aggregation_s"] = float(aggregation_time)
            round_metrics[-1]["testing_s"] = float(testing_time)
        other_time = epoch_total_time - (
            client_selection_time
            + client_training_time
            + aggregation_time
            + defense_time
            + testing_time
        )
        timing_stats["other"] += other_time
        try:
            if bool(handle.params.get("log_memory_each_epoch", True)):
                logger.info(memory_snapshot(tag=f"epoch={epoch}/end"))
        except Exception:
            pass
        try:
            clear_cuda_memory(tag=f"epoch={epoch}/end")
        except Exception:
            pass
        try:
            del w_locals, w_locals_clean, w_locals_noisy, w_glob, g_imgs, trigger_tensor
        except Exception:
            pass
        if epoch % 10 == 0:
            total_time = sum(timing_stats.values())
            print(f"\n=== Cumulative timing stats for first {epoch} rounds ===")
            for stage, time_spent in timing_stats.items():
                percentage = time_spent / total_time * 100 if total_time > 0 else 0
                print(f" {stage}: {time_spent:.3f}s ({percentage:.1f}%)")
            print(f" Total elapsed: {total_time:.3f}s")
            print(f" Average per round: {total_time / epoch:.3f}s")
            print(" ========================")
    if handle.params["defence_method"] is not None:
        final_precision = global_tp / (global_tp + global_fp) if global_tp + global_fp > 0 else 0.0
        final_recall = global_tp / (global_tp + global_fn) if global_tp + global_fn > 0 else 0.0
        final_accuracy = (
            (global_tp + global_tn) / (global_tp + global_tn + global_fp + global_fn)
            if global_tp + global_tn + global_fp + global_fn > 0
            else 0.0
        )
        final_f1_score = (
            2 * (final_precision * final_recall) / (final_precision + final_recall)
            if final_precision + final_recall > 0
            else 0.0
        )
        logger.info(f"\n=== Final summary stats ({handle.params['defence_method']}) ===")
        logger.info(f"Totals TP={global_tp}, FP={global_fp}, TN={global_tn}, FN={global_fn}")
        logger.info(f"Final Precision={final_precision:.3f}, Recall={final_recall:.3f}")
        logger.info(f"Final Accuracy={final_accuracy:.3f}, F1 Score={final_f1_score:.3f}")
        logger.info(f"Final Clean Accuracy={acc:.3f}, Attack Success Rate={asr:.3f}")
        logger.info("=" * 50)
    dba_local_attack_success_rates = {}
    if is_dba_attack(handle.params) and bool(handle.params.get("dba_report_local_asr", True)):
        handle.model.to(device)
        for pattern_index in range(int(handle.params.get("trigger_num", 1))):
            local_asr = test.normal_test(
                handle.model,
                handle.test_un_target_label_data,
                handle.params,
                handle,
                poison=True,
                poison_pattern_index=pattern_index,
            )
            dba_local_attack_success_rates[str(pattern_index)] = float(local_asr)
        logger.info(f"[DBA] Final local-shard ASR: {dba_local_attack_success_rates}")
    handle.params.pop("_shieldfl_runtime", None)
    handle.params.pop("_input_validation_runtime", None)
    score_records = getattr(handle, "detection_score_records", []) or []
    malicious_scores = [float(row["p_target"]) for row in score_records if row.get("malicious_gt")]
    benign_scores = [float(row["p_target"]) for row in score_records if not row.get("malicious_gt")]
    malicious_accept_rate = (
        (count_all_attack - global_tp) / count_all_attack if count_all_attack > 0 else 0.0
    )
    benign_total_seen = global_tn + global_fp
    benign_reject_rate = global_fp / benign_total_seen if benign_total_seen > 0 else 0.0
    benign_acceptance_rate = 1.0 - benign_reject_rate if benign_total_seen > 0 else 0.0
    mean_accepted_client_count = (
        float(sum(accepted_client_counts)) / float(len(accepted_client_counts))
        if accepted_client_counts
        else 0.0
    )
    return {
        "tp": global_tp,
        "fp": global_fp,
        "tn": global_tn,
        "fn": global_fn,
        "precision": final_precision,
        "recall": final_recall,
        "accuracy": final_accuracy,
        "f1": final_f1_score,
        "clean_accuracy": acc,
        "attack_success_rate": asr,
        "malicious_accept_rate": malicious_accept_rate,
        "benign_reject_rate": benign_reject_rate,
        "benign_acceptance_rate": benign_acceptance_rate,
        "mean_accepted_client_count": mean_accepted_client_count,
        "accepted_client_counts": accepted_client_counts,
        "round_metrics": round_metrics,
        "dba_local_attack_success_rates": dba_local_attack_success_rates,
        "detection_score_records": score_records,
        "detector_canary": {
            "enabled": bool(handle.params.get("detector_canary_enabled", False)),
            "ratio_threshold": float(handle.params.get("detector_canary_ratio", 0.01)),
            "window": int(handle.params.get("detector_canary_window", 20)),
            "rollback_lag": int(handle.params.get("detector_canary_rollback_lag", 5)),
            "response": str(handle.params.get("detector_canary_response", "halt")),
            "escalate": float(handle.params.get("detector_canary_escalate", 1.0)),
            "fired_round": canary_fired_round
            if canary_fired_round is not None
            else canary_first_fired_round,
            "restored_round": canary_restored_round,
            "halted": bool(canary_halted),
            "fire_log": canary_fire_log,
            "fire_count": len(canary_fire_log) + (1 if canary_halted else 0),
            "total_attempts": int(canary_attempts),
            "attempt_cap_hit": bool(canary_attempt_cap_hit),
            "final_q": float(handle.params.get("robust_detector_threshold_q", 0.0)),
            "readings": canary_readings,
        },
        "detector_output_records": getattr(handle, "detector_output_records", []) or [],
        "bn_discrepancy_records": getattr(handle, "bn_discrepancy_records", []) or [],
        "adaptive_evasion_records": getattr(handle, "adaptive_evasion_records", []) or [],
        "empty_client_train_records": getattr(handle, "empty_client_train_records", []) or [],
        "cryptgpu_runtime_records": getattr(handle, "cryptgpu_runtime_records", []) or [],
        "secure_aggregation_records": getattr(handle, "secure_aggregation_records", []) or [],
        "shieldfl_round_records": getattr(handle, "shieldfl_round_records", []) or [],
        "input_validation_records": getattr(handle, "input_validation_records", []) or [],
        "detection_confusion_semantics": "derived_soft_weight_threshold_reporting_only"
        if str(handle.params.get("defence_method", "")).strip().lower() == "shieldfl"
        else "hard_decision",
        "client_sample_counts": [
            int(value) for value in getattr(handle, "clients_data_num", []) or []
        ],
        "training_batch_size": int(handle.params.get("batch_size", 0)),
        "mean_malicious_score": float(sum(malicious_scores) / len(malicious_scores))
        if malicious_scores
        else "",
        "mean_benign_score": float(sum(benign_scores) / len(benign_scores))
        if benign_scores
        else "",
    }
