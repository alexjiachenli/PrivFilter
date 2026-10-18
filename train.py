import copy
import hashlib
import logging
import math
import types
from collections.abc import Mapping
import torch
from torch import nn
import main
from backdoor_attack import apply_pixel_patterns, poison_batch, resolve_client_pattern_index

from device import device
from differential_privacy import create_ldp_instance, apply_ldp_to_weights
from memory_utils import state_dict_to_cpu, clear_cuda_memory

logger = logging.getLogger("logger")


_DETECTOR_EVASION_ATTACK_MODES = {
    "priv_filterfl_detector_evasion_adaptive",
    "priv_filterfl_split_upload_detector_evasion_adaptive",
}

_ADAPTIVE_EVASION_SURROGATE_LEGACY = "legacy_mean_logit_margin_v1"
_ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_V2 = "fixed_tau_dual_hinge_v2"
_ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3 = "fixed_tau_dual_hinge_band_v3"
_PUBLICATION_GRADE_ADAPTIVE_SURROGATES = {
    _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_V2,
    _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3,
}
_ADAPTIVE_EVASION_PROBE_LEGACY_TRAIN = "legacy_train"
_ADAPTIVE_EVASION_PROBE_FROZEN_EVAL = "frozen_eval"
_ADAPTIVE_EVASION_PROBE_DEPLOYED_NOISY_BN_HYBRID = "deployed_noisy_bn_hybrid"
_DETECTOR_PROBE_DIGEST_SCHEMA = "fldd-detector-probe-sha256-v1"
_MODEL_STATE_DIGEST_SCHEMA = "fldd-model-state-sha256-v1"
_BATCH_BN_STATS_DIGEST_SCHEMA = "fldd-batch-bn-stats-sha256-v1"
_HYBRID_REFERENCE_STATE_SOURCE = "split_upload_clean_shadow_client_ldp_noisy"
_HYBRID_REFERENCE_BN_STATS_SOURCE = (
    "previous_public_probe_batch_stats_from_split_upload_clean_shadow_client_ldp_noisy"
)
_HYBRID_TRAINING_ORDER = "clean_shadow_ldp_then_poisoned"
_HYBRID_PROBE_EVAL_SCOPE = (
    "deployed_noisy_reference_batch_bn_stats_then_poisoned_eval_with_fixed_stats_and_gradients"
)
_ADAPTIVE_EVASION_SURROGATE_SCHEMA_V2 = "fldd-adaptive-evasion-surrogate-v2"
_ADAPTIVE_EVASION_SURROGATE_FORMULA_V2 = (
    "CE+lambda_prob*relu(mean_softmax_p_target-(tau-safety_margin))"
    "+lambda_margin*mean_relu(per_sample_target_minus_best_other_margin-guard)"
)
_ADAPTIVE_EVASION_SURROGATE_SCHEMA_V3 = "fldd-adaptive-evasion-surrogate-v3"
_ADAPTIVE_EVASION_SURROGATE_FORMULA_V3 = (
    "CE+lambda_prob*relu(mean_softmax_p_target-(tau-safety_margin))"
    "+lambda_margin*mean(relu(lower_guard-margin)"
    "+relu(margin-upper_guard))"
)


def _detector_probe_digest(trigger):
    """Hash the public probe dtype, shape and contiguous tensor bytes."""
    if not isinstance(trigger, torch.Tensor):
        raise TypeError("detector probe digest requires a torch.Tensor")
    canonical = trigger.detach().to("cpu", dtype=torch.float32)
    if canonical.ndim == 3:
        canonical = canonical.unsqueeze(0)
    canonical = canonical.contiguous()
    digest = hashlib.sha256()
    digest.update((_DETECTOR_PROBE_DIGEST_SCHEMA + "\0").encode("ascii"))
    digest.update((str(canonical.dtype) + "\0").encode("ascii"))
    digest.update((repr(tuple(int(v) for v in canonical.shape)) + "\0").encode("ascii"))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _verify_previous_detection_trigger_digest(
    handle, trigger, require_digest, expected=None, schema=None
):
    """Verify the cached probe against its expected content digest."""
    actual = _detector_probe_digest(trigger)
    if expected is None:
        expected = getattr(handle, "previous_detection_trigger_digest", None)
    if schema is None:
        schema = getattr(handle, "previous_detection_trigger_digest_schema", None)
    if expected is None and schema is None:
        if require_digest:
            return actual, None, False, "missing_previous_trigger_digest"
        return actual, None, None, "ready"
    if schema != _DETECTOR_PROBE_DIGEST_SCHEMA:
        return actual, expected, False, "invalid_previous_trigger_digest_schema"
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(ch not in "0123456789abcdef" for ch in expected)
    ):
        return actual, expected, False, "invalid_previous_trigger_digest"
    if expected != actual:
        return actual, expected, False, "previous_trigger_digest_mismatch"
    return actual, expected, True, "ready"


def _canonical_adaptive_evasion_surrogate(value):
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "legacy": _ADAPTIVE_EVASION_SURROGATE_LEGACY,
        "legacy_logit_margin": _ADAPTIVE_EVASION_SURROGATE_LEGACY,
        _ADAPTIVE_EVASION_SURROGATE_LEGACY: (_ADAPTIVE_EVASION_SURROGATE_LEGACY),
        "dual_hinge_v2": _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_V2,
        _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_V2: (
            _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_V2
        ),
        "dual_hinge_band_v3": (_ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3),
        "fixed_tau_margin_band_v3": (_ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3),
        _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3: (
            _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3
        ),
    }
    if normalized not in aliases:
        raise ValueError(
            f"adaptive_evasion_surrogate must be one of {sorted(aliases)}, got {value!r}"
        )
    return aliases[normalized]


def _required_finite_config_number(params, key):
    if key not in params:
        raise ValueError(f"{key} is required for the configured adaptive surrogate")
    try:
        value = float(params[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def _target_logit_margins(logits, target):
    """Per-sample target-vs-best-other logit margins."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("detector-evasion logits must have shape [batch, classes]")
    target = int(target)
    class_count = int(logits.shape[1])
    if class_count < 2 or target < 0 or target >= class_count:
        raise ValueError(f"detector-evasion target={target} is invalid for {class_count} classes")
    if target == 0:
        other_logits = logits[:, 1:]
    elif target == class_count - 1:
        other_logits = logits[:, :-1]
    else:
        other_logits = torch.cat((logits[:, :target], logits[:, target + 1 :]), dim=1)
    return logits[:, target] - other_logits.max(dim=1).values


def _batchnorm_buffer_snapshot(model):
    snapshot = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(module, buffer_name, None)
            if isinstance(value, torch.Tensor):
                key = f"{module_name}.{buffer_name}" if module_name else buffer_name
                snapshot[key] = value.detach().to("cpu").contiguous().clone()
    return snapshot


def _named_tensor_digest_with_schema(snapshot, schema):
    digest = hashlib.sha256()
    digest.update((str(schema) + "\0").encode("ascii"))
    for name in sorted(snapshot):
        tensor = snapshot[name].detach().to("cpu").contiguous()
        digest.update((name + "\0").encode("utf-8"))
        digest.update((str(tensor.dtype) + "\0").encode("ascii"))
        digest.update((repr(tuple(int(v) for v in tensor.shape)) + "\0").encode("ascii"))
        # Flatten before viewing bytes to support scalar buffers and bfloat16.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _named_tensor_digest(snapshot):
    return _named_tensor_digest_with_schema(snapshot, "fldd-bn-buffer-snapshot-sha256-v1")


def _batchnorm_snapshots_equal(before, after):
    return set(before) == set(after) and all(
        torch.equal(before[name], after[name]) for name in before
    )


def _restore_batchnorm_buffer_snapshot(model, snapshot):
    current = _batchnorm_buffer_snapshot(model)
    if set(current) != set(snapshot):
        raise RuntimeError("BatchNorm buffer schema changed during probe forward")
    modules = dict(model.named_modules())
    with torch.no_grad():
        for key, saved in snapshot.items():
            if "." in key:
                module_name, buffer_name = key.rsplit(".", 1)
            else:
                module_name, buffer_name = "", key
            module = modules.get(module_name)
            value = getattr(module, buffer_name, None) if module is not None else None
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"BatchNorm buffer {key!r} disappeared")
            value.copy_(saved.to(device=value.device, dtype=value.dtype))


def _empty_probe_forward_audit(mode):
    return {
        "probe_forward_mode": mode,
        "probe_eval_scope": None,
        "probe_bn_buffer_count": None,
        "probe_bn_digest_before": None,
        "probe_bn_digest_after": None,
        "probe_bn_buffers_unchanged": None,
        "probe_training_state_restored": None,
        "probe_reference_state_digest_schema": None,
        "probe_reference_state_digest": None,
        "probe_reference_state_tensor_count": None,
        "probe_reference_state_source": None,
        "probe_reference_bn_stats_digest_schema": None,
        "probe_reference_bn_stats_digest": None,
        "probe_reference_bn_stats_source": None,
        "probe_reference_bn_stats_count": None,
        "probe_reference_bn_buffer_count": None,
        "probe_reference_bn_digest_before": None,
        "probe_reference_bn_digest_after": None,
        "probe_reference_bn_buffers_unchanged": None,
        "probe_reference_training_state_restored": None,
        "probe_external_bn_stats_applied_count": None,
        "split_upload_training_order": None,
    }


def _extract_reference_batch_bn_stats(reference_model, trigger_batch):
    """Collect population BN statistics in one train-mode forward pass.

    Restore module modes and BN buffers before returning.
    """

    training_states = [(module, bool(module.training)) for module in reference_model.modules()]
    bn_before = _batchnorm_buffer_snapshot(reference_model)
    digest_before = _named_tensor_digest(bn_before)
    named_bn = {
        name: module
        for name, module in reference_model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    if not named_bn:
        raise ValueError("deployed_noisy_bn_hybrid requires at least one BatchNorm layer")
    stats = {}
    invocations = {name: 0 for name in named_bn}
    hooks = []

    def make_hook(name):
        def hook(_module, inputs, _output):
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"BatchNorm layer {name!r} has an invalid probe input")
            value = inputs[0]
            if value.ndim < 2:
                raise RuntimeError(f"BatchNorm layer {name!r} probe input has rank < 2")
            dims = (0,) + tuple(range(2, value.ndim))
            mean = value.mean(dim=dims)
            var = value.var(dim=dims, unbiased=False)
            if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
                raise RuntimeError(f"BatchNorm layer {name!r} produced non-finite statistics")
            stats[name] = {
                "mean": mean.detach().to("cpu").contiguous().clone(),
                "var": var.detach().to("cpu").contiguous().clone(),
            }
            invocations[name] += 1

        return hook

    for name, module in named_bn.items():
        hooks.append(module.register_forward_hook(make_hook(name)))
    try:
        reference_model.train()
        with torch.no_grad():
            reference_model(trigger_batch)
    finally:
        for hook in hooks:
            hook.remove()
        _restore_batchnorm_buffer_snapshot(reference_model, bn_before)
        for module, was_training in training_states:
            module.training = was_training

    bn_after = _batchnorm_buffer_snapshot(reference_model)
    digest_after = _named_tensor_digest(bn_after)
    unchanged = _batchnorm_snapshots_equal(bn_before, bn_after)
    restored = all(
        bool(module.training) == was_training for module, was_training in training_states
    )
    if set(stats) != set(named_bn) or any(invocations[name] != 1 for name in named_bn):
        raise RuntimeError("reference probe did not visit every BatchNorm layer exactly once")
    if not unchanged:
        raise RuntimeError("reference BatchNorm buffers were not restored")
    if not restored:
        raise RuntimeError("reference module training state was not restored")
    return stats, {
        "probe_reference_bn_buffer_count": int(len(bn_before)),
        "probe_reference_bn_digest_before": digest_before,
        "probe_reference_bn_digest_after": digest_after,
        "probe_reference_bn_buffers_unchanged": True,
        "probe_reference_training_state_restored": True,
    }


def _prepare_deployed_noisy_bn_hybrid(model, noisy_reference_state, trigger_batch):
    if not isinstance(noisy_reference_state, Mapping) or not noisy_reference_state:
        raise ValueError(
            "deployed_noisy_bn_hybrid requires an explicit non-empty noisy reference state"
        )
    reference_state = {}
    for name, value in noisy_reference_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("noisy reference state must map string names to tensors")
        canonical = value.detach().to("cpu").contiguous()
        if canonical.is_floating_point() and not torch.isfinite(canonical).all():
            raise ValueError(f"noisy reference tensor {name!r} is non-finite")
        reference_state[name] = canonical.clone()
    state_digest = _named_tensor_digest_with_schema(reference_state, _MODEL_STATE_DIGEST_SCHEMA)

    reference_model = copy.deepcopy(model)
    reference_model.load_state_dict(reference_state, strict=True)
    try:
        parameter = next(reference_model.parameters())
        reference_trigger = trigger_batch.to(device=parameter.device, dtype=parameter.dtype)
    except StopIteration as exc:
        raise ValueError("noisy reference model has no parameters") from exc
    stats, reference_audit = _extract_reference_batch_bn_stats(reference_model, reference_trigger)
    current_bn_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    if set(stats) != current_bn_names:
        raise RuntimeError("noisy-reference and poisoned-model BatchNorm layer sets differ")
    flattened_stats = {
        f"{name}.{field}": tensor
        for name, values in stats.items()
        for field, tensor in values.items()
    }
    stats_digest = _named_tensor_digest_with_schema(flattened_stats, _BATCH_BN_STATS_DIGEST_SCHEMA)
    return {
        "bn_stats": stats,
        "audit": {
            "probe_reference_state_digest_schema": _MODEL_STATE_DIGEST_SCHEMA,
            "probe_reference_state_digest": state_digest,
            "probe_reference_state_tensor_count": int(len(reference_state)),
            "probe_reference_state_source": _HYBRID_REFERENCE_STATE_SOURCE,
            "probe_reference_bn_stats_digest_schema": _BATCH_BN_STATS_DIGEST_SCHEMA,
            "probe_reference_bn_stats_digest": stats_digest,
            "probe_reference_bn_stats_source": _HYBRID_REFERENCE_BN_STATS_SOURCE,
            "probe_reference_bn_stats_count": int(len(stats)),
            "split_upload_training_order": _HYBRID_TRAINING_ORDER,
            **reference_audit,
        },
    }


def _adaptive_probe_forward(model, trigger_batch, mode, *, hybrid_context=None):
    """Evaluate the public probe with the selected BN statistics."""
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == _ADAPTIVE_EVASION_PROBE_LEGACY_TRAIN:
        return model(trigger_batch), _empty_probe_forward_audit(normalized)
    if normalized not in {
        _ADAPTIVE_EVASION_PROBE_FROZEN_EVAL,
        _ADAPTIVE_EVASION_PROBE_DEPLOYED_NOISY_BN_HYBRID,
    }:
        raise ValueError(
            "adaptive_evasion_probe_forward_mode must be legacy_train or "
            "frozen_eval or deployed_noisy_bn_hybrid, "
            f"got {mode!r}"
        )

    training_states = [(module, bool(module.training)) for module in model.modules()]
    bn_before = _batchnorm_buffer_snapshot(model)
    digest_before = _named_tensor_digest(bn_before)
    audit = _empty_probe_forward_audit(normalized)
    audit.update(
        {
            "probe_eval_scope": (
                "full_model_eval_with_gradients"
                if normalized == _ADAPTIVE_EVASION_PROBE_FROZEN_EVAL
                else _HYBRID_PROBE_EVAL_SCOPE
            ),
            "probe_bn_buffer_count": int(len(bn_before)),
            "probe_bn_digest_before": digest_before,
        }
    )
    patched_forwards = []
    logits = None
    bn_after = None
    try:
        model.eval()
        if normalized == _ADAPTIVE_EVASION_PROBE_DEPLOYED_NOISY_BN_HYBRID:
            if not isinstance(hybrid_context, dict):
                raise ValueError("deployed_noisy_bn_hybrid context is unavailable")
            stats = hybrid_context.get("bn_stats")
            context_audit = hybrid_context.get("audit")
            if not isinstance(stats, dict) or not isinstance(context_audit, dict):
                raise ValueError("deployed_noisy_bn_hybrid context is malformed")
            named_bn = {
                name: module
                for name, module in model.named_modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            }
            if set(stats) != set(named_bn) or not named_bn:
                raise RuntimeError(
                    "external BatchNorm statistics do not exactly cover the poisoned model"
                )
            for name, module in named_bn.items():
                values = stats[name]
                mean = values.get("mean") if isinstance(values, dict) else None
                var = values.get("var") if isinstance(values, dict) else None
                if not isinstance(mean, torch.Tensor) or not isinstance(var, torch.Tensor):
                    raise TypeError(f"external BatchNorm statistics for {name!r} are malformed")
                if (
                    mean.ndim != 1
                    or var.shape != mean.shape
                    or int(mean.numel()) != int(module.num_features)
                ):
                    raise ValueError(
                        f"external BatchNorm statistics for {name!r} have the wrong shape"
                    )
                had_instance_forward = "forward" in module.__dict__
                prior_instance_forward = module.__dict__.get("forward")

                def external_forward(this, value, *, fixed_mean=mean, fixed_var=var):
                    return torch.nn.functional.batch_norm(
                        value,
                        fixed_mean.to(device=value.device, dtype=value.dtype),
                        fixed_var.to(device=value.device, dtype=value.dtype),
                        this.weight,
                        this.bias,
                        training=False,
                        momentum=0.0,
                        eps=this.eps,
                    )

                module.forward = types.MethodType(external_forward, module)
                patched_forwards.append((module, had_instance_forward, prior_instance_forward))
            audit.update(context_audit)
            audit["probe_external_bn_stats_applied_count"] = int(len(named_bn))
        logits = model(trigger_batch)
        bn_after = _batchnorm_buffer_snapshot(model)
    finally:
        for module, had_instance_forward, prior_instance_forward in reversed(patched_forwards):
            if had_instance_forward:
                module.forward = prior_instance_forward
            else:
                del module.forward
        for module, was_training in training_states:
            module.training = was_training
    if bn_after is None:
        bn_after = _batchnorm_buffer_snapshot(model)
    digest_after = _named_tensor_digest(bn_after)
    unchanged = _batchnorm_snapshots_equal(bn_before, bn_after)
    restored = all(
        bool(module.training) == was_training for module, was_training in training_states
    )
    audit.update(
        {
            "probe_bn_digest_after": digest_after,
            "probe_bn_buffers_unchanged": bool(unchanged),
            "probe_training_state_restored": bool(restored),
        }
    )
    return logits, audit


def _validate_probe_tensor(trigger):
    """Validate the probe shape and finite values."""
    if not isinstance(trigger, torch.Tensor):
        return None, "invalid_previous_trigger_type"
    trigger = trigger.detach().to("cpu")
    if trigger.ndim == 3:
        trigger = trigger.unsqueeze(0)
    if trigger.ndim != 4 or trigger.shape[0] == 0:
        return None, "invalid_previous_trigger_shape"
    if not trigger.is_floating_point():
        trigger = trigger.float()
    if not torch.isfinite(trigger).all():
        return None, "nonfinite_previous_trigger"
    return trigger.contiguous(), "ready"


def _prepare_previous_detection_trigger(handle, epoch, staleness=1):
    """Return the public probe from round ``epoch - staleness`` and its digest.

    A missing round returns a status describing why the probe is unavailable.
    """
    staleness = int(staleness)
    target_round = int(epoch) - staleness
    if staleness == 1:
        trigger = getattr(handle, "previous_detection_trigger", None)
        if trigger is None:
            return None, "missing_previous_trigger", None, None
        trigger_round = getattr(handle, "previous_detection_trigger_round", None)
        expected_digest = getattr(handle, "previous_detection_trigger_digest", None)
        expected_schema = getattr(handle, "previous_detection_trigger_digest_schema", None)
    else:
        ring = getattr(handle, "detector_trigger_ring", None)
        entry = None
        if ring is not None:
            for item in ring:
                if int(item.get("round", -1)) == target_round:
                    entry = item
                    break
        if entry is None:
            return None, "stale_previous_trigger_unavailable", None, None
        trigger = entry.get("trigger")
        trigger_round = entry.get("round")
        expected_digest = entry.get("digest")
        expected_schema = entry.get("digest_schema")
    if trigger_round is None:
        return None, "missing_previous_trigger_round", None, None
    if isinstance(trigger_round, bool) or not isinstance(trigger_round, int):
        return None, "invalid_previous_trigger_round", None, None
    if trigger_round != target_round:
        return None, "stale_previous_trigger_round", None, None
    validated, status = _validate_probe_tensor(trigger)
    if status != "ready":
        return None, status, None, None
    return validated, "ready", expected_digest, expected_schema


def _prepare_self_synth_trigger(handle, epoch):
    """Return the attacker-synthesized probe and its bound content digest."""
    status_flag = getattr(handle, "attacker_self_synth_trigger_status", None)
    trigger = getattr(handle, "attacker_self_synth_trigger", None)
    if trigger is None:
        reason = (
            status_flag
            if isinstance(status_flag, str) and status_flag not in ("ready", "inactive")
            else "self_synth_trigger_unavailable"
        )
        return None, reason, None, None
    trigger_round = getattr(handle, "attacker_self_synth_trigger_round", None)
    if trigger_round != int(epoch):
        return None, "self_synth_trigger_round_mismatch", None, None
    validated, status = _validate_probe_tensor(trigger)
    if status != "ready":
        return None, status, None, None
    expected_digest = getattr(handle, "attacker_self_synth_trigger_digest", None)
    expected_schema = getattr(handle, "attacker_self_synth_trigger_digest_schema", None)
    return validated, "ready", expected_digest, expected_schema


def _target_logit_margin(logits, target):
    """Mean target-vs-best-other logit margin used by the final detector."""
    return _target_logit_margins(logits, target).mean()


def _fixed_tau_dual_hinge_surrogate(
    logits,
    target,
    *,
    tau,
    safety_margin,
    logit_margin_guard,
):
    """Compute probability and per-sample logit-margin hinge losses.

    The probability term pushes the mean response below ``tau-safety_margin``;
    the margin term pushes each response below the configured guard.
    """
    margins = _target_logit_margins(logits, target)
    target = int(target)
    p_target = torch.softmax(logits, dim=1)[:, target]
    p_target_mean = p_target.mean()
    probability_boundary = float(tau) - float(safety_margin)
    probability_hinge = torch.relu(p_target_mean - probability_boundary)
    margin_excess = torch.relu(margins - float(logit_margin_guard))
    margin_hinge = margin_excess.mean()
    return {
        "p_target": p_target,
        "p_target_mean": p_target_mean,
        "probability_boundary": probability_boundary,
        "probability_hinge": probability_hinge,
        "probability_hinge_active_fraction": (p_target_mean > probability_boundary).to(
            dtype=logits.dtype
        ),
        "margins": margins,
        "margin_hinge_sum": margin_excess.sum(),
        "margin_hinge": margin_hinge,
        "margin_active_count": (margins > float(logit_margin_guard)).sum(),
        "margin_active_fraction": (margins > float(logit_margin_guard))
        .to(dtype=logits.dtype)
        .mean(),
    }


def _fixed_tau_dual_hinge_band_surrogate(
    logits,
    target,
    *,
    tau,
    safety_margin,
    logit_margin_lower_guard,
    logit_margin_upper_guard,
):
    """Compute a probability hinge and a two-sided logit-margin band loss."""

    lower_guard = float(logit_margin_lower_guard)
    upper_guard = float(logit_margin_upper_guard)
    if not lower_guard < upper_guard:
        raise ValueError("logit-margin band requires lower_guard < upper_guard")
    margins = _target_logit_margins(logits, target)
    target = int(target)
    p_target = torch.softmax(logits, dim=1)[:, target]
    p_target_mean = p_target.mean()
    probability_boundary = float(tau) - float(safety_margin)
    probability_hinge = torch.relu(p_target_mean - probability_boundary)
    lower_excess = torch.relu(lower_guard - margins)
    upper_excess = torch.relu(margins - upper_guard)
    band_excess = lower_excess + upper_excess
    outside = (margins < lower_guard) | (margins > upper_guard)
    within = (margins >= lower_guard) & (margins <= upper_guard)
    return {
        "p_target": p_target,
        "p_target_mean": p_target_mean,
        "probability_boundary": probability_boundary,
        "probability_hinge": probability_hinge,
        "probability_hinge_active_fraction": (p_target_mean > probability_boundary).to(
            dtype=logits.dtype
        ),
        "margins": margins,
        "margin_hinge_sum": band_excess.sum(),
        "margin_hinge": band_excess.mean(),
        "margin_active_count": outside.sum(),
        "margin_active_fraction": outside.to(dtype=logits.dtype).mean(),
        "margin_lower_hinge_sum": lower_excess.sum(),
        "margin_lower_hinge": lower_excess.mean(),
        "margin_lower_active_count": (margins < lower_guard).sum(),
        "margin_lower_active_fraction": (margins < lower_guard).to(dtype=logits.dtype).mean(),
        "margin_upper_hinge_sum": upper_excess.sum(),
        "margin_upper_hinge": upper_excess.mean(),
        "margin_upper_active_count": (margins > upper_guard).sum(),
        "margin_upper_active_fraction": (margins > upper_guard).to(dtype=logits.dtype).mean(),
        "margin_within_band_count": within.sum(),
        "margin_within_band_fraction": within.to(dtype=logits.dtype).mean(),
    }


def _append_adaptive_evasion_record(handle, record):
    records = getattr(handle, "adaptive_evasion_records", None)
    if records is None:
        records = []
        handle.adaptive_evasion_records = records
    records.append(record)


def _adaptive_evasion_fail_closed(params):
    """Determine whether unavailable attack prerequisites stop local training."""
    campaign_id = params.get("campaign_id")
    campaign_mode = campaign_id is not None and bool(str(campaign_id).strip())
    return campaign_mode or bool(params.get("adaptive_evasion_fail_closed", False))


def _raise_adaptive_evasion_fail_closed(epoch, client, status):
    raise RuntimeError(
        "[adaptive_attack][detector_evasion] fail-closed: "
        f"round={int(epoch)}, client={int(client)}, status={status}"
    )


def _lockdown_is_maskable_key(name: str, tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if not tensor.is_floating_point():
        return False
    if ("running_mean" in name) or ("running_var" in name) or ("num_batches_tracked" in name):
        return False
    return True


def _lockdown_init_mask(state_dict, config, reference_mask=None):
    density = config.get("lockdown_density", None)
    if density is None:
        sparsity = float(config.get("lockdown_sparsity", 0.5))
        density = max(0.0, min(1.0, 1.0 - sparsity))
    density = max(0.0, min(1.0, float(density)))
    init_mode = str(config.get("lockdown_mask_init", "random")).lower()

    mask = {}
    for name, tensor in state_dict.items():
        if not _lockdown_is_maskable_key(name, tensor):
            mask[name] = torch.ones_like(tensor, dtype=torch.float32)
            continue
        flat = tensor.detach().view(-1)
        numel = flat.numel()
        if numel == 0:
            mask[name] = torch.ones_like(tensor, dtype=torch.float32)
            continue
        k = max(1, int(round(numel * density)))
        if init_mode == "magnitude":
            _, idx = torch.topk(flat.abs(), k, largest=True)
        else:
            idx = torch.randperm(numel)[:k]
        flat_mask = torch.zeros(numel, dtype=torch.float32)
        flat_mask[idx] = 1.0
        mask[name] = flat_mask.view_as(tensor)

    if reference_mask is not None:
        aligned = {}
        for name, tensor in state_dict.items():
            if name in reference_mask:
                aligned[name] = reference_mask[name].detach().clone()
            else:
                aligned[name] = torch.ones_like(tensor, dtype=torch.float32)
        mask = aligned
    return mask


def _lockdown_screen_gradients(model, data_iterator, max_batches, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    gradient = {
        name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()
    }
    for batch_idx, (images, labels) in enumerate(data_iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        model.zero_grad()
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            gradient[name] += param.grad.detach()
    return gradient


def _lockdown_update_mask(mask, num_remove, gradient, use_random=False, device=None):
    updated = {}
    for name, m in mask.items():
        if name not in gradient or name not in num_remove:
            updated[name] = m
            continue
        nr = int(num_remove[name])
        if nr <= 0:
            updated[name] = m
            continue
        m_dev = m.to(device) if device is not None else m
        g = gradient[name]
        if use_random:
            available = (m_dev == 0).view(-1)
            idx_pool = torch.nonzero(available, as_tuple=False).view(-1)
            if idx_pool.numel() == 0:
                updated[name] = m
                continue
            pick = idx_pool[torch.randperm(idx_pool.numel())[:nr]]
            m_flat = m_dev.view(-1)
            m_flat[pick] = 1.0
            updated[name] = m_flat.view_as(m_dev).detach().cpu()
        else:
            temp = torch.where(m_dev == 0, g.abs(), -1e6 * torch.ones_like(g))
            _, idx = torch.topk(temp.view(-1), k=min(nr, temp.numel()), largest=True)
            m_flat = m_dev.view(-1)
            m_flat[idx] = 1.0
            updated[name] = m_flat.view_as(m_dev).detach().cpu()
    return updated


def _lockdown_fire_mask(weights, mask, round_idx, total_rounds, anneal_factor, device=None):
    updated = {}
    num_remove = {}
    total_rounds = max(1, int(total_rounds))
    drop_ratio = float(anneal_factor) / 2.0 * (1.0 + math.cos((round_idx * math.pi) / total_rounds))
    for name, m in mask.items():
        w = weights.get(name)
        if w is None or not _lockdown_is_maskable_key(name, w):
            updated[name] = m.detach().cpu()
            num_remove[name] = 0
            continue
        m_dev = m.to(device) if device is not None else m
        active = (m_dev > 0).sum().item()
        nr = int(math.ceil(drop_ratio * active))
        num_remove[name] = nr
        if nr <= 0:
            updated[name] = m_dev.detach().cpu()
            continue
        temp_w = torch.where(m_dev > 0, w.abs(), 1e6 * torch.ones_like(w))
        _, idx = torch.topk(-temp_w.view(-1), k=min(nr, temp_w.numel()), largest=True)
        m_flat = m_dev.view(-1)
        m_flat[idx] = 0.0
        updated[name] = m_flat.view_as(m_dev).detach().cpu()
    return updated, num_remove


def add_pixel_pattern(ori_image, params, index, handle):
    return apply_pixel_patterns(ori_image, params, [index], handle.target)


def standard_train(
    epoch,
    dataset_size,
    client,
    params,
    model,
    dataset,
    handle,
    poison=False,
    enable_ldp=False,
    lockdown_state=None,
    lockdown_config=None,
    force_cpu_data=False,
    adaptive_evasion_noisy_reference=None,
):
    dataset_size = int(dataset_size)
    if dataset_size < 0:
        raise ValueError(f"client {client} has invalid negative dataset_size={dataset_size}")
    empty_client = dataset_size == 0
    if empty_client:
        empty_policy = str(params.get("empty_client_policy", "error")).strip().lower()
        if empty_policy != "no_op":
            raise RuntimeError(
                f"client {client} has zero samples; set empty_client_policy=no_op "
                "to preserve its selected slot with an unchanged clean model"
            )
        records = getattr(handle, "empty_client_train_records", None)
        if records is None:
            records = []
            handle.empty_client_train_records = records
        records.append(
            {
                "round": int(epoch),
                "client_id": int(client),
                "poison": bool(poison),
                "enable_ldp": bool(enable_ldp),
                "action": "no_op_local_update",
            }
        )
        main.logger.warning(
            f"[empty_client] round={epoch} client={client} has zero samples; "
            "preserve the selected-client slot and use an unchanged clean model"
        )
    try:
        model = model.to(device)
    except Exception:
        pass
    model.train()
    index = -1
    local_epoch = params["local_epochs"]
    lr = params["lr"]
    attack_mode = str(params.get("attack_mode", params.get("attack.mode", ""))).strip().lower()
    effective_poison_rate = float(params.get("poison_rate", 0.0))
    detector_evasion_enabled = poison is True and attack_mode in _DETECTOR_EVASION_ATTACK_MODES
    evasion_surrogate = _ADAPTIVE_EVASION_SURROGATE_LEGACY
    evasion_lambda = 0.0
    evasion_kappa = 0.0
    evasion_probe_forward_mode = _ADAPTIVE_EVASION_PROBE_LEGACY_TRAIN
    evasion_require_probe_digest = False
    evasion_probe_digest = None
    evasion_cached_probe_digest = None
    evasion_probe_digest_verified = None
    evasion_fixed_tau = None
    evasion_fixed_tau_safety_margin = None
    evasion_fixed_tau_boundary = None
    evasion_fixed_tau_lambda = None
    evasion_logit_margin_guard = None
    evasion_logit_margin_lower_guard = None
    evasion_logit_margin_lambda = None
    evasion_hybrid_context = None
    evasion_start_round = 2
    previous_trigger = None
    evasion_readiness = "not_detector_evasion_attack"
    evasion_trigger_round = None
    evasion_expected_trigger_round = None
    evasion_trigger_source = (
        str(
            params.get(
                "adaptive_evasion_trigger_source",
                "previous_public_detector_trigger",
            )
        )
        .strip()
        .lower()
    )
    evasion_trigger_staleness = int(params.get("adaptive_evasion_trigger_staleness", 1))
    evasion_fail_closed = detector_evasion_enabled and _adaptive_evasion_fail_closed(params)
    evasion_strict = evasion_fail_closed
    if poison is True:
        local_epoch = params["poison_epochs"]
        lr = params["poison_lr"]
        index = resolve_client_pattern_index(params, client, handle)
        if attack_mode == "filterfl_low_poison_rate_adaptive":
            effective_poison_rate = float(
                params.get("adaptive_poison_rate", params.get("poison_rate", 0.0))
            )
            main.logger.info(
                f"[adaptive_attack] mode=filterfl_low_poison_rate_adaptive, client={client}, poison_rate={effective_poison_rate}, poison_epochs={local_epoch}, poison_lr={lr}"
            )
        elif detector_evasion_enabled:
            evasion_surrogate = _canonical_adaptive_evasion_surrogate(
                params.get(
                    "adaptive_evasion_surrogate",
                    _ADAPTIVE_EVASION_SURROGATE_LEGACY,
                )
            )
            evasion_strict = (
                evasion_fail_closed or evasion_surrogate in _PUBLICATION_GRADE_ADAPTIVE_SURROGATES
            )
            evasion_start_round = int(params.get("adaptive_evasion_start_round", 2))
            if evasion_start_round < 1:
                raise ValueError(
                    f"adaptive_evasion_start_round must be >= 1, got {evasion_start_round}"
                )
            if evasion_surrogate == _ADAPTIVE_EVASION_SURROGATE_LEGACY:
                evasion_lambda = float(params.get("adaptive_evasion_lambda", 1.0))
                evasion_kappa = float(params.get("adaptive_evasion_kappa", 0.0))
                if not math.isfinite(evasion_lambda) or evasion_lambda < 0.0:
                    raise ValueError(
                        "adaptive_evasion_lambda must be finite and non-negative, "
                        f"got {evasion_lambda}"
                    )
                if not math.isfinite(evasion_kappa):
                    raise ValueError(f"adaptive_evasion_kappa must be finite, got {evasion_kappa}")
            else:
                split_upload_v2 = (
                    attack_mode == "priv_filterfl_split_upload_detector_evasion_adaptive"
                )
                expected_probe_forward_mode = (
                    _ADAPTIVE_EVASION_PROBE_DEPLOYED_NOISY_BN_HYBRID
                    if split_upload_v2
                    else _ADAPTIVE_EVASION_PROBE_FROZEN_EVAL
                )
                evasion_probe_forward_mode = (
                    str(
                        params.get(
                            "adaptive_evasion_probe_forward_mode",
                            expected_probe_forward_mode,
                        )
                    )
                    .strip()
                    .lower()
                    .replace("-", "_")
                )
                if evasion_probe_forward_mode != expected_probe_forward_mode:
                    raise ValueError(
                        f"{evasion_surrogate} requires adaptive_evasion_"
                        f"probe_forward_mode={expected_probe_forward_mode} for "
                        f"attack_mode={attack_mode}"
                    )
                if split_upload_v2:
                    if adaptive_evasion_noisy_reference is None:
                        raise ValueError(
                            "deployed_noisy_bn_hybrid requires the exact split-upload "
                            "clean-shadow LDP noisy reference"
                        )
                    if bool(params.get("bn_folding_use_running_stats", False)):
                        raise ValueError(
                            "deployed_noisy_bn_hybrid requires bn_folding_use_running_stats=false"
                        )
                    if params.get("cryptgpu_bn_use_noisy") is not True:
                        raise ValueError(
                            "deployed_noisy_bn_hybrid requires cryptgpu_bn_use_noisy=true"
                        )
                    stats_source = (
                        str(params.get("dp_cbf_stats_source", "per_client_noisy_model"))
                        .strip()
                        .lower()
                        .replace("-", "_")
                    )
                    if stats_source != "per_client_noisy_model":
                        raise ValueError(
                            "deployed_noisy_bn_hybrid requires "
                            "dp_cbf_stats_source=per_client_noisy_model"
                        )
                require_digest_value = params.get("adaptive_evasion_require_probe_digest", True)
                if type(require_digest_value) is not bool:
                    raise ValueError("adaptive_evasion_require_probe_digest must be a boolean")
                evasion_require_probe_digest = require_digest_value
                if not evasion_require_probe_digest:
                    raise ValueError(
                        f"{evasion_surrogate} requires adaptive_evasion_require_probe_digest=true"
                    )
                evasion_fixed_tau = _required_finite_config_number(
                    params, "adaptive_evasion_fixed_tau"
                )
                evasion_fixed_tau_safety_margin = _required_finite_config_number(
                    params, "adaptive_evasion_fixed_tau_safety_margin"
                )
                evasion_fixed_tau_lambda = _required_finite_config_number(
                    params, "adaptive_evasion_fixed_tau_lambda"
                )
                evasion_logit_margin_guard = _required_finite_config_number(
                    params, "adaptive_evasion_logit_margin_guard"
                )
                evasion_logit_margin_lambda = _required_finite_config_number(
                    params, "adaptive_evasion_logit_margin_lambda"
                )
                if evasion_surrogate == _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3:
                    evasion_logit_margin_lower_guard = _required_finite_config_number(
                        params,
                        "adaptive_evasion_logit_margin_lower_guard",
                    )
                    if not (evasion_logit_margin_lower_guard < evasion_logit_margin_guard):
                        raise ValueError(
                            "fixed_tau_dual_hinge_band_v3 requires "
                            "adaptive_evasion_logit_margin_lower_guard < "
                            "adaptive_evasion_logit_margin_guard"
                        )
                if not 0.0 < evasion_fixed_tau < 1.0:
                    raise ValueError("adaptive_evasion_fixed_tau must be in (0, 1)")
                if (
                    evasion_fixed_tau_safety_margin < 0.0
                    or evasion_fixed_tau_safety_margin >= evasion_fixed_tau
                ):
                    raise ValueError(
                        "adaptive_evasion_fixed_tau_safety_margin must be in "
                        "[0, adaptive_evasion_fixed_tau)"
                    )
                if evasion_fixed_tau_lambda <= 0.0:
                    raise ValueError("adaptive_evasion_fixed_tau_lambda must be positive")
                if evasion_logit_margin_lambda <= 0.0:
                    raise ValueError("adaptive_evasion_logit_margin_lambda must be positive")
                evasion_fixed_tau_boundary = evasion_fixed_tau - evasion_fixed_tau_safety_margin
            if int(epoch) < evasion_start_round:
                evasion_readiness = "before_start_round"
            elif evasion_surrogate == _ADAPTIVE_EVASION_SURROGATE_LEGACY and evasion_lambda == 0.0:
                evasion_readiness = "zero_evasion_lambda"
                if evasion_strict:
                    _raise_adaptive_evasion_fail_closed(epoch, client, evasion_readiness)
            elif evasion_trigger_source == "self_synth_recent_published_globals":
                # Bind this round's independently synthesized attacker probe.
                evasion_expected_trigger_round = int(epoch)
                (
                    previous_trigger,
                    evasion_readiness,
                    expected_probe_digest,
                    expected_probe_digest_schema,
                ) = _prepare_self_synth_trigger(handle, epoch)
                if evasion_readiness == "ready":
                    (
                        evasion_probe_digest,
                        evasion_cached_probe_digest,
                        evasion_probe_digest_verified,
                        digest_status,
                    ) = _verify_previous_detection_trigger_digest(
                        handle,
                        previous_trigger,
                        evasion_require_probe_digest,
                        expected=expected_probe_digest,
                        schema=expected_probe_digest_schema,
                    )
                    if digest_status != "ready":
                        evasion_readiness = digest_status
                evasion_trigger_round = (
                    evasion_expected_trigger_round
                    if evasion_readiness == "ready"
                    else getattr(handle, "attacker_self_synth_trigger_round", None)
                )
                if evasion_strict and evasion_readiness != "ready":
                    _raise_adaptive_evasion_fail_closed(epoch, client, evasion_readiness)
            else:
                # Use the published probe from the specified prior round.
                evasion_expected_trigger_round = int(epoch) - evasion_trigger_staleness
                (
                    previous_trigger,
                    evasion_readiness,
                    expected_probe_digest,
                    expected_probe_digest_schema,
                ) = _prepare_previous_detection_trigger(handle, epoch, evasion_trigger_staleness)
                if evasion_readiness == "ready":
                    (
                        evasion_probe_digest,
                        evasion_cached_probe_digest,
                        evasion_probe_digest_verified,
                        digest_status,
                    ) = _verify_previous_detection_trigger_digest(
                        handle,
                        previous_trigger,
                        evasion_require_probe_digest,
                        expected=expected_probe_digest,
                        schema=expected_probe_digest_schema,
                    )
                    if digest_status != "ready":
                        evasion_readiness = digest_status
                evasion_trigger_round = (
                    evasion_expected_trigger_round
                    if evasion_readiness == "ready"
                    else getattr(handle, "previous_detection_trigger_round", None)
                )
                if evasion_strict and evasion_readiness != "ready":
                    _raise_adaptive_evasion_fail_closed(epoch, client, evasion_readiness)
            main.logger.info(
                f"[adaptive_attack][detector_evasion] round={epoch}, client={client}, "
                f"status={evasion_readiness}, surrogate={evasion_surrogate}, "
                f"probe_forward={evasion_probe_forward_mode}, "
                f"start_round={evasion_start_round}; "
                "uses previous-round trigger only"
            )
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=params["momentum"], weight_decay=params["decay"]
    )

    ldp_instance = None
    if enable_ldp:
        ldp_instance = create_ldp_instance(params)

    # The adaptive shadow model uses unpoisoned DataLoader batches.
    use_gpu_preload = (
        params.get("gpu_preload", False)
        and hasattr(handle, "gpu_train_data")
        and not force_cpu_data
    )
    if use_gpu_preload:
        data_iterator = handle.get_gpu_train_data(client)
    else:
        _, data_iterator = dataset
    # Measure updates from the shared starting model.
    initial_state = copy.deepcopy(model.state_dict())

    lockdown_enabled = lockdown_state is not None
    lockdown_mask = None
    if lockdown_enabled:
        lockdown_config = lockdown_config or {}
        lockdown_mask = lockdown_state.get("mask")
        if lockdown_mask is None:
            same_mask = bool(lockdown_config.get("lockdown_same_mask", False))
            if same_mask and hasattr(handle, "lockdown_initial_mask"):
                lockdown_mask = copy.deepcopy(handle.lockdown_initial_mask)
            else:
                lockdown_mask = _lockdown_init_mask(model.state_dict(), lockdown_config)
                if same_mask:
                    handle.lockdown_initial_mask = copy.deepcopy(lockdown_mask)
            lockdown_state["mask"] = lockdown_mask

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in lockdown_mask:
                    param.data = param.data * lockdown_mask[name].to(param.device)

        # Regrow mask based on gradients if requested.
        num_remove = lockdown_state.get("num_remove")
        if num_remove:
            max_batches = lockdown_config.get("lockdown_grad_batches", 1)
            grad = _lockdown_screen_gradients(model, data_iterator, max_batches, device)
            lockdown_mask = _lockdown_update_mask(
                lockdown_mask,
                num_remove,
                grad,
                use_random=bool(lockdown_config.get("lockdown_regrow_random", False)),
                device=device,
            )
            lockdown_state["mask"] = lockdown_mask

    for i in range(local_epoch):
        Predict_loss = 0
        correct = 0
        batches_seen = 0
        for batch_idx, (images, labels) in enumerate(data_iterator):
            batches_seen += 1
            # Preloaded malicious batches already contain their poison pattern.
            if poison is True and not use_gpu_preload:
                images, labels = poison_batch(
                    images,
                    labels,
                    params,
                    client,
                    handle,
                    poison_rate=effective_poison_rate,
                )

            if not use_gpu_preload:
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model(images)

            ce_loss = loss_func(log_probs, labels)
            loss = ce_loss
            evasion_margin_value = None
            evasion_hinge_value = None
            v2_values = {}
            probe_forward_audit = _empty_probe_forward_audit(evasion_probe_forward_mode)
            batch_evasion_status = evasion_readiness
            if detector_evasion_enabled and evasion_readiness == "ready":
                if tuple(previous_trigger.shape[1:]) != tuple(images.shape[1:]):
                    batch_evasion_status = "previous_trigger_shape_mismatch"
                    if evasion_strict:
                        _raise_adaptive_evasion_fail_closed(epoch, client, batch_evasion_status)
                else:
                    trigger_batch = previous_trigger.to(
                        device=images.device,
                        dtype=images.dtype,
                        non_blocking=True,
                    )
                    if (
                        evasion_probe_forward_mode
                        == _ADAPTIVE_EVASION_PROBE_DEPLOYED_NOISY_BN_HYBRID
                        and evasion_hybrid_context is None
                    ):
                        try:
                            evasion_hybrid_context = _prepare_deployed_noisy_bn_hybrid(
                                model,
                                adaptive_evasion_noisy_reference,
                                trigger_batch,
                            )
                        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                            batch_evasion_status = f"invalid_deployed_noisy_bn_hybrid:{exc}"
                            if evasion_strict:
                                _raise_adaptive_evasion_fail_closed(
                                    epoch, client, batch_evasion_status
                                )
                    trigger_logits, probe_forward_audit = _adaptive_probe_forward(
                        model,
                        trigger_batch,
                        evasion_probe_forward_mode,
                        hybrid_context=evasion_hybrid_context,
                    )
                    if not isinstance(trigger_logits, torch.Tensor):
                        batch_evasion_status = "invalid_trigger_logits_type"
                        if evasion_strict:
                            _raise_adaptive_evasion_fail_closed(epoch, client, batch_evasion_status)
                    elif not torch.isfinite(trigger_logits).all():
                        batch_evasion_status = "nonfinite_trigger_logits"
                        if evasion_strict:
                            _raise_adaptive_evasion_fail_closed(epoch, client, batch_evasion_status)
                    elif (
                        evasion_surrogate in _PUBLICATION_GRADE_ADAPTIVE_SURROGATES
                        and probe_forward_audit["probe_bn_buffers_unchanged"] is not True
                    ):
                        batch_evasion_status = "probe_bn_buffers_changed"
                        if evasion_strict:
                            _raise_adaptive_evasion_fail_closed(epoch, client, batch_evasion_status)
                    elif (
                        evasion_surrogate in _PUBLICATION_GRADE_ADAPTIVE_SURROGATES
                        and probe_forward_audit["probe_training_state_restored"] is not True
                    ):
                        batch_evasion_status = "probe_training_state_not_restored"
                        if evasion_strict:
                            _raise_adaptive_evasion_fail_closed(epoch, client, batch_evasion_status)
                    elif evasion_surrogate == _ADAPTIVE_EVASION_SURROGATE_LEGACY:
                        try:
                            evasion_margin = _target_logit_margin(
                                trigger_logits, getattr(handle, "target", None)
                            )
                        except (TypeError, ValueError) as exc:
                            batch_evasion_status = f"invalid_target_or_logits:{exc}"
                            if evasion_strict:
                                _raise_adaptive_evasion_fail_closed(
                                    epoch, client, batch_evasion_status
                                )
                        else:
                            if not torch.isfinite(evasion_margin).all():
                                batch_evasion_status = "nonfinite_evasion_margin"
                                if evasion_strict:
                                    _raise_adaptive_evasion_fail_closed(
                                        epoch, client, batch_evasion_status
                                    )
                            else:
                                evasion_hinge = torch.relu(evasion_margin - evasion_kappa)
                                candidate_loss = ce_loss + evasion_lambda * evasion_hinge
                                if not torch.isfinite(evasion_hinge).all():
                                    batch_evasion_status = "nonfinite_evasion_hinge"
                                    if evasion_strict:
                                        _raise_adaptive_evasion_fail_closed(
                                            epoch, client, batch_evasion_status
                                        )
                                elif not torch.isfinite(candidate_loss).all():
                                    batch_evasion_status = "nonfinite_adaptive_loss"
                                    if evasion_strict:
                                        _raise_adaptive_evasion_fail_closed(
                                            epoch, client, batch_evasion_status
                                        )
                                else:
                                    loss = candidate_loss
                                    evasion_margin_value = float(evasion_margin.detach().item())
                                    evasion_hinge_value = float(evasion_hinge.detach().item())
                                    batch_evasion_status = "applied"
                    else:
                        try:
                            if (
                                evasion_surrogate
                                == _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3
                            ):
                                surrogate_values = _fixed_tau_dual_hinge_band_surrogate(
                                    trigger_logits,
                                    getattr(handle, "target", None),
                                    tau=evasion_fixed_tau,
                                    safety_margin=(evasion_fixed_tau_safety_margin),
                                    logit_margin_lower_guard=(evasion_logit_margin_lower_guard),
                                    logit_margin_upper_guard=(evasion_logit_margin_guard),
                                )
                            else:
                                surrogate_values = _fixed_tau_dual_hinge_surrogate(
                                    trigger_logits,
                                    getattr(handle, "target", None),
                                    tau=evasion_fixed_tau,
                                    safety_margin=(evasion_fixed_tau_safety_margin),
                                    logit_margin_guard=(evasion_logit_margin_guard),
                                )
                        except (TypeError, ValueError) as exc:
                            batch_evasion_status = f"invalid_target_or_logits:{exc}"
                            if evasion_strict:
                                _raise_adaptive_evasion_fail_closed(
                                    epoch, client, batch_evasion_status
                                )
                        else:
                            finite_tensors = (
                                surrogate_values["p_target"],
                                surrogate_values["p_target_mean"],
                                surrogate_values["probability_hinge"],
                                surrogate_values["margins"],
                                surrogate_values["margin_hinge_sum"],
                                surrogate_values["margin_hinge"],
                            )
                            if any(not torch.isfinite(value).all() for value in finite_tensors):
                                batch_evasion_status = "nonfinite_v2_surrogate"
                                if evasion_strict:
                                    _raise_adaptive_evasion_fail_closed(
                                        epoch, client, batch_evasion_status
                                    )
                            else:
                                fixed_tau_weighted_loss = (
                                    evasion_fixed_tau_lambda * surrogate_values["probability_hinge"]
                                )
                                logit_margin_weighted_loss = (
                                    evasion_logit_margin_lambda * surrogate_values["margin_hinge"]
                                )
                                evasion_loss = fixed_tau_weighted_loss + logit_margin_weighted_loss
                                candidate_loss = ce_loss + evasion_loss
                                if not torch.isfinite(candidate_loss).all():
                                    batch_evasion_status = "nonfinite_adaptive_loss"
                                    if evasion_strict:
                                        _raise_adaptive_evasion_fail_closed(
                                            epoch, client, batch_evasion_status
                                        )
                                else:
                                    loss = candidate_loss
                                    margins_detached = surrogate_values["margins"].detach()
                                    v2_values = {
                                        "p_target_per_sample": [
                                            float(value)
                                            for value in surrogate_values["p_target"]
                                            .detach()
                                            .to("cpu")
                                            .tolist()
                                        ],
                                        "p_target_mean": float(
                                            surrogate_values["p_target_mean"].detach().item()
                                        ),
                                        "fixed_tau_hinge": float(
                                            surrogate_values["probability_hinge"].detach().item()
                                        ),
                                        "fixed_tau_hinge_active_fraction": float(
                                            surrogate_values["probability_hinge_active_fraction"]
                                            .detach()
                                            .item()
                                        ),
                                        "logit_margin_sample_count": int(margins_detached.numel()),
                                        "logit_margin_per_sample": [
                                            float(value)
                                            for value in margins_detached.to("cpu").tolist()
                                        ],
                                        "logit_margin_min": float(margins_detached.min().item()),
                                        "logit_margin_mean": float(margins_detached.mean().item()),
                                        "logit_margin_max": float(margins_detached.max().item()),
                                        "logit_margin_hinge_sum": float(
                                            surrogate_values["margin_hinge_sum"].detach().item()
                                        ),
                                        "logit_margin_hinge_mean": float(
                                            surrogate_values["margin_hinge"].detach().item()
                                        ),
                                        "logit_margin_active_count": int(
                                            surrogate_values["margin_active_count"].detach().item()
                                        ),
                                        "logit_margin_active_fraction": float(
                                            surrogate_values["margin_active_fraction"]
                                            .detach()
                                            .item()
                                        ),
                                        "fixed_tau_weighted_loss": float(
                                            fixed_tau_weighted_loss.detach().item()
                                        ),
                                        "logit_margin_weighted_loss": float(
                                            logit_margin_weighted_loss.detach().item()
                                        ),
                                        "evasion_loss": float(evasion_loss.detach().item()),
                                    }
                                    if (
                                        evasion_surrogate
                                        == _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3
                                    ):
                                        v2_values.update(
                                            {
                                                "logit_margin_lower_hinge_sum": float(
                                                    surrogate_values["margin_lower_hinge_sum"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_lower_hinge_mean": float(
                                                    surrogate_values["margin_lower_hinge"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_lower_active_count": int(
                                                    surrogate_values["margin_lower_active_count"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_lower_active_fraction": float(
                                                    surrogate_values["margin_lower_active_fraction"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_upper_hinge_sum": float(
                                                    surrogate_values["margin_upper_hinge_sum"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_upper_hinge_mean": float(
                                                    surrogate_values["margin_upper_hinge"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_upper_active_count": int(
                                                    surrogate_values["margin_upper_active_count"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_upper_active_fraction": float(
                                                    surrogate_values["margin_upper_active_fraction"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_within_band_count": int(
                                                    surrogate_values["margin_within_band_count"]
                                                    .detach()
                                                    .item()
                                                ),
                                                "logit_margin_within_band_fraction": float(
                                                    surrogate_values["margin_within_band_fraction"]
                                                    .detach()
                                                    .item()
                                                ),
                                            }
                                        )
                                    batch_evasion_status = "applied"
            loss.backward()
            if lockdown_enabled and lockdown_mask is not None:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if name in lockdown_mask:
                        param.grad.data = param.grad.data * lockdown_mask[name].to(param.device)
            optimizer.step()
            if detector_evasion_enabled:
                common_record = {
                    "round": int(epoch),
                    "client_id": int(client),
                    "local_epoch": int(i + 1),
                    "batch_idx": int(batch_idx),
                    "status": batch_evasion_status,
                    "previous_trigger_only": True,
                    "trigger_round": evasion_trigger_round,
                    "expected_trigger_round": evasion_expected_trigger_round,
                    "trigger_source": evasion_trigger_source,
                    "start_round": int(evasion_start_round),
                    "trigger_batch_size": (
                        int(previous_trigger.shape[0])
                        if isinstance(previous_trigger, torch.Tensor)
                        else 0
                    ),
                    "ce_loss": float(ce_loss.detach().item()),
                    "total_loss": float(loss.detach().item()),
                }
                if evasion_surrogate == _ADAPTIVE_EVASION_SURROGATE_LEGACY:
                    legacy_record = dict(common_record)
                    legacy_record.update(
                        {
                            "lambda": float(evasion_lambda),
                            "kappa": float(evasion_kappa),
                            "margin": evasion_margin_value,
                            "hinge": evasion_hinge_value,
                        }
                    )
                    _append_adaptive_evasion_record(handle, legacy_record)
                else:
                    v2_record = dict(common_record)
                    v2_record.update(
                        {
                            "surrogate_schema": (
                                _ADAPTIVE_EVASION_SURROGATE_SCHEMA_V3
                                if evasion_surrogate
                                == _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3
                                else _ADAPTIVE_EVASION_SURROGATE_SCHEMA_V2
                            ),
                            "surrogate": evasion_surrogate,
                            "surrogate_formula": (
                                _ADAPTIVE_EVASION_SURROGATE_FORMULA_V3
                                if evasion_surrogate
                                == _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3
                                else _ADAPTIVE_EVASION_SURROGATE_FORMULA_V2
                            ),
                            "measurement_stage": "pre_optimizer_step",
                            "probe_forward_mode": probe_forward_audit["probe_forward_mode"],
                            "probe_eval_scope": probe_forward_audit["probe_eval_scope"],
                            "probe_digest_schema": _DETECTOR_PROBE_DIGEST_SCHEMA,
                            "probe_digest": evasion_probe_digest,
                            "cached_probe_digest": evasion_cached_probe_digest,
                            "probe_digest_verified": evasion_probe_digest_verified,
                            "probe_bn_buffer_count": probe_forward_audit["probe_bn_buffer_count"],
                            "probe_bn_digest_before": probe_forward_audit["probe_bn_digest_before"],
                            "probe_bn_digest_after": probe_forward_audit["probe_bn_digest_after"],
                            "probe_bn_buffers_unchanged": probe_forward_audit[
                                "probe_bn_buffers_unchanged"
                            ],
                            "probe_training_state_restored": probe_forward_audit[
                                "probe_training_state_restored"
                            ],
                            "probe_reference_state_digest_schema": (
                                probe_forward_audit["probe_reference_state_digest_schema"]
                            ),
                            "probe_reference_state_digest": probe_forward_audit[
                                "probe_reference_state_digest"
                            ],
                            "probe_reference_state_tensor_count": (
                                probe_forward_audit["probe_reference_state_tensor_count"]
                            ),
                            "probe_reference_state_source": probe_forward_audit[
                                "probe_reference_state_source"
                            ],
                            "probe_reference_bn_stats_digest_schema": (
                                probe_forward_audit["probe_reference_bn_stats_digest_schema"]
                            ),
                            "probe_reference_bn_stats_digest": probe_forward_audit[
                                "probe_reference_bn_stats_digest"
                            ],
                            "probe_reference_bn_stats_source": probe_forward_audit[
                                "probe_reference_bn_stats_source"
                            ],
                            "probe_reference_bn_stats_count": probe_forward_audit[
                                "probe_reference_bn_stats_count"
                            ],
                            "probe_reference_bn_buffer_count": probe_forward_audit[
                                "probe_reference_bn_buffer_count"
                            ],
                            "probe_reference_bn_digest_before": probe_forward_audit[
                                "probe_reference_bn_digest_before"
                            ],
                            "probe_reference_bn_digest_after": probe_forward_audit[
                                "probe_reference_bn_digest_after"
                            ],
                            "probe_reference_bn_buffers_unchanged": (
                                probe_forward_audit["probe_reference_bn_buffers_unchanged"]
                            ),
                            "probe_reference_training_state_restored": (
                                probe_forward_audit["probe_reference_training_state_restored"]
                            ),
                            "probe_external_bn_stats_applied_count": (
                                probe_forward_audit["probe_external_bn_stats_applied_count"]
                            ),
                            "split_upload_training_order": probe_forward_audit[
                                "split_upload_training_order"
                            ],
                            "fixed_tau": evasion_fixed_tau,
                            "fixed_tau_safety_margin": (evasion_fixed_tau_safety_margin),
                            "fixed_tau_boundary": evasion_fixed_tau_boundary,
                            "fixed_tau_lambda": evasion_fixed_tau_lambda,
                            "p_target_per_sample": v2_values.get("p_target_per_sample"),
                            "p_target_mean": v2_values.get("p_target_mean"),
                            "fixed_tau_hinge": v2_values.get("fixed_tau_hinge"),
                            "fixed_tau_hinge_active_fraction": v2_values.get(
                                "fixed_tau_hinge_active_fraction"
                            ),
                            "logit_margin_guard": evasion_logit_margin_guard,
                            "logit_margin_band_lower_guard": (evasion_logit_margin_lower_guard),
                            "logit_margin_band_upper_guard": (
                                evasion_logit_margin_guard
                                if evasion_surrogate
                                == _ADAPTIVE_EVASION_SURROGATE_FIXED_TAU_DUAL_HINGE_BAND_V3
                                else None
                            ),
                            "logit_margin_lambda": evasion_logit_margin_lambda,
                            "logit_margin_sample_count": v2_values.get("logit_margin_sample_count"),
                            "logit_margin_per_sample": v2_values.get("logit_margin_per_sample"),
                            "logit_margin_min": v2_values.get("logit_margin_min"),
                            "logit_margin_mean": v2_values.get("logit_margin_mean"),
                            "logit_margin_max": v2_values.get("logit_margin_max"),
                            "logit_margin_hinge_sum": v2_values.get("logit_margin_hinge_sum"),
                            "logit_margin_hinge_mean": v2_values.get("logit_margin_hinge_mean"),
                            "logit_margin_active_count": v2_values.get("logit_margin_active_count"),
                            "logit_margin_active_fraction": v2_values.get(
                                "logit_margin_active_fraction"
                            ),
                            "logit_margin_lower_hinge_sum": v2_values.get(
                                "logit_margin_lower_hinge_sum"
                            ),
                            "logit_margin_lower_hinge_mean": v2_values.get(
                                "logit_margin_lower_hinge_mean"
                            ),
                            "logit_margin_lower_active_count": v2_values.get(
                                "logit_margin_lower_active_count"
                            ),
                            "logit_margin_lower_active_fraction": v2_values.get(
                                "logit_margin_lower_active_fraction"
                            ),
                            "logit_margin_upper_hinge_sum": v2_values.get(
                                "logit_margin_upper_hinge_sum"
                            ),
                            "logit_margin_upper_hinge_mean": v2_values.get(
                                "logit_margin_upper_hinge_mean"
                            ),
                            "logit_margin_upper_active_count": v2_values.get(
                                "logit_margin_upper_active_count"
                            ),
                            "logit_margin_upper_active_fraction": v2_values.get(
                                "logit_margin_upper_active_fraction"
                            ),
                            "logit_margin_within_band_count": v2_values.get(
                                "logit_margin_within_band_count"
                            ),
                            "logit_margin_within_band_fraction": v2_values.get(
                                "logit_margin_within_band_fraction"
                            ),
                            "fixed_tau_weighted_loss": v2_values.get("fixed_tau_weighted_loss"),
                            "logit_margin_weighted_loss": v2_values.get(
                                "logit_margin_weighted_loss"
                            ),
                            "evasion_loss": v2_values.get("evasion_loss"),
                        }
                    )
                    _append_adaptive_evasion_record(handle, v2_record)
            Predict_loss += loss.item()
            pred = log_probs.data.max(1)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        if empty_client and batches_seen:
            raise RuntimeError(
                f"client {client} declared dataset_size=0 but yielded {batches_seen} batches"
            )
        # An empty selected client contributes a zero clean update.
        if empty_client:
            acc = 0.0
            total_l = 0.0
        else:
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = Predict_loss / dataset_size
        main.logger.info(
            "___Local_Train , g_epoch {:3d}, l_epoch {:3d}, local model {},  Average loss: {:.4f}, "
            "Accuracy: {}/{} ({:.4f}%)".format(
                epoch, i + 1, client, total_l, correct, dataset_size, acc
            )
        )

    clean_weights = model.state_dict()

    if lockdown_enabled and lockdown_mask is not None:
        # Update the client's active subspace for the next round.
        total_rounds = params.get("epochs", 1)
        anneal_factor = float(lockdown_config.get("lockdown_anneal_factor", 0.0001))
        lockdown_mask, num_remove = _lockdown_fire_mask(
            clean_weights, lockdown_mask, epoch, total_rounds, anneal_factor, device=device
        )
        lockdown_state["mask"] = lockdown_mask
        lockdown_state["num_remove"] = num_remove

        # Restrict the update to the client's active subspace.
        masked_weights = {}
        for k in clean_weights.keys():
            if k in lockdown_mask and _lockdown_is_maskable_key(k, clean_weights[k]):
                m = lockdown_mask[k].to(clean_weights[k].device)
                masked_weights[k] = initial_state[k] + (clean_weights[k] - initial_state[k]) * m
            else:
                masked_weights[k] = clean_weights[k]
        clean_weights = masked_weights

    if enable_ldp and ldp_instance is not None:
        # Noise applies to trainable floating-point updates; BN buffers remain unnoised.
        def _is_float_tensor(t: torch.Tensor) -> bool:
            return t.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16)

        def _is_excluded_key(k: str) -> bool:
            return ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k)

        update_clean = {}
        float_update_keys = []
        excluded_keys = []
        for k in clean_weights.keys():
            if _is_excluded_key(k):
                excluded_keys.append(k)
                continue
            if _is_float_tensor(initial_state[k]):
                update_clean[k] = clean_weights[k] - initial_state[k]
                float_update_keys.append(k)
            else:
                excluded_keys.append(k)

        if len(excluded_keys) > 0:
            logger.debug(
                f"[LDP] Excluded {len(excluded_keys)} parameter keys from noise; examples={excluded_keys[:5]}"
            )

        # Apply clipping and noise to the update and record its RDP cost.
        scheme2a_clipped_clean = (
            str(params.get("dp_release_source", "client_ldp")).strip().lower().replace("-", "_")
            == "client_ldp"
            and str(params.get("consistency_mechanism", "none")).strip().lower() == "mpc_l2_screen"
            and str(params.get("validation_clip_mode", "")).strip().lower()
            == "aggregation_update_already_clipped"
        )
        clean_update_copy, noisy_update = apply_ldp_to_weights(
            update_clean,
            ldp_instance,
            return_clipped_clean=scheme2a_clipped_clean,
        )
        if scheme2a_clipped_clean:
            logger.info(
                f"[Scheme2A] Client {client} uses the L2-clipped clean update for validation and aggregation"
            )

        # Reconstruct model views while preserving local BN buffers and integer state.
        clean_weights_reconstructed = {}
        noisy_weights_reconstructed = {}
        for k in initial_state.keys():
            if k in float_update_keys:
                clean_weights_reconstructed[k] = initial_state[k] + clean_update_copy[k]
                noisy_weights_reconstructed[k] = initial_state[k] + noisy_update[k]
            else:
                clean_weights_reconstructed[k] = clean_weights[k]
                noisy_weights_reconstructed[k] = clean_weights[k]

        logger.info(
            f"[LDP] Client {client} produced clean and noisy model views with unnoised BN buffers and integer state"
        )
        cpu_clean = state_dict_to_cpu(clean_weights_reconstructed)
        cpu_noisy = state_dict_to_cpu(noisy_weights_reconstructed)
        try:
            del (
                optimizer,
                loss_func,
                ldp_instance,
                initial_state,
                clean_weights,
                clean_update_copy,
                noisy_update,
            )
        except Exception:
            pass
        clear_cuda_memory("train_ldp_done")
        return {"clean": cpu_clean, "noisy": cpu_noisy}
    else:
        cpu_weights = state_dict_to_cpu(clean_weights)
        try:
            del optimizer, loss_func, initial_state, clean_weights
        except Exception:
            pass
        clear_cuda_memory("train_clean_done")
        return cpu_weights
