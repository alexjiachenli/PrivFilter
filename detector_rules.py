"""Calibrated centered response rule and FilterFL original-rule control.

Plaintext diagnostics and secure decisions use the same public constants.
"""

from __future__ import annotations
import math
from typing import Iterable, List, Sequence

FIXED_TAU = "fixed_tau"
CALIBRATED_CENTERED_SCORE = "calibrated_centered_score"
CENTER_NONE = "none"
CENTER_MEAN = "mean"
CENTER_LOO_MEAN = "loo_mean"
CENTER_MEDIAN = "median"
VALID_CENTER_MODES = {CENTER_NONE, CENTER_MEAN, CENTER_LOO_MEAN, CENTER_MEDIAN}
SCORE_TARGET_PROBABILITY = "target_probability"
SCORE_LOGIT_MARGIN = "logit_margin"
VALID_CALIBRATED_SCORE_TYPES = {SCORE_TARGET_PROBABILITY, SCORE_LOGIT_MARGIN}
DEBUG_LOGITS = "debug_logits"
ACCEPT_BITS_ONLY = "accept_bits_only"
DIAGNOSTIC_SCORES = "diagnostic_scores"
VALID_DETECTOR_OUTPUT_MODES = {DEBUG_LOGITS, ACCEPT_BITS_ONLY, DIAGNOSTIC_SCORES}
MASK_OPENED = "opened"
MASK_SECRET_SHARED = "secret_shared"
VALID_SECURE_MASK_MODES = {MASK_OPENED, MASK_SECRET_SHARED}


def canonical_secure_mask_mode(value: object) -> str:
    """Resolve the acceptance-mask output mode.

    ``opened`` reveals acceptance bits before aggregation. ``secret_shared``
    keeps the mask in party-local shares and applies it inside MPC; the public
    accepted count supplies the averaging denominator.
    """
    mode = str(value or MASK_OPENED).strip().lower().replace("-", "_")
    aliases = {
        "plain": MASK_OPENED,
        "plaintext": MASK_OPENED,
        "reveal": MASK_OPENED,
        "revealed": MASK_OPENED,
        "secret": MASK_SECRET_SHARED,
        "shared": MASK_SECRET_SHARED,
        "secret_mask": MASK_SECRET_SHARED,
        "in_mpc": MASK_SECRET_SHARED,
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_SECURE_MASK_MODES:
        raise ValueError(
            f"unsupported secure_mask_mode={value!r}; expected one of {sorted(VALID_SECURE_MASK_MODES)}"
        )
    return mode


def canonical_final_detector(value):
    mode = str(value or FIXED_TAU).strip().lower().replace("-", "_")
    if mode not in {FIXED_TAU, CALIBRATED_CENTERED_SCORE}:
        raise ValueError(f"Unsupported final_detector={value!r}")
    return mode


def canonical_center_mode(value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"center_mode must be one of {sorted(VALID_CENTER_MODES)}")
    mode = value.strip().lower().replace("-", "_")
    if mode not in VALID_CENTER_MODES:
        raise ValueError(
            f"unsupported center_mode={value!r}; expected one of {sorted(VALID_CENTER_MODES)}"
        )
    return mode


def canonical_calibrated_score_type(value: object) -> str:
    if type(value) is not str:
        raise ValueError(
            f"score_type must be explicitly set to one of {sorted(VALID_CALIBRATED_SCORE_TYPES)}"
        )
    score_type = value.strip().lower().replace("-", "_")
    aliases = {
        "probability": SCORE_TARGET_PROBABILITY,
        "p_target": SCORE_TARGET_PROBABILITY,
        "margin": SCORE_LOGIT_MARGIN,
    }
    score_type = aliases.get(score_type, score_type)
    if score_type not in VALID_CALIBRATED_SCORE_TYPES:
        raise ValueError(
            f"unsupported score_type={value!r}; expected one of {sorted(VALID_CALIBRATED_SCORE_TYPES)}"
        )
    return score_type


def canonical_detector_output_mode(value):
    mode = str(value or DEBUG_LOGITS).strip().lower().replace("-", "_")
    mode = {
        "bits": ACCEPT_BITS_ONLY,
        "bit_only": ACCEPT_BITS_ONLY,
        "diagnostic": DIAGNOSTIC_SCORES,
    }.get(mode, mode)
    if mode not in {DEBUG_LOGITS, ACCEPT_BITS_ONLY, DIAGNOSTIC_SCORES}:
        raise ValueError(f"Unsupported detector_output_mode={value!r}")
    return mode


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted((float(value) for value in values))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _validated_calibrated_centered_constants(
    *, threshold_q: float, center_weight: float, center_mode: str, min_clients: int
):
    if type(min_clients) is not int or min_clients < 2:
        raise ValueError("min_clients must be an integer at least 2")
    if type(threshold_q) not in (int, float) or not math.isfinite(float(threshold_q)):
        raise ValueError("threshold_q must be finite")
    if type(center_weight) not in (int, float) or not math.isfinite(float(center_weight)):
        raise ValueError("center_weight must be finite and in [0, 1]")
    weight = float(center_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("center_weight must be finite and in [0, 1]")
    return (float(threshold_q), weight, canonical_center_mode(center_mode), min_clients)


def calibrated_centered_reject_bits(
    scores: Iterable[float],
    *,
    threshold_q: float,
    center_weight: float,
    center_mode: str,
    min_clients: int = 5,
) -> List[bool]:
    """Reject when score_i - center_weight * center_i > threshold_q.

    The cutoff is calibrated externally. Mean and median use a cohort center;
    leave-one-out mean excludes the current client; ``none`` uses zero.
    """
    threshold, weight, mode, required_clients = _validated_calibrated_centered_constants(
        threshold_q=threshold_q,
        center_weight=center_weight,
        center_mode=center_mode,
        min_clients=min_clients,
    )
    values = [float(value) for value in scores]
    if not values or any((not math.isfinite(value) for value in values)):
        raise ValueError("calibrated centered detector scores must be finite")
    if len(values) < required_clients:
        raise ValueError(
            f"calibrated centered detector requires at least {required_clients} clients; received {len(values)}"
        )
    if mode == CENTER_NONE or weight == 0.0:
        centers = [0.0] * len(values)
    elif mode == CENTER_MEAN:
        center = sum(values) / float(len(values))
        centers = [center] * len(values)
    elif mode == CENTER_LOO_MEAN:
        total = sum(values)
        denominator = float(len(values) - 1)
        centers = [(total - value) / denominator for value in values]
    else:
        center = _median(values)
        centers = [center] * len(values)
    return [bool(value - weight * center > threshold) for value, center in zip(values, centers)]


def _secure_compare_swap(left, right):
    """Return an ascending compare/swap pair without opening either value."""
    high = right + (left - right).relu()
    low = left + right - high
    return (low, high)


def _secure_sort(values: Sequence[object]) -> List[object]:
    """Small deterministic sorting network suitable for K=10 cohorts.

    Odd-even transposition uses a fixed comparison schedule, so control flow
    does not depend on secret values.  It uses K(K-1)/2 secure comparisons.
    """
    ordered = list(values)
    for phase in range(len(ordered)):
        start = phase % 2
        for index in range(start, len(ordered) - 1, 2):
            ordered[index], ordered[index + 1] = _secure_compare_swap(
                ordered[index], ordered[index + 1]
            )
    return ordered


def _secure_median(values: Sequence[object]):
    if not values:
        raise ValueError("secure median requires at least one value")
    ordered = _secure_sort(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def secure_mean_target_logit_margin(encrypted_logits, *, num_classes: int):
    """Compute the encrypted mean target-versus-other logit margin.

    A fixed sequence of encrypted ReLUs computes the other-class maximum
    without opening intermediate values.
    """
    num_classes = int(num_classes)
    if num_classes < 2:
        raise ValueError("target-logit margin requires at least two classes")
    shape = tuple((int(value) for value in encrypted_logits.size()))
    if len(shape) != 2 or shape[1] != num_classes or shape[0] <= 0:
        raise ValueError(f"encrypted logits must have shape [batch, num_classes]; received {shape}")
    other_indices = list(range(1, num_classes))
    max_other = encrypted_logits[:, other_indices[0]]
    for class_index in other_indices[1:]:
        candidate = encrypted_logits[:, class_index]
        max_other = candidate + (max_other - candidate).relu()
    margin_sum = (encrypted_logits[:, 0] - max_other).sum(dim=0)
    return margin_sum / float(shape[0])


def secure_mean_target_probability(encrypted_probabilities, *, num_classes: int):
    """Average encrypted response probabilities over the public batch size."""
    num_classes = int(num_classes)
    if num_classes < 2:
        raise ValueError("target probability requires at least two classes")
    shape = tuple((int(value) for value in encrypted_probabilities.size()))
    if len(shape) != 2 or shape[1] != num_classes or shape[0] <= 0:
        raise ValueError(
            f"encrypted probabilities must have shape [batch, num_classes]; received {shape}"
        )
    target_sum = encrypted_probabilities[:, 0].sum(dim=0)
    return target_sum / float(shape[0])


def secure_calibrated_centered_decision_values(
    scores: Sequence[object],
    *,
    threshold_q: float,
    center_weight: float,
    center_mode: str,
    min_clients: int = 5,
) -> List[object]:
    """Return encrypted signed centered decisions.

    Mean and leave-one-out rules clear positive public denominators before
    the sign test to avoid reciprocal fixed-point rounding.
    """
    threshold, weight, mode, required_clients = _validated_calibrated_centered_constants(
        threshold_q=threshold_q,
        center_weight=center_weight,
        center_mode=center_mode,
        min_clients=min_clients,
    )
    if len(scores) < required_clients:
        raise ValueError(
            f"calibrated centered detector requires at least {required_clients} clients; received {len(scores)}"
        )
    if mode == CENTER_NONE or weight == 0.0:
        return [score - threshold for score in scores]
    if mode == CENTER_MEDIAN:
        center = _secure_median(scores)
        return [score - center * weight - threshold for score in scores]
    total = scores[0]
    for score in scores[1:]:
        total = total + score
    if mode == CENTER_MEAN:
        cohort_size = float(len(scores))
        return [score * cohort_size - total * weight - threshold * cohort_size for score in scores]
    other_count = len(scores) - 1
    return [
        score * other_count - (total - score) * weight - threshold * other_count for score in scores
    ]
