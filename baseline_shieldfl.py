"""ShieldFL comparison rule in gradient-direction coordinates.

Reference: Z. Ma et al., "ShieldFL: Mitigating Model Poisoning Attacks in
Privacy-Preserving Federated Learning," IEEE TIFS 17, 1639-1654, 2022.
doi:10.1109/TIFS.2022.3169918. See Algorithms 2-3 and Equations 9, 11-13.
The SAFEFL reference is https://github.com/encryptogroup/SAFEFL.

Client directions g_i = W - w_i undergo selective thresholding and L2
normalization. Unit-norm inputs enter the aggregation. Round one uses their
mean; later rounds weight them by distance from an elected poison baseline.
The server rescales by the median selected-input norm. ``minmax_l2`` selects
SAFEFL preprocessing; ``l2`` follows Equation 11. The returned rejection flags
are reporting statistics; ``checked`` determines aggregation membership.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


DEFAULT_KAPPA = 0.0
DEFAULT_REJECT_THRESHOLD_FRAC = 0.5
SCALE_RULE_MEDIAN_SELECTED_NORM = "median_selected_norm"
NORMALIZATION_L2 = "l2"  # paper Eq. 11
NORMALIZATION_MINMAX_L2 = "minmax_l2"  # SAFEFL aggregation_rules.py::shieldfl
NORMALIZATIONS = (NORMALIZATION_L2, NORMALIZATION_MINMAX_L2)
DEFAULT_NORMALIZATION = NORMALIZATION_L2
_DEGENERATE_WEIGHT_SUM = 1e-9
_NORM_CHECK_REL_TOL = 1e-5
_NORM_CHECK_ABS_TOL = 1e-8


@dataclass
class ShieldFLOutput:
    """Result of one ShieldFL round."""

    confidence: List[float]
    """Normalized confidence per input; all zero for a degenerate round."""

    checked: List[bool]
    """SecJudge outcome per input: the protocol's only hard decision."""

    poison_baseline_index: Optional[int]
    """Index (into the input list) of the elected poison-baseline gradient."""

    rejected: List[bool] = field(default_factory=list)
    """Derived reporting bit only.  Never consumed by aggregation or state."""

    cos_to_previous_global: List[Optional[float]] = field(default_factory=list)
    cos_to_poison_baseline: List[Optional[float]] = field(default_factory=list)

    normalized: List[torch.Tensor] = field(default_factory=list)
    """Client-side normalised gradient ghat_i per input (zeros when degenerate)."""

    selected_norms: List[float] = field(default_factory=list)
    """||select_kappa(g_i)||_2 per input, the scale normalisation removed."""

    aggregate: Optional[torch.Tensor] = None
    """sum_i eta_i ghat_i over checked inputs (paper Alg. 3 line 18), or the
    plain mean at round 1 (lines 1-3).  None when nothing was admitted."""

    election_performed: bool = False
    scale: Optional[float] = None
    """median over checked of selected_norms; the deterministic step scale."""

    weight_sum: float = 0.0
    """sum of raw 1 - cos weights before normalisation; 0 marks a degenerate round."""


def _normalize_client_update(
    update: torch.Tensor,
    previous_update: Optional[torch.Tensor],
    kappa: float,
    is_first_round: bool,
    normalization: str = DEFAULT_NORMALIZATION,
) -> torch.Tensor:
    """Apply selective thresholding and normalization to a client direction.

    ``previous_update`` is the same client's previous unnormalized direction.
    """

    return _normalize_with_norm(update, previous_update, kappa, is_first_round, normalization)[0]


def _normalize_with_norm(
    update, previous_update, kappa, is_first_round, normalization=DEFAULT_NORMALIZATION
):
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}")
    vec = update.detach().to(torch.float32).clone()
    if is_first_round or previous_update is None:
        over_threshold = vec >= kappa
    else:
        prev = previous_update.detach().to(torch.float32)
        if prev.shape != vec.shape:
            over_threshold = vec >= kappa
        else:
            over_threshold = torch.logical_or(vec >= kappa, (vec - prev) >= kappa)
    vec = over_threshold.to(vec.dtype) * vec
    selected_norm = float(torch.norm(vec.to(torch.float64), p=2.0).item())

    if normalization == NORMALIZATION_MINMAX_L2:
        max_value = torch.max(vec)
        min_value = torch.min(vec)
        spread = max_value - min_value
        if float(spread.abs().item()) < 1e-12:
            # Zero vectors remain zero and fail the unit-norm predicate.
            return torch.zeros_like(vec), selected_norm
        vec = (vec - min_value) / spread

    # Float64 normalization keeps roundoff within the unit-norm tolerance.
    vec64 = vec.to(torch.float64)
    l2 = torch.norm(vec64, p=2.0)
    if float(l2.item()) < 1e-12:
        return torch.zeros_like(vec), selected_norm
    return (vec64 / l2).to(vec.dtype), selected_norm


def _cosine64(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().to(torch.float64)
    b64 = b.detach().to(torch.float64)
    denom = float(torch.norm(a64).item()) * float(torch.norm(b64).item())
    if denom < 1e-18:
        return 0.0
    return float(torch.dot(a64, b64).item()) / denom


def shieldfl_round(
    updates: Sequence[torch.Tensor],
    previous_global_update: Optional[torch.Tensor],
    previous_updates: Optional[Sequence[Optional[torch.Tensor]]] = None,
    *,
    kappa: float = DEFAULT_KAPPA,
    is_first_round: bool = False,
    reject_threshold_frac: float = DEFAULT_REJECT_THRESHOLD_FRAC,
    normalization: str = DEFAULT_NORMALIZATION,
) -> ShieldFLOutput:
    """Aggregate flattened directions g_i = W - w_i with ShieldFL.

    Inputs share a shape. ``previous_global_update`` is the previous normalized
    aggregate; ``previous_updates`` holds each client's previous raw direction.
    ``reject_threshold_frac`` controls the derived reporting flags only.
    """

    n = len(updates)
    if n == 0:
        return ShieldFLOutput(confidence=[], checked=[], poison_baseline_index=None)

    if previous_updates is None:
        previous_updates = [None] * n

    normalized: List[torch.Tensor] = []
    selected_norms: List[float] = []
    for i, upd in enumerate(updates):
        prev = previous_updates[i] if i < len(previous_updates) else None
        vec, sel_norm = _normalize_with_norm(upd, prev, kappa, is_first_round, normalization)
        normalized.append(vec)
        selected_norms.append(sel_norm)

    # Apply the unit-norm predicate using float64 accumulation (Alg. 3, 5-7).
    checked: List[bool] = []
    for vec in normalized:
        sq = float(torch.sum(torch.square(vec.to(torch.float64))).item())
        checked.append(
            math.isclose(sq, 1.0, rel_tol=_NORM_CHECK_REL_TOL, abs_tol=_NORM_CHECK_ABS_TOL)
        )
    checked_idx = [i for i, ok in enumerate(checked) if ok]
    cos_prev: List[Optional[float]] = [None] * n
    cos_poison: List[Optional[float]] = [None] * n

    if not checked_idx:
        return ShieldFLOutput(
            confidence=[0.0] * n,
            checked=checked,
            poison_baseline_index=None,
            rejected=[True] * n,
            cos_to_previous_global=cos_prev,
            cos_to_poison_baseline=cos_poison,
            normalized=normalized,
            selected_norms=selected_norms,
            aggregate=None,
            election_performed=False,
            scale=None,
            weight_sum=0.0,
        )

    scale = float(statistics.median([selected_norms[i] for i in checked_idx]))
    confidence = [0.0] * n

    if previous_global_update is None:
        # Round 1 (Alg. 3 lines 1-3): the admitted normalised gradients are
        # averaged; there is no reference to elect against yet.
        uniform = 1.0 / float(len(checked_idx))
        for i in checked_idx:
            confidence[i] = uniform
        aggregate = torch.zeros_like(normalized[checked_idx[0]])
        for i in checked_idx:
            aggregate = aggregate + normalized[i] * uniform
        rejected = [(not checked[i]) for i in range(n)]
        return ShieldFLOutput(
            confidence=confidence,
            checked=checked,
            poison_baseline_index=None,
            rejected=rejected,
            cos_to_previous_global=cos_prev,
            cos_to_poison_baseline=cos_poison,
            normalized=normalized,
            selected_norms=selected_norms,
            aggregate=aggregate,
            election_performed=False,
            scale=scale,
            weight_sum=float(len(checked_idx)),
        )

    ref = previous_global_update.detach().to(torch.float64)
    for i in checked_idx:
        if ref.shape == normalized[i].shape:
            cos_prev[i] = _cosine64(normalized[i], ref)
    electable = [i for i in checked_idx if cos_prev[i] is not None]
    if not electable:
        raise ValueError("previous_global_update shape does not match the updates")

    # The submission least aligned with the previous aggregate is elected as
    # the poison baseline (Alg. 3 lines 9-10).
    baseline_idx = min(electable, key=lambda i: cos_prev[i])
    baseline = normalized[baseline_idx]

    raw_conf: List[float] = []
    baseline64 = baseline.detach().to(torch.float64)
    for i in checked_idx:
        c = _cosine64(normalized[i], baseline64)
        cos_poison[i] = c
        # Clamp roundoff so eta_i = 1 - cos_i remains nonnegative (Alg. 3, 15).
        raw_conf.append(max(0.0, 1.0 - c))

    weight_sum = float(sum(raw_conf))
    if weight_sum <= _DEGENERATE_WEIGHT_SUM * len(checked_idx):
        # A zero confidence sum produces a zero update.
        weight_sum = 0.0
        normalized_conf = [0.0 for _ in raw_conf]
    else:
        normalized_conf = [c / weight_sum for c in raw_conf]  # line 17
    for slot, i in enumerate(checked_idx):
        confidence[i] = float(normalized_conf[slot])

    aggregate = torch.zeros_like(normalized[checked_idx[0]])
    for i in checked_idx:  # line 18: sum_i eta_i ghat_i over admitted inputs
        if confidence[i] != 0.0:
            aggregate = aggregate + normalized[i] * confidence[i]

    uniform = 1.0 / float(len(checked_idx))
    cutoff = float(reject_threshold_frac) * uniform
    rejected = [(not checked[i]) or (confidence[i] < cutoff) for i in range(n)]

    return ShieldFLOutput(
        confidence=confidence,
        checked=checked,
        poison_baseline_index=baseline_idx,
        rejected=rejected,
        cos_to_previous_global=cos_prev,
        cos_to_poison_baseline=cos_poison,
        normalized=normalized,
        selected_norms=selected_norms,
        aggregate=aggregate,
        election_performed=True,
        scale=scale,
        weight_sum=weight_sum,
    )


def aggregate_with_confidence(
    updates: Sequence[torch.Tensor],
    normalized_updates: Optional[Sequence[torch.Tensor]],
    confidence: Sequence[float],
    *,
    aggregate_normalized: bool = True,
) -> torch.Tensor:
    """Confidence-weighted sum, over normalised (default, the paper) or raw updates."""

    source = normalized_updates if aggregate_normalized else updates
    if not source:
        raise ValueError("aggregate_with_confidence requires at least one update")
    out = torch.zeros_like(source[0].detach().to(torch.float32))
    for vec, weight in zip(source, confidence):
        if weight == 0.0:
            continue
        out = out + vec.detach().to(torch.float32) * float(weight)
    return out


def server_step(
    global_flat: torch.Tensor,
    output: ShieldFLOutput,
    *,
    eta: float = 1.0,
) -> torch.Tensor:
    """Apply W - eta * scale * aggregate to the global parameter vector."""

    if output.aggregate is None or output.scale is None:
        raise ValueError("ShieldFL admitted no client this round; no aggregate exists")
    direction = output.aggregate.detach().to(torch.float32).to(global_flat.device)
    return global_flat.detach().to(torch.float32) - float(eta) * float(output.scale) * direction
