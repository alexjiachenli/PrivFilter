"""RoFL and ELSA input-validation reference rules.

References: H. Lycklama et al., "RoFL: Robustness of Secure Federated
Learning," IEEE S&P 2023, arXiv:2107.03311v4, Sections 2.2-2.3 and 3.2-3.4;
M. Rathee et al., "ELSA: Secure Aggregation for Federated Learning with
Malicious Actors," IEEE S&P 2023, ePrint 2022/1695, Algorithm 3 and Appendix A.

The public L2 bound is 1.5 times the median update norm unless supplied
explicitly. Client clipping precedes validation of joint L2 and per-element
bounds on the submitted vector. Both predicates use plaintext reference
arithmetic; the baseline protocols' proof and comparison systems are external
to this module. The returned validated tensors are the objects to aggregate.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch


DEFAULT_MEDIAN_MULTIPLIER = 1.5
VALIDATION_BACKEND = "plaintext_reference"
RECEIPT_SCHEMA = "fldd-input-validation-receipt-v1"
_PREDICATE_REL_TOL = 1e-6
CLIENT_CLIP_SCOPES = ("all", "honest", "none")


def rule_id(method: str) -> str:
    method = str(method).strip().lower()
    if method == "rofl":
        return "rofl-l2-median-r1.5-with-per-element-bound"
    if method == "elsa":
        return "elsa-l2sq-and-lcap-median-bound-from-rofl"
    raise ValueError(f"unsupported input-validation method {method!r}")


@dataclass
class InputValidationSpec:
    method: str
    median_multiplier: float = DEFAULT_MEDIAN_MULTIPLIER
    l2_bound: Optional[float] = None
    """Explicit public bound; None selects the median rule."""
    enforce_linf: bool = True
    linf_bound: Optional[float] = None
    """Per-element bound; None means equal to the l2 bound (RoFL 3.4 / ELSA l_cap)."""
    client_clip_scope: str = "all"

    def __post_init__(self):
        self.method = str(self.method).strip().lower()
        rule_id(self.method)
        if self.client_clip_scope not in CLIENT_CLIP_SCOPES:
            raise ValueError(
                f"client_clip_scope must be one of {CLIENT_CLIP_SCOPES}, got {self.client_clip_scope!r}"
            )
        if self.l2_bound is not None and float(self.l2_bound) <= 0.0:
            raise ValueError("an explicit l2 bound must be positive")
        if float(self.median_multiplier) <= 0.0:
            raise ValueError("median_multiplier must be positive")


@dataclass
class InputValidationResult:
    method: str
    rule: str
    bound_rule: str
    median_multiplier: float
    l2_bound: float
    linf_bound: Optional[float]
    norms_submitted: List[float]
    """||dw_i||_2 as revealed, before any client-side rule."""
    client_rule_applied: List[bool]
    norms_after_client_rule: List[float]
    max_abs_after_client_rule: List[float]
    valid: List[bool]
    invalid_reasons: List[str]
    validated: List[Optional[Dict[str, torch.Tensor]]]
    """State dicts to aggregate for valid inputs (clipped copies where the
    client rule applied, otherwise the original object); None when invalid."""
    degenerate: bool = False
    median_norm: float = 0.0

    @property
    def accepted_indices(self) -> List[int]:
        return [i for i, ok in enumerate(self.valid) if ok]

    def receipt(self, *, round_id: int, client_ids: Optional[Sequence] = None) -> dict:
        """Per-round public receipt: rule, threshold and its source, object summary,
        counts, members, weight sum and backend label.  No client tensors."""
        accepted = self.accepted_indices
        rec = {
            "receipt_schema": RECEIPT_SCHEMA,
            "round": int(round_id),
            "method": self.method,
            "rule": self.rule,
            "bound_rule": self.bound_rule,
            "median_multiplier": float(self.median_multiplier),
            "median_submitted_norm": float(self.median_norm),
            "l2_bound": float(self.l2_bound),
            "linf_bound": None if self.linf_bound is None else float(self.linf_bound),
            "validation_backend": VALIDATION_BACKEND,
            "n_submitted": len(self.valid),
            "norms_submitted": [round(v, 6) for v in self.norms_submitted],
            "client_rule_applied": list(self.client_rule_applied),
            "norms_validated": [round(v, 6) for v in self.norms_after_client_rule],
            "max_abs_validated": [round(v, 8) for v in self.max_abs_after_client_rule],
            "n_valid": len(accepted),
            "n_invalid": len(self.valid) - len(accepted),
            "invalid_indices": [i for i, ok in enumerate(self.valid) if not ok],
            "invalid_reasons": [r for r in self.invalid_reasons if r],
            "accepted_indices": accepted,
            "weight_sum": float(len(accepted)),
            "degenerate": bool(self.degenerate),
        }
        if client_ids is not None:
            rec["accepted_client_ids"] = [str(client_ids[i]) for i in accepted]
            rec["invalid_client_ids"] = [
                str(client_ids[i]) for i, ok in enumerate(self.valid) if not ok
            ]
        return rec


def _float_keys(reference: Dict[str, torch.Tensor]) -> List[str]:
    return [
        k for k, v in reference.items() if isinstance(v, torch.Tensor) and v.is_floating_point()
    ]


def update_norms(
    weights: Dict[str, torch.Tensor], global_weights: Dict[str, torch.Tensor], keys: Sequence[str]
):
    """(||dw||_2, max_j |dw_j|) of dw = weights - global over ``keys`` in float64."""
    sq = 0.0
    max_abs = 0.0
    for k in keys:
        v = weights.get(k)
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        d = v.detach().to("cpu", torch.float64) - global_weights[k].detach().to(
            "cpu", torch.float64
        )
        sq += float(torch.sum(d * d).item())
        if d.numel():
            max_abs = max(max_abs, float(torch.max(torch.abs(d)).item()))
    return math.sqrt(sq), max_abs


def median_bound(norms: Sequence[float], multiplier: float) -> float:
    """RoFL Sec. 2.3: B = r * median of the revealed client update norms."""
    if not norms:
        raise ValueError("median bound needs at least one revealed norm")
    return float(multiplier) * float(statistics.median([float(x) for x in norms]))


def clip_update(
    weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor],
    keys: Sequence[str],
    bound: float,
    norm: float,
):
    """Honest-client rule: dw <- dw * min(1, B / ||dw||_2); returns a new state dict."""
    factor = min(1.0, float(bound) / (float(norm) if norm > 0.0 else 1.0))
    if factor >= 1.0:
        return weights, False
    out = dict(weights)
    for k in keys:
        v = weights.get(k)
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        g = global_weights[k].detach().to(v.device, v.dtype)
        out[k] = (g + (v.detach() - g) * factor).to(v.dtype)
    return out, True


def validate_submissions(
    submissions: Sequence[Dict[str, torch.Tensor]],
    global_weights: Dict[str, torch.Tensor],
    spec: InputValidationSpec,
    *,
    honest_mask: Optional[Sequence[bool]] = None,
) -> InputValidationResult:
    """Run the client-side rule and the server predicate for one round."""

    n = len(submissions)
    if n == 0:
        raise ValueError("input validation needs at least one submission")
    keys = _float_keys(global_weights)
    if honest_mask is None:
        honest_mask = [True] * n
    if len(honest_mask) != n:
        raise ValueError("honest_mask length does not match submissions")

    norms_submitted: List[float] = []
    for w in submissions:
        norm, _ = update_norms(w, global_weights, keys)
        norms_submitted.append(norm)

    median_norm = float(statistics.median(norms_submitted))
    if spec.l2_bound is not None:
        l2_bound = float(spec.l2_bound)
        bound_rule = "explicit_public_bound"
    else:
        l2_bound = median_bound(norms_submitted, spec.median_multiplier)
        bound_rule = "median_of_revealed_norms_times_multiplier"
    degenerate = not (l2_bound > 0.0)
    if spec.enforce_linf:
        linf_bound = l2_bound if spec.linf_bound is None else float(spec.linf_bound)
    else:
        linf_bound = None

    validated: List[Optional[Dict[str, torch.Tensor]]] = []
    applied: List[bool] = []
    norms_after: List[float] = []
    max_after: List[float] = []
    valid: List[bool] = []
    reasons: List[str] = []
    for i, w in enumerate(submissions):
        runs_client_rule = spec.client_clip_scope == "all" or (
            spec.client_clip_scope == "honest" and bool(honest_mask[i])
        )
        if runs_client_rule and not degenerate:
            w_sub, did = clip_update(w, global_weights, keys, l2_bound, norms_submitted[i])
        else:
            w_sub, did = w, False
        applied.append(did)
        norm, max_abs = update_norms(w_sub, global_weights, keys)
        norms_after.append(norm)
        max_after.append(max_abs)
        why = []
        if degenerate:
            why.append("degenerate_bound")
        if norm > l2_bound * (1.0 + _PREDICATE_REL_TOL):
            why.append(f"l2_norm_exceeds_bound:{norm:.6g}>{l2_bound:.6g}")
        if linf_bound is not None and max_abs > linf_bound * (1.0 + _PREDICATE_REL_TOL):
            why.append(f"element_exceeds_bound:{max_abs:.6g}>{linf_bound:.6g}")
        if not all(
            torch.isfinite(w_sub[k]).all() for k in keys if isinstance(w_sub.get(k), torch.Tensor)
        ):
            why.append("non_finite_submission")
        ok = not why
        valid.append(ok)
        reasons.append(";".join(why))
        validated.append(w_sub if ok else None)

    return InputValidationResult(
        method=spec.method,
        rule=rule_id(spec.method),
        bound_rule=bound_rule,
        median_multiplier=float(spec.median_multiplier),
        l2_bound=l2_bound,
        linf_bound=linf_bound,
        norms_submitted=norms_submitted,
        client_rule_applied=applied,
        norms_after_client_rule=norms_after,
        max_abs_after_client_rule=max_after,
        valid=valid,
        invalid_reasons=reasons,
        validated=validated,
        degenerate=degenerate,
        median_norm=median_norm,
    )


def spec_from_params(params: dict, method: str) -> InputValidationSpec:
    """Read public bounds and client-clipping scope from the baseline configuration."""
    method = str(method).strip().lower()
    prefix = method
    for old_key, old_default in (
        (f"{prefix}_l2_scale", 1.0),
        (f"{prefix}_linf_scale", 1.0),
        (f"{prefix}_value_bound", None),
    ):
        if old_key in params and params[old_key] not in (None, old_default):
            raise ValueError(f"{old_key} is unsupported for the {method} rule; remove it")
    explicit = params.get(f"{prefix}_l2_bound", None)
    linf = params.get(f"{prefix}_linf_bound", None)
    return InputValidationSpec(
        method=method,
        median_multiplier=float(
            params.get(f"{prefix}_median_multiplier", DEFAULT_MEDIAN_MULTIPLIER)
        ),
        l2_bound=None if explicit in (None, "", 0, 0.0) else float(explicit),
        enforce_linf=bool(params.get(f"{prefix}_enforce_linf", True)),
        linf_bound=None if linf in (None, "", 0, 0.0) else float(linf),
        client_clip_scope=str(params.get(f"{prefix}_client_clip_scope", "all")).strip().lower(),
    )
