"""FLTrust weighting with a trusted root-data update.

References: X. Cao et al., "FLTrust: Byzantine-robust Federated Learning via
Trust Bootstrapping," NDSS 2021; T. Gehlhar et al., "SAFEFL: MPC-friendly
Framework for Private and Robust Federated Learning," IEEE SPW 2023, 69-76,
doi:10.1109/SPW59333.2023.00012. Reference implementation:
https://github.com/encryptogroup/SAFEFL, aggregation_rules.py::fltrust.

The root update defines the trust direction and magnitude. Nonnegative cosine
scores determine continuous aggregation weights. Zero-trust rejection flags
provide confusion-matrix statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


DEFAULT_ROOT_SIZE = 100
_EPS = 1e-9


@dataclass
class FLTrustOutput:
    """Result of one FLTrust round."""

    trust_scores: List[float]
    """ReLU'd cosine similarity to the server update, before normalisation."""

    normalized_weights: List[float]
    """Trust scores normalised to sum to 1."""

    scale_factors: List[float]
    """Per-client magnitude rescaling toward the server update's norm."""

    rejected: List[bool] = field(default_factory=list)
    """Derived hard decision (zero trust), for confusion-matrix reporting."""


def fltrust_round(
    updates: Sequence[torch.Tensor],
    server_update: torch.Tensor,
) -> FLTrustOutput:
    """Weight client updates by cosine trust in the root-data server update.

    Inputs are flattened w_i - W vectors; ``server_update`` has the same shape.
    """

    n = len(updates)
    if n == 0:
        return FLTrustOutput([], [], [], [])

    baseline = server_update.detach().to(torch.float32).reshape(-1)
    baseline_norm = float(torch.norm(baseline).item())

    sims: List[float] = []
    for upd in updates:
        vec = upd.detach().to(torch.float32).reshape(-1)
        if vec.shape != baseline.shape:
            sims.append(0.0)
            continue
        dot = float(torch.dot(baseline, vec).item())
        sims.append(dot / (baseline_norm + _EPS) / (float(torch.norm(vec).item()) + _EPS))

    trust = [float(F.relu(torch.tensor(s)).item()) for s in sims]
    total = sum(trust)
    normalized = [t / (total + _EPS) for t in trust]

    scale: List[float] = []
    for upd in updates:
        vec = upd.detach().to(torch.float32).reshape(-1)
        scale.append(baseline_norm / (float(torch.norm(vec).item()) + _EPS))

    rejected = [t <= 0.0 for t in trust]

    return FLTrustOutput(
        trust_scores=trust,
        normalized_weights=normalized,
        scale_factors=scale,
        rejected=rejected,
    )


def aggregate(
    updates: Sequence[torch.Tensor],
    output: FLTrustOutput,
) -> torch.Tensor:
    """FLTrust's global update: magnitude-normalised, trust-weighted sum."""

    if not updates:
        raise ValueError("aggregate requires at least one update")
    out = torch.zeros_like(updates[0].detach().to(torch.float32).reshape(-1))
    for i, upd in enumerate(updates):
        w = output.normalized_weights[i]
        if w == 0.0:
            continue
        vec = upd.detach().to(torch.float32).reshape(-1)
        out = out + vec * w * output.scale_factors[i]
    return out
