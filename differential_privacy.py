"""Gaussian clipping, noise calibration and per-client Renyi-DP accounting."""

import copy
import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("logger")


ALPHAS = [1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128]

ACCOUNTANT_SUBMITTED_EXPECTED_QT = "submitted_expected_qt"
ACCOUNTANT_BALANCED_ANONYMOUS = "balanced_anonymous"
ACCOUNTANT_BALANCED_DIRECT = "balanced_direct_composition"


def rdp_gaussian(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """Return Gaussian RDP: alpha * sensitivity**2 / (2 * sigma**2)."""
    return alpha * (sensitivity**2) / (2.0 * (sigma**2))


def _shuffled_unlabeled_rdp_approx(
    alpha: float, sigma: float, batch_size: int, sensitivity: float = 1.0
) -> float:
    """Approximate position-mixture RDP for an unlabeled batch of K releases.

    eps_shuf = log(1 + (exp(eps_local) - 1) / K).
    """
    if batch_size <= 1:
        return rdp_gaussian(alpha, sigma, sensitivity=sensitivity)
    local_rdp = rdp_gaussian(alpha, sigma, sensitivity=sensitivity)
    if local_rdp <= 0.0:
        return 0.0
    # Avoid overflow for large local_rdp.
    if local_rdp > 40.0:
        return max(0.0, local_rdp - math.log(float(batch_size)))
    return max(0.0, math.log1p(math.expm1(local_rdp) / float(batch_size)))


RDP_CONVERSION_LEGACY = "legacy_divided"
RDP_CONVERSION_PAPER = "paper_tight"


RDP_CONVERSION_DEFAULT = RDP_CONVERSION_PAPER


def _canonical_rdp_conversion(name: Optional[str]) -> str:
    value = str(name or RDP_CONVERSION_DEFAULT).strip().lower()
    aliases = {
        "legacy": RDP_CONVERSION_LEGACY,
        "legacy_divided": RDP_CONVERSION_LEGACY,
        "submitted": RDP_CONVERSION_LEGACY,
        "paper": RDP_CONVERSION_PAPER,
        "paper_tight": RDP_CONVERSION_PAPER,
        "balle": RDP_CONVERSION_PAPER,
        "tight": RDP_CONVERSION_PAPER,
    }
    canonical = aliases.get(value, value)
    if canonical not in (RDP_CONVERSION_LEGACY, RDP_CONVERSION_PAPER):
        raise ValueError(
            f"Unsupported RDP->DP conversion: {name!r}; expected one of "
            f"{(RDP_CONVERSION_LEGACY, RDP_CONVERSION_PAPER)}"
        )
    return canonical


def rdp_to_dp(
    rdp_total_by_alpha: Dict[float, float], delta: float, conversion: str = RDP_CONVERSION_DEFAULT
) -> float:
    """Minimize the RDP-to-DP conversion over the supplied Renyi orders.

    The default ``paper_tight`` conversion follows Canonne, Kamath and Steinke
    (2020, Proposition 12). ``legacy_divided`` reproduces an obsolete numerical
    convention and is not a valid privacy-loss upper bound.
    """
    mode = _canonical_rdp_conversion(conversion)
    best_eps = float("inf")
    for alpha, rdp in rdp_total_by_alpha.items():
        # Conversion constant from Canonne, Kamath and Steinke, Proposition 12.
        c_alpha = (alpha - 1) * math.log(1 - 1 / alpha) - math.log(alpha)
        if mode == RDP_CONVERSION_PAPER:
            eps_alpha = rdp + (math.log(1.0 / delta) + c_alpha) / (alpha - 1)
        else:
            eps_alpha = (rdp - math.log(delta) + (-c_alpha)) / (alpha - 1)
        if eps_alpha < best_eps:
            best_eps = eps_alpha
    return best_eps


def _log_standard_normal_cdf(x: float) -> float:
    """log Phi(x), accurate into the far left tail where erfc underflows."""
    if x > -30.0:
        return math.log(math.erfc(-x / math.sqrt(2.0)) / 2.0)
    # Asymptotic expansion of the Mills ratio; the neglected term is O(x^-6).
    inv = 1.0 / (x * x)
    series = 1.0 - inv * (1.0 - 3.0 * inv * (1.0 - 5.0 * inv))
    return -0.5 * x * x - math.log(-x) - 0.5 * math.log(2.0 * math.pi) + math.log(series)


def analytic_gaussian_epsilon(
    releases: float, sigma: float, delta: float, clip_norm: float = 1.0
) -> float:
    """Return the exact composed-Gaussian reference epsilon.

    Solve delta = Phi(mu/2 - eps/mu) - exp(eps) * Phi(-mu/2 - eps/mu),
    where mu = sqrt(releases) / sigma (Balle and Wang, 2018).
    Calibration uses the RDP conversion; this function provides a reference.
    """
    releases = float(releases)
    sigma = float(sigma)
    if releases <= 0.0:
        return 0.0
    if sigma <= 0.0:
        return float("inf")
    mu = math.sqrt(releases) / sigma

    def delta_at(eps: float) -> float:
        upper = _log_standard_normal_cdf(mu / 2.0 - eps / mu)
        lower = _log_standard_normal_cdf(-mu / 2.0 - eps / mu)
        return math.exp(upper) - math.exp(min(eps + lower, 700.0))

    lo, hi = 0.0, max(1.0, mu * mu)
    while delta_at(hi) > delta:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("analytic Gaussian epsilon exceeded the search bound")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if delta_at(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


# Per-client composition count for the coordinator-visible release transcript.
DP_RELEASES_PER_ROUND_KEY = "ldp_releases_per_round"
_MPC_AGGREGATE_ROUTES = frozenset(
    {
        "mpc_aggregate_ref_dp_fast",
        "mpc_aggregate_ref_dp_secure_clip_debug",
        "aggregate",
        "aggregate_ref",
        "mpc_aggregate_ref",
        "mpc_assisted_aggregate_ref_dp",
    }
)
_FILTERED_REFERENCE_SOURCES = frozenset(
    {
        "noisy_agg",
        "aggregate_ref",
        "aggregate_noisy_reference",
    }
)


def releases_per_round(params: Dict[str, Any]) -> Tuple[int, str]:
    """Return per-client release compositions per round and their basis.

    Partition-correlated releases compose once per participating round.
    ``ldp_releases_per_round`` provides an explicit count for other routes.
    """
    override = params.get(DP_RELEASES_PER_ROUND_KEY)
    if override is not None:
        count = int(override)
        if count < 1:
            raise ValueError(f"{DP_RELEASES_PER_ROUND_KEY} must be >= 1, got {override!r}")
        return count, "config override"

    route = str(params.get("dp_release_source", "client_ldp")).strip().lower().replace("-", "_")
    baseline = str(params.get("ldp_ori_weight_source", "clean")).strip().lower()
    if route in _MPC_AGGREGATE_ROUTES and baseline in _FILTERED_REFERENCE_SOURCES:
        pairing = (
            str(params.get("mpc_dp_release_pairing", "independent"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if pairing in ("partition", "partition_correlated", "correlated", "one_release"):
            # (S_all + Z_a + Z_b, S_acc + Z_a) is an invertible transform of
            # (S_acc + Z_a, S_rej + Z_b), with one contribution per client.
            return 1, (
                "MPC aggregate route with partition-correlated noise: the two "
                "opened aggregates are an invertible transform of a partition, "
                "so a client's data enters one release per round"
            )
        return 2, (
            "MPC aggregate route with a post-detection filtered reference: the "
            "pre-detection aggregate and the filtered aggregate are two opened "
            "releases per round"
        )
    return 1, "one coordinator-visible DP release per participating round"


def _canonical_accountant_variant(name: Optional[str]) -> str:
    value = str(name or ACCOUNTANT_SUBMITTED_EXPECTED_QT).strip().lower()
    aliases = {
        "submitted": ACCOUNTANT_SUBMITTED_EXPECTED_QT,
        "submitted_expected_qt": ACCOUNTANT_SUBMITTED_EXPECTED_QT,
        "submitted_expected_qt_accountant": ACCOUNTANT_SUBMITTED_EXPECTED_QT,
        "legacy": ACCOUNTANT_SUBMITTED_EXPECTED_QT,
        "legacy_qt": ACCOUNTANT_SUBMITTED_EXPECTED_QT,
        "balanced_anonymous": ACCOUNTANT_BALANCED_ANONYMOUS,
        "balanced+anonymous": ACCOUNTANT_BALANCED_ANONYMOUS,
        "anonymous": ACCOUNTANT_BALANCED_ANONYMOUS,
        "balanced_direct_composition": ACCOUNTANT_BALANCED_DIRECT,
        "balanced_direct": ACCOUNTANT_BALANCED_DIRECT,
        "direct_composition": ACCOUNTANT_BALANCED_DIRECT,
        "direct": ACCOUNTANT_BALANCED_DIRECT,
    }
    return aliases.get(value, value)


def _compose_rdp(
    num_steps: float, sigma: float, clip_norm: float, per_step_fn
) -> Dict[float, float]:
    return {
        alpha: float(num_steps) * float(per_step_fn(alpha, sigma, clip_norm)) for alpha in ALPHAS
    }


def _normalize_participations(participations: Optional[Sequence[Any]]) -> Tuple[int, ...]:
    if participations is None:
        return tuple()
    return tuple(int(x) for x in participations)


def _infer_sampling_rate(total_clients: int, sampled_clients: int) -> float:
    if total_clients <= 0 or sampled_clients <= 0:
        return 0.0
    return max(0.0, min(1.0, float(sampled_clients) / float(total_clients)))


def _infer_balanced_block_participations(
    total_clients: int, sampled_clients: int, total_rounds: int
) -> Tuple[int, ...]:
    if total_clients <= 0 or sampled_clients <= 0:
        return tuple()
    if total_clients % sampled_clients != 0:
        raise ValueError(
            f"balanced_block requires N divisible by K, got N={total_clients}, K={sampled_clients}"
        )
    rounds_per_block = total_clients // sampled_clients
    full_blocks = total_rounds // rounds_per_block
    remainder_rounds = total_rounds % rounds_per_block
    base = full_blocks
    extra = 1 if remainder_rounds > 0 else 0
    counts = [base for _ in range(total_clients)]
    for idx in range(min(total_clients, remainder_rounds * sampled_clients)):
        counts[idx] += extra
    return tuple(counts)


def submitted_expected_qT_accountant(
    total_rounds: int,
    sampled_clients: int,
    total_clients: int,
    sigma: float,
    delta: float,
    clip_norm: float,
    conversion: str = RDP_CONVERSION_DEFAULT,
) -> float:
    q = _infer_sampling_rate(total_clients, sampled_clients)
    effective_steps = float(total_rounds) * q
    rdp_totals = _compose_rdp(
        effective_steps,
        sigma,
        clip_norm,
        lambda alpha, s, c: rdp_gaussian(alpha, s, sensitivity=c),
    )
    return rdp_to_dp(rdp_totals, delta, conversion=conversion)


def balanced_direct_composition_accountant(
    participations: Sequence[int],
    sigma: float,
    delta: float,
    clip_norm: float,
    conversion: str = RDP_CONVERSION_DEFAULT,
) -> float:
    counts = _normalize_participations(participations)
    steps = max(counts) if counts else 0
    rdp_totals = _compose_rdp(
        steps,
        sigma,
        clip_norm,
        lambda alpha, s, c: rdp_gaussian(alpha, s, sensitivity=c),
    )
    return rdp_to_dp(rdp_totals, delta, conversion=conversion)


def balanced_anonymous_accountant(
    participations: Sequence[int],
    batch_size: int,
    sigma: float,
    delta: float,
    clip_norm: float,
    conversion: str = RDP_CONVERSION_DEFAULT,
) -> float:
    counts = _normalize_participations(participations)
    steps = max(counts) if counts else 0
    rdp_totals = _compose_rdp(
        steps,
        sigma,
        clip_norm,
        lambda alpha, s, c: _shuffled_unlabeled_rdp_approx(
            alpha, s, batch_size=batch_size, sensitivity=c
        ),
    )
    return rdp_to_dp(rdp_totals, delta, conversion=conversion)


def evaluate_accountant_for_sigma(
    params: Dict[str, Any],
    sigma: float,
    actual_participations: Optional[Sequence[int]] = None,
    participations_are_releases: bool = False,
) -> Dict[str, Any]:
    """Evaluate the configured accountant at a specified noise scale.

    ``actual_participations`` contains client round counts. Set
    ``participations_are_releases`` when counts already include each release.
    """
    variant = _canonical_accountant_variant(params.get("ldp_accountant_variant"))
    conversion = _canonical_rdp_conversion(params.get("ldp_rdp_conversion"))
    total_rounds = int(params.get("epochs", params.get("ldp_total_steps", 100)))
    total_clients = int(params.get("number_of_total_participants", 0))
    sampled_clients = int(params.get("num_clients", 0))
    delta = float(params.get("ldp_target_delta", params.get("ldp_delta", 1e-5)))
    clip_norm = float(params.get("ldp_max_grad_norm", params.get("ldp_sensitivity", 1.0)))
    sampling_strategy = (
        str(
            params.get(
                "client_sampling_strategy",
                "balanced_block" if params.get("balanced_schedule", False) else "uniform_random",
            )
        )
        .strip()
        .lower()
    )

    # Inferred balanced participation requires an enforcing public schedule.
    schedule_assumed = actual_participations is None and variant in [
        ACCOUNTANT_BALANCED_ANONYMOUS,
        ACCOUNTANT_BALANCED_DIRECT,
    ]
    schedule_is_balanced = sampling_strategy in (
        "balanced_block",
        "balanced_block_randomized",
        "block_balanced",
        "legacy_balanced",
        "balanced_legacy",
        "frozen_manifest",
        "frozen_schedule",
    )
    schedule_claim_holds = not schedule_assumed or schedule_is_balanced
    if not schedule_claim_holds:
        logger.warning(
            "[LDP][accountant] client_sampling_strategy=%s does not enforce the "
            "accounted participation cap. Use balanced_block or measured per-client counts.",
            sampling_strategy,
        )

    if actual_participations is not None:
        participations = _normalize_participations(actual_participations)
    elif variant in [ACCOUNTANT_BALANCED_ANONYMOUS, ACCOUNTANT_BALANCED_DIRECT]:
        participations = _infer_balanced_block_participations(
            total_clients, sampled_clients, total_rounds
        )
    else:
        participations = tuple()

    # Convert participation rounds to release compositions exactly once.
    per_round, release_basis = releases_per_round(params)
    if participations_are_releases:
        release_counts = participations
        rounds_per_client: Tuple[int, ...] = tuple()
        release_basis = "caller supplied opened-release counts"
    else:
        rounds_per_client = participations
        release_counts = tuple(int(count) * per_round for count in participations)
    participations = release_counts

    if variant == ACCOUNTANT_SUBMITTED_EXPECTED_QT:
        epsilon = submitted_expected_qT_accountant(
            total_rounds=total_rounds * per_round,
            sampled_clients=sampled_clients,
            total_clients=total_clients,
            sigma=sigma,
            delta=delta,
            clip_norm=clip_norm,
            conversion=conversion,
        )
        description = "Expected-participation accountant using T_eff = qT under uniform random round sampling."
        approximation = None
    elif variant == ACCOUNTANT_BALANCED_ANONYMOUS:
        epsilon = balanced_anonymous_accountant(
            participations=participations,
            batch_size=sampled_clients,
            sigma=sigma,
            delta=delta,
            clip_norm=clip_norm,
            conversion=conversion,
        )
        description = "Balanced schedule with anonymized shuffled noisy uploads; accountant uses an unlabeled-batch RDP approximation."
        approximation = "Per-release anonymous RDP uses log(1 + (exp(eps_local(alpha)) - 1) / K)."
    elif variant == ACCOUNTANT_BALANCED_DIRECT:
        epsilon = balanced_direct_composition_accountant(
            participations=participations,
            sigma=sigma,
            delta=delta,
            clip_norm=clip_norm,
            conversion=conversion,
        )
        description = "Balanced schedule with direct composition over actual per-client participations and no anonymity amplification."
        approximation = None
    else:
        raise ValueError(f"Unsupported LDP accountant variant: {variant}")

    worst_case_releases = max(participations) if participations else total_rounds * per_round
    return {
        "variant": variant,
        "rdp_conversion": conversion,
        "epsilon": float(epsilon),
        # Exact Gaussian epsilon provides an independent accounting reference.
        "epsilon_exact_gaussian": float(
            analytic_gaussian_epsilon(worst_case_releases, sigma, delta, clip_norm)
        ),
        "sigma": float(sigma),
        "delta": delta,
        "clip_norm": clip_norm,
        "sampling_strategy": sampling_strategy,
        # Retain both release counts and their originating participation rounds.
        "participations": list(participations),
        "releases_per_client": list(participations),
        "rounds_per_client": list(rounds_per_client),
        "releases_per_round": per_round,
        "release_basis": release_basis,
        "worst_case_releases": int(worst_case_releases),
        # Validity of the schedule used to infer per-client counts.
        "schedule_claim_holds": bool(schedule_claim_holds),
        "description": description,
        "approximation": approximation,
        "sampled_clients": sampled_clients,
        "total_clients": total_clients,
        "total_rounds": total_rounds,
    }


def calibrate_sigma_for_variant(
    params: Dict[str, Any],
    target_epsilon: float,
    actual_participations: Optional[Sequence[int]] = None,
    lo: float = 1e-6,
    hi: Optional[float] = None,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """Find the smallest noise scale meeting the configured composed epsilon."""
    clip_norm = float(params.get("ldp_max_grad_norm", params.get("ldp_sensitivity", 1.0)))
    hi = float(hi) if hi is not None else max(clip_norm, 1.0)

    def eps_with_sigma(sigma_val: float) -> float:
        return evaluate_accountant_for_sigma(
            params, sigma=sigma_val, actual_participations=actual_participations
        )["epsilon"]

    while eps_with_sigma(hi) > target_epsilon:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError(
                f"Failed to calibrate sigma for target epsilon={target_epsilon}; sigma exceeded search bound."
            )

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if eps_with_sigma(mid) <= target_epsilon:
            hi = mid
        else:
            lo = mid

    calibrated = evaluate_accountant_for_sigma(
        params, sigma=hi, actual_participations=actual_participations
    )
    calibrated["target_epsilon"] = float(target_epsilon)
    calibrated["noise_multiplier"] = float(hi / max(clip_norm, 1e-12))
    return calibrated


class RDPAccountant:
    """Accumulate Gaussian RDP and convert it to an epsilon bound."""

    def __init__(self, clip_norm: float, conversion: str = RDP_CONVERSION_DEFAULT):
        self.clip_norm = clip_norm
        self.conversion = _canonical_rdp_conversion(conversion)
        self.rdp_totals: Dict[float, float] = {a: 0.0 for a in ALPHAS}
        self.steps = 0

    def add_step(self, noise_multiplier: float):
        C = self.clip_norm
        sigma = noise_multiplier * C
        for a in ALPHAS:
            self.rdp_totals[a] += rdp_gaussian(a, sigma, sensitivity=C)
        self.steps += 1

    def epsilon(self, delta: float) -> float:
        return rdp_to_dp(self.rdp_totals, delta, conversion=self.conversion)


def find_noise_multiplier(
    target_eps: float,
    target_delta: float,
    T: int,
    clip_norm: float,
    lo: float = 0.1,
    hi: float = 20.0,
    tol: float = 1e-3,
    conversion: str = RDP_CONVERSION_DEFAULT,
) -> float:
    """Bisect the smallest noise multiplier meeting the target epsilon and delta."""

    def eps_with_nm(nm: float) -> float:
        acc_local = RDPAccountant(clip_norm, conversion=conversion)
        for _ in range(T):
            acc_local.add_step(nm)
        return acc_local.epsilon(target_delta)

    while eps_with_nm(hi) > target_eps:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("noise_multiplier too large, check T/target/clip_norm.")

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if eps_with_nm(mid) <= target_eps:
            hi = mid
        else:
            lo = mid
    return hi


def _clip_update_dict_l2(
    model_update: Dict[str, torch.Tensor], clip_norm: float
) -> Dict[str, torch.Tensor]:
    """Clip the concatenated update tensors to a joint L2 radius."""
    params = [p.view(-1) for p in model_update.values()]
    if len(params) == 0:
        return {k: v.clone() for k, v in model_update.items()}
    vec = torch.cat(params)
    norm = torch.linalg.norm(vec).item()
    if norm <= clip_norm or norm == 0.0:
        return {k: v.clone() for k, v in model_update.items()}
    scale = clip_norm / norm
    clipped = {}
    for name, p in model_update.items():
        clipped[name] = p * scale
    return clipped


class LocalDifferentialPrivacy:
    """Client-side L2 clipping, Gaussian noise and RDP accounting."""

    def __init__(
        self,
        target_epsilon: float = 10.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        renyi_alpha: float = 2.0,
        noise_multiplier: Optional[float] = None,
        total_steps: int = 100,
        sampling_rate: Optional[float] = None,
        accountant_variant: str = ACCOUNTANT_SUBMITTED_EXPECTED_QT,
        achieved_epsilon: Optional[float] = None,
        rdp_conversion: str = RDP_CONVERSION_DEFAULT,
    ):
        """Configure the client-side Gaussian mechanism.

        ``sensitivity`` sets the clipping radius. If ``noise_multiplier`` is omitted,
        calibrate it for the expected participation count and target budget.
        ``renyi_alpha`` is retained as an interface parameter; accounting uses ALPHAS.
        """
        self.target_epsilon = float(target_epsilon)
        self.delta = float(delta)
        self.clip_norm = float(sensitivity)
        self.total_steps = int(total_steps)
        self.sampling_rate = float(sampling_rate) if sampling_rate is not None else None
        self.accountant_variant = _canonical_accountant_variant(accountant_variant)
        self.rdp_conversion = _canonical_rdp_conversion(rdp_conversion)
        self.achieved_epsilon = float(achieved_epsilon) if achieved_epsilon is not None else None
        # Client-side calibration uses ceil(T * q) expected participation steps.
        if self.sampling_rate is not None and self.sampling_rate > 0.0:
            self.effective_total_steps = max(
                1, int(math.ceil(self.total_steps * self.sampling_rate))
            )
        else:
            self.effective_total_steps = self.total_steps
        self.noise_multiplier = float(noise_multiplier) if noise_multiplier is not None else None

        if self.noise_multiplier is None:
            try:
                self.noise_multiplier = find_noise_multiplier(
                    target_eps=self.target_epsilon,
                    target_delta=self.delta,
                    T=self.effective_total_steps,
                    clip_norm=self.clip_norm,
                    conversion=self.rdp_conversion,
                )
            except Exception as e:
                logger.warning(f"[LDP] Noise calibration failed: {e}; using noise_multiplier=1.0")
                self.noise_multiplier = 1.0

        self.sigma = self.noise_multiplier * self.clip_norm
        self.round_count = 0

        self.accountant = RDPAccountant(self.clip_norm, conversion=self.rdp_conversion)

        logger.info(
            f"[LDP] RDP configuration: target_epsilon={self.target_epsilon}, delta={self.delta}, "
            f"clip_norm={self.clip_norm}, noise_multiplier={self.noise_multiplier:.6f}, sigma={self.sigma:.6f}, "
            f"total_steps={self.total_steps}, sampling_rate={(self.sampling_rate if self.sampling_rate is not None else 'None')}, "
            f"effective_steps={self.effective_total_steps}, accountant_variant={self.accountant_variant}, "
            f"rdp_conversion={self.rdp_conversion}, "
            f"achieved_epsilon={(self.achieved_epsilon if self.achieved_epsilon is not None else 'None')}"
        )

    def add_gaussian_noise(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply joint L2 clipping and independent Gaussian noise to update tensors."""
        clipped = _clip_update_dict_l2(model_update, self.clip_norm)
        noisy_update = {}
        total_params = 0
        for name, param in clipped.items():
            noise = torch.normal(
                0, self.sigma, size=param.shape, device=param.device, dtype=param.dtype
            )
            noisy_update[name] = param + noise
            total_params += param.numel()
        logger.debug(
            f"[LDP] Clipped and added Gaussian noise to {total_params} parameters (sigma={self.sigma:.6f})"
        )
        return noisy_update

    def track_privacy_budget(self) -> bool:
        """Record a release and return whether the target budget is met.

        Use the calibrated final epsilon when supplied; otherwise compose the
        releases recorded by this instance.
        """
        self.round_count += 1
        if self.achieved_epsilon is not None:
            remaining = self.target_epsilon - self.achieved_epsilon
            logger.info(
                f"[LDP] calibrated_final_epsilon={self.achieved_epsilon:.6f}, remaining={remaining:.6f}, "
                f"variant={self.accountant_variant}, nm={self.noise_multiplier:.6f}, sigma={self.sigma:.6f}"
            )
            return remaining >= -1e-4

        self.accountant.add_step(self.noise_multiplier)
        current_eps = self.accountant.epsilon(self.delta)
        remaining = self.target_epsilon - current_eps
        logger.info(
            f"[LDP] round={self.round_count}, epsilon={current_eps:.6f}, remaining={remaining:.6f}, "
            f"nm={self.noise_multiplier:.6f}, sigma={self.sigma:.6f}"
        )
        if remaining < 0:
            logger.warning(
                f"[LDP] Budget exceeded: epsilon={current_eps:.6f} > target={self.target_epsilon:.6f}"
            )
            return False
        return True

    def get_privacy_stats(self) -> Dict[str, float]:
        """Return the accounted epsilon, remaining budget and release statistics."""
        if self.achieved_epsilon is not None:
            return {
                "total_epsilon_spent": self.achieved_epsilon,
                "remaining_epsilon": self.target_epsilon - self.achieved_epsilon,
                "round_count": self.round_count,
                "sigma": self.sigma,
                "epsilon_per_round": self.achieved_epsilon / max(1, self.total_steps),
            }
        current_eps = self.accountant.epsilon(self.delta)
        return {
            "total_epsilon_spent": current_eps,
            "remaining_epsilon": self.target_epsilon - current_eps,
            "round_count": self.round_count,
            "sigma": self.sigma,
            "epsilon_per_round": current_eps / max(1, self.round_count),
        }

    def reset_budget(self):
        """Reset the release counter and reported cumulative budget."""
        self.total_epsilon_spent = 0.0
        self.round_count = 0
        logger.info("[LDP] Privacy-budget counters reset")


BALANCED_SCHEDULES = frozenset(
    {
        "balanced_block",
        "balanced_block_randomized",
        "block_balanced",
        "legacy_balanced",
        "balanced_legacy",
        "frozen_manifest",
        "frozen_schedule",
    }
)
ALLOW_UNBALANCED_KEY = "allow_unbalanced_dp_schedule"


def require_balanced_schedule(params: Dict[str, Any]) -> None:
    """Require a public schedule that caps per-client participation.

    ``allow_unbalanced_dp_schedule`` permits other schedules and marks the
    configured participation-based budget invalid.
    """
    if not params.get("enable_ldp", False):
        return
    strategy = (
        str(
            params.get(
                "client_sampling_strategy",
                "balanced_block" if params.get("balanced_schedule", False) else "uniform_random",
            )
        )
        .strip()
        .lower()
    )
    if strategy in BALANCED_SCHEDULES:
        params["ldp_release_cap_enforced"] = True
        return
    params["ldp_release_cap_enforced"] = False
    if params.get(ALLOW_UNBALANCED_KEY, False):
        logger.warning(
            "[LDP][schedule] client_sampling_strategy=%s with %s: "
            "per-client participation is uncapped; the configured budget is invalid.",
            strategy,
            ALLOW_UNBALANCED_KEY,
        )
        return
    raise ValueError(
        f"DP accounting requires capped participation; received client_sampling_strategy={strategy!r}. "
        "Set client_sampling_strategy=balanced_block, or "
        f"{ALLOW_UNBALANCED_KEY}=true to run with the configured budget marked invalid."
    )


def _record_release_accounting(params: Dict[str, Any], calibration: Dict[str, Any]) -> None:
    """Store composition counts, conversion and schedule validity with the run."""
    params["ldp_releases_per_round"] = int(calibration["releases_per_round"])
    params["ldp_release_basis"] = calibration["release_basis"]
    params["ldp_worst_case_releases"] = int(calibration["worst_case_releases"])
    params["ldp_epsilon_exact_gaussian"] = float(calibration["epsilon_exact_gaussian"])
    params["ldp_schedule_claim_holds"] = bool(calibration["schedule_claim_holds"])
    params["ldp_rdp_conversion"] = calibration["rdp_conversion"]


def create_ldp_instance(params: Dict) -> Optional[LocalDifferentialPrivacy]:
    """Build a configured client-side Gaussian mechanism, or None when disabled."""
    if not params.get("enable_ldp", False):
        return None

    require_balanced_schedule(params)

    target_epsilon = float(params.get("ldp_target_epsilon", params.get("ldp_epsilon", 10.0)))
    delta = float(params.get("ldp_delta", 1e-5))
    clip_norm = float(params.get("ldp_max_grad_norm", params.get("ldp_sensitivity", 1.0)))
    noise_multiplier = params.get("ldp_noise_multiplier", None)
    total_steps = int(params.get("epochs", params.get("ldp_total_steps", 100)))
    renyi_alpha = float(params.get("renyi_alpha", 2.0))
    accountant_variant = _canonical_accountant_variant(params.get("ldp_accountant_variant"))
    rdp_conversion = _canonical_rdp_conversion(params.get("ldp_rdp_conversion"))
    params["ldp_rdp_conversion"] = rdp_conversion

    sampling_rate = None
    try:
        num_clients = float(params.get("num_clients", 0))
        total_participants = float(params.get("number_of_total_participants", 0))
        if total_participants > 0 and num_clients > 0:
            sampling_rate = max(0.0, min(1.0, num_clients / total_participants))
            logger.info(
                f"[LDP] sampling_rate={sampling_rate:.6f} (num_clients={int(num_clients)}, total={int(total_participants)})"
            )
        else:
            logger.info("[LDP] Sampling configuration unavailable; using full participation")
    except Exception as e:
        logger.warning(f"[LDP] Sampling-rate calculation failed: {e}; using full participation")

    try:
        sr_override = params.get("ldp_sampling_rate", None)
        if sr_override is not None:
            sr_val = float(sr_override)
            if 0.0 < sr_val <= 1.0:
                sampling_rate = sr_val
                logger.info(f"[LDP] Explicit sampling_rate={sampling_rate:.6f}")
            else:
                logger.warning(
                    f"[LDP] Ignoring invalid ldp_sampling_rate={sr_override}; expected (0,1]"
                )
    except Exception as e:
        logger.warning(f"[LDP] Sampling-rate override failed: {e}; keeping the inferred value")

    calibrated_sigma = params.get("ldp_calibrated_sigma", None)
    achieved_epsilon = params.get("ldp_achieved_epsilon", None)

    if noise_multiplier is None:
        calibration = calibrate_sigma_for_variant(params, target_epsilon=target_epsilon)
        calibrated_sigma = calibration["sigma"]
        noise_multiplier = calibration["noise_multiplier"]
        achieved_epsilon = calibration["epsilon"]
        params["ldp_noise_multiplier"] = float(noise_multiplier)
        params["ldp_calibrated_sigma"] = float(calibrated_sigma)
        params["ldp_achieved_epsilon"] = float(achieved_epsilon)
        params["ldp_accountant_variant"] = calibration["variant"]
        params["ldp_rdp_conversion"] = calibration["rdp_conversion"]
        params["ldp_accountant_description"] = calibration["description"]
        if calibration.get("approximation") is not None:
            params["ldp_accountant_approximation"] = calibration["approximation"]
        params["ldp_calibration_participations"] = calibration["participations"]
        _record_release_accounting(params, calibration)
        logger.info(
            f"[LDP] Calibrated sigma via {calibration['variant']}: sigma={calibrated_sigma:.6f}, "
            f"noise_multiplier={noise_multiplier:.6f}, achieved_epsilon={achieved_epsilon:.6f}, "
            f"target_epsilon={target_epsilon:.6f}, "
            f"releases/round={calibration['releases_per_round']}, "
            f"worst-case releases={calibration['worst_case_releases']}, "
            f"exact-Gaussian epsilon={calibration['epsilon_exact_gaussian']:.6f}"
        )
    else:
        calibrated_sigma = float(noise_multiplier) * clip_norm
        calibration = evaluate_accountant_for_sigma(params, sigma=calibrated_sigma)
        achieved_epsilon = calibration["epsilon"]
        params.setdefault("ldp_calibrated_sigma", float(calibrated_sigma))
        params.setdefault("ldp_achieved_epsilon", float(achieved_epsilon))
        params.setdefault("ldp_accountant_variant", calibration["variant"])
        params.setdefault("ldp_accountant_description", calibration["description"])
        if calibration.get("approximation") is not None:
            params.setdefault("ldp_accountant_approximation", calibration["approximation"])
        params.setdefault("ldp_calibration_participations", calibration["participations"])
        _record_release_accounting(params, calibration)
        logger.info(
            f"[LDP] Reused configured noise_multiplier={float(noise_multiplier):.6f} -> "
            f"sigma={calibrated_sigma:.6f}, achieved_epsilon={achieved_epsilon:.6f}, "
            f"accountant_variant={calibration['variant']}, "
            f"releases/round={calibration['releases_per_round']}, "
            f"worst-case releases={calibration['worst_case_releases']}"
        )
        # Verify the budget achieved by an explicitly configured multiplier.
        if achieved_epsilon > target_epsilon * (1.0 + 1e-6):
            logger.warning(
                "[LDP][accountant] pinned noise_multiplier=%.6f composes to "
                "epsilon=%.4f against a declared target of %.4f (%d releases per "
                "worst-case client). The configured budget is not met.",
                float(noise_multiplier),
                achieved_epsilon,
                target_epsilon,
                calibration["worst_case_releases"],
            )
            params["ldp_budget_met"] = False
        else:
            params["ldp_budget_met"] = True

    return LocalDifferentialPrivacy(
        target_epsilon=target_epsilon,
        delta=delta,
        sensitivity=clip_norm,
        renyi_alpha=renyi_alpha,
        noise_multiplier=(float(noise_multiplier) if noise_multiplier is not None else None),
        total_steps=total_steps,
        sampling_rate=sampling_rate,
        accountant_variant=accountant_variant,
        achieved_epsilon=achieved_epsilon,
        rdp_conversion=rdp_conversion,
    )


def apply_ldp_to_weights(
    model_weights: Dict[str, torch.Tensor],
    ldp_instance: LocalDifferentialPrivacy,
    return_clipped_clean: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return clean and noisy update dictionaries and record one release."""
    if return_clipped_clean:
        clean_update = _clip_update_dict_l2(model_weights, ldp_instance.clip_norm)
    else:
        clean_update = copy.deepcopy(model_weights)
    noisy_update = ldp_instance.add_gaussian_noise(model_weights)

    budget_ok = ldp_instance.track_privacy_budget()
    if not budget_ok:
        logger.warning("[LDP] Epsilon exceeds the target; continuing the configured run")

    return clean_update, noisy_update
