from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .likelihood import loss_rows
from .objective import SupportRecord
from .report import Certificate
from .response import Context, ModelMatrix, ResponseEngine
from .rules import ClosureTerm, EMPTY_SUPPORT, RuleIdentity, Support, hierarchy_closure
from .search import SupportOptimizer, support_key
from .solver import FitResult


@dataclass(frozen=True)
class EffectTest:
    mean: float
    standard_error: float
    statistic: float
    pvalue: float
    testable: bool


@dataclass(frozen=True)
class CertifiedModel:
    record: SupportRecord
    certificate: Certificate
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class CertificationResult:
    models: tuple[CertifiedModel, ...]
    certified: tuple[CertifiedModel, ...]
    family_size: int


@dataclass(frozen=True)
class _EnvironmentSpec:
    """Natural held-out environments and their calibrated mixture set."""

    inverse: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    probabilities: np.ndarray
    l1_radius: float
    source: str


def _multinomial_l1_radius(n: int, environments: int, alpha: float) -> float:
    """Finite-sample confidence radius for multinomial mixture weights.

    The Bretagnolle--Huber--Carol inequality gives
    ``P(||p_hat-p||_1 >= eps) <= 2**K exp(-n eps**2/2)``.  Thus the radius is
    fixed by sample size, environment count and the pre-registered alpha; no
    hand-selected stress magnitude or inferred market regime is introduced.
    """
    if n < 1:
        raise ValueError("environment calibration requires observations")
    if environments <= 1:
        return 0.0
    radius = math.sqrt(
        2.0
        * (float(environments) * math.log(2.0) + math.log(1.0 / float(alpha)))
        / float(n)
    )
    return min(2.0, radius)


def _environment_spec(context: Context, config: RunConfig) -> _EnvironmentSpec:
    """Use pre-existing financial cohorts rather than inferred regimes."""
    raw = context.dataset.split_groups[context.entity_codes]
    source = "dataset.split_groups"
    if len(np.unique(raw)) <= 1:
        # Recurrent datasets such as IBM intentionally use random entity
        # splitting and therefore have no split-group cohort.  Their entity
        # entry times still define an outcome-blind temporal cohort.  The block
        # width is the complete formation-plus-impact horizon, not a tuned
        # market-regime threshold.
        width = max(
            1,
            (max(config.formation_windows) + config.impact_lag)
            * context.dataset.ticks_per_unit,
        )
        origin = int(np.min(context.starts))
        raw = (context.starts - origin) // width
        source = "entity_start_time/formation_plus_impact_horizon"
    labels, inverse, counts = np.unique(raw, return_inverse=True, return_counts=True)
    counts = counts.astype(np.float64)
    probabilities = counts / float(np.sum(counts))
    return _EnvironmentSpec(
        inverse=np.asarray(inverse, dtype=np.int64),
        labels=np.asarray(labels),
        counts=counts,
        probabilities=probabilities,
        l1_radius=_multinomial_l1_radius(len(raw), len(labels), config.alpha),
        source=source,
    )


def _worst_case_total_variation_mean(
    values: np.ndarray,
    probabilities: np.ndarray,
    l1_radius: float,
) -> float:
    """Exact minimum mean over an L1 ball on a finite environment simplex."""
    values = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or probabilities.shape != values.shape
        or not len(values)
        or np.any(probabilities < 0.0)
        or not np.isclose(float(np.sum(probabilities)), 1.0)
        or not 0.0 <= l1_radius <= 2.0
    ):
        raise ValueError("invalid finite-environment ambiguity problem")
    if len(values) == 1 or l1_radius == 0.0:
        return float(probabilities @ values)
    weights = probabilities.copy()
    low_order = np.argsort(values, kind="stable")
    high_order = low_order[::-1]
    budget = min(1.0, 0.5 * float(l1_radius))
    low_index = high_index = 0
    tolerance = 16.0 * np.finfo(float).eps
    while budget > tolerance:
        while (
            low_index < len(values) and weights[low_order[low_index]] >= 1.0 - tolerance
        ):
            low_index += 1
        while high_index < len(values) and weights[high_order[high_index]] <= tolerance:
            high_index += 1
        if low_index >= len(values) or high_index >= len(values):
            break
        low = int(low_order[low_index])
        high = int(high_order[high_index])
        if low == high or values[low] >= values[high]:
            break
        moved = min(budget, 1.0 - weights[low], weights[high])
        if moved <= tolerance:
            break
        weights[low] += moved
        weights[high] -= moved
        budget -= moved
    return float(weights @ values)


def _environment_robust_mean(
    values: np.ndarray, environments: _EnvironmentSpec
) -> tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != environments.inverse.shape:
        raise ValueError("entity gain and environment arrays must align")
    sums = np.bincount(
        environments.inverse,
        weights=values,
        minlength=len(environments.labels),
    )
    means = sums / environments.counts
    worst = _worst_case_total_variation_mean(
        means, environments.probabilities, environments.l1_radius
    )
    return worst, means


def one_sided_mean_test(values: np.ndarray, threshold: float = 0.0) -> EffectTest:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] - float(threshold)
    if len(values) < 2:
        return EffectTest(math.nan, math.inf, -math.inf, 1.0, False)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    if (
        not math.isfinite(standard_deviation)
        or standard_deviation <= np.finfo(float).eps
    ):
        return EffectTest(mean, math.inf, -math.inf, 1.0, False)
    standard_error = standard_deviation / math.sqrt(len(values))
    statistic = mean / standard_error
    pvalue = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    return EffectTest(mean, standard_error, statistic, min(1.0, max(0.0, pvalue)), True)


def _events_at_rows(context: Context, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    event = np.zeros(len(rows), dtype=np.float64)
    if not len(rows) or not len(context.target_rows):
        return event
    positions = np.searchsorted(context.target_rows, rows)
    matched = positions < len(context.target_rows)
    safe = np.minimum(positions, len(context.target_rows) - 1)
    matched &= context.target_rows[safe] == rows
    event[matched] = context.target_counts[positions[matched]]
    return event


def _loss_at_rows(context: Context, eta: np.ndarray, rows: np.ndarray) -> np.ndarray:
    event = _events_at_rows(context, rows)
    exposure = np.full(
        len(rows),
        1.0 / context.dataset.ticks_per_unit
        if context.dataset.likelihood == "poisson"
        else 1.0,
        dtype=np.float64,
    )
    noevent = (
        exposure - event
        if context.dataset.likelihood == "first_event_cloglog"
        else exposure
    )
    values, _, _ = loss_rows(
        eta,
        likelihood=context.dataset.likelihood,
        exposure_weight=exposure,
        noevent_weight=noevent,
        event_weight=event,
    )
    return values


def _entity_losses_sparse(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
) -> np.ndarray:
    """Exact entity losses without materializing the complete time grid."""
    baseline_dimension = matrix.free_dimension - matrix.closure_dimension
    baseline_eta = np.full(baseline_dimension, fit.coefficients[0], dtype=np.float64)
    if baseline_dimension > 1:
        baseline_eta[1:] += fit.coefficients[1:baseline_dimension]
    baseline_loss = (1.0 / context.dataset.ticks_per_unit) * np.exp(
        np.clip(baseline_eta, -745.0, 700.0)
    )
    if context.dataset.likelihood == "first_event_cloglog":
        baseline_loss *= context.dataset.ticks_per_unit
    entity = engine.entity_age_counts(context)[:, :baseline_dimension] @ baseline_loss
    rows = matrix.active_rows
    if not len(rows):
        return entity
    eta = engine.linear_predictor_at_rows(context, matrix, fit.coefficients, rows)
    full_loss = _loss_at_rows(context, eta, rows)
    default_loss = baseline_loss[matrix.active_age_bins]
    local, _ = context.rows_to_entity_time(rows)
    np.add.at(entity, local, full_loss - default_loss)
    return entity


def _branch_drop(support: Support, root: RuleIdentity) -> Support:
    root_set = set(root.antecedent)
    return Support.of(
        rule
        for rule in support.rules
        if rule != root and not root_set.issubset(rule.antecedent)
    )


def _branch_null_closure(
    full_closure: tuple[ClosureTerm, ...], drop_support: Support, root: RuleIdentity
) -> tuple[ClosureTerm, ...]:
    root_set = set(root.antecedent)
    retained = {term for term in full_closure if not root_set.issubset(term.antecedent)}
    retained.update(hierarchy_closure(drop_support))
    return tuple(sorted(retained))


def _evaluate_frozen(
    engine: ResponseEngine,
    context: Context,
    fit_matrix: ModelMatrix,
    fit: FitResult,
) -> tuple[ModelMatrix, np.ndarray]:
    matrix = engine.model_matrix(
        context, fit_matrix.support, forced_closure=fit_matrix.closure
    )
    if matrix.dimension != len(fit.coefficients):
        raise ValueError("frozen model dimension changed across split")
    entity_loss = _entity_losses_sparse(engine, context, matrix, fit)
    return matrix, entity_loss


def _fit_on_discovery(
    optimizer: SupportOptimizer,
    support: Support,
    *,
    closure: tuple[ClosureTerm, ...] | None = None,
) -> tuple[ModelMatrix, FitResult]:
    resolved = hierarchy_closure(support) if closure is None else closure
    return optimizer.fit_fixed(support, resolved)


def _holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: (pvalues[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def certify_family(
    optimizer: SupportOptimizer,
    certification_context: Context,
    family: tuple[SupportRecord, ...],
    config: RunConfig,
) -> CertificationResult:
    cert_engine = ResponseEngine(
        certification_context.dataset,
        lag=config.impact_lag,
        knot_count=config.knot_count,
        cache_bytes=config.cache_bytes // 4,
    )
    environments = _environment_spec(certification_context, config)
    interim: list[tuple[SupportRecord, float, dict[str, object], tuple[str, ...]]] = []
    for record in family:
        reasons: list[str] = []
        diagnostics: dict[str, object] = {}
        full_matrix, full_entity = _evaluate_frozen(
            cert_engine, certification_context, record.matrix, record.fit
        )
        closure = hierarchy_closure(record.support)
        null_matrix, null_fit = _fit_on_discovery(
            optimizer, EMPTY_SUPPORT, closure=closure
        )
        if not null_fit.converged:
            reasons.append("closure_null_nonconvergence")
            interim.append((record, 1.0, diagnostics, tuple(reasons)))
            continue
        null_cert_matrix, null_entity = _evaluate_frozen(
            cert_engine, certification_context, null_matrix, null_fit
        )
        f1 = one_sided_mean_test(null_entity - full_entity)
        diagnostics["f1"] = f1.__dict__
        if not f1.testable:
            reasons.append("f1_not_testable")
        support_robust_gain, support_environment_gains = _environment_robust_mean(
            null_entity - full_entity, environments
        )
        support_dimension = max(0, full_matrix.dimension - null_cert_matrix.dimension)
        support_mdl_threshold = (
            support_dimension
            * math.log(max(2, len(certification_context.entity_codes)))
            / (2.0 * len(certification_context.entity_codes))
        )
        distribution_shift_testable = len(environments.labels) > 1
        support_f3 = bool(
            distribution_shift_testable and support_robust_gain > support_mdl_threshold
        )
        if not distribution_shift_testable:
            reasons.append("f3_distribution_shift_not_testable")
        if not support_f3:
            if distribution_shift_testable:
                reasons.append("f3_support_robust_gain_below_mdl")
        branch_specifications = []
        for root in record.support.rules:
            drop_support = _branch_drop(record.support, root)
            drop_closure = _branch_null_closure(
                record.matrix.closure, drop_support, root
            )
            branch_specifications.append((drop_support, drop_closure))
        branch_models = optimizer.fit_fixed_many(branch_specifications)
        rule_pvalues: list[float] = []
        rule_diagnostics: list[dict[str, object]] = []
        rule_f3: list[bool] = []
        for root, (drop_matrix, drop_fit) in zip(
            record.support.rules, branch_models, strict=True
        ):
            if not drop_fit.converged:
                rule_pvalues.append(1.0)
                rule_f3.append(False)
                reasons.append(f"branch_nonconvergence:{root}")
                rule_diagnostics.append(
                    {
                        "rule": repr(root),
                        "global": None,
                        "horizon": None,
                        "probability": None,
                        "footprint_rows": 0,
                        "pvalue": 1.0,
                        "f3_robust_global_gain": None,
                        "f3_robust_probability_contribution": None,
                        "f3": False,
                        "reason": drop_fit.message,
                    }
                )
                continue
            drop_cert_matrix, drop_entity = _evaluate_frozen(
                cert_engine, certification_context, drop_matrix, drop_fit
            )
            global_test = one_sided_mean_test(drop_entity - full_entity)
            footprint = cert_engine.footprint_rows(
                certification_context, root, config.early_warning_horizon
            )
            local_difference = np.zeros(
                len(certification_context.entity_codes), dtype=np.float64
            )
            probability_difference = np.zeros_like(local_difference)
            if len(footprint):
                local, _ = certification_context.rows_to_entity_time(footprint)
                full_eta = cert_engine.linear_predictor_at_rows(
                    certification_context,
                    full_matrix,
                    record.fit.coefficients,
                    footprint,
                )
                drop_eta = cert_engine.linear_predictor_at_rows(
                    certification_context,
                    drop_cert_matrix,
                    drop_fit.coefficients,
                    footprint,
                )
                full_rows = _loss_at_rows(certification_context, full_eta, footprint)
                drop_rows = _loss_at_rows(certification_context, drop_eta, footprint)
                np.add.at(local_difference, local, drop_rows - full_rows)
                full_hazard = (
                    np.bincount(
                        local,
                        weights=np.exp(np.clip(full_eta, -745.0, 700.0)),
                        minlength=len(local_difference),
                    )
                    / certification_context.dataset.ticks_per_unit
                )
                drop_hazard = (
                    np.bincount(
                        local,
                        weights=np.exp(np.clip(drop_eta, -745.0, 700.0)),
                        minlength=len(local_difference),
                    )
                    / certification_context.dataset.ticks_per_unit
                )
                probability_difference = root.sign * (
                    -np.expm1(-full_hazard) + np.expm1(-drop_hazard)
                )
            local_test = one_sided_mean_test(local_difference)
            probability_test = one_sided_mean_test(
                probability_difference, config.probability_materiality
            )
            robust_global_gain, global_environment_gains = _environment_robust_mean(
                drop_entity - full_entity, environments
            )
            robust_probability, probability_environment_gains = (
                _environment_robust_mean(probability_difference, environments)
            )
            rule_f3_passed = bool(
                robust_global_gain > 0.0
                and robust_probability > config.probability_materiality
            )
            rule_f3.append(rule_f3_passed)
            if robust_global_gain <= 0.0:
                reasons.append(f"f3_rule_robust_gain_nonpositive:{root}")
            if robust_probability <= config.probability_materiality:
                reasons.append(f"f3_rule_robust_probability_nonpositive:{root}")
            pvalue = max(global_test.pvalue, local_test.pvalue, probability_test.pvalue)
            rule_pvalues.append(pvalue)
            rule_diagnostics.append(
                {
                    "rule": repr(root),
                    "global": global_test.__dict__,
                    "horizon": local_test.__dict__,
                    "probability": probability_test.__dict__,
                    "footprint_rows": int(len(footprint)),
                    "pvalue": pvalue,
                    "f3_robust_global_gain": robust_global_gain,
                    "f3_robust_probability_contribution": robust_probability,
                    "f3_global_environment_gain_min": float(
                        np.min(global_environment_gains)
                    ),
                    "f3_global_environment_gain_max": float(
                        np.max(global_environment_gains)
                    ),
                    "f3_probability_environment_gain_min": float(
                        np.min(probability_environment_gains)
                    ),
                    "f3_probability_environment_gain_max": float(
                        np.max(probability_environment_gains)
                    ),
                    "f3": rule_f3_passed,
                }
            )
            if not (
                global_test.testable
                and local_test.testable
                and probability_test.testable
            ):
                reasons.append(f"f2_not_testable:{root}")
        diagnostics["rules"] = rule_diagnostics
        f3 = bool(support_f3 and all(rule_f3))
        diagnostics["f3"] = {
            "passed": f3,
            "environment_source": environments.source,
            "environment_count": int(len(environments.labels)),
            "distribution_shift_testable": distribution_shift_testable,
            "ambiguity": "finite-sample multinomial L1 confidence set",
            "ambiguity_l1_radius": environments.l1_radius,
            "support_robust_gain": support_robust_gain,
            "support_mdl_threshold": support_mdl_threshold,
            "support_environment_gain_min": float(np.min(support_environment_gains)),
            "support_environment_gain_max": float(np.max(support_environment_gains)),
        }
        support_pvalue = max([f1.pvalue, *rule_pvalues], default=1.0)
        interim.append((record, support_pvalue, diagnostics, tuple(reasons)))
    adjusted = _holm_adjust([item[1] for item in interim])
    models: list[CertifiedModel] = []
    f0 = all(
        optimizer.context.dataset.f0_contract.get(name) is True
        for name in (
            "dynamic_predicates",
            "outcome_blind_predicate_construction",
            "direct_target_proxy_excluded",
            "strict_future_effect_required",
            "atomic_predicates",
        )
    )
    for (record, pvalue, diagnostics, reasons), adjusted_pvalue in zip(
        interim, adjusted, strict=True
    ):
        f1_pvalue = (
            float(diagnostics.get("f1", {}).get("pvalue", 1.0))
            if isinstance(diagnostics.get("f1"), dict)
            else 1.0
        )
        rule_pvalues = tuple(
            float(item["pvalue"]) for item in diagnostics.get("rules", [])
        )
        f3 = bool(diagnostics.get("f3", {}).get("passed", False))
        certified = bool(f0 and f3 and not reasons and adjusted_pvalue <= config.alpha)
        certificate = Certificate(
            support_key=support_key(record.support),
            f0=f0,
            f1_pvalue=f1_pvalue,
            f2_pvalues=rule_pvalues,
            f3=f3,
            family_pvalue=pvalue,
            holm_adjusted_pvalue=adjusted_pvalue,
            certified=certified,
            reasons=reasons,
        )
        models.append(CertifiedModel(record, certificate, diagnostics))
    certified_models = tuple(model for model in models if model.certificate.certified)
    return CertificationResult(tuple(models), certified_models, len(models))
