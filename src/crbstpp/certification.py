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
        _, null_entity = _evaluate_frozen(
            cert_engine, certification_context, null_matrix, null_fit
        )
        f1 = one_sided_mean_test(null_entity - full_entity)
        diagnostics["f1"] = f1.__dict__
        if not f1.testable:
            reasons.append("f1_not_testable")
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
        for root, (drop_matrix, drop_fit) in zip(
            record.support.rules, branch_models, strict=True
        ):
            if not drop_fit.converged:
                rule_pvalues.append(1.0)
                reasons.append(f"branch_nonconvergence:{root}")
                rule_diagnostics.append(
                    {
                        "rule": repr(root),
                        "global": None,
                        "horizon": None,
                        "probability": None,
                        "footprint_rows": 0,
                        "pvalue": 1.0,
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
                }
            )
            if not (
                global_test.testable
                and local_test.testable
                and probability_test.testable
            ):
                reasons.append(f"f2_not_testable:{root}")
        diagnostics["rules"] = rule_diagnostics
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
        certified = bool(f0 and not reasons and adjusted_pvalue <= config.alpha)
        certificate = Certificate(
            support_key=support_key(record.support),
            f0=f0,
            f1_pvalue=f1_pvalue,
            f2_pvalues=rule_pvalues,
            family_pvalue=pvalue,
            holm_adjusted_pvalue=adjusted_pvalue,
            certified=certified,
            reasons=reasons,
        )
        models.append(CertifiedModel(record, certificate, diagnostics))
    certified_models = tuple(model for model in models if model.certificate.certified)
    return CertificationResult(tuple(models), certified_models, len(models))
