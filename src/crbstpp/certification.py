from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .likelihood import loss_rows
from .objective import SupportRecord
from .report import Certificate
from .response import Context, ModelMatrix, ResponseEngine
from .rules import EMPTY_SUPPORT, RuleIdentity, Support, hierarchy_closure
from .search import SupportOptimizer, support_key
from .solver import FitResult, fit_model_matrix


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
    if not math.isfinite(standard_deviation) or standard_deviation <= np.finfo(float).eps:
        return EffectTest(mean, math.inf, -math.inf, 1.0, False)
    standard_error = standard_deviation / math.sqrt(len(values))
    statistic = mean / standard_error
    pvalue = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    return EffectTest(mean, standard_error, statistic, min(1.0, max(0.0, pvalue)), True)


def _grid_weights(context: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exposure = np.ones(context.n_grid, dtype=np.float64)
    event = np.zeros(context.n_grid, dtype=np.float64)
    if len(context.target_rows):
        np.add.at(event, context.target_rows, context.target_counts)
    noevent = exposure - event if context.dataset.likelihood == "first_event_cloglog" else exposure.copy()
    return exposure, noevent, event


def entity_losses(context: Context, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exposure, noevent, event = _grid_weights(context)
    rows, _, _ = loss_rows(
        eta,
        likelihood=context.dataset.likelihood,
        exposure_weight=exposure,
        noevent_weight=noevent,
        event_weight=event,
    )
    entity = np.add.reduceat(rows, context.offsets[:-1])
    return rows, entity


def _branch_drop(support: Support, root: RuleIdentity) -> Support:
    root_set = set(root.antecedent)
    return Support.of(
        rule
        for rule in support.rules
        if rule != root and not root_set.issubset(rule.antecedent)
    )


def _evaluate_frozen(
    engine: ResponseEngine,
    context: Context,
    fit_matrix: ModelMatrix,
    fit: FitResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = engine.model_matrix(
        context, fit_matrix.support, forced_closure=fit_matrix.closure
    )
    if matrix.dimension != len(fit.coefficients):
        raise ValueError("frozen model dimension changed across split")
    eta = engine.linear_predictor(context, matrix, fit.coefficients)
    row_loss, entity_loss = entity_losses(context, eta)
    return eta, row_loss, entity_loss


def _fit_on_discovery(
    optimizer: SupportOptimizer,
    support: Support,
    *,
    closure: tuple | None = None,
) -> tuple[ModelMatrix, FitResult]:
    matrix = optimizer.engine.model_matrix(
        optimizer.context, support, forced_closure=closure
    )
    fit = fit_model_matrix(
        matrix,
        likelihood=optimizer.context.dataset.likelihood,
        tolerance=optimizer.config.solver_tolerance,
        max_iter=optimizer.config.solver_max_iter,
    )
    return matrix, fit


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
        cache_bytes=config.cache_bytes,
    )
    interim: list[tuple[SupportRecord, float, dict[str, object], tuple[str, ...]]] = []
    for record in family:
        reasons: list[str] = []
        diagnostics: dict[str, object] = {}
        full_eta, full_rows, full_entity = _evaluate_frozen(
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
        _, _, null_entity = _evaluate_frozen(
            cert_engine, certification_context, null_matrix, null_fit
        )
        f1 = one_sided_mean_test(null_entity - full_entity)
        diagnostics["f1"] = f1.__dict__
        if not f1.testable:
            reasons.append("f1_not_testable")
        rule_pvalues: list[float] = []
        rule_diagnostics: list[dict[str, object]] = []
        for root in record.support.rules:
            drop_support = _branch_drop(record.support, root)
            drop_matrix, drop_fit = _fit_on_discovery(optimizer, drop_support)
            if not drop_fit.converged:
                rule_pvalues.append(1.0)
                reasons.append(f"branch_nonconvergence:{root}")
                continue
            drop_eta, drop_rows, drop_entity = _evaluate_frozen(
                cert_engine, certification_context, drop_matrix, drop_fit
            )
            global_test = one_sided_mean_test(drop_entity - full_entity)
            footprint = cert_engine.footprint_rows(
                certification_context, root, config.early_warning_horizon
            )
            local_difference = np.zeros(len(certification_context.entity_codes), dtype=np.float64)
            probability_difference = np.zeros_like(local_difference)
            if len(footprint):
                local, _ = certification_context.rows_to_entity_time(footprint)
                np.add.at(local_difference, local, drop_rows[footprint] - full_rows[footprint])
                full_hazard = np.bincount(
                    local, weights=np.exp(np.clip(full_eta[footprint], -745.0, 700.0)),
                    minlength=len(local_difference),
                )
                drop_hazard = np.bincount(
                    local, weights=np.exp(np.clip(drop_eta[footprint], -745.0, 700.0)),
                    minlength=len(local_difference),
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
            rule_diagnostics.append({
                "rule": repr(root),
                "global": global_test.__dict__,
                "horizon": local_test.__dict__,
                "probability": probability_test.__dict__,
                "footprint_rows": int(len(footprint)),
                "pvalue": pvalue,
            })
            if not (global_test.testable and local_test.testable and probability_test.testable):
                reasons.append(f"f2_not_testable:{root}")
        diagnostics["rules"] = rule_diagnostics
        support_pvalue = max([f1.pvalue, *rule_pvalues], default=1.0)
        interim.append((record, support_pvalue, diagnostics, tuple(reasons)))
    adjusted = _holm_adjust([item[1] for item in interim])
    models: list[CertifiedModel] = []
    f0 = all(optimizer.context.dataset.f0_contract.get(name) is True for name in (
        "dynamic_predicates", "outcome_blind_predicate_construction",
        "direct_target_proxy_excluded", "strict_future_effect_required", "atomic_predicates",
    ))
    for (record, pvalue, diagnostics, reasons), adjusted_pvalue in zip(interim, adjusted, strict=True):
        f1_pvalue = float(diagnostics.get("f1", {}).get("pvalue", 1.0)) if isinstance(diagnostics.get("f1"), dict) else 1.0
        rule_pvalues = tuple(float(item["pvalue"]) for item in diagnostics.get("rules", []))
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

