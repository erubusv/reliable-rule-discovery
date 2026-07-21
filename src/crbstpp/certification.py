from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .likelihood import loss_rows
from .objective import SupportRecord
from .report import Certificate
from .reliability import (
    environment_robust_lcb as _environment_robust_lcb,
    environment_spec as _environment_spec,
    multinomial_l1_radius,
    worst_case_total_variation_mean,
)
from .response import Context, ModelMatrix, ResponseEngine
from .rules import ClosureTerm, EMPTY_SUPPORT, RuleIdentity, Support, hierarchy_closure
from .search import SupportOptimizer, support_key
from .solver import FitResult

# Backward-compatible private test hooks; implementation is shared with search.
_multinomial_l1_radius = multinomial_l1_radius
_worst_case_total_variation_mean = worst_case_total_variation_mean


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
    source: SupportRecord | None = None,
) -> tuple[ModelMatrix, FitResult]:
    resolved = hierarchy_closure(support) if closure is None else closure
    return optimizer.fit_fixed(support, resolved, source=source)


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
    # Split the declared family error budget between ordinary F1/F2 tests and
    # simultaneous F3 confidence bounds.  F3 then divides its mean-bound half
    # over every support/branch functional in the frozen family; the BHC
    # mixture-weight event is shared and paid only once.
    test_alpha = config.alpha / 2.0
    f3_alpha = config.alpha / 2.0
    environments = _environment_spec(
        certification_context, config, alpha=f3_alpha
    )
    # One support functional plus global, horizon and sign-aligned probability
    # functionals for every reported rule.
    robust_metric_count = sum(1 + 3 * len(record.support.rules) for record in family)
    robust_metric_alpha = f3_alpha / max(1, robust_metric_count)
    interim: list[tuple[SupportRecord, float, dict[str, object], tuple[str, ...]]] = []
    for record in family:
        reasons: list[str] = []
        diagnostics: dict[str, object] = {}
        full_matrix, full_entity = _evaluate_frozen(
            cert_engine, certification_context, record.matrix, record.fit
        )
        branch_specifications = []
        for root in record.support.rules:
            drop_support = _branch_drop(record.support, root)
            drop_closure = _branch_null_closure(
                record.matrix.closure, drop_support, root
            )
            branch_specifications.append((drop_support, drop_closure))
        # Build and fit every nested comparison exactly once.  Each null is a
        # lossless column projection of the already fitted full matrix inside
        # SupportOptimizer, so certification never reconstructs its kernels.
        comparison_models = optimizer.fit_fixed_many(
            [(EMPTY_SUPPORT, ()), *branch_specifications],
            sources=[record] * (1 + len(branch_specifications)),
        )
        (null_matrix, null_fit), *branch_models = comparison_models
        if not null_fit.converged:
            reasons.append("baseline_null_nonconvergence")
            interim.append((record, 1.0, diagnostics, tuple(reasons)))
            continue
        null_cert_matrix, null_entity = _evaluate_frozen(
            cert_engine, certification_context, null_matrix, null_fit
        )
        f1 = one_sided_mean_test(null_entity - full_entity)
        diagnostics["f1"] = f1.__dict__
        if not f1.testable:
            reasons.append("f1_not_testable")
        (
            support_robust_gain,
            support_environment_gains,
            support_environment_lcbs,
        ) = _environment_robust_lcb(
            null_entity - full_entity,
            environments,
            alpha=robust_metric_alpha,
        )
        support_dimension = max(0, full_matrix.dimension - null_cert_matrix.dimension)
        support_mdl_threshold = (
            optimizer.objective.penalty_for_dimension(
                record.support,
                support_dimension,
                n_entities=len(certification_context.entity_codes),
            )
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
            (
                robust_global_gain,
                global_environment_gains,
                global_environment_lcbs,
            ) = _environment_robust_lcb(
                drop_entity - full_entity,
                environments,
                alpha=robust_metric_alpha,
            )
            (
                robust_probability,
                probability_environment_gains,
                probability_environment_lcbs,
            ) = (
                _environment_robust_lcb(
                    probability_difference,
                    environments,
                    alpha=robust_metric_alpha,
                )
            )
            (
                robust_horizon_gain,
                horizon_environment_gains,
                horizon_environment_lcbs,
            ) = _environment_robust_lcb(
                local_difference,
                environments,
                alpha=robust_metric_alpha,
            )
            rule_f3_passed = bool(
                robust_global_gain > 0.0
                and robust_horizon_gain > 0.0
                and robust_probability > config.probability_materiality
            )
            rule_f3.append(rule_f3_passed)
            if robust_global_gain <= 0.0:
                reasons.append(f"f3_rule_robust_gain_nonpositive:{root}")
            if robust_horizon_gain <= 0.0:
                reasons.append(f"f3_rule_robust_horizon_nonpositive:{root}")
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
                    "f3_robust_horizon_gain": robust_horizon_gain,
                    "f3_robust_probability_contribution": robust_probability,
                    "f3_global_environment_gain_min": float(
                        np.min(global_environment_gains)
                    ),
                    "f3_global_environment_gain_max": float(
                        np.max(global_environment_gains)
                    ),
                    "f3_global_environment_lcb_min": float(
                        np.min(global_environment_lcbs)
                    ),
                    "f3_horizon_environment_gain_min": float(
                        np.min(horizon_environment_gains)
                    ),
                    "f3_horizon_environment_gain_max": float(
                        np.max(horizon_environment_gains)
                    ),
                    "f3_horizon_environment_lcb_min": float(
                        np.min(horizon_environment_lcbs)
                    ),
                    "f3_probability_environment_gain_min": float(
                        np.min(probability_environment_gains)
                    ),
                    "f3_probability_environment_gain_max": float(
                        np.max(probability_environment_gains)
                    ),
                    "f3_probability_environment_lcb_min": float(
                        np.min(probability_environment_lcbs)
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
            "ambiguity": (
                "BHC finite-sample cohort-mixture set plus simultaneous "
                "one-sided normal cohort-effect lower bounds"
            ),
            "ambiguity_l1_radius": environments.l1_radius,
            "support_robust_gain": support_robust_gain,
            "support_mdl_threshold": support_mdl_threshold,
            "support_environment_gain_min": float(np.min(support_environment_gains)),
            "support_environment_gain_max": float(np.max(support_environment_gains)),
            "support_environment_lcb_min": float(np.min(support_environment_lcbs)),
            "family_f3_alpha": f3_alpha,
            "family_robust_metric_count": robust_metric_count,
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
    ) and optimizer.context.dataset.f0_contract.get(
        # Independence is a certification prerequisite, not a dataset-schema
        # prerequisite (IBM is deliberately loadable but uncertifiable).  A
        # missing declaration must therefore fail closed, never silently pass.
        "independent_certification_units", False
    ) is True
    for (record, pvalue, diagnostics, reasons), adjusted_pvalue in zip(
        interim, adjusted, strict=True
    ):
        final_reasons = reasons if f0 else (*reasons, "f0_contract_failed")
        if adjusted_pvalue > test_alpha:
            final_reasons = (*final_reasons, "holm_family_not_significant")
        f1_pvalue = (
            float(diagnostics.get("f1", {}).get("pvalue", 1.0))
            if isinstance(diagnostics.get("f1"), dict)
            else 1.0
        )
        rule_pvalues = tuple(
            float(item["pvalue"]) for item in diagnostics.get("rules", [])
        )
        f3 = bool(diagnostics.get("f3", {}).get("passed", False))
        certified = bool(
            f0 and f3 and not final_reasons and adjusted_pvalue <= test_alpha
        )
        certificate = Certificate(
            support_key=support_key(record.support),
            f0=f0,
            f1_pvalue=f1_pvalue,
            f2_pvalues=rule_pvalues,
            f3=f3,
            family_pvalue=pvalue,
            holm_adjusted_pvalue=adjusted_pvalue,
            certified=certified,
            reasons=final_reasons,
        )
        models.append(CertifiedModel(record, certificate, diagnostics))
    certified_models = tuple(model for model in models if model.certificate.certified)
    return CertificationResult(tuple(models), certified_models, len(models))
