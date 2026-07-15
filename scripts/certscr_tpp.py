#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from certscr.data import EventData, load_event_data
from certscr.pipeline import CertSCRConfig, CertSCRPipeline, save_result
from certscr.predicate_policy import PREDICATE_POLICIES, resolve_predicate_policy


def _default_solver_workers() -> int:
    """Use available physical cores; this changes scheduling only."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return max(1, int(physical))
    except ImportError:
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, int(os.cpu_count() or 1))


def _csv_values(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(out))


def synthetic_data(seed: int = 7, n_sequences: int = 2400) -> EventData:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    bounds: list[dict] = []
    for seq in range(n_sequences):
        has_a = bool(rng.random() < 0.78)
        has_b = bool(rng.random() < 0.68)
        a_time = int(rng.integers(2, 6)) if has_a else -1
        if has_b and has_a:
            b_time = a_time + int(rng.integers(1, 4))
        elif has_b:
            b_time = int(rng.integers(3, 9))
        else:
            b_time = -1
        for t in range(0, 25):
            pred_a = int(t == a_time)
            pred_b = int(t == b_time)
            eta = -3.8
            lag_a = t - a_time if has_a else -1
            if has_a and 1 <= lag_a <= 6:
                eta += 1.75 * (1.0 - 0.09 * (lag_a - 1))
            if has_a and has_b:
                lag_ab = t - b_time
                if 1 <= lag_ab <= 4:
                    eta -= 4.25 * (1.0 - 0.12 * (lag_ab - 1))
            probability = -np.expm1(-np.exp(eta))
            target = int(rng.random() < probability)
            # Keep one zero-valued row so sequences with no events remain part
            # of the independent-cluster sample and exposure denominator.
            if t == 0 or pred_a or pred_b or target:
                rows.append(
                    {
                        "sequence_id": f"s{seq:05d}",
                        "position": 0,
                        "month_index": t,
                        "target_token": target,
                        "pred_a": pred_a,
                        "pred_b": pred_b,
                    }
                )
        bounds.append({"sequence_id": f"s{seq:05d}", "start_month": 0, "end_month": 24})
    frame = pd.DataFrame(rows).sort_values(["sequence_id", "month_index"], kind="stable")
    frame["position"] = frame.groupby("sequence_id", sort=False).cumcount().astype(np.int32)
    return EventData.from_frame(
        frame,
        predicate_names=("pred_a", "pred_b"),
        bounds=pd.DataFrame(bounds),
    )


def run_self_test(device: str) -> dict:
    predictive_data = synthetic_data()
    financial_weights = 1.0 + (np.arange(predictive_data.n_sequences, dtype=np.float64) % 5.0) / 4.0
    data = replace(
        predictive_data,
        sequence_financial_weights=financial_weights,
        financial_weight_name="synthetic_business_loss_exposure",
    )
    config = CertSCRConfig(
        q_max=2,
        impact_lag=6,
        knot_count=3,
        max_formation_window=4,
        max_support_size=2,
        split_fractions=(0.60, 0.20, 0.20),
        alpha_fit_screen=0.10,
        alpha_family=0.10,
        solver_device=device,
        solver_dtype="float64",
        solver_max_iter=100,
        solver_tolerance=3.0e-5,
        identity_profile="exact",
        triplet_generation="all",
    )
    pipeline = CertSCRPipeline(data, rule_predicates=("pred_a", "pred_b"), config=config)
    result = pipeline.run()
    profile = {tuple(item["antecedent_names"]): item for item in result["window_profiles"] if item.get("status") == "profiled"}
    if profile[("pred_a",)]["selected_sign"] != 1:
        raise AssertionError("synthetic A excitation was not recovered")
    if profile[("pred_a", "pred_b")]["selected_sign"] != -1:
        raise AssertionError("synthetic AB inhibition was not recovered")
    if not all(value >= 0 for item in result["certification"]["all_supports"] for rule in item["rules"] for value in rule["shape"]):
        raise AssertionError("negative rule-kernel shape detected")
    for item in result["certification"]["all_supports"]:
        for rule in item["rules"]:
            if "hierarchical_conditional_contribution" in rule:
                raise AssertionError("obsolete hierarchical auxiliary gate detected")
            impact = rule.get("kernel_impact")
            if impact is not None and not np.isclose(
                abs(float(impact["integrated_log_intensity_impact"])),
                float(rule["support_amplitude"]),
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise AssertionError("reported impact and prediction-support kernel are not aligned")
    if result["certification"]["certified_count"] <= 0:
        raise AssertionError("synthetic pipeline produced no certified support")
    if not result["certification"].get("financially_grounded_loss"):
        raise AssertionError("synthetic financial loss was not used for certification")
    if result["certification"].get("financially_certified"):
        raise AssertionError(
            "zero materiality thresholds must not produce a financial-certification claim"
        )
    if result["certification"].get("loss") != "financial_weighted_tpp_nll[synthetic_business_loss_exposure]":
        raise AssertionError("unexpected synthetic financial certification loss")
    if result["certification"]["family_size"] != result["fit_screen"]["selected_support_count"]:
        raise AssertionError("certification family is not exactly the family frozen by D_fit")
    selected_identities = {
        tuple(
            (tuple(rule["antecedent_ids"]), int(rule["window"]), str(rule["sign"]))
            for rule in item["support"]
        )
        for item in result["fit_screen"]["selected_supports"]
    }
    for item in result["certification"]["all_supports"]:
        identity = tuple(
            (tuple(rule["antecedent_ids"]), int(rule["window"]), str(rule["sign"]))
            for rule in item["support"]
        )
        if identity not in selected_identities:
            raise AssertionError("certification evaluated a support not frozen by D_fit")
    desired_pair = {
        (("pred_a",), "exc"),
        (("pred_a", "pred_b"), "inh"),
    }
    matching = []
    for support in result["certification"]["certified_supports"]:
        identities = {(tuple(rule["antecedents"]), rule["sign"]) for rule in support["support"]}
        if all(identity in identities for identity in desired_pair):
            matching.append(support)
    if not matching:
        raise AssertionError("synthetic A-excitation/AB-inhibition structure was not certified")
    for support in matching:
        closure = {tuple(term["antecedents"]) for term in support["closure_terms"]}
        if closure != {("pred_b",)}:
            raise AssertionError(f"unexpected A/AB hierarchy closure: {closure}")
    if not result["ensemble"].get("fitted"):
        raise AssertionError("certified-support ensemble was not fitted")
    if not result["ensemble"].get("final_test_superiority"):
        raise AssertionError("synthetic financial ensemble did not pass the untouched contribution test")
    if result["ensemble"].get("deployment_certified"):
        raise AssertionError("deployment was certified without a pre-specified calibration equivalence gate")
    multiplicity = result["ensemble"].get("explanation_multiplicity", {})
    if multiplicity.get("certified_support_count") != result["certification"]["certified_count"]:
        raise AssertionError("ensemble explanation multiplicity is not aligned with certified supports")
    for item in multiplicity.get("rule_inclusion", []):
        refinement = item.get("support_conditioned_kernel_refinement", {})
        if len(refinement.get("signed_integrated_impacts", [])) != item.get("support_count"):
            raise AssertionError("support-conditioned kernel impacts are not aligned with rule inclusion")
        if item.get("absolute_ensemble_weight", 0.0) > 0 and (
            refinement.get("ensemble_weighted_signed_curve") is None
            or refinement.get("ensemble_weighted_signed_curve_sd") is None
        ):
            raise AssertionError("weighted kernel refinement summary is missing")
    result["self_test"] = "passed"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CertSCR-TPP certified multi-rule discovery")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/certscr_result.json"))
    predicate_group = parser.add_mutually_exclusive_group()
    predicate_group.add_argument("--rule-predicate", action="append")
    predicate_group.add_argument("--predicate-policy", choices=tuple(sorted(PREDICATE_POLICIES)))
    parser.add_argument("--control-predicate", action="append")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument(
        "--fit-negative-sample-size",
        type=int,
        help="after the full cohort split, keep every target-positive D_fit sequence and sample this many negatives with IPW",
    )
    parser.add_argument(
        "--hybrid-full-acceptance",
        action="store_true",
        help=(
            "use the sampled/IPW D_fit only for gradient dictionary pricing, then exact-fit the priced "
            "atoms, supports, and terminal kernels on the complete D_fit population"
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=111)
    parser.add_argument(
        "--financial-weight-column",
        help=(
            "sequence-level nonnegative business cost/exposure column in the sibling sequences parquet; "
            "when supplied, support certification and ensemble fitting use financially weighted TPP NLL"
        ),
    )
    parser.add_argument(
        "--mark-column",
        help=(
            "event-row column containing a list of one strictly positive monetary mark per target; "
            "enables marked-TPP discovery and financial-exposure certification"
        ),
    )
    parser.add_argument("--q-max", type=int, default=3)
    parser.add_argument("--impact-lag", type=int, default=12)
    parser.add_argument("--knots", type=int, default=4)
    parser.add_argument("--max-window", type=int, default=12)
    parser.add_argument(
        "--max-support-size",
        type=int,
        default=None,
        help=(
            "optional computational ablation; unset searches up to the complete "
            "frozen profiled-rule library and stops by exact one-exchange stationarity"
        ),
    )
    parser.add_argument("--fit-fraction", type=float, default=0.60)
    parser.add_argument("--cert-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument(
        "--stratify-target-sequences",
        action="store_true",
        help="stratify independent sequence assignment by whether a sequence contains any target event",
    )
    parser.add_argument("--support-search", choices=("active_set", "exhaustive"), default="active_set")
    parser.add_argument(
        "--identity-profile",
        choices=("exact", "score_mdl", "dictionary_mdl"),
        default="dictionary_mdl",
        help=(
            "exact fits every full-M W/sign candidate; score_mdl and dictionary_mdl use "
            "quadratic scores only for ordering, safely eliminate identities by a rigorous "
            "global MDL upper bound, and exact-fit every remaining finite W/sign working atom"
        ),
    )
    parser.add_argument(
        "--triplet-generation",
        choices=(
            "all",
            "weak_mdl_heredity",
            "connected_mdl_heredity",
            "strong_mdl_heredity",
        ),
        default="all",
        help=(
            "all prices the complete finite triplet dictionary (default, no heredity assumption); "
            "weak_mdl_heredity requires one admitted constituent pair; connected_mdl_heredity "
            "requires two; strong_mdl_heredity requires all three (heuristic ablations)"
        ),
    )
    parser.add_argument(
        "--active-start-policy",
        choices=("all_atoms", "stratified_budget"),
        default="all_atoms",
        help=(
            "all_atoms starts exact one-exchange ascent from the empty support and every "
            "profiled rule atom; stratified_budget retains the legacy --active-restarts ablation"
        ),
    )
    parser.add_argument(
        "--active-restarts",
        type=int,
        default=8,
        help="restart budget used only by --active-start-policy stratified_budget",
    )
    parser.add_argument(
        "--support-family",
        choices=("terminal_atoms", "visited_pool"),
        default="terminal_atoms",
        help=(
            "terminal_atoms certifies positive standalone atoms, strict hierarchy links, and unique local terminals; "
            "visited_pool retains the legacy family of positive intermediate search states"
        ),
    )
    parser.add_argument(
        "--max-gradient-triplets",
        type=int,
        default=None,
        help=(
            "optional heuristic cap used only with a heredity triplet generator; unset by default "
            "so no skeleton is removed by a compute budget"
        ),
    )
    parser.add_argument(
        "--support-pool-size",
        type=int,
        default=None,
        help="optional heuristic cap on non-required visited supports; unset by default",
    )
    parser.add_argument("--search-improvement-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--safe-mdl-screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "skip only supports whose global group-saturated likelihood bound proves that "
            "their best possible block-MDL is nonpositive"
        ),
    )
    parser.add_argument(
        "--no-support-conditioned-refinement",
        action="store_true",
        help="disable exact pooled-support coordinate profiling of formation window and sign",
    )
    parser.add_argument("--alpha-fit-screen", type=float, default=0.05)
    parser.add_argument("--alpha-family", type=float, default=0.05)
    parser.add_argument(
        "--joint-loss-threshold",
        "--financial-threshold",
        dest="financial_threshold",
        type=float,
        default=0.0,
        help=(
            "minimum held-out joint loss improvement; --financial-threshold is a "
            "deprecated compatibility alias"
        ),
    )
    parser.add_argument("--rule-threshold", type=float, default=0.0)
    parser.add_argument(
        "--certification-mode",
        choices=("auto", "early_warning", "predictive"),
        default="auto",
        help=(
            "auto uses marked/financial certification only with a supplied financial estimand, "
            "and otherwise certifies adverse-event early-warning rules"
        ),
    )
    parser.add_argument(
        "--adverse-event-name",
        help=(
            "pre-specified semantic name of the adverse financial target; together with a "
            "registered --predicate-policy and no unreviewed predicate controls, required for F0; "
            "the registered target-history nuisance is allowed"
        ),
    )
    parser.add_argument(
        "--early-warning-horizon",
        type=int,
        help="primary warning horizon in grid lags; defaults to --impact-lag",
    )
    parser.add_argument(
        "--early-warning-threshold",
        type=float,
        default=0.0,
        help=(
            "minimum sign-aligned entity-average adverse-event probability shift in "
            "the primary warning horizon"
        ),
    )
    parser.add_argument(
        "--target-history-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "condition every support on a shared M-knot history of prior target events "
            "(default: true; use --no-target-history-control only for first-event/renewal ablations)"
        ),
    )
    parser.add_argument("--calibration-tolerance", type=float)
    parser.add_argument(
        "--occurrence-likelihood",
        choices=("auto", "poisson", "first_event_cloglog"),
        default="auto",
        help=(
            "auto uses monthly first-event cloglog for target_process=first_event "
            "and event-time Poisson for recurrent streams"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="solver device, e.g. cpu, cuda, cuda:0, or cuda:1",
    )
    parser.add_argument("--solver-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--solver-workers",
        type=int,
        default=None,
        help=(
            "independent exact-fit workers on --device; changes scheduling only, "
            "not the evaluated dictionary, objective, or acceptance decisions"
        ),
    )
    parser.add_argument("--solver-max-iter", type=int, default=80)
    parser.add_argument("--solver-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--feature-cache-gb",
        type=float,
        default=16.0,
        help=(
            "bounded shared CPU cache for exact completion streams, raw rule responses, "
            "and dictionary-projected responses"
        ),
    )
    parser.add_argument(
        "--loss-summary-cache-gb",
        type=float,
        default=1.0,
        help=(
            "byte-bounded cache for exact repeated hierarchy-null entity losses; "
            "changes reuse only, not fitting or certification"
        ),
    )
    parser.add_argument(
        "--fit-summary-cache-gb",
        type=float,
        default=8.0,
        help=(
            "byte-bounded cache for exact full sparse summaries reused by frozen "
            "support/drop comparisons; changes memoization only"
        ),
    )
    parser.add_argument(
        "--response-workers",
        type=int,
        default=8,
        help="CPU workers that precompute exact completion streams ahead of GPU fitting",
    )
    parser.add_argument(
        "--support-device",
        action="append",
        help=(
            "exact support-fit worker device (repeat, e.g. cuda:0 and cuda:1); "
            "every support not rejected by a certified safe bound is exact-fitted"
        ),
    )
    parser.add_argument(
        "--support-workers-per-device",
        type=int,
        default=2,
        help=(
            "independent exact-support workers feeding each listed device; "
            "two overlaps CPU sparse-design preparation with GPU fitting "
            "without changing the evaluated support set or objective"
        ),
    )
    parser.add_argument(
        "--exact-identity-audit",
        action="store_true",
        help=(
            "oracle audit: treat every exact W/sign parameterization as a separate rule identity; "
            "the default estimator profiles one canonical W/sign per antecedent skeleton"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = run_self_test(args.device)
        print(json.dumps({
            "self_test": result["self_test"],
            "profiled_rules": result["profiled_rules"],
            "certified_count": result["certification"]["certified_count"],
        }, indent=2))
        return
    if args.data is None:
        raise SystemExit("--data is required unless --self-test is used")
    if args.mark_column and args.financial_weight_column:
        raise SystemExit(
            "--mark-column and legacy --financial-weight-column are mutually exclusive"
        )
    checkpoint_path = Path(f"{args.output}.checkpoint.json")
    rule_predicates = (
        list(resolve_predicate_policy(args.predicate_policy))
        if args.predicate_policy is not None
        else _csv_values(args.rule_predicate)
    )
    control_predicates = _csv_values(args.control_predicate)
    if not rule_predicates:
        raise SystemExit("at least one --rule-predicate is required")
    selected = [*control_predicates, *rule_predicates]
    data = load_event_data(
        args.data,
        predicate_names=selected,
        max_sequences=args.max_sequences,
        sample_seed=args.sample_seed,
        mark_col=args.mark_column,
        financial_weight_col=args.financial_weight_column,
    )
    config = CertSCRConfig(
        q_max=args.q_max,
        impact_lag=args.impact_lag,
        knot_count=args.knots,
        max_formation_window=args.max_window,
        max_support_size=args.max_support_size,
        split_fractions=(args.fit_fraction, args.cert_fraction, args.test_fraction),
        stratify_target_sequences=bool(args.stratify_target_sequences),
        fit_negative_sample_size=args.fit_negative_sample_size,
        split_seed=args.sample_seed,
        alpha_fit_screen=args.alpha_fit_screen,
        alpha_family=args.alpha_family,
        financial_threshold=args.financial_threshold,
        rule_threshold=args.rule_threshold,
        certification_mode=args.certification_mode,
        adverse_event_name=args.adverse_event_name,
        early_warning_horizon=args.early_warning_horizon,
        early_warning_threshold=args.early_warning_threshold,
        target_history_control=bool(args.target_history_control),
        occurrence_likelihood=args.occurrence_likelihood,
        calibration_tolerance=args.calibration_tolerance,
        solver_device=args.device,
        solver_dtype=args.solver_dtype,
        solver_workers=(
            int(args.solver_workers)
            if args.solver_workers is not None
            else 1
            if args.support_device
            else _default_solver_workers()
        ),
        solver_max_iter=args.solver_max_iter,
        solver_tolerance=args.solver_tolerance,
        feature_cache_bytes=int(args.feature_cache_gb * 1024**3),
        loss_summary_cache_bytes=int(args.loss_summary_cache_gb * 1024**3),
        fit_summary_cache_bytes=int(args.fit_summary_cache_gb * 1024**3),
        response_workers=args.response_workers,
        exhaustive_profile=bool(args.exact_identity_audit),
        support_devices=tuple(_csv_values(args.support_device)),
        support_workers_per_device=args.support_workers_per_device,
        support_search=args.support_search,
        # Oracle identity audit must enter the exact branch; merely toggling
        # exhaustive_profile while leaving dictionary_mdl selected continued
        # to price only dictionary atoms and made the flag's claim false.
        identity_profile=("exact" if args.exact_identity_audit else args.identity_profile),
        triplet_generation=args.triplet_generation,
        active_start_policy=args.active_start_policy,
        active_restarts=args.active_restarts,
        support_family=args.support_family,
        max_gradient_triplets=args.max_gradient_triplets,
        support_pool_size=args.support_pool_size,
        search_improvement_tolerance=args.search_improvement_tolerance,
        support_conditioned_refinement=not args.no_support_conditioned_refinement,
        safe_mdl_screen=bool(args.safe_mdl_screen),
    )
    checkpoint_algorithm = (
        "FR-Marked-SCR-TPP"
        if args.mark_column
        else "CER-SCR-TPP"
        if args.financial_weight_column or args.certification_mode == "predictive"
        else "EW-CertSCR-TPP"
    )
    if args.hybrid_full_acceptance:
        if args.fit_negative_sample_size is None:
            raise SystemExit("--hybrid-full-acceptance requires --fit-negative-sample-size")
        hybrid_started = time.perf_counter()
        pricing_config = replace(config, gradient_pricing_only=True)
        pricing_pipeline = CertSCRPipeline(
            data,
            rule_predicates=rule_predicates,
            control_predicates=control_predicates,
            config=pricing_config,
            predicate_policy_name=args.predicate_policy,
        )
        pricing_started = time.perf_counter()
        pricing_pipeline.fit_baseline()
        priced_rules = pricing_pipeline.profile_rule_identities()
        pricing_done = time.perf_counter()
        pricing_summary = {
            "method": "sampled_ipw_gradient_dictionary_pricing",
            "sampled_negative_sequences": pricing_pipeline.splits.fit_sampled_negative_count,
            "sampled_fit_sequences": pricing_pipeline.splits.fit.n_sequences,
            "kish_effective_sample_size": pricing_pipeline.fit_sampling_ess,
            "priced_rule_count": len(priced_rules),
            "priced_rules": [pricing_pipeline._rule_dict(rule) for rule in priced_rules],
            "window_profiles": [
                {key: value for key, value in row.items() if key != "candidates"}
                for row in pricing_pipeline.profile_logs
            ],
            "timing_seconds": pricing_done - pricing_started,
        }
        save_result(
            {
                "algorithm": checkpoint_algorithm,
                "checkpoint_stage": "hybrid_pricing_complete",
                "elapsed_seconds": pricing_done - hybrid_started,
                "hybrid_pricing": pricing_summary,
            },
            checkpoint_path,
        )
        identity_candidates = dict(pricing_pipeline.identity_candidates)
        dictionary_shapes = {
            rule: shape.copy()
            for rule, shape in pricing_pipeline.rule_dictionary_shapes.items()
        }
        del pricing_pipeline
        gc.collect()

        full_config = replace(
            config,
            fit_negative_sample_size=None,
            gradient_pricing_only=False,
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=rule_predicates,
            control_predicates=control_predicates,
            config=full_config,
            predicate_policy_name=args.predicate_policy,
        )
        pipeline.seed_profiled_library(
            priced_rules,
            identity_candidates=identity_candidates,
            dictionary_shapes=dictionary_shapes,
        )
        acceptance_started = time.perf_counter()
        full_acceptance = pipeline.accept_seeded_rules_on_fit()
        acceptance_done = time.perf_counter()
        save_result(
            {
                "algorithm": checkpoint_algorithm,
                "checkpoint_stage": "full_d_fit_acceptance_complete",
                "elapsed_seconds": acceptance_done - hybrid_started,
                "hybrid_pricing": pricing_summary,
                "hybrid_full_d_fit_acceptance": {
                    **full_acceptance,
                    "timing_seconds": acceptance_done - acceptance_started,
                },
            },
            checkpoint_path,
        )
        result = pipeline.run(checkpoint_path=checkpoint_path)
        result["hybrid_pricing"] = pricing_summary
        result["hybrid_full_d_fit_acceptance"] = {
            **full_acceptance,
            "timing_seconds": acceptance_done - acceptance_started,
        }
        result["timing_seconds"]["hybrid_end_to_end_total"] = (
            time.perf_counter() - hybrid_started
        )
    else:
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=rule_predicates,
            control_predicates=control_predicates,
            config=config,
            predicate_policy_name=args.predicate_policy,
        )
        result = pipeline.run(checkpoint_path=checkpoint_path)
    save_result(result, args.output)
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(args.output),
        "candidate_rule_count": result["candidate_rule_count"],
        "candidate_support_count": result["candidate_support_count"],
        "active_support_count": result["active_support_count"],
        "fit_screen_support_count": result["fit_screen"]["selected_support_count"],
        "certified_count": result["certification"]["certified_count"],
        "timing_seconds": result["timing_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
