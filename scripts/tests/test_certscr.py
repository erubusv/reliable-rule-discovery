from __future__ import annotations

import math
import json
import os
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import certscr.pipeline as pipeline_module

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from certscr.data import (
    EventData,
    load_event_data,
    make_context,
    split_contexts,
)
from certscr.ensemble import (
    _simplex_project,
    evaluate_ensemble,
    evaluate_ensemble_sufficient,
    fit_intensity_ensemble,
)
from certscr.loss import financial_weighted_nll_loss
from certscr.marked import (
    expected_mark,
    fit_mark_head,
    make_mark_base_residualizer,
    mark_score_moments,
)
from certscr.model import (
    FitResult,
    IncrementalSupportPartition,
    PreparedFixedSupportDesign,
    aggregate_duplicate_design_rows,
    append_rules_to_incremental_partition,
    assemble_compressed_design,
    assemble_design,
    canonical_nll,
    cluster_nll,
    cloglog_event_terms,
    compress_zero_grid_rows,
    fixed_support_projected_kkt,
    fit_fixed_support,
    fit_delta_factorized_support,
    fit_sparse_delta_support,
    fit_sparse_delta_closure,
    factorized_rule_recession_columns,
    fit_constrained_prepared_batch,
    fit_unconstrained_prepared_batch,
    _fit_fixed_support_native64,
    _fit_fixed_support_numpy64,
    group_saturated_poisson_lower_bound,
    prepare_fixed_support_design,
    prepare_delta_factorized_support_design,
    prepare_sparse_delta_support_design,
    prepare_sparse_nuisance_partition,
    sparse_delta_rule_recession_columns,
    project_prepared_support_design,
    refine_sparse_nuisance_partition,
    update_incremental_support_partition,
    _factorized_objective_grad_hessian_numpy,
    _factorized_objective_numpy,
    _prepared_objective_grad_hessian_numpy,
    _prepared_objective_numpy,
    predict_eta,
)
from certscr.native import (
    batched_sparse_rule_moments,
    sorted_unique_int64_union,
    update_sparse_design_partitioned,
)
from certscr.occurrence import (
    RuleIdentity,
    RuleOccurrenceEngine,
    SourceEvents,
    SparseKernelResponse,
    make_triangular_basis,
)
from certscr.pipeline import CertSCRConfig, CertSCRPipeline, SupportRecord, save_result
from certscr.statistics import (
    cluster_directional_score,
    efficient_information_matrix,
    holm_adjust,
    one_sided_mean_test,
    one_sided_mean_test_zero_padded,
    student_t_cdf,
    student_t_ppf,
)
from certscr_tpp import synthetic_data
from certscr.predicate_policy import (
    FREDDIE_PRIMITIVE_DYNAMIC_V3,
    FREDDIE_PRIMITIVE_DYNAMIC_V4,
    FREDDIE_STRUCTURAL_DYNAMIC_V2,
    HOME_CREDIT_BEHAVIORAL_NONPROXY,
    HOME_CREDIT_BEHAVIORAL_NONPROXY_EXPANDED,
    IBM_AML_DYNAMIC_NONPROXY_V2,
    IBM_AML_DYNAMIC_NONPROXY_V3,
    IBM_AML_PRIMITIVE_DYNAMIC_V1,
    IBM_AML_TYPOLOGY_DYNAMIC_V1,
    PredicatePolicyContract,
    resolve_predicate_policy,
    resolve_predicate_policy_contract,
)
from preprocess_home_credit_tpp import (
    BEHAVIORAL_NONPROXY_EXPANDED_PREDICATES,
    BEHAVIORAL_NONPROXY_PREDICATES,
    selected_predicates,
)
from preprocess_ibm_aml_tpp import compute_source_prior_incoming, infer_currency_usd_rates
from preprocess_freddiemac_dynamic_events import (
    QuarterFile,
    build_sequence_outputs,
    observable_sequence_prefix,
)


class StatisticsTests(unittest.TestCase):
    def test_native_wide_sorted_union_matches_numpy(self) -> None:
        rng = np.random.default_rng(29)
        arrays = [
            np.unique(rng.integers(0, 20000, size=700, dtype=np.int64))
            for _ in range(100)
        ]
        union = sorted_unique_int64_union(arrays, allow_wide=True)
        if union is None:
            self.skipTest("native compiler is unavailable")
        np.testing.assert_array_equal(
            union, np.unique(np.concatenate(arrays))
        )

    def test_native_batched_rule_moments_match_numpy(self) -> None:
        rng = np.random.default_rng(17)
        grid = rng.random((257, 4), dtype=np.float32)
        grid_weights = rng.random((257, 5))
        events = rng.random((31, 4), dtype=np.float32)
        event_first = rng.normal(size=(31, 5))
        event_second = np.abs(rng.normal(size=(31, 5)))
        native = batched_sparse_rule_moments(
            grid,
            grid_weights.T,
            events,
            event_first.T,
            event_second.T,
            include_event_second=True,
            worker_count=4,
        )
        if native is None:
            self.skipTest("native compiler is unavailable")
        gradient, information = native
        grid64 = grid.astype(np.float64)
        events64 = events.astype(np.float64)
        expected_gradient = (
            grid64.T @ grid_weights + events64.T @ event_first
        )
        expected_information = np.zeros((5, 4, 4), dtype=np.float64)
        for left in range(4):
            for right in range(left + 1):
                values = (
                    (grid64[:, left] * grid64[:, right]) @ grid_weights
                    + (events64[:, left] * events64[:, right])
                    @ event_second
                )
                expected_information[:, left, right] = values
                expected_information[:, right, left] = values
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=2e-13, atol=2e-11
        )
        np.testing.assert_allclose(
            information,
            expected_information,
            rtol=2e-13,
            atol=2e-11,
        )

    def test_cloglog_event_terms_remain_finite_below_exp_underflow(self) -> None:
        loss, gradient, hessian = cloglog_event_terms(
            np.asarray([-1000.0], dtype=np.float64)
        )
        self.assertEqual(float(loss[0]), 1000.0)
        self.assertEqual(float(gradient[0]), -1.0)
        self.assertEqual(float(hessian[0]), 0.0)

    def test_native_cone_solver_matches_portable_cloglog_optimum(self) -> None:
        rows = []
        bounds = []
        for sequence in range(80):
            target_time = 4 + sequence % 3 if sequence % 5 == 0 else None
            end_time = target_time if target_time is not None else 8 + sequence % 3
            serialized = []
            for time in range(end_time + 1):
                source = int(time in {1, 3} and time < end_time and sequence % 3 != 0)
                target = int(target_time is not None and time == target_time)
                if source or target:
                    serialized.append(
                        {
                            "sequence_id": f"n{sequence}",
                            "position": len(serialized),
                            "month_index": time,
                            "target_token": target,
                            "a": source,
                        }
                    )
            rows.extend(serialized)
            bounds.append(
                {
                    "sequence_id": f"n{sequence}",
                    "start_month": 0,
                    "end_month": end_time,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
        )
        ctx = make_context(data, "native-cloglog", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=4, knot_count=2)
        rule = RuleIdentity((0,), 0, 1)
        feature = engine.sparse_response(ctx, rule.antecedent, rule.window)
        prepared = prepare_fixed_support_design(
            ctx,
            (),
            (feature,),
            (rule,),
            occurrence_likelihood="first_event_cloglog",
        )
        native = _fit_fixed_support_native64(
            prepared,
            (rule,),
            (),
            max_iter=120,
            tolerance=1.0e-8,
            initial=None,
        )
        portable = _fit_fixed_support_numpy64(
            prepared,
            (rule,),
            (),
            max_iter=120,
            tolerance=1.0e-8,
            initial=None,
        )
        self.assertIsNotNone(native)
        self.assertTrue(portable.converged)
        self.assertAlmostEqual(native.nll, portable.nll, places=10)  # type: ignore[union-attr]
        self.assertTrue(
            np.allclose(native.theta, portable.theta, rtol=1.0e-9, atol=1.0e-9)  # type: ignore[union-attr]
        )

    def test_sparse_event_storage_preserves_dense_risk_process(self) -> None:
        dense_rows = []
        bounds = []
        for sequence, end_time, target_time in (
            ("s0", 6, 5),
            ("s1", 7, None),
            ("s2", 4, 4),
        ):
            for time in range(end_time + 1):
                dense_rows.append(
                    {
                        "sequence_id": sequence,
                        "position": time,
                        "month_index": time,
                        "target_token": int(target_time == time),
                        "a": int(time in {1, 3} and time != target_time),
                    }
                )
            bounds.append(
                {
                    "sequence_id": sequence,
                    "start_month": 0,
                    "end_month": end_time,
                }
            )
        dense_frame = pd.DataFrame(dense_rows)
        sparse_frame = dense_frame.loc[
            dense_frame[["target_token", "a"]].any(axis=1)
        ].copy()
        sparse_frame["position"] = sparse_frame.groupby(
            "sequence_id", sort=False
        ).cumcount()
        bound_frame = pd.DataFrame(bounds)
        dense = EventData.from_frame(
            dense_frame,
            predicate_names=("a",),
            bounds=bound_frame,
        )
        sparse = EventData.from_frame(
            sparse_frame,
            predicate_names=("a",),
            bounds=bound_frame,
        )
        dense_ctx = make_context(dense, "dense", np.arange(dense.n_sequences))
        sparse_ctx = make_context(sparse, "sparse", np.arange(sparse.n_sequences))
        self.assertTrue(np.array_equal(dense_ctx.grid_offsets, sparse_ctx.grid_offsets))
        self.assertTrue(np.array_equal(dense_ctx.event_times, sparse_ctx.event_times))
        dense_engine = RuleOccurrenceEngine(dense, lag=4, knot_count=2)
        sparse_engine = RuleOccurrenceEngine(sparse, lag=4, knot_count=2)
        dense_response = dense_engine.sparse_response(dense_ctx, (0,), 0)
        sparse_response = sparse_engine.sparse_response(sparse_ctx, (0,), 0)
        self.assertTrue(
            np.array_equal(dense_response.grid_indices, sparse_response.grid_indices)
        )
        self.assertTrue(
            np.array_equal(dense_response.grid_values, sparse_response.grid_values)
        )

    def test_freddie_post_target_gap_cannot_exclude_a_target_history(self) -> None:
        frame = pd.DataFrame(
            {
                "loan_id": ["a", "a", "a", "b", "b", "c", "c", "c"],
                "month_index": [1, 2, 4, 1, 3, 1, 2, 4],
                "is_target_month": [False, True, False, False, True, False, False, False],
            }
        )
        prefix, gap_loans = observable_sequence_prefix(frame)
        self.assertEqual(
            prefix.loc[prefix["loan_id"].eq("a"), "month_index"].tolist(),
            [1, 2],
        )
        self.assertEqual(
            int(prefix.loc[prefix["loan_id"].eq("a"), "is_target_month"].sum()),
            1,
        )
        self.assertNotIn("a", set(gap_loans))
        self.assertEqual(
            prefix.loc[prefix["loan_id"].eq("b"), "month_index"].tolist(),
            [1],
        )
        self.assertEqual(
            prefix.loc[prefix["loan_id"].eq("c"), "month_index"].tolist(),
            [1, 2],
        )
        self.assertEqual(set(gap_loans), {"b", "c"})

    def test_freddie_unmarked_preprocessing_does_not_require_a_upb_mark(self) -> None:
        raw = pd.DataFrame(
            {
                "loan_id": ["a", "a"],
                "monthly_reporting_period": [202301, 202302],
                "current_actual_upb": [100.0, np.nan],
                "current_interest_rate": [5.0, 5.0],
                "current_deferred_upb": [0.0, 0.0],
                "loan_age": [1.0, 2.0],
                "modification_flag": ["", ""],
                "zero_balance_code": ["", ""],
                "deferred_payment_plan": ["", ""],
                "eltv": [80.0, 80.0],
                "delinquency_due_to_disaster": ["", ""],
                "borrower_assistance_status_code": ["", ""],
                "month_index": [2023 * 12 + 1, 2023 * 12 + 2],
                "reporting_year": [2023, 2023],
                "reporting_month": [1, 2],
                "delq_status": ["0", "3"],
                "delq_numeric": [0.0, 3.0],
                "is_target_month": [False, True],
                "is_terminated": [False, False],
            }
        )
        quarter = QuarterFile(2023, 1, Path("unused.txt"))
        import preprocess_freddiemac_dynamic_events as freddie_preprocess

        original = freddie_preprocess.read_performance
        freddie_preprocess.read_performance = lambda _path: raw.copy()
        try:
            _sequences, months, _tokens, metadata = build_sequence_outputs(quarter)
            self.assertNotIn("target_mark_values", months.columns)
            self.assertIsNone(metadata["target_mark"])
            self.assertEqual(metadata["risk_month_rows"], 2)
            self.assertEqual(metadata["sequence_month_rows"], 1)
            self.assertEqual(months["target_token"].tolist(), [1])
            with self.assertRaisesRegex(ValueError, "nonpositive current UPB mark"):
                build_sequence_outputs(quarter, emit_target_mark=True)
        finally:
            freddie_preprocess.read_performance = original

    def test_first_event_cloglog_intercept_and_grouped_likelihood_are_exact(self) -> None:
        frame = pd.DataFrame(
            [
                {"sequence_id": "s0", "position": 0, "month_index": 0, "target_token": 0, "a": 0},
                {"sequence_id": "s0", "position": 1, "month_index": 2, "target_token": 1, "a": 0},
                {"sequence_id": "s1", "position": 0, "month_index": 0, "target_token": 0, "a": 0},
                {"sequence_id": "s1", "position": 1, "month_index": 1, "target_token": 1, "a": 0},
            ]
        )
        bounds = pd.DataFrame(
            [
                {"sequence_id": "s0", "start_month": 0, "end_month": 2},
                {"sequence_id": "s1", "start_month": 0, "end_month": 2},
            ]
        )
        data = EventData.from_frame(frame, predicate_names=("a",), bounds=bounds)
        ctx = make_context(data, "first-event", np.asarray([0, 1]))
        prepared = prepare_fixed_support_design(
            ctx,
            (),
            (),
            (),
            occurrence_likelihood="first_event_cloglog",
        )
        self.assertEqual(float(np.sum(prepared.grid_weights)), 4.0)
        self.assertEqual(float(np.sum(prepared.event_weights)), 2.0)
        fit = fit_fixed_support(
            ctx,
            (),
            (),
            (),
            device="cpu",
            dtype="float64",
            max_iter=100,
            tolerance=1.0e-9,
            prepared_design=prepared,
            occurrence_likelihood="first_event_cloglog",
        )
        expected_alpha = math.log(math.log1p(2.0 / 4.0))
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.alpha, expected_alpha, places=12)
        eta = np.full(ctx.n_queries, fit.alpha, dtype=np.float64)
        direct = canonical_nll(
            eta,
            ctx,
            occurrence_likelihood="first_event_cloglog",
        )
        clustered = cluster_nll(
            eta,
            ctx,
            occurrence_likelihood="first_event_cloglog",
        )
        self.assertAlmostEqual(direct, float(np.sum(clustered)), places=12)
        self.assertAlmostEqual(direct, fit.nll, places=12)

    def test_batched_first_event_cloglog_nulls_match_scalar_fits(self) -> None:
        data = EventData.from_frame(
            pd.DataFrame(
                [
                    {
                        "sequence_id": "batch-0",
                        "position": 0,
                        "month_index": 1,
                        "target_token": 1,
                        "a": 0,
                    },
                    {
                        "sequence_id": "batch-1",
                        "position": 0,
                        "month_index": 2,
                        "target_token": 1,
                        "a": 0,
                    },
                ]
            ),
            predicate_names=("a",),
            bounds=pd.DataFrame(
                [
                    {"sequence_id": "batch-0", "start_month": 0, "end_month": 1},
                    {"sequence_id": "batch-1", "start_month": 0, "end_month": 2},
                ]
            ),
        )
        ctx = make_context(data, "batched-cloglog", np.arange(data.n_sequences))
        closure_terms = [(((0,), 0),), (((0,), 2),)]
        controls = [
            (np.zeros((ctx.n_queries, 1), dtype=np.float64),),
            (np.zeros((ctx.n_queries, 1), dtype=np.float64),),
        ]
        prepared = [
            PreparedFixedSupportDesign(
                design=np.asarray(design, dtype=np.float64),
                n_events=3,
                event_weights=np.asarray(event_weights, dtype=np.float64),
                grid_weights=np.asarray(grid_weights, dtype=np.float64),
                constrained_start=2,
                control_width=1,
                knot_count=0,
                active_grid_rows=len(grid_weights),
                rules=(),
                occurrence_likelihood="first_event_cloglog",
            )
            for design, event_weights, grid_weights in (
                (
                    [
                        [1.0, -0.8],
                        [1.0, 0.1],
                        [1.0, 0.9],
                        [1.0, -1.0],
                        [1.0, -0.2],
                        [1.0, 0.7],
                        [1.0, 1.2],
                    ],
                    [4.0, 6.0, 5.0],
                    [18.0, 22.0, 17.0, 11.0],
                ),
                (
                    [
                        [1.0, -0.5],
                        [1.0, 0.3],
                        [1.0, 1.1],
                        [1.0, -1.2],
                        [1.0, 0.0],
                        [1.0, 0.6],
                    ],
                    [3.0, 7.0, 4.0],
                    [21.0, 19.0, 16.0],
                ),
            )
        ]
        batched = fit_unconstrained_prepared_batch(
            ctx,
            controls,
            prepared,
            closure_terms,
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-9,
        )
        scalar = [
            fit_fixed_support(
                ctx,
                control,
                (),
                (),
                device="cpu",
                dtype="float64",
                max_iter=120,
                tolerance=1.0e-9,
                closure_terms=terms,
                prepared_design=item,
                occurrence_likelihood="first_event_cloglog",
            )
            for control, terms, item in zip(
                controls, closure_terms, prepared, strict=True
            )
        ]
        for item, batch_fit, scalar_fit in zip(
            prepared, batched, scalar, strict=True
        ):
            self.assertTrue(batch_fit.converged)
            self.assertTrue(scalar_fit.converged)
            self.assertAlmostEqual(batch_fit.nll, scalar_fit.nll, places=9)
            self.assertAlmostEqual(batch_fit.alpha, scalar_fit.alpha, places=8)
            batch_eta = item.design.astype(np.float64) @ np.concatenate(
                ([batch_fit.alpha], batch_fit.gamma)
            )
            scalar_eta = item.design.astype(np.float64) @ np.concatenate(
                ([scalar_fit.alpha], scalar_fit.gamma)
            )
            self.assertTrue(
                np.allclose(batch_eta, scalar_eta, rtol=0.0, atol=1.0e-8),
                msg=str(float(np.max(np.abs(batch_eta - scalar_eta)))),
            )

    def test_batched_constrained_supports_match_scalar_cone_fits(self) -> None:
        data = EventData.from_frame(
            pd.DataFrame(
                [
                    {
                        "sequence_id": "cone-0",
                        "position": 0,
                        "month_index": 1,
                        "target_token": 1,
                        "a": 0,
                    },
                    {
                        "sequence_id": "cone-1",
                        "position": 0,
                        "month_index": 2,
                        "target_token": 1,
                        "a": 0,
                    },
                ]
            ),
            predicate_names=("a",),
            bounds=pd.DataFrame(
                [
                    {"sequence_id": "cone-0", "start_month": 0, "end_month": 1},
                    {"sequence_id": "cone-1", "start_month": 0, "end_month": 2},
                ]
            ),
        )
        ctx = make_context(data, "batched-cone", np.arange(data.n_sequences))
        rule = RuleIdentity((0,), 0, 1)
        terms = [(((0,), 0),), (((0,), 1),)]
        controls = [(), ()]
        prepared = [
            PreparedFixedSupportDesign(
                design=np.asarray(design, dtype=np.float64),
                n_events=2,
                event_weights=np.asarray([4.0, 5.0], dtype=np.float64),
                grid_weights=np.asarray([18.0, 20.0, 16.0], dtype=np.float64),
                constrained_start=2,
                control_width=1,
                knot_count=1,
                active_grid_rows=3,
                rules=(rule,),
                occurrence_likelihood="first_event_cloglog",
            )
            for design in (
                [
                    [1.0, -0.4, 1.0],
                    [1.0, 0.6, 0.9],
                    [1.0, -1.0, 0.1],
                    [1.0, 0.0, 0.2],
                    [1.0, 1.0, 0.1],
                ],
                [
                    [1.0, -0.4, 0.0],
                    [1.0, 0.6, 0.1],
                    [1.0, -1.0, 1.0],
                    [1.0, 0.0, 0.9],
                    [1.0, 1.0, 1.0],
                ],
            )
        ]
        batched = fit_constrained_prepared_batch(
            ctx,
            controls,
            prepared,
            terms,
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-9,
        )
        scalar = [
            fit_fixed_support(
                ctx,
                control,
                (),
                (rule,),
                device="cpu",
                dtype="float64",
                max_iter=120,
                tolerance=1.0e-9,
                closure_terms=item_terms,
                prepared_design=item,
                occurrence_likelihood="first_event_cloglog",
            )
            for control, item_terms, item in zip(
                controls, terms, prepared, strict=True
            )
        ]
        for batch_fit, scalar_fit in zip(batched, scalar, strict=True):
            self.assertTrue(batch_fit.converged)
            self.assertTrue(scalar_fit.converged)
            self.assertLessEqual(batch_fit.kkt_residual, 1.0e-9)
            np.testing.assert_allclose(
                batch_fit.gamma, scalar_fit.gamma, rtol=1.0e-7, atol=1.0e-8
            )
            np.testing.assert_allclose(
                batch_fit.theta, scalar_fit.theta, rtol=1.0e-7, atol=1.0e-8
            )
            self.assertAlmostEqual(batch_fit.nll, scalar_fit.nll, places=8)
        if torch is not None and torch.cuda.is_available():
            gpu_batched = fit_constrained_prepared_batch(
                ctx,
                controls,
                prepared,
                terms,
                device="cuda:0",
                dtype="float32",
                max_iter=120,
                tolerance=1.0e-9,
            )
            for gpu_fit, scalar_fit in zip(gpu_batched, scalar, strict=True):
                self.assertTrue(gpu_fit.converged)
                self.assertLessEqual(gpu_fit.kkt_residual, 1.0e-9)
                self.assertAlmostEqual(gpu_fit.nll, scalar_fit.nll, places=8)

    def test_freddie_primitive_dictionary_has_thirteen_unique_atoms(self) -> None:
        self.assertEqual(len(FREDDIE_PRIMITIVE_DYNAMIC_V3), 13)
        self.assertEqual(len(set(FREDDIE_PRIMITIVE_DYNAMIC_V3)), 13)
        contract = resolve_predicate_policy_contract("freddie_primitive_dynamic_v3")
        self.assertTrue(contract.atomic_events)
        self.assertTrue(contract.f0_eligible)

    def test_freddie_financial_primitive_v4_has_twelve_unique_atoms(self) -> None:
        self.assertEqual(len(FREDDIE_PRIMITIVE_DYNAMIC_V4), 12)
        self.assertEqual(len(set(FREDDIE_PRIMITIVE_DYNAMIC_V4)), 12)
        contract = resolve_predicate_policy_contract(
            "freddie_primitive_dynamic_v4"
        )
        self.assertEqual(contract.predicates, FREDDIE_PRIMITIVE_DYNAMIC_V4)
        self.assertTrue(contract.atomic_events)
        self.assertTrue(contract.f0_eligible)

    def test_ordered_group_split_keeps_cohorts_disjoint_and_ordered(self) -> None:
        rows = []
        bounds = []
        for sequence in range(6):
            group = 1 + sequence // 2
            rows.append(
                {
                    "sequence_id": f"g{sequence}",
                    "position": 0,
                    "month_index": 0,
                    "target_token": int(sequence % 2 == 0),
                    "a": 0,
                }
            )
            bounds.append(
                {
                    "sequence_id": f"g{sequence}",
                    "start_month": 0,
                    "end_month": 1,
                    "vintage": group,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
            split_group_col="vintage",
        )
        splits = split_contexts(
            data,
            fractions=(1 / 3, 1 / 3, 1 / 3),
            strategy="ordered_group",
        )
        self.assertEqual(splits.split_strategy, "ordered_group")
        self.assertEqual(splits.split_groups, ((1,), (2,), (3,)))
        observed = [
            set(data.sequence_split_groups[ctx.global_sequence_ids].tolist())
            for ctx in (splits.fit, splits.cert, splits.test)
        ]
        self.assertEqual(observed, [{1}, {2}, {3}])

    def test_first_event_loan_age_baseline_is_registered_nuisance(self) -> None:
        rows = []
        bounds = []
        for sequence in range(12):
            target_time = 5 if sequence % 3 == 0 else None
            end = target_time if target_time is not None else 8
            rows.append(
                {
                    "sequence_id": f"a{sequence}",
                    "position": 0,
                    "month_index": target_time if target_time is not None else 1,
                    "target_token": int(target_time is not None),
                    "a": 0,
                }
            )
            bounds.append(
                {
                    "sequence_id": f"a{sequence}",
                    "start_month": 0,
                    "end_month": end,
                    "start_loan_age": 0,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
            start_age_col="start_loan_age",
            preprocessing_provenance={"target_process": "first_event"},
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                target_history_control=False,
                loan_age_baseline=True,
                occurrence_likelihood="first_event_cloglog",
            ),
        )
        self.assertEqual(pipeline.loan_age_baseline_milestones, (0, 4))
        self.assertEqual(len(pipeline.loan_age_baseline_source_ids), 2)
        self.assertTrue(
            set(pipeline.loan_age_baseline_source_ids).isdisjoint(
                pipeline.rule_source_ids
            )
        )
        fit = pipeline.fit_baseline()
        self.assertTrue(fit.converged)
        self.assertEqual(len(fit.gamma), 4)

    def test_zero_padded_mean_test_matches_explicit_dense_test(self) -> None:
        rng = np.random.default_rng(20260715)
        for total_count in (2, 7, 1000):
            for active_count in (0, 1, min(5, total_count)):
                values = rng.normal(size=active_count)
                dense = np.zeros(total_count, dtype=np.float64)
                dense[:active_count] = values
                expected = one_sided_mean_test(
                    dense,
                    null=0.013,
                    alpha=0.025,
                )
                actual = one_sided_mean_test_zero_padded(
                    values,
                    total_count=total_count,
                    null=0.013,
                    alpha=0.025,
                )
                self.assertEqual(actual.n_clusters, expected.n_clusters)
                self.assertAlmostEqual(actual.estimate, expected.estimate, places=14)
                self.assertAlmostEqual(
                    actual.standard_error,
                    expected.standard_error,
                    places=14,
                )
                self.assertAlmostEqual(actual.statistic, expected.statistic, places=12)
                self.assertAlmostEqual(actual.p_value, expected.p_value, places=14)
                self.assertAlmostEqual(actual.lower_bound, expected.lower_bound, places=14)

    def test_zero_padded_mean_test_preserves_degenerate_and_invalid_cases(self) -> None:
        for values, total_count in (([], 10), ([1.0], 1), ([], 0)):
            dense = np.pad(
                np.asarray(values, dtype=np.float64),
                (0, max(0, total_count - len(values))),
            )
            expected = one_sided_mean_test(dense, null=0.0, alpha=0.05)
            actual = one_sided_mean_test_zero_padded(
                values,
                total_count=total_count,
                null=0.0,
                alpha=0.05,
            )
            self.assertEqual(actual.n_clusters, expected.n_clusters)
            if math.isnan(expected.estimate):
                self.assertTrue(math.isnan(actual.estimate))
            else:
                self.assertEqual(actual.estimate, expected.estimate)
            self.assertEqual(actual.standard_error, expected.standard_error)
            self.assertEqual(actual.statistic, expected.statistic)
            self.assertEqual(actual.p_value, expected.p_value)
            self.assertEqual(actual.lower_bound, expected.lower_bound)

        invalid = one_sided_mean_test_zero_padded(
            [math.nan], total_count=4
        )
        self.assertEqual(invalid.p_value, 1.0)
        self.assertEqual(invalid.n_clusters, 4)
        self.assertTrue(math.isnan(invalid.estimate))

    def test_duplicate_design_aggregation_preserves_poisson_moments(self) -> None:
        design = np.asarray(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 3.0],
                [1.0, 3.0],
            ],
            dtype=np.float32,
        )
        event_weights = np.asarray([0.4, 1.1, 2.0], dtype=np.float64)
        grid_weights = np.asarray([0.5, 1.5, 0.2, 0.8], dtype=np.float64)
        grouped, n_events, grouped_events, grouped_grid = aggregate_duplicate_design_rows(
            design, 3, event_weights, grid_weights
        )
        self.assertEqual(n_events, 2)
        self.assertEqual(len(grouped), 4)
        beta = np.asarray([-0.3, 0.17], dtype=np.float64)

        def moments(
            x: np.ndarray,
            boundary: int,
            event_mass: np.ndarray,
            grid_mass: np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray]:
            eta = x.astype(np.float64) @ beta
            intensity = grid_mass * np.exp(eta[boundary:])
            value = float(np.sum(intensity) - np.dot(event_mass, eta[:boundary]))
            gradient = (
                x[boundary:].astype(np.float64).T @ intensity
                - x[:boundary].astype(np.float64).T @ event_mass
            )
            fisher = x[boundary:].astype(np.float64).T @ (
                intensity[:, None] * x[boundary:].astype(np.float64)
            )
            return value, gradient, fisher

        original = moments(design, 3, event_weights, grid_weights)
        reduced = moments(grouped, n_events, grouped_events, grouped_grid)
        self.assertAlmostEqual(original[0], reduced[0], places=14)
        self.assertTrue(np.allclose(original[1], reduced[1], rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(original[2], reduced[2], rtol=1e-14, atol=1e-14))

    def test_prepared_design_reuse_preserves_fit_and_saturated_bound(self) -> None:
        data = synthetic_data(seed=613, n_sequences=75)
        ctx = make_context(data, "prepared-design", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=4, knot_count=2)
        rule = RuleIdentity((0,), 0, 1)
        controls = (engine.sparse_response(ctx, (1,), 0),)
        features = [engine.sparse_response(ctx, (0,), 0)]
        prepared = prepare_fixed_support_design(ctx, controls, features, (rule,))
        ordinary_bound = group_saturated_poisson_lower_bound(
            ctx, controls, features, (rule,)
        )
        reused_bound = group_saturated_poisson_lower_bound(
            ctx,
            controls,
            features,
            (rule,),
            prepared_design=prepared,
        )
        self.assertEqual(ordinary_bound.finite, reused_bound.finite)
        self.assertAlmostEqual(ordinary_bound.lower_bound, reused_bound.lower_bound, places=12)
        ordinary = fit_fixed_support(
            ctx,
            controls,
            features,
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
        )
        reused = fit_fixed_support(
            ctx,
            controls,
            features,
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=prepared,
        )
        self.assertEqual(ordinary.converged, reused.converged)
        self.assertAlmostEqual(ordinary.nll, reused.nll, places=12)
        self.assertTrue(np.allclose(ordinary.gamma, reused.gamma, rtol=0.0, atol=1e-12))
        self.assertTrue(np.allclose(ordinary.theta, reused.theta, rtol=0.0, atol=1e-12))
        certified, residual, objective = fixed_support_projected_kkt(
            prepared,
            reused,
            tolerance=1.0e-7,
        )
        self.assertTrue(certified)
        self.assertLessEqual(residual, 1.0e-7)
        self.assertAlmostEqual(objective, reused.nll, places=10)

    def test_fitted_null_hessian_reuse_preserves_child_optimum(self) -> None:
        data = synthetic_data(seed=1607, n_sequences=115)
        ctx = make_context(data, "null-hessian-reuse", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=5, knot_count=2)
        controls = (
            engine.sparse_response(ctx, (1,), 0),
            engine.sparse_response(ctx, (0,), 1),
        )
        rule = RuleIdentity((0, 1), 2, -1)
        feature = engine.sparse_response(ctx, rule.antecedent, rule.window).projected(
            np.asarray([0.35, 0.65], dtype=np.float64)
        )
        null_prepared = prepare_fixed_support_design(ctx, controls, (), ())
        child_prepared = prepare_fixed_support_design(
            ctx, controls, (feature,), (rule,)
        )
        null = _fit_fixed_support_numpy64(
            null_prepared,
            (),
            (),
            max_iter=120,
            tolerance=1.0e-8,
            initial=None,
        )
        self.assertTrue(null.converged)
        self.assertIsNotNone(null.solver_hessian)
        reused = _fit_fixed_support_numpy64(
            child_prepared,
            (rule,),
            (),
            max_iter=120,
            tolerance=1.0e-8,
            initial=null,
        )
        cold_state = replace(null, solver_hessian=None)
        ordinary = _fit_fixed_support_numpy64(
            child_prepared,
            (rule,),
            (),
            max_iter=120,
            tolerance=1.0e-8,
            initial=cold_state,
        )
        self.assertTrue(reused.converged)
        self.assertTrue(ordinary.converged)
        self.assertAlmostEqual(reused.nll, ordinary.nll, places=10)
        self.assertTrue(
            np.allclose(reused.gamma, ordinary.gamma, rtol=0.0, atol=2.0e-8)
        )
        self.assertTrue(
            np.allclose(reused.theta, ordinary.theta, rtol=0.0, atol=2.0e-8)
        )
        self.assertLessEqual(reused.kkt_residual, 1.0e-8)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is unavailable",
    )
    def test_cuda_first_event_cloglog_matches_host_certificate(self) -> None:
        rule = RuleIdentity((0,), 0, 1)
        prepared = PreparedFixedSupportDesign(
            design=np.asarray(
                [
                    [1.0, 0.2],
                    [1.0, 1.0],
                    [1.0, 1.5],
                    [1.0, 0.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                    [1.0, 2.0],
                ],
                dtype=np.float32,
            ),
            n_events=3,
            event_weights=np.ones(3, dtype=np.float64),
            grid_weights=np.full(4, 5.0, dtype=np.float64),
            constrained_start=1,
            control_width=0,
            knot_count=1,
            active_grid_rows=4,
            rules=(rule,),
            occurrence_likelihood="first_event_cloglog",
        )
        host = fit_fixed_support(
            None,
            (),
            (),
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-8,
            prepared_design=prepared,
            occurrence_likelihood="first_event_cloglog",
        )
        device = fit_fixed_support(
            None,
            (),
            (),
            (rule,),
            device="cuda:0",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-8,
            prepared_design=prepared,
            occurrence_likelihood="first_event_cloglog",
        )
        self.assertTrue(host.converged)
        self.assertTrue(device.converged)
        self.assertAlmostEqual(device.nll, host.nll, places=11)
        self.assertTrue(
            np.allclose(device.theta, host.theta, rtol=0.0, atol=1.0e-10)
        )

    def test_sparse_nuisance_partition_refinement_preserves_exact_fit(self) -> None:
        data = synthetic_data(seed=817, n_sequences=110)
        ctx = make_context(data, "partition-refinement", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=4, knot_count=2)
        weights = 0.75 + (np.arange(ctx.n_sequences, dtype=np.float64) % 5) / 8.0
        fixed = (engine.sparse_response(ctx, (1,), 0),)
        dynamic = (engine.sparse_response(ctx, (0,), 1),)
        rule = RuleIdentity((0, 1), 2, -1)
        feature = engine.sparse_response(ctx, rule.antecedent, rule.window)
        direct = prepare_fixed_support_design(
            ctx,
            (*fixed, *dynamic),
            (feature,),
            (rule,),
            cluster_weights=weights,
        )
        base = prepare_sparse_nuisance_partition(
            ctx,
            fixed,
            cluster_weights=weights,
        )
        refined = refine_sparse_nuisance_partition(
            ctx,
            base,
            dynamic,
            (feature,),
            (rule,),
            cluster_weights=weights,
        )
        direct_fit = fit_fixed_support(
            ctx,
            (),
            (),
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=direct,
        )
        refined_fit = fit_fixed_support(
            ctx,
            (),
            (),
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=refined,
        )
        self.assertTrue(direct_fit.converged)
        self.assertTrue(refined_fit.converged)
        self.assertAlmostEqual(direct_fit.nll, refined_fit.nll, places=10)
        self.assertTrue(
            np.allclose(direct_fit.gamma, refined_fit.gamma, rtol=0.0, atol=1e-9)
        )
        self.assertTrue(
            np.allclose(direct_fit.theta, refined_fit.theta, rtol=0.0, atol=1e-9)
        )

    def test_sparse_delta_native_fit_matches_materialized_poisson(self) -> None:
        data = synthetic_data(seed=881, n_sequences=180)
        ctx = make_context(data, "sparse-delta-poisson", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=4, knot_count=2)
        weights = 0.7 + (np.arange(ctx.n_sequences) % 4) * 0.15
        fixed = (engine.sparse_response(ctx, (1,), 0),)
        closure = (engine.sparse_response(ctx, (0,), 1),)
        closure_terms = (((0,), 1),)
        rule = RuleIdentity((0, 1), 2, 1)
        feature = engine.sparse_response(ctx, rule.antecedent, rule.window)
        direct = prepare_fixed_support_design(
            ctx,
            (*fixed, *closure),
            (feature,),
            (rule,),
            cluster_weights=weights,
        )
        base = prepare_sparse_nuisance_partition(
            ctx, fixed, cluster_weights=weights
        )
        sparse = prepare_sparse_delta_support_design(
            ctx,
            base,
            closure,
            closure_terms,
            (feature,),
            (rule,),
            cluster_weights=weights,
        )
        direct_fit = fit_fixed_support(
            ctx,
            (),
            (),
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=160,
            tolerance=1.0e-7,
            prepared_design=direct,
            closure_terms=closure_terms,
        )
        sparse_fit = fit_sparse_delta_support(
            sparse, max_iter=160, tolerance=1.0e-7
        )
        direct_closure = prepare_fixed_support_design(
            ctx,
            (*fixed, *closure),
            (),
            (),
            cluster_weights=weights,
        )
        direct_closure_fit = fit_fixed_support(
            ctx,
            (),
            (),
            (),
            device="cpu",
            dtype="float64",
            max_iter=160,
            tolerance=1.0e-7,
            prepared_design=direct_closure,
            closure_terms=closure_terms,
        )
        sparse_closure_fit = fit_sparse_delta_closure(
            sparse, max_iter=160, tolerance=1.0e-7
        )
        self.assertIsNotNone(sparse_fit)
        assert sparse_fit is not None
        self.assertTrue(direct_fit.converged)
        self.assertTrue(sparse_fit.converged)
        self.assertIsNotNone(sparse_closure_fit)
        assert sparse_closure_fit is not None
        self.assertTrue(sparse_closure_fit.converged)
        self.assertAlmostEqual(
            direct_closure_fit.nll, sparse_closure_fit.nll, places=9
        )
        self.assertAlmostEqual(direct_fit.nll, sparse_fit.nll, places=9)
        self.assertTrue(
            np.allclose(direct_fit.gamma, sparse_fit.gamma, rtol=0.0, atol=1e-8)
        )
        self.assertTrue(
            np.allclose(direct_fit.theta, sparse_fit.theta, rtol=0.0, atol=1e-8)
        )
        self.assertEqual(
            sparse_delta_rule_recession_columns(sparse),
            CertSCRPipeline._nonattained_rule_recession_columns(direct),
        )

    def test_incremental_window_partition_equals_fresh_cumulative_design(self) -> None:
        data = synthetic_data(seed=819, n_sequences=125)
        ctx = make_context(data, "incremental-window", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=5, knot_count=3)
        weights = 0.9 + (np.arange(ctx.n_sequences) % 4) * 0.15
        fixed: tuple[SparseKernelResponse, ...] = ()
        base = prepare_sparse_nuisance_partition(
            ctx, fixed, cluster_weights=weights
        )
        rule_one = RuleIdentity((0, 1), 1, 1)
        dynamic_one = (
            engine.sparse_response(ctx, (0,), 1),
            engine.sparse_response(ctx, (1,), 1),
        )
        initial = refine_sparse_nuisance_partition(
            ctx,
            base,
            dynamic_one,
            (engine.sparse_response(ctx, rule_one.antecedent, 1),),
            (rule_one,),
            cluster_weights=weights,
            return_partition=True,
        )
        self.assertIsInstance(initial, IncrementalSupportPartition)
        assert isinstance(initial, IncrementalSupportPartition)
        rule_three = RuleIdentity((0, 1), 3, 1)
        updated = update_incremental_support_partition(
            ctx,
            initial,
            (
                engine.sparse_window_delta_response(ctx, (0,), 1, 3),
                engine.sparse_window_delta_response(ctx, (1,), 1, 3),
            ),
            (
                engine.sparse_window_delta_response(
                    ctx, rule_three.antecedent, 1, 3
                ),
            ),
            (rule_three,),
            cluster_weights=weights,
        )
        direct = prepare_fixed_support_design(
            ctx,
            (
                *fixed,
                engine.sparse_response(ctx, (0,), 3),
                engine.sparse_response(ctx, (1,), 3),
            ),
            (engine.sparse_response(ctx, rule_three.antecedent, 3),),
            (rule_three,),
            cluster_weights=weights,
        )
        probe = FitResult(
            rules=(rule_three,),
            closure_terms=(),
            alpha=-2.4,
            gamma=np.linspace(-0.12, 0.14, direct.control_width),
            theta=np.asarray([[0.08, 0.03, 0.11]]),
            nll=math.inf,
            kkt_residual=math.inf,
            converged=False,
            iterations=0,
            device="test",
        )
        _, direct_probe_kkt, direct_probe_nll = fixed_support_projected_kkt(
            direct, probe, tolerance=1.0e-9
        )
        _, updated_probe_kkt, updated_probe_nll = fixed_support_projected_kkt(
            updated.prepared, probe, tolerance=1.0e-9
        )
        self.assertAlmostEqual(direct_probe_nll, updated_probe_nll, places=12)
        self.assertAlmostEqual(direct_probe_kkt, updated_probe_kkt, places=11)
        direct_fit = fit_fixed_support(
            ctx,
            (),
            (),
            (rule_three,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=direct,
        )
        updated_fit = fit_fixed_support(
            ctx,
            (),
            (),
            (rule_three,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=updated.prepared,
        )
        self.assertTrue(direct_fit.converged)
        self.assertTrue(updated_fit.converged)
        self.assertAlmostEqual(direct_fit.nll, updated_fit.nll, places=10)
        self.assertTrue(
            np.allclose(direct_fit.gamma, updated_fit.gamma, rtol=0.0, atol=1e-7)
        )
        self.assertTrue(
            np.allclose(direct_fit.theta, updated_fit.theta, rtol=0.0, atol=1e-7)
        )

    def test_incremental_native_update_clamps_only_accumulation_roundoff(self) -> None:
        # IPW groups can contain thousands of equal positive masses.  The old
        # group sum and the later row-wise subtraction are legal but different
        # floating-point summation orders.  Simulate the Freddie deficit:
        # larger than 128 eps, yet inside the standard gamma_k error bound.
        row_count = 4096
        active_rows = np.arange(row_count, dtype=np.int64)
        active_weights = np.full(row_count, 50.0 / row_count, dtype=np.float64)
        exact_active_mass = float(np.sum(active_weights, dtype=np.float64))
        rounded_old_mass = exact_active_mass - 2.0e-11
        grouped = update_sparse_design_partitioned(
            active_rows,
            active_weights,
            np.zeros(row_count, dtype=np.int32),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([rounded_old_mass], dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float64),
            (active_rows,),
            (np.ones((row_count, 1), dtype=np.float32),),
            (np.empty((0, 1), dtype=np.float32),),
            (1,),
            (1.0,),
        )
        self.assertIsNotNone(grouped)
        assert grouped is not None
        _design, event_rows, mass, _representatives = grouped[:4]
        self.assertEqual(event_rows, 0)
        self.assertAlmostEqual(float(np.sum(mass)), exact_active_mass, places=10)

        with self.assertRaisesRegex(RuntimeError, "status -64"):
            update_sparse_design_partitioned(
                active_rows,
                active_weights,
                np.zeros(row_count, dtype=np.int32),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([exact_active_mass - 1.0e-6], dtype=np.float64),
                np.empty(0, dtype=np.int32),
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                (active_rows,),
                (np.ones((row_count, 1), dtype=np.float32),),
                (np.empty((0, 1), dtype=np.float32),),
                (1,),
                (1.0,),
            )

    def test_appended_rule_partition_matches_cold_hierarchy_support_fit(self) -> None:
        data = synthetic_data(seed=1441, n_sequences=140)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                solver_device="cpu",
                solver_dtype="float64",
                solver_max_iter=120,
                solver_tolerance=1.0e-7,
            ),
        )
        rule = RuleIdentity((0, 1), 2, -1)
        closure = pipeline.hierarchy_closure((rule,))
        features = pipeline.sparse_features(pipeline.splits.fit, (rule,))
        weights = np.ones(pipeline.splits.fit.n_sequences, dtype=np.float64)
        weights[::11] = 0.0
        pipeline.fit_cluster_weights = weights
        base = pipeline.prepare_partitioned_support_design(
            pipeline.splits.fit,
            closure,
            [],
            [],
            cluster_weights=weights,
            return_partition=True,
        )
        self.assertIsInstance(base, IncrementalSupportPartition)
        assert isinstance(base, IncrementalSupportPartition)
        appended = append_rules_to_incremental_partition(
            pipeline.splits.fit,
            base,
            features,
            (rule,),
            cluster_weights=weights,
            occurrence_likelihood=pipeline.occurrence_likelihood,
        )
        self.assertIsNotNone(appended)
        assert appended is not None
        factorized = prepare_delta_factorized_support_design(
            pipeline.splits.fit,
            base,
            features,
            (rule,),
            cluster_weights=weights,
            occurrence_likelihood=pipeline.occurrence_likelihood,
        )
        probe = np.linspace(
            -0.25,
            0.35,
            appended.prepared.design.shape[1],
            dtype=np.float64,
        )
        probe[appended.prepared.constrained_start :] = np.abs(
            probe[appended.prepared.constrained_start :]
        )
        dense_objective, dense_gradient, dense_hessian = (
            _prepared_objective_grad_hessian_numpy(
                appended.prepared, probe
            )
        )
        delta_objective, delta_gradient, delta_hessian = (
            _factorized_objective_grad_hessian_numpy(factorized, probe)
        )
        self.assertAlmostEqual(
            _prepared_objective_numpy(appended.prepared, probe),
            _factorized_objective_numpy(factorized, probe),
            places=10,
        )
        self.assertAlmostEqual(dense_objective, delta_objective, places=10)
        self.assertTrue(
            np.allclose(dense_gradient, delta_gradient, rtol=1e-11, atol=1e-10)
        )
        assert dense_hessian is not None
        self.assertTrue(
            np.allclose(dense_hessian, delta_hessian, rtol=1e-10, atol=1e-10)
        )
        _, _, base_hessian = _prepared_objective_grad_hessian_numpy(
            factorized.base,
            probe[: factorized.constrained_start],
        )
        assert base_hessian is not None
        cached_objective, cached_gradient, cached_hessian = (
            _factorized_objective_grad_hessian_numpy(
                factorized,
                probe,
                closure_hessian=base_hessian,
            )
        )
        self.assertAlmostEqual(delta_objective, cached_objective, places=12)
        self.assertTrue(np.array_equal(delta_gradient, cached_gradient))
        self.assertTrue(
            np.allclose(delta_hessian, cached_hessian, rtol=0.0, atol=1e-12)
        )
        self.assertEqual(
            factorized_rule_recession_columns(factorized),
            CertSCRPipeline._nonattained_rule_recession_columns(
                appended.prepared
            ),
        )
        # Strong inhibition must be evaluated as residual-base plus the
        # active likelihood, not by subtracting two nearly equal large base
        # terms.  The compact path remains identical to the materialized child
        # even when exp(delta) is close to machine zero.
        extreme = probe.copy()
        extreme[factorized.constrained_start :] = 80.0
        dense_extreme = _prepared_objective_grad_hessian_numpy(
            appended.prepared, extreme
        )
        factorized_extreme = _factorized_objective_grad_hessian_numpy(
            factorized, extreme
        )
        self.assertTrue(np.isfinite(factorized_extreme[0]))
        self.assertAlmostEqual(
            dense_extreme[0], factorized_extreme[0], places=9
        )
        self.assertTrue(
            np.allclose(
                dense_extreme[1],
                factorized_extreme[1],
                rtol=1e-11,
                atol=1e-9,
            )
        )
        assert dense_extreme[2] is not None
        self.assertTrue(
            np.allclose(
                dense_extreme[2],
                factorized_extreme[2],
                rtol=1e-10,
                atol=1e-9,
            )
        )
        cold = pipeline.prepare_partitioned_support_design(
            pipeline.splits.fit,
            closure,
            features,
            (rule,),
            cluster_weights=weights,
        )
        direct_fit = fit_fixed_support(
            pipeline.splits.fit,
            (),
            (),
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=cold,
        )
        appended_fit = fit_fixed_support(
            pipeline.splits.fit,
            (),
            (),
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=appended.prepared,
        )
        factorized_fit = fit_delta_factorized_support(
            factorized,
            closure,
            max_iter=120,
            tolerance=1.0e-7,
        )
        self.assertTrue(direct_fit.converged)
        self.assertTrue(appended_fit.converged)
        self.assertTrue(factorized_fit.converged)
        self.assertAlmostEqual(direct_fit.nll, appended_fit.nll, places=10)
        self.assertAlmostEqual(
            direct_fit.nll, factorized_fit.nll, places=9
        )
        self.assertTrue(
            np.allclose(direct_fit.gamma, appended_fit.gamma, rtol=0.0, atol=1e-8)
        )
        self.assertTrue(
            np.allclose(direct_fit.theta, appended_fit.theta, rtol=0.0, atol=1e-8)
        )
        self.assertTrue(
            np.allclose(
                direct_fit.theta,
                factorized_fit.theta,
                rtol=0.0,
                atol=1e-7,
            )
        )

    def test_sparse_partition_refinement_preserves_first_event_cloglog(self) -> None:
        rows: list[dict] = []
        bounds: list[dict] = []
        for sequence in range(72):
            target_time = 6 if sequence % 4 == 0 else None
            end_time = target_time if target_time is not None else 9
            serialized: list[dict] = []
            b_time = 2 + sequence % 3
            for month in range(end_time + 1):
                a = int(month == 1 and sequence % 3 != 0)
                b = int(month == b_time and sequence % 5 != 0)
                target = int(target_time is not None and month == target_time)
                if a or b or target:
                    serialized.append(
                        {
                            "sequence_id": f"c{sequence}",
                            "position": len(serialized),
                            "month_index": month,
                            "target_token": target,
                            "a": a,
                            "b": b,
                        }
                    )
            rows.extend(serialized)
            bounds.append(
                {
                    "sequence_id": f"c{sequence}",
                    "start_month": 0,
                    "end_month": end_time,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a", "b"),
            bounds=pd.DataFrame(bounds),
        )
        ctx = make_context(data, "partition-cloglog", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=4, knot_count=2)
        fixed = (engine.sparse_response(ctx, (0,), 0),)
        dynamic = (engine.sparse_response(ctx, (1,), 0),)
        rule = RuleIdentity((0, 1), 1, 1)
        feature = engine.sparse_response(ctx, rule.antecedent, rule.window)
        weights = 0.8 + (np.arange(ctx.n_sequences) % 3) * 0.2
        direct = prepare_fixed_support_design(
            ctx,
            (*fixed, *dynamic),
            (feature,),
            (rule,),
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
        )
        base = prepare_sparse_nuisance_partition(
            ctx,
            fixed,
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
        )
        sparse_delta = prepare_sparse_delta_support_design(
            ctx,
            base,
            dynamic,
            (((1,), 0),),
            (feature,),
            (rule,),
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
        )
        partitioned = refine_sparse_nuisance_partition(
            ctx,
            base,
            dynamic,
            (feature,),
            (rule,),
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
            return_partition=True,
        )
        self.assertIsInstance(partitioned, IncrementalSupportPartition)
        assert isinstance(partitioned, IncrementalSupportPartition)
        refined = partitioned.prepared
        closure_partition = refine_sparse_nuisance_partition(
            ctx,
            base,
            dynamic,
            (),
            (),
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
            return_partition=True,
        )
        self.assertIsInstance(closure_partition, IncrementalSupportPartition)
        assert isinstance(closure_partition, IncrementalSupportPartition)
        factorized = prepare_delta_factorized_support_design(
            ctx,
            closure_partition,
            (feature,),
            (rule,),
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
        )
        probe = np.linspace(
            -0.35,
            0.3,
            direct.design.shape[1],
            dtype=np.float64,
        )
        probe[direct.constrained_start :] = np.abs(
            probe[direct.constrained_start :]
        )
        dense_objective, dense_gradient, dense_hessian = (
            _prepared_objective_grad_hessian_numpy(direct, probe)
        )
        delta_objective, delta_gradient, delta_hessian = (
            _factorized_objective_grad_hessian_numpy(factorized, probe)
        )
        self.assertAlmostEqual(dense_objective, delta_objective, places=9)
        self.assertTrue(
            np.allclose(dense_gradient, delta_gradient, rtol=1e-10, atol=1e-9)
        )
        assert dense_hessian is not None
        self.assertTrue(
            np.allclose(dense_hessian, delta_hessian, rtol=1e-9, atol=1e-9)
        )
        dense_fit = fit_fixed_support(
            ctx,
            (),
            (),
            (rule,),
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
            prepared_design=direct,
            occurrence_likelihood="first_event_cloglog",
        )
        factorized_fit = fit_delta_factorized_support(
            factorized,
            (),
            max_iter=120,
            tolerance=1.0e-7,
        )
        sparse_delta_fit = fit_sparse_delta_support(
            sparse_delta,
            max_iter=120,
            tolerance=1.0e-7,
        )
        self.assertTrue(dense_fit.converged)
        self.assertTrue(factorized_fit.converged)
        self.assertIsNotNone(sparse_delta_fit)
        assert sparse_delta_fit is not None
        self.assertTrue(sparse_delta_fit.converged)
        self.assertAlmostEqual(dense_fit.nll, factorized_fit.nll, places=8)
        self.assertAlmostEqual(dense_fit.nll, sparse_delta_fit.nll, places=8)
        self.assertTrue(
            np.allclose(
                dense_fit.theta,
                factorized_fit.theta,
                rtol=0.0,
                atol=1e-7,
            )
        )
        self.assertTrue(
            np.allclose(
                dense_fit.theta,
                sparse_delta_fit.theta,
                rtol=0.0,
                atol=1e-7,
            )
        )
        initial = FitResult(
            rules=(rule,),
            closure_terms=(),
            alpha=-2.0,
            gamma=np.asarray([0.1, -0.2, 0.05, -0.1]),
            theta=np.asarray([[0.15, 0.05]]),
            nll=math.inf,
            kkt_residual=math.inf,
            converged=False,
            iterations=0,
            device="test",
        )
        direct_ok, direct_kkt, direct_objective = fixed_support_projected_kkt(
            direct, initial, tolerance=1.0e-7
        )
        refined_ok, refined_kkt, refined_objective = fixed_support_projected_kkt(
            refined, initial, tolerance=1.0e-7
        )
        self.assertEqual(direct_ok, refined_ok)
        self.assertAlmostEqual(direct_objective, refined_objective, places=11)
        self.assertAlmostEqual(direct_kkt, refined_kkt, places=10)
        rule_three = RuleIdentity((0, 1), 3, 1)
        updated = update_incremental_support_partition(
            ctx,
            partitioned,
            (),
            (
                engine.sparse_window_delta_response(
                    ctx, rule_three.antecedent, 1, 3
                ),
            ),
            (rule_three,),
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
        )
        direct_three = prepare_fixed_support_design(
            ctx,
            (*fixed, *dynamic),
            (engine.sparse_response(ctx, rule_three.antecedent, 3),),
            (rule_three,),
            cluster_weights=weights,
            occurrence_likelihood="first_event_cloglog",
        )
        probe_three = replace(initial, rules=(rule_three,))
        _, updated_kkt, updated_objective = fixed_support_projected_kkt(
            updated.prepared, probe_three, tolerance=1.0e-7
        )
        _, direct_three_kkt, direct_three_objective = fixed_support_projected_kkt(
            direct_three, probe_three, tolerance=1.0e-7
        )
        self.assertAlmostEqual(updated_objective, direct_three_objective, places=11)
        self.assertAlmostEqual(updated_kkt, direct_three_kkt, places=10)

    def test_minimum_event_activating_span_matches_full_sparse_responses(self) -> None:
        data = synthetic_data(seed=821, n_sequences=95)
        ctx = make_context(data, "event-activation-span", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=5, knot_count=3)
        windows = [0, 1, 2, 3, 4]
        for antecedent in ((0,), (1,), (0, 1)):
            exact = [
                window
                for window, response in engine.iter_window_sparse_responses(
                    ctx, antecedent, windows
                )
                if np.any(response.event_values != 0.0)
            ]
            expected = min(exact) if exact else None
            observed = engine.minimum_event_activating_span(
                ctx, antecedent, max_window=max(windows)
            )
            self.assertEqual(observed, expected)

    def test_group_saturated_bound_is_below_exact_signed_kernel_optimum(self) -> None:
        frame = pd.DataFrame(
            [
                {"sequence_id": "s0", "position": 0, "month_index": 0, "target_token": 0, "a": 0},
                {"sequence_id": "s0", "position": 1, "month_index": 1, "target_token": 1, "a": 0},
                {"sequence_id": "s1", "position": 0, "month_index": 0, "target_token": 0, "a": 0},
                {"sequence_id": "s1", "position": 1, "month_index": 2, "target_token": 1, "a": 0},
            ]
        )
        bounds = pd.DataFrame(
            [
                {"sequence_id": "s0", "start_month": 0, "end_month": 2},
                {"sequence_id": "s1", "start_month": 0, "end_month": 2},
            ]
        )
        data = EventData.from_frame(frame, predicate_names=("a",), bounds=bounds)
        ctx = make_context(data, "safe-bound", np.asarray([0, 1]))
        grid_feature = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        event_grid = (
            ctx.grid_offsets[ctx.event_sequence_local]
            + ctx.event_times
            - ctx.start_times[ctx.event_sequence_local]
        )
        feature = np.concatenate(
            (grid_feature[event_grid], grid_feature)
        ).reshape(-1, 1)
        rule = RuleIdentity((0,), 0, 1)
        bound = group_saturated_poisson_lower_bound(
            ctx,
            (),
            [feature],
            [rule],
        )
        fit = fit_fixed_support(
            ctx,
            (),
            [feature],
            [rule],
            device="cpu",
            dtype="float64",
            max_iter=100,
            tolerance=2.0e-5,
        )
        self.assertTrue(bound.finite)
        self.assertTrue(fit.converged)
        self.assertLessEqual(bound.lower_bound, fit.nll + 1.0e-10)

    def test_direct_block_compression_equals_dense_assembly(self) -> None:
        control_a = np.asarray([[2.0], [0.0], [1.0], [0.0]], dtype=np.float32)
        control_b = np.asarray([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        feature = np.asarray([[1.0], [0.0], [0.0], [3.0]], dtype=np.float32)
        rule = RuleIdentity((0,), 0, -1)
        weights = np.asarray([0.5, 2.0, 4.0], dtype=np.float64)
        dense, constrained = assemble_design(
            np.concatenate([control_a, control_b], axis=1), [feature], [rule]
        )
        expected_x, expected_w, expected_active = compress_zero_grid_rows(
            dense, 1, weights
        )
        actual_x, actual_w, actual_constrained, actual_active = assemble_compressed_design(
            (control_a, control_b),
            [feature],
            [rule],
            n_events=1,
            grid_weights=weights,
        )
        self.assertEqual(actual_constrained, constrained)
        self.assertEqual(actual_active, expected_active)
        self.assertTrue(np.array_equal(actual_x, expected_x))
        self.assertTrue(np.array_equal(actual_w, expected_w))

    def test_factorized_grid_weights_equal_materialized_weights(self) -> None:
        control = np.asarray(
            [[1.0], [0.0], [2.0], [0.0], [3.0]], dtype=np.float32
        )
        feature = np.asarray(
            [[0.5], [0.0], [0.0], [4.0], [0.0]], dtype=np.float32
        )
        rule = RuleIdentity((0,), 0, 1)
        base = np.asarray([0.5, 2.0, 1.5, 3.0], dtype=np.float32)
        sequence_local = np.asarray([0, 0, 1, 1], dtype=np.int32)
        sequence_weights = np.asarray([1.25, 0.4], dtype=np.float64)
        materialized = base.astype(np.float64) * sequence_weights[sequence_local]
        expected = assemble_compressed_design(
            (control,),
            [feature],
            [rule],
            n_events=1,
            grid_weights=materialized,
        )
        actual = assemble_compressed_design(
            (control,),
            [feature],
            [rule],
            n_events=1,
            base_grid_weights=base,
            grid_sequence_local=sequence_local,
            sequence_weights=sequence_weights,
            weighted_exposure=float(np.sum(materialized)),
        )
        self.assertTrue(np.array_equal(actual[0], expected[0]))
        self.assertTrue(np.allclose(actual[1], expected[1], rtol=0.0, atol=1.0e-14))
        self.assertEqual(actual[2:], expected[2:])

    def test_zero_grid_compression_preserves_objective_gradient_and_fisher(self) -> None:
        design = np.asarray(
            [
                [1.0, 2.0, 0.0],       # event
                [1.0, 0.0, 0.0],       # zero grid
                [1.0, 1.0, 0.0],       # active grid
                [1.0, 0.0, 0.0],       # zero grid
                [1.0, 0.0, 3.0],       # active grid
            ],
            dtype=np.float32,
        )
        grid_weights = np.asarray([2.0, 0.5, 4.0, 1.5], dtype=np.float64)
        compressed, compressed_weights, active = compress_zero_grid_rows(
            design, 1, grid_weights
        )
        self.assertEqual(active, 2)
        self.assertEqual(len(compressed), 4)
        params = np.asarray([-1.2, 0.3, -0.1], dtype=np.float64)

        def moments(x: np.ndarray, weights: np.ndarray):
            grid = x[1:].astype(np.float64)
            mu = weights * np.exp(grid @ params)
            objective = float(np.sum(mu) - x[0].astype(np.float64) @ params)
            gradient = grid.T @ mu - x[0].astype(np.float64)
            fisher = grid.T @ (grid * mu[:, None])
            return objective, gradient, fisher

        dense = moments(design, grid_weights)
        reduced = moments(compressed, compressed_weights)
        self.assertAlmostEqual(dense[0], reduced[0], places=12)
        self.assertTrue(np.allclose(dense[1], reduced[1], rtol=1e-12, atol=1e-12))
        self.assertTrue(np.allclose(dense[2], reduced[2], rtol=1e-12, atol=1e-12))

    def test_mark_multiplicity_stays_aligned_with_repeated_target_queries(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sequence_id": "s0",
                    "position": 0,
                    "month_index": 0,
                    "target_token": 2,
                    "target_mark_values": [10.0, 25.0],
                    "a": 1,
                },
                {
                    "sequence_id": "s1",
                    "position": 0,
                    "month_index": 1,
                    "target_token": 1,
                    "target_mark_values": [7.0],
                    "a": 0,
                },
            ]
        )
        data = EventData.from_frame(
            frame,
            predicate_names=("a",),
            mark_col="target_mark_values",
        )
        ctx = make_context(data, "marked", np.asarray([0, 1]))
        self.assertEqual(ctx.n_events, 3)
        self.assertEqual(ctx.event_times.tolist(), [0, 0, 1])
        self.assertEqual(ctx.event_marks.tolist(), [10.0, 25.0, 7.0])

    def test_mark_head_and_score_use_positive_log_normal_likelihood(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sequence_id": f"s{i}",
                    "position": 0,
                    "month_index": i,
                    "target_token": 1,
                    "target_mark_values": [float(2 ** (i + 1))],
                    "a": 0,
                }
                for i in range(4)
            ]
        )
        data = EventData.from_frame(
            frame,
            predicate_names=("a",),
            mark_col="target_mark_values",
        )
        ctx = make_context(data, "marked", np.arange(4))
        nuisance = np.zeros((ctx.n_queries, 0), dtype=np.float64)
        fit = fit_mark_head(
            ctx,
            nuisance,
            [],
            unit=4.0,
            variance=None,
        )
        mean = expected_mark(fit, nuisance, [])
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(mean > 0))
        candidate = np.arange(1, ctx.n_events + 1, dtype=np.float64).reshape(-1, 1)
        gradient, information = mark_score_moments(
            fit,
            ctx,
            nuisance,
            [],
            candidate,
        )
        self.assertEqual(gradient.shape, (1,))
        self.assertGreater(float(information[0, 0]), 0.0)

        constant = np.ones((ctx.n_events, 1), dtype=np.float64)
        _gradient_constant, information_constant = mark_score_moments(
            fit,
            ctx,
            nuisance,
            [],
            constant,
        )
        self.assertLessEqual(abs(float(information_constant[0, 0])), 1.0e-24)

    def test_cached_fwl_mark_fit_matches_direct_weighted_least_squares(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sequence_id": f"m{i}",
                    "position": 0,
                    "month_index": i,
                    "target_token": 1,
                    "target_mark_values": [float(np.exp(0.2 + 0.3 * i + 0.1 * (i % 2)))],
                    "a": 0,
                }
                for i in range(8)
            ]
        )
        data = EventData.from_frame(
            frame,
            predicate_names=("a",),
            mark_col="target_mark_values",
        )
        ctx = make_context(data, "fwl", np.arange(8))
        nuisance = np.arange(8, dtype=np.float64).reshape(-1, 1)
        activation = (np.arange(8) % 2).astype(np.float64)
        weights = 1.0 + np.arange(8, dtype=np.float64) / 8.0
        residualizer = make_mark_base_residualizer(
            ctx,
            nuisance,
            cluster_weights=weights,
        )
        fit = fit_mark_head(
            ctx,
            nuisance,
            [activation],
            unit=1.0,
            variance=0.7,
            cluster_weights=weights,
            base_residualizer=residualizer,
        )
        design = np.column_stack((np.ones(8), nuisance[:, 0], activation))
        y = np.log(ctx.event_marks)
        direct = np.linalg.lstsq(
            design * np.sqrt(weights)[:, None],
            y * np.sqrt(weights),
            rcond=None,
        )[0]
        fitted_fwl = design @ np.concatenate(
            ([fit.intercept], fit.nuisance_beta, fit.rule_beta)
        )
        self.assertTrue(np.allclose(fitted_fwl, design @ direct, rtol=1.0e-12, atol=1.0e-12))

    def test_nonfinite_cluster_value_invalidates_instead_of_dropping_a_test_row(self) -> None:
        result = one_sided_mean_test([1.0, math.inf, 2.0])
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual(result.n_clusters, 3)

    def test_prior_incoming_distinguishes_transactions_from_unique_senders(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": np.arange(6),
                "from_code": [1, 1, 2, 0, 3, 0],
                "to_code": [0, 0, 0, 9, 0, 9],
                "day_index": [0] * 6,
                "amount_received": [10.0] * 6,
            }
        )
        counts, _amounts, unique = compute_source_prior_incoming(frame)
        self.assertEqual(int(counts[3]), 3)
        self.assertEqual(int(unique[3]), 2)
        self.assertEqual(int(counts[5]), 4)
        self.assertEqual(int(unique[5]), 3)

    def test_currency_conversion_is_identified_from_label_free_amount_pairs(self) -> None:
        frame = pd.DataFrame(
            {
                "time_index": [0, 1, 2],
                "amount_paid": [10.0, 20.0, 5.0],
                "amount_received": [20.0, 10.0, 10.0],
                "payment_currency": ["EUR", "US Dollar", "EUR"],
                "receiving_currency": ["US Dollar", "EUR", "US Dollar"],
            }
        )
        rates = infer_currency_usd_rates(frame, 1.0)
        self.assertAlmostEqual(rates["US Dollar"], 1.0, places=12)
        self.assertAlmostEqual(rates["EUR"], 2.0, places=12)

    def test_result_writer_replaces_nonfinite_diagnostics_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_result({"finite": 1.0, "diagnostic": -math.inf}, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["finite"], 1.0)
            self.assertIsNone(payload["diagnostic"])

    def test_ibm_dynamic_nonproxy_policy_is_frozen_and_unique(self) -> None:
        selected = resolve_predicate_policy("ibm_aml_dynamic_nonproxy_v2")
        self.assertEqual(selected, IBM_AML_DYNAMIC_NONPROXY_V2)
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(set(selected)), len(selected))
        self.assertNotIn("pred_out_format_switch_after_history", selected)
        self.assertNotIn("pred_out_bank_route_switch_after_history", selected)
        expanded = resolve_predicate_policy("ibm_aml_dynamic_nonproxy_v3")
        self.assertEqual(expanded, IBM_AML_DYNAMIC_NONPROXY_V3)
        self.assertEqual(len(expanded), 12)
        primitive = resolve_predicate_policy("ibm_aml_primitive_dynamic_v1")
        self.assertEqual(primitive, IBM_AML_PRIMITIVE_DYNAMIC_V1)
        self.assertEqual(len(primitive), 12)
        self.assertEqual(len(set(primitive)), len(primitive))
        self.assertNotIn("pred_in_to_out_turnaround_starts", primitive)
        self.assertNotIn("pred_out_format_switch_after_history", primitive)
        self.assertNotIn("pred_out_bank_route_switch_after_history", primitive)
        self.assertNotIn("pred_out_structured_small_repeats_day", primitive)
        self.assertNotIn("pred_fan_in_then_out_day", primitive)
        self.assertNotIn("pred_cycle_return_72h", primitive)
        typology = resolve_predicate_policy("ibm_aml_typology_dynamic_v1")
        self.assertEqual(typology, IBM_AML_TYPOLOGY_DYNAMIC_V1)
        self.assertEqual(len(typology), 13)
        self.assertEqual(len(set(typology)), len(typology))
        self.assertIn("pred_cycle_return_72h", typology)
        self.assertIn("pred_fan_in_then_out_day", typology)

    def test_nonheuristic_triplet_defaults_do_not_require_heredity_or_budget(self) -> None:
        config = CertSCRConfig()
        self.assertEqual(config.triplet_generation, "all")
        self.assertIsNone(config.max_gradient_triplets)
        self.assertIsNone(config.support_pool_size)
        self.assertEqual(config.active_start_policy, "all_atoms")
        self.assertEqual(
            config.active_neighbor_strategy,
            "exact_one_exchange",
        )
        self.assertEqual(config.support_family, "terminal_atoms")
        self.assertEqual(config.certification_mode, "auto")
        self.assertEqual(config.solver_device, "cpu")
        self.assertEqual(config.solver_dtype, "float64")

    def test_coincident_target_counts_are_preserved_as_point_events(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sequence_id": "s",
                    "position": 0,
                    "month_index": 1,
                    "target_token": 3,
                    "a": 1,
                }
            ]
        )
        bounds = pd.DataFrame(
            [{"sequence_id": "s", "start_month": 0, "end_month": 2}]
        )
        data = EventData.from_frame(frame, predicate_names=("a",), bounds=bounds)
        ctx = make_context(data, "counts", np.array([0]))
        self.assertEqual(ctx.n_events, 3)
        self.assertTrue(np.array_equal(ctx.event_times, np.array([1, 1, 1])))

    def test_invalid_scientific_and_solver_thresholds_are_rejected(self) -> None:
        for kwargs in (
            {"financial_threshold": -1.0},
            {"rule_threshold": math.nan},
            {"certification_mode": "unknown"},
            {"adverse_event_name": "  "},
            {"early_warning_horizon": 0},
            {"early_warning_horizon": 13},
            {"early_warning_threshold": 1.0},
            {"calibration_tolerance": 0.0},
            {"solver_tolerance": -1.0},
            {"solver_max_iter": 0},
            {"solver_workers": 0},
            {"solver_workers": 2, "support_devices": ("cpu", "cpu:0")},
            {"support_devices": ("cuda:0",)},
            {"support_search": "beam"},
            {"identity_profile": "unknown"},
            {"triplet_generation": "unknown"},
            {"target_history_control": 1},
            {"active_restarts": 0},
            {"active_start_policy": "random_top_k"},
            {"active_neighbor_strategy": "beam"},
            {"support_family": "all_visited_unconditionally"},
            {"support_pool_size": 0},
            {"search_improvement_tolerance": -1.0},
            {"split_fractions": (0.4, 0.4, 0.3)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CertSCRConfig(**kwargs).validate()

    def test_connected_mdl_heredity_is_a_valid_triplet_policy(self) -> None:
        CertSCRConfig(triplet_generation="connected_mdl_heredity").validate()
        CertSCRConfig(triplet_generation="strong_mdl_heredity").validate()

    def test_nonnegative_cone_quadratic_gain_enumerates_active_face(self) -> None:
        information = np.eye(2, dtype=np.float64)
        self.assertAlmostEqual(
            CertSCRPipeline._cone_quadratic_gain(np.array([-2.0, 1.0]), information),
            2.0,
        )
        self.assertAlmostEqual(
            CertSCRPipeline._cone_quadratic_gain(np.array([2.0, -1.0]), information),
            0.5,
        )

    def test_target_stratified_split_preserves_independent_sequence_parts(self) -> None:
        data = synthetic_data(seed=19, n_sequences=300)
        splits = split_contexts(data, fractions=(0.5, 0.3, 0.2), seed=23, stratify_target=True)
        ids = [set(ctx.global_sequence_ids.tolist()) for ctx in (splits.fit, splits.cert, splits.test)]
        self.assertFalse(ids[0] & ids[1])
        self.assertFalse(ids[0] & ids[2])
        self.assertFalse(ids[1] & ids[2])
        self.assertEqual(len(set.union(*ids)), data.n_sequences)
        self.assertTrue(all(ctx.n_events > 0 for ctx in (splits.fit, splits.cert, splits.test)))

    def test_fit_negative_sampling_ipw_recovers_population_sequence_mass(self) -> None:
        data = synthetic_data(seed=29, n_sequences=300)
        splits = split_contexts(
            data,
            fractions=(0.5, 0.3, 0.2),
            seed=31,
            stratify_target=True,
            fit_negative_sample_size=30,
        )
        selected_target_counts = np.bincount(
            data.sequence_codes,
            weights=data.targets.astype(np.float64),
            minlength=data.n_sequences,
        )[splits.fit.global_sequence_ids]
        negative = selected_target_counts <= 0
        self.assertAlmostEqual(
            float(np.sum(splits.fit_sampling_weights[negative])),
            float(splits.fit_population_negative_count),
        )
        self.assertTrue(np.allclose(splits.fit_sampling_weights[~negative], 1.0))

    def test_fit_screen_ipw_is_normalized_to_population_mean_units(self) -> None:
        data = synthetic_data(seed=37, n_sequences=300)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a",),
            config=CertSCRConfig(
                q_max=1,
                fit_negative_sample_size=25,
                solver_device="cpu",
            ),
        )
        normalized = pipeline._inference_weights(pipeline.splits.fit)
        self.assertAlmostEqual(float(np.mean(normalized)), 1.0, places=14)
        self.assertTrue(
            np.allclose(
                normalized,
                pipeline.fit_sampling_weights / pipeline.fit_sampling_scale,
            )
        )

    def test_financial_ipw_mdl_scale_is_invariant_to_weight_units(self) -> None:
        base = synthetic_data(seed=43, n_sequences=300)
        weights = 1.0 + np.arange(base.n_sequences, dtype=np.float64) % 7.0
        config = CertSCRConfig(
            q_max=1,
            fit_negative_sample_size=20,
            solver_device="cpu",
        )
        first = CertSCRPipeline(
            replace(
                base,
                sequence_financial_weights=weights,
                financial_weight_name="exposure",
            ),
            rule_predicates=("pred_a",),
            config=config,
        )
        second = CertSCRPipeline(
            replace(
                base,
                sequence_financial_weights=1000.0 * weights,
                financial_weight_name="exposure_milliunits",
            ),
            rule_predicates=("pred_a",),
            config=config,
        )
        self.assertAlmostEqual(
            first.fit_objective_population_scale,
            second.fit_objective_population_scale,
            places=13,
        )

    def test_contiguous_kernel_dictionary_is_normalized_and_contains_one_hot_basis(self) -> None:
        dictionary = CertSCRPipeline._contiguous_kernel_dictionary(4)
        self.assertEqual(dictionary.shape, (10, 4))
        self.assertTrue(np.all(dictionary >= 0.0))
        self.assertTrue(np.allclose(np.sum(dictionary, axis=1), 1.0))
        for row in np.eye(4):
            self.assertTrue(any(np.allclose(candidate, row) for candidate in dictionary))

    def test_student_t_known_value_and_inverse(self) -> None:
        self.assertAlmostEqual(student_t_cdf(1.0, 1), 0.75, places=12)
        self.assertAlmostEqual(student_t_ppf(0.75, 1), 1.0, places=10)

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        raw = np.array([0.03, 0.001, 0.02, 0.8])
        adjusted = holm_adjust(raw)
        self.assertTrue(np.allclose(adjusted, [0.06, 0.004, 0.06, 0.8]))

    def test_independent_family_selection_then_holm_controls_all_null_fwer(self) -> None:
        rng = np.random.default_rng(20260711)
        repetitions = 50_000
        hypotheses = 25
        fit_screen_p = rng.random((repetitions, hypotheses))
        certification_p = rng.random((repetitions, hypotheses))
        any_false_rejection = np.zeros(repetitions, dtype=bool)
        for index in range(repetitions):
            selected = fit_screen_p[index] <= 0.10
            if np.any(selected):
                any_false_rejection[index] = bool(
                    np.any(holm_adjust(certification_p[index, selected]) <= 0.05)
                )
        observed = float(np.mean(any_false_rejection))
        # Fixed-seed Monte Carlo audit of the conditional-FWER construction;
        # the theoretical guarantee comes from independence + Holm, not from
        # this numerical tolerance.
        self.assertLess(observed, 0.052)

    def test_degenerate_one_sided_test_is_conservative_at_null(self) -> None:
        result = one_sided_mean_test(np.zeros(10), null=0.0)
        self.assertEqual(result.p_value, 1.0)

    def test_directional_score_and_information_use_the_same_cluster_weights(self) -> None:
        data = OccurrenceTests.data()
        ctx = make_context(data, "weighted-score", np.array([0, 1]))
        residual = np.ones(ctx.n_queries, dtype=np.float64)
        eta = np.zeros(ctx.n_queries, dtype=np.float64)
        weights = np.array([3.0, 7.0])
        ordinary = cluster_directional_score(residual, eta, ctx, sign=1)
        weighted = cluster_directional_score(
            residual, eta, ctx, sign=1, cluster_weights=weights
        )
        self.assertTrue(np.allclose(weighted, weights * ordinary))
        _residual, information, rank = efficient_information_matrix(
            residual[:, None],
            np.zeros((ctx.n_queries, 0)),
            eta,
            ctx,
            cluster_weights=weights,
        )
        expected = float(
            np.dot(ctx.grid_weights, weights[ctx.grid_sequence_local])
        )
        self.assertEqual(rank, 0)
        self.assertAlmostEqual(float(information[0, 0]), expected, places=12)


class PredicatePolicyTests(unittest.TestCase):
    def test_behavioral_nonproxy_tier_is_explicit_and_excludes_direct_or_nonbehavioral_markers(self) -> None:
        expected = {
            "pred_prev_revolving_application",
            "pred_prev_multi_application_month",
            "pred_bureau_credit_card_opened",
            "pred_bureau_large_credit_opened",
            "pred_pos_installment_count_increases",
            "pred_card_utilization_jump_20pp",
            "pred_card_cash_withdrawal",
        }
        excluded = {
            "pred_prev_application_refused",
            "pred_prev_application_canceled",
            "pred_bureau_unknown_status_3m_starts",
            "pred_bureau_recovers_to_current",
            "pred_card_utilization_cross_95",
            "pred_card_limit_cut_20pct",
        }
        selected = set(selected_predicates("behavioral_nonproxy"))
        self.assertEqual(selected, expected)
        self.assertEqual(selected, set(BEHAVIORAL_NONPROXY_PREDICATES))
        self.assertEqual(selected, set(HOME_CREDIT_BEHAVIORAL_NONPROXY))
        self.assertEqual(selected, set(resolve_predicate_policy("home_credit_behavioral_nonproxy")))
        self.assertTrue(selected.isdisjoint(excluded))

    def test_expanded_behavioral_nonproxy_policy_has_twelve_dynamic_nonproxy_predicates(self) -> None:
        selected = set(selected_predicates("behavioral_nonproxy_expanded"))
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected, set(BEHAVIORAL_NONPROXY_EXPANDED_PREDICATES))
        self.assertEqual(selected, set(HOME_CREDIT_BEHAVIORAL_NONPROXY_EXPANDED))
        self.assertEqual(
            selected,
            set(resolve_predicate_policy("home_credit_behavioral_nonproxy_expanded")),
        )
        forbidden = {
            "pred_prev_application_refused",
            "pred_prev_application_canceled",
            "pred_bureau_status_1_starts",
            "pred_pos_mild_dpd_starts",
            "pred_card_mild_dpd_starts",
            "pred_card_payment_shortfall_min_due",
            "pred_card_limit_cut_20pct",
            "pred_inst_late_1_15d",
            "pred_inst_late_16_29d",
            "pred_inst_underpaid_5pct",
            "pred_inst_underpaid_50pct",
            "pred_pos_future_installments_not_decreasing_2m",
            "pred_inst_paid_early_7d",
        }
        self.assertTrue(selected.isdisjoint(forbidden))

    def test_freddie_structural_policy_excludes_target_adjacent_and_time_markers(self) -> None:
        selected = set(resolve_predicate_policy("freddie_structural_dynamic_v2"))
        self.assertEqual(len(selected), 18)
        self.assertEqual(selected, set(FREDDIE_STRUCTURAL_DYNAMIC_V2))
        self.assertTrue(
            selected.isdisjoint(
                {
                    "pred_forbearance_starts",
                    "pred_payment_deferral_current",
                    "pred_loan_age_reaches_12",
                    "pred_loan_age_reaches_24",
                }
            )
        )


class OccurrenceTests(unittest.TestCase):
    @staticmethod
    def data() -> EventData:
        frame = pd.DataFrame(
            [
                {"sequence_id": "s0", "position": 0, "month_index": 0, "target_token": 0, "a": 0, "b": 0},
                {"sequence_id": "s0", "position": 1, "month_index": 2, "target_token": 1, "a": 1, "b": 0},
                {"sequence_id": "s0", "position": 2, "month_index": 3, "target_token": 1, "a": 0, "b": 0},
                {"sequence_id": "s0", "position": 3, "month_index": 4, "target_token": 0, "a": 0, "b": 1},
                {"sequence_id": "s1", "position": 0, "month_index": 0, "target_token": 0, "a": 0, "b": 0},
                {"sequence_id": "s1", "position": 1, "month_index": 1, "target_token": 0, "a": 0, "b": 1},
                {"sequence_id": "s1", "position": 2, "month_index": 4, "target_token": 0, "a": 1, "b": 0},
            ]
        )
        bounds = pd.DataFrame(
            [
                {"sequence_id": "s0", "start_month": 0, "end_month": 6},
                {"sequence_id": "s1", "start_month": 0, "end_month": 6},
            ]
        )
        return EventData.from_frame(frame, predicate_names=("a", "b"), bounds=bounds)

    def test_implicit_grid_metadata_preserves_array_contract_without_storage(self) -> None:
        data = self.data()
        ctx = make_context(data, "implicit-grid", np.asarray([0, 1]))
        expected_sequence = np.repeat(
            np.arange(ctx.n_sequences, dtype=np.int32), np.diff(ctx.grid_offsets)
        )
        self.assertEqual(int(ctx.grid_sequence_local.nbytes), 0)
        self.assertEqual(int(ctx.grid_weights.nbytes), 0)
        self.assertTrue(np.array_equal(np.asarray(ctx.grid_sequence_local), expected_sequence))
        self.assertTrue(
            np.array_equal(np.asarray(ctx.grid_weights), np.ones(ctx.n_grid, dtype=np.float32))
        )
        selected = np.asarray([0, 3, ctx.n_grid - 1], dtype=np.int64)
        self.assertTrue(np.array_equal(ctx.grid_sequences_at(selected), expected_sequence[selected]))
        self.assertTrue(np.array_equal(ctx.grid_weights_at(selected), np.ones(len(selected))))
        values = np.arange(ctx.n_grid, dtype=np.float64)
        expected_sum = np.bincount(
            expected_sequence, weights=values, minlength=ctx.n_sequences
        )
        self.assertTrue(np.array_equal(ctx.aggregate_weighted_grid(values), expected_sum))

    def test_basis_is_nonnegative_and_unit_area(self) -> None:
        basis = make_triangular_basis(7, 4)
        self.assertTrue(np.all(basis >= 0))
        self.assertTrue(np.allclose(np.sum(basis, axis=1), 1.0))

    def test_strictly_past_response_and_exact_pair_breakpoints(self) -> None:
        data = self.data()
        ctx = make_context(data, "all", np.array([0, 1]))
        engine = RuleOccurrenceEngine(data, lag=3, knot_count=2)
        response = engine.response(ctx, (0,), 0)
        # Target events are ordered as s0/t=2 then s0/t=3.  A at t=2
        # cannot affect the simultaneous event but must affect t=3.
        self.assertTrue(np.all(response[0] == 0))
        self.assertGreater(float(np.sum(response[1])), 0.0)
        breakpoints = engine.window_breakpoints((0, 1), np.array([0, 1]), max_window=5)
        self.assertTrue(np.array_equal(breakpoints, np.array([2, 3], dtype=np.int32)))

    def test_global_scatter_response_equals_sequence_convolution(self) -> None:
        data = synthetic_data(seed=73, n_sequences=80)
        ctx = make_context(data, "response-equivalence", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=6, knot_count=3)

        def sequence_convolution(antecedent: tuple[int, ...], window: int) -> np.ndarray:
            events = engine.completions(antecedent)
            local_seq = ctx.sequence_lookup[events.sequence_codes]
            keep = (local_seq >= 0) & (events.spans <= window)
            local_seq = local_seq[keep].astype(np.int32, copy=False)
            occ_times = events.times[keep].astype(np.int32, copy=False)
            out_grid = np.zeros((ctx.n_grid, engine.knot_count), dtype=np.float32)
            order = np.argsort(local_seq, kind="stable")
            local_seq = local_seq[order]
            occ_times = occ_times[order]
            populated = np.unique(local_seq)
            kernels = np.pad(engine.basis, ((0, 0), (1, 0)))
            for seq_local in populated.tolist():
                occ = occ_times[local_seq == seq_local]
                start = int(ctx.start_times[seq_local])
                end = int(ctx.end_times[seq_local])
                offset = int(ctx.grid_offsets[seq_local])
                length = end - start + 1
                indices = occ.astype(np.int64) - start
                valid = (indices >= 0) & (indices < length)
                counts = np.bincount(indices[valid], minlength=length).astype(np.float32)
                for knot in range(engine.knot_count):
                    out_grid[offset : offset + length, knot] = np.convolve(
                        counts, kernels[knot], mode="full"
                    )[:length]
            event_indices = (
                ctx.grid_offsets[ctx.event_sequence_local]
                + ctx.event_times.astype(np.int64)
                - ctx.start_times[ctx.event_sequence_local].astype(np.int64)
            )
            return np.concatenate((out_grid[event_indices], out_grid), axis=0)

        for antecedent, window in (((0,), 0), ((1,), 0), ((0, 1), 1), ((0, 1), 4)):
            with self.subTest(antecedent=antecedent, window=window):
                expected = sequence_convolution(antecedent, window)
                actual = engine.response(ctx, antecedent, window)
                self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=2.0e-7))

    def test_sparse_native_response_and_projection_equal_dense_response(self) -> None:
        data = synthetic_data(seed=731, n_sequences=90)
        ctx = make_context(data, "sparse-response", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=6, knot_count=3)
        shape = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
        for antecedent, window in (((0,), 0), ((1,), 0), ((0, 1), 2), ((0, 1), 4)):
            with self.subTest(antecedent=antecedent, window=window):
                dense = engine.response(ctx, antecedent, window)
                sparse = engine.sparse_response(ctx, antecedent, window)
                self.assertIsInstance(sparse, SparseKernelResponse)
                self.assertTrue(np.allclose(sparse.dense(), dense, rtol=0.0, atol=2.0e-7))
                gathered_rows = np.asarray(
                    sorted({0, max(0, ctx.n_grid // 2), max(0, ctx.n_grid - 1)}),
                    dtype=np.int64,
                )
                self.assertTrue(
                    np.array_equal(
                        sparse.grid_at(gathered_rows),
                        dense[ctx.n_events + gathered_rows],
                    )
                )
                coefficients = np.asarray([0.2, -0.3, 0.7], dtype=np.float64)
                for requested in (
                    gathered_rows,
                    gathered_rows[::-1],
                    np.concatenate((gathered_rows, gathered_rows[:1])),
                    np.arange(ctx.n_grid, dtype=np.int64),
                ):
                    expected_predictor = (
                        sparse.grid_at(requested) @ coefficients
                    )
                    actual_predictor = np.full(len(requested), 1.25)
                    sparse.add_grid_linear_predictor(
                        requested,
                        coefficients,
                        actual_predictor,
                        scale=-2.0,
                    )
                    self.assertTrue(
                        np.allclose(
                            actual_predictor,
                            1.25 - 2.0 * expected_predictor,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                    )
                    scattered = np.empty(
                        (len(requested), sparse.shape[1]), dtype=np.float64
                    )
                    sparse.scatter_grid_at(requested, scattered)
                    self.assertTrue(
                        np.array_equal(scattered, sparse.grid_at(requested))
                    )
                projected = engine.sparse_projected_response(
                    ctx, antecedent, window, shape
                )
                self.assertTrue(
                    np.allclose(
                        projected.dense(),
                        dense @ shape.reshape(-1, 1),
                        rtol=0.0,
                        atol=2.0e-7,
                    )
                )

    def test_horizon_response_contains_only_the_prespecified_future_lags(self) -> None:
        data = self.data()
        ctx = make_context(data, "horizon-response", np.asarray([0, 1]))
        engine = RuleOccurrenceEngine(data, lag=3, knot_count=2)
        one_lag = engine.sparse_horizon_response(ctx, (0,), 0, 1)
        expected_rows = np.asarray(
            [
                ctx.grid_offsets[0] + 3,
                ctx.grid_offsets[1] + 5,
            ],
            dtype=np.int64,
        )
        self.assertTrue(np.array_equal(one_lag.grid_indices, expected_rows))
        self.assertTrue(
            np.allclose(
                one_lag.grid_values,
                np.repeat(engine.basis[:, :1].T, 2, axis=0),
                rtol=0.0,
                atol=0.0,
            )
        )
        self.assertIs(
            engine.sparse_horizon_response(ctx, (0,), 0, engine.lag),
            engine.sparse_response(ctx, (0,), 0),
        )

    def test_bit_identical_rule_responses_share_storage_without_merging_identities(self) -> None:
        base = self.data()
        duplicated = replace(
            base,
            predicates=np.column_stack((base.predicates[:, 0], base.predicates[:, 0])),
            predicate_names=("a", "a_duplicate"),
        )
        ctx = make_context(duplicated, "response-alias", np.asarray([0, 1]))
        engine = RuleOccurrenceEngine(
            duplicated,
            lag=3,
            knot_count=2,
            feature_cache_bytes=1024**2,
        )
        first = engine.sparse_response(ctx, (0,), 0)
        second = engine.sparse_response(ctx, (1,), 0)
        self.assertIs(first, second)
        self.assertEqual(engine.equivalent_response_hits, 1)
        self.assertNotEqual(RuleIdentity((0,), 0, 1), RuleIdentity((1,), 0, 1))
        self.assertIs(engine.sparse_response(ctx, (1,), 0), first)

    def test_sparse_response_alias_metadata_is_released_with_context_cache(self) -> None:
        base = self.data()
        duplicated = replace(
            base,
            predicates=np.column_stack((base.predicates[:, 0], base.predicates[:, 0])),
            predicate_names=("a", "a_duplicate"),
        )
        ctx = make_context(duplicated, "response-alias-release", np.asarray([0, 1]))
        engine = RuleOccurrenceEngine(
            duplicated,
            lag=3,
            knot_count=2,
            feature_cache_bytes=1024**2,
        )
        first = engine.sparse_response(ctx, (0,), 0)
        self.assertIs(engine.sparse_response(ctx, (1,), 0), first)
        self.assertTrue(engine._sparse_aliases)
        self.assertTrue(engine._sparse_equivalence_keys)
        engine.clear_context_cache(ctx.name)
        self.assertFalse(engine._sparse_aliases)
        self.assertFalse(engine._sparse_equivalence_keys)
        self.assertFalse(engine._sparse_key_fingerprints)
        self.assertFalse(engine._sparse_canonical_aliases)
        rebuilt = engine.sparse_response(ctx, (1,), 0)
        self.assertIsNot(rebuilt, first)

    def test_persistent_sparse_response_store_is_exact_and_cross_run_mmap(self) -> None:
        data = self.data()
        ctx = make_context(data, "persistent-response", np.asarray([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            writer = RuleOccurrenceEngine(
                data,
                lag=3,
                knot_count=2,
                feature_cache_bytes=1,
                persistent_response_dir=directory,
                persistent_response_bytes=1024**2,
            )
            expected = writer.persist_sparse_response(ctx, (0,), 0)
            shape = np.asarray([0.25, 0.75], dtype=np.float64)
            expected_projected = writer.persist_sparse_response(
                ctx, (0,), 0, shape=shape
            )
            self.assertEqual(writer.persistent_response_writes, 2)

            reader = RuleOccurrenceEngine(
                data,
                lag=3,
                knot_count=2,
                feature_cache_bytes=0,
                persistent_response_dir=directory,
                persistent_response_bytes=1024**2,
            )
            reader._build_sparse_response = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("persistent response was rebuilt")
            )
            observed = reader.sparse_response(ctx, (0,), 0)
            observed_projected = reader.sparse_projected_response(
                ctx, (0,), 0, shape
            )
            np.testing.assert_array_equal(
                observed.grid_indices, expected.grid_indices
            )
            np.testing.assert_array_equal(
                observed.grid_values, expected.grid_values
            )
            np.testing.assert_array_equal(
                observed.event_values, expected.event_values
            )
            np.testing.assert_array_equal(
                observed_projected.grid_values,
                expected_projected.grid_values,
            )
            self.assertIsInstance(observed.grid_values, np.memmap)
            self.assertGreaterEqual(reader.persistent_response_hits, 2)

    def test_persistent_mmaps_share_the_resident_feature_cache_budget(self) -> None:
        data = self.data()
        ctx = make_context(data, "persistent-resident-budget", np.asarray([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            writer = RuleOccurrenceEngine(
                data,
                lag=3,
                knot_count=2,
                feature_cache_bytes=1,
                persistent_response_dir=directory,
                persistent_response_bytes=1024**2,
            )
            raw = writer.persist_sparse_response(ctx, (0,), 0)
            shape = np.asarray([0.25, 0.75], dtype=np.float64)
            projected = writer.persist_sparse_response(
                ctx, (0,), 0, shape=shape
            )
            resident_limit = max(int(raw.nbytes), int(projected.nbytes))
            reader = RuleOccurrenceEngine(
                data,
                lag=3,
                knot_count=2,
                feature_cache_bytes=resident_limit,
                persistent_response_dir=directory,
                persistent_response_bytes=1024**2,
            )
            observed_raw = reader.sparse_response(ctx, (0,), 0)
            observed_projected = reader.sparse_projected_response(
                ctx, (0,), 0, shape
            )
            np.testing.assert_array_equal(
                observed_raw.grid_values, raw.grid_values
            )
            np.testing.assert_array_equal(
                observed_projected.grid_values, projected.grid_values
            )
            self.assertLessEqual(
                reader._feature_cache_bytes
                + reader._persistent_sparse_cache_bytes,
                resident_limit,
            )
            self.assertGreaterEqual(reader.persistent_response_evictions, 1)

    def test_persistent_response_digest_rejects_changed_event_data(self) -> None:
        data = self.data()
        with tempfile.TemporaryDirectory() as directory:
            original_ctx = make_context(
                data, "persistent-invalidation", np.asarray([0, 1])
            )
            writer = RuleOccurrenceEngine(
                data,
                lag=3,
                knot_count=2,
                persistent_response_dir=directory,
                persistent_response_bytes=1024**2,
            )
            original = writer.persist_sparse_response(
                original_ctx, (0,), 0
            )
            self.assertGreater(len(original.grid_indices), 0)
            changed_predicates = data.predicates.copy()
            changed_predicates[:, 0] = 0
            changed = replace(data, predicates=changed_predicates)
            changed_ctx = make_context(
                changed, "persistent-invalidation", np.asarray([0, 1])
            )
            reader = RuleOccurrenceEngine(
                changed,
                lag=3,
                knot_count=2,
                feature_cache_bytes=0,
                persistent_response_dir=directory,
                persistent_response_bytes=1024**2,
            )
            observed = reader.sparse_response(changed_ctx, (0,), 0)
            self.assertEqual(reader.persistent_response_hits, 0)
            self.assertEqual(len(observed.grid_indices), 0)

    def test_sparse_native_support_fit_equals_dense_support_fit(self) -> None:
        data = synthetic_data(seed=733, n_sequences=120)
        ctx = make_context(data, "sparse-fit", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=6, knot_count=3)
        rule = RuleIdentity((0,), 0, 1)
        dense_control = engine.response(ctx, (1,), 0)
        sparse_control = engine.sparse_response(ctx, (1,), 0)
        dense_feature = engine.response(ctx, (0,), 0)
        sparse_feature = engine.sparse_response(ctx, (0,), 0)
        dense_fit = fit_fixed_support(
            ctx,
            (dense_control,),
            [dense_feature],
            [rule],
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
        )
        sparse_fit = fit_fixed_support(
            ctx,
            (sparse_control,),
            [sparse_feature],
            [rule],
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-7,
        )
        self.assertTrue(dense_fit.converged and sparse_fit.converged)
        self.assertAlmostEqual(dense_fit.nll, sparse_fit.nll, places=9)
        self.assertTrue(np.allclose(dense_fit.gamma, sparse_fit.gamma, atol=1.0e-9))
        self.assertTrue(np.allclose(dense_fit.theta, sparse_fit.theta, atol=1.0e-9))

    def test_sparse_native_first_event_fit_equals_dense_fit(self) -> None:
        rows = []
        bounds = []
        for sequence in range(40):
            target_time = 5 if sequence % 4 == 0 else None
            end_time = target_time if target_time is not None else 8
            serialized = []
            for time in range(end_time + 1):
                source = int(time in {1, 3} and sequence % 3 != 0)
                target = int(target_time is not None and time == target_time)
                if source or target:
                    serialized.append(
                        {
                            "sequence_id": f"f{sequence}",
                            "position": len(serialized),
                            "month_index": time,
                            "target_token": target,
                            "a": source,
                        }
                    )
            rows.extend(serialized)
            bounds.append(
                {
                    "sequence_id": f"f{sequence}",
                    "start_month": 0,
                    "end_month": end_time,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
        )
        ctx = make_context(data, "sparse-first-event", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=4, knot_count=2)
        rule = RuleIdentity((0,), 0, 1)
        sparse = engine.sparse_response(ctx, (0,), 0)
        dense = sparse.dense()
        dense_fit = fit_fixed_support(
            ctx,
            (),
            [dense],
            [rule],
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-8,
            occurrence_likelihood="first_event_cloglog",
        )
        sparse_fit = fit_fixed_support(
            ctx,
            (),
            [sparse],
            [rule],
            device="cpu",
            dtype="float64",
            max_iter=120,
            tolerance=1.0e-8,
            occurrence_likelihood="first_event_cloglog",
        )
        self.assertTrue(dense_fit.converged and sparse_fit.converged)
        self.assertAlmostEqual(dense_fit.nll, sparse_fit.nll, places=10)
        self.assertTrue(
            np.allclose(dense_fit.theta, sparse_fit.theta, rtol=0.0, atol=1.0e-10)
        )

    def test_cumulative_window_responses_equal_independent_responses(self) -> None:
        data = synthetic_data(seed=79, n_sequences=90)
        ctx = make_context(data, "cumulative-window-equivalence", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=6, knot_count=3)
        antecedent = (0, 1)
        windows = engine.window_breakpoints(
            antecedent,
            ctx.global_sequence_ids,
            max_window=5,
        ).tolist()
        expected = {
            int(window): engine._build_response(ctx, antecedent, int(window))
            for window in windows
        }
        for window, event_part, grid_part, _updates in engine.iter_window_response_parts(
            ctx,
            antecedent,
            windows,
        ):
            actual = np.concatenate((event_part, grid_part), axis=0)
            self.assertTrue(
                np.allclose(actual, expected[window], rtol=0.0, atol=2.0e-7)
            )

    def test_cumulative_sparse_window_responses_equal_dense_responses(self) -> None:
        data = synthetic_data(seed=797, n_sequences=90)
        ctx = make_context(data, "cumulative-sparse", np.arange(data.n_sequences))
        engine = RuleOccurrenceEngine(data, lag=6, knot_count=3)
        antecedent = (0, 1)
        windows = engine.window_breakpoints(
            antecedent,
            ctx.global_sequence_ids,
            max_window=5,
            context=ctx,
        ).tolist()
        for window, sparse in engine.iter_window_sparse_responses(
            ctx, antecedent, windows
        ):
            dense = engine._build_response(ctx, antecedent, window)
            self.assertTrue(
                np.allclose(sparse.dense(), dense, rtol=0.0, atol=2.0e-7)
            )

    def test_rejected_formation_window_responses_can_be_evicted_exactly(self) -> None:
        data = self.data()
        ctx = make_context(data, "window-cache", np.array([0, 1]))
        engine = RuleOccurrenceEngine(data, lag=3, knot_count=2)
        antecedent = (0, 1)
        kept = engine.response(ctx, antecedent, 3).copy()
        engine.response(ctx, antecedent, 2)
        self.assertEqual(
            sum(key[0] == ctx.name and key[1] == antecedent for key in engine._feature_cache),
            2,
        )
        engine.retain_antecedent_windows(ctx.name, antecedent, (3,))
        self.assertEqual(
            sum(key[0] == ctx.name and key[1] == antecedent for key in engine._feature_cache),
            1,
        )
        self.assertTrue(np.array_equal(engine.response(ctx, antecedent, 3), kept))

    def test_completion_and_response_artifacts_share_the_byte_bounded_cache(self) -> None:
        data = self.data()
        ctx = make_context(data, "bounded-occurrence-cache", np.array([0, 1]))
        cache_limit = 256
        engine = RuleOccurrenceEngine(
            data,
            lag=3,
            knot_count=2,
            feature_cache_bytes=cache_limit,
        )
        expected = engine.response(ctx, (0, 1), 3).copy()
        engine.response(ctx, (0,), 0)
        engine.response(ctx, (1,), 0)
        self.assertLessEqual(engine._feature_cache_bytes, cache_limit)
        # An evicted exact artifact is recomputed, not approximated.
        self.assertTrue(np.array_equal(engine.response(ctx, (0, 1), 3), expected))

    def test_indexed_completions_equal_legacy_sequence_scan(self) -> None:
        base = self.data()
        third = ((base.times % 2) == 0).astype(np.uint8)
        data = replace(
            base,
            predicates=np.column_stack((base.predicates, third)),
            predicate_names=("a", "b", "c"),
        )
        engine = RuleOccurrenceEngine(data, lag=3, knot_count=2)

        def legacy(antecedent: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            seq_parts: list[np.ndarray] = []
            time_parts: list[np.ndarray] = []
            span_parts: list[np.ndarray] = []
            source_array = np.asarray(antecedent, dtype=np.int64)
            for seq_code, (left, right) in enumerate(data.sequence_slices):
                local_x = data.predicates[left:right, :][:, source_array]
                active_rows = np.flatnonzero(np.any(local_x > 0, axis=1))
                if active_rows.size == 0:
                    continue
                local_times = data.times[left:right]
                last = np.full(len(antecedent), np.iinfo(np.int64).min, dtype=np.int64)
                previous: tuple[int, ...] | None = None
                times: list[int] = []
                spans: list[int] = []
                for row in active_rows.tolist():
                    active_sources = np.flatnonzero(local_x[row] > 0)
                    now = int(local_times[row])
                    last[active_sources] = now
                    if np.any(last == np.iinfo(np.int64).min):
                        continue
                    witness = tuple(int(value) for value in last.tolist())
                    if witness == previous:
                        continue
                    previous = witness
                    times.append(now)
                    spans.append(int(np.max(last) - np.min(last)))
                if times:
                    seq_parts.append(np.full(len(times), seq_code, dtype=np.int32))
                    time_parts.append(np.asarray(times, dtype=np.int32))
                    span_parts.append(np.asarray(spans, dtype=np.int32))
            return (
                np.concatenate(seq_parts) if seq_parts else np.zeros(0, dtype=np.int32),
                np.concatenate(time_parts) if time_parts else np.zeros(0, dtype=np.int32),
                np.concatenate(span_parts) if span_parts else np.zeros(0, dtype=np.int32),
            )

        for antecedent in ((0,), (1,), (2,), (0, 1), (0, 1, 2)):
            expected = legacy(antecedent)
            actual = engine.completions(antecedent)
            self.assertTrue(np.array_equal(actual.sequence_codes, expected[0]))
            self.assertTrue(np.array_equal(actual.times, expected[1]))
            self.assertTrue(np.array_equal(actual.spans, expected[2]))

    def test_bounded_completion_index_preserves_every_reachable_window_response(self) -> None:
        data = synthetic_data(seed=907, n_sequences=75)
        ctx = make_context(data, "bounded-completions", np.arange(data.n_sequences))
        unrestricted = RuleOccurrenceEngine(data, lag=5, knot_count=3)
        bounded = RuleOccurrenceEngine(
            data, lag=5, knot_count=3, max_completion_span=2
        )
        antecedent = (0, 1)
        self.assertTrue(np.all(bounded.completions(antecedent).spans <= 2))
        for window in range(3):
            with self.subTest(window=window):
                expected = unrestricted.sparse_response(ctx, antecedent, window).dense()
                actual = bounded.sparse_response(ctx, antecedent, window).dense()
                self.assertTrue(np.array_equal(actual, expected))

    def test_control_free_baseline_matches_closed_form(self) -> None:
        data = self.data()
        ctx = make_context(data, "baseline", np.array([0, 1]))
        fit = fit_fixed_support(
            ctx,
            np.zeros((ctx.n_queries, 0), dtype=np.float32),
            [],
            [],
            device="cpu",
            dtype="float64",
            tolerance=1.0e-10,
        )
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.alpha, math.log(ctx.n_events / ctx.exposure), places=9)

    def test_fisher_standardized_kkt_is_feature_scale_invariant(self) -> None:
        data = self.data()
        ctx = make_context(data, "scale-invariance", np.array([0, 1]))
        feature = RuleOccurrenceEngine(data, lag=3, knot_count=2).response(ctx, (0,), 0)
        controls = np.zeros((ctx.n_queries, 0), dtype=np.float32)
        rule = RuleIdentity((0,), 0, 1)
        baseline = fit_fixed_support(
            ctx, controls, [], [], device="cpu", dtype="float32", max_iter=100
        )
        ordinary = fit_fixed_support(
            ctx,
            controls,
            [feature],
            [rule],
            device="cpu",
            dtype="float32",
            max_iter=100,
            initial=baseline,
        )
        scaled_feature = feature * np.float32(1.0e-3)
        scaled = fit_fixed_support(
            ctx,
            controls,
            [scaled_feature],
            [rule],
            device="cpu",
            dtype="float32",
            max_iter=100,
            initial=baseline,
        )
        self.assertTrue(ordinary.converged and scaled.converged)
        self.assertTrue(np.allclose(
            predict_eta(ordinary, controls, [feature]),
            predict_eta(scaled, controls, [scaled_feature]),
            rtol=2.0e-4,
            atol=2.0e-4,
        ))

    def test_nonfinite_warm_start_backs_off_without_changing_the_fit(self) -> None:
        data = self.data()
        ctx = make_context(data, "warm-start", np.array([0, 1]))
        unsafe = FitResult(
            rules=(),
            closure_terms=(),
            alpha=1_000.0,
            gamma=np.zeros(0, dtype=np.float64),
            theta=np.zeros((0, 0), dtype=np.float64),
            nll=math.inf,
            kkt_residual=math.inf,
            converged=False,
            iterations=0,
            device="cpu",
        )
        fit = fit_fixed_support(
            ctx,
            np.zeros((ctx.n_queries, 0), dtype=np.float32),
            [],
            [],
            device="cpu",
            dtype="float64",
            tolerance=1.0e-10,
            initial=unsafe,
        )
        self.assertTrue(fit.converged)
        self.assertTrue(math.isfinite(fit.nll))
        self.assertAlmostEqual(fit.alpha, math.log(ctx.n_events / ctx.exposure), places=9)

    def test_finite_worse_warm_start_uses_lower_objective_start(self) -> None:
        data = self.data()
        ctx = make_context(data, "finite-warm-start", np.array([0, 1]))
        finite_but_bad = FitResult(
            rules=(),
            closure_terms=(),
            alpha=20.0,
            gamma=np.zeros(0, dtype=np.float64),
            theta=np.zeros((0, 0), dtype=np.float64),
            nll=0.0,
            kkt_residual=0.0,
            converged=True,
            iterations=0,
            device="cpu",
        )
        fit = fit_fixed_support(
            ctx,
            np.zeros((ctx.n_queries, 0), dtype=np.float32),
            [],
            [],
            device="cpu",
            dtype="float64",
            tolerance=1.0e-10,
            initial=finite_but_bad,
        )
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(
            fit.alpha,
            math.log(ctx.n_events / ctx.exposure),
            places=9,
        )

    def test_bounds_only_sequence_is_kept_as_zero_event_exposure(self) -> None:
        frame = pd.DataFrame(
            [{"sequence_id": "seen", "position": 0, "month_index": 1, "target_token": 0, "a": 1}]
        )
        bounds = pd.DataFrame(
            [
                {"sequence_id": "seen", "start_month": 0, "end_month": 2},
                {"sequence_id": "empty", "start_month": 3, "end_month": 5},
            ]
        )
        data = EventData.from_frame(frame, predicate_names=("a",), bounds=bounds)
        self.assertEqual(data.n_sequences, 2)
        empty_code = int(np.flatnonzero(data.sequence_ids == "empty")[0])
        ctx = make_context(data, "with-empty", np.array([empty_code]))
        self.assertEqual(ctx.n_events, 0)
        self.assertEqual(ctx.exposure, 3.0)

    def test_sparse_loader_samples_from_sequence_exposure_not_event_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sequence_months").mkdir()
            (root / "sequences").mkdir()
            pd.DataFrame(
                [
                    {
                        "sequence_id": "eventful",
                        "position": 0,
                        "month_index": 0,
                        "target_token": 0,
                        "a": 1,
                    }
                ]
            ).to_parquet(root / "sequence_months" / "part.parquet", index=False)
            pd.DataFrame(
                [
                    {"sequence_id": "eventful", "start_month": 0, "end_month": 1},
                    {"sequence_id": "exposure_only", "start_month": 0, "end_month": 1},
                ]
            ).to_parquet(root / "sequences" / "part.parquet", index=False)
            loaded = load_event_data(
                root / "sequence_months",
                predicate_names=("a",),
                max_sequences=2,
            )
            self.assertEqual(set(loaded.sequence_ids.tolist()), {"eventful", "exposure_only"})

    def test_loader_prefers_numeric_month_index_bounds_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sequence_months").mkdir()
            (root / "sequences").mkdir()
            pd.DataFrame(
                [
                    {
                        "sequence_id": "loan",
                        "position": 0,
                        "month_index": 600,
                        "target_token": 0,
                        "a": 1,
                    }
                ]
            ).to_parquet(root / "sequence_months" / "part.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "sequence_id": "loan",
                        "start_month": "202001",
                        "end_month": "202003",
                        "start_month_index": 600,
                        "end_month_index": 602,
                    }
                ]
            ).to_parquet(root / "sequences" / "part.parquet", index=False)
            loaded = load_event_data(root / "sequence_months", predicate_names=("a",))
            self.assertTrue(np.array_equal(loaded.start_times, np.array([600])))
            self.assertTrue(np.array_equal(loaded.end_times, np.array([602])))

    def test_three_way_splits_are_disjoint_and_cover_every_sequence(self) -> None:
        data = synthetic_data(n_sequences=25)
        splits = split_contexts(data, seed=17)
        parts = (
            splits.fit.global_sequence_ids,
            splits.cert.global_sequence_ids,
            splits.test.global_sequence_ids,
        )
        combined = np.concatenate(parts)
        self.assertEqual(len(combined), data.n_sequences)
        self.assertEqual(len(np.unique(combined)), data.n_sequences)
        self.assertTrue(np.array_equal(np.sort(combined), np.arange(data.n_sequences)))

    def test_target_stratification_accepts_an_all_positive_marked_cohort(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sequence_id": f"s{i}",
                    "position": 0,
                    "month_index": 0,
                    "target_token": 1,
                    "target_mark_values": [float(i + 1)],
                    "a": 0,
                }
                for i in range(9)
            ]
        )
        data = EventData.from_frame(
            frame,
            predicate_names=("a",),
            mark_col="target_mark_values",
        )
        splits = split_contexts(data, seed=3, stratify_target=True)
        self.assertEqual(
            sum(part.n_sequences for part in (splits.fit, splits.cert, splits.test)),
            9,
        )
        self.assertTrue(
            all(part.n_events > 0 for part in (splits.fit, splits.cert, splits.test))
        )

    def test_explicit_financial_weights_define_the_certification_loss(self) -> None:
        frame = pd.DataFrame(
            [
                {"sequence_id": "s0", "position": 0, "month_index": 0, "target_token": 1, "a": 1},
                {"sequence_id": "s1", "position": 0, "month_index": 0, "target_token": 0, "a": 0},
            ]
        )
        bounds = pd.DataFrame(
            [
                {"sequence_id": "s0", "start_month": 0, "end_month": 1, "loss_exposure": 2.0},
                {"sequence_id": "s1", "start_month": 0, "end_month": 1, "loss_exposure": 5.0},
            ]
        )
        data = EventData.from_frame(
            frame,
            predicate_names=("a",),
            bounds=bounds,
            financial_weight_col="loss_exposure",
        )
        self.assertTrue(np.array_equal(data.sequence_financial_weights, np.array([2.0, 5.0])))
        ctx = make_context(data, "financial", np.array([0, 1]))
        eta = np.zeros(ctx.n_queries, dtype=np.float64)
        loss = financial_weighted_nll_loss(
            data.sequence_financial_weights,
            weight_name=data.financial_weight_name or "missing",
        )
        self.assertTrue(loss.financially_grounded)
        self.assertTrue(np.allclose(loss.values(eta, ctx), np.array([2.0, 5.0]) * cluster_nll(eta, ctx)))

    def test_financial_weights_change_ensemble_optimization_on_the_same_loss(self) -> None:
        frame = pd.DataFrame(
            [
                {"sequence_id": "s0", "position": 0, "month_index": 0, "target_token": 1, "a": 0},
                {"sequence_id": "s1", "position": 0, "month_index": 0, "target_token": 1, "a": 0},
            ]
        )
        data = EventData.from_frame(frame, predicate_names=("a",))
        ctx = make_context(data, "ensemble-financial", np.array([0, 1]))
        component_zero = np.array([math.log(10.0), math.log(0.1), 0.0, 0.0])
        component_one = np.array([math.log(0.1), math.log(10.0), 0.0, 0.0])
        unweighted = fit_intensity_ensemble(
            [component_zero, component_one], ctx, device="cpu", tolerance=1.0e-10
        )
        weighted = fit_intensity_ensemble(
            [component_zero, component_one],
            ctx,
            device="cpu",
            tolerance=1.0e-10,
            cluster_weights=np.array([100.0, 1.0]),
        )
        self.assertAlmostEqual(float(unweighted.weights[0]), 0.5, places=7)
        self.assertGreater(float(weighted.weights[0]), 0.95)

    def test_ensemble_event_and_grid_sufficient_statistics_equal_full_query_fit(self) -> None:
        frame = pd.DataFrame(
            [
                {"sequence_id": "s0", "position": 0, "month_index": 0, "target_token": 1, "a": 0},
                {"sequence_id": "s1", "position": 0, "month_index": 0, "target_token": 1, "a": 0},
            ]
        )
        bounds = pd.DataFrame(
            [
                {"sequence_id": "s0", "start_month": 0, "end_month": 2},
                {"sequence_id": "s1", "start_month": 0, "end_month": 2},
            ]
        )
        data = EventData.from_frame(frame, predicate_names=("a",), bounds=bounds)
        ctx = make_context(data, "ensemble-sufficient-statistics", np.array([0, 1]))
        components = [
            np.linspace(-0.7, 0.4, ctx.n_queries, dtype=np.float64),
            np.linspace(0.5, -0.3, ctx.n_queries, dtype=np.float64),
            np.full(ctx.n_queries, -0.1, dtype=np.float64),
        ]
        cluster_weights = np.array([2.5, 0.7], dtype=np.float64)
        full = fit_intensity_ensemble(
            components,
            ctx,
            device="cpu",
            tolerance=1.0e-11,
            cluster_weights=cluster_weights,
        )
        weighted_grid = (
            ctx.grid_weights.astype(np.float64)
            * cluster_weights[ctx.grid_sequence_local]
        )
        grid_integrals = np.asarray(
            [
                np.dot(weighted_grid, np.exp(component[ctx.n_events :]))
                for component in components
            ],
            dtype=np.float64,
        )
        summarized = fit_intensity_ensemble(
            [component[: ctx.n_events] for component in components],
            ctx,
            component_grid_integrals=grid_integrals,
            device="cpu",
            tolerance=1.0e-11,
            cluster_weights=cluster_weights,
        )
        self.assertEqual(full.converged, summarized.converged)
        self.assertTrue(np.allclose(full.weights, summarized.weights, rtol=1e-11, atol=1e-11))
        self.assertAlmostEqual(full.nll, summarized.nll, places=11)

    def test_ensemble_evaluation_sufficient_statistics_equal_dense_evaluation(self) -> None:
        data = synthetic_data(seed=812, n_sequences=45)
        ctx = make_context(data, "ensemble-evaluation", np.arange(data.n_sequences))
        baseline = np.linspace(-0.8, 0.2, ctx.n_queries, dtype=np.float64)
        ensemble = np.linspace(0.15, -0.55, ctx.n_queries, dtype=np.float64)
        weights = 0.5 + np.arange(ctx.n_sequences, dtype=np.float64) / ctx.n_sequences
        dense = evaluate_ensemble(
            ensemble,
            baseline,
            ctx,
            contribution_threshold=0.01,
            calibration_tolerance=0.4,
            alpha=0.1,
            cluster_weights=weights,
        )
        sufficient = evaluate_ensemble_sufficient(
            ensemble[: ctx.n_events],
            baseline[: ctx.n_events],
            ctx.aggregate_weighted_grid(np.exp(ensemble[ctx.n_events :])),
            ctx.aggregate_weighted_grid(np.exp(baseline[ctx.n_events :])),
            ctx,
            contribution_threshold=0.01,
            calibration_tolerance=0.4,
            alpha=0.1,
            cluster_weights=weights,
        )
        self.assertEqual(dense.keys(), sufficient.keys())
        self.assertEqual(dense["n_sequences"], sufficient["n_sequences"])
        self.assertEqual(dense["n_events"], sufficient["n_events"])
        for section in ("contribution", "calibration"):
            self.assertEqual(dense[section].keys(), sufficient[section].keys())
            for key, expected in dense[section].items():
                actual = sufficient[section][key]
                if isinstance(expected, bool):
                    self.assertEqual(expected, actual)
                else:
                    self.assertAlmostEqual(float(expected), float(actual), places=13)

    def test_cluster_weights_define_the_fixed_support_fit_objective(self) -> None:
        frame = pd.DataFrame(
            [
                {"sequence_id": "s0", "position": 0, "month_index": 0, "target_token": 1, "a": 0},
                {"sequence_id": "s1", "position": 0, "month_index": 0, "target_token": 0, "a": 0},
            ]
        )
        bounds = pd.DataFrame(
            [
                {"sequence_id": "s0", "start_month": 0, "end_month": 1},
                {"sequence_id": "s1", "start_month": 0, "end_month": 1},
            ]
        )
        data = EventData.from_frame(frame, predicate_names=("a",), bounds=bounds)
        ctx = make_context(data, "weighted-fit", np.array([0, 1]))
        weights = np.array([100.0, 1.0])
        fit = fit_fixed_support(
            ctx,
            np.zeros((ctx.n_queries, 0), dtype=np.float32),
            [],
            [],
            device="cpu",
            dtype="float64",
            tolerance=1.0e-10,
            cluster_weights=weights,
        )
        expected_rate = 100.0 / (101.0 * 2.0)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.alpha, math.log(expected_rate), places=9)
        eta = np.full(ctx.n_queries, fit.alpha)
        self.assertAlmostEqual(
            canonical_nll(eta, ctx, weights),
            float(np.dot(weights, cluster_nll(eta, ctx))),
            places=9,
        )


class HierarchyClosureTests(unittest.TestCase):
    def test_exact_fit_worker_limit_changes_scheduling_only(self) -> None:
        name = "CERTSCR_MAX_CONCURRENT_EXACT_FITS"
        previous = os.environ.get(name)
        try:
            os.environ[name] = "3"
            self.assertEqual(CertSCRPipeline._exact_fit_worker_limit(12), 3)
            self.assertEqual(CertSCRPipeline._exact_fit_worker_limit(2), 2)
            os.environ[name] = "0"
            with self.assertRaises(ValueError):
                CertSCRPipeline._exact_fit_worker_limit(12)
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    def test_first_event_support_calibration_is_not_mislabeled_hazard_error(self) -> None:
        rows: list[dict] = []
        bounds: list[dict] = []
        for sequence in range(100):
            exposed = sequence % 2 == 0
            event = sequence % 10 < (7 if exposed else 2)
            for month in range(6):
                rows.append(
                    {
                        "sequence_id": f"s{sequence}",
                        "position": month,
                        "month_index": month,
                        "target_token": int(event and month == 4),
                        "pred_a": int(exposed and month == 1),
                    }
                )
                if event and month == 4:
                    break
            bounds.append(
                {
                    "sequence_id": f"s{sequence}",
                    "start_month": 0,
                    "end_month": 4 if event else 5,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("pred_a",),
            bounds=pd.DataFrame(bounds),
            preprocessing_provenance={"target_process": "first_event"},
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=3,
                knot_count=2,
                max_formation_window=0,
                occurrence_likelihood="first_event_cloglog",
                solver_dtype="float64",
                solver_tolerance=1.0e-7,
                stratify_target_sequences=True,
            ),
        )
        rule = RuleIdentity((0,), 0, 1)
        fit = pipeline.fit_support((rule,))
        closure = pipeline.fit_model((), fit.closure_terms)
        self.assertTrue(fit.converged)
        record = SupportRecord(
            (rule,),
            fit,
            closure,
            float(closure.nll - fit.nll),
            "calibration-regression",
        )
        item = pipeline._evaluate_supports(
            [record], pipeline.splits.cert, alpha=0.05
        )[0]
        self.assertEqual(
            item["calibration"],
            {
                "gated": False,
                "estimate": None,
                "note": "probability calibration requires per-cell survival probabilities",
            },
        )
        profile_closures = ((((0,), 0),), (((0,), 1),))
        pipeline._prefit_profile_nulls(profile_closures)
        self.assertEqual(
            pipeline._safe_screen_stats["profile_null_models_batched"], 2
        )
        self.assertTrue(
            all(((), terms) in pipeline._fit_cache for terms in profile_closures)
        )

    def test_full_m_mdl_keeps_w_sign_code_but_drops_dictionary_shape_code(self) -> None:
        data = synthetic_data(n_sequences=20)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a",),
            config=CertSCRConfig(
                knot_count=3,
                max_formation_window=1,
                identity_profile="dictionary_mdl",
            ),
        )
        rule = RuleIdentity((0,), 0, 1)
        identities = tuple(
            RuleIdentity((0,), window, sign)
            for window in (0, 1)
            for sign in (-1, 1)
        )
        pipeline.identity_candidates[(0,)] = identities
        log_n = math.log(max(2, pipeline.splits.fit_population_sequence_count))
        identity_code = 2.0 * math.log(len(identities))
        shape_code = 2.0 * math.log(len(pipeline.kernel_dictionary))
        scalar_penalty = pipeline._support_complexity_penalty((rule,), 1)
        full_m_penalty = pipeline._support_complexity_penalty((rule,), 3)
        self.assertAlmostEqual(
            scalar_penalty,
            log_n + identity_code + shape_code,
            places=12,
        )
        self.assertAlmostEqual(
            full_m_penalty,
            3.0 * log_n + identity_code,
            places=12,
        )

    def test_dictionary_profile_selects_exact_best_finite_identity(self) -> None:
        data = synthetic_data(n_sequences=100)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                solver_dtype="float64",
                solver_tolerance=1.0e-7,
                identity_profile="dictionary_mdl",
            ),
        )
        selected = pipeline.profile_rule_identities()
        identities = pipeline.identity_candidates[(0,)]
        exact_scores = {
            rule: pipeline._support_search_score(
                pipeline._support_record((rule,), profile="test-exact-identity")
            )
            for rule in identities
        }
        best = min(
            identities,
            key=lambda rule: (-exact_scores[rule], rule),
        )
        log = next(item for item in pipeline.profile_logs if item["antecedent"] == [0])
        reported = RuleIdentity(
            (0,), int(log["selected_window"]), 1 if log["selected_sign"] > 0 else -1
        )
        self.assertEqual(reported, best)
        self.assertEqual(selected, [best] if exact_scores[best] > 0.0 else [])
        self.assertEqual(
            int(log["exact_fit_count"])
            + int(log["safely_eliminated_count"])
            + sum(
                item.get("exact_fit_status")
                == "zero_boundary_certified_by_null_cone_KKT"
                for item in log["candidates"]
            ),
            len(identities),
        )

    def test_fused_profile_reuses_live_window_response_for_exact_fit(self) -> None:
        data = synthetic_data(seed=2207, n_sequences=100)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                solver_device="cpu",
                solver_dtype="float64",
                solver_tolerance=1.0e-7,
                identity_profile="dictionary_mdl",
                safe_mdl_screen=False,
            ),
        )
        ordinary = pipeline.engine.sparse_response
        candidate_rebuilds: list[tuple[tuple[int, ...], int]] = []

        def counted(ctx, antecedent, window):
            if tuple(antecedent) == (0,):
                candidate_rebuilds.append((tuple(antecedent), int(window)))
            return ordinary(ctx, antecedent, window)

        pipeline.engine.sparse_response = counted
        pipeline.profile_rule_identities()
        self.assertEqual(candidate_rebuilds, [])
        log = next(
            item for item in pipeline.profile_logs if item["antecedent"] == [0]
        )
        self.assertEqual(
            pipeline._safe_screen_stats["identity_fused_profile_exact_fits"],
            int(log["exact_fit_count"]),
        )

    def test_profile_parent_representatives_preserve_canonical_atom_design(self) -> None:
        data = synthetic_data(seed=2208, n_sequences=120)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=2,
                max_formation_window=3,
                solver_device="cpu",
                solver_dtype="float64",
                identity_profile="dictionary_mdl",
                safe_mdl_screen=False,
            ),
        )
        rule = RuleIdentity((0, 1), 2, 1)
        closure = pipeline.hierarchy_closure((rule,))
        response = pipeline.engine.sparse_response(
            pipeline.splits.fit, rule.antecedent, rule.window
        )
        nuisance = pipeline.sparse_nuisance_blocks(
            pipeline.splits.fit, closure
        )
        parent = prepare_fixed_support_design(
            pipeline.splits.fit,
            nuisance,
            [response],
            [rule],
            cluster_weights=pipeline.fit_cluster_weights,
            sequence_exposures=pipeline.sequence_exposures(
                pipeline.splits.fit
            ),
            occurrence_likelihood=pipeline.occurrence_likelihood,
        )
        self.assertIsNotNone(parent.representative_rows)
        shape = pipeline.kernel_dictionary[-1]
        candidates = (
            RuleIdentity(rule.antecedent, rule.window, -1),
            rule,
        )
        for candidate in candidates:
            pipeline.rule_dictionary_shapes[candidate] = shape.copy()
        pipeline._stage_profile_window_designs(
            candidates,
            response,
            full_m_parent=parent,
        )
        for candidate in candidates:
            staged = pipeline._prepared_design_cache.pop(
                ((candidate,), closure)
            )
            direct = prepare_fixed_support_design(
                pipeline.splits.fit,
                nuisance,
                [response.projected(shape)],
                [candidate],
                cluster_weights=pipeline.fit_cluster_weights,
                sequence_exposures=pipeline.sequence_exposures(
                    pipeline.splits.fit
                ),
                occurrence_likelihood=pipeline.occurrence_likelihood,
            )
            self.assertTrue(np.array_equal(staged.design, direct.design))
            self.assertTrue(
                np.array_equal(staged.event_weights, direct.event_weights)
            )
            self.assertTrue(
                np.array_equal(staged.grid_weights, direct.grid_weights)
            )

    def test_grouped_parent_identity_moments_equal_sparse_moments(self) -> None:
        data = synthetic_data(seed=2211, n_sequences=130)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=2,
                max_formation_window=3,
                solver_device="cpu",
                solver_dtype="float64",
                identity_profile="dictionary_mdl",
                safe_mdl_screen=False,
            ),
        )
        rule = RuleIdentity((0, 1), 2, 1)
        closure = pipeline.hierarchy_closure((rule,))
        null = pipeline.fit_model((), closure)
        response = pipeline.engine.sparse_response(
            pipeline.splits.fit, rule.antecedent, rule.window
        )
        parent = pipeline.prepare_partitioned_support_design(
            pipeline.splits.fit,
            closure,
            (response,),
            (rule,),
            cluster_weights=pipeline.fit_cluster_weights,
        )
        sparse_gradient, sparse_information = pipeline._identity_moments_at_null(
            null, response
        )
        grouped_gradient, grouped_information = (
            pipeline._identity_moments_from_grouped_parent(null, parent)
        )
        self.assertTrue(
            np.allclose(sparse_gradient, grouped_gradient, rtol=1e-12, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(sparse_information, grouped_information, rtol=1e-12, atol=1e-12)
        )

    def test_rule_irreducibility_uses_global_and_horizon_iut(self) -> None:
        self.assertEqual(
            CertSCRPipeline._rule_irreducibility_p_value(
                0.19, {"p_value": 1.0e-9}
            ),
            0.19,
        )
        self.assertEqual(
            CertSCRPipeline._rule_irreducibility_p_value(0.03, None),
            0.03,
        )

    def test_inhibition_recession_column_is_rejected_without_count_threshold(self) -> None:
        rule = RuleIdentity((0, 1, 2), 9, -1)
        unsupported = PreparedFixedSupportDesign(
            design=np.asarray([[1.0, 0.0], [1.0, -1.0]]),
            n_events=1,
            event_weights=np.asarray([1.0]),
            grid_weights=np.asarray([1.0]),
            constrained_start=1,
            control_width=0,
            knot_count=1,
            active_grid_rows=1,
            rules=(rule,),
        )
        self.assertEqual(
            CertSCRPipeline._nonattained_rule_recession_columns(unsupported),
            ((0, 0),),
        )
        supported = replace(
            unsupported,
            design=np.asarray([[1.0, -1.0], [1.0, -1.0]]),
        )
        self.assertEqual(
            CertSCRPipeline._nonattained_rule_recession_columns(supported),
            (),
        )
        excitation = replace(
            unsupported,
            rules=(replace(rule, sign=1),),
            design=np.asarray([[1.0, 0.0], [1.0, 1.0]]),
        )
        self.assertEqual(
            CertSCRPipeline._nonattained_rule_recession_columns(excitation),
            (),
        )

    def test_primary_family_keeps_pair_and_triplet_standalone_supports(self) -> None:
        base = synthetic_data(seed=1201, n_sequences=60)
        data = replace(
            base,
            predicates=np.column_stack(
                (base.predicates, np.zeros(len(base.predicates), dtype=np.int8))
            ),
            predicate_names=("pred_a", "pred_b", "pred_c"),
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b", "pred_c"),
            config=CertSCRConfig(
                q_max=3,
                impact_lag=3,
                knot_count=1,
                max_formation_window=2,
                max_support_size=2,
                solver_device="cpu",
                identity_profile="exact",
                support_conditioned_refinement=False,
                safe_mdl_screen=False,
            ),
        )
        pair = RuleIdentity((0, 1), 1, 1)
        triplet = RuleIdentity((0, 1, 2), 2, -1)
        pipeline.profiled_rules = [pair, triplet]

        def fake_batch(
            owner: CertSCRPipeline,
            rule_sets: list[tuple[RuleIdentity, ...]],
            *,
            profile: str,
        ) -> list[SupportRecord]:
            output: list[SupportRecord] = []
            for values in rule_sets:
                rules = tuple(sorted(values))
                null = FitResult(
                    rules=(),
                    closure_terms=(),
                    alpha=-3.0,
                    gamma=np.zeros(0),
                    theta=np.zeros((0, 1)),
                    nll=100.0,
                    kkt_residual=0.0,
                    converged=True,
                    iterations=0,
                    device="cpu",
                )
                fit = replace(
                    null,
                    rules=rules,
                    theta=np.ones((len(rules), 1)),
                    nll=100.0 - 25.0 * len(rules),
                )
                output.append(
                    SupportRecord(
                        rules=rules,
                        fit=fit,
                        closure_baseline_fit=null,
                        search_nll_improvement=100.0 - fit.nll,
                        profile=profile,
                    )
                )
            return output

        pipeline._fit_or_safe_screen_records_batch = types.MethodType(
            fake_batch, pipeline
        )
        records = pipeline._search_supports_active_set()
        returned = {record.rules for record in records}
        self.assertIn((pair,), returned)
        self.assertIn((triplet,), returned)
        self.assertEqual(
            pipeline.search_diagnostics["triplet_anchor_policy"],
            "every_positive_profiled_triplet_has_a_standalone_support",
        )
        self.assertEqual(
            pipeline.search_diagnostics["returned_family_definition"],
            "all_positive_standalone_atoms_and_unique_atom_start_local_terminals",
        )

    def test_fit_summary_reuse_metadata_can_be_replanned_and_cleared(self) -> None:
        data = synthetic_data(seed=1200, n_sequences=50)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=2,
                max_formation_window=3,
                solver_device="cpu",
                solver_dtype="float64",
                solver_workers=1,
                fit_summary_cache_bytes=1024**2,
            ),
        )
        fit = pipeline.fit_baseline()
        record = SupportRecord(
            rules=(),
            fit=fit,
            closure_baseline_fit=fit,
            search_nll_improvement=0.0,
            profile="test",
        )
        # The same model appears as both full and branch-drop in this empty-rule
        # test record, making it eligible without requiring a discovery run.
        key = pipeline._fit_summary_key(fit, pipeline.splits.fit)
        pipeline._fit_summary_cacheable_keys.add(key)
        pipeline._prepare_fit_summary_reuse([record], pipeline.splits.fit)
        pipeline._fit_summary_cacheable_keys.add(key)
        pipeline._cached_sparse_fit_summary(fit, pipeline.splits.fit)
        pipeline._clear_fit_summary_context(pipeline.splits.fit)
        self.assertNotIn(key, pipeline._fit_summary_cacheable_keys)
        self.assertEqual(pipeline._fit_summary_cache_size[0], 0)

    def test_hierarchy_null_loss_cache_preserves_exact_entity_statistics(self) -> None:
        data = synthetic_data(seed=1201, n_sequences=90)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=2,
                max_formation_window=3,
                solver_device="cpu",
                solver_dtype="float64",
                solver_workers=1,
                loss_summary_cache_bytes=1024**2,
            ),
        )
        fit = pipeline.fit_baseline()
        direct = pipeline._sparse_fit_summary(fit, pipeline.splits.fit)
        first = pipeline._closure_loss_summary(fit, pipeline.splits.fit)
        second = pipeline._closure_loss_summary(fit, pipeline.splits.fit)
        self.assertIs(first, second)
        self.assertTrue(np.array_equal(first.cluster_nll, direct.cluster_nll))
        self.assertTrue(
            np.array_equal(first.cluster_intensity, direct.cluster_intensity)
        )
        self.assertEqual(pipeline._loss_summary_cache_stats["misses"], 1)
        self.assertEqual(pipeline._loss_summary_cache_stats["hits"], 1)
        self.assertGreater(pipeline._loss_summary_cache_size[0], 0)
        pipeline._clear_loss_summary_context(pipeline.splits.fit)
        self.assertEqual(pipeline._loss_summary_cache_size[0], 0)
        self.assertEqual(len(pipeline._loss_summary_cache), 0)
        self.assertFalse(
            any(
                key[1] == id(pipeline.splits.fit)
                for key in pipeline._loss_summary_key_locks
            )
        )

    def test_final_cell_target_history_is_removed_as_an_exact_zero_block(self) -> None:
        rows = []
        bounds = []
        for sequence in range(6):
            rows.extend(
                [
                    {
                        "sequence_id": f"f{sequence}",
                        "position": 0,
                        "month_index": 0,
                        "target_token": 0,
                        "a": sequence % 2,
                    },
                    {
                        "sequence_id": f"f{sequence}",
                        "position": 1,
                        "month_index": 2,
                        "target_token": 1,
                        "a": 0,
                    },
                ]
            )
            bounds.append(
                {
                    "sequence_id": f"f{sequence}",
                    "start_month": 0,
                    "end_month": 2,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
            preprocessing_provenance={"target_process": "first_event"},
        )
        common = dict(
            q_max=1,
            impact_lag=2,
            knot_count=2,
            split_seed=17,
            solver_device="cpu",
            solver_dtype="float64",
        )
        requested = CertSCRPipeline(
            data,
            rule_predicates=("a",),
            config=CertSCRConfig(**common, target_history_control=True),
        )
        ordinary = CertSCRPipeline(
            data,
            rule_predicates=("a",),
            config=CertSCRConfig(**common, target_history_control=False),
        )
        self.assertTrue(requested.target_history_control_requested)
        self.assertTrue(requested.target_history_structural_zero)
        self.assertIsNone(requested.target_history_source_id)
        left = requested.fit_baseline()
        right = ordinary.fit_baseline()
        self.assertAlmostEqual(left.nll, right.nll, places=14)
        self.assertAlmostEqual(left.alpha, right.alpha, places=14)
        self.assertEqual(len(left.gamma), 0)

    def test_target_history_control_is_causal_and_not_a_rule_source(self) -> None:
        rows: list[dict] = []
        bounds: list[dict] = []
        for sequence in range(3):
            for time, target, pred in ((0, 0, 1), (1, 2, 0), (3, 1, 0)):
                rows.append(
                    {
                        "sequence_id": f"r{sequence}",
                        "position": len([row for row in rows if row["sequence_id"] == f"r{sequence}"]),
                        "month_index": time,
                        "target_token": target,
                        "a": pred,
                    }
                )
            bounds.append(
                {"sequence_id": f"r{sequence}", "start_month": 0, "end_month": 4}
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=3,
                knot_count=3,
                target_history_control=True,
                solver_device="cpu",
            ),
        )
        ctx = make_context(data, "target-history-audit", np.arange(3))
        controls = pipeline.control_blocks(ctx)
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0].shape, (ctx.n_queries, 3))
        first_events = np.flatnonzero(ctx.event_times == 1)
        recurrent_events = np.flatnonzero(ctx.event_times == 3)
        self.assertTrue(np.allclose(controls[0][first_events], 0.0))
        self.assertTrue(np.all(np.sum(controls[0][recurrent_events], axis=1) > 0.0))
        self.assertTrue(
            np.allclose(np.sum(controls[0][recurrent_events], axis=1), 2.0)
        )
        self.assertNotIn(pipeline.target_history_source_id, pipeline.rule_source_ids)
        report = pipeline._f0_contract()
        self.assertEqual(report["unreviewed_control_count"], 0)
        self.assertTrue(report["target_history_control"]["enabled"])
        self.assertNotIn(
            "control_predicates_have_no_registered_predictability_contract",
            report["failure_reasons"],
        )

    def test_recurrent_f0_requires_the_registered_target_history_nuisance(self) -> None:
        data = replace(
            synthetic_data(seed=993, n_sequences=24),
            preprocessing_provenance={
                "target_process": "recurrent",
                "f0_contract": {
                    "dynamic_predicates": True,
                    "outcome_blind_predicate_construction": True,
                    "direct_target_proxy_excluded": True,
                    "predicate_history_includes_target_labeled_observations": True,
                    "strict_future_effect_required": True,
                },
            },
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                adverse_event_name="pre-specified adverse event",
                target_history_control=False,
                solver_device="cpu",
            ),
        )
        pipeline.predicate_policy_name = "test_registered_policy"
        pipeline.predicate_policy_contract = PredicatePolicyContract(
            predicates=("pred_a", "pred_b"),
            dynamic=True,
            outcome_blind_construction=True,
            direct_target_proxy_excluded=True,
            review_basis="unit test",
            atomic_events=True,
        )
        report = pipeline._f0_contract()
        self.assertFalse(report["passed"])
        self.assertIn(
            "recurrent_target_history_control_missing",
            report["failure_reasons"],
        )

    def test_extra_source_stream_offsets_and_order_are_validated(self) -> None:
        data = synthetic_data(seed=994, n_sequences=12)
        invalid = SourceEvents(
            sequence_codes=np.asarray([1, 0], dtype=np.int32),
            times=np.asarray([2, 1], dtype=np.int32),
            offsets=np.asarray([0, 1, 2] + [2] * 10, dtype=np.int64),
            populated_sequences=np.asarray([0, 1], dtype=np.int32),
        )
        with self.assertRaisesRegex(ValueError, "sequence/time ordered"):
            RuleOccurrenceEngine(
                data,
                lag=2,
                knot_count=2,
                extra_source_events={data.n_predicates: invalid},
            )

    def test_f0_rejects_unregistered_manual_predicates(self) -> None:
        data = synthetic_data(seed=991, n_sequences=12)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                adverse_event_name="pre-specified adverse event",
                solver_device="cpu",
            ),
        )
        report = pipeline._f0_contract()
        self.assertFalse(report["passed"])
        self.assertIn(
            "rule_predicate_policy_not_pre_registered",
            report["failure_reasons"],
        )

    def test_f0_rejects_outcome_conditioned_recurrent_history_metadata(self) -> None:
        data = replace(
            synthetic_data(seed=992, n_sequences=12),
            preprocessing_provenance={
                "leakage_policy": {
                    "is_laundering": "used only as target_token",
                    "pattern_or_typology_labels": "not used",
                    "laundering_transaction_predicates": False,
                }
            },
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                adverse_event_name="pre-specified adverse event",
                solver_device="cpu",
            ),
        )
        pipeline.predicate_policy_name = "test_registered_policy"
        pipeline.predicate_policy_contract = PredicatePolicyContract(
            predicates=("pred_a", "pred_b"),
            dynamic=True,
            outcome_blind_construction=True,
            direct_target_proxy_excluded=True,
            review_basis="unit test",
            atomic_events=True,
        )
        report = pipeline._f0_contract()
        self.assertFalse(report["passed"])
        self.assertIn(
            "dataset_preprocessing_provenance_not_F0_verified",
            report["failure_reasons"],
        )

    def test_registered_predicate_policies_have_target_blind_f0_contracts(self) -> None:
        for name in (
            "home_credit_behavioral_nonproxy_expanded",
            "freddie_structural_dynamic_v2",
            "ibm_aml_primitive_dynamic_v1",
        ):
            contract = resolve_predicate_policy_contract(name)
            self.assertEqual(contract.predicates, resolve_predicate_policy(name))
            self.assertTrue(contract.f0_eligible)

    def test_early_warning_probability_contrast_respects_rule_sign(self) -> None:
        data = synthetic_data(seed=991, n_sequences=90)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                certification_mode="early_warning",
                early_warning_horizon=2,
                solver_device="cpu",
                solver_dtype="float64",
            ),
        )
        for sign in (1, -1):
            rule = RuleIdentity((0,), 0, sign)
            fit = FitResult(
                rules=(rule,),
                closure_terms=(),
                alpha=-3.0,
                gamma=np.zeros(0, dtype=np.float64),
                theta=np.asarray([[0.8, 0.2]], dtype=np.float64),
                nll=0.0,
                kkt_residual=0.0,
                converged=True,
                iterations=0,
                device="cpu",
            )
            drop_fit = FitResult(
                rules=(),
                closure_terms=(),
                alpha=-3.0,
                gamma=np.zeros(0, dtype=np.float64),
                theta=np.zeros((0, 2), dtype=np.float64),
                nll=0.0,
                kkt_residual=0.0,
                converged=True,
                iterations=0,
                device="cpu",
            )
            report = pipeline._early_warning_rule_report(
                fit,
                drop_fit,
                pipeline._sparse_fit_summary(fit, pipeline.splits.cert),
                pipeline._sparse_fit_summary(drop_fit, pipeline.splits.cert),
                pipeline.splits.cert,
                0,
                alpha=0.05,
            )
            self.assertTrue(report["testable"])
            self.assertGreater(
                report["sign_aligned_probability_shift"]["estimate"], 0.0
            )
            self.assertEqual(report["raw_probability_shift"] > 0.0, sign > 0)
            self.assertFalse(report["causal_interpretation"])

    def test_default_support_search_has_no_artificial_size_cap(self) -> None:
        data = synthetic_data(seed=903, n_sequences=80)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                solver_device="cpu",
            ),
        )
        rules = (
            RuleIdentity((0,), 0, 1),
            RuleIdentity((1,), 0, 1),
            RuleIdentity((0, 1), 1, 1),
        )
        pipeline.profiled_rules = list(rules)
        self.assertIsNone(pipeline.config.max_support_size)
        self.assertTrue(pipeline._eligible_support(rules))
        self.assertIn(tuple(sorted(rules)), pipeline._one_exchange_neighbors(rules[:2]))

    def test_nested_null_warm_start_preserves_cold_fit_optimum(self) -> None:
        data = synthetic_data(seed=904, n_sequences=180)
        config = CertSCRConfig(
            q_max=2,
            impact_lag=4,
            knot_count=2,
            max_formation_window=2,
            max_support_size=1,
            solver_device="cpu",
            solver_dtype="float64",
            solver_tolerance=1.0e-6,
            solver_max_iter=200,
        )
        warm = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        small_closure = (((0,), 0),)
        large_closure = (((0,), 0), ((1,), 0))
        self.assertTrue(warm.fit_model((), small_closure).converged)
        warm_fit = warm.fit_model((), large_closure)

        cold = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        cold_fit = cold.fit_model((), large_closure)
        self.assertTrue(warm_fit.converged)
        self.assertTrue(cold_fit.converged)
        self.assertLessEqual(abs(warm_fit.nll - cold_fit.nll), 1.0e-6)
        self.assertEqual(warm._safe_screen_stats["nested_null_warm_starts"], 1)

    def test_child_kkt_shortcut_returns_the_same_zero_rule_optimum(self) -> None:
        rows = []
        bounds = []
        for index in range(18):
            rows.append(
                {
                    "sequence_id": f"k{index}",
                    "position": 0,
                    "month_index": 0,
                    "target_token": int(index % 3 == 0),
                    "a": 0,
                }
            )
            bounds.append(
                {"sequence_id": f"k{index}", "start_month": 0, "end_month": 2}
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=2,
                knot_count=2,
                max_support_size=1,
                solver_device="cpu",
                solver_dtype="float64",
                solver_tolerance=1.0e-8,
            ),
        )
        baseline = pipeline.fit_baseline()
        fit = pipeline.fit_support((RuleIdentity((0,), 0, 1),))
        self.assertTrue(fit.converged)
        self.assertEqual(fit.iterations, 0)
        self.assertTrue(np.array_equal(fit.theta, np.zeros((1, 2))))
        self.assertAlmostEqual(fit.nll, baseline.nll, places=12)
        self.assertEqual(pipeline._safe_screen_stats["child_kkt_shortcuts"], 1)

    def test_marked_child_kkt_shortcut_preserves_the_joint_null_objective(self) -> None:
        rows = []
        bounds = []
        for index in range(30):
            count = int(index % 2 == 0)
            rows.append(
                {
                    "sequence_id": f"mk{index}",
                    "position": 0,
                    "month_index": 0,
                    "target_token": count,
                    "target_marks": ([float(index + 1)] if count else []),
                    "a": 0,
                }
            )
            bounds.append(
                {"sequence_id": f"mk{index}", "start_month": 0, "end_month": 2}
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
            mark_col="target_marks",
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=2,
                knot_count=2,
                max_support_size=1,
                identity_profile="exact",
                solver_device="cpu",
                solver_dtype="float64",
                solver_tolerance=1.0e-8,
            ),
        )
        baseline = pipeline.fit_baseline()
        fit = pipeline.fit_support((RuleIdentity((0,), 0, 1),))
        self.assertTrue(fit.converged)
        self.assertEqual(fit.iterations, 0)
        self.assertIsNotNone(fit.mark_fit)
        self.assertTrue(np.array_equal(fit.theta, np.zeros((1, 2))))
        self.assertTrue(np.array_equal(fit.mark_fit.rule_beta, np.zeros(1)))
        self.assertAlmostEqual(fit.nll, baseline.nll, places=12)

    def test_sparse_fit_summary_equals_dense_predictor_and_cluster_loss(self) -> None:
        data = synthetic_data(seed=911, n_sequences=70)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                max_support_size=1,
                solver_device="cpu",
                solver_dtype="float64",
            ),
        )
        ctx = pipeline.splits.cert
        rule = RuleIdentity((0,), 0, 1)
        fit = FitResult(
            rules=(rule,),
            closure_terms=(),
            alpha=-1.2,
            gamma=np.zeros(0, dtype=np.float64),
            theta=np.asarray([[0.15, 0.35]], dtype=np.float64),
            nll=0.0,
            kkt_residual=0.0,
            converged=True,
            iterations=0,
            device="cpu",
        )
        dense_eta = pipeline._eta_on(fit, ctx)
        summary = pipeline._sparse_fit_summary(fit, ctx)
        self.assertTrue(
            np.allclose(summary.event_eta, dense_eta[: ctx.n_events], rtol=0.0, atol=1e-14)
        )
        self.assertTrue(
            np.allclose(summary.cluster_nll, cluster_nll(dense_eta, ctx), rtol=1e-13, atol=1e-13)
        )
        self.assertLess(len(summary.active_grid_indices), ctx.n_grid)
        rows = np.arange(ctx.n_grid, dtype=np.int64)
        self.assertTrue(
            np.allclose(
                pipeline._summary_eta_at(fit, summary, rows),
                dense_eta[ctx.n_events :],
                rtol=0.0,
                atol=1e-14,
            )
        )

    def test_sparse_fisher_projection_and_score_equal_dense_reference(self) -> None:
        data = synthetic_data(seed=919, n_sequences=80)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                max_support_size=1,
                solver_device="cpu",
                solver_dtype="float64",
            ),
        )
        ctx = pipeline.splits.cert
        null_fit = FitResult(
            rules=(),
            closure_terms=(),
            alpha=-0.7,
            gamma=np.zeros(0, dtype=np.float64),
            theta=np.zeros((0, 0), dtype=np.float64),
            nll=0.0,
            kkt_residual=0.0,
            converged=True,
            iterations=0,
            device="cpu",
        )
        sparse_feature = pipeline.engine.sparse_response(ctx, (0,), 0)
        dense_feature = sparse_feature.dense()
        shape = np.asarray([0.3, 0.7], dtype=np.float64)
        sequence_weights = np.linspace(0.4, 1.6, ctx.n_sequences, dtype=np.float64)
        sparse_info, sparse_rank, sparse_score, sparse_raw = pipeline._sparse_rule_information(
            null_fit,
            ctx,
            sparse_feature,
            (),
            shape,
            sequence_weights,
        )
        eta_null = pipeline._eta_on(null_fit, ctx)
        intercept = np.ones((ctx.n_queries, 1), dtype=np.float64)
        residual, dense_info, dense_rank = efficient_information_matrix(
            dense_feature,
            intercept,
            eta_null,
            ctx,
            sequence_weights,
            projected_shape=shape,
        )
        dense_score = cluster_directional_score(
            residual,
            eta_null,
            ctx,
            sign=1,
            cluster_weights=sequence_weights,
        )
        raw_z = dense_feature[ctx.n_events :] @ shape
        raw_weight = (
            ctx.expand_sequence_values(sequence_weights)
            * np.exp(eta_null[ctx.n_events :])
        )
        dense_raw = float(np.dot(raw_weight, raw_z * raw_z))
        self.assertEqual(sparse_rank, dense_rank)
        self.assertTrue(np.allclose(sparse_info, dense_info, rtol=2e-12, atol=2e-12))
        self.assertTrue(np.allclose(sparse_score, dense_score, rtol=2e-12, atol=2e-12))
        self.assertAlmostEqual(sparse_raw, dense_raw, places=12)

    def test_grouped_fisher_reuse_equals_sparse_path_with_inhibitory_nuisance(self) -> None:
        data = synthetic_data(seed=920, n_sequences=100)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                solver_device="cpu",
                solver_dtype="float64",
            ),
        )
        ctx = pipeline.splits.fit
        focal = RuleIdentity((0,), 0, 1)
        inhibitory = RuleIdentity((1,), 0, -1)
        full_fit = FitResult(
            rules=(focal, inhibitory),
            closure_terms=(),
            alpha=-0.8,
            gamma=np.zeros(0, dtype=np.float64),
            theta=np.asarray([[0.2, 0.1], [0.15, 0.05]], dtype=np.float64),
            nll=0.0,
            kkt_residual=0.0,
            converged=True,
            iterations=0,
            device="cpu",
        )
        drop_fit = replace(
            full_fit,
            rules=(inhibitory,),
            theta=np.asarray([[0.15, 0.05]], dtype=np.float64),
        )
        weights = np.linspace(0.5, 1.5, ctx.n_sequences, dtype=np.float64)
        raw_feature = pipeline.engine.sparse_response(ctx, focal.antecedent, 0)
        nuisance = tuple(pipeline.sparse_features(ctx, (inhibitory,)))
        shape = np.asarray([0.35, 0.65], dtype=np.float64)
        summary = pipeline._sparse_fit_summary(drop_fit, ctx)
        sparse = pipeline._sparse_rule_information(
            drop_fit,
            ctx,
            raw_feature,
            nuisance,
            shape,
            weights,
            null_summary=summary,
        )
        prepared = pipeline._prepare_support_information_design(
            full_fit, ctx, weights
        )
        grouped = pipeline._sparse_rule_information(
            drop_fit,
            ctx,
            raw_feature,
            nuisance,
            shape,
            weights,
            null_summary=summary,
            prepared_full=prepared,
            focal_rule=focal,
            remaining_rules=(inhibitory,),
        )
        self.assertEqual(sparse[1], grouped[1])
        for expected, actual in zip(
            (sparse[0], sparse[2], sparse[3]),
            (grouped[0], grouped[2], grouped[3]),
            strict=True,
        ):
            self.assertTrue(
                np.allclose(expected, actual, rtol=5e-11, atol=5e-11)
            )

    def test_nested_full_design_projection_preserves_changed_hierarchy_closure(self) -> None:
        data = synthetic_data(seed=921, n_sequences=120)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                solver_device="cpu",
                solver_dtype="float64",
            ),
        )
        ctx = pipeline.splits.fit
        a = RuleIdentity((0,), 0, -1)
        ab = RuleIdentity((0, 1), 2, 1)
        parent_rules = (a, ab)
        child_rules = (ab,)
        parent_closure = pipeline.hierarchy_closure(parent_rules)
        child_closure = pipeline.hierarchy_closure(child_rules)
        self.assertEqual(parent_closure, (((1,), 0),))
        self.assertEqual(child_closure, (((0,), 0), ((1,), 0)))
        parent = prepare_fixed_support_design(
            ctx,
            pipeline.sparse_nuisance_blocks(ctx, parent_closure),
            pipeline.sparse_features(ctx, parent_rules),
            parent_rules,
            sequence_exposures=pipeline.sequence_exposures(ctx),
        )
        projected = project_prepared_support_design(
            parent,
            child_rules,
            source_closure_terms=parent_closure,
            target_closure_terms=child_closure,
            regroup=True,
        )
        fresh = prepare_fixed_support_design(
            ctx,
            pipeline.sparse_nuisance_blocks(ctx, child_closure),
            pipeline.sparse_features(ctx, child_rules),
            child_rules,
            sequence_exposures=pipeline.sequence_exposures(ctx),
        )

        def ordered_parts(prepared):
            output = []
            for design, weights in (
                (
                    prepared.design[: prepared.n_events],
                    prepared.event_weights,
                ),
                (
                    prepared.design[prepared.n_events :],
                    prepared.grid_weights,
                ),
            ):
                # Numeric lexicographic order treats +0 and -0 as the same
                # design value; the parent inhibitory column is sign-restored
                # and may retain only that irrelevant sign bit.
                order = np.lexsort(
                    tuple(
                        design[:, column]
                        for column in reversed(range(design.shape[1]))
                    )
                )
                output.append((design[order], weights[order]))
            return output

        self.assertEqual(projected.constrained_start, fresh.constrained_start)
        self.assertEqual(projected.control_width, fresh.control_width)
        for (projected_x, projected_w), (fresh_x, fresh_w) in zip(
            ordered_parts(projected), ordered_parts(fresh), strict=True
        ):
            self.assertTrue(np.array_equal(projected_x, fresh_x))
            self.assertTrue(np.allclose(projected_w, fresh_w, rtol=2e-15, atol=2e-15))

    def test_parallel_support_evaluation_preserves_serial_results(self) -> None:
        data = synthetic_data(seed=929, n_sequences=120)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=2,
                max_formation_window=2,
                solver_workers=2,
                solver_device="cpu",
                solver_dtype="float64",
                certification_mode="predictive",
                target_history_control=False,
            ),
        )
        atoms = (
            RuleIdentity((0,), 0, 1),
            RuleIdentity((1,), 0, 1),
            RuleIdentity((0, 1), 2, -1),
        )
        supports = (
            (atoms[0],),
            (atoms[1],),
            (atoms[2],),
            (atoms[0], atoms[2]),
        )
        records = [
            pipeline._support_record(rules, profile="parallel-regression")
            for rules in supports
        ]
        serial = pipeline._evaluate_supports(
            records,
            pipeline.splits.fit,
            alpha=0.05,
            short_circuit_alpha=0.05,
            _parallel=False,
        )
        parallel = pipeline._evaluate_supports(
            records,
            pipeline.splits.fit,
            alpha=0.05,
            short_circuit_alpha=0.05,
            _parallel=True,
        )
        self.assertEqual(
            json.dumps(serial, sort_keys=True, allow_nan=True),
            json.dumps(parallel, sort_keys=True, allow_nan=True),
        )

    def test_closure_group_reuse_matches_cold_support_fits(self) -> None:
        data = synthetic_data(seed=1931, n_sequences=180)
        config = CertSCRConfig(
            q_max=2,
            impact_lag=4,
            knot_count=3,
            max_formation_window=2,
            solver_workers=1,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            target_history_control=False,
            safe_mdl_screen=False,
        )
        optimized = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        cold = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        a_rule = RuleIdentity((0,), 0, 1)
        b_rule = RuleIdentity((1,), 0, -1)
        supports = ((a_rule,), (b_rule,), (a_rule, b_rule))
        observed = optimized._fit_one_closure_group(
            supports, profile="closure-reuse-regression"
        )
        expected = [
            cold._support_record(rules, profile="cold-regression")
            for rules in supports
        ]
        self.assertEqual(
            [record.rules for record in observed],
            [record.rules for record in expected],
        )
        self.assertTrue(
            np.allclose(
                [record.fit.nll for record in observed],
                [record.fit.nll for record in expected],
                rtol=0.0,
                atol=1.0e-8,
            )
        )
        self.assertTrue(
            np.allclose(
                [record.search_nll_improvement for record in observed],
                [record.search_nll_improvement for record in expected],
                rtol=0.0,
                atol=1.0e-8,
            )
        )
        self.assertEqual(
            optimized._safe_screen_stats[
                "support_sparse_delta_closure_groups"
            ],
            1,
        )
        self.assertGreater(
            optimized._safe_screen_stats["support_sparse_delta_fits"],
            0,
        )

    def test_safe_mdl_screen_rejects_a_provably_zero_response_without_fitting(self) -> None:
        rows = []
        bounds = []
        for index in range(30):
            rows.append(
                {
                    "sequence_id": f"z{index}",
                    "position": 0,
                    "month_index": 0,
                    "target_token": int(index % 3 == 0),
                    "a": 0,
                }
            )
            bounds.append(
                {
                    "sequence_id": f"z{index}",
                    "start_month": 0,
                    "end_month": 2,
                }
            )
        data = EventData.from_frame(
            pd.DataFrame(rows),
            predicate_names=("a",),
            bounds=pd.DataFrame(bounds),
        )
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("a",),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=2,
                knot_count=2,
                max_support_size=1,
                solver_device="cpu",
                solver_dtype="float64",
                identity_profile="dictionary_mdl",
            ),
        )
        rule = RuleIdentity((0,), 0, 1)
        pipeline.identity_candidates = {(0,): (rule,)}
        pipeline.rule_dictionary_shapes[rule] = np.asarray([1.0, 0.0])
        screened = pipeline._safe_screened_support_record((rule,), profile="test")
        self.assertIsNotNone(screened)
        self.assertFalse(screened.fit.converged)  # type: ignore[union-attr]
        self.assertNotIn(((rule,), ()), pipeline._fit_cache)

    def test_safe_mdl_upper_bound_never_understates_exact_support_score(self) -> None:
        data = synthetic_data(n_sequences=100)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                max_support_size=1,
                solver_device="cpu",
                solver_dtype="float64",
                solver_max_iter=120,
                solver_tolerance=1.0e-7,
                identity_profile="dictionary_mdl",
            ),
        )
        rules = pipeline.profile_rule_identities()
        self.assertTrue(rules)
        for rule in rules:
            bound = pipeline._support_score_upper_bound((rule,))
            exact = pipeline._support_record((rule,), profile="safe-bound-test")
            exact_score = pipeline._support_search_score(exact)
            self.assertTrue(bound["finite"])
            self.assertLessEqual(
                exact_score,
                float(bound["score_upper_bound"]) + 1.0e-8,
            )

    def test_safe_mdl_screen_preserves_every_positive_active_search_support(self) -> None:
        data = synthetic_data(n_sequences=140)
        common = dict(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            max_support_size=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            support_search="active_set",
            active_restarts=4,
            support_conditioned_refinement=False,
            identity_profile="dictionary_mdl",
        )
        reference = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, safe_mdl_screen=False),
        )
        screened = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, safe_mdl_screen=True),
        )
        reference_records = reference.search_supports()
        screened_records = screened.search_supports()
        reference_positive = {
            record.rules: reference._support_search_score(record)
            for record in reference_records
            if reference._support_search_score(record) > 0.0
        }
        screened_positive = {
            record.rules: screened._support_search_score(record)
            for record in screened_records
            if screened._support_search_score(record) > 0.0
        }
        self.assertEqual(set(reference_positive), set(screened_positive))
        self.assertTrue(
            np.allclose(
                [reference_positive[key] for key in sorted(reference_positive)],
                [screened_positive[key] for key in sorted(screened_positive)],
                rtol=1.0e-8,
                atol=1.0e-8,
            )
        )

    def test_disabled_safe_screen_never_builds_identity_bounds(self) -> None:
        data = synthetic_data(seed=947, n_sequences=100)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                solver_device="cpu",
                solver_dtype="float64",
                identity_profile="dictionary_mdl",
                safe_mdl_screen=False,
            ),
        )

        def forbidden_bound(_rules):
            raise AssertionError("disabled safe screen evaluated a bound")

        pipeline._support_score_upper_bound = forbidden_bound  # type: ignore[method-assign]
        pipeline.profile_rule_identities()

    def test_rule_drop_removes_higher_order_hierarchy_descendants(self) -> None:
        a = RuleIdentity((0,), 0, 1)
        c = RuleIdentity((2,), 0, 1)
        ab = RuleIdentity((0, 1), 3, -1)
        abc = RuleIdentity((0, 1, 2), 3, 1)
        remaining, removed = CertSCRPipeline.hierarchy_preserving_drop(
            (a, c, ab, abc), a
        )
        self.assertEqual(remaining, (c,))
        self.assertEqual(removed, (a, ab, abc))

    def test_reported_a_and_ab_keep_only_b_as_nuisance(self) -> None:
        data = synthetic_data(n_sequences=8)
        pipeline = CertSCRPipeline(data, rule_predicates=("pred_a", "pred_b"))
        a = RuleIdentity((0,), 0, 1)
        ab = RuleIdentity((0, 1), 2, -1)
        self.assertEqual(pipeline.hierarchy_closure((a, ab)), (((1,), 0),))

    def test_ab_alone_adjusts_both_main_effects(self) -> None:
        data = synthetic_data(n_sequences=8)
        pipeline = CertSCRPipeline(data, rule_predicates=("pred_a", "pred_b"))
        ab = RuleIdentity((0, 1), 2, -1)
        self.assertEqual(pipeline.hierarchy_closure((ab,)), (((0,), 0), ((1,), 0)))

    def test_certification_family_is_exactly_the_family_frozen_on_fit(self) -> None:
        data = synthetic_data(n_sequences=10)
        pipeline = CertSCRPipeline(data, rule_predicates=("pred_a", "pred_b"))
        records = ["support-0", "support-1", "support-2"]
        pipeline.support_records = records  # type: ignore[assignment]
        test_case = self

        def item(p_value: float) -> dict:
            return {
                "fit_converged": True,
                "closure_baseline_converged": True,
                "all_rule_blocks_active": True,
                "structurally_testable": True,
                "p_value": p_value,
            }

        def fake_evaluate(
            self: object,
            selected: object,
            ctx: object,
            *,
            alpha: float,
            short_circuit_alpha: float | None = None,
        ) -> list[dict]:
            del self, alpha, short_circuit_alpha
            if ctx.name == "fit":
                test_case.assertEqual(selected, records)
                return [item(0.01), item(0.20), item(0.03)]
            test_case.assertEqual(selected, [records[0], records[2]])
            return [item(0.001), item(0.04)]

        pipeline._evaluate_supports = types.MethodType(fake_evaluate, pipeline)  # type: ignore[method-assign]
        fit_screen = pipeline.screen_supports_on_fit()
        certification = pipeline.certify_supports()
        self.assertEqual(fit_screen["selected_support_count"], 2)
        self.assertEqual(pipeline.candidate_records, [records[0], records[2]])
        self.assertEqual(certification["family_size"], 2)
        self.assertEqual(certification["certified_count"], 2)
        self.assertEqual(certification["financially_reliable_support_count"], 0)
        self.assertIn("F1", certification["all_supports"][0])
        self.assertIn("F2", certification["all_supports"][0])
        self.assertEqual(
            certification["claim"],
            "certified_event_early_warning_support_family",
        )
        self.assertFalse(
            certification["reliability_contract"][
                "adverse_financial_event_semantics_pre_specified"
            ]
        )
        self.assertEqual(
            certification["reliability_contract"]["family_error_control"],
            "Holm strong FWER conditional on the family frozen by D_fit",
        )
        self.assertFalse(
            certification["reliability_contract"][
                "calibration_equivalence_required_for_rule_certification"
            ]
        )
        self.assertFalse(certification["financially_grounded_loss"])
        self.assertFalse(certification["financially_certified"])
        self.assertEqual(certification["financially_reliable_supports"], [])
        ensemble = pipeline.fit_and_evaluate_ensemble()
        self.assertFalse(ensemble["fitted"])
        self.assertEqual(ensemble["reason"], "no_financially_reliable_support")
        pipeline.certification_loss = financial_weighted_nll_loss(
            np.ones(data.n_sequences), weight_name="test_business_cost"
        )
        pipeline.last_certification = None
        financial_certification = pipeline.certify_supports()
        self.assertTrue(financial_certification["financially_grounded_loss"])
        self.assertFalse(financial_certification["financial_contract_complete"])
        self.assertFalse(financial_certification["financially_certified"])

        complete_pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                financial_threshold=1.0e-6,
                rule_threshold=1.0e-6,
                calibration_tolerance=1.0,
            ),
            certification_loss=financial_weighted_nll_loss(
                np.ones(data.n_sequences), weight_name="test_business_cost"
            ),
        )
        complete_pipeline.support_records = records  # type: ignore[assignment]
        complete_pipeline._evaluate_supports = types.MethodType(  # type: ignore[method-assign]
            fake_evaluate, complete_pipeline
        )
        complete_certification = complete_pipeline.certify_supports()
        self.assertTrue(complete_certification["financial_contract_complete"])
        self.assertTrue(complete_certification["financially_certified"])

    def test_parallel_support_devices_preserve_the_exact_support_universe(self) -> None:
        data = synthetic_data(n_sequences=120)
        common = dict(
            q_max=1,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            max_support_size=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            support_search="exhaustive",
        )
        sequential = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common),
        )
        parallel = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, support_devices=("cpu", "cpu")),
        )
        sequential.profile_rule_identities()
        parallel.profile_rule_identities()
        sequential_records = sequential.search_supports()
        parallel_records = parallel.search_supports()
        self.assertEqual(
            [record.rules for record in sequential_records],
            [record.rules for record in parallel_records],
        )
        self.assertTrue(
            np.allclose(
                [record.fit.nll for record in sequential_records],
                [record.fit.nll for record in parallel_records],
                rtol=1.0e-8,
                atol=1.0e-8,
            )
        )

    def test_forked_cpu_profiling_preserves_selected_rule_library(self) -> None:
        data = synthetic_data(n_sequences=140)
        common = dict(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            identity_profile="dictionary_mdl",
        )
        sequential = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=1),
        )
        forked = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=2),
        )
        sequential_rules = sequential.profile_rule_identities()
        forked_rules = forked.profile_rule_identities()
        self.assertEqual(sequential_rules, forked_rules)
        sequential_logs = {
            tuple(row["antecedent"]): row for row in sequential.profile_logs
        }
        forked_logs = {
            tuple(row["antecedent"]): row for row in forked.profile_logs
        }
        self.assertEqual(set(sequential_logs), set(forked_logs))
        for antecedent in sequential_logs:
            left = sequential_logs[antecedent]
            right = forked_logs[antecedent]
            self.assertEqual(left["status"], right["status"])
            if left["status"].startswith("profiled"):
                self.assertEqual(left["selected_window"], right["selected_window"])
                self.assertEqual(left["selected_sign"], right["selected_sign"])
                self.assertAlmostEqual(
                    left["selected_block_mdl"],
                    right["selected_block_mdl"],
                    places=7,
                )
        for key in (
            "identity_zero_boundary_kkt_screens",
            "identity_sign_pair_parent_designs",
            "identity_sign_pair_child_reuses",
            "identity_fused_profile_exact_fits",
            "identity_incremental_partition_rebuilds",
        ):
            self.assertEqual(
                sequential._safe_screen_stats[key],
                forked._safe_screen_stats[key],
            )

    def test_skeleton_subset_reprofiles_all_identities_like_complete_family(self) -> None:
        data = synthetic_data(seed=1517, n_sequences=140)
        config = CertSCRConfig(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_workers=1,
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            identity_profile="dictionary_mdl",
            safe_mdl_screen=False,
        )
        complete = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        restricted = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        complete_rules = complete.profile_rule_identities()
        requested = ((0,), (0, 1))
        restricted_rules = restricted.profile_rule_identities(
            antecedent_subset=requested
        )
        self.assertEqual(
            restricted_rules,
            [rule for rule in complete_rules if rule.antecedent in requested],
        )
        complete_logs = {
            tuple(row["antecedent"]): row for row in complete.profile_logs
        }
        restricted_logs = {
            tuple(row["antecedent"]): row for row in restricted.profile_logs
        }
        self.assertEqual(set(restricted_logs), set(requested))
        for antecedent, right in restricted_logs.items():
            left = complete_logs[antecedent]
            self.assertEqual(left["status"], right["status"])
            self.assertEqual(
                [(item["window"], item["sign"]) for item in left["candidates"]],
                [(item["window"], item["sign"]) for item in right["candidates"]],
            )
            if left["status"].startswith("profiled"):
                self.assertEqual(left["selected_window"], right["selected_window"])
                self.assertEqual(left["selected_sign"], right["selected_sign"])
                self.assertAlmostEqual(
                    left["selected_block_mdl"],
                    right["selected_block_mdl"],
                    places=8,
                )
        with self.assertRaisesRegex(ValueError, "outside the finite family"):
            CertSCRPipeline(
                data,
                rule_predicates=("pred_a", "pred_b"),
                config=config,
            ).profile_rule_identities(antecedent_subset=((0, 1, 2),))

    def test_incremental_profile_failure_rebuilds_the_same_exact_identity(self) -> None:
        base = synthetic_data(seed=1611, n_sequences=180)
        data = replace(
            base,
            predicates=np.column_stack(
                (
                    base.predicates,
                    np.logical_or(
                        base.predicates[:, 0], base.predicates[:, 1]
                    ).astype(np.int8),
                )
            ),
            predicate_names=("pred_a", "pred_b", "pred_c"),
        )
        common = dict(
            q_max=3,
            impact_lag=4,
            knot_count=2,
            max_formation_window=3,
            solver_device="cpu",
            solver_dtype="float64",
            solver_workers=1,
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            identity_profile="dictionary_mdl",
            safe_mdl_screen=False,
        )
        ordinary = CertSCRPipeline(
            data,
            rule_predicates=data.predicate_names,
            config=CertSCRConfig(**common),
        )
        rebuilt = CertSCRPipeline(
            data,
            rule_predicates=data.predicate_names,
            config=CertSCRConfig(**common),
        )
        expected = ordinary.profile_rule_identities()
        original_update = pipeline_module.update_incremental_support_partition

        def fail_incremental(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("forced native incremental failure")

        pipeline_module.update_incremental_support_partition = fail_incremental  # type: ignore[assignment]
        try:
            observed = rebuilt.profile_rule_identities()
        finally:
            pipeline_module.update_incremental_support_partition = original_update
        self.assertEqual(observed, expected)
        self.assertGreater(
            rebuilt._safe_screen_stats["identity_incremental_partition_rebuilds"],
            0,
        )
        expected_logs = {
            tuple(row["antecedent"]): row for row in ordinary.profile_logs
        }
        observed_logs = {
            tuple(row["antecedent"]): row for row in rebuilt.profile_logs
        }
        for antecedent, left in expected_logs.items():
            right = observed_logs[antecedent]
            self.assertEqual(left["status"], right["status"])
            if left["status"].startswith("profiled"):
                self.assertEqual(left["selected_window"], right["selected_window"])
                self.assertEqual(left["selected_sign"], right["selected_sign"])
                self.assertAlmostEqual(
                    left["selected_block_mdl"],
                    right["selected_block_mdl"],
                    places=8,
                )

    def test_active_set_returns_exact_one_exchange_stationarity_certificate(self) -> None:
        data = synthetic_data(n_sequences=160)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                max_support_size=2,
                solver_device="cpu",
                solver_dtype="float64",
                solver_max_iter=120,
                solver_tolerance=1.0e-7,
                support_search="active_set",
                active_neighbor_strategy="exact_one_exchange",
                active_restarts=4,
                support_pool_size=8,
            ),
        )
        records = pipeline.search_supports()
        self.assertTrue(records)
        diagnostics = pipeline.search_diagnostics
        self.assertEqual(diagnostics["method"], "multi-start-exact-one-exchange-active-set")
        self.assertTrue(diagnostics["runs"])
        self.assertTrue(all(run["stationary_within_tolerance"] for run in diagnostics["runs"]))
        self.assertTrue(
            all(
                float(run["one_exchange_stationarity_gap"])
                <= pipeline.config.search_improvement_tolerance
                for run in diagnostics["runs"]
            )
        )
        refinement = diagnostics["identity_refinement"]
        self.assertTrue(refinement["enabled"])
        self.assertTrue(refinement["runs"])
        self.assertTrue(
            all(run["stationary_within_tolerance"] for run in refinement["runs"])
        )
        for record in records:
            for rule in record.rules:
                self.assertIn(rule, pipeline.identity_candidates[rule.antecedent])

    def test_gradient_multistart_returns_exact_one_exchange_stationarity(self) -> None:
        data = synthetic_data(n_sequences=180)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                max_support_size=2,
                solver_device="cpu",
                solver_dtype="float64",
                solver_max_iter=120,
                solver_tolerance=1.0e-7,
                support_search="active_set",
                active_neighbor_strategy="gradient_first_exact_audit",
                active_start_policy="all_atoms",
            ),
        )
        records = pipeline.search_supports()
        self.assertTrue(records)
        diagnostics = pipeline.search_diagnostics
        self.assertEqual(
            diagnostics["method"],
            "multi-start-gradient-first-exact-one-exchange-column-generation",
        )
        self.assertEqual(
            diagnostics["gradient_pricing"]["role"],
            "deterministic_ordering_only_no_candidate_removal",
        )
        self.assertTrue(diagnostics["runs"])
        self.assertTrue(
            all(run["stationary_within_tolerance"] for run in diagnostics["runs"])
        )
        self.assertTrue(
            all(
                float(run["one_exchange_stationarity_gap"])
                <= pipeline.config.search_improvement_tolerance
                for run in diagnostics["runs"]
            )
        )
        self.assertGreaterEqual(
            diagnostics["gradient_pricing"]["complete_terminal_audits"],
            diagnostics["terminal_support_count"],
        )
        self.assertEqual(
            diagnostics["atom_anchor_support_count"],
            diagnostics["start_atom_count"],
        )

    def test_mdl_score_working_set_has_no_candidate_budget(self) -> None:
        data = synthetic_data(n_sequences=180)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                max_support_size=2,
                solver_device="cpu",
                solver_dtype="float64",
                solver_max_iter=120,
                solver_tolerance=1.0e-7,
                support_search="active_set",
                active_neighbor_strategy="mdl_score_working_set",
                active_start_policy="all_atoms",
            ),
        )
        records = pipeline.search_supports()
        self.assertTrue(records)
        diagnostics = pipeline.search_diagnostics
        self.assertEqual(
            diagnostics["method"],
            "multi-start-mdl-block-score-working-set",
        )
        self.assertEqual(
            diagnostics["stationarity_claim"],
            "block_score_stationary_not_exact_one_exchange_stationary",
        )
        pricing = diagnostics["gradient_pricing"]
        self.assertEqual(
            pricing["role"],
            "MDL_calibrated_full_dictionary_working_set_no_top_k_or_budget",
        )
        self.assertEqual(pricing["complete_terminal_audits"], 0)
        self.assertGreater(pricing["block_score_terminal_certificates"], 0)
        self.assertTrue(diagnostics["runs"])
        self.assertTrue(
            all(run["stationary_within_tolerance"] for run in diagnostics["runs"])
        )
        self.assertTrue(
            all("block_score_stationarity_gap" in run for run in diagnostics["runs"])
        )
        scheduled = pricing["ordered_work_conserving_exact_refits"]
        self.assertTrue(scheduled["enabled"])
        self.assertIn("next_contiguous_prefix", scheduled["dispatch"])
        self.assertTrue(scheduled["preserves_candidate_order"])
        self.assertTrue(scheduled["preserves_accepted_moves_and_terminal"])
        self.assertFalse(scheduled["changes_objective_or_candidate_family"])

    def test_ordered_lazy_refits_preserve_mdl_working_set_across_worker_widths(self) -> None:
        data = synthetic_data(seed=1979, n_sequences=150)
        common = dict(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            max_support_size=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            support_search="active_set",
            active_neighbor_strategy="mdl_score_working_set",
            active_start_policy="all_atoms",
            support_conditioned_refinement=False,
        )
        serial = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=1),
        )
        parallel = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=4),
        )
        serial_records = serial.search_supports()
        parallel_records = parallel.search_supports()
        self.assertEqual(
            [record.rules for record in serial_records],
            [record.rules for record in parallel_records],
        )
        np.testing.assert_allclose(
            [serial._support_search_score(record) for record in serial_records],
            [parallel._support_search_score(record) for record in parallel_records],
            rtol=0.0,
            atol=1.0e-8,
        )
        serial_runs = {
            tuple(
                (
                    tuple(rule["antecedent_ids"]),
                    rule["window"],
                    rule["sign"],
                )
                for rule in run["start"]
            ): run
            for run in serial.search_diagnostics["runs"]
        }
        parallel_runs = {
            tuple(
                (
                    tuple(rule["antecedent_ids"]),
                    rule["window"],
                    rule["sign"],
                )
                for rule in run["start"]
            ): run
            for run in parallel.search_diagnostics["runs"]
        }
        self.assertEqual(serial_runs.keys(), parallel_runs.keys())
        for start in serial_runs:
            self.assertEqual(
                serial_runs[start]["terminal"], parallel_runs[start]["terminal"]
            )
            self.assertAlmostEqual(
                float(serial_runs[start]["terminal_score"]),
                float(parallel_runs[start]["terminal_score"]),
                places=8,
            )

    def test_conditional_safe_bound_preserves_working_set_output(self) -> None:
        data = synthetic_data(seed=1967, n_sequences=140)
        common = dict(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            solver_workers=2,
            target_history_control=False,
            safe_mdl_screen=False,
            support_search="active_set",
            active_start_policy="all_atoms",
            active_neighbor_strategy="mdl_score_working_set",
            support_conditioned_refinement=False,
        )
        ordinary = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                **common, conditional_safe_mdl_screen=False
            ),
        )
        bounded = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                **common, conditional_safe_mdl_screen=True
            ),
        )
        ordinary_records = ordinary.search_supports()
        bounded_records = bounded.search_supports()
        self.assertEqual(
            [record.rules for record in ordinary_records],
            [record.rules for record in bounded_records],
        )
        np.testing.assert_allclose(
            [ordinary._support_search_score(record) for record in ordinary_records],
            [bounded._support_search_score(record) for record in bounded_records],
            rtol=0.0,
            atol=1.0e-8,
        )

    def test_batched_multistate_rule_prices_match_scalar_prices(self) -> None:
        data = synthetic_data(n_sequences=240)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=3,
                knot_count=2,
                max_formation_window=2,
                solver_device="cpu",
                solver_dtype="float64",
                solver_max_iter=120,
                solver_tolerance=1.0e-7,
                support_search="active_set",
            ),
        )
        rules = [
            RuleIdentity((0,), 1, 1),
            RuleIdentity((1,), 1, -1),
            RuleIdentity((0, 1), 2, -1),
        ]
        fits = [pipeline.fit_baseline(), pipeline.fit_support((rules[0],))]
        candidate_sets = [tuple(rules), tuple(rules[1:])]
        scalar = [
            pipeline._support_rule_gradient_prices(fit, candidates)
            for fit, candidates in zip(fits, candidate_sets, strict=True)
        ]
        batched = pipeline._support_rule_gradient_prices_batch(
            list(zip(fits, candidate_sets, strict=True))
        )
        self.assertEqual(
            [set(item) for item in batched],
            [set(item) for item in scalar],
        )
        for expected, observed in zip(scalar, batched, strict=True):
            for rule in expected:
                np.testing.assert_allclose(
                    observed[rule], expected[rule], rtol=1.0e-10, atol=1.0e-10
                )
    def test_fused_support_gradient_prices_equal_direct_moments(self) -> None:
        data = synthetic_data(seed=1602, n_sequences=220)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=2,
                impact_lag=4,
                knot_count=3,
                max_formation_window=3,
                solver_device="cpu",
                solver_dtype="float64",
                solver_tolerance=1.0e-7,
                identity_profile="dictionary_mdl",
                safe_mdl_screen=False,
            ),
        )
        candidates = pipeline.profile_rule_identities()
        self.assertTrue(candidates)
        fit = pipeline.fit_baseline()
        fused = pipeline._support_rule_gradient_prices(fit, candidates)
        for rule in candidates:
            response = pipeline.sparse_features(pipeline.splits.fit, (rule,))[0]
            gradient, information = pipeline._identity_moments_at_null(
                fit,
                response,
            )
            signed = float(rule.sign) * gradient
            expected_gain = pipeline._cone_quadratic_gain(
                signed, information
            )
            expected_kkt = float(
                np.max(
                    np.maximum(-signed, 0.0)
                    / np.sqrt(
                        np.maximum(
                            np.diag(information),
                            np.finfo(np.float64).tiny,
                        )
                    ),
                    initial=0.0,
                )
            )
            self.assertAlmostEqual(fused[rule][0], expected_gain, places=12)
            self.assertAlmostEqual(fused[rule][1], expected_kkt, places=12)

    def test_parallel_active_set_preserves_refined_supports_and_objectives(self) -> None:
        data = synthetic_data(n_sequences=180)
        common = dict(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            max_support_size=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            support_search="active_set",
            active_restarts=4,
            support_pool_size=8,
        )
        sequential = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common),
        )
        parallel = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, support_devices=("cpu", "cpu")),
        )
        sequential_records = sequential.search_supports()
        parallel_records = parallel.search_supports()
        self.assertEqual(
            [record.rules for record in sequential_records],
            [record.rules for record in parallel_records],
        )
        self.assertTrue(
            np.allclose(
                [record.fit.nll for record in sequential_records],
                [record.fit.nll for record in parallel_records],
                rtol=1.0e-9,
                atol=1.0e-9,
            )
        )
        self.assertTrue(
            all(
                run["stationary_within_tolerance"]
                for run in parallel.search_diagnostics["identity_refinement"]["runs"]
            )
        )

    def test_shared_cpu_fit_caches_preserve_parallel_active_search(self) -> None:
        data = synthetic_data(n_sequences=140)
        common = dict(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=1,
            max_support_size=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            support_search="active_set",
        )
        sequential = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=1),
        )
        parallel = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=2),
        )
        sequential_records = sequential.search_supports()
        parallel_records = parallel.search_supports()
        self.assertEqual(
            [record.rules for record in sequential_records],
            [record.rules for record in parallel_records],
        )
        self.assertTrue(
            np.allclose(
                [record.fit.nll for record in sequential_records],
                [record.fit.nll for record in parallel_records],
                rtol=1.0e-9,
                atol=1.0e-9,
            )
        )
        self.assertGreaterEqual(
            parallel.search_diagnostics.get("memoized_neighborhood_hits", 0),
            0,
        )

    def test_shared_thread_support_scheduler_preserves_input_order(self) -> None:
        data = synthetic_data(seed=1973, n_sequences=60)
        pipeline = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(
                q_max=1,
                impact_lag=3,
                knot_count=2,
                max_formation_window=7,
                solver_workers=2,
                solver_device="cpu",
                solver_dtype="float64",
                solver_max_iter=80,
                solver_tolerance=1.0e-7,
                target_history_control=False,
                safe_mdl_screen=False,
            ),
        )
        supports = tuple(
            (RuleIdentity((source,), window, sign),)
            for source in (0, 1)
            for window in range(8)
            for sign in (-1, 1)
        )
        self.assertEqual(len(supports), 32)
        pipeline._start_active_support_workers()
        records = pipeline._fit_support_records_batch(
            supports, profile="rolling-fork-regression"
        )
        self.assertEqual(
            [record.rules for record in records],
            [tuple(sorted(rules)) for rules in supports],
        )
        self.assertEqual(
            pipeline._safe_screen_stats["active_fit_thread_batches"], 1
        )
        self.assertEqual(
            pipeline._safe_screen_stats["active_fit_process_batches"], 0
        )
        self.assertEqual(
            pipeline._safe_screen_stats["active_fit_dynamic_worker_launches"],
            0,
        )

    def test_shared_closure_children_are_sharded_without_changing_exact_fits(self) -> None:
        data = synthetic_data(seed=1981, n_sequences=180)
        common = dict(
            q_max=2,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            support_search="active_set",
            support_conditioned_refinement=False,
        )
        serial = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=1),
        )
        parallel = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=CertSCRConfig(**common, solver_workers=4),
        )
        rules = (
            RuleIdentity((0,), 1, -1),
            RuleIdentity((0,), 1, 1),
            RuleIdentity((1,), 1, -1),
            RuleIdentity((1,), 1, 1),
        )
        keys = [
            tuple(sorted((left, right)))
            for left in rules[:2]
            for right in rules[2:]
        ]
        self.assertEqual(
            len({serial.hierarchy_closure(key) for key in keys}), 1
        )
        serial_records = serial._fit_support_records_batch(
            keys, profile="shared-closure-shard-regression"
        )
        parallel._start_active_support_workers()
        parallel_records = parallel._fit_support_records_batch(
            keys, profile="shared-closure-shard-regression"
        )
        self.assertEqual(
            [record.rules for record in serial_records],
            [record.rules for record in parallel_records],
        )
        np.testing.assert_allclose(
            [record.fit.nll for record in serial_records],
            [record.fit.nll for record in parallel_records],
            rtol=0.0,
            atol=1.0e-8,
        )
        self.assertGreater(
            parallel._safe_screen_stats["active_fit_closure_shards"], 0
        )

    def test_profile_and_fit_checkpoint_round_trip_is_exact(self) -> None:
        data = synthetic_data(seed=2027, n_sequences=80)
        config = CertSCRConfig(
            q_max=1,
            impact_lag=3,
            knot_count=2,
            max_formation_window=2,
            solver_workers=1,
            solver_device="cpu",
            solver_dtype="float64",
            solver_max_iter=120,
            solver_tolerance=1.0e-7,
            target_history_control=False,
            safe_mdl_screen=False,
            identity_profile="dictionary_mdl",
        )
        source = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        rule = RuleIdentity((0,), 1, 1)
        alternative = RuleIdentity((0,), 2, -1)
        source.seed_profiled_library(
            (rule,),
            identity_candidates={(0,): (rule, alternative)},
            dictionary_shapes={
                rule: np.asarray([0.25, 0.75]),
                alternative: np.asarray([0.75, 0.25]),
            },
        )
        source_fit = source.fit_model((rule,), source.hierarchy_closure((rule,)))
        self.assertTrue(source_fit.converged)
        profile_state = source.export_profile_state()
        fit_state = source.export_fit_cache_state()
        # Execution-only parallelism and cache sizes may change across a
        # resumed run; the statistical model and data may not.
        resumed = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=replace(
                config,
                solver_workers=2,
                feature_cache_bytes=1024,
                loss_summary_cache_bytes=0,
                fit_summary_cache_bytes=0,
            ),
        )
        resumed.restore_profile_state(profile_state)
        restored_count = resumed.restore_fit_cache_state(fit_state)
        self.assertGreaterEqual(restored_count, 1)
        self.assertEqual(resumed.profiled_rules, [rule])
        np.testing.assert_array_equal(
            resumed.rule_dictionary_shapes[rule],
            np.asarray([0.25, 0.75]),
        )
        np.testing.assert_array_equal(
            resumed.rule_dictionary_shapes[alternative],
            np.asarray([0.75, 0.25]),
        )
        restored = resumed.fit_model((rule,), resumed.hierarchy_closure((rule,)))
        self.assertEqual(restored.nll, source_fit.nll)
        self.assertEqual(restored.alpha, source_fit.alpha)
        np.testing.assert_array_equal(restored.gamma, source_fit.gamma)
        np.testing.assert_array_equal(restored.theta, source_fit.theta)
        self.assertTrue(restored.device.startswith("resume:"))

    def test_checkpoint_signature_rejects_changed_statistical_inputs(self) -> None:
        data = synthetic_data(seed=2028, n_sequences=50)
        config = CertSCRConfig(
            q_max=1,
            impact_lag=3,
            knot_count=2,
            solver_device="cpu",
            solver_dtype="float64",
            target_history_control=False,
        )
        source = CertSCRPipeline(
            data,
            rule_predicates=("pred_a",),
            control_predicates=("pred_b",),
            config=config,
        )
        rule = RuleIdentity((0,), 0, 1)
        source.seed_profiled_library(
            (rule,),
            identity_candidates={(0,): (rule,)},
            dictionary_shapes={rule: np.asarray([1.0, 0.0])},
        )
        state = source.export_profile_state()
        changed = CertSCRPipeline(
            data,
            rule_predicates=("pred_a", "pred_b"),
            config=config,
        )
        with self.assertRaisesRegex(ValueError, "signature differs"):
            changed.restore_profile_state(state)


@unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA is unavailable")
class CudaPrimitiveTests(unittest.TestCase):
    def test_simplex_projection(self) -> None:
        import torch

        values = torch.tensor([0.8, -0.2, 0.9], dtype=torch.float64, device="cuda")
        projected = _simplex_project(values).cpu().numpy()
        self.assertAlmostEqual(float(np.sum(projected)), 1.0, places=12)
        self.assertTrue(np.all(projected >= 0))


if __name__ == "__main__":
    unittest.main()
