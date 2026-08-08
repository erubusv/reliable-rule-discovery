from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crbstpp.config import RunConfig
from crbstpp.certification import (
    CertifiedModel,
    compact_certified_models,
)
from crbstpp.data import Dataset, write_dataset
from crbstpp.native import cuda_available
from crbstpp.objective import ObjectiveSpec, SupportRecord, freeze_support_record
from crbstpp.report import Certificate
from crbstpp.response import Context, ModelMatrix, ResponseEngine
from crbstpp.rules import RuleIdentity, Support, hierarchy_closure
from crbstpp.search import SupportOptimizer
from crbstpp.solver import _general_recession, fit_model_matrix


F0 = {
    "dynamic_predicates": True,
    "outcome_blind_predicate_construction": True,
    "direct_target_proxy_excluded_from_reported_dictionary": True,
    "strict_future_effect_required": True,
    "atomic_predicates": True,
    "primitive_event_provenance": True,
    "independent_certification_units": True,
}


def cell_dataset(
    root: Path,
    predicates: int,
    probabilities: dict[tuple[int, ...], float],
    per_cell: int = 180,
    *,
    stratified: bool = False,
) -> Dataset:
    entities = []
    events = []
    targets = []
    code = 0
    for cell in sorted(probabilities):
        for replicate in range(per_cell):
            environment_start = 3 * (replicate % 10)
            entity = (
                f"e{code:06d}",
                environment_start,
                environment_start + 2,
                0,
                0,
            )
            entities.append((*entity, code % 2) if stratified else entity)
            for predicate, active in enumerate(cell):
                if active:
                    events.append((code, environment_start + 1, predicate))
            # Give every one of the ten temporal cohorts the same deterministic
            # cell probability; otherwise a synthetic recovery test would
            # confound its rule effect with the F3 environment by construction.
            cohort_size = per_cell // 10
            cohort_target = int(round(probabilities[cell] * cohort_size))
            within_cohort = replicate // 10
            if (within_cohort * 7_919) % cohort_size < cohort_target:
                targets.append((code, environment_start + 2, 1))
            code += 1
    write_dataset(
        root,
        entities=pd.DataFrame(
            entities,
            columns=[
                "entity_id",
                "start_time",
                "end_time",
                "baseline_origin",
                "split_group",
                *(("baseline_stratum",) if stratified else ()),
            ],
        ),
        events=pd.DataFrame(events, columns=["entity_code", "time", "predicate_code"]),
        targets=pd.DataFrame(targets, columns=["entity_code", "time", "multiplicity"]),
        predicate_names=tuple(chr(ord("A") + index) for index in range(predicates)),
        likelihood="first_event_cloglog",
        time_unit="month",
        adverse_event_name="synthetic adverse event",
        f0_contract=F0,
        provenance={"generator": "factorial-cell-test"},
    )
    return Dataset.load(root)


def optimizer(
    data: Dataset,
    q_max: int,
    *,
    search_mode: str = "exact_safe",
    adaptive_gradient_racing: bool = False,
    terminal_add_audit: str = "exact",
) -> SupportOptimizer:
    fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
    config = RunConfig(
        dataset=str(data.root),
        q_max=q_max,
        impact_lag=3,
        knot_count=1,
        formation_windows=(0,),
        solver_tolerance=1e-8,
        solver_max_iter=150,
        cache_bytes=32 * 1024**2,
        early_warning_horizon=3,
        pricing_devices=(),
        # These tests isolate structural higher-order recovery.  Reliability
        # screening has its own finite-cohort tests and can legitimately reject
        # these deliberately tiny discovery samples before D_cert exists.
        reliability_aware_search=False,
        search_mode=search_mode,
        adaptive_gradient_racing=adaptive_gradient_racing,
        terminal_add_audit=terminal_add_audit,
    )
    return SupportOptimizer(Context.make(data, fit_codes), config)


class HigherOrderRecoveryTests(unittest.TestCase):
    def test_fused_source_safe_shell_matches_completion_reference(self) -> None:
        probabilities = {
            cell: (0.55 if sum(cell) >= 2 else 0.12)
            for cell in __import__("itertools").product((0, 1), repeat=3)
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                3,
                probabilities,
                per_cell=35,
            )
            fused = optimizer(data, 3)
            reference = optimizer(data, 3)
            try:
                fused_empty = fused.records[Support(())]
                reference_empty = reference.records[Support(())]
                identities = tuple(fused.dictionary)
                fused_survivors = fused._safe_identity_survivors(
                    fused_empty, identities, 0.0
                )
                with patch(
                    "crbstpp.search.safe_shell_counts_sources",
                    return_value=None,
                ):
                    reference_survivors = reference._safe_identity_survivors(
                        reference_empty, identities, 0.0
                    )
                self.assertEqual(fused_survivors, reference_survivors)
                for rule in identities:
                    trial = Support.of((rule,))
                    fused_unsigned = fused._relaxed_upper_cache[
                        fused._unsigned_geometry_key(trial)
                    ]
                    reference_unsigned = reference._relaxed_upper_cache[
                        reference._unsigned_geometry_key(trial)
                    ]
                    self.assertAlmostEqual(
                        fused_unsigned, reference_unsigned, places=10
                    )
                    self.assertAlmostEqual(
                        fused._directional_upper_cache[trial],
                        reference._directional_upper_cache[trial],
                        places=10,
                    )
            finally:
                fused.close()
                reference.close()

    def test_state_quotient_bound_contains_signed_triplet_exact_optima(self) -> None:
        probabilities = {
            cell: (0.65 if sum(cell) % 2 == 1 else 0.15)
            for cell in __import__("itertools").product((0, 1), repeat=3)
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                3,
                probabilities,
                per_cell=60,
            )
            search = optimizer(data, 3)
            try:
                empty = search.records[Support(())]
                a = RuleIdentity((0,), 0, 1)
                current = search.fit(Support.of((a,)), empty)
                self.assertTrue(current.fit.converged, current.fit.message)
                for sign in (-1, 1):
                    abc = RuleIdentity((0, 1, 2), 0, sign)
                    trial = current.support.add(abc)
                    self.assertTrue(
                        search.engine.total_state_geometry_changed(
                            current.support,
                            trial,
                        )
                    )
                    exact = search.fit(trial, current)
                    self.assertTrue(exact.fit.converged, exact.fit.message)
                    upper = search._state_splice_group_upper_score(current, abc)
                    slack = 1.0e-7 * max(1.0, abs(exact.score))
                    self.assertGreaterEqual(upper + slack, exact.score)
            finally:
                search.close()

    @unittest.skipUnless(cuda_available(), "CUDA implicit pricing is unavailable")
    def test_implicit_triplet_tiles_long_entity_horizon_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                3,
                {(0, 0, 0): 0.05, (1, 1, 1): 0.5},
                per_cell=30,
            )
            # IBM has 425 hourly rows per entity.  A 7-block triplet with M=4
            # exceeds the default CUDA shared-memory limit if materialized as
            # one entity-wide response; the tiled kernel must remain exact.
            data = replace(
                data,
                end_times=np.full(data.n_entities, 424, dtype=np.int64),
                likelihood="poisson",
            )
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=3,
                impact_lag=48,
                knot_count=4,
                formation_windows=(0, 3, 12, 48),
                solver_tolerance=1e-8,
                solver_max_iter=100,
                cache_bytes=256 * 1024**2,
                early_warning_horizon=24,
                pricing_devices=("cuda:0",),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(context, config)
            try:
                current = optimizer.records[Support(())]
                first, second = optimizer._baseline_grid_derivatives(current)
                antecedent = (0, 1, 2)
                window = 12
                trial = current.support.add(RuleIdentity(antecedent, window, 1))
                closure = hierarchy_closure(trial)
                keys = tuple(
                    (term.antecedent, int(term.window)) for term in closure
                ) + ((antecedent, window),)
                actual = optimizer._implicit_hierarchy_moments(
                    current,
                    ((antecedent, window, closure, keys),),
                    first,
                    second,
                    device="cuda:0",
                )
                self.assertIsNotNone(actual)
                assert actual is not None
                blocks = tuple(
                    optimizer.engine.block(context, term.antecedent, term.window)
                    for term in closure
                ) + (optimizer.engine.block(context, antecedent, window),)
                reference = optimizer._joint_hierarchy_moments(
                    current, blocks, device="cpu"
                )
                for output, expected in zip(
                    actual,
                    (
                        reference[0][None, :],
                        reference[1][None, :, :],
                        reference[2][None, :, :],
                    ),
                    strict=True,
                ):
                    np.testing.assert_allclose(output, expected, rtol=2e-10, atol=2e-10)

                # Two different same-dimensional supports must not share the
                # compact Poisson derivative/group token on the GPU.
                for predicate in (0, 1):
                    current = optimizer.fit(
                        Support.of((RuleIdentity((predicate,), 0, 1),)),
                        device="cpu",
                    )
                    trial = current.support.add(RuleIdentity(antecedent, window, 1))
                    additions = tuple(
                        sorted(
                            set(hierarchy_closure(trial)) - set(current.matrix.closure)
                        )
                    )
                    keys = tuple(
                        (term.antecedent, int(term.window)) for term in additions
                    ) + ((antecedent, window),)
                    actual = optimizer._implicit_hierarchy_moments(
                        current,
                        ((antecedent, window, additions, keys),),
                        None,
                        None,
                        device="cuda:0",
                    )
                    self.assertIsNotNone(actual)
                    assert actual is not None
                    blocks = tuple(
                        optimizer.engine.block(context, term.antecedent, term.window)
                        for term in additions
                    ) + (optimizer.engine.block(context, antecedent, window),)
                    reference = optimizer._joint_hierarchy_moments(
                        current, blocks, device="cpu"
                    )
                    for output, expected in zip(
                        actual,
                        (
                            reference[0][None, :],
                            reference[1][None, :, :],
                            reference[2][None, :, :],
                        ),
                        strict=True,
                    ):
                        np.testing.assert_allclose(
                            output, expected, rtol=2e-10, atol=2e-10
                        )
            finally:
                optimizer.close()

    @unittest.skipUnless(cuda_available(), "CUDA implicit pricing is unavailable")
    def test_implicit_triplet_window_batch_matches_sparse_reference(self) -> None:
        probabilities = {
            tuple((mask >> index) & 1 for index in range(3)): (
                0.6 if mask.bit_count() >= 2 else 0.1
            )
            for mask in range(8)
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(Path(directory) / "data", 3, probabilities, per_cell=30)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=3,
                impact_lag=2,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=256 * 1024**2,
                early_warning_horizon=2,
                pricing_devices=("cuda:0",),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                current = optimizer.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)),
                    device="cuda:0",
                )
                optimizer._build_fast_derivative_grid(current)
                assert optimizer._fast_derivative_state is not None
                first, second = optimizer._fast_derivative_state[1:]
                specifications = []
                references = []
                antecedent = (0, 1, 2)
                for window in (0, 2):
                    trial = current.support.add(RuleIdentity(antecedent, window, 1))
                    additions = tuple(
                        sorted(
                            set(hierarchy_closure(trial)) - set(current.matrix.closure)
                        )
                    )
                    keys = tuple(
                        (term.antecedent, int(term.window)) for term in additions
                    ) + ((antecedent, window),)
                    specifications.append((antecedent, window, additions, keys))
                    blocks = tuple(
                        optimizer.engine.block(
                            optimizer.context,
                            term.antecedent,
                            term.window,
                        )
                        for term in additions
                    ) + (optimizer.engine.block(optimizer.context, antecedent, window),)
                    references.append(
                        optimizer._joint_hierarchy_moments(
                            current, blocks, device="cuda:0"
                        )
                    )
                actual = optimizer._implicit_hierarchy_moments(
                    current,
                    tuple(specifications),
                    first,
                    second,
                    device="cuda:0",
                )
                self.assertIsNotNone(actual)
                assert actual is not None
                for candidate, reference in enumerate(references):
                    for output, expected in zip(actual, reference, strict=True):
                        np.testing.assert_allclose(
                            output[candidate], expected, rtol=2e-11, atol=2e-11
                        )
            finally:
                optimizer.close()

    def test_compact_certification_keeps_one_dfit_mdl_nested_representative(
        self,
    ) -> None:
        singleton = Support.of((RuleIdentity((0,), 0, 1),))
        larger = Support.of((RuleIdentity((0,), 0, 1), RuleIdentity((1,), 0, 1)))

        def model(support: Support, score: float) -> CertifiedModel:
            record = SupportRecord(  # type: ignore[arg-type]
                support, None, None, 0.0, score
            )
            certificate = Certificate(
                support_key="test",
                f0=True,
                f1_pvalue=0.001,
                f2_pvalues=(0.001,) * len(support.rules),
                f3=True,
                family_pvalue=0.001,
                holm_adjusted_pvalue=0.002,
                certified=True,
            )
            return CertifiedModel(record, certificate, {"cert_mean_nll": 1.0})

        small = model(singleton, 2.0)
        worse_large = model(larger, 1.0)
        self.assertEqual(compact_certified_models((small, worse_large)), (small,))
        better_large = model(larger, 3.0)
        self.assertEqual(
            compact_certified_models((small, better_large)),
            (better_large,),
        )

    def test_triplet_incremental_window_pricing_matches_snapshot_reference(
        self,
    ) -> None:
        probabilities = {
            tuple((mask >> index) & 1 for index in range(3)): (
                0.7 if mask.bit_count() >= 2 else 0.1
            )
            for mask in range(8)
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                3,
                probabilities,
                per_cell=100,
            )
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=3,
                impact_lag=2,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=128 * 1024**2,
                early_warning_horizon=2,
                pricing_devices=(),
            )
            reference_optimizer = SupportOptimizer(
                Context.make(data, fit_codes), config
            )
            reference_empty = reference_optimizer.records[Support(())]
            reference = reference_optimizer._price_hierarchy_skeleton(
                reference_empty,
                (0, 1, 2),
                config.formation_windows,
                device="cpu",
                allow_incremental=False,
            )
            incremental_optimizer = SupportOptimizer(
                Context.make(data, fit_codes), config
            )
            incremental_empty = incremental_optimizer.records[Support(())]
            incremental = incremental_optimizer._price_hierarchy_incremental(
                incremental_empty,
                (0, 1, 2),
                config.formation_windows,
                device="cpu",
            )
            self.assertIsNotNone(incremental)
            assert incremental is not None
            for window in config.formation_windows:
                np.testing.assert_allclose(
                    incremental[window][:4],
                    reference[window][:4],
                    rtol=2e-10,
                    atol=2e-10,
                )
                self.assertEqual(incremental[window][4:], reference[window][4:])

    def test_combined_recession_ray_is_detected_when_axes_are_not(self) -> None:
        x = np.asarray([[1.0, -2.0], [-2.0, 1.0]])
        matrix = ModelMatrix(
            x=x,
            exposure_weight=np.ones(2),
            noevent_weight=np.ones(2),
            event_weight=np.zeros(2),
            free_dimension=2,
            closure_dimension=0,
            rule_slices=(),
            support=Support(()),
            closure=(),
            closure_signs=(),
            active_rows=np.asarray([0, 1]),
            active_design_groups=np.asarray([0, 1]),
            active_baseline_groups=np.zeros(2, dtype=np.int64),
            aggregate_baseline_groups=np.zeros(2, dtype=np.int64),
        )
        self.assertTrue(_general_recession(matrix, "poisson"))

    def test_solver_rejects_recession_and_rank_deficiency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recession_data = cell_dataset(
                root / "recession", 1, {(0,): 0.0, (1,): 1.0}, per_cell=80
            )
            context = Context.make(
                recession_data,
                np.arange(recession_data.n_entities, dtype=np.int32),
            )
            engine = ResponseEngine(
                recession_data, lag=3, knot_count=1, cache_bytes=8 * 1024**2
            )
            matrix = engine.model_matrix(
                context, Support.of((RuleIdentity((0,), 0, 1),))
            )
            recession = fit_model_matrix(
                matrix,
                likelihood=recession_data.likelihood,
                tolerance=1e-9,
                max_iter=100,
            )
            self.assertFalse(recession.converged)
            self.assertTrue(recession.recession)

            rank_data = cell_dataset(
                root / "rank", 2, {(0, 0): 0.1, (1, 1): 0.6}, per_cell=100
            )
            rank_context = Context.make(
                rank_data, np.arange(rank_data.n_entities, dtype=np.int32)
            )
            rank_engine = ResponseEngine(
                rank_data, lag=3, knot_count=1, cache_bytes=8 * 1024**2
            )
            rank_matrix = rank_engine.model_matrix(
                rank_context,
                Support.of((RuleIdentity((0,), 0, 1), RuleIdentity((1,), 0, 1))),
            )
            rank_fit = fit_model_matrix(
                rank_matrix,
                likelihood=rank_data.likelihood,
                tolerance=1e-9,
                max_iter=100,
            )
            self.assertFalse(rank_fit.converged)
            self.assertLess(rank_fit.rank, rank_matrix.dimension)

    def test_pair_only_interaction_is_recovered_without_heredity(self) -> None:
        probabilities = {
            (0, 0): 0.75,
            (0, 1): 0.05,
            (1, 0): 0.05,
            (1, 1): 0.75,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = optimizer(
                cell_dataset(Path(directory) / "data", 2, probabilities), 2
            ).search()
            self.assertTrue(
                any(
                    any(rule.antecedent == (0, 1) for rule in record.support.rules)
                    for record in result.family
                )
            )

    def test_triplet_only_interaction_is_recovered(self) -> None:
        probabilities = {
            cell: (0.75 if sum(cell) % 2 == 1 else 0.05)
            for cell in __import__("itertools").product((0, 1), repeat=3)
        }
        with tempfile.TemporaryDirectory() as directory:
            result = optimizer(
                cell_dataset(Path(directory) / "data", 3, probabilities), 3
            ).search()
            self.assertTrue(
                any(
                    any(rule.antecedent == (0, 1, 2) for rule in record.support.rules)
                    for record in result.family
                )
            )

    def test_A_excitation_plus_AB_inhibition_is_recovered(self) -> None:
        probabilities = {
            # The intercept covers the two pre-source months as well as the
            # outcome month, so use a background cell rate high enough that
            # the AB outcome state is below the fitted per-row baseline.
            (0, 0): 0.10,
            (0, 1): 0.10,
            (1, 0): 0.80,
            # Below the fitted per-row background intensity, so the complete AB
            # state is an inhibition rather than merely a negative
            # interaction correction.
            (1, 1): 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=1_000,
                ),
                2,
            ).search()
            # A applies only to A-without-B rows; AB directly owns the joint
            # state and is therefore a literal total-state inhibition.
            recovered = next(
                record
                for record in result.family
                if RuleIdentity((0, 1), 0, -1) in record.support.rules
            )
            self.assertTrue(
                any(
                    rule.antecedent == (0,) and rule.sign > 0
                    for rule in recovered.support.rules
                )
            )

    def test_safe_column_generation_recovers_signed_pair_without_fisher_pruning(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.10,
            (0, 1): 0.10,
            (1, 0): 0.80,
            (1, 1): 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=1_000,
                ),
                2,
                search_mode="safe_column_generation",
            )
            result = fitted.search()
            self.assertTrue(
                any(
                    RuleIdentity((0, 1), 0, -1) in record.support.rules
                    for record in result.family
                )
            )
            self.assertGreater(result.diagnostics.safe_column_candidates, 0)
            self.assertGreater(result.diagnostics.safe_column_exact_audits, 0)
            self.assertGreaterEqual(
                result.diagnostics.rashomon_atom_candidates,
                result.diagnostics.rashomon_basin_representatives,
            )

    def test_gap_safe_rashomon_path_recovers_pair_and_exactifies_terminals(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.10,
            (0, 1): 0.10,
            (1, 0): 0.80,
            (1, 1): 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=1_000,
                ),
                2,
                search_mode="gap_safe_rashomon_path",
            )
            result = fitted.search()
            self.assertTrue(
                any(
                    RuleIdentity((0, 1), 0, -1) in record.support.rules
                    for record in result.family
                )
            )
            self.assertTrue(all(record.fit.converged for record in result.terminals))
            self.assertGreater(result.diagnostics.gap_path_candidate_skeletons, 0)
            self.assertGreater(result.diagnostics.gap_path_root_exact_fits, 0)
            self.assertLessEqual(
                result.diagnostics.gap_path_representative_skeletons,
                result.diagnostics.gap_path_candidate_skeletons,
            )

    def test_gap_root_basins_do_not_merge_unrelated_same_order_rules(self) -> None:
        probabilities = {
            cell: (0.30 if sum(cell) % 2 else 0.10)
            for cell in __import__("itertools").product((0, 1), repeat=3)
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    3,
                    probabilities,
                    per_cell=60,
                ),
                3,
                search_mode="gap_safe_rashomon_path",
            )
            try:
                ab = RuleIdentity((0, 1), 0, 1)
                ac = RuleIdentity((0, 2), 0, 1)
                bc = RuleIdentity((1, 2), 0, 1)
                fitted._objective_root_candidates(
                    [(3.0, 0.0, ab), (2.0, 0.0, ac), (1.0, 0.0, bc)]
                )
                # None of these one-rule supports is reachable from another
                # by one Add or Drop.  A same-order exchange chain must not
                # collapse the three distinct mechanisms into one route root.
                self.assertEqual(
                    fitted._route_root_antecedents,
                    {(0, 1), (0, 2), (1, 2)},
                )
            finally:
                fitted.close()

    def test_safe_column_generation_never_uses_negative_fisher_as_a_screen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                1,
                {(0,): 0.10, (1,): 0.60},
                per_cell=100,
            )
            fitted = optimizer(
                data,
                1,
                search_mode="safe_column_generation",
            )
            current = fitted.records[Support(())]
            identities = tuple(
                rule for rule in fitted.dictionary if rule.antecedent == (0,)
            )
            ranked = [
                (-10.0 - index, 0.0, rule) for index, rule in enumerate(identities)
            ]
            with (
                patch.object(
                    fitted,
                    "_structural_upper_survivors",
                    return_value=identities,
                ),
                patch.object(
                    fitted,
                    "_rank_profiled_identities",
                    return_value=ranked,
                ),
                patch.object(
                    fitted,
                    "_safe_identity_survivors",
                    return_value=identities,
                ),
                patch.object(
                    fitted,
                    "_best_safe_column_addition",
                    return_value=None,
                ) as oracle,
            ):
                fitted._best_conditional_addition(
                    current,
                    antecedents={(0,)},
                )
            passed = oracle.call_args.args[1]
            self.assertEqual(
                {item[2] for item in passed},
                set(identities),
            )

    def test_adaptive_gradient_route_refines_feasible_pair_without_certifying_it(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.10,
            (0, 1): 0.10,
            (1, 0): 0.80,
            (1, 1): 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=160,
                ),
                2,
                search_mode="fast_block_score",
                adaptive_gradient_racing=True,
            )
            current = fitted.records[Support(())]
            rule = RuleIdentity((0, 1), 0, -1)
            proposal = fitted._conditional_one_step(
                current,
                current.support.add(rule),
                device="cpu",
            )
            routed = fitted._adaptive_gradient_route_record(
                current,
                proposal.record,
                structural_upper=proposal.upper_score,
                device="cpu",
            )
            self.assertFalse(routed.fit.converged)
            self.assertTrue(
                routed.fit.message.startswith("batched-feasible approximate route")
            )
            self.assertLessEqual(routed.fit.nll, proposal.record.fit.nll + 1.0e-10)
            self.assertGreaterEqual(routed.score, proposal.record.score - 1.0e-10)
            self.assertEqual(
                fitted.diagnostics.adaptive_gradient_candidates,
                1,
            )
            # A feasible route point must never be inserted into the exact
            # fixed-support cache or become reportable before terminal polish.
            stored = fitted._stored_records.get(routed.support)
            self.assertTrue(stored is None or not stored.fit.converged)

    def test_adaptive_gradient_search_exactifies_every_reported_terminal(self) -> None:
        probabilities = {
            (0, 0): 0.10,
            (0, 1): 0.10,
            (1, 0): 0.80,
            (1, 1): 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=1_000,
                ),
                2,
                search_mode="fast_block_score",
                adaptive_gradient_racing=True,
            )
            result = fitted.search()
            self.assertTrue(result.terminals)
            self.assertTrue(all(record.fit.converged for record in result.terminals))
            self.assertTrue(all(record.fit.converged for record in result.family))
            self.assertTrue(
                any(
                    any(rule.order == 2 for rule in record.support.rules)
                    for record in (*result.positive_atoms, *result.family)
                )
            )
            if result.diagnostics.adaptive_gradient_route_moves:
                self.assertGreater(
                    result.diagnostics.adaptive_gradient_terminal_exactifications
                    + result.diagnostics.composite_add_exactifications,
                    0,
                )
            self.assertGreater(
                result.diagnostics.block_score_evaluations,
                0,
            )

    def test_rashomon_basin_merges_routes_without_removing_positive_atoms(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.10,
            (0, 1): 0.10,
            (1, 0): 0.80,
            (1, 1): 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=1_000,
                ),
                2,
                search_mode="safe_column_generation",
            )
            result = fitted.search()
            atom_antecedents = {
                record.support.rules[0].antecedent for record in result.positive_atoms
            }
            self.assertEqual(atom_antecedents, {(0,), (1,), (0, 1)})
            self.assertEqual(result.diagnostics.rashomon_atom_candidates, 3)
            self.assertEqual(
                result.diagnostics.rashomon_basin_representatives,
                3,
            )
            # Empty plus every distinct singleton and the pair representative
            # is evaluated.  Singleton mechanisms are not collapsed into an
            # artificial all-to-all exchange basin; all exact-positive atoms
            # also remain available to family assembly.
            self.assertEqual(len(result.paths), 4)
            self.assertEqual(
                sum(path["start"] != "empty" for path in result.paths),
                3,
            )

    def test_successor_rashomon_keeps_exact_roots_until_supports_merge(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.10,
            (0, 1): 0.10,
            (1, 0): 0.80,
            (1, 1): 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=1_000,
                ),
                2,
                search_mode="successor_rashomon_path",
                adaptive_gradient_racing=True,
            )
            result = fitted.search()
            atom_antecedents = {
                record.support.rules[0].antecedent for record in result.positive_atoms
            }
            self.assertEqual(atom_antecedents, {(0,), (1,), (0, 1)})
            self.assertEqual(result.diagnostics.rashomon_atom_candidates, 3)
            self.assertEqual(
                result.diagnostics.rashomon_basin_representatives,
                3,
            )
            # Empty plus every exact-positive atom starts a route.  A route is
            # compressed only after an accepted move reaches a support already
            # owned by the shared DAG; no standalone exchange score removes it.
            self.assertEqual(len(result.paths), 4)
            self.assertEqual(
                sum(path["start"] != "empty" for path in result.paths),
                3,
            )
            self.assertTrue(all(record.fit.converged for record in result.terminals))
            self.assertTrue(
                any(
                    any(rule.order == 2 for rule in record.support.rules)
                    for record in (*result.positive_atoms, *result.family)
                )
            )

    def test_one_df_proposal_code_is_order_fair_but_final_code_is_full_knot(
        self,
    ) -> None:
        objective = ObjectiveSpec(
            n_entities=1_000,
            skeleton_count=7,
            knot_count=4,
            window_count_by_order=(1, 1, 1),
        )
        singleton = Support.of((RuleIdentity((0,), 0, 1),))
        pair = Support.of((RuleIdentity((0, 1), 0, 1),))
        self.assertAlmostEqual(
            objective.proposal_penalty(singleton),
            objective.proposal_penalty(pair),
        )
        self.assertAlmostEqual(
            objective.structural_penalty(singleton),
            objective.structural_penalty(pair),
        )
        self.assertGreater(
            objective.structural_penalty(pair),
            objective.proposal_penalty(pair),
        )

    def test_proposal_add_delta_matches_materialized_child(self) -> None:
        objective = ObjectiveSpec(
            n_entities=1_000,
            skeleton_count=20,
            knot_count=4,
            window_count_by_order=(1, 3, 5),
        )
        parent = Support.of((RuleIdentity((0,), 0, 1),))
        for rule in (
            RuleIdentity((0, 1), 3, -1),
            RuleIdentity((0, 1, 2), 5, 1),
        ):
            expected = objective.proposal_penalty(
                parent.add(rule)
            ) - objective.proposal_penalty(parent)
            actual = objective.proposal_add_penalty_delta(parent, rule.pattern_key)
            self.assertAlmostEqual(actual, expected, places=12)

    def test_joint_specific_pair_survives_global_lower_order_winner(self) -> None:
        probabilities = {
            (0, 0): 0.05,
            (0, 1): 0.05,
            (1, 0): 0.55,
            (1, 1): 0.60,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=80,
                ),
                2,
            )
            empty = fitted.records[Support(())]
            pair = RuleIdentity((0, 1), 0, 1)
            origin = fitted.fit(Support.of((pair,)), empty)
            self.assertGreater(origin.score, 0.0)
            fitted._standalone_profiled_atoms(empty)
            audited = fitted._local_representation_audit((origin,))
            self.assertEqual(
                {record.support.antecedents for record in audited},
                {((0,),), ((1,),), ((0, 1),)},
            )
            self.assertEqual(
                fitted.diagnostics.representation_specificity_retained,
                1,
            )

    def test_frozen_family_record_is_never_projected_as_empty_data(self) -> None:
        probabilities = {
            (0, 0): 0.05,
            (0, 1): 0.05,
            (1, 0): 0.80,
            (1, 1): 0.08,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(Path(directory) / "data", 2, probabilities), 2
            )
            support = Support.of(
                (
                    RuleIdentity((0,), 0, 1),
                    RuleIdentity((0, 1), 0, -1),
                )
            )
            source = fitted.fit(support)
            self.assertTrue(source.fit.converged)
            frozen = freeze_support_record(source)
            matrix, refit = fitted.fit_fixed(
                support,
                (),
                source=frozen,
            )
            self.assertGreater(matrix.x.shape[0], 0)
            self.assertTrue(refit.converged)
            self.assertGreater(refit.rank, 0)

    def test_pair_representation_audit_matches_local_exhaustive_mdl(self) -> None:
        probabilities = {
            (0, 0): 0.05,
            (0, 1): 0.05,
            (1, 0): 0.80,
            (1, 1): 0.80,
        }
        with tempfile.TemporaryDirectory() as directory:
            fitted = optimizer(
                cell_dataset(
                    Path(directory) / "data",
                    2,
                    probabilities,
                    per_cell=400,
                ),
                2,
            )
            empty = fitted.records[Support(())]
            pair = RuleIdentity((0, 1), 0, 1)
            child = fitted.fit(Support.of((pair,)), empty)
            # Pair-only is a legitimate total state.  The terminal
            # representation audit, rather than a hidden closure null, decides
            # whether A/B/AB should describe these cells.
            self.assertGreater(child.score, 0.0)
            validated, branch_net = fitted._exact_add_branch_validation(
                empty, freeze_support_record(child), pair
            )
            self.assertIsNotNone(validated)
            self.assertGreater(branch_net, 0.0)
            self.assertAlmostEqual(
                branch_net,
                child.score - empty.score,
                places=10,
            )
            assert validated is not None
            self.assertEqual(validated.matrix.x.shape[0], 0)
            fitted._standalone_profiled_atoms(empty)
            # Two route terminals may expose the same representation lattice.
            # The global audit must solve each unique support once, preserve
            # independently MDL-positive lower-order Rashomon alternatives,
            # and include the same best exact local representation.
            audited = fitted._local_representation_audit((child, child))
            self.assertGreater(
                fitted.diagnostics.representation_global_duplicate_fits_avoided,
                0,
            )
            identities = {
                antecedent: fitted._profiled_by_antecedent[antecedent]
                for antecedent in ((0,), (1,))
            }
            identities[(0, 1)] = pair
            exact = []
            ordered = ((0,), (1,), (0, 1))
            for mask in range(1, 8):
                support = Support.of(
                    identities[antecedent]
                    for index, antecedent in enumerate(ordered)
                    if mask & (1 << index)
                )
                record = fitted.fit(support)
                if record.fit.converged:
                    exact.append(record)
            expected = max(exact, key=lambda record: record.score)
            actual_best = max(audited, key=lambda record: record.score)
            self.assertAlmostEqual(actual_best.score, expected.score, places=8)
            self.assertEqual(actual_best.support, expected.support)
            self.assertEqual(
                {record.support for record in audited},
                {
                    Support.of((identities[(0,)],)),
                    Support.of((identities[(1,)],)),
                    expected.support,
                },
            )

    def test_shared_identity_replacements_match_materialized_exact_fits(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.05,
            (0, 1): 0.12,
            (1, 0): 0.72,
            (1, 1): 0.31,
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data", 2, probabilities, per_cell=120
            )
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                solver_tolerance=1e-9,
                solver_max_iter=180,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                reliability_aware_search=False,
                search_mode="fast_block_score",
                effect_model="support_additive",
            )
            a = RuleIdentity((0,), 0, 1, support_additive=True)
            b = RuleIdentity((1,), 0, 1, support_additive=True)
            current_support = Support.of((a, b))
            replacements = (
                RuleIdentity((0,), 0, -1, support_additive=True),
                RuleIdentity((0, 1), 0, 1, support_additive=True),
                RuleIdentity((0, 1), 0, -1, support_additive=True),
            )
            trials = tuple(current_support.replace(a, rule) for rule in replacements)
            specifications = {
                trial: (a, rule)
                for trial, rule in zip(trials, replacements, strict=True)
            }
            shared = SupportOptimizer(Context.make(data, fit_codes), config)
            reference = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                current = shared.fit(current_support)
                self.assertTrue(current.fit.converged, current.fit.message)
                projected = shared._fit_identity_replacements_shared(
                    current, trials, specifications
                )
                self.assertIsNotNone(projected)
                materialized = [reference.fit(trial) for trial in trials]
                for accelerated, exact in zip(
                    projected or (), materialized, strict=True
                ):
                    self.assertIsNotNone(accelerated)
                    assert accelerated is not None
                    self.assertEqual(accelerated.fit.converged, exact.fit.converged)
                    self.assertAlmostEqual(accelerated.fit.nll, exact.fit.nll, places=8)
                    self.assertAlmostEqual(accelerated.score, exact.score, places=8)
            finally:
                shared.close()
                reference.close()

    def test_shared_representation_lattice_matches_materialized_exact_fits(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.05,
            (0, 1): 0.12,
            (1, 0): 0.72,
            (1, 1): 0.31,
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                2,
                probabilities,
                per_cell=120,
            )
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                solver_tolerance=1e-9,
                solver_max_iter=180,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                reliability_aware_search=False,
                search_mode="fast_block_score",
            )
            a = RuleIdentity((0,), 0, 1)
            b = RuleIdentity((1,), 0, -1)
            ab = RuleIdentity((0, 1), 0, 1)
            supports = tuple(
                Support.of(rules)
                for rules in (
                    (a,),
                    (b,),
                    (ab,),
                    (a, b),
                    (a, ab),
                    (b, ab),
                    (a, b, ab),
                )
            )
            shared = SupportOptimizer(Context.make(data, fit_codes), config)
            reference = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                shared_records = shared._fit_representation_supports_shared(
                    supports, shared.records[Support(())]
                )
                materialized = [
                    reference.fit(support, reference.records[Support(())])
                    for support in supports
                ]
                for projected, exact in zip(shared_records, materialized, strict=True):
                    self.assertEqual(projected.fit.converged, exact.fit.converged)
                    self.assertAlmostEqual(projected.fit.nll, exact.fit.nll, places=8)
                    self.assertAlmostEqual(projected.score, exact.score, places=8)
                self.assertEqual(shared.diagnostics.representation_shared_fail_open, 0)
                # Representation fitting prefers the exact sparse-grid
                # backend and fails open to the shared projected lattice.
                # Both solve the same convex fixed-support objective.
                self.assertEqual(
                    shared.diagnostics.safe_column_sparse_exact_fits
                    + shared.diagnostics.representation_shared_projected_fits,
                    len(supports),
                )
            finally:
                shared.close()
                reference.close()

    def test_shared_sparse_representation_matches_stratified_dense_fits(
        self,
    ) -> None:
        probabilities = {
            (0, 0): 0.05,
            (0, 1): 0.14,
            (1, 0): 0.68,
            (1, 1): 0.37,
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                2,
                probabilities,
                per_cell=120,
                stratified=True,
            )
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                solver_tolerance=1e-9,
                solver_max_iter=180,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                reliability_aware_search=False,
                search_mode="fast_block_score",
            )
            a = RuleIdentity((0,), 0, 1)
            b = RuleIdentity((1,), 0, -1)
            ab = RuleIdentity((0, 1), 0, 1)
            supports = (
                Support.of((a,)),
                Support.of((b,)),
                Support.of((ab,)),
                Support.of((a, ab)),
            )
            shared = SupportOptimizer(Context.make(data, fit_codes), config)
            reference = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                sparse = shared._fit_sparse_supports_exact_many(
                    supports, shared.records[Support(())]
                )
                dense = [
                    reference.fit(support, reference.records[Support(())])
                    for support in supports
                ]
                for projected, exact in zip(sparse, dense, strict=True):
                    self.assertIsNotNone(projected)
                    assert projected is not None
                    self.assertEqual(projected.fit.converged, exact.fit.converged)
                    self.assertAlmostEqual(projected.fit.nll, exact.fit.nll, places=8)
                    self.assertAlmostEqual(projected.score, exact.score, places=8)
            finally:
                shared.close()
                reference.close()

    def test_total_state_dimension_counts_only_reported_rules(self) -> None:
        probabilities = {(0, 0, 0): 0.1, (1, 1, 1): 0.5}
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(Path(directory) / "data", 3, probabilities, per_cell=30)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            ab = Support.of((RuleIdentity((0, 1), 0, 1),))
            abc = Support.of((RuleIdentity((0, 1, 2), 0, 1),))
            a_ab = Support.of((RuleIdentity((0,), 0, 1), RuleIdentity((0, 1), 0, -1)))
            baseline = engine.model_matrix(context, Support(())).dimension
            self.assertEqual(
                engine.model_matrix(context, ab).dimension - baseline, 1 * 2
            )
            self.assertEqual(
                engine.model_matrix(context, abc).dimension - baseline, 1 * 2
            )
            self.assertEqual(
                engine.model_matrix(context, a_ab).dimension - baseline, 2 * 2
            )


if __name__ == "__main__":
    unittest.main()
