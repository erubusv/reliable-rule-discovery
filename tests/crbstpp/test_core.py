from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crbstpp.config import RunConfig
from crbstpp.data import Dataset, write_dataset
from crbstpp.dual import dual_certificate
from crbstpp.likelihood import cloglog_event_terms
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import (
    RuleIdentity,
    Support,
    hierarchy_closure,
)
from crbstpp.search import SupportOptimizer, support_key
from crbstpp.solver import fit_model_matrix, fit_model_matrix_continued


def synthetic_dataset(
    root: Path, n_entities: int = 90, *, explicit_partition: bool = False
) -> Dataset:
    entities = pd.DataFrame(
        {
            "entity_id": [f"e{index:04d}" for index in range(n_entities)],
            "start_time": np.zeros(n_entities, dtype=np.int64),
            "end_time": np.full(n_entities, 8, dtype=np.int64),
            "baseline_origin": np.zeros(n_entities, dtype=np.int64),
            "split_group": np.zeros(n_entities, dtype=np.int64),
        }
    )
    if explicit_partition:
        entities["partition"] = np.arange(n_entities, dtype=np.int64) * 3 // n_entities
    events = []
    targets = []
    for entity in range(n_entities):
        if entity % 3 != 0:
            events.append((entity, 1, 0))
        if entity % 5 != 0:
            events.append((entity, 2, 1))
        # Strong A excitation, with some negative controls retained.
        if entity % 3 != 0 and entity % 7 != 0:
            targets.append((entity, 3, 1))
        elif entity % 17 == 0:
            targets.append((entity, 6, 1))
    event_frame = pd.DataFrame(
        events, columns=["entity_code", "time", "predicate_code"]
    )
    target_frame = pd.DataFrame(
        targets, columns=["entity_code", "time", "multiplicity"]
    )
    write_dataset(
        root,
        entities=entities,
        events=event_frame,
        targets=target_frame,
        predicate_names=("A", "B"),
        likelihood="first_event_cloglog",
        time_unit="month",
        adverse_event_name="synthetic adverse event",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
        },
        provenance={"generator": "test"},
    )
    return Dataset.load(root)


class DataRuleTests(unittest.TestCase):
    def test_roundtrip_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            self.assertEqual(data.n_entities, 90)
            split = data.split((0.6, 0.2, 0.2), 111)
            self.assertEqual(sum(map(len, split)), 90)
            self.assertEqual(len(set(np.concatenate(split).tolist())), 90)

    def test_explicit_partition_overrides_fraction_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(
                Path(directory) / "data", explicit_partition=True
            )
            first = data.split((0.98, 0.01, 0.01), 1)
            second = data.split((0.1, 0.1, 0.8), 999)
            for code, (left, right) in enumerate(zip(first, second, strict=True)):
                np.testing.assert_array_equal(left, right)
                self.assertTrue(np.all(data.partitions[left] == code))

    def test_hierarchy_closure(self) -> None:
        a = RuleIdentity((0,), 0, 1)
        ab = RuleIdentity((0, 1), 2, -1)
        abc = RuleIdentity((0, 1, 2), 3, 1)
        self.assertEqual(
            {
                (term.antecedent, term.window)
                for term in hierarchy_closure(Support.of((ab,)))
            },
            {((0,), 0), ((1,), 0)},
        )
        self.assertEqual(
            {
                (term.antecedent, term.window)
                for term in hierarchy_closure(Support.of((a, ab)))
            },
            {((1,), 0)},
        )
        closure = hierarchy_closure(Support.of((abc,)))
        self.assertEqual(len(closure), 6)
        self.assertIn(((0, 1), 3), {(term.antecedent, term.window) for term in closure})

    def test_completion_is_nonnegative_and_strictly_future(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 12)
            context = Context.make(data, np.arange(12))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=1024**2)
            block = engine.block(context, (0, 1), 2)
            self.assertTrue(np.all(block.values >= 0))
            local, times = context.rows_to_entity_time(block.rows)
            self.assertTrue(np.all(times >= 3))
            # A completion at B's t=2 cannot affect t=2 itself.
            self.assertFalse(np.any(times == 2))


class LikelihoodSolverTests(unittest.TestCase):
    def test_iteration_window_continuation_reaches_same_exact_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            matrix = engine.model_matrix(
                context, Support.of((RuleIdentity((0,), 0, 1),))
            )
            reference = fit_model_matrix(
                matrix, likelihood=data.likelihood, tolerance=1e-8, max_iter=150
            )
            continued = fit_model_matrix_continued(
                matrix, likelihood=data.likelihood, tolerance=1e-8, max_iter=1
            )
            self.assertTrue(reference.converged, reference.message)
            self.assertTrue(continued.converged, continued.message)
            self.assertGreater(continued.iterations, 1)
            self.assertAlmostEqual(continued.nll, reference.nll, places=10)

    def test_cloglog_derivatives(self) -> None:
        eta = np.linspace(-5, 3, 21)
        value, first, second = cloglog_event_terms(eta)
        epsilon = 1.0e-5
        plus = cloglog_event_terms(eta + epsilon)[0]
        minus = cloglog_event_terms(eta - epsilon)[0]
        numeric_first = (plus - minus) / (2 * epsilon)
        numeric_second = (plus - 2 * value + minus) / epsilon**2
        np.testing.assert_allclose(first, numeric_first, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(second, numeric_second, rtol=2e-3, atol=2e-5)

    def test_dual_bound_contains_exact_optimum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            matrix = engine.model_matrix(context, support)
            fit = fit_model_matrix(
                matrix, likelihood=data.likelihood, tolerance=1e-8, max_iter=150
            )
            self.assertTrue(fit.converged, fit.message)
            certificate = dual_certificate(
                matrix,
                likelihood=data.likelihood,
                beta=fit.coefficients,
                tolerance=1e-8,
                max_iter=1000,
            )
            self.assertTrue(certificate.feasible)
            self.assertLessEqual(certificate.nll_lower_bound, fit.nll + 1e-6)

    def test_small_block_search_has_exact_conditional_score_terminals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            result = optimizer.search()
            self.assertGreater(len(result.family), 0)
            self.assertGreater(len(result.terminals), 0)
            self.assertIsNotNone(optimizer._profiled_dictionary)
            self.assertEqual(
                len({rule.antecedent for rule in optimizer._profiled_dictionary}),
                len(optimizer._profiled_dictionary),
            )
            self.assertLessEqual(
                len(optimizer._profiled_dictionary), len(optimizer.skeletons)
            )
            frozen = set(optimizer._profiled_dictionary)
            self.assertTrue(
                all(
                    rule in frozen
                    for record in result.family
                    for rule in record.support.rules
                )
            )
            expected_starts = {
                "empty",
                *(support_key(record.support) for record in result.positive_atoms),
            }
            self.assertEqual(
                {str(path["start"]) for path in result.paths}, expected_starts
            )
            self.assertEqual(len(result.paths), len(result.positive_atoms) + 1)
            self.assertEqual(
                result.diagnostics.multi_source_roots,
                len(result.positive_atoms) + 1,
            )
            self.assertEqual(
                result.diagnostics.standalone_branch_audits,
                len(result.positive_atoms)
                + result.diagnostics.standalone_branch_rejections,
            )
            for record in result.family:
                self.assertEqual(
                    len(record.support.antecedents),
                    len(set(record.support.antecedents)),
                )
            for terminal in result.terminals:
                self.assertTrue(terminal.fit.converged, terminal.fit.message)
                self.assertIsNone(optimizer._best_profiled_move(terminal))
                inactive = tuple(
                    rule
                    for rule in (optimizer._profiled_dictionary or ())
                    if rule.antecedent not in terminal.support.antecedents
                )
                score_admissible = tuple(
                    item[2]
                    for item in optimizer._rank_profiled_identities(
                        terminal, inactive
                    )
                    if item[0] > config.search_tolerance
                )
                trials = [terminal.support.add(rule) for rule in score_admissible]
                trials.extend(
                    terminal.support.drop(rule) for rule in terminal.support.rules
                )
                for trial in trials:
                    proposal = optimizer._conditional_one_step(terminal, trial)
                    self.assertLessEqual(
                        proposal.lower_score,
                        terminal.score + config.search_tolerance,
                    )
                    exact = optimizer.fit(trial, terminal)
                    if exact.fit.converged:
                        slack = 1e-6 * max(1.0, abs(exact.score))
                        self.assertLessEqual(
                            proposal.lower_score, exact.score + slack
                        )
                        self.assertGreaterEqual(
                            proposal.upper_score + slack, exact.score
                        )

    def test_parallel_search_matches_serial_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            common = dict(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            families = []
            for workers in (1, 3):
                result = SupportOptimizer(
                    Context.make(data, fit_codes),
                    RunConfig(**common, exact_workers=workers),
                ).search()
                families.append(
                    [(record.support, record.score) for record in result.family]
                )
            self.assertEqual(
                [item[0] for item in families[0]],
                [item[0] for item in families[1]],
            )
            np.testing.assert_allclose(
                [item[1] for item in families[0]],
                [item[1] for item in families[1]],
                rtol=0,
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
