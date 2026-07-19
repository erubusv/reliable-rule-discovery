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
from crbstpp.search import SupportOptimizer
from crbstpp.solver import fit_model_matrix


def synthetic_dataset(root: Path, n_entities: int = 90) -> Dataset:
    entities = pd.DataFrame(
        {
            "entity_id": [f"e{index:04d}" for index in range(n_entities)],
            "start_time": np.zeros(n_entities, dtype=np.int64),
            "end_time": np.full(n_entities, 8, dtype=np.int64),
            "baseline_origin": np.zeros(n_entities, dtype=np.int64),
            "split_group": np.zeros(n_entities, dtype=np.int64),
        }
    )
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

    def test_small_search_returns_stationary_family(self) -> None:
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
            for terminal in result.terminals:
                for neighbor in __import__(
                    "crbstpp.rules", fromlist=["one_exchange_neighbors"]
                ).one_exchange_neighbors(terminal.support, optimizer.dictionary):
                    record = optimizer.fit(neighbor, terminal)
                    self.assertLessEqual(
                        record.score, terminal.score + config.search_tolerance + 1e-7
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
