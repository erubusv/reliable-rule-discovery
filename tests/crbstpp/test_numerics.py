from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from crbstpp.likelihood import conjugate_sum, loss_rows
from crbstpp.native import completion_events, cpu_available, cuda_available, moments
from crbstpp.response import Context, ResponseEngine
from crbstpp.config import RunConfig
from crbstpp.data import Dataset, write_dataset
from crbstpp.rules import Support
from crbstpp.search import SupportOptimizer
from crbstpp.solver import fit_model_matrix

from tests.crbstpp.test_core import synthetic_dataset


class NumericalParityTests(unittest.TestCase):
    def test_poisson_value_gradient_hessian_and_conjugate(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(31, 5))
        beta = rng.normal(size=5)
        exposure = rng.uniform(0.1, 2.0, size=31)
        event = rng.poisson(0.7, size=31).astype(float)
        eta = x @ beta
        value, first, second = loss_rows(
            eta,
            likelihood="poisson",
            exposure_weight=exposure,
            noevent_weight=exposure,
            event_weight=event,
        )
        epsilon = 2.0e-6
        direction = rng.normal(size=5)

        def objective(coefficient: np.ndarray) -> float:
            return float(
                np.sum(
                    loss_rows(
                        x @ coefficient,
                        likelihood="poisson",
                        exposure_weight=exposure,
                        noevent_weight=exposure,
                        event_weight=event,
                    )[0]
                )
            )

        numeric_first = (
            objective(beta + epsilon * direction)
            - objective(beta - epsilon * direction)
        ) / (2 * epsilon)
        analytic_first = float(direction @ (x.T @ first))
        self.assertAlmostEqual(numeric_first, analytic_first, places=5)
        numeric_second = (
            objective(beta + epsilon * direction)
            - 2 * float(np.sum(value))
            + objective(beta - epsilon * direction)
        ) / epsilon**2
        analytic_second = float(direction @ (x.T @ (second[:, None] * x)) @ direction)
        self.assertAlmostEqual(numeric_second, analytic_second, delta=2.0e-2)
        dual = first
        conjugate = conjugate_sum(
            dual,
            likelihood="poisson",
            exposure_weight=exposure,
            noevent_weight=exposure,
            event_weight=event,
        )
        self.assertAlmostEqual(
            float(np.sum(value)), float(dual @ eta - conjugate), places=9
        )

    def test_compiled_cpu_and_cuda_moments_match_reference(self) -> None:
        self.assertTrue(cpu_available())
        rng = np.random.default_rng(10)
        x = rng.normal(size=(257, 11))
        first = rng.normal(size=257)
        second = rng.uniform(0.01, 3.0, size=257)
        reference_gradient = x.T @ first
        reference_hessian = x.T @ (second[:, None] * x)
        cpu_gradient, cpu_hessian = moments(x, first, second, device="cpu")
        np.testing.assert_allclose(
            cpu_gradient, reference_gradient, rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            cpu_hessian, reference_hessian, rtol=1e-13, atol=1e-13
        )
        if cuda_available():
            for device in ("cuda:0", "cuda:1"):
                gradient, hessian = moments(x, first, second, device=device)
                np.testing.assert_allclose(
                    gradient, reference_gradient, rtol=1e-13, atol=1e-13
                )
                np.testing.assert_allclose(
                    hessian, reference_hessian, rtol=1e-13, atol=1e-13
                )

    def test_compiled_completion_matches_python_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 30)
            context = Context.make(data, np.arange(30, dtype=np.int32))
            compiled = ResponseEngine(data, lag=3, knot_count=3, cache_bytes=1024**2)
            compiled_block = compiled.block(context, (0, 1), 2)
            with (
                mock.patch("crbstpp.response.completion_events", return_value=None),
                mock.patch("crbstpp.response.kernel_contributions", return_value=None),
            ):
                reference = ResponseEngine(
                    data, lag=3, knot_count=3, cache_bytes=1024**2
                )
                reference_block = reference.block(context, (0, 1), 2)
            np.testing.assert_array_equal(compiled_block.rows, reference_block.rows)
            np.testing.assert_allclose(
                compiled_block.values, reference_block.values, rtol=0, atol=0
            )

    def test_latest_witness_completion_updates_all_three_sources(self) -> None:
        sources = [
            (np.asarray([0, 0]), np.asarray([1, 5])),
            (np.asarray([0, 0]), np.asarray([2, 4])),
            (np.asarray([0, 0]), np.asarray([3, 5])),
        ]
        result = completion_events(sources)
        self.assertIsNotNone(result)
        entities, times, spans = result
        np.testing.assert_array_equal(entities, [0, 0, 0])
        np.testing.assert_array_equal(times, [3, 4, 5])
        np.testing.assert_array_equal(spans, [2, 3, 1])

    def test_lru_does_not_retain_oversized_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 30)
            context = Context.make(data, np.arange(30, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=3, cache_bytes=1)
            block = engine.block(context, (0,), 0)
            self.assertGreater(block.nbytes, 1)
            self.assertLessEqual(engine._cache_size, engine.cache_bytes)

    def test_every_primal_dual_sandwich_contains_exact_support_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            for rule in optimizer.dictionary:
                support = Support.of((rule,))
                bounds = optimizer.bounds(empty, support)
                exact = optimizer.fit(support, empty)
                if exact.fit.converged:
                    self.assertLessEqual(bounds.lower_score, exact.score + 1e-7)
                    self.assertGreaterEqual(bounds.upper_score + 1e-7, exact.score)

    def test_continuous_poisson_uses_exact_tick_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            write_dataset(
                root,
                entities=pd.DataFrame(
                    {
                        "entity_id": ["a", "b", "c"],
                        "start_time": [0, 0, 0],
                        "end_time": [59, 59, 59],
                        "baseline_origin": [0, 0, 0],
                        "split_group": [0, 0, 0],
                    }
                ),
                events=pd.DataFrame(
                    [(0, 1, 0), (1, 1, 0), (2, 1, 0)],
                    columns=["entity_code", "time", "predicate_code"],
                ),
                targets=pd.DataFrame(
                    [(0, 20, 1), (1, 30, 1), (2, 40, 1)],
                    columns=["entity_code", "time", "multiplicity"],
                ),
                predicate_names=("A",),
                likelihood="poisson",
                time_unit="hour",
                ticks_per_unit=60,
                adverse_event_name="synthetic recurrent event",
                f0_contract={
                    "dynamic_predicates": True,
                    "outcome_blind_predicate_construction": True,
                    "direct_target_proxy_excluded": True,
                    "strict_future_effect_required": True,
                    "atomic_predicates": True,
                },
                provenance={"generator": "continuous-poisson-test"},
            )
            data = Dataset.load(root)
            context = Context.make(data, np.arange(3, dtype=np.int32))
            engine = ResponseEngine(data, lag=1, knot_count=2, cache_bytes=1024**2)
            matrix = engine.model_matrix(context, Support(()))
            fit = fit_model_matrix(
                matrix, likelihood="poisson", tolerance=1e-10, max_iter=100
            )
            self.assertTrue(fit.converged, fit.message)
            total_exposure_hours = 3 * 60 / 60
            self.assertAlmostEqual(
                np.exp(fit.coefficients[0]), 3 / total_exposure_hours, places=8
            )
            self.assertEqual(engine.lag, 60)


if __name__ == "__main__":
    unittest.main()
