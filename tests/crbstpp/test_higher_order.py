from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crbstpp.config import RunConfig
from crbstpp.certification import certify_family
from crbstpp.data import Dataset, write_dataset
from crbstpp.ensemble import fit_ensemble
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import RuleIdentity, Support
from crbstpp.search import SupportOptimizer
from crbstpp.solver import fit_model_matrix


F0 = {
    "dynamic_predicates": True,
    "outcome_blind_predicate_construction": True,
    "direct_target_proxy_excluded": True,
    "strict_future_effect_required": True,
    "atomic_predicates": True,
}


def cell_dataset(
    root: Path,
    predicates: int,
    probabilities: dict[tuple[int, ...], float],
    per_cell: int = 180,
) -> Dataset:
    entities = []
    events = []
    targets = []
    code = 0
    for cell in sorted(probabilities):
        target_count = int(round(probabilities[cell] * per_cell))
        for replicate in range(per_cell):
            entities.append((f"e{code:06d}", 0, 2, 0, 0))
            for predicate, active in enumerate(cell):
                if active:
                    events.append((code, 1, predicate))
            if replicate < target_count:
                targets.append((code, 2, 1))
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


def optimizer(data: Dataset, q_max: int) -> SupportOptimizer:
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
    )
    return SupportOptimizer(Context.make(data, fit_codes), config)


class HigherOrderRecoveryTests(unittest.TestCase):
    def test_pair_support_passes_f0_f1_f2_on_held_out_entities(self) -> None:
        probabilities = {
            (0, 0): 0.005,
            (0, 1): 0.005,
            (1, 0): 0.005,
            (1, 1): 0.99,
        }
        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data", 2, probabilities, per_cell=1000
            )
            fit_codes, cert_codes, test_codes = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            fit_optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            record = fit_optimizer.fit(
                Support.of((RuleIdentity((0, 1), 0, 1),)),
                fit_optimizer.records[Support(())],
            )
            self.assertGreater(record.score, 0)
            certification = certify_family(
                fit_optimizer, Context.make(data, cert_codes), (record,), config
            )
            self.assertEqual(len(certification.certified), 1)
            certificate = certification.certified[0].certificate
            self.assertTrue(certificate.f0)
            self.assertLessEqual(certificate.holm_adjusted_pvalue, config.alpha)
            combined = np.sort(np.concatenate([fit_codes, cert_codes])).astype(np.int32)
            ensemble = fit_ensemble(
                Context.make(data, combined),
                Context.make(data, test_codes),
                (record.support,),
                config,
            )
            np.testing.assert_allclose(ensemble.weights, [1.0])
            self.assertTrue(np.isfinite(ensemble.test_nll))

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
            (0, 0): 0.05,
            (0, 1): 0.05,
            (1, 0): 0.80,
            (1, 1): 0.08,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = optimizer(
                cell_dataset(Path(directory) / "data", 2, probabilities), 2
            ).search()
            # The closure-equivalent pair model is deliberately preferred by
            # MDL over redundantly reporting A inside the same support.  Both
            # directional identities must nevertheless enter the frozen family
            # and can therefore coexist in the certified ensemble.
            self.assertTrue(
                any(
                    RuleIdentity((0,), 0, 1) in record.support.rules
                    for record in result.family
                )
            )
            self.assertTrue(
                any(
                    RuleIdentity((0, 1), 0, -1) in record.support.rules
                    for record in result.family
                )
            )

    def test_closure_dimension_is_counted_once(self) -> None:
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
                engine.model_matrix(context, ab).dimension - baseline, 3 * 2
            )
            self.assertEqual(
                engine.model_matrix(context, abc).dimension - baseline, 7 * 2
            )
            self.assertEqual(
                engine.model_matrix(context, a_ab).dimension - baseline, 3 * 2
            )


if __name__ == "__main__":
    unittest.main()
