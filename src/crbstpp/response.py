from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .rules import Antecedent, ClosureTerm, RuleIdentity, Support, hierarchy_closure


def triangular_basis(lag: int, count: int) -> np.ndarray:
    if lag < 1 or count < 1:
        raise ValueError("lag and count must be positive")
    x = np.arange(1, lag + 1, dtype=np.float64)
    if count == 1:
        basis = np.ones((1, lag), dtype=np.float64)
    else:
        centers = np.linspace(1.0, float(lag), count)
        width = max(1.0, float(centers[1] - centers[0]))
        basis = np.maximum(0.0, 1.0 - np.abs(x[None, :] - centers[:, None]) / width)
    area = basis.sum(axis=1, keepdims=True)
    if np.any(area <= 0):
        raise ValueError("degenerate temporal basis")
    return basis / area


@dataclass(frozen=True)
class Context:
    dataset: Dataset
    entity_codes: np.ndarray
    entity_lookup: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    baseline_origins: np.ndarray
    offsets: np.ndarray
    n_grid: int
    target_rows: np.ndarray
    target_counts: np.ndarray

    @classmethod
    def make(cls, dataset: Dataset, entity_codes: np.ndarray) -> "Context":
        entity_codes = np.sort(np.unique(np.asarray(entity_codes, dtype=np.int32)))
        lookup = np.full(dataset.n_entities, -1, dtype=np.int32)
        lookup[entity_codes] = np.arange(len(entity_codes), dtype=np.int32)
        starts = dataset.start_times[entity_codes]
        ends = dataset.end_times[entity_codes]
        origins = dataset.baseline_origins[entity_codes]
        lengths = ends - starts + 1
        offsets = np.zeros(len(entity_codes) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)
        local = lookup[dataset.target_entities]
        keep = local >= 0
        local = local[keep]
        rows = offsets[local] + dataset.target_times[keep] - starts[local]
        counts = dataset.target_multiplicity[keep]
        if len(rows):
            order = np.argsort(rows, kind="stable")
            rows, counts = rows[order], counts[order]
            unique, first = np.unique(rows, return_index=True)
            counts = np.add.reduceat(counts, first)
            rows = unique
        return cls(
            dataset=dataset,
            entity_codes=entity_codes,
            entity_lookup=lookup,
            starts=starts,
            ends=ends,
            baseline_origins=origins,
            offsets=offsets,
            n_grid=int(offsets[-1]),
            target_rows=np.asarray(rows, dtype=np.int64),
            target_counts=np.asarray(counts, dtype=np.float64),
        )

    def rows_to_entity_time(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        local = np.searchsorted(self.offsets, rows, side="right") - 1
        times = self.starts[local] + rows - self.offsets[local]
        return local.astype(np.int32), times.astype(np.int64)


@dataclass(frozen=True)
class SparseBlock:
    rows: np.ndarray
    values: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.rows.nbytes + self.values.nbytes)


@dataclass(frozen=True)
class ModelMatrix:
    x: np.ndarray
    exposure_weight: np.ndarray
    noevent_weight: np.ndarray
    event_weight: np.ndarray
    free_dimension: int
    closure_dimension: int
    rule_slices: tuple[slice, ...]
    support: Support
    closure: tuple[ClosureTerm, ...]

    @property
    def dimension(self) -> int:
        return int(self.x.shape[1])


class ResponseEngine:
    def __init__(self, dataset: Dataset, *, lag: int, knot_count: int, cache_bytes: int):
        self.dataset = dataset
        self.lag = int(lag)
        self.knot_count = int(knot_count)
        self.basis = triangular_basis(lag, knot_count)
        maximum_age = int(
            np.max(dataset.baseline_origins + dataset.end_times - dataset.start_times)
        )
        self.baseline_dimension = max(1, maximum_age // self.lag + 1)
        self.cache_bytes = max(0, int(cache_bytes))
        self._cache: OrderedDict[tuple, SparseBlock] = OrderedDict()
        self._cache_size = 0
        self._lock = threading.RLock()

    def _source(self, predicate: int, context: Context) -> tuple[np.ndarray, np.ndarray]:
        entities, times = self.dataset.predicate_stream(predicate)
        local = context.entity_lookup[entities]
        keep = local >= 0
        return local[keep].astype(np.int32), times[keep].astype(np.int64)

    def completions(self, context: Context, antecedent: Antecedent) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sources = [self._source(predicate, context) for predicate in antecedent]
        per_source: list[dict[int, np.ndarray]] = []
        for entities, times in sources:
            mapping: dict[int, np.ndarray] = {}
            if len(entities):
                unique, first = np.unique(entities, return_index=True)
                for index, entity in enumerate(unique):
                    right = first[index + 1] if index + 1 < len(first) else len(entities)
                    mapping[int(entity)] = np.unique(times[first[index]:right])
            per_source.append(mapping)
        eligible = set(per_source[0]) if per_source else set()
        for mapping in per_source[1:]:
            eligible.intersection_update(mapping)
        completion_entities: list[int] = []
        completion_times: list[int] = []
        completion_spans: list[int] = []
        for entity in sorted(eligible):
            streams = [mapping[entity] for mapping in per_source]
            if len(streams) == 1:
                for time in streams[0]:
                    completion_entities.append(entity)
                    completion_times.append(int(time))
                    completion_spans.append(0)
                continue
            union = np.unique(np.concatenate(streams))
            latest = np.full(len(streams), np.iinfo(np.int64).min, dtype=np.int64)
            positions = np.zeros(len(streams), dtype=np.int64)
            for time in union:
                for source_index, stream in enumerate(streams):
                    while positions[source_index] < len(stream) and stream[positions[source_index]] <= time:
                        latest[source_index] = stream[positions[source_index]]
                        positions[source_index] += 1
                if np.all(latest != np.iinfo(np.int64).min):
                    completion_entities.append(entity)
                    completion_times.append(int(time))
                    completion_spans.append(int(latest.max() - latest.min()))
        return (
            np.asarray(completion_entities, dtype=np.int32),
            np.asarray(completion_times, dtype=np.int64),
            np.asarray(completion_spans, dtype=np.int64),
        )

    def block(self, context: Context, antecedent: Antecedent, window: int) -> SparseBlock:
        key = (context.entity_codes.tobytes(), antecedent, int(window))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        entities, times, spans = self.completions(context, antecedent)
        keep = spans <= int(window)
        entities, times = entities[keep], times[keep]
        values: dict[int, np.ndarray] = {}
        for entity, completion in zip(entities.tolist(), times.tolist(), strict=True):
            base = int(context.offsets[entity] + completion - context.starts[entity])
            remaining = int(context.ends[entity] - completion)
            for lag in range(1, min(self.lag, remaining) + 1):
                row = base + lag
                if row not in values:
                    values[row] = np.zeros(self.knot_count, dtype=np.float64)
                values[row] += self.basis[:, lag - 1]
        rows = np.asarray(sorted(values), dtype=np.int64)
        matrix = (
            np.stack([values[int(row)] for row in rows])
            if len(rows)
            else np.zeros((0, self.knot_count), dtype=np.float64)
        )
        result = SparseBlock(rows, matrix)
        with self._lock:
            self._cache[key] = result
            self._cache_size += result.nbytes
            while self._cache_size > self.cache_bytes and len(self._cache) > 1:
                _, removed = self._cache.popitem(last=False)
                self._cache_size -= removed.nbytes
        return result

    def model_matrix(
        self,
        context: Context,
        support: Support,
        *,
        forced_closure: tuple[ClosureTerm, ...] | None = None,
    ) -> ModelMatrix:
        closure = hierarchy_closure(support) if forced_closure is None else tuple(sorted(forced_closure))
        closure_blocks = [self.block(context, term.antecedent, term.window) for term in closure]
        rule_blocks = [self.block(context, rule.antecedent, rule.window) for rule in support.rules]
        all_blocks = [*closure_blocks, *rule_blocks]
        active_rows = [block.rows for block in all_blocks if len(block.rows)]
        if len(context.target_rows):
            active_rows.append(context.target_rows)
        union = np.unique(np.concatenate(active_rows)) if active_rows else np.zeros(0, dtype=np.int64)
        local, times = context.rows_to_entity_time(union)
        ages = context.baseline_origins[local] + times - context.starts[local]
        age_count = self.baseline_dimension
        baseline_dim = age_count
        dimension = baseline_dim + self.knot_count * len(all_blocks)
        x_active = np.zeros((len(union), dimension), dtype=np.float64)
        x_active[:, 0] = 1.0
        age_bins = np.minimum(ages // self.lag, age_count - 1).astype(np.int64)
        for row_index, age_bin in enumerate(age_bins):
            if age_bin > 0:
                x_active[row_index, int(age_bin)] = 1.0
        for block_index, block in enumerate(all_blocks):
            if not len(block.rows):
                continue
            positions = np.searchsorted(union, block.rows)
            sign = 1.0
            if block_index >= len(closure_blocks):
                sign = float(support.rules[block_index - len(closure_blocks)].sign)
            left = baseline_dim + block_index * self.knot_count
            x_active[positions, left:left + self.knot_count] = sign * block.values
        # Aggregate all rows outside the active union exactly by baseline age bin.
        total_by_age = np.zeros(age_count, dtype=np.float64)
        for origin, start, end in zip(
            context.baseline_origins.tolist(), context.starts.tolist(), context.ends.tolist(), strict=True
        ):
            ages_entity = origin + np.arange(end - start + 1, dtype=np.int64)
            total_by_age += np.bincount(
                np.minimum(ages_entity // self.lag, age_count - 1), minlength=age_count
            )
        active_by_age = np.bincount(age_bins, minlength=age_count).astype(np.float64)
        inactive_by_age = total_by_age - active_by_age
        aggregate_bins = np.flatnonzero(inactive_by_age > 0)
        x_aggregate = np.zeros((len(aggregate_bins), dimension), dtype=np.float64)
        x_aggregate[:, 0] = 1.0
        for row_index, age_bin in enumerate(aggregate_bins):
            if age_bin > 0:
                x_aggregate[row_index, int(age_bin)] = 1.0
        x = np.concatenate([x_active, x_aggregate], axis=0)
        target_position = np.searchsorted(union, context.target_rows)
        matched = (target_position < len(union)) & (union[np.minimum(target_position, max(0, len(union) - 1))] == context.target_rows) if len(union) else np.zeros(len(context.target_rows), dtype=bool)
        event_active = np.zeros(len(union), dtype=np.float64)
        if np.any(matched):
            np.add.at(event_active, target_position[matched], context.target_counts[matched])
        # target rows were explicitly included, so the unmatched branch is an invariant check.
        if len(context.target_rows) and not np.all(matched):
            raise AssertionError("target row missing from active union")
        event_weight = np.concatenate([event_active, np.zeros(len(aggregate_bins))])
        exposure_weight = np.concatenate([np.ones(len(union)), inactive_by_age[aggregate_bins]])
        if context.dataset.likelihood == "first_event_cloglog":
            noevent_weight = exposure_weight - event_weight
        else:
            noevent_weight = exposure_weight.copy()
        free_dimension = baseline_dim + len(closure_blocks) * self.knot_count
        rule_slices = tuple(
            slice(
                free_dimension + index * self.knot_count,
                free_dimension + (index + 1) * self.knot_count,
            )
            for index in range(len(rule_blocks))
        )
        return ModelMatrix(
            x=x,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event_weight,
            free_dimension=free_dimension,
            closure_dimension=len(closure_blocks) * self.knot_count,
            rule_slices=rule_slices,
            support=support,
            closure=closure,
        )

    def linear_predictor(
        self,
        context: Context,
        matrix: ModelMatrix,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (matrix.dimension,):
            raise ValueError("coefficient dimension mismatch")
        rows = np.arange(context.n_grid, dtype=np.int64)
        local, times = context.rows_to_entity_time(rows)
        ages = context.baseline_origins[local] + times - context.starts[local]
        baseline_dimension = matrix.free_dimension - matrix.closure_dimension
        age_bins = np.minimum(ages // self.lag, baseline_dimension - 1).astype(np.int64)
        eta = np.full(context.n_grid, coefficients[0], dtype=np.float64)
        nonzero_age = age_bins > 0
        eta[nonzero_age] += coefficients[age_bins[nonzero_age]]
        left = baseline_dimension
        for index, term in enumerate(matrix.closure):
            block = self.block(context, term.antecedent, term.window)
            beta = coefficients[left + index * self.knot_count : left + (index + 1) * self.knot_count]
            if len(block.rows):
                eta[block.rows] += block.values @ beta
        left = matrix.free_dimension
        for index, rule in enumerate(matrix.support.rules):
            block = self.block(context, rule.antecedent, rule.window)
            beta = coefficients[left + index * self.knot_count : left + (index + 1) * self.knot_count]
            if len(block.rows):
                eta[block.rows] += rule.sign * (block.values @ beta)
        return eta

    def footprint_rows(
        self, context: Context, rule: RuleIdentity, horizon: int
    ) -> np.ndarray:
        entities, times, spans = self.completions(context, rule.antecedent)
        keep = spans <= rule.window
        entities, times = entities[keep], times[keep]
        rows: list[np.ndarray] = []
        for entity, completion in zip(entities.tolist(), times.tolist(), strict=True):
            length = min(int(horizon), int(context.ends[entity] - completion))
            if length > 0:
                base = int(context.offsets[entity] + completion - context.starts[entity])
                rows.append(base + np.arange(1, length + 1, dtype=np.int64))
        return np.unique(np.concatenate(rows)) if rows else np.zeros(0, dtype=np.int64)
