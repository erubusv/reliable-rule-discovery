from __future__ import annotations

import heapq
import math
import threading
from itertools import count, product
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .data import Dataset
from .native import (
    aggregate_design_rows,
    aggregate_design_rows_with_groups,
    accumulate_kernel,
    bounded_span_order,
    completion_events,
    completion_window_counts,
    future_rows,
    kernel_contributions,
    response_min_spans,
    sorted_unique_union,
    subtract_group_weights,
)
from .rules import (
    Antecedent,
    ClosureTerm,
    PatternKey,
    RuleIdentity,
    Support,
    hierarchy_closure,
    normalize_pattern,
    normalize_relation,
)
from .state import (
    StateIntervals,
    active_at,
    history_state_intervals,
    transition_state_intervals,
)


_MODEL_MATRIX_TOKENS = count(1)
_TEMPORAL_BASELINE_LAYOUTS: dict[tuple[str, int], np.ndarray] = {}
_TEMPORAL_BASELINE_LAYOUT_LOCK = threading.Lock()


def _dataset_temporal_baseline_edges(
    dataset: Dataset, time_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    bins = int(time_bins)
    calendar_min = int(np.min(dataset.start_times))
    calendar_max = int(np.max(dataset.end_times))
    maximum_age = int(
        np.max(
            np.maximum(0, dataset.end_times - dataset.baseline_origins),
            initial=0,
        )
    )

    def edges(left: int, right_inclusive: int) -> np.ndarray:
        width = max(1, right_inclusive - left + 1)
        return left + (np.arange(bins + 1, dtype=np.int64) * width) // bins

    return edges(calendar_min, calendar_max), edges(0, maximum_age)


def _temporal_baseline_layout(dataset: Dataset, time_bins: int) -> np.ndarray:
    """Map possible raw time cells to a fixed compact coefficient layout."""
    bins = int(time_bins)
    strata_count = dataset.n_baseline_strata
    raw_dimension = strata_count * bins * bins
    if bins == 1:
        return np.arange(raw_dimension, dtype=np.int32)
    key = (dataset.digest, bins)
    with _TEMPORAL_BASELINE_LAYOUT_LOCK:
        cached = _TEMPORAL_BASELINE_LAYOUTS.get(key)
        if cached is not None:
            return cached
    calendar_edges, age_edges = _dataset_temporal_baseline_edges(dataset, bins)
    active = np.zeros(raw_dimension, dtype=bool)
    if dataset.baseline_cell_strata is not None:
        if dataset.likelihood == "continuous_poisson":
            if (
                dataset.baseline_cell_entities is None
                or dataset.baseline_cell_times is None
            ):
                raise ValueError("continuous risk-interval provenance is absent")
            total = len(dataset.baseline_cell_times)
        else:
            lengths = dataset.end_times - dataset.start_times + 1
            offsets = np.zeros(dataset.n_entities + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(lengths, dtype=np.int64)
            total = int(offsets[-1])
        chunk = 1_000_000
        for left in range(0, total, chunk):
            right = min(total, left + chunk)
            if dataset.likelihood == "continuous_poisson":
                entity = dataset.baseline_cell_entities[left:right]
                times = dataset.baseline_cell_times[left:right]
            else:
                rows = np.arange(left, right, dtype=np.int64)
                entity = np.searchsorted(offsets, rows, side="right") - 1
                times = dataset.start_times[entity] + rows - offsets[entity]
            calendar_bin = np.searchsorted(calendar_edges[1:-1], times, side="right")
            age = np.maximum(0, times - dataset.baseline_origins[entity])
            age_bin = np.searchsorted(age_edges[1:-1], age, side="right")
            raw = dataset.baseline_cell_strata[left:right].astype(np.int64) + (
                strata_count * (age_bin + bins * calendar_bin)
            )
            active[np.unique(raw)] = True
    else:
        strata = (
            np.zeros(dataset.n_entities, dtype=np.int16)
            if dataset.baseline_strata is None
            else dataset.baseline_strata
        )
        for calendar_bin in range(bins):
            cal_left = calendar_edges[calendar_bin]
            cal_right = calendar_edges[calendar_bin + 1] - 1
            for age_bin in range(bins):
                age_left = dataset.baseline_origins + age_edges[age_bin]
                age_right = dataset.baseline_origins + age_edges[age_bin + 1] - 1
                left = np.maximum(dataset.start_times, np.maximum(cal_left, age_left))
                right = np.minimum(dataset.end_times, np.minimum(cal_right, age_right))
                present = np.unique(strata[right >= left])
                raw = present.astype(np.int64) + strata_count * (
                    age_bin + bins * calendar_bin
                )
                active[raw] = True
    mapping = np.full(raw_dimension, -1, dtype=np.int32)
    mapping[active] = np.arange(np.count_nonzero(active), dtype=np.int32)
    mapping.setflags(write=False)
    with _TEMPORAL_BASELINE_LAYOUT_LOCK:
        incumbent = _TEMPORAL_BASELINE_LAYOUTS.setdefault(key, mapping)
    return incumbent


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
    baseline_strata: np.ndarray
    offsets: np.ndarray
    row_times: np.ndarray | None
    baseline_row_strata: np.ndarray | None
    baseline_row_exposure: np.ndarray | None
    n_grid: int
    target_rows: np.ndarray
    target_counts: np.ndarray
    entity_weights: np.ndarray
    uniform_entity_weight: float | None
    population_entities: int

    @classmethod
    def make(
        cls,
        dataset: Dataset,
        entity_codes: np.ndarray,
        *,
        entity_weights: np.ndarray | None = None,
        population_entities: int | None = None,
    ) -> "Context":
        supplied_codes = np.asarray(entity_codes, dtype=np.int32)
        if entity_weights is None:
            entity_codes = np.sort(np.unique(supplied_codes))
            weights = np.ones(len(entity_codes), dtype=np.float64)
        else:
            supplied_weights = np.asarray(entity_weights, dtype=np.float64)
            if supplied_weights.shape != supplied_codes.shape:
                raise ValueError("entity weights must align with entity codes")
            order = np.argsort(supplied_codes, kind="stable")
            entity_codes = supplied_codes[order]
            weights = supplied_weights[order]
            if len(entity_codes) and np.any(entity_codes[1:] == entity_codes[:-1]):
                raise ValueError("weighted context entity codes must be unique")
            if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("entity weights must be finite and positive")
        population = (
            len(entity_codes)
            if population_entities is None
            else int(population_entities)
        )
        if population < len(entity_codes):
            raise ValueError("population size cannot be smaller than the sample")
        lookup = np.full(dataset.n_entities, -1, dtype=np.int32)
        lookup[entity_codes] = np.arange(len(entity_codes), dtype=np.int32)
        starts = dataset.start_times[entity_codes]
        ends = dataset.end_times[entity_codes]
        origins = dataset.baseline_origins[entity_codes]
        strata = (
            np.zeros(len(entity_codes), dtype=np.int16)
            if dataset.baseline_strata is None
            else dataset.baseline_strata[entity_codes]
        )
        irregular = dataset.likelihood == "continuous_poisson"
        if irregular:
            if (
                dataset.baseline_cell_entities is None
                or dataset.baseline_cell_times is None
                or dataset.baseline_cell_strata is None
                or dataset.baseline_cell_exposure is None
            ):
                raise ValueError("continuous risk intervals are incomplete")
            global_lengths = np.bincount(
                dataset.baseline_cell_entities, minlength=dataset.n_entities
            ).astype(np.int64)
            global_offsets = np.zeros(dataset.n_entities + 1, dtype=np.int64)
            global_offsets[1:] = np.cumsum(global_lengths, dtype=np.int64)
            lengths = global_lengths[entity_codes]
        else:
            lengths = ends - starts + 1
        offsets = np.zeros(len(entity_codes) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)
        local = lookup[dataset.target_entities]
        row_strata: np.ndarray | None = None
        row_exposure: np.ndarray | None = None
        row_times: np.ndarray | None = None
        if dataset.baseline_cell_strata is not None:
            if not irregular:
                global_lengths = dataset.end_times - dataset.start_times + 1
                global_offsets = np.zeros(dataset.n_entities + 1, dtype=np.int64)
                global_offsets[1:] = np.cumsum(global_lengths, dtype=np.int64)
            row_strata = np.empty(int(offsets[-1]), dtype=np.int16)
            if dataset.baseline_cell_exposure is not None:
                row_exposure = np.empty(int(offsets[-1]), dtype=np.float64)
            if irregular:
                row_times = np.empty(int(offsets[-1]), dtype=np.int64)
            chunk = 1_000_000
            for left in range(0, len(row_strata), chunk):
                right = min(len(row_strata), left + chunk)
                rows_chunk = np.arange(left, right, dtype=np.int64)
                local_chunk = np.searchsorted(offsets, rows_chunk, side="right") - 1
                global_rows = (
                    global_offsets[entity_codes[local_chunk]]
                    + rows_chunk
                    - offsets[local_chunk]
                )
                row_strata[left:right] = dataset.baseline_cell_strata[global_rows]
                if row_exposure is not None:
                    row_exposure[left:right] = dataset.baseline_cell_exposure[
                        global_rows
                    ]
                if row_times is not None:
                    assert dataset.baseline_cell_times is not None
                    row_times[left:right] = dataset.baseline_cell_times[global_rows]
            row_strata = np.ascontiguousarray(row_strata, dtype=np.int16)
            if row_exposure is not None:
                row_exposure = np.ascontiguousarray(row_exposure, dtype=np.float64)
            if row_times is not None:
                row_times = np.ascontiguousarray(row_times, dtype=np.int64)
        keep = local >= 0
        local = local[keep]
        if irregular:
            assert row_times is not None
            kept_times = dataset.target_times[keep]
            rows = np.empty(len(kept_times), dtype=np.int64)
            for entity in np.unique(local):
                selected = np.flatnonzero(local == entity)
                left = int(offsets[entity])
                right = int(offsets[entity + 1])
                positions = np.searchsorted(row_times[left:right], kept_times[selected])
                if np.any(positions >= right - left) or not np.array_equal(
                    row_times[left + positions], kept_times[selected]
                ):
                    raise ValueError("continuous target is not a risk-grid boundary")
                rows[selected] = left + positions
        else:
            rows = offsets[local] + dataset.target_times[keep] - starts[local]
        counts = dataset.target_multiplicity[keep] * weights[local]
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
            baseline_strata=np.ascontiguousarray(strata, dtype=np.int16),
            offsets=offsets,
            row_times=row_times,
            n_grid=int(offsets[-1]),
            baseline_row_strata=row_strata,
            baseline_row_exposure=row_exposure,
            target_rows=np.asarray(rows, dtype=np.int64),
            target_counts=np.asarray(counts, dtype=np.float64),
            entity_weights=np.ascontiguousarray(weights),
            uniform_entity_weight=(
                float(weights[0])
                if len(weights) and np.all(weights == weights[0])
                else None
            ),
            population_entities=population,
        )

    def rows_to_entity_time(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        local = np.searchsorted(self.offsets, rows, side="right") - 1
        times = (
            self.starts[local] + rows - self.offsets[local]
            if self.row_times is None
            else self.row_times[rows]
        )
        return local.astype(np.int32), times.astype(np.int64)

    def interval_positions(
        self,
        entities: np.ndarray,
        times: np.ndarray,
        *,
        allow_right_endpoint: bool = False,
    ) -> np.ndarray:
        """Map raw continuous timestamps to irregular risk-row boundaries."""
        if self.row_times is None:
            raise ValueError("interval positions require a continuous context")
        entities = np.asarray(entities, dtype=np.int32)
        times = np.asarray(times, dtype=np.int64)
        if entities.shape != times.shape:
            raise ValueError("entity/time arrays are misaligned")
        output = np.empty(len(times), dtype=np.int64)
        for entity in np.unique(entities):
            selected = np.flatnonzero(entities == entity)
            left = int(self.offsets[entity])
            right = int(self.offsets[entity + 1])
            positions = np.searchsorted(self.row_times[left:right], times[selected])
            global_positions = left + positions
            if not allow_right_endpoint:
                valid = positions < right - left
                if np.any(~valid) or not np.array_equal(
                    self.row_times[global_positions], times[selected]
                ):
                    raise ValueError("timestamp is not a continuous risk boundary")
            else:
                valid = positions <= right - left
                if np.any(~valid):
                    raise ValueError("timestamp lies after the continuous risk set")
                interior = positions < right - left
                if np.any(interior) and not np.array_equal(
                    self.row_times[global_positions[interior]],
                    times[selected][interior],
                ):
                    raise ValueError("timestamp is not a continuous risk boundary")
                endpoint = ~interior
                # ``np.array_equal(array, scalar)`` is always false because
                # their shapes differ, even when every requested boundary is
                # the valid terminal endpoint.  Compare elementwise instead.
                if np.any(endpoint) and not np.all(
                    times[selected][endpoint] == self.ends[entity] + 1
                ):
                    raise ValueError("invalid continuous right endpoint")
            output[selected] = global_positions
        return output

    def weights_at_rows(self, rows: np.ndarray) -> np.ndarray:
        """Return the Horvitz--Thompson entity weight for each grid row."""
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) and (np.any(rows < 0) or np.any(rows >= self.n_grid)):
            raise ValueError("grid row is outside the context")
        if self.uniform_entity_weight is not None:
            result = np.full(rows.shape, self.uniform_entity_weight, dtype=np.float64)
        else:
            local = np.searchsorted(self.offsets, rows, side="right") - 1
            result = self.entity_weights[local].copy()
        if self.baseline_row_exposure is not None:
            result *= self.baseline_row_exposure[rows]
        return result

    def baseline_groups_at_rows(self, rows: np.ndarray) -> np.ndarray:
        """Return the pre-registered baseline stratum for each grid row."""
        return self.temporal_baseline_groups_at_rows(rows, time_bins=1)

    def temporal_baseline_groups_at_rows(
        self, rows: np.ndarray, *, time_bins: int
    ) -> np.ndarray:
        """Return outcome-blind stratum x age x calendar baseline cells.

        The bin edges use only the dataset observation clock and declared
        baseline origins.  They never use predicate histories, target labels,
        D_cert, or D_test.  ``time_bins=1`` is exactly the former static
        stratum intercept.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) and (np.any(rows < 0) or np.any(rows >= self.n_grid)):
            raise ValueError("grid row is outside the context")
        local, times = self.rows_to_entity_time(rows)
        row_strata = (
            self.baseline_strata[local]
            if self.baseline_row_strata is None
            else self.baseline_row_strata[rows]
        )
        bins = int(time_bins)
        if bins == 1:
            return np.ascontiguousarray(row_strata, dtype=np.int32)
        calendar_edges, age_edges = self._temporal_baseline_edges(bins)
        calendar_bin = np.searchsorted(calendar_edges[1:-1], times, side="right")
        age = np.maximum(0, times - self.baseline_origins[local])
        age_bin = np.searchsorted(age_edges[1:-1], age, side="right")
        strata = self.dataset.n_baseline_strata
        raw_groups = row_strata.astype(np.int64) + strata * (
            age_bin + bins * calendar_bin
        )
        groups = _temporal_baseline_layout(self.dataset, bins)[raw_groups]
        if np.any(groups < 0):
            raise AssertionError("row mapped to an impossible temporal baseline cell")
        return np.ascontiguousarray(groups, dtype=np.int32)

    def _temporal_baseline_edges(self, time_bins: int) -> tuple[np.ndarray, np.ndarray]:
        return _dataset_temporal_baseline_edges(self.dataset, time_bins)

    @property
    def temporal_baseline_strata(self) -> int:
        """Number of static protocol/cohort strata before time crossing."""
        return self.dataset.n_baseline_strata

    def temporal_baseline_counts(self, *, time_bins: int) -> np.ndarray:
        """Exact row counts by entity and frozen structural-time cell."""
        bins = int(time_bins)
        strata = self.dataset.n_baseline_strata
        layout = _temporal_baseline_layout(self.dataset, bins)
        dimension = int(np.count_nonzero(layout >= 0))
        if self.baseline_row_strata is not None:
            counts = np.zeros((len(self.entity_codes), dimension), dtype=np.float64)
            expected = np.zeros(len(self.entity_codes), dtype=np.float64)
            chunk = 1_000_000
            for left in range(0, self.n_grid, chunk):
                right = min(self.n_grid, left + chunk)
                rows = np.arange(left, right, dtype=np.int64)
                local = np.searchsorted(self.offsets, rows, side="right") - 1
                groups = self.temporal_baseline_groups_at_rows(rows, time_bins=bins)
                exposure = (
                    np.ones(len(rows), dtype=np.float64)
                    if self.baseline_row_exposure is None
                    else self.baseline_row_exposure[left:right]
                )
                np.add.at(counts, (local, groups), exposure)
                np.add.at(expected, local, exposure)
            if not np.allclose(
                counts.sum(axis=1), expected, rtol=1.0e-12, atol=1.0e-12
            ):
                raise AssertionError(
                    "dynamic baseline cells do not cover observation intervals"
                )
            return counts
        if bins == 1:
            counts = np.zeros((len(self.entity_codes), strata), dtype=np.float64)
            counts[np.arange(len(self.entity_codes)), self.baseline_strata] = (
                self.ends - self.starts + 1
            )
            return counts
        calendar_edges, age_edges = self._temporal_baseline_edges(bins)
        counts = np.zeros((len(self.entity_codes), dimension), dtype=np.float64)
        for calendar_bin in range(bins):
            cal_left = calendar_edges[calendar_bin]
            cal_right = calendar_edges[calendar_bin + 1] - 1
            for age_bin in range(bins):
                age_left = self.baseline_origins + age_edges[age_bin]
                age_right = self.baseline_origins + age_edges[age_bin + 1] - 1
                left = np.maximum(self.starts, np.maximum(cal_left, age_left))
                right = np.minimum(self.ends, np.minimum(cal_right, age_right))
                lengths = np.maximum(0, right - left + 1).astype(np.float64)
                raw_group = self.baseline_strata.astype(np.int64) + strata * (
                    age_bin + bins * calendar_bin
                )
                group = layout[raw_group]
                present = lengths > 0
                if np.any(group[present] < 0):
                    raise AssertionError("observed temporal baseline cell is absent")
                counts[np.flatnonzero(present), group[present]] = lengths[present]
        if not np.array_equal(
            counts.sum(axis=1), (self.ends - self.starts + 1).astype(np.float64)
        ):
            raise AssertionError(
                "temporal baseline cells do not cover observation intervals"
            )
        return counts

    def temporal_baseline_segments(
        self, *, time_bins: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the exact piecewise-constant baseline layout by entity.

        The sparse likelihood aggregates rows by structural time cells, while
        frequency-robust evidence needs calendar-time risk-set totals.  This
        method exposes the same frozen cell partition as half-open segments
        ``[left, right)`` without expanding the entity-by-time grid.  Segment
        construction uses observation bounds and baseline origins only; it is
        therefore target and predicate blind.
        """

        bins = int(time_bins)
        if bins < 1:
            raise ValueError("time_bins must be positive")
        if self.baseline_row_strata is not None:
            rows = np.arange(self.n_grid, dtype=np.int64)
            local, times = self.rows_to_entity_time(rows)
            row_groups = self.temporal_baseline_groups_at_rows(rows, time_bins=bins)
            if self.baseline_row_exposure is not None:
                observed = self.baseline_row_exposure > 0.0
                local = local[observed]
                times = times[observed]
                row_groups = row_groups[observed]
                exposures = self.baseline_row_exposure[observed]
            else:
                exposures = np.ones(len(times), dtype=np.float64)
            if not len(times):
                empty_i32 = np.zeros(0, dtype=np.int32)
                empty_i64 = np.zeros(0, dtype=np.int64)
                return empty_i32, empty_i64, empty_i64.copy(), empty_i32.copy()
            right_times = times + np.rint(
                exposures * self.dataset.ticks_per_unit
            ).astype(np.int64)
            boundary = (
                np.r_[True, local[1:] != local[:-1]]
                | np.r_[True, row_groups[1:] != row_groups[:-1]]
                | np.r_[True, times[1:] != right_times[:-1]]
            )
            first = np.flatnonzero(boundary)
            last = np.r_[first[1:] - 1, len(times) - 1]
            return (
                np.ascontiguousarray(local[first], dtype=np.int32),
                np.ascontiguousarray(times[first], dtype=np.int64),
                np.ascontiguousarray(right_times[last], dtype=np.int64),
                np.ascontiguousarray(row_groups[first], dtype=np.int32),
            )
        if bins == 1:
            return (
                np.arange(len(self.entity_codes), dtype=np.int32),
                np.ascontiguousarray(self.starts, dtype=np.int64),
                np.ascontiguousarray(self.ends + 1, dtype=np.int64),
                np.ascontiguousarray(self.baseline_strata, dtype=np.int32),
            )
        calendar_edges, age_edges = self._temporal_baseline_edges(bins)
        entities: list[np.ndarray] = []
        lefts: list[np.ndarray] = []
        rights: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        local_all = np.arange(len(self.entity_codes), dtype=np.int32)
        strata = self.dataset.n_baseline_strata
        layout = _temporal_baseline_layout(self.dataset, bins)
        for calendar_bin in range(bins):
            calendar_left = int(calendar_edges[calendar_bin])
            calendar_right = int(calendar_edges[calendar_bin + 1])
            for age_bin in range(bins):
                left = np.maximum(
                    self.starts,
                    np.maximum(
                        calendar_left,
                        self.baseline_origins + int(age_edges[age_bin]),
                    ),
                )
                right = np.minimum(
                    self.ends + 1,
                    np.minimum(
                        calendar_right,
                        self.baseline_origins + int(age_edges[age_bin + 1]),
                    ),
                )
                present = right > left
                if not np.any(present):
                    continue
                local = local_all[present]
                raw_group = self.baseline_strata[local].astype(np.int64) + strata * (
                    age_bin + bins * calendar_bin
                )
                group = layout[raw_group]
                if np.any(group < 0):
                    raise AssertionError("observed temporal baseline segment is absent")
                entities.append(local)
                lefts.append(np.asarray(left[present], dtype=np.int64))
                rights.append(np.asarray(right[present], dtype=np.int64))
                groups.append(np.asarray(group, dtype=np.int32))
        if not entities:
            empty_i32 = np.zeros(0, dtype=np.int32)
            empty_i64 = np.zeros(0, dtype=np.int64)
            return empty_i32, empty_i64, empty_i64.copy(), empty_i32.copy()
        entity = np.ascontiguousarray(np.concatenate(entities), dtype=np.int32)
        left = np.ascontiguousarray(np.concatenate(lefts), dtype=np.int64)
        right = np.ascontiguousarray(np.concatenate(rights), dtype=np.int64)
        group = np.ascontiguousarray(np.concatenate(groups), dtype=np.int32)
        covered = np.bincount(
            entity,
            weights=(right - left).astype(np.float64),
            minlength=len(self.entity_codes),
        )
        if not np.array_equal(covered.astype(np.int64), self.ends - self.starts + 1):
            raise AssertionError(
                "temporal baseline segments do not cover risk intervals"
            )
        return entity, left, right, group

    def weighted_baseline_totals(
        self, group_count: int, *, time_bins: int = 1
    ) -> np.ndarray:
        counts = self.temporal_baseline_counts(time_bins=time_bins)
        if counts.shape[1] != int(group_count):
            raise ValueError("baseline group count does not match temporal contract")
        return self.entity_weights @ counts

    def entity_exposure_totals(self) -> np.ndarray:
        """Return declared observation opportunity per selected entity."""
        if self.baseline_row_exposure is None:
            return (self.ends - self.starts + 1).astype(np.float64)
        return np.add.reduceat(self.baseline_row_exposure, self.offsets[:-1]).astype(
            np.float64
        )

    def all_row_weights(self) -> np.ndarray:
        if self.uniform_entity_weight is not None:
            result = np.full(self.n_grid, self.uniform_entity_weight, dtype=np.float64)
        else:
            lengths = np.diff(self.offsets)
            result = np.repeat(self.entity_weights, lengths)
        if self.baseline_row_exposure is not None:
            result *= self.baseline_row_exposure
        return result

    def target_counts_at_sorted_rows(self, rows: np.ndarray) -> np.ndarray:
        """Gather sparse target counts onto sorted unique grid rows.

        Target rows are much rarer than observation rows in the financial
        datasets.  Searching the short target stream in the long requested
        row vector avoids an ``O(len(rows))`` int64 position temporary while
        returning the same dense event vector used by the likelihood.
        """
        rows = np.asarray(rows, dtype=np.int64)
        event = np.zeros(len(rows), dtype=np.float64)
        if not len(rows) or not len(self.target_rows):
            return event
        left = int(np.searchsorted(self.target_rows, rows[0], side="left"))
        right = int(np.searchsorted(self.target_rows, rows[-1], side="right"))
        targets = self.target_rows[left:right]
        if not len(targets):
            return event
        positions = np.searchsorted(rows, targets)
        matched = positions < len(rows)
        safe = np.minimum(positions, len(rows) - 1)
        matched &= rows[safe] == targets
        if np.any(matched):
            event[positions[matched]] = self.target_counts[left:right][matched]
        return event

    @property
    def weighted_n_grid(self) -> float:
        return float(np.sum(self.all_row_weights(), dtype=np.float64))


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
    closure_signs: tuple[int, ...]
    active_rows: np.ndarray
    active_design_groups: np.ndarray
    active_baseline_groups: np.ndarray
    aggregate_baseline_groups: np.ndarray
    # Fixed baseline controls are one-direction nonnegative-kernel nuisance
    # blocks. Exposure-decrease controls have a frozen negative design sign.
    # are excluded from MDL complexity like the intercept, but unlike the
    # intercept they belong to the projected cone.  Keeping their width
    # explicit separates "unpenalized baseline" from "unconstrained".
    control_dimension: int = 0
    resident_token: int = field(
        default_factory=lambda: next(_MODEL_MATRIX_TOKENS),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if len(self.closure_signs) != len(self.closure) or any(
            sign not in {-1, 1} for sign in self.closure_signs
        ):
            raise ValueError("closure terms require one frozen sign each")
        if not 0 <= self.free_dimension <= self.dimension:
            raise ValueError("invalid free baseline dimension")
        if (
            self.control_dimension < 0
            or self.free_dimension + self.control_dimension > self.dimension
        ):
            raise ValueError("invalid constrained control dimension")
        if (
            self.closure_dimension < 0
            or self.baseline_dimension + self.closure_dimension > self.dimension
        ):
            raise ValueError("invalid constrained closure dimension")
        rule_start = self.baseline_dimension + self.closure_dimension
        if self.rule_slices and self.rule_slices[0].start != rule_start:
            raise ValueError("reported rules must follow constrained closure blocks")

    @property
    def dimension(self) -> int:
        return int(self.x.shape[1])

    @property
    def baseline_dimension(self) -> int:
        return int(self.free_dimension + self.control_dimension)

    @property
    def nbytes(self) -> int:
        return int(
            self.x.nbytes
            + self.exposure_weight.nbytes
            + self.noevent_weight.nbytes
            + self.event_weight.nbytes
            + self.active_rows.nbytes
            + self.active_design_groups.nbytes
            + self.active_baseline_groups.nbytes
            + self.aggregate_baseline_groups.nbytes
        )


class ResponseEngine:
    def __init__(
        self,
        dataset: Dataset,
        *,
        lag: int,
        knot_count: int,
        cache_bytes: int,
        baseline_time_bins: int = 1,
        effect_model: str = "total_state",
    ):
        self.dataset = dataset
        if effect_model not in {
            "total_state",
            "additive_hierarchy",
            "support_additive",
        }:
            raise ValueError("invalid rule effect model")
        self.effect_model = str(effect_model)
        self.lag_units = int(lag)
        self.lag = int(lag) * dataset.ticks_per_unit
        self.continuous = dataset.likelihood == "continuous_poisson"
        self.tick_exposure = (
            1.0 / dataset.ticks_per_unit if dataset.likelihood == "poisson" else 1.0
        )
        # Hierarchy nuisance uses the same one-direction kernel family as a
        # reported rule.  Signs are profiled once on D_fit by SupportOptimizer
        # and then frozen; an absent entry is a deterministic excitation
        # default used only by low-level matrix tests.
        self._closure_signs: dict[ClosureTerm, int] = {}
        self.knot_count = int(knot_count)
        if self.continuous:
            if self.lag_units < self.knot_count:
                raise ValueError(
                    "continuous impact horizon must span every kernel interval"
                )
            self.continuous_edges = (
                np.arange(self.knot_count + 1, dtype=np.int64) * self.lag
            ) // self.knot_count
            self.continuous_edges[-1] = self.lag
            unit_edges = (
                np.arange(self.knot_count + 1, dtype=np.int64) * self.lag_units
            ) // self.knot_count
            unit_edges[-1] = self.lag_units
            self.basis = np.zeros((self.knot_count, self.lag_units), dtype=np.float64)
            for index in range(self.knot_count):
                left, right = int(unit_edges[index]), int(unit_edges[index + 1])
                width = max(1, right - left)
                self.basis[index, left:right] = 1.0 / width
        else:
            self.continuous_edges = np.zeros(0, dtype=np.int64)
            self.basis = triangular_basis(self.lag, knot_count)
        if dataset.likelihood == "poisson":
            # Unit integral in the declared continuous-time unit (hours for
            # IBM), rather than unit sum in implementation ticks.
            self.basis *= dataset.ticks_per_unit
        # The preregistered null is an intercept plus fixed, non-reportable
        # control histories.  Controls use the same fixed M-knot lag basis as
        # reported rules and remain non-reportable nuisance blocks.  Their
        # pre-registered positive direction and nonnegative coefficients
        # prevent a single control from fitting alternating lag signs.  Giving
        # a rule four lag degrees of freedom while allowing an
        # activity control only one flat rolling-count coefficient lets rules
        # win merely by supplying the missing baseline lag shape.  Equal basis
        # capacity removes that artifact without selecting any control from the
        # target.  Because response blocks are strictly future, status at t can
        # affect only t+1 onward.
        self.control_predicates = dataset.baseline_control_predicates
        self.control_signs = dataset.baseline_control_signs
        self.baseline_time_bins = int(baseline_time_bins)
        if not 1 <= self.baseline_time_bins <= 8:
            raise ValueError("baseline_time_bins must lie in [1, 8]")
        self.free_baseline_dimension = int(
            np.count_nonzero(
                _temporal_baseline_layout(dataset, self.baseline_time_bins) >= 0
            )
        )
        self.baseline_dimension = (
            self.free_baseline_dimension
            + len(self.control_predicates) * self.knot_count
        )
        self.cache_bytes = max(0, int(cache_bytes))
        # A quantile-band dictionary is frozen on D_fit by SupportOptimizer.
        # Missing entries retain the historical cumulative ``span <= W``
        # response.  For a registered higher-order pattern, W=0 is the exact
        # same-tick identity and every positive W is the disjoint interval
        # ``previous_W < span <= W``.  The map lives in the response engine so
        # route pricing, exact fitting and held-out footprints cannot silently
        # disagree about the meaning of W.
        self._window_band_lower: dict[tuple[str, Antecedent, int], int] = {}
        # Numeric LRUs together stay within ``cache_bytes``.  Threshold rows
        # and their dense int32 lookup are reused by many triplet closures;
        # reserving explicit space for them avoids rebuilding an O(n_grid)
        # array for every high-order skeleton.
        self._block_cache_limit = self.cache_bytes // 2
        self._cache: OrderedDict[tuple, SparseBlock] = OrderedDict()
        self._cache_size = 0

        self._completion_cache: OrderedDict[
            tuple, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._completion_cache_size = 0
        # Lower-order completions are reused by every pair/triplet hierarchy
        # closure.  The former hard 2-GiB cap forced exact recomputation even
        # when the configured bounded cache had ample room (Freddie allocates
        # 16 GiB to this engine).  All numeric LRUs still sum to at most
        # ``cache_bytes``; only the artificial cap is removed.
        self._completion_cache_limit = self.cache_bytes // 4
        self._source_cache: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._marked_source_cache: OrderedDict[
            tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._marked_source_cache_size = 0
        self._marked_source_cache_limit = self.cache_bytes // 32
        self._history_level_cache: dict[tuple[int, int, int], tuple[int, ...]] = {}
        self._history_count_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._history_quantile_cache: dict[
            tuple[int, int, int, tuple[float, ...]], tuple[int, ...]
        ] = {}
        self._state_interval_cache: dict[tuple[int, int], StateIntervals] = {}
        self._baseline_cache: dict[int, np.ndarray] = {}
        # The exact intercept/control model is the immutable parent of every
        # support in one Context.  Reusing it lets arbitrary fresh or
        # forced-closure matrices use the touched-only extension path instead
        # of rebuilding control responses and hashing the complete row union.
        self._baseline_model_cache: dict[int, ModelMatrix] = {}
        self._entity_age_cache: dict[int, np.ndarray] = {}
        self._footprint_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._footprint_cache_size = 0
        self._footprint_cache_limit = self.cache_bytes // 32
        self._row_threshold_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._row_threshold_cache_size = 0
        self._row_threshold_cache_limit = 3 * self.cache_bytes // 32
        self._row_lookup_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._row_lookup_cache_size = 0
        self._row_lookup_cache_limit = self.cache_bytes // 8
        self._lock = threading.RLock()
        # Different exact-fit workers frequently request the same hierarchy
        # blocks.  The LRU itself is thread-safe, but without key-scoped locks
        # every worker can still perform the same expensive miss concurrently.
        # Keep these locks for the engine lifetime: the number of structural
        # keys is finite and tiny compared with the cached numeric arrays.
        self._compute_locks: dict[tuple, threading.Lock] = {}

    @property
    def has_window_bands(self) -> bool:
        return bool(self._window_band_lower)

    def configure_window_bands(
        self,
        window_dictionary: dict[PatternKey, tuple[int, ...]],
    ) -> None:
        """Freeze disjoint formation-lag intervals for this engine.

        The dictionary must already have been learned without target values on
        D_fit.  Calling this method after a response block has been built would
        make an existing cache key ambiguous, so that misuse is rejected.
        """

        if self._cache or self._footprint_cache:
            raise RuntimeError("window bands must be configured before response use")
        bands: dict[tuple[str, Antecedent, int], int] = {}
        for raw_pattern, raw_windows in window_dictionary.items():
            relation, antecedent = normalize_pattern(raw_pattern)
            if len(antecedent) == 1:
                continue
            previous = 0
            for window in sorted(set(map(int, raw_windows))):
                if window < 0:
                    raise ValueError("window-band boundaries must be nonnegative")
                lower = -1 if window == 0 else previous
                bands[(relation, antecedent, window)] = int(lower)
                previous = window
        self._window_band_lower = bands

    def window_band_lower(
        self,
        antecedent: Antecedent,
        window: int,
        relation: str = "unordered",
    ) -> int | None:
        relation = "atomic" if len(antecedent) == 1 else str(relation)
        return self._window_band_lower.get((relation, antecedent, int(window)))

    def _keep_unaggregated_design(self, x: np.ndarray) -> bool:
        """Return whether exact hash aggregation is predictably unproductive.

        ``x`` has already been allocated by the caller.  Retaining it therefore
        cannot require more memory than entering the hash aggregator, whose
        compact result is copied before the input can be released.  A
        deterministic sample estimates whether compaction can remove at least
        half the rows.  Below that break-even, the measured full-data cost of
        the single-thread hash exceeds the extra resident Poisson work, so the
        original weighted rows are retained.  The 6-GiB cap is the exact
        solver's resident-device limit.  These are storage/backend decisions:
        every likelihood row and weight remains exact.
        """
        rows = int(x.shape[0])
        if not rows:
            return False
        sample_count = min(rows, 65_536)
        if rows < 65_536 or sample_count <= 1:
            return False
        positions = np.linspace(0, rows - 1, sample_count, dtype=np.int64)
        sample = np.ascontiguousarray(x[positions])
        weights = np.linspace(
            1.0,
            2.0,
            x.shape[1],
            dtype=np.float64,
        )
        projection = sample @ weights
        unique_fraction = len(np.unique(projection)) / sample_count
        return bool(x.nbytes <= 6 * 1024**3 and unique_fraction >= 0.5)

    def aggregate_or_keep_design_rows(
        self,
        x: np.ndarray,
        exposure_weight: np.ndarray,
        noevent_weight: np.ndarray,
        event_weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Losslessly aggregate rows without allocating an unused inverse map."""
        if self._keep_unaggregated_design(x):
            return (
                np.ascontiguousarray(x),
                np.ascontiguousarray(exposure_weight),
                np.ascontiguousarray(noevent_weight),
                np.ascontiguousarray(event_weight),
            )
        # The compiled hash aggregator compacts its input arrays in place.
        # ``x`` is caller-owned scratch in every construction path, whereas
        # likelihood weights may be views of an existing parent ModelMatrix.
        # Copy only the three narrow vectors so a projected child can never
        # corrupt its parent's sufficient statistics.
        return aggregate_design_rows(
            x,
            np.asarray(exposure_weight, dtype=np.float64).copy(),
            np.asarray(noevent_weight, dtype=np.float64).copy(),
            np.asarray(event_weight, dtype=np.float64).copy(),
            copy_input=False,
        )

    def _aggregate_or_keep_design_rows(
        self,
        x: np.ndarray,
        exposure_weight: np.ndarray,
        noevent_weight: np.ndarray,
        event_weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Losslessly choose between aggregation and the original row design.

        Hash aggregation is valuable when millions of response rows collapse
        to a small number of repeated kernel profiles.  For some alternating
        high-order histories, however, essentially every row is distinct; the
        old implementation then spent tens of seconds hashing a multi-GiB
        matrix only to return the same number of rows.

        A deterministic, evenly spaced probe is used solely as a computational
        routing decision.  If aggregation cannot halve the sampled row count,
        the original likelihood rows and weights are returned unchanged.
        This is mathematically identical to summing equal rows: no event,
        support, coefficient, likelihood term, or optimization tolerance is
        approximated.  A missed duplicate merely forgoes compression.
        """
        rows = int(x.shape[0])
        if self._keep_unaggregated_design(x):
            return (
                np.ascontiguousarray(x),
                np.ascontiguousarray(exposure_weight),
                np.ascontiguousarray(noevent_weight),
                np.ascontiguousarray(event_weight),
                np.arange(rows, dtype=np.int64),
            )
        return aggregate_design_rows_with_groups(
            x,
            # The native routine aggregates in place.  These weights often
            # belong to a live parent support, so they must not be borrowed
            # mutably.  Copying three vectors is far smaller than copying the
            # multi-GiB design and preserves exact parent/child objectives.
            np.asarray(exposure_weight, dtype=np.float64).copy(),
            np.asarray(noevent_weight, dtype=np.float64).copy(),
            np.asarray(event_weight, dtype=np.float64).copy(),
            copy_input=False,
        )

    def set_closure_sign(self, term: ClosureTerm, sign: int) -> None:
        sign = int(sign)
        if sign not in {-1, 1}:
            raise ValueError("closure sign must be -1 or +1")
        incumbent = self._closure_signs.setdefault(term, sign)
        if incumbent != sign:
            raise ValueError("a frozen closure sign cannot be changed")

    def closure_sign(self, term: ClosureTerm) -> int:
        return int(self._closure_signs.get(term, 1))

    def _compute_lock(self, namespace: str, key: tuple) -> threading.Lock:
        namespaced = (namespace, *key)
        with self._lock:
            lock = self._compute_locks.get(namespaced)
            if lock is None:
                lock = threading.Lock()
                self._compute_locks[namespaced] = lock
            return lock

    def clear_caches(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cache_size = 0
            self._completion_cache.clear()
            self._completion_cache_size = 0
            self._source_cache.clear()
            self._marked_source_cache.clear()
            self._marked_source_cache_size = 0
            self._history_level_cache.clear()
            self._history_count_cache.clear()
            self._history_quantile_cache.clear()
            self._state_interval_cache.clear()
            self._baseline_cache.clear()
            self._entity_age_cache.clear()
            self._footprint_cache.clear()
            self._footprint_cache_size = 0
            self._row_threshold_cache.clear()
            self._row_threshold_cache_size = 0
            self._row_lookup_cache.clear()
            self._row_lookup_cache_size = 0
            self._compute_locks.clear()

    def clear_completion_cache(self) -> None:
        """Release transient completion arrays without dropping other caches.

        Discovery can materialize one immutable completion automaton spanning
        the complete finite skeleton dictionary.  Once that automaton owns
        the arrays, retaining the per-antecedent LRU entries only duplicates
        memory.  Clearing this cache changes no response or fitted objective.
        """
        with self._lock:
            self._completion_cache.clear()
            self._completion_cache_size = 0
            for key in tuple(self._compute_locks):
                if key and key[0] == "completion":
                    self._compute_locks.pop(key, None)

    def retain_support_blocks(
        self, context: Context, supports: tuple[Support, ...]
    ) -> None:
        """Drop discovery-only W blocks while retaining frozen-family models."""
        block_keys: set[tuple] = set()
        completion_keys: set[tuple] = set()
        for support in supports:
            for term in hierarchy_closure(support):
                if term.history_marks:
                    block_keys.add(
                        (
                            id(context),
                            "history-rule",
                            "atomic" if len(term.antecedent) == 1 else "unordered",
                            term.antecedent,
                            term.window,
                            term.history_marks,
                        )
                    )
                    completion_keys.add(
                        (
                            id(context),
                            "atomic" if len(term.antecedent) == 1 else "unordered",
                            term.antecedent,
                            term.history_marks,
                        )
                    )
                else:
                    closure_relation = (
                        "atomic" if len(term.antecedent) == 1 else "unordered"
                    )
                    block_keys.add(
                        (id(context), closure_relation, term.antecedent, term.window)
                    )
                    completion_keys.add(
                        (id(context), closure_relation, term.antecedent)
                    )
            for rule in support.rules:
                if rule.history_marks:
                    block_keys.add(
                        (
                            id(context),
                            "history-rule",
                            rule.relation,
                            rule.antecedent,
                            rule.window,
                            rule.history_marks,
                        )
                    )
                    completion_keys.add(
                        (
                            id(context),
                            rule.relation,
                            rule.antecedent,
                            rule.history_marks,
                        )
                    )
                else:
                    block_keys.add(
                        (id(context), rule.relation, rule.antecedent, rule.window)
                    )
                    completion_keys.add((id(context), rule.relation, rule.antecedent))
        context_id = id(context)
        with self._lock:
            for key in tuple(self._cache):
                if key[0] != context_id or key not in block_keys:
                    removed = self._cache.pop(key)
                    self._cache_size -= removed.nbytes
            for key in tuple(self._completion_cache):
                if key[0] != context_id or key not in completion_keys:
                    removed = self._completion_cache.pop(key)
                    self._completion_cache_size -= sum(
                        array.nbytes for array in removed
                    )
            self._footprint_cache.clear()
            self._footprint_cache_size = 0
            self._row_threshold_cache.clear()
            self._row_threshold_cache_size = 0
            self._row_lookup_cache.clear()
            self._row_lookup_cache_size = 0
            self._compute_locks.clear()

    def _source(
        self, predicate: int, context: Context
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (id(context), int(predicate))
        with self._lock:
            cached = self._source_cache.get(key)
            if cached is not None:
                return cached
        with self._compute_lock("source", key):
            with self._lock:
                cached = self._source_cache.get(key)
                if cached is not None:
                    return cached
            if self.dataset.is_state_predicate(predicate):
                intervals = self._state_intervals(context, predicate)
                result = (
                    intervals.entities,
                    intervals.starts,
                    intervals.entry_primitive_ids,
                )
            else:
                entities, times, primitive_ids = self.dataset.predicate_stream_with_ids(
                    predicate
                )
                local = context.entity_lookup[entities]
                keep = local >= 0
                result = (
                    local[keep].astype(np.int32),
                    times[keep].astype(np.int64),
                    primitive_ids[keep].astype(np.int64),
                )
            with self._lock:
                self._source_cache[key] = result
            return result

    def _history_prior_counts(
        self,
        predicate: int,
        lookback: int,
        context: Context,
    ) -> np.ndarray:
        """Return exact [t-L,t) counts once for every source event.

        Quantile freezing, level enumeration and each finite c identity all
        use this identical target-blind count.  Keeping the immutable vector
        avoids rerunning the same entity-wise search for every threshold.
        """

        key = (id(context), int(predicate), int(lookback))
        with self._lock:
            cached = self._history_count_cache.get(key)
            if cached is not None:
                return cached
        entities, times, _ = self._source(predicate, context)
        counts = np.zeros(len(entities), dtype=np.int64)
        ticks = int(lookback) * self.dataset.ticks_per_unit
        if len(entities):
            _, starts = np.unique(entities, return_index=True)
            ends = np.r_[starts[1:], len(entities)]
            for left, right in zip(starts.tolist(), ends.tolist(), strict=True):
                local = times[left:right]
                lower = np.searchsorted(local, local - ticks, side="left")
                upper = np.searchsorted(local, local, side="left")
                counts[left:right] = upper - lower
        counts = np.ascontiguousarray(counts, dtype=np.int64)
        counts.setflags(write=False)
        with self._lock:
            incumbent = self._history_count_cache.setdefault(key, counts)
        return incumbent

    def _history_marked_source(
        self,
        predicate: int,
        lookback: int,
        threshold: int,
        context: Context,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return events whose same-predicate history satisfies one mark.

        For an event at t the count interval is exactly [t-L, t).  All events
        sharing t are excluded from one another, so input row order and
        duplicated log attributes cannot fabricate a repeat.  The operation
        uses predicate history only and is consequently safe to freeze on
        D_fit before any target likelihood is inspected.
        """
        if lookback < 1 or threshold < 1:
            raise ValueError("history mark requires positive lookback/count")
        key = (id(context), int(predicate), int(lookback), int(threshold))
        with self._lock:
            cached = self._marked_source_cache.get(key)
            if cached is not None:
                self._marked_source_cache.move_to_end(key)
                return cached
        with self._compute_lock("marked-source", key):
            with self._lock:
                cached = self._marked_source_cache.get(key)
                if cached is not None:
                    self._marked_source_cache.move_to_end(key)
                    return cached
            entities, times, primitive_ids = self._source(predicate, context)
            keep = self._history_prior_counts(predicate, lookback, context) >= int(
                threshold
            )
            result = (
                np.ascontiguousarray(entities[keep], dtype=np.int32),
                np.ascontiguousarray(times[keep], dtype=np.int64),
                np.ascontiguousarray(primitive_ids[keep], dtype=np.int64),
            )
            size = sum(value.nbytes for value in result)
            if size <= self._marked_source_cache_limit:
                with self._lock:
                    incumbent = self._marked_source_cache.get(key)
                    if incumbent is not None:
                        return incumbent
                    self._marked_source_cache[key] = result
                    self._marked_source_cache_size += size
                    while (
                        self._marked_source_cache_size > self._marked_source_cache_limit
                        and len(self._marked_source_cache) > 1
                    ):
                        _, removed = self._marked_source_cache.popitem(last=False)
                        self._marked_source_cache_size -= sum(
                            value.nbytes for value in removed
                        )
            return result

    def history_count_levels(
        self,
        context: Context,
        predicate: int,
        lookback: int,
    ) -> tuple[int, ...]:
        """Return every response-distinct positive count threshold on D_fit."""
        key = (id(context), int(predicate), int(lookback))
        with self._lock:
            cached = self._history_level_cache.get(key)
            if cached is not None:
                return cached
        counts = self._history_prior_counts(predicate, lookback, context)
        result = tuple(int(value) for value in np.unique(counts) if value > 0)
        with self._lock:
            self._history_level_cache[key] = result
        return result

    def history_count_quantiles(
        self,
        context: Context,
        predicate: int,
        lookback: int,
        quantiles: tuple[float, ...],
    ) -> tuple[int, ...]:
        """Freeze distinct positive prior-count quantiles without target use."""
        quantiles = tuple(map(float, quantiles))
        key = (id(context), int(predicate), int(lookback), quantiles)
        with self._lock:
            cached = self._history_quantile_cache.get(key)
            if cached is not None:
                return cached
        counts = self._history_prior_counts(predicate, lookback, context)
        positive = counts[counts > 0]
        if not len(positive):
            result: tuple[int, ...] = ()
            with self._lock:
                self._history_quantile_cache[key] = result
            return result
        counts = np.sort(positive)
        selected = {
            int(counts[max(0, int(math.ceil(float(q) * len(counts))) - 1)])
            for q in quantiles
        }
        result = tuple(sorted(selected))
        with self._lock:
            self._history_quantile_cache[key] = result
        return result

    def rule_completions(
        self,
        context: Context,
        rule: RuleIdentity,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact completions of an ordinary or history-marked rule."""
        if not rule.history_marks:
            return self.completions(context, rule.antecedent, rule.relation)
        key = (
            id(context),
            rule.relation,
            rule.antecedent,
            rule.history_marks,
        )
        with self._lock:
            cached = self._completion_cache.get(key)
            if cached is not None:
                self._completion_cache.move_to_end(key)
                return cached
        with self._compute_lock("marked-completion", key):
            with self._lock:
                cached = self._completion_cache.get(key)
                if cached is not None:
                    self._completion_cache.move_to_end(key)
                    return cached
            sources = [
                (
                    self._source(predicate, context)
                    if mark == (0, 0)
                    else self._history_marked_source(
                        predicate, mark[0], mark[1], context
                    )
                )
                for predicate, mark in zip(
                    rule.antecedent, rule.history_marks, strict=True
                )
            ]
            compiled = completion_events(sources, relation=rule.relation)
            if compiled is None:
                raise RuntimeError(
                    "history-marked completions require the compiled completion operator"
                )
            result = (
                np.ascontiguousarray(compiled[0], dtype=np.int32),
                np.ascontiguousarray(compiled[1], dtype=np.int64),
                np.ascontiguousarray(compiled[2], dtype=np.int64),
            )
            return self._retain_completions(key, result)

    def _state_intervals(self, context: Context, predicate: int) -> StateIntervals:
        key = (id(context), int(predicate))
        with self._lock:
            cached = self._state_interval_cache.get(key)
            if cached is not None:
                return cached
        definition = self.dataset.predicate_definition(predicate)
        if not self.dataset.is_state_predicate(predicate):
            raise ValueError("requested predicate is not a history state")
        if definition.get("kind") == "transition_state":
            entry_entities, entry_times, entry_ids = self._source(
                int(definition["entry_predicate"]), context
            )
            exit_entities, exit_times, _ = self._source(
                int(definition["exit_predicate"]), context
            )
            result = transition_state_intervals(
                entry_entities,
                entry_times,
                entry_ids,
                exit_entities,
                exit_times,
                context.ends,
            )
        else:
            source = int(definition["source_predicate"])
            source_entities, source_times, source_ids = self._source(source, context)
            result = history_state_intervals(
                source_entities,
                source_times,
                source_ids,
                context.ends,
                transform=str(definition["transform"]),
                horizon_ticks=(
                    int(definition["horizon"]) * self.dataset.ticks_per_unit
                ),
            )
        with self._lock:
            incumbent = self._state_interval_cache.setdefault(key, result)
        return incumbent

    def completions(
        self,
        context: Context,
        antecedent: Antecedent,
        relation: str = "unordered",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relation = normalize_relation(antecedent, relation)
        key = (id(context), relation, antecedent)
        with self._lock:
            cached = self._completion_cache.get(key)
            if cached is not None:
                self._completion_cache.move_to_end(key)
                return cached
        with self._compute_lock("completion", key):
            with self._lock:
                cached = self._completion_cache.get(key)
                if cached is not None:
                    self._completion_cache.move_to_end(key)
                    return cached
            return self._compute_completions(context, antecedent, relation, key)

    def seed_completions(
        self,
        context: Context,
        antecedent: Antecedent,
        result: tuple[np.ndarray, np.ndarray, np.ndarray],
        relation: str = "unordered",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Seed the ordinary exact-response cache from a validated store.

        Search can restore immutable completion streams from its
        content-addressed on-disk cache.  Registering the same arrays here
        lets later exact support fits reuse them instead of repeating the
        latest-witness merge.  This is a cache insertion only: callers must
        validate provenance, dtype and shape before invoking it.
        """
        relation = normalize_relation(antecedent, relation)
        key = (id(context), relation, antecedent)
        return self._retain_completions(key, result)

    def _compute_completions(
        self,
        context: Context,
        antecedent: Antecedent,
        relation: str,
        key: tuple[int, str, Antecedent],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_predicates = tuple(
            predicate
            for predicate in antecedent
            if self.dataset.is_state_predicate(predicate)
        )
        if state_predicates:
            event_predicates = tuple(
                predicate
                for predicate in antecedent
                if predicate not in state_predicates
            )
            if relation == "ordered":
                raise ValueError("ordered relations cannot contain history-state atoms")
            if event_predicates:
                event_relation = "atomic" if len(event_predicates) == 1 else "unordered"
                entities, times, spans = self.completions(
                    context, event_predicates, event_relation
                )
                keep = np.ones(len(entities), dtype=bool)
                for predicate in state_predicates:
                    keep &= active_at(
                        self._state_intervals(context, predicate), entities, times
                    )
                result = (entities[keep], times[keep], spans[keep])
                return self._retain_completions(key, result)
            # The v14 grammar admits state singleton atoms.  This generic
            # fallback also gives exact state-state entry semantics if a fixed
            # support is inspected manually: each distinct entry is a witness.
        sources = [self._source(predicate, context) for predicate in antecedent]
        compiled = completion_events(sources, relation=relation)
        if compiled is not None:
            result = (
                compiled[0].astype(np.int32, copy=False),
                compiled[1],
                compiled[2],
            )
            return self._retain_completions(key, result)
        per_source: list[dict[int, np.ndarray]] = []
        per_source_ids: list[dict[int, np.ndarray]] = []
        for entities, times, primitive_ids in sources:
            mapping: dict[int, np.ndarray] = {}
            id_mapping: dict[int, np.ndarray] = {}
            if len(entities):
                unique, first = np.unique(entities, return_index=True)
                for index, entity in enumerate(unique):
                    right = (
                        first[index + 1] if index + 1 < len(first) else len(entities)
                    )
                    local_times = times[first[index] : right]
                    local_ids = primitive_ids[first[index] : right]
                    order = np.lexsort((local_ids, local_times))
                    mapping[int(entity)] = local_times[order]
                    id_mapping[int(entity)] = local_ids[order]
            per_source.append(mapping)
            per_source_ids.append(id_mapping)
        eligible = set(per_source[0]) if per_source else set()
        for mapping in per_source[1:]:
            eligible.intersection_update(mapping)
        completion_entities: list[int] = []
        completion_times: list[int] = []
        completion_spans: list[int] = []
        for entity in sorted(eligible):
            streams = [mapping[entity] for mapping in per_source]
            id_streams = [mapping[entity] for mapping in per_source_ids]
            if len(streams) == 1:
                for time in np.unique(streams[0]):
                    completion_entities.append(entity)
                    completion_times.append(int(time))
                    completion_spans.append(0)
                continue
            union = sorted_unique_union(streams)
            if union is None:
                union = np.unique(np.concatenate(streams))
            latest_candidates: list[list[tuple[int, int]]] = [[] for _ in streams]
            positions = np.zeros(len(streams), dtype=np.int64)
            for time in union:
                for source_index, stream in enumerate(streams):
                    while (
                        positions[source_index] < len(stream)
                        and stream[positions[source_index]] <= time
                    ):
                        event_time = int(stream[positions[source_index]])
                        primitive_id = int(
                            id_streams[source_index][positions[source_index]]
                        )
                        candidates = latest_candidates[source_index]
                        candidates = [
                            item for item in candidates if item[1] != primitive_id
                        ]
                        candidates.append((event_time, primitive_id))
                        candidates.sort(key=lambda item: (-item[0], item[1]))
                        # With q sources, retaining the q newest distinct
                        # primitive IDs per source is exact: no matching can
                        # have more than q-1 of them occupied elsewhere.
                        latest_candidates[source_index] = candidates[: len(streams)]
                        positions[source_index] += 1
                assignments = [
                    (
                        sum(item[0] for item in witnesses),
                        max(item[0] for item in witnesses)
                        - min(item[0] for item in witnesses),
                    )
                    for witnesses in product(*latest_candidates)
                    if len({item[1] for item in witnesses}) == len(witnesses)
                ]
                if assignments:
                    _, span = max(
                        assignments,
                        key=lambda item: (item[0], -item[1]),
                    )
                    completion_entities.append(entity)
                    completion_times.append(int(time))
                    completion_spans.append(int(span))
        result = (
            np.asarray(completion_entities, dtype=np.int32),
            np.asarray(completion_times, dtype=np.int64),
            np.asarray(completion_spans, dtype=np.int64),
        )
        return self._retain_completions(key, result)

    def effective_windows(
        self,
        context: Context,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        relation: str = "unordered",
    ) -> tuple[int, ...]:
        """Return exactly the windows with distinct nonzero response blocks.

        Response blocks are nested in ``W``.  A newly admitted completion
        changes the block iff it has at least one strictly-future observed
        row.  Kernel bases are nonnegative and every productive completion
        contributes a positive basis entry, so counting productive spans is
        an exact response-equivalence test rather than a screening heuristic.
        """
        if not windows:
            return ()
        ordered_windows = tuple(sorted(set(map(int, windows))))
        sources = [self._source(predicate, context) for predicate in antecedent]
        compiled_counts = completion_window_counts(
            sources,
            context.ends,
            np.asarray(ordered_windows, dtype=np.int64) * self.dataset.ticks_per_unit,
            relation=relation,
        )
        if compiled_counts is not None:
            previous = 0
            effective: list[int] = []
            for window, admitted in zip(
                ordered_windows, compiled_counts.tolist(), strict=True
            ):
                if admitted > previous:
                    effective.append(window)
                    previous = admitted
            return tuple(effective)
        entities, times, spans = self.completions(context, antecedent, relation)
        if not len(entities):
            return ()
        productive = times < context.ends[entities]
        productive_spans = np.sort(spans[productive])
        if not len(productive_spans):
            return ()
        previous = 0
        effective: list[int] = []
        for window in ordered_windows:
            admitted = int(
                np.searchsorted(
                    productive_spans,
                    int(window) * self.dataset.ticks_per_unit,
                    side="right",
                )
            )
            if admitted > previous:
                effective.append(int(window))
                previous = admitted
        return tuple(effective)

    def quantile_windows(
        self,
        context: Context,
        antecedent: Antecedent,
        quantiles: tuple[float, ...],
        *,
        maximum_window: int | None = None,
        relation: str = "unordered",
    ) -> tuple[int, ...]:
        windows, _ = self.quantile_windows_with_provenance(
            context,
            antecedent,
            quantiles,
            maximum_window=maximum_window,
            relation=relation,
        )
        return windows

    def quantile_windows_with_provenance(
        self,
        context: Context,
        antecedent: Antecedent,
        quantiles: tuple[float, ...],
        *,
        maximum_window: int | None = None,
        relation: str = "unordered",
    ) -> tuple[tuple[int, ...], dict[int, tuple[float, ...]]]:
        """Freeze distinct target-blind productive-span quantiles on D_fit.

        The cumulative completion counts used to locate the quantiles also
        prove response distinctness: every returned positive window admits at
        least one productive completion not admitted by the preceding
        returned window.  Consequently callers need not repeat the complete
        stream pass through :meth:`effective_windows`.

        ``maximum_window`` is the pre-registered formation horizon, not a
        fitted cutoff.  Quantile knots adapt resolution inside that finite
        horizon while preserving the original bound on every response
        footprint.  This prevents a single observation-horizon outlier from
        turning Q100 into an effectively global co-occurrence rule.

        The second return value records every quantile label that collapsed
        to each distinct integer window.  Window zero is represented by an
        empty label tuple because it is the exact same-tick identity rather
        than an empirical positive-span quantile.
        """
        if len(antecedent) == 1:
            return (0,), {0: ()}
        quantiles = tuple(float(value) for value in quantiles)
        if (
            not quantiles
            or quantiles != tuple(sorted(set(quantiles)))
            or any(value <= 0.0 or value > 1.0 for value in quantiles)
        ):
            raise ValueError("invalid productive-span quantiles")
        observed_ticks = int(np.max(context.ends - context.starts, initial=0))
        observed_window = int(math.ceil(observed_ticks / self.dataset.ticks_per_unit))
        horizon = (
            observed_window
            if maximum_window is None
            else min(observed_window, int(maximum_window))
        )
        if horizon < 0:
            raise ValueError("maximum formation window must be nonnegative")
        unit_windows = np.arange(horizon + 1, dtype=np.int64)
        sources = [self._source(predicate, context) for predicate in antecedent]
        counts = completion_window_counts(
            sources,
            context.ends,
            unit_windows * self.dataset.ticks_per_unit,
            relation=relation,
        )
        if counts is None:
            entities, times, spans = self.completions(context, antecedent, relation)
            productive = times < context.ends[entities]
            cap_ticks = horizon * self.dataset.ticks_per_unit
            positive = np.sort(spans[productive & (spans > 0) & (spans <= cap_ticks)])
            zero_count = int(np.count_nonzero(productive & (spans == 0)))
            if not len(positive):
                return ((0,), {0: ()}) if zero_count else ((), {})
            selected: dict[int, list[float]] = {0: []} if zero_count else {}
            for quantile in quantiles:
                rank = max(1, int(math.ceil(quantile * len(positive))))
                ticks = int(positive[rank - 1])
                window = int(math.ceil(ticks / self.dataset.ticks_per_unit))
                selected.setdefault(window, []).append(quantile)
            return (
                tuple(sorted(selected)),
                {window: tuple(labels) for window, labels in sorted(selected.items())},
            )
        zero_count = int(counts[0])
        positive_total = int(counts[-1]) - zero_count
        if positive_total <= 0:
            return ((0,), {0: ()}) if zero_count else ((), {})
        positive_cumulative = counts - zero_count
        selected = {0: []} if zero_count else {}
        for quantile in quantiles:
            rank = max(1, int(math.ceil(quantile * positive_total)))
            window = int(np.searchsorted(positive_cumulative, rank, side="left"))
            selected.setdefault(window, []).append(quantile)
        return (
            tuple(sorted(selected)),
            {window: tuple(labels) for window, labels in sorted(selected.items())},
        )

    def _retain_completions(
        self,
        key: tuple[int, str, Antecedent],
        result: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = sum(array.nbytes for array in result)
        if size > self._completion_cache_limit:
            return result
        with self._lock:
            existing = self._completion_cache.get(key)
            if existing is not None:
                self._completion_cache.move_to_end(key)
                return existing
            self._completion_cache[key] = result
            self._completion_cache_size += size
            while self._completion_cache_size > self._completion_cache_limit:
                # Higher-order completions are private to one skeleton, while
                # a lower-order stream is shared by many hierarchy closures.
                # Evict a triplet first, otherwise use ordinary deterministic
                # LRU order.  This is cache scheduling only; the returned
                # arrays and every fitted objective remain identical.
                victim = next(
                    (
                        candidate
                        for candidate in self._completion_cache
                        if len(candidate[2]) >= 3
                    ),
                    next(iter(self._completion_cache)),
                )
                removed = self._completion_cache.pop(victim)
                self._completion_cache_size -= sum(array.nbytes for array in removed)
        return result

    def _baseline_totals(self, context: Context) -> np.ndarray:
        key = id(context)
        with self._lock:
            cached = self._baseline_cache.get(key)
            if cached is not None:
                return cached
        # Every row touched by a control response is kept in ``active_rows``.
        # Remaining rows share an intercept-only design *within* each
        # pre-registered entity stratum and can be aggregated exactly by that
        # stratum.
        # Reuse the per-entity cell table required later by certification and
        # diagnostics.  Building temporal interval intersections twice was
        # pure preprocessing overhead; the weighted reduction is exact.
        counts = self.entity_age_counts(context)
        totals = context.entity_weights @ counts
        with self._lock:
            self._baseline_cache[key] = totals
        return totals

    def entity_age_counts(self, context: Context) -> np.ndarray:
        key = id(context)
        with self._lock:
            cached = self._entity_age_cache.get(key)
            if cached is not None:
                return cached
        counts = context.temporal_baseline_counts(time_bins=self.baseline_time_bins)
        if not np.allclose(counts.sum(axis=1), context.entity_exposure_totals()):
            raise AssertionError(
                "entity baseline counts do not cover observation intervals"
            )
        with self._lock:
            self._entity_age_cache[key] = counts
        return counts

    def _continuous_ranges(
        self,
        context: Context,
        entities: np.ndarray,
        times: np.ndarray,
        *,
        left_offset: int,
        right_offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact irregular-row ranges and the retained-input mask."""
        if context.row_times is None:
            raise ValueError("continuous ranges require irregular risk intervals")
        entities = np.asarray(entities, dtype=np.int32)
        times = np.asarray(times, dtype=np.int64)
        starts = times + np.int64(1 + int(left_offset))
        rights = times + np.int64(1 + int(right_offset))
        terminal = context.ends[entities] + 1
        starts = np.minimum(starts, terminal)
        rights = np.minimum(rights, terminal)
        keep = rights > starts
        if not np.any(keep):
            empty = np.zeros(0, dtype=np.int64)
            return (
                empty,
                empty.copy(),
                np.ascontiguousarray(keep, dtype=np.bool_),
            )
        selected_entities = entities[keep]
        left = context.interval_positions(
            selected_entities,
            starts[keep],
            allow_right_endpoint=True,
        )
        right = context.interval_positions(
            selected_entities,
            rights[keep],
            allow_right_endpoint=True,
        )
        return left, right, np.ascontiguousarray(keep, dtype=np.bool_)

    def _continuous_footprint_rows(
        self,
        context: Context,
        entities: np.ndarray,
        times: np.ndarray,
        *,
        horizon_ticks: int,
    ) -> np.ndarray:
        left, right, _ = self._continuous_ranges(
            context,
            entities,
            times,
            left_offset=0,
            right_offset=int(horizon_ticks),
        )
        if not len(left):
            return np.zeros(0, dtype=np.int64)
        order = np.argsort(left, kind="stable")
        left, right = left[order], right[order]
        merged_left: list[int] = []
        merged_right: list[int] = []
        current_left, current_right = int(left[0]), int(right[0])
        for next_left, next_right in zip(
            left[1:].tolist(), right[1:].tolist(), strict=True
        ):
            if int(next_left) <= current_right:
                current_right = max(current_right, int(next_right))
                continue
            merged_left.append(current_left)
            merged_right.append(current_right)
            current_left, current_right = int(next_left), int(next_right)
        merged_left.append(current_left)
        merged_right.append(current_right)
        return np.ascontiguousarray(
            np.concatenate(
                [
                    np.arange(start, stop, dtype=np.int64)
                    for start, stop in zip(merged_left, merged_right, strict=True)
                ]
            ),
            dtype=np.int64,
        )

    def _continuous_block_from_completions(
        self,
        context: Context,
        entities: np.ndarray,
        times: np.ndarray,
        spans: np.ndarray,
        *,
        window_ticks: int,
    ) -> SparseBlock:
        admitted = np.asarray(spans, dtype=np.int64) <= int(window_ticks)
        entities = np.asarray(entities, dtype=np.int32)[admitted]
        times = np.asarray(times, dtype=np.int64)[admitted]
        if not len(times):
            return SparseBlock(
                np.zeros(0, dtype=np.int64),
                np.zeros((0, self.knot_count), dtype=np.float64),
            )
        rows = self._continuous_footprint_rows(
            context,
            entities,
            times,
            horizon_ticks=self.lag,
        )
        if not len(rows):
            return SparseBlock(
                rows,
                np.zeros((0, self.knot_count), dtype=np.float64),
            )
        values = np.zeros((len(rows), self.knot_count), dtype=np.float64)
        for index in range(self.knot_count):
            left_edge = int(self.continuous_edges[index])
            right_edge = int(self.continuous_edges[index + 1])
            left, right, _ = self._continuous_ranges(
                context,
                entities,
                times,
                left_offset=left_edge,
                right_offset=right_edge,
            )
            if not len(left):
                continue
            difference = np.zeros(len(rows) + 1, dtype=np.float64)
            range_left = np.searchsorted(rows, left)
            range_right = np.searchsorted(rows, right)
            np.add.at(difference, range_left, 1.0)
            np.add.at(difference, range_right, -1.0)
            np.cumsum(difference, out=difference)
            width_seconds = (right_edge - left_edge) / self.dataset.ticks_per_unit
            values[:, index] = difference[:-1] / width_seconds
        nonzero = np.any(values != 0.0, axis=1)
        return SparseBlock(
            np.ascontiguousarray(rows[nonzero], dtype=np.int64),
            np.ascontiguousarray(values[nonzero], dtype=np.float64),
        )

    def _continuous_response_thresholds(
        self,
        context: Context,
        entities: np.ndarray,
        times: np.ndarray,
        spans: np.ndarray,
        *,
        maximum_span: int,
        horizon_ticks: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        admitted = np.asarray(spans, dtype=np.int64) <= int(maximum_span)
        entities = np.asarray(entities, dtype=np.int32)[admitted]
        times = np.asarray(times, dtype=np.int64)[admitted]
        spans = np.asarray(spans, dtype=np.int64)[admitted]
        left, right, retained = self._continuous_ranges(
            context,
            entities,
            times,
            left_offset=0,
            right_offset=int(horizon_ticks),
        )
        if not len(left):
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.copy()
        # Completions at an entity's terminal time have no strictly future
        # interval. _continuous_ranges removes them, so apply its exact same
        # mask to the completion spans before sorting.
        spans = spans[retained]
        if not (left.shape == right.shape == spans.shape):
            raise AssertionError("continuous response ranges lost input alignment")
        order = np.lexsort((spans, left))
        left, right, spans = left[order], right[order], spans[order]
        heap: list[tuple[int, int, int]] = []
        row_parts: list[np.ndarray] = []
        span_parts: list[np.ndarray] = []
        cursor = 0
        position = int(left[0])
        while cursor < len(left) or heap:
            while cursor < len(left) and int(left[cursor]) <= position:
                heapq.heappush(
                    heap,
                    (int(spans[cursor]), int(right[cursor]), int(cursor)),
                )
                cursor += 1
            while heap and heap[0][1] <= position:
                heapq.heappop(heap)
            if not heap:
                if cursor >= len(left):
                    break
                position = int(left[cursor])
                continue
            next_start = int(left[cursor]) if cursor < len(left) else context.n_grid
            next_position = min(next_start, int(heap[0][1]))
            if next_position <= position:
                position += 1
                continue
            row_parts.append(np.arange(position, next_position, dtype=np.int64))
            span_parts.append(
                np.full(next_position - position, int(heap[0][0]), dtype=np.int64)
            )
            position = next_position
        return (
            np.ascontiguousarray(
                np.concatenate(row_parts) if row_parts else np.zeros(0, dtype=np.int64),
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                np.concatenate(span_parts)
                if span_parts
                else np.zeros(0, dtype=np.int64),
                dtype=np.int64,
            ),
        )

    def block(
        self,
        context: Context,
        antecedent: Antecedent,
        window: int,
        relation: str = "unordered",
    ) -> SparseBlock:
        relation = "atomic" if len(antecedent) == 1 else str(relation)
        key = (id(context), relation, antecedent, int(window))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        with self._compute_lock("block", key):
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached
            # The single-W path used to expand ``completion x impact_lag``
            # rows and sort/reduce them.  Route, certification and control
            # callers frequently request a single block, so that path became
            # the dominant 30-day allocation.  Reuse the exact nested-W
            # accumulator even for one identity: it builds the union footprint
            # once and performs entity-local range accumulation in native code.
            return next(
                self.iter_blocks_many(
                    context,
                    antecedent,
                    (int(window),),
                    relation=relation,
                    retain=True,
                )
            )[1]

    def _block_from_completion_band(
        self,
        context: Context,
        entities: np.ndarray,
        times: np.ndarray,
        spans: np.ndarray,
        *,
        lower_ticks: int,
        upper_ticks: int,
    ) -> SparseBlock:
        """Build one exact ``lower < span <= upper`` response block."""

        admitted = (spans > int(lower_ticks)) & (spans <= int(upper_ticks))
        entities = entities[admitted]
        times = times[admitted]
        spans = spans[admitted]
        if self.continuous:
            return self._continuous_block_from_completions(
                context,
                entities,
                times,
                spans,
                window_ticks=int(upper_ticks),
            )
        contributions = kernel_contributions(
            entities,
            times,
            spans,
            context.starts,
            context.ends,
            context.offsets,
            self.basis,
            int(upper_ticks),
        )
        if contributions is None:
            rows_list: list[int] = []
            values_list: list[np.ndarray] = []
            for entity, completion in zip(
                entities.tolist(), times.tolist(), strict=True
            ):
                base = int(
                    context.offsets[entity] + completion - context.starts[entity]
                )
                remaining = min(self.lag, int(context.ends[entity] - completion))
                for lag in range(1, remaining + 1):
                    rows_list.append(base + lag)
                    values_list.append(self.basis[:, lag - 1])
            raw_rows = np.asarray(rows_list, dtype=np.int64)
            raw_values = (
                np.asarray(values_list, dtype=np.float64)
                if values_list
                else np.zeros((0, self.knot_count), dtype=np.float64)
            )
        else:
            raw_rows, raw_values = contributions
        if not len(raw_rows):
            return SparseBlock(
                np.zeros(0, dtype=np.int64),
                np.zeros((0, self.knot_count), dtype=np.float64),
            )
        order = np.argsort(raw_rows, kind="stable")
        sorted_rows = np.asarray(raw_rows[order], dtype=np.int64)
        sorted_values = np.asarray(raw_values[order], dtype=np.float64)
        rows, first = np.unique(sorted_rows, return_index=True)
        values = np.add.reduceat(sorted_values, first, axis=0)
        return SparseBlock(
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(values, dtype=np.float64),
        )

    def rule_block(self, context: Context, rule: RuleIdentity) -> SparseBlock:
        """Return the exact response block for one finite rule identity."""
        if not rule.history_marks:
            return self.block(context, rule.antecedent, rule.window, rule.relation)
        key = (
            id(context),
            "history-rule",
            rule.relation,
            rule.antecedent,
            int(rule.window),
            rule.history_marks,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        with self._compute_lock("history-rule-block", key):
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached
            entities, times, spans = self.rule_completions(context, rule)
            window_ticks = int(rule.window) * self.dataset.ticks_per_unit
            lower = self.window_band_lower(rule.antecedent, rule.window, rule.relation)
            if lower is not None:
                result = self._block_from_completion_band(
                    context,
                    entities,
                    times,
                    spans,
                    lower_ticks=int(lower) * self.dataset.ticks_per_unit,
                    upper_ticks=window_ticks,
                )
                return self._retain_block(key, result)
            if self.continuous:
                result = self._continuous_block_from_completions(
                    context,
                    entities,
                    times,
                    spans,
                    window_ticks=window_ticks,
                )
                return self._retain_block(key, result)
            admitted = spans <= window_ticks
            entities = entities[admitted]
            times = times[admitted]
            spans = spans[admitted]
            contributions = kernel_contributions(
                entities,
                times,
                spans,
                context.starts,
                context.ends,
                context.offsets,
                self.basis,
                window_ticks,
            )
            if contributions is None:
                rows_list: list[int] = []
                values_list: list[np.ndarray] = []
                for entity, completion in zip(
                    entities.tolist(), times.tolist(), strict=True
                ):
                    base = int(
                        context.offsets[entity] + completion - context.starts[entity]
                    )
                    remaining = min(self.lag, int(context.ends[entity] - completion))
                    for lag in range(1, remaining + 1):
                        rows_list.append(base + lag)
                        values_list.append(self.basis[:, lag - 1])
                raw_rows = np.asarray(rows_list, dtype=np.int64)
                raw_values = (
                    np.asarray(values_list, dtype=np.float64)
                    if values_list
                    else np.zeros((0, self.knot_count), dtype=np.float64)
                )
            else:
                raw_rows, raw_values = contributions
            if not len(raw_rows):
                result = SparseBlock(
                    np.zeros(0, dtype=np.int64),
                    np.zeros((0, self.knot_count), dtype=np.float64),
                )
            else:
                order = np.argsort(raw_rows, kind="stable")
                sorted_rows = np.asarray(raw_rows[order], dtype=np.int64)
                sorted_values = np.asarray(raw_values[order], dtype=np.float64)
                rows, first = np.unique(sorted_rows, return_index=True)
                values = np.add.reduceat(sorted_values, first, axis=0)
                result = SparseBlock(
                    np.ascontiguousarray(rows, dtype=np.int64),
                    np.ascontiguousarray(values, dtype=np.float64),
                )
            return self._retain_block(key, result)

    def closure_block(self, context: Context, term: ClosureTerm) -> SparseBlock:
        """Return a hierarchy nuisance block with the same marked semantics."""
        return self.rule_block(
            context,
            RuleIdentity(
                term.antecedent,
                term.window,
                1,
                relation=("atomic" if len(term.antecedent) == 1 else "unordered"),
                history_marks=term.history_marks,
            ),
        )

    @staticmethod
    def _strict_subset(left: Antecedent, right: Antecedent) -> bool:
        return len(left) < len(right) and set(left).issubset(right)

    @classmethod
    def _more_specific(cls, lower: RuleIdentity, higher: RuleIdentity) -> bool:
        """Whether ``higher`` owns a more specific total-state footprint."""
        if cls._strict_subset(lower.antecedent, higher.antecedent):
            return True
        return (
            lower.relation == "unordered"
            and higher.relation == "ordered"
            and set(lower.antecedent) == set(higher.antecedent)
        )

    def total_state_geometry_changed(
        self,
        source: Support,
        target: Support,
    ) -> bool:
        """Whether common columns change under total-state specificity.

        A higher-order active state masks every selected strict lower-order
        state.  Adding, removing, or changing the W of a higher-order rule can
        therefore change an already present lower-order column.  The check is
        structural and coefficient independent; a sign-only replacement does
        not alter the mask.
        """

        source_by_pattern = {rule.pattern_key: rule for rule in source.rules}
        target_by_pattern = {rule.pattern_key: rule for rule in target.rules}
        if any(
            source_by_pattern[key].response_geometry
            != target_by_pattern[key].response_geometry
            for key in set(source_by_pattern).intersection(target_by_pattern)
        ):
            return True

        # Support-relative additive rules never mask or rewrite a retained
        # rule block.  A retained identity geometry change was handled above;
        # an ordinary Add or Drop is therefore an exact column append/remove.
        if self.effect_model == "support_additive" or any(
            rule.support_additive for rule in (*source.rules, *target.rules)
        ):
            return False

        if any(rule.hierarchical for rule in (*source.rules, *target.rules)):
            source_closure = set(hierarchy_closure(source))
            target_closure = set(hierarchy_closure(target))
            source_ranks = {
                (rule.pattern_key, int(rule.window)): int(rule.kernel_rank)
                for rule in source.rules
            }
            target_ranks = {
                (rule.pattern_key, int(rule.window)): int(rule.kernel_rank)
                for rule in target.rules
            }
            return bool(
                # Additive hierarchy does not mask an existing lower-order
                # response when a higher-order modifier is added.  A monotone
                # closure extension therefore only appends/repositions
                # columns and is handled exactly by the incremental builder.
                # Geometry truly changes only when an old closure disappears
                # (usually because it is promoted to a reported rule), or a
                # retained block changes width.  Treating every unequal
                # closure set as a state splice forced a full child rebuild
                # and exact DAG fit for every high-order proposal.
                not source_closure.issubset(target_closure)
                or any(
                    source_ranks[key] != target_ranks[key]
                    for key in set(source_ranks).intersection(target_ranks)
                )
            )

        # Sign-only replacements have identical response geometry, so compare
        # identities by antecedent/window rather than by the signed atom.
        source_geometry = {
            (rule.pattern_key, int(rule.window), int(rule.kernel_rank)): rule
            for rule in source.rules
        }
        target_geometry = {
            (rule.pattern_key, int(rule.window), int(rule.kernel_rank)): rule
            for rule in target.rules
        }
        source_ranks = {
            (rule.pattern_key, int(rule.window)): int(rule.kernel_rank)
            for rule in source.rules
        }
        target_ranks = {
            (rule.pattern_key, int(rule.window)): int(rule.kernel_rank)
            for rule in target.rules
        }
        if any(
            source_ranks[key] != target_ranks[key]
            for key in set(source_ranks).intersection(target_ranks)
        ):
            return True
        common_keys = set(source_geometry).intersection(target_geometry)

        def dominators(rule: RuleIdentity, support: Support) -> frozenset[tuple]:
            return frozenset(
                (candidate.pattern_key, int(candidate.window))
                for candidate in support.rules
                if self._more_specific(rule, candidate)
            )

        if any(
            dominators(source_geometry[key], source)
            != dominators(target_geometry[key], target)
            for key in common_keys
        ):
            return True

        # A newly added lower-order state may itself require masking even when
        # none of the parent's existing columns changes (AB -> A+AB).  Raw
        # block Add algebra is invalid in that case and must fail open to the
        # sparse state-splice builder.
        for key in set(target_geometry).difference(source_geometry):
            rule = target_geometry[key]
            if any(self._more_specific(rule, other) for other in target.rules):
                return True
        return False

    def rule_design_values(self, block: SparseBlock, rule: RuleIdentity) -> np.ndarray:
        """Return the exact design for a rule's declared kernel family.

        ``kernel_rank=1`` is the normalized constant mixture of the configured
        triangular basis.  It is a genuine one-parameter submodel of the full
        M-knot cone, so scalar/full MDL scores are directly comparable.
        """
        if rule.kernel_rank == 0:
            return block.values
        return np.mean(block.values, axis=1, keepdims=True)

    def rule_slices(
        self, support: Support, start: int
    ) -> tuple[tuple[slice, ...], int]:
        slices: list[slice] = []
        left = int(start)
        for rule in support.rules:
            right = left + rule.kernel_dimension(self.knot_count)
            slices.append(slice(left, right))
            left = right
        return tuple(slices), left

    def total_state_rule_blocks(
        self,
        context: Context,
        support: Support,
    ) -> tuple[SparseBlock, ...]:
        """Return directly interpretable rule blocks with nested-state masking.

        Raw completion/kernel blocks remain support independent and cached.
        For a selected lower-order rule, rows covered by any selected strict
        superset are removed.  Thus ``A + AB`` means ``A-only`` plus the
        complete ``AB`` state, rather than an interaction coefficient added on
        top of ``A``.  Incomparable rules remain ordinary additive components;
        only strict subset/superset states are made mutually exclusive.
        """

        raw = tuple(self.rule_block(context, rule) for rule in support.rules)
        if (
            self.effect_model in {"additive_hierarchy", "support_additive"}
            or any(rule.hierarchical for rule in support.rules)
            or any(rule.support_additive for rule in support.rules)
        ):
            return raw
        output: list[SparseBlock] = []
        for index, (rule, block) in enumerate(zip(support.rules, raw, strict=True)):
            dominator_rows = [
                raw[other_index].rows
                for other_index, other in enumerate(support.rules)
                if self._more_specific(rule, other) and len(raw[other_index].rows)
            ]
            if not dominator_rows or not len(block.rows):
                output.append(block)
                continue
            shadow = sorted_unique_union(dominator_rows)
            if shadow is None:
                shadow = np.unique(np.concatenate(dominator_rows))
            positions = np.searchsorted(shadow, block.rows)
            matched = positions < len(shadow)
            if len(shadow):
                safe = np.minimum(positions, len(shadow) - 1)
                matched &= shadow[safe] == block.rows
            keep = ~matched
            if np.all(keep):
                output.append(block)
            else:
                output.append(
                    SparseBlock(
                        np.ascontiguousarray(block.rows[keep], dtype=np.int64),
                        np.ascontiguousarray(block.values[keep], dtype=np.float64),
                    )
                )
        return tuple(output)

    def total_state_added_block(
        self,
        context: Context,
        support: Support,
        added_rule: RuleIdentity,
    ) -> SparseBlock:
        """Return one target block with the same total-state masking.

        Conditional state-splice pricing needs only the newly added block.
        Rebuilding and masking every retained block for each W/sign identity
        is algebraically redundant.
        """

        if added_rule not in support.rules:
            raise ValueError("added rule is absent from target support")
        block = self.rule_block(context, added_rule)
        if (
            self.effect_model in {"additive_hierarchy", "support_additive"}
            or added_rule.hierarchical
            or added_rule.support_additive
        ):
            return block
        return self.mask_total_state_added_block(
            context,
            support,
            added_rule,
            block,
        )

    def mask_total_state_added_block(
        self,
        context: Context,
        support: Support,
        added_rule: RuleIdentity,
        block: SparseBlock,
    ) -> SparseBlock:
        """Apply exact total-state masking to an already generated Add block.

        Terminal audits generate all nested W snapshots of one antecedent in a
        single recurrence.  Reusing those immutable snapshots here avoids
        rebuilding the same completion stream once per W; the returned block is
        exactly the one produced by :meth:`total_state_added_block`.
        """

        if added_rule not in support.rules:
            raise ValueError("added rule is absent from target support")
        if not len(block.rows):
            return block
        dominator_rows = [
            self.rule_block(context, other).rows
            for other in support.rules
            if self._more_specific(added_rule, other)
        ]
        dominator_rows = [rows for rows in dominator_rows if len(rows)]
        if not dominator_rows:
            return block
        shadow = sorted_unique_union(dominator_rows)
        if shadow is None:
            shadow = np.unique(np.concatenate(dominator_rows))
        positions = np.searchsorted(shadow, block.rows)
        matched = positions < len(shadow)
        if len(shadow):
            safe = np.minimum(positions, len(shadow) - 1)
            matched &= shadow[safe] == block.rows
        keep = ~matched
        if np.all(keep):
            return block
        return SparseBlock(
            np.ascontiguousarray(block.rows[keep], dtype=np.int64),
            np.ascontiguousarray(block.values[keep], dtype=np.float64),
        )

    def control_block(self, context: Context, predicate: int) -> SparseBlock:
        """Return a strictly-future M-knot history block for one control."""
        # A control is the W=0 singleton response with a non-reportable role;
        # its geometry is otherwise identical.  Sharing the canonical block
        # eliminates a second completion expansion and cache entry.
        return self.block(context, (int(predicate),), 0, "atomic")

    def _retain_block(self, key: tuple, result: SparseBlock) -> SparseBlock:
        if result.nbytes > self._block_cache_limit:
            return result
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = result
            self._cache_size += result.nbytes
            while self._cache_size > self._block_cache_limit and len(self._cache) > 1:
                _, removed = self._cache.popitem(last=False)
                self._cache_size -= removed.nbytes
        return result

    def blocks_many(
        self,
        context: Context,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        *,
        relation: str = "unordered",
        retain: bool = True,
    ) -> dict[int, SparseBlock]:
        """Materialize every nested W block after one pass over completions.

        Discovery pricing must use :meth:`iter_blocks_many` instead: collecting
        a dense Freddie W family here intentionally keeps every snapshot alive
        and can require tens of GiB.  This collecting API remains for small
        callers and exact parity tests.
        """
        return dict(
            self.iter_blocks_many(
                context,
                antecedent,
                windows,
                relation=relation,
                retain=retain,
            )
        )

    def iter_blocks_many(
        self,
        context: Context,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        *,
        relation: str = "unordered",
        retain: bool = False,
        worker_count: int = 0,
    ) -> Iterator[tuple[int, SparseBlock]]:
        """Yield exact nested W blocks while retaining only one snapshot.

        Newly admitted completions are accumulated exactly once.  A compact
        global-row lookup maps them into the maximum-W footprint, after which
        each requested block is a deterministic snapshot of the accumulator.
        The caller consumes that snapshot before advancing the iterator, so
        memory is ``O(max-W footprint)`` rather than the sum of all W
        footprints.  Accumulation order and returned arrays are identical to
        :meth:`blocks_many`; only their lifetime differs.
        """
        requested = tuple(sorted(set(map(int, windows))))
        if not requested:
            return
        relation = "atomic" if len(antecedent) == 1 else str(relation)
        keys = {
            window: (id(context), relation, antecedent, window) for window in requested
        }
        with self._lock:
            cached = {
                window: self._cache[key]
                for window, key in keys.items()
                if key in self._cache
            }
            for window in cached:
                self._cache.move_to_end(keys[window])
        if len(cached) == len(requested):
            for window in requested:
                yield window, cached[window]
            return
        band_lowers = {
            window: self.window_band_lower(antecedent, window, relation)
            for window in requested
        }
        if any(lower is not None for lower in band_lowers.values()):
            if any(lower is None for lower in band_lowers.values()):
                raise ValueError("one pattern cannot mix cumulative and band windows")
            entities, times, spans = self.completions(context, antecedent, relation)
            for window in requested:
                result = cached.get(window)
                if result is None:
                    result = self._block_from_completion_band(
                        context,
                        entities,
                        times,
                        spans,
                        lower_ticks=(
                            int(band_lowers[window]) * self.dataset.ticks_per_unit
                        ),
                        upper_ticks=window * self.dataset.ticks_per_unit,
                    )
                    if retain:
                        result = self._retain_block(keys[window], result)
                yield window, result
            return
        if self.continuous:
            entities, times, spans = self.completions(context, antecedent, relation)
            for window in requested:
                result = cached.get(window)
                if result is None:
                    result = self._continuous_block_from_completions(
                        context,
                        entities,
                        times,
                        spans,
                        window_ticks=window * self.dataset.ticks_per_unit,
                    )
                    if retain:
                        result = self._retain_block(keys[window], result)
                yield window, result
            return
        group_key = (id(context), relation, antecedent, retain, *requested)
        with self._compute_lock("blocks-many", group_key):
            with self._lock:
                cached = {
                    window: self._cache[key]
                    for window, key in keys.items()
                    if key in self._cache
                }
                for window in cached:
                    self._cache.move_to_end(keys[window])
            if len(cached) == len(requested):
                for window in requested:
                    yield window, cached[window]
                return
            maximum_window = max(requested)
            threshold_rows, minimum_spans = self.response_row_thresholds(
                context, antecedent, maximum_window, relation=relation
            )
            if not len(threshold_rows):
                empty = SparseBlock(
                    np.zeros(0, dtype=np.int64),
                    np.zeros((0, self.knot_count), dtype=np.float64),
                )
                for window in requested:
                    result = cached.get(window)
                    if result is None:
                        result = (
                            self._retain_block(keys[window], empty) if retain else empty
                        )
                    yield window, result
                return
            # A dense global-row lookup is the fastest exact representation on
            # ordinary grids.  On fine continuous-time grids it can dwarf the
            # actual response footprint, so use exact sorted-row lookup there.
            # int32 is sufficient because the accumulator itself cannot have
            # more than 2^31 rows in memory.
            dense_lookup = context.n_grid <= max(64_000_000, 8 * len(threshold_rows))
            lookup: np.ndarray | None
            if dense_lookup:
                lookup = np.full(context.n_grid, -1, dtype=np.int32)
                lookup[threshold_rows] = np.arange(len(threshold_rows), dtype=np.int32)
            else:
                lookup = None
            accumulator = np.zeros(
                (len(threshold_rows), self.knot_count), dtype=np.float64
            )
            entities, times, spans = self.completions(context, antecedent, relation)
            maximum_ticks = maximum_window * self.dataset.ticks_per_unit
            order = bounded_span_order(spans, maximum_ticks)
            if order is None:
                admitted = spans <= maximum_ticks
                indices = np.flatnonzero(admitted)
                order = indices[np.argsort(spans[indices], kind="stable")]
            entities, times, spans = entities[order], times[order], spans[order]
            left = 0
            for window in requested:
                window_ticks = window * self.dataset.ticks_per_unit
                right = int(np.searchsorted(spans, window_ticks, side="right"))
                if right > left:
                    compiled = False
                    if lookup is not None:
                        compiled = accumulate_kernel(
                            entities[left:right],
                            times[left:right],
                            context.starts,
                            context.ends,
                            context.offsets,
                            self.basis,
                            lookup,
                            accumulator,
                            worker_count=worker_count,
                        )
                    if not compiled:
                        contributions = kernel_contributions(
                            entities[left:right],
                            times[left:right],
                            spans[left:right],
                            context.starts,
                            context.ends,
                            context.offsets,
                            self.basis,
                            window_ticks,
                        )
                        if contributions is None:
                            for entity, completion in zip(
                                entities[left:right].tolist(),
                                times[left:right].tolist(),
                                strict=True,
                            ):
                                base = int(
                                    context.offsets[entity]
                                    + completion
                                    - context.starts[entity]
                                )
                                remaining = min(
                                    self.lag,
                                    int(context.ends[entity] - completion),
                                )
                                for lag in range(1, remaining + 1):
                                    row = base + lag
                                    position = (
                                        int(lookup[row])
                                        if lookup is not None
                                        else int(np.searchsorted(threshold_rows, row))
                                    )
                                    if (
                                        position >= len(threshold_rows)
                                        or threshold_rows[position] != row
                                    ):
                                        raise AssertionError(
                                            "kernel row missing from response footprint"
                                        )
                                    accumulator[position] += self.basis[:, lag - 1]
                        else:
                            raw_rows, raw_values = contributions
                            if lookup is not None:
                                positions = lookup[raw_rows].astype(
                                    np.int64, copy=False
                                )
                            else:
                                positions = np.searchsorted(threshold_rows, raw_rows)
                                if np.any(
                                    positions >= len(threshold_rows)
                                ) or not np.array_equal(
                                    threshold_rows[positions], raw_rows
                                ):
                                    raise AssertionError(
                                        "kernel rows are outside the response footprint"
                                    )
                            np.add.at(accumulator, positions, raw_values)
                left = right
                cached_result = cached.get(window)
                if cached_result is not None:
                    # Accumulation must still advance for later W values, but
                    # an exact retained snapshot avoids another potentially
                    # multi-GiB boolean gather and copy at this W.
                    yield window, cached_result
                    continue
                footprint = minimum_spans <= window_ticks
                rows = threshold_rows[footprint]
                values = accumulator[footprint].copy()
                result = SparseBlock(rows, values)
                result = self._retain_block(keys[window], result) if retain else result
                yield window, result

    def model_matrix(
        self,
        context: Context,
        support: Support,
        *,
        forced_closure: tuple[ClosureTerm, ...] | None = None,
        _allow_extension: bool = True,
        _aggregate_rows: bool = True,
    ) -> ModelMatrix:
        if any(
            predicate >= self.dataset.n_reported_predicates
            for rule in support.rules
            for predicate in rule.antecedent
        ):
            raise ValueError("baseline-control predicates cannot enter a support")
        closure = (
            hierarchy_closure(support)
            if forced_closure is None
            else tuple(sorted(forced_closure))
        )
        if closure and self.effect_model != "additive_hierarchy":
            raise ValueError("total-state models do not admit hidden closure blocks")
        baseline_key = id(context)
        with self._lock:
            baseline_matrix = (
                self._baseline_model_cache.get(baseline_key)
                if _aggregate_rows
                else None
            )
        if baseline_matrix is not None:
            if support == Support(()) and not closure:
                return baseline_matrix
            if (
                _allow_extension
                and all(rule.kernel_rank == 0 for rule in support.rules)
                and not self.total_state_geometry_changed(
                    baseline_matrix.support, support
                )
            ):
                return self.extend_model_matrix(
                    context,
                    support,
                    baseline_matrix,
                    forced_closure=closure,
                )
        closure_blocks = [self.closure_block(context, term) for term in closure]
        closure_signs = tuple(self.closure_sign(term) for term in closure)
        rule_blocks = list(self.total_state_rule_blocks(context, support))
        control_blocks = [
            self.control_block(context, predicate)
            for predicate in self.control_predicates
        ]
        all_blocks = [*control_blocks, *closure_blocks, *rule_blocks]
        active_rows = [block.rows for block in all_blocks if len(block.rows)]
        if len(context.target_rows):
            active_rows.append(context.target_rows)
        union = sorted_unique_union(active_rows)
        if union is None:
            union = (
                np.unique(np.concatenate(active_rows))
                if active_rows
                else np.zeros(0, dtype=np.int64)
            )
        baseline_dim = self.baseline_dimension
        rule_start = baseline_dim + self.knot_count * len(closure_blocks)
        rule_slices, dimension = self.rule_slices(support, rule_start)
        x_active = np.zeros((len(union), dimension), dtype=np.float64)
        baseline_groups = context.temporal_baseline_groups_at_rows(
            union, time_bins=self.baseline_time_bins
        )
        x_active[np.arange(len(union)), baseline_groups] = 1.0
        for control_index, (block, sign) in enumerate(
            zip(control_blocks, self.control_signs, strict=True)
        ):
            if not len(block.rows):
                continue
            positions = np.searchsorted(union, block.rows)
            left = self.free_baseline_dimension + control_index * self.knot_count
            x_active[positions, left : left + self.knot_count] = sign * block.values
        for block_index, block in enumerate(closure_blocks):
            if not len(block.rows):
                continue
            positions = np.searchsorted(union, block.rows)
            sign = float(closure_signs[block_index])
            left = baseline_dim + block_index * self.knot_count
            x_active[positions, left : left + self.knot_count] = sign * block.values
        for rule, block, destination in zip(
            support.rules, rule_blocks, rule_slices, strict=True
        ):
            if not len(block.rows):
                continue
            positions = np.searchsorted(union, block.rows)
            x_active[positions, destination] = float(
                rule.sign
            ) * self.rule_design_values(block, rule)
        matrix = self._finalize_model_matrix(
            context,
            support,
            closure,
            closure_signs,
            union,
            baseline_groups,
            x_active,
            aggregate_rows=_aggregate_rows,
        )
        if _aggregate_rows and support == Support(()) and not closure:
            with self._lock:
                incumbent = self._baseline_model_cache.setdefault(baseline_key, matrix)
            return incumbent
        return matrix

    def model_metadata(
        self,
        support: Support,
        *,
        forced_closure: tuple[ClosureTerm, ...] | None = None,
    ) -> ModelMatrix:
        """Return the exact block layout without constructing observation rows.

        Frozen validation and prediction use coefficients estimated on another
        split.  They need the support, closure directions and column layout,
        but not a dense sufficient-statistic matrix.  Keeping that distinction
        explicit prevents D_cert/D_test from rebuilding a multi-GiB design
        merely to evaluate fixed coefficients.
        """
        if any(
            predicate >= self.dataset.n_reported_predicates
            for rule in support.rules
            for predicate in rule.antecedent
        ):
            raise ValueError("baseline-control predicates cannot enter a support")
        closure = (
            hierarchy_closure(support)
            if forced_closure is None
            else tuple(sorted(forced_closure))
        )
        if closure and self.effect_model != "additive_hierarchy":
            raise ValueError("total-state models do not admit hidden closure blocks")
        baseline_dimension = self.baseline_dimension
        closure_dimension = len(closure) * self.knot_count
        rule_start = baseline_dimension + closure_dimension
        rule_slices, dimension = self.rule_slices(support, rule_start)
        empty_float = np.zeros((0, dimension), dtype=np.float64)
        empty_weight = np.zeros(0, dtype=np.float64)
        empty_int64 = np.zeros(0, dtype=np.int64)
        empty_int32 = np.zeros(0, dtype=np.int32)
        return ModelMatrix(
            x=empty_float,
            exposure_weight=empty_weight,
            noevent_weight=empty_weight,
            event_weight=empty_weight,
            free_dimension=self.free_baseline_dimension,
            closure_dimension=closure_dimension,
            rule_slices=rule_slices,
            support=support,
            closure=closure,
            closure_signs=tuple(self.closure_sign(term) for term in closure),
            active_rows=empty_int64,
            active_design_groups=empty_int32,
            active_baseline_groups=empty_int32,
            aggregate_baseline_groups=empty_int32,
            control_dimension=baseline_dimension - self.free_baseline_dimension,
        )

    def _frozen_blocks(
        self,
        context: Context,
        matrix: ModelMatrix,
    ) -> tuple[tuple[SparseBlock, float, slice], ...]:
        """Resolve the sparse signed blocks of one frozen model."""
        if matrix.baseline_dimension != self.baseline_dimension:
            raise ValueError("matrix baseline-control dimension mismatch")
        blocks: list[tuple[SparseBlock, float, slice]] = []
        for index, (predicate, sign) in enumerate(
            zip(self.control_predicates, self.control_signs, strict=True)
        ):
            left = self.free_baseline_dimension + index * self.knot_count
            blocks.append(
                (
                    self.control_block(context, predicate),
                    float(sign),
                    slice(left, left + self.knot_count),
                )
            )
        closure_left = matrix.baseline_dimension
        for index, (term, sign) in enumerate(
            zip(matrix.closure, matrix.closure_signs, strict=True)
        ):
            left = closure_left + index * self.knot_count
            blocks.append(
                (
                    self.closure_block(context, term),
                    float(sign),
                    slice(left, left + self.knot_count),
                )
            )
        for rule, block, coefficient_slice in zip(
            matrix.support.rules,
            self.total_state_rule_blocks(context, matrix.support),
            matrix.rule_slices,
            strict=True,
        ):
            blocks.append(
                (
                    SparseBlock(
                        block.rows,
                        np.ascontiguousarray(self.rule_design_values(block, rule)),
                    ),
                    float(rule.sign),
                    coefficient_slice,
                )
            )
        return tuple(blocks)

    def frozen_linear_predictor_at_rows(
        self,
        context: Context,
        matrix: ModelMatrix,
        coefficients: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a frozen model directly from sparse response blocks.

        This is algebraically identical to ``model_matrix(...).x @ beta`` on
        the requested rows.  It avoids constructing zero columns and is used
        only after the coefficient vector and closure directions are frozen.
        """
        coefficients = np.asarray(coefficients, dtype=np.float64)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if coefficients.shape != (matrix.dimension,):
            raise ValueError("coefficient dimension mismatch")
        if rows.ndim != 1 or (
            len(rows) and (int(rows[0]) < 0 or int(rows[-1]) >= context.n_grid)
        ):
            raise ValueError("requested rows are outside the context")
        if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
            raise ValueError("requested rows must be sorted and unique")
        eta = coefficients[
            context.temporal_baseline_groups_at_rows(
                rows, time_bins=self.baseline_time_bins
            )
        ].astype(np.float64, copy=True)
        if not len(rows):
            return eta
        for block, sign, coefficient_slice in self._frozen_blocks(context, matrix):
            if not len(block.rows):
                continue
            positions = np.searchsorted(block.rows, rows)
            matched = positions < len(block.rows)
            safe = np.minimum(positions, len(block.rows) - 1)
            matched &= block.rows[safe] == rows
            if np.any(matched):
                eta[matched] += sign * (
                    block.values[positions[matched]] @ coefficients[coefficient_slice]
                )
        return eta

    def frozen_contextual_rule_contribution_at_rows(
        self,
        context: Context,
        matrix: ModelMatrix,
        coefficients: np.ndarray,
        *,
        rule_index: int,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Return one reported rule's direct nested-state-masked contribution."""

        coefficients = np.asarray(coefficients, dtype=np.float64)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if coefficients.shape != (matrix.dimension,):
            raise ValueError("coefficient dimension mismatch")
        if not 0 <= rule_index < len(matrix.support.rules):
            raise IndexError("reported-rule index is outside the frozen support")
        if rows.ndim != 1 or (
            len(rows) and (int(rows[0]) < 0 or int(rows[-1]) >= context.n_grid)
        ):
            raise ValueError("requested rows are outside the context")
        if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
            raise ValueError("requested rows must be sorted and unique")
        output = np.zeros(len(rows), dtype=np.float64)
        if not len(rows):
            return output

        root = matrix.support.rules[rule_index]
        block = self.total_state_rule_blocks(context, matrix.support)[rule_index]
        if len(block.rows):
            positions = np.searchsorted(block.rows, rows)
            matched = positions < len(block.rows)
            safe = np.minimum(positions, len(block.rows) - 1)
            matched &= block.rows[safe] == rows
            if np.any(matched):
                output[matched] = float(root.sign) * (
                    self.rule_design_values(block, root)[positions[matched]]
                    @ coefficients[matrix.rule_slices[rule_index]]
                )
        return output

    def frozen_hierarchical_total_contribution_at_rows(
        self,
        context: Context,
        matrix: ModelMatrix,
        coefficients: np.ndarray,
        *,
        rule_index: int,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Return a modifier together with all of its lower-order main effects."""

        root = matrix.support.rules[rule_index]
        direct = self.frozen_contextual_rule_contribution_at_rows(
            context,
            matrix,
            coefficients,
            rule_index=rule_index,
            rows=rows,
        )
        if not root.hierarchical:
            return direct
        output = direct.copy()
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        coefficients = np.asarray(coefficients, dtype=np.float64)

        def add_block(
            block: SparseBlock,
            values: np.ndarray,
            sign: int,
            destination: slice,
        ) -> None:
            if not len(block.rows):
                return
            positions = np.searchsorted(block.rows, rows)
            matched = positions < len(block.rows)
            safe = np.minimum(positions, len(block.rows) - 1)
            matched &= block.rows[safe] == rows
            if np.any(matched):
                output[matched] += float(sign) * (
                    values[positions[matched]] @ coefficients[destination]
                )

        closure_left = matrix.baseline_dimension
        for index, (term, sign) in enumerate(
            zip(matrix.closure, matrix.closure_signs, strict=True)
        ):
            if not self._strict_subset(term.antecedent, root.antecedent):
                continue
            destination = slice(
                closure_left + index * self.knot_count,
                closure_left + (index + 1) * self.knot_count,
            )
            block = self.closure_block(context, term)
            add_block(block, block.values, int(sign), destination)
        raw_rules = self.total_state_rule_blocks(context, matrix.support)
        for index, (rule, block, destination) in enumerate(
            zip(
                matrix.support.rules,
                raw_rules,
                matrix.rule_slices,
                strict=True,
            )
        ):
            if index == rule_index or not self._strict_subset(
                rule.antecedent, root.antecedent
            ):
                continue
            add_block(
                block,
                self.rule_design_values(block, rule),
                int(rule.sign),
                destination,
            )
        return output

    def frozen_active_predictor(
        self,
        context: Context,
        matrix: ModelMatrix,
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return every non-intercept/target row and its frozen predictor."""
        blocks = self._frozen_blocks(context, matrix)
        active = [context.target_rows]
        active.extend(block.rows for block, _, _ in blocks if len(block.rows))
        rows = sorted_unique_union(active)
        if rows is None:
            rows = (
                np.unique(np.concatenate(active))
                if active
                else np.zeros(0, dtype=np.int64)
            )
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (matrix.dimension,):
            raise ValueError("coefficient dimension mismatch")
        eta = coefficients[
            context.temporal_baseline_groups_at_rows(
                rows, time_bins=self.baseline_time_bins
            )
        ].astype(np.float64, copy=True)
        for block, sign, coefficient_slice in blocks:
            if not len(block.rows):
                continue
            positions = np.searchsorted(rows, block.rows)
            if np.any(positions >= len(rows)) or not np.array_equal(
                rows[positions], block.rows
            ):
                raise AssertionError("frozen response row missing from active union")
            eta[positions] += sign * (block.values @ coefficients[coefficient_slice])
        return rows, eta

    def extend_model_matrix(
        self,
        context: Context,
        support: Support,
        source: ModelMatrix,
        *,
        forced_closure: tuple[ClosureTerm, ...] | None = None,
    ) -> ModelMatrix:
        """Losslessly append rules/closure to a monotone support matrix."""
        if any(
            predicate >= self.dataset.n_reported_predicates
            for rule in support.rules
            for predicate in rule.antecedent
        ):
            raise ValueError("baseline-control predicates cannot enter a support")
        # A compact state-splice matrix is an exact likelihood object for one
        # transient proposal, but deliberately omits row-to-group provenance.
        # It must not become the parent of another incremental extension:
        # doing so loses target-only rows from the active union.  Rebuilding
        # this uncommon transition is exact and preserves the compact fast
        # path for its intended one-step use.
        if len(source.active_rows) == 0 and (source.support.rules or source.closure):
            return self.model_matrix(context, support, forced_closure=forced_closure)
        closure = (
            hierarchy_closure(support)
            if forced_closure is None
            else tuple(sorted(forced_closure))
        )
        if any(
            rule.kernel_rank != 0 for rule in (*source.support.rules, *support.rules)
        ):
            # Variable-width representation changes are rare terminal audits.
            # Rebuilding is exact and avoids routing the fixed-M incremental
            # builder through an invalid column layout.
            return self.model_matrix(context, support, forced_closure=closure)
        if self.total_state_geometry_changed(source.support, support):
            return self.model_matrix(context, support, forced_closure=closure)
        if not set(source.support.rules).issubset(support.rules) or not set(
            source.closure
        ).issubset(closure):
            return self.model_matrix(context, support, forced_closure=forced_closure)
        new_closure = tuple(term for term in closure if term not in source.closure)
        new_rules = tuple(
            rule for rule in support.rules if rule not in source.support.rules
        )
        specifications = [
            (
                term.antecedent,
                term.window,
                float(self.closure_sign(term)),
                "closure",
                term,
            )
            for term in new_closure
        ] + [
            (rule.antecedent, rule.window, float(rule.sign), "rule", rule)
            for rule in new_rules
        ]
        effective_rules = {
            rule: block
            for rule, block in zip(
                support.rules,
                self.total_state_rule_blocks(context, support),
                strict=True,
            )
        }
        blocks = [
            (
                effective_rules[identity]
                if kind == "rule"
                else self.block(context, antecedent, window)
            )
            for antecedent, window, _, kind, identity in specifications
        ]
        active_parts = [source.active_rows]
        active_parts.extend(block.rows for block in blocks if len(block.rows))
        union = sorted_unique_union(active_parts)
        if union is None:
            union = np.unique(np.concatenate(active_parts))
        baseline_dim = source.baseline_dimension
        dimension = baseline_dim + self.knot_count * (len(closure) + len(support.rules))
        closure_index = {term: index for index, term in enumerate(closure)}
        new_rule_start = baseline_dim + len(closure) * self.knot_count
        rule_index = {rule: index for index, rule in enumerate(support.rules)}
        baseline_groups = context.temporal_baseline_groups_at_rows(
            union, time_bins=self.baseline_time_bins
        )
        return self._finalize_extended_model_matrix(
            context=context,
            support=support,
            closure=closure,
            source=source,
            union=union,
            baseline_groups=baseline_groups,
            specifications=specifications,
            blocks=blocks,
            closure_index=closure_index,
            rule_index=rule_index,
            dimension=dimension,
            baseline_dim=baseline_dim,
            new_rule_start=new_rule_start,
        )

    def standalone_window_family_matrix(
        self,
        context: Context,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        source: ModelMatrix,
        relation: str = "auto",
    ) -> ModelMatrix:
        """Build one exact shared design for every W of one antecedent.

        A standalone W/sign profile used to build and aggregate an almost
        identical observation matrix for every identity.  Here every distinct
        unsigned W block is appended to the same already-fitted baseline
        matrix once.  A caller obtains an individual signed rule by projecting
        onto the baseline columns and one W block, multiplying that block by
        ``+1`` or ``-1``.

        Keeping extra W columns in the source only refines its sufficient-
        statistic partition.  Projecting them away therefore gives exactly
        the same weighted likelihood as a separately materialized one-rule
        matrix; it is not a screening approximation.
        """
        relation = normalize_relation(antecedent, relation)
        if source.support.rules or source.closure:
            raise ValueError("standalone window families require the baseline source")
        return self.nested_add_window_family_matrix(
            context, antecedent, windows, source, relation=relation
        )

    def hierarchical_standalone_family_matrix(
        self,
        context: Context,
        supports: tuple[Support, ...],
        source: ModelMatrix,
    ) -> tuple[ModelMatrix, dict[Support, tuple[tuple[int, int], ...]]]:
        """Build one exact projected family for additive standalone identities.

        A hierarchical pair/triplet contains its fixed lower-order nuisance
        blocks in addition to the reported block.  The ordinary standalone
        window family contains only the latter and is therefore not a valid
        projection for an additive model.  This family stores every distinct
        unsigned nuisance/rule block once.  A candidate projection selects
        its closure blocks followed by its reported block and applies the
        exact frozen signs, reproducing the canonical model column-for-column.

        Extra family columns only refine sufficient-statistic row groups and
        disappear in the projection.  Consequently this is an execution
        accelerator, not an approximation to the likelihood or hierarchy.
        """

        ordered_supports = tuple(dict.fromkeys(supports))
        if not ordered_supports:
            raise ValueError("a hierarchical standalone family must be nonempty")
        if source.support.rules or source.closure:
            raise ValueError("hierarchical standalone families require baseline")
        if any(len(support.rules) != 1 for support in ordered_supports):
            raise ValueError("hierarchical standalone families require one rule")
        if any(support.rules[0].kernel_rank != 0 for support in ordered_supports):
            raise ValueError("hierarchical family requires full-rank kernels")

        block_indices: dict[tuple[object, ...], int] = {}
        blocks: list[SparseBlock] = []
        projections: dict[Support, tuple[tuple[int, int], ...]] = {}

        def retain(key: tuple[object, ...], block: SparseBlock) -> int:
            index = block_indices.get(key)
            if index is None:
                index = len(blocks)
                block_indices[key] = index
                blocks.append(block)
            return index

        for support in ordered_supports:
            rule = support.rules[0]
            selected: list[tuple[int, int]] = []
            for term in hierarchy_closure(support):
                selected.append(
                    (
                        retain(
                            ("closure", term),
                            self.closure_block(context, term),
                        ),
                        int(self.closure_sign(term)),
                    )
                )
            raw = self.total_state_rule_blocks(context, support)[0]
            selected.append(
                (
                    retain(
                        ("rule", rule.pattern_key, int(rule.window)),
                        SparseBlock(
                            raw.rows,
                            np.ascontiguousarray(self.rule_design_values(raw, rule)),
                        ),
                    ),
                    int(rule.sign),
                )
            )
            projections[support] = tuple(selected)

        # Artificial identities describe temporary unsigned column slots.
        # They are internal to this matrix and cannot collide with a real
        # reportable predicate.
        fake_base = int(self.dataset.n_predicates) + 2_000_000
        fake_rules = tuple(
            RuleIdentity((fake_base + index,), 0, 1) for index in range(len(blocks))
        )
        family_support = Support.of(fake_rules)
        specifications = [
            (rule.antecedent, 0, 1.0, "rule", rule) for rule in fake_rules
        ]
        active_parts = [source.active_rows]
        active_parts.extend(block.rows for block in blocks if len(block.rows))
        union = sorted_unique_union(active_parts)
        if union is None:
            union = np.unique(np.concatenate(active_parts))
        baseline_dim = source.baseline_dimension
        family = self._finalize_extended_model_matrix(
            context=context,
            support=family_support,
            closure=(),
            source=source,
            union=union,
            baseline_groups=context.temporal_baseline_groups_at_rows(
                union, time_bins=self.baseline_time_bins
            ),
            specifications=specifications,
            blocks=blocks,
            closure_index={},
            rule_index={rule: index for index, rule in enumerate(fake_rules)},
            dimension=baseline_dim + len(blocks) * self.knot_count,
            baseline_dim=baseline_dim,
            new_rule_start=baseline_dim,
        )
        return family, projections

    def representation_lattice_family_matrix(
        self,
        context: Context,
        supports: tuple[Support, ...],
        source: ModelMatrix,
    ) -> tuple[ModelMatrix, dict[Support, tuple[int, ...]]]:
        """Build one exact sufficient-statistic design for a support lattice.

        Every candidate keeps its ordinary total-state semantics.  A temporary
        column block is created for each distinct ``(rule, dominator set)``;
        this distinguishes, for example, ``A`` from the ``A-only`` block in
        ``A+AB``.  Selecting the baseline columns and the mapped temporary
        blocks therefore reproduces the candidate's materialized design
        exactly.  Extra temporary columns only refine the row partition and
        disappear under projection, so likelihood values are unchanged.
        """

        ordered_supports = tuple(dict.fromkeys(supports))
        if not ordered_supports:
            raise ValueError("a representation family must be nonempty")
        if source.support.rules or source.closure:
            raise ValueError("representation families require the baseline source")

        block_indices: dict[tuple[object, ...], int] = {}
        blocks: list[SparseBlock] = []
        projection_blocks: dict[Support, tuple[int, ...]] = {}
        for support in ordered_supports:
            effective = self.total_state_rule_blocks(context, support)
            indices: list[int] = []
            for rule, block in zip(support.rules, effective, strict=True):
                dominators = tuple(
                    (other.pattern_key, int(other.window))
                    for other in support.rules
                    if self._more_specific(rule, other)
                )
                key = (
                    rule.pattern_key,
                    int(rule.window),
                    dominators,
                )
                index = block_indices.get(key)
                if index is None:
                    index = len(blocks)
                    block_indices[key] = index
                    blocks.append(block)
                indices.append(index)
            projection_blocks[support] = tuple(indices)

        # Artificial identities describe only temporary column layout.  They
        # are never exposed as rules and deliberately use predicate IDs beyond
        # the dataset dictionary so they cannot collide with a real support.
        fake_base = int(self.dataset.n_predicates) + 1_000_000
        fake_rules = tuple(
            RuleIdentity((fake_base + index,), 0, 1) for index in range(len(blocks))
        )
        family_support = Support.of(fake_rules)
        specifications = [
            (rule.antecedent, 0, 1.0, "rule", rule) for rule in fake_rules
        ]
        active_parts = [source.active_rows]
        active_parts.extend(block.rows for block in blocks if len(block.rows))
        union = sorted_unique_union(active_parts)
        if union is None:
            union = np.unique(np.concatenate(active_parts))
        baseline_dim = source.baseline_dimension
        family = self._finalize_extended_model_matrix(
            context=context,
            support=family_support,
            closure=(),
            source=source,
            union=union,
            baseline_groups=context.temporal_baseline_groups_at_rows(
                union, time_bins=self.baseline_time_bins
            ),
            specifications=specifications,
            blocks=blocks,
            closure_index={},
            rule_index={rule: index for index, rule in enumerate(fake_rules)},
            dimension=baseline_dim + len(blocks) * self.knot_count,
            baseline_dim=baseline_dim,
            new_rule_start=baseline_dim,
        )
        return family, projection_blocks

    def nested_add_window_family_matrix(
        self,
        context: Context,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        source: ModelMatrix,
        relation: str = "auto",
    ) -> ModelMatrix:
        """Build a shared exact W design for a geometry-preserving Add.

        The source support columns are copied once and all unsigned W blocks
        are appended as temporary columns.  Selecting the source columns plus
        one signed W block is exactly the ordinary child model whenever adding
        the antecedent does not alter another total-state rule's response.
        Callers must check ``total_state_geometry_changed`` before using this
        accelerator; geometry-changing children use the canonical fail-open
        builder.
        """
        relation = normalize_relation(antecedent, relation)
        ordered_windows = tuple(sorted(set(map(int, windows))))
        if not ordered_windows:
            raise ValueError("a window family must be nonempty")
        if source.closure:
            raise ValueError("window families do not accept hidden source closure")
        if len(antecedent) == 1 and ordered_windows != (0,):
            raise ValueError("singleton window family must contain only W=0")
        representative = RuleIdentity(
            antecedent, ordered_windows[0], 1, relation=relation
        )
        if representative.pattern_key in source.support.patterns:
            raise ValueError("window-family Add temporal pattern is already active")
        if self.total_state_geometry_changed(
            source.support, source.support.add(representative)
        ):
            raise ValueError("window-family Add changes total-state geometry")
        # These terms describe only temporary unsigned column slots.  Their
        # antecedent is canonicalized because ClosureTerm predates temporal
        # relations; the actual response below retains ``relation``.
        closure_antecedent = tuple(sorted(antecedent))
        terms = tuple(
            ClosureTerm(closure_antecedent, window) for window in ordered_windows
        )
        specifications = [
            (antecedent, window, 1.0, "closure", term)
            for window, term in zip(ordered_windows, terms, strict=True)
        ]
        blocks = [
            self.block(context, antecedent, window, relation)
            for window in ordered_windows
        ]
        active_parts = [source.active_rows]
        active_parts.extend(block.rows for block in blocks if len(block.rows))
        union = sorted_unique_union(active_parts)
        if union is None:
            union = np.unique(np.concatenate(active_parts))
        baseline_dim = source.baseline_dimension
        dimension = baseline_dim + len(terms) * self.knot_count
        matrix = self._finalize_extended_model_matrix(
            context=context,
            support=source.support,
            closure=terms,
            source=source,
            union=union,
            baseline_groups=context.temporal_baseline_groups_at_rows(
                union, time_bins=self.baseline_time_bins
            ),
            specifications=specifications,
            blocks=blocks,
            closure_index={term: index for index, term in enumerate(terms)},
            rule_index={rule: index for index, rule in enumerate(source.support.rules)},
            dimension=dimension + len(source.support.rules) * self.knot_count,
            baseline_dim=baseline_dim,
            new_rule_start=dimension,
        )
        # The artificial closure metadata is used only to describe the shared
        # column layout.  Its columns are unsigned; the projected solver
        # applies the requested rule sign explicitly.
        return ModelMatrix(
            x=matrix.x,
            exposure_weight=matrix.exposure_weight,
            noevent_weight=matrix.noevent_weight,
            event_weight=matrix.event_weight,
            free_dimension=matrix.free_dimension,
            closure_dimension=matrix.closure_dimension,
            rule_slices=matrix.rule_slices,
            support=source.support,
            closure=terms,
            closure_signs=(1,) * len(terms),
            active_rows=matrix.active_rows,
            active_design_groups=matrix.active_design_groups,
            active_baseline_groups=matrix.active_baseline_groups,
            aggregate_baseline_groups=matrix.aggregate_baseline_groups,
            control_dimension=matrix.control_dimension,
        )

    def identity_replacement_family_matrix(
        self,
        context: Context,
        identities: tuple[RuleIdentity, ...],
        source: ModelMatrix,
    ) -> tuple[ModelMatrix, dict[RuleIdentity, int]]:
        """Build one exact shared design for replacement identities.

        ``source`` is the exact current support. Its signed rule columns are
        retained once. Every unsigned replacement response is appended as a
        temporary block. A caller projects away the replaced source block and
        selects one signed temporary block, reproducing the canonical child
        likelihood exactly. Extra temporary columns only refine sufficient-
        statistic groups and therefore cannot change a fitted optimum.
        """
        ordered = tuple(dict.fromkeys(identities))
        if not ordered:
            raise ValueError("an identity replacement family must be nonempty")
        if source.closure:
            raise ValueError("replacement families do not accept hidden closure")
        if self.effect_model != "support_additive":
            raise ValueError("replacement family requires additive support effects")
        if any(rule.kernel_rank != 0 for rule in ordered):
            raise ValueError("replacement family requires full-rank kernels")

        geometry_indices: dict[tuple[object, ...], int] = {}
        blocks: list[SparseBlock] = []
        identity_blocks: dict[RuleIdentity, int] = {}
        for rule in ordered:
            key = (
                rule.pattern_key,
                int(rule.window),
                rule.history_marks,
                bool(rule.support_additive),
            )
            block_index = geometry_indices.get(key)
            if block_index is None:
                raw = self.rule_block(context, rule)
                block_index = len(blocks)
                geometry_indices[key] = block_index
                blocks.append(
                    SparseBlock(
                        raw.rows,
                        np.ascontiguousarray(
                            self.rule_design_values(raw, rule), dtype=np.float64
                        ),
                    )
                )
            identity_blocks[rule] = block_index

        fake_base = int(self.dataset.n_predicates) + 3_000_000
        terms = tuple(
            ClosureTerm((fake_base + index,), 0) for index in range(len(blocks))
        )
        specifications = [(term.antecedent, 0, 1.0, "closure", term) for term in terms]
        active_parts = [source.active_rows]
        active_parts.extend(block.rows for block in blocks if len(block.rows))
        union = sorted_unique_union(active_parts)
        if union is None:
            union = np.unique(np.concatenate(active_parts))
        baseline_dim = source.baseline_dimension
        temporary_dimension = len(terms) * self.knot_count
        matrix = self._finalize_extended_model_matrix(
            context=context,
            support=source.support,
            closure=terms,
            source=source,
            union=union,
            baseline_groups=context.temporal_baseline_groups_at_rows(
                union, time_bins=self.baseline_time_bins
            ),
            specifications=specifications,
            blocks=blocks,
            closure_index={term: index for index, term in enumerate(terms)},
            rule_index={rule: index for index, rule in enumerate(source.support.rules)},
            dimension=(
                baseline_dim
                + temporary_dimension
                + len(source.support.rules) * self.knot_count
            ),
            baseline_dim=baseline_dim,
            new_rule_start=baseline_dim + temporary_dimension,
        )
        family = ModelMatrix(
            x=matrix.x,
            exposure_weight=matrix.exposure_weight,
            noevent_weight=matrix.noevent_weight,
            event_weight=matrix.event_weight,
            free_dimension=matrix.free_dimension,
            closure_dimension=matrix.closure_dimension,
            rule_slices=matrix.rule_slices,
            support=source.support,
            closure=terms,
            closure_signs=(1,) * len(terms),
            active_rows=matrix.active_rows,
            active_design_groups=matrix.active_design_groups,
            active_baseline_groups=matrix.active_baseline_groups,
            aggregate_baseline_groups=matrix.aggregate_baseline_groups,
            control_dimension=matrix.control_dimension,
        )
        return family, identity_blocks

    def splice_total_state_add(
        self,
        context: Context,
        support: Support,
        source: ModelMatrix,
        added_rule: RuleIdentity,
        *,
        include_active_metadata: bool = True,
    ) -> ModelMatrix:
        """Losslessly add one rule whose total-state mask changes geometry.

        Only the effective response rows of ``added_rule`` can change: a new
        higher-order state zeros selected strict lower-order columns there,
        while a new lower-order state is already stripped of selected
        superset rows.  This builder subtracts those rows from the source
        sufficient statistics, reconstructs just the touched designs, and
        reuses every untouched source group.  It is algebraically identical to
        :meth:`model_matrix` but avoids rebuilding the full observation union.
        """

        if self.effect_model in {"additive_hierarchy", "support_additive"}:
            return self.model_matrix(context, support)
        if source.closure or hierarchy_closure(support):
            raise ValueError("total-state splice does not admit hidden closure")
        new_rules = set(support.rules).difference(source.support.rules)
        if (
            new_rules != {added_rule}
            or not set(source.support.rules).issubset(support.rules)
            or not self.total_state_geometry_changed(source.support, support)
        ):
            return self.model_matrix(context, support, forced_closure=())

        target_blocks = self.total_state_rule_blocks(context, support)
        added_index = support.rules.index(added_rule)
        added_block = target_blocks[added_index]
        touched = added_block.rows
        if not len(touched):
            return self.model_matrix(context, support, forced_closure=())

        source_group_count = len(source.x)
        positions = np.searchsorted(source.active_rows, touched)
        matched = positions < len(source.active_rows)
        if len(source.active_rows):
            safe = np.minimum(positions, len(source.active_rows) - 1)
            matched &= source.active_rows[safe] == touched

        # Rows outside active_rows have the intercept-only design and contain
        # no target.  The largest identical intercept-only sufficient-
        # statistic row owns their aggregate mass.  Assigning new untouched
        # no-event rows to any identical design row is algebraically exact,
        # and this O(number of design groups) lookup avoids scanning millions
        # of source active rows for every candidate.
        intercept_only = np.flatnonzero(
            np.all(np.abs(source.x[:, 1:]) <= 1.0e-14, axis=1)
        )
        outside_group = (
            int(intercept_only[np.argmax(source.exposure_weight[intercept_only])])
            if len(intercept_only)
            else -1
        )
        scale = max(1.0, float(np.max(source.exposure_weight, initial=0.0)))
        tolerance = 1.0e-9 * scale
        if np.any(~matched) and outside_group < 0:
            return self.model_matrix(context, support, forced_closure=())
        parent_groups = np.full(
            len(touched),
            outside_group if outside_group >= 0 else 0,
            dtype=np.int64,
        )
        if np.any(matched):
            parent_groups[matched] = source.active_design_groups[positions[matched]]

        baseline_dimension = source.baseline_dimension
        dimension = baseline_dimension + len(support.rules) * self.knot_count
        target_slices = {
            rule: slice(
                baseline_dimension + index * self.knot_count,
                baseline_dimension + (index + 1) * self.knot_count,
            )
            for index, rule in enumerate(support.rules)
        }
        source_slices = {
            rule: block
            for rule, block in zip(
                source.support.rules, source.rule_slices, strict=True
            )
        }

        def remap(
            source_design: np.ndarray,
            rows: np.ndarray | None = None,
        ) -> np.ndarray:
            """Remap parent columns without a whole-row advanced-index copy.

            ``source.x[rows]`` creates a second dense parent-sized temporary
            before the target matrix is allocated.  A large total-state Add
            therefore held parent, gathered parent and target simultaneously.
            Gather one column block at a time directly into the target and
            initialize only the genuinely new rule slice.
            """

            count = len(source_design) if rows is None else len(rows)
            output = np.empty((count, dimension), dtype=np.float64)
            if rows is None:
                output[:, :baseline_dimension] = source_design[:, :baseline_dimension]
            else:
                output[:, :baseline_dimension] = source_design[
                    rows, :baseline_dimension
                ]
            for rule, source_slice in source_slices.items():
                if rows is None:
                    output[:, target_slices[rule]] = source_design[:, source_slice]
                else:
                    output[:, target_slices[rule]] = source_design[rows, source_slice]
            output[:, target_slices[added_rule]] = 0.0
            return output

        touched_design = remap(source.x, parent_groups)
        for rule in source.support.rules:
            if self._more_specific(rule, added_rule):
                touched_design[:, target_slices[rule]] = 0.0
        touched_design[:, target_slices[added_rule]] = (
            float(added_rule.sign) * added_block.values
        )

        row_weights = context.weights_at_rows(touched)
        touched_exposure = self.tick_exposure * row_weights
        touched_event = context.target_counts_at_sorted_rows(touched)
        touched_noevent = (
            touched_exposure - touched_event
            if context.dataset.likelihood == "first_event_cloglog"
            else touched_exposure.copy()
        )
        removed_exposure = np.bincount(
            parent_groups,
            weights=touched_exposure,
            minlength=source_group_count,
        )
        removed_noevent = np.bincount(
            parent_groups,
            weights=touched_noevent,
            minlength=source_group_count,
        )
        removed_event = np.bincount(
            parent_groups,
            weights=touched_event,
            minlength=source_group_count,
        )
        residual_exposure = source.exposure_weight - removed_exposure
        residual_noevent = source.noevent_weight - removed_noevent
        residual_event = source.event_weight - removed_event
        if (
            np.min(residual_exposure, initial=0.0) < -tolerance
            or np.min(residual_noevent, initial=0.0) < -tolerance
            or np.min(residual_event, initial=0.0) < -tolerance
        ):
            return self.model_matrix(context, support, forced_closure=())
        residual_exposure = np.maximum(residual_exposure, 0.0)
        residual_noevent = np.maximum(residual_noevent, 0.0)
        residual_event = np.maximum(residual_event, 0.0)
        retained = (
            (residual_exposure > tolerance)
            | (residual_noevent > tolerance)
            | (residual_event > tolerance)
        )
        residual_map = np.full(source_group_count, -1, dtype=np.int64)
        residual_map[retained] = np.arange(np.count_nonzero(retained), dtype=np.int64)
        residual_source_groups = np.flatnonzero(retained)
        residual_design = remap(source.x, residual_source_groups)

        if include_active_metadata:
            (
                touched_design,
                touched_exposure,
                touched_noevent,
                touched_event,
                touched_groups,
            ) = self._aggregate_or_keep_design_rows(
                touched_design,
                touched_exposure,
                touched_noevent,
                touched_event,
            )
        else:
            # Conditional route validation consumes this matrix only for a
            # one-step likelihood/Fisher calculation.  Duplicate rows with
            # their original weights are an exact sufficient-statistic
            # representation, while hashing and copying the touched design
            # dominated state-splice construction.  Terminal/exact records
            # still request active metadata and retain canonical aggregation.
            touched_groups = np.arange(len(touched_design), dtype=np.int64)
        residual_count = len(residual_design)
        x = np.concatenate([residual_design, touched_design], axis=0)
        exposure_weight = np.concatenate(
            [residual_exposure[retained], touched_exposure]
        )
        noevent_weight = np.concatenate([residual_noevent[retained], touched_noevent])
        event_weight = np.concatenate([residual_event[retained], touched_event])

        if include_active_metadata:
            union = sorted_unique_union([source.active_rows, touched])
            if union is None:
                union = np.union1d(source.active_rows, touched)
            active_design_groups = np.full(len(union), -1, dtype=np.int64)
            if len(source.active_rows):
                source_union_positions = np.searchsorted(union, source.active_rows)
                touched_positions = np.searchsorted(touched, source.active_rows)
                source_touched = touched_positions < len(touched)
                safe = np.minimum(touched_positions, len(touched) - 1)
                source_touched &= touched[safe] == source.active_rows
                untouched = ~source_touched
                inherited = residual_map[source.active_design_groups[untouched]]
                if np.any(inherited < 0):
                    return self.model_matrix(context, support, forced_closure=())
                active_design_groups[source_union_positions[untouched]] = inherited
            touched_union_positions = np.searchsorted(union, touched)
            active_design_groups[touched_union_positions] = (
                residual_count + touched_groups
            )
            if np.any(active_design_groups < 0):
                return self.model_matrix(context, support, forced_closure=())

            active_baseline_groups = context.temporal_baseline_groups_at_rows(
                union, time_bins=self.baseline_time_bins
            )
            active_by_group = np.bincount(
                active_baseline_groups,
                weights=context.weights_at_rows(union),
                minlength=self.free_baseline_dimension,
            ).astype(np.float64)
            inactive_by_group = self._baseline_totals(context) - active_by_group
            if np.any(inactive_by_group < -tolerance):
                return self.model_matrix(context, support, forced_closure=())
            aggregate_baseline_groups = np.flatnonzero(
                inactive_by_group > tolerance
            ).astype(np.int64)
        else:
            union = np.zeros(0, dtype=np.int64)
            active_design_groups = np.zeros(0, dtype=np.int32)
            active_baseline_groups = np.zeros(0, dtype=np.int32)
            aggregate_baseline_groups = np.zeros(0, dtype=np.int64)
        return ModelMatrix(
            x=np.ascontiguousarray(x),
            exposure_weight=np.ascontiguousarray(exposure_weight),
            noevent_weight=np.ascontiguousarray(noevent_weight),
            event_weight=np.ascontiguousarray(event_weight),
            free_dimension=source.free_dimension,
            closure_dimension=0,
            rule_slices=tuple(target_slices[rule] for rule in support.rules),
            support=support,
            closure=(),
            closure_signs=(),
            active_rows=np.ascontiguousarray(union, dtype=np.int64),
            active_design_groups=np.ascontiguousarray(
                active_design_groups, dtype=np.int32
            ),
            active_baseline_groups=active_baseline_groups,
            aggregate_baseline_groups=aggregate_baseline_groups,
            control_dimension=source.control_dimension,
        )

    def _finalize_extended_model_matrix(
        self,
        *,
        context: Context,
        support: Support,
        closure: tuple[ClosureTerm, ...],
        source: ModelMatrix,
        union: np.ndarray,
        baseline_groups: np.ndarray,
        specifications: list[tuple[Antecedent, int, float, str, object]],
        blocks: list[SparseBlock],
        closure_index: dict[ClosureTerm, int],
        rule_index: dict[RuleIdentity, int],
        dimension: int,
        baseline_dim: int,
        new_rule_start: int,
    ) -> ModelMatrix:
        """Split existing sufficient-statistic groups only by new blocks.

        The touched-only path handles the ordinary monotone-add case without
        revisiting unchanged rows.  The original full-union implementation
        remains an exact fail-open path for unusual external SparseBlocks.
        """
        incremental = self._finalize_touched_extended_model_matrix(
            context=context,
            support=support,
            closure=closure,
            source=source,
            union=union,
            baseline_groups=baseline_groups,
            specifications=specifications,
            blocks=blocks,
            closure_index=closure_index,
            rule_index=rule_index,
            dimension=dimension,
            baseline_dim=baseline_dim,
            new_rule_start=new_rule_start,
        )
        if incremental is not None:
            return incremental
        active_count = len(union)
        total_by_group = self._baseline_totals(context)
        active_weights = context.weights_at_rows(union)
        active_by_group = np.bincount(
            baseline_groups,
            weights=active_weights,
            minlength=self.free_baseline_dimension,
        ).astype(np.float64)
        inactive_by_group = total_by_group - active_by_group
        aggregate_baseline_groups = np.flatnonzero(inactive_by_group > 0)

        positions = np.searchsorted(source.active_rows, union)
        matched = positions < len(source.active_rows)
        if len(source.active_rows):
            safe = np.minimum(positions, len(source.active_rows) - 1)
            matched &= source.active_rows[safe] == union
        source_group_count = source.x.shape[0]
        old_groups = source_group_count + baseline_groups.astype(np.int64)
        if np.any(matched):
            old_groups[matched] = source.active_design_groups[positions[matched]]

        new_width = len(specifications) * self.knot_count
        keys = np.zeros(
            (active_count + len(aggregate_baseline_groups), 1 + new_width),
            dtype=np.float64,
        )
        keys[:active_count, 0] = old_groups
        keys[active_count:, 0] = source_group_count + aggregate_baseline_groups
        for index, (block, specification) in enumerate(
            zip(blocks, specifications, strict=True)
        ):
            if not len(block.rows):
                continue
            sign = float(specification[2])
            destination = slice(
                1 + index * self.knot_count,
                1 + (index + 1) * self.knot_count,
            )
            block_positions = np.searchsorted(union, block.rows)
            keys[block_positions, destination] = sign * block.values

        target_position = np.searchsorted(union, context.target_rows)
        target_matched = target_position < len(union)
        if len(union):
            safe = np.minimum(target_position, len(union) - 1)
            target_matched &= union[safe] == context.target_rows
        if len(context.target_rows) and not np.all(target_matched):
            raise AssertionError("target row missing from extended active union")
        event_active = np.zeros(active_count, dtype=np.float64)
        if np.any(target_matched):
            # Context.make has already stably reduced duplicate target rows,
            # so these destinations are unique. Direct assignment is exactly
            # equivalent to add.at here and avoids a serialized scatter in
            # every fail-open matrix construction.
            event_active[target_position[target_matched]] = context.target_counts[
                target_matched
            ]
        event_weight = np.concatenate(
            [
                event_active,
                np.zeros(len(aggregate_baseline_groups), dtype=np.float64),
            ]
        )
        exposure_weight = self.tick_exposure * np.concatenate(
            [active_weights, inactive_by_group[aggregate_baseline_groups]]
        )
        noevent_weight = (
            exposure_weight - event_weight
            if context.dataset.likelihood == "first_event_cloglog"
            else exposure_weight.copy()
        )
        (
            grouped_keys,
            exposure_weight,
            noevent_weight,
            event_weight,
            design_groups,
        ) = self._aggregate_or_keep_design_rows(
            keys,
            exposure_weight,
            noevent_weight,
            event_weight,
        )

        group_codes = np.rint(grouped_keys[:, 0]).astype(np.int64)
        x = np.zeros((len(grouped_keys), dimension), dtype=np.float64)
        inherited = group_codes < source_group_count
        inherited_rows = np.flatnonzero(inherited)
        if len(inherited_rows):
            parent_rows = group_codes[inherited]
            x[inherited_rows, :baseline_dim] = source.x[parent_rows, :baseline_dim]
            for old_index, term in enumerate(source.closure):
                old_left = baseline_dim + old_index * self.knot_count
                new_left = baseline_dim + closure_index[term] * self.knot_count
                x[inherited_rows, new_left : new_left + self.knot_count] = source.x[
                    parent_rows, old_left : old_left + self.knot_count
                ]
            for rule, old_slice in zip(
                source.support.rules, source.rule_slices, strict=True
            ):
                new_left = new_rule_start + rule_index[rule] * self.knot_count
                x[inherited_rows, new_left : new_left + self.knot_count] = source.x[
                    parent_rows, old_slice
                ]
        baseline_rows = np.flatnonzero(~inherited)
        if len(baseline_rows):
            baseline_bins = group_codes[~inherited] - source_group_count
            if np.any(
                (baseline_bins < 0) | (baseline_bins >= self.free_baseline_dimension)
            ):
                raise AssertionError("unknown intercept-only baseline group")
            x[baseline_rows, baseline_bins] = 1.0
        for index, (_, _, _, kind, identity) in enumerate(specifications):
            source_slice = slice(
                1 + index * self.knot_count,
                1 + (index + 1) * self.knot_count,
            )
            if kind == "closure":
                left = baseline_dim + closure_index[identity] * self.knot_count
            else:
                left = new_rule_start + rule_index[identity] * self.knot_count
            x[:, left : left + self.knot_count] = grouped_keys[:, source_slice]

        rule_slices = tuple(
            slice(
                new_rule_start + index * self.knot_count,
                new_rule_start + (index + 1) * self.knot_count,
            )
            for index in range(len(support.rules))
        )
        return ModelMatrix(
            x=x,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event_weight,
            free_dimension=source.free_dimension,
            closure_dimension=len(closure) * self.knot_count,
            rule_slices=rule_slices,
            support=support,
            closure=closure,
            closure_signs=tuple(self.closure_sign(term) for term in closure),
            active_rows=union,
            active_design_groups=np.ascontiguousarray(
                design_groups[:active_count], dtype=np.int32
            ),
            active_baseline_groups=baseline_groups,
            aggregate_baseline_groups=aggregate_baseline_groups,
            control_dimension=source.control_dimension,
        )

    def _finalize_touched_extended_model_matrix(
        self,
        *,
        context: Context,
        support: Support,
        closure: tuple[ClosureTerm, ...],
        source: ModelMatrix,
        union: np.ndarray,
        baseline_groups: np.ndarray,
        specifications: list[tuple[Antecedent, int, float, str, object]],
        blocks: list[SparseBlock],
        closure_index: dict[ClosureTerm, int],
        rule_index: dict[RuleIdentity, int],
        dimension: int,
        baseline_dim: int,
        new_rule_start: int,
    ) -> ModelMatrix | None:
        """Exactly refine only parent groups touched by newly appended blocks.

        The parent ``ModelMatrix`` is already an exact sufficient-statistic
        aggregation.  Appending columns can split an old group, but it cannot
        change any row outside the new blocks' union.  Subtracting the touched
        row weights from their parent groups and hashing only those rows is
        therefore algebraically identical to rebuilding and hashing the full
        active design.  Returning ``None`` invokes the general exact path for
        an unexpected structural case.
        """
        block_rows = [block.rows for block in blocks if len(block.rows)]
        touched = sorted_unique_union(block_rows)
        if touched is None:
            touched = (
                np.unique(np.concatenate(block_rows))
                if block_rows
                else np.zeros(0, dtype=np.int64)
            )
        touched = np.ascontiguousarray(touched, dtype=np.int64)
        source_group_count = int(source.x.shape[0])

        # Every row not explicitly active in the parent has its intercept-only
        # parent design.  There must be one such exact aggregate group whenever
        # a newly touched row lies outside ``source.active_rows``.
        source_positions = np.searchsorted(source.active_rows, touched)
        source_matched = source_positions < len(source.active_rows)
        if len(source.active_rows):
            safe = np.minimum(source_positions, len(source.active_rows) - 1)
            source_matched &= source.active_rows[safe] == touched
        parent_groups = np.empty(len(touched), dtype=np.int64)
        if np.any(source_matched):
            parent_groups[source_matched] = source.active_design_groups[
                source_positions[source_matched]
            ]
        if np.any(~source_matched):
            missing_positions = np.flatnonzero(~source_matched)
            missing_strata = context.temporal_baseline_groups_at_rows(
                touched[missing_positions], time_bins=self.baseline_time_bins
            )
            for stratum in np.unique(missing_strata):
                expected = np.zeros(source.dimension, dtype=np.float64)
                expected[int(stratum)] = 1.0
                intercept_groups = np.flatnonzero(
                    np.all(source.x == expected[None, :], axis=1)
                )
                if not len(intercept_groups):
                    return None
                # An unaggregated parent can contain many numerically identical
                # intercept-only rows.  The first such row is commonly one
                # active observation with unit mass; the actual inactive
                # baseline aggregate is another identical row with all of the
                # remaining stratum mass.  Assigning every newly touched row
                # to the first match therefore over-subtracted that unit row,
                # forced the complete-union fallback, and rebuilt a multi-GiB
                # matrix for every Add candidate.  Identical designs are
                # interchangeable sufficient-statistic groups, so select the
                # group with the largest removable no-event mass.  The later
                # signed residual check remains the exact fail-open guard if a
                # nonstandard weighted context cannot be represented by one
                # such aggregate.
                removable_mass = np.minimum(
                    source.exposure_weight[intercept_groups],
                    source.noevent_weight[intercept_groups],
                )
                parent_groups[missing_positions[missing_strata == stratum]] = int(
                    intercept_groups[int(np.argmax(removable_mass))]
                )

        new_width = len(specifications) * self.knot_count
        touched_keys = np.zeros((len(touched), 1 + new_width), dtype=np.float64)
        if len(touched):
            touched_keys[:, 0] = parent_groups
        for index, (block, specification) in enumerate(
            zip(blocks, specifications, strict=True)
        ):
            if not len(block.rows):
                continue
            positions = np.searchsorted(touched, block.rows)
            if np.any(positions >= len(touched)) or not np.array_equal(
                touched[positions], block.rows
            ):
                raise AssertionError("new block row missing from touched union")
            destination = slice(
                1 + index * self.knot_count,
                1 + (index + 1) * self.knot_count,
            )
            touched_keys[positions, destination] = (
                float(specification[2]) * block.values
            )

        # SparseBlock rows are the nonzero response footprint.  If an external
        # block implementation violates that contract, use the general exact
        # path so a zero-valued touched row can still merge with its parent.
        if len(touched) and np.any(np.all(touched_keys[:, 1:] == 0.0, axis=1)):
            return None

        touched_exposure = self.tick_exposure * context.weights_at_rows(touched)
        touched_event = context.target_counts_at_sorted_rows(touched)
        touched_noevent = (
            touched_exposure - touched_event
            if context.dataset.likelihood == "first_event_cloglog"
            else touched_exposure.copy()
        )

        (
            residual_exposure,
            residual_noevent,
            residual_event,
        ) = subtract_group_weights(
            parent_groups,
            touched_exposure,
            touched_noevent,
            touched_event,
            source.exposure_weight,
            source.noevent_weight,
            source.event_weight,
        )

        scale = max(
            1.0,
            float(np.max(np.abs(source.exposure_weight), initial=0.0)),
            float(np.max(np.abs(source.noevent_weight), initial=0.0)),
            float(np.max(np.abs(source.event_weight), initial=0.0)),
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
        for weights in (residual_exposure, residual_noevent, residual_event):
            if np.any(weights < -tolerance):
                # The touched-only representation assumes that every newly
                # touched observation is a literal subset of its stored parent
                # sufficient-statistic group.  Total-state masking and a long
                # chain of aggregate/split operations can invalidate that
                # representation even though the requested support itself is
                # perfectly valid.  This helper's contract is to return None
                # for precisely such an unexpected structural case; the
                # caller then rebuilds the complete union and hashes it
                # exactly.  Clipping a material negative value would change
                # the likelihood, while raising here incorrectly aborts a
                # valid optimization path.
                return None
            weights[np.abs(weights) <= tolerance] = 0.0
        retained = (
            (residual_exposure != 0.0)
            | (residual_noevent != 0.0)
            | (residual_event != 0.0)
        )

        # Observation masks can leave an active response row with exactly
        # zero likelihood mass.  If newly touched positive-mass rows consume
        # the rest of that row's parent sufficient-statistic group, the group
        # is numerically empty but is still required as provenance for the
        # untouched active row.  Retain precisely those zero-mass parent
        # groups.  They do not change the objective, gradient or Hessian, and
        # keeping them avoids an unnecessary complete-union rebuild.
        source_is_touched = np.zeros(len(source.active_rows), dtype=bool)
        if len(source.active_rows) and len(touched):
            touched_positions_in_source = np.searchsorted(touched, source.active_rows)
            source_is_touched = touched_positions_in_source < len(touched)
            touched_safe = np.minimum(touched_positions_in_source, len(touched) - 1)
            source_is_touched &= touched[touched_safe] == source.active_rows
        if len(source.active_rows):
            untouched_parent_groups = source.active_design_groups[~source_is_touched]
            if np.any(
                (untouched_parent_groups < 0)
                | (untouched_parent_groups >= source_group_count)
            ):
                return None
            retained[untouched_parent_groups] = True
        residual_map = np.full(source_group_count, -1, dtype=np.int64)
        residual_map[retained] = np.arange(np.count_nonzero(retained))

        if len(touched):
            (
                grouped_touched_keys,
                grouped_touched_exposure,
                grouped_touched_noevent,
                grouped_touched_event,
                touched_design_groups,
            ) = self._aggregate_or_keep_design_rows(
                touched_keys,
                touched_exposure,
                touched_noevent,
                touched_event,
            )
        else:
            grouped_touched_keys = np.zeros((0, 1 + new_width), dtype=np.float64)
            grouped_touched_exposure = np.zeros(0, dtype=np.float64)
            grouped_touched_noevent = np.zeros(0, dtype=np.float64)
            grouped_touched_event = np.zeros(0, dtype=np.float64)
            touched_design_groups = np.zeros(0, dtype=np.int64)

        residual_count = int(np.count_nonzero(retained))
        touched_group_count = len(grouped_touched_keys)
        x = np.zeros(
            (residual_count + touched_group_count, dimension), dtype=np.float64
        )

        def copy_parent_columns(
            destination_rows: np.ndarray,
            parent_rows: np.ndarray,
        ) -> None:
            if not len(destination_rows):
                return
            # Never gather every parent column at once.  On a full-data
            # support that advanced-index temporary can be as large as the
            # complete tens-of-GiB destination matrix.
            x[destination_rows, :baseline_dim] = source.x[parent_rows, :baseline_dim]
            for old_index, term in enumerate(source.closure):
                old_left = baseline_dim + old_index * self.knot_count
                new_left = baseline_dim + closure_index[term] * self.knot_count
                x[destination_rows, new_left : new_left + self.knot_count] = source.x[
                    parent_rows, old_left : old_left + self.knot_count
                ]
            for rule, old_slice in zip(
                source.support.rules, source.rule_slices, strict=True
            ):
                new_left = new_rule_start + rule_index[rule] * self.knot_count
                x[destination_rows, new_left : new_left + self.knot_count] = source.x[
                    parent_rows, old_slice
                ]

        residual_source_groups = np.flatnonzero(retained)
        copy_parent_columns(
            np.arange(residual_count, dtype=np.int64),
            residual_source_groups,
        )
        if touched_group_count:
            touched_parent_groups = np.rint(grouped_touched_keys[:, 0]).astype(np.int64)
            if np.any(touched_parent_groups < 0) or np.any(
                touched_parent_groups >= source_group_count
            ):
                raise AssertionError("invalid touched parent design group")
            touched_destinations = residual_count + np.arange(
                touched_group_count, dtype=np.int64
            )
            copy_parent_columns(touched_destinations, touched_parent_groups)
            for index, (_, _, _, kind, identity) in enumerate(specifications):
                source_slice = slice(
                    1 + index * self.knot_count,
                    1 + (index + 1) * self.knot_count,
                )
                if kind == "closure":
                    left = baseline_dim + closure_index[identity] * self.knot_count
                else:
                    left = new_rule_start + rule_index[identity] * self.knot_count
                x[
                    touched_destinations,
                    left : left + self.knot_count,
                ] = grouped_touched_keys[:, source_slice]

        exposure_weight = np.concatenate(
            [residual_exposure[retained], grouped_touched_exposure]
        )
        noevent_weight = np.concatenate(
            [residual_noevent[retained], grouped_touched_noevent]
        )
        event_weight = np.concatenate([residual_event[retained], grouped_touched_event])

        active_design_groups = np.full(len(union), -1, dtype=np.int64)
        if len(source.active_rows):
            source_union_positions = np.searchsorted(union, source.active_rows)
            if np.any(source_union_positions >= len(union)) or not np.array_equal(
                union[source_union_positions], source.active_rows
            ):
                raise AssertionError("source active row missing from extended union")
            untouched = ~source_is_touched
            inherited_groups = residual_map[source.active_design_groups[untouched]]
            if np.any(inherited_groups < 0):
                # Fail open to the complete-union exact builder.  A malformed
                # or externally supplied parent matrix must never terminate a
                # valid support search merely because the incremental
                # quotient cannot preserve its metadata.
                return None
            active_design_groups[source_union_positions[untouched]] = inherited_groups
        if len(touched):
            touched_union_positions = np.searchsorted(union, touched)
            if np.any(touched_union_positions >= len(union)) or not np.array_equal(
                union[touched_union_positions], touched
            ):
                raise AssertionError("touched row missing from extended union")
            active_design_groups[touched_union_positions] = (
                residual_count + touched_design_groups
            )
        if np.any(active_design_groups < 0):
            raise AssertionError("extended active row has no design group")

        active_by_group = np.bincount(
            baseline_groups,
            weights=context.weights_at_rows(union),
            minlength=self.free_baseline_dimension,
        ).astype(np.float64)
        inactive_by_group = self._baseline_totals(context) - active_by_group
        if np.any(inactive_by_group < -tolerance):
            raise AssertionError("extended active weights exceed baseline total")
        aggregate_baseline_groups = np.flatnonzero(
            inactive_by_group > tolerance
        ).astype(np.int64)
        rule_slices = tuple(
            slice(
                new_rule_start + index * self.knot_count,
                new_rule_start + (index + 1) * self.knot_count,
            )
            for index in range(len(support.rules))
        )
        return ModelMatrix(
            x=np.ascontiguousarray(x),
            exposure_weight=np.ascontiguousarray(exposure_weight),
            noevent_weight=np.ascontiguousarray(noevent_weight),
            event_weight=np.ascontiguousarray(event_weight),
            free_dimension=source.free_dimension,
            closure_dimension=len(closure) * self.knot_count,
            rule_slices=rule_slices,
            support=support,
            closure=closure,
            closure_signs=tuple(self.closure_sign(term) for term in closure),
            active_rows=union,
            active_design_groups=np.ascontiguousarray(
                active_design_groups, dtype=np.int32
            ),
            active_baseline_groups=baseline_groups,
            aggregate_baseline_groups=aggregate_baseline_groups,
            control_dimension=source.control_dimension,
        )

    def _finalize_model_matrix(
        self,
        context: Context,
        support: Support,
        closure: tuple[ClosureTerm, ...],
        closure_signs: tuple[int, ...],
        union: np.ndarray,
        baseline_groups: np.ndarray,
        x_active: np.ndarray,
        *,
        aggregate_rows: bool = True,
    ) -> ModelMatrix:
        """Attach likelihood weights and losslessly aggregate an active design."""
        baseline_group_count = self.free_baseline_dimension
        baseline_dim = self.baseline_dimension
        dimension = x_active.shape[1]
        # Every row outside the active union has the same intercept-only design.
        total_by_group = self._baseline_totals(context)
        active_weights = context.weights_at_rows(union)
        active_by_group = np.bincount(
            baseline_groups,
            weights=active_weights,
            minlength=baseline_group_count,
        ).astype(np.float64)
        inactive_by_group = total_by_group - active_by_group
        aggregate_baseline_groups = np.flatnonzero(inactive_by_group > 0)
        x_aggregate = np.zeros(
            (len(aggregate_baseline_groups), dimension), dtype=np.float64
        )
        x_aggregate[
            np.arange(len(aggregate_baseline_groups)), aggregate_baseline_groups
        ] = 1.0
        x = np.concatenate([x_active, x_aggregate], axis=0)
        target_position = np.searchsorted(union, context.target_rows)
        matched = (
            (target_position < len(union))
            & (
                union[np.minimum(target_position, max(0, len(union) - 1))]
                == context.target_rows
            )
            if len(union)
            else np.zeros(len(context.target_rows), dtype=bool)
        )
        event_active = np.zeros(len(union), dtype=np.float64)
        if np.any(matched):
            event_active[target_position[matched]] = context.target_counts[matched]
        # target rows were explicitly included, so the unmatched branch is an invariant check.
        if len(context.target_rows) and not np.all(matched):
            raise AssertionError("target row missing from active union")
        event_weight = np.concatenate(
            [event_active, np.zeros(len(aggregate_baseline_groups))]
        )
        exposure_weight = self.tick_exposure * np.concatenate(
            [active_weights, inactive_by_group[aggregate_baseline_groups]]
        )
        if context.dataset.likelihood == "first_event_cloglog":
            noevent_weight = exposure_weight - event_weight
        else:
            noevent_weight = exposure_weight.copy()
        active_count = len(union)
        if aggregate_rows:
            (
                x,
                exposure_weight,
                noevent_weight,
                event_weight,
                design_groups,
            ) = self._aggregate_or_keep_design_rows(
                x,
                exposure_weight,
                noevent_weight,
                event_weight,
            )
        else:
            # Dependency scoring needs the exact entity/time allocation of
            # active rows and immediately consumes the matrix.  Hashing and
            # copying those rows into duplicate-design groups first changes no
            # score or Godambe dimension, but dominated full-data rescoring.
            # Keep the lossless unaggregated sufficient statistic instead.
            design_groups = np.arange(len(x), dtype=np.int32)
        rule_start = baseline_dim + len(closure) * self.knot_count
        rule_slices, expected_dimension = self.rule_slices(support, rule_start)
        if expected_dimension != dimension:
            raise AssertionError("rule representation dimension mismatch")
        return ModelMatrix(
            x=x,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event_weight,
            free_dimension=self.free_baseline_dimension,
            closure_dimension=len(closure) * self.knot_count,
            rule_slices=rule_slices,
            support=support,
            closure=closure,
            closure_signs=closure_signs,
            active_rows=union,
            active_design_groups=np.ascontiguousarray(
                design_groups[:active_count], dtype=np.int32
            ),
            active_baseline_groups=baseline_groups,
            aggregate_baseline_groups=aggregate_baseline_groups,
            control_dimension=baseline_dim - self.free_baseline_dimension,
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
        baseline_dimension = matrix.baseline_dimension
        rows = np.arange(context.n_grid, dtype=np.int64)
        eta = coefficients[
            context.temporal_baseline_groups_at_rows(
                rows, time_bins=self.baseline_time_bins
            )
        ].astype(np.float64, copy=True)
        for index, (predicate, sign) in enumerate(
            zip(self.control_predicates, self.control_signs, strict=True)
        ):
            block = self.control_block(context, predicate)
            left = self.free_baseline_dimension + index * self.knot_count
            beta = coefficients[left : left + self.knot_count]
            if len(block.rows):
                eta[block.rows] += sign * (block.values @ beta)
        left = baseline_dimension
        for index, term in enumerate(matrix.closure):
            block = self.closure_block(context, term)
            beta = coefficients[
                left + index * self.knot_count : left + (index + 1) * self.knot_count
            ]
            if len(block.rows):
                eta[block.rows] += matrix.closure_signs[index] * (block.values @ beta)
        for rule, block, coefficient_slice in zip(
            matrix.support.rules,
            self.total_state_rule_blocks(context, matrix.support),
            matrix.rule_slices,
            strict=True,
        ):
            beta = coefficients[coefficient_slice]
            if len(block.rows):
                eta[block.rows] += rule.sign * (
                    self.rule_design_values(block, rule) @ beta
                )
        return eta

    def design_at_rows_with_context(
        self, context: Context, matrix: ModelMatrix, rows: np.ndarray
    ) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= context.n_grid):
            raise ValueError("grid row is outside the context")
        baseline_dimension = matrix.baseline_dimension
        output = np.zeros((len(rows), matrix.dimension), dtype=np.float64)
        output[
            np.arange(len(rows)),
            context.temporal_baseline_groups_at_rows(
                rows, time_bins=self.baseline_time_bins
            ),
        ] = 1.0
        if baseline_dimension != self.baseline_dimension:
            raise ValueError("matrix baseline-control dimension mismatch")
        # ``active_rows`` contains every target or response-block row and maps
        # exactly to the compact, deduplicated design group in ``matrix.x``.
        # Rows outside it have baseline columns only.  Reusing that map avoids
        # rebuilding every closure/rule block for every line-search tile.
        positions = np.searchsorted(matrix.active_rows, rows)
        matched = positions < len(matrix.active_rows)
        if len(matrix.active_rows):
            safe = np.minimum(positions, len(matrix.active_rows) - 1)
            matched &= matrix.active_rows[safe] == rows
        if np.any(matched):
            output[matched] = matrix.x[matrix.active_design_groups[positions[matched]]]
        return output

    def linear_predictor_at_rows(
        self,
        context: Context,
        matrix: ModelMatrix,
        coefficients: np.ndarray,
        rows: np.ndarray,
    ) -> np.ndarray:
        design = self.design_at_rows_with_context(context, matrix, rows)
        return design @ np.asarray(coefficients, dtype=np.float64)

    def footprint_rows(
        self, context: Context, rule: RuleIdentity, horizon: int
    ) -> np.ndarray:
        if rule.history_marks:
            entities, times, spans = self.rule_completions(context, rule)
            maximum_span = int(rule.window) * self.dataset.ticks_per_unit
            lower = self.window_band_lower(rule.antecedent, rule.window, rule.relation)
            minimum_span = (
                int(lower) * self.dataset.ticks_per_unit if lower is not None else -1
            )
            admitted = (spans > minimum_span) & (spans <= maximum_span)
            if self.continuous:
                return self._continuous_footprint_rows(
                    context,
                    entities[admitted],
                    times[admitted],
                    horizon_ticks=int(horizon) * self.dataset.ticks_per_unit,
                )
            expanded = future_rows(
                entities,
                times,
                spans,
                context.starts,
                context.ends,
                context.offsets,
                window=int(rule.window) * self.dataset.ticks_per_unit,
                horizon=int(horizon) * self.dataset.ticks_per_unit,
            )
            if expanded is not None:
                return np.unique(expanded[0])
            rows: list[int] = []
            maximum_lag = int(horizon) * self.dataset.ticks_per_unit
            for entity, completion, span in zip(
                entities.tolist(), times.tolist(), spans.tolist(), strict=True
            ):
                if span > maximum_span:
                    continue
                remaining = min(maximum_lag, int(context.ends[entity] - completion))
                base = int(
                    context.offsets[entity] + completion - context.starts[entity]
                )
                rows.extend(base + lag for lag in range(1, remaining + 1))
            return np.unique(np.asarray(rows, dtype=np.int64))
        return self.response_rows_many(
            context,
            rule.antecedent,
            (rule.window,),
            horizon=horizon,
            relation=rule.relation,
        )[rule.window]

    def response_rows_many(
        self,
        context: Context,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        *,
        horizon: int | None = None,
        relation: str = "unordered",
    ) -> dict[int, np.ndarray]:
        """Return nested exact footprints after one bounded expansion pass."""
        unique_windows = tuple(sorted(set(map(int, windows))))
        if not unique_windows:
            return {}
        horizon_units = self.lag_units if horizon is None else int(horizon)
        band_lowers = {
            window: self.window_band_lower(antecedent, window, relation)
            for window in unique_windows
        }
        if any(lower is not None for lower in band_lowers.values()):
            if any(lower is None for lower in band_lowers.values()):
                raise ValueError("one pattern cannot mix cumulative and band windows")
            if horizon_units == self.lag_units:
                return {
                    window: self.block(context, antecedent, window, relation).rows
                    for window in unique_windows
                }
            entities, times, spans = self.completions(context, antecedent, relation)
            output: dict[int, np.ndarray] = {}
            horizon_ticks = horizon_units * self.dataset.ticks_per_unit
            for window in unique_windows:
                lower_ticks = int(band_lowers[window]) * self.dataset.ticks_per_unit
                upper_ticks = int(window) * self.dataset.ticks_per_unit
                admitted = (spans > lower_ticks) & (spans <= upper_ticks)
                selected_entities = entities[admitted]
                selected_times = times[admitted]
                if self.continuous:
                    output[window] = self._continuous_footprint_rows(
                        context,
                        selected_entities,
                        selected_times,
                        horizon_ticks=horizon_ticks,
                    )
                    continue
                rows: list[int] = []
                for entity, completion in zip(
                    selected_entities.tolist(), selected_times.tolist(), strict=True
                ):
                    remaining = min(
                        horizon_ticks, int(context.ends[entity] - completion)
                    )
                    base = int(
                        context.offsets[entity] + completion - context.starts[entity]
                    )
                    rows.extend(base + lag for lag in range(1, remaining + 1))
                output[window] = np.unique(np.asarray(rows, dtype=np.int64))
            return output
        rows, minimum_spans = self.response_row_thresholds(
            context,
            antecedent,
            max(unique_windows),
            horizon=horizon_units,
            relation=relation,
        )
        output: dict[int, np.ndarray] = {}
        for window in unique_windows:
            relation = "atomic" if len(antecedent) == 1 else str(relation)
            key = (id(context), relation, antecedent, window, horizon_units)
            with self._lock:
                cached = self._footprint_cache.get(key)
                if cached is not None:
                    self._footprint_cache.move_to_end(key)
                    output[window] = cached
                    continue
            result = rows[minimum_spans <= window * self.dataset.ticks_per_unit]
            if result.nbytes <= self._footprint_cache_limit:
                with self._lock:
                    existing = self._footprint_cache.get(key)
                    if existing is not None:
                        self._footprint_cache.move_to_end(key)
                        output[window] = existing
                        continue
                    self._footprint_cache[key] = result
                    self._footprint_cache_size += result.nbytes
                    while (
                        self._footprint_cache_size > self._footprint_cache_limit
                        and len(self._footprint_cache) > 1
                    ):
                        _, removed = self._footprint_cache.popitem(last=False)
                        self._footprint_cache_size -= removed.nbytes
            output[window] = result
        return output

    def response_row_thresholds(
        self,
        context: Context,
        antecedent: Antecedent,
        maximum_window: int,
        *,
        horizon: int | None = None,
        relation: str = "unordered",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact exact representation shared by every nested W footprint."""
        horizon_units = self.lag_units if horizon is None else int(horizon)
        return self._response_row_thresholds(
            context,
            antecedent,
            horizon_units * self.dataset.ticks_per_unit,
            int(maximum_window) * self.dataset.ticks_per_unit,
            relation=relation,
        )

    def response_row_lookup(
        self,
        context: Context,
        antecedent: Antecedent,
        maximum_window: int,
        *,
        horizon: int | None = None,
        relation: str = "unordered",
    ) -> np.ndarray | None:
        """Return a shared exact dense row-to-threshold-position lookup.

        Fine continuous grids for which the dense index dwarfs the active
        footprint deliberately return ``None`` and use the existing sparse
        fail-open path.  On bounded Freddie grids, the same lower-order pair
        lookup is shared by every triplet that owns it.
        """
        horizon_units = self.lag_units if horizon is None else int(horizon)
        relation = "atomic" if len(antecedent) == 1 else str(relation)
        key = (
            id(context),
            relation,
            antecedent,
            horizon_units * self.dataset.ticks_per_unit,
            int(maximum_window) * self.dataset.ticks_per_unit,
        )
        rows, _ = self._response_row_thresholds(
            context,
            antecedent,
            key[3],
            key[4],
            relation=relation,
        )
        if context.n_grid > max(64_000_000, 8 * len(rows)):
            return None
        with self._lock:
            cached = self._row_lookup_cache.get(key)
            if cached is not None:
                self._row_lookup_cache.move_to_end(key)
                return cached
        with self._compute_lock("row-lookup", key):
            with self._lock:
                cached = self._row_lookup_cache.get(key)
                if cached is not None:
                    self._row_lookup_cache.move_to_end(key)
                    return cached
            lookup = np.full(context.n_grid, -1, dtype=np.int32)
            lookup[rows] = np.arange(len(rows), dtype=np.int32)
            if lookup.nbytes <= self._row_lookup_cache_limit:
                with self._lock:
                    self._row_lookup_cache[key] = lookup
                    self._row_lookup_cache_size += lookup.nbytes
                    while (
                        self._row_lookup_cache_size > self._row_lookup_cache_limit
                        and len(self._row_lookup_cache) > 1
                    ):
                        _, removed = self._row_lookup_cache.popitem(last=False)
                        self._row_lookup_cache_size -= removed.nbytes
            return lookup

    def _response_row_thresholds(
        self,
        context: Context,
        antecedent: Antecedent,
        horizon_ticks: int,
        maximum_window_ticks: int,
        *,
        relation: str = "unordered",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return every future row and its minimum admitting witness span."""
        relation = "atomic" if len(antecedent) == 1 else str(relation)
        key = (
            id(context),
            relation,
            antecedent,
            int(horizon_ticks),
            int(maximum_window_ticks),
        )
        with self._lock:
            cached = self._row_threshold_cache.get(key)
            if cached is not None:
                self._row_threshold_cache.move_to_end(key)
                return cached
        with self._compute_lock("row-threshold", key):
            with self._lock:
                cached = self._row_threshold_cache.get(key)
                if cached is not None:
                    self._row_threshold_cache.move_to_end(key)
                    return cached
            return self._compute_response_row_thresholds(
                context,
                antecedent,
                int(horizon_ticks),
                int(maximum_window_ticks),
                relation,
                key,
            )

    def _compute_response_row_thresholds(
        self,
        context: Context,
        antecedent: Antecedent,
        horizon_ticks: int,
        maximum_window_ticks: int,
        relation: str,
        key: tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        entities, times, spans = self.completions(context, antecedent, relation)
        admitted = spans <= int(maximum_window_ticks)
        entities, times, spans = (
            entities[admitted],
            times[admitted],
            spans[admitted],
        )
        if self.continuous:
            rows, minimum_spans = self._continuous_response_thresholds(
                context,
                entities,
                times,
                spans,
                maximum_span=int(maximum_window_ticks),
                horizon_ticks=int(horizon_ticks),
            )
            result = (rows, minimum_spans)
            size = rows.nbytes + minimum_spans.nbytes
            if size <= self._row_threshold_cache_limit:
                with self._lock:
                    self._row_threshold_cache[key] = result
                    self._row_threshold_cache_size += size
                    while (
                        self._row_threshold_cache_size > self._row_threshold_cache_limit
                        and len(self._row_threshold_cache) > 1
                    ):
                        _, removed = self._row_threshold_cache.popitem(last=False)
                        self._row_threshold_cache_size -= (
                            removed[0].nbytes + removed[1].nbytes
                        )
            return result
        # A dense minimum-span bitmap avoids materializing and sorting
        # O(completions * horizon) temporary rows on bounded discrete grids.
        # This changes only representation; fine continuous-time grids retain
        # the sparse expansion below.
        dense_representation = context.n_grid <= 64_000_000
        compiled = (
            response_min_spans(
                entities,
                times,
                spans,
                context.starts,
                context.ends,
                context.offsets,
                horizon=int(horizon_ticks),
                n_grid=context.n_grid,
            )
            if dense_representation
            else future_rows(
                entities,
                times,
                spans,
                context.starts,
                context.ends,
                context.offsets,
                window=int(maximum_window_ticks),
                horizon=int(horizon_ticks),
            )
        )
        if compiled is None:
            raw_rows: list[np.ndarray] = []
            raw_spans: list[np.ndarray] = []
            for entity, completion, span in zip(
                entities.tolist(), times.tolist(), spans.tolist(), strict=True
            ):
                length = min(int(horizon_ticks), int(context.ends[entity] - completion))
                if length > 0:
                    base = int(
                        context.offsets[entity] + completion - context.starts[entity]
                    )
                    raw_rows.append(base + np.arange(1, length + 1, dtype=np.int64))
                    raw_spans.append(np.full(length, span, dtype=np.int64))
            generated_rows = (
                np.concatenate(raw_rows) if raw_rows else np.zeros(0, dtype=np.int64)
            )
            generated_spans = (
                np.concatenate(raw_spans) if raw_spans else np.zeros(0, dtype=np.int64)
            )
        else:
            generated_rows, generated_spans = compiled
        if dense_representation and compiled is not None:
            rows, minimum_spans = generated_rows, generated_spans
        elif len(generated_rows):
            order = np.argsort(generated_rows, kind="stable")
            ordered_rows = generated_rows[order]
            ordered_spans = generated_spans[order]
            rows, first = np.unique(ordered_rows, return_index=True)
            minimum_spans = np.minimum.reduceat(ordered_spans, first)
        else:
            rows = np.zeros(0, dtype=np.int64)
            minimum_spans = np.zeros(0, dtype=np.int64)
        result = (rows, minimum_spans)
        size = rows.nbytes + minimum_spans.nbytes
        if size <= self._row_threshold_cache_limit:
            with self._lock:
                self._row_threshold_cache[key] = result
                self._row_threshold_cache_size += size
                while (
                    self._row_threshold_cache_size > self._row_threshold_cache_limit
                    and len(self._row_threshold_cache) > 1
                ):
                    _, removed = self._row_threshold_cache.popitem(last=False)
                    self._row_threshold_cache_size -= (
                        removed[0].nbytes + removed[1].nbytes
                    )
        return result

    def response_rows(
        self,
        context: Context,
        antecedent: Antecedent,
        window: int,
        relation: str = "auto",
    ) -> np.ndarray:
        """Exact nonzero footprint without materializing M-knot values."""
        return self.footprint_rows(
            context,
            RuleIdentity(antecedent, int(window), 1, relation=relation),
            self.lag_units,
        )
