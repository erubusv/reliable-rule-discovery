from __future__ import annotations

import hashlib
import itertools
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import EventData, QueryContext
from .native import (
    linear_completions,
    sorted_unique_int64_union,
    sorted_unique_int64_union_with_positions,
    sparse_kernel_block,
)


Antecedent = tuple[int, ...]


@dataclass(frozen=True, order=True)
class RuleIdentity:
    antecedent: Antecedent
    window: int
    sign: int

    def __post_init__(self) -> None:
        if not self.antecedent or tuple(sorted(set(self.antecedent))) != self.antecedent:
            raise ValueError("antecedent must be a nonempty sorted unique tuple")
        if self.window < 0:
            raise ValueError("window must be nonnegative")
        if self.sign not in (-1, 1):
            raise ValueError("rule sign must be -1 or +1")

    @property
    def order(self) -> int:
        return len(self.antecedent)


@dataclass(frozen=True)
class CompletionEvents:
    sequence_codes: np.ndarray
    times: np.ndarray
    spans: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.times))

    @property
    def nbytes(self) -> int:
        return int(self.sequence_codes.nbytes + self.times.nbytes + self.spans.nbytes)


@dataclass(frozen=True)
class SparseKernelResponse:
    """Kernel response stored only on grid rows with a nonzero value."""

    n_events: int
    n_grid: int
    grid_indices: np.ndarray
    grid_values: np.ndarray
    event_values: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.grid_indices)
        grid = np.asarray(self.grid_values)
        events = np.asarray(self.event_values)
        if indices.ndim != 1 or grid.ndim != 2 or events.ndim != 2:
            raise ValueError("invalid sparse kernel response dimensions")
        if len(indices) != len(grid) or events.shape[0] != int(self.n_events):
            raise ValueError("sparse kernel response row mismatch")
        if grid.shape[1] != events.shape[1]:
            raise ValueError("sparse kernel response column mismatch")
        if len(indices) and (
            np.any(indices < 0)
            or np.any(indices >= int(self.n_grid))
            or np.any(indices[1:] <= indices[:-1])
        ):
            raise ValueError("sparse grid indices must be sorted, unique, and in range")
        if np.any(~np.isfinite(grid)) or np.any(~np.isfinite(events)):
            raise ValueError("sparse kernel response must be finite")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.n_events + self.n_grid), int(self.grid_values.shape[1]))

    @property
    def nbytes(self) -> int:
        return int(
            self.grid_indices.nbytes
            + self.grid_values.nbytes
            + self.event_values.nbytes
        )

    def dense(self) -> np.ndarray:
        grid = np.zeros((self.n_grid, self.shape[1]), dtype=np.float32)
        if len(self.grid_indices):
            grid[self.grid_indices] = self.grid_values
        return np.concatenate((self.event_values, grid), axis=0)

    def grid_at(self, indices: np.ndarray) -> np.ndarray:
        """Gather arbitrary grid rows without expanding the zero background."""
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(requested < 0) or np.any(requested >= int(self.n_grid)):
            raise IndexError("sparse grid row is out of range")
        output = np.zeros((len(requested), self.shape[1]), dtype=np.float32)
        if not len(requested) or not len(self.grid_indices):
            return output
        positions = np.searchsorted(self.grid_indices, requested)
        safe = np.minimum(positions, len(self.grid_indices) - 1)
        matched = (positions < len(self.grid_indices)) & (
            self.grid_indices[safe] == requested
        )
        output[matched] = self.grid_values[positions[matched]]
        return output

    def add_grid_linear_predictor(
        self,
        indices: np.ndarray,
        coefficients: np.ndarray,
        output: np.ndarray,
        *,
        scale: float = 1.0,
        assume_sorted_unique: bool = False,
    ) -> None:
        """Add ``scale * X[indices] @ coefficients`` without materializing zeros.

        ``grid_at(indices)`` is convenient, but creates an ``n_indices`` by
        ``n_columns`` zero matrix even when only a small fraction of the rows
        are stored.  Certification repeatedly needs only the resulting linear
        predictor, so intersect the requested and stored rows and multiply the
        nonzero rows directly.  This is algebraically identical to
        ``grid_at(indices) @ coefficients`` and preserves repeated or unsorted
        requests.
        """
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        beta = np.asarray(coefficients)
        target = np.asarray(output)
        if target.shape != (len(requested),):
            raise ValueError("output does not align with requested sparse rows")
        if beta.shape != (self.shape[1],):
            raise ValueError("coefficient vector does not match sparse response")
        if np.any(requested < 0) or np.any(requested >= int(self.n_grid)):
            raise IndexError("sparse grid row is out of range")
        if not len(requested) or not len(self.grid_indices):
            return
        # Certification passes sorted unions of sparse row sets.  In that hot
        # path the stored block is usually much smaller than the union, so map
        # the stored rows into the union instead of binary-searching once per
        # (mostly zero) requested row.  Fall back to the fully general gather
        # semantics for unsorted/repeated requests.
        sorted_unique = bool(
            assume_sorted_unique
            or len(requested) < 2
            or np.all(requested[1:] > requested[:-1])
        )
        if sorted_unique and len(self.grid_indices) < len(requested):
            output_positions = np.searchsorted(requested, self.grid_indices)
            safe = np.minimum(output_positions, len(requested) - 1)
            matched = (output_positions < len(requested)) & (
                requested[safe] == self.grid_indices
            )
            if np.any(matched):
                target[output_positions[matched]] += float(scale) * (
                    self.grid_values[matched] @ beta
                )
            return
        positions = np.searchsorted(self.grid_indices, requested)
        safe = np.minimum(positions, len(self.grid_indices) - 1)
        matched = (positions < len(self.grid_indices)) & (
            self.grid_indices[safe] == requested
        )
        if np.any(matched):
            target[matched] += float(scale) * (
                self.grid_values[positions[matched]] @ beta
            )

    def scatter_grid_at(
        self,
        indices: np.ndarray,
        output: np.ndarray,
        *,
        assume_sorted_unique: bool = False,
    ) -> None:
        """Write ``X[indices]`` into a preallocated matrix without a temporary."""
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        target = np.asarray(output)
        if target.shape != (len(requested), self.shape[1]):
            raise ValueError("output does not align with requested sparse rows")
        if np.any(requested < 0) or np.any(requested >= int(self.n_grid)):
            raise IndexError("sparse grid row is out of range")
        target.fill(0.0)
        if not len(requested) or not len(self.grid_indices):
            return
        sorted_unique = bool(
            assume_sorted_unique
            or len(requested) < 2
            or np.all(requested[1:] > requested[:-1])
        )
        if sorted_unique and len(self.grid_indices) < len(requested):
            output_positions = np.searchsorted(requested, self.grid_indices)
            safe = np.minimum(output_positions, len(requested) - 1)
            matched = (output_positions < len(requested)) & (
                requested[safe] == self.grid_indices
            )
            if np.any(matched):
                target[output_positions[matched]] = self.grid_values[matched]
            return
        positions = np.searchsorted(self.grid_indices, requested)
        safe = np.minimum(positions, len(self.grid_indices) - 1)
        matched = (positions < len(self.grid_indices)) & (
            self.grid_indices[safe] == requested
        )
        if np.any(matched):
            target[matched] = self.grid_values[positions[matched]]

    def projected(self, shape: np.ndarray) -> SparseKernelResponse:
        shape32 = np.asarray(shape, dtype=np.float32).reshape(-1)
        if shape32.shape != (self.shape[1],) or np.any(~np.isfinite(shape32)):
            raise ValueError("projection shape does not match sparse response")
        grid = (self.grid_values @ shape32).reshape(-1, 1).astype(np.float32, copy=False)
        events = (self.event_values @ shape32).reshape(-1, 1).astype(np.float32, copy=False)
        keep = np.any(grid != 0.0, axis=1)
        return SparseKernelResponse(
            n_events=self.n_events,
            n_grid=self.n_grid,
            grid_indices=self.grid_indices[keep].copy(),
            grid_values=grid[keep].copy(),
            event_values=events,
        )


@dataclass(frozen=True)
class SourceEvents:
    sequence_codes: np.ndarray
    times: np.ndarray
    offsets: np.ndarray
    populated_sequences: np.ndarray


def make_triangular_basis(lag: int, count: int) -> np.ndarray:
    if lag < 1 or count < 1:
        raise ValueError("lag and knot count must be positive")
    x = np.arange(1, lag + 1, dtype=np.float64)
    if count == 1:
        raw = np.ones((1, lag), dtype=np.float64)
    else:
        centers = np.linspace(1.0, float(lag), count)
        width = max(float(centers[1] - centers[0]), 1.0)
        raw = np.stack([np.maximum(0.0, 1.0 - np.abs(x - c) / width) for c in centers])
    area = np.sum(raw, axis=1, keepdims=True)
    if np.any(area <= 0):
        raise RuntimeError("degenerate temporal basis")
    return (raw / area).astype(np.float32)


class RuleOccurrenceEngine:
    def __init__(
        self,
        data: EventData,
        *,
        lag: int,
        knot_count: int,
        feature_cache_bytes: int = 4 * 1024**3,
        max_completion_span: int | None = None,
        extra_source_events: dict[int, SourceEvents] | None = None,
    ):
        self.data = data
        self.lag = int(lag)
        self.knot_count = int(knot_count)
        self.max_completion_span = (
            None if max_completion_span is None else int(max_completion_span)
        )
        if self.max_completion_span is not None and self.max_completion_span < 0:
            raise ValueError("maximum completion span must be nonnegative")
        self.basis = make_triangular_basis(self.lag, self.knot_count)
        # Reporting and information calculations are float64.  Preserve the
        # same values while avoiding a fresh M-by-L conversion per rule.
        self.basis64 = self.basis.astype(np.float64)
        # Raw response keys have three fields; projected dictionary-shape keys
        # append shape bytes, and completion-stream keys use a reserved prefix.
        # All expensive occurrence artifacts share one byte-bounded LRU so the
        # configured cache limit is a real total, not response bytes plus an
        # unbounded completion dictionary.
        self._feature_cache: OrderedDict[
            tuple, np.ndarray | CompletionEvents | SparseKernelResponse
        ] = OrderedDict()
        self._feature_cache_bytes = 0
        self._feature_cache_limit = max(0, int(feature_cache_bytes))
        self._cache_guard = threading.RLock()
        self._response_locks: dict[tuple, threading.Lock] = {}
        self._source_event_cache: dict[int, SourceEvents] = {}
        for source_id, source in (extra_source_events or {}).items():
            source_id = int(source_id)
            if source_id < self.data.n_predicates:
                raise ValueError("extra source ids must not overlap predicate columns")
            sequence_codes = np.asarray(source.sequence_codes)
            times = np.asarray(source.times)
            offsets = np.asarray(source.offsets)
            populated = np.asarray(source.populated_sequences)
            if (
                sequence_codes.ndim != 1
                or times.ndim != 1
                or sequence_codes.shape != times.shape
                or offsets.shape != (self.data.n_sequences + 1,)
                or populated.ndim != 1
                or not np.issubdtype(sequence_codes.dtype, np.integer)
                or not np.issubdtype(times.dtype, np.integer)
                or not np.issubdtype(offsets.dtype, np.integer)
                or not np.issubdtype(populated.dtype, np.integer)
                or int(offsets[0]) != 0
                or int(offsets[-1]) != len(times)
                or np.any(offsets[1:] < offsets[:-1])
                or np.any(sequence_codes < 0)
                or np.any(sequence_codes >= self.data.n_sequences)
                or np.any(times < np.iinfo(np.int32).min)
                or np.any(times > np.iinfo(np.int32).max)
            ):
                raise ValueError("invalid extra source event stream")
            expected_counts = np.diff(offsets)
            actual_counts = np.bincount(
                sequence_codes.astype(np.int64, copy=False),
                minlength=self.data.n_sequences,
            )
            expected_populated = np.flatnonzero(expected_counts).astype(
                np.int64, copy=False
            )
            if (
                not np.array_equal(expected_counts, actual_counts)
                or not np.array_equal(
                    populated.astype(np.int64, copy=False), expected_populated
                )
                or (
                    len(sequence_codes) > 1
                    and np.any(
                        (sequence_codes[1:] < sequence_codes[:-1])
                        | (
                            (sequence_codes[1:] == sequence_codes[:-1])
                            & (times[1:] < times[:-1])
                        )
                    )
                )
            ):
                raise ValueError(
                    "extra source events must be sequence/time ordered with consistent offsets"
                )
            self._source_event_cache[source_id] = source
        self._context_sequences: dict[str, np.ndarray] = {}
        self._sparse_equivalence_keys: dict[tuple, list[tuple]] = {}
        self._sparse_aliases: dict[tuple, tuple] = {}
        self._sparse_key_fingerprints: dict[tuple, tuple] = {}
        self._sparse_canonical_aliases: dict[tuple, set[tuple]] = {}
        self.equivalent_response_hits = 0
        self._all_sequence_lookup = np.zeros(self.data.n_sequences, dtype=np.int32)
        # These bounds are dataset invariants.  Recomputing four full-array
        # reductions for every antecedent was an accidental O(skeletons*N)
        # cost in the portable fallback.
        self._time_origin = int(
            min(np.min(self.data.start_times), np.min(self.data.times))
        )
        self._time_stride = int(
            max(np.max(self.data.end_times), np.max(self.data.times))
        ) - self._time_origin + 2

    def _validate_context_identity(self, ctx: QueryContext) -> None:
        """Prevent a name-keyed cache from serving a different cohort."""
        with self._cache_guard:
            known = self._context_sequences.get(ctx.name)
            if known is None:
                self._context_sequences[ctx.name] = ctx.global_sequence_ids.copy()
            elif not np.array_equal(known, ctx.global_sequence_ids):
                raise ValueError(
                    f"query-context name {ctx.name!r} was reused for a different sequence cohort"
                )

    def antecedents(self, source_ids: Sequence[int], q_max: int = 3) -> list[Antecedent]:
        values = tuple(sorted(set(int(v) for v in source_ids)))
        return [
            tuple(combo)
            for order in range(1, min(int(q_max), len(values)) + 1)
            for combo in itertools.combinations(values, order)
        ]

    def completions(self, antecedent: Antecedent) -> CompletionEvents:
        antecedent = tuple(antecedent)
        key = ("__completion_global__", antecedent)
        with self._cache_guard:
            cached = self._feature_cache.get(key)
            if isinstance(cached, CompletionEvents):
                self._feature_cache.move_to_end(key)
                return cached
            key_lock = self._response_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._cache_guard:
                cached = self._feature_cache.get(key)
                if isinstance(cached, CompletionEvents):
                    self._feature_cache.move_to_end(key)
                    return cached
            result = self._compute_completions(antecedent, self._all_sequence_lookup)
            self._cache_value(key, result)
            return result

    def completions_for_context(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
    ) -> CompletionEvents:
        self._validate_context_identity(ctx)
        antecedent = tuple(antecedent)
        key = ("__completion_context__", ctx.name, antecedent)
        with self._cache_guard:
            cached = self._feature_cache.get(key)
            if isinstance(cached, CompletionEvents):
                self._feature_cache.move_to_end(key)
                return cached
            key_lock = self._response_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._cache_guard:
                cached = self._feature_cache.get(key)
                if isinstance(cached, CompletionEvents):
                    self._feature_cache.move_to_end(key)
                    return cached
            result = self._compute_completions(antecedent, ctx.sequence_lookup)
            self._cache_value(key, result)
            return result

    def _cache_value(
        self,
        key: tuple,
        value: np.ndarray | CompletionEvents | SparseKernelResponse,
    ) -> None:
        size = int(value.nbytes)
        with self._cache_guard:
            existing = self._feature_cache.get(key)
            if existing is not None:
                self._feature_cache.move_to_end(key)
                self._response_locks.pop(key, None)
                return
            if self._feature_cache_limit <= 0 or size > self._feature_cache_limit:
                self._response_locks.pop(key, None)
                return
            self._feature_cache[key] = value
            self._feature_cache.move_to_end(key)
            self._feature_cache_bytes += size
            while (
                self._feature_cache_bytes > self._feature_cache_limit
                and len(self._feature_cache) > 1
            ):
                old_key, old_value = self._feature_cache.popitem(last=False)
                self._feature_cache_bytes -= int(old_value.nbytes)
                self._response_locks.pop(old_key, None)
                self._remove_sparse_metadata(old_key)
            self._response_locks.pop(key, None)

    def _remove_sparse_metadata(self, key: tuple) -> None:
        """Remove all interning metadata owned by one cache/alias key.

        This is called with ``_cache_guard`` held.  Keeping the reverse maps in
        sync matters because alias keys deliberately do not own another strong
        reference to the response object.
        """
        canonical = self._sparse_aliases.pop(key, None)
        if canonical is not None:
            aliases = self._sparse_canonical_aliases.get(canonical)
            if aliases is not None:
                aliases.discard(key)
                if not aliases:
                    self._sparse_canonical_aliases.pop(canonical, None)
            return

        fingerprint = self._sparse_key_fingerprints.pop(key, None)
        if fingerprint is not None:
            candidates = self._sparse_equivalence_keys.get(fingerprint)
            if candidates is not None:
                remaining = [candidate for candidate in candidates if candidate != key]
                if remaining:
                    self._sparse_equivalence_keys[fingerprint] = remaining
                else:
                    self._sparse_equivalence_keys.pop(fingerprint, None)
        for alias in self._sparse_canonical_aliases.pop(key, set()):
            self._sparse_aliases.pop(alias, None)

    @staticmethod
    def _sparse_fingerprint(value: SparseKernelResponse) -> tuple:
        digest = hashlib.blake2b(digest_size=16)
        for array in (
            value.grid_indices,
            value.grid_values,
            value.event_values,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.view(np.uint8))
        return (
            int(value.n_events),
            int(value.n_grid),
            int(value.shape[1]),
            digest.digest(),
        )

    @staticmethod
    def _same_sparse_response(
        left: SparseKernelResponse,
        right: SparseKernelResponse,
    ) -> bool:
        return bool(
            left.n_events == right.n_events
            and left.n_grid == right.n_grid
            and np.array_equal(left.grid_indices, right.grid_indices)
            and np.array_equal(left.grid_values, right.grid_values)
            and np.array_equal(left.event_values, right.event_values)
        )

    def _cached_sparse(self, key: tuple) -> SparseKernelResponse | None:
        with self._cache_guard:
            cached = self._feature_cache.get(key)
            if isinstance(cached, SparseKernelResponse):
                self._feature_cache.move_to_end(key)
                return cached
            canonical = self._sparse_aliases.get(key)
            if canonical is None:
                return None
            cached = self._feature_cache.get(canonical)
            if isinstance(cached, SparseKernelResponse):
                self._feature_cache.move_to_end(canonical)
                return cached
            self._remove_sparse_metadata(key)
            return None

    def _intern_sparse(
        self,
        key: tuple,
        value: SparseKernelResponse,
    ) -> SparseKernelResponse:
        fingerprint = self._sparse_fingerprint(value)
        with self._cache_guard:
            candidates = self._sparse_equivalence_keys.setdefault(fingerprint, [])
            live_candidates: list[tuple] = []
            for position, canonical in enumerate(candidates):
                existing = self._feature_cache.get(canonical)
                if not isinstance(existing, SparseKernelResponse):
                    continue
                live_candidates.append(canonical)
                if self._same_sparse_response(existing, value):
                    self._sparse_aliases[key] = canonical
                    self._sparse_canonical_aliases.setdefault(canonical, set()).add(key)
                    self._feature_cache.move_to_end(canonical)
                    self.equivalent_response_hits += 1
                    self._response_locks.pop(key, None)
                    # Preserve unvisited candidates.  Stale ones are removed
                    # lazily on the next lookup; dropping them here would make
                    # interning depend on which equal response was encountered
                    # first.
                    self._sparse_equivalence_keys[fingerprint] = (
                        live_candidates + candidates[position + 1 :]
                    )
                    return existing
            self._sparse_equivalence_keys[fingerprint] = live_candidates
            self._cache_value(key, value)
            if self._feature_cache.get(key) is value:
                self._sparse_equivalence_keys.setdefault(fingerprint, []).append(key)
                self._sparse_key_fingerprints[key] = fingerprint
            return value

    def _compute_completions(
        self,
        antecedent: Antecedent,
        sequence_lookup: np.ndarray,
    ) -> CompletionEvents:
        if tuple(sorted(set(antecedent))) != antecedent:
            raise ValueError("antecedent must be sorted and unique")
        sources = [self._source_events(source) for source in antecedent]
        if len(sources) == 1:
            # A singleton completion is the source stream itself.  Ordinary
            # Boolean predicates are unique by entity/time, while the reserved
            # recurrent-target source may deliberately repeat a time to retain
            # exact same-bin target multiplicity.  The general latest-witness
            # sweep collapses equal timestamps, so bypass it here.
            source = sources[0]
            keep = np.asarray(sequence_lookup, dtype=np.int32)[
                source.sequence_codes
            ] >= 0
            return CompletionEvents(
                sequence_codes=source.sequence_codes[keep].astype(
                    np.int32, copy=False
                ),
                times=source.times[keep].astype(np.int32, copy=False),
                spans=np.zeros(int(np.sum(keep)), dtype=np.int32),
            )
        native = linear_completions(
            [source.sequence_codes for source in sources],
            [source.times for source in sources],
            sequence_lookup,
            max_span=self.max_completion_span,
        )
        if native is not None:
            return CompletionEvents(
                sequence_codes=native[0],
                times=native[1],
                spans=native[2],
            )
        eligible_sequences = sources[0].populated_sequences
        for source in sources[1:]:
            eligible_sequences = np.intersect1d(
                eligible_sequences,
                source.populated_sequences,
                assume_unique=True,
            ).astype(np.int32, copy=False)
        eligible_sequences = eligible_sequences[
            np.asarray(sequence_lookup, dtype=np.int32)[eligible_sequences] >= 0
        ]
        if not len(eligible_sequences):
            return CompletionEvents(
                sequence_codes=np.zeros(0, dtype=np.int32),
                times=np.zeros(0, dtype=np.int32),
                spans=np.zeros(0, dtype=np.int32),
            )
        selected_mask = np.zeros(self.data.n_sequences, dtype=bool)
        selected_mask[eligible_sequences] = True
        time_origin = self._time_origin
        time_stride = self._time_stride
        source_sequences: list[np.ndarray] = []
        source_times: list[np.ndarray] = []
        source_keys: list[np.ndarray] = []
        for source in sources:
            keep = selected_mask[source.sequence_codes]
            sequences = source.sequence_codes[keep].astype(np.int32, copy=False)
            times = source.times[keep].astype(np.int32, copy=False)
            if len(times) == 0:
                return CompletionEvents(
                    sequence_codes=np.zeros(0, dtype=np.int32),
                    times=np.zeros(0, dtype=np.int32),
                    spans=np.zeros(0, dtype=np.int32),
                )
            keys = (
                sequences.astype(np.int64) * time_stride
                + times.astype(np.int64)
                - time_origin
            )
            source_sequences.append(sequences)
            source_times.append(times)
            source_keys.append(keys)
        try:
            native_union = sorted_unique_int64_union(source_keys)
        except RuntimeError:
            # Event inputs from supported loaders are strictly increasing, but
            # retain the exact NumPy behavior for third-party EventData that
            # contains duplicate source-time rows.
            native_union = None
        union_keys = (
            native_union
            if native_union is not None
            else np.unique(np.concatenate(source_keys))
        )
        union_sequences = (union_keys // time_stride).astype(np.int32, copy=False)
        union_times = (union_keys % time_stride + time_origin).astype(np.int32, copy=False)
        latest = np.empty((len(union_keys), len(sources)), dtype=np.int32)
        valid = np.ones(len(union_keys), dtype=bool)
        for source_index, (keys, sequences, times) in enumerate(
            zip(source_keys, source_sequences, source_times, strict=True)
        ):
            positions = np.searchsorted(keys, union_keys, side="right") - 1
            safe_positions = np.maximum(positions, 0)
            same_sequence = (positions >= 0) & (
                sequences[safe_positions] == union_sequences
            )
            valid &= same_sequence
            latest[:, source_index] = times[safe_positions]
        if not np.any(valid):
            return CompletionEvents(
                sequence_codes=np.zeros(0, dtype=np.int32),
                times=np.zeros(0, dtype=np.int32),
                spans=np.zeros(0, dtype=np.int32),
            )
        latest = latest[valid]
        spans = (np.max(latest, axis=1) - np.min(latest, axis=1)).astype(
            np.int32, copy=False
        )
        if self.max_completion_span is not None:
            within_span = spans <= self.max_completion_span
            latest_sequences = union_sequences[valid][within_span]
            latest_times = union_times[valid][within_span]
            spans = spans[within_span]
        else:
            latest_sequences = union_sequences[valid]
            latest_times = union_times[valid]
        return CompletionEvents(
            sequence_codes=latest_sequences,
            times=latest_times,
            spans=spans,
        )

    def _source_events(self, source: int) -> SourceEvents:
        source = int(source)
        cached = self._source_event_cache.get(source)
        if cached is not None:
            return cached
        rows = np.flatnonzero(self.data.predicates[:, source] > 0)
        sequence_codes = self.data.sequence_codes[rows].astype(np.int32, copy=False)
        times = self.data.times[rows].astype(np.int32, copy=False)
        # Event rows are already ordered by sequence and time. Duplicate
        # source activations at an identical sequence/time represent one state
        # update and are removed exactly as the legacy witness-tuple check did.
        if len(rows) > 1:
            keep = np.ones(len(rows), dtype=bool)
            keep[1:] = (sequence_codes[1:] != sequence_codes[:-1]) | (times[1:] != times[:-1])
            sequence_codes = sequence_codes[keep]
            times = times[keep]
        counts = np.bincount(sequence_codes, minlength=self.data.n_sequences)
        offsets = np.zeros(self.data.n_sequences + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        result = SourceEvents(
            sequence_codes=sequence_codes,
            times=times,
            offsets=offsets,
            populated_sequences=np.flatnonzero(counts).astype(np.int32, copy=False),
        )
        self._source_event_cache[source] = result
        return result

    def window_breakpoints(
        self,
        antecedent: Antecedent,
        fit_sequence_ids: np.ndarray,
        *,
        max_window: int,
        context: QueryContext | None = None,
    ) -> np.ndarray:
        events = (
            self.completions_for_context(context, antecedent)
            if context is not None
            else self.completions(antecedent)
        )
        if len(antecedent) == 1:
            return np.asarray([0], dtype=np.int32) if events.size else np.zeros(0, dtype=np.int32)
        if context is not None:
            keep = events.spans <= int(max_window)
        else:
            selected = np.zeros(self.data.n_sequences, dtype=bool)
            selected[np.asarray(fit_sequence_ids, dtype=np.int64)] = True
            keep = selected[events.sequence_codes] & (events.spans <= int(max_window))
        if not np.any(keep):
            return np.zeros(0, dtype=np.int32)
        return np.unique(events.spans[keep]).astype(np.int32, copy=False)

    def response(self, ctx: QueryContext, antecedent: Antecedent, window: int) -> np.ndarray:
        self._validate_context_identity(ctx)
        key = (ctx.name, tuple(antecedent), int(window))
        with self._cache_guard:
            cached = self._feature_cache.get(key)
            if isinstance(cached, np.ndarray):
                self._feature_cache.move_to_end(key)
                return cached
            key_lock = self._response_locks.setdefault(key, threading.Lock())
        # Different rule terms remain parallel, while duplicate closure terms
        # requested by two GPU workers are built exactly once.
        with key_lock:
            with self._cache_guard:
                cached = self._feature_cache.get(key)
                if isinstance(cached, np.ndarray):
                    self._feature_cache.move_to_end(key)
                    return cached
            result = self._build_response(ctx, tuple(antecedent), int(window))
            self._cache_value(key, result)
            return result

    def projected_response(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        window: int,
        shape: np.ndarray,
    ) -> np.ndarray:
        """Return one dictionary-shape column from the shared bounded LRU."""
        self._validate_context_identity(ctx)
        shape32 = np.asarray(shape, dtype=np.float32).reshape(-1)
        if shape32.shape != (self.knot_count,) or np.any(~np.isfinite(shape32)):
            raise ValueError("projected response shape must be a finite M-vector")
        key = (ctx.name, tuple(antecedent), int(window), shape32.tobytes())
        with self._cache_guard:
            cached = self._feature_cache.get(key)
            if isinstance(cached, np.ndarray):
                self._feature_cache.move_to_end(key)
                return cached
            key_lock = self._response_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._cache_guard:
                cached = self._feature_cache.get(key)
                if isinstance(cached, np.ndarray):
                    self._feature_cache.move_to_end(key)
                    return cached
            result = self.response(ctx, antecedent, int(window)) @ shape32.reshape(-1, 1)
            result = result.astype(np.float32, copy=False)
            self._cache_value(key, result)
            return result

    def sparse_response(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        window: int,
    ) -> SparseKernelResponse:
        """Build a response without allocating an ``n_grid × M`` zero matrix."""
        self._validate_context_identity(ctx)
        key = (ctx.name, tuple(antecedent), int(window), "__sparse_raw__")
        with self._cache_guard:
            cached = self._cached_sparse(key)
            if cached is not None:
                return cached
            key_lock = self._response_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._cache_guard:
                cached = self._cached_sparse(key)
                if cached is not None:
                    return cached
            result = self._build_sparse_response(ctx, tuple(antecedent), int(window))
            return self._intern_sparse(key, result)

    def sparse_horizon_response(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        window: int,
        horizon: int,
    ) -> SparseKernelResponse:
        """Return the exact rule response restricted to the first ``horizon`` lags.

        The discovery model continues to use the complete ``impact_lag``
        response.  This restricted response is only an evaluation contrast for
        the pre-specified early-warning horizon; it never changes the fitted
        objective or the selected support.  Reusing the same completion stream
        also preserves the rule identity and formation-window semantics.
        """
        self._validate_context_identity(ctx)
        horizon = int(horizon)
        if not 1 <= horizon <= self.lag:
            raise ValueError("early-warning horizon must lie in [1, impact_lag]")
        if horizon == self.lag:
            return self.sparse_response(ctx, antecedent, window)
        key = (
            ctx.name,
            tuple(antecedent),
            int(window),
            "__sparse_horizon_raw__",
            horizon,
        )
        with self._cache_guard:
            cached = self._cached_sparse(key)
            if cached is not None:
                return cached
            key_lock = self._response_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._cache_guard:
                cached = self._cached_sparse(key)
                if cached is not None:
                    return cached
            result = self._build_sparse_response(
                ctx,
                tuple(antecedent),
                int(window),
                lag_limit=horizon,
            )
            return self._intern_sparse(key, result)

    def sparse_projected_response(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        window: int,
        shape: np.ndarray,
    ) -> SparseKernelResponse:
        shape32 = np.asarray(shape, dtype=np.float32).reshape(-1)
        if shape32.shape != (self.knot_count,) or np.any(~np.isfinite(shape32)):
            raise ValueError("projected response shape must be a finite M-vector")
        key = (
            ctx.name,
            tuple(antecedent),
            int(window),
            "__sparse_projected__",
            shape32.tobytes(),
        )
        with self._cache_guard:
            cached = self._cached_sparse(key)
            if cached is not None:
                return cached
            key_lock = self._response_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._cache_guard:
                cached = self._cached_sparse(key)
                if cached is not None:
                    return cached
            result = self.sparse_response(
                ctx,
                antecedent,
                int(window),
            ).projected(shape32)
            return self._intern_sparse(key, result)

    def _build_sparse_response(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        window: int,
        *,
        lag_limit: int | None = None,
    ) -> SparseKernelResponse:
        events = self.completions_for_context(ctx, antecedent)
        local_seq = ctx.sequence_lookup[events.sequence_codes]
        keep = (local_seq >= 0) & (events.spans <= int(window))
        local_seq = local_seq[keep].astype(np.int32, copy=False)
        occurrence_times = events.times[keep].astype(np.int64, copy=False)
        grid_indices = np.zeros(0, dtype=np.int64)
        grid_values = np.zeros((0, self.knot_count), dtype=np.float32)
        if len(occurrence_times):
            starts = ctx.start_times[local_seq].astype(np.int64, copy=False)
            ends = ctx.end_times[local_seq].astype(np.int64, copy=False)
            valid = (occurrence_times >= starts) & (occurrence_times <= ends)
            local_seq = local_seq[valid]
            occurrence_times = occurrence_times[valid]
            ends = ends[valid]
            base_indices = (
                ctx.grid_offsets[local_seq]
                + occurrence_times
                - ctx.start_times[local_seq].astype(np.int64, copy=False)
            )
            grid_indices, grid_values = self._sparse_block_from_occurrences(
                base_indices,
                occurrence_times,
                ends,
                lag_limit=lag_limit,
            )
        event_values = np.zeros((ctx.n_events, self.knot_count), dtype=np.float32)
        if ctx.n_events and len(grid_indices):
            event_grid_indices = ctx.event_grid_rows
            positions = np.searchsorted(grid_indices, event_grid_indices)
            safe = np.minimum(positions, len(grid_indices) - 1)
            matched = (positions < len(grid_indices)) & (
                grid_indices[safe] == event_grid_indices
            )
            event_values[matched] = grid_values[positions[matched]]
        if np.any(grid_values < 0.0) or np.any(event_values < 0.0):
            raise AssertionError("rule response must be nonnegative")
        return SparseKernelResponse(
            n_events=ctx.n_events,
            n_grid=ctx.n_grid,
            grid_indices=grid_indices,
            grid_values=grid_values,
            event_values=event_values,
        )

    def _sparse_block_from_occurrences(
        self,
        base_indices: np.ndarray,
        occurrence_times: np.ndarray,
        ends: np.ndarray,
        *,
        lag_limit: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accumulate only destinations reached by the supplied occurrences."""
        if len(occurrence_times) == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, self.knot_count), dtype=np.float32),
            )
        effective_lag = self.lag if lag_limit is None else int(lag_limit)
        if not 1 <= effective_lag <= self.lag:
            raise ValueError("lag limit must lie in [1, impact_lag]")
        basis = self.basis[:, :effective_lag]
        max_cells = 4_000_000
        chunk_size = max(1, max_cells // max(1, effective_lag))
        lags = np.arange(1, effective_lag + 1, dtype=np.int64)
        lag_indices = np.arange(effective_lag, dtype=np.int64)
        chunks: list[tuple[np.ndarray, np.ndarray]] = []
        for left in range(0, len(occurrence_times), chunk_size):
            right = min(left + chunk_size, len(occurrence_times))
            native = sparse_kernel_block(
                base_indices[left:right],
                occurrence_times[left:right],
                ends[left:right],
                basis,
            )
            if native is not None:
                chunks.append(native)
                continue
            valid = occurrence_times[left:right, None] + lags <= ends[left:right, None]
            if not np.any(valid):
                continue
            destinations = (base_indices[left:right, None] + lags)[valid]
            selected_lags = np.broadcast_to(lag_indices, valid.shape)[valid]
            rows, inverse = np.unique(destinations, return_inverse=True)
            values = np.zeros((len(rows), self.knot_count), dtype=np.float32)
            for knot_index in range(self.knot_count):
                coefficients = basis[knot_index, selected_lags].astype(
                    np.float64, copy=False
                )
                if np.any(coefficients):
                    values[:, knot_index] = np.bincount(
                        inverse,
                        weights=coefficients,
                        minlength=len(rows),
                    ).astype(np.float32, copy=False)
            chunks.append((rows.astype(np.int64, copy=False), values))
        if not chunks:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, self.knot_count), dtype=np.float32),
            )
        row_parts = [rows for rows, _values in chunks]
        native_layout = sorted_unique_int64_union_with_positions(
            row_parts,
            assume_sorted=True,
        )
        native_union = (
            native_layout[0]
            if native_layout is not None
            else sorted_unique_int64_union(row_parts)
        )
        indices = (
            native_union
            if native_union is not None
            else np.unique(np.concatenate(row_parts)).astype(np.int64, copy=False)
        )
        values = np.zeros((len(indices), self.knot_count), dtype=np.float32)
        if native_layout is not None:
            for positions, (_rows, chunk_values) in zip(
                native_layout[1], chunks, strict=True
            ):
                values[positions] += chunk_values
        else:
            for rows, chunk_values in chunks:
                values[np.searchsorted(indices, rows)] += chunk_values
        keep = np.any(values != 0.0, axis=1)
        return indices[keep], values[keep]

    @staticmethod
    def _merge_sparse_blocks(
        left_indices: np.ndarray,
        left_values: np.ndarray,
        right_indices: np.ndarray,
        right_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(left_indices):
            return right_indices.copy(), right_values.copy()
        if not len(right_indices):
            return left_indices, left_values
        native_layout = sorted_unique_int64_union_with_positions(
            (left_indices, right_indices),
            assume_sorted=True,
        )
        if native_layout is None:
            indices = np.union1d(left_indices, right_indices).astype(
                np.int64, copy=False
            )
            left_positions = np.searchsorted(indices, left_indices)
            right_positions = np.searchsorted(indices, right_indices)
        else:
            indices, positions = native_layout
            left_positions, right_positions = positions
        values = np.zeros((len(indices), left_values.shape[1]), dtype=np.float32)
        values[left_positions] = left_values
        values[right_positions] += right_values
        keep = np.any(values != 0.0, axis=1)
        return indices[keep], values[keep]

    def iter_window_sparse_responses(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        windows: Sequence[int],
    ):
        """Yield cumulative W responses while touching each completion once.

        Yielded arrays are immutable to the consumer contract but not copied;
        the next iteration allocates a new merged grid block.
        """
        ordered_windows = np.asarray(
            sorted(set(int(value) for value in windows)), dtype=np.int32
        )
        if ordered_windows.size == 0:
            return
        events = self.completions_for_context(ctx, tuple(antecedent))
        local_seq = ctx.sequence_lookup[events.sequence_codes]
        keep = (local_seq >= 0) & (events.spans <= int(ordered_windows[-1]))
        local_seq = local_seq[keep].astype(np.int32, copy=False)
        occurrence_times = events.times[keep].astype(np.int64, copy=False)
        spans = events.spans[keep].astype(np.int32, copy=False)
        starts = ctx.start_times[local_seq].astype(np.int64, copy=False)
        ends = ctx.end_times[local_seq].astype(np.int64, copy=False)
        valid = (occurrence_times >= starts) & (occurrence_times <= ends)
        local_seq = local_seq[valid]
        occurrence_times = occurrence_times[valid]
        spans = spans[valid]
        ends = ends[valid]
        base_indices = (
            ctx.grid_offsets[local_seq]
            + occurrence_times
            - ctx.start_times[local_seq].astype(np.int64, copy=False)
        )
        order = np.argsort(spans, kind="stable")
        spans = spans[order]
        occurrence_times = occurrence_times[order]
        ends = ends[order]
        base_indices = base_indices[order]
        event_grid_indices = ctx.event_grid_rows
        grid_indices = np.zeros(0, dtype=np.int64)
        grid_values = np.zeros((0, self.knot_count), dtype=np.float32)
        left = 0
        for window in ordered_windows.tolist():
            right = int(np.searchsorted(spans, int(window), side="right"))
            if right > left:
                rows, values = self._sparse_block_from_occurrences(
                    base_indices[left:right],
                    occurrence_times[left:right],
                    ends[left:right],
                )
                grid_indices, grid_values = self._merge_sparse_blocks(
                    grid_indices,
                    grid_values,
                    rows,
                    values,
                )
                left = right
            event_values = np.zeros(
                (ctx.n_events, self.knot_count), dtype=np.float32
            )
            if ctx.n_events and len(grid_indices):
                positions = np.searchsorted(grid_indices, event_grid_indices)
                safe = np.minimum(positions, len(grid_indices) - 1)
                matched = (positions < len(grid_indices)) & (
                    grid_indices[safe] == event_grid_indices
                )
                event_values[matched] = grid_values[positions[matched]]
            yield int(window), SparseKernelResponse(
                n_events=ctx.n_events,
                n_grid=ctx.n_grid,
                grid_indices=grid_indices,
                grid_values=grid_values,
                event_values=event_values,
            )

    def _build_response(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        window: int,
    ) -> np.ndarray:
        events = self.completions_for_context(ctx, antecedent)
        local_seq = ctx.sequence_lookup[events.sequence_codes]
        keep = (local_seq >= 0) & (events.spans <= int(window))
        local_seq = local_seq[keep].astype(np.int32, copy=False)
        occurrence_times = events.times[keep].astype(np.int64, copy=False)
        out_grid = np.zeros((ctx.n_grid, self.knot_count), dtype=np.float32)
        if len(occurrence_times):
            starts = ctx.start_times[local_seq].astype(np.int64, copy=False)
            ends = ctx.end_times[local_seq].astype(np.int64, copy=False)
            valid_occurrence = (occurrence_times >= starts) & (occurrence_times <= ends)
            local_seq = local_seq[valid_occurrence]
            occurrence_times = occurrence_times[valid_occurrence]
            ends = ends[valid_occurrence]
            base_indices = (
                ctx.grid_offsets[local_seq]
                + occurrence_times
                - ctx.start_times[local_seq].astype(np.int64, copy=False)
            )
            self._accumulate_grid(
                out_grid,
                base_indices,
                occurrence_times,
                ends,
            )
        if ctx.n_events:
            event_grid_indices = ctx.event_grid_rows
            out_event = out_grid[event_grid_indices, :]
        else:
            out_event = np.zeros((0, self.knot_count), dtype=np.float32)
        result = np.concatenate([out_event, out_grid], axis=0)
        if np.any(result < 0):
            raise AssertionError("rule response must be nonnegative")
        return result

    def _accumulate_grid(
        self,
        out_grid: np.ndarray,
        base_indices: np.ndarray,
        occurrence_times: np.ndarray,
        ends: np.ndarray,
        *,
        capture_updates: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Add a batch of occurrence convolutions using M, not L, bincounts."""
        if len(occurrence_times) == 0:
            return []
        updates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # Bound the temporary occurrence-by-lag block to roughly 64 MiB before
        # NumPy indexing temporaries.  This changes only chunking, never sums.
        max_cells = 4_000_000
        chunk_size = max(1, max_cells // max(1, self.lag))
        lags = np.arange(1, self.lag + 1, dtype=np.int64)
        lag_indices = np.arange(self.lag, dtype=np.int64)
        for left in range(0, len(occurrence_times), chunk_size):
            right = min(left + chunk_size, len(occurrence_times))
            valid = occurrence_times[left:right, None] + lags <= ends[left:right, None]
            if not np.any(valid):
                continue
            destinations = (base_indices[left:right, None] + lags)[valid]
            selected_lags = np.broadcast_to(lag_indices, valid.shape)[valid]
            changed_rows, inverse = np.unique(destinations, return_inverse=True)
            old_values = out_grid[changed_rows, :].copy() if capture_updates else None
            for knot_index in range(self.knot_count):
                coefficients = self.basis[knot_index, selected_lags].astype(np.float64, copy=False)
                if not np.any(coefficients):
                    continue
                counts = np.bincount(
                    inverse,
                    weights=coefficients,
                    minlength=len(changed_rows),
                )
                out_grid[changed_rows, knot_index] += counts.astype(np.float32, copy=False)
            if capture_updates and old_values is not None:
                updates.append((changed_rows, old_values, out_grid[changed_rows, :].copy()))
        return updates

    def iter_window_response_parts(
        self,
        ctx: QueryContext,
        antecedent: Antecedent,
        windows: Sequence[int],
        *,
        capture_updates: bool = False,
    ):
        """Yield cumulative event/grid responses for increasing formation windows.

        The yielded grid array is reused and mutated on the next iteration;
        callers must consume it immediately.  This avoids rebuilding all
        occurrences with span <= W from scratch for every breakpoint.
        """
        ordered_windows = np.asarray(sorted(set(int(value) for value in windows)), dtype=np.int32)
        if ordered_windows.size == 0:
            return
        events = self.completions_for_context(ctx, tuple(antecedent))
        local_seq = ctx.sequence_lookup[events.sequence_codes]
        keep = (local_seq >= 0) & (events.spans <= int(ordered_windows[-1]))
        local_seq = local_seq[keep].astype(np.int32, copy=False)
        occurrence_times = events.times[keep].astype(np.int64, copy=False)
        spans = events.spans[keep].astype(np.int32, copy=False)
        starts = ctx.start_times[local_seq].astype(np.int64, copy=False)
        ends = ctx.end_times[local_seq].astype(np.int64, copy=False)
        valid_occurrence = (occurrence_times >= starts) & (occurrence_times <= ends)
        local_seq = local_seq[valid_occurrence]
        occurrence_times = occurrence_times[valid_occurrence]
        spans = spans[valid_occurrence]
        ends = ends[valid_occurrence]
        base_indices = (
            ctx.grid_offsets[local_seq]
            + occurrence_times
            - ctx.start_times[local_seq].astype(np.int64, copy=False)
        )
        order = np.argsort(spans, kind="stable")
        spans = spans[order]
        occurrence_times = occurrence_times[order]
        ends = ends[order]
        base_indices = base_indices[order]
        event_grid_indices = ctx.event_grid_rows
        out_grid = np.zeros((ctx.n_grid, self.knot_count), dtype=np.float32)
        left = 0
        for window in ordered_windows.tolist():
            right = int(np.searchsorted(spans, int(window), side="right"))
            updates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            if right > left:
                updates = self._accumulate_grid(
                    out_grid,
                    base_indices[left:right],
                    occurrence_times[left:right],
                    ends[left:right],
                    capture_updates=capture_updates,
                )
                left = right
            out_event = (
                out_grid[event_grid_indices, :]
                if ctx.n_events
                else np.zeros((0, self.knot_count), dtype=np.float32)
            )
            yield int(window), out_event, out_grid, updates

    def clear_context_cache(self, context_name: str | None = None) -> None:
        with self._cache_guard:
            if context_name is None:
                self._feature_cache.clear()
                self._feature_cache_bytes = 0
                self._context_sequences.clear()
                self._response_locks.clear()
                self._sparse_equivalence_keys.clear()
                self._sparse_aliases.clear()
                self._sparse_key_fingerprints.clear()
                self._sparse_canonical_aliases.clear()
                return
            self._remove_feature_keys([key for key in self._feature_cache if key[0] == context_name])
            self._remove_feature_keys([
                key
                for key in self._feature_cache
                if len(key) == 3
                and key[0] == "__completion_context__"
                and key[1] == context_name
            ])
            for key in list(self._sparse_aliases):
                if key and key[0] == context_name:
                    self._remove_sparse_metadata(key)
            self._context_sequences.pop(context_name, None)

    def evict_context_completion(
        self,
        context_name: str,
        antecedent: Antecedent,
    ) -> None:
        """Drop a rejected skeleton's completion stream without touching features."""
        key = ("__completion_context__", str(context_name), tuple(antecedent))
        with self._cache_guard:
            self._remove_feature_keys([key])

    def _remove_feature_keys(self, keys: Sequence[tuple]) -> None:
        for key in keys:
            value = self._feature_cache.pop(key, None)
            if value is not None:
                self._feature_cache_bytes -= int(value.nbytes)
            self._response_locks.pop(key, None)
            self._remove_sparse_metadata(key)

    def _remove_alias_keys(self, keys: Sequence[tuple]) -> None:
        """Drop identity-only aliases selected by cache-retention policies."""
        for key in keys:
            if key in self._sparse_aliases:
                self._response_locks.pop(key, None)
                self._remove_sparse_metadata(key)

    def retain_antecedent_windows(
        self,
        context_name: str,
        antecedent: Antecedent,
        windows: Sequence[int],
    ) -> None:
        """Evict response matrices for profiled-but-rejected formation windows."""
        antecedent = tuple(antecedent)
        keep = {int(window) for window in windows}
        with self._cache_guard:
            self._remove_feature_keys([
                key
                for key in self._feature_cache
                if key[0] == context_name and key[1] == antecedent and key[2] not in keep
            ])
            self._remove_alias_keys([
                key
                for key in self._sparse_aliases
                if key[0] == context_name
                and key[1] == antecedent
                and key[2] not in keep
            ])

    def evict_context_terms(
        self,
        context_name: str,
        terms: Sequence[tuple[Antecedent, int]],
    ) -> None:
        remove = {(tuple(antecedent), int(window)) for antecedent, window in terms}
        with self._cache_guard:
            self._remove_feature_keys([
                key
                for key in self._feature_cache
                if key[0] == context_name and (key[1], key[2]) in remove
            ])
            self._remove_alias_keys([
                key
                for key in self._sparse_aliases
                if key[0] == context_name and (key[1], key[2]) in remove
            ])

    def retain_context_terms(
        self,
        context_name: str,
        terms: Sequence[tuple[Antecedent, int]],
    ) -> None:
        keep = {(tuple(antecedent), int(window)) for antecedent, window in terms}
        with self._cache_guard:
            self._remove_feature_keys([
                key
                for key in self._feature_cache
                if key[0] == context_name and (key[1], key[2]) not in keep
            ])
            self._remove_alias_keys([
                key
                for key in self._sparse_aliases
                if key[0] == context_name and (key[1], key[2]) not in keep
            ])

    def rule_name(self, rule: RuleIdentity) -> str:
        names = [self.data.predicate_names[idx] for idx in rule.antecedent]
        sign = "exc" if rule.sign > 0 else "inh"
        window = "singleton" if len(rule.antecedent) == 1 else f"W={rule.window}"
        return f"{' AND '.join(names)} -> target ({sign}, {window})"
