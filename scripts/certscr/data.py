from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as papq

from .native import sorted_grid_sequences


@dataclass(frozen=True)
class TargetMarkCSR:
    """Compact row-aligned storage for sparse event-mark lists."""

    values: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        return max(0, len(self.offsets) - 1)

    def __getitem__(self, row: int) -> np.ndarray:
        left = int(self.offsets[row])
        right = int(self.offsets[row + 1])
        return self.values[left:right]

    def __iter__(self) -> Iterator[np.ndarray]:
        for row in range(len(self)):
            yield self[row]


@dataclass(frozen=True)
class EventData:
    sequence_ids: np.ndarray
    sequence_codes: np.ndarray
    positions: np.ndarray
    times: np.ndarray
    predicates: np.ndarray
    targets: np.ndarray
    predicate_names: tuple[str, ...]
    sequence_slices: tuple[tuple[int, int], ...]
    start_times: np.ndarray
    end_times: np.ndarray
    sequence_split_groups: np.ndarray | None = None
    sequence_start_ages: np.ndarray | None = None
    target_marks: TargetMarkCSR | None = None
    mark_name: str | None = None
    sequence_financial_weights: np.ndarray | None = None
    financial_weight_name: str | None = None
    preprocessing_provenance: dict[str, object] | None = None

    @property
    def n_sequences(self) -> int:
        return len(self.sequence_slices)

    @property
    def n_predicates(self) -> int:
        return int(self.predicates.shape[1])

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        predicate_names: Sequence[str],
        sequence_col: str = "sequence_id",
        position_col: str = "position",
        time_col: str = "month_index",
        target_col: str = "target_token",
        bounds: pd.DataFrame | None = None,
        start_col: str = "start_month",
        end_col: str = "end_month",
        mark_col: str | None = None,
        financial_weight_col: str | None = None,
        split_group_col: str | None = None,
        start_age_col: str | None = None,
        preprocessing_provenance: dict[str, object] | None = None,
    ) -> "EventData":
        predicate_names = tuple(str(v) for v in predicate_names)
        if len(set(predicate_names)) != len(predicate_names):
            raise ValueError("predicate names must be unique")
        reserved = {sequence_col, position_col, time_col, target_col}
        if mark_col is not None:
            reserved.add(mark_col)
        overlap = sorted(set(predicate_names) & reserved)
        if overlap:
            raise ValueError(f"predicate names overlap reserved event columns: {overlap}")
        required = (
            sequence_col,
            position_col,
            time_col,
            target_col,
            *((mark_col,) if mark_col else ()),
            *predicate_names,
        )
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"missing event columns: {missing}")
        bounds_map: dict[str, tuple[int, int]] = {}
        financial_weight_map: dict[str, float] = {}
        split_group_map: dict[str, object] = {}
        start_age_map: dict[str, int] = {}
        if bounds is not None:
            bound_columns = (sequence_col, start_col, end_col)
            for name in (
                *bound_columns,
                *((financial_weight_col,) if financial_weight_col else ()),
                *((split_group_col,) if split_group_col else ()),
                *((start_age_col,) if start_age_col else ()),
            ):
                if name not in bounds.columns:
                    raise ValueError(f"sequence bounds missing {name}")
            selected_bound_columns = [
                *bound_columns,
                *((financial_weight_col,) if financial_weight_col else ()),
                *((split_group_col,) if split_group_col else ()),
                *((start_age_col,) if start_age_col else ()),
            ]
            bound_frame = bounds.loc[:, selected_bound_columns]
            bound_ids = bound_frame[sequence_col].astype("string").astype(str).to_numpy(
                dtype=object
            )
            duplicate = pd.Series(bound_ids, copy=False).duplicated().to_numpy()
            if np.any(duplicate):
                raise ValueError(
                    f"duplicate sequence bounds for {bound_ids[int(np.flatnonzero(duplicate)[0])]}"
                )
            start_values = pd.to_numeric(
                bound_frame[start_col], errors="coerce"
            ).to_numpy(dtype=np.float64)
            end_values = pd.to_numeric(
                bound_frame[end_col], errors="coerce"
            ).to_numpy(dtype=np.float64)
            numeric_bounds = np.column_stack((start_values, end_values))
            int32 = np.iinfo(np.int32)
            invalid_numeric = (
                np.any(~np.isfinite(numeric_bounds), axis=1)
                | np.any(numeric_bounds != np.floor(numeric_bounds), axis=1)
                | np.any(numeric_bounds < int32.min, axis=1)
                | np.any(numeric_bounds > int32.max, axis=1)
            )
            if np.any(invalid_numeric):
                index = int(np.flatnonzero(invalid_numeric)[0])
                raise ValueError(
                    f"bounds for {bound_ids[index]} must be finite int32 values"
                )
            starts_i32 = start_values.astype(np.int32, copy=False)
            ends_i32 = end_values.astype(np.int32, copy=False)
            invalid_order = ends_i32 < starts_i32
            if np.any(invalid_order):
                index = int(np.flatnonzero(invalid_order)[0])
                raise ValueError(
                    f"invalid bounds for {bound_ids[index]}: "
                    f"{(int(starts_i32[index]), int(ends_i32[index]))}"
                )
            bounds_map = dict(
                zip(
                    bound_ids.tolist(),
                    zip(starts_i32.tolist(), ends_i32.tolist(), strict=True),
                    strict=True,
                )
            )
            if financial_weight_col is not None:
                weight_values = pd.to_numeric(
                    bound_frame[financial_weight_col], errors="coerce"
                ).to_numpy(dtype=np.float64)
                invalid_weight = ~np.isfinite(weight_values) | (weight_values < 0.0)
                if np.any(invalid_weight):
                    index = int(np.flatnonzero(invalid_weight)[0])
                    raise ValueError(
                        f"invalid financial weight for {bound_ids[index]}: "
                        f"{weight_values[index]}"
                    )
                financial_weight_map = dict(
                    zip(bound_ids.tolist(), weight_values.tolist(), strict=True)
                )
            if split_group_col is not None:
                group_values = bound_frame[split_group_col]
                if bool(group_values.isna().any()):
                    raise ValueError("sequence split groups must be nonmissing")
                split_group_map = dict(
                    zip(bound_ids.tolist(), group_values.tolist(), strict=True)
                )
            if start_age_col is not None:
                age_values = pd.to_numeric(
                    bound_frame[start_age_col], errors="coerce"
                ).to_numpy(dtype=np.float64)
                invalid_age = (
                    ~np.isfinite(age_values)
                    | (age_values < 0.0)
                    | (age_values != np.floor(age_values))
                    | (age_values > np.iinfo(np.int16).max)
                )
                if np.any(invalid_age):
                    index = int(np.flatnonzero(invalid_age)[0])
                    raise ValueError(
                        f"invalid sequence start age for {bound_ids[index]}: {age_values[index]}"
                    )
                start_age_map = dict(
                    zip(
                        bound_ids.tolist(),
                        age_values.astype(np.int16).tolist(),
                        strict=True,
                    )
                )
        elif financial_weight_col is not None:
            raise ValueError("financial weights require a sequence bounds table")

        df = frame.loc[:, required].copy()
        df[sequence_col] = df[sequence_col].astype("string").astype(str)
        if bounds_map:
            observed_ids = set(df[sequence_col].tolist())
            empty_ids = sorted(set(bounds_map) - observed_ids)
            if empty_ids:
                dummy = {
                    sequence_col: empty_ids,
                    position_col: np.zeros(len(empty_ids), dtype=np.int64),
                    time_col: np.asarray([bounds_map[seq_id][0] for seq_id in empty_ids], dtype=np.int64),
                    target_col: np.zeros(len(empty_ids), dtype=np.int8),
                }
                if mark_col is not None:
                    dummy[mark_col] = [[] for _ in empty_ids]
                for name in predicate_names:
                    dummy[name] = np.zeros(len(empty_ids), dtype=np.uint8)
                df = pd.concat([df, pd.DataFrame(dummy)], ignore_index=True)
        # Sort numerically.  Sorting object/string positions first would put
        # position 10 before 2 and can silently alter event chronology.
        raw_positions = pd.to_numeric(df[position_col], errors="coerce").to_numpy(dtype=np.float64)
        if (
            np.any(~np.isfinite(raw_positions))
            or np.any(raw_positions != np.floor(raw_positions))
            or np.any(raw_positions < 0)
            or np.any(raw_positions >= float(2**63))
        ):
            raise ValueError(f"{position_col} must contain nonnegative finite integers")
        df[position_col] = raw_positions.astype(np.int64, copy=False)
        df = df.sort_values([sequence_col, position_col], kind="mergesort").reset_index(drop=True)

        seq_values = df[sequence_col].to_numpy(dtype=object)
        def integer_values(name: str, dtype: np.dtype, *, minimum: int | None = None) -> np.ndarray:
            numeric = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
                raise ValueError(f"{name} must contain finite integers")
            info = np.iinfo(dtype)
            if np.any(numeric < info.min) or np.any(numeric > info.max):
                raise ValueError(f"{name} is outside the {np.dtype(dtype).name} range")
            if minimum is not None and np.any(numeric < minimum):
                raise ValueError(f"{name} must be at least {minimum}")
            return numeric.astype(dtype, copy=False)

        positions = integer_values(position_col, np.dtype(np.int64), minimum=0)
        times = integer_values(time_col, np.dtype(np.int32))
        targets = integer_values(target_col, np.dtype(np.int32), minimum=0)
        predicates = np.empty((len(df), len(predicate_names)), dtype=np.uint8)
        for column_index, name in enumerate(predicate_names):
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(values)) or np.any(
                (values != 0.0) & (values != 1.0)
            ):
                raise ValueError(f"predicate {name} must contain finite binary values")
            predicates[:, column_index] = values.astype(np.uint8, copy=False)

        target_marks: TargetMarkCSR | None = None
        if mark_col is not None:
            mark_offsets = np.zeros(len(df) + 1, dtype=np.int64)
            mark_offsets[1:] = np.cumsum(targets, dtype=np.int64)
            mark_values = np.empty(int(mark_offsets[-1]), dtype=np.float64)
            # Iterate the existing arrays directly; converting both columns to
            # Python lists duplicates millions of scalar objects on large
            # marked datasets.
            for row_index, (count, value) in enumerate(
                zip(targets, df[mark_col].array, strict=True)
            ):
                if int(count) == 0 and (
                    value is None
                    or (isinstance(value, float) and math.isnan(value))
                    or (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0)
                ):
                    continue
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    marks = np.zeros(0, dtype=np.float64)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    marks = np.asarray(value, dtype=np.float64).reshape(-1)
                else:
                    marks = np.asarray([value], dtype=np.float64)
                if len(marks) != int(count):
                    raise ValueError(
                        f"{mark_col} row {row_index} has {len(marks)} marks for target count {count}"
                    )
                if np.any(~np.isfinite(marks)) or np.any(marks <= 0):
                    raise ValueError(
                        f"{mark_col} must contain one finite, strictly positive financial mark "
                        "per target event"
                    )
                if count:
                    mark_values[mark_offsets[row_index] : mark_offsets[row_index + 1]] = marks
            target_marks = TargetMarkCSR(values=mark_values, offsets=mark_offsets)

        if len(df) == 0:
            raise ValueError("empty event frame")
        boundaries = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(seq_values[1:] != seq_values[:-1]).astype(np.int64) + 1,
                np.asarray([len(df)], dtype=np.int64),
            )
        )
        lefts = boundaries[:-1]
        rights = boundaries[1:]
        lengths = rights - lefts
        ids = [str(value) for value in seq_values[lefts].tolist()]
        codes = np.repeat(np.arange(len(lefts), dtype=np.int32), lengths)
        same_sequence = seq_values[1:] == seq_values[:-1]
        time_differences = np.diff(times)
        position_differences = np.diff(positions)
        invalid_time = np.flatnonzero(same_sequence & (time_differences < 0))
        if invalid_time.size:
            raise ValueError(f"times are not ordered in {seq_values[int(invalid_time[0])]}")
        invalid_position = np.flatnonzero(same_sequence & (position_differences != 1))
        if invalid_position.size:
            raise ValueError(
                f"positions are not consecutive in {seq_values[int(invalid_position[0])]}"
            )
        observed_starts = times[lefts]
        observed_ends = times[rights - 1]
        if bounds_map:
            missing_bounds = sorted(set(ids) - set(bounds_map))
            if missing_bounds:
                raise ValueError(f"event sequence is missing from the authoritative bounds table: {missing_bounds[0]}")
            bound_values = np.asarray(
                [
                    bounds_map[seq_id]
                    for seq_id in ids
                ],
                dtype=np.int64,
            )
            starts_array = bound_values[:, 0]
            ends_array = bound_values[:, 1]
        else:
            starts_array = observed_starts.astype(np.int64, copy=False)
            ends_array = observed_ends.astype(np.int64, copy=False)
        invalid_bounds = np.flatnonzero(
            (starts_array > observed_starts) | (ends_array < observed_ends)
        )
        if invalid_bounds.size:
            index = int(invalid_bounds[0])
            raise ValueError(f"bounds do not contain observations for {ids[index]}")
        slices = tuple((int(left), int(right)) for left, right in zip(lefts, rights, strict=True))
        financial_weights = []
        if financial_weight_col is not None:
            missing_weights = [seq_id for seq_id in ids if seq_id not in financial_weight_map]
            if missing_weights:
                raise ValueError(f"missing financial weight for {missing_weights[0]}")
            financial_weights = [financial_weight_map[seq_id] for seq_id in ids]

        return cls(
            sequence_ids=np.asarray(ids, dtype=object),
            sequence_codes=codes,
            positions=positions,
            times=times,
            predicates=predicates,
            targets=targets,
            predicate_names=predicate_names,
            sequence_slices=tuple(slices),
            start_times=np.asarray(starts_array, dtype=np.int32),
            end_times=np.asarray(ends_array, dtype=np.int32),
            sequence_split_groups=(
                np.asarray([split_group_map[seq_id] for seq_id in ids])
                if split_group_col is not None
                else None
            ),
            sequence_start_ages=(
                np.asarray(
                    [start_age_map[seq_id] for seq_id in ids],
                    dtype=np.int16,
                )
                if start_age_col is not None
                else None
            ),
            target_marks=target_marks,
            mark_name=mark_col,
            sequence_financial_weights=(
                np.asarray(financial_weights, dtype=np.float64) if financial_weight_col is not None else None
            ),
            financial_weight_name=financial_weight_col,
            preprocessing_provenance=preprocessing_provenance,
        )


class ImplicitUnitGridWeights:
    """Array-compatible unit quadrature weights with zero O(n_grid) storage."""

    def __init__(self, size: int):
        self.size = int(size)
        self.shape = (self.size,)
        self.dtype = np.dtype(np.float32)
        self.nbytes = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, key: object) -> np.ndarray | np.float32:
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if not -self.size <= index < self.size:
                raise IndexError(index)
            return np.float32(1.0)
        if isinstance(key, slice):
            start, stop, step = key.indices(self.size)
            return np.ones(len(range(start, stop, step)), dtype=np.float32)
        requested = np.asarray(key)
        return np.ones(requested.shape, dtype=np.float32)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return np.ones(self.shape, dtype=(np.float32 if dtype is None else dtype))

    def astype(
        self,
        dtype: np.dtype,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        del order, casting, subok, copy
        return np.ones(self.shape, dtype=dtype)


class ImplicitGridSequence:
    """Array-compatible sequence labels derived exactly from contiguous offsets."""

    def __init__(self, offsets: np.ndarray):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.size = int(self.offsets[-1])
        self.shape = (self.size,)
        self.dtype = np.dtype(np.int32)
        self.nbytes = 0

    def __len__(self) -> int:
        return self.size

    def at(
        self,
        rows: np.ndarray,
        *,
        assume_sorted: bool = False,
    ) -> np.ndarray:
        requested = np.asarray(rows, dtype=np.int64)
        if assume_sorted:
            native = sorted_grid_sequences(self.offsets, requested)
            if native is not None:
                return native
        return np.searchsorted(
            self.offsets[1:], requested, side="right"
        ).astype(np.int32, copy=False)

    def __getitem__(self, key: object) -> np.ndarray | np.int32:
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += self.size
            if not 0 <= index < self.size:
                raise IndexError(index)
            return np.int32(np.searchsorted(self.offsets[1:], index, side="right"))
        if isinstance(key, slice):
            start, stop, step = key.indices(self.size)
            rows = np.arange(start, stop, step, dtype=np.int64)
        else:
            raw = np.asarray(key)
            rows = (
                np.flatnonzero(raw).astype(np.int64, copy=False)
                if raw.dtype == np.bool_
                else raw.astype(np.int64, copy=False)
            )
        return self.at(rows)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        lengths = np.diff(self.offsets)
        values = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        return values.astype(dtype, copy=False) if dtype is not None else values


@dataclass(frozen=True)
class QueryContext:
    name: str
    global_sequence_ids: np.ndarray
    sequence_lookup: np.ndarray
    event_sequence_local: np.ndarray
    event_times: np.ndarray
    event_marks: np.ndarray | None
    grid_sequence_local: np.ndarray | ImplicitGridSequence
    # Absolute grid times are not consumed after the sequence-local offsets
    # have been built.  Keeping them would cost another int32 vector over the
    # entire exposure grid (often hundreds of millions of rows).
    grid_times: np.ndarray | None
    grid_weights: np.ndarray | ImplicitUnitGridWeights
    grid_offsets: np.ndarray
    start_times: np.ndarray
    end_times: np.ndarray
    # These two vectors depend only on the immutable split geometry.  They are
    # consumed by every support fit / response construction, so constructing
    # them once here avoids O(number-of-supports) identical allocations.
    sequence_exposure_values: np.ndarray
    event_grid_rows: np.ndarray

    @property
    def n_sequences(self) -> int:
        return int(len(self.global_sequence_ids))

    @property
    def n_events(self) -> int:
        return int(len(self.event_times))

    @property
    def n_grid(self) -> int:
        return int(self.grid_offsets[-1])

    @property
    def n_queries(self) -> int:
        return self.n_events + self.n_grid

    @property
    def exposure(self) -> float:
        return float(np.sum(self.sequence_exposures(), dtype=np.float64))

    def grid_sequences_at(
        self,
        indices: np.ndarray,
        *,
        assume_valid: bool = False,
        assume_sorted: bool = False,
    ) -> np.ndarray:
        rows = np.asarray(indices, dtype=np.int64)
        if not assume_valid and (np.any(rows < 0) or np.any(rows >= self.n_grid)):
            raise IndexError("grid row is out of range")
        if isinstance(self.grid_sequence_local, ImplicitGridSequence):
            return self.grid_sequence_local.at(
                rows, assume_sorted=assume_sorted
            )
        return np.asarray(self.grid_sequence_local[rows], dtype=np.int32)

    def grid_weights_at(
        self,
        indices: np.ndarray,
        *,
        assume_valid: bool = False,
    ) -> np.ndarray:
        rows = np.asarray(indices, dtype=np.int64)
        if not assume_valid and (np.any(rows < 0) or np.any(rows >= self.n_grid)):
            raise IndexError("grid row is out of range")
        if isinstance(self.grid_weights, ImplicitUnitGridWeights):
            return np.ones(rows.shape, dtype=np.float64)
        return np.asarray(self.grid_weights[rows], dtype=np.float64)

    def sequence_exposures(self) -> np.ndarray:
        return self.sequence_exposure_values

    def aggregate_grid(self, values: np.ndarray) -> np.ndarray:
        """Sum one full-grid vector per sequence without a sequence-id vector."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.n_grid,):
            raise ValueError("grid values do not match query context")
        return np.add.reduceat(array, self.grid_offsets[:-1])

    def aggregate_weighted_grid(self, values: np.ndarray) -> np.ndarray:
        """Sum quadrature-weighted grid values per sequence."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.n_grid,):
            raise ValueError("grid values do not match query context")
        if not isinstance(self.grid_weights, ImplicitUnitGridWeights):
            array = array * np.asarray(self.grid_weights, dtype=np.float64)
        return np.add.reduceat(array, self.grid_offsets[:-1])

    def expand_sequence_values(self, values: np.ndarray) -> np.ndarray:
        """Expand one scalar per sequence without storing grid sequence labels."""
        array = np.asarray(values)
        if array.shape != (self.n_sequences,):
            raise ValueError("sequence values do not match query context")
        return np.repeat(array, np.diff(self.grid_offsets))


@dataclass(frozen=True)
class ThreeWayContexts:
    fit: QueryContext
    cert: QueryContext
    test: QueryContext
    split_seed: int
    fractions: tuple[float, float, float]
    fit_sampling_weights: np.ndarray
    fit_population_global_ids: np.ndarray
    fit_population_sequence_count: int
    fit_population_negative_count: int
    fit_sampled_negative_count: int
    split_strategy: str = "random_sequence"
    split_groups: tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]] | None = None


def _parquet_schema_names(path: Path) -> tuple[str, ...]:
    if path.is_file():
        return tuple(papq.ParquetFile(path).schema_arrow.names)
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {path}")
    return tuple(papq.ParquetFile(files[0]).schema_arrow.names)


def _read_parquet_dir(
    path: Path,
    columns: Sequence[str],
    *,
    sequence_col: str | None = None,
    selected_sequence_ids: Sequence[object] | None = None,
) -> pd.DataFrame:
    dataset = pads.dataset(str(path), format="parquet")
    filter_expression = None
    if selected_sequence_ids is not None:
        if sequence_col is None:
            raise ValueError("sequence_col is required for a filtered parquet scan")
        selected = list(selected_sequence_ids)
        if not selected:
            return pd.DataFrame(columns=list(columns))
        filter_expression = pads.field(sequence_col).isin(selected)
    table = dataset.to_table(columns=list(columns), filter=filter_expression, use_threads=True)
    return table.to_pandas(split_blocks=True, self_destruct=True)


def load_event_data(
    path: str | Path,
    *,
    predicate_names: Sequence[str] | None = None,
    max_sequences: int | None = None,
    sample_seed: int = 111,
    sequence_col: str = "sequence_id",
    position_col: str = "position",
    time_col: str = "month_index",
    target_col: str = "target_token",
    mark_col: str | None = None,
    financial_weight_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    split_group_col: str | None = None,
    start_age_col: str | None = None,
) -> EventData:
    path = Path(path)
    if max_sequences is not None and max_sequences < 1:
        raise ValueError("max_sequences must be positive when supplied")
    probe = path if path.is_file() else next(iter(sorted(path.glob("*.parquet"))), None)
    if probe is None:
        raise FileNotFoundError(path)
    all_columns = _parquet_schema_names(probe)
    dataset_root = path.parent if path.is_dir() else path.parent.parent
    summary_path = dataset_root / "metadata" / "summary.json"
    preprocessing_provenance: dict[str, object] | None = None
    if summary_path.is_file():
        try:
            raw_provenance = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid preprocessing metadata: {summary_path}") from exc
        if not isinstance(raw_provenance, dict):
            raise ValueError(f"preprocessing metadata must be a JSON object: {summary_path}")
        preprocessing_provenance = raw_provenance
    if predicate_names is None:
        predicate_names = tuple(name for name in all_columns if str(name).startswith("pred_"))
    predicate_names = tuple(predicate_names)
    if not predicate_names:
        raise ValueError("empty predicate set")
    columns = (
        sequence_col,
        position_col,
        time_col,
        target_col,
        *((mark_col,) if mark_col else ()),
        *predicate_names,
    )
    bounds = None
    resolved_start_col = start_col
    resolved_end_col = end_col
    sibling = path.parent / "sequences" if path.is_dir() else path.parent.parent / "sequences"
    if sibling.is_dir():
        bound_probe = next(iter(sorted(sibling.glob("*.parquet"))), None)
        if bound_probe is None:
            raise FileNotFoundError(f"no parquet files in {sibling}")
        available_bounds = set(_parquet_schema_names(bound_probe))
        if (resolved_start_col is None) != (resolved_end_col is None):
            raise ValueError("start_col and end_col must be supplied together")
        if resolved_start_col is None:
            for candidate_start, candidate_end in (
                ("start_month_index", "end_month_index"),
                ("start_month", "end_month"),
            ):
                if candidate_start in available_bounds and candidate_end in available_bounds:
                    resolved_start_col, resolved_end_col = candidate_start, candidate_end
                    break
        if resolved_start_col is None or resolved_end_col is None:
            raise ValueError("sequence bounds require a recognized start/end column pair")
        bound_columns = [sequence_col, resolved_start_col, resolved_end_col]
        if financial_weight_col is not None:
            bound_columns.append(financial_weight_col)
        if split_group_col is not None:
            bound_columns.append(split_group_col)
        if start_age_col is not None:
            bound_columns.append(start_age_col)
        bounds = _read_parquet_dir(sibling, bound_columns)

    # In a sparse event stream a valid exposed sequence may have no event row.
    # The sequence table, when present, is therefore the authoritative entity
    # sampling frame.  Sampling from event rows would condition on observing an
    # event and bias both the baseline and financial loss.
    sampling_frame = (
        bounds
        if bounds is not None
        else _read_parquet_dir(path, [sequence_col])
    )
    unique_ids = sampling_frame[sequence_col].drop_duplicates().to_numpy(dtype=object)
    selected_ids: np.ndarray | None = None
    if max_sequences is not None and 0 < max_sequences < len(unique_ids):
        rng = np.random.default_rng(sample_seed)
        selected_ids = rng.choice(unique_ids, size=max_sequences, replace=False)
        chosen = set(selected_ids.tolist())
        if bounds is not None:
            bounds = bounds.loc[bounds[sequence_col].isin(chosen)].copy()
    frame = _read_parquet_dir(
        path,
        columns,
        sequence_col=sequence_col,
        selected_sequence_ids=selected_ids,
    )
    frame[sequence_col] = frame[sequence_col].astype("string").astype(str)
    if bounds is not None:
        bounds[sequence_col] = bounds[sequence_col].astype("string").astype(str)
    return EventData.from_frame(
        frame,
        predicate_names=predicate_names,
        sequence_col=sequence_col,
        position_col=position_col,
        time_col=time_col,
        target_col=target_col,
        bounds=bounds,
        start_col=resolved_start_col or "start_month",
        end_col=resolved_end_col or "end_month",
        mark_col=mark_col,
        financial_weight_col=financial_weight_col,
        split_group_col=split_group_col,
        start_age_col=start_age_col,
        preprocessing_provenance=preprocessing_provenance,
    )


def make_context(data: EventData, name: str, sequence_ids: np.ndarray) -> QueryContext:
    global_ids = np.asarray(sorted(set(int(v) for v in sequence_ids)), dtype=np.int32)
    if global_ids.size == 0:
        raise ValueError(f"empty split: {name}")
    lookup = np.full(data.n_sequences, -1, dtype=np.int32)
    lookup[global_ids] = np.arange(len(global_ids), dtype=np.int32)

    target_rows = np.flatnonzero((data.targets > 0) & (lookup[data.sequence_codes] >= 0))
    # Multiple target transactions may share an account/time bin.  Repeating
    # their query row gives the exact point-process event term while preserving
    # the compact, one-row-per-account-time parquet representation.
    event_global_rows = np.repeat(target_rows, data.targets[target_rows].astype(np.int64))
    event_local = lookup[data.sequence_codes[event_global_rows]].astype(np.int32, copy=False)
    event_times = data.times[event_global_rows].astype(np.int32, copy=False)
    if data.target_marks is None:
        event_marks = None
    else:
        event_count = int(len(event_global_rows))
        if event_count:
            repeated_group_starts = np.repeat(
                np.cumsum(data.targets[target_rows], dtype=np.int64)
                - data.targets[target_rows],
                data.targets[target_rows],
            )
            within_row = np.arange(event_count, dtype=np.int64) - repeated_group_starts
            mark_indices = data.target_marks.offsets[event_global_rows] + within_row
            event_marks = data.target_marks.values[mark_indices].astype(np.float64, copy=False)
        else:
            event_marks = np.zeros(0, dtype=np.float64)
        if len(event_marks) != len(event_global_rows):
            raise ValueError("expanded target marks do not match expanded target events")

    starts = data.start_times[global_ids].astype(np.int32, copy=False)
    ends = data.end_times[global_ids].astype(np.int32, copy=False)
    lengths = (ends.astype(np.int64) - starts.astype(np.int64) + 1)
    if np.any(lengths <= 0):
        raise ValueError(f"nonpositive sequence exposure in {name}")
    offsets = np.zeros(len(global_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total_grid = int(offsets[-1])
    sequence_exposure_values = lengths.astype(np.float64, copy=False)
    event_grid_rows = (
        offsets[event_local]
        + event_times.astype(np.int64, copy=False)
        - starts[event_local].astype(np.int64, copy=False)
        if len(event_times)
        else np.zeros(0, dtype=np.int64)
    )
    sequence_exposure_values.setflags(write=False)
    event_grid_rows.setflags(write=False)
    return QueryContext(
        name=name,
        global_sequence_ids=global_ids,
        sequence_lookup=lookup,
        event_sequence_local=event_local,
        event_times=event_times,
        event_marks=event_marks,
        grid_sequence_local=ImplicitGridSequence(offsets),
        grid_times=None,
        grid_weights=ImplicitUnitGridWeights(total_grid),
        grid_offsets=offsets,
        start_times=starts,
        end_times=ends,
        sequence_exposure_values=sequence_exposure_values,
        event_grid_rows=event_grid_rows,
    )


def split_contexts(
    data: EventData,
    *,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
    seed: int = 111,
    stratify_target: bool = False,
    fit_negative_sample_size: int | None = None,
    strategy: str = "random_sequence",
) -> ThreeWayContexts:
    n_parts = 3
    if len(fractions) != n_parts or any(v <= 0 for v in fractions) or not math.isclose(sum(fractions), 1.0):
        raise ValueError("three positive split fractions must sum to one")
    if data.n_sequences < n_parts:
        raise ValueError("at least three sequences are required for three-way split")
    rng = np.random.default_rng(seed)
    def apportioned_sizes(total: int) -> list[int]:
        # Hamilton apportionment with a one-sequence lower bound gives the
        # closest integer split while guaranteeing nonempty partitions.
        if total < n_parts:
            raise ValueError("each split class needs at least three sequences")
        raw_sizes = np.asarray(fractions, dtype=np.float64) * total
        sizes = np.maximum(np.floor(raw_sizes).astype(np.int64), 1)
        while int(np.sum(sizes)) > total:
            eligible = np.flatnonzero(sizes > 1)
            chosen = int(eligible[np.argmax(sizes[eligible] - raw_sizes[eligible])])
            sizes[chosen] -= 1
        while int(np.sum(sizes)) < total:
            chosen = int(np.argmax(raw_sizes - sizes))
            sizes[chosen] += 1
        return sizes.tolist()

    desired_sizes = apportioned_sizes(data.n_sequences)

    split_groups: tuple[
        tuple[object, ...], tuple[object, ...], tuple[object, ...]
    ] | None = None
    if strategy not in {"random_sequence", "ordered_group"}:
        raise ValueError("split strategy must be random_sequence or ordered_group")
    if strategy == "ordered_group":
        if stratify_target:
            raise ValueError(
                "ordered-group splitting cannot also stratify on the target"
            )
        if data.sequence_split_groups is None:
            raise ValueError(
                "ordered-group splitting requires sequence split-group metadata"
            )
        group_values = np.asarray(data.sequence_split_groups)
        if group_values.shape != (data.n_sequences,):
            raise ValueError("sequence split groups do not align with entities")
        unique_groups = np.unique(group_values)
        if len(unique_groups) < n_parts:
            raise ValueError("ordered-group splitting requires at least three groups")
        group_members = [
            np.flatnonzero(group_values == group) for group in unique_groups
        ]
        target_sizes = np.asarray(desired_sizes, dtype=np.float64)
        best: tuple[float, int, int, np.ndarray] | None = None
        for first_cut in range(1, len(unique_groups) - 1):
            for second_cut in range(first_cut + 1, len(unique_groups)):
                sizes = np.asarray(
                    [
                        sum(len(v) for v in group_members[:first_cut]),
                        sum(len(v) for v in group_members[first_cut:second_cut]),
                        sum(len(v) for v in group_members[second_cut:]),
                    ],
                    dtype=np.int64,
                )
                score = float(
                    np.sum(
                        ((sizes.astype(np.float64) - target_sizes) / data.n_sequences)
                        ** 2
                    )
                )
                candidate = (score, first_cut, second_cut, sizes)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        assert best is not None
        _score, first_cut, second_cut, _sizes = best
        grouped_parts = (
            group_members[:first_cut],
            group_members[first_cut:second_cut],
            group_members[second_cut:],
        )
        parts = [
            np.concatenate(values).astype(np.int64, copy=False)
            for values in grouped_parts
        ]
        split_groups = (
            tuple(unique_groups[:first_cut].tolist()),
            tuple(unique_groups[first_cut:second_cut].tolist()),
            tuple(unique_groups[second_cut:].tolist()),
        )
        target_counts = np.bincount(
            data.sequence_codes,
            weights=data.targets.astype(np.float64),
            minlength=data.n_sequences,
        )
        if any(not np.any(target_counts[part] > 0) for part in parts):
            raise ValueError(
                "ordered-group split produced a target-free evaluation partition"
            )
    elif stratify_target:
        target_counts = np.bincount(
            data.sequence_codes,
            weights=data.targets.astype(np.float64),
            minlength=data.n_sequences,
        )
        positive = np.flatnonzero(target_counts > 0)
        negative = np.flatnonzero(target_counts <= 0)
        if len(positive) < n_parts:
            raise ValueError(
                "marked/target-stratified three-way evaluation requires at least "
                "one target-positive sequence per split"
            )
        # Respect the exact overall split sizes while keeping at least one
        # positive sequence in each part.  Requiring three negatives was an
        # unnecessary restriction: an all-positive marked cohort is valid.
        raw_positive = np.asarray(fractions, dtype=np.float64) * len(positive)
        positive_sizes_array = np.clip(
            np.floor(raw_positive).astype(np.int64),
            1,
            np.asarray(desired_sizes, dtype=np.int64),
        )
        while int(np.sum(positive_sizes_array)) < len(positive):
            capacity = positive_sizes_array < np.asarray(desired_sizes, dtype=np.int64)
            if not np.any(capacity):
                raise AssertionError("positive split allocation exhausted capacity")
            deficit = raw_positive - positive_sizes_array
            deficit[~capacity] = -math.inf
            positive_sizes_array[int(np.argmax(deficit))] += 1
        while int(np.sum(positive_sizes_array)) > len(positive):
            removable = positive_sizes_array > 1
            excess = positive_sizes_array - raw_positive
            excess[~removable] = -math.inf
            positive_sizes_array[int(np.argmax(excess))] -= 1
        positive_sizes = positive_sizes_array.tolist()
        negative_sizes = [
            int(total - positive_count)
            for total, positive_count in zip(
                desired_sizes, positive_sizes, strict=True
            )
        ]
        if sum(negative_sizes) != len(negative) or any(size < 0 for size in negative_sizes):
            raise AssertionError("stratified split allocation is inconsistent")
        positive_parts = np.split(
            rng.permutation(positive), np.cumsum(positive_sizes[:-1]).tolist()
        )
        negative_parts = np.split(
            rng.permutation(negative), np.cumsum(negative_sizes[:-1]).tolist()
        )
        parts = [
            rng.permutation(np.concatenate([positive_part, negative_part]))
            for positive_part, negative_part in zip(
                positive_parts, negative_parts, strict=True
            )
        ]
    else:
        order = rng.permutation(data.n_sequences)
        cuts = np.cumsum(desired_sizes[:-1])
        parts = list(np.split(order, cuts.tolist()))
    if any(len(part) == 0 for part in parts):
        raise ValueError("three-way split produced an empty partition")
    fit_population_ids = np.asarray(parts[0], dtype=np.int64)
    fit_population_count = int(len(fit_population_ids))
    fit_target_counts = np.bincount(
        data.sequence_codes,
        weights=data.targets.astype(np.float64),
        minlength=data.n_sequences,
    )[fit_population_ids]
    fit_positive = fit_population_ids[fit_target_counts > 0]
    fit_negative = fit_population_ids[fit_target_counts <= 0]
    fit_population_negative_count = int(len(fit_negative))
    sampled_negative_count = fit_population_negative_count
    sampling_weight_by_global = np.ones(data.n_sequences, dtype=np.float64)
    if fit_negative_sample_size is not None:
        if fit_negative_sample_size < 1:
            raise ValueError("fit negative sample size must be positive")
        if fit_negative_sample_size < fit_population_negative_count:
            sampled_negative = rng.choice(
                fit_negative,
                size=int(fit_negative_sample_size),
                replace=False,
            )
            sampled_negative_count = int(len(sampled_negative))
            negative_weight = float(fit_population_negative_count / sampled_negative_count)
            sampling_weight_by_global[sampled_negative] = negative_weight
            parts[0] = rng.permutation(np.concatenate([fit_positive, sampled_negative]))
    fit_ctx = make_context(data, "fit", parts[0])
    fit_sampling_weights = sampling_weight_by_global[fit_ctx.global_sequence_ids]
    return ThreeWayContexts(
        fit=fit_ctx,
        cert=make_context(data, "cert", parts[1]),
        test=make_context(data, "test", parts[2]),
        split_seed=seed,
        fractions=fractions,
        fit_sampling_weights=fit_sampling_weights,
        fit_population_global_ids=fit_population_ids.astype(np.int32, copy=True),
        fit_population_sequence_count=fit_population_count,
        fit_population_negative_count=fit_population_negative_count,
        fit_sampled_negative_count=sampled_negative_count,
        split_strategy=strategy,
        split_groups=split_groups,
    )
