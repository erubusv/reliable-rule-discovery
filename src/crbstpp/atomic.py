from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Hashable, Iterable

import numpy as np

from .likelihood import is_poisson_likelihood
from .native import aggregate_design_rows, response_min_spans, sorted_unique_union
from .objective import ObjectiveSpec
from .response import Context, ModelMatrix, ResponseEngine
from .rules import PatternKey, Support, normalize_pattern


@dataclass(frozen=True)
class AtomicSubtreeCertificate:
    """A certified score envelope for an add-descendant support subtree.

    ``required`` is present in every descendant.  Each antecedent in
    ``optional_antecedents`` may be absent or may contribute one of its
    preregistered W/sign identities.  The likelihood relaxation gives every
    *observed atomic history signature* its own predictor.  This contains all
    exact total-state descendants while retaining equality between rows that
    no admissible descendant can distinguish.
    """

    required: Support
    optional_antecedents: tuple[PatternKey, ...]
    nll_lower_bound: float
    penalty_lower_bound: float
    score_upper_bound: float
    coarse_score_upper_bound: float
    refined_components: int
    saturated_components: int
    atomic_groups: int
    active_rows: int
    peak_workspace_bytes: int

    @property
    def tightened(self) -> bool:
        return self.score_upper_bound < self.coarse_score_upper_bound


@dataclass(frozen=True)
class SidetrackItem:
    key: Hashable
    upper_score: float
    lower_score: float
    payload: object


class CertifiedSidetrackQueue:
    """Deterministic best-upper-bound queue with exact key deduplication.

    Queue ordering is an acceleration only.  An item is removed solely when
    its certified upper endpoint is below ``threshold``.  Reaching the same
    state through another route updates the existing envelope rather than
    repeating its suffix.
    """

    def __init__(self, threshold: float, *, tolerance: float = 0.0):
        self.threshold = float(threshold)
        self.tolerance = max(0.0, float(tolerance))
        self._serial = count()
        self._heap: list[tuple[float, int, Hashable]] = []
        self._items: dict[Hashable, SidetrackItem] = {}
        self.pruned = 0
        self.merged = 0

    def push(
        self,
        key: Hashable,
        *,
        upper_score: float,
        lower_score: float,
        payload: object,
    ) -> bool:
        upper = float(upper_score)
        lower = float(lower_score)
        # ``+inf`` is the correct fail-open endpoint: it must stay in the
        # frontier.  Only NaN, ``-inf`` or a finite non-improving endpoint can
        # be discarded.
        if math.isnan(upper) or upper <= self.threshold + self.tolerance:
            self.pruned += 1
            return False
        incumbent = self._items.get(key)
        if incumbent is not None:
            self.merged += 1
            same_upper = upper == incumbent.upper_score or (
                math.isfinite(upper)
                and math.isfinite(incumbent.upper_score)
                and abs(upper - incumbent.upper_score) <= self.tolerance
            )
            if upper < incumbent.upper_score - self.tolerance or (
                same_upper and lower <= incumbent.lower_score + self.tolerance
            ):
                return False
        item = SidetrackItem(key, upper, lower, payload)
        self._items[key] = item
        heappush(self._heap, (-upper, next(self._serial), key))
        return True

    def pop(self) -> SidetrackItem | None:
        while self._heap:
            negative_upper, _, key = heappop(self._heap)
            item = self._items.get(key)
            if item is None or not math.isclose(
                -negative_upper,
                item.upper_score,
                rel_tol=0.0,
                abs_tol=self.tolerance,
            ):
                continue
            del self._items[key]
            return item
        return None

    @property
    def remaining_upper_score(self) -> float:
        while self._heap:
            negative_upper, _, key = self._heap[0]
            item = self._items.get(key)
            if item is not None and math.isclose(
                -negative_upper,
                item.upper_score,
                rel_tol=0.0,
                abs_tol=self.tolerance,
            ):
                return item.upper_score
            heappop(self._heap)
        return -math.inf

    def __len__(self) -> int:
        return len(self._items)


def minimum_descendant_penalty(
    objective: ObjectiveSpec,
    required: Support,
    optional_antecedents: Iterable[PatternKey],
) -> float:
    """Return the exact minimum MDL code over an add-descendant subtree.

    The support-size code is not assumed monotone.  Every feasible final
    cardinality is scanned, while additive identity costs are handled by one
    sort.  This is ``O(K log K)`` rather than support enumeration.
    """

    required_antecedents = required.patterns
    optional = tuple(
        sorted(
            {
                normalize_pattern(pattern)
                for pattern in optional_antecedents
                if normalize_pattern(pattern) not in required_antecedents
            }
        )
    )
    required_count = len(required.rules)
    if required_count > objective.skeleton_count:
        raise ValueError("required support exceeds the skeleton dictionary")
    if required_count + len(optional) > objective.skeleton_count:
        raise ValueError("descendant subtree exceeds the skeleton dictionary")
    required_identity = sum(
        2.0 * math.log(
            2 * objective.window_count(rule.antecedent, rule.relation)
        )
        for rule in required.rules
    )
    optional_identity = sorted(
        2.0 * math.log(2 * objective.window_count(pattern[1], pattern[0]))
        for pattern in optional
    )
    prefix = np.zeros(len(optional_identity) + 1, dtype=np.float64)
    if optional_identity:
        prefix[1:] = np.cumsum(optional_identity, dtype=np.float64)
    best = math.inf
    for added in range(len(optional_identity) + 1):
        size = required_count + added
        if size == 0:
            candidate = 0.0
        else:
            parameter_code = objective.knot_count * size * math.log(
                max(2, objective.n_entities)
            )
            support_code = 2.0 * (
                math.lgamma(objective.skeleton_count + 1)
                - math.lgamma(size + 1)
                - math.lgamma(objective.skeleton_count - size + 1)
            )
            candidate = (
                parameter_code
                + support_code
                + required_identity
                + float(prefix[added])
            )
        best = min(best, candidate)
    return float(best)


def _group_saturated_nll(
    exposure: np.ndarray,
    noevent: np.ndarray,
    event: np.ndarray,
    *,
    likelihood: str,
) -> float:
    exposure = np.maximum(np.asarray(exposure, dtype=np.float64), 0.0)
    noevent = np.maximum(np.asarray(noevent, dtype=np.float64), 0.0)
    event = np.maximum(np.asarray(event, dtype=np.float64), 0.0)
    if is_poisson_likelihood(likelihood):
        positive = event > 0.0
        if np.any(positive & (exposure <= 0.0)):
            return -math.inf
        output = np.zeros(np.broadcast_shapes(exposure.shape, event.shape))
        finite = positive & (exposure > 0.0)
        output[finite] = event[finite] * (
            1.0 - np.log(event[finite] / exposure[finite])
        )
        return float(np.sum(output))
    if likelihood != "first_event_cloglog":
        raise ValueError(f"unknown likelihood: {likelihood}")
    mixed = (event > 0.0) & (noevent > 0.0)
    if not np.any(mixed):
        return 0.0
    total = event[mixed] + noevent[mixed]
    probability = event[mixed] / total
    return float(
        np.sum(
            -event[mixed] * np.log(probability)
            - noevent[mixed] * np.log1p(-probability)
        )
    )


def _score_upper(
    *, baseline_nll: float, nll_lower_bound: float, penalty_lower_bound: float
) -> float:
    """Convert certified lower bounds into a score upper endpoint.

    The ordinary reportable-support helper intentionally rejects every
    non-finite fitted NLL.  A relaxation is different: ``-inf`` means that a
    finite lower endpoint could not be certified and therefore maps to
    ``+inf`` (fail open), never to a pruning score.
    """

    if math.isnan(nll_lower_bound) or nll_lower_bound == -math.inf:
        return math.inf
    if nll_lower_bound == math.inf:
        return -math.inf
    return float(2.0 * (baseline_nll - nll_lower_bound) - penalty_lower_bound)


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        value = int(value)
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class AtomicSignatureRelaxation:
    """Closed-form, dominance-aware relaxation for total-state descendants.

    Rows are grouped only when their required-support design and every raw
    optional W block are identical.  For any optional support, its total-state
    masking is a deterministic function of that signature.  Hence all exact
    descendants assign the same predictor within a group, whereas this
    relaxation may choose an arbitrary predictor for each group.  The model
    family is therefore contained without constructing a tall universal
    design or running Newton iterations.

    Overlap components that exceed ``workspace_bytes`` fail open to independent
    row saturation.  This weakens the bound but cannot change a pruning result
    incorrectly.
    """

    def __init__(
        self,
        context: Context,
        engine: ResponseEngine,
        objective: ObjectiveSpec,
        *,
        baseline_nll: float,
        window_dictionary: dict[PatternKey, tuple[int, ...]],
        workspace_bytes: int,
        global_nll_lower_bound: float,
    ):
        self.context = context
        self.engine = engine
        self.objective = objective
        self.baseline_nll = float(baseline_nll)
        self.window_dictionary = {
            normalize_pattern(pattern): tuple(sorted(set(map(int, windows))))
            for pattern, windows in window_dictionary.items()
        }
        self.workspace_bytes = max(1, int(workspace_bytes))
        self.global_nll_lower_bound = float(global_nll_lower_bound)
        self._footprints: dict[PatternKey, np.ndarray] = {}
        self._primitive_covers: dict[tuple[int, ...], np.ndarray] = {}

    def _windows(self, pattern: PatternKey) -> tuple[int, ...]:
        pattern = normalize_pattern(pattern)
        windows = self.window_dictionary.get(pattern)
        if windows is None:
            raise ValueError(f"pattern is outside the frozen dictionary: {pattern}")
        return windows

    def _footprint(self, pattern: PatternKey) -> np.ndarray:
        cached = self._footprints.get(pattern)
        if cached is not None:
            return cached
        windows = self._windows(pattern)
        # ``response_row_thresholds`` is the canonical exact representation
        # shared by every W in the frozen latest-witness dictionary.  Its row
        # vector at max(W) is precisely the union of all smaller-W footprints;
        # requesting each W separately rebuilt and merged the same completion
        # expansion up to |W| times per skeleton on Aave.
        rows, _ = self.engine.response_row_thresholds(
            self.context,
            pattern[1],
            max(windows),
            relation=pattern[0],
        )
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        self._footprints[pattern] = rows
        return rows

    def _primitive_cover(self, predicates: tuple[int, ...]) -> np.ndarray:
        """Return the exact union of all singleton future footprints.

        Expanding each singleton and then sorting their union repeats the same
        dense strict-future scan once per predicate.  The union is exactly the
        response footprint of the combined primitive event stream, so one
        compiled minimum-span pass returns the identical row set.  Falling
        back to the former construction preserves correctness on platforms
        without the compiled operator.
        """

        predicates = tuple(sorted(set(map(int, predicates))))
        cached = self._primitive_covers.get(predicates)
        if cached is not None:
            return cached
        dataset = self.context.dataset
        allowed = np.zeros(dataset.n_predicates, dtype=np.bool_)
        allowed[np.asarray(predicates, dtype=np.int64)] = True
        event_keep = allowed[dataset.event_predicates]
        source_entities = dataset.event_entities[event_keep]
        local = self.context.entity_lookup[source_entities]
        keep = local >= 0
        entities = np.ascontiguousarray(local[keep], dtype=np.int64)
        times = np.ascontiguousarray(dataset.event_times[event_keep][keep], dtype=np.int64)
        compiled = response_min_spans(
            entities,
            times,
            np.zeros(len(entities), dtype=np.int64),
            self.context.starts,
            self.context.ends,
            self.context.offsets,
            horizon=self.engine.lag_units * dataset.ticks_per_unit,
            n_grid=self.context.n_grid,
        )
        if compiled is None:
            parts = [self._footprint(("atomic", (predicate,))) for predicate in predicates]
            nonempty = [rows for rows in parts if len(rows)]
            rows = sorted_unique_union(nonempty)
            if rows is None:
                rows = (
                    np.unique(np.concatenate(nonempty))
                    if nonempty
                    else np.zeros(0, dtype=np.int64)
                )
        else:
            rows = compiled[0]
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        self._primitive_covers[predicates] = rows
        return rows

    @staticmethod
    def _default_group(matrix: ModelMatrix) -> int | None:
        if not len(matrix.x):
            return None
        nonintercept = matrix.x[:, 1:]
        candidates = np.flatnonzero(
            np.all(np.abs(nonintercept) <= 1.0e-14, axis=1)
        )
        if len(candidates) != 1:
            return None
        return int(candidates[0])

    def _groups_at_rows(
        self,
        matrix: ModelMatrix,
        rows: np.ndarray,
        default_group: int,
    ) -> np.ndarray:
        groups = np.full(len(rows), int(default_group), dtype=np.int64)
        if not len(rows) or not len(matrix.active_rows):
            return groups
        positions = np.searchsorted(matrix.active_rows, rows)
        matched = positions < len(matrix.active_rows)
        safe = np.minimum(positions, len(matrix.active_rows) - 1)
        matched &= matrix.active_rows[safe] == rows
        groups[matched] = matrix.active_design_groups[positions[matched]]
        return groups

    def _row_weights(
        self, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entity_weight = self.context.weights_at_rows(rows)
        exposure = self.engine.tick_exposure * entity_weight
        event = self.context.target_counts_at_sorted_rows(rows)
        noevent = (
            exposure - event
            if self.context.dataset.likelihood == "first_event_cloglog"
            else exposure.copy()
        )
        scale = max(
            1.0,
            float(np.max(np.abs(exposure), initial=0.0)),
            float(np.max(np.abs(event), initial=0.0)),
        )
        tolerance = 128.0 * np.finfo(np.float64).eps * scale
        if np.min(noevent, initial=0.0) < -tolerance:
            raise AssertionError("atomic relaxation produced negative no-event mass")
        return exposure, np.maximum(noevent, 0.0), event

    def _independent_row_nll(self, rows: np.ndarray) -> float:
        """Saturated loss without allocating dense row-weight vectors."""

        if not len(rows) or self.context.dataset.likelihood == "first_event_cloglog":
            return 0.0
        targets = self.context.target_rows
        if not len(targets):
            return 0.0
        positions = np.searchsorted(rows, targets)
        matched = positions < len(rows)
        safe = np.minimum(positions, len(rows) - 1)
        matched &= rows[safe] == targets
        if not np.any(matched):
            return 0.0
        target_rows = targets[matched]
        event = self.context.target_counts[matched]
        exposure = self.engine.tick_exposure * self.context.weights_at_rows(
            target_rows
        )
        return _group_saturated_nll(
            exposure,
            exposure,
            event,
            likelihood="poisson",
        )

    def _subtract_rows(
        self,
        matrix: ModelMatrix,
        rows: np.ndarray,
        groups: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        exposure, noevent, event = self._row_weights(rows)
        count = len(matrix.x)
        removed_exposure = np.bincount(
            groups, weights=exposure, minlength=count
        ).astype(np.float64)
        removed_noevent = np.bincount(
            groups, weights=noevent, minlength=count
        ).astype(np.float64)
        removed_event = np.bincount(
            groups, weights=event, minlength=count
        ).astype(np.float64)
        residual = (
            matrix.exposure_weight - removed_exposure,
            matrix.noevent_weight - removed_noevent,
            matrix.event_weight - removed_event,
        )
        scale = max(
            1.0,
            *(float(np.max(np.abs(value), initial=0.0)) for value in residual),
        )
        tolerance = 1.0e-9 * scale
        if any(np.min(value, initial=0.0) < -tolerance for value in residual):
            raise ArithmeticError("atomic row-to-group map is inconsistent")
        return tuple(np.maximum(value, 0.0) for value in residual)  # type: ignore[return-value]

    def _signature_nll(
        self,
        matrix: ModelMatrix,
        rows: np.ndarray,
        antecedents: tuple[PatternKey, ...],
        default_group: int,
    ) -> tuple[float, int, int]:
        if not len(rows):
            return 0.0, 0, 0
        dimension = 1 + sum(
            len(self._windows(antecedent)) * (self.engine.knot_count + 1)
            for antecedent in antecedents
        )
        estimated = self._signature_workspace_bytes(len(rows), dimension)
        if estimated > self.workspace_bytes:
            return (
                self._independent_row_nll(rows),
                len(rows),
                0,
            )
        signature = np.zeros((len(rows), dimension), dtype=np.float64)
        signature[:, 0] = self._groups_at_rows(matrix, rows, default_group)
        left = 1
        for pattern in antecedents:
            for window in self._windows(pattern):
                block = self.engine.block(
                    self.context, pattern[1], window, pattern[0]
                )
                destination = slice(left + 1, left + 1 + self.engine.knot_count)
                if len(block.rows):
                    positions = np.searchsorted(rows, block.rows)
                    matched = positions < len(rows)
                    safe = np.minimum(positions, len(rows) - 1)
                    matched &= rows[safe] == block.rows
                    if not np.all(matched):
                        # ``rows`` is a component union and must contain every
                        # row of every member block.  Fail open rather than
                        # accepting a potentially unsafe merged signature.
                        return (
                            self._independent_row_nll(rows),
                            len(rows),
                            0,
                        )
                    signature[positions, left] = 1.0
                    signature[positions, destination] = block.values
                left = destination.stop
        exposure, noevent, event = self._row_weights(rows)
        grouped, exposure, noevent, event = aggregate_design_rows(
            signature,
            exposure,
            noevent,
            event,
            copy_input=False,
        )
        return (
            _group_saturated_nll(
                exposure,
                noevent,
                event,
                likelihood=self.context.dataset.likelihood,
            ),
            len(grouped),
            estimated,
        )

    @staticmethod
    def _signature_workspace_bytes(row_count: int, dimension: int) -> int:
        # Dense signature plus three likelihood vectors, required-group IDs,
        # native row hashes and conservative hash-table/equality bookkeeping.
        # The estimate is intentionally high: crossing the cap merely selects
        # the mathematically safe row-saturated fallback.
        return int(row_count) * (8 * int(dimension) + 128)

    def complete_signature_fits(
        self, antecedents: Iterable[PatternKey]
    ) -> bool:
        """Whether the complete atomic map fits the declared workspace.

        ``context.n_grid`` is an exact upper bound on its row count.  Returning
        false merely defers to candidate-wise safe bounds and the exact
        one-exchange audit; it never discards a support.
        """

        patterns = tuple(sorted({normalize_pattern(item) for item in antecedents}))
        dimension = 1 + sum(
            len(self._windows(pattern)) * (self.engine.knot_count + 1)
            for pattern in patterns
        )
        return (
            self._signature_workspace_bytes(self.context.n_grid, dimension)
            <= self.workspace_bytes
        )

    def bound(
        self,
        required: Support,
        required_matrix: ModelMatrix,
        optional_antecedents: Iterable[PatternKey],
        *,
        prune_threshold: float | None = None,
    ) -> AtomicSubtreeCertificate:
        if required_matrix.support != required or required_matrix.closure:
            raise ValueError("atomic bound requires the exact total-state matrix")
        if required_matrix.x.shape[0] == 0:
            raise ValueError("atomic bound cannot use a metadata-only matrix")
        optional = tuple(
            sorted(
                {
                    normalize_pattern(pattern)
                    for pattern in optional_antecedents
                    if normalize_pattern(pattern) not in required.patterns
                    and self._windows(normalize_pattern(pattern))
                }
            )
        )
        penalty = minimum_descendant_penalty(self.objective, required, optional)
        default_group = self._default_group(required_matrix)
        if default_group is None:
            # The all-row saturated likelihood is a universal fail-open bound.
            lower_nll = self.global_nll_lower_bound
            upper = _score_upper(
                baseline_nll=self.baseline_nll,
                nll_lower_bound=lower_nll,
                penalty_lower_bound=penalty,
            )
            return AtomicSubtreeCertificate(
                required,
                optional,
                lower_nll,
                penalty,
                upper,
                upper,
                0,
                1,
                len(self.context.target_rows),
                self.context.n_grid,
                0,
            )
        # Decision-level coarse map.  A higher-order latest-witness completion
        # occurs at an event time of one of its primitive predicates.  Its
        # strictly-future response is therefore contained in the union of the
        # corresponding singleton response footprints.  This cover needs at
        # most P expansions instead of K=P+choose(P,2)+choose(P,3), while row
        # saturation on a superset remains a valid descendant relaxation.
        #
        # When the full optional signature cannot fit the declared workspace,
        # an overlap-component refinement may be tighter but requires the
        # expensive K-by-footprint incidence map.  Return the coarse endpoint
        # instead.  This is an execution fail-open, not a candidate rejection:
        # an overlapping bound keeps the region and the ordinary candidate
        # oracle remains responsible for it.
        if prune_threshold is not None and optional:
            primitive = tuple(
                sorted(
                    {
                        predicate
                        for pattern in optional
                        for predicate in pattern[1]
                    }
                )
            )
            complete_cover = all(
                ("atomic", (int(predicate),)) in self.window_dictionary
                for predicate in primitive
            )
            if complete_cover:
                cover = self._primitive_cover(primitive)
                cover_groups = self._groups_at_rows(
                    required_matrix, cover, default_group
                )
                try:
                    cover_residual = self._subtract_rows(
                        required_matrix, cover, cover_groups
                    )
                except ArithmeticError:
                    cover_residual = None
                if cover_residual is not None:
                    cover_nll = _group_saturated_nll(
                        *cover_residual,
                        likelihood=self.context.dataset.likelihood,
                    ) + self._independent_row_nll(cover)
                    cover_upper = _score_upper(
                        baseline_nll=self.baseline_nll,
                        nll_lower_bound=cover_nll,
                        penalty_lower_bound=penalty,
                    )
                    complete_dimension = 1 + sum(
                        len(self._windows(antecedent))
                        * (self.engine.knot_count + 1)
                        for antecedent in optional
                    )
                    refinement_fits = (
                        self._signature_workspace_bytes(
                            len(cover), complete_dimension
                        )
                        <= self.workspace_bytes
                    )
                    if (
                        math.isfinite(cover_upper)
                        and (
                            cover_upper <= float(prune_threshold)
                            or not refinement_fits
                        )
                    ):
                        return AtomicSubtreeCertificate(
                            required,
                            optional,
                            cover_nll,
                            penalty,
                            cover_upper,
                            cover_upper,
                            0,
                            1,
                            len(cover),
                            len(cover),
                            0,
                        )
        footprints = [self._footprint(antecedent) for antecedent in optional]
        nonempty = [rows for rows in footprints if len(rows)]
        union = sorted_unique_union(nonempty)
        if union is None:
            union = (
                np.unique(np.concatenate(nonempty))
                if nonempty
                else np.zeros(0, dtype=np.int64)
            )
        if not len(union):
            lower_nll = _group_saturated_nll(
                required_matrix.exposure_weight,
                required_matrix.noevent_weight,
                required_matrix.event_weight,
                likelihood=self.context.dataset.likelihood,
            )
            upper = _score_upper(
                baseline_nll=self.baseline_nll,
                nll_lower_bound=lower_nll,
                penalty_lower_bound=penalty,
            )
            return AtomicSubtreeCertificate(
                required,
                optional,
                lower_nll,
                penalty,
                upper,
                upper,
                0,
                0,
                len(required_matrix.x),
                0,
                0,
            )

        required_groups = self._groups_at_rows(
            required_matrix, union, default_group
        )
        try:
            residual = self._subtract_rows(required_matrix, union, required_groups)
        except ArithmeticError:
            # Never turn a bookkeeping failure into a candidate rejection.
            lower_nll = self.global_nll_lower_bound
            upper = _score_upper(
                baseline_nll=self.baseline_nll,
                nll_lower_bound=lower_nll,
                penalty_lower_bound=penalty,
            )
            return AtomicSubtreeCertificate(
                required,
                optional,
                lower_nll,
                penalty,
                upper,
                upper,
                0,
                1,
                len(self.context.target_rows),
                len(union),
                0,
            )
        residual_nll = _group_saturated_nll(
            *residual,
            likelihood=self.context.dataset.likelihood,
        )
        coarse_active_nll = self._independent_row_nll(union)
        coarse_nll = residual_nll + coarse_active_nll
        coarse_upper = _score_upper(
            baseline_nll=self.baseline_nll,
            nll_lower_bound=coarse_nll,
            penalty_lower_bound=penalty,
        )
        # Progressive tightening must stop as soon as a valid level proves
        # the requested decision.  Building the overlap graph below requires
        # one subset-to-union lookup for every optional skeleton; on a large
        # financial panel that can dominate the complete search even though a
        # coarse saturated endpoint has already closed the region.  Returning
        # this looser certificate is exact-safe because it is itself an upper
        # bound on every descendant.  Calls without a threshold retain the
        # fully refined diagnostic behavior used by exhaustive tests.
        if (
            prune_threshold is not None
            and math.isfinite(coarse_upper)
            and coarse_upper <= float(prune_threshold)
        ):
            return AtomicSubtreeCertificate(
                required,
                optional,
                coarse_nll,
                penalty,
                coarse_upper,
                coarse_upper,
                0,
                1,
                len(union),
                len(union),
                0,
            )

        # Build the exact overlap graph in one sparse pass.  Components have
        # disjoint row unions, so each can be refined and released separately.
        owner = np.full(len(union), -1, dtype=np.int32)
        touch_count = np.zeros(len(union), dtype=np.uint16)
        sets = _DisjointSet(len(optional))
        positions_by_index: list[np.ndarray] = []
        for index, rows in enumerate(footprints):
            positions = np.searchsorted(union, rows)
            positions_by_index.append(positions)
            previous = owner[positions]
            for other in np.unique(previous[previous >= 0]):
                sets.union(index, int(other))
            empty = previous < 0
            owner[positions[empty]] = index
            touch_count[positions] = np.minimum(
                np.iinfo(np.uint16).max,
                touch_count[positions].astype(np.uint32) + 1,
            ).astype(np.uint16)
        components: dict[int, list[int]] = {}
        for index, footprint in enumerate(footprints):
            # Empty atoms still participate in the finite support-code lower
            # bound, but change no likelihood term and need no component.
            if not len(footprint):
                continue
            components.setdefault(sets.find(index), []).append(index)

        active_nll = 0.0
        refined_components = 0
        saturated_components = 0
        atomic_groups = 0
        peak_workspace = 0
        for indices in components.values():
            component_rows = sorted_unique_union(
                [footprints[index] for index in indices if len(footprints[index])]
            )
            if component_rows is None:
                component_rows = np.unique(
                    np.concatenate(
                        [
                            footprints[index]
                            for index in indices
                            if len(footprints[index])
                        ]
                    )
                )
            antecedents = tuple(optional[index] for index in indices)
            dimension = 1 + sum(
                len(self._windows(antecedent)) * (self.engine.knot_count + 1)
                for antecedent in antecedents
            )
            estimate = self._signature_workspace_bytes(
                len(component_rows), dimension
            )
            if estimate <= self.workspace_bytes:
                nll, groups, workspace = self._signature_nll(
                    required_matrix,
                    component_rows,
                    antecedents,
                    default_group,
                )
                active_nll += nll
                atomic_groups += groups
                peak_workspace = max(peak_workspace, workspace)
                refined_components += 1
                continue

            # Memory-bounded level-1 refinement: exclusive footprints retain
            # their complete W signature, while genuinely overlapping rows
            # remain independently saturated.  This is still a strict
            # refinement of the coarse all-row relaxation whenever repeated
            # exclusive signatures occur.
            component_positions = np.searchsorted(union, component_rows)
            multiple = touch_count[component_positions] > 1
            if np.any(multiple):
                multi_rows = component_rows[multiple]
                active_nll += self._independent_row_nll(multi_rows)
                atomic_groups += len(multi_rows)
            for index in indices:
                rows = footprints[index]
                positions = positions_by_index[index]
                exclusive = rows[touch_count[positions] == 1]
                nll, groups, workspace = self._signature_nll(
                    required_matrix,
                    exclusive,
                    (optional[index],),
                    default_group,
                )
                active_nll += nll
                atomic_groups += groups
                peak_workspace = max(peak_workspace, workspace)
            saturated_components += 1

        refined_nll = residual_nll + active_nll
        # A refinement can differ by a few ulps after regrouping.  Taking the
        # smaller upper score is safe only when both NLL values are valid lower
        # bounds; equivalently use the larger certified NLL lower bound.
        lower_nll = max(coarse_nll, refined_nll)
        upper = _score_upper(
            baseline_nll=self.baseline_nll,
            nll_lower_bound=lower_nll,
            penalty_lower_bound=penalty,
        )
        upper = min(upper, coarse_upper)
        return AtomicSubtreeCertificate(
            required,
            optional,
            lower_nll,
            penalty,
            upper,
            coarse_upper,
            refined_components,
            saturated_components,
            atomic_groups,
            len(union),
            peak_workspace,
        )
