from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator


Antecedent = tuple[int, ...]
Relation = str
PatternKey = tuple[Relation, Antecedent]
HistoryMark = tuple[int, int]


def normalize_relation(antecedent: Antecedent, relation: str) -> str:
    """Return the canonical temporal relation for one rule pattern."""
    value = str(relation)
    if value == "auto":
        return "atomic" if len(antecedent) == 1 else "unordered"
    return value


def normalize_pattern(
    value: PatternKey | Antecedent,
    relation: str = "auto",
) -> PatternKey:
    """Return an explicit v13 pattern key.

    Fixed-support helpers historically accepted ``(A, B)`` directly.  This
    compatibility input remains supported while all internal/cache keys carry
    an explicit temporal relation.
    """
    if (
        len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], tuple)
    ):
        antecedent = tuple(map(int, value[1]))
        return (normalize_relation(antecedent, str(value[0])), antecedent)
    antecedent = tuple(map(int, value))
    return (normalize_relation(antecedent, relation), antecedent)


@dataclass(frozen=True, order=True)
class RuleIdentity:
    antecedent: Antecedent
    window: int
    sign: int
    # Zero denotes the configured full M-knot kernel.  One denotes the
    # normalized one-amplitude kernel used by adaptive MDL refinement.  The
    # sentinel keeps legacy checkpoints/configurations backward compatible.
    kernel_rank: int = 0
    # ``atomic`` is required for singleton rules.  Higher-order rules are
    # either latest-witness unordered ANDs or strict temporal sequences.
    # ``auto`` is accepted only as a construction compatibility sentinel and
    # is normalized immediately, so serialized identities are unambiguous.
    relation: Relation = "auto"
    # A higher-order additive identity owns a signed interaction block and
    # requires all strict lower-order states as shared nuisance main effects.
    # The flag is serialized with the identity so v13 total-state checkpoints
    # can never be interpreted as v14 additive models.
    hierarchical: bool = False
    # Optional target-blind refinement of each antecedent atom.  ``(L, c)``
    # means that the atom occurring at time t is retained only when the same
    # primitive predicate occurred at least c times in [t-L, t).  The current
    # event is therefore never counted.  An empty tuple is the ordinary
    # unmarked event representation.  Marks live in the rule identity rather
    # than in the dataset predicate dictionary, so P primitive predicates do
    # not become P x L x c materialized state predicates.
    history_marks: tuple[HistoryMark, ...] = ()
    # No automatic lower-order closure and no nested-state masking.  The
    # coefficient is conditional only on the other reported rules selected in
    # the same support.  Serializing this bit prevents the v16 contract from
    # being decoded as either v13 total-state or v14 automatic hierarchy.
    support_additive: bool = False

    def __post_init__(self) -> None:
        relation = normalize_relation(self.antecedent, self.relation)
        object.__setattr__(self, "relation", relation)
        if not 1 <= len(self.antecedent) <= 3:
            raise ValueError("antecedent order must be in [1, 3]")
        if len(set(self.antecedent)) != len(self.antecedent):
            raise ValueError("antecedent predicates must be unique")
        if relation not in {"atomic", "unordered", "ordered"}:
            raise ValueError("relation must be atomic, unordered or ordered")
        if len(self.antecedent) == 1 and relation != "atomic":
            raise ValueError("singleton relation must be atomic")
        if len(self.antecedent) > 1 and relation == "atomic":
            raise ValueError("higher-order relation cannot be atomic")
        if relation == "unordered" and tuple(sorted(self.antecedent)) != self.antecedent:
            raise ValueError("unordered antecedent must be sorted")
        if self.window < 0 or (len(self.antecedent) == 1 and self.window != 0):
            raise ValueError(
                "singleton window is zero; higher-order windows are nonnegative"
            )
        if self.sign not in {-1, 1}:
            raise ValueError("sign must be -1 or +1")
        if self.kernel_rank not in {0, 1}:
            raise ValueError("kernel_rank must be 0 (full) or 1 (scalar)")
        if relation == "ordered" and self.window == 0:
            raise ValueError("strictly ordered rules require a positive window")
        if self.hierarchical and len(self.antecedent) == 1:
            raise ValueError("singleton rules cannot be hierarchy modifiers")
        if self.hierarchical and self.support_additive:
            raise ValueError(
                "automatic hierarchy and support-relative additive semantics "
                "are mutually exclusive"
            )
        marks = tuple((int(window), int(count)) for window, count in self.history_marks)
        if marks and len(marks) != len(self.antecedent):
            raise ValueError("history marks must align with antecedent atoms")
        if any(
            (window, count) != (0, 0) and (window < 1 or count < 1)
            for window, count in marks
        ):
            raise ValueError(
                "each atom history mark must be (0,0) or a positive lookback/count"
            )
        if marks and not any(mark != (0, 0) for mark in marks):
            marks = ()
        object.__setattr__(self, "history_marks", marks)

    @property
    def order(self) -> int:
        return len(self.antecedent)

    def kernel_dimension(self, knot_count: int) -> int:
        """Return this rule's actual number of fitted kernel parameters."""
        return int(knot_count) if self.kernel_rank == 0 else 1

    def with_kernel_rank(self, kernel_rank: int) -> "RuleIdentity":
        return RuleIdentity(
            self.antecedent,
            self.window,
            self.sign,
            kernel_rank,
            self.relation,
            self.hierarchical,
            self.history_marks,
            self.support_additive,
        )

    @property
    def is_history_marked(self) -> bool:
        return bool(self.history_marks)

    @property
    def response_geometry(self) -> tuple:
        """Return the unsigned response identity used by exact matrix caches."""
        return (
            self.relation,
            self.antecedent,
            int(self.window),
            int(self.kernel_rank),
            self.history_marks,
            bool(self.support_additive),
        )

    @property
    def pattern_key(self) -> PatternKey:
        return (self.relation, self.antecedent)


@dataclass(frozen=True, order=True)
class ClosureTerm:
    antecedent: Antecedent
    window: int
    history_marks: tuple[HistoryMark, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.antecedent
            or tuple(sorted(set(self.antecedent))) != self.antecedent
        ):
            raise ValueError("invalid closure antecedent")
        if len(self.antecedent) == 1 and self.window != 0:
            raise ValueError("singleton closure window must be zero")
        marks = tuple((int(window), int(count)) for window, count in self.history_marks)
        if marks and len(marks) != len(self.antecedent):
            raise ValueError("closure history marks must align with antecedent")
        if any(
            mark != (0, 0) and (mark[0] < 1 or mark[1] < 1)
            for mark in marks
        ):
            raise ValueError("invalid closure history mark")
        if marks and not any(mark != (0, 0) for mark in marks):
            marks = ()
        object.__setattr__(self, "history_marks", marks)


@dataclass(frozen=True)
class Support:
    rules: tuple[RuleIdentity, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.rules))
        if ordered != self.rules or len(set(ordered)) != len(ordered):
            raise ValueError("support rules must be sorted and unique")
        patterns = [rule.pattern_key for rule in ordered]
        if len(set(patterns)) != len(patterns):
            raise ValueError(
                "one support cannot contain two identities of one temporal pattern"
            )
        if any(rule.support_additive for rule in ordered) and not all(
            rule.support_additive for rule in ordered
        ):
            raise ValueError(
                "support-relative additive identities cannot be mixed with "
                "total-state or automatic-hierarchy identities"
            )

    @classmethod
    def of(cls, rules: Iterable[RuleIdentity]) -> "Support":
        return cls(tuple(sorted(rules)))

    @property
    def antecedents(self) -> tuple[Antecedent, ...]:
        return tuple(rule.antecedent for rule in self.rules)

    @property
    def patterns(self) -> tuple[PatternKey, ...]:
        return tuple(rule.pattern_key for rule in self.rules)

    def add(self, rule: RuleIdentity) -> "Support":
        return Support.of((*self.rules, rule))

    def drop(self, rule: RuleIdentity) -> "Support":
        return Support.of(item for item in self.rules if item != rule)

    def replace(self, old: RuleIdentity, new: RuleIdentity) -> "Support":
        return Support.of(new if item == old else item for item in self.rules)


EMPTY_SUPPORT = Support(())


def skeletons(n_predicates: int, q_max: int) -> tuple[Antecedent, ...]:
    return tuple(
        combination
        for order in range(1, min(3, int(q_max)) + 1)
        for combination in itertools.combinations(range(int(n_predicates)), order)
    )


def identities_for(
    antecedent: Antecedent,
    formation_windows: tuple[int, ...],
    relation: Relation = "auto",
    *,
    additive_hierarchy: bool = False,
    support_additive: bool = False,
) -> tuple[RuleIdentity, ...]:
    relation = normalize_relation(antecedent, relation)
    windows = (0,) if len(antecedent) == 1 else formation_windows
    if relation == "ordered":
        windows = tuple(window for window in windows if int(window) > 0)
    return tuple(
        RuleIdentity(
            antecedent,
            int(window),
            sign,
            relation=relation,
            hierarchical=bool(additive_hierarchy and len(antecedent) > 1),
            support_additive=bool(support_additive),
        )
        for window in windows
        for sign in (-1, 1)
    )


def temporal_patterns(
    n_predicates: int,
    q_max: int,
    relations: tuple[str, ...] = ("unordered",),
) -> tuple[PatternKey, ...]:
    """Return the finite atomic/unordered/ordered structural dictionary."""
    requested = tuple(dict.fromkeys(map(str, relations)))
    if any(value not in {"unordered", "ordered"} for value in requested):
        raise ValueError("temporal relations must contain unordered and/or ordered")
    output: list[PatternKey] = [
        ("atomic", (predicate,)) for predicate in range(int(n_predicates))
    ]
    maximum = min(3, int(q_max))
    for order in range(2, maximum + 1):
        for combination in itertools.combinations(range(int(n_predicates)), order):
            if "unordered" in requested:
                output.append(("unordered", combination))
            if "ordered" in requested:
                output.extend(("ordered", value) for value in itertools.permutations(combination))
    return tuple(output)


def state_aware_temporal_patterns(
    event_predicates: tuple[int, ...],
    state_predicates: tuple[int, ...],
    q_max: int,
    relations: tuple[str, ...] = ("unordered",),
) -> tuple[PatternKey, ...]:
    """Return the finite v14 event/state rule grammar.

    Primitive events retain singleton/pair/triplet motifs.  A predictable
    history state may stand alone or contextualize exactly one current event.
    State-state and state-plus-two-event motifs are deliberately excluded:
    the state already summarizes a temporal history, and this restriction
    prevents an automatic state lift from turning P primitive predicates into
    an uninterpretable O(P^3) duplicate dictionary.
    """

    events = tuple(sorted(set(map(int, event_predicates))))
    states = tuple(sorted(set(map(int, state_predicates))))
    if set(events).intersection(states):
        raise ValueError("event and state predicate dictionaries must be disjoint")
    requested = tuple(dict.fromkeys(map(str, relations)))
    if any(value not in {"unordered", "ordered"} for value in requested):
        raise ValueError("invalid temporal relation")
    output: list[PatternKey] = [
        *(('atomic', (predicate,)) for predicate in (*events, *states)),
    ]
    maximum = min(3, int(q_max))
    for order in range(2, maximum + 1):
        for combination in itertools.combinations(events, order):
            if "unordered" in requested:
                output.append(("unordered", combination))
            if "ordered" in requested:
                output.extend(
                    ("ordered", value) for value in itertools.permutations(combination)
                )
    if maximum >= 2 and "unordered" in requested:
        output.extend(
            ("unordered", tuple(sorted((state, event))))
            for state in states
            for event in events
        )
    return tuple(output)


def rule_dictionary(
    n_predicates: int,
    q_max: int,
    formation_windows: tuple[int, ...],
    relations: tuple[str, ...] = ("unordered",),
) -> tuple[RuleIdentity, ...]:
    return tuple(
        identity
        for relation, antecedent in temporal_patterns(n_predicates, q_max, relations)
        for identity in identities_for(antecedent, formation_windows, relation)
    )


@lru_cache(maxsize=262_144)
def hierarchy_closure(support: Support) -> tuple[ClosureTerm, ...]:
    """Return shared, complexity-counted additive lower-order main effects."""

    reported = {
        (rule.antecedent, rule.window, rule.history_marks)
        for rule in support.rules
        if rule.relation != "ordered"
    }
    required: set[ClosureTerm] = set()
    for rule in support.rules:
        if not rule.hierarchical:
            continue
        if rule.relation == "ordered":
            raise ValueError("ordered additive hierarchy is not implemented")
        for order in range(1, len(rule.antecedent)):
            for indices in itertools.combinations(range(len(rule.antecedent)), order):
                subset = tuple(rule.antecedent[index] for index in indices)
                window = 0 if order == 1 else rule.window
                subset_marks = (
                    tuple(rule.history_marks[index] for index in indices)
                    if rule.history_marks
                    else ()
                )
                if (subset, window, subset_marks) not in reported:
                    required.add(ClosureTerm(subset, window, subset_marks))
    return tuple(sorted(required))


def hierarchy_branch_drop(support: Support, root: RuleIdentity) -> Support:
    """Drop exactly one total-state rule.

    Descendants are independent reported states, not hierarchy modifiers, so
    removing a singleton must not remove a pair/triplet with it.
    """
    if root not in support.rules:
        raise ValueError("branch root is not present in the support")
    return support.drop(root)


def hierarchy_branch_null_closure(
    full_closure: tuple[ClosureTerm, ...],
    drop_support: Support,
    root: RuleIdentity,
) -> tuple[ClosureTerm, ...]:
    """Return the hierarchy-complete closure after one reported-rule drop."""
    del full_closure, root
    return hierarchy_closure(drop_support)


def one_exchange_neighbors(
    support: Support,
    dictionary: tuple[RuleIdentity, ...],
) -> Iterator[Support]:
    existing = set(support.rules)
    by_pattern = {rule.pattern_key: rule for rule in support.rules}
    emitted: set[Support] = set()
    for rule in support.rules:
        trial = support.drop(rule)
        if trial not in emitted:
            emitted.add(trial)
            yield trial
    for candidate in dictionary:
        current = by_pattern.get(candidate.pattern_key)
        if current is None:
            trial = support.add(candidate)
        elif current != candidate:
            trial = support.replace(current, candidate)
        else:
            continue
        if trial not in emitted:
            emitted.add(trial)
            yield trial
    # Exact one-drop/one-add swaps across distinct antecedents.
    for removed in support.rules:
        reduced = support.drop(removed)
        reduced_patterns = set(reduced.patterns)
        for candidate in dictionary:
            if candidate in existing or candidate.pattern_key in reduced_patterns:
                continue
            trial = reduced.add(candidate)
            if trial not in emitted:
                emitted.add(trial)
                yield trial
