from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator


Antecedent = tuple[int, ...]


@dataclass(frozen=True, order=True)
class RuleIdentity:
    antecedent: Antecedent
    window: int
    sign: int

    def __post_init__(self) -> None:
        if not 1 <= len(self.antecedent) <= 3:
            raise ValueError("antecedent order must be in [1, 3]")
        if tuple(sorted(set(self.antecedent))) != self.antecedent:
            raise ValueError("antecedent must be sorted and unique")
        if self.window < 0 or (len(self.antecedent) == 1 and self.window != 0):
            raise ValueError(
                "singleton window is zero; higher-order windows are nonnegative"
            )
        if self.sign not in {-1, 1}:
            raise ValueError("sign must be -1 or +1")

    @property
    def order(self) -> int:
        return len(self.antecedent)


@dataclass(frozen=True, order=True)
class ClosureTerm:
    antecedent: Antecedent
    window: int

    def __post_init__(self) -> None:
        if (
            not self.antecedent
            or tuple(sorted(set(self.antecedent))) != self.antecedent
        ):
            raise ValueError("invalid closure antecedent")
        if len(self.antecedent) == 1 and self.window != 0:
            raise ValueError("singleton closure window must be zero")


@dataclass(frozen=True)
class Support:
    rules: tuple[RuleIdentity, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.rules))
        if ordered != self.rules or len(set(ordered)) != len(ordered):
            raise ValueError("support rules must be sorted and unique")
        antecedents = [rule.antecedent for rule in ordered]
        if len(set(antecedents)) != len(antecedents):
            raise ValueError(
                "one support cannot contain two identities of one antecedent"
            )

    @classmethod
    def of(cls, rules: Iterable[RuleIdentity]) -> "Support":
        return cls(tuple(sorted(rules)))

    @property
    def antecedents(self) -> tuple[Antecedent, ...]:
        return tuple(rule.antecedent for rule in self.rules)

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
    antecedent: Antecedent, formation_windows: tuple[int, ...]
) -> tuple[RuleIdentity, ...]:
    windows = (0,) if len(antecedent) == 1 else formation_windows
    return tuple(
        RuleIdentity(antecedent, int(window), sign)
        for window in windows
        for sign in (-1, 1)
    )


def rule_dictionary(
    n_predicates: int, q_max: int, formation_windows: tuple[int, ...]
) -> tuple[RuleIdentity, ...]:
    return tuple(
        identity
        for antecedent in skeletons(n_predicates, q_max)
        for identity in identities_for(antecedent, formation_windows)
    )


def hierarchy_closure(support: Support) -> tuple[ClosureTerm, ...]:
    reported = {(rule.antecedent, rule.window) for rule in support.rules}
    required: set[ClosureTerm] = set()
    for rule in support.rules:
        for order in range(1, len(rule.antecedent)):
            for subset in itertools.combinations(rule.antecedent, order):
                window = 0 if order == 1 else rule.window
                if (subset, window) not in reported:
                    required.add(ClosureTerm(subset, window))
    return tuple(sorted(required))


def one_exchange_neighbors(
    support: Support,
    dictionary: tuple[RuleIdentity, ...],
) -> Iterator[Support]:
    existing = set(support.rules)
    by_antecedent = {rule.antecedent: rule for rule in support.rules}
    emitted: set[Support] = set()
    for rule in support.rules:
        trial = support.drop(rule)
        if trial not in emitted:
            emitted.add(trial)
            yield trial
    for candidate in dictionary:
        current = by_antecedent.get(candidate.antecedent)
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
        reduced_antecedents = set(reduced.antecedents)
        for candidate in dictionary:
            if candidate in existing or candidate.antecedent in reduced_antecedents:
                continue
            trial = reduced.add(candidate)
            if trial not in emitted:
                emitted.add(trial)
                yield trial
