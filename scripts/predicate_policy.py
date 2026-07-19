"""Predicate taxonomy and selection policies for rule discovery experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


GROUP_DESCRIPTIONS: dict[str, str] = {
    "target_leakage": "Target, post-target, termination, or target-definition fields. Never valid as rule antecedents.",
    "loss_mitigation_sentinel": "Servicer or administrative hardship/loss-mitigation actions that are target-adjacent sentinels.",
    "delinquency_precursor": "Early delinquency states or transitions. Useful controls, but too direct for nontrivial SDQ3 rule discovery.",
    "collateral_dynamics": "Loan-to-value, collateral value, or home-price dynamics.",
    "balance_dynamics": "UPB, amortization, balance, or negative-amortization-like dynamics.",
    "rate_dynamics": "Interest-rate or coupon level/change dynamics.",
    "seasoning_time_control": "Loan-age, vintage, calendar, or seasonality markers. Usually better as controls than discovered rules.",
    "macro_context": "External macro or geography-level context such as HPI, unemployment, or market-rate changes.",
    "data_quality": "Missingness, stale value, correction, or outlier/anomaly indicators.",
    "unknown": "Unclassified predicate. Excluded by restrictive policies until explicitly reviewed.",
}


CURRENT_PREDICATE_GROUPS: dict[str, str] = {
    "pred_eltv_enters_high_ltv": "collateral_dynamics",
    "pred_eltv_exits_high_ltv": "collateral_dynamics",
    "pred_eltv_enters_negative_equity": "collateral_dynamics",
    "pred_eltv_exits_negative_equity": "collateral_dynamics",
    "pred_eltv_rises_within_band": "collateral_dynamics",
    "pred_eltv_falls_within_band": "collateral_dynamics",
    "pred_upb_increase_starts": "balance_dynamics",
    "pred_upb_increase_continues": "balance_dynamics",
    "pred_upb_flat_starts": "balance_dynamics",
    "pred_upb_paydown_resumes": "balance_dynamics",
    "pred_upb_paydown_accelerates": "balance_dynamics",
    "pred_upb_paydown_decelerates": "balance_dynamics",
    "pred_upb_paydown_steady": "balance_dynamics",
    "pred_eltv_deterioration_starts_within_band": "collateral_dynamics",
    "pred_eltv_improvement_starts_within_band": "collateral_dynamics",
    "pred_upb_paydown_acceleration_starts": "balance_dynamics",
    "pred_upb_paydown_deceleration_starts": "balance_dynamics",
    "pred_upb_paydown_steady_starts": "balance_dynamics",
    "pred_payment_deferral_current": "loss_mitigation_sentinel",
    "pred_deferred_upb_starts": "loss_mitigation_sentinel",
    "pred_deferred_upb_clears": "loss_mitigation_sentinel",
    "pred_forbearance_starts": "loss_mitigation_sentinel",
    "pred_forbearance_ends": "loss_mitigation_sentinel",
    "pred_repayment_plan_starts": "loss_mitigation_sentinel",
    "pred_repayment_plan_ends": "loss_mitigation_sentinel",
    "pred_trial_plan_starts": "loss_mitigation_sentinel",
    "pred_trial_plan_ends": "loss_mitigation_sentinel",
    "pred_disaster_hardship_starts": "loss_mitigation_sentinel",
    "pred_disaster_hardship_ends": "loss_mitigation_sentinel",
    "pred_eltv_cross_90": "collateral_dynamics",
    "pred_eltv_cross_down_90": "collateral_dynamics",
    "pred_eltv_jump_10pp": "collateral_dynamics",
    "pred_eltv_drop_10pp": "collateral_dynamics",
    "pred_upb_increase_1pct": "balance_dynamics",
    "pred_eltv_cross_up_80": "collateral_dynamics",
    "pred_eltv_cross_up_90": "collateral_dynamics",
    "pred_eltv_cross_up_100": "collateral_dynamics",
    "pred_eltv_jump_5pp_1m": "collateral_dynamics",
    "pred_eltv_jump_10pp_3m": "collateral_dynamics",
    "pred_eltv_drop_10pp_3m": "collateral_dynamics",
    "pred_eltv_sustained_ge_90_3m_starts": "collateral_dynamics",
    "pred_eltv_sustained_ge_100_3m_starts": "collateral_dynamics",
    "pred_eltv_range_ge_15pp_6m_starts": "collateral_dynamics",
    "pred_eltv_missing_999_starts": "collateral_dynamics",
    "pred_eltv_enters_monthly_p90": "collateral_dynamics",
    "pred_eltv_rank_worsens_20pctile_3m": "collateral_dynamics",
    "pred_upb_increase_0p5pct": "balance_dynamics",
    "pred_upb_increase_0p5pct_1m_after_6m": "balance_dynamics",
    "pred_upb_jump_1pct_3m_after_6m": "balance_dynamics",
    "pred_upb_flat_3m_after_6m_starts": "balance_dynamics",
    "pred_upb_reduction_lt_1pct_6m_after_12m_starts": "balance_dynamics",
    "pred_upb_reduction_bottom_decile_by_age_6m_starts": "balance_dynamics",
    "pred_upb_to_initial_upb_ratio_ge_0p98_after_12m": "balance_dynamics",
    "pred_loan_age_reaches_12": "seasoning_time_control",
    "pred_loan_age_reaches_24": "seasoning_time_control",
}


DIRECT_RULE_GROUPS = frozenset(
    {
        "target_leakage",
        "loss_mitigation_sentinel",
        "delinquency_precursor",
    }
)


PREDICATE_POLICY_ALLOWED_GROUPS: dict[str, frozenset[str] | None] = {
    "all": None,
    "exclude-direct": frozenset(
        {
            "collateral_dynamics",
            "balance_dynamics",
            "rate_dynamics",
            "seasoning_time_control",
            "macro_context",
            "data_quality",
        }
    ),
    "upstream": frozenset(
        {
            "collateral_dynamics",
            "balance_dynamics",
            "rate_dynamics",
            "seasoning_time_control",
            "macro_context",
        }
    ),
    "structural": frozenset(
        {
            "collateral_dynamics",
            "balance_dynamics",
            "rate_dynamics",
        }
    ),
}


PREDICATE_POLICY_DESCRIPTIONS: dict[str, str] = {
    "all": "Legacy behavior: use every requested or inferred predicate.",
    "exclude-direct": "Drop target/leakage, delinquency-precursor, and loss-mitigation sentinel predicates.",
    "upstream": "Keep reviewed upstream risk dynamics and seasoning/context controls; drop direct sentinel predicates.",
    "structural": "Keep only structural loan-risk dynamics; drop direct sentinels and time/seasoning controls.",
}


_PATTERN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "target_leakage",
        (
            r"target",
            r"sdq3",
            r"first_?90",
            r"reo",
            r"foreclosure",
            r"zero_?balance",
            r"termination",
        ),
    ),
    (
        "loss_mitigation_sentinel",
        (
            r"forbear",
            r"repayment",
            r"trial_?plan",
            r"deferr",
            r"assistance",
            r"hardship",
            r"disaster",
            r"modification",
            r"loss_?mitigation",
        ),
    ),
    (
        "delinquency_precursor",
        (
            r"delq",
            r"delin",
            r"dpd",
            r"dq",
            r"missed_?payment",
            r"arrear",
        ),
    ),
    (
        "collateral_dynamics",
        (
            r"eltv",
            r"cltv",
            r"ltv",
            r"equity",
            r"hpi",
            r"home_?price",
            r"collateral",
        ),
    ),
    (
        "balance_dynamics",
        (
            r"upb",
            r"balance",
            r"amort",
            r"principal",
            r"interest_?bearing_?upb",
        ),
    ),
    (
        "rate_dynamics",
        (
            r"interest_?rate",
            r"coupon",
            r"note_?rate",
            r"mortgage_?rate",
            r"rate_?(jump|rise|drop|change|reset|high|low)",
        ),
    ),
    (
        "seasoning_time_control",
        (
            r"loan_?age",
            r"season",
            r"vintage",
            r"calendar",
            r"month_?of_?year",
        ),
    ),
    (
        "macro_context",
        (
            r"unemployment",
            r"macro",
            r"market",
            r"msa",
            r"metro",
            r"state",
        ),
    ),
    (
        "data_quality",
        (
            r"missing",
            r"unknown",
            r"stale",
            r"correct",
            r"outlier",
            r"anomal",
        ),
    ),
)


@dataclass(frozen=True)
class PredicateSelection:
    policy: str
    selected: tuple[str, ...]
    excluded_by_policy: tuple[str, ...]
    excluded_explicit: tuple[str, ...]
    groups: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "selected": list(self.selected),
            "excluded_by_policy": list(self.excluded_by_policy),
            "excluded_explicit": list(self.excluded_explicit),
            "groups": self.groups,
        }


def classify_predicate_name(name: str) -> str:
    if name in CURRENT_PREDICATE_GROUPS:
        return CURRENT_PREDICATE_GROUPS[name]
    low = name.lower()
    for group, patterns in _PATTERN_GROUPS:
        if any(re.search(pattern, low) for pattern in patterns):
            return group
    return "unknown"


def select_predicates(
    predicate_cols: Iterable[str],
    policy: str = "all",
    exclude_predicates: Iterable[str] | None = None,
) -> PredicateSelection:
    if policy not in PREDICATE_POLICY_ALLOWED_GROUPS:
        valid = ", ".join(sorted(PREDICATE_POLICY_ALLOWED_GROUPS))
        raise ValueError(f"unknown predicate policy {policy!r}; valid policies: {valid}")

    excluded_names = set(exclude_predicates or [])
    allowed_groups = PREDICATE_POLICY_ALLOWED_GROUPS[policy]
    selected: list[str] = []
    excluded_by_policy: list[str] = []
    excluded_explicit: list[str] = []
    groups: dict[str, str] = {}

    for col in predicate_cols:
        group = classify_predicate_name(col)
        groups[col] = group
        if col in excluded_names:
            excluded_explicit.append(col)
            continue
        if allowed_groups is not None and group not in allowed_groups:
            excluded_by_policy.append(col)
            continue
        selected.append(col)

    return PredicateSelection(
        policy=policy,
        selected=tuple(selected),
        excluded_by_policy=tuple(excluded_by_policy),
        excluded_explicit=tuple(excluded_explicit),
        groups=groups,
    )


def predicate_policy_metadata() -> dict:
    return {
        "group_descriptions": GROUP_DESCRIPTIONS,
        "current_predicate_groups": CURRENT_PREDICATE_GROUPS,
        "direct_rule_groups": sorted(DIRECT_RULE_GROUPS),
        "policies": {
            name: {
                "description": PREDICATE_POLICY_DESCRIPTIONS[name],
                "allowed_groups": "all"
                if groups is None
                else sorted(groups),
            }
            for name, groups in PREDICATE_POLICY_ALLOWED_GROUPS.items()
        },
    }
