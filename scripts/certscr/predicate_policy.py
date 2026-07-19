from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


HOME_CREDIT_BEHAVIORAL_NONPROXY: tuple[str, ...] = (
    "pred_prev_revolving_application",
    "pred_prev_multi_application_month",
    "pred_bureau_credit_card_opened",
    "pred_bureau_large_credit_opened",
    "pred_pos_installment_count_increases",
    "pred_card_utilization_jump_20pp",
    "pred_card_cash_withdrawal",
)

# Expanded dynamic-only policy. The additions cover product choice, external
# credit acquisition/closure, and contract lifecycle without introducing
# refusal/cancellation, delinquency, underpayment, lender limit actions, or
# reporting-state proxies. The final set was frequency/overlap audited without
# consulting target outcomes (20k smoke sequence prevalence: 1.1%--88.4%).
HOME_CREDIT_BEHAVIORAL_NONPROXY_EXPANDED: tuple[str, ...] = (
    "pred_prev_cash_application",
    "pred_prev_consumer_application",
    "pred_prev_revolving_application",
    "pred_prev_multi_application_month",
    "pred_bureau_credit_card_opened",
    "pred_bureau_microloan_opened",
    "pred_bureau_large_credit_opened",
    "pred_bureau_credit_closed",
    "pred_pos_contract_completed",
    "pred_pos_installment_count_increases",
    "pred_card_utilization_jump_20pp",
    "pred_card_cash_withdrawal",
)

# Freddie Mac SDQ3 upstream structural dynamics. Delinquency, servicing/loss
# mitigation, termination, static origination, and loan-age/calendar markers
# are deliberately excluded from the discovered-rule family.
FREDDIE_STRUCTURAL_DYNAMIC_V2: tuple[str, ...] = (
    "pred_eltv_cross_up_80",
    "pred_eltv_cross_up_90",
    "pred_eltv_cross_up_100",
    "pred_eltv_cross_down_90",
    "pred_eltv_jump_5pp_1m",
    "pred_eltv_jump_10pp_3m",
    "pred_eltv_drop_10pp_3m",
    "pred_eltv_sustained_ge_90_3m_starts",
    "pred_eltv_range_ge_15pp_6m_starts",
    "pred_eltv_missing_999_starts",
    "pred_eltv_enters_monthly_p90",
    "pred_eltv_rank_worsens_20pctile_3m",
    "pred_upb_increase_0p5pct_1m_after_6m",
    "pred_upb_jump_1pct_3m_after_6m",
    "pred_upb_flat_3m_after_6m_starts",
    "pred_upb_reduction_lt_1pct_6m_after_12m_starts",
    "pred_upb_reduction_bottom_decile_by_age_6m_starts",
    "pred_upb_to_initial_upb_ratio_ge_0p98_after_12m",
)

# Outcome-blind primitive monthly state transitions for Freddie Mac SDQ3.
# The dictionary intentionally excludes delinquency, servicing actions,
# termination, rolling motifs, cross-sectional ranks, and duplicate threshold
# aliases. Pair/triplet structure is composed by the rule grammar itself.
FREDDIE_PRIMITIVE_DYNAMIC_V3: tuple[str, ...] = (
    "pred_eltv_enters_high_ltv",
    "pred_eltv_exits_high_ltv",
    "pred_eltv_enters_negative_equity",
    "pred_eltv_exits_negative_equity",
    "pred_eltv_rises_within_band",
    "pred_eltv_falls_within_band",
    "pred_upb_increase_starts",
    "pred_upb_increase_continues",
    "pred_upb_flat_starts",
    "pred_upb_paydown_resumes",
    "pred_upb_paydown_accelerates",
    "pred_upb_paydown_decelerates",
    "pred_upb_paydown_steady",
)

# Financially interpretable, target-blind transition onsets for the primary
# Freddie experiment. Within each ELTV and UPB family the events
# are mutually exclusive in a reporting month. Dense persistent directions
# and the nearly empty "UPB increase continues" atom from v3 are replaced by
# onsets, so pair/triplet rules describe combinations of distinct changes
# rather than repeated encodings of one persistent state.
FREDDIE_PRIMITIVE_DYNAMIC_V4: tuple[str, ...] = (
    "pred_eltv_enters_high_ltv",
    "pred_eltv_exits_high_ltv",
    "pred_eltv_enters_negative_equity",
    "pred_eltv_exits_negative_equity",
    "pred_eltv_deterioration_starts_within_band",
    "pred_eltv_improvement_starts_within_band",
    "pred_upb_increase_starts",
    "pred_upb_flat_starts",
    "pred_upb_paydown_resumes",
    "pred_upb_paydown_acceleration_starts",
    "pred_upb_paydown_deceleration_starts",
    "pred_upb_paydown_steady_starts",
)

# IBM AMLworld account-history transitions selected without consulting the AML
# outcome.  On the complete HI-Small stream these seven columns have maximum
# pairwise event-row Jaccard 0.156 after calibration-tail refinement; all but
# the novelty/dormancy pair are at most 0.128.
IBM_AML_DYNAMIC_NONPROXY_V2: tuple[str, ...] = (
    "pred_out_amount_spike_rel_mean",
    "pred_out_receiver_novelty_after_history",
    "pred_in_sender_novelty_after_history",
    "pred_out_burst_hour_starts",
    "pred_in_burst_hour_starts",
    "pred_out_currency_switch_after_history",
    "pred_out_dormancy_reactivation",
)

IBM_AML_DYNAMIC_NONPROXY_V3: tuple[str, ...] = (
    *IBM_AML_DYNAMIC_NONPROXY_V2,
    "pred_out_amount_drop_rel_mean",
    "pred_in_amount_spike_rel_mean",
    "pred_out_receiver_revisit_after_alternative",
    "pred_in_sender_revisit_after_alternative",
    "pred_out_cadence_acceleration",
)

# Atomic account-state transitions for interaction discovery.  Unlike the
# typology policy below, no entry already encodes a multi-leg AML motif such as
# structuring, fan-in/fan-out, pass-through, gather-scatter, or a cycle.  A
# predicate may use frozen, outcome-blind history to decide that the current
# transaction is a transition/onset, but it contributes exactly one event to
# the TPP rule grammar.  `pred_in_to_out_turnaround_starts` is deliberately
# excluded because it already combines an incoming and an outgoing event.
IBM_AML_PRIMITIVE_DYNAMIC_V1: tuple[str, ...] = IBM_AML_DYNAMIC_NONPROXY_V3

# Domain-hypothesis dictionary for AMLworld.  These are causal onsets or
# observable transitions, not labels: structuring, fan-in/fan-out,
# pass-through/gather-scatter, cycles, velocity, amount/counterparty and route
# changes.  Unlike the low-overlap primitive dictionary, this policy preserves
# the graph/flow motifs that AMLworld actually injects.
IBM_AML_TYPOLOGY_DYNAMIC_V1: tuple[str, ...] = (
    "pred_out_structured_small_repeats_day",
    "pred_many_new_receivers_day",
    "pred_many_unique_senders_day",
    "pred_rapid_in_to_out_day",
    "pred_fan_in_then_out_day",
    "pred_cycle_return_72h",
    "pred_out_burst_hour_starts",
    "pred_out_amount_spike_rel_mean",
    "pred_in_amount_spike_rel_mean",
    "pred_out_receiver_novelty_after_history",
    "pred_out_currency_switch_after_history",
    "pred_out_bank_route_switch_after_history",
    "pred_out_cadence_acceleration",
)


PREDICATE_POLICIES: dict[str, tuple[str, ...]] = {
    "home_credit_behavioral_nonproxy": HOME_CREDIT_BEHAVIORAL_NONPROXY,
    "home_credit_behavioral_nonproxy_expanded": HOME_CREDIT_BEHAVIORAL_NONPROXY_EXPANDED,
    "freddie_structural_dynamic_v2": FREDDIE_STRUCTURAL_DYNAMIC_V2,
    "freddie_primitive_dynamic_v3": FREDDIE_PRIMITIVE_DYNAMIC_V3,
    "freddie_primitive_dynamic_v4": FREDDIE_PRIMITIVE_DYNAMIC_V4,
    "ibm_aml_dynamic_nonproxy_v2": IBM_AML_DYNAMIC_NONPROXY_V2,
    "ibm_aml_dynamic_nonproxy_v3": IBM_AML_DYNAMIC_NONPROXY_V3,
    "ibm_aml_primitive_dynamic_v1": IBM_AML_PRIMITIVE_DYNAMIC_V1,
    "ibm_aml_typology_dynamic_v1": IBM_AML_TYPOLOGY_DYNAMIC_V1,
}


@dataclass(frozen=True)
class PredicatePolicyContract:
    """Pre-outcome semantic audit attached to a frozen predicate dictionary.

    These declarations are not estimated from the target and never enter the
    likelihood.  They record the parts of F0 that code can actually enforce:
    the named dictionary was reviewed as dynamic, outcome-blind and free of a
    direct target proxy.  Strict temporal predictability is supplied by the
    TPP response, whose first possible effect lag is one grid step.
    """

    predicates: tuple[str, ...]
    dynamic: bool
    outcome_blind_construction: bool
    direct_target_proxy_excluded: bool
    review_basis: str
    atomic_events: bool = False

    @property
    def f0_eligible(self) -> bool:
        return bool(
            self.dynamic
            and self.outcome_blind_construction
            and self.direct_target_proxy_excluded
        )


PREDICATE_POLICY_CONTRACTS: dict[str, PredicatePolicyContract] = {
    name: PredicatePolicyContract(
        predicates=predicates,
        dynamic=True,
        outcome_blind_construction=True,
        direct_target_proxy_excluded=True,
        review_basis=(
            "frozen named policy reviewed from source-variable semantics before "
            "target-conditioned rule fitting"
        ),
        atomic_events=(
            name
            in {
                "ibm_aml_primitive_dynamic_v1",
                "freddie_primitive_dynamic_v3",
                "freddie_primitive_dynamic_v4",
            }
        ),
    )
    for name, predicates in PREDICATE_POLICIES.items()
}


def resolve_predicate_policy(name: str) -> tuple[str, ...]:
    try:
        return PREDICATE_POLICIES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown predicate policy: {name}") from exc


def resolve_predicate_policy_contract(name: str) -> PredicatePolicyContract:
    try:
        return PREDICATE_POLICY_CONTRACTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown predicate policy contract: {name}") from exc


def validate_policy_columns(policy: Sequence[str], available: Sequence[str]) -> None:
    missing = sorted(set(policy) - set(available))
    if missing:
        raise ValueError(f"predicate policy columns are missing from the dataset: {missing}")
