from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Complete, serializable configuration for one CRBS-TPP run."""

    dataset: str
    run_root: str = "runs/crbstpp"
    run_id: str | None = None
    q_max: int = 3
    impact_lag: int = 12
    knot_count: int = 4
    # Fit route identities with the ordinary full M-knot kernel, then compare
    # the exact scalar and full representations by the same MDL objective.
    adaptive_kernel_mdl: bool = False
    formation_windows: tuple[int, ...] = tuple(range(13))
    formation_window_mode: str = "fixed"
    formation_window_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)
    temporal_relations: tuple[str, ...] = ("unordered",)
    # ``total_state`` keeps mutually exclusive nested state blocks.  The v14
    # model uses hierarchy-complete additive main effects plus a signed
    # higher-order modifier, so interaction and total contextual directions
    # can be reported separately.
    effect_model: str = "total_state"
    split_fractions: tuple[float, float, float] = (0.60, 0.20, 0.20)
    split_seed: int = 111
    alpha: float = 0.05
    romano_wolf_resamples: int = 200_000
    romano_wolf_seed: int = 314159
    early_warning_horizon: int = 12
    # Outcome-blind structural time baseline.  One means the legacy static
    # stratum intercept.  Values above one cross fixed calendar-time and
    # episode-age cells, while reported financial actions remain outside the
    # null model.
    baseline_time_bins: int = 1
    probability_materiality: float = 0.0
    # Separate a candidate's calendar-wide prevalence component from its
    # entity-relative component using the exact risk-set Fisher metric.  This
    # affects route ordering and adds an independently frozen D_cert direction
    # diagnostic; exact MDL acceptance and the F0--F3 definitions are intact.
    frequency_effect_separation: bool = False
    # Dependency-aware composite-likelihood MDL.  Coefficients continue to be
    # fitted by the declared point-process likelihood; model selection uses a
    # wallet/entity x calendar two-way Godambe complexity correction.
    dependency_aware_mdl: bool = False
    # Treat the target-blind recent/recurrent/accelerating/decelerating lifts
    # of one primitive predicate as alternative representations of one
    # history mechanism while constructing provisional roots and routes.  The
    # complete flat identity dictionary is retained for exact terminal Add and
    # representation audits, so this compresses exploration without changing
    # the terminal finite-dictionary certificate.
    history_state_family_search: bool = False
    # Target-blind count refinements of primitive events.  A marked atom at t
    # requires c earlier occurrences of that same primitive in [t-L,t).  The
    # count levels are response-distinct D_fit values and are not materialized
    # as extra dataset predicates.
    history_marked_events: bool = False
    history_lookback_windows: tuple[int, ...] = (3, 7, 30)
    history_count_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)
    history_max_marked_atoms: int = 1
    # D_fit-only reduction performed before the certification family is
    # frozen.  Only deletion-minimal, predictively distinct and nested-MDL
    # nondominated supports are exposed to D_cert multiplicity correction.
    precert_family_compaction: bool = False
    # Freeze the D_fit family with the exact fixed-intensity simplex MDL before
    # D_cert, and report only nonzero ensemble components after refitting.
    # This makes every final support both independently reliable and necessary
    # to the learned family rather than a zero-weight certified alternative.
    ensemble_irreducible_family: bool = False
    reliability_aware_search: bool = True
    ensemble_aware_roots: bool = False
    ensemble_residual_search: bool = False
    # Prioritize (never prune) standalone routes by the common-baseline
    # gradient/Fisher residual of their individual fitted rule effects, then
    # combine the certified rule library with an exact nonnegative effect
    # stack. Support Block-MDL Add/Drop and support-level certification stay
    # unchanged.
    rule_effect_stacking_search: bool = False
    # Separate discovery into exact family-MDL-positive predictive columns
    # and Fisher/MDL-resolution Rashomon basins. Predictive roots are always
    # explored; zero-reduced-cost roots launch one route per predictively
    # distinguishable basin while every exact-positive atom remains eligible
    # for the frozen certification family.
    predictive_basin_rashomon_search: bool = False
    # Expand the exact-positive standalone atom pool through the complete
    # objective-admissible fallback ledger, without protecting predicates,
    # forcing one route per predicate, or exempting roots from Fisher/MDL
    # equivalence packing.
    standalone_positive_atom_expansion: bool = False
    # Audit selected high-order rules together with visible lower-order anchors.
    # This permits H(exc) + H-and-R(inh) without hidden hierarchy nuisance.
    anchor_aware_representation_audit: bool = False
    # Reorder (never prune) exact-positive route tails by Fisher novelty.
    fisher_orthogonal_route_order: bool = False
    # Discover and report an MDL-resolution-separated Rashomon family. Route
    # roots are ordered by residual Fisher novelty, while final D_fit supports
    # closer than the minimum one-rule MDL resolution are represented by the
    # stronger/smaller member. In block-score mode this also prevents the
    # later exact-family frontier from collapsing all independently positive
    # basins into one greedy Add descendant; accepted supports retain exact
    # fixed-support, Drop, W/sign and representation fitting/audits.
    fisher_separated_rashomon: bool = False
    # At every nonterminal Add table, order the complete candidate dictionary
    # by Pareto fronts of family Block-MDL gain and primitive-predicate novelty
    # relative to previously completed D_fit terminals. No candidate is
    # removed and the terminal certificate remains the ordinary family-level
    # block-stationarity audit. This changes the explored local basin without
    # rewarding a negative-MDL rule or weakening F0--F3.
    online_predicate_pareto_frontier: bool = False
    # Give every primitive predicate with an objective-admissible standalone
    # envelope one exact constrained route opportunity.  The constraint is
    # released at the first exact terminal and the ordinary unrestricted
    # Add/Drop/W-sign objective is audited before reporting.  D_fit family
    # packing preserves an exact-positive root when it contributes a predicate
    # not represented by any stronger retained root; F0--F3 are unchanged.
    predicate_coverage_rashomon: bool = False
    # Follow every support-conditioned positive score-basin maximum in one
    # shared DAG instead of committing each state to only its best Add.
    conditional_basin_branching: bool = False
    # Re-optimize the exact support-family mixture after each frontier round.
    ensemble_residual_dictionary_repricing: bool = False
    # Preserve every exact-positive standalone objective basin as an
    # independent shared-DAG route. Each state follows one certified improving
    # Add, while equal downstream supports are merged. This explores distinct
    # basins without enumerating every W/sign neighbour at one parent.
    rashomon_branching: bool = False
    search_mode: str = "exact_safe"
    # ``exact`` proves complete Add stationarity by refitting every unresolved
    # terminal neighbour.  ``block_score`` keeps the monotone multi-step route
    # certificate and performs exact fitting/audits only for selected terminal
    # supports.  Drop, W/sign and representation audits remain exact.
    terminal_add_audit: str = "exact"
    adaptive_gradient_racing: bool = False
    route_refinement_max_steps: int = 4
    route_refinement_kkt_tolerance: float | None = None
    max_rules_per_support: int | None = None
    discovery_sampling: str = "full"
    discovery_reference_dataset: str | None = None
    discovery_noncase_fraction: float = 0.10
    discovery_sampling_seed: int = 271828
    solver_tolerance: float = 2.0e-7
    solver_max_iter: int = 100
    search_tolerance: float = 1.0e-8
    exact_workers: int = 3
    pricing_workers: int = 12
    pricing_devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    route_workers: int = 1
    cache_bytes: int = 8 * 1024**3
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("run config must be a YAML mapping")
        for name in (
            "formation_windows",
            "formation_window_quantiles",
            "temporal_relations",
            "split_fractions",
            "pricing_devices",
            "history_lookback_windows",
            "history_count_quantiles",
        ):
            if name in payload:
                payload[name] = tuple(payload[name])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError("dataset is required")
        if not 1 <= self.q_max <= 3:
            raise ValueError("q_max must be in [1, 3]")
        if self.impact_lag < 1 or self.knot_count < 1:
            raise ValueError("impact_lag and knot_count must be positive")
        if not isinstance(self.adaptive_kernel_mdl, bool):
            raise ValueError("adaptive_kernel_mdl must be boolean")
        windows = tuple(int(value) for value in self.formation_windows)
        if not windows or windows != tuple(sorted(set(windows))) or windows[0] < 0:
            raise ValueError("formation_windows must be sorted, unique and nonnegative")
        if self.formation_window_mode not in {
            "fixed",
            "fit_quantile",
            "fit_quantile_band",
        }:
            raise ValueError(
                "formation_window_mode must be 'fixed', 'fit_quantile' or "
                "'fit_quantile_band'"
            )
        quantiles = tuple(float(value) for value in self.formation_window_quantiles)
        if (
            not quantiles
            or quantiles != tuple(sorted(set(quantiles)))
            or any(value <= 0.0 or value > 1.0 for value in quantiles)
        ):
            raise ValueError(
                "formation_window_quantiles must be sorted, unique, lie in (0, 1]"
            )
        relations = tuple(map(str, self.temporal_relations))
        if (
            not relations
            or relations != tuple(dict.fromkeys(relations))
            or any(value not in {"unordered", "ordered"} for value in relations)
        ):
            raise ValueError(
                "temporal_relations must be a unique nonempty subset of "
                "unordered/ordered"
            )
        if self.effect_model not in {
            "total_state",
            "additive_hierarchy",
            "support_additive",
        }:
            raise ValueError(
                "effect_model must be 'total_state', 'additive_hierarchy' or "
                "'support_additive'"
            )
        if self.effect_model == "additive_hierarchy" and "ordered" in relations:
            raise ValueError(
                "v14 additive hierarchy currently requires unordered relations"
            )
        if len(self.split_fractions) != 3 or any(
            value <= 0 for value in self.split_fractions
        ):
            raise ValueError("three positive split fractions are required")
        if not math.isclose(sum(self.split_fractions), 1.0, abs_tol=1.0e-12):
            raise ValueError("split fractions must sum to one")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must lie in (0, 0.5)")
        if self.romano_wolf_resamples < 1_000:
            raise ValueError("romano_wolf_resamples must be at least 1000")
        if self.romano_wolf_seed < 0:
            raise ValueError("romano_wolf_seed must be nonnegative")
        if not 1 <= self.early_warning_horizon <= self.impact_lag:
            raise ValueError("early_warning_horizon must lie in [1, impact_lag]")
        if (
            isinstance(self.baseline_time_bins, bool)
            or not 1 <= self.baseline_time_bins <= 8
        ):
            raise ValueError("baseline_time_bins must be an integer in [1, 8]")
        if not 0 <= self.probability_materiality < 1:
            raise ValueError("probability_materiality must lie in [0, 1)")
        if not isinstance(self.frequency_effect_separation, bool):
            raise ValueError("frequency_effect_separation must be boolean")
        if not isinstance(self.dependency_aware_mdl, bool):
            raise ValueError("dependency_aware_mdl must be boolean")
        if not isinstance(self.history_state_family_search, bool):
            raise ValueError("history_state_family_search must be boolean")
        if not isinstance(self.history_marked_events, bool):
            raise ValueError("history_marked_events must be boolean")
        history_windows = tuple(map(int, self.history_lookback_windows))
        if (
            not history_windows
            or history_windows != tuple(sorted(set(history_windows)))
            or history_windows[0] < 1
        ):
            raise ValueError(
                "history_lookback_windows must be sorted, unique and positive"
            )
        if self.history_max_marked_atoms != 1:
            raise ValueError(
                "v15 finite grammar supports exactly one history-marked atom per rule"
            )
        count_quantiles = tuple(map(float, self.history_count_quantiles))
        if (
            not count_quantiles
            or count_quantiles != tuple(sorted(set(count_quantiles)))
            or any(value <= 0.0 or value > 1.0 for value in count_quantiles)
        ):
            raise ValueError(
                "history_count_quantiles must be sorted, unique and in (0,1]"
            )
        if self.history_marked_events and self.history_state_family_search:
            raise ValueError(
                "history-marked events replace materialized history-state search"
            )
        if not isinstance(self.precert_family_compaction, bool):
            raise ValueError("precert_family_compaction must be boolean")
        if not isinstance(self.ensemble_irreducible_family, bool):
            raise ValueError("ensemble_irreducible_family must be boolean")
        if self.ensemble_irreducible_family and not self.reliability_aware_search:
            raise ValueError(
                "ensemble-irreducible reporting requires reliability-aware search"
            )
        if self.dependency_aware_mdl and self.discovery_sampling != "full":
            raise ValueError("dependency-aware MDL requires complete D_fit histories")
        if not isinstance(self.reliability_aware_search, bool):
            raise ValueError("reliability_aware_search must be boolean")
        if not isinstance(self.ensemble_aware_roots, bool):
            raise ValueError("ensemble_aware_roots must be boolean")
        if not isinstance(self.ensemble_residual_search, bool):
            raise ValueError("ensemble_residual_search must be boolean")
        if not isinstance(self.rule_effect_stacking_search, bool):
            raise ValueError("rule_effect_stacking_search must be boolean")
        if self.ensemble_residual_search and self.rule_effect_stacking_search:
            raise ValueError(
                "support-intensity and rule-effect residual search are mutually exclusive"
            )
        if self.rule_effect_stacking_search and self.ensemble_irreducible_family:
            raise ValueError(
                "rule-effect stacking reports every certified rule and is incompatible "
                "with pre-emptive support-simplex pruning"
            )
        if not isinstance(self.predictive_basin_rashomon_search, bool):
            raise ValueError("predictive_basin_rashomon_search must be boolean")
        if not isinstance(self.standalone_positive_atom_expansion, bool):
            raise ValueError("standalone_positive_atom_expansion must be boolean")
        if self.predictive_basin_rashomon_search and not (
            (self.ensemble_residual_search or self.rule_effect_stacking_search)
            and self.fisher_separated_rashomon
            and self.search_mode == "atomic_rashomon_frontier"
        ):
            raise ValueError(
                "predictive_basin_rashomon_search requires ensemble residual "
                "search, Fisher-separated Rashomon discovery, and the atomic "
                "Rashomon frontier"
            )
        if self.predictive_basin_rashomon_search and (
            self.online_predicate_pareto_frontier or self.predicate_coverage_rashomon
        ):
            raise ValueError(
                "predictive-basin search replaces Pareto and predicate-coverage "
                "route forcing"
            )
        if not isinstance(self.anchor_aware_representation_audit, bool):
            raise ValueError("anchor_aware_representation_audit must be boolean")
        if not isinstance(self.fisher_orthogonal_route_order, bool):
            raise ValueError("fisher_orthogonal_route_order must be boolean")
        if not isinstance(self.fisher_separated_rashomon, bool):
            raise ValueError("fisher_separated_rashomon must be boolean")
        if not isinstance(self.online_predicate_pareto_frontier, bool):
            raise ValueError("online_predicate_pareto_frontier must be boolean")
        if not isinstance(self.predicate_coverage_rashomon, bool):
            raise ValueError("predicate_coverage_rashomon must be boolean")
        if not isinstance(self.conditional_basin_branching, bool):
            raise ValueError("conditional_basin_branching must be boolean")
        if not isinstance(self.ensemble_residual_dictionary_repricing, bool):
            raise ValueError("ensemble_residual_dictionary_repricing must be boolean")
        if (
            self.conditional_basin_branching
            and self.search_mode != "atomic_rashomon_frontier"
        ):
            raise ValueError(
                "conditional_basin_branching requires "
                "search_mode='atomic_rashomon_frontier'"
            )
        if self.ensemble_residual_dictionary_repricing and not (
            self.ensemble_residual_search or self.rule_effect_stacking_search
        ):
            raise ValueError(
                "ensemble_residual_dictionary_repricing requires a residual search mode"
            )
        if not isinstance(self.rashomon_branching, bool):
            raise ValueError("rashomon_branching must be boolean")
        if self.rashomon_branching and self.search_mode != "atomic_rashomon_frontier":
            raise ValueError(
                "rashomon_branching requires search_mode='atomic_rashomon_frontier'"
            )
        if self.search_mode not in {
            "exact_safe",
            "fast_block_score",
            "safe_column_generation",
            "gap_safe_rashomon_path",
            "successor_rashomon_path",
            "atomic_rashomon_frontier",
        }:
            raise ValueError(
                "search_mode must be 'exact_safe', 'fast_block_score', "
                "'safe_column_generation', 'gap_safe_rashomon_path', or "
                "'successor_rashomon_path', or 'atomic_rashomon_frontier'"
            )
        if self.terminal_add_audit not in {"exact", "block_score"}:
            raise ValueError("terminal_add_audit must be 'exact' or 'block_score'")
        if (
            self.terminal_add_audit == "block_score"
            and self.search_mode != "atomic_rashomon_frontier"
        ):
            raise ValueError(
                "terminal_add_audit='block_score' requires "
                "search_mode='atomic_rashomon_frontier'"
            )
        if not isinstance(self.adaptive_gradient_racing, bool):
            raise ValueError("adaptive_gradient_racing must be boolean")
        if self.adaptive_gradient_racing and self.search_mode not in {
            "fast_block_score",
            "gap_safe_rashomon_path",
            "successor_rashomon_path",
            "atomic_rashomon_frontier",
        }:
            raise ValueError(
                "adaptive_gradient_racing requires search_mode='fast_block_score' "
                "'gap_safe_rashomon_path', 'successor_rashomon_path', or "
                "'atomic_rashomon_frontier'"
            )
        if self.route_refinement_max_steps < 1:
            raise ValueError("route_refinement_max_steps must be positive")
        if (
            self.route_refinement_kkt_tolerance is not None
            and self.route_refinement_kkt_tolerance <= 0.0
        ):
            raise ValueError("route_refinement_kkt_tolerance must be positive or null")
        if self.max_rules_per_support is not None and (
            isinstance(self.max_rules_per_support, bool)
            or self.max_rules_per_support < 1
        ):
            raise ValueError("max_rules_per_support must be positive or null")
        if self.discovery_sampling not in {
            "full",
            "reference_cohort",
            "case_cohort_ipw",
        }:
            raise ValueError(
                "discovery_sampling must be 'full', 'reference_cohort', or "
                "'case_cohort_ipw'"
            )
        if self.discovery_reference_dataset is not None and (
            self.discovery_sampling != "reference_cohort"
        ):
            raise ValueError(
                "discovery_reference_dataset requires "
                "discovery_sampling='reference_cohort'; full always means "
                "complete D_fit"
            )
        if (
            self.discovery_sampling == "reference_cohort"
            and self.discovery_reference_dataset is None
        ):
            raise ValueError(
                "reference_cohort discovery requires discovery_reference_dataset"
            )
        if (
            self.formation_window_mode == "fit_quantile"
            and self.discovery_sampling not in {"full", "reference_cohort"}
        ):
            raise ValueError(
                "fit_quantile W requires complete D_fit or a frozen reference "
                "cohort; sampled discovery would change the finite dictionary"
            )
        if not 0 < self.discovery_noncase_fraction <= 1:
            raise ValueError("discovery_noncase_fraction must lie in (0, 1]")
        if self.discovery_sampling_seed < 0:
            raise ValueError("discovery_sampling_seed must be nonnegative")
        if self.solver_tolerance <= 0 or self.solver_max_iter < 1:
            raise ValueError("invalid solver controls")
        if self.search_tolerance < 0:
            raise ValueError("search_tolerance must be nonnegative")
        if self.exact_workers < 1 or self.pricing_workers < 1 or self.route_workers < 1:
            raise ValueError("worker counts must be positive")
        # Exact workers are response-matrix builders as well as CUDA solver
        # submitters.  More than one builder per physical GPU is useful when
        # matrix construction dominates Newton time: native CUDA workspaces
        # serialize per device while the CPU prepares the next exact design.
        # Keep the bound tied to the declared CPU pool so resident host
        # matrices remain deterministically bounded.
        if self.exact_workers > self.pricing_workers:
            raise ValueError("exact fit concurrency cannot exceed pricing_workers")
        if self.route_workers > max(1, len(self.pricing_devices)):
            raise ValueError(
                "route_workers cannot exceed the number of pricing devices"
            )
        for device in self.pricing_devices:
            if device != "cpu" and not (
                device.startswith("cuda:") and device[5:].isdigit()
            ):
                raise ValueError(f"invalid pricing device: {device}")
        if self.cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
