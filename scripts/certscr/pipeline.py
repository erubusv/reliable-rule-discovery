from __future__ import annotations

import copy
import ctypes
import ctypes.util
import gc
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .data import (
    EventData,
    ImplicitUnitGridWeights,
    QueryContext,
    ThreeWayContexts,
    make_context,
    split_contexts,
)
from .ensemble import evaluate_ensemble_sufficient, fit_intensity_ensemble
from .loss import ClusterLoss, financial_weighted_nll_loss, predictive_nll_loss
from .marked import (
    MarkBaseResidualizer,
    cluster_financial_mean_loss,
    cluster_mark_nll,
    event_mark_log_density,
    fit_mark_head,
    make_mark_base_residualizer,
    mark_score_moments,
)
from .model import (
    ClosureTerm,
    DeltaFactorizedSupportDesign,
    FitResult,
    IncrementalSupportPartition,
    PreparedFixedSupportDesign,
    SparseNuisancePartition,
    SparseDeltaSupportDesign,
    aggregate_duplicate_design_rows,
    append_rules_to_incremental_partition,
    cluster_exposure,
    cluster_nll,
    cloglog_event_nll,
    cloglog_event_terms,
    fixed_support_projected_kkt,
    fit_fixed_support,
    fit_delta_factorized_support,
    fit_sparse_delta_support,
    fit_sparse_delta_closure,
    fit_unconstrained_prepared_batch,
    group_saturated_poisson_lower_bound,
    prepare_fixed_support_design,
    prepare_delta_factorized_support_design,
    prepare_sparse_delta_support_design,
    prepare_sparse_nuisance_partition,
    project_prepared_support_design,
    promote_prepared_design_float64,
    refine_sparse_nuisance_partition,
    factorized_rule_recession_columns,
    sparse_delta_rule_recession_columns,
    update_incremental_support_partition,
)
from .occurrence import (
    CompletionEvents,
    RuleIdentity,
    RuleOccurrenceEngine,
    SourceEvents,
    SparseKernelResponse,
)
from .native import (
    add_sparse_linear_predictor,
    batched_sparse_rule_moments,
    sparse_component_integral,
    sorted_unique_int64_union,
    sorted_unique_int64_union_with_positions,
)
from .predicate_policy import resolve_predicate_policy_contract
from .statistics import (
    equivalence_mean_test,
    holm_adjust,
    numeric_information_positive_from_raw,
    one_sided_mean_test,
    one_sided_mean_test_zero_padded,
)


_MKL_LOCAL_THREAD_SETTER: object | None = None
_MKL_LOCAL_THREAD_LOOKUP_DONE = False
_MKL_LOCAL_THREAD_GUARD = threading.Lock()
_NATIVE_HEAP_TRIM_GUARD = threading.Lock()
_NATIVE_HEAP_TRIMMER: object | None = None
_NATIVE_HEAP_TRIM_LOOKUP_DONE = False
_NATIVE_HEAP_LAST_TRIM = -math.inf


def _release_unused_native_heap() -> bool:
    """Return dead NumPy/native arenas to the OS after a support-fit wave.

    Large closure partitions are intentionally short lived.  CPython releases
    their owners after a wave, but glibc may otherwise retain the underlying
    pages in a worker arena indefinitely.  That allocator retention, rather
    than live model state, caused the Freddie exact-audit process to grow until
    it was killed.  Collection and ``malloc_trim`` change neither a cached
    sufficient statistic nor a numerical result; unsupported allocators simply
    take the no-op path.
    """
    global _NATIVE_HEAP_TRIMMER, _NATIVE_HEAP_TRIM_LOOKUP_DONE
    global _NATIVE_HEAP_LAST_TRIM
    if os.environ.get("CERTSCR_MALLOC_TRIM", "1") == "0":
        return False
    with _NATIVE_HEAP_TRIM_GUARD:
        now = time.monotonic()
        try:
            interval = max(
                0.0,
                float(
                    os.environ.get(
                        "CERTSCR_MALLOC_TRIM_INTERVAL_SECONDS", "30"
                    )
                ),
            )
        except ValueError:
            interval = 30.0
        if now - _NATIVE_HEAP_LAST_TRIM < interval:
            return False
        _NATIVE_HEAP_LAST_TRIM = now
        gc.collect()
        if not _NATIVE_HEAP_TRIM_LOOKUP_DONE:
            try:
                function = ctypes.CDLL(None).malloc_trim
                function.argtypes = [ctypes.c_size_t]
                function.restype = ctypes.c_int
                _NATIVE_HEAP_TRIMMER = function
            except (AttributeError, OSError):
                _NATIVE_HEAP_TRIMMER = None
            _NATIVE_HEAP_TRIM_LOOKUP_DONE = True
        function = _NATIVE_HEAP_TRIMMER
        if function is None:
            return False
        try:
            return bool(function(0))
        except (OSError, TypeError, ValueError):
            return False


def _mkl_local_thread_setter():
    """Resolve MKL's thread-local limiter once without adding a dependency."""
    global _MKL_LOCAL_THREAD_SETTER, _MKL_LOCAL_THREAD_LOOKUP_DONE
    if _MKL_LOCAL_THREAD_LOOKUP_DONE:
        return _MKL_LOCAL_THREAD_SETTER
    with _MKL_LOCAL_THREAD_GUARD:
        if _MKL_LOCAL_THREAD_LOOKUP_DONE:
            return _MKL_LOCAL_THREAD_SETTER
        try:
            path = ctypes.util.find_library("mkl_rt")
            if path:
                library = ctypes.CDLL(path)
                setter = library.MKL_Set_Num_Threads_Local
                setter.argtypes = [ctypes.c_int]
                setter.restype = ctypes.c_int
                # Retain the library through the bound ctypes function.
                _MKL_LOCAL_THREAD_SETTER = setter
        except (AttributeError, OSError):
            _MKL_LOCAL_THREAD_SETTER = None
        _MKL_LOCAL_THREAD_LOOKUP_DONE = True
        return _MKL_LOCAL_THREAD_SETTER


@contextmanager
def _single_threaded_local_blas():
    """Prevent nested MKL pools inside independent exact-fit workers."""
    setter = _mkl_local_thread_setter()
    previous = int(setter(1)) if setter is not None else 0
    try:
        yield
    finally:
        if setter is not None:
            setter(previous)


@dataclass(frozen=True)
class CertSCRConfig:
    q_max: int = 3
    impact_lag: int = 12
    knot_count: int = 4
    max_formation_window: int = 12
    # None means the complete profiled rule library.  A finite value is only a
    # computational ablation and changes the feasible support space.
    max_support_size: int | None = None
    split_fractions: tuple[float, float, float] = (0.60, 0.20, 0.20)
    split_seed: int = 111
    split_strategy: str = "random_sequence"
    alpha_fit_screen: float = 0.05
    alpha_family: float = 0.05
    financial_threshold: float = 0.0
    rule_threshold: float = 0.0
    # ``auto`` selects marked-financial certification when genuine event marks
    # exist, legacy financial-loss certification when an explicit business
    # loss is supplied, and adverse-event early-warning certification
    # otherwise.  ``predictive`` is retained as an occurrence-only ablation.
    certification_mode: str = "auto"
    # Semantic metadata only: it never enters fitting.  Supplying it before a
    # run is required to make a domain-level adverse-financial-event claim.
    adverse_event_name: str | None = None
    early_warning_horizon: int | None = None
    # Absolute probability-point change, averaged over independent entities,
    # in the union of grid cells reached within the primary warning horizon.
    early_warning_threshold: float = 0.0
    # Recurrent-event models should condition every candidate support on the
    # target's own predictable history.  This is one pre-registered nuisance
    # block, never a rule antecedent.  CLI runs enable it by default; direct
    # library callers opt in explicitly to preserve existing estimands.
    target_history_control: bool = False
    # Outcome-blind seasoning nuisance for monthly first-event data. Separate
    # age-epoch clocks receive the same M-knot basis used by the TPP over each
    # impact_lag-month epoch, preventing rule atoms from proxying the ordinary
    # duration dependence of mortgage hazard.
    loan_age_baseline: bool = False
    # ``auto`` maps recurrent streams to the event-time Poisson likelihood and
    # monthly terminal outcomes to the exact interval-censored cloglog hazard.
    occurrence_likelihood: str = "auto"
    calibration_tolerance: float | None = None
    # Grouped sufficient-statistic fits are small, launch-bound convex solves.
    # Host float64 both reaches the declared KKT tolerance and is faster than
    # consumer-GPU float32 on the profiled financial datasets.
    solver_device: str = "cpu"
    solver_dtype: str = "float64"
    # Pure execution parallelism; it never caps or filters the rule family.
    solver_workers: int = 1
    solver_max_iter: int = 80
    solver_tolerance: float = 2.0e-5
    exhaustive_profile: bool = False
    support_devices: tuple[str, ...] = ()
    support_workers_per_device: int = 2
    support_search: str = "active_set"
    # ``all_atoms`` removes the arbitrary restart budget from the primary
    # estimator: the empty model and every D_fit-admitted rule atom are exact
    # deterministic starts.  ``stratified_budget`` is retained as a runtime
    # ablation for older experiments.
    active_start_policy: str = "all_atoms"
    active_restarts: int = 8
    # ``gradient_first_exact_audit`` uses score/Fisher pricing only to order
    # the complete add/drop coordinate neighborhood, takes the first exactly
    # fitted improvement, and stops only after a complete exact add/drop/swap
    # terminal audit. No budget or threshold removes a candidate, so its
    # reported terminal has the same one-exchange stationarity guarantee as
    # best-improvement, although the deterministic ascent path and attained
    # local terminal may differ. ``exact_one_exchange`` retains full-batch
    # best-improvement as a computational/comparative estimator.
    # ``mdl_score_working_set`` is the scalable estimator: every feasible
    # absent block is priced in one fused pass, but an exact child solve is
    # launched only when its cone-score gain pays its *exact incremental MDL
    # code length*.  There is no top-k, beam width, or candidate budget.  Its
    # terminal is block-score stationary (not exact one-exchange stationary),
    # while every accepted move is still decided by the ordinary exact fit and
    # scalar support objective.
    active_neighbor_strategy: str = "exact_one_exchange"
    # The reportable discovery family is fixed before D_cert.  The primary
    # estimator returns every positive standalone atom (including triplets)
    # and every unique atom-start terminal under the configured exact
    # neighborhood. Intermediate path
    # states are optimization traces, not additional scientific hypotheses.
    # ``visited_pool`` preserves the legacy broad-family ablation.
    support_family: str = "terminal_atoms"
    # No outcome-dependent family cap by default.  An explicit finite cap is
    # retained only for computational ablations; required anchors/terminals are
    # never removed even when it is supplied.
    support_pool_size: int | None = None
    search_improvement_tolerance: float = 1.0e-8
    support_conditioned_refinement: bool = True
    identity_profile: str = "dictionary_mdl"
    stratify_target_sequences: bool = False
    # Exhausting the finite skeleton dictionary is the non-heuristic default.
    # Heredity modes remain available only as explicitly requested ablations.
    triplet_generation: str = "all"
    fit_negative_sample_size: int | None = None
    feature_cache_bytes: int = 16 * 1024**3
    # Optional exact mmap store for retained D_fit rule/closure responses.
    # It changes only recomputation and resident-memory behavior; zero bytes
    # disables it.  The context/cohort digest is part of every on-disk key.
    persistent_response_dir: str | None = None
    persistent_response_bytes: int = 0
    # Exact entity-loss summaries for repeated hierarchy-null fits.  This is a
    # pure byte-bounded memoization layer; zero disables it without changing
    # any fitted model or test.
    loss_summary_cache_bytes: int = 1024**3
    # Full sparse summaries are retained only for model/context keys whose
    # frozen evaluation graph uses them more than once.  The byte cap affects
    # memoization only; zero disables it without changing any statistic.
    fit_summary_cache_bytes: int = 8 * 1024**3
    # CPU-native completion producers run ahead of the one/two GPU consumers.
    # This changes scheduling only; antecedents are still consumed in the same
    # deterministic batches and every exact result is keyed by identity.
    response_workers: int = 8
    gradient_pricing_only: bool = False
    max_gradient_triplets: int | None = None
    # Algebraic group-saturated likelihood bounds may reject only supports
    # whose best possible block-MDL is nonpositive.  Disabling this is a
    # performance ablation; it does not define a different statistical model.
    safe_mdl_screen: bool = True
    # After the local score working set admits an Add, compare a rigorous
    # support-specific likelihood upper bound with the current incumbent.
    # This is stronger than screening only against zero MDL and cannot remove
    # an exact improving move.  It is ignored by the legacy search modes.
    conditional_safe_mdl_screen: bool = True

    def validate(self) -> None:
        if not 1 <= self.q_max <= 3:
            raise ValueError("q_max must be between one and three")
        if self.impact_lag < 1 or self.knot_count < 1:
            raise ValueError("impact lag and knot count must be positive")
        if self.max_formation_window < 0:
            raise ValueError("formation window must be nonnegative")
        if self.max_support_size is not None and self.max_support_size < 1:
            raise ValueError("support size must be positive")
        if (
            len(self.split_fractions) != 3
            or any(not math.isfinite(value) or value <= 0 for value in self.split_fractions)
            or not math.isclose(sum(self.split_fractions), 1.0)
        ):
            raise ValueError("three positive split fractions must sum to one")
        if self.split_strategy not in {"random_sequence", "ordered_group"}:
            raise ValueError(
                "split strategy must be random_sequence or ordered_group"
            )
        if self.support_search not in {"active_set", "exhaustive"}:
            raise ValueError("support search must be active_set or exhaustive")
        if self.active_start_policy not in {"all_atoms", "stratified_budget"}:
            raise ValueError(
                "active start policy must be all_atoms or stratified_budget"
            )
        if self.active_neighbor_strategy not in {
            "gradient_first_exact_audit",
            "mdl_score_working_set",
            "exact_one_exchange",
        }:
            raise ValueError(
                "active neighbor strategy must be gradient_first_exact_audit, "
                "mdl_score_working_set, or exact_one_exchange"
            )
        if self.support_family not in {"terminal_atoms", "visited_pool"}:
            raise ValueError(
                "support family must be terminal_atoms or visited_pool"
            )
        if self.identity_profile not in {"exact", "score_mdl", "dictionary_mdl"}:
            raise ValueError("identity profile must be exact, score_mdl, or dictionary_mdl")
        if self.triplet_generation not in {
            "all",
            "weak_mdl_heredity",
            "connected_mdl_heredity",
            "strong_mdl_heredity",
        }:
            raise ValueError(
                "triplet generation must be all, weak_mdl_heredity, connected_mdl_heredity, "
                "or strong_mdl_heredity"
            )
        if self.fit_negative_sample_size is not None and self.fit_negative_sample_size < 1:
            raise ValueError("fit negative sample size must be positive")
        if self.feature_cache_bytes < 0:
            raise ValueError("feature cache bytes must be nonnegative")
        if self.persistent_response_bytes < 0:
            raise ValueError("persistent response bytes must be nonnegative")
        if self.persistent_response_bytes and not self.persistent_response_dir:
            raise ValueError(
                "persistent response bytes require a response-store directory"
            )
        if self.loss_summary_cache_bytes < 0:
            raise ValueError("loss summary cache bytes must be nonnegative")
        if self.fit_summary_cache_bytes < 0:
            raise ValueError("fit summary cache bytes must be nonnegative")
        if self.response_workers < 1:
            raise ValueError("response workers must be positive")
        if self.max_gradient_triplets is not None and self.max_gradient_triplets < 1:
            raise ValueError("max gradient triplets must be positive")
        if not isinstance(self.target_history_control, bool):
            raise ValueError("target history control flag must be boolean")
        if not isinstance(self.loan_age_baseline, bool):
            raise ValueError("loan-age baseline flag must be boolean")
        if self.occurrence_likelihood not in {
            "auto",
            "poisson",
            "first_event_cloglog",
        }:
            raise ValueError(
                "occurrence likelihood must be auto, poisson, or first_event_cloglog"
            )
        if not isinstance(self.safe_mdl_screen, bool):
            raise ValueError("safe MDL screen flag must be boolean")
        if not isinstance(self.conditional_safe_mdl_screen, bool):
            raise ValueError(
                "conditional safe MDL screen flag must be boolean"
            )
        if self.active_restarts < 1:
            raise ValueError("active-set restarts must be positive")
        if self.support_pool_size is not None and self.support_pool_size < 1:
            raise ValueError("support pool size must be positive")
        if not math.isfinite(self.search_improvement_tolerance) or self.search_improvement_tolerance < 0:
            raise ValueError("search improvement tolerance must be finite and nonnegative")
        if any(not str(device).strip() for device in self.support_devices):
            raise ValueError("support devices must be nonempty device names")
        if self.solver_workers < 1:
            raise ValueError("solver workers must be positive")
        if self.support_devices and self.solver_workers != 1:
            raise ValueError("solver workers and explicit support devices are mutually exclusive")
        if len(self.support_devices) == 1:
            raise ValueError("one support device is ambiguous; use solver_device, or provide at least two workers")
        if self.support_workers_per_device < 1:
            raise ValueError("support workers per device must be positive")
        if not (0 < self.alpha_fit_screen < 0.5):
            raise ValueError("fit-screen alpha must lie in (0, 0.5)")
        if not (0 < self.alpha_family < 0.5):
            raise ValueError("family alpha must lie in (0, 0.5)")
        if not math.isfinite(self.financial_threshold) or self.financial_threshold < 0:
            raise ValueError("financial contribution threshold must be finite and nonnegative")
        if not math.isfinite(self.rule_threshold) or self.rule_threshold < 0:
            raise ValueError("rule contribution threshold must be finite and nonnegative")
        if self.certification_mode not in {"auto", "early_warning", "predictive"}:
            raise ValueError(
                "certification mode must be auto, early_warning, or predictive"
            )
        if self.adverse_event_name is not None and not str(
            self.adverse_event_name
        ).strip():
            raise ValueError("adverse-event name must be nonempty when supplied")
        if self.early_warning_horizon is not None and not (
            1 <= int(self.early_warning_horizon) <= self.impact_lag
        ):
            raise ValueError(
                "early-warning horizon must lie in [1, impact_lag]"
            )
        if (
            not math.isfinite(self.early_warning_threshold)
            or not 0.0 <= self.early_warning_threshold < 1.0
        ):
            raise ValueError(
                "early-warning probability threshold must lie in [0, 1)"
            )
        if self.calibration_tolerance is not None and (
            not math.isfinite(self.calibration_tolerance) or self.calibration_tolerance <= 0
        ):
            raise ValueError("calibration tolerance must be finite and positive")
        if self.solver_max_iter < 1:
            raise ValueError("solver max iterations must be positive")
        if not math.isfinite(self.solver_tolerance) or self.solver_tolerance <= 0:
            raise ValueError("solver tolerance must be finite and positive")
        if self.solver_dtype not in {"float32", "float64"}:
            raise ValueError("solver dtype must be float32 or float64")
        if not str(self.solver_device).strip():
            raise ValueError("solver device must be nonempty")


@dataclass
class SupportRecord:
    rules: tuple[RuleIdentity, ...]
    fit: FitResult
    closure_baseline_fit: FitResult
    search_nll_improvement: float
    profile: str

    @property
    def active(self) -> bool:
        return bool(self.fit.theta.size and np.all(self.fit.amplitudes > 0.0))


@dataclass(frozen=True)
class SparseFitSummary:
    """Exact event predictor and per-sequence compensator for a sparse fit."""

    event_eta: np.ndarray
    active_grid_indices: np.ndarray
    active_grid_eta: np.ndarray
    cluster_intensity: np.ndarray
    cluster_nll: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(
            self.event_eta.nbytes
            + self.active_grid_indices.nbytes
            + self.active_grid_eta.nbytes
            + self.cluster_intensity.nbytes
            + self.cluster_nll.nbytes
        )


@dataclass(frozen=True)
class SparseLossSummary:
    """Entity sufficient statistics retained without active-grid predictors."""

    cluster_intensity: np.ndarray
    cluster_nll: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.cluster_intensity.nbytes + self.cluster_nll.nbytes)


def _mean_test_dict(test: object) -> dict:
    def finite(value: float) -> float | None:
        value = float(value)
        return value if math.isfinite(value) else None

    return {
        "estimate": finite(test.estimate),
        "standard_error": finite(test.standard_error),
        "statistic": finite(test.statistic),
        "p_value": float(test.p_value),
        "lower_bound": finite(test.lower_bound),
        "n_clusters": int(test.n_clusters),
    }


class CertSCRPipeline:
    def __init__(
        self,
        data: EventData,
        *,
        rule_predicates: Sequence[str],
        control_predicates: Sequence[str] = (),
        config: CertSCRConfig | None = None,
        certification_loss: ClusterLoss | None = None,
        predicate_policy_name: str | None = None,
    ):
        self.data = data
        self.config = config or CertSCRConfig()
        self.config.validate()
        if (
            torch is not None
            and str(self.config.solver_device).startswith("cpu")
            and len(self._support_worker_devices()) > 1
            and torch.get_num_threads() != 1
        ):
            # Independent small convex fits parallelize across solver workers;
            # nested BLAS pools only oversubscribe the same cores.  This is a
            # scheduling change and leaves every fitted objective untouched.
            torch.set_num_threads(1)
        self.marked = data.target_marks is not None
        if self.marked and self.config.identity_profile not in {"dictionary_mdl", "exact"}:
            raise ValueError(
                "marked discovery requires dictionary_mdl (joint score) or exact identity profiling"
            )
        if self.marked and data.sequence_financial_weights is not None:
            raise ValueError(
                "event-level financial marks and legacy sequence financial weights cannot be combined"
            )
        if certification_loss is not None:
            self.certification_loss = certification_loss
        elif data.sequence_financial_weights is not None:
            self.certification_loss = financial_weighted_nll_loss(
                data.sequence_financial_weights,
                weight_name=data.financial_weight_name or "unnamed_financial_weight",
            )
        else:
            self.certification_loss = predictive_nll_loss()
        self.certification_mode = self._resolve_certification_mode()
        name_to_id = {name: idx for idx, name in enumerate(data.predicate_names)}
        unknown = [name for name in (*rule_predicates, *control_predicates) if name not in name_to_id]
        if unknown:
            raise ValueError(f"unknown predicates: {unknown}")
        self.rule_source_ids = tuple(name_to_id[name] for name in rule_predicates)
        self.control_source_ids = tuple(name_to_id[name] for name in control_predicates)
        if set(self.rule_source_ids) & set(self.control_source_ids):
            raise ValueError("rule and control predicates must be disjoint")
        if not self.rule_source_ids:
            raise ValueError("empty rule predicate set")
        provenance = (
            data.preprocessing_provenance
            if isinstance(data.preprocessing_provenance, dict)
            else {}
        )
        raw_target_mode = provenance.get(
            "target_process", provenance.get("target_mode")
        )
        self.target_process_source = (
            "metadata.target_process"
            if "target_process" in provenance
            else "metadata.target_mode"
            if "target_mode" in provenance
            else "unknown"
        )
        self.target_process_mode = {
            "event_stream": "recurrent",
            "recurrent": "recurrent",
            "first_truncated": "first_event",
            "first_event": "first_event",
        }.get(str(raw_target_mode), "unknown")
        if self.target_process_mode == "unknown":
            legacy_leakage = provenance.get("leakage_policy")
            if (
                isinstance(legacy_leakage, dict)
                and legacy_leakage.get("is_laundering")
                == "used only as target_token"
            ):
                self.target_process_mode = "recurrent"
                self.target_process_source = "metadata.leakage_policy_legacy_adapter"
            elif provenance.get("forward_bias_rule") == (
                "Predicate tokens use only t and consecutive t-1. Rows after first T are removed. "
                "Predicate columns on the T row are set to 0 so T is consequent-only."
            ):
                self.target_process_mode = "first_event"
                self.target_process_source = "metadata.forward_bias_rule_legacy_adapter"
        self.occurrence_likelihood = (
            "first_event_cloglog"
            if self.config.occurrence_likelihood == "auto"
            and self.target_process_mode == "first_event"
            else "poisson"
            if self.config.occurrence_likelihood == "auto"
            else self.config.occurrence_likelihood
        )
        if (
            self.occurrence_likelihood == "first_event_cloglog"
            and self.target_process_mode != "first_event"
        ):
            raise ValueError(
                "first_event_cloglog requires preprocessing target_process=first_event"
            )
        if self.marked and self.occurrence_likelihood == "first_event_cloglog":
            raise ValueError(
                "marked two-stage fitting is not defined for the monthly first-event likelihood"
            )
        if (
            self.occurrence_likelihood == "first_event_cloglog"
            and self.config.calibration_tolerance is not None
        ):
            raise ValueError(
                "intensity calibration tolerance is not a probability calibration test for first-event cloglog"
            )
        self.target_history_source_id: int | None = None
        self.loan_age_baseline_source_ids: tuple[int, ...] = ()
        self.loan_age_baseline_milestones: tuple[int, ...] = ()
        self.target_history_control_requested = bool(
            self.config.target_history_control
        )
        raw_target_rows = np.flatnonzero(data.targets > 0)
        target_counts_by_sequence = np.bincount(
            data.sequence_codes,
            weights=data.targets.astype(np.float64, copy=False),
            minlength=data.n_sequences,
        )
        self.first_event_target_multiplicity_valid = bool(
            np.all(target_counts_by_sequence <= 1.0)
        )
        self.target_has_post_event_exposure = bool(
            len(raw_target_rows)
            and np.any(
                data.times[raw_target_rows]
                < data.end_times[data.sequence_codes[raw_target_rows]]
            )
        )
        # With strict lag >= 1, a target occurring at the final exposed grid
        # cell cannot contribute to any event or exposure row.  Dropping this
        # identically-zero nuisance block removes a singular M-column system
        # while preserving the objective and its fitted intensity exactly.
        self.target_history_response_structural_zero = bool(
            not self.target_has_post_event_exposure
        )
        self.target_history_structural_zero = bool(
            self.target_history_control_requested
            and self.target_history_response_structural_zero
        )
        target_history_sources: dict[int, SourceEvents] = {}
        if (
            self.target_history_control_requested
            and not self.target_history_structural_zero
        ):
            target_rows = raw_target_rows
            # QueryContext expands target_token multiplicity in the event term;
            # repeat it here as well so the predictable self-history is the
            # exact counting-process convolution after that bin, not merely a
            # binary "some target happened" flag.  Lag zero remains excluded.
            target_rows = np.repeat(
                target_rows,
                data.targets[target_rows].astype(np.int64, copy=False),
            )
            target_sequences = data.sequence_codes[target_rows].astype(
                np.int32, copy=False
            )
            target_times = data.times[target_rows].astype(np.int32, copy=False)
            counts = np.bincount(
                target_sequences, minlength=data.n_sequences
            ).astype(np.int64, copy=False)
            offsets = np.zeros(data.n_sequences + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(counts, dtype=np.int64)
            self.target_history_source_id = int(data.n_predicates)
            target_history_sources[self.target_history_source_id] = SourceEvents(
                sequence_codes=target_sequences,
                times=target_times,
                offsets=offsets,
                populated_sequences=np.flatnonzero(counts).astype(
                    np.int32, copy=False
                ),
            )
        age_baseline_sources: dict[int, SourceEvents] = {}
        if self.config.loan_age_baseline:
            if self.target_process_mode != "first_event":
                raise ValueError(
                    "loan-age baseline is defined only for first-event data"
                )
            if data.sequence_start_ages is None:
                raise ValueError(
                    "loan-age baseline requires start_loan_age in the sequence table"
                )
            start_ages = np.asarray(data.sequence_start_ages, dtype=np.int64)
            if start_ages.shape != (data.n_sequences,) or np.any(start_ages < 0):
                raise ValueError("invalid sequence start ages")
            start_times = data.start_times.astype(np.int64)
            end_times = data.end_times.astype(np.int64)
            end_ages = start_ages + end_times - start_times
            interval = int(self.config.impact_lag)
            max_age = int(np.max(end_ages, initial=0))
            milestones = tuple(range(0, max_age + 1, interval))
            next_source_id = int(data.n_predicates) + int(
                self.target_history_source_id is not None
            )
            source_ids: list[int] = []
            retained_milestones: list[int] = []
            for milestone in milestones:
                source_time = start_times - start_ages + int(milestone)
                # A clock may be left of the observed prefix. Retain it exactly
                # when one of its predictable lags 1..L intersects observed
                # risk time; no target value enters this construction.
                included = (
                    (source_time + 1 <= end_times)
                    & (source_time + interval >= start_times)
                )
                if not np.any(included):
                    continue
                sequences = np.flatnonzero(included).astype(np.int32)
                times = source_time[included].astype(np.int32)
                offsets = np.zeros(data.n_sequences + 1, dtype=np.int64)
                offsets[1:] = np.cumsum(
                    included.astype(np.int64), dtype=np.int64
                )
                source_id = next_source_id + len(source_ids)
                age_baseline_sources[source_id] = SourceEvents(
                    sequence_codes=sequences,
                    times=times,
                    offsets=offsets,
                    populated_sequences=sequences.copy(),
                )
                source_ids.append(source_id)
                retained_milestones.append(int(milestone))
            self.loan_age_baseline_source_ids = tuple(source_ids)
            self.loan_age_baseline_milestones = tuple(retained_milestones)
        self.predicate_policy_name = (
            str(predicate_policy_name) if predicate_policy_name is not None else None
        )
        self.predicate_policy_contract = (
            resolve_predicate_policy_contract(self.predicate_policy_name)
            if self.predicate_policy_name is not None
            else None
        )
        if self.predicate_policy_contract is not None and tuple(rule_predicates) != tuple(
            self.predicate_policy_contract.predicates
        ):
            raise ValueError(
                "rule predicates do not exactly match the named pre-registered policy"
            )
        self.splits: ThreeWayContexts = split_contexts(
            data,
            fractions=self.config.split_fractions,
            seed=self.config.split_seed,
            # A marked fit and both held-out financial tests require observed
            # target marks.  Stratification is automatic here; it does not use
            # the mark amount and therefore cannot select favorable outcomes.
            stratify_target=(self.config.stratify_target_sequences or self.marked),
            fit_negative_sample_size=self.config.fit_negative_sample_size,
            strategy=self.config.split_strategy,
        )
        self.fit_sampling_weights = self.splits.fit_sampling_weights.astype(np.float64, copy=True)
        raw_fit_weights = self.certification_loss.weights(self.splits.fit) * self.fit_sampling_weights
        self.fit_sampling_scale = float(np.mean(self.fit_sampling_weights))
        self.fit_sampling_ess = float(
            np.sum(self.fit_sampling_weights) ** 2
            / np.sum(self.fit_sampling_weights**2)
        )
        self.fit_weight_scale = float(np.mean(raw_fit_weights))
        if not math.isfinite(self.fit_weight_scale) or self.fit_weight_scale <= 0:
            raise ValueError("fit-split loss weights must have a finite positive mean")
        # A common positive scale does not change the exact optimizer.  Mean-one
        # weights keep Newton/KKT numerics comparable between currency-valued
        # financial exposure and the ordinary predictive likelihood.
        self.fit_cluster_weights = raw_fit_weights / self.fit_weight_scale
        self.fit_cluster_weights_are_unit = bool(
            np.all(self.fit_cluster_weights == 1.0)
        )
        population_loss_weights = self.certification_loss.global_sequence_weights
        self.fit_population_loss_weight_mean = (
            1.0
            if population_loss_weights is None
            else float(
                np.mean(
                    np.asarray(population_loss_weights, dtype=np.float64)[
                        self.splits.fit_population_global_ids
                    ]
                )
            )
        )
        if (
            not math.isfinite(self.fit_population_loss_weight_mean)
            or self.fit_population_loss_weight_mean <= 0.0
        ):
            raise ValueError("D_fit population loss weights must have a finite positive mean")
        # The solver uses sample-mean-one weights for stable KKT numerics.  MDL
        # must restore the HT estimate of the *population-normalized* loss.  For
        # ordinary/marked likelihood this equals mean(IPW); for legacy financial
        # weights it also removes arbitrary currency/exposure units.
        self.fit_objective_population_scale = (
            self.fit_weight_scale / self.fit_population_loss_weight_mean
        )
        self.engine = RuleOccurrenceEngine(
            data,
            lag=self.config.impact_lag,
            knot_count=self.config.knot_count,
            feature_cache_bytes=self.config.feature_cache_bytes,
            persistent_response_dir=self.config.persistent_response_dir,
            persistent_response_bytes=self.config.persistent_response_bytes,
            max_completion_span=self.config.max_formation_window,
            extra_source_events={
                **target_history_sources,
                **age_baseline_sources,
            },
        )
        # Weighted total exposure is needed by every fixed-support fit.  Cache
        # the exact per-sequence quadrature sums once instead of scanning the
        # full grid or materializing a weighted grid vector for every support.
        self._sequence_exposures: dict[str, np.ndarray] = {
            ctx.name: ctx.sequence_exposures()
            for ctx in (self.splits.fit, self.splits.cert, self.splits.test)
        }
        self._fit_cache: dict[tuple[tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]], FitResult] = {}
        # Evaluation workers share fitted immutable models.  A per-model lock
        # prevents two supports from solving the same hierarchy branch-drop
        # model concurrently; unrelated convex fits remain parallel.
        self._fit_key_locks: dict[
            tuple[tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]],
            threading.Lock,
        ] = {}
        self._fit_key_locks_guard = threading.Lock()
        # Hierarchy-null lookup is hot during active support search.  Keeping a
        # closure-only index avoids scanning every cached nonnull support fit
        # whenever a new closure is encountered.
        self._null_fit_cache: dict[tuple[ClosureTerm, ...], FitResult] = {}
        self._hierarchy_closure_cache: dict[
            tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]
        ] = {}
        self._safe_bound_cache: dict[tuple[RuleIdentity, ...], dict] = {}
        self._prepared_design_cache: dict[
            tuple[tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]],
            PreparedFixedSupportDesign,
        ] = {}
        self._fixed_nuisance_partitions: dict[str, SparseNuisancePartition] = {}
        self._fixed_nuisance_partition_guard = threading.Lock()
        # Profiling computes the exact new-block KKT at the converged closure
        # null before staging a child design.  Keys enter this set only when
        # that residual already proves the expanded null is not optimal; the
        # scalar fitter can then skip an otherwise duplicate full Hessian pass.
        # The marker changes no warm start, objective or convergence test.
        self._prepared_profile_nonoptimal_keys: set[
            tuple[tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]]
        ] = set()
        # Stage-local parent design used only while one nested support group is
        # screened.  The mutable slot is worker-local and released at the group
        # boundary, so peak memory is independent of the number of supports.
        self._information_design_parent: list[object] | None = None
        self._safe_screened_records: dict[tuple[RuleIdentity, ...], SupportRecord] = {}
        self._nuisance_event_design_cache: dict[
            tuple[str, tuple[ClosureTerm, ...]], np.ndarray
        ] = {}
        self._mark_base_residualizer_cache: dict[
            tuple[str, tuple[ClosureTerm, ...]], MarkBaseResidualizer
        ] = {}
        self._event_grid_count_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._inference_weight_cache: dict[str, np.ndarray] = {}
        # Stage-local union of dictionary response rows for fused multi-state
        # pricing.  Atom-start waves usually differ only in which one rule is
        # absent, so their union dictionary is identical.  Retain one int64
        # row vector, not another copy of any response value, and release it
        # when support search completes.
        self._support_pricing_union_cache: tuple[
            tuple[RuleIdentity, ...], np.ndarray
        ] | None = None
        # Early-warning geometry depends only on the frozen split and rule
        # identity, yet the same rule is tested in many support models.  Cache
        # sequence labels, quadrature and event multiplicities once per rule;
        # no fitted predictor or split outcome decision is cached here.
        self._early_warning_geometry_cache: dict[
            tuple[str, int, tuple[int, ...], int, int],
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
        ] = {}
        self._early_warning_geometry_guard = threading.Lock()
        self._marked_response_cache: dict[
            tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        # Many candidate supports share one hierarchy-closure null (2,824 IBM
        # supports reduced to 388 nulls in the audited run).  Retain only its
        # entity-level loss/intensity, never the much larger active-grid
        # predictor.  Ordered eviction makes peak memory independent of the
        # number of searched supports.
        self._loss_summary_cache: OrderedDict[
            tuple[str, int, int], tuple[FitResult, QueryContext, SparseLossSummary]
        ] = OrderedDict()
        # Mutable scalar is deliberately shared by shallow worker copies along
        # with the OrderedDict and lock.
        self._loss_summary_cache_size = [0]
        self._loss_summary_cache_guard = threading.Lock()
        self._loss_summary_key_locks: dict[tuple[str, int, int], threading.Lock] = {}
        self._loss_summary_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        # Full summaries are substantially larger than closure-loss summaries,
        # so cache only keys proven reusable by the frozen support/drop graph.
        # These objects are shared by shallow evaluation-worker copies.
        self._fit_summary_cache: OrderedDict[
            tuple[str, int, tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]],
            tuple[FitResult, QueryContext, SparseFitSummary],
        ] = OrderedDict()
        self._fit_summary_cache_size = [0]
        self._fit_summary_cache_guard = threading.Lock()
        self._fit_summary_key_locks: dict[
            tuple[str, int, tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]],
            threading.Lock,
        ] = {}
        self._fit_summary_cacheable_keys: set[
            tuple[str, int, tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]]
        ] = set()
        self._fit_summary_cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "eligible_model_keys": 0,
        }
        self._safe_screen_stats = {
            "bound_evaluations": 0,
            "screened_supports": 0,
            "exact_fit_supports": 0,
            "child_kkt_shortcuts": 0,
            "nested_null_warm_starts": 0,
            "active_fit_thread_batches": 0,
            "active_fit_process_batches": 0,
            "active_fit_closure_groups": 0,
            "active_fit_dynamic_jobs": 0,
            "active_fit_dynamic_worker_launches": 0,
            "active_fit_dynamic_thread_tasks": 0,
            "active_fit_shared_closure_null_prefits": 0,
            "active_fit_closure_shards": 0,
            "support_closure_partitions": 0,
            "support_closure_partition_child_reuses": 0,
            "support_closure_partition_fallbacks": 0,
            "support_joint_null_kkt_screens": 0,
            "support_delta_factorized_fits": 0,
            "support_sparse_delta_fits": 0,
            "support_sparse_delta_closure_groups": 0,
            "support_sparse_delta_fallbacks": 0,
            "support_nested_exact_kkt_screens": 0,
            "support_semantic_warm_starts": 0,
            "profile_fit_results_reused": 0,
            "profile_null_fit_batches": 0,
            "profile_null_models_batched": 0,
            "profile_null_models_scalar": 0,
            "profile_window_null_warm_starts": 0,
            "identity_zero_boundary_kkt_screens": 0,
            "identity_sign_pair_parent_designs": 0,
            "identity_sign_pair_child_reuses": 0,
            "identity_fused_profile_exact_fits": 0,
            "identity_redundant_null_kkt_passes_skipped": 0,
            "identity_no_event_activation_screens": 0,
            "identity_incremental_partition_rebuilds": 0,
            "nested_prepared_parent_designs": 0,
            "nested_prepared_child_reuses": 0,
            "rule_recession_support_rejections": 0,
            "rule_recession_columns": 0,
        }
        self._diagnostic_guard = threading.Lock()
        self.baseline_fit: FitResult | None = None
        self.profiled_rules: list[RuleIdentity] = []
        self._profile_completed = False
        self.identity_candidates: dict[tuple[int, ...], tuple[RuleIdentity, ...]] = {}
        self.profile_logs: list[dict] = []
        self.support_records: list[SupportRecord] = []
        self.candidate_records: list[SupportRecord] = []
        self.last_fit_screen: dict | None = None
        self.last_certification: dict | None = None
        self.search_diagnostics: dict = {}
        self._active_support_workers: list[CertSCRPipeline] = []
        self._runtime_checkpoint_path: Path | None = None
        self._runtime_checkpoint_started = 0.0
        self._runtime_checkpoint_last_write = -math.inf
        self._runtime_checkpoint_interval = 60.0
        self._checkpoint_signature_cache: str | None = None
        self._identity_refinement_keys: set[
            tuple[RuleIdentity, ...]
        ] = set()
        self.rule_dictionary_shapes: dict[RuleIdentity, np.ndarray] = {}
        self.kernel_dictionary = self._contiguous_kernel_dictionary(self.config.knot_count)
        if self.marked:
            marks = np.asarray(self.splits.fit.event_marks, dtype=np.float64)
            if marks.size == 0 or np.any(~np.isfinite(marks)) or np.any(marks <= 0):
                raise ValueError("D_fit must contain finite positive financial marks")
            # A fixed, outcome-scale normalizer only changes units.  The sample
            # median is equivariant to currency rescaling and is frozen before
            # any rule support is searched.
            self.mark_unit = float(np.median(marks))
        else:
            self.mark_unit = None
        self.mark_variance: float | None = None

    def _resolve_certification_mode(self) -> str:
        """Resolve the reportable estimand without inventing a financial mark."""
        if self.marked:
            return "marked_financial"
        if self.certification_loss.financially_grounded:
            return "financial_loss"
        if self.config.certification_mode == "auto":
            return "early_warning"
        return self.config.certification_mode

    def _f0_contract(self) -> dict:
        """Return the auditable, target-independent F0 eligibility screen.

        F0 cannot establish that a human-provided target is economically
        adverse from its array values.  It can enforce that the semantic label
        was supplied, that the rule dictionary is a frozen reviewed policy,
        and that no unreviewed predicate control enters the predictor.  The
        registered target self-history nuisance is allowed because it is
        fixed for every null/full comparison and can never enter a rule.
        The occurrence
        construction then guarantees a minimum effect lag of one, so target
        rows at time t cannot affect their own prediction.
        """
        adverse_name = (
            str(self.config.adverse_event_name).strip()
            if self.config.adverse_event_name is not None
            else None
        )
        contract = self.predicate_policy_contract
        provenance = self.data.preprocessing_provenance
        raw_f0 = (
            provenance.get("f0_contract")
            if isinstance(provenance, dict)
            else None
        )
        provenance_verified = False
        provenance_source = None
        if isinstance(raw_f0, dict):
            target_history_contract = raw_f0.get(
                "predicate_history_includes_target_labeled_observations"
            )
            history_contract_valid = (
                target_history_contract is True
                if self.target_process_mode == "recurrent"
                else True
                if self.target_process_mode == "first_event"
                else target_history_contract is not False
            )
            provenance_verified = bool(
                raw_f0.get("dynamic_predicates") is True
                and raw_f0.get("outcome_blind_predicate_construction") is True
                and raw_f0.get("direct_target_proxy_excluded") is True
                and raw_f0.get("strict_future_effect_required") is True
                and history_contract_valid
            )
            provenance_source = "metadata.f0_contract"
        elif isinstance(provenance, dict):
            # Backward-compatible audit for IBM outputs produced before the
            # explicit F0 block was added.  ``False`` means the target label
            # was used to delete observable transactions from later history.
            leakage = provenance.get("leakage_policy")
            if isinstance(leakage, dict) and "laundering_transaction_predicates" in leakage:
                provenance_verified = bool(
                    leakage.get("laundering_transaction_predicates") is True
                    and leakage.get("is_laundering") == "used only as target_token"
                    and leakage.get("pattern_or_typology_labels") == "not used"
                )
                provenance_source = "metadata.leakage_policy_legacy_adapter"
        reasons: list[str] = []
        if adverse_name is None:
            reasons.append("adverse_event_name_not_pre_specified")
        if contract is None:
            reasons.append("rule_predicate_policy_not_pre_registered")
        elif not contract.f0_eligible:
            reasons.append("predicate_policy_failed_semantic_review")
        if not provenance_verified:
            reasons.append("dataset_preprocessing_provenance_not_F0_verified")
        if self.control_source_ids:
            reasons.append("control_predicates_have_no_registered_predictability_contract")
        if (
            self.target_process_mode == "recurrent"
            and self.target_history_source_id is None
            and not self.target_history_response_structural_zero
        ):
            reasons.append("recurrent_target_history_control_missing")
        if (
            self.target_process_mode == "first_event"
            and self.target_has_post_event_exposure
        ):
            reasons.append("first_event_risk_set_continues_after_target")
        if (
            self.target_process_mode == "first_event"
            and not self.first_event_target_multiplicity_valid
        ):
            reasons.append("first_event_process_contains_repeated_targets")
        if self.certification_mode != "early_warning":
            reasons.append("F0_is_defined_for_the_unmarked_early_warning_claim")
        passed = not reasons
        return {
            "name": "F0_pre_registered_financial_forecast_eligibility",
            "passed": passed,
            "target": adverse_name,
            "target_semantics_pre_specified": adverse_name is not None,
            "predicate_policy": self.predicate_policy_name,
            "policy_registered": contract is not None,
            "dynamic": bool(contract.dynamic) if contract is not None else False,
            "outcome_blind_construction": (
                bool(contract.outcome_blind_construction) if contract is not None else False
            ),
            "direct_target_proxy_excluded": (
                bool(contract.direct_target_proxy_excluded) if contract is not None else False
            ),
            "atomic_events": bool(contract.atomic_events) if contract is not None else None,
            "dataset_preprocessing_provenance_verified": provenance_verified,
            "dataset_preprocessing_provenance_source": provenance_source,
            "strict_future_effect_lag": 1,
            "target_process": self.target_process_mode,
            "target_process_source": self.target_process_source,
            "first_event_risk_set_valid": bool(
                not self.target_has_post_event_exposure
                and self.first_event_target_multiplicity_valid
            ) if self.target_process_mode == "first_event" else None,
            "unreviewed_control_count": len(self.control_source_ids),
            "target_history_control": {
                "requested": self.target_history_control_requested,
                "enabled": self.target_history_source_id is not None,
                "omitted_as_structural_zero": self.target_history_structural_zero,
                "role": "registered recurrent-target nuisance; excluded from rule grammar",
                "strict_minimum_lag": 1,
            },
            "loan_age_baseline": {
                "requested": bool(self.config.loan_age_baseline),
                "enabled": bool(self.loan_age_baseline_source_ids),
                "milestones": list(self.loan_age_baseline_milestones),
                "epoch_width_months": int(self.config.impact_lag),
                "role": (
                    "registered target-blind first-event seasoning nuisance; "
                    "excluded from rule grammar"
                ),
            },
            "failure_reasons": reasons,
            "scope": (
                "semantic and temporal eligibility only; not a causal, monetary-utility, "
                "or regime-invariance claim"
            ),
        }

    @property
    def early_warning_horizon(self) -> int:
        return int(
            self.config.impact_lag
            if self.config.early_warning_horizon is None
            else self.config.early_warning_horizon
        )

    @property
    def algorithm_name(self) -> str:
        if self.marked:
            return "FR-Marked-SCR-TPP"
        if self.certification_mode == "early_warning":
            return "EW-CertSCR-TPP"
        return "CER-SCR-TPP"

    @staticmethod
    def _contiguous_kernel_dictionary(knot_count: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        for left in range(int(knot_count)):
            for right in range(left, int(knot_count)):
                row = np.zeros(int(knot_count), dtype=np.float64)
                row[left : right + 1] = 1.0 / float(right - left + 1)
                rows.append(row)
        return np.stack(rows, axis=0)

    def _support_worker_devices(self) -> tuple[str, ...]:
        """Expand physical devices into CPU-preparation/GPU-consumer workers."""
        if not self.config.support_devices:
            return tuple(
                str(self.config.solver_device)
                for _ in range(int(self.config.solver_workers))
            )
        return tuple(
            device
            for device in self.config.support_devices
            for _ in range(self.config.support_workers_per_device)
        )

    def _effective_max_support_size(self) -> int:
        """Finite cap induced only by the frozen profiled rule dictionary."""
        library_size = len(self.profiled_rules)
        if self.config.max_support_size is None:
            return library_size
        return min(int(self.config.max_support_size), library_size)

    @staticmethod
    def _union_sparse_indices(
        blocks: Sequence[SparseKernelResponse],
    ) -> np.ndarray:
        """Exact sparse-row union with one concatenate/unique pass."""
        parts = [block.grid_indices for block in blocks if len(block.grid_indices)]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        if len(parts) == 1:
            return parts[0].copy()
        native = sorted_unique_int64_union(parts)
        if native is not None:
            return native
        return np.unique(np.concatenate(parts)).astype(np.int64, copy=False)

    @classmethod
    def _sparse_union_layout(
        cls,
        blocks: Sequence[SparseKernelResponse],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Return one row union and each block's exact positions in that union."""
        block_tuple = tuple(blocks)
        native = sorted_unique_int64_union_with_positions(
            [block.grid_indices for block in block_tuple],
            assume_sorted=True,
        )
        if native is not None:
            return native
        union = cls._union_sparse_indices(block_tuple)
        position_dtype = np.int32 if len(union) <= np.iinfo(np.int32).max else np.int64
        positions: list[np.ndarray] = []
        for block in block_tuple:
            if not len(block.grid_indices):
                positions.append(np.zeros(0, dtype=position_dtype))
                continue
            mapped = np.searchsorted(union, block.grid_indices)
            if (
                np.any(mapped >= len(union))
                or not np.array_equal(union[mapped], block.grid_indices)
            ):
                raise AssertionError("sparse block rows are missing from their union")
            positions.append(mapped.astype(position_dtype, copy=False))
        return union, tuple(positions)

    @staticmethod
    def _cone_quadratic_gain(gradient: np.ndarray, information: np.ndarray) -> float:
        """Exact nonnegative-cone gain for the local quadratic model."""
        gradient = np.asarray(gradient, dtype=np.float64)
        information = np.asarray(information, dtype=np.float64)
        m = len(gradient)
        best = 0.0
        scale = max(1.0, float(np.linalg.norm(information, ord=2)))
        tolerance = np.finfo(np.float64).eps * scale * max(1, m)
        for mask in range(1, 1 << m):
            active = np.asarray([index for index in range(m) if mask & (1 << index)], dtype=np.int64)
            h = information[np.ix_(active, active)]
            g = gradient[active]
            try:
                left, singular, right = np.linalg.svd(
                    h, full_matrices=False
                )
                if not singular.size or singular[0] <= 0.0:
                    continue
                rank_tolerance = np.finfo(np.float64).eps * max(h.shape) * singular[0]
                inverse = np.divide(
                    1.0,
                    singular,
                    out=np.zeros_like(singular),
                    where=singular > rank_tolerance,
                )
                theta = -right.T @ (inverse * (left.T @ g))
            except np.linalg.LinAlgError:
                continue
            if np.any(theta <= tolerance):
                continue
            full_theta = np.zeros(m, dtype=np.float64)
            full_theta[active] = theta
            final_gradient = gradient + information @ full_theta
            inactive = np.ones(m, dtype=bool)
            inactive[active] = False
            if np.any(final_gradient[inactive] < -tolerance):
                continue
            gain = float(-(gradient @ full_theta + 0.5 * full_theta @ information @ full_theta))
            if math.isfinite(gain):
                best = max(best, gain)
        return best

    def _identity_moments(
        self,
        null_fit: FitResult,
        closure_terms: Sequence[ClosureTerm],
        antecedent: tuple[int, ...],
        window: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score gradient/information shared by both signs and all shapes."""
        ctx = self.splits.fit
        response = self.engine.sparse_response(ctx, antecedent, int(window))
        return self._identity_moments_at_null(
            null_fit,
            response,
        )

    def _eta_on_sparse_grid(
        self,
        fit: FitResult,
        ctx: QueryContext,
        grid_indices: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a frozen fitted predictor on selected quadrature rows."""
        rows = np.asarray(grid_indices, dtype=np.int64).reshape(-1)
        eta = np.full(len(rows), float(fit.alpha), dtype=np.float64)
        gamma_offset = 0
        for block in self.sparse_nuisance_blocks(ctx, fit.closure_terms):
            width = int(block.shape[1])
            block.add_grid_linear_predictor(
                rows,
                fit.gamma[gamma_offset : gamma_offset + width],
                eta,
                assume_sorted_unique=True,
            )
            gamma_offset += width
        if gamma_offset != len(fit.gamma):
            raise ValueError("fit/design mismatch")
        for index, (rule, feature) in enumerate(
            zip(fit.rules, self.sparse_features(ctx, fit.rules), strict=True)
        ):
            if feature.shape[1] != fit.theta.shape[1]:
                raise ValueError("fit/design mismatch")
            feature.add_grid_linear_predictor(
                rows,
                fit.theta[index],
                eta,
                scale=float(rule.sign),
                assume_sorted_unique=True,
            )
        return eta

    def _eta_on_events(self, fit: FitResult, ctx: QueryContext) -> np.ndarray:
        """Evaluate a frozen fit only on observed-event query rows."""
        eta = np.full(ctx.n_events, float(fit.alpha), dtype=np.float64)
        offset = 0
        for block in self.sparse_nuisance_blocks(ctx, fit.closure_terms):
            width = int(block.shape[1])
            eta += block.event_values @ fit.gamma[offset : offset + width]
            offset += width
        if offset != len(fit.gamma):
            raise ValueError("fit/design mismatch")
        for index, (rule, block) in enumerate(
            zip(fit.rules, self.sparse_features(ctx, fit.rules), strict=True)
        ):
            eta += float(rule.sign) * (block.event_values @ fit.theta[index])
        return eta

    def _identity_moments_at_null(
        self,
        null_fit: FitResult,
        response: SparseKernelResponse,
        *,
        grid_eta: np.ndarray | None = None,
        event_eta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact score/Fisher moments without any full-grid response or weight."""
        m = int(response.shape[1])
        if not len(response.grid_indices):
            return np.zeros(m, dtype=np.float64), np.zeros((m, m), dtype=np.float64)
        ctx = self.splits.fit
        rows = response.grid_indices
        if (
            self.occurrence_likelihood == "first_event_cloglog"
            and len(rows)
            and ctx.n_events
        ):
            event_rows = np.asarray(ctx.event_grid_rows, dtype=np.int64)
            positions = np.searchsorted(event_rows, rows)
            safe = np.minimum(positions, len(event_rows) - 1)
            keep = (positions >= len(event_rows)) | (event_rows[safe] != rows)
            rows = rows[keep]
            grid_values = response.grid_values[keep]
        else:
            grid_values = response.grid_values
        eta = (
            self._eta_on_sparse_grid(null_fit, ctx, rows)
            if grid_eta is None
            else np.asarray(grid_eta, dtype=np.float64)
        )
        if eta.shape != (len(rows),):
            raise ValueError("precomputed sparse predictor does not align with response")
        sequence_weights = self.fit_cluster_weights
        grid_weights = (
            ctx.grid_weights_at(rows, assume_valid=True)
            * sequence_weights[
                ctx.grid_sequences_at(
                    rows, assume_valid=True, assume_sorted=True
                )
            ]
            * np.exp(eta)
        )
        if np.any(~np.isfinite(grid_weights)):
            raise FloatingPointError("nonfinite fitted intensity during rule pricing")
        block = grid_values.astype(np.float64, copy=False)
        gradient = block.T @ grid_weights
        if ctx.n_events:
            event_design = response.event_values.astype(np.float64, copy=False)
            event_weights = sequence_weights[ctx.event_sequence_local]
            if self.occurrence_likelihood == "poisson":
                gradient -= event_design.T @ event_weights
                event_information_weight = np.zeros(ctx.n_events, dtype=np.float64)
            else:
                event_predictor = (
                    self._eta_on_events(null_fit, ctx)
                    if event_eta is None
                    else np.asarray(event_eta, dtype=np.float64)
                )
                if event_predictor.shape != (ctx.n_events,):
                    raise ValueError(
                        "precomputed event predictor does not align with response"
                    )
                _loss, event_gradient, event_hessian = cloglog_event_terms(
                    event_predictor
                )
                gradient += event_design.T @ (event_weights * event_gradient)
                event_information_weight = event_weights * event_hessian
        information = block.T @ (grid_weights[:, None] * block)
        if ctx.n_events and self.occurrence_likelihood == "first_event_cloglog":
            information += event_design.T @ (
                event_information_weight[:, None] * event_design
            )
        return gradient, 0.5 * (information + information.T)

    def _identity_moments_from_grouped_parent(
        self,
        null_fit: FitResult,
        parent: PreparedFixedSupportDesign,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact new-block score/Fisher from an already grouped W parent.

        The full-M parent is a finer sufficient-statistic partition of the
        closure null and contains the candidate block in its final columns.
        Re-evaluating the same moments from sparse source streams regenerated
        every hierarchy convolution and then scanned the ungrouped active
        grid.  Chunked promotion of the grouped rows preserves the float32
        design values and performs the identical weighted derivatives with a
        bounded temporary.
        """
        if len(parent.rules) != 1:
            raise ValueError("identity moment parent must contain one full-M rule")
        left = int(parent.constrained_start)
        m = int(parent.knot_count)
        if m < 1 or parent.design.shape[1] != left + m:
            raise ValueError("identity moment parent has an invalid block layout")
        if len(null_fit.gamma) != int(parent.control_width):
            raise ValueError("identity null coefficients do not match grouped parent")
        coefficients = np.concatenate(
            (np.asarray([null_fit.alpha], dtype=np.float64), null_fit.gamma)
        )
        gradient = np.zeros(m, dtype=np.float64)
        information = np.zeros((m, m), dtype=np.float64)
        chunk_size = 262_144

        def accumulate(start: int, stop: int, *, event: bool) -> None:
            weights_source = parent.event_weights if event else parent.grid_weights
            source_offset = 0 if event else int(parent.n_events)
            for row_left in range(start, stop, chunk_size):
                row_right = min(row_left + chunk_size, stop)
                raw = parent.design[row_left:row_right]
                prefix = raw[:, :left].astype(np.float64, copy=False)
                block = raw[:, left:].astype(np.float64, copy=False)
                eta = prefix @ coefficients
                weights = weights_source[
                    row_left - source_offset : row_right - source_offset
                ]
                if event:
                    if parent.occurrence_likelihood == "poisson":
                        first = -weights
                        second = np.zeros_like(weights)
                    else:
                        _loss, first, second = cloglog_event_terms(eta)
                        first = weights * first
                        second = weights * second
                else:
                    first = weights * np.exp(eta)
                    second = first
                gradient[:] += block.T @ first
                information[:] += block.T @ (block * second[:, None])

        if parent.n_events:
            accumulate(0, int(parent.n_events), event=True)
        if parent.n_events < len(parent.design):
            accumulate(int(parent.n_events), len(parent.design), event=False)
        return gradient, 0.5 * (information + information.T)

    def _scores_from_moments(
        self,
        gradient: np.ndarray,
        information: np.ndarray,
        *,
        dictionary_mode: bool,
        mark_gradient: np.ndarray | None = None,
        mark_information: np.ndarray | None = None,
    ) -> dict[int, tuple[float, int | None]]:
        if not dictionary_mode:
            return {
                sign: (
                    self._cone_quadratic_gain(float(sign) * gradient, information),
                    None,
                )
                for sign in (-1, 1)
            }
        shapes = self.kernel_dictionary
        projected_gradient = shapes @ gradient
        projected_information = np.einsum(
            "ki,ij,kj->k",
            shapes,
            information,
            shapes,
            optimize=True,
        )
        if (mark_gradient is None) != (mark_information is None):
            raise ValueError("mark gradient and information must be supplied together")
        if mark_gradient is None:
            mark_gains = np.zeros(len(shapes), dtype=np.float64)
        else:
            projected_mark_gradient = shapes @ np.asarray(mark_gradient, dtype=np.float64)
            projected_mark_information = np.einsum(
                "ki,ij,kj->k",
                shapes,
                np.asarray(mark_information, dtype=np.float64),
                shapes,
                optimize=True,
            )
            # The mark coefficient is unrestricted: either a larger or a
            # smaller conditional amount can make the same occurrence rule
            # financially informative.
            mark_gains = np.zeros_like(projected_mark_information)
            mark_identified = projected_mark_information > 0.0
            np.divide(
                0.5 * projected_mark_gradient**2,
                projected_mark_information,
                out=mark_gains,
                where=mark_identified,
            )
        output: dict[int, tuple[float, int | None]] = {}
        for sign in (-1, 1):
            signed_gradient = float(sign) * projected_gradient
            occurrence_gains = np.zeros_like(projected_information)
            occurrence_identified = (signed_gradient < 0.0) & (
                projected_information > 0.0
            )
            np.divide(
                0.5 * signed_gradient**2,
                projected_information,
                out=occurrence_gains,
                where=occurrence_identified,
            )
            # For a fixed signed shape the occurrence likelihood is convex in
            # its nonnegative amplitude.  A nonnegative derivative at zero
            # proves that its exact optimum is the zero boundary, irrespective
            # of the mark coefficient.  Removing such mark-only atoms here is
            # therefore exact, not a heuristic screen.
            gains = np.where(
                occurrence_gains > 0.0,
                occurrence_gains + mark_gains,
                0.0,
            )
            best_index = int(np.argmax(gains))
            output[sign] = (float(gains[best_index]), best_index)
        return output

    def _identity_score(
        self,
        null_fit: FitResult,
        closure_terms: Sequence[ClosureTerm],
        antecedent: tuple[int, ...],
        window: int,
        sign: int,
    ) -> float:
        """Conditional M-knot cone score at a fitted closure/support null."""
        gradient, information = self._identity_moments(
            null_fit, closure_terms, antecedent, window
        )
        return self._cone_quadratic_gain(float(sign) * gradient, information)

    def _scores_for_both_signs(
        self,
        null_fit: FitResult,
        closure_terms: Sequence[ClosureTerm],
        antecedent: tuple[int, ...],
        window: int,
        *,
        dictionary_mode: bool,
    ) -> dict[int, tuple[float, int | None]]:
        gradient, information = self._identity_moments(
            null_fit, closure_terms, antecedent, window
        )
        return self._scores_from_moments(
            gradient,
            information,
            dictionary_mode=dictionary_mode,
        )

    def _identity_dictionary_score(
        self,
        null_fit: FitResult,
        closure_terms: Sequence[ClosureTerm],
        antecedent: tuple[int, ...],
        window: int,
        sign: int,
    ) -> tuple[float, int]:
        score, index = self._scores_for_both_signs(
            null_fit,
            closure_terms,
            antecedent,
            window,
            dictionary_mode=True,
        )[int(sign)]
        assert index is not None
        return score, index

    def _start_active_support_workers(self) -> None:
        devices = self._support_worker_devices()
        if len(devices) <= 1 or self._active_support_workers:
            return
        workers: list[CertSCRPipeline] = []
        share_host_fit_caches = bool(
            not self.config.support_devices
            and all(str(device).startswith("cpu") for device in devices)
        )
        for device in devices:
            worker = copy.copy(self)
            worker.config = CertSCRConfig(
                **{
                    **asdict(self.config),
                    "solver_device": device,
                    "support_devices": (),
                }
            )
            if share_host_fit_caches:
                # Equal CPU workers solve the same float64 convex problems.
                # Sharing immutable completed fits avoids repeatedly copying a
                # cache that grows with every support wave. Per-key locks below
                # guarantee one owner for every model key.
                worker._fit_cache = self._fit_cache
                worker._fit_key_locks = self._fit_key_locks
                worker._fit_key_locks_guard = self._fit_key_locks_guard
                worker._null_fit_cache = self._null_fit_cache
                worker._safe_bound_cache = self._safe_bound_cache
                worker._safe_screened_records = self._safe_screened_records
            else:
                worker._fit_cache = dict(self._fit_cache)
                worker._fit_key_locks = {}
                worker._fit_key_locks_guard = threading.Lock()
                worker._null_fit_cache = dict(self._null_fit_cache)
                worker._safe_bound_cache = dict(self._safe_bound_cache)
                worker._safe_screened_records = dict(self._safe_screened_records)
            worker._hierarchy_closure_cache = dict(self._hierarchy_closure_cache)
            worker._prepared_design_cache = {}
            worker._prepared_profile_nonoptimal_keys = set()
            worker._nuisance_event_design_cache = dict(self._nuisance_event_design_cache)
            worker._mark_base_residualizer_cache = dict(self._mark_base_residualizer_cache)
            worker._marked_response_cache = dict(self._marked_response_cache)
            worker.profiled_rules = list(self.profiled_rules)
            worker.identity_candidates = dict(self.identity_candidates)
            worker.rule_dictionary_shapes = {
                rule: shape.copy() for rule, shape in self.rule_dictionary_shapes.items()
            }
            worker.profile_logs = list(self.profile_logs)
            worker.support_records = []
            worker.candidate_records = []
            worker._active_support_workers = []
            worker._identity_refinement_keys = set()
            workers.append(worker)
        self._active_support_workers = workers

    @staticmethod
    def _exact_fit_worker_limit(available_workers: int) -> int:
        """Bound simultaneous large designs without changing fitted models.

        Worker objects remain available for response pricing. Only the number
        of concurrently resident exact support/refinement designs is limited;
        every requested model is still solved with the same objective and KKT
        tolerance. The environment value is execution-only and deliberately
        excluded from the statistical checkpoint signature.
        """
        available = max(1, int(available_workers))
        raw = os.environ.get("CERTSCR_MAX_CONCURRENT_EXACT_FITS")
        if raw is None:
            return available
        try:
            requested = int(raw)
        except ValueError as error:
            raise ValueError(
                "CERTSCR_MAX_CONCURRENT_EXACT_FITS must be an integer"
            ) from error
        if requested < 1:
            raise ValueError(
                "CERTSCR_MAX_CONCURRENT_EXACT_FITS must be positive"
            )
        return min(available, requested)

    def _stage_nested_prepared_designs(
        self,
        rule_sets: Sequence[Sequence[RuleIdentity]],
    ) -> None:
        """Cache exact child designs projected from maximal same-closure models."""
        keys = list(
            dict.fromkeys(tuple(sorted(rules)) for rules in rule_sets if rules)
        )
        pending = [
            key
            for key in keys
            if (key, self.hierarchy_closure(key)) not in self._fit_cache
            and (key, self.hierarchy_closure(key))
            not in self._prepared_design_cache
        ]
        if len(pending) < 2:
            return
        by_closure: dict[
            tuple[ClosureTerm, ...], list[tuple[RuleIdentity, ...]]
        ] = {}
        for key in pending:
            by_closure.setdefault(self.hierarchy_closure(key), []).append(key)
        for closure_terms, group in by_closure.items():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda item: (-len(item), item))
            maximal: list[tuple[RuleIdentity, ...]] = []
            for key in ordered:
                key_set = set(key)
                if not any(key_set.issubset(set(parent)) for parent in maximal):
                    maximal.append(key)
            children_by_parent: dict[
                tuple[RuleIdentity, ...], list[tuple[RuleIdentity, ...]]
            ] = {parent: [] for parent in maximal}
            for key in group:
                parents = [
                    parent
                    for parent in maximal
                    if set(key).issubset(set(parent))
                ]
                parent = min(parents, key=lambda item: (len(item), item))
                children_by_parent[parent].append(key)
            for parent, children in children_by_parent.items():
                # An incomparable group offers no reuse; leave its ordinary
                # lazy preparation path untouched.
                if len(children) < 2:
                    continue
                parent_prepared = self.prepare_partitioned_support_design(
                    self.splits.fit,
                    closure_terms,
                    self.sparse_features(self.splits.fit, parent),
                    parent,
                    cluster_weights=self.fit_cluster_weights,
                )
                self._prepared_design_cache[(parent, closure_terms)] = (
                    parent_prepared
                )
                reused = 0
                for child in children:
                    if child == parent:
                        continue
                    self._prepared_design_cache[(child, closure_terms)] = (
                        project_prepared_support_design(
                            parent_prepared,
                            child,
                            source_closure_terms=closure_terms,
                            target_closure_terms=closure_terms,
                            regroup=True,
                        )
                    )
                    reused += 1
                with self._diagnostic_guard:
                    self._safe_screen_stats[
                        "nested_prepared_parent_designs"
                    ] += 1
                    self._safe_screen_stats[
                        "nested_prepared_child_reuses"
                    ] += reused

    def _semantic_support_warm_start(
        self,
        prepared: DeltaFactorizedSupportDesign,
        closure_terms: tuple[ClosureTerm, ...],
        closure_baseline: FitResult,
    ) -> FitResult:
        """Map the closest cached model into the target's exact column semantics.

        This is initialization only.  Missing or non-cone-compatible columns
        remain zero, and the factorized solver independently rejects any warm
        point whose objective is worse than its intercept initialization.
        """
        rules = prepared.rules
        knot_count = int(self.config.knot_count)
        target_rule_width = int(prepared.knot_count)
        target_fixed_width = int(prepared.control_width) - knot_count * len(
            closure_terms
        )
        if target_fixed_width < 0:
            return closure_baseline
        with self._fit_key_locks_guard:
            cached = tuple(self._fit_cache.values())

        def raw_rule_coefficients(
            fit: FitResult,
            rule: RuleIdentity,
        ) -> np.ndarray | None:
            try:
                index = fit.rules.index(rule)
            except ValueError:
                return None
            row = np.asarray(fit.theta[index], dtype=np.float64)
            if len(row) == knot_count:
                return float(rule.sign) * row
            shape = self.rule_dictionary_shapes.get(rule)
            if len(row) == 1 and shape is not None and len(shape) == knot_count:
                return float(rule.sign) * float(row[0]) * np.asarray(
                    shape, dtype=np.float64
                )
            return None

        best = closure_baseline
        best_score = target_fixed_width + knot_count * len(closure_terms)
        for source in cached:
            if not source.converged:
                continue
            source_fixed_width = len(source.gamma) - knot_count * len(
                source.closure_terms
            )
            if source_fixed_width != target_fixed_width:
                continue
            gamma = np.zeros(prepared.control_width, dtype=np.float64)
            if target_fixed_width:
                gamma[:target_fixed_width] = source.gamma[:target_fixed_width]
            theta = np.zeros(
                (len(rules), target_rule_width), dtype=np.float64
            )
            score = target_fixed_width
            source_closure_index = {
                term: index
                for index, term in enumerate(source.closure_terms)
            }
            for target_index, term in enumerate(closure_terms):
                left = target_fixed_width + target_index * knot_count
                source_index = source_closure_index.get(term)
                if source_index is not None:
                    source_left = (
                        source_fixed_width + source_index * knot_count
                    )
                    gamma[left : left + knot_count] = source.gamma[
                        source_left : source_left + knot_count
                    ]
                    score += knot_count
                    continue
                source_rule = next(
                    (
                        rule
                        for rule in source.rules
                        if (rule.antecedent, int(rule.window)) == term
                    ),
                    None,
                )
                if source_rule is not None:
                    raw = raw_rule_coefficients(source, source_rule)
                    if raw is not None:
                        gamma[left : left + knot_count] = raw
                        score += knot_count
            for target_index, rule in enumerate(rules):
                if rule in source.rules:
                    source_index = source.rules.index(rule)
                    source_theta = np.asarray(
                        source.theta[source_index], dtype=np.float64
                    )
                    if len(source_theta) == target_rule_width:
                        theta[target_index] = np.maximum(source_theta, 0.0)
                        score += 2 * target_rule_width
                        continue
                term = (rule.antecedent, int(rule.window))
                source_index = source_closure_index.get(term)
                if source_index is None:
                    continue
                source_left = source_fixed_width + source_index * knot_count
                raw = float(rule.sign) * source.gamma[
                    source_left : source_left + knot_count
                ]
                if target_rule_width == knot_count:
                    theta[target_index] = np.maximum(raw, 0.0)
                    score += target_rule_width
                else:
                    shape = self.rule_dictionary_shapes.get(rule)
                    if shape is not None and len(shape) == knot_count:
                        shape64 = np.asarray(shape, dtype=np.float64)
                        denominator = float(shape64 @ shape64)
                        if denominator > 0.0:
                            theta[target_index, 0] = max(
                                0.0, float(shape64 @ raw) / denominator
                            )
                            score += 1
            if score <= best_score:
                continue
            best_score = score
            best = replace(
                source,
                rules=rules,
                closure_terms=closure_terms,
                gamma=gamma,
                theta=theta,
                mark_fit=None,
                solver_hessian=None,
            )
        if best is not closure_baseline:
            with self._diagnostic_guard:
                self._safe_screen_stats[
                    "support_semantic_warm_starts"
                ] += 1
        return best

    def _fit_one_closure_group_sparse(
        self,
        ordered: Sequence[tuple[RuleIdentity, ...]],
        closure_terms: tuple[ClosureTerm, ...],
        *,
        profile: str,
    ) -> list[SupportRecord] | None:
        """Execute one closure group without materializing dense closure rows.

        This is an exact CPU-float64 backend.  Any construction or convergence
        failure returns ``None`` and the caller runs the canonical grouped
        implementation, so the acceleration cannot remove a candidate or
        weaken its KKT requirement.
        """
        if (
            self.marked
            or not str(self.config.solver_device).startswith("cpu")
            or self.config.solver_dtype != "float64"
        ):
            return None
        try:
            fixed = self.fixed_nuisance_partition(self.splits.fit)
            closure_blocks = self.sparse_closure_blocks(
                self.splits.fit, closure_terms
            )
            null_key = ((), closure_terms)
            closure_baseline = self._fit_cache.get(null_key)
            if closure_baseline is None and not closure_terms:
                closure_baseline = self.fit_model((), ())
            if closure_baseline is not None and not closure_baseline.converged:
                return None

            output: list[SupportRecord] = []
            nested_screened: dict[tuple[RuleIdentity, ...], FitResult] = {}
            if closure_baseline is not None:
                by_parent: dict[
                    tuple[RuleIdentity, ...],
                    list[tuple[tuple[RuleIdentity, ...], tuple[RuleIdentity, ...]]],
                ] = {}
                for key in ordered:
                    if (key, closure_terms) in self._fit_cache:
                        continue
                    parents = [
                        subset
                        for subset in (
                            key[:index] + key[index + 1 :]
                            for index in range(len(key))
                        )
                        if subset
                        and (
                            candidate := self._fit_cache.get(
                                (subset, closure_terms)
                            )
                        ) is not None
                        and candidate.converged
                    ]
                    if not parents:
                        continue
                    parent = max(parents, key=lambda item: (len(item), item))
                    missing = tuple(rule for rule in key if rule not in parent)
                    by_parent.setdefault(parent, []).append((key, missing))
                for parent_key, children in by_parent.items():
                    parent_fit = self._fit_cache[(parent_key, closure_terms)]
                    missing_rules = tuple(
                        sorted(
                            {
                                rule
                                for _key, missing in children
                                for rule in missing
                            }
                        )
                    )
                    prices = self._support_rule_gradient_prices(
                        parent_fit, missing_rules
                    )
                    for key, missing in children:
                        joint_kkt = max(
                            float(parent_fit.kkt_residual),
                            max(
                                (float(prices[rule][1]) for rule in missing),
                                default=0.0,
                            ),
                        )
                        if joint_kkt > float(self.config.solver_tolerance):
                            continue
                        widths = {
                            int(block.shape[1])
                            for block in self.sparse_features(
                                self.splits.fit, key
                            )
                        }
                        if len(widths) != 1:
                            raise ValueError(
                                "nested KKT screen found inconsistent rule widths"
                            )
                        width = widths.pop()
                        theta = np.zeros(
                            (len(key), width), dtype=np.float64
                        )
                        parent_index = {
                            rule: index
                            for index, rule in enumerate(parent_fit.rules)
                        }
                        compatible = parent_fit.theta.shape[1] == width
                        if compatible:
                            for index, rule in enumerate(key):
                                source = parent_index.get(rule)
                                if source is not None:
                                    theta[index] = parent_fit.theta[source]
                        if not compatible:
                            continue
                        fit = replace(
                            parent_fit,
                            rules=key,
                            theta=theta,
                            kkt_residual=float(joint_kkt),
                            converged=True,
                            iterations=0,
                            mark_fit=None,
                            solver_hessian=None,
                        )
                        nested_screened[key] = fit
                        with self._fit_key_locks_guard:
                            self._fit_cache[(key, closure_terms)] = fit
                        with self._diagnostic_guard:
                            self._safe_screen_stats[
                                "support_nested_exact_kkt_screens"
                            ] += 1
            for key in ordered:
                model_key = (key, closure_terms)
                fit = nested_screened.get(key) or self._fit_cache.get(model_key)
                if fit is None:
                    sparse_support = prepare_sparse_delta_support_design(
                        self.splits.fit,
                        fixed,
                        closure_blocks,
                        closure_terms,
                        self.sparse_features(self.splits.fit, key),
                        key,
                        cluster_weights=self.fit_cluster_weights,
                        occurrence_likelihood=self.occurrence_likelihood,
                    )
                    if closure_baseline is None:
                        fixed_baseline = self.fit_model((), ())
                        closure_baseline = fit_sparse_delta_closure(
                            sparse_support,
                            max_iter=self.config.solver_max_iter,
                            tolerance=self.config.solver_tolerance,
                            baseline=fixed_baseline,
                        )
                        if closure_baseline is None:
                            return None
                        with self._fit_key_locks_guard:
                            self._fit_cache[null_key] = closure_baseline
                            self._null_fit_cache[
                                closure_terms
                            ] = closure_baseline
                    recession_columns = sparse_delta_rule_recession_columns(
                        sparse_support
                    )
                    if recession_columns:
                        fit = self._nonattained_support_fit(
                            key,
                            closure_baseline,
                            int(sparse_support.knot_count),
                            recession_columns,
                        )
                    else:
                        initial = self._semantic_support_warm_start(
                            sparse_support,
                            closure_terms,
                            closure_baseline,
                        )
                        fit = fit_sparse_delta_support(
                            sparse_support,
                            max_iter=self.config.solver_max_iter,
                            tolerance=self.config.solver_tolerance,
                            initial=initial,
                            baseline=closure_baseline,
                        )
                        if fit is None:
                            return None
                    with self._fit_key_locks_guard:
                        self._fit_cache[model_key] = fit
                    with self._diagnostic_guard:
                        self._safe_screen_stats[
                            "support_sparse_delta_fits"
                        ] += 1
                output.append(
                    SupportRecord(
                        rules=key,
                        fit=fit,
                        closure_baseline_fit=closure_baseline,
                        search_nll_improvement=float(
                            closure_baseline.nll - fit.nll
                        ),
                        profile=profile,
                    )
                )
            if closure_baseline is None:
                return None
            with self._diagnostic_guard:
                self._safe_screen_stats[
                    "support_sparse_delta_closure_groups"
                ] += 1
            return output
        except (FloatingPointError, MemoryError, RuntimeError, ValueError):
            with self._diagnostic_guard:
                self._safe_screen_stats[
                    "support_sparse_delta_fallbacks"
                ] += 1
            return None

    def _fit_one_closure_group(
        self,
        keys: Sequence[tuple[RuleIdentity, ...]],
        *,
        profile: str,
    ) -> list[SupportRecord]:
        """Fit one closure-local support group from one exact partition.

        The old path built ``fixed + closure`` once for the null and again for
        every support.  Here the closure partition is retained, its null is
        solved once, and each rule set is appended by sparse partition
        refinement.  A native failure falls back to the canonical cold design;
        no candidate, objective, row mass or KKT tolerance changes.
        """
        ordered = list(dict.fromkeys(tuple(sorted(key)) for key in keys))
        if not ordered:
            return []
        closure_terms = self.hierarchy_closure(ordered[0])
        if any(self.hierarchy_closure(key) != closure_terms for key in ordered):
            raise ValueError("closure-local fit group contains different closures")
        if all(
            (key, closure_terms) in self._fit_cache
            or (key, closure_terms) in self._prepared_design_cache
            for key in ordered
        ):
            return [self._support_record(key, profile=profile) for key in ordered]
        sparse_output = self._fit_one_closure_group_sparse(
            ordered, closure_terms, profile=profile
        )
        if sparse_output is not None:
            return sparse_output
        partition = self.prepare_partitioned_support_design(
            self.splits.fit,
            closure_terms,
            [],
            [],
            cluster_weights=self.fit_cluster_weights,
            return_partition=True,
        )
        if not isinstance(partition, IncrementalSupportPartition):
            return [self._support_record(key, profile=profile) for key in ordered]
        with self._diagnostic_guard:
            self._safe_screen_stats["support_closure_partitions"] += 1

        null_key = ((), closure_terms)
        if null_key not in self._fit_cache:
            self._prepared_design_cache[null_key] = partition.prepared
        closure_baseline = self.fit_model((), closure_terms)
        output: list[SupportRecord] = []
        remaining = list(ordered)

        # KKT conditions at a converged convex null are jointly sufficient for
        # a zero rule block.  Price a shared closure group in one sparse-union
        # pass; this exact screen is worthwhile only when at least two support
        # candidates reuse that pass.  Marked models are excluded because an
        # occurrence-zero block can still own a nonzero conditional-mark head.
        if (
            not self.marked
            and closure_baseline.converged
            and len(remaining) > 1
        ):
            group_rules = tuple(
                sorted({rule for key in remaining for rule in key})
            )
            prices = self._support_rule_gradient_prices(
                closure_baseline,
                group_rules,
            )
            unscreened: list[tuple[RuleIdentity, ...]] = []
            for key in remaining:
                rule_kkt = max(
                    (float(prices[rule][1]) for rule in key), default=0.0
                )
                joint_kkt = max(
                    float(closure_baseline.kkt_residual), rule_kkt
                )
                if joint_kkt > float(self.config.solver_tolerance):
                    unscreened.append(key)
                    continue
                widths = {
                    int(block.shape[1])
                    for block in self.sparse_features(self.splits.fit, key)
                }
                if len(widths) != 1:
                    raise ValueError("joint-null screen found inconsistent rule widths")
                width = widths.pop()
                zero_fit = replace(
                    closure_baseline,
                    rules=key,
                    theta=np.zeros((len(key), width), dtype=np.float64),
                    kkt_residual=joint_kkt,
                    converged=True,
                    iterations=0,
                    mark_fit=None,
                    solver_hessian=None,
                )
                with self._fit_key_locks_guard:
                    self._fit_cache[(key, closure_terms)] = zero_fit
                output.append(
                    SupportRecord(
                        rules=key,
                        fit=zero_fit,
                        closure_baseline_fit=closure_baseline,
                        search_nll_improvement=0.0,
                        profile=f"{profile}-exact-joint-null-kkt-screen",
                    )
                )
                with self._diagnostic_guard:
                    self._safe_screen_stats[
                        "support_joint_null_kkt_screens"
                    ] += 1
            remaining = unscreened

        for key in remaining:
            model_key = (key, closure_terms)
            cached_fit = self._fit_cache.get(model_key)
            if (
                cached_fit is None
                and model_key not in self._prepared_design_cache
            ):
                features = self.sparse_features(self.splits.fit, key)
                try:
                    factorized = prepare_delta_factorized_support_design(
                        self.splits.fit,
                        partition,
                        features,
                        key,
                        cluster_weights=self.fit_cluster_weights,
                        occurrence_likelihood=self.occurrence_likelihood,
                    )
                except RuntimeError:
                    factorized = None
                    with self._diagnostic_guard:
                        self._safe_screen_stats[
                            "support_closure_partition_fallbacks"
                        ] += 1
                if factorized is not None:
                    recession_columns = factorized_rule_recession_columns(
                        factorized
                    )
                    if recession_columns:
                        fit = self._nonattained_support_fit(
                            key,
                            closure_baseline,
                            int(factorized.knot_count),
                            recession_columns,
                        )
                    else:
                        initial = self._semantic_support_warm_start(
                            factorized,
                            closure_terms,
                            closure_baseline,
                        )
                        fit = fit_delta_factorized_support(
                            factorized,
                            closure_terms,
                            max_iter=self.config.solver_max_iter,
                            tolerance=self.config.solver_tolerance,
                            initial=initial,
                            baseline=closure_baseline,
                        )
                        fit = self._attach_mark_head(fit, self.splits.fit)
                    with self._fit_key_locks_guard:
                        self._fit_cache[model_key] = fit
                    with self._diagnostic_guard:
                        self._safe_screen_stats[
                            "support_closure_partition_child_reuses"
                        ] += 1
                        self._safe_screen_stats[
                            "support_delta_factorized_fits"
                        ] += 1
            output.append(self._support_record(key, profile=profile))
        by_key = {record.rules: record for record in output}
        return [by_key[key] for key in ordered]

    def _fit_support_records_batch(
        self,
        rule_sets: Sequence[Sequence[RuleIdentity]],
        *,
        profile: str,
    ) -> list[SupportRecord]:
        keys = list(dict.fromkeys(tuple(sorted(rules)) for rules in rule_sets))
        if not keys:
            return []
        workers = self._active_support_workers
        if len(workers) <= 1:
            by_closure: dict[
                tuple[ClosureTerm, ...], list[tuple[RuleIdentity, ...]]
            ] = {}
            for key in keys:
                by_closure.setdefault(self.hierarchy_closure(key), []).append(
                    key
                )
            fitted: list[SupportRecord] = []
            for group in by_closure.values():
                fitted.extend(
                    self._fit_one_closure_group(group, profile=profile)
                )
            by_key = {record.rules: record for record in fitted}
            _release_unused_native_heap()
            self._maybe_checkpoint_support_fits()
            return [by_key[key] for key in keys]

        # Safe-bound evaluation can fit new hierarchy-null models on the
        # parent between batches.  Publish them before dispatch so workers do
        # not repeat the same closure optimization on separate GPUs.
        for worker in workers:
            if worker._fit_cache is not self._fit_cache:
                worker._fit_cache.update(self._fit_cache)
                worker._null_fit_cache.update(self._null_fit_cache)
            worker._prepared_design_cache.clear()
            worker._nuisance_event_design_cache.update(self._nuisance_event_design_cache)
            worker._mark_base_residualizer_cache.update(self._mark_base_residualizer_cache)

        # Keep every support with the same hierarchy closure on one worker.
        # Otherwise private worker/process caches can solve the identical null
        # once per chunk.  Within a closure, smaller supports go first so their
        # converged coefficients are available as exact warm starts for nested
        # supports.  This changes only scheduling/initialization: every key is
        # still solved over the same convex feasible set and checked against the
        # same host-float64 KKT tolerance.
        by_closure: dict[
            tuple[ClosureTerm, ...], list[tuple[RuleIdentity, ...]]
        ] = {}
        for key in keys:
            by_closure.setdefault(self.hierarchy_closure(key), []).append(key)
        closure_groups = [
            sorted(group, key=lambda item: (len(item), item))
            for group in by_closure.values()
        ]
        closure_groups.sort(
            key=lambda group: (
                -sum(1 + len(key) for key in group),
                self.hierarchy_closure(group[0]),
                group[0],
            )
        )
        # Compact closure/delta arrays and completed fits are immutable and can
        # be shared by long-lived thread workers.  Making rolling fork the
        # default copied Python/cache pages in every microjob and exhausted
        # memory on the full Freddie run.  NumPy matrix products execute in
        # compiled MKL outside the GIL, while the local limiter prevents nested
        # BLAS pools.  Retain the old exact process route only as an explicit
        # systems fallback; it changes neither candidates nor optimization.
        use_fork_batch = bool(
            os.environ.get("CERTSCR_PROCESS_FITS", "0") == "1"
            and os.name == "posix"
            and not self.config.support_devices
            and not self.config.safe_mdl_screen
            and str(self.config.solver_device).startswith("cpu")
            and int(self.config.solver_workers) > 1
            and len(keys) >= 16 * len(workers)
        )
        chunks: list[list[tuple[RuleIdentity, ...]]] = [
            [] for _ in range(len(workers))
        ]
        chunk_loads = [0 for _ in workers]
        for group in closure_groups:
            worker_index = min(
                range(len(workers)),
                key=lambda index: (chunk_loads[index], index),
            )
            chunks[worker_index].extend(group)
            chunk_loads[worker_index] += sum(1 + len(key) for key in group)
        def fit_chunk(
            worker: CertSCRPipeline,
            chunk: list[tuple[RuleIdentity, ...]],
        ) -> list[SupportRecord]:
            with _single_threaded_local_blas():
                output: list[SupportRecord] = []
                # Build each distinct hierarchy-closure partition once and
                # append the rule blocks of its children through exact native
                # repartitioning.  No child candidate or sufficient-statistic
                # mass is changed.
                closure_chunks: dict[
                    tuple[ClosureTerm, ...],
                    list[tuple[RuleIdentity, ...]],
                ] = {}
                for key in chunk:
                    closure_chunks.setdefault(
                        worker.hierarchy_closure(key), []
                    ).append(key)
                for closure_keys in closure_chunks.values():
                    output.extend(
                        worker._fit_one_closure_group(
                            closure_keys,
                            profile=profile,
                        )
                    )
                return output

        records: list[SupportRecord] = []
        with self._diagnostic_guard:
            self._safe_screen_stats[
                "active_fit_process_batches"
                if use_fork_batch
                else "active_fit_thread_batches"
            ] += 1
            self._safe_screen_stats["active_fit_closure_groups"] += len(
                closure_groups
            )
        if use_fork_batch:
            context = mp.get_context("fork")
            result_queue = context.Queue()

            # Static one-chunk-per-worker scheduling treated every pair as
            # equally expensive and left most cores idle behind two closure
            # stragglers.  Preserve each closure group, but pack it into small
            # bounded jobs consumed by a rolling process pool.  Job order and
            # completion order affect only warm starts; every convex key still
            # receives the same fit and host-float64 KKT certificate.
            target_job_supports = 16
            dynamic_chunks: list[list[tuple[RuleIdentity, ...]]] = []
            current_chunk: list[tuple[RuleIdentity, ...]] = []
            for group in closure_groups:
                if (
                    current_chunk
                    and len(current_chunk) + len(group) > target_job_supports
                ):
                    dynamic_chunks.append(current_chunk)
                    current_chunk = []
                current_chunk.extend(group)
                if len(current_chunk) >= target_job_supports:
                    dynamic_chunks.append(current_chunk)
                    current_chunk = []
            if current_chunk:
                dynamic_chunks.append(current_chunk)

            def process_fit_chunk(
                chunk_index: int,
                chunk: list[tuple[RuleIdentity, ...]],
            ) -> None:
                try:
                    setter = _mkl_local_thread_setter()
                    if setter is not None:
                        setter(1)
                    worker = copy.copy(self)
                    worker.config = replace(
                        self.config,
                        solver_workers=1,
                        support_devices=(),
                    )
                    worker._active_support_workers = []
                    chunk_set = set(chunk)
                    worker._prepared_design_cache = {
                        model_key: prepared
                        for model_key, prepared in self._prepared_design_cache.items()
                        if model_key[0] in chunk_set
                    }
                    worker._information_design_parent = None
                    diagnostic_before = dict(worker._safe_screen_stats)
                    values = fit_chunk(worker, chunk)
                    diagnostic_updates = {
                        key: int(value) - int(diagnostic_before.get(key, 0))
                        for key, value in worker._safe_screen_stats.items()
                        if int(value) != int(diagnostic_before.get(key, 0))
                    }
                    result_queue.put(
                        (True, chunk_index, values, diagnostic_updates, None)
                    )
                except BaseException:
                    result_queue.put(
                        (
                            False,
                            chunk_index,
                            None,
                            None,
                            traceback.format_exc(),
                        )
                    )

            process_jobs = [
                (index, chunk)
                for index, chunk in enumerate(dynamic_chunks)
                if chunk
            ]
            received: dict[int, list[SupportRecord]] = {}
            pending = iter(process_jobs)
            active: dict[int, mp.Process] = {}

            def launch_next() -> bool:
                try:
                    chunk_index, chunk = next(pending)
                except StopIteration:
                    return False
                process = context.Process(
                    target=process_fit_chunk,
                    args=(chunk_index, chunk),
                )
                process.start()
                active[chunk_index] = process
                with self._diagnostic_guard:
                    self._safe_screen_stats[
                        "active_fit_dynamic_worker_launches"
                    ] += 1
                return True

            for _ in range(min(len(workers), len(process_jobs))):
                launch_next()
            with self._diagnostic_guard:
                self._safe_screen_stats["active_fit_dynamic_jobs"] += len(
                    process_jobs
                )
            try:
                while len(received) < len(process_jobs):
                    try:
                        (
                            ok,
                            chunk_index,
                            values,
                            diagnostic_updates,
                            error,
                        ) = result_queue.get(timeout=5.0)
                    except queue.Empty:
                        failed = [
                            (index, process.exitcode)
                            for index, process in active.items()
                            if not process.is_alive()
                            and process.exitcode not in {0, None}
                        ]
                        if not failed and any(
                            process.is_alive() for process in active.values()
                        ):
                            continue
                        raise RuntimeError(
                            "rolling support-fit workers exited before returning a result: "
                            + ", ".join(
                                f"job {index} (exit {code})"
                                for index, code in failed
                            )
                        )
                    chunk_index = int(chunk_index)
                    process = active.pop(chunk_index, None)
                    if process is None:
                        raise RuntimeError(
                            "rolling support worker returned unknown job "
                            f"{chunk_index}"
                        )
                    process.join()
                    if not ok:
                        raise RuntimeError(
                            "forked support fitting failed:\n" + str(error)
                        )
                    if process.exitcode not in {0, None}:
                        raise RuntimeError(
                            f"rolling support worker {chunk_index} exited with code "
                            f"{process.exitcode}"
                        )
                    chunk_records = values or []
                    received[chunk_index] = chunk_records
                    # Publish completed immutable fits before launching the
                    # next fork.  The new child inherits them copy-on-write
                    # and can use exact nested-null warm starts without
                    # retaining the finished worker's large response/design
                    # arenas.
                    for record in chunk_records:
                        self._fit_cache[
                            (record.fit.rules, record.fit.closure_terms)
                        ] = record.fit
                        self._fit_cache[
                            ((), record.fit.closure_terms)
                        ] = record.closure_baseline_fit
                        self._null_fit_cache[
                            record.fit.closure_terms
                        ] = record.closure_baseline_fit
                        self._prepared_design_cache.pop(
                            (record.fit.rules, record.fit.closure_terms), None
                        )
                    if diagnostic_updates:
                        with self._diagnostic_guard:
                            for key, delta in diagnostic_updates.items():
                                self._safe_screen_stats[key] = int(
                                    self._safe_screen_stats.get(key, 0)
                                ) + int(delta)
                    launch_next()
            finally:
                for outstanding in active.values():
                    if outstanding.is_alive():
                        outstanding.terminate()
                for outstanding in active.values():
                    outstanding.join()
                result_queue.close()
            for chunk_index, _chunk in process_jobs:
                records.extend(received[chunk_index])
        else:
            # A closure group is an execution-reuse unit, not an optimization
            # dependency.  When fewer closure groups than physical workers
            # remain, keeping all children on one worker serializes expensive
            # high-order refits.  Host workers share immutable caches: fit each
            # missing closure null once, then shard its independent exact
            # children across the idle workers.  Every child retains the same
            # convex feasible set, objective and KKT tolerance; only its warm
            # start and dispatch worker may differ.
            fit_worker_count = self._exact_fit_worker_limit(len(workers))
            fit_workers = workers[:fit_worker_count]
            available: queue.SimpleQueue[CertSCRPipeline] = queue.SimpleQueue()
            for worker in fit_workers:
                available.put(worker)

            shared_host_caches = all(
                worker._fit_cache is self._fit_cache
                and worker._null_fit_cache is self._null_fit_cache
                for worker in workers
            )
            shard_closures = bool(
                shared_host_caches
                and len(closure_groups) < len(workers)
                and any(len(group) > 1 for group in closure_groups)
            )

            def prefit_closure_null(
                closure_terms: tuple[ClosureTerm, ...],
            ) -> None:
                worker = available.get()
                try:
                    with _single_threaded_local_blas():
                        worker.fit_model((), closure_terms)
                finally:
                    available.put(worker)

            thread_groups = closure_groups
            prefitted_closures = 0
            if shard_closures:
                missing_closures = [
                    self.hierarchy_closure(group[0])
                    for group in closure_groups
                    if ((), self.hierarchy_closure(group[0]))
                    not in self._fit_cache
                ]
                if missing_closures:
                    # Nulls are distinct convex cache keys and can be solved
                    # concurrently.  The per-key fit lock is still active, so
                    # a concurrent safe-bound request cannot duplicate one.
                    with ThreadPoolExecutor(
                        max_workers=min(
                            fit_worker_count, len(missing_closures)
                        )
                    ) as null_executor:
                        list(
                            null_executor.map(
                                prefit_closure_null, missing_closures
                            )
                        )
                    prefitted_closures = len(missing_closures)
                # A singleton task is deliberate here.  At most
                # ``fit_worker_count``
                # tasks execute concurrently, so live design memory has the
                # same bound as an ordinary multi-closure wave while one large
                # closure can no longer strand the remaining workers.
                thread_groups = [
                    [key]
                    for group in closure_groups
                    for key in group
                ]

            def fit_closure_task(
                closure_keys: list[tuple[RuleIdentity, ...]],
            ) -> list[SupportRecord]:
                worker = available.get()
                try:
                    for rules in closure_keys:
                        model_key = (
                            rules,
                            worker.hierarchy_closure(rules),
                        )
                        prepared = self._prepared_design_cache.pop(
                            model_key, None
                        )
                        if prepared is not None:
                            worker._prepared_design_cache[model_key] = prepared
                    return fit_chunk(worker, closure_keys)
                finally:
                    available.put(worker)

            with self._diagnostic_guard:
                self._safe_screen_stats[
                    "active_fit_dynamic_thread_tasks"
                ] += len(thread_groups)
                self._safe_screen_stats[
                    "active_fit_shared_closure_null_prefits"
                ] += prefitted_closures
                self._safe_screen_stats[
                    "active_fit_closure_shards"
                ] += max(0, len(thread_groups) - len(closure_groups))
            with ThreadPoolExecutor(max_workers=fit_worker_count) as executor:
                futures = [
                    executor.submit(fit_closure_task, group)
                    for group in thread_groups
                ]
                for future in futures:
                    records.extend(future.result())
        # Make every fitted state available to the parent and to either worker
        # in later batches. Numerical objectives are persisted in float64, so
        # device scheduling cannot alter search ordering beyond solver tolerance.
        for record in records:
            self._fit_cache[(record.fit.rules, record.fit.closure_terms)] = record.fit
            self._fit_cache[((), record.fit.closure_terms)] = record.closure_baseline_fit
            self._null_fit_cache[record.fit.closure_terms] = record.closure_baseline_fit
        for worker in workers:
            if worker._fit_cache is not self._fit_cache:
                worker._fit_cache.update(self._fit_cache)
                worker._null_fit_cache.update(self._null_fit_cache)
            # Every staged design in this batch has now either been consumed
            # or superseded by its immutable FitResult.  Keeping the grouped
            # matrix on a worker cannot accelerate a later request for the
            # same cache key and was the principal source of live peak-memory
            # growth in long all-atom searches.
            worker._prepared_design_cache.clear()
            worker._information_design_parent = None
        _release_unused_native_heap()
        self._maybe_checkpoint_support_fits()
        by_key = {record.rules: record for record in records}
        return [by_key[key] for key in keys]

    def control_blocks(self, ctx: QueryContext) -> tuple[np.ndarray, ...]:
        # RuleOccurrenceEngine already owns the bounded response cache.  A
        # second concatenated control cache duplicated the largest matrices and
        # was never needed by the blockwise solver/predictor.
        source_ids = (
            *((self.target_history_source_id,) if self.target_history_source_id is not None else ()),
            *self.control_source_ids,
            *self.loan_age_baseline_source_ids,
        )
        return tuple(self.engine.response(ctx, (source,), 0) for source in source_ids)

    def controls(self, ctx: QueryContext) -> np.ndarray:
        blocks = self.control_blocks(ctx)
        return (
            np.concatenate(blocks, axis=1)
            if blocks
            else np.zeros((ctx.n_queries, 0), dtype=np.float32)
        )

    def features(self, ctx: QueryContext, rules: Sequence[RuleIdentity]) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for rule in rules:
            shape = self.rule_dictionary_shapes.get(rule)
            output.append(
                self.engine.projected_response(
                    ctx, rule.antecedent, rule.window, shape
                )
                if shape is not None
                else self.engine.response(ctx, rule.antecedent, rule.window)
            )
        return output

    def sparse_features(
        self,
        ctx: QueryContext,
        rules: Sequence[RuleIdentity],
    ) -> list[SparseKernelResponse]:
        """Return support-fit responses without materializing zero grid rows."""
        output: list[SparseKernelResponse] = []
        for rule in rules:
            shape = self.rule_dictionary_shapes.get(rule)
            output.append(
                self.engine.sparse_projected_response(
                    ctx, rule.antecedent, rule.window, shape
                )
                if shape is not None
                else self.engine.sparse_response(ctx, rule.antecedent, rule.window)
            )
        return output

    def _stage_profile_window_designs(
        self,
        rules: Sequence[RuleIdentity],
        response: SparseKernelResponse,
        *,
        full_m_parent: PreparedFixedSupportDesign | None = None,
    ) -> None:
        """Stage exact one-rule designs directly from one fused W response.

        ``iter_window_sparse_responses`` has already swept every completion
        exactly once.  Re-entering ``sparse_response`` during the subsequent
        exact identity fit used to repeat that convolution for each W/sign.
        Build the signed candidates' common parent while the cumulative W
        response is live, then project/regroup each one-rule child.  A parent
        partition is only a finer exact sufficient statistic, so the child
        objective, derivatives and KKT problem are unchanged.
        """
        candidates = tuple(sorted(set(rules)))
        if not candidates:
            return
        closure_terms = self.hierarchy_closure((candidates[0],))
        if any(
            self.hierarchy_closure((candidate,)) != closure_terms
            or candidate.antecedent != candidates[0].antecedent
            or int(candidate.window) != int(candidates[0].window)
            for candidate in candidates
        ):
            raise ValueError("fused profile candidates must share one W and closure")
        if full_m_parent is not None:
            source_rules = tuple(full_m_parent.rules)
            if (
                len(source_rules) != 1
                or source_rules[0].antecedent != candidates[0].antecedent
                or int(source_rules[0].window) != int(candidates[0].window)
                or int(full_m_parent.knot_count) != self.config.knot_count
            ):
                raise ValueError("full-M profile parent does not match candidate W")
            source_rule = source_rules[0]
            raw_block = full_m_parent.design[
                :,
                full_m_parent.constrained_start :
                full_m_parent.constrained_start + full_m_parent.knot_count,
            ]
            by_shape: dict[bytes, list[RuleIdentity]] = {}
            shape_values: dict[bytes, np.ndarray] = {}
            for candidate in candidates:
                shape = self.rule_dictionary_shapes.get(candidate)
                if shape is None:
                    raise ValueError("dictionary child is missing its frozen shape")
                canonical_shape = np.ascontiguousarray(shape, dtype=np.float32)
                shape_key = canonical_shape.tobytes()
                by_shape.setdefault(shape_key, []).append(candidate)
                shape_values[shape_key] = canonical_shape
            for shape_key, shape_candidates in by_shape.items():
                # Every parent group has one bit-identical full-M rule vector.
                # Projecting that vector is therefore exactly the same rowwise
                # float32 GEMV used by SparseKernelResponse.projected, including
                # for background groups carried across incremental W updates.
                activation = (raw_block @ shape_values[shape_key]).astype(
                    np.float32, copy=False
                )
                activation *= np.float32(source_rule.sign)
                selected = np.ascontiguousarray(
                    np.column_stack(
                        (
                            full_m_parent.design[
                                :, : full_m_parent.constrained_start
                            ],
                            activation,
                        )
                    ),
                    dtype=full_m_parent.design.dtype,
                )
                (
                    selected,
                    selected_events,
                    selected_event_weights,
                    selected_grid_weights,
                ) = aggregate_duplicate_design_rows(
                    selected,
                    full_m_parent.n_events,
                    full_m_parent.event_weights,
                    full_m_parent.grid_weights,
                    inplace=True,
                )
                for candidate in shape_candidates:
                    if candidate.sign > 0:
                        candidate_design = selected
                    elif len(shape_candidates) == 1:
                        candidate_design = selected
                        candidate_design[:, -1] *= np.float32(-1.0)
                    else:
                        candidate_design = selected.copy()
                        candidate_design[:, -1] *= np.float32(-1.0)
                    child = (candidate,)
                    self._prepared_design_cache[(child, closure_terms)] = (
                        PreparedFixedSupportDesign(
                            design=candidate_design,
                            n_events=selected_events,
                            event_weights=selected_event_weights,
                            grid_weights=selected_grid_weights,
                            constrained_start=int(full_m_parent.constrained_start),
                            control_width=int(full_m_parent.control_width),
                            knot_count=1,
                            active_grid_rows=int(full_m_parent.active_grid_rows),
                            rules=child,
                            occurrence_likelihood=(
                                full_m_parent.occurrence_likelihood
                            ),
                        )
                    )
            with self._diagnostic_guard:
                self._safe_screen_stats["identity_sign_pair_parent_designs"] += 1
                self._safe_screen_stats[
                    "identity_sign_pair_child_reuses"
                ] += len(candidates)
            self._prepared_profile_nonoptimal_keys.update(
                ((candidate,), closure_terms) for candidate in candidates
            )
            return
        features: list[SparseKernelResponse] = []
        for candidate in candidates:
            shape = self.rule_dictionary_shapes.get(candidate)
            features.append(
                response.projected(shape) if shape is not None else response
            )
        parent = self.prepare_partitioned_support_design(
            self.splits.fit,
            closure_terms,
            features,
            candidates,
            cluster_weights=self.fit_cluster_weights,
        )
        if len(candidates) == 1:
            self._prepared_design_cache[(candidates, closure_terms)] = parent
            self._prepared_profile_nonoptimal_keys.add(
                (candidates, closure_terms)
            )
            return
        for candidate in candidates:
            child = (candidate,)
            self._prepared_design_cache[(child, closure_terms)] = (
                project_prepared_support_design(
                    parent,
                    child,
                    source_closure_terms=closure_terms,
                    target_closure_terms=closure_terms,
                    regroup=True,
                )
            )
        with self._diagnostic_guard:
            self._safe_screen_stats["identity_sign_pair_parent_designs"] += 1
            self._safe_screen_stats["identity_sign_pair_child_reuses"] += len(
                candidates
            )
        self._prepared_profile_nonoptimal_keys.update(
            ((candidate,), closure_terms) for candidate in candidates
        )

    def sequence_exposures(self, ctx: QueryContext) -> np.ndarray:
        """Return exact per-sequence quadrature mass, cached by context."""
        cached = self._sequence_exposures.get(ctx.name)
        if cached is not None:
            return cached
        values = ctx.sequence_exposures()
        self._sequence_exposures[ctx.name] = values
        return values

    def sparse_nuisance_blocks(
        self,
        ctx: QueryContext,
        closure_terms: Sequence[ClosureTerm],
    ) -> tuple[SparseKernelResponse, ...]:
        source_ids = (
            *((self.target_history_source_id,) if self.target_history_source_id is not None else ()),
            *self.control_source_ids,
            *self.loan_age_baseline_source_ids,
        )
        blocks = [self.engine.sparse_response(ctx, (source,), 0) for source in source_ids]
        blocks.extend(
            self.engine.sparse_response(ctx, antecedent, window)
            for antecedent, window in sorted(closure_terms)
        )
        return tuple(blocks)

    def sparse_closure_blocks(
        self,
        ctx: QueryContext,
        closure_terms: Sequence[ClosureTerm],
    ) -> tuple[SparseKernelResponse, ...]:
        """Return only candidate-dependent hierarchy nuisance blocks."""
        return tuple(
            self.engine.sparse_response(ctx, antecedent, window)
            for antecedent, window in sorted(closure_terms)
        )

    def fixed_nuisance_partition(
        self,
        ctx: QueryContext,
    ) -> SparseNuisancePartition:
        """Return the exact fixed control/age partition shared by all supports."""
        cached = self._fixed_nuisance_partitions.get(ctx.name)
        if cached is not None:
            return cached
        with self._fixed_nuisance_partition_guard:
            cached = self._fixed_nuisance_partitions.get(ctx.name)
            if cached is not None:
                return cached
            # Only the empty-closure blocks are candidate invariant. Hierarchy
            # terms remain in every fitted model and are added by exact
            # partition refinement below; no coefficient is frozen or omitted.
            fixed = self.sparse_nuisance_blocks(ctx, ())
            partition = prepare_sparse_nuisance_partition(
                ctx,
                fixed,
                cluster_weights=(
                    self.fit_cluster_weights
                    if ctx.name == self.splits.fit.name
                    else None
                ),
                sequence_exposures=self.sequence_exposures(ctx),
                occurrence_likelihood=self.occurrence_likelihood,
            )
            self._fixed_nuisance_partitions[ctx.name] = partition
            return partition

    def prepare_partitioned_support_design(
        self,
        ctx: QueryContext,
        closure_terms: Sequence[ClosureTerm],
        features: Sequence[SparseKernelResponse],
        rules: Sequence[RuleIdentity],
        *,
        cluster_weights: np.ndarray | None = None,
        return_partition: bool = False,
    ) -> PreparedFixedSupportDesign | IncrementalSupportPartition:
        """Assemble a support by refining the shared fixed-nuisance partition."""
        return refine_sparse_nuisance_partition(
            ctx,
            self.fixed_nuisance_partition(ctx),
            self.sparse_closure_blocks(ctx, closure_terms),
            features,
            rules,
            cluster_weights=cluster_weights,
            occurrence_likelihood=self.occurrence_likelihood,
            return_partition=return_partition,
        )

    def mark_rule_activations(
        self,
        fit: FitResult,
        ctx: QueryContext,
        *,
        event_only: bool = False,
    ) -> list[np.ndarray]:
        """One shared-shape activation column per occurrence rule.

        The occurrence amplitude is intentionally omitted: its scale is
        absorbed by the unrestricted mark coefficient.  This leaves the
        temporal shape shared while allowing occurrence and amount effects to
        have different magnitudes and signs.
        """
        activations: list[np.ndarray] = []
        feature_blocks = (
            self.sparse_features(ctx, fit.rules)
            if event_only
            else self.features(ctx, fit.rules)
        )
        stop = ctx.n_events if event_only else ctx.n_queries
        for rule_index, feature in enumerate(feature_blocks):
            if fit.theta.shape[1] != feature.shape[1]:
                raise ValueError("marked activation and occurrence kernel dimensions differ")
            event_or_dense = (
                feature.event_values
                if isinstance(feature, SparseKernelResponse)
                else feature[:stop]
            )
            activations.append(event_or_dense @ fit.shapes[rule_index])
        return activations

    def _attach_mark_head(self, intensity_fit: FitResult, ctx: QueryContext) -> FitResult:
        if not self.marked:
            return intensity_fit
        if ctx.event_marks is None or self.mark_unit is None:
            raise ValueError("marked model requires event marks in every split")
        if self.mark_variance is None and (intensity_fit.rules or intensity_fit.closure_terms):
            # Estimate the mark noise scale once under the global D_fit null,
            # before comparing any support.  This call terminates at the empty
            # support/empty closure case below.
            self.fit_model((), ())
        mark_fit = fit_mark_head(
            ctx,
            self.nuisance_event_design(ctx, intensity_fit.closure_terms),
            self.mark_rule_activations(intensity_fit, ctx, event_only=True),
            unit=self.mark_unit,
            variance=self.mark_variance,
            cluster_weights=(self.fit_cluster_weights if ctx.name == self.splits.fit.name else None),
            base_residualizer=self.mark_base_residualizer(
                ctx,
                intensity_fit.closure_terms,
            ),
        )
        if self.mark_variance is None:
            if intensity_fit.rules or intensity_fit.closure_terms:
                raise RuntimeError("mark variance was not initialized by the D_fit baseline")
            self.mark_variance = float(mark_fit.variance)
        intensity_nll = float(
            intensity_fit.nll
            if intensity_fit.intensity_nll is None
            else intensity_fit.intensity_nll
        )
        return replace(
            intensity_fit,
            nll=float(intensity_nll + mark_fit.nll),
            intensity_nll=intensity_nll,
            mark_fit=mark_fit,
        )

    def hierarchy_closure(self, rules: Sequence[RuleIdentity]) -> tuple[ClosureTerm, ...]:
        rules = tuple(sorted(rules))
        cached = self._hierarchy_closure_cache.get(rules)
        if cached is not None:
            return cached
        represented = {(rule.antecedent, int(rule.window)) for rule in rules}
        closure: set[ClosureTerm] = set()
        for rule in rules:
            for order in range(1, len(rule.antecedent)):
                for subset in itertools.combinations(rule.antecedent, order):
                    term = (tuple(subset), 0 if order == 1 else int(rule.window))
                    if term not in represented:
                        closure.add(term)
        result = tuple(sorted(closure))
        self._hierarchy_closure_cache[rules] = result
        return result

    @staticmethod
    def hierarchy_preserving_drop(
        rules: Sequence[RuleIdentity],
        focal: RuleIdentity,
    ) -> tuple[tuple[RuleIdentity, ...], tuple[RuleIdentity, ...]]:
        """Drop a rule together with higher-order rules that depend on it.

        With A and AB in one support, keeping AB after removing A would also
        remove A from AB's hierarchy adjustment.  The resulting null changes
        the meaning of AB and is not a valid conditional comparison.  The
        smallest nested comparison therefore removes the whole branch rooted
        at A.  Candidate generation itself remains free of heredity rules.
        """
        focal_sources = set(focal.antecedent)

        def depends_on_focal(rule: RuleIdentity) -> bool:
            if not focal_sources < set(rule.antecedent):
                return False
            required_window = 0 if len(focal.antecedent) == 1 else int(rule.window)
            return int(focal.window) == required_window

        removed = tuple(
            rule
            for rule in rules
            if rule == focal or depends_on_focal(rule)
        )
        removed_set = set(removed)
        remaining = tuple(rule for rule in rules if rule not in removed_set)
        return tuple(sorted(remaining)), tuple(sorted(removed))

    def nuisance_design(self, ctx: QueryContext, closure_terms: Sequence[ClosureTerm]) -> np.ndarray:
        closure_terms = tuple(sorted(closure_terms))
        parts = list(self.control_blocks(ctx))
        parts.extend(self.engine.response(ctx, antecedent, window) for antecedent, window in closure_terms)
        return (
            np.concatenate(parts, axis=1)
            if parts
            else np.zeros((ctx.n_queries, 0), dtype=np.float32)
        )

    def nuisance_blocks(
        self,
        ctx: QueryContext,
        closure_terms: Sequence[ClosureTerm],
        *,
        event_only: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Return nuisance blocks without a full-grid concatenation."""
        if event_only:
            return tuple(
                block.event_values
                for block in self.sparse_nuisance_blocks(ctx, closure_terms)
            )
        stop = ctx.n_events if event_only else ctx.n_queries
        blocks: list[np.ndarray] = []
        blocks.extend(block[:stop] for block in self.control_blocks(ctx))
        blocks.extend(
            self.engine.response(ctx, antecedent, window)[:stop]
            for antecedent, window in sorted(closure_terms)
        )
        return tuple(blocks)

    def nuisance_event_design(
        self,
        ctx: QueryContext,
        closure_terms: Sequence[ClosureTerm],
    ) -> np.ndarray:
        key = (str(ctx.name), tuple(sorted(closure_terms)))
        cached = self._nuisance_event_design_cache.get(key)
        if cached is not None:
            return cached
        blocks = self.nuisance_blocks(ctx, key[1], event_only=True)
        result = (
            np.concatenate(blocks, axis=1)
            if blocks
            else np.zeros((ctx.n_events, 0), dtype=np.float32)
        )
        # Query contexts and hierarchy terms are immutable after construction;
        # the event-only matrix can therefore be shared by every mark fit and
        # score calculation without changing numerical results.
        result.setflags(write=False)
        self._nuisance_event_design_cache[key] = result
        return result

    def mark_base_residualizer(
        self,
        ctx: QueryContext,
        closure_terms: Sequence[ClosureTerm],
    ) -> MarkBaseResidualizer:
        """Reuse the exact weighted nuisance factorization across mark fits."""
        key = (str(ctx.name), tuple(sorted(closure_terms)))
        cached = self._mark_base_residualizer_cache.get(key)
        if cached is not None:
            return cached
        residualizer = make_mark_base_residualizer(
            ctx,
            self.nuisance_event_design(ctx, key[1]),
            cluster_weights=(
                self.fit_cluster_weights
                if ctx.name == self.splits.fit.name
                else None
            ),
        )
        self._mark_base_residualizer_cache[key] = residualizer
        return residualizer

    def fit_model(
        self,
        rules: Sequence[RuleIdentity],
        closure_terms: Sequence[ClosureTerm],
        *,
        initial: FitResult | None = None,
    ) -> FitResult:
        rules = tuple(sorted(rules))
        closure_terms = tuple(sorted(closure_terms))
        key = (rules, closure_terms)
        cached = self._fit_cache.get(key)
        if cached is not None:
            return cached
        with self._fit_key_locks_guard:
            key_lock = self._fit_key_locks.setdefault(key, threading.Lock())
        with key_lock:
            cached = self._fit_cache.get(key)
            if cached is not None:
                return cached
            return self._fit_model_uncached(
                rules,
                closure_terms,
                initial=initial,
            )

    @staticmethod
    def _nonattained_rule_recession_columns(
        prepared: PreparedFixedSupportDesign,
    ) -> tuple[tuple[int, int], ...]:
        """Find rule columns that prove a nonattained Poisson optimum.

        A nonnegative coefficient whose signed design is zero on every
        positive-weight event row and nonpositive on every positive-weight
        exposure row (strictly negative somewhere) is a feasible recession
        direction.  Sending it to infinity lowers the integrated intensity
        without an event penalty.  This is an exact design-cone condition, not
        a minimum event-count or effect-size threshold.
        """
        knot_count = int(prepared.knot_count)
        rule_count = len(prepared.rules)
        if knot_count <= 0 or rule_count <= 0:
            return ()
        left = int(prepared.constrained_start)
        right = left + rule_count * knot_count
        design = np.asarray(prepared.design)
        if right > design.shape[1]:
            raise ValueError("prepared rule block exceeds design width")
        rule_design = design[:, left:right]
        event_mask = np.asarray(prepared.event_weights) > 0.0
        grid_mask = np.asarray(prepared.grid_weights) > 0.0
        event_rows = rule_design[: prepared.n_events]
        grid_rows = rule_design[prepared.n_events :]
        if not np.any(grid_mask):
            return ()
        active_grid = grid_rows[grid_mask]
        nonpositive = np.all(active_grid <= 0.0, axis=0)
        strict_negative = np.any(active_grid < 0.0, axis=0)
        if prepared.occurrence_likelihood == "poisson":
            event_compatible = (
                ~np.any(event_rows[event_mask] != 0.0, axis=0)
                if np.any(event_mask)
                else np.ones(rule_count * knot_count, dtype=bool)
            )
            strict_direction = strict_negative
        else:
            active_events = event_rows[event_mask]
            event_compatible = (
                np.all(active_events >= 0.0, axis=0)
                if len(active_events)
                else np.ones(rule_count * knot_count, dtype=bool)
            )
            strict_direction = strict_negative | (
                np.any(active_events > 0.0, axis=0)
                if len(active_events)
                else False
            )
        flat_columns = np.flatnonzero(
            event_compatible & nonpositive & strict_direction
        )
        return tuple(
            (int(column // knot_count), int(column % knot_count))
            for column in flat_columns
        )

    def _nonattained_support_fit(
        self,
        rules: tuple[RuleIdentity, ...],
        closure_baseline: FitResult,
        knot_count: int,
        recession_columns: Sequence[tuple[int, int]],
    ) -> FitResult:
        with self._diagnostic_guard:
            self._safe_screen_stats["rule_recession_support_rejections"] += 1
            self._safe_screen_stats["rule_recession_columns"] += len(
                recession_columns
            )
        return replace(
            closure_baseline,
            rules=rules,
            theta=np.zeros(
                (len(rules), int(knot_count)), dtype=np.float64
            ),
            nll=float(closure_baseline.nll),
            kkt_residual=math.inf,
            converged=False,
            iterations=0,
            mark_fit=None,
            solver_hessian=None,
        )

    def _fit_model_uncached(
        self,
        rules: tuple[RuleIdentity, ...],
        closure_terms: tuple[ClosureTerm, ...],
        *,
        initial: FitResult | None,
    ) -> FitResult:
        """Fit one canonical cache key while its per-key lock is held."""
        key = (rules, closure_terms)
        nested_null_warm_start = False
        # Hierarchy-null models form a nested family.  Reuse the largest
        # converged cached subset closure and insert exactly-zero coefficients
        # for only the new nuisance terms.  This changes no model, objective,
        # constraint, or stopping rule; it only avoids restarting every null
        # fit from the intercept when active search repeatedly enlarges a
        # hierarchy closure.
        if not rules and closure_terms and initial is None:
            target_set = set(closure_terms)
            # Host support workers can share the null cache. Snapshot while
            # holding the cache/lock registry guard so iteration cannot race a
            # different closure fit being published.
            with self._fit_key_locks_guard:
                cached_null_items = tuple(self._null_fit_cache.items())
            subset_candidates = [
                fit
                for cached_closure, fit in cached_null_items
                if fit.converged
                and set(cached_closure).issubset(target_set)
                and cached_closure != closure_terms
            ]
            if subset_candidates:
                source = max(
                    subset_candidates,
                    key=lambda fit: (len(fit.closure_terms), -float(fit.nll)),
                )
                control_width = sum(
                    int(block.shape[1])
                    for block in self.sparse_nuisance_blocks(
                        self.splits.fit,
                        (),
                    )
                )
                knot_count = int(self.config.knot_count)
                expected_source_width = control_width + knot_count * len(
                    source.closure_terms
                )
                if len(source.gamma) == expected_source_width:
                    expanded_gamma = np.zeros(
                        control_width + knot_count * len(closure_terms),
                        dtype=np.float64,
                    )
                    expanded_gamma[:control_width] = source.gamma[:control_width]
                    source_index = {
                        term: index for index, term in enumerate(source.closure_terms)
                    }
                    for target_index, term in enumerate(closure_terms):
                        old_index = source_index.get(term)
                        if old_index is None:
                            continue
                        old_left = control_width + old_index * knot_count
                        new_left = control_width + target_index * knot_count
                        expanded_gamma[new_left : new_left + knot_count] = (
                            source.gamma[old_left : old_left + knot_count]
                        )
                    initial = replace(
                        source,
                        closure_terms=closure_terms,
                        gamma=expanded_gamma,
                        mark_fit=None,
                        solver_hessian=None,
                    )
                    nested_null_warm_start = True
                    with self._diagnostic_guard:
                        self._safe_screen_stats["nested_null_warm_starts"] += 1
        if rules:
            closure_baseline = self.fit_model((), closure_terms)
            if initial is None:
                subset_fits = [
                    self._fit_cache.get(
                        (tuple(rule for rule in rules if rule != omitted), closure_terms)
                    )
                    for omitted in rules
                ]
                converged_subsets = [
                    candidate
                    for candidate in subset_fits
                    if candidate is not None and candidate.converged
                ]
                initial = (
                    min(converged_subsets, key=lambda candidate: candidate.nll)
                    if converged_subsets
                    else closure_baseline
                )
            elif initial.closure_terms != closure_terms:
                raise ValueError("warm start and target model must use the same hierarchy closure")
        if (
            not rules
            and closure_terms
            and not self.marked
            and str(self.config.solver_device).startswith("cpu")
            and self.config.solver_dtype == "float64"
            and key not in self._prepared_design_cache
        ):
            try:
                sparse_null = prepare_sparse_delta_support_design(
                    self.splits.fit,
                    self.fixed_nuisance_partition(self.splits.fit),
                    self.sparse_closure_blocks(
                        self.splits.fit, closure_terms
                    ),
                    closure_terms,
                    (),
                    (),
                    cluster_weights=self.fit_cluster_weights,
                    occurrence_likelihood=self.occurrence_likelihood,
                )
                baseline = self.fit_model((), ())
                sparse_fit = fit_sparse_delta_support(
                    sparse_null,
                    max_iter=self.config.solver_max_iter,
                    tolerance=self.config.solver_tolerance,
                    initial=initial,
                    baseline=baseline,
                )
            except (FloatingPointError, MemoryError, RuntimeError, ValueError):
                sparse_fit = None
            if sparse_fit is not None:
                with self._fit_key_locks_guard:
                    self._fit_cache[key] = sparse_fit
                    self._null_fit_cache[closure_terms] = sparse_fit
                with self._diagnostic_guard:
                    self._safe_screen_stats[
                        "support_sparse_delta_closure_groups"
                    ] += 1
                return sparse_fit
        prepared = self._prepared_design_cache.pop(key, None)
        profile_null_known_nonoptimal = (
            key in self._prepared_profile_nonoptimal_keys
        )
        self._prepared_profile_nonoptimal_keys.discard(key)
        nuisance: Sequence[SparseKernelResponse] = ()
        # Profiling can hand the exact grouped design produced by its fused
        # cumulative-W sweep directly to the solver.  In that case rebuilding
        # the rule response merely to validate its width would repeat the
        # dominant convolution. ``fit_fixed_support`` validates the rule
        # identities and knot width against ``prepared`` itself.
        features = (
            []
            if prepared is not None
            else self.sparse_features(self.splits.fit, rules)
        )
        if prepared is None:
            prepared = self.prepare_partitioned_support_design(
                self.splits.fit,
                closure_terms,
                features,
                rules,
                cluster_weights=self.fit_cluster_weights,
            )
            # ``prepared`` is now the complete sufficient statistic.  Avoid
            # regenerating fixed/closure responses merely for width checks.
            features = []
        if (
            str(self.config.solver_device).startswith("cpu")
            and self.config.solver_dtype == "float64"
        ):
            # Drop the grouped float32 matrix before the host Newton loop.  Its
            # values promote exactly, and the promoted matrix is also reused by
            # the warm-start KKT check below.
            prepared = promote_prepared_design_float64(prepared)
        recession_columns = self._nonattained_rule_recession_columns(prepared)
        if recession_columns:
            if not rules:
                raise RuntimeError(
                    "nuisance-only fit cannot own rule recession columns"
                )
            fit = self._nonattained_support_fit(
                rules,
                closure_baseline,
                int(prepared.knot_count),
                recession_columns,
            )
            with self._fit_key_locks_guard:
                self._fit_cache[key] = fit
            return fit
        if initial is not None and initial.converged:
            knot_count = int(prepared.knot_count) if rules else 0
            expanded_theta = np.zeros((len(rules), knot_count), dtype=np.float64)
            initial_rule_index = {rule: index for index, rule in enumerate(initial.rules)}
            for new_index, rule in enumerate(rules):
                old_index = initial_rule_index.get(rule)
                if old_index is not None and initial.theta.shape[1] == knot_count:
                    expanded_theta[new_index] = initial.theta[old_index]
            expanded_mark = initial.mark_fit
            if expanded_mark is not None:
                expanded_rule_beta = np.zeros(len(rules), dtype=np.float64)
                for new_index, rule in enumerate(rules):
                    old_index = initial_rule_index.get(rule)
                    if old_index is not None and old_index < len(expanded_mark.rule_beta):
                        expanded_rule_beta[new_index] = expanded_mark.rule_beta[old_index]
                expanded_mark = replace(
                    expanded_mark,
                    rule_beta=expanded_rule_beta,
                )
            expanded_initial = replace(
                initial,
                rules=rules,
                closure_terms=closure_terms,
                theta=expanded_theta,
                mark_fit=expanded_mark,
                solver_hessian=None,
            )
            # ``prepared`` already contains the exact grouped sufficient
            # statistics.  A former sparse rule-only prefilter scanned the
            # ungrouped active grid before this complete KKT check.  It could
            # neither accept a fit nor inspect nuisance coordinates and was
            # 40x slower than the complete check on IBM; bypassing it changes
            # no objective, tolerance, fitted point, or shortcut decision.
            if (
                rules
                and not nested_null_warm_start
                and not profile_null_known_nonoptimal
            ):
                already_optimal, initial_kkt, _initial_objective = (
                    fixed_support_projected_kkt(
                        prepared,
                        expanded_initial,
                        tolerance=self.config.solver_tolerance,
                    )
                )
            else:
                already_optimal = False
                initial_kkt = math.inf
                if rules and profile_null_known_nonoptimal:
                    with self._diagnostic_guard:
                        self._safe_screen_stats[
                            "identity_redundant_null_kkt_passes_skipped"
                        ] += 1
            if already_optimal:
                fit = replace(
                    expanded_initial,
                    kkt_residual=float(initial_kkt),
                    converged=True,
                    iterations=0,
                )
                if self.marked and fit.mark_fit is None:
                    fit = self._attach_mark_head(fit, self.splits.fit)
                with self._fit_key_locks_guard:
                    self._fit_cache[key] = fit
                if not rules:
                    with self._fit_key_locks_guard:
                        self._null_fit_cache[closure_terms] = fit
                with self._diagnostic_guard:
                    self._safe_screen_stats["child_kkt_shortcuts"] += 1
                return fit
        fit = fit_fixed_support(
            self.splits.fit,
            nuisance,
            features,
            rules,
            device=self.config.solver_device,
            dtype=self.config.solver_dtype,
            max_iter=self.config.solver_max_iter,
            tolerance=self.config.solver_tolerance,
            initial=initial,
            closure_terms=closure_terms,
            cluster_weights=self.fit_cluster_weights,
            sequence_exposures=self.sequence_exposures(self.splits.fit),
            prepared_design=prepared,
            occurrence_likelihood=self.occurrence_likelihood,
        )
        fit = self._attach_mark_head(fit, self.splits.fit)
        with self._fit_key_locks_guard:
            self._fit_cache[key] = fit
        if not rules:
            with self._fit_key_locks_guard:
                self._null_fit_cache[closure_terms] = fit
        return fit

    def fit_support(self, rules: Sequence[RuleIdentity]) -> FitResult:
        rules = tuple(sorted(rules))
        return self.fit_model(rules, self.hierarchy_closure(rules))

    def _prefit_profile_nulls(
        self,
        closure_sets: Sequence[Sequence[ClosureTerm]],
    ) -> None:
        """Fit profiling nulls with the fastest measured exact backend.

        Triplet W profiling uses up to 13 hierarchy-null occurrence GLMs with
        the same parameter width.  The batched solver pads only zero-weight
        grouped rows and certifies every result with
        ``fixed_support_projected_kkt`` in host float64; any item that misses
        that tolerance falls back to the ordinary scalar solver.  Thus this
        changes dispatch, not an objective, candidate, constraint, or
        convergence requirement.
        """
        ordered = list(
            dict.fromkeys(tuple(sorted(terms)) for terms in closure_sets)
        )
        missing = [
            terms
            for terms in ordered
            if ((), terms) not in self._fit_cache
        ]
        if len(missing) < 2:
            for terms in missing:
                self.fit_model((), terms)
            return

        if (
            str(self.config.solver_device).startswith("cpu")
            and self.occurrence_likelihood == "poisson"
        ):
            # On the audited IBM grouped designs, host scalar solves are faster
            # end-to-end than tensor padding. Do not pass the preceding W fit
            # positionally: its closure columns denote different W-specific
            # functions.
            for terms in missing:
                self.fit_model((), terms)
            with self._diagnostic_guard:
                self._safe_screen_stats["profile_null_models_scalar"] += len(
                    missing
                )
            return

        controls: list[tuple[SparseKernelResponse, ...]] = []
        prepared: list[PreparedFixedSupportDesign] = []
        for terms in missing:
            nuisance = self.sparse_nuisance_blocks(self.splits.fit, terms)
            controls.append(nuisance)
            prepared.append(
                self.prepare_partitioned_support_design(
                    self.splits.fit,
                    terms,
                    [],
                    [],
                    cluster_weights=self.fit_cluster_weights,
                )
            )

        # A given antecedent order normally has one width.  Retain exact scalar
        # handling for any structurally unusual singleton-width group.
        by_width: dict[int, list[int]] = {}
        for index, item in enumerate(prepared):
            by_width.setdefault(int(item.design.shape[1]), []).append(index)
        for indices in by_width.values():
            if len(indices) == 1:
                index = indices[0]
                self._prepared_design_cache[((), missing[index])] = prepared[index]
                self.fit_model((), missing[index])
                continue
            batch_fits = fit_unconstrained_prepared_batch(
                self.splits.fit,
                [controls[index] for index in indices],
                [prepared[index] for index in indices],
                [missing[index] for index in indices],
                device=self.config.solver_device,
                dtype=self.config.solver_dtype,
                max_iter=self.config.solver_max_iter,
                tolerance=self.config.solver_tolerance,
            )
            with self._diagnostic_guard:
                self._safe_screen_stats["profile_null_fit_batches"] += 1
                self._safe_screen_stats["profile_null_models_batched"] += len(
                    batch_fits
                )
            for index, intensity_fit in zip(indices, batch_fits, strict=True):
                terms = missing[index]
                fit = self._attach_mark_head(intensity_fit, self.splits.fit)
                with self._fit_key_locks_guard:
                    self._fit_cache[((), terms)] = fit
                    self._null_fit_cache[terms] = fit

    def fit_frozen_support_on_context(
        self,
        rules: Sequence[RuleIdentity],
        ctx: QueryContext,
        *,
        initial: FitResult | None = None,
    ) -> FitResult:
        """Refit a frozen support without reopening discovery or certification."""
        rules = tuple(sorted(rules))
        closure_terms = self.hierarchy_closure(rules)
        prepared = self.prepare_partitioned_support_design(
            ctx,
            closure_terms,
            self.sparse_features(ctx, rules),
            rules,
            cluster_weights=(
                self.fit_cluster_weights
                if ctx.name == self.splits.fit.name
                else None
            ),
        )
        fit = fit_fixed_support(
            ctx,
            (),
            (),
            rules,
            device=self.config.solver_device,
            dtype=self.config.solver_dtype,
            max_iter=self.config.solver_max_iter,
            tolerance=self.config.solver_tolerance,
            initial=initial,
            closure_terms=closure_terms,
            cluster_weights=(
                self.fit_cluster_weights
                if ctx.name == self.splits.fit.name
                else None
            ),
            sequence_exposures=self.sequence_exposures(ctx),
            prepared_design=prepared,
            occurrence_likelihood=self.occurrence_likelihood,
        )
        return self._attach_mark_head(fit, ctx)

    def fit_baseline(self) -> FitResult:
        if self.baseline_fit is None:
            self.baseline_fit = self.fit_support(())
            if not self.baseline_fit.converged:
                raise RuntimeError(
                    "control baseline did not satisfy the configured KKT tolerance; "
                    "the candidate family cannot be certified"
                )
        return self.baseline_fit

    def seed_profiled_library(
        self,
        rules: Sequence[RuleIdentity],
        *,
        identity_candidates: dict[tuple[int, ...], tuple[RuleIdentity, ...]],
        dictionary_shapes: dict[RuleIdentity, np.ndarray],
    ) -> None:
        """Freeze a D_fit-pricing library for exact fitting on this pipeline."""
        self.profiled_rules = sorted(set(rules))
        self.identity_candidates = {
            tuple(antecedent): tuple(sorted(candidates))
            for antecedent, candidates in identity_candidates.items()
            if any(rule.antecedent == tuple(antecedent) for rule in self.profiled_rules)
        }
        self.rule_dictionary_shapes = {
            rule: np.asarray(shape, dtype=np.float64).copy()
            for rule, shape in dictionary_shapes.items()
            if rule.antecedent in self.identity_candidates
        }
        self._profile_completed = True

    def accept_seeded_rules_on_fit(self) -> dict:
        """Exact block-MDL acceptance of priced atoms on this pipeline's D_fit."""
        self.fit_baseline()
        if not self._profile_completed:
            raise RuntimeError("a priced rule library must be seeded before full-fit acceptance")
        self._start_active_support_workers()
        records = self._fit_or_safe_screen_records_batch(
            [(rule,) for rule in self.profiled_rules],
            profile="full-d-fit-exact-atom-acceptance-after-ipw-pricing",
        )
        rows: list[dict] = []
        accepted: list[RuleIdentity] = []
        for record in records:
            score = self._support_search_score(record)
            passed = bool(math.isfinite(score) and score > 0.0)
            safe_bound = self._safe_bound_cache.get(record.rules)
            safely_screened = record.rules in self._safe_screened_records
            if passed:
                accepted.extend(record.rules)
            rows.append(
                {
                    "rule": self._rule_dict(record.rules[0]),
                    "full_fit_nll": float(record.fit.nll),
                    "closure_baseline_nll": float(record.closure_baseline_fit.nll),
                    "full_fit_block_mdl_score": float(score) if math.isfinite(score) else None,
                    "fit_converged": bool(record.fit.converged),
                    "accepted": passed,
                    "safe_mdl_screened": safely_screened,
                    "safe_score_upper_bound": (
                        float(safe_bound["score_upper_bound"])
                        if safely_screened
                        and safe_bound is not None
                        and math.isfinite(float(safe_bound["score_upper_bound"]))
                        else None
                    ),
                }
            )
        self.profiled_rules = sorted(set(accepted))
        for worker in self._active_support_workers:
            worker.profiled_rules = list(self.profiled_rules)
        return {
            "claim": "full_d_fit_exact_atom_acceptance_after_ipw_gradient_pricing",
            "priced_rule_count": len(records),
            "accepted_rule_count": len(self.profiled_rules),
            "safe_mdl_screen": {
                "enabled": bool(self.config.safe_mdl_screen),
                **self._safe_screen_stats,
            },
            "rows": rows,
        }

    def _active_amplitudes(self, fit: FitResult) -> np.ndarray:
        amplitude = fit.amplitudes
        if amplitude.size == 0:
            return np.zeros(0, dtype=bool)
        solver_type = np.float32 if self.config.solver_dtype == "float32" else np.float64
        numeric = (
            np.finfo(solver_type).eps
            * max(1, int(fit.theta.size))
            * max(1.0, float(np.max(amplitude)))
        )
        return amplitude > numeric

    def profile_rule_identities(
        self,
        *,
        antecedent_subset: Sequence[tuple[int, ...]] | None = None,
    ) -> list[RuleIdentity]:
        """Profile every finite W/sign identity for the requested skeletons.

        ``antecedent_subset`` restricts only the antecedent skeleton population.
        It does not seed or freeze a W/sign identity: every finite window and
        both signs for each retained skeleton still enter the ordinary exact
        profile below.  This distinction is required by the multi-fidelity
        screen, whose sampled/IPW stage may reject skeletons but may not choose
        the identity that is later fitted on complete D_fit.
        """
        baseline = self.fit_baseline()
        # Build the candidate-invariant control/age row partition once in the
        # parent. POSIX workers inherit it copy-on-write, so no child rescans
        # the full risk grid merely to reconstruct the same nuisance patterns.
        self.fixed_nuisance_partition(self.splits.fit)
        fit_seq = self.splits.fit.global_sequence_ids
        complete_antecedents = self.engine.antecedents(
            self.rule_source_ids, self.config.q_max
        )
        if antecedent_subset is None:
            antecedents = complete_antecedents
        else:
            complete_set = set(complete_antecedents)
            requested = {tuple(int(value) for value in item) for item in antecedent_subset}
            invalid = requested - complete_set
            if invalid:
                raise ValueError(
                    "antecedent subset contains skeletons outside the finite family: "
                    + ", ".join(str(item) for item in sorted(invalid))
                )
            # Retain the canonical finite-family order so sequential and forked
            # execution have identical tie-breaking and reporting order.
            antecedents = [
                antecedent
                for antecedent in complete_antecedents
                if antecedent in requested
            ]
        candidate_rules: list[RuleIdentity] = []
        logs: list[dict] = []

        def profile_one(
            pipeline: CertSCRPipeline,
            antecedent: tuple[int, ...],
        ) -> tuple[list[RuleIdentity], tuple[RuleIdentity, ...], dict]:
            windows = pipeline.engine.window_breakpoints(
                antecedent,
                fit_seq,
                max_window=pipeline.config.max_formation_window,
                context=pipeline.splits.fit,
            )
            if windows.size == 0:
                return [], (), {"antecedent": list(antecedent), "status": "no_fit_completion"}
            if pipeline.config.identity_profile in {"score_mdl", "dictionary_mdl"}:
                dictionary_mode = pipeline.config.identity_profile == "dictionary_mdl"
                minimum_event_span = pipeline.engine.minimum_event_activating_span(
                    pipeline.splits.fit,
                    antecedent,
                    max_window=int(windows[-1]),
                )
                if minimum_event_span is None:
                    identities = tuple(
                        RuleIdentity(antecedent, int(window), sign)
                        for window in windows.tolist()
                        for sign in (-1, 1)
                    )
                    with pipeline._diagnostic_guard:
                        pipeline._safe_screen_stats[
                            "identity_no_event_activation_screens"
                        ] += len(identities)
                    return [], identities, {
                        "antecedent": list(antecedent),
                        "antecedent_names": [
                            pipeline.data.predicate_names[idx]
                            for idx in antecedent
                        ],
                        "status": "structurally_excluded_no_event_activation",
                        "candidate_count": len(identities),
                        "minimum_event_activating_span": None,
                        "candidates": [
                            {
                                "window": int(rule.window),
                                "sign": int(rule.sign),
                                "joint_occurrence_mark_quadratic_gain": 0.0,
                                "dictionary_shape_index": None,
                                "exact_fit_status": (
                                    "zero_boundary_certified_no_event_activation"
                                    if rule.sign > 0
                                    else "rejected_nonattained_no_event_activation"
                                ),
                            }
                            for rule in identities
                        ],
                    }
                fused_dictionary_designs = bool(
                    dictionary_mode
                    and not pipeline.config.gradient_pricing_only
                    and not pipeline.marked
                    and not pipeline.config.safe_mdl_screen
                )
                scored: list[tuple[float, RuleIdentity, FitResult, int | None]] = []
                candidate_log: list[dict] = []
                identities: list[RuleIdentity] = []
                prefitted_records: dict[RuleIdentity, SupportRecord] = {}
                if (
                    not pipeline.config.gradient_pricing_only
                    and not fused_dictionary_designs
                ):
                    # All W values remain in the finite identity family.  Only
                    # their equal-width hierarchy nulls are solved together;
                    # each is still independently certified at the configured
                    # host-float64 KKT tolerance before any score is computed.
                    pipeline._prefit_profile_nulls(
                        [
                            pipeline.hierarchy_closure(
                                (
                                    RuleIdentity(
                                        antecedent=antecedent,
                                        window=int(window),
                                        sign=1,
                                    ),
                                )
                            )
                            for window in windows.tolist()
                        ]
                    )
                # Each successive response is cumulative in W, but remains
                # sparse over the quadrature grid.  Score moments are evaluated
                # exactly under the corresponding fitted null on only those
                # rows where this candidate can change the objective.
                response_iterator = pipeline.engine.iter_window_sparse_responses(
                    pipeline.splits.fit,
                    antecedent,
                    windows.tolist(),
                )
                previous_full_partition: IncrementalSupportPartition | None = None
                previous_partition_window: int | None = None
                previous_profile_null: FitResult | None = None
                for window, sparse_response in response_iterator:
                    window_candidates: list[tuple[RuleIdentity, dict]] = []
                    probe = RuleIdentity(antecedent=antecedent, window=int(window), sign=1)
                    # A nonnegative activation that is zero at every observed
                    # target event cannot own a finite nonzero rule optimum.
                    # Excitation has an everywhere nonnegative objective
                    # derivative and is minimized at amplitude zero; inhibition
                    # either has the same zero solution (no grid activation) or
                    # a nonattained recession direction.  This is an exact
                    # structural likelihood certificate, independent of the
                    # nuisance coefficients, and avoids constructing its dense
                    # hierarchy design.
                    if not np.any(sparse_response.event_values != 0.0):
                        for sign in (-1, 1):
                            rule = RuleIdentity(
                                antecedent=antecedent,
                                window=int(window),
                                sign=sign,
                            )
                            identities.append(rule)
                            candidate_log.append(
                                {
                                    "window": int(window),
                                    "sign": int(sign),
                                    "joint_occurrence_mark_quadratic_gain": 0.0,
                                    "dictionary_shape_index": None,
                                    "null_boundary_projected_kkt": 0.0,
                                    "null_boundary_kkt_certified": sign > 0,
                                    "exact_fit_status": (
                                        "zero_boundary_certified_no_event_activation"
                                        if sign > 0
                                        else "rejected_nonattained_no_event_activation"
                                    ),
                                    "exact_nll": None,
                                    "exact_block_mdl": None,
                                    "exact_converged": False,
                                }
                            )
                        with pipeline._diagnostic_guard:
                            pipeline._safe_screen_stats[
                                "identity_no_event_activation_screens"
                            ] += 2
                        continue
                    full_m_parent: PreparedFixedSupportDesign | None = None
                    if pipeline.config.gradient_pricing_only:
                        # Sampled pricing is a deliberately inclusive gradient
                        # screen.  Re-fitting a different hierarchical null for
                        # every triplet/window duplicates the later full-D_fit
                        # exact acceptance and dominated runtime.  The global
                        # fitted baseline prices potential atoms; full D_fit
                        # then applies the exact hierarchy closure, full M-knot
                        # cone, KKT contract and block-MDL objective once.
                        closure_terms = ()
                        closure_baseline = baseline
                    else:
                        closure_terms = pipeline.hierarchy_closure((probe,))
                        null_key = ((), closure_terms)
                        if (
                            fused_dictionary_designs
                            and null_key not in pipeline._fit_cache
                        ):
                            if (
                                previous_full_partition is not None
                                and previous_partition_window is not None
                            ):
                                delta_controls = [
                                    pipeline.engine.sparse_window_delta_response(
                                        pipeline.splits.fit,
                                        closure_antecedent,
                                        previous_partition_window,
                                        int(window),
                                    )
                                    for closure_antecedent, _closure_window in closure_terms
                                ]
                                delta_rule = (
                                    pipeline.engine.sparse_window_delta_response(
                                        pipeline.splits.fit,
                                        antecedent,
                                        previous_partition_window,
                                        int(window),
                                    )
                                )
                                try:
                                    previous_full_partition = (
                                        update_incremental_support_partition(
                                            pipeline.splits.fit,
                                            previous_full_partition,
                                            delta_controls,
                                            (delta_rule,),
                                            (probe,),
                                            cluster_weights=pipeline.fit_cluster_weights,
                                            occurrence_likelihood=(
                                                pipeline.occurrence_likelihood
                                            ),
                                        )
                                    )
                                except RuntimeError:
                                    # The native incremental partition is only
                                    # an accelerator. Repeated IPW regrouping
                                    # can accumulate a tiny mass-roundoff
                                    # history that is not recoverable from the
                                    # compressed parent alone. Rebuild this W
                                    # from its cumulative sparse responses;
                                    # this is the same canonical design used
                                    # by a cold exact fit, not an approximation.
                                    rebuilt = (
                                        pipeline.prepare_partitioned_support_design(
                                            pipeline.splits.fit,
                                            closure_terms,
                                            [sparse_response],
                                            [probe],
                                            cluster_weights=(
                                                pipeline.fit_cluster_weights
                                            ),
                                            return_partition=True,
                                        )
                                    )
                                    if not isinstance(
                                        rebuilt, IncrementalSupportPartition
                                    ):
                                        raise RuntimeError(
                                            "exact incremental fallback did not retain its partition"
                                        )
                                    previous_full_partition = rebuilt
                                    with pipeline._diagnostic_guard:
                                        pipeline._safe_screen_stats[
                                            "identity_incremental_partition_rebuilds"
                                        ] += 1
                            else:
                                initial_partition = (
                                    pipeline.prepare_partitioned_support_design(
                                        pipeline.splits.fit,
                                        closure_terms,
                                        [sparse_response],
                                        [probe],
                                        cluster_weights=pipeline.fit_cluster_weights,
                                        return_partition=True,
                                    )
                                )
                                if not isinstance(
                                    initial_partition, IncrementalSupportPartition
                                ):
                                    raise RuntimeError(
                                        "incremental identity partition was not retained"
                                    )
                                previous_full_partition = initial_partition
                            previous_partition_window = int(window)
                            full_m_parent = previous_full_partition.prepared
                            pipeline._prepared_design_cache[null_key] = (
                                project_prepared_support_design(
                                    full_m_parent,
                                    (),
                                    source_closure_terms=closure_terms,
                                    target_closure_terms=closure_terms,
                                    # The full-M groups are a finer exact null
                                    # partition. Re-hashing removed at most 2%
                                    # of Freddie rows but cost another complete
                                    # pass. Project directly into the solver's
                                    # float64 dtype and retain the finer masses.
                                    regroup=False,
                                    output_dtype=(
                                        np.float64
                                        if str(
                                            pipeline.config.solver_device
                                        ).startswith("cpu")
                                        and pipeline.config.solver_dtype == "float64"
                                        else None
                                    ),
                                )
                            )
                        null_initial = (
                            replace(
                                previous_profile_null,
                                closure_terms=closure_terms,
                                mark_fit=None,
                                solver_hessian=None,
                            )
                            if previous_profile_null is not None
                            else None
                        )
                        closure_baseline = pipeline.fit_model(
                            (), closure_terms, initial=null_initial
                        )
                        if null_initial is not None:
                            with pipeline._diagnostic_guard:
                                pipeline._safe_screen_stats[
                                    "profile_window_null_warm_starts"
                                ] += 1
                        if closure_baseline.converged:
                            previous_profile_null = closure_baseline
                    if not closure_baseline.converged:
                        continue
                    if full_m_parent is not None:
                        gradient, information = (
                            pipeline._identity_moments_from_grouped_parent(
                                closure_baseline,
                                full_m_parent,
                            )
                        )
                    else:
                        gradient, information = pipeline._identity_moments_at_null(
                            closure_baseline,
                            sparse_response,
                        )
                    mark_gradient = None
                    mark_information = None
                    if pipeline.marked and dictionary_mode:
                        if closure_baseline.mark_fit is None:
                            raise RuntimeError("marked closure null is missing its mark head")
                        mark_gradient, mark_information = mark_score_moments(
                            closure_baseline.mark_fit,
                            pipeline.splits.fit,
                            pipeline.nuisance_event_design(
                                pipeline.splits.fit,
                                closure_baseline.closure_terms,
                            ),
                            pipeline.mark_rule_activations(
                                closure_baseline,
                                pipeline.splits.fit,
                                event_only=True,
                            ),
                            sparse_response.event_values,
                            cluster_weights=pipeline.fit_cluster_weights,
                            base_residualizer=pipeline.mark_base_residualizer(
                                pipeline.splits.fit,
                                closure_baseline.closure_terms,
                            ),
                        )
                    sign_scores = pipeline._scores_from_moments(
                        gradient,
                        information,
                        dictionary_mode=dictionary_mode,
                        mark_gradient=mark_gradient,
                        mark_information=mark_information,
                    )
                    for sign in (-1, 1):
                        rule = RuleIdentity(antecedent=antecedent, window=int(window), sign=int(sign))
                        identities.append(rule)
                        score, shape_index = sign_scores[int(sign)]
                        if dictionary_mode:
                            assert shape_index is not None
                            shape = pipeline.kernel_dictionary[
                                int(shape_index)
                            ]
                            signed_gradient = float(sign) * float(
                                shape @ gradient
                            )
                            signed_information = float(
                                shape @ information @ shape
                            )
                            null_boundary_kkt = float(
                                max(0.0, -signed_gradient)
                                / math.sqrt(
                                    max(
                                        signed_information,
                                        np.finfo(np.float64).tiny,
                                    )
                                )
                            )
                        else:
                            signed_gradient_vector = float(sign) * gradient
                            signed_information_diagonal = np.diag(information)
                            null_boundary_kkt = float(
                                np.max(
                                    np.maximum(-signed_gradient_vector, 0.0)
                                    / np.sqrt(
                                        np.maximum(
                                            signed_information_diagonal,
                                            np.finfo(np.float64).tiny,
                                        )
                                    ),
                                    initial=0.0,
                                )
                            )
                        if dictionary_mode and shape_index is not None:
                            pipeline.rule_dictionary_shapes[rule] = pipeline.kernel_dictionary[
                                int(shape_index)
                            ].copy()
                        scored.append((score, rule, closure_baseline, shape_index))
                        row = {
                                "window": int(window),
                                "sign": int(sign),
                                "joint_occurrence_mark_quadratic_gain": float(score),
                                "dictionary_shape_index": shape_index,
                                "null_boundary_projected_kkt": null_boundary_kkt,
                                "null_boundary_kkt_certified": bool(
                                    closure_baseline.converged
                                    and null_boundary_kkt
                                    <= pipeline.config.solver_tolerance
                                ),
                                "closure_baseline_nll": float(closure_baseline.nll),
                            }
                        candidate_log.append(row)
                        window_candidates.append((rule, row))
                    # On the unmarked exact-enumeration path, consume the
                    # cumulative W response before advancing the iterator.
                    # Previously it was discarded here and rebuilt once per
                    # sign during the later exact-fit tournament.
                    if (
                        not pipeline.config.gradient_pricing_only
                        and not pipeline.marked
                        and not pipeline.config.safe_mdl_screen
                    ):
                        live_rules = [
                            rule
                            for rule, row in window_candidates
                            if not bool(
                                row.get("null_boundary_kkt_certified", False)
                            )
                        ]
                        if live_rules:
                            pipeline._stage_profile_window_designs(
                                live_rules,
                                sparse_response,
                                full_m_parent=full_m_parent,
                            )
                            if not pipeline.config.safe_mdl_screen:
                                for rule in live_rules:
                                    prefitted_records[rule] = pipeline._support_record(
                                        (rule,),
                                        profile="finite-identity-exact-mdl-profile",
                                    )
                                with pipeline._diagnostic_guard:
                                    pipeline._safe_screen_stats[
                                        "identity_fused_profile_exact_fits"
                                    ] += len(live_rules)
                if not scored:
                    return [], tuple(sorted(identities)), {
                        "antecedent": list(antecedent),
                        "status": "no_converged_score_null",
                        "candidates": candidate_log,
                    }
                scored.sort(key=lambda item: (-item[0], item[1].window, -item[1].sign))
                screen_score, selected, selected_baseline, selected_shape_index = scored[0]
                if dictionary_mode:
                    assert selected_shape_index is not None
                    pipeline.rule_dictionary_shapes[selected] = pipeline.kernel_dictionary[
                        int(selected_shape_index)
                    ].copy()
                if pipeline.config.gradient_pricing_only:
                    identity_count = max(1, len(identities))
                    score_block_mdl = float(
                        2.0 * screen_score * pipeline.fit_objective_population_scale
                        - (1 + int(pipeline.marked))
                        * math.log(max(2, pipeline.splits.fit_population_sequence_count))
                        - 2.0 * math.log(len(pipeline.kernel_dictionary))
                        - 2.0 * math.log(identity_count)
                    )
                    admissible = bool(
                        selected_baseline.converged
                        and math.isfinite(score_block_mdl)
                        and score_block_mdl > 0.0
                    )
                    pipeline.engine.retain_antecedent_windows(
                        pipeline.splits.fit.name,
                        antecedent,
                        (),
                    )
                    return ([selected] if admissible else []), tuple(sorted(identities)), {
                        "antecedent": list(antecedent),
                        "antecedent_names": [
                            pipeline.data.predicate_names[idx] for idx in antecedent
                        ],
                        "status": (
                            "priced_gradient_dictionary_mdl"
                            if admissible
                            else "rejected_nonpositive_gradient_dictionary_mdl"
                        ),
                        "selected_window": int(selected.window),
                        "selected_sign": int(selected.sign),
                        "selected_quadratic_cone_gain": float(screen_score),
                        "selected_dictionary_shape_index": selected_shape_index,
                        "selected_dictionary_shape": (
                            pipeline.kernel_dictionary[int(selected_shape_index)].tolist()
                            if selected_shape_index is not None
                            else None
                        ),
                        "selected_score_block_mdl": score_block_mdl,
                        "exact_fit_deferred_to_full_d_fit_acceptance": True,
                        "candidate_count": len(candidate_log),
                        "candidates": candidate_log,
                    }
                # Quadratic cone scores determine evaluation order only.  They
                # are not valid upper bounds on the nuisance-refitted exact
                # likelihood and therefore cannot select one identity by
                # themselves.  Give every finite W/sign identity its proper
                # code length, use the group-saturated global bound only when
                # it proves that an identity cannot beat the current exact
                # incumbent, and exact-fit every remaining candidate.
                identity_tuple = tuple(sorted(identities))
                pipeline.identity_candidates[antecedent] = identity_tuple
                log_by_rule = {
                    RuleIdentity(
                        antecedent=antecedent,
                        window=int(item["window"]),
                        sign=int(item["sign"]),
                    ): item
                    for item in candidate_log
                }
                best_record: SupportRecord | None = None
                best_score = -math.inf
                best_rule: RuleIdentity | None = None
                exact_fit_count = 0
                safely_eliminated_count = 0
                if pipeline.config.safe_mdl_screen:
                    evaluation_order = scored
                    scored_by_window: dict[int, list[tuple]] = {}
                else:
                    scored_by_window = {}
                    for item in scored:
                        scored_by_window.setdefault(int(item[1].window), []).append(item)
                    ordered_windows = sorted(
                        scored_by_window,
                        key=lambda window: (
                            -max(item[0] for item in scored_by_window[window]),
                            window,
                        ),
                    )
                    evaluation_order = [
                        item
                        for window in ordered_windows
                        for item in sorted(
                            scored_by_window[window],
                            key=lambda value: (-value[0], -value[1].sign),
                        )
                    ]
                staged_window: int | None = None
                for candidate_screen_score, candidate, _baseline, _shape_index in evaluation_order:
                    row = log_by_rule[candidate]
                    # At the converged hierarchy null, the nuisance gradient
                    # already satisfies its KKT condition.  Pricing computed
                    # the exact signed block gradient and Fisher diagonal at
                    # amplitude zero.  If their projected residual satisfies
                    # the configured solver tolerance, the expanded null is
                    # already a certified solution of this convex atom model.
                    # This is the same block KKT shortcut used after design
                    # assembly, not a score/effect threshold.
                    if bool(row.get("null_boundary_kkt_certified", False)):
                        row["exact_fit_status"] = (
                            "zero_boundary_certified_by_null_cone_KKT"
                        )
                        row["exact_nll"] = float(_baseline.nll)
                        row["exact_block_mdl"] = None
                        row["exact_converged"] = bool(_baseline.converged)
                        with pipeline._diagnostic_guard:
                            pipeline._safe_screen_stats[
                                "identity_zero_boundary_kkt_screens"
                            ] += 1
                        continue
                    if (
                        not pipeline.config.safe_mdl_screen
                        and candidate not in prefitted_records
                        and staged_window != int(candidate.window)
                    ):
                        staged_window = int(candidate.window)
                        sign_pair = [
                            item[1]
                            for item in scored_by_window[staged_window]
                            if float(item[0]) > 0.0
                        ]
                        if len(sign_pair) == 2:
                            parent_rules = tuple(sorted(sign_pair))
                            closure_terms = pipeline.hierarchy_closure(
                                (parent_rules[0],)
                            )
                            parent_prepared = pipeline.prepare_partitioned_support_design(
                                pipeline.splits.fit,
                                closure_terms,
                                pipeline.sparse_features(
                                    pipeline.splits.fit, parent_rules
                                ),
                                parent_rules,
                                cluster_weights=pipeline.fit_cluster_weights,
                            )
                            for child in parent_rules:
                                pipeline._prepared_design_cache[
                                    ((child,), closure_terms)
                                ] = project_prepared_support_design(
                                    parent_prepared,
                                    (child,),
                                    source_closure_terms=closure_terms,
                                    target_closure_terms=closure_terms,
                                    regroup=True,
                                )
                            with pipeline._diagnostic_guard:
                                pipeline._safe_screen_stats[
                                    "identity_sign_pair_parent_designs"
                                ] += 1
                                pipeline._safe_screen_stats[
                                    "identity_sign_pair_child_reuses"
                                ] += 2
                    # A disabled safe screen means direct exact enumeration,
                    # not "compute the expensive bound but ignore its final
                    # decision".  The latter accidentally dominated Freddie
                    # profiling while screening no identities.  Skipping this
                    # calculation changes no candidate, fit or objective.
                    bound = (
                        pipeline._support_score_upper_bound((candidate,))
                        if pipeline.config.safe_mdl_screen
                        else None
                    )
                    bound_value = float(
                        math.inf
                        if bound is None
                        else bound.get("score_upper_bound", math.inf)
                    )
                    numeric_margin = float(
                        0.0 if bound is None else bound.get("numeric_margin", 0.0)
                    )
                    row["exact_block_mdl_upper_bound"] = (
                        bound_value if math.isfinite(bound_value) else None
                    )
                    safely_dominated = bool(
                        best_record is not None
                        and math.isfinite(bound_value)
                        and bound_value + numeric_margin < best_score
                    )
                    if safely_dominated:
                        row["exact_fit_status"] = "safely_dominated_by_global_upper_bound"
                        pipeline._prepared_design_cache.pop(
                            ((candidate,), pipeline.hierarchy_closure((candidate,))),
                            None,
                        )
                        safely_eliminated_count += 1
                        continue
                    candidate_record = prefitted_records.get(candidate)
                    if candidate_record is None:
                        candidate_record = pipeline._support_record(
                            (candidate,),
                            profile="finite-identity-exact-mdl-profile",
                        )
                    candidate_score = pipeline._support_search_score(candidate_record)
                    exact_fit_count += 1
                    row["exact_fit_status"] = "fitted"
                    row["exact_nll"] = float(candidate_record.fit.nll)
                    row["exact_block_mdl"] = (
                        float(candidate_score)
                        if math.isfinite(candidate_score)
                        else None
                    )
                    row["exact_converged"] = bool(
                        candidate_record.fit.converged
                        and candidate_record.closure_baseline_fit.converged
                    )
                    if (
                        best_record is None
                        or candidate_score > best_score
                        or (
                            candidate_score == best_score
                            and best_rule is not None
                            and candidate < best_rule
                        )
                    ):
                        best_record = candidate_record
                        best_score = float(candidate_score)
                        best_rule = candidate
                if best_record is None or best_rule is None:
                    pipeline.engine.retain_antecedent_windows(
                        pipeline.splits.fit.name, antecedent, ()
                    )
                    return [], identity_tuple, {
                        "antecedent": list(antecedent),
                        "antecedent_names": [
                            pipeline.data.predicate_names[idx] for idx in antecedent
                        ],
                        "status": "no_exact_finite_identity_fit",
                        "candidate_count": len(candidate_log),
                        "exact_fit_count": exact_fit_count,
                        "safely_eliminated_count": safely_eliminated_count,
                        "candidates": candidate_log,
                    }
                selected = best_rule
                selected_fit = best_record.fit
                selected_baseline = best_record.closure_baseline_fit
                block_mdl = best_score
                improvement = float(best_record.search_nll_improvement)
                selected_row = log_by_rule[selected]
                screen_score = float(
                    selected_row["joint_occurrence_mark_quadratic_gain"]
                )
                selected_shape_index = selected_row["dictionary_shape_index"]
                active = bool(
                    selected_fit.theta.size
                    and np.all(pipeline._active_amplitudes(selected_fit))
                )
                admissible = bool(math.isfinite(block_mdl) and block_mdl > 0.0)
                pipeline.engine.retain_antecedent_windows(
                    pipeline.splits.fit.name,
                    antecedent,
                    (selected.window,) if admissible else (),
                )
                return ([selected] if admissible else []), tuple(sorted(identities)), {
                    "antecedent": list(antecedent),
                    "antecedent_names": [pipeline.data.predicate_names[idx] for idx in antecedent],
                    "status": (
                        "profiled_exact_finite_identity_dictionary_mdl"
                        if dictionary_mode and admissible
                        else "profiled_exact_finite_identity_score_mdl" if admissible
                        else "rejected_nonpositive_block_mdl"
                    ),
                    "selected_window": int(selected.window),
                    "selected_sign": int(selected.sign),
                    "selected_quadratic_cone_gain": float(screen_score),
                    "selected_dictionary_shape_index": selected_shape_index,
                    "selected_dictionary_shape": (
                        pipeline.kernel_dictionary[int(selected_shape_index)].tolist()
                        if selected_shape_index is not None else None
                    ),
                    "selected_nll": float(selected_fit.nll),
                    "closure_baseline_nll": float(selected_baseline.nll),
                    "selected_improvement": improvement,
                    "selected_block_mdl": block_mdl,
                    "candidate_count": len(candidate_log),
                    "exact_fit_count": exact_fit_count,
                    "safely_eliminated_count": safely_eliminated_count,
                    "identity_selection_guarantee": (
                        "exact maximum over the finite W/sign working atoms after each dictionary shape is "
                        "frozen by exhaustive score pricing; only globally safe bound eliminations follow"
                    ),
                    "candidates": candidate_log,
                }
            candidates: list[tuple[float, RuleIdentity, FitResult, FitResult]] = []
            candidate_log: list[dict] = []
            for window in windows.tolist():
                for sign in (-1, 1):
                    rule = RuleIdentity(antecedent=antecedent, window=int(window), sign=int(sign))
                    fit = pipeline.fit_support((rule,))
                    closure_baseline = pipeline.fit_model((), fit.closure_terms)
                    amplitude = float(fit.amplitudes[0])
                    improvement = float(closure_baseline.nll - fit.nll)
                    candidates.append((-improvement, rule, fit, closure_baseline))
                    candidate_log.append(
                        {
                            "window": int(window),
                            "sign": int(sign),
                            "nll": float(fit.nll),
                            "closure_baseline_nll": float(closure_baseline.nll),
                            "closure_adjusted_improvement": improvement,
                            "amplitude": amplitude,
                            "kkt": float(fit.kkt_residual),
                            "converged": bool(fit.converged),
                        }
                    )
            all_candidates = list(candidates)
            identity_candidates = tuple(sorted(item[1] for item in all_candidates))
            converged_candidates = [item for item in candidates if item[2].converged and item[3].converged]
            if not converged_candidates and not pipeline.config.exhaustive_profile:
                pipeline.engine.retain_antecedent_windows(pipeline.splits.fit.name, antecedent, ())
                return [], identity_candidates, {
                    "antecedent": list(antecedent),
                    "antecedent_names": [pipeline.data.predicate_names[idx] for idx in antecedent],
                    "status": "no_converged_profile",
                    "candidates": candidate_log,
                }
            selection_pool = converged_candidates or all_candidates
            selection_pool.sort(key=lambda item: (item[0], item[1].window, -item[1].sign))
            _score, selected, selected_fit, selected_baseline = selection_pool[0]
            if pipeline.config.exhaustive_profile:
                # No standalone-effect screen is allowed here: an atom that is
                # weak alone can be necessary in a multi-rule support.  The
                # standalone fits above are diagnostics and warm-cache entries.
                selected_rules = [item[1] for item in all_candidates]
            else:
                selected_rules = [selected]
                # Profiling can touch many exact span breakpoints. Only the
                # frozen canonical response is needed by support search.
                pipeline.engine.retain_antecedent_windows(
                    pipeline.splits.fit.name,
                    antecedent,
                    (selected.window,),
                )
            # Candidate high-order identities can create many lower-order
            # closure responses at rejected W values. Their fitted parameters
            # are already cached; retain only closure terms needed by the
            # selected identity and reproduce any evicted shared term later.
            candidate_closure_terms = {
                term
                for _score_value, _rule_value, fit_value, _baseline_value in all_candidates
                for term in fit_value.closure_terms
            }
            selected_closure_terms = set(selected_fit.closure_terms)
            pipeline.engine.evict_context_terms(
                pipeline.splits.fit.name,
                tuple(candidate_closure_terms - selected_closure_terms),
            )
            return selected_rules, identity_candidates, {
                "antecedent": list(antecedent),
                "antecedent_names": [pipeline.data.predicate_names[idx] for idx in antecedent],
                "status": "profiled",
                "selected_window": int(selected.window),
                "selected_sign": int(selected.sign),
                "selected_nll": float(selected_fit.nll),
                "global_baseline_nll": float(baseline.nll),
                "closure_baseline_nll": float(selected_baseline.nll),
                "selected_improvement": float(selected_baseline.nll - selected_fit.nll),
                "candidate_count": len(candidate_log),
                "candidates": candidate_log,
            }

        # Share immutable event/response caches by default.  One-process-per-
        # antecedent duplicated allocator arenas and was observed to terminate
        # without a result under the full Freddie workload.  Native completion
        # and NumPy/MKL kernels release the GIL, so long-lived CPU thread
        # workers retain parallel compiled execution without that replication.
        # The process implementation remains an explicit exact fallback.
        fork_process_count = (
            int(self.config.solver_workers)
            if os.environ.get("CERTSCR_PROCESS_PROFILE", "0") == "1"
            and os.name == "posix"
            and not self.config.support_devices
            and str(self.config.solver_device).startswith("cpu")
            and int(self.config.solver_workers) > 1
            else 0
        )
        # Count every primitive column in one contiguous pass.  Scanning one
        # strided column per predicate reread the complete event matrix up to
        # 12--15 times merely to obtain an outcome-blind LPT scheduling key.
        primitive_event_counts = np.count_nonzero(
            self.data.predicates, axis=0
        )
        source_event_counts = {
            int(source_id): int(primitive_event_counts[int(source_id)])
            for source_id in self.rule_source_ids
            if int(source_id) < self.data.n_predicates
        }

        def structural_profile_load(antecedent: tuple[int, ...]) -> int:
            counts = [source_event_counts.get(int(source), 0) for source in antecedent]
            if not counts:
                return 0
            # The completion sweep reads the source streams once and emits at
            # most the smallest stream. Kernel construction then touches at
            # most ``impact_lag`` rows per completion. This outcome-free upper
            # workload orders tasks only; it never admits or removes a rule.
            return int(sum(counts) + self.config.impact_lag * min(counts))
        devices = self._support_worker_devices()
        results_by_antecedent: dict[
            tuple[int, ...],
            tuple[list[RuleIdentity], tuple[RuleIdentity, ...], dict],
        ] = {}
        process_fit_updates: dict[
            tuple[int, ...],
            tuple[
                dict[tuple[tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]], FitResult],
                dict[tuple[ClosureTerm, ...], FitResult],
            ],
        ] = {}
        workers: list[CertSCRPipeline] = []
        if not fork_process_count and len(devices) > 1:
            self._start_active_support_workers()
            workers = self._active_support_workers

        def process_batch(
            batch: list[tuple[int, ...]],
            *,
            evict_worker_fit_features_after_each: bool = False,
        ) -> None:
            if not batch:
                return

            # Longest-processing-time scheduling avoids leaving a few dense
            # triplets on the final workers after the others have gone idle.
            # It changes queue order only and preserves input-order result
            # merging below.
            batch = sorted(
                batch,
                key=lambda antecedent: (
                    -structural_profile_load(antecedent), antecedent
                ),
            )

            if fork_process_count:
                # CPU profile fits contain many short Python-controlled Newton
                # iterations. Threads serialize those GIL sections. POSIX fork
                # shares immutable data arrays copy-on-write while independent
                # processes evaluate the exact same complete skeleton family.
                context = mp.get_context("fork")
                result_queue = context.Queue()

                process_count = min(fork_process_count, len(batch))
                # ``feature_cache_bytes`` is a host-wide budget.  Forked
                # workers own private caches, so divide that budget exactly;
                # in particular, keep an explicitly disabled (zero-byte)
                # cache disabled.  Cache size changes recomputation only and
                # therefore cannot change the fitted objective or family.
                per_worker_feature_cache = (
                    int(self.config.feature_cache_bytes) // process_count
                )

                def process_worker(antecedent: tuple[int, ...]) -> None:
                    setter = _mkl_local_thread_setter()
                    if setter is not None:
                        setter(1)
                    worker = copy.copy(self)
                    worker.config = replace(
                        self.config,
                        solver_workers=1,
                        support_devices=(),
                        feature_cache_bytes=per_worker_feature_cache,
                    )
                    worker._active_support_workers = []
                    # Fork gives every process a private copy-on-write cache.
                    # Treat the configured byte cap as one host-wide budget,
                    # rather than allowing every worker to consume the full
                    # cap independently. Cache eviction changes recomputation
                    # only; response values and the finite family are exact.
                    worker.engine._feature_cache_limit = per_worker_feature_cache
                    try:
                        diagnostic_before = dict(worker._safe_screen_stats)
                        profile_started = time.perf_counter()
                        selected_rules, identities, log = profile_one(
                            worker, antecedent
                        )
                        log["profile_seconds"] = time.perf_counter() - profile_started
                        result = selected_rules, identities, log
                        if evict_worker_fit_features_after_each:
                            worker.engine.retain_antecedent_windows(
                                worker.splits.fit.name, antecedent, ()
                            )
                        # Publish only reportable fits.  The process exits after
                        # this antecedent, releasing every rejected-W design and
                        # convolution regardless of allocator behaviour.
                        candidate_fit_keys = [
                            ((rule,), worker.hierarchy_closure((rule,)))
                            for rule in selected_rules
                        ]
                        fit_updates = {
                            key: worker._fit_cache[key]
                            for key in candidate_fit_keys
                            if key in worker._fit_cache
                        }
                        candidate_null_keys = {
                            worker.hierarchy_closure((rule,))
                            for rule in identities
                        }
                        null_updates = {
                            key: worker._null_fit_cache[key]
                            for key in candidate_null_keys
                            if key in worker._null_fit_cache
                        }
                        diagnostic_updates = {
                            key: int(value) - int(diagnostic_before.get(key, 0))
                            for key, value in worker._safe_screen_stats.items()
                            if int(value) != int(diagnostic_before.get(key, 0))
                        }
                        result_queue.put(
                            (
                                True,
                                antecedent,
                                result,
                                fit_updates,
                                null_updates,
                                diagnostic_updates,
                                None,
                            )
                        )
                    except BaseException:
                        result_queue.put(
                            (
                                False,
                                antecedent,
                                None,
                                None,
                                None,
                                None,
                                traceback.format_exc(),
                            )
                        )

                # One antecedent per child is deliberate: long-lived workers
                # retained allocator arenas and response graphs across many
                # triplets, so 24 nominally bounded caches still exhausted the
                # 125-GiB host.  A rolling one-shot pool keeps identical LPT task
                # order and exact fits while making peak memory independent of
                # the number of completed skeletons.
                pending = iter(batch)
                active: dict[tuple[int, ...], mp.Process] = {}

                def launch_next() -> bool:
                    try:
                        antecedent = next(pending)
                    except StopIteration:
                        return False
                    process = context.Process(
                        target=process_worker, args=(antecedent,)
                    )
                    process.start()
                    active[antecedent] = process
                    return True

                for _ in range(process_count):
                    if not launch_next():
                        break
                received = 0
                while received < len(batch):
                    try:
                        (
                            ok,
                            antecedent,
                            result,
                            fit_updates,
                            null_updates,
                            diagnostic_updates,
                            error,
                        ) = result_queue.get(
                            timeout=5.0
                        )
                    except queue.Empty:
                        failed = [
                            (antecedent, process.exitcode)
                            for antecedent, process in active.items()
                            if not process.is_alive() and process.exitcode not in {0, None}
                        ]
                        if failed:
                            raise RuntimeError(
                                "forked identity worker exited before returning a result: "
                                + ", ".join(
                                    f"{antecedent} (exit {code})"
                                    for antecedent, code in failed
                                )
                            )
                        continue
                    received += 1
                    process = active.pop(antecedent, None)
                    if process is None:
                        raise RuntimeError(
                            f"identity worker returned an unknown task {antecedent}"
                        )
                    process.join()
                    if process.exitcode not in {0, None} and ok:
                        raise RuntimeError(
                            f"identity worker {antecedent} exited with code {process.exitcode}"
                        )
                    if ok:
                        assert result is not None
                        results_by_antecedent[antecedent] = result
                        process_fit_updates[antecedent] = (
                            fit_updates or {},
                            null_updates or {},
                        )
                        if diagnostic_updates:
                            with self._diagnostic_guard:
                                for key, delta in diagnostic_updates.items():
                                    self._safe_screen_stats[key] = int(
                                        self._safe_screen_stats.get(key, 0)
                                    ) + int(delta)
                    else:
                        # The batch result is unusable after one exact worker
                        # fails. Do not launch and wait for the remaining
                        # hundreds of skeletons only to raise the saved error
                        # at the end of the batch.
                        for outstanding in active.values():
                            if outstanding.is_alive():
                                outstanding.terminate()
                        for outstanding in active.values():
                            outstanding.join()
                        result_queue.close()
                        raise RuntimeError(
                            "forked identity profiling failed:\n"
                            f"antecedent={antecedent}:\n{error}"
                        )
                    launch_next()
                for process in active.values():
                    process.join()
                result_queue.close()
                # POSIX workers fitted the exact singleton and hierarchy-null
                # models used by the next search stage.  Preserve those small
                # immutable results instead of discarding them and immediately
                # solving the same cache keys again.  Merge in input order so
                # equal convex keys have deterministic ownership.
                for antecedent in batch:
                    fit_updates, null_updates = process_fit_updates.get(
                        antecedent, ({}, {})
                    )
                    for key, value in fit_updates.items():
                        if key not in self._fit_cache:
                            self._fit_cache[key] = value
                            with self._diagnostic_guard:
                                self._safe_screen_stats[
                                    "profile_fit_results_reused"
                                ] += 1
                    for key, value in null_updates.items():
                        self._null_fit_cache.setdefault(key, value)
                return

            def timed_profile_one(
                pipeline: CertSCRPipeline,
                antecedent: tuple[int, ...],
            ) -> tuple[list[RuleIdentity], tuple[RuleIdentity, ...], dict]:
                profile_started = time.perf_counter()
                selected_rules, identities, log = profile_one(pipeline, antecedent)
                log["profile_seconds"] = time.perf_counter() - profile_started
                return selected_rules, identities, log

            def profile_wave(wave: list[tuple[int, ...]]) -> None:
                if not workers:
                    for antecedent in wave:
                        results_by_antecedent[antecedent] = timed_profile_one(
                            self, antecedent
                        )
                    return
                fit_worker_count = self._exact_fit_worker_limit(len(workers))
                fit_workers = workers[:fit_worker_count]
                available: queue.SimpleQueue[CertSCRPipeline] = queue.SimpleQueue()
                for worker in fit_workers:
                    available.put(worker)

                def profile_task(
                    antecedent: tuple[int, ...],
                ) -> tuple[
                    tuple[int, ...],
                    tuple[list[RuleIdentity], tuple[RuleIdentity, ...], dict],
                ]:
                    worker = available.get()
                    try:
                        with _single_threaded_local_blas():
                            result = timed_profile_one(worker, antecedent)
                        if evict_worker_fit_features_after_each:
                            worker.engine.retain_antecedent_windows(
                                worker.splits.fit.name,
                                antecedent,
                                (),
                            )
                        return antecedent, result
                    finally:
                        available.put(worker)

                with ThreadPoolExecutor(max_workers=fit_worker_count) as fit_executor:
                    futures = [
                        fit_executor.submit(profile_task, antecedent)
                        for antecedent in wave
                    ]
                    for future in futures:
                        antecedent, result = future.result()
                        results_by_antecedent[antecedent] = result

            producer_count = min(
                len(batch),
                int(self.config.response_workers),
                max(1, int(os.cpu_count() or 1)),
            )

            def prefetch(antecedent: tuple[int, ...]) -> CompletionEvents:
                return self.engine.completions_for_context(
                    self.splits.fit, antecedent
                )

            wave_size = max(1, producer_count * 2)
            waves = [
                batch[left : left + wave_size]
                for left in range(0, len(batch), wave_size)
            ]
            with ThreadPoolExecutor(max_workers=producer_count) as response_executor:
                pending = [
                    response_executor.submit(prefetch, antecedent)
                    for antecedent in waves[0]
                ]
                for wave_index, wave in enumerate(waves):
                    for future in pending:
                        future.result()
                    pending = (
                        [
                            response_executor.submit(prefetch, antecedent)
                            for antecedent in waves[wave_index + 1]
                        ]
                        if wave_index + 1 < len(waves)
                        else []
                    )
                    profile_wave(wave)
            for worker in workers:
                self._fit_cache.update(worker._fit_cache)
                self._null_fit_cache.update(worker._null_fit_cache)
            for worker in workers:
                worker._fit_cache.update(self._fit_cache)
                worker._null_fit_cache.update(self._null_fit_cache)

        staged_triplets = bool(
            self.config.identity_profile in {"score_mdl", "dictionary_mdl"}
            and self.config.triplet_generation
            in {
                "weak_mdl_heredity",
                "connected_mdl_heredity",
                "strong_mdl_heredity",
            }
        )
        if staged_triplets:
            lower = [antecedent for antecedent in antecedents if len(antecedent) <= 2]
            triplets = [antecedent for antecedent in antecedents if len(antecedent) == 3]
            process_batch(lower)
            admitted_pairs = {
                antecedent
                for antecedent, result in results_by_antecedent.items()
                if len(antecedent) == 2 and bool(result[0])
            }
            required_pair_edges = {
                "weak_mdl_heredity": 1,
                "connected_mdl_heredity": 2,
                "strong_mdl_heredity": 3,
            }[self.config.triplet_generation]
            eligible_triplets = []
            for antecedent in triplets:
                admitted_edge_count = sum(
                    tuple(pair) in admitted_pairs
                    for pair in itertools.combinations(antecedent, 2)
                )
                if admitted_edge_count >= required_pair_edges:
                    eligible_triplets.append(antecedent)
            budget_excluded: set[tuple[int, ...]] = set()
            if (
                self.config.gradient_pricing_only
                and self.config.max_gradient_triplets is not None
                and len(eligible_triplets) > self.config.max_gradient_triplets
            ):
                def heredity_strength(antecedent: tuple[int, ...]) -> tuple[float, float]:
                    edge_scores = sorted(
                        (
                            float(
                                results_by_antecedent[tuple(pair)][2].get(
                                    "selected_score_block_mdl",
                                    results_by_antecedent[tuple(pair)][2].get(
                                        "selected_block_mdl", -math.inf
                                    ),
                                )
                            )
                            for pair in itertools.combinations(antecedent, 2)
                            if tuple(pair) in admitted_pairs
                        ),
                        reverse=True,
                    )
                    bottleneck = edge_scores[required_pair_edges - 1]
                    return bottleneck, float(sum(edge_scores))

                ranked_triplets = sorted(
                    eligible_triplets,
                    key=lambda item: (*(-value for value in heredity_strength(item)), item),
                )
                kept = ranked_triplets[: self.config.max_gradient_triplets]
                budget_excluded = set(eligible_triplets) - set(kept)
                eligible_triplets = kept
            eligible_set = set(eligible_triplets)
            for antecedent in triplets:
                if antecedent not in eligible_set:
                    status = (
                        "excluded_by_gradient_triplet_budget"
                        if antecedent in budget_excluded
                        else f"excluded_by_{self.config.triplet_generation}"
                    )
                    results_by_antecedent[antecedent] = (
                        [],
                        (),
                        {
                            "antecedent": list(antecedent),
                            "antecedent_names": [self.data.predicate_names[idx] for idx in antecedent],
                            "status": status,
                            "required_admitted_pair_edges": required_pair_edges,
                            "max_gradient_triplets": self.config.max_gradient_triplets,
                        },
                    )
            process_batch(
                eligible_triplets,
                evict_worker_fit_features_after_each=True,
            )
        else:
            # High-order completions and rejected W-specific responses are not
            # reusable once that triplet has been exactly profiled. Processing
            # lower orders first and evicting only the completed triplet's own
            # artifacts keeps the same exhaustive family while preventing 12
            # forked workers from retaining hundreds of disjoint triplet
            # response graphs until process exit.
            lower = [
                antecedent for antecedent in antecedents if len(antecedent) <= 2
            ]
            triplets = [
                antecedent for antecedent in antecedents if len(antecedent) == 3
            ]
            process_batch(lower)
            process_batch(
                triplets,
                evict_worker_fit_features_after_each=True,
            )

        for antecedent in antecedents:
            selected_rules, identities, log = results_by_antecedent[antecedent]
            candidate_rules.extend(selected_rules)
            if not selected_rules:
                self.engine.evict_context_completion(
                    self.splits.fit.name,
                    antecedent,
                )
            if selected_rules and log.get("selected_dictionary_shape") is not None:
                self.rule_dictionary_shapes[selected_rules[0]] = np.asarray(
                    log["selected_dictionary_shape"], dtype=np.float64
                )
            if self.config.identity_profile == "dictionary_mdl":
                for candidate in log.get("candidates", []):
                    shape_index = candidate.get("dictionary_shape_index")
                    if shape_index is None:
                        continue
                    identity = RuleIdentity(
                        antecedent=antecedent,
                        window=int(candidate["window"]),
                        sign=int(candidate["sign"]),
                    )
                    self.rule_dictionary_shapes[identity] = self.kernel_dictionary[
                        int(shape_index)
                    ].copy()
            if identities:
                self.identity_candidates[antecedent] = identities
            logs.append(log)
        self.profiled_rules = sorted(set(candidate_rules))
        self.profile_logs = logs
        self._profile_completed = True
        canonical_terms = {
            (rule.antecedent, int(rule.window)) for rule in self.profiled_rules
        }
        for rule in self.profiled_rules:
            canonical_terms.update(self.hierarchy_closure((rule,)))
        self.engine.retain_context_terms(self.splits.fit.name, tuple(canonical_terms))
        # Persist only the frozen discovery dictionary, never the much larger
        # pre-profile W/sign population.  Projected one-column rule atoms are
        # what support search consumes; unrestricted closure blocks retain M
        # columns.  The mmap store is an exact cache and may decline writes at
        # its byte cap without changing any subsequent response.
        if self.config.persistent_response_bytes > 0:
            for rule in self.profiled_rules:
                shape = self.rule_dictionary_shapes.get(rule)
                self.engine.persist_sparse_response(
                    self.splits.fit,
                    rule.antecedent,
                    int(rule.window),
                    shape=shape,
                )
            rule_terms = {
                (rule.antecedent, int(rule.window))
                for rule in self.profiled_rules
            }
            for antecedent, window in sorted(canonical_terms - rule_terms):
                self.engine.persist_sparse_response(
                    self.splits.fit,
                    antecedent,
                    int(window),
                )
        for worker in self._active_support_workers:
            worker.profiled_rules = list(self.profiled_rules)
            worker.identity_candidates = dict(self.identity_candidates)
            worker.rule_dictionary_shapes = {
                rule: shape.copy() for rule, shape in self.rule_dictionary_shapes.items()
            }
        return self.profiled_rules

    def _support_record(self, rules: Sequence[RuleIdentity], *, profile: str) -> SupportRecord:
        rules = tuple(sorted(rules))
        fit = self.fit_support(rules)
        closure_baseline = self.fit_model((), fit.closure_terms)
        return SupportRecord(
            rules=rules,
            fit=fit,
            closure_baseline_fit=closure_baseline,
            search_nll_improvement=float(closure_baseline.nll - fit.nll),
            profile=profile,
        )

    def _support_complexity_penalty(
        self,
        rules: Sequence[RuleIdentity],
        fitted_dimension: int,
    ) -> float:
        """Return the rule-block and finite-identity MDL complexity.

        The W/sign identity is selected in every profiling mode and therefore
        remains part of the code length after a dictionary atom is expanded to
        a full M-knot block.  Only the temporary dictionary-shape index ceases
        to be a parameter of the final full-cone model.
        """
        rules = tuple(sorted(rules))
        rule_dimension = len(rules) * (int(fitted_dimension) + int(self.marked))
        penalty = rule_dimension * math.log(
            max(2, self.splits.fit_population_sequence_count)
        )
        if int(fitted_dimension) == 1 and self.config.identity_profile == "dictionary_mdl":
            penalty += 2.0 * len(rules) * math.log(len(self.kernel_dictionary))
        penalty += 2.0 * sum(
            math.log(
                max(
                    1,
                    len(self.identity_candidates.get(rule.antecedent, (rule,))),
                )
            )
            for rule in rules
        )
        return float(penalty)

    def _support_score_upper_bound(self, rules: Sequence[RuleIdentity]) -> dict:
        """Safe global block-MDL upper bound before an exact support fit.

        The occurrence lower bound independently saturates every distinct row
        of the complete augmented design.  The marked lower bound gives every
        candidate knot its own unrestricted amount coefficient.  Both are
        relaxations of the fitted model, so subtracting them from the closure
        null yields an upper—not approximate—gain bound.
        """
        rules = tuple(sorted(rules))
        cached = self._safe_bound_cache.get(rules)
        if cached is not None:
            return cached
        with self._diagnostic_guard:
            # Count fail-open/nonfinite attempts too.  The previous increment
            # at the final return made runtime diagnostics under-report work.
            self._safe_screen_stats["bound_evaluations"] += 1
        closure_terms = self.hierarchy_closure(rules)
        closure_baseline = self.fit_model((), closure_terms)
        features = self.sparse_features(self.splits.fit, rules)
        dimensions = {int(feature.shape[1]) for feature in features}
        if len(dimensions) != 1 or not closure_baseline.converged:
            result = {
                "finite": False,
                "score_upper_bound": math.inf,
                "reason": "nonconverged_closure_or_mixed_kernel_dimension",
            }
            self._safe_bound_cache[rules] = result
            return result
        fitted_dimension = dimensions.pop()
        nuisance = self.sparse_nuisance_blocks(
            self.splits.fit, closure_terms
        )
        prepared_key = (rules, closure_terms)
        # Identity profiling has already assembled this exact child from the
        # live full-M W parent.  Rebuilding it here made the safe bound more
        # expensive than the fit it was intended to avoid, so the full runner
        # had to disable an otherwise exact screen.  Ownership is transferred
        # out of the stage cache and returned below only when a fit is still
        # required; no objective row or grouping is changed.
        prepared = self._prepared_design_cache.pop(prepared_key, None)
        if prepared is None:
            prepared = self.prepare_partitioned_support_design(
                self.splits.fit,
                closure_terms,
                features,
                rules,
                cluster_weights=self.fit_cluster_weights,
            )
        occurrence_bound = group_saturated_poisson_lower_bound(
            self.splits.fit,
            nuisance,
            features,
            rules,
            cluster_weights=self.fit_cluster_weights,
            sequence_exposures=self.sequence_exposures(self.splits.fit),
            prepared_design=prepared,
            occurrence_likelihood=self.occurrence_likelihood,
        )
        null_occurrence_nll = float(
            closure_baseline.nll
            if closure_baseline.intensity_nll is None
            else closure_baseline.intensity_nll
        )
        if not occurrence_bound.finite:
            result = {
                "finite": False,
                "score_upper_bound": math.inf,
                "reason": "nonfinite_occurrence_lower_bound",
                "active_grid_rows": occurrence_bound.active_grid_rows,
            }
            self._prepared_design_cache[prepared_key] = prepared
            self._safe_bound_cache[rules] = result
            return result
        # The null itself is feasible.  Min with its objective protects the
        # relaxation inequality against harmless floating summation noise.
        occurrence_lower_nll = min(
            null_occurrence_nll,
            float(occurrence_bound.lower_bound),
        )
        occurrence_gain_upper = max(0.0, null_occurrence_nll - occurrence_lower_nll)

        mark_gain_upper = 0.0
        mark_lower_nll: float | None = None
        if self.marked:
            if (
                closure_baseline.mark_fit is None
                or self.mark_unit is None
                or self.mark_variance is None
            ):
                result = {
                    "finite": False,
                    "score_upper_bound": math.inf,
                    "reason": "marked_closure_not_initialized",
                }
                self._prepared_design_cache[prepared_key] = prepared
                self._safe_bound_cache[rules] = result
                return result
            # Independent coefficients for every knot strictly contain the
            # actual one-shared-shape coefficient per rule.
            relaxed_activations = [
                feature.event_values[:, column]
                for feature in features
                for column in range(feature.shape[1])
            ]
            relaxed_mark = fit_mark_head(
                self.splits.fit,
                self.nuisance_event_design(self.splits.fit, closure_terms),
                relaxed_activations,
                unit=self.mark_unit,
                variance=self.mark_variance,
                cluster_weights=self.fit_cluster_weights,
                base_residualizer=self.mark_base_residualizer(
                    self.splits.fit,
                    closure_terms,
                ),
            )
            if not relaxed_mark.converged:
                result = {
                    "finite": False,
                    "score_upper_bound": math.inf,
                    "reason": "nonconverged_relaxed_mark_bound",
                }
                self._prepared_design_cache[prepared_key] = prepared
                self._safe_bound_cache[rules] = result
                return result
            null_mark_nll = float(closure_baseline.mark_fit.nll)
            mark_lower_nll = min(null_mark_nll, float(relaxed_mark.nll))
            mark_gain_upper = max(0.0, null_mark_nll - mark_lower_nll)

        gain_upper = occurrence_gain_upper + mark_gain_upper
        penalty = self._support_complexity_penalty(rules, fitted_dimension)
        score_upper = float(
            2.0 * gain_upper * self.fit_objective_population_scale - penalty
        )
        # Only a strictly negative bound is screened.  The margin is scaled to
        # all floating terms so a roundoff-sized negative value never removes a
        # boundary candidate.
        numeric_margin = 128.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(score_upper),
            abs(penalty),
            abs(2.0 * gain_upper * self.fit_objective_population_scale),
        )
        result = {
            "finite": math.isfinite(score_upper),
            "score_upper_bound": score_upper,
            "screenable": bool(math.isfinite(score_upper) and score_upper < -numeric_margin),
            "numeric_margin": float(numeric_margin),
            "gain_upper_bound": float(gain_upper),
            "occurrence_gain_upper_bound": float(occurrence_gain_upper),
            "mark_gain_upper_bound": float(mark_gain_upper),
            "occurrence_lower_nll": float(occurrence_lower_nll),
            "mark_lower_nll": mark_lower_nll,
            "penalty": float(penalty),
            "saturated_group_count": int(occurrence_bound.group_count),
            "active_grid_rows": int(occurrence_bound.active_grid_rows),
            "reason": "group_saturated_global_likelihood_bound",
            "fitted_dimension": int(fitted_dimension),
        }
        if not result["screenable"]:
            # Safe screening and the exact solver use the same complete
            # augmented row partition. Reuse it once; fit_model pops the entry
            # so large designs cannot accumulate across the search.
            self._prepared_design_cache[prepared_key] = prepared
        else:
            self._prepared_profile_nonoptimal_keys.discard(prepared_key)
        self._safe_bound_cache[rules] = result
        return result

    def _safe_screened_support_record(
        self,
        rules: Sequence[RuleIdentity],
        *,
        profile: str,
    ) -> SupportRecord | None:
        """Return a fail-closed record only when positive MDL is impossible."""
        rules = tuple(sorted(rules))
        if not self.config.safe_mdl_screen:
            return None
        existing = self._fit_cache.get((rules, self.hierarchy_closure(rules)))
        if existing is not None:
            return None
        cached = self._safe_screened_records.get(rules)
        if cached is not None:
            return cached
        bound = self._support_score_upper_bound(rules)
        if not bound.get("screenable", False):
            return None
        closure_terms = self.hierarchy_closure(rules)
        closure_baseline = self.fit_model((), closure_terms)
        fitted_dimension = int(bound["fitted_dimension"])
        screened_fit = replace(
            closure_baseline,
            rules=rules,
            theta=np.zeros((len(rules), fitted_dimension), dtype=np.float64),
            converged=False,
        )
        record = SupportRecord(
            rules=rules,
            fit=screened_fit,
            closure_baseline_fit=closure_baseline,
            search_nll_improvement=0.0,
            profile=f"{profile}-safe-nonpositive-mdl-screen",
        )
        self._safe_screened_records[rules] = record
        with self._diagnostic_guard:
            self._safe_screen_stats["screened_supports"] += 1
        return record

    def _fit_or_safe_screen_records_batch(
        self,
        rule_sets: Sequence[Sequence[RuleIdentity]],
        *,
        profile: str,
    ) -> list[SupportRecord]:
        """Screen provably nonpositive supports, exact-fit every other one."""
        keys = list(dict.fromkeys(tuple(sorted(rules)) for rules in rule_sets))
        if not self.config.safe_mdl_screen:
            # With screening disabled there is no bound state to synchronize
            # between waves.  The old 2*worker wave loop rebuilt a thread pool
            # and copied every fit cache roughly O(n_supports/workers) times.
            # One balanced batch evaluates the identical key set with the same
            # exact solver and KKT tolerance; only executor scheduling changes.
            fitted = self._fit_support_records_batch(keys, profile=profile)
            with self._diagnostic_guard:
                self._safe_screen_stats["exact_fit_supports"] += len(fitted)
            return fitted
        records: dict[tuple[RuleIdentity, ...], SupportRecord] = {}
        worker_count = max(1, len(self._active_support_workers))
        wave_size = 2 * worker_count
        for left in range(0, len(keys), wave_size):
            wave = keys[left : left + wave_size]
            if len(self._active_support_workers) > 1:
                workers = self._active_support_workers
                for worker in workers:
                    if worker._fit_cache is not self._fit_cache:
                        worker._fit_cache.update(self._fit_cache)
                        worker._null_fit_cache.update(self._null_fit_cache)
                        worker._safe_bound_cache.update(self._safe_bound_cache)
                        worker._safe_screened_records.update(
                            self._safe_screened_records
                        )
                    worker._nuisance_event_design_cache.update(
                        self._nuisance_event_design_cache
                    )
                    worker._mark_base_residualizer_cache.update(
                        self._mark_base_residualizer_cache
                    )
                fit_worker_count = self._exact_fit_worker_limit(len(workers))
                fit_workers = workers[:fit_worker_count]
                chunks = [
                    wave[index::fit_worker_count]
                    for index in range(fit_worker_count)
                ]
                for worker, chunk in zip(fit_workers, chunks, strict=True):
                    for rules in chunk:
                        model_key = (rules, self.hierarchy_closure(rules))
                        prepared = self._prepared_design_cache.pop(model_key, None)
                        if prepared is not None:
                            worker._prepared_design_cache[model_key] = prepared

                def screen_and_fit_chunk(
                    worker: CertSCRPipeline,
                    chunk: list[tuple[RuleIdentity, ...]],
                ) -> tuple[list[SupportRecord], int]:
                    with _single_threaded_local_blas():
                        output: list[SupportRecord] = []
                        exact_count = 0
                        for key in chunk:
                            screened = worker._safe_screened_support_record(
                                key,
                                profile=profile,
                            )
                            if screened is not None:
                                output.append(screened)
                                continue
                            output.append(
                                worker._support_record(key, profile=profile)
                            )
                            exact_count += 1
                        return output, exact_count

                exact_count = 0
                with ThreadPoolExecutor(max_workers=fit_worker_count) as executor:
                    futures = [
                        executor.submit(screen_and_fit_chunk, worker, chunk)
                        for worker, chunk in zip(fit_workers, chunks, strict=True)
                        if chunk
                    ]
                    for future in futures:
                        chunk_records, chunk_exact = future.result()
                        exact_count += chunk_exact
                        for record in chunk_records:
                            records[record.rules] = record
                with self._diagnostic_guard:
                    self._safe_screen_stats["exact_fit_supports"] += exact_count
                for worker in workers:
                    if worker._safe_bound_cache is not self._safe_bound_cache:
                        self._safe_bound_cache.update(worker._safe_bound_cache)
                        self._safe_screened_records.update(
                            worker._safe_screened_records
                        )
                    # Screened pseudo-fits are not inserted into a worker's
                    # fit cache; publishing the actual worker cache therefore
                    # transfers only optimized models and hierarchy nulls.
                    if worker._fit_cache is not self._fit_cache:
                        self._fit_cache.update(worker._fit_cache)
                        self._null_fit_cache.update(worker._null_fit_cache)
                for worker in workers:
                    if worker._fit_cache is not self._fit_cache:
                        worker._fit_cache.update(self._fit_cache)
                        worker._null_fit_cache.update(self._null_fit_cache)
                continue
            to_fit: list[tuple[RuleIdentity, ...]] = []
            for key in wave:
                screened = self._safe_screened_support_record(key, profile=profile)
                if screened is None:
                    to_fit.append(key)
                else:
                    records[key] = screened
            fitted = self._fit_support_records_batch(to_fit, profile=profile)
            with self._diagnostic_guard:
                self._safe_screen_stats["exact_fit_supports"] += len(fitted)
            records.update({record.rules: record for record in fitted})
        return [records[key] for key in keys]

    def _support_search_score(self, record: SupportRecord) -> float:
        """Conditional BIC evidence of the rule blocks over their fixed closure.

        The closure null is support-specific by construction.  Its nuisance fit
        is therefore subtracted before the rule-block complexity penalty is
        applied; hierarchy terms cannot masquerade as discovered-rule impact.
        """
        if not record.fit.converged or not record.closure_baseline_fit.converged:
            return -math.inf
        if not np.all(self._active_amplitudes(record.fit)):
            return -math.inf
        fitted_dimension = int(record.fit.theta.shape[1]) if record.fit.theta.ndim == 2 and record.fit.theta.size else 0
        improvement_scale = self.fit_objective_population_scale
        penalty = self._support_complexity_penalty(record.rules, fitted_dimension)
        return float(
            2.0 * record.search_nll_improvement * improvement_scale
            - penalty
        )

    def _eligible_support(self, rules: Sequence[RuleIdentity]) -> bool:
        rules = tuple(rules)
        return (
            0 < len(rules) <= self._effective_max_support_size()
            and len({rule.antecedent for rule in rules}) == len(rules)
        )

    def _one_exchange_neighbors(
        self,
        rules: Sequence[RuleIdentity],
    ) -> list[tuple[RuleIdentity, ...]]:
        """The exact finite add/drop/swap neighborhood used by the certificate."""
        current = tuple(sorted(rules))
        current_set = set(current)
        current_antecedents = {rule.antecedent for rule in current}
        neighbors: set[tuple[RuleIdentity, ...]] = set()
        if len(current) < self._effective_max_support_size():
            for rule in self.profiled_rules:
                if rule in current_set or rule.antecedent in current_antecedents:
                    continue
                neighbors.add(tuple(sorted((*current, rule))))
        if current:
            for drop_index in range(len(current)):
                dropped = current[:drop_index] + current[drop_index + 1 :]
                neighbors.add(dropped)
                dropped_antecedents = {
                    rule.antecedent for rule in dropped
                }
                for rule in self.profiled_rules:
                    if rule.antecedent in dropped_antecedents:
                        continue
                    trial = tuple(sorted((*dropped, rule)))
                    if trial != current:
                        neighbors.add(trial)
        return sorted(neighbors)

    def _add_drop_neighbors(
        self,
        rules: Sequence[RuleIdentity],
    ) -> list[tuple[RuleIdentity, ...]]:
        """Complete one-coordinate neighborhood for column generation.

        A coordinate is either one admitted rule block entering at zero or one
        active rule block being removed.  No gradient budget is used: every
        feasible coordinate is present in the terminal exact audit.
        """
        current = tuple(sorted(rules))
        current_set = set(current)
        current_antecedents = {rule.antecedent for rule in current}
        neighbors: set[tuple[RuleIdentity, ...]] = set()
        if len(current) < self._effective_max_support_size():
            for rule in self.profiled_rules:
                if rule in current_set or rule.antecedent in current_antecedents:
                    continue
                neighbors.add(tuple(sorted((*current, rule))))
        for index in range(len(current)):
            neighbors.add(current[:index] + current[index + 1 :])
        return sorted(neighbors)

    def _support_rule_gradient_prices(
        self,
        fit: FitResult,
        rules: Sequence[RuleIdentity],
        *,
        worker_limit: int | None = None,
    ) -> dict[RuleIdentity, tuple[float, float]]:
        """Price absent rule blocks at one fitted support in a fused row pass.

        The quadratic gain orders candidates only; exact support likelihood
        and block-MDL decide every accepted move.  The normalized projected
        KKT value is also returned for diagnostics.  Computing the fitted
        predictor once on the union of all candidate rows avoids one complete
        nuisance/support traversal per rule without changing a response or
        derivative.
        """
        candidates = tuple(sorted(set(rules)))
        if not candidates:
            return {}
        responses = tuple(
            self.sparse_features(self.splits.fit, (rule,))[0]
            for rule in candidates
        )
        ctx = self.splits.fit
        event_rows = (
            np.asarray(ctx.event_grid_rows, dtype=np.int64)
            if self.occurrence_likelihood == "first_event_cloglog"
            and ctx.n_events
            else np.zeros(0, dtype=np.int64)
        )
        effective_rows: list[np.ndarray] = []
        effective_values: list[np.ndarray] = []
        for response in responses:
            rows = response.grid_indices
            values = response.grid_values
            if len(rows) and len(event_rows):
                positions = np.searchsorted(event_rows, rows)
                safe = np.minimum(positions, len(event_rows) - 1)
                keep = (positions >= len(event_rows)) | (
                    event_rows[safe] != rows
                )
                rows = rows[keep]
                values = values[keep]
            effective_rows.append(rows)
            effective_values.append(values)
        row_parts = [rows for rows in effective_rows if len(rows)]
        if not row_parts:
            union_rows = np.zeros(0, dtype=np.int64)
        elif len(row_parts) == 1:
            union_rows = row_parts[0].copy()
        else:
            union_rows = sorted_unique_int64_union(
                row_parts,
                allow_wide=True,
            )
            if union_rows is None:
                union_rows = np.unique(np.concatenate(row_parts)).astype(
                    np.int64, copy=False
                )
        union_eta = self._eta_on_sparse_grid(fit, ctx, union_rows)
        sequence_weights = self.fit_cluster_weights
        with np.errstate(over="ignore", invalid="ignore"):
            union_mu = (
                ctx.grid_weights_at(union_rows, assume_valid=True)
                * sequence_weights[
                    ctx.grid_sequences_at(
                        union_rows,
                        assume_valid=True,
                        assume_sorted=True,
                    )
                ]
                * np.exp(union_eta)
            )
        if np.any(~np.isfinite(union_mu)):
            raise FloatingPointError(
                "nonfinite fitted intensity during support pricing"
            )
        shared_event_eta = (
            self._eta_on_events(fit, ctx)
            if self.occurrence_likelihood == "first_event_cloglog"
            and ctx.n_events
            else None
        )
        event_weights = sequence_weights[ctx.event_sequence_local]
        if ctx.n_events:
            if self.occurrence_likelihood == "poisson":
                shared_event_gradient_weight = -event_weights
                shared_event_information_weight = np.zeros(
                    ctx.n_events, dtype=np.float64
                )
            else:
                assert shared_event_eta is not None
                (
                    _event_loss,
                    event_gradient,
                    event_hessian,
                ) = cloglog_event_terms(shared_event_eta)
                shared_event_gradient_weight = (
                    event_weights * event_gradient
                )
                shared_event_information_weight = (
                    event_weights * event_hessian
                )
        else:
            shared_event_gradient_weight = np.zeros(0, dtype=np.float64)
            shared_event_information_weight = np.zeros(0, dtype=np.float64)
        def price_one(
            item: tuple[
                RuleIdentity,
                SparseKernelResponse,
                np.ndarray,
                np.ndarray,
            ],
        ) -> tuple[RuleIdentity, tuple[float, float]]:
            rule, response, rows, grid_block = item
            if len(rows):
                positions = np.searchsorted(union_rows, rows)
                if (
                    np.any(positions >= len(union_rows))
                    or not np.array_equal(union_rows[positions], rows)
                ):
                    raise AssertionError(
                        "pricing rows are missing from the shared union"
                    )
                row_mu = union_mu[positions]
            else:
                row_mu = np.zeros(0, dtype=np.float64)
            block = grid_block.astype(np.float64, copy=False)
            gradient = block.T @ row_mu
            information = block.T @ (row_mu[:, None] * block)
            if ctx.n_events:
                event_block = response.event_values.astype(
                    np.float64, copy=False
                )
                gradient += (
                    event_block.T @ shared_event_gradient_weight
                )
                if self.occurrence_likelihood == "first_event_cloglog":
                    information += event_block.T @ (
                        shared_event_information_weight[:, None]
                        * event_block
                    )
            information = 0.5 * (
                information + information.T
            )
            signed_gradient = float(rule.sign) * gradient
            gain = self._cone_quadratic_gain(signed_gradient, information)
            diagonal = np.diag(information)
            kkt = float(
                np.max(
                    np.maximum(-signed_gradient, 0.0)
                    / np.sqrt(
                        np.maximum(diagonal, np.finfo(np.float64).tiny)
                    ),
                    initial=0.0,
                )
            )
            return rule, (float(gain), kkt)

        items = tuple(
            zip(
                candidates,
                responses,
                effective_rows,
                effective_values,
                strict=True,
            )
        )
        worker_count = min(
            len(items),
            max(
                1,
                int(self.config.response_workers)
                if worker_limit is None
                else int(worker_limit),
            ),
            max(1, int(os.cpu_count() or 1)),
        )
        if worker_count <= 1:
            priced = [price_one(item) for item in items]
        else:
            # Candidate blocks are independent read-only reductions over the
            # shared fitted intensity. NumPy releases the GIL in the matrix
            # products, so threads occupy otherwise idle cores without
            # changing accumulation order inside any individual rule.
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                priced = list(executor.map(price_one, items))
        return dict(priced)

    def _support_rule_gradient_prices_batch(
        self,
        items: Sequence[
            tuple[FitResult, Sequence[RuleIdentity]]
        ],
    ) -> list[dict[RuleIdentity, tuple[float, float]]]:
        """Price several support states in one exact response traversal.

        All states use the same frozen D_fit rows and rule dictionary.  The
        scalar implementation rebuilt the union row set and reread every rule
        block once per atom start.  Here predictors form columns of one small
        matrix and each sparse rule block multiplies all state residuals at
        once.  Each state's gradient and Fisher matrix are algebraically the
        same reductions as :meth:`_support_rule_gradient_prices`; only their
        scheduling is fused.  Callers bound the state batch, so peak memory is
        no larger than the former concurrent scalar tasks.
        """
        normalized = [
            (fit, tuple(sorted(set(rules)))) for fit, rules in items
        ]
        if not normalized:
            return []
        if len(normalized) == 1:
            fit, rules = normalized[0]
            return [
                self._support_rule_gradient_prices(
                    fit, rules, worker_limit=1
                )
            ]
        all_rules = tuple(
            sorted(
                {
                    rule
                    for _fit, rules in normalized
                    for rule in rules
                }
            )
        )
        if not all_rules:
            return [{} for _fit, _rules in normalized]
        response_by_rule = {
            rule: self.sparse_features(self.splits.fit, (rule,))[0]
            for rule in all_rules
        }
        ctx = self.splits.fit
        event_rows = (
            np.asarray(ctx.event_grid_rows, dtype=np.int64)
            if self.occurrence_likelihood == "first_event_cloglog"
            and ctx.n_events
            else np.zeros(0, dtype=np.int64)
        )
        effective: dict[RuleIdentity, tuple[np.ndarray, np.ndarray]] = {}
        row_parts: list[np.ndarray] = []
        for rule in all_rules:
            response = response_by_rule[rule]
            rows = response.grid_indices
            values = response.grid_values
            if len(rows) and len(event_rows):
                positions = np.searchsorted(event_rows, rows)
                safe = np.minimum(positions, len(event_rows) - 1)
                keep = (positions >= len(event_rows)) | (
                    event_rows[safe] != rows
                )
                rows = rows[keep]
                values = values[keep]
            effective[rule] = (rows, values)
            if len(rows):
                row_parts.append(rows)
        cached_union = self._support_pricing_union_cache
        if cached_union is not None and cached_union[0] == all_rules:
            union_rows = cached_union[1]
        else:
            if not row_parts:
                union_rows = np.zeros(0, dtype=np.int64)
            elif len(row_parts) == 1:
                union_rows = row_parts[0].copy()
            else:
                union_rows = sorted_unique_int64_union(
                    row_parts, allow_wide=True
                )
                if union_rows is None:
                    union_rows = np.unique(np.concatenate(row_parts)).astype(
                        np.int64, copy=False
                    )
            self._support_pricing_union_cache = (
                all_rules,
                np.ascontiguousarray(union_rows, dtype=np.int64),
            )
            union_rows = self._support_pricing_union_cache[1]
        state_count = len(normalized)
        union_mu = np.empty(
            (state_count, len(union_rows)), dtype=np.float64
        )
        sequence_weights = self.fit_cluster_weights
        exposure = ctx.grid_weights_at(union_rows, assume_valid=True)
        exposure = exposure * sequence_weights[
            ctx.grid_sequences_at(
                union_rows,
                assume_valid=True,
                assume_sorted=True,
            )
        ]
        def populate_union_mu(
            item: tuple[int, tuple[FitResult, tuple[RuleIdentity, ...]]],
        ) -> None:
            state, (fit, _rules) = item
            eta = self._eta_on_sparse_grid(fit, ctx, union_rows)
            with np.errstate(over="ignore", invalid="ignore"):
                # Reuse the state-local eta buffer.  ``np.exp(eta)`` followed by
                # multiplication formerly allocated two additional union-sized
                # float64 arrays per pricing thread.  The operation order and
                # dtype are unchanged; only temporary ownership differs.
                np.exp(eta, out=eta)
                np.multiply(eta, exposure, out=eta)
            union_mu[state] = eta

        eta_workers = min(
            state_count,
            max(1, int(self.config.response_workers)),
            max(1, int(os.cpu_count() or 1)),
        )
        indexed_states = tuple(enumerate(normalized))
        if eta_workers <= 1:
            for item in indexed_states:
                populate_union_mu(item)
        else:
            # Every column is a frozen, independent fitted state. Threads write
            # disjoint rows of the state-major output and each predictor keeps
            # its original sparse-row accumulation order.
            with ThreadPoolExecutor(max_workers=eta_workers) as executor:
                list(executor.map(populate_union_mu, indexed_states))
        if np.any(~np.isfinite(union_mu)):
            raise FloatingPointError(
                "nonfinite fitted intensity during batched support pricing"
            )

        event_weights = sequence_weights[ctx.event_sequence_local]
        event_first = np.zeros(
            (state_count, ctx.n_events), dtype=np.float64
        )
        event_second = np.zeros_like(event_first)
        if ctx.n_events:
            if self.occurrence_likelihood == "poisson":
                event_first[:] = -event_weights[None, :]
            else:
                def populate_event_terms(
                    item: tuple[
                        int,
                        tuple[FitResult, tuple[RuleIdentity, ...]],
                    ],
                ) -> None:
                    state, (fit, _rules) = item
                    eta = self._eta_on_events(fit, ctx)
                    _loss, first, second = cloglog_event_terms(eta)
                    event_first[state] = event_weights * first
                    event_second[state] = event_weights * second

                if eta_workers <= 1:
                    for item in indexed_states:
                        populate_event_terms(item)
                else:
                    with ThreadPoolExecutor(
                        max_workers=eta_workers
                    ) as executor:
                        list(
                            executor.map(
                                populate_event_terms, indexed_states
                            )
                        )

        requested = [set(rules) for _fit, rules in normalized]
        output: list[dict[RuleIdentity, tuple[float, float]]] = [
            {} for _ in normalized
        ]
        for rule in all_rules:
            response = response_by_rule[rule]
            rows, raw_block = effective[rule]
            if len(rows):
                positions = np.searchsorted(union_rows, rows)
                if (
                    np.any(positions >= len(union_rows))
                    or not np.array_equal(union_rows[positions], rows)
                ):
                    raise AssertionError(
                        "batched pricing rows are missing from their union"
                    )
            else:
                positions = np.zeros(0, dtype=np.int64)
            width = int(response.shape[1])
            native_moments = batched_sparse_rule_moments(
                raw_block,
                union_mu,
                response.event_values,
                event_first,
                event_second,
                include_event_second=(
                    ctx.n_events > 0
                    and self.occurrence_likelihood
                    == "first_event_cloglog"
                ),
                worker_count=min(
                    state_count,
                    max(1, int(self.config.response_workers)),
                ),
                grid_weight_positions=(
                    positions if len(rows) else np.zeros(0, dtype=np.int64)
                ),
            )
            if native_moments is not None:
                gradient, information_by_state = native_moments
            else:
                row_mu = np.take(union_mu, positions, axis=1).T
                block = raw_block.astype(np.float64, copy=False)
                if len(rows):
                    gradient = block.T @ row_mu
                else:
                    gradient = np.zeros(
                        (width, state_count), dtype=np.float64
                    )
                event_block = response.event_values.astype(
                    np.float64, copy=False
                )
                if ctx.n_events:
                    gradient += event_block.T @ event_first.T
                information_by_state = np.zeros(
                    (state_count, width, width), dtype=np.float64
                )
                for left in range(width):
                    for right in range(left + 1):
                        values = (
                            (block[:, left] * block[:, right]) @ row_mu
                            if len(rows)
                            else np.zeros(state_count, dtype=np.float64)
                        )
                        if (
                            ctx.n_events
                            and self.occurrence_likelihood
                            == "first_event_cloglog"
                        ):
                            values = values + (
                                event_block[:, left]
                                * event_block[:, right]
                            ) @ event_second.T
                        information_by_state[:, left, right] = values
                        information_by_state[:, right, left] = values
            for state in range(state_count):
                if rule not in requested[state]:
                    continue
                information = information_by_state[state]
                signed_gradient = float(rule.sign) * gradient[:, state]
                gain = self._cone_quadratic_gain(
                    signed_gradient, information
                )
                diagonal = np.diag(information)
                kkt = float(
                    np.max(
                        np.maximum(-signed_gradient, 0.0)
                        / np.sqrt(
                            np.maximum(
                                diagonal,
                                np.finfo(np.float64).tiny,
                            )
                        ),
                        initial=0.0,
                    )
                )
                output[state][rule] = (float(gain), kkt)
        return output

    def _search_supports_active_set(self) -> list[SupportRecord]:
        score_working_set = (
            self.config.active_neighbor_strategy == "mdl_score_working_set"
        )
        gradient_strategy = self.config.active_neighbor_strategy in {
            "gradient_first_exact_audit",
            "mdl_score_working_set",
        }
        if score_working_set:
            profile = "multi-start-mdl-block-score-working-set"
        elif gradient_strategy:
            profile = (
                "multi-start-gradient-first-exact-one-exchange-column-generation"
            )
        else:
            profile = "multi-start-exact-one-exchange-active-set"
        record_cache: dict[tuple[RuleIdentity, ...], SupportRecord] = {}

        def record(rules: Sequence[RuleIdentity]) -> SupportRecord:
            key = tuple(sorted(rules))
            cached = record_cache.get(key)
            if cached is None:
                cached = self._support_record(key, profile=profile)
                record_cache[key] = cached
            return cached

        def records(rule_sets: Sequence[Sequence[RuleIdentity]]) -> list[SupportRecord]:
            keys = list(dict.fromkeys(tuple(sorted(rules)) for rules in rule_sets))
            missing = [key for key in keys if key not in record_cache]
            for item in self._fit_or_safe_screen_records_batch(missing, profile=profile):
                record_cache[item.rules] = item
            return [record_cache[key] for key in keys]

        singleton_records = records([(rule,) for rule in self.profiled_rules])
        strata: dict[tuple[int, int], list[SupportRecord]] = {}
        for item in singleton_records:
            rule = item.rules[0]
            strata.setdefault((len(rule.antecedent), int(rule.sign)), []).append(item)
        for values in strata.values():
            values.sort(key=lambda item: (-self._support_search_score(item), item.rules))
        positive_singleton_records = [
            item
            for item in singleton_records
            if self._support_search_score(item) > 0.0
        ]
        search_fitted_dimensions = {
            int(item.fit.theta.shape[1]) for item in singleton_records
        }
        if len(search_fitted_dimensions) > 1:
            raise ValueError(
                "active support library mixes fitted kernel dimensions"
            )
        search_fitted_dimension = (
            search_fitted_dimensions.pop()
            if search_fitted_dimensions
            else int(self.config.knot_count)
        )
        starts: list[tuple[RuleIdentity, ...]] = [()]
        if self.config.active_start_policy == "all_atoms":
            # Every admitted atom is a deterministic basin witness.  This has
            # no restart budget or random seed and, unlike a top-k start rule,
            # cannot erase a pair/triplet merely because another atom has a
            # larger standalone D_fit score.
            starts.extend(item.rules for item in positive_singleton_records)
        else:
            # Legacy runtime ablation: deterministic round-robin makes
            # excitation/inhibition and every rule order reachable only up to
            # the explicitly configured restart budget.
            depth = 0
            ordered_strata = sorted(strata)
            while len(starts) < self.config.active_restarts + 1:
                added = False
                for key in ordered_strata:
                    values = strata[key]
                    if depth < len(values):
                        starts.append(values[depth].rules)
                        added = True
                        if len(starts) >= self.config.active_restarts + 1:
                            break
                if not added:
                    break
                depth += 1

        visited: set[tuple[RuleIdentity, ...]] = set()
        terminals: set[tuple[RuleIdentity, ...]] = set()
        runs: list[dict] = []
        transition_cache: dict[tuple[RuleIdentity, ...], dict] = {}
        gradient_price_cache: dict[
            tuple[RuleIdentity, ...],
            dict[RuleIdentity, tuple[float, float]],
        ] = {}
        transition_cache_hits = 0
        priced_state_count = 0
        priced_rule_count = 0
        pricing_seconds = 0.0
        exact_neighbor_requests = 0
        full_terminal_audits = 0
        score_terminal_certificates = 0
        score_screened_add_requests = 0
        score_admitted_add_requests = 0
        maximum_terminal_score_surplus = 0.0
        conditional_bound_evaluations = 0
        conditional_bound_screens = 0
        conditional_bound_seconds = 0.0
        skipped_after_first_improvement = 0
        speculative_exact_requests_after_first_improvement = 0
        ordered_lazy_refit_rounds = 0
        tolerance = float(self.config.search_improvement_tolerance)
        execution_wave_size = max(1, len(self._active_support_workers))
        # Under all-atom starts, every singleton state receives a complete
        # exact one-exchange audit.  Consequently the union below (all
        # singleton and pair supports) is not speculative work: the serial
        # traversal must fit exactly the same model keys before it can certify
        # those starts.  Fit the union once as a large closure-aware batch so
        # the process/thread pool stays occupied and repeated small-batch
        # startup is removed.  Only scheduling changes; the finite family,
        # objective, optimizer and subsequent transition decisions are
        # identical.  Gradient-first deliberately keeps its ordered waves.
        initial_frontier_fit_count = 0
        initial_frontier_seconds = 0.0
        if (
            not gradient_strategy
            and self.config.active_start_policy == "all_atoms"
            and starts
        ):
            initial_frontier = sorted(
                {
                    trial
                    for start in starts
                    for trial in self._one_exchange_neighbors(start)
                    if trial
                }
            )
            uncached_frontier = [
                trial for trial in initial_frontier if trial not in record_cache
            ]
            frontier_started = time.perf_counter()
            records(uncached_frontier)
            initial_frontier_seconds = time.perf_counter() - frontier_started
            initial_frontier_fit_count = len(uncached_frontier)

        # Traverse all exact best-improvement starts breadth-synchronously.
        # At a fixed depth, every current state has already been determined by
        # earlier exact scores, hence its complete neighborhood is mandatory
        # work.  Taking the union only coalesces those mandatory model keys
        # into a larger solver batch.  Per-state ranking below is unchanged,
        # so every path, terminal and one-exchange certificate is the same as
        # the serial multi-start traversal (up to solver floating tolerance).
        exact_frontier_rounds = 0
        exact_frontier_requested_keys = 0
        exact_frontier_unique_new_keys = 0
        exact_frontier_fit_seconds = 0.0
        if not gradient_strategy:
            run_states: list[dict] = []
            for start in starts:
                current = tuple(start)
                run_states.append(
                    {
                        "start": current,
                        "current": current,
                        "score": (
                            0.0
                            if not current
                            else self._support_search_score(record(current))
                        ),
                        "path": [],
                        "done": False,
                    }
                )
            run_results: list[dict | None] = [None] * len(run_states)
            while any(not state["done"] for state in run_states):
                exact_frontier_rounds += 1
                current_keys = sorted(
                    {
                        tuple(state["current"])
                        for state in run_states
                        if not state["done"]
                        and tuple(state["current"]) not in transition_cache
                    }
                )
                neighborhoods = {
                    current: self._one_exchange_neighbors(current)
                    for current in current_keys
                }
                requested = [
                    trial
                    for current in current_keys
                    for trial in neighborhoods[current]
                    if trial
                ]
                exact_neighbor_requests += len(requested)
                exact_frontier_requested_keys += len(requested)
                new_keys = sorted(
                    {
                        trial
                        for trial in requested
                        if trial not in record_cache
                    }
                )
                exact_frontier_unique_new_keys += len(new_keys)
                frontier_fit_started = time.perf_counter()
                records(new_keys)
                exact_frontier_fit_seconds += (
                    time.perf_counter() - frontier_fit_started
                )
                for current in current_keys:
                    current_score = next(
                        float(state["score"])
                        for state in run_states
                        if not state["done"]
                        and tuple(state["current"]) == current
                    )
                    evaluated: list[
                        tuple[float, tuple[RuleIdentity, ...]]
                    ] = []
                    for trial in neighborhoods[current]:
                        trial_score = (
                            0.0
                            if not trial
                            else self._support_search_score(record_cache[trial])
                        )
                        evaluated.append((trial_score, trial))
                        if trial:
                            visited.add(trial)
                    evaluated.sort(key=lambda item: (-item[0], item[1]))
                    best_score, best_rules = (
                        evaluated[0]
                        if evaluated
                        else (-math.inf, current)
                    )
                    finite_gains = [
                        score - current_score
                        for score, _trial in evaluated
                        if math.isfinite(score)
                        and math.isfinite(current_score)
                    ]
                    stationarity_gap = (
                        max(0.0, max(finite_gains, default=-math.inf))
                        if math.isfinite(current_score)
                        else math.inf
                    )
                    transition_cache[current] = {
                        "evaluated": tuple(evaluated),
                        "best_score": float(best_score),
                        "best_rules": best_rules,
                        "stationarity_gap": stationarity_gap,
                        "full_audit": True,
                        "neighborhood_count": len(neighborhoods[current]),
                    }

                claimed_this_round: set[tuple[RuleIdentity, ...]] = set()
                for run_index, state in enumerate(run_states):
                    if state["done"]:
                        continue
                    current = tuple(state["current"])
                    current_score = float(state["score"])
                    if current in claimed_this_round or current not in current_keys:
                        transition_cache_hits += 1
                    claimed_this_round.add(current)
                    transition = transition_cache[current]
                    best_score = float(transition["best_score"])
                    best_rules = tuple(transition["best_rules"])
                    gain = float(best_score - current_score)
                    improving = bool(
                        math.isfinite(best_score)
                        and (
                            not math.isfinite(current_score)
                            or gain > tolerance
                        )
                    )
                    if improving:
                        state["path"].append(
                            {
                                "from": [
                                    self._rule_dict(rule) for rule in current
                                ],
                                "to": [
                                    self._rule_dict(rule) for rule in best_rules
                                ],
                                "score_gain": gain,
                            }
                        )
                        state["current"] = best_rules
                        state["score"] = best_score
                        continue
                    if current:
                        visited.add(current)
                        terminals.add(current)
                    stationarity_gap = float(
                        transition["stationarity_gap"]
                    )
                    if not bool(transition["full_audit"]):
                        raise RuntimeError(
                            "terminal support is missing its complete exact neighborhood audit"
                        )
                    run_results[run_index] = {
                        "start": [
                            self._rule_dict(rule) for rule in state["start"]
                        ],
                        "terminal": [
                            self._rule_dict(rule) for rule in current
                        ],
                        "terminal_score": (
                            current_score
                            if math.isfinite(current_score)
                            else None
                        ),
                        "one_exchange_stationarity_gap": stationarity_gap,
                        "stationary_within_tolerance": bool(
                            stationarity_gap <= tolerance
                        ),
                        "moves": state["path"],
                    }
                    state["done"] = True
            runs.extend(
                result for result in run_results if result is not None
            )

        if gradient_strategy:
            # Advance every deterministic start concurrently, but expose only
            # the next gradient-ordered coordinate(s) of each current state.
            # The previous implementation completed one start at a time and
            # fitted a full hardware-sized wave for it; with 100 atom starts
            # that eagerly materialized much of the 4,950-pair frontier even
            # when the first candidate in each wave was already improving.
            # Fair inter-start batching keeps all workers occupied while each
            # path still takes the same first exact improvement in its fixed
            # pricing order.  A state with no improvement necessarily reaches
            # the end of both lists and therefore retains the complete exact
            # one-exchange terminal audit.
            run_states = [
                {
                    "start": tuple(start),
                    "current": tuple(start),
                    "score": (
                        0.0
                        if not start
                        else self._support_search_score(record(start))
                    ),
                    "path": [],
                    "done": False,
                }
                for start in starts
            ]
            run_results: list[dict | None] = [None] * len(run_states)

            def apply_gradient_transition(
                run_index: int,
                transition: dict,
            ) -> None:
                state = run_states[run_index]
                current = tuple(state["current"])
                current_score = float(state["score"])
                best_score = float(transition["best_score"])
                best_rules = tuple(transition["best_rules"])
                gain = float(best_score - current_score)
                improving = bool(
                    math.isfinite(best_score)
                    and (
                        not math.isfinite(current_score)
                        or gain > tolerance
                    )
                )
                if improving:
                    state["path"].append(
                        {
                            "from": [
                                self._rule_dict(rule) for rule in current
                            ],
                            "to": [
                                self._rule_dict(rule) for rule in best_rules
                            ],
                            "score_gain": gain,
                        }
                    )
                    state["current"] = best_rules
                    state["score"] = best_score
                    return
                if not bool(transition["full_audit"]):
                    raise RuntimeError("gradient terminal certificate is incomplete")
                if current:
                    visited.add(current)
                    terminals.add(current)
                stationarity_gap = float(transition["stationarity_gap"])
                gap_name = (
                    "block_score_stationarity_gap"
                    if score_working_set
                    else "one_exchange_stationarity_gap"
                )
                omitted_score_surplus = float(
                    transition.get("omitted_score_surplus", -math.inf)
                )
                stationary = bool(
                    stationarity_gap <= tolerance
                    and (
                        not score_working_set
                        or omitted_score_surplus <= tolerance
                    )
                )
                run_results[run_index] = {
                    "start": [
                        self._rule_dict(rule) for rule in state["start"]
                    ],
                    "terminal": [
                        self._rule_dict(rule) for rule in current
                    ],
                    "terminal_score": (
                        current_score if math.isfinite(current_score) else None
                    ),
                    gap_name: stationarity_gap,
                    "maximum_screened_add_score_surplus": (
                        omitted_score_surplus
                        if score_working_set
                        and math.isfinite(omitted_score_surplus)
                        else None
                    ),
                    "stationary_within_tolerance": stationary,
                    "moves": state["path"],
                }
                state["done"] = True

            while any(not state["done"] for state in run_states):
                waiting: dict[
                    tuple[RuleIdentity, ...], list[int]
                ] = {}
                for run_index, state in enumerate(run_states):
                    if state["done"]:
                        continue
                    current = tuple(state["current"])
                    cached_transition = transition_cache.get(current)
                    if cached_transition is not None:
                        transition_cache_hits += 1
                        apply_gradient_transition(
                            run_index, cached_transition
                        )
                        continue
                    waiting.setdefault(current, []).append(run_index)
                if not waiting:
                    continue

                # Pricing one state evaluates its fitted predictor on the
                # union of all absent-rule rows.  On Freddie this is tens of
                # millions of rows, while the former loop priced up to 100
                # atom starts serially and left eleven physical cores idle.
                # States are independent and each rule reduction retains its
                # original row order, so state-level threads alter only
                # scheduling, not a score, KKT value or candidate order.
                pricing_work: list[
                    tuple[
                        tuple[RuleIdentity, ...],
                        FitResult,
                        tuple[RuleIdentity, ...],
                    ]
                ] = []
                for current in sorted(waiting):
                    if current in gradient_price_cache:
                        continue
                    full_neighborhood = self._one_exchange_neighbors(current)
                    added_rules = tuple(
                        sorted(
                            {
                                rule
                                for trial in full_neighborhood
                                for rule in set(trial) - set(current)
                            }
                        )
                    )
                    pricing_fit = (
                        self.fit_baseline()
                        if not current
                        else record(current).fit
                    )
                    pricing_work.append(
                        (current, pricing_fit, added_rules)
                    )
                if pricing_work:
                    pricing_started = time.perf_counter()

                    def price_state(
                        item: tuple[
                            tuple[RuleIdentity, ...],
                            FitResult,
                            tuple[RuleIdentity, ...],
                        ],
                    ) -> tuple[
                        tuple[RuleIdentity, ...],
                        dict[RuleIdentity, tuple[float, float]],
                        int,
                    ]:
                        current, pricing_fit, added_rules = item
                        with _single_threaded_local_blas():
                            prices = self._support_rule_gradient_prices(
                                pricing_fit,
                                added_rules,
                                worker_limit=1,
                            )
                        return current, prices, len(added_rules)

                    price_workers = min(len(pricing_work), execution_wave_size)
                    if score_working_set and price_workers > 1:
                        # Fuse at most one former concurrency wave.  Thus the
                        # residual matrix replaces, rather than adds to, the
                        # memory that independent scalar workers allocated.
                        priced_states = []
                        for left in range(
                            0, len(pricing_work), price_workers
                        ):
                            wave = pricing_work[left : left + price_workers]
                            fused = self._support_rule_gradient_prices_batch(
                                [(fit, rules) for _current, fit, rules in wave]
                            )
                            priced_states.extend(
                                (current, prices, len(rules))
                                for (current, _fit, rules), prices in zip(
                                    wave, fused, strict=True
                                )
                            )
                    elif price_workers <= 1:
                        priced_states = [price_state(item) for item in pricing_work]
                    else:
                        with ThreadPoolExecutor(
                            max_workers=price_workers
                        ) as executor:
                            priced_states = list(
                                executor.map(price_state, pricing_work)
                            )
                    pricing_seconds += (
                        time.perf_counter() - pricing_started
                    )
                    for current, prices, rule_count in priced_states:
                        gradient_price_cache[current] = prices
                        priced_state_count += 1
                        priced_rule_count += rule_count

                explorations: dict[tuple[RuleIdentity, ...], dict] = {}
                for current in sorted(waiting):
                    current_score = float(
                        run_states[waiting[current][0]]["score"]
                    )
                    full_neighborhood = self._one_exchange_neighbors(current)
                    coordinate_neighborhood = self._add_drop_neighbors(current)
                    prices = gradient_price_cache.get(current)
                    if prices is None:
                        added_rules = sorted(
                            {
                                rule
                                for trial in full_neighborhood
                                for rule in set(trial) - set(current)
                            }
                        )
                        pricing_started = time.perf_counter()
                        pricing_fit = (
                            self.fit_baseline()
                            if not current
                            else record(current).fit
                        )
                        prices = self._support_rule_gradient_prices(
                            pricing_fit,
                            added_rules,
                        )
                        pricing_seconds += (
                            time.perf_counter() - pricing_started
                        )
                        priced_state_count += 1
                        priced_rule_count += len(added_rules)
                        gradient_price_cache[current] = prices

                    def order_key(
                        trial: tuple[RuleIdentity, ...],
                        *,
                        current_rules: tuple[RuleIdentity, ...] = current,
                        current_prices: dict[
                            RuleIdentity, tuple[float, float]
                        ] = prices,
                    ) -> tuple:
                        added = tuple(
                            sorted(set(trial) - set(current_rules))
                        )
                        removed = tuple(
                            sorted(set(current_rules) - set(trial))
                        )
                        if not added:
                            amplitude = (
                                min(
                                    float(
                                        record(current_rules).fit.amplitudes[
                                            current_rules.index(rule)
                                        ]
                                    )
                                    for rule in removed
                                )
                                if current_rules and removed
                                else 0.0
                            )
                            return (0, amplitude, trial)
                        gain_value, kkt_value = current_prices[added[0]]
                        return (
                            1,
                            -float(gain_value),
                            -float(kkt_value),
                            trial,
                        )

                    score_surpluses: dict[
                        tuple[RuleIdentity, ...], float
                    ] = {}
                    omitted_score_surplus = -math.inf
                    if score_working_set:
                        # This is the block hard-thresholding rule obtained by
                        # minimizing the local Fisher score surrogate plus the
                        # exact incremental MDL code.  Every absent block is priced;
                        # only its exact nonlinear refit is conditional.  A
                        # strict comparison, with the same roundoff margin used
                        # by support acceptance, avoids promoting a numerical
                        # zero to the working set.
                        fitted_dimension = int(search_fitted_dimension)
                        current_penalty = self._support_complexity_penalty(
                            current, fitted_dimension
                        )
                        admitted_coordinates: list[
                            tuple[RuleIdentity, ...]
                        ] = []
                        for trial in coordinate_neighborhood:
                            added = tuple(sorted(set(trial) - set(current)))
                            if not added:
                                admitted_coordinates.append(trial)
                                continue
                            if len(added) != 1:
                                raise AssertionError(
                                    "a score-working-set coordinate must add one block"
                                )
                            gain_value, _kkt_value = prices[added[0]]
                            incremental_code = (
                                self._support_complexity_penalty(
                                    trial, fitted_dimension
                                )
                                - current_penalty
                            )
                            surplus = float(
                                2.0
                                * float(gain_value)
                                * self.fit_objective_population_scale
                                - incremental_code
                            )
                            score_surpluses[trial] = surplus
                            score_admissible = surplus > tolerance
                            safely_nonimproving = False
                            if (
                                score_admissible
                                and self.config.conditional_safe_mdl_screen
                            ):
                                bound_started = time.perf_counter()
                                bound = self._support_score_upper_bound(trial)
                                conditional_bound_seconds += (
                                    time.perf_counter() - bound_started
                                )
                                conditional_bound_evaluations += 1
                                certified_upper = float(
                                    bound.get("score_upper_bound", math.inf)
                                ) + float(bound.get("numeric_margin", 0.0))
                                safely_nonimproving = bool(
                                    bound.get("finite", False)
                                    and certified_upper
                                    <= current_score + tolerance
                                )
                            if score_admissible and not safely_nonimproving:
                                admitted_coordinates.append(trial)
                                score_admitted_add_requests += 1
                            else:
                                score_screened_add_requests += 1
                                if safely_nonimproving:
                                    conditional_bound_screens += 1
                                else:
                                    omitted_score_surplus = max(
                                        omitted_score_surplus, surplus
                                    )
                        coordinate_neighborhood = admitted_coordinates
                        # A swap changes two rule blocks and therefore is not a
                        # one-block stationary move.  The exact legacy mode is
                        # retained when a one-exchange certificate is needed.
                        full_neighborhood = sorted(coordinate_neighborhood)
                    coordinate_set = set(coordinate_neighborhood)
                    explorations[current] = {
                        "current_score": current_score,
                        "full": full_neighborhood,
                        "coordinate": sorted(
                            coordinate_neighborhood, key=order_key
                        ),
                        "swaps": (
                            []
                            if score_working_set
                            else sorted(
                                (
                                    trial
                                    for trial in full_neighborhood
                                    if trial not in coordinate_set
                                ),
                                key=order_key,
                            )
                        ),
                        "phase": "coordinate",
                        "cursor": 0,
                        "evaluated": [],
                        "score_surpluses": score_surpluses,
                        "omitted_score_surplus": omitted_score_surplus,
                    }

                while explorations:
                    # Keep the exact ordered first-improvement rule while
                    # filling the physical workers.  Results are committed in
                    # candidate order below, so speculative evaluation can
                    # neither skip an earlier improvement nor change a
                    # terminal audit.  At most one hardware wave is in flight.
                    fair_capacity = max(
                        1,
                        execution_wave_size // max(1, len(explorations)),
                    )
                    requests: list[
                        tuple[
                            tuple[RuleIdentity, ...],
                            tuple[RuleIdentity, ...],
                        ]
                    ] = []
                    finished_without_request: list[
                        tuple[RuleIdentity, ...]
                    ] = []
                    for current, exploration in list(explorations.items()):
                        phase = str(exploration["phase"])
                        candidates = exploration[
                            "coordinate" if phase == "coordinate" else "swaps"
                        ]
                        cursor = int(exploration["cursor"])
                        if cursor >= len(candidates):
                            if phase == "coordinate":
                                exploration["phase"] = "swaps"
                                exploration["cursor"] = 0
                                candidates = exploration["swaps"]
                                cursor = 0
                            if cursor >= len(candidates):
                                finished_without_request.append(current)
                                continue
                        quota = fair_capacity
                        right = min(cursor + quota, len(candidates))
                        requests.extend(
                            (current, candidates[index])
                            for index in range(cursor, right)
                        )
                        exploration["cursor"] = right

                    for current in finished_without_request:
                        exploration = explorations.pop(current)
                        evaluated = exploration["evaluated"]
                        full_neighborhood = exploration["full"]
                        if len(evaluated) != len(full_neighborhood):
                            raise RuntimeError(
                                "gradient terminal audit did not cover its "
                                "complete one-exchange neighborhood"
                            )
                        if score_working_set:
                            score_terminal_certificates += 1
                            omitted_surplus = float(
                                exploration["omitted_score_surplus"]
                            )
                            if math.isfinite(omitted_surplus):
                                maximum_terminal_score_surplus = max(
                                    maximum_terminal_score_surplus,
                                    max(0.0, omitted_surplus),
                                )
                        else:
                            full_terminal_audits += 1
                        finite_gains = [
                            score - float(exploration["current_score"])
                            for score, _trial in evaluated
                            if math.isfinite(score)
                            and math.isfinite(
                                float(exploration["current_score"])
                            )
                        ]
                        transition_cache[current] = {
                            "evaluated": tuple(evaluated),
                            "best_score": float(exploration["current_score"]),
                            "best_rules": current,
                            "stationarity_gap": (
                                max(
                                    0.0,
                                    max(finite_gains, default=-math.inf),
                                )
                                if math.isfinite(
                                    float(exploration["current_score"])
                                )
                                else math.inf
                            ),
                            "full_audit": True,
                            "neighborhood_count": len(full_neighborhood),
                            "omitted_score_surplus": float(
                                exploration["omitted_score_surplus"]
                            ),
                        }
                    if not requests:
                        continue

                    nonempty = [trial for _current, trial in requests if trial]
                    request_records = {
                        item.rules: item for item in records(nonempty)
                    }
                    exact_neighbor_requests += len(nonempty)
                    by_current: dict[
                        tuple[RuleIdentity, ...],
                        list[tuple[RuleIdentity, ...]],
                    ] = {}
                    for current, trial in requests:
                        by_current.setdefault(current, []).append(trial)
                    for current, trials in by_current.items():
                        exploration = explorations.get(current)
                        if exploration is None:
                            continue
                        chosen: tuple[
                            float, tuple[RuleIdentity, ...]
                        ] | None = None
                        for trial in trials:
                            trial_score = (
                                0.0
                                if not trial
                                else self._support_search_score(
                                    request_records[trial]
                                )
                            )
                            exploration["evaluated"].append(
                                (trial_score, trial)
                            )
                            if trial:
                                visited.add(trial)
                            if (
                                chosen is None
                                and math.isfinite(trial_score)
                                and (
                                    not math.isfinite(
                                        float(exploration["current_score"])
                                    )
                                    or trial_score
                                    - float(exploration["current_score"])
                                    > tolerance
                                )
                            ):
                                chosen = (trial_score, trial)
                        if chosen is None:
                            ordered_lazy_refit_rounds += 1
                            continue
                        chosen_index = next(
                            index
                            for index, trial in enumerate(trials)
                            if trial == chosen[1]
                        )
                        speculative_exact_requests_after_first_improvement += max(
                            0, len(trials) - chosen_index - 1
                        )
                        evaluated = exploration["evaluated"]
                        skipped_after_first_improvement += (
                            len(exploration["full"]) - len(evaluated)
                        )
                        transition_cache[current] = {
                            "evaluated": tuple(evaluated),
                            "best_score": float(chosen[0]),
                            "best_rules": chosen[1],
                            "stationarity_gap": None,
                            "full_audit": bool(
                                len(evaluated) == len(exploration["full"])
                            ),
                            "neighborhood_count": len(exploration["full"]),
                        }
                        del explorations[current]

                for current, run_indices in waiting.items():
                    transition = transition_cache.get(current)
                    if transition is None:
                        raise RuntimeError(
                            "gradient exploration did not publish a transition"
                        )
                    for run_index in run_indices:
                        apply_gradient_transition(run_index, transition)
            runs.extend(
                result for result in run_results if result is not None
            )

        # Retained as an unreachable reference implementation while the
        # breadth-synchronous scheduler above is regression-tested.
        for start in ():
            current = tuple(start)
            current_score = 0.0 if not current else self._support_search_score(record(current))
            path: list[dict] = []
            while True:
                cached_transition = transition_cache.get(current)
                if cached_transition is None:
                    full_neighborhood = self._one_exchange_neighbors(current)
                    neighborhood = (
                        self._add_drop_neighbors(current)
                        if gradient_strategy
                        else full_neighborhood
                    )
                    if gradient_strategy:
                        prices = gradient_price_cache.get(current)
                        if prices is None:
                            added_rules = sorted(
                                {
                                    rule
                                    for trial in full_neighborhood
                                    for rule in set(trial) - set(current)
                                }
                            )
                            pricing_started = time.perf_counter()
                            pricing_fit = (
                                self.fit_baseline()
                                if not current
                                else record(current).fit
                            )
                            prices = self._support_rule_gradient_prices(
                                pricing_fit,
                                added_rules,
                            )
                            pricing_seconds += (
                                time.perf_counter() - pricing_started
                            )
                            priced_state_count += 1
                            priced_rule_count += len(added_rules)
                            gradient_price_cache[current] = prices

                        def pricing_order(
                            trial: tuple[RuleIdentity, ...],
                        ) -> tuple:
                            added = tuple(sorted(set(trial) - set(current)))
                            removed = tuple(sorted(set(current) - set(trial)))
                            # Exact drops are few and can expose a cheaper
                            # support before any new column is considered.
                            # Adds then follow decreasing local quadratic
                            # gain. These values only order exact fits.
                            if not added:
                                amplitude = (
                                    min(
                                        float(record(current).fit.amplitudes[
                                            current.index(rule)
                                        ])
                                        for rule in removed
                                    )
                                    if current and removed
                                    else 0.0
                                )
                                return (0, amplitude, trial)
                            gain, kkt = prices[added[0]]
                            return (1, -float(gain), -float(kkt), trial)

                        ordered_neighborhood = sorted(
                            neighborhood,
                            key=pricing_order,
                        )
                        evaluated: list[
                            tuple[float, tuple[RuleIdentity, ...]]
                        ] = []
                        chosen: tuple[
                            float, tuple[RuleIdentity, ...]
                        ] | None = None
                        for left in range(
                            0, len(ordered_neighborhood), execution_wave_size
                        ):
                            wave = ordered_neighborhood[
                                left : left + execution_wave_size
                            ]
                            nonempty = [trial for trial in wave if trial]
                            wave_records = {
                                item.rules: item for item in records(nonempty)
                            }
                            exact_neighbor_requests += len(nonempty)
                            for trial in wave:
                                trial_score = (
                                    0.0
                                    if not trial
                                    else self._support_search_score(
                                        wave_records[trial]
                                    )
                                )
                                evaluated.append((trial_score, trial))
                                if trial:
                                    visited.add(trial)
                                if (
                                    chosen is None
                                    and math.isfinite(trial_score)
                                    and (
                                        not math.isfinite(current_score)
                                        or trial_score - current_score
                                        > tolerance
                                    )
                                ):
                                    chosen = (trial_score, trial)
                            if chosen is not None:
                                skipped_after_first_improvement += (
                                    len(full_neighborhood) - len(evaluated)
                                )
                                break
                        # A direct swap is a two-coordinate move, but auditing
                        # it only after no add/drop improves preserves exact
                        # one-exchange terminal stationarity without paying
                        # for every swap at every intermediate state.
                        if chosen is None:
                            coordinate_set = set(neighborhood)
                            ordered_swaps = sorted(
                                (
                                    trial
                                    for trial in full_neighborhood
                                    if trial not in coordinate_set
                                ),
                                key=pricing_order,
                            )
                            for left in range(
                                0, len(ordered_swaps), execution_wave_size
                            ):
                                wave = ordered_swaps[
                                    left : left + execution_wave_size
                                ]
                                wave_records = {
                                    item.rules: item
                                    for item in records(wave)
                                }
                                exact_neighbor_requests += len(wave)
                                for trial in wave:
                                    trial_score = self._support_search_score(
                                        wave_records[trial]
                                    )
                                    evaluated.append((trial_score, trial))
                                    visited.add(trial)
                                    if (
                                        chosen is None
                                        and math.isfinite(trial_score)
                                        and (
                                            not math.isfinite(current_score)
                                            or trial_score - current_score
                                            > tolerance
                                        )
                                    ):
                                        chosen = (trial_score, trial)
                                if chosen is not None:
                                    skipped_after_first_improvement += (
                                        len(full_neighborhood)
                                        - len(evaluated)
                                    )
                                    break
                        full_audit = len(evaluated) == len(full_neighborhood)
                        if chosen is None:
                            full_terminal_audits += 1
                            best_score, _best_trial = max(
                                evaluated,
                                key=lambda item: item[0],
                                default=(-math.inf, current),
                            )
                            best_rules = current
                            finite_gains = [
                                score - current_score
                                for score, _trial in evaluated
                                if math.isfinite(score)
                                and math.isfinite(current_score)
                            ]
                            stationarity_gap = (
                                max(
                                    0.0,
                                    max(finite_gains, default=-math.inf),
                                )
                                if math.isfinite(current_score)
                                else math.inf
                            )
                        else:
                            best_score, best_rules = chosen
                            stationarity_gap = None
                        transition = {
                            "evaluated": tuple(evaluated),
                            "best_score": float(best_score),
                            "best_rules": best_rules,
                            "stationarity_gap": stationarity_gap,
                            "full_audit": bool(full_audit),
                            "neighborhood_count": len(full_neighborhood),
                        }
                    else:
                        evaluated = []
                        nonempty = [trial for trial in neighborhood if trial]
                        neighborhood_records = {
                            item.rules: item for item in records(nonempty)
                        }
                        exact_neighbor_requests += len(nonempty)
                        for trial in neighborhood:
                            trial_score = (
                                0.0
                                if not trial
                                else self._support_search_score(
                                    neighborhood_records[trial]
                                )
                            )
                            evaluated.append((trial_score, trial))
                            if trial:
                                visited.add(trial)
                        evaluated.sort(key=lambda item: (-item[0], item[1]))
                        best_score, best_rules = (
                            evaluated[0]
                            if evaluated
                            else (-math.inf, current)
                        )
                        finite_gains = [
                            score - current_score
                            for score, _trial in evaluated
                            if math.isfinite(score)
                            and math.isfinite(current_score)
                        ]
                        stationarity_gap = (
                            max(0.0, max(finite_gains, default=-math.inf))
                            if math.isfinite(current_score)
                            else math.inf
                        )
                        transition = {
                            "evaluated": tuple(evaluated),
                            "best_score": float(best_score),
                            "best_rules": best_rules,
                            "stationarity_gap": stationarity_gap,
                            "full_audit": True,
                            "neighborhood_count": len(neighborhood),
                        }
                    transition_cache[current] = transition
                else:
                    transition = cached_transition
                    transition_cache_hits += 1
                best_score = float(transition["best_score"])
                best_rules = tuple(transition["best_rules"])
                gain = float(best_score - current_score)
                improving = bool(
                    math.isfinite(best_score)
                    and (
                        not math.isfinite(current_score)
                        or gain > tolerance
                    )
                )
                if improving:
                    path.append(
                        {
                            "from": [self._rule_dict(rule) for rule in current],
                            "to": [self._rule_dict(rule) for rule in best_rules],
                            "score_gain": gain,
                        }
                    )
                    current = best_rules
                    current_score = best_score
                    continue
                if current:
                    visited.add(current)
                    terminals.add(current)
                stationarity_gap = float(transition["stationarity_gap"])
                if not bool(transition["full_audit"]):
                    raise RuntimeError(
                        "terminal support is missing its complete exact neighborhood audit"
                    )
                gap_name = "one_exchange_stationarity_gap"
                runs.append(
                    {
                        "start": [self._rule_dict(rule) for rule in start],
                        "terminal": [self._rule_dict(rule) for rule in current],
                        "terminal_score": current_score if math.isfinite(current_score) else None,
                        gap_name: stationarity_gap,
                        "stationary_within_tolerance": bool(stationarity_gap <= tolerance),
                        "moves": path,
                    }
                )
                break

        ranked = sorted(
            (
                record_cache[key]
                for key in visited
                if self._support_search_score(record_cache[key]) > 0.0
            ),
            key=lambda item: (-self._support_search_score(item), len(item.rules), item.rules),
        )
        terminal_records = [record_cache[key] for key in terminals]
        terminal_records.sort(
            key=lambda item: (-self._support_search_score(item), len(item.rules), item.rules)
        )
        # Antecedent inclusion alone does not make a two-rule support a local
        # optimum.  Earlier code unconditionally fitted every A+AB-style link,
        # which added hundreds of arbitrary intermediate hypotheses after the
        # stationary search. Multi-rule supports are now reportable only when
        # reached as exact atom-start local terminals; standalone high-order
        # atoms remain anchors, so pair/triplet rules are never hidden.
        hierarchy_link_records: list[SupportRecord] = []
        # Preserve the best visited support in every size/order/sign stratum
        # before filling the remaining pool by Q. This is deterministic and
        # prevents one dominant singleton basin from erasing high-order or
        # inhibitory support structures.
        best_by_stratum: dict[tuple, SupportRecord] = {}
        for item in ranked:
            signature = (
                len(item.rules),
                tuple(sorted(rule.order for rule in item.rules)),
                tuple(sorted(rule.sign for rule in item.rules)),
            )
            best_by_stratum.setdefault(signature, item)
        stratified_records = sorted(
            best_by_stratum.values(),
            key=lambda item: (-self._support_search_score(item), len(item.rules), item.rules),
        )
        # Every admitted rule atom, including every triplet, retains its
        # one-rule support.  Profiling has already paid for these exact
        # singleton-support fits; dropping only order-three atoms here did not
        # save discovery work and made triplets reportable only when attached
        # to another rule.
        anchor_records = positive_singleton_records
        # Exact identity refinement is required only for genuine multi-rule
        # local terminals.  Standalone atoms already received their finite
        # D_fit W/sign profile; intermediate path states make no stationarity
        # claim and are not part of the primary family.
        required_support_keys = (
            set(terminals)
            | {record.rules for record in anchor_records}
            | {record.rules for record in hierarchy_link_records}
        )
        if self.config.support_family == "visited_pool":
            required_support_keys.update(
                record.rules for record in stratified_records
            )
        # Multi-rule local terminals receive support-conditioned W/sign
        # coordinate refinement. Standalone atoms
        # already received their exact identity profile in their one-rule
        # context.
        self._identity_refinement_keys = set(terminals)

        family_candidates = (
            [*terminal_records, *hierarchy_link_records, *anchor_records]
            if self.config.support_family == "terminal_atoms"
            else [
                *terminal_records,
                *hierarchy_link_records,
                *stratified_records,
                *anchor_records,
                *ranked,
            ]
        )
        selected: list[SupportRecord] = []
        selected_keys: set[tuple[RuleIdentity, ...]] = set()
        for item in family_candidates:
            if item.rules in selected_keys:
                continue
            if (
                self.config.support_family == "visited_pool"
                and
                self.config.support_pool_size is not None
                and len(selected) >= self.config.support_pool_size
                and item.rules not in required_support_keys
            ):
                continue
            selected.append(item)
            selected_keys.add(item.rules)
        selected.sort(key=lambda item: (-self._support_search_score(item), len(item.rules), item.rules))
        self.search_diagnostics = {
            "estimator_name": (
                "MDL-triggered fused Rashomon support search"
                if score_working_set
                else None
            ),
            "method": profile,
            "objective": (
                "2*(closure_null_nll-support_nll)-rule_parameter_BIC-"
                "finite_W_sign_identity_code-and_dictionary_shape_code_when_scalar"
            ),
            "neighbor_strategy": self.config.active_neighbor_strategy,
            "neighborhood": (
                "all exact drops plus every add whose conditional cone score pays its incremental MDL code; swaps are two-block moves"
                if score_working_set
                else "gradient-ordered add/drop moves followed by exact terminal swap audit"
                if gradient_strategy
                else "all feasible add/drop/swap moves"
            ),
            "move_selection": (
                "MDL-score working set followed by first exact objective improvement"
                if score_working_set
                else "gradient-ordered first exact improvement with complete exact terminal audit"
                if gradient_strategy
                else "maximum exact one-exchange improvement"
            ),
            "finite_convergence_argument": (
                "each accepted move strictly increases the scalar objective over a finite support space"
            ),
            "stationarity_bound": (
                "all drops and every MDL-admissible score block were exact-fitted; every omitted add has nonpositive conditional Fisher-gain minus incremental MDL code"
                if score_working_set
                else "reported gap is the maximum exact objective gain over the complete one-exchange neighborhood; swaps are deferred until add/drop stationarity"
                if gradient_strategy
                else "reported gap is the maximum exact objective gain over the complete one-exchange neighborhood"
            ),
            "stationarity_claim": (
                "block_score_stationary_not_exact_one_exchange_stationary"
                if score_working_set
                else "exact_one_exchange_stationary"
            ),
            "candidate_budget": None,
            "candidate_coverage": (
                "every feasible absent rule block is scored at every attained state; "
                "there is no top-k, beam width, or triplet budget"
                if score_working_set
                else "complete configured finite neighborhood"
            ),
            "stationarity_objective_scope": (
                "the frozen discovery dictionary objective; terminal full-M refinement is a subsequent "
                "acceptance/refinement stage and is not claimed to preserve the dictionary-stage stationarity certificate"
            ),
            "restart_count": len(starts),
            "active_start_policy": self.config.active_start_policy,
            "start_atom_count": len(starts) - 1,
            "initial_exact_frontier_batch": {
                "enabled": bool(
                    not gradient_strategy
                    and self.config.active_start_policy == "all_atoms"
                ),
                "new_model_key_count": int(initial_frontier_fit_count),
                "seconds": float(initial_frontier_seconds),
                "changes_candidate_family_or_objective": False,
            },
            "exact_multi_start_frontier_batching": {
                "enabled": bool(not gradient_strategy),
                "round_count": int(exact_frontier_rounds),
                "neighborhood_key_requests": int(
                    exact_frontier_requested_keys
                ),
                "unique_new_model_keys": int(
                    exact_frontier_unique_new_keys
                ),
                "fit_seconds": float(exact_frontier_fit_seconds),
                "changes_candidate_family_or_objective": False,
            },
            "evaluated_support_count": len(record_cache),
            "memoized_neighborhood_count": len(transition_cache),
            "memoized_neighborhood_hits": transition_cache_hits,
            "gradient_pricing": {
                "enabled": bool(gradient_strategy),
                "role": (
                    "MDL_calibrated_full_dictionary_working_set_no_top_k_or_budget"
                    if score_working_set
                    else "deterministic_ordering_only_no_candidate_removal"
                ),
                "priced_state_count": int(priced_state_count),
                "priced_rule_count": int(priced_rule_count),
                "pricing_seconds": float(pricing_seconds),
                "exact_neighbor_requests": int(exact_neighbor_requests),
                "complete_terminal_audits": int(full_terminal_audits),
                "block_score_terminal_certificates": int(
                    score_terminal_certificates
                ),
                "score_admitted_add_requests": int(
                    score_admitted_add_requests
                ),
                "score_screened_add_requests": int(
                    score_screened_add_requests
                ),
                "conditional_safe_mdl_bound": {
                    "enabled": bool(
                        score_working_set
                        and self.config.conditional_safe_mdl_screen
                    ),
                    "evaluations": int(conditional_bound_evaluations),
                    "screened_exact_fits": int(
                        conditional_bound_screens
                    ),
                    "seconds": float(conditional_bound_seconds),
                    "criterion": (
                        "candidate_absolute_MDL_upper_bound <= "
                        "current_exact_objective + acceptance_tolerance"
                    ),
                    "changes_accepted_move": False,
                },
                "maximum_terminal_screened_score_surplus": float(
                    maximum_terminal_score_surplus
                ),
                "neighbors_not_fitted_after_first_improvement": int(
                    skipped_after_first_improvement
                ),
                "ordered_work_conserving_exact_refits": {
                    "enabled": bool(gradient_strategy),
                    "dispatch": (
                        "one_ordered_candidate_per_active_state_then_fill_"
                        "idle_workers_with_the_next_contiguous_prefix"
                    ),
                    "maximum_exact_batch_size": int(execution_wave_size),
                    "exactly_nonimproving_wave_count": int(
                        ordered_lazy_refit_rounds
                    ),
                    "speculative_exact_requests_after_first_improvement": int(
                        speculative_exact_requests_after_first_improvement
                    ),
                    "preserves_candidate_order": True,
                    "preserves_accepted_moves_and_terminal": True,
                    "changes_objective_or_candidate_family": False,
                },
                "execution_wave_size": int(execution_wave_size),
                "inter_start_batching": (
                    "fair_one-next-coordinate-per-active-state_then_fill_"
                    "workers_near_terminal"
                    if gradient_strategy
                    else None
                ),
                "hardware_wave_changes_move": False,
            },
            "visited_positive_support_count": len(ranked),
            "hierarchy_link_support_count": len(hierarchy_link_records),
            "hierarchy_link_definition": (
                "none: antecedent inclusion alone is not a reportable support; any A+AB structure must be an atom-start local terminal"
            ),
            "returned_pool_count": len(selected),
            "returned_family_definition": (
                (
                    "all_positive_standalone_atoms_and_unique_atom_start_block_score_terminals"
                    if score_working_set
                    else "all_positive_standalone_atoms_and_unique_atom_start_local_terminals"
                )
                if self.config.support_family == "terminal_atoms"
                else "legacy_positive_visited_pool_plus_required_anchors_and_terminals"
            ),
            "terminal_support_count": len(terminals),
            "structural_strata_preserved": (
                len(stratified_records)
                if self.config.support_family == "visited_pool"
                else 0
            ),
            "atom_anchor_support_count": len(anchor_records),
            "atom_anchor_max_antecedent_order": self.config.q_max,
            "triplet_anchor_policy": "every_positive_profiled_triplet_has_a_standalone_support",
            "support_family": self.config.support_family,
            "pool_cap": self.config.support_pool_size,
            "pool_cap_excludes_required_anchors": True,
            "safe_mdl_screen": {
                "enabled": bool(self.config.safe_mdl_screen),
                "method": "exact-group-saturated-occurrence-plus-relaxed-mark-global-bound",
                **self._safe_screen_stats,
                "guarantee": (
                    "only supports with a strictly negative global block-MDL upper bound are skipped"
                ),
            },
            "runs": runs,
        }
        self._support_pricing_union_cache = None
        return selected

    def _refine_support_identities(
        self,
        records: Sequence[SupportRecord],
    ) -> tuple[list[SupportRecord], dict]:
        """Exact coordinate profiling of W/sign conditional on each support.

        Adjacent formation-window moves and the sign flip for one existing
        antecedent skeleton are evaluated while all other identities are held
        fixed. Strict Q ascent over the connected finite state graph guarantees
        termination. The reported gap is exact for this local coordinate
        neighborhood; it is not a global identity-combination certificate.
        """
        profile = "support-conditioned-exact-coordinate-w-sign"
        tolerance = float(self.config.search_improvement_tolerance)
        cache: dict[tuple[RuleIdentity, ...], SupportRecord] = {
            record.rules: record for record in records
        }

        def get(rules: Sequence[RuleIdentity]) -> SupportRecord:
            key = tuple(sorted(rules))
            item = cache.get(key)
            if item is None:
                item = self._support_record(key, profile=profile)
                cache[key] = item
            return item

        def get_many(rule_sets: Sequence[Sequence[RuleIdentity]]) -> list[SupportRecord]:
            keys = list(dict.fromkeys(tuple(sorted(rules)) for rules in rule_sets))
            missing = [key for key in keys if key not in cache]
            for item in self._fit_or_safe_screen_records_batch(missing, profile=profile):
                cache[item.rules] = item
            return [cache[key] for key in keys]

        def local_identity_neighbors(rule: RuleIdentity) -> tuple[RuleIdentity, ...]:
            candidates = self.identity_candidates.get(rule.antecedent, (rule,))
            by_window: dict[int, dict[int, RuleIdentity]] = {}
            for candidate in candidates:
                by_window.setdefault(int(candidate.window), {})[int(candidate.sign)] = candidate
            windows = sorted(by_window)
            if int(rule.window) not in by_window:
                return tuple(candidate for candidate in candidates if candidate != rule)
            position = windows.index(int(rule.window))
            neighbor_windows = [windows[position]]
            if position > 0:
                neighbor_windows.append(windows[position - 1])
            if position + 1 < len(windows):
                neighbor_windows.append(windows[position + 1])
            output = {
                candidate
                for window in neighbor_windows
                for candidate in by_window[window].values()
                if candidate != rule
            }
            return tuple(sorted(output))

        refined: list[SupportRecord] = []
        runs: list[dict] = []
        gradient_ordering = (
            self.config.active_neighbor_strategy
            == "gradient_first_exact_audit"
        )
        transition_cache: dict[tuple[RuleIdentity, ...], dict] = {}
        pricing_seconds = 0.0
        priced_rule_count = 0
        exact_fit_requests = 0
        skipped_after_first_improvement = 0
        terminal_audits = 0
        execution_wave_size = max(1, len(self._active_support_workers))
        for start_record in records:
            current = get(start_record.rules)
            current_score = self._support_search_score(current)
            moves: list[dict] = []
            while True:
                transition = transition_cache.get(current.rules)
                if transition is None:
                    trial_candidate: dict[
                        tuple[RuleIdentity, ...], RuleIdentity
                    ] = {}
                    for position, old_rule in enumerate(current.rules):
                        for candidate in local_identity_neighbors(old_rule):
                            if candidate == old_rule:
                                continue
                            trial_rules = list(current.rules)
                            trial_rules[position] = candidate
                            trial_candidate[
                                tuple(sorted(trial_rules))
                            ] = candidate
                    trial_keys = list(trial_candidate)
                    if gradient_ordering and trial_keys:
                        pricing_started = time.perf_counter()
                        candidate_prices = self._support_rule_gradient_prices(
                            current.fit,
                            tuple(trial_candidate.values()),
                        )
                        pricing_seconds += time.perf_counter() - pricing_started
                        priced_rule_count += len(candidate_prices)
                        ordered_keys = sorted(
                            trial_keys,
                            key=lambda key: (
                                -candidate_prices[trial_candidate[key]][0],
                                -candidate_prices[trial_candidate[key]][1],
                                key,
                            ),
                        )
                    else:
                        ordered_keys = sorted(trial_keys)
                    alternatives: list[
                        tuple[float, tuple[RuleIdentity, ...], SupportRecord]
                    ] = []
                    chosen: tuple[
                        float, tuple[RuleIdentity, ...], SupportRecord
                    ] | None = None
                    if gradient_ordering:
                        for left in range(
                            0, len(ordered_keys), execution_wave_size
                        ):
                            wave = ordered_keys[
                                left : left + execution_wave_size
                            ]
                            wave_records = {
                                item.rules: item for item in get_many(wave)
                            }
                            exact_fit_requests += len(wave)
                            for key in wave:
                                item = wave_records[key]
                                score = self._support_search_score(item)
                                alternatives.append((score, key, item))
                                if (
                                    chosen is None
                                    and math.isfinite(score)
                                    and (
                                        not math.isfinite(current_score)
                                        or score - current_score > tolerance
                                    )
                                ):
                                    chosen = (score, key, item)
                            if chosen is not None:
                                skipped_after_first_improvement += (
                                    len(ordered_keys) - len(alternatives)
                                )
                                break
                    else:
                        all_records = {
                            item.rules: item for item in get_many(ordered_keys)
                        }
                        exact_fit_requests += len(ordered_keys)
                        alternatives = [
                            (
                                self._support_search_score(all_records[key]),
                                key,
                                all_records[key],
                            )
                            for key in ordered_keys
                        ]
                        alternatives.sort(key=lambda item: (-item[0], item[1]))
                        chosen = alternatives[0] if alternatives else None
                    full_audit = len(alternatives) == len(ordered_keys)
                    if gradient_ordering and chosen is None:
                        terminal_audits += 1
                    if chosen is None:
                        best_score, best_key, best = (
                            -math.inf,
                            current.rules,
                            current,
                        )
                    else:
                        best_score, best_key, best = chosen
                    finite_gains = [
                        score - current_score
                        for score, _key, _record in alternatives
                        if math.isfinite(score) and math.isfinite(current_score)
                    ]
                    gap = (
                        max(0.0, max(finite_gains, default=-math.inf))
                        if full_audit and math.isfinite(current_score)
                        else None
                    )
                    transition = {
                        "best_score": float(best_score),
                        "best_key": best_key,
                        "best": best,
                        "full_audit": bool(full_audit),
                        "gap": gap,
                    }
                    transition_cache[current.rules] = transition
                best_score = float(transition["best_score"])
                best = transition["best"]
                gain = float(best_score - current_score)
                improving = bool(
                    math.isfinite(best_score)
                    and (not math.isfinite(current_score) or gain > tolerance)
                )
                if improving:
                    moves.append(
                        {
                            "from": [self._rule_dict(rule) for rule in current.rules],
                            "to": [self._rule_dict(rule) for rule in best.rules],
                            "score_gain": gain,
                        }
                    )
                    current = best
                    current_score = best_score
                    continue
                if not bool(transition["full_audit"]):
                    raise RuntimeError(
                        "identity terminal is missing its complete exact coordinate audit"
                    )
                gap = float(transition["gap"])
                refined.append(current)
                runs.append(
                    {
                        "start": [self._rule_dict(rule) for rule in start_record.rules],
                        "terminal": [self._rule_dict(rule) for rule in current.rules],
                        "terminal_score": current_score if math.isfinite(current_score) else None,
                        "coordinate_stationarity_gap": gap,
                        "stationary_within_tolerance": bool(gap <= tolerance),
                        "moves": moves,
                    }
                )
                break

        # Multiple canonical starts can converge to the same conditional
        # identity. Keep one exact fit per resulting support.
        unique = {record.rules: record for record in refined}
        output = sorted(
            unique.values(),
            key=lambda item: (-self._support_search_score(item), len(item.rules), item.rules),
        )
        diagnostics = {
            "enabled": True,
            "method": profile,
            "input_support_count": len(records),
            "output_support_count": len(output),
            "evaluated_identity_support_count": len(cache),
            "finite_convergence_argument": (
                "each accepted adjacent-W/sign-flip coordinate move strictly increases Q over a finite state graph"
            ),
            "stationarity_bound": (
                "reported gap is the maximum exact Q gain over every adjacent-W or sign-flip move of one rule"
            ),
            "gradient_ordering": {
                "enabled": bool(gradient_ordering),
                "role": "deterministic_ordering_only_no_identity_removal",
                "pricing_seconds": float(pricing_seconds),
                "priced_rule_count": int(priced_rule_count),
                "exact_fit_requests": int(exact_fit_requests),
                "neighbors_not_fitted_after_first_improvement": int(
                    skipped_after_first_improvement
                ),
                "complete_terminal_audits": int(terminal_audits),
                "execution_wave_size": int(execution_wave_size),
            },
            "runs": runs,
        }
        return output, diagnostics

    def _refit_returned_full_kernels(
        self,
        records: Sequence[SupportRecord],
    ) -> tuple[list[SupportRecord], dict]:
        """Lift every support returned for D_fit screening to the full M-knot cone."""
        rules_to_expand = {rule for record in records for rule in record.rules}
        selected_shapes = {
            rule: self.rule_dictionary_shapes[rule].tolist()
            for rule in rules_to_expand
            if rule in self.rule_dictionary_shapes
        }
        def prepare_pipeline(pipeline: CertSCRPipeline) -> None:
            for rule in rules_to_expand:
                pipeline.rule_dictionary_shapes.pop(rule, None)
            # Dictionary discovery fits use one scalar column per rule. Once a
            # terminal rule is lifted to full M, every cached model containing
            # it has an incompatible parameter dimension.
            for key in list(pipeline._fit_cache):
                cached_rules, _closure_terms = key
                if any(rule in rules_to_expand for rule in cached_rules):
                    del pipeline._fit_cache[key]

        prepare_pipeline(self)
        workers = self._active_support_workers
        for worker in workers:
            prepare_pipeline(worker)

        def refit_one(
            pipeline: CertSCRPipeline,
            record: SupportRecord,
            *,
            prepared_design: PreparedFixedSupportDesign | None = None,
        ) -> SupportRecord:
            rules = record.rules
            closure_terms = pipeline.hierarchy_closure(rules)
            closure_baseline = pipeline.fit_model((), closure_terms)
            dictionary_theta = np.stack(
                [
                    float(record.fit.amplitudes[index])
                    * np.asarray(selected_shapes[rule], dtype=np.float64)
                    for index, rule in enumerate(rules)
                ],
                axis=0,
            )
            expanded_dictionary_fit = replace(record.fit, theta=dictionary_theta)
            nuisance = pipeline.sparse_nuisance_blocks(
                pipeline.splits.fit, closure_terms
            )
            raw_features = [
                pipeline.engine.sparse_response(
                    pipeline.splits.fit,
                    rule.antecedent,
                    rule.window,
                )
                for rule in rules
            ]
            prepared = prepared_design
            if prepared is None:
                prepared = pipeline.prepare_partitioned_support_design(
                    pipeline.splits.fit,
                    closure_terms,
                    raw_features,
                    rules,
                    cluster_weights=pipeline.fit_cluster_weights,
                )
            if (
                str(pipeline.config.solver_device).startswith("cpu")
                and pipeline.config.solver_dtype == "float64"
            ):
                prepared = promote_prepared_design_float64(prepared)
            recession_columns = (
                pipeline._nonattained_rule_recession_columns(prepared)
            )
            if recession_columns:
                full_fit = pipeline._nonattained_support_fit(
                    rules,
                    closure_baseline,
                    int(prepared.knot_count),
                    recession_columns,
                )
            else:
                # The grouped full-design KKT scan is itself the exact
                # necessary and sufficient convex-optimality check once the
                # explicit nonattainment directions above are absent.
                already_optimal, initial_kkt, _initial_objective = (
                    fixed_support_projected_kkt(
                        prepared,
                        expanded_dictionary_fit,
                        tolerance=pipeline.config.solver_tolerance,
                    )
                )
                if already_optimal:
                    # Convexity makes the full-cone KKT condition sufficient.
                    # The expanded coefficients represent exactly the
                    # dictionary predictor, so retaining its joint objective
                    # and mark head avoids a roundoff-only ordering change.
                    full_fit = replace(
                        expanded_dictionary_fit,
                        kkt_residual=float(initial_kkt),
                        converged=True,
                        iterations=0,
                    )
                else:
                    full_fit = fit_fixed_support(
                        pipeline.splits.fit,
                        nuisance,
                        raw_features,
                        rules,
                        device=pipeline.config.solver_device,
                        dtype=pipeline.config.solver_dtype,
                        max_iter=pipeline.config.solver_max_iter,
                        tolerance=pipeline.config.solver_tolerance,
                        initial=expanded_dictionary_fit,
                        closure_terms=closure_terms,
                        cluster_weights=pipeline.fit_cluster_weights,
                        sequence_exposures=pipeline.sequence_exposures(
                            pipeline.splits.fit
                        ),
                        prepared_design=prepared,
                        occurrence_likelihood=pipeline.occurrence_likelihood,
                    )
                    full_fit = pipeline._attach_mark_head(
                        full_fit, pipeline.splits.fit
                    )
            # Full-M mode is an occurrence-first two-stage estimator.  Its
            # occurrence head must therefore satisfy the full-cone KKT
            # contract; retaining a lower-dimensional dictionary point after
            # KKT failed mislabeled that point as a converged full-M fit and
            # applied the wrong parameter penalty.  The conditional mark head
            # is fitted at the resulting occurrence shape even if its joint
            # diagnostic NLL happens to be worse than at the screening shape.
            chosen_fit = full_fit
            pipeline._fit_cache[(rules, closure_terms)] = chosen_fit
            return SupportRecord(
                rules=rules,
                fit=chosen_fit,
                closure_baseline_fit=closure_baseline,
                search_nll_improvement=float(closure_baseline.nll - chosen_fit.nll),
                profile="returned-pool-full-m-knot-refit-after-dictionary-search",
            )

        # A design grouped on a maximal support is an exact finer partition for
        # every nested support.  Select the nested closure/rule columns and
        # regroup identical rows, rather than rebuilding each model from the
        # complete time grid.  On the audited IBM pool this reduces full-grid
        # assemblies from 2,824 supports to 334 maximal supports.  Every child
        # still solves its original convex objective at the original KKT
        # tolerance; this is sufficient-statistic reuse, not candidate pruning.
        ordered_records = sorted(
            records,
            key=lambda item: (-len(item.rules), item.rules),
        )
        maximal_records: list[SupportRecord] = []
        for record in ordered_records:
            rule_set = set(record.rules)
            if not any(
                rule_set.issubset(set(parent.rules))
                for parent in maximal_records
            ):
                maximal_records.append(record)
        children_by_parent: dict[
            tuple[RuleIdentity, ...], list[SupportRecord]
        ] = {parent.rules: [] for parent in maximal_records}
        for record in records:
            candidates = [
                parent
                for parent in maximal_records
                if set(record.rules).issubset(set(parent.rules))
            ]
            parent = min(candidates, key=lambda item: (len(item.rules), item.rules))
            children_by_parent[parent.rules].append(record)

        def refit_parent_group(
            pipeline: CertSCRPipeline,
            parent: SupportRecord,
            children: Sequence[SupportRecord],
        ) -> list[SupportRecord]:
            parent_rules = parent.rules
            parent_closure = pipeline.hierarchy_closure(parent_rules)
            parent_features = [
                pipeline.engine.sparse_response(
                    pipeline.splits.fit,
                    rule.antecedent,
                    rule.window,
                )
                for rule in parent_rules
            ]
            parent_prepared = pipeline.prepare_partitioned_support_design(
                pipeline.splits.fit,
                parent_closure,
                parent_features,
                parent_rules,
                cluster_weights=pipeline.fit_cluster_weights,
            )
            output_group: list[SupportRecord] = []
            for child in sorted(
                children,
                key=lambda item: (-len(item.rules), item.rules),
            ):
                if child.rules == parent_rules:
                    child_prepared = parent_prepared
                else:
                    child_prepared = project_prepared_support_design(
                        parent_prepared,
                        child.rules,
                        source_closure_terms=parent_closure,
                        target_closure_terms=pipeline.hierarchy_closure(
                            child.rules
                        ),
                        regroup=True,
                    )
                output_group.append(
                    refit_one(
                        pipeline,
                        child,
                        prepared_design=child_prepared,
                    )
                )
            return output_group

        fit_worker_count = self._exact_fit_worker_limit(
            len(workers) if workers else 1
        )
        worker_pool = (
            workers[:fit_worker_count]
            if len(workers) > 1
            else [self]
        )
        # Greedy structural load balance changes scheduling only.  The weight is
        # the number of fitted rule columns in each parent group and uses no
        # outcome, score, p-value, or convergence result.
        assignments: list[list[SupportRecord]] = [
            [] for _ in range(len(worker_pool))
        ]
        assignment_weights = [0 for _ in worker_pool]
        for parent in sorted(
            maximal_records,
            key=lambda item: (
                -sum(
                    1 + len(child.rules)
                    for child in children_by_parent[item.rules]
                ),
                item.rules,
            ),
        ):
            worker_index = min(
                range(len(worker_pool)),
                key=lambda index: (assignment_weights[index], index),
            )
            assignments[worker_index].append(parent)
            assignment_weights[worker_index] += sum(
                1 + len(child.rules)
                for child in children_by_parent[parent.rules]
            )

        def refit_assignment(
            worker: CertSCRPipeline,
            parents: Sequence[SupportRecord],
        ) -> list[SupportRecord]:
            result: list[SupportRecord] = []
            with _single_threaded_local_blas():
                for parent in parents:
                    result.extend(
                        refit_parent_group(
                            worker,
                            parent,
                            children_by_parent[parent.rules],
                        )
                    )
            return result

        output: list[SupportRecord] = []
        use_fork_refit = bool(
            os.environ.get("CERTSCR_PROCESS_REFIT", "0") == "1"
            and os.name == "posix"
            and not self.config.support_devices
            and str(self.config.solver_device).startswith("cpu")
            and int(self.config.solver_workers) > 1
            and len(maximal_records) > 1
        )
        if use_fork_refit:
            # Full-M groups are independent and return only small FitResult
            # objects.  Threads serialize the Python-controlled sparse assembly
            # under the GIL; fork shares immutable EventData/response arrays
            # copy-on-write and runs the same deterministic group assignments in
            # separate interpreters.  No support or optimization step is added,
            # removed, or approximated.
            context = mp.get_context("fork")
            result_queue = context.Queue()

            def process_refit_assignment(
                assignment_index: int,
                parents: Sequence[SupportRecord],
            ) -> None:
                try:
                    setter = _mkl_local_thread_setter()
                    if setter is not None:
                        setter(1)
                    worker = copy.copy(self)
                    worker.config = replace(
                        self.config,
                        solver_workers=1,
                        support_devices=(),
                    )
                    worker._active_support_workers = []
                    worker._fit_cache = dict(self._fit_cache)
                    worker._null_fit_cache = dict(self._null_fit_cache)
                    worker._prepared_design_cache = {}
                    worker._information_design_parent = None
                    values = refit_assignment(worker, parents)
                    result_queue.put(
                        (True, assignment_index, values, None)
                    )
                except BaseException:
                    result_queue.put(
                        (
                            False,
                            assignment_index,
                            None,
                            traceback.format_exc(),
                        )
                    )

            process_jobs = [
                (index, assignment)
                for index, assignment in enumerate(assignments)
                if assignment
            ]
            processes = [
                context.Process(
                    target=process_refit_assignment,
                    args=(index, assignment),
                )
                for index, assignment in process_jobs
            ]
            for process in processes:
                process.start()
            received: dict[int, list[SupportRecord]] = {}
            first_error: str | None = None
            while len(received) < len(processes):
                try:
                    ok, assignment_index, values, error = result_queue.get(
                        timeout=5.0
                    )
                except queue.Empty:
                    if any(process.is_alive() for process in processes):
                        continue
                    raise RuntimeError(
                        "forked full-M workers exited before returning all results"
                    )
                if ok:
                    received[int(assignment_index)] = values
                else:
                    received[int(assignment_index)] = []
                    if first_error is None:
                        first_error = str(error)
            for process in processes:
                process.join()
            result_queue.close()
            if first_error is not None:
                raise RuntimeError(
                    "forked full-M refinement failed:\n" + first_error
                )
            bad_exit = [
                process.exitcode
                for process in processes
                if process.exitcode not in {0, None}
            ]
            if bad_exit:
                raise RuntimeError(
                    f"forked full-M workers exited with codes {bad_exit}"
                )
            for assignment_index, _assignment in process_jobs:
                output.extend(received[assignment_index])
        elif len(worker_pool) > 1 and len(maximal_records) > 1:
            with ThreadPoolExecutor(max_workers=len(worker_pool)) as executor:
                futures = [
                    executor.submit(refit_assignment, worker, assignment)
                    for worker, assignment in zip(
                        worker_pool, assignments, strict=True
                    )
                    if assignment
                ]
                for future in futures:
                    output.extend(future.result())
        else:
            output = refit_assignment(self, maximal_records)
        for refined in output:
            self._fit_cache[(refined.fit.rules, refined.fit.closure_terms)] = refined.fit

        dictionary_record_by_rules = {record.rules: record for record in records}
        rows: list[dict] = []
        for refined in output:
            rules = refined.rules
            full_fit = refined.fit
            record = dictionary_record_by_rules[rules]
            full_m_score = self._support_search_score(refined)
            rows.append(
                {
                    "support": [self._rule_dict(rule) for rule in rules],
                    "dictionary_shapes": [selected_shapes.get(rule) for rule in rules],
                    "dictionary_nll": float(record.fit.nll),
                    "full_m_nll": float(full_fit.nll),
                    "full_m_converged": bool(full_fit.converged),
                    "full_m_kkt_shortcut": bool(full_fit.iterations == 0),
                    "full_m_block_mdl_score": (
                        float(full_m_score) if math.isfinite(full_m_score) else None
                    ),
                }
            )
        positive_output = [
            item for item in output if self._support_search_score(item) > 0.0
        ]
        positive_output.sort(
            key=lambda item: (
                -self._support_search_score(item), len(item.rules), item.rules
            )
        )
        return positive_output, {
            "enabled": True,
            "method": "returned-pool-full-m-knot-refit-after-contiguous-dictionary-search",
            "input_support_count": len(records),
            "maximal_prepared_design_count": len(maximal_records),
            "nested_prepared_design_reuse_count": (
                len(records) - len(maximal_records)
            ),
            "execution": (
                "posix-copy-on-write-process-groups"
                if use_fork_refit
                else "independent-thread-groups"
                if len(worker_pool) > 1
                else "serial-groups"
            ),
            "output_support_count": len(positive_output),
            "removed_after_full_m_nonpositive_or_nonconverged_mdl": (
                len(output) - len(positive_output)
            ),
            "rows": rows,
        }

    def search_supports(self) -> list[SupportRecord]:
        self.fit_baseline()
        if not self._profile_completed:
            self.profile_rule_identities()
        if self.config.support_search == "active_set":
            self._start_active_support_workers()
            try:
                active_search_started = time.perf_counter()
                records = self._search_supports_active_set()
                self.search_diagnostics["active_set_seconds"] = (
                    time.perf_counter() - active_search_started
                )
                if self.config.support_conditioned_refinement:
                    identity_refinement_started = time.perf_counter()
                    refinement_records = [
                        record
                        for record in records
                        if record.rules in self._identity_refinement_keys
                    ]
                    refined_supports, refinement = self._refine_support_identities(
                        refinement_records
                    )
                    # A refined support can move to a new identity. Replace
                    # exactly the predeclared refinement subset and leave
                    # standalone atom anchors untouched.
                    merged = [
                        record
                        for record in records
                        if record.rules not in self._identity_refinement_keys
                    ] + list(refined_supports)
                    records = list({record.rules: record for record in merged}.values())
                    records.sort(
                        key=lambda item: (
                            -self._support_search_score(item), len(item.rules), item.rules
                        )
                    )
                    self.search_diagnostics["identity_refinement"] = refinement
                    self.search_diagnostics["identity_refinement_seconds"] = (
                        time.perf_counter() - identity_refinement_started
                    )
                    self.search_diagnostics["returned_pool_count_after_identity_refinement"] = len(records)
                    self.search_diagnostics["refinement_scope"] = (
                        "local_terminal_and_hierarchy_link_supports"
                    )
                if self.config.identity_profile == "dictionary_mdl":
                    kernel_refinement_started = time.perf_counter()
                    records, kernel_refinement = self._refit_returned_full_kernels(records)
                    self.search_diagnostics["returned_pool_full_kernel_refinement"] = kernel_refinement
                    self.search_diagnostics["full_kernel_refinement_seconds"] = (
                        time.perf_counter() - kernel_refinement_started
                    )
                pooled_terms = {
                    (rule.antecedent, int(rule.window))
                    for record in records
                    for rule in record.rules
                }
                for record in records:
                    pooled_terms.update(record.fit.closure_terms)
                self.engine.retain_context_terms(self.splits.fit.name, tuple(pooled_terms))
                self.search_diagnostics["parallel_support_devices"] = list(
                    self._support_worker_devices()
                )
                if "safe_mdl_screen" in self.search_diagnostics:
                    self.search_diagnostics["safe_mdl_screen"].update(self._safe_screen_stats)
                self.support_records = records
                return records
            finally:
                self._active_support_workers = []
        records: list[SupportRecord] = []

        def eligible_supports(size: int) -> list[tuple[RuleIdentity, ...]]:
            return [
                tuple(rules)
                for rules in itertools.combinations(self.profiled_rules, size)
                # A support cannot contain two sign/window identities for the
                # same antecedent skeleton.
                if len({rule.antecedent for rule in rules}) == len(rules)
            ]

        def fit_one(pipeline: CertSCRPipeline, rules: tuple[RuleIdentity, ...]) -> SupportRecord:
            return pipeline._support_record(
                rules,
                profile=(
                    "oracle-exact-identity-exhaustive-support"
                    if self.config.exhaustive_profile
                    else "canonical-rule-atom-exhaustive-support"
                ),
            )

        max_size = self._effective_max_support_size()
        devices = self._support_worker_devices()
        if len(devices) <= 1:
            for size in range(1, max_size + 1):
                for rules in eligible_supports(size):
                    records.append(fit_one(self, rules))
        else:
            # Workers have independent fit caches and one explicitly assigned
            # device.  Their shared occurrence engine owns a thread-safe,
            # byte-bounded feature cache.  They receive the exact same support
            # list; only its partition changes.  No loss/rank/p-value screen is
            # used.
            workers = []
            for device in devices:
                worker = copy.copy(self)
                worker.config = CertSCRConfig(
                    **{
                        **asdict(self.config),
                        "solver_device": device,
                        "support_devices": (),
                    }
                )
                worker._fit_cache = dict(self._fit_cache)
                worker._fit_key_locks = {}
                worker._fit_key_locks_guard = threading.Lock()
                worker._null_fit_cache = dict(self._null_fit_cache)
                worker._hierarchy_closure_cache = dict(self._hierarchy_closure_cache)
                worker._nuisance_event_design_cache = dict(self._nuisance_event_design_cache)
                worker._mark_base_residualizer_cache = dict(self._mark_base_residualizer_cache)
                worker._marked_response_cache = dict(self._marked_response_cache)
                worker._active_support_workers = []
                worker.support_records = []
                worker.candidate_records = []
                workers.append(worker)
            for worker in workers:
                worker._fit_cache = dict(self._fit_cache)
                worker._null_fit_cache = dict(self._null_fit_cache)
                worker._nuisance_event_design_cache = dict(self._nuisance_event_design_cache)
                worker._mark_base_residualizer_cache = dict(self._mark_base_residualizer_cache)
                worker._marked_response_cache = dict(self._marked_response_cache)
                worker.baseline_fit = self.baseline_fit
                worker.profiled_rules = list(self.profiled_rules)

            def fit_chunk(
                worker: CertSCRPipeline,
                chunk: list[tuple[RuleIdentity, ...]],
            ) -> list[SupportRecord]:
                return [fit_one(worker, rules) for rules in chunk]

            fit_worker_count = self._exact_fit_worker_limit(len(workers))
            fit_workers = workers[:fit_worker_count]
            with ThreadPoolExecutor(max_workers=fit_worker_count) as executor:
                for size in range(1, max_size + 1):
                    universe = eligible_supports(size)
                    chunks = [
                        universe[index::fit_worker_count]
                        for index in range(fit_worker_count)
                    ]
                    futures = [
                        executor.submit(fit_chunk, worker, chunk)
                        for worker, chunk in zip(fit_workers, chunks, strict=True)
                        if chunk
                    ]
                    for future in futures:
                        records.extend(future.result())
            for worker in workers:
                worker.engine.clear_context_cache(worker.splits.fit.name)
            # Subsequent fit-screen/certification uses the frozen fits in the
            # parent pipeline, independently of which device produced them.
            for record in records:
                self._fit_cache[(record.fit.rules, record.fit.closure_terms)] = record.fit
                self._fit_cache[((), record.fit.closure_terms)] = record.closure_baseline_fit
                self._null_fit_cache[record.fit.closure_terms] = record.closure_baseline_fit
        records.sort(key=lambda record: (-record.search_nll_improvement, len(record.rules), record.rules))
        self.search_diagnostics = {
            "method": "exhaustive",
            "evaluated_support_count": len(records),
            "returned_pool_count": len(records),
        }
        self.support_records = records
        return records

    def _eta_on(self, fit: FitResult, ctx: QueryContext) -> np.ndarray:
        eta = np.full(ctx.n_queries, float(fit.alpha), dtype=np.float64)
        gamma_offset = 0
        for controls in self.control_blocks(ctx):
            width = int(controls.shape[1])
            eta += controls @ fit.gamma[gamma_offset : gamma_offset + width]
            gamma_offset += width
        for antecedent, window in fit.closure_terms:
            feature = self.engine.response(ctx, antecedent, window)
            width = int(feature.shape[1])
            eta += feature @ fit.gamma[gamma_offset : gamma_offset + width]
            gamma_offset += width
        if gamma_offset != len(fit.gamma):
            raise ValueError("fit/design mismatch")
        for index, (rule, feature) in enumerate(
            zip(fit.rules, self.features(ctx, fit.rules), strict=True)
        ):
            if feature.shape[1] != fit.theta.shape[1]:
                raise ValueError("fit/design mismatch")
            eta += float(rule.sign) * (feature @ fit.theta[index])
        return eta

    def _sparse_fit_summary(
        self,
        fit: FitResult,
        ctx: QueryContext,
    ) -> SparseFitSummary:
        """Evaluate a fit exactly without constructing any full-grid vector."""
        nuisance = self.sparse_nuisance_blocks(ctx, fit.closure_terms)
        features = self.sparse_features(ctx, fit.rules)
        all_blocks = (*nuisance, *features)
        active_indices, block_positions = self._sparse_union_layout(all_blocks)

        event_eta = np.full(ctx.n_events, float(fit.alpha), dtype=np.float64)
        active_eta = np.full(len(active_indices), float(fit.alpha), dtype=np.float64)
        gamma_offset = 0
        for block_index, block in enumerate(nuisance):
            width = int(block.shape[1])
            coefficients = fit.gamma[gamma_offset : gamma_offset + width]
            event_eta += block.event_values @ coefficients
            if len(active_indices):
                positions = block_positions[block_index]
                if not add_sparse_linear_predictor(
                    positions,
                    block.grid_values,
                    coefficients,
                    active_eta,
                ):
                    active_eta[positions] += block.grid_values @ coefficients
            gamma_offset += width
        if gamma_offset != len(fit.gamma):
            raise ValueError("fit/design mismatch")
        for index, (rule, block) in enumerate(
            zip(fit.rules, features, strict=True)
        ):
            coefficients = fit.theta[index]
            event_eta += float(rule.sign) * (block.event_values @ coefficients)
            if len(active_indices):
                positions = block_positions[len(nuisance) + index]
                if not add_sparse_linear_predictor(
                    positions,
                    block.grid_values,
                    coefficients,
                    active_eta,
                    scale=float(rule.sign),
                ):
                    active_eta[positions] += float(rule.sign) * (
                        block.grid_values @ coefficients
                    )

        with np.errstate(over="ignore", invalid="ignore"):
            baseline_intensity = float(np.exp(float(fit.alpha)))
            active_intensity = np.exp(active_eta)
        if (
            not math.isfinite(baseline_intensity)
            or np.any(~np.isfinite(active_intensity))
            or np.any(~np.isfinite(event_eta))
        ):
            cluster_intensity = np.full(ctx.n_sequences, math.inf, dtype=np.float64)
        else:
            cluster_intensity = (
                self.sequence_exposures(ctx) * baseline_intensity
            ).astype(np.float64, copy=False)
            if len(active_indices):
                correction = ctx.grid_weights_at(
                    active_indices, assume_valid=True
                ) * (
                    active_intensity - baseline_intensity
                )
                cluster_intensity += np.bincount(
                    ctx.grid_sequences_at(
                        active_indices,
                        assume_valid=True,
                        assume_sorted=True,
                    ),
                    weights=correction,
                    minlength=ctx.n_sequences,
                )
        if self.occurrence_likelihood == "poisson":
            event_loss = np.bincount(
                ctx.event_sequence_local,
                weights=-event_eta,
                minlength=ctx.n_sequences,
            )
            likelihood_grid = cluster_intensity
        else:
            event_hazard = np.exp(event_eta)
            likelihood_grid = cluster_intensity - np.bincount(
                ctx.event_sequence_local,
                weights=event_hazard,
                minlength=ctx.n_sequences,
            )
            event_loss = np.bincount(
                ctx.event_sequence_local,
                weights=cloglog_event_nll(event_eta),
                minlength=ctx.n_sequences,
            )
        return SparseFitSummary(
            event_eta=event_eta,
            active_grid_indices=active_indices,
            active_grid_eta=active_eta,
            cluster_intensity=cluster_intensity,
            cluster_nll=likelihood_grid + event_loss,
        )

    @staticmethod
    def _fit_summary_key(
        fit: FitResult,
        ctx: QueryContext,
    ) -> tuple[
        str,
        int,
        tuple[RuleIdentity, ...],
        tuple[ClosureTerm, ...],
    ]:
        return (
            str(ctx.name),
            id(ctx),
            tuple(fit.rules),
            tuple(fit.closure_terms),
        )

    def _prepare_fit_summary_reuse(
        self,
        records: Sequence[SupportRecord],
        ctx: QueryContext,
    ) -> None:
        """Freeze cache eligibility from the already-frozen evaluation graph."""
        usage: Counter[
            tuple[
                str,
                int,
                tuple[RuleIdentity, ...],
                tuple[ClosureTerm, ...],
            ]
        ] = Counter()
        for record in records:
            fit = record.fit
            usage[self._fit_summary_key(fit, ctx)] += 1
            if self.marked:
                usage[
                    (
                        str(ctx.name),
                        id(ctx),
                        (),
                        tuple(fit.closure_terms),
                    )
                ] += 1
            for rule in fit.rules:
                other_rules, _removed = self.hierarchy_preserving_drop(
                    fit.rules, rule
                )
                usage[
                    (
                        str(ctx.name),
                        id(ctx),
                        tuple(other_rules),
                        tuple(fit.closure_terms),
                    )
                ] += 1
        reusable = {key for key, count in usage.items() if count > 1}
        with self._fit_summary_cache_guard:
            # Freeze removals before mutating the set.  ``difference_update``
            # consumes its iterable lazily, so a generator over this same set can
            # otherwise raise ``RuntimeError: Set changed size during iteration``.
            stale_cacheable_keys = {
                key
                for key in self._fit_summary_cacheable_keys
                if key[0] == str(ctx.name) and key[1] == id(ctx)
            }
            self._fit_summary_cacheable_keys.difference_update(
                stale_cacheable_keys
            )
            self._fit_summary_cacheable_keys.update(reusable)
            self._fit_summary_cache_stats["eligible_model_keys"] = len(reusable)
            self._fit_summary_cache_stats["planned_summary_reuses"] = int(
                sum(count - 1 for count in usage.values() if count > 1)
            )

    def _cached_sparse_fit_summary(
        self,
        fit: FitResult,
        ctx: QueryContext,
    ) -> SparseFitSummary:
        """Memoize an exact full summary only when the frozen graph reuses it."""
        key = self._fit_summary_key(fit, ctx)
        with self._fit_summary_cache_guard:
            if key not in self._fit_summary_cacheable_keys:
                cacheable = False
                key_lock = None
            else:
                cacheable = True
                cached = self._fit_summary_cache.get(key)
                if cached is not None and cached[0] is fit and cached[1] is ctx:
                    self._fit_summary_cache_stats["hits"] += 1
                    self._fit_summary_cache.move_to_end(key)
                    return cached[2]
                key_lock = self._fit_summary_key_locks.setdefault(
                    key, threading.Lock()
                )
        if not cacheable or key_lock is None:
            return self._sparse_fit_summary(fit, ctx)
        with key_lock:
            with self._fit_summary_cache_guard:
                cached = self._fit_summary_cache.get(key)
                if cached is not None and cached[0] is fit and cached[1] is ctx:
                    self._fit_summary_cache_stats["hits"] += 1
                    self._fit_summary_cache.move_to_end(key)
                    return cached[2]
                self._fit_summary_cache_stats["misses"] += 1
            summary = self._sparse_fit_summary(fit, ctx)
            limit = int(self.config.fit_summary_cache_bytes)
            if limit <= 0 or summary.nbytes > limit:
                return summary
            for value in (
                summary.event_eta,
                summary.active_grid_indices,
                summary.active_grid_eta,
                summary.cluster_intensity,
                summary.cluster_nll,
            ):
                value.setflags(write=False)
            with self._fit_summary_cache_guard:
                old = self._fit_summary_cache.pop(key, None)
                if old is not None:
                    self._fit_summary_cache_size[0] -= old[2].nbytes
                self._fit_summary_cache[key] = (fit, ctx, summary)
                self._fit_summary_cache_size[0] += summary.nbytes
                while (
                    self._fit_summary_cache
                    and self._fit_summary_cache_size[0] > limit
                ):
                    _old_key, (_old_fit, _old_ctx, old_summary) = (
                        self._fit_summary_cache.popitem(last=False)
                    )
                    self._fit_summary_cache_size[0] -= old_summary.nbytes
                    self._fit_summary_cache_stats["evictions"] += 1
            return summary

    def _clear_fit_summary_context(self, ctx: QueryContext) -> None:
        """Release full summaries and reuse metadata at a split boundary."""
        with self._fit_summary_cache_guard:
            keys = [
                key
                for key, (_fit, cached_ctx, _summary) in self._fit_summary_cache.items()
                if cached_ctx is ctx
            ]
            released = 0
            for key in keys:
                _fit, _cached_ctx, summary = self._fit_summary_cache.pop(key)
                released += int(summary.nbytes)
            metadata_keys = [
                key
                for key in self._fit_summary_key_locks
                if key[0] == str(ctx.name) and key[1] == id(ctx)
            ]
            for key in metadata_keys:
                self._fit_summary_key_locks.pop(key, None)
            stale_cacheable_keys = {
                key
                for key in self._fit_summary_cacheable_keys
                if key[0] == str(ctx.name) and key[1] == id(ctx)
            }
            self._fit_summary_cacheable_keys.difference_update(
                stale_cacheable_keys
            )
            self._fit_summary_cache_size[0] -= released
            if self._fit_summary_cache_size[0] < 0:
                raise AssertionError("fit-summary cache byte accounting became negative")
            self._fit_summary_cache_stats["context_clears"] = int(
                self._fit_summary_cache_stats.get("context_clears", 0)
            ) + len(keys)
            self._fit_summary_cache_stats["released_bytes"] = int(
                self._fit_summary_cache_stats.get("released_bytes", 0)
            ) + released

    def _closure_loss_summary(
        self,
        fit: FitResult,
        ctx: QueryContext,
    ) -> SparseLossSummary:
        """Return an exact byte-bounded entity summary for a hierarchy null."""
        if fit.rules:
            raise ValueError("closure loss cache accepts null-rule fits only")
        key = (str(ctx.name), id(ctx), id(fit))
        with self._loss_summary_cache_guard:
            cached = self._loss_summary_cache.get(key)
            if cached is not None and cached[0] is fit and cached[1] is ctx:
                self._loss_summary_cache_stats["hits"] += 1
                self._loss_summary_cache.move_to_end(key)
                return cached[2]
            key_lock = self._loss_summary_key_locks.setdefault(
                key, threading.Lock()
            )
        with key_lock:
            with self._loss_summary_cache_guard:
                cached = self._loss_summary_cache.get(key)
                if cached is not None and cached[0] is fit and cached[1] is ctx:
                    self._loss_summary_cache_stats["hits"] += 1
                    self._loss_summary_cache.move_to_end(key)
                    return cached[2]
                self._loss_summary_cache_stats["misses"] += 1
            full = self._sparse_fit_summary(fit, ctx)
            summary = SparseLossSummary(
                cluster_intensity=full.cluster_intensity,
                cluster_nll=full.cluster_nll,
            )
            summary.cluster_intensity.setflags(write=False)
            summary.cluster_nll.setflags(write=False)
            limit = int(self.config.loss_summary_cache_bytes)
            if limit <= 0 or summary.nbytes > limit:
                return summary
            with self._loss_summary_cache_guard:
                old = self._loss_summary_cache.pop(key, None)
                if old is not None:
                    self._loss_summary_cache_size[0] -= old[2].nbytes
                self._loss_summary_cache[key] = (fit, ctx, summary)
                self._loss_summary_cache_size[0] += summary.nbytes
                while (
                    self._loss_summary_cache
                    and self._loss_summary_cache_size[0] > limit
                ):
                    _old_key, (_old_fit, _old_ctx, old_summary) = (
                        self._loss_summary_cache.popitem(last=False)
                    )
                    self._loss_summary_cache_size[0] -= old_summary.nbytes
                    self._loss_summary_cache_stats["evictions"] += 1
            return summary

    def _clear_loss_summary_context(self, ctx: QueryContext) -> None:
        """Release entity-loss cache entries that cannot cross split boundaries."""
        with self._loss_summary_cache_guard:
            keys = [
                key
                for key, (_fit, cached_ctx, _summary) in self._loss_summary_cache.items()
                if cached_ctx is ctx
            ]
            released = 0
            for key in keys:
                _fit, _cached_ctx, summary = self._loss_summary_cache.pop(key)
                released += int(summary.nbytes)
            lock_keys = [
                key
                for key in self._loss_summary_key_locks
                if key[0] == str(ctx.name) and key[1] == id(ctx)
            ]
            for key in lock_keys:
                self._loss_summary_key_locks.pop(key, None)
            self._loss_summary_cache_size[0] -= released
            if self._loss_summary_cache_size[0] < 0:
                raise AssertionError("loss-summary cache byte accounting became negative")
            self._loss_summary_cache_stats["context_clears"] = int(
                self._loss_summary_cache_stats.get("context_clears", 0)
            ) + len(keys)
            self._loss_summary_cache_stats["released_bytes"] = int(
                self._loss_summary_cache_stats.get("released_bytes", 0)
            ) + released

    @staticmethod
    def _summary_eta_at(
        fit: FitResult,
        summary: SparseFitSummary,
        grid_indices: np.ndarray,
    ) -> np.ndarray:
        """Gather exact sparse-summary predictors without reevaluating a model."""
        rows = np.asarray(grid_indices, dtype=np.int64).reshape(-1)
        eta = np.full(len(rows), float(fit.alpha), dtype=np.float64)
        if not len(rows) or not len(summary.active_grid_indices):
            return eta
        positions = np.searchsorted(summary.active_grid_indices, rows)
        safe = np.minimum(positions, len(summary.active_grid_indices) - 1)
        matched = (positions < len(summary.active_grid_indices)) & (
            summary.active_grid_indices[safe] == rows
        )
        eta[matched] = summary.active_grid_eta[positions[matched]]
        return eta

    def _event_grid_counts(self, ctx: QueryContext) -> tuple[np.ndarray, np.ndarray]:
        """Cache unique target-event grid rows and multiplicities per context."""
        cached = self._event_grid_count_cache.get(ctx.name)
        if cached is not None:
            return cached
        if ctx.n_events:
            rows, counts = np.unique(ctx.event_grid_rows, return_counts=True)
            value = (
                rows.astype(np.int64, copy=False),
                counts.astype(np.float64, copy=False),
            )
        else:
            value = (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
            )
        self._event_grid_count_cache[ctx.name] = value
        return value

    def _early_warning_geometry(
        self,
        ctx: QueryContext,
        rule: RuleIdentity,
        rows: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return exact rule/split geometry reused by every support contrast."""
        key = (
            str(ctx.name),
            id(ctx),
            tuple(rule.antecedent),
            int(rule.window),
            int(self.early_warning_horizon),
        )
        with self._early_warning_geometry_guard:
            cached = self._early_warning_geometry_cache.get(key)
            if cached is not None:
                if not np.array_equal(cached[0], rows):
                    raise ValueError(
                        "early-warning rule identity produced inconsistent grid rows"
                    )
                return cached
        sequences = ctx.grid_sequences_at(
            rows, assume_valid=True, assume_sorted=True
        )
        quadrature = ctx.grid_weights_at(rows, assume_valid=True)
        event_grid_rows, event_grid_counts = self._event_grid_counts(ctx)
        event_positions = np.searchsorted(rows, event_grid_rows)
        safe_positions = np.minimum(event_positions, len(rows) - 1)
        matched_events = (event_positions < len(rows)) & (
            rows[safe_positions] == event_grid_rows
        )
        event_counts = np.bincount(
            event_positions[matched_events],
            weights=event_grid_counts[matched_events],
            minlength=len(rows),
        ).astype(np.float64)
        group_starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(sequences[1:] != sequences[:-1]).astype(
                    np.int64, copy=False
                )
                + 1,
            )
        )
        active_sequences = sequences[group_starts]
        value = (
            rows,
            sequences,
            quadrature,
            event_counts,
            group_starts,
            active_sequences,
        )
        for array in value:
            array.setflags(write=False)
        with self._early_warning_geometry_guard:
            existing = self._early_warning_geometry_cache.setdefault(key, value)
            if not np.array_equal(existing[0], rows):
                raise ValueError(
                    "early-warning rule identity produced inconsistent grid rows"
                )
            return existing

    def _clear_early_warning_geometry(self, ctx: QueryContext) -> None:
        with self._early_warning_geometry_guard:
            for key in [
                key
                for key in self._early_warning_geometry_cache
                if key[0] == str(ctx.name) and key[1] == id(ctx)
            ]:
                self._early_warning_geometry_cache.pop(key, None)

    def _inference_weights(self, ctx: QueryContext) -> np.ndarray:
        cached = self._inference_weight_cache.get(ctx.name)
        if cached is not None:
            return cached
        value = (
            self.fit_sampling_weights / self.fit_sampling_scale
            if ctx.name == self.splits.fit.name
            else np.ones(ctx.n_sequences, dtype=np.float64)
        )
        self._inference_weight_cache[ctx.name] = value
        return value

    def _sparse_loss_values(
        self,
        summary: SparseFitSummary | SparseLossSummary,
        ctx: QueryContext,
    ) -> np.ndarray:
        values = self.certification_loss.weights(ctx) * summary.cluster_nll
        if ctx.name == self.splits.fit.name:
            # Horvitz-Thompson weights estimate the population total.  The
            # one-sample tests below target a population *mean*, so divide by
            # the sample mean IPW (= N_population / n_sample).  Omitting this
            # factor left zero-null p-values unchanged but mis-scaled every
            # positive materiality threshold and reported estimate.
            values = values * self._inference_weights(ctx)
        return values

    def _prepare_support_information_design(
        self,
        fit: FitResult,
        ctx: QueryContext,
        cluster_weights: np.ndarray,
    ) -> PreparedFixedSupportDesign:
        """Build one exact grouped full-support design for every rule Fisher test."""
        parent_state = self._information_design_parent
        if parent_state is not None:
            parent_ctx, parent_fit, parent_prepared = parent_state
            if parent_ctx is ctx and set(fit.rules).issubset(set(parent_fit.rules)):
                if parent_prepared is None:
                    parent_prepared = prepare_fixed_support_design(
                        ctx,
                        self.sparse_nuisance_blocks(
                            ctx, parent_fit.closure_terms
                        ),
                        tuple(
                            self.engine.sparse_response(
                                ctx, rule.antecedent, int(rule.window)
                            )
                            for rule in parent_fit.rules
                        ),
                        parent_fit.rules,
                        cluster_weights=cluster_weights,
                        sequence_exposures=self.sequence_exposures(ctx),
                        occurrence_likelihood=self.occurrence_likelihood,
                    )
                    parent_prepared = promote_prepared_design_float64(
                        parent_prepared
                    )
                    parent_state[2] = parent_prepared
                if fit.rules == parent_fit.rules:
                    return parent_prepared
                return promote_prepared_design_float64(
                    project_prepared_support_design(
                        parent_prepared,
                        fit.rules,
                        source_closure_terms=parent_fit.closure_terms,
                        target_closure_terms=fit.closure_terms,
                        regroup=True,
                    )
                )
        raw_features = tuple(
            self.engine.sparse_response(
                ctx, rule.antecedent, int(rule.window)
            )
            for rule in fit.rules
        )
        prepared = prepare_fixed_support_design(
            ctx,
            self.sparse_nuisance_blocks(ctx, fit.closure_terms),
            raw_features,
            fit.rules,
            cluster_weights=cluster_weights,
            sequence_exposures=self.sequence_exposures(ctx),
            occurrence_likelihood=self.occurrence_likelihood,
        )
        # Response values originate in float32, so promotion is exact.  Keeping
        # one promoted matrix avoids copying it inside every rule's grouped
        # Fisher calculation.
        return promote_prepared_design_float64(prepared)

    def _sparse_rule_information(
        self,
        null_fit: FitResult,
        ctx: QueryContext,
        rule_feature: SparseKernelResponse,
        nuisance_blocks: Sequence[SparseKernelResponse],
        shape: np.ndarray,
        cluster_weights: np.ndarray,
        *,
        null_summary: SparseFitSummary | None = None,
        prepared_full: PreparedFixedSupportDesign | None = None,
        focal_rule: RuleIdentity | None = None,
        remaining_rules: Sequence[RuleIdentity] | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray, float]:
        """Exact Fisher projection and cluster score on sparse sufficient statistics.

        When the full-support grouped design is supplied, its grid rows are an
        exact sufficient statistic for the Fisher Gram matrix.  Entity scores
        still use the ungrouped sparse rows below.  Grouping can therefore
        remove the expensive M-dimensional grid pass without changing either
        the projection or the independent-entity inference sample.
        """
        shape64 = np.asarray(shape, dtype=np.float64).reshape(-1)
        if shape64.shape != (rule_feature.shape[1],):
            raise ValueError("rule shape does not match sparse response")
        sequence_weights = np.asarray(cluster_weights, dtype=np.float64)
        if sequence_weights.shape != (ctx.n_sequences,):
            raise ValueError("cluster weights do not match query context")
        if (
            self.occurrence_likelihood == "first_event_cloglog"
            and prepared_full is None
        ):
            raise ValueError(
                "first-event rule information requires the exact grouped support design"
            )
        nuisance_width = 1 + sum(int(block.shape[1]) for block in nuisance_blocks)
        kernel_width = int(rule_feature.shape[1])
        fisher_chunk_rows = 262_144

        def sparse_grid_state() -> tuple[
            np.ndarray,
            np.ndarray,
            tuple[np.ndarray, ...],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
        ]:
            active_indices, layout_positions = self._sparse_union_layout(
                (rule_feature, *nuisance_blocks)
            )
            rule_positions = layout_positions[0]
            nuisance_positions = layout_positions[1:]
            eta_active = (
                self._summary_eta_at(
                    null_fit, null_summary, active_indices
                )
                if null_summary is not None
                else self._eta_on_sparse_grid(
                    null_fit, ctx, active_indices
                )
            )
            active_sequences = ctx.grid_sequences_at(
                active_indices, assume_valid=True, assume_sorted=True
            )
            active_quadrature = ctx.grid_weights_at(
                active_indices, assume_valid=True
            )
            with np.errstate(over="ignore", invalid="ignore"):
                active_intensity = np.exp(eta_active)
                active_mu = (
                    active_quadrature
                    * sequence_weights[active_sequences]
                    * active_intensity
                )
                baseline_mu = float(np.exp(float(null_fit.alpha)))
            total_baseline_weight = float(
                np.dot(self.sequence_exposures(ctx), sequence_weights)
                * baseline_mu
            )
            active_baseline_weight = float(
                baseline_mu
                * np.sum(
                    active_quadrature
                    * sequence_weights[active_sequences],
                    dtype=np.float64,
                )
            )
            inactive_weight = total_baseline_weight - active_baseline_weight
            mass_roundoff = (
                32.0
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    abs(total_baseline_weight),
                    abs(active_baseline_weight),
                )
                * max(1, len(active_indices))
            )
            if inactive_weight < -mass_roundoff:
                raise FloatingPointError(
                    "active Fisher mass exceeds total exposure mass"
                )
            return (
                active_indices,
                rule_positions,
                nuisance_positions,
                active_sequences,
                active_quadrature,
                active_intensity,
                active_mu,
                baseline_mu,
                max(0.0, inactive_weight),
            )

        if prepared_full is not None:
            if focal_rule is None or remaining_rules is None:
                raise ValueError(
                    "grouped Fisher reuse requires focal and remaining rule identities"
                )
            full_rules = tuple(prepared_full.rules)
            if focal_rule not in full_rules:
                raise ValueError("focal rule is absent from grouped full-support design")
            remaining = tuple(remaining_rules)
            if tuple(null_fit.rules) != remaining:
                raise ValueError("drop fit and grouped Fisher nuisance rules differ")
            if prepared_full.knot_count != kernel_width:
                raise ValueError("grouped Fisher kernel width differs from response")
            if len(null_fit.gamma) != prepared_full.control_width:
                raise ValueError("drop fit and grouped Fisher control widths differ")
            design = prepared_full.design.astype(np.float64, copy=False)
            constrained_start = int(prepared_full.constrained_start)
            values = np.zeros(design.shape[1], dtype=np.float64)
            values[0] = float(null_fit.alpha)
            values[1:constrained_start] = null_fit.gamma
            null_rule_index = {
                rule: index for index, rule in enumerate(null_fit.rules)
            }
            for full_index, rule in enumerate(full_rules):
                null_index = null_rule_index.get(rule)
                if null_index is None:
                    continue
                left = constrained_start + full_index * kernel_width
                values[left : left + kernel_width] = null_fit.theta[null_index]
            nuisance_columns = list(range(constrained_start))
            unsigned_column_scale = [1.0] * constrained_start
            full_rule_index = {rule: index for index, rule in enumerate(full_rules)}
            for rule in remaining:
                rule_index = full_rule_index[rule]
                left = constrained_start + rule_index * kernel_width
                nuisance_columns.extend(range(left, left + kernel_width))
                unsigned_column_scale.extend(
                    [float(rule.sign)] * kernel_width
                )
            focal_index = full_rule_index[focal_rule]
            focal_left = constrained_start + focal_index * kernel_width
            focal_columns = list(range(focal_left, focal_left + kernel_width))
            selected_columns = np.asarray(
                [*nuisance_columns, *focal_columns], dtype=np.int64
            )
            unsigned_column_scale.extend(
                [float(focal_rule.sign)] * kernel_width
            )
            grid_design = design[prepared_full.n_events :]
            with np.errstate(over="ignore", invalid="ignore"):
                grouped_mu = prepared_full.grid_weights * np.exp(
                    grid_design @ values
                )
            if np.any(~np.isfinite(grouped_mu)):
                raise FloatingPointError(
                    "nonfinite grouped intensity during Fisher projection"
                )
            # Fixed-support design columns carry their rule signs, whereas the
            # structural score projection is defined over unsigned activation
            # blocks and applies the focal direction only after entity scores
            # are formed.  Undo every rule-column sign before taking moments.
            selected_design = grid_design[:, selected_columns] * np.asarray(
                unsigned_column_scale, dtype=np.float64
            )
            grouped_information = selected_design.T @ (
                grouped_mu[:, None] * selected_design
            )
            if self.occurrence_likelihood == "first_event_cloglog":
                event_design = design[: prepared_full.n_events]
                event_eta = event_design @ values
                _loss, _gradient, event_hessian = cloglog_event_terms(event_eta)
                selected_event_design = event_design[:, selected_columns] * np.asarray(
                    unsigned_column_scale, dtype=np.float64
                )
                grouped_information += selected_event_design.T @ (
                    (prepared_full.event_weights * event_hessian)[:, None]
                    * selected_event_design
                )
            gram = grouped_information[:nuisance_width, :nuisance_width]
            rhs = grouped_information[:nuisance_width, nuisance_width:]
            raw_rule_information = grouped_information[
                nuisance_width:, nuisance_width:
            ]
        else:
            # Portable exact fallback when no grouped full-support design is
            # available.  Materialize only bounded active-grid chunks.
            (
                active_indices,
                rule_positions,
                nuisance_positions,
                active_sequences,
                active_quadrature,
                active_intensity,
                active_mu,
                baseline_mu,
                inactive_weight,
            ) = sparse_grid_state()
            gram = np.zeros((nuisance_width, nuisance_width), dtype=np.float64)
            rhs = np.zeros((nuisance_width, kernel_width), dtype=np.float64)
            raw_rule_information = np.zeros(
                (kernel_width, kernel_width), dtype=np.float64
            )
            chunk_edges = np.arange(
                0, len(active_indices), fisher_chunk_rows, dtype=np.int64
            )
            chunk_edges = np.append(chunk_edges, len(active_indices))
            nuisance_offsets = tuple(
                np.searchsorted(positions, chunk_edges)
                for positions in nuisance_positions
            )
            rule_offsets = np.searchsorted(rule_positions, chunk_edges)
            for chunk_index in range(len(chunk_edges) - 1):
                chunk_left = int(chunk_edges[chunk_index])
                chunk_right = int(chunk_edges[chunk_index + 1])
                rows = active_indices[chunk_left:chunk_right]
                f_chunk = np.zeros(
                    (len(rows), nuisance_width), dtype=np.float64
                )
                f_chunk[:, 0] = 1.0
                offset = 1
                for block, positions, boundaries in zip(
                    nuisance_blocks,
                    nuisance_positions,
                    nuisance_offsets,
                    strict=True,
                ):
                    width = int(block.shape[1])
                    stored_left = int(boundaries[chunk_index])
                    stored_right = int(boundaries[chunk_index + 1])
                    if stored_right > stored_left:
                        local_rows = (
                            positions[stored_left:stored_right].astype(
                                np.int64, copy=False
                            )
                            - chunk_left
                        )
                        f_chunk[local_rows, offset : offset + width] = (
                            block.grid_values[stored_left:stored_right]
                        )
                    offset += width
                x_chunk = np.zeros((len(rows), kernel_width), dtype=np.float64)
                stored_left = int(rule_offsets[chunk_index])
                stored_right = int(rule_offsets[chunk_index + 1])
                if stored_right > stored_left:
                    local_rows = (
                        rule_positions[stored_left:stored_right].astype(
                            np.int64, copy=False
                        )
                        - chunk_left
                    )
                    x_chunk[local_rows] = rule_feature.grid_values[
                        stored_left:stored_right
                    ]
                mu_chunk = active_mu[chunk_left:chunk_right]
                weighted_f = mu_chunk[:, None] * f_chunk
                gram += f_chunk.T @ weighted_f
                rhs += weighted_f.T @ x_chunk
                raw_rule_information += x_chunk.T @ (
                    mu_chunk[:, None] * x_chunk
                )
            gram[0, 0] += inactive_weight
        # One SVD supplies both rank and the Moore-Penrose solve.  Calling
        # ``svd`` and then ``pinv`` decomposed the same nuisance Gram matrix
        # twice for every rule in every support.
        left, singular, right = np.linalg.svd(gram, full_matrices=False)
        if singular.size and singular[0] > 0.0:
            tolerance = np.finfo(np.float64).eps * max(gram.shape) * singular[0]
            rank = int(np.sum(singular > tolerance))
            inverse = np.divide(
                1.0,
                singular,
                out=np.zeros_like(singular),
                where=singular > tolerance,
            )
            coefficients = right.T @ (inverse[:, None] * (left.T @ rhs))
        else:
            rank = 0
            coefficients = np.zeros((nuisance_width, kernel_width))

        projected_coefficients = coefficients @ shape64
        projected_event = (
            rule_feature.event_values.astype(np.float64, copy=False) @ shape64
        )
        projected_event -= projected_coefficients[0]
        offset = 1
        for block in nuisance_blocks:
            width = int(block.shape[1])
            projected_event -= (
                block.event_values
                @ projected_coefficients[offset : offset + width]
            )
            offset += width
        projected_inactive = -float(projected_coefficients[0])
        # Weighted least-squares projection identity:
        #   (X-F G^+ F'WX)' W (X-F G^+ F'WX)
        #       = X'WX - (F'WX)' G^+ (F'WX).
        # ``coefficients`` is exactly G^+ F'WX from the SVD above.  Reusing
        # these sufficient statistics eliminates a second M-dimensional
        # residual construction while preserving the same Fisher matrix.
        information = raw_rule_information - rhs.T @ coefficients
        information = 0.5 * (information + information.T)
        if self.occurrence_likelihood == "poisson":
            event_score_values = projected_event
        else:
            event_eta_for_score = (
                null_summary.event_eta
                if null_summary is not None
                else self._eta_on_events(null_fit, ctx)
            )
            _loss, event_nll_gradient, _event_hessian = cloglog_event_terms(
                event_eta_for_score
            )
            event_score_values = -event_nll_gradient * projected_event
        event_score = np.bincount(
            ctx.event_sequence_local,
            weights=event_score_values,
            minlength=ctx.n_sequences,
        ).astype(np.float64)
        raw_information = float(shape64 @ raw_rule_information @ shape64)
        if prepared_full is not None and null_summary is not None:
            # Linearity permits integrating the projected scalar one sparse
            # component at a time.  The intercept integral is already the
            # exact per-entity compensator in ``null_summary``.  This avoids
            # rebuilding a multi-block union solely for the entity score while
            # preserving every entity as a separate inference observation.
            grid_score = (
                projected_inactive * null_summary.cluster_intensity.copy()
            )

            def add_component(
                block: SparseKernelResponse,
                component_coefficients: np.ndarray,
                scale: float,
            ) -> None:
                rows = block.grid_indices
                if not len(rows):
                    return
                implicit_unit_weights = isinstance(
                    ctx.grid_weights, ImplicitUnitGridWeights
                )
                quadrature = (
                    None
                    if implicit_unit_weights
                    else ctx.grid_weights_at(rows, assume_valid=True)
                )
                native = sparse_component_integral(
                    null_summary.active_grid_indices,
                    null_summary.active_grid_eta,
                    float(null_fit.alpha),
                    rows,
                    block.grid_values,
                    component_coefficients,
                    quadrature,
                    ctx.grid_offsets,
                    assume_sorted=True,
                )
                if native is not None:
                    grid_score[:] += float(scale) * native
                    return
                eta = self._summary_eta_at(null_fit, null_summary, rows)
                if quadrature is None:
                    quadrature = np.ones(len(rows), dtype=np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    row_mass = (
                        quadrature * np.exp(eta)
                        * (block.grid_values @ component_coefficients)
                    )
                if np.any(~np.isfinite(row_mass)):
                    raise FloatingPointError(
                        "nonfinite sparse component during entity score"
                    )
                grid_score[:] += float(scale) * np.bincount(
                    ctx.grid_sequences_at(
                        rows, assume_valid=True, assume_sorted=True
                    ),
                    weights=row_mass,
                    minlength=ctx.n_sequences,
                )

            offset = 1
            for block in nuisance_blocks:
                width = int(block.shape[1])
                add_component(
                    block,
                    projected_coefficients[offset : offset + width],
                    -1.0,
                )
                offset += width
            add_component(rule_feature, shape64, 1.0)
            if self.occurrence_likelihood == "first_event_cloglog":
                event_eta_for_score = null_summary.event_eta
                grid_score -= np.bincount(
                    ctx.event_sequence_local,
                    weights=np.exp(event_eta_for_score) * projected_event,
                    minlength=ctx.n_sequences,
                )
            cluster_score = sequence_weights * (event_score - grid_score)
            return information, rank, cluster_score, raw_information

        if prepared_full is not None:
            # Grouped moments without a fit summary use the portable ungrouped
            # entity-score path.  Production evaluation always supplies the
            # summary; this branch preserves the public helper's generality.
            (
                active_indices,
                rule_positions,
                nuisance_positions,
                active_sequences,
                active_quadrature,
                active_intensity,
                _active_mu,
                baseline_mu,
                _inactive_weight,
            ) = sparse_grid_state()
        active_grid_score = np.zeros(ctx.n_sequences, dtype=np.float64)
        chunk_edges = np.arange(
            0, len(active_indices), fisher_chunk_rows, dtype=np.int64
        )
        chunk_edges = np.append(chunk_edges, len(active_indices))
        nuisance_offsets = tuple(
            np.searchsorted(positions, chunk_edges)
            for positions in nuisance_positions
        )
        rule_offsets = np.searchsorted(rule_positions, chunk_edges)
        for chunk_index in range(len(chunk_edges) - 1):
            chunk_left = int(chunk_edges[chunk_index])
            chunk_right = int(chunk_edges[chunk_index + 1])
            rows = active_indices[chunk_left:chunk_right]
            projected_chunk = np.full(
                len(rows), projected_inactive, dtype=np.float64
            )
            offset = 1
            for block, positions, boundaries in zip(
                nuisance_blocks,
                nuisance_positions,
                nuisance_offsets,
                strict=True,
            ):
                width = int(block.shape[1])
                stored_left = int(boundaries[chunk_index])
                stored_right = int(boundaries[chunk_index + 1])
                if stored_right > stored_left:
                    local_rows = (
                        positions[stored_left:stored_right].astype(
                            np.int64, copy=False
                        )
                        - chunk_left
                    )
                    projected_chunk[local_rows] -= (
                        block.grid_values[stored_left:stored_right]
                        @ projected_coefficients[offset : offset + width]
                    )
                offset += width
            stored_left = int(rule_offsets[chunk_index])
            stored_right = int(rule_offsets[chunk_index + 1])
            if stored_right > stored_left:
                local_rows = (
                    rule_positions[stored_left:stored_right].astype(
                        np.int64, copy=False
                    )
                    - chunk_left
                )
                projected_chunk[local_rows] += (
                    rule_feature.grid_values[stored_left:stored_right] @ shape64
                )
            active_grid_score += np.bincount(
                active_sequences[chunk_left:chunk_right],
                weights=(
                    active_quadrature[chunk_left:chunk_right]
                    * active_intensity[chunk_left:chunk_right]
                    * projected_chunk
                ),
                minlength=ctx.n_sequences,
            ).astype(np.float64, copy=False)
        active_exposure = np.zeros(ctx.n_sequences, dtype=np.float64)
        if len(active_indices):
            active_exposure = np.bincount(
                active_sequences,
                weights=active_quadrature,
                minlength=ctx.n_sequences,
            ).astype(np.float64, copy=False)
        inactive_exposure = self.sequence_exposures(ctx) - active_exposure
        grid_score = active_grid_score + (
            inactive_exposure * baseline_mu * projected_inactive
        )
        cluster_score = sequence_weights * (event_score - grid_score)
        return information, rank, cluster_score, raw_information

    def _fitted_rule_shape(self, fit: FitResult, rule_index: int) -> np.ndarray:
        rule = fit.rules[rule_index]
        dictionary_shape = self.rule_dictionary_shapes.get(rule)
        if dictionary_shape is not None and fit.theta.shape[1] == 1:
            return dictionary_shape.astype(np.float64, copy=True)
        return fit.shapes[rule_index].astype(np.float64, copy=True)

    def _expanded_rule_theta(self, fit: FitResult, rule_index: int) -> np.ndarray:
        shape = self._fitted_rule_shape(fit, rule_index)
        return float(fit.amplitudes[rule_index]) * shape

    def _loss_values(self, eta: np.ndarray, ctx: QueryContext) -> np.ndarray:
        values = self.certification_loss.values(eta, ctx)
        if ctx.name == self.splits.fit.name:
            values = values * self._inference_weights(ctx)
        return values

    def _kernel_impact_report(
        self,
        rule: RuleIdentity,
        theta: np.ndarray,
        information: np.ndarray,
    ) -> dict:
        theta = np.asarray(theta, dtype=np.float64)
        unsigned_curve = theta @ self.engine.basis64
        signed_curve = float(rule.sign) * unsigned_curve
        total = float(np.sum(theta))
        normalized = np.divide(theta, total, out=np.zeros_like(theta), where=total > 0)
        eigenvalues = np.linalg.eigvalsh(information) if information.size else np.zeros(0)
        largest = float(np.max(eigenvalues, initial=0.0))
        rank_tolerance = np.finfo(np.float64).eps * max(information.shape, default=1) * max(1.0, largest)
        identifiable_rank = int(np.sum(eigenvalues > rank_tolerance))
        if unsigned_curve.size and float(np.sum(unsigned_curve)) > 0:
            lags = np.arange(1, len(unsigned_curve) + 1, dtype=np.float64)
            center = float(np.dot(lags, unsigned_curve) / np.sum(unsigned_curve))
            peak_index = int(np.argmax(unsigned_curve))
            peak_lag = peak_index + 1
            peak_impact = float(signed_curve[peak_index])
        else:
            center = None
            peak_lag = None
            peak_impact = 0.0
        return {
            "normalized_knot_shape": normalized.tolist(),
            "signed_log_intensity_curve": signed_curve.tolist(),
            "integrated_log_intensity_impact": float(rule.sign) * total,
            "peak_log_intensity_impact": peak_impact,
            "peak_lag": peak_lag,
            "impact_center_of_mass_lag": center,
            "identifiable_fisher_rank": identifiable_rank,
            "kernel_shape_resolved": bool(total > 0 and identifiable_rank == len(theta)),
            "uncertainty_bands_available": False,
            "uncertainty_note": (
                "Selection-aware constrained cluster-bootstrap bands are not computed in this run; "
                "curve values are frozen point estimates."
            ),
        }

    def _early_warning_rule_report(
        self,
        fit: FitResult,
        drop_fit: FitResult,
        summary_full: SparseFitSummary,
        summary_drop: SparseFitSummary,
        ctx: QueryContext,
        rule_index: int,
        *,
        alpha: float,
    ) -> dict:
        """Evaluate one frozen rule's prospective event-risk footprint.

        For each independent entity, let A_r(H) be the union of grid cells
        reached one through H lags after an eligible rule completion.  The
        fitted full intensity is compared with the frozen hierarchy-preserving
        branch-drop predictor inside A_r(H) to score observed held-out events.
        Direction and magnitude instead remove only r's frozen term, so a valid
        A-excitation/AB-inhibition support is not assigned the net sign of the
        entire hierarchy branch.  Neither contrast is a causal intervention.
        """
        rule = fit.rules[rule_index]
        response = self.engine.sparse_horizon_response(
            ctx,
            rule.antecedent,
            rule.window,
            self.early_warning_horizon,
        )
        rows = response.grid_indices
        if not len(rows):
            invalid = one_sided_mean_test(
                [],
                null=self.config.early_warning_threshold,
                alpha=alpha,
            )
            return {
                "horizon": self.early_warning_horizon,
                "estimand": "hierarchy_branch_early_warning_contribution",
                "horizon_predictive_contribution": _mean_test_dict(invalid),
                "sign_aligned_probability_shift": _mean_test_dict(invalid),
                "p_value": 1.0,
                "testable": False,
                "activated_sequences": 0,
                "active_grid_rows": 0,
                "invalid_reason": "the rule reaches no future grid cell on this split",
                "causal_interpretation": False,
            }

        theta = self._expanded_rule_theta(fit, rule_index)
        signed_rule_effect = float(rule.sign) * (
            response.grid_values.astype(np.float64) @ theta
        )
        eta_full = self._summary_eta_at(fit, summary_full, rows)
        eta_drop = self._summary_eta_at(drop_fit, summary_drop, rows)
        (
            _geometry_rows,
            sequences,
            quadrature,
            event_counts,
            group_starts,
            active_sequences,
        ) = self._early_warning_geometry(ctx, rule, rows)
        activated_count = int(len(active_sequences))
        with np.errstate(over="ignore", invalid="ignore"):
            full_mass_rows = quadrature * np.exp(eta_full)
            drop_mass_rows = quadrature * np.exp(eta_drop)
            no_rule_mass_rows = quadrature * np.exp(
                eta_full - signed_rule_effect
            )
        if (
            np.any(~np.isfinite(full_mass_rows))
            or np.any(~np.isfinite(drop_mass_rows))
            or np.any(~np.isfinite(no_rule_mass_rows))
            or np.any(~np.isfinite(eta_full))
            or np.any(~np.isfinite(eta_drop))
        ):
            invalid = one_sided_mean_test_zero_padded(
                [math.nan],
                total_count=ctx.n_sequences,
                null=self.config.early_warning_threshold,
                alpha=alpha,
            )
            return {
                "horizon": self.early_warning_horizon,
                "estimand": "hierarchy_branch_early_warning_contribution",
                "horizon_predictive_contribution": _mean_test_dict(invalid),
                "sign_aligned_probability_shift": _mean_test_dict(invalid),
                "p_value": 1.0,
                "testable": False,
                "activated_sequences": int(
                    len(
                        np.unique(
                            sequences
                        )
                    )
                ),
                "active_grid_rows": int(len(rows)),
                "invalid_reason": "nonfinite frozen intensity contrast",
                "causal_interpretation": False,
            }

        # Rows and their sequence labels are sorted.  Reduce only the entities
        # reached by this rule, and account for every unreached entity as an
        # implicit zero in the exact one-sample tests below.  The previous
        # minlength=n_sequences bincounts allocated several dense vectors per
        # rule, including an entirely unused drop-mass vector.
        full_mass = np.add.reduceat(full_mass_rows, group_starts)
        no_rule_mass = np.add.reduceat(no_rule_mass_rows, group_starts)
        if self.occurrence_likelihood == "poisson":
            row_loss_full = full_mass_rows - event_counts * eta_full
            row_loss_drop = drop_mass_rows - event_counts * eta_drop
        else:
            row_loss_full = full_mass_rows + event_counts * (
                cloglog_event_nll(eta_full) - full_mass_rows
            )
            row_loss_drop = drop_mass_rows + event_counts * (
                cloglog_event_nll(eta_drop) - drop_mass_rows
            )
        horizon_loss_contribution = np.add.reduceat(
            row_loss_drop - row_loss_full, group_starts
        )
        inference_weights = self._inference_weights(ctx)
        active_inference_weights = inference_weights[active_sequences]
        horizon_test = one_sided_mean_test_zero_padded(
            active_inference_weights * horizon_loss_contribution,
            total_count=ctx.n_sequences,
            null=self.config.rule_threshold,
            alpha=alpha,
        )
        # Conditional on the predictable fitted intensity, exp(-Lambda) is the
        # no-event probability over the specified union of future cells.
        probability_full = -np.expm1(-full_mass)
        probability_no_rule = -np.expm1(-no_rule_mass)
        raw_probability_shift = probability_full - probability_no_rule
        signed_probability_shift = float(rule.sign) * raw_probability_shift
        signed_expected_count_shift = float(rule.sign) * (full_mass - no_rule_mass)
        risk_test = one_sided_mean_test_zero_padded(
            active_inference_weights * signed_probability_shift,
            total_count=ctx.n_sequences,
            null=self.config.early_warning_threshold,
            alpha=alpha,
        )
        p_value = max(float(horizon_test.p_value), float(risk_test.p_value))
        return {
            "horizon": self.early_warning_horizon,
            "estimand": "hierarchy_branch_early_warning_contribution",
            "threshold_probability_points": self.config.early_warning_threshold,
            "horizon_predictive_contribution": _mean_test_dict(horizon_test),
            "sign_aligned_probability_shift": _mean_test_dict(risk_test),
            "p_value": p_value,
            "testable": bool(ctx.n_sequences >= 2 and activated_count > 0),
            "activated_sequences": activated_count,
            "activation_coverage": float(activated_count / ctx.n_sequences),
            "active_grid_rows": int(len(rows)),
            "raw_probability_shift": float(
                np.sum(
                    active_inference_weights * raw_probability_shift,
                    dtype=np.float64,
                )
                / ctx.n_sequences
            ),
            "fit_screen_ipw_hajek_normalized": bool(
                ctx.name == self.splits.fit.name
            ),
            "signed_expected_target_count_shift": float(
                np.sum(
                    active_inference_weights * signed_expected_count_shift,
                    dtype=np.float64,
                )
                / ctx.n_sequences
            ),
            "interpretation": (
                "The branch must improve observed horizon-local point-process loss. Positive raw "
                "probability shift means excitation raises event risk; negative raw shift means "
                "inhibition lowers event risk. Certification also requires the sign-aligned shift."
            ),
            "causal_interpretation": False,
        }

    @staticmethod
    def _rule_irreducibility_p_value(
        global_drop_p_value: float,
        early_warning: dict | None,
    ) -> float:
        """Intersection-union p-value for one hierarchy-preserving branch."""
        global_value = float(global_drop_p_value)
        return (
            max(global_value, float(early_warning["p_value"]))
            if early_warning is not None
            else global_value
        )

    def _marked_losses(
        self,
        fit: FitResult,
        ctx: QueryContext,
        eta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if fit.mark_fit is None:
            raise ValueError("marked support is missing its conditional mark head")
        nuisance = self.nuisance_event_design(ctx, fit.closure_terms)
        activations = self.mark_rule_activations(fit, ctx, event_only=True)
        occurrence = cluster_nll(eta, ctx)
        mark_nll = cluster_mark_nll(fit.mark_fit, ctx, nuisance, activations)
        mark_mean = self._mark_mean_on(fit, ctx)
        financial, observed, predicted = cluster_financial_mean_loss(
            eta,
            mark_mean,
            ctx,
            unit=float(fit.mark_fit.unit),
        )
        if ctx.name == self.splits.fit.name:
            ipw = self._inference_weights(ctx)
            occurrence = occurrence * ipw
            mark_nll = mark_nll * ipw
            financial = financial * ipw
        return occurrence, mark_nll, financial, observed, predicted

    def _sparse_marked_losses(
        self,
        fit: FitResult,
        ctx: QueryContext,
        summary: SparseFitSummary,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Marked cluster losses from sparse event/active-grid sufficient statistics."""
        if fit.mark_fit is None or ctx.event_marks is None:
            raise ValueError("marked support is missing marks or its conditional head")
        head = fit.mark_fit
        nuisance = self.sparse_nuisance_blocks(ctx, fit.closure_terms)
        features = self.sparse_features(ctx, fit.rules)
        event_mu = np.full(ctx.n_events, float(head.intercept), dtype=np.float64)
        active_mu = np.full(
            len(summary.active_grid_indices), float(head.intercept), dtype=np.float64
        )
        offset = 0
        for block in nuisance:
            width = int(block.shape[1])
            coefficients = head.nuisance_beta[offset : offset + width]
            event_mu += block.event_values @ coefficients
            if len(summary.active_grid_indices):
                block.add_grid_linear_predictor(
                    summary.active_grid_indices,
                    coefficients,
                    active_mu,
                    assume_sorted_unique=True,
                )
            offset += width
        if offset != len(head.nuisance_beta):
            raise ValueError("mark nuisance fit/design mismatch")
        if len(head.rule_beta) != len(features):
            raise ValueError("mark rule fit/design mismatch")
        for index, block in enumerate(features):
            shape = fit.shapes[index]
            event_mu += float(head.rule_beta[index]) * (
                block.event_values @ shape
            )
            if len(summary.active_grid_indices):
                block.add_grid_linear_predictor(
                    summary.active_grid_indices,
                    shape,
                    active_mu,
                    scale=float(head.rule_beta[index]),
                    assume_sorted_unique=True,
                )

        response_key = (str(ctx.name), float(head.unit))
        marked_response = self._marked_response_cache.get(response_key)
        if marked_response is None:
            marks = np.asarray(ctx.event_marks, dtype=np.float64)
            y = np.log(marks / float(head.unit))
            log_marks = np.log(marks)
            observed = np.bincount(
                ctx.event_sequence_local,
                weights=marks,
                minlength=ctx.n_sequences,
            ).astype(np.float64)
            marks.setflags(write=False)
            y.setflags(write=False)
            log_marks.setflags(write=False)
            observed.setflags(write=False)
            marked_response = (marks, y, log_marks, observed)
            self._marked_response_cache[response_key] = marked_response
        marks, y, log_marks, observed = marked_response
        event_mark_nll = (
            0.5
            * (
                math.log(2.0 * math.pi * float(head.variance))
                + (y - event_mu) ** 2 / float(head.variance)
            )
            + log_marks
        )
        mark_nll = np.bincount(
            ctx.event_sequence_local,
            weights=event_mark_nll,
            minlength=ctx.n_sequences,
        ).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            baseline_rate = float(head.unit) * float(
                np.exp(float(fit.alpha) + float(head.intercept) + 0.5 * float(head.variance))
            )
            active_rate = float(head.unit) * np.exp(
                summary.active_grid_eta + active_mu + 0.5 * float(head.variance)
            )
        predicted = self.sequence_exposures(ctx) * baseline_rate
        if len(summary.active_grid_indices):
            correction = ctx.grid_weights_at(
                summary.active_grid_indices, assume_valid=True
            ) * (
                active_rate - baseline_rate
            )
            predicted += np.bincount(
                ctx.grid_sequences_at(
                    summary.active_grid_indices,
                    assume_valid=True,
                    assume_sorted=True,
                ),
                weights=correction,
                minlength=ctx.n_sequences,
            )
        financial = ((observed - predicted) / float(head.unit)) ** 2
        occurrence = summary.cluster_nll.copy()
        if ctx.name == self.splits.fit.name:
            ipw = self._inference_weights(ctx)
            occurrence *= ipw
            mark_nll *= ipw
            financial *= ipw
        return occurrence, mark_nll, financial, observed, predicted

    def _mark_mean_on(self, fit: FitResult, ctx: QueryContext) -> np.ndarray:
        """Predict marked means blockwise without a full nuisance matrix."""
        if fit.mark_fit is None:
            raise ValueError("marked support is missing its conditional mark head")
        head = fit.mark_fit
        mu = np.full(ctx.n_queries, float(head.intercept), dtype=np.float64)
        offset = 0
        for block in self.nuisance_blocks(ctx, fit.closure_terms):
            width = int(block.shape[1])
            mu += block @ head.nuisance_beta[offset : offset + width]
            offset += width
        if offset != len(head.nuisance_beta):
            raise ValueError("mark nuisance fit/design mismatch")
        if len(head.rule_beta) != len(fit.rules):
            raise ValueError("mark rule fit/design mismatch")
        for index, feature in enumerate(self.features(ctx, fit.rules)):
            activation = feature @ fit.shapes[index]
            mu += float(head.rule_beta[index]) * activation
        with np.errstate(over="ignore", invalid="ignore"):
            return float(head.unit) * np.exp(mu + 0.5 * float(head.variance))

    def _evaluate_marked_supports(
        self,
        records: Sequence[SupportRecord],
        ctx: QueryContext,
        *,
        alpha: float,
        short_circuit_alpha: float | None,
    ) -> list[dict]:
        """Certify occurrence rules by occurrence and monetary exposure jointly.

        A reported rule must improve held-out event-time likelihood and the
        conditional mean of cumulative marked exposure.  Consequently a
        mark-only association can never pass, while a rule can be financially
        useful through occurrence frequency, conditional amount, or both.
        """
        if ctx.event_marks is None:
            raise ValueError("marked certification split has no event marks")
        raw_score_weights = (
            self.fit_sampling_weights
            if ctx.name == self.splits.fit.name
            else np.ones(ctx.n_sequences, dtype=np.float64)
        )
        score_weights = raw_score_weights / float(np.mean(raw_score_weights))
        def failed_item(record: SupportRecord, reason: str) -> dict:
            fit = record.fit
            return {
                "support": [self._rule_dict(rule) for rule in fit.rules],
                "search_joint_nll": float(fit.nll),
                "search_intensity_nll": (
                    float(fit.intensity_nll) if fit.intensity_nll is not None else None
                ),
                "fit_kkt_residual": float(fit.kkt_residual),
                "fit_converged": bool(fit.converged),
                "all_rule_blocks_active": False,
                "structurally_testable": False,
                "rules": [],
                "p_value": 1.0,
                "invalid_reason": reason,
            }

        items: list[dict] = []
        for record in records:
            # Summaries contain O(n_sequences + active_grid_rows) arrays.  A
            # run-wide cache retained several such arrays for every searched
            # support (tens of GB on IBM) even though later records do not need
            # them.  The only guaranteed reuse is within the current support
            # (full/null/drop comparisons), so give the cache that exact
            # lifetime.  This changes no fitted model or statistic.
            summary_cache: dict[int, SparseFitSummary] = {}
            information_design: PreparedFixedSupportDesign | None = None

            def summary_for(candidate: FitResult) -> SparseFitSummary:
                cached = summary_cache.get(id(candidate))
                if cached is None:
                    cached = self._cached_sparse_fit_summary(candidate, ctx)
                    summary_cache[id(candidate)] = cached
                return cached

            fit = record.fit
            closure_baseline = self.fit_model((), fit.closure_terms)
            if (
                not fit.converged
                or not closure_baseline.converged
                or fit.mark_fit is None
                or closure_baseline.mark_fit is None
                or not fit.mark_fit.converged
                or not closure_baseline.mark_fit.converged
            ):
                items.append(
                    failed_item(
                        record,
                        "full or hierarchy-preserving branch null fit did not converge",
                    )
                )
                continue
            active = bool(np.all(self._active_amplitudes(fit)))
            if not active:
                items.append(
                    failed_item(record, "at least one occurrence rule block is on the zero boundary")
                )
                continue

            summary_full = summary_for(fit)
            summary_null = summary_for(closure_baseline)
            occ_full, mark_full, financial_full, observed, predicted_full = self._sparse_marked_losses(
                fit, ctx, summary_full
            )
            occ_null, mark_null, financial_null, _observed_null, predicted_null = self._sparse_marked_losses(
                closure_baseline, ctx, summary_null
            )
            joint_occurrence = one_sided_mean_test(
                occ_null - occ_full,
                null=0.0,
                alpha=alpha,
            )
            joint_financial = one_sided_mean_test(
                financial_null - financial_full,
                null=self.config.financial_threshold,
                alpha=alpha,
            )
            joint_mark_diagnostic = one_sided_mean_test(
                mark_null - mark_full,
                null=0.0,
                alpha=alpha,
            )
            joint_p = max(joint_occurrence.p_value, joint_financial.p_value)
            if short_circuit_alpha is not None and joint_p > short_circuit_alpha:
                items.append(
                    {
                        "support": [self._rule_dict(rule) for rule in fit.rules],
                        "search_joint_nll": float(fit.nll),
                        "search_nll_improvement": float(record.search_nll_improvement),
                        "fit_kkt_residual": float(fit.kkt_residual),
                        "fit_converged": True,
                        "all_rule_blocks_active": True,
                        "joint_occurrence_contribution": _mean_test_dict(joint_occurrence),
                        "joint_financial_exposure_contribution": _mean_test_dict(joint_financial),
                        "joint_conditional_mark_nll_diagnostic": _mean_test_dict(
                            joint_mark_diagnostic
                        ),
                        "rules": [],
                        "structurally_testable": False,
                        "p_value": 1.0,
                        "evaluation_short_circuited": True,
                        "short_circuit_reason": (
                            "the support already failed occurrence or cumulative financial-exposure "
                            "contribution; branch-null tests cannot rescue the conjunction"
                        ),
                    }
                )
                continue

            rule_items: list[dict] = []
            p_values = [float(joint_occurrence.p_value), float(joint_financial.p_value)]
            structural = True
            conditional_short_circuit = False
            mark_design_dimension = (
                1 + len(fit.mark_fit.nuisance_beta) + len(fit.mark_fit.rule_beta)
            )
            mark_coefficients_identifiable = bool(
                fit.mark_fit.rank == mark_design_dimension
            )
            for rule_index, rule in enumerate(fit.rules):
                other_rules, removed_rules = self.hierarchy_preserving_drop(fit.rules, rule)
                drop_key = (other_rules, tuple(fit.closure_terms))
                if (
                    ctx.name == self.splits.fit.name
                    and drop_key not in self._fit_cache
                    and drop_key not in self._prepared_design_cache
                ):
                    if information_design is None:
                        information_design = self._prepare_support_information_design(
                            fit, ctx, score_weights
                        )
                    self._prepared_design_cache[drop_key] = (
                        project_prepared_support_design(
                            information_design, other_rules
                        )
                    )
                drop_fit = self.fit_model(other_rules, fit.closure_terms, initial=fit)
                # A concurrent worker may have published the same canonical
                # drop fit before this worker acquired its per-key lock.  In
                # that case the locally projected prepared design was never
                # consumed and must not accumulate for the rest of the stage.
                self._prepared_design_cache.pop(drop_key, None)
                if not drop_fit.converged or drop_fit.mark_fit is None or not drop_fit.mark_fit.converged:
                    structural = False
                    rule_items.append(
                        {
                            "rule": self._rule_dict(rule),
                            "hierarchy_branch_removed": [
                                self._rule_dict(item) for item in removed_rules
                            ],
                            "testable": False,
                            "p_value": 1.0,
                            "invalid_reason": "hierarchy-preserving branch-drop fit did not converge",
                        }
                    )
                    p_values.append(1.0)
                    if (
                        ctx.name == self.splits.fit.name
                        and short_circuit_alpha is not None
                    ):
                        conditional_short_circuit = True
                        for remaining_index in range(
                            rule_index + 1, len(fit.rules)
                        ):
                            rule_items.append(
                                {
                                    "rule": self._rule_dict(
                                        fit.rules[remaining_index]
                                    ),
                                    "support_amplitude": float(
                                        fit.amplitudes[remaining_index]
                                    ),
                                    "shape": self._fitted_rule_shape(
                                        fit, remaining_index
                                    ).tolist(),
                                    "conditional_gates_evaluated": False,
                                    "p_value": 1.0,
                                    "short_circuit_reason": (
                                        "an earlier D_fit marked branch-drop fit "
                                        "made support selection impossible"
                                    ),
                                }
                            )
                        break
                    continue
                summary_drop = summary_for(drop_fit)
                occ_drop, mark_drop, financial_drop, _observed_drop, predicted_drop = (
                    self._sparse_marked_losses(drop_fit, ctx, summary_drop)
                )
                occurrence_contribution = one_sided_mean_test(
                    occ_drop - occ_full,
                    null=0.0,
                    alpha=alpha,
                )
                financial_contribution = one_sided_mean_test(
                    financial_drop - financial_full,
                    null=self.config.rule_threshold,
                    alpha=alpha,
                )
                mark_diagnostic = one_sided_mean_test(
                    mark_drop - mark_full,
                    null=0.0,
                    alpha=alpha,
                )
                shape = self._fitted_rule_shape(fit, rule_index)
                rule_p = max(
                    float(occurrence_contribution.p_value),
                    float(financial_contribution.p_value),
                )
                if (
                    ctx.name == self.splits.fit.name
                    and short_circuit_alpha is not None
                    and rule_p > short_circuit_alpha
                ):
                    p_values.append(1.0)
                    structural = False
                    conditional_short_circuit = True
                    invalid_directional = one_sided_mean_test(
                        [], null=0.0, alpha=alpha
                    )
                    rule_items.append(
                        {
                            "rule": self._rule_dict(rule),
                            "hierarchy_branch_removed": [
                                self._rule_dict(item) for item in removed_rules
                            ],
                            "conditional_comparison": "hierarchy_preserving_branch_drop",
                            "support_amplitude": float(fit.amplitudes[rule_index]),
                            "shape": shape.tolist(),
                            "kernel_impact": None,
                            "effective_information": None,
                            "nuisance_rank": None,
                            "testable": False,
                            "structural_testability_evaluated": False,
                            "directional_occurrence_diagnostic": _mean_test_dict(
                                invalid_directional
                            ),
                            "hierarchy_branch_occurrence_contribution": _mean_test_dict(
                                occurrence_contribution
                            ),
                            "hierarchy_branch_financial_exposure_contribution": _mean_test_dict(
                                financial_contribution
                            ),
                            "hierarchy_branch_conditional_mark_nll_diagnostic": _mean_test_dict(
                                mark_diagnostic
                            ),
                            "mean_predicted_exposure_full": float(
                                np.mean(predicted_full)
                            ),
                            "mean_predicted_exposure_branch_null": float(
                                np.mean(predicted_drop)
                            ),
                            "p_value": 1.0,
                            "conditional_gates_evaluated": False,
                            "short_circuit_reason": (
                                "this D_fit marked rule condition already failed the conjunction"
                            ),
                        }
                    )
                    for remaining_index in range(rule_index + 1, len(fit.rules)):
                        rule_items.append(
                            {
                                "rule": self._rule_dict(fit.rules[remaining_index]),
                                "support_amplitude": float(
                                    fit.amplitudes[remaining_index]
                                ),
                                "shape": self._fitted_rule_shape(
                                    fit, remaining_index
                                ).tolist(),
                                "conditional_gates_evaluated": False,
                                "p_value": 1.0,
                            }
                        )
                    if drop_fit is not fit and drop_fit is not closure_baseline:
                        summary_cache.pop(id(drop_fit), None)
                    break
                raw_feature = self.engine.sparse_response(
                    ctx, rule.antecedent, rule.window
                )
                if information_design is None:
                    information_design = self._prepare_support_information_design(
                        fit, ctx, score_weights
                    )
                nuisance = (
                    *self.sparse_nuisance_blocks(ctx, fit.closure_terms),
                    *self.sparse_features(ctx, other_rules),
                )
                information_matrix, nuisance_rank, score_values, raw_information = (
                    self._sparse_rule_information(
                        drop_fit,
                        ctx,
                        raw_feature,
                        nuisance,
                        shape,
                        score_weights,
                        null_summary=summary_drop,
                        prepared_full=information_design,
                        focal_rule=rule,
                        remaining_rules=other_rules,
                    )
                )
                information = float(shape @ information_matrix @ shape)
                testable = bool(
                    ctx.n_sequences >= 2
                    and numeric_information_positive_from_raw(
                        information,
                        raw_information,
                        ctx.n_grid,
                    )
                )
                directional = (
                    one_sided_mean_test(
                        float(rule.sign) * score_values,
                        null=0.0,
                        alpha=alpha,
                    )
                    if testable
                    else one_sided_mean_test([], null=0.0, alpha=alpha)
                )
                p_values.extend(
                    [
                        float(occurrence_contribution.p_value),
                        float(financial_contribution.p_value),
                    ]
                )
                structural = structural and testable
                rule_items.append(
                    {
                        "rule": self._rule_dict(rule),
                        "hierarchy_branch_removed": [
                            self._rule_dict(item) for item in removed_rules
                        ],
                        "conditional_comparison": "hierarchy_preserving_branch_drop",
                        "support_amplitude": float(fit.amplitudes[rule_index]),
                        "conditional_mark_coefficient": (
                            float(fit.mark_fit.rule_beta[rule_index])
                            if mark_coefficients_identifiable
                            else None
                        ),
                        "conditional_mark_coefficient_identifiable": (
                            mark_coefficients_identifiable
                        ),
                        "shape": shape.tolist(),
                        "kernel_impact": self._kernel_impact_report(
                            rule,
                            self._expanded_rule_theta(fit, rule_index),
                            information_matrix,
                        ),
                        "effective_information": information,
                        "nuisance_rank": int(nuisance_rank),
                        "testable": testable,
                        "directional_occurrence_diagnostic": _mean_test_dict(directional),
                        "hierarchy_branch_occurrence_contribution": _mean_test_dict(
                            occurrence_contribution
                        ),
                        "hierarchy_branch_financial_exposure_contribution": _mean_test_dict(
                            financial_contribution
                        ),
                        "hierarchy_branch_conditional_mark_nll_diagnostic": _mean_test_dict(
                            mark_diagnostic
                        ),
                        "mean_predicted_exposure_full": float(np.mean(predicted_full)),
                        "mean_predicted_exposure_branch_null": float(np.mean(predicted_drop)),
                        "p_value": rule_p,
                    }
                )
                if drop_fit is not fit and drop_fit is not closure_baseline:
                    summary_cache.pop(id(drop_fit), None)
            support_p = max(p_values, default=1.0) if structural else 1.0
            items.append(
                {
                    "support": [self._rule_dict(rule) for rule in fit.rules],
                    "search_joint_nll": float(fit.nll),
                    "search_intensity_nll": float(fit.intensity_nll),
                    "search_mark_nll": float(fit.mark_fit.nll),
                    "mark_design_rank": int(fit.mark_fit.rank),
                    "mark_design_dimension": int(mark_design_dimension),
                    "search_nll_improvement": float(record.search_nll_improvement),
                    "fit_kkt_residual": float(fit.kkt_residual),
                    "fit_converged": True,
                    "closure_terms": [self._closure_dict(term) for term in fit.closure_terms],
                    "closure_baseline_converged": True,
                    "all_rule_blocks_active": True,
                    "joint_occurrence_contribution": _mean_test_dict(joint_occurrence),
                    "joint_financial_exposure_contribution": _mean_test_dict(joint_financial),
                    "joint_conditional_mark_nll_diagnostic": _mean_test_dict(
                        joint_mark_diagnostic
                    ),
                    "mean_observed_financial_exposure": float(np.mean(observed)),
                    "mean_predicted_exposure_full": float(np.mean(predicted_full)),
                    "mean_predicted_exposure_null": float(np.mean(predicted_null)),
                    "rules": rule_items,
                    "structurally_testable": structural,
                    "within_support_test": (
                        "intersection-union maximum p over joint and every rule-rooted branch occurrence + "
                        "financial-exposure contributions"
                    ),
                    "p_value": float(support_p),
                    "evaluation_short_circuited": conditional_short_circuit,
                }
            )
        return items

    def _evaluate_supports(
        self,
        records: Sequence[SupportRecord],
        ctx: QueryContext,
        *,
        alpha: float,
        short_circuit_alpha: float | None = None,
        _parallel: bool = True,
    ) -> list[dict]:
        if _parallel:
            # Cache eligibility is determined solely by the frozen support and
            # hierarchy-preserving drop graph, before any split outcomes are
            # evaluated.  Recursive worker calls reuse this shared plan.
            self._prepare_fit_summary_reuse(records, ctx)
        devices = self._support_worker_devices()
        if _parallel and len(records) > 1 and len(devices) > 1:
            # Support evaluation may have to solve hierarchy-preserving drop
            # models.  Those exact designs have the same resident-memory scale
            # as discovery fits, so apply the execution-only admission limit
            # here as well.  Every record, drop and test is still evaluated.
            worker_count = self._exact_fit_worker_limit(
                min(len(records), len(devices))
            )
            indexed_records = list(enumerate(records))
            nested_chunks: list[
                list[tuple[SupportRecord, list[tuple[int, SupportRecord]]]]
            ] | None = None
            if (
                not self.marked
                and ctx.name == self.splits.fit.name
                and short_circuit_alpha is not None
            ):
                # The full-support Fisher/drop design is the expensive D_fit
                # setup.  Maximal supports provide an exact finer row partition
                # for all nested supports, including hierarchy terms that move
                # from a parent rule block into a child nuisance block.  Keep a
                # parent and its children on one worker and construct that parent
                # lazily only if a child reaches a conditional gate.
                ordered = sorted(
                    indexed_records,
                    key=lambda pair: (-len(pair[1].fit.rules), pair[1].fit.rules),
                )
                maximal: list[tuple[int, SupportRecord]] = []
                maximal_sets: list[frozenset[RuleIdentity]] = []
                for indexed_record in ordered:
                    rule_set = frozenset(indexed_record[1].fit.rules)
                    if not any(rule_set.issubset(parent) for parent in maximal_sets):
                        maximal.append(indexed_record)
                        maximal_sets.append(rule_set)
                groups: list[
                    tuple[SupportRecord, list[tuple[int, SupportRecord]]]
                ] = [(record, []) for _index, record in maximal]
                for indexed_record in indexed_records:
                    rule_set = frozenset(indexed_record[1].fit.rules)
                    parent_index = min(
                        (
                            index
                            for index, parent_set in enumerate(maximal_sets)
                            if rule_set.issubset(parent_set)
                        ),
                        key=lambda index: (
                            len(maximal[index][1].fit.rules),
                            maximal[index][1].fit.rules,
                        ),
                    )
                    groups[parent_index][1].append(indexed_record)
                groups.sort(
                    key=lambda pair: (
                        -sum(1 + len(record.fit.rules) for _index, record in pair[1]),
                        pair[0].fit.rules,
                    )
                )
                nested_chunks = [[] for _ in range(worker_count)]
                chunk_loads = [0] * worker_count
                for group in groups:
                    worker_index = min(
                        range(worker_count),
                        key=lambda index: (chunk_loads[index], index),
                    )
                    nested_chunks[worker_index].append(group)
                    chunk_loads[worker_index] += sum(
                        1 + len(record.fit.rules)
                        for _index, record in group[1]
                    )
                chunks = [
                    [item for _parent, group in groups_for_worker for item in group]
                    for groups_for_worker in nested_chunks
                ]
                with self._diagnostic_guard:
                    self._fit_summary_cache_stats[
                        "scheduled_nested_information_groups"
                    ] = len(groups)
            elif not self.marked:
                # The hierarchy-null loss is identical for every support with
                # the same closure.  Keep each closure group consecutive on a
                # single worker, while greedily balancing group sizes.  Output
                # is restored by original index below, so this is scheduling
                # only: models, losses, tests, and the frozen family are
                # unchanged.  It also makes the byte-bounded null cache attain
                # one computation per closure instead of depending on LRU luck.
                by_closure: dict[
                    tuple[ClosureTerm, ...],
                    list[tuple[int, SupportRecord]],
                ] = {}
                for indexed_record in indexed_records:
                    by_closure.setdefault(
                        indexed_record[1].fit.closure_terms, []
                    ).append(indexed_record)
                closure_groups = sorted(
                    by_closure.values(),
                    key=lambda group: (
                        -sum(1 + len(record.fit.rules) for _index, record in group),
                        group[0][0],
                    ),
                )
                chunks = [[] for _ in range(worker_count)]
                chunk_loads = [0] * worker_count
                for group in closure_groups:
                    worker_index = min(
                        range(worker_count),
                        key=lambda index: (chunk_loads[index], index),
                    )
                    chunks[worker_index].extend(group)
                    # Rule-wise drop/Fisher work grows approximately linearly
                    # with support width.  This deterministic structural weight
                    # is known before looking at split outcomes, so it improves
                    # tail balance without changing or outcome-screening the
                    # frozen family.
                    chunk_loads[worker_index] += sum(
                        1 + len(record.fit.rules)
                        for _index, record in group
                    )
                with self._diagnostic_guard:
                    self._loss_summary_cache_stats["scheduled_closure_groups"] = (
                        len(closure_groups)
                    )
            else:
                chunks = [[] for _ in range(worker_count)]
                chunk_loads = [0] * worker_count
                for indexed_record in sorted(
                    indexed_records,
                    key=lambda pair: (-1 - len(pair[1].fit.rules), pair[0]),
                ):
                    worker_index = min(
                        range(worker_count),
                        key=lambda index: (chunk_loads[index], index),
                    )
                    chunks[worker_index].append(indexed_record)
                    chunk_loads[worker_index] += 1 + len(
                        indexed_record[1].fit.rules
                    )
            workers: list[CertSCRPipeline] = []
            for device in devices[:worker_count]:
                worker = copy.copy(self)
                worker.config = replace(
                    self.config,
                    solver_workers=1,
                    support_devices=(),
                    solver_device=device,
                )
                # Immutable fitted models and per-key locks are shared during
                # certification so repeated branch-drop models are solved once.
                # Other mutable design/summary caches remain worker-local; the
                # byte-bounded occurrence engine is internally locked.
                worker._fit_cache = self._fit_cache
                worker._fit_key_locks = self._fit_key_locks
                worker._fit_key_locks_guard = self._fit_key_locks_guard
                worker._null_fit_cache = dict(self._null_fit_cache)
                worker._hierarchy_closure_cache = dict(
                    self._hierarchy_closure_cache
                )
                worker._safe_bound_cache = dict(self._safe_bound_cache)
                worker._prepared_design_cache = {}
                worker._information_design_parent = None
                worker._safe_screened_records = dict(
                    self._safe_screened_records
                )
                worker._nuisance_event_design_cache = dict(
                    self._nuisance_event_design_cache
                )
                worker._mark_base_residualizer_cache = dict(
                    self._mark_base_residualizer_cache
                )
                worker._event_grid_count_cache = dict(
                    self._event_grid_count_cache
                )
                worker._inference_weight_cache = dict(
                    self._inference_weight_cache
                )
                worker._marked_response_cache = dict(
                    self._marked_response_cache
                )
                worker._active_support_workers = []
                worker.support_records = []
                worker.candidate_records = []
                workers.append(worker)

            def evaluate_chunk(
                worker: CertSCRPipeline,
                chunk: list[tuple[int, SupportRecord]],
                nested_groups: list[
                    tuple[SupportRecord, list[tuple[int, SupportRecord]]]
                ] | None,
            ) -> list[tuple[int, dict]]:
                if nested_groups is not None:
                    output: list[tuple[int, dict]] = []
                    with _single_threaded_local_blas():
                        for parent, group in nested_groups:
                            indices = [index for index, _record in group]
                            group_records = [record for _index, record in group]
                            worker._information_design_parent = [
                                ctx,
                                parent.fit,
                                None,
                            ]
                            try:
                                values = worker._evaluate_supports(
                                    group_records,
                                    ctx,
                                    alpha=alpha,
                                    short_circuit_alpha=short_circuit_alpha,
                                    _parallel=False,
                                )
                            finally:
                                worker._information_design_parent = None
                            output.extend(zip(indices, values, strict=True))
                    return output
                indices = [index for index, _record in chunk]
                chunk_records = [record for _index, record in chunk]
                with _single_threaded_local_blas():
                    values = worker._evaluate_supports(
                        chunk_records,
                        ctx,
                        alpha=alpha,
                        short_circuit_alpha=short_circuit_alpha,
                        _parallel=False,
                    )
                return list(zip(indices, values, strict=True))

            indexed_items: list[tuple[int, dict]] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        evaluate_chunk,
                        worker,
                        chunk,
                        (
                            nested_chunks[index]
                            if nested_chunks is not None
                            else None
                        ),
                    )
                    for index, (worker, chunk) in enumerate(
                        zip(workers, chunks, strict=True)
                    )
                    if chunk
                ]
                for future in futures:
                    indexed_items.extend(future.result())

            # Branch-drop fits found during D_fit screening are exact frozen
            # models reused on D_cert.  Publish them in deterministic worker
            # order; an existing parent fit always wins.
            for worker in workers:
                for key, fit in worker._fit_cache.items():
                    self._fit_cache.setdefault(key, fit)
                for key, fit in worker._null_fit_cache.items():
                    self._null_fit_cache.setdefault(key, fit)
            indexed_items.sort(key=lambda pair: pair[0])
            return [item for _index, item in indexed_items]

        if self.marked:
            return self._evaluate_marked_supports(
                records,
                ctx,
                alpha=alpha,
                short_circuit_alpha=short_circuit_alpha,
            )
        if self.occurrence_likelihood == "first_event_cloglog":
            # The stored cluster_intensity is cumulative hazard, not an
            # expected Bernoulli event count.  Reporting observed-hazard as a
            # probability calibration error is dimensionally invalid.
            exposure = None
            observed = None
        else:
            exposure = cluster_exposure(ctx)
            observed = np.bincount(
                ctx.event_sequence_local, minlength=ctx.n_sequences
            ).astype(np.float64)
        raw_score_weights = self.certification_loss.weights(ctx)
        if ctx.name == self.splits.fit.name:
            raw_score_weights = raw_score_weights * self.fit_sampling_weights
        score_weight_scale = float(np.mean(raw_score_weights))
        score_weights = raw_score_weights / score_weight_scale
        items: list[dict] = []

        for record in records:
            # Bound exact sufficient-statistic storage to one support.  Full,
            # closure-null and branch-drop summaries are reused within this
            # record, then are provably dead for its remaining calculations.
            # Keeping every prior support summary was an unbounded memory leak
            # in the screening schedule, not a statistical cache requirement.
            summary_cache: dict[int, SparseFitSummary] = {}
            information_design: PreparedFixedSupportDesign | None = None

            def summary_for(candidate: FitResult) -> SparseFitSummary:
                key = id(candidate)
                value = summary_cache.get(key)
                if value is None:
                    value = self._cached_sparse_fit_summary(candidate, ctx)
                    summary_cache[key] = value
                return value

            fit = record.fit
            closure_baseline = self.fit_model((), fit.closure_terms)
            if not fit.converged or not closure_baseline.converged:
                invalid_test = _mean_test_dict(
                    one_sided_mean_test([], null=self.config.financial_threshold, alpha=alpha)
                )
                items.append(
                    {
                        "support": [self._rule_dict(rule) for rule in fit.rules],
                        "search_nll": float(fit.nll),
                        "search_nll_improvement": float(record.search_nll_improvement),
                        "fit_kkt_residual": float(fit.kkt_residual),
                        "fit_converged": bool(fit.converged),
                        "closure_terms": [self._closure_dict(term) for term in fit.closure_terms],
                        "closure_baseline_converged": bool(closure_baseline.converged),
                        "all_rule_blocks_active": False,
                        "joint_comparison": "hierarchy_closure_baseline_vs_full_support",
                        "joint_contribution": invalid_test,
                        "calibration": {"gated": False, "p_value": 1.0, "passed": False},
                        "rules": [],
                        "structurally_testable": False,
                        "p_value": 1.0,
                        "invalid_reason": "full or hierarchy-closure baseline did not satisfy the KKT tolerance",
                    }
                )
                continue
            active = bool(np.all(self._active_amplitudes(fit)))
            if not active:
                invalid_rule_test = one_sided_mean_test(
                    [],
                    null=self.config.rule_threshold,
                    alpha=alpha,
                )
                items.append(
                    {
                        "support": [self._rule_dict(rule) for rule in fit.rules],
                        "search_nll": float(fit.nll),
                        "search_nll_improvement": float(record.search_nll_improvement),
                        "fit_kkt_residual": float(fit.kkt_residual),
                        "fit_converged": True,
                        "closure_terms": [self._closure_dict(term) for term in fit.closure_terms],
                        "closure_baseline_converged": True,
                        "all_rule_blocks_active": False,
                        "joint_comparison": "hierarchy_closure_baseline_vs_full_support",
                        "joint_contribution": _mean_test_dict(
                            one_sided_mean_test(
                                [],
                                null=self.config.financial_threshold,
                                alpha=alpha,
                            )
                        ),
                        "calibration": {"gated": False, "p_value": 1.0, "passed": False},
                        "rules": [
                            {
                                "rule": self._rule_dict(rule),
                                "support_amplitude": float(fit.amplitudes[index]),
                                "shape": self._fitted_rule_shape(fit, index).tolist(),
                                "branch_null_model_converged": True,
                                "effective_information": 0.0,
                                "nuisance_rank": 0,
                                "testable": False,
                                "directional": _mean_test_dict(invalid_rule_test),
                                "hierarchy_branch_contribution": _mean_test_dict(invalid_rule_test),
                                "early_warning_effect": (
                                    {
                                        "testable": False,
                                        "horizon_predictive_contribution": _mean_test_dict(
                                            invalid_rule_test
                                        ),
                                        "sign_aligned_probability_shift": _mean_test_dict(
                                            invalid_rule_test
                                        ),
                                        "p_value": 1.0,
                                        "invalid_reason": "rule block is on the zero boundary",
                                    }
                                    if self.certification_mode == "early_warning"
                                    else None
                                ),
                                "p_value": 1.0,
                            }
                            for index, rule in enumerate(fit.rules)
                        ],
                        "structurally_testable": False,
                        "p_value": 1.0,
                        "invalid_reason": "at least one fitted rule block is on the zero boundary",
                    }
                )
                continue
            summary_full = summary_for(fit)
            loss_full = self._sparse_loss_values(summary_full, ctx)
            # Closure-null fits repeat heavily across distinct supports.  The
            # active-grid predictor is not needed for the joint entity-loss
            # test, so cache only its exact O(n_entities) sufficient statistic.
            # If a later singleton/drop comparison needs grid predictors it is
            # reconstructed once in the current support-local cache.
            summary_closure = self._closure_loss_summary(
                closure_baseline, ctx
            )
            loss_closure = self._sparse_loss_values(summary_closure, ctx)
            joint = one_sided_mean_test(
                loss_closure - loss_full,
                null=self.config.financial_threshold,
                alpha=alpha,
            )
            if self.occurrence_likelihood == "first_event_cloglog":
                calibration = {
                    "gated": False,
                    "estimate": None,
                    "note": "probability calibration requires per-cell survival probabilities",
                }
            else:
                assert observed is not None and exposure is not None
                expected = summary_full.cluster_intensity
                calibration_error = np.divide(
                    observed - expected,
                    exposure,
                    out=np.zeros_like(expected),
                    where=exposure > 0,
                )
                if self.config.calibration_tolerance is None:
                    calibration = {
                        "gated": False,
                        "estimate": float(np.mean(calibration_error)),
                    }
                else:
                    calibration = equivalence_mean_test(
                        calibration_error,
                        tolerance=float(self.config.calibration_tolerance),
                        alpha=alpha,
                    )
                    calibration["gated"] = True

            # Where a valid calibration estimand is implemented, it is an
            # absolute deployment diagnostic, not an additional rule-discovery
            # hypothesis. First-event cloglog leaves it explicitly unassessed.
            preliminary_p = float(joint.p_value)
            if short_circuit_alpha is not None and preliminary_p > short_circuit_alpha:
                items.append(
                    {
                        "support": [self._rule_dict(rule) for rule in fit.rules],
                        "search_nll": float(fit.nll),
                        "search_nll_improvement": float(record.search_nll_improvement),
                        "fit_kkt_residual": float(fit.kkt_residual),
                        "fit_converged": bool(fit.converged),
                        "closure_terms": [self._closure_dict(term) for term in fit.closure_terms],
                        "closure_baseline_converged": bool(closure_baseline.converged),
                        "all_rule_blocks_active": active,
                        "joint_comparison": "hierarchy_closure_baseline_vs_full_support",
                        "joint_contribution": _mean_test_dict(joint),
                        "calibration": calibration,
                        "rules": [
                            {
                                "rule": self._rule_dict(rule),
                                "support_amplitude": float(fit.amplitudes[index]),
                                "shape": self._fitted_rule_shape(fit, index).tolist(),
                                "conditional_gates_evaluated": False,
                            }
                            for index, rule in enumerate(fit.rules)
                        ],
                        "structurally_testable": False,
                        "p_value": 1.0,
                        "evaluation_short_circuited": True,
                        "short_circuit_reason": (
                            "joint materiality already failed the current screen threshold; later "
                            "branch-null conjunction conditions cannot make the support selectable"
                        ),
                    }
                )
                continue

            rule_items: list[dict] = []
            rule_p_values: list[float] = []
            structural = True
            conditional_short_circuit = False
            for rule_index, rule in enumerate(fit.rules):
                other_rules, removed_rules = self.hierarchy_preserving_drop(fit.rules, rule)
                drop_key = (other_rules, tuple(fit.closure_terms))
                if (
                    ctx.name == self.splits.fit.name
                    and drop_key not in self._fit_cache
                    and drop_key not in self._prepared_design_cache
                ):
                    if information_design is None:
                        information_design = self._prepare_support_information_design(
                            fit, ctx, score_weights
                        )
                    self._prepared_design_cache[drop_key] = (
                        project_prepared_support_design(
                            information_design, other_rules
                        )
                    )
                drop_fit = self.fit_model(other_rules, fit.closure_terms, initial=fit)
                self._prepared_design_cache.pop(drop_key, None)
                if not drop_fit.converged:
                    invalid_directional = one_sided_mean_test(
                        [], null=0.0, alpha=alpha
                    )
                    rule_p_values.append(1.0)
                    structural = False
                    rule_items.append(
                        {
                            "rule": self._rule_dict(rule),
                            "hierarchy_branch_removed": [
                                self._rule_dict(item) for item in removed_rules
                            ],
                            "conditional_comparison": "hierarchy_preserving_branch_drop",
                            "support_amplitude": float(fit.amplitudes[rule_index]),
                            "shape": self._fitted_rule_shape(
                                fit, rule_index
                            ).tolist(),
                            "kernel_impact": None,
                            "branch_null_model_converged": False,
                            "effective_information": None,
                            "nuisance_rank": None,
                            "testable": False,
                            "directional": _mean_test_dict(invalid_directional),
                            "hierarchy_branch_contribution": _mean_test_dict(
                                invalid_directional
                            ),
                            "early_warning_effect": None,
                            "p_value": 1.0,
                            "invalid_reason": (
                                "hierarchy-preserving branch-drop fit did not satisfy "
                                "the configured KKT tolerance"
                            ),
                        }
                    )
                    # D_fit selection is a conjunction over every rule-rooted
                    # hierarchy branch. Once a
                    # nested null is invalid, no later rule can restore this
                    # support.  D_cert retains per-rule diagnostics and thus
                    # continues.  This is logical short-circuiting only; it does
                    # not change a fit, test, threshold, or selected support.
                    if (
                        ctx.name == self.splits.fit.name
                        and short_circuit_alpha is not None
                    ):
                        conditional_short_circuit = True
                        for remaining_index in range(
                            rule_index + 1, len(fit.rules)
                        ):
                            remaining_rule = fit.rules[remaining_index]
                            rule_items.append(
                                {
                                    "rule": self._rule_dict(remaining_rule),
                                    "support_amplitude": float(
                                        fit.amplitudes[remaining_index]
                                    ),
                                    "shape": self._fitted_rule_shape(
                                        fit, remaining_index
                                    ).tolist(),
                                    "conditional_gates_evaluated": False,
                                    "p_value": 1.0,
                                    "short_circuit_reason": (
                                        "an earlier D_fit branch-drop fit made "
                                        "support selection impossible"
                                    ),
                                }
                            )
                        break
                    continue
                summary_drop = summary_for(drop_fit)
                contribution = one_sided_mean_test(
                    self._sparse_loss_values(summary_drop, ctx) - loss_full,
                    null=self.config.rule_threshold,
                    alpha=alpha,
                )
                shape = self._fitted_rule_shape(fit, rule_index)
                early_warning = (
                    self._early_warning_rule_report(
                        fit,
                        drop_fit,
                        summary_full,
                        summary_drop,
                        ctx,
                        rule_index,
                        alpha=alpha,
                    )
                    if self.certification_mode == "early_warning"
                    else None
                )
                preliminary_rule_p = self._rule_irreducibility_p_value(
                    contribution.p_value,
                    early_warning,
                )
                preliminary_rule_testable = bool(
                    early_warning is None or early_warning["testable"]
                )
                fit_screen_rule_failure = bool(
                    ctx.name == self.splits.fit.name
                    and short_circuit_alpha is not None
                    and (
                        not preliminary_rule_testable
                        or preliminary_rule_p > short_circuit_alpha
                    )
                )
                if fit_screen_rule_failure:
                    invalid_directional = one_sided_mean_test(
                        [], null=0.0, alpha=alpha
                    )
                    rule_p_values.append(1.0)
                    structural = False
                    conditional_short_circuit = True
                    rule_items.append(
                        {
                            "rule": self._rule_dict(rule),
                            "hierarchy_branch_removed": [
                                self._rule_dict(item) for item in removed_rules
                            ],
                            "conditional_comparison": "hierarchy_preserving_branch_drop",
                            "support_amplitude": float(fit.amplitudes[rule_index]),
                            "shape": shape.tolist(),
                            "kernel_impact": None,
                            "branch_null_model_converged": bool(drop_fit.converged),
                            "effective_information": None,
                            "nuisance_rank": None,
                            "testable": False,
                            "structural_testability_evaluated": False,
                            "directional": _mean_test_dict(invalid_directional),
                            "hierarchy_branch_contribution": _mean_test_dict(
                                contribution
                            ),
                            "early_warning_effect": early_warning,
                            "p_value": 1.0,
                            "conditional_gates_evaluated": False,
                            "short_circuit_reason": (
                                "this D_fit rule condition already failed the conjunction; "
                                "later Fisher diagnostics and rules cannot restore selection"
                            ),
                        }
                    )
                    for remaining_index in range(rule_index + 1, len(fit.rules)):
                        remaining_rule = fit.rules[remaining_index]
                        rule_items.append(
                            {
                                "rule": self._rule_dict(remaining_rule),
                                "support_amplitude": float(
                                    fit.amplitudes[remaining_index]
                                ),
                                "shape": self._fitted_rule_shape(
                                    fit, remaining_index
                                ).tolist(),
                                "conditional_gates_evaluated": False,
                                "p_value": 1.0,
                                "short_circuit_reason": (
                                    "an earlier D_fit rule condition made support selection impossible"
                                ),
                            }
                        )
                    if drop_fit is not fit and drop_fit is not closure_baseline:
                        summary_cache.pop(id(drop_fit), None)
                    break
                raw_feature = self.engine.sparse_response(
                    ctx, rule.antecedent, rule.window
                )
                if information_design is None:
                    information_design = self._prepare_support_information_design(
                        fit, ctx, score_weights
                    )
                nuisance = (
                    *self.sparse_nuisance_blocks(ctx, fit.closure_terms),
                    *self.sparse_features(ctx, other_rules),
                )
                information_matrix, nuisance_rank, score_values, raw_information = (
                    self._sparse_rule_information(
                        drop_fit,
                        ctx,
                        raw_feature,
                        nuisance,
                        shape,
                        score_weights,
                        null_summary=summary_drop,
                        prepared_full=information_design,
                        focal_rule=rule,
                        remaining_rules=other_rules,
                    )
                )
                information = float(shape @ information_matrix @ shape)
                impact = self._kernel_impact_report(
                    rule,
                    self._expanded_rule_theta(fit, rule_index),
                    information_matrix,
                )
                drop_converged = bool(drop_fit.converged)
                testable = (
                    drop_converged
                    and ctx.n_sequences >= 2
                    and numeric_information_positive_from_raw(
                        information,
                        raw_information,
                        ctx.n_grid,
                    )
                )
                if testable:
                    directional = one_sided_mean_test(
                        float(rule.sign) * score_values,
                        null=0.0,
                        alpha=alpha,
                    )
                else:
                    directional = one_sided_mean_test([], null=0.0, alpha=alpha)
                # Rule irreducibility is an intersection-union claim.  In
                # early-warning mode the same hierarchy-preserving branch must
                # improve global held-out point-process loss *and* its
                # horizon-local observed loss/risk footprint.  The previous
                # implementation replaced the global branch-null p-value with the
                # horizon p-value, allowing a rare inhibition branch to ride on
                # another rule's joint support gain.
                p_rule = self._rule_irreducibility_p_value(
                    contribution.p_value,
                    early_warning,
                )
                if early_warning is not None and not early_warning["testable"]:
                    p_rule = 1.0
                rule_p_values.append(p_rule)
                structural = structural and testable and (
                    early_warning is None or bool(early_warning["testable"])
                )
                rule_items.append(
                    {
                        "rule": self._rule_dict(rule),
                        "hierarchy_branch_removed": [
                            self._rule_dict(item) for item in removed_rules
                        ],
                        "conditional_comparison": "hierarchy_preserving_branch_drop",
                        "support_amplitude": float(fit.amplitudes[rule_index]),
                        "shape": shape.tolist(),
                        "kernel_impact": impact,
                        "branch_null_model_converged": drop_converged,
                        "effective_information": float(information),
                        "nuisance_rank": int(nuisance_rank),
                        "testable": bool(testable),
                        "directional": _mean_test_dict(directional),
                        "hierarchy_branch_contribution": _mean_test_dict(contribution),
                        "early_warning_effect": early_warning,
                        "p_value": float(p_rule),
                    }
                )
                if drop_fit is not fit and drop_fit is not closure_baseline:
                    summary_cache.pop(id(drop_fit), None)
            p_full = max(float(joint.p_value), max(rule_p_values, default=1.0))
            if not structural:
                p_full = 1.0
            items.append(
                {
                    "support": [self._rule_dict(rule) for rule in fit.rules],
                    "search_nll": float(fit.nll),
                    "search_nll_improvement": float(record.search_nll_improvement),
                    "fit_kkt_residual": float(fit.kkt_residual),
                    "fit_converged": bool(fit.converged),
                    "closure_terms": [self._closure_dict(term) for term in fit.closure_terms],
                    "closure_baseline_converged": bool(closure_baseline.converged),
                    "all_rule_blocks_active": active,
                    "joint_comparison": "hierarchy_closure_baseline_vs_full_support",
                    "joint_contribution": _mean_test_dict(joint),
                    "calibration": calibration,
                    "rules": rule_items,
                    "structurally_testable": bool(structural),
                    "p_value": float(p_full),
                    "evaluation_short_circuited": conditional_short_circuit,
                }
            )
        return items

    def screen_supports_on_fit(self) -> dict:
        self.certification_mode = self._resolve_certification_mode()
        if not self.support_records:
            self.search_supports()
        items = self._evaluate_supports(
            self.support_records,
            self.splits.fit,
            alpha=self.config.alpha_fit_screen,
            short_circuit_alpha=self.config.alpha_fit_screen,
        )
        selected_mask = [
            bool(
                item.get("fit_converged", False)
                and item.get("closure_baseline_converged", False)
                and item.get("all_rule_blocks_active", False)
                and item.get("structurally_testable", False)
                and float(item.get("p_value", 1.0)) <= self.config.alpha_fit_screen
            )
            for item in items
        ]
        for item, selected in zip(items, selected_mask, strict=True):
            item["selected_for_certification"] = selected
        self.candidate_records = [
            record
            for record, selected in zip(self.support_records, selected_mask, strict=True)
            if selected
        ]
        result = {
            "claim": "fit_screen_only_not_a_reportable_claim",
            "selection_rule": (
                "all fitted blocks active and structurally testable; the support and every "
                "rule-rooted hierarchy branch improve D_fit loss"
                + (
                    "; every branch root also has a sign-aligned primary-horizon adverse-event risk effect"
                    if self.certification_mode == "early_warning"
                    else ""
                )
                + "; direction and calibration remain diagnostics; only independent D_cert is reportable"
            ),
            "certification_mode": self.certification_mode,
            "alpha_fit_screen": self.config.alpha_fit_screen,
            "searched_support_count": len(items),
            "selected_support_count": len(self.candidate_records),
            "selected_supports": [item for item in items if item["selected_for_certification"]],
            "all_supports": items,
        }
        self.last_fit_screen = result
        return result

    def certify_supports(self) -> dict:
        self.certification_mode = self._resolve_certification_mode()
        if self.last_fit_screen is None:
            self.screen_supports_on_fit()
        items = self._evaluate_supports(
            self.candidate_records,
            self.splits.cert,
            alpha=self.config.alpha_family,
            short_circuit_alpha=self.config.alpha_family,
        )
        f0 = self._f0_contract() if self.certification_mode == "early_warning" else None
        adjusted = holm_adjust([item["p_value"] for item in items])
        for item, adjusted_p in zip(items, adjusted, strict=True):
            item["holm_p_value"] = float(adjusted_p)
            if self.certification_mode == "early_warning":
                joint_p = float(
                    item.get("joint_contribution", {}).get(
                        "p_value", item.get("p_value", 1.0)
                    )
                )
                rule_p_values = [
                    float(rule.get("p_value", 1.0))
                    for rule in item.get("rules", [])
                ]
                if not rule_p_values:
                    rule_p_values = [float(item.get("p_value", 1.0))]
                f1_passed = bool(
                    item.get("fit_converged", False)
                    and item.get("closure_baseline_converged", False)
                    and item.get("all_rule_blocks_active", False)
                    and joint_p <= self.config.alpha_family
                )
                f2_passed = bool(
                    item.get("structurally_testable", False)
                    and rule_p_values
                    and max(rule_p_values) <= self.config.alpha_family
                )
                item["F1"] = {
                    "name": "held_out_support_predictive_irreducibility",
                    "raw_p_value": joint_p,
                    "passed_before_family_correction": f1_passed,
                }
                item["F2"] = {
                    "name": "every_rule_rooted_hierarchy_branch_global_and_horizon_irreducibility",
                    "raw_max_branch_p_value": max(rule_p_values, default=1.0),
                    "passed_before_family_correction": f2_passed,
                }
                item["statistically_certified"] = bool(
                    f1_passed
                    and f2_passed
                    and adjusted_p <= self.config.alpha_family
                )
            else:
                item["statistically_certified"] = bool(
                    adjusted_p <= self.config.alpha_family
                )
            # ``certified`` is retained as the backwards-compatible generic
            # statistical certificate.  Only the conjunction with F0 is
            # allowed to carry the financial early-warning reliability label.
            item["certified"] = item["statistically_certified"]
            item["financially_reliable"] = bool(
                item["statistically_certified"]
                and (
                    self.marked
                    or (f0 is not None and f0["passed"])
                )
            )
        certified = [item for item in items if item["certified"]]
        if self.marked:
            financially_reliable = [
                item for item in items if item["financially_reliable"]
            ]
            result = {
                "claim": "financially_reliable_marked_tpp_support_set",
                "financial_estimand": "per-entity cumulative marked exposure sum_i M_i",
                "exposure_rate": "rho(t|H_t)=lambda(t|H_t) E[M|t,H_t]",
                "reliability_contract": {
                    "base_requirement": (
                        "every signed occurrence rule is active and structurally testable under the "
                        "same hierarchy closure"
                    ),
                    "condition_1_occurrence": (
                        "the full support and every hierarchy-preserving rule branch improve held-out "
                        "occurrence-process NLL"
                    ),
                    "condition_2_financial": (
                        "the full support and every hierarchy-preserving rule branch improve held-out "
                        "squared prediction loss for cumulative marked exposure"
                    ),
                    "mark_only_rules_reportable": False,
                    "conditional_mark_nll_role": "joint discovery objective and diagnostic, not a separate gate",
                    "minimum_joint_financial_contribution": self.config.financial_threshold,
                    "minimum_each_rule_financial_contribution": self.config.rule_threshold,
                    "selection_split": "fit",
                    "independent_certification_split": "cert",
                    "within_support_test": "intersection-union maximum p",
                    "family_error_control": (
                        "Holm strong FWER over the support family frozen without D_cert"
                    ),
                    "alpha_family": self.config.alpha_family,
                    "cluster_unit": "independent sequence/entity",
                    "cluster_inference": (
                        "one-sided studentized entity-mean tests; large-cluster asymptotic, "
                        "not finite-sample exact"
                    ),
                },
                "mark_name": self.data.mark_name,
                "mark_distribution": "positive log-normal with variance frozen at the D_fit null",
                "family_size": len(items),
                "alpha_family": self.config.alpha_family,
                "certified_count": len(certified),
                "financially_certified": bool(certified),
                "financially_reliable_support_count": len(financially_reliable),
                "financially_reliable_supports": financially_reliable,
                "certified_supports": certified,
                "all_supports": items,
            }
            self.last_certification = result
            return result
        if self.certification_mode == "early_warning":
            financially_reliable = [
                item for item in items if item["financially_reliable"]
            ]
            adverse_name = (
                str(self.config.adverse_event_name).strip()
                if self.config.adverse_event_name is not None
                else None
            )
            result = {
                "claim": (
                    "financially_reliable_adverse_event_early_warning_support_family"
                    if f0 is not None and f0["passed"]
                    else "certified_event_early_warning_support_family"
                ),
                "reliability_contract": {
                    "F0": f0,
                    "F1": (
                        "on independent D_cert, the full support improves point-process NLL "
                        "over the same hierarchy closure"
                    ),
                    "F2": (
                        "on independent D_cert, every hierarchy-preserving rule branch improves "
                        "global and horizon-local observed point-process loss, and its frozen rule "
                        "term has the fitted sign's event-probability effect"
                    ),
                    "target": adverse_name or "unnamed target event",
                    "adverse_financial_event_semantics_pre_specified": bool(
                        adverse_name is not None
                    ),
                    "condition_1_predictive_irreducibility": (
                        "the full support improves held-out point-process NLL over its hierarchy closure"
                    ),
                    "condition_2_early_warning_materiality": (
                        "each hierarchy-preserving branch improves global held-out loss and, inside "
                        "its primary-horizon future cells, observed point-process loss; the frozen "
                        "rule term also has a positive sign-aligned model-implied probability shift"
                    ),
                    "primary_horizon": self.early_warning_horizon,
                    "forecast_protocol": (
                        "rolling predictable-intensity evaluation; overlapping warning cells are "
                        "counted once, not fixed-origin forecasting with future covariates"
                    ),
                    "minimum_probability_point_shift": self.config.early_warning_threshold,
                    "structural_testability_required": True,
                    "selection_split": "fit",
                    "independent_certification_split": "cert",
                    "within_support_test": "intersection_union_max_p",
                    "family_error_control": (
                        "Holm strong FWER conditional on the family frozen by D_fit"
                    ),
                    "alpha_family": self.config.alpha_family,
                    "cluster_unit": "independent sequence/entity",
                    "cluster_inference": (
                        "one-sided studentized entity-mean tests; large-cluster asymptotic, "
                        "not finite-sample exact"
                    ),
                    "calibration_equivalence_required_for_rule_certification": False,
                    "causal_claim": False,
                    "monetary_loss_claim": False,
                },
                "financially_grounded_loss": False,
                "financial_contract_complete": False,
                "financially_certified": False,
                "adverse_event_early_warning_certified": bool(
                    f0 is not None and f0["passed"] and certified
                ),
                "financially_reliable_support_count": int(
                    len(financially_reliable)
                ),
                "financially_reliable_supports": financially_reliable,
                "ensemble_support_gate": "F0_and_statistical_F1_F2_certification",
                "claim_scope": (
                    (
                        "statistically certified predictive early-warning rules for the named "
                        "adverse financial event"
                        if f0 is not None and f0["passed"]
                        else "statistically certified predictive early-warning rules for an unnamed event"
                    )
                    + "; not monetary utility, causal effect, or regime invariance"
                ),
                "family_size": len(items),
                "alpha_family": self.config.alpha_family,
                "certified_count": len(certified),
                "statistically_certified_count": len(certified),
                "certified_supports": certified,
                "all_supports": items,
            }
            self.last_certification = result
            return result
        materiality_pre_specified = bool(
            self.config.financial_threshold > 0 and self.config.rule_threshold > 0
        )
        financial_contract_complete = bool(
            self.certification_loss.financially_grounded
            and materiality_pre_specified
        )
        financially_certified = bool(financial_contract_complete and certified)
        result = {
            "claim": (
                "financially_certified_support_set"
                if financial_contract_complete
                else (
                    "financially_weighted_predictive_support_set"
                    if self.certification_loss.financially_grounded
                    else "predictively_certified_support_set"
                )
            ),
            "loss": self.certification_loss.name,
            "reliability_contract": {
                "estimand": self.certification_loss.name,
                "minimum_joint_contribution": self.config.financial_threshold,
                "minimum_each_rule_rooted_branch_contribution": self.config.rule_threshold,
                "conditions": (
                    "joint financial materiality AND every hierarchy-preserving rule-branch financial materiality"
                ),
                "directional_score_role": "diagnostic_not_certification_gate",
                "structural_testability_required": True,
                "selection_split": "fit",
                "independent_certification_split": "cert",
                "within_support_test": "intersection_union_max_p",
                "family_error_control": "Holm strong FWER conditional on the family frozen by D_fit",
                "alpha_family": self.config.alpha_family,
                "calibration_equivalence_required_for_rule_certification": False,
                "calibration_role": "diagnostic_for_supports; optional deployment gate for the final ensemble",
                "calibration_tolerance": self.config.calibration_tolerance,
                "cluster_inference": (
                    "one-sided studentized independent-entity mean test; "
                    "large-cluster asymptotic, not finite-sample exact"
                ),
            },
            "financially_grounded_loss": bool(self.certification_loss.financially_grounded),
            "positive_materiality_thresholds_pre_specified": materiality_pre_specified,
            "financial_contract_complete": financial_contract_complete,
            "financially_certified": financially_certified,
            "financial_claim_reason": (
                "At least one support passed joint and every rule-rooted branch materiality under the "
                "pre-specified financial contract."
                if financially_certified
                else (
                    "The financial contract is complete, but no support passed simultaneous certification."
                    if financial_contract_complete
                    else (
                        "A financial weight was supplied, but positive joint and rule-wise materiality "
                        "thresholds were not both pre-specified."
                        if self.certification_loss.financially_grounded
                        else "No business-defined financial loss was supplied; TPP NLL is a predictive loss."
                    )
                )
            ),
            "family_size": len(items),
            "alpha_family": self.config.alpha_family,
            "certified_count": len(certified),
            "certified_supports": certified,
            "all_supports": items,
        }
        self.last_certification = result
        return result

    @staticmethod
    def _sparse_intensity_disagreement(
        fits: Sequence[FitResult],
        summaries: Sequence[SparseFitSummary],
        weights: np.ndarray,
        ctx: QueryContext,
    ) -> dict:
        """Exact query-row disagreement with inactive rows represented by a count."""
        coefficients = np.asarray(weights, dtype=np.float64)
        if not fits or coefficients.shape != (len(fits),):
            raise ValueError("support disagreement inputs do not align")
        if len(fits) == 1:
            # The weighted standard deviation of a singleton component is
            # identically zero on every event and grid row; no sparse union or
            # quantile materialization is necessary.
            return {
                "defined": True,
                "mean_relative_intensity_sd": 0.0,
                "p95_relative_intensity_sd": 0.0,
                "max_relative_intensity_sd": 0.0,
            }
        summary_parts = [
            summary.active_grid_indices
            for summary in summaries
            if len(summary.active_grid_indices)
        ]
        active_indices = (
            np.unique(np.concatenate(summary_parts)).astype(np.int64, copy=False)
            if summary_parts
            else np.zeros(0, dtype=np.int64)
        )

        def relative_sd_columns(
            row_count: int,
            columns: Iterable[np.ndarray],
        ) -> np.ndarray:
            log_mean = np.full(row_count, -math.inf, dtype=np.float64)
            log_second = np.full(row_count, -math.inf, dtype=np.float64)
            for log_values, coefficient in zip(
                columns, coefficients, strict=True
            ):
                if coefficient <= 0.0:
                    continue
                values = np.asarray(log_values, dtype=np.float64)
                if values.shape != (row_count,):
                    raise ValueError("disagreement predictor column is misaligned")
                log_weight = math.log(float(coefficient))
                log_mean = np.logaddexp(
                    log_mean, log_weight + values
                )
                log_second = np.logaddexp(
                    log_second, log_weight + 2.0 * values
                )
            with np.errstate(over="ignore", invalid="ignore"):
                variance = np.expm1(log_second - 2.0 * log_mean)
            return np.sqrt(np.maximum(variance, 0.0))

        explicit_parts: list[np.ndarray] = []
        if ctx.n_events:
            explicit_parts.append(
                relative_sd_columns(
                    ctx.n_events,
                    [summary.event_eta for summary in summaries],
                )
            )
        if len(active_indices):
            # Stream one component column at a time.  The old active-by-model
            # matrix can dwarf the actual fit when many certified supports are
            # ensembled, while log-sum-exp needs only two row accumulators.
            def grid_columns():
                for fit, summary in zip(fits, summaries, strict=True):
                    grid_log = np.full(
                        len(active_indices), float(fit.alpha), dtype=np.float64
                    )
                    if len(summary.active_grid_indices):
                        positions = np.searchsorted(
                            active_indices, summary.active_grid_indices
                        )
                        grid_log[positions] = summary.active_grid_eta
                    yield grid_log

            explicit_parts.append(
                relative_sd_columns(len(active_indices), grid_columns())
            )
        explicit = (
            np.concatenate(explicit_parts)
            if explicit_parts
            else np.zeros(0, dtype=np.float64)
        )
        inactive_count = int(ctx.n_grid - len(active_indices))
        inactive_value = (
            float(
                relative_sd_columns(
                    1,
                    [np.asarray([float(fit.alpha)]) for fit in fits],
                )[0]
            )
            if inactive_count
            else None
        )
        total_count = int(len(explicit) + inactive_count)
        if total_count <= 0:
            return {"defined": False, "reason": "empty query context"}
        mean_numerator = float(np.sum(explicit, dtype=np.float64))
        if inactive_value is not None:
            mean_numerator += inactive_count * inactive_value

        values = explicit
        counts = np.ones(len(explicit), dtype=np.int64)
        if inactive_value is not None:
            values = np.concatenate((values, np.asarray([inactive_value])))
            counts = np.concatenate((counts, np.asarray([inactive_count], dtype=np.int64)))
        order = np.argsort(values, kind="stable")
        ordered_values = values[order]
        cumulative = np.cumsum(counts[order], dtype=np.int64)

        def order_value(index: int) -> float:
            position = int(np.searchsorted(cumulative, int(index), side="right"))
            return float(ordered_values[position])

        quantile_position = 0.95 * float(total_count - 1)
        lower_index = int(math.floor(quantile_position))
        upper_index = int(math.ceil(quantile_position))
        fraction = quantile_position - lower_index
        p95 = (1.0 - fraction) * order_value(lower_index) + fraction * order_value(
            upper_index
        )
        return {
            "defined": True,
            "mean_relative_intensity_sd": mean_numerator / float(total_count),
            "p95_relative_intensity_sd": float(p95),
            "max_relative_intensity_sd": float(ordered_values[-1]),
        }

    def fit_and_evaluate_ensemble(self) -> dict:
        if self.last_certification is None:
            self.certify_supports()
        assert self.last_certification is not None
        # In adverse-event early-warning mode, the financial semantics contract
        # F0 is part of the reliability gate.  A statistically significant
        # support for an unnamed/non-adverse event remains reportable as a
        # statistical result, but must never enter the reliability ensemble.
        ensemble_gate = (
            "financially_reliable"
            if self.certification_mode == "early_warning"
            else "certified"
        )
        certified_mask = [
            bool(item.get(ensemble_gate, False))
            for item in self.last_certification["all_supports"]
        ]
        certified_records = [
            record for record, passed in zip(self.candidate_records, certified_mask, strict=True) if passed
        ]
        if not certified_records:
            return {
                "fitted": False,
                "reason": (
                    "no_financially_reliable_support"
                    if self.certification_mode == "early_warning"
                    else "no_certified_support"
                ),
            }
        # Support identities, signs and hierarchy contracts are frozen after
        # D_cert. Only then are component parameters (including the M-knot
        # coefficients) and ensemble weights refitted on all
        # non-test entities (the complete, unsampled D_fit population + D_cert).
        # D_test is never touched until the final evaluation.
        ensemble_ids = np.unique(
            np.concatenate(
                [
                    self.splits.fit_population_global_ids,
                    self.splits.cert.global_sequence_ids,
                ]
            )
        )
        ensemble_ctx = make_context(self.data, "ensemble_train_fit_plus_cert", ensemble_ids)
        baseline = self.fit_baseline()
        refit_jobs = [
            ((), baseline),
            *[(record.rules, record.fit) for record in certified_records],
        ]
        devices = self._support_worker_devices()
        if len(refit_jobs) > 1 and len(devices) > 1:
            # Fit+cert component refits use the complete population and can be
            # larger than discovery designs.  Limit only simultaneous ownership;
            # the frozen component family, objective and KKT target are unchanged.
            worker_count = self._exact_fit_worker_limit(
                min(len(refit_jobs), len(devices))
            )
            workers: list[CertSCRPipeline] = []
            for device in devices[:worker_count]:
                worker = copy.copy(self)
                worker.config = replace(
                    self.config,
                    solver_workers=1,
                    support_devices=(),
                    solver_device=device,
                )
                worker._nuisance_event_design_cache = dict(
                    self._nuisance_event_design_cache
                )
                worker._mark_base_residualizer_cache = dict(
                    self._mark_base_residualizer_cache
                )
                worker._active_support_workers = []
                workers.append(worker)
            indexed_jobs = list(enumerate(refit_jobs))
            chunks = [
                indexed_jobs[index::worker_count]
                for index in range(worker_count)
            ]

            def refit_chunk(
                worker: CertSCRPipeline,
                chunk: list[
                    tuple[
                        int,
                        tuple[tuple[RuleIdentity, ...], FitResult],
                    ]
                ],
            ) -> list[tuple[int, FitResult]]:
                output: list[tuple[int, FitResult]] = []
                with _single_threaded_local_blas():
                    for index, (rules, initial) in chunk:
                        output.append(
                            (
                                index,
                                worker.fit_frozen_support_on_context(
                                    rules, ensemble_ctx, initial=initial
                                ),
                            )
                        )
                return output

            indexed_refits: list[tuple[int, FitResult]] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(refit_chunk, worker, chunk)
                    for worker, chunk in zip(workers, chunks, strict=True)
                    if chunk
                ]
                for future in futures:
                    indexed_refits.extend(future.result())
            indexed_refits.sort(key=lambda pair: pair[0])
            component_fits = [fit for _index, fit in indexed_refits]
        else:
            component_fits = [
                self.fit_frozen_support_on_context(
                    rules, ensemble_ctx, initial=initial
                )
                for rules, initial in refit_jobs
            ]
        refit_baseline = component_fits[0]
        refit_supports = component_fits[1:]
        component_refits_converged = all(
            fit.converged
            and (
                not self.marked
                or (
                    fit.mark_fit is not None
                    and fit.mark_fit.converged
                )
            )
            for fit in component_fits
        )
        if not component_refits_converged:
            return {
                "fitted": False,
                "reason": "a frozen fit-plus-cert component did not converge",
            }
        support_refit_active = [
            bool(np.all(self._active_amplitudes(fit))) for fit in refit_supports
        ]
        collapsed_components = [
            {
                "support": [self._rule_dict(rule) for rule in record.rules],
                "fit_plus_cert_amplitudes": fit.amplitudes.tolist(),
                "reason": "at least one frozen rule block reached the numerical zero boundary",
            }
            for record, fit, active in zip(
                certified_records, refit_supports, support_refit_active, strict=True
            )
            if not active
        ]
        if collapsed_components:
            retained = [
                (record, fit)
                for record, fit, active in zip(
                    certified_records, refit_supports, support_refit_active, strict=True
                )
                if active
            ]
            certified_records = [record for record, _fit in retained]
            refit_supports = [fit for _record, fit in retained]
            component_fits = [refit_baseline, *refit_supports]
        if not certified_records:
            return {
                "fitted": False,
                "reason": "all certified supports collapsed on fit-plus-cert refitting",
                "collapsed_components": collapsed_components,
            }
        ensemble_cluster_weights = (
            np.ones(ensemble_ctx.n_sequences, dtype=np.float64)
            if self.marked
            else self.certification_loss.weights(ensemble_ctx)
        )
        component_event_eta: list[np.ndarray] = []
        component_grid_integrals = np.empty(len(component_fits), dtype=np.float64)
        component_mark_density: list[np.ndarray] | None = [] if self.marked else None
        for component_index, fit in enumerate(component_fits):
            summary = self._sparse_fit_summary(fit, ensemble_ctx)
            component_event_eta.append(summary.event_eta)
            if np.any(~np.isfinite(summary.cluster_intensity)):
                return {
                    "fitted": False,
                    "reason": "nonfinite component intensity on ensemble training data",
                }
            component_grid_integrals[component_index] = float(
                np.dot(ensemble_cluster_weights, summary.cluster_intensity)
            )
            if component_mark_density is not None:
                if fit.mark_fit is None:
                    raise RuntimeError("marked ensemble component is missing a mark head")
                component_mark_density.append(
                    event_mark_log_density(
                        fit.mark_fit,
                        ensemble_ctx,
                        self.nuisance_event_design(ensemble_ctx, fit.closure_terms),
                        self.mark_rule_activations(fit, ensemble_ctx, event_only=True),
                    )
                )
        ensemble_fit = fit_intensity_ensemble(
            component_event_eta,
            ensemble_ctx,
            component_event_log_density=component_mark_density,
            component_grid_integrals=component_grid_integrals,
            device=self.config.solver_device,
            cluster_weights=ensemble_cluster_weights,
            occurrence_likelihood=self.occurrence_likelihood,
        )
        if not ensemble_fit.converged:
            return {
                "fitted": False,
                "reason": "ensemble simplex optimization did not converge",
                "projected_residual": ensemble_fit.projected_residual,
            }
        test_ctx = self.splits.test
        support_weights = np.asarray(ensemble_fit.weights[1:], dtype=np.float64)
        support_weight_total = float(np.sum(support_weights))
        conditional_weights = (
            support_weights / support_weight_total
            if support_weight_total > 0
            else np.zeros_like(support_weights)
        )
        eta_test = np.full(test_ctx.n_events, -math.inf, dtype=np.float64)
        ensemble_test_intensity = np.zeros(test_ctx.n_sequences, dtype=np.float64)
        baseline_summary_test: SparseFitSummary | None = None
        test_summaries: list[SparseFitSummary] = []
        empty_summary_values = np.zeros(0, dtype=np.float64)
        ensemble_expected_financial = (
            np.zeros(test_ctx.n_sequences, dtype=np.float64)
            if self.marked
            else None
        )
        baseline_expected_financial: np.ndarray | None = None
        for component_index, (fit, weight) in enumerate(
            zip(component_fits, ensemble_fit.weights, strict=True)
        ):
            summary = self._sparse_fit_summary(fit, test_ctx)
            # Disagreement needs only predictors, not the two per-sequence loss
            # arrays.  Retaining those arrays for every ensemble component was
            # another O(models * entities) memory accumulation.
            test_summaries.append(
                SparseFitSummary(
                    event_eta=summary.event_eta,
                    active_grid_indices=summary.active_grid_indices,
                    active_grid_eta=summary.active_grid_eta,
                    cluster_intensity=empty_summary_values,
                    cluster_nll=empty_summary_values,
                )
            )
            if component_index == 0:
                baseline_summary_test = summary
            if weight > 0:
                eta_test = np.logaddexp(
                    eta_test,
                    math.log(float(weight)) + summary.event_eta,
                )
                ensemble_test_intensity += float(weight) * summary.cluster_intensity
            if self.marked:
                _occurrence, _mark, _financial, _observed, predicted = (
                    self._sparse_marked_losses(fit, test_ctx, summary)
                )
                assert ensemble_expected_financial is not None
                ensemble_expected_financial += float(weight) * predicted
                if component_index == 0:
                    baseline_expected_financial = predicted
        if (
            baseline_summary_test is None
            or np.any(~np.isfinite(eta_test))
            or np.any(~np.isfinite(ensemble_test_intensity))
        ):
            return {"fitted": False, "reason": "nonfinite streamed test mixture"}
        test_evaluation = evaluate_ensemble_sufficient(
            eta_test,
            baseline_summary_test.event_eta,
            ensemble_test_intensity,
            baseline_summary_test.cluster_intensity,
            test_ctx,
            contribution_threshold=(0.0 if self.marked else self.config.financial_threshold),
            calibration_tolerance=self.config.calibration_tolerance,
            alpha=self.config.alpha_family,
            cluster_weights=(None if self.marked else self.certification_loss.weights(test_ctx)),
            occurrence_likelihood=self.occurrence_likelihood,
        )
        if self.marked:
            if (
                ensemble_expected_financial is None
                or baseline_expected_financial is None
            ):
                raise RuntimeError("marked ensemble financial predictions are missing")
            observed_financial = np.bincount(
                test_ctx.event_sequence_local,
                weights=test_ctx.event_marks,
                minlength=test_ctx.n_sequences,
            ).astype(np.float64)
            unit = float(component_fits[0].mark_fit.unit)
            baseline_financial_loss = (
                (observed_financial - baseline_expected_financial) / unit
            ) ** 2
            ensemble_financial_loss = (
                (observed_financial - ensemble_expected_financial) / unit
            ) ** 2
            financial_contribution = one_sided_mean_test(
                baseline_financial_loss - ensemble_financial_loss,
                null=self.config.financial_threshold,
                alpha=self.config.alpha_family,
            )
            test_evaluation["financial_exposure_contribution"] = _mean_test_dict(
                financial_contribution
            )
            test_evaluation["mean_observed_financial_exposure"] = float(
                np.mean(observed_financial)
            )
            test_evaluation["mean_predicted_financial_exposure"] = float(
                np.mean(ensemble_expected_financial)
            )
        final_test_superiority = bool(
            float(test_evaluation["contribution"]["p_value"]) <= self.config.alpha_family
        )
        if self.marked:
            final_test_superiority = bool(
                final_test_superiority
                and float(test_evaluation["financial_exposure_contribution"]["p_value"])
                <= self.config.alpha_family
            )
        test_calibration = test_evaluation["calibration"]
        calibration_certified = bool(
            test_calibration.get("gated", False) and test_calibration.get("passed", False)
        )
        deployment_assessed = bool(
            self.marked or self.occurrence_likelihood != "first_event_cloglog"
        )
        deployment_certified: bool | None = (
            bool(
                final_test_superiority
                and (True if self.marked else calibration_certified)
            )
            if deployment_assessed
            else None
        )
        component_labels = [
            {"type": "baseline", "support": []},
            *[
                {"type": "support", "support": [self._rule_dict(rule) for rule in record.rules]}
                for record in certified_records
            ],
        ]
        all_rules = sorted({rule for record in certified_records for rule in record.rules})
        inclusion = []
        for rule in all_rules:
            present = np.asarray([rule in record.rules for record in certified_records], dtype=bool)
            absolute_weight = float(np.sum(support_weights[present]))
            signed_impacts: list[float] = []
            signed_curves: list[np.ndarray] = []
            # Ensemble weights multiply the fit+cert refits, not the original
            # D_fit discovery estimates.  Reporting D_fit kernels here mixed
            # coefficients from one model with weights learned for another.
            for fit, is_present in zip(refit_supports, present, strict=True):
                if not is_present:
                    continue
                rule_index = fit.rules.index(rule)
                theta = self._expanded_rule_theta(fit, rule_index)
                unsigned_curve = theta @ self.engine.basis64
                signed_impacts.append(float(rule.sign) * float(np.sum(theta)))
                signed_curves.append(float(rule.sign) * unsigned_curve)
            curves = np.stack(signed_curves, axis=0)
            present_weights = support_weights[present]
            if absolute_weight > 0:
                conditional_present_weights = present_weights / absolute_weight
                weighted_impact = float(np.dot(conditional_present_weights, signed_impacts))
                weighted_curve = conditional_present_weights @ curves
                weighted_curve_variance = conditional_present_weights @ (
                    (curves - weighted_curve[None, :]) ** 2
                )
                weighted_curve_sd = np.sqrt(np.maximum(weighted_curve_variance, 0.0))
            else:
                weighted_impact = None
                weighted_curve = None
                weighted_curve_sd = None
            inclusion.append(
                {
                    "rule": self._rule_dict(rule),
                    "support_count": int(np.sum(present)),
                    "support_fraction": float(np.mean(present)),
                    "absolute_ensemble_weight": absolute_weight,
                    "conditional_rule_inclusion": (
                        absolute_weight / support_weight_total if support_weight_total > 0 else 0.0
                    ),
                    "core_across_certified_supports": bool(np.all(present)),
                    "support_conditioned_kernel_refinement": {
                        "signed_integrated_impacts": signed_impacts,
                        "min_signed_integrated_impact": float(np.min(signed_impacts)),
                        "max_signed_integrated_impact": float(np.max(signed_impacts)),
                        "ensemble_weighted_signed_integrated_impact": weighted_impact,
                        "ensemble_weighted_signed_curve": (
                            weighted_curve.tolist() if weighted_curve is not None else None
                        ),
                        "ensemble_weighted_signed_curve_sd": (
                            weighted_curve_sd.tolist() if weighted_curve_sd is not None else None
                        ),
                    },
                }
            )
        if support_weight_total > 0:
            disagreement = self._sparse_intensity_disagreement(
                component_fits[1:],
                test_summaries[1:],
                conditional_weights,
                test_ctx,
            )
        else:
            disagreement = {
                "defined": False,
                "reason": "the fitted ensemble assigned zero total weight to certified support models",
            }
        return {
            "fitted": True,
            "component_count": len(component_fits),
            "weights": [
                {**label, "weight": float(weight)}
                for label, weight in zip(component_labels, ensemble_fit.weights, strict=True)
            ],
            "ensemble_fit_nll": ensemble_fit.nll,
            "ensemble_objective": (
                "marked_point_process_mixture_nll"
                if self.marked
                else "monthly_first_event_cloglog_mixture_nll"
                if self.occurrence_likelihood == "first_event_cloglog"
                else "point_process_mixture_nll"
            ),
            "ensemble_training_split": "complete_D_fit_population_plus_D_cert_after_support_freeze",
            "ensemble_fit_converged": ensemble_fit.converged,
            "ensemble_fit_iterations": ensemble_fit.iterations,
            "ensemble_projected_residual": ensemble_fit.projected_residual,
            "final_test_superiority": final_test_superiority,
            "final_test_superiority_claim": (
                "occurrence_and_cumulative_marked_exposure_superiority"
                if self.marked
                else "adverse_event_process_predictive_superiority"
                if self.certification_mode == "early_warning"
                else "financial_loss_superiority"
                if self.certification_loss.financially_grounded
                else "predictive_loss_superiority"
            ),
            "calibration_certified": calibration_certified,
            "deployment_assessed": deployment_assessed,
            "deployment_certified": deployment_certified,
            "deployment_reason": (
                "Not assessed: first-event cloglog probability calibration is not implemented."
                if not deployment_assessed
                else
                (
                    "Final occurrence and cumulative marked-exposure contributions both passed; "
                    "calibration is diagnostic, not a reliability gate."
                    if self.marked
                    else "Final contribution and pre-specified calibration equivalence both passed."
                )
                if deployment_certified
                else (
                    "Final contribution did not pass on untouched D_test."
                    if not final_test_superiority
                    else "Final contribution passed, but calibration equivalence was not pre-specified and passed."
                )
            ),
            "explanation_multiplicity": {
                "certified_support_count": int(sum(certified_mask)),
                "active_ensemble_support_count": len(certified_records),
                "collapsed_fit_plus_cert_components": collapsed_components,
                "distinct_rule_count": len(all_rules),
                "support_weight_total": support_weight_total,
                "rule_inclusion": inclusion,
                "test_intensity_disagreement": disagreement,
            },
            "test": test_evaluation,
        }

    def _closure_dict(self, term: ClosureTerm) -> dict:
        antecedent, window = term
        names = [self.data.predicate_names[idx] for idx in antecedent]
        return {
            "antecedent_ids": list(antecedent),
            "antecedents": names,
            "order": len(antecedent),
            "window": int(window),
            "role": "unrestricted_hierarchy_nuisance",
            "text": f"{' AND '.join(names)} (hierarchy nuisance, W={window})",
        }

    def _rule_dict(self, rule: RuleIdentity) -> dict:
        return {
            "antecedent_ids": list(rule.antecedent),
            "antecedents": [self.data.predicate_names[idx] for idx in rule.antecedent],
            "order": len(rule.antecedent),
            "window": int(rule.window),
            "sign": "exc" if rule.sign > 0 else "inh",
            "text": self.engine.rule_name(rule),
        }

    @staticmethod
    def _rule_state(rule: RuleIdentity) -> dict:
        return {
            "antecedent": list(rule.antecedent),
            "window": int(rule.window),
            "sign": int(rule.sign),
        }

    @staticmethod
    def _rule_from_state(payload: dict) -> RuleIdentity:
        return RuleIdentity(
            tuple(int(value) for value in payload["antecedent"]),
            int(payload["window"]),
            int(payload["sign"]),
        )

    def export_profile_state(self) -> dict:
        """Serialize the frozen dictionary independently of Python caches."""
        state_rules = set(self.profiled_rules)
        for candidates in self.identity_candidates.values():
            state_rules.update(candidates)
        return {
            "schema": 1,
            "checkpoint_signature": self.checkpoint_signature(),
            "profiled_rules": [
                self._rule_state(rule) for rule in self.profiled_rules
            ],
            "dictionary_shapes": [
                {
                    "rule": self._rule_state(rule),
                    "shape": np.asarray(shape, dtype=np.float64).tolist(),
                }
                for rule, shape in sorted(
                    self.rule_dictionary_shapes.items()
                )
                if rule in state_rules
            ],
            "identity_candidates": [
                {
                    "antecedent": list(antecedent),
                    "rules": [self._rule_state(rule) for rule in rules],
                }
                for antecedent, rules in sorted(
                    self.identity_candidates.items()
                )
            ],
            "profile_logs": self.profile_logs,
        }

    def restore_profile_state(self, payload: dict) -> None:
        if int(payload.get("schema", -1)) != 1:
            raise ValueError("unsupported profile checkpoint schema")
        if payload.get("checkpoint_signature") != self.checkpoint_signature():
            raise ValueError("profile checkpoint data/config signature differs")
        rules = sorted(
            {
                self._rule_from_state(item)
                for item in payload.get("profiled_rules", [])
            }
        )
        if any(
            any(source not in self.rule_source_ids for source in rule.antecedent)
            for rule in rules
        ):
            raise ValueError("profile checkpoint contains an ineligible predicate")
        identities: dict[
            tuple[int, ...], tuple[RuleIdentity, ...]
        ] = {}
        for item in payload.get("identity_candidates", []):
            antecedent = tuple(int(value) for value in item["antecedent"])
            values = tuple(
                sorted(
                    self._rule_from_state(rule)
                    for rule in item.get("rules", [])
                )
            )
            if values:
                identities[antecedent] = values
        allowed_shape_rules = set(rules)
        for values in identities.values():
            allowed_shape_rules.update(values)
        # Re-read shapes after the identity graph is known.  Alternative W/sign
        # atoms are required by support-conditioned identity refinement even
        # though only one canonical atom per skeleton is in profiled_rules.
        shapes = {}
        for item in payload.get("dictionary_shapes", []):
            rule = self._rule_from_state(item["rule"])
            shape = np.asarray(item["shape"], dtype=np.float64)
            if shape.shape != (self.config.knot_count,) or np.any(
                ~np.isfinite(shape)
            ):
                raise ValueError("profile checkpoint contains an invalid shape")
            if rule in allowed_shape_rules:
                shapes[rule] = shape.copy()
        if self.config.identity_profile == "dictionary_mdl" and any(
            rule not in shapes for rule in rules
        ):
            raise ValueError("profile checkpoint is missing a dictionary shape")
        self.profiled_rules = rules
        self.rule_dictionary_shapes = shapes
        self.identity_candidates = identities
        self.profile_logs = list(payload.get("profile_logs", []))
        self._profile_completed = True
        canonical_terms = {
            (rule.antecedent, int(rule.window)) for rule in rules
        }
        for rule in rules:
            canonical_terms.update(self.hierarchy_closure((rule,)))
        self.engine.retain_context_terms(
            self.splits.fit.name, tuple(canonical_terms)
        )

    def export_fit_cache_state(self) -> dict:
        """Serialize converged unmarked occurrence fits for exact resume."""
        entries: list[dict] = []
        seen: set[tuple[tuple[RuleIdentity, ...], tuple[ClosureTerm, ...]]] = set()
        with self._fit_key_locks_guard:
            items = tuple(self._fit_cache.items())
        for (_cache_rules, _cache_closure), fit in items:
            key = (tuple(fit.rules), tuple(fit.closure_terms))
            if (
                key in seen
                or not fit.converged
                or fit.mark_fit is not None
                or not math.isfinite(float(fit.nll))
            ):
                continue
            seen.add(key)
            entries.append(
                {
                    "rules": [self._rule_state(rule) for rule in fit.rules],
                    "closure": [
                        {
                            "antecedent": list(antecedent),
                            "window": int(window),
                        }
                        for antecedent, window in fit.closure_terms
                    ],
                    "alpha": float(fit.alpha),
                    "gamma": np.asarray(fit.gamma, dtype=np.float64).tolist(),
                    "theta": np.asarray(fit.theta, dtype=np.float64).tolist(),
                    "nll": float(fit.nll),
                    "kkt_residual": float(fit.kkt_residual),
                    "iterations": int(fit.iterations),
                    "device": str(fit.device),
                    "intensity_nll": (
                        None
                        if fit.intensity_nll is None
                        else float(fit.intensity_nll)
                    ),
                }
            )
        entries.sort(
            key=lambda item: (
                len(item["rules"]),
                json.dumps(item["rules"], sort_keys=True),
                json.dumps(item["closure"], sort_keys=True),
            )
        )
        return {
            "schema": 1,
            "checkpoint_signature": self.checkpoint_signature(),
            "fits": entries,
        }

    def restore_fit_cache_state(self, payload: dict) -> int:
        if int(payload.get("schema", -1)) != 1:
            raise ValueError("unsupported fit-cache checkpoint schema")
        if payload.get("checkpoint_signature") != self.checkpoint_signature():
            raise ValueError("fit-cache checkpoint data/config signature differs")
        restored = 0
        for item in payload.get("fits", []):
            rules = tuple(
                sorted(self._rule_from_state(rule) for rule in item["rules"])
            )
            closure = tuple(
                sorted(
                    (
                        tuple(int(value) for value in term["antecedent"]),
                        int(term["window"]),
                    )
                    for term in item["closure"]
                )
            )
            gamma = np.asarray(item["gamma"], dtype=np.float64)
            theta = np.asarray(item["theta"], dtype=np.float64)
            if theta.ndim == 1 and not rules and theta.size == 0:
                theta = np.zeros((0, 0), dtype=np.float64)
            if theta.ndim != 2 or theta.shape[0] != len(rules):
                raise ValueError("fit checkpoint rule dimension mismatch")
            if any(
                any(source not in self.rule_source_ids for source in rule.antecedent)
                for rule in rules
            ) or any(
                any(source not in self.rule_source_ids for source in antecedent)
                or not 0 <= int(window) <= self.config.max_formation_window
                for antecedent, window in closure
            ):
                raise ValueError("fit checkpoint contains an ineligible rule term")
            expected_gamma = self.config.knot_count * (
                int(self.target_history_source_id is not None)
                + len(self.control_source_ids)
                + len(self.loan_age_baseline_source_ids)
                + len(closure)
            )
            if gamma.shape != (expected_gamma,):
                raise ValueError("fit checkpoint nuisance dimension mismatch")
            if rules:
                widths = {
                    1 if rule in self.rule_dictionary_shapes else self.config.knot_count
                    for rule in rules
                }
                if len(widths) != 1 or theta.shape[1] != widths.pop():
                    raise ValueError("fit checkpoint kernel dimension mismatch")
            elif theta.size:
                raise ValueError("null fit checkpoint contains rule coefficients")
            if (
                np.any(~np.isfinite(gamma))
                or np.any(~np.isfinite(theta))
                or np.any(theta < 0.0)
                or not math.isfinite(float(item["alpha"]))
                or not math.isfinite(float(item["nll"]))
                or not math.isfinite(float(item["kkt_residual"]))
            ):
                raise ValueError("fit checkpoint contains invalid coefficients")
            fit = FitResult(
                rules=rules,
                closure_terms=closure,
                alpha=float(item["alpha"]),
                gamma=gamma,
                theta=theta,
                nll=float(item["nll"]),
                kkt_residual=float(item["kkt_residual"]),
                converged=True,
                iterations=int(item["iterations"]),
                device=f"resume:{item['device']}",
                intensity_nll=(
                    None
                    if item.get("intensity_nll") is None
                    else float(item["intensity_nll"])
                ),
                mark_fit=None,
                solver_hessian=None,
            )
            self._fit_cache[(rules, closure)] = fit
            if not rules:
                self._null_fit_cache[closure] = fit
            restored += 1
        return restored

    def checkpoint_signature(self) -> str:
        cached = self._checkpoint_signature_cache
        if cached is not None:
            return cached
        digest = hashlib.blake2b(digest_size=24)
        execution_only = {
            "solver_device",
            "solver_workers",
            "support_devices",
            "support_workers_per_device",
            "feature_cache_bytes",
            "persistent_response_dir",
            "persistent_response_bytes",
            "loss_summary_cache_bytes",
            "fit_summary_cache_bytes",
            "response_workers",
        }
        statistical_config = {
            key: value
            for key, value in asdict(self.config).items()
            if key not in execution_only
        }
        digest.update(
            json.dumps(
                {
                    "schema": 1,
                    "predicate_names": self.data.predicate_names,
                    "rule_source_ids": self.rule_source_ids,
                    "control_source_ids": self.control_source_ids,
                    "predicate_policy_name": self.predicate_policy_name,
                    "target_process_mode": self.target_process_mode,
                    "occurrence_likelihood": self.occurrence_likelihood,
                    "mark_name": self.data.mark_name,
                    "financial_weight_name": self.data.financial_weight_name,
                    "certification_loss": {
                        "name": self.certification_loss.name,
                        "financially_grounded": bool(
                            self.certification_loss.financially_grounded
                        ),
                    },
                    "config": statistical_config,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        def update_array(array: np.ndarray | None) -> None:
            if array is None:
                digest.update(b"<none>")
                return
            value = np.asarray(array)
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(str(value.dtype).encode("ascii"))
            if value.dtype.hasobject or value.dtype.kind in {"U", "S"}:
                digest.update(
                    json.dumps(
                        value.tolist(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                return
            contiguous = np.ascontiguousarray(value)
            view = memoryview(contiguous).cast("B")
            chunk = 8 * 1024**2
            for left in range(0, len(view), chunk):
                digest.update(view[left : left + chunk])

        for array in (
            self.data.sequence_ids,
            self.data.sequence_codes,
            self.data.positions,
            self.data.times,
            self.data.predicates,
            self.data.targets,
            self.data.start_times,
            self.data.end_times,
            self.data.sequence_split_groups,
            self.data.sequence_start_ages,
            self.data.sequence_financial_weights,
            self.certification_loss.global_sequence_weights,
        ):
            update_array(array)
        if self.data.target_marks is None:
            update_array(None)
            update_array(None)
        else:
            update_array(self.data.target_marks.offsets)
            update_array(self.data.target_marks.values)
        value = digest.hexdigest()
        self._checkpoint_signature_cache = value
        return value

    def _maybe_checkpoint_support_fits(self, *, force: bool = False) -> None:
        path = self._runtime_checkpoint_path
        if path is None:
            return
        now = time.monotonic()
        if (
            not force
            and now - self._runtime_checkpoint_last_write
            < self._runtime_checkpoint_interval
        ):
            return
        self._runtime_checkpoint_last_write = now
        preserved: dict = {}
        if path.exists():
            try:
                preserved = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                preserved = {}
        preserved.update(
            {
                "algorithm": self.algorithm_name,
                "checkpoint_stage": "support_search_in_progress",
                "elapsed_seconds": time.perf_counter()
                - self._runtime_checkpoint_started,
                "profile_state": self.export_profile_state(),
                "support_fit_cache": self.export_fit_cache_state(),
            }
        )
        save_result(preserved, path)

    def run(self, *, checkpoint_path: str | Path | None = None) -> dict:
        started = time.perf_counter()
        self._runtime_checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path)
        )
        self._runtime_checkpoint_started = started
        self._runtime_checkpoint_last_write = -math.inf
        def checkpoint(stage: str, **payload: object) -> None:
            if checkpoint_path is None:
                return
            checkpoint_file = Path(checkpoint_path)
            preserved: dict[str, object] = {}
            if checkpoint_file.exists():
                try:
                    previous = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                    preserved = {
                        key: previous[key]
                        for key in (
                            "hybrid_pricing",
                            "hybrid_full_d_fit_acceptance",
                            "multifidelity_skeleton_screen",
                            "full_survivor_reprofiling",
                            "profile_state",
                            "support_fit_cache",
                        )
                        if key in previous
                    }
                except (OSError, ValueError, TypeError):
                    preserved = {}
            save_result(
                {
                    "algorithm": self.algorithm_name,
                    **preserved,
                    "checkpoint_stage": stage,
                    "elapsed_seconds": time.perf_counter() - started,
                    **payload,
                },
                checkpoint_path,
            )

        baseline = self.fit_baseline()
        baseline_done = time.perf_counter()
        profiled = (
            list(self.profiled_rules)
            if self._profile_completed
            else self.profile_rule_identities()
        )
        profile_done = time.perf_counter()
        checkpoint("profile_complete", profiled_rule_count=len(profiled))
        self._maybe_checkpoint_support_fits(force=True)
        supports = self.search_supports()
        search_done = time.perf_counter()
        checkpoint(
            "support_search_complete",
            profiled_rule_count=len(profiled),
            candidate_support_count=len(supports),
            support_search_diagnostics=self.search_diagnostics,
        )
        fit_screen = self.screen_supports_on_fit()
        fit_screen_done = time.perf_counter()
        checkpoint(
            "fit_screen_complete",
            candidate_support_count=len(supports),
            selected_support_count=fit_screen["selected_support_count"],
        )
        # Fitted objects are frozen in _fit_cache. Drop D_fit response matrices
        # only after the fit-screen has frozen the certification family.
        self._clear_loss_summary_context(self.splits.fit)
        self._clear_fit_summary_context(self.splits.fit)
        self._clear_early_warning_geometry(self.splits.fit)
        self.engine.clear_context_cache(self.splits.fit.name)
        certification = self.certify_supports()
        certification_done = time.perf_counter()
        checkpoint(
            "certification_complete",
            family_size=certification["family_size"],
            certified_count=certification["certified_count"],
        )
        self._clear_loss_summary_context(self.splits.cert)
        self._clear_fit_summary_context(self.splits.cert)
        self._clear_early_warning_geometry(self.splits.cert)
        ensemble = self.fit_and_evaluate_ensemble()
        ensemble_done = time.perf_counter()
        self.engine.clear_context_cache(self.splits.cert.name)
        self.engine.clear_context_cache(self.splits.test.name)
        self.engine.clear_context_cache("ensemble_train_fit_plus_cert")
        elapsed = ensemble_done - started
        return {
            "algorithm": self.algorithm_name,
            "schema_version": 14,
            "support_model": (
                "three_way_standalone_atoms_and_atom_start_local_terminals_with_independent_certification_and_test"
            ),
            "profile": (
                "oracle-exact-window-sign-and-support-enumeration"
                if self.config.exhaustive_profile
                else (
                    "finite_identity_exact_score_mdl_atoms_then_terminal_exact_refinement"
                    if self.config.identity_profile == "score_mdl"
                    else "finite_identity_exact_dictionary_mdl_then_primary_family_full_m_refinement"
                    if self.config.identity_profile == "dictionary_mdl"
                    else "canonical-rule-atom_then_explanatory-support-set"
                )
            ),
            "profile_limitation": (
                (
                    "Oracle audit mode treats every W/sign parameterization as a distinct identity and is not "
                    "the primary set-valued estimator."
                )
                if self.config.exhaustive_profile
                else (
                    "Every finite W/sign identity not eliminated by a rigorous global MDL upper bound is "
                    "exact-fitted; the exact best standalone identity is admitted. Exact W/sign coordinate "
                    "refinement is restricted to local terminal supports. weak_mdl_heredity, "
                    "when configured, excludes pure triplets whose three constituent pairs all have "
                    "nonpositive standalone block-MDL."
                ) if self.config.identity_profile == "score_mdl" else (
                    "Every W/sign is ordered by its deterministic contiguous-dictionary score, then every "
                    "identity not eliminated by a rigorous global MDL upper bound is exact-fitted. Occurrence "
                    "and conditional-mark gains share one shape in marked mode; block dimension, dictionary "
                    "shape code, and W/sign identity code define discovery MDL. Every positive standalone "
                    "atom of order 1..q_max and every unique atom-start local terminal "
                    "local terminal are offered "
                    "a full nonnegative M-knot occurrence refinement; only full-cone KKT-converged, positive "
                    "full-M block-MDL supports proceed to the D_fit reliability screen. The scalar-dictionary "
                    "configured-neighborhood certificate and the terminal full-M acceptance are both exact for their stated "
                    "objectives. The per-identity scalar dictionary shape is the exhaustively best null-score "
                    "shape, not an exact likelihood-profiled shape; full-M discrete-neighborhood stationarity over "
                    "unreturned supports is not claimed."
                ) if self.config.identity_profile == "dictionary_mdl" else (
                    "Each antecedent skeleton defines one stable canonical rule atom; W/sign are exactly profiled "
                    "over their finite D_fit candidates and then frozen across every multi-rule support."
                )
            ),
            "config": asdict(self.config),
            "occurrence_model": {
                "likelihood": self.occurrence_likelihood,
                "target_process": self.target_process_mode,
                "event_cell_probability": (
                    "1-exp(-exp(eta))"
                    if self.occurrence_likelihood == "first_event_cloglog"
                    else None
                ),
                "event_reporting_cell_excluded_from_no_event_compensator": bool(
                    self.occurrence_likelihood == "first_event_cloglog"
                ),
            },
            "certification_loss": {
                "name": self.certification_loss.name,
                "certification_mode": self.certification_mode,
                "financially_grounded": self.certification_loss.financially_grounded,
                "used_for_fit_profile_and_support_refinement": not self.marked,
                "fit_weight_normalization": {
                    "raw_fit_split_mean": self.fit_weight_scale,
                    "normalized_mean": 1.0,
                    "optimizer_invariance": (
                        "division by one common positive constant preserves every fixed-support minimizer, "
                        "profile ordering, and support ordering"
                    ),
                },
                "directional_estimand": "the same loss-weighted target population",
            },
            "marked_process": (
                {
                    "enabled": True,
                    "mark_column": self.data.mark_name,
                    "mark_unit_D_fit_median": self.mark_unit,
                    "mark_log_variance_frozen_at_D_fit_null": self.mark_variance,
                    "discovery_score": "occurrence NLL + conditional log-normal mark NLL",
                    "estimator": (
                        "two-stage: occurrence kernel is fitted first; the shared-shape conditional "
                        "mark head is then fitted exactly with frozen null variance"
                    ),
                    "financial_exposure_rate": "lambda(t|H_t) E[M|t,H_t]",
                }
                if self.marked
                else {"enabled": False}
            ),
            "data": {
                "n_sequences": self.data.n_sequences,
                "n_rows": int(len(self.data.targets)),
                "target_events": int(np.sum(self.data.targets)),
                "target_mark_sum": (
                    float(np.sum(self.data.target_marks.values, dtype=np.float64))
                    if self.data.target_marks is not None
                    else None
                ),
                "rule_predicates": [self.data.predicate_names[idx] for idx in self.rule_source_ids],
                "control_predicates": [self.data.predicate_names[idx] for idx in self.control_source_ids],
                "target_history_control": {
                    "requested": self.target_history_control_requested,
                    "enabled": self.target_history_source_id is not None,
                    "omitted_as_structural_zero": self.target_history_structural_zero,
                    "target_process": self.target_process_mode,
                    "target_process_source": self.target_process_source,
                    "candidate_rule_eligible": False,
                    "kernel_basis": (
                        f"unrestricted {self.config.knot_count}-knot nuisance over lags "
                        f"1..{self.config.impact_lag}"
                    ),
                    "same_bin_target_predicts_itself": False,
                },
                "loan_age_baseline": {
                    "enabled": bool(self.loan_age_baseline_source_ids),
                    "milestones": list(self.loan_age_baseline_milestones),
                    "epoch_width_months": int(self.config.impact_lag),
                    "knot_count_per_epoch": int(self.config.knot_count),
                    "candidate_rule_eligible": False,
                },
                "predicate_policy": self.predicate_policy_name,
                "F0_contract": (
                    self._f0_contract()
                    if self.certification_mode == "early_warning"
                    else None
                ),
                "split_sequences": {
                    "fit": self.splits.fit.n_sequences,
                    "cert": self.splits.cert.n_sequences,
                    "test": self.splits.test.n_sequences,
                },
                "split_strategy": self.splits.split_strategy,
                "split_groups": (
                    {
                        "fit": list(self.splits.split_groups[0]),
                        "cert": list(self.splits.split_groups[1]),
                        "test": list(self.splits.split_groups[2]),
                    }
                    if self.splits.split_groups is not None
                    else None
                ),
                "fit_sampling": {
                    "population_sequences": self.splits.fit_population_sequence_count,
                    "population_negative_sequences": self.splits.fit_population_negative_count,
                    "sampled_negative_sequences": self.splits.fit_sampled_negative_count,
                    "sampled_fit_sequences": self.splits.fit.n_sequences,
                    "ipw_mean": self.fit_sampling_scale,
                    "population_loss_weight_mean": self.fit_population_loss_weight_mean,
                    "mdl_population_scale": self.fit_objective_population_scale,
                    "kish_effective_sample_size": self.fit_sampling_ess,
                },
                "split_target_events": {
                    "fit": self.splits.fit.n_events,
                    "cert": self.splits.cert.n_events,
                    "test": self.splits.test.n_events,
                },
            },
            "baseline": {
                "nll": baseline.nll,
                "intensity_nll": baseline.intensity_nll,
                "mark_nll": baseline.mark_fit.nll if baseline.mark_fit is not None else None,
                "alpha": baseline.alpha,
                "gamma": baseline.gamma.tolist(),
                "target_history_gamma": (
                    baseline.gamma[: self.config.knot_count].tolist()
                    if self.target_history_source_id is not None
                    else None
                ),
                "loan_age_baseline_gamma": (
                    baseline.gamma[
                        (
                            self.config.knot_count
                            if self.target_history_source_id is not None
                            else 0
                        )
                        + len(self.control_source_ids) * self.config.knot_count :
                    ].reshape(
                        len(self.loan_age_baseline_source_ids),
                        self.config.knot_count,
                    ).tolist()
                    if self.loan_age_baseline_source_ids
                    else None
                ),
                "kkt_residual": baseline.kkt_residual,
                "converged": baseline.converged,
            },
            "profiled_rule_count": len(profiled),
            "candidate_rule_count": len(profiled),
            "profiled_rules": [self._rule_dict(rule) for rule in profiled],
            "window_profiles": self.profile_logs,
            "candidate_support_count": len(supports),
            "support_search_diagnostics": self.search_diagnostics,
            "runtime_optimization": {
                "equivalent_sparse_response_hits": int(
                    self.engine.equivalent_response_hits
                ),
                "persistent_response_store": {
                    "enabled": bool(
                        self.config.persistent_response_bytes > 0
                    ),
                    "directory": self.config.persistent_response_dir,
                    "byte_limit": int(
                        self.config.persistent_response_bytes
                    ),
                    "mapped_bytes": int(
                        self.engine._persistent_response_bytes
                    ),
                    "resident_mmap_entries": int(
                        len(self.engine._persistent_sparse_cache)
                    ),
                    "resident_mmap_bytes": int(
                        self.engine._persistent_sparse_cache_bytes
                    ),
                    "resident_heap_feature_bytes": int(
                        self.engine._feature_cache_bytes
                    ),
                    "combined_resident_byte_limit": int(
                        self.engine._feature_cache_limit
                    ),
                    "resident_mmap_evictions": int(
                        self.engine.persistent_response_evictions
                    ),
                    "hits": int(self.engine.persistent_response_hits),
                    "writes": int(self.engine.persistent_response_writes),
                    "skipped_bytes": int(
                        self.engine.persistent_response_skipped_bytes
                    ),
                },
                "child_support_kkt_shortcuts": int(
                    self._safe_screen_stats["child_kkt_shortcuts"]
                ),
                "native_sparse_kernels": (
                    "completion_sweep_union_layout_kernel_accumulation_row_grouping_"
                    "component_integral_and_predictor_with_exact_numpy_fallback"
                ),
                "native_cone_solver": {
                    "contract": (
                        "compiled_float64_active_set_with_host_projected_KKT_"
                        "certification_and_exact_numpy_fallback"
                    ),
                    "cached_fit_backend_counts": dict(
                        Counter(fit.device for fit in self._fit_cache.values())
                    ),
                },
                "nested_blas_oversubscription_guard": bool(
                    _mkl_local_thread_setter() is not None
                    and len(self._support_worker_devices()) > 1
                ),
                "sparse_query_evaluation": True,
                "parallel_support_evaluation_workers": len(
                    self._support_worker_devices()
                ),
                "maximum_concurrent_exact_fits": int(
                    self._exact_fit_worker_limit(
                        len(self._support_worker_devices())
                    )
                ),
                "support_local_summary_scope": "current_support_only",
                "cross_support_summary_reuse": (
                    "frozen_graph_keys_only_with_byte_bounded_LRU"
                ),
                "active_fit_execution": {
                    "closure_local_batches": True,
                    "default_backend": (
                        "shared_long_lived_threads_with_compiled_MKL_kernels"
                    ),
                    "process_backend_opt_in_environment": (
                        "CERTSCR_PROCESS_FITS=1"
                    ),
                    "nested_support_order": (
                        "ascending_size_semantic_feasible_warm_start"
                    ),
                    "thread_batches": int(
                        self._safe_screen_stats["active_fit_thread_batches"]
                    ),
                    "posix_process_batches": int(
                        self._safe_screen_stats["active_fit_process_batches"]
                    ),
                    "scheduled_closure_groups": int(
                        self._safe_screen_stats["active_fit_closure_groups"]
                    ),
                    "shared_closure_null_prefits": int(
                        self._safe_screen_stats[
                            "active_fit_shared_closure_null_prefits"
                        ]
                    ),
                    "closure_child_shards": int(
                        self._safe_screen_stats[
                            "active_fit_closure_shards"
                        ]
                    ),
                    "rolling_microjobs": int(
                        self._safe_screen_stats["active_fit_dynamic_jobs"]
                    ),
                    "rolling_worker_launches": int(
                        self._safe_screen_stats[
                            "active_fit_dynamic_worker_launches"
                        ]
                    ),
                    "support_closure_partitions": int(
                        self._safe_screen_stats["support_closure_partitions"]
                    ),
                    "support_closure_child_reuses": int(
                        self._safe_screen_stats[
                            "support_closure_partition_child_reuses"
                        ]
                    ),
                    "support_delta_factorized_fits": int(
                        self._safe_screen_stats[
                            "support_delta_factorized_fits"
                        ]
                    ),
                    "support_semantic_warm_starts": int(
                        self._safe_screen_stats[
                            "support_semantic_warm_starts"
                        ]
                    ),
                    "support_delta_factorization_fallbacks": int(
                        self._safe_screen_stats[
                            "support_closure_partition_fallbacks"
                        ]
                    ),
                    "joint_null_kkt_screens": int(
                        self._safe_screen_stats[
                            "support_joint_null_kkt_screens"
                        ]
                    ),
                    "profile_fit_results_reused": int(
                        self._safe_screen_stats["profile_fit_results_reused"]
                    ),
                    "profile_null_fit_batches": int(
                        self._safe_screen_stats["profile_null_fit_batches"]
                    ),
                    "profile_null_models_batched": int(
                        self._safe_screen_stats["profile_null_models_batched"]
                    ),
                    "profile_null_models_scalar": int(
                        self._safe_screen_stats["profile_null_models_scalar"]
                    ),
                    "identity_zero_boundary_kkt_screens": int(
                        self._safe_screen_stats[
                            "identity_zero_boundary_kkt_screens"
                        ]
                    ),
                    "identity_sign_pair_parent_designs": int(
                        self._safe_screen_stats[
                            "identity_sign_pair_parent_designs"
                        ]
                    ),
                    "identity_sign_pair_child_reuses": int(
                        self._safe_screen_stats[
                            "identity_sign_pair_child_reuses"
                        ]
                    ),
                    "identity_fused_profile_exact_fits": int(
                        self._safe_screen_stats[
                            "identity_fused_profile_exact_fits"
                        ]
                    ),
                    "identity_incremental_partition_rebuilds": int(
                        self._safe_screen_stats[
                            "identity_incremental_partition_rebuilds"
                        ]
                    ),
                    "nested_prepared_parent_designs": int(
                        self._safe_screen_stats["nested_prepared_parent_designs"]
                    ),
                    "nested_prepared_child_reuses": int(
                        self._safe_screen_stats["nested_prepared_child_reuses"]
                    ),
                },
                "hierarchy_null_loss_cache": {
                    **self._loss_summary_cache_stats,
                    "bytes": int(self._loss_summary_cache_size[0]),
                    "byte_limit": int(self.config.loss_summary_cache_bytes),
                    "entries": len(self._loss_summary_cache),
                    "payload": "entity_intensity_and_nll_only",
                },
                "reused_full_fit_summary_cache": {
                    **self._fit_summary_cache_stats,
                    "bytes": int(self._fit_summary_cache_size[0]),
                    "byte_limit": int(self.config.fit_summary_cache_bytes),
                    "entries": len(self._fit_summary_cache),
                    "eligibility": "frozen_full_or_drop_model_used_more_than_once",
                },
                "fisher_sufficient_statistic_chunk_rows": 262_144,
                "sparse_zero_background_materialized": False,
                "immutable_split_geometry_cached": [
                    "sequence_exposure",
                    "event_grid_rows",
                    "event_grid_multiplicity",
                ],
                "hierarchy_closure_and_null_fit_index_cached": True,
                "final_solver_derivatives_reused": True,
                "optimization_invariance": (
                    "all runtime shortcuts reuse algebraically identical sufficient statistics, "
                    "cached immutable geometry, or a compiled evaluation of the same convex objective; "
                    "every native solution is re-certified by the ordinary host-float64 projected KKT test"
                ),
            },
            "active_support_count": int(
                sum(record.fit.converged and np.all(self._active_amplitudes(record.fit)) for record in supports)
            ),
            "fit_screen": fit_screen,
            "certification": certification,
            "ensemble": ensemble,
            "timing_seconds": {
                "baseline": baseline_done - started,
                "rule_identity_library": profile_done - baseline_done,
                "support_search": search_done - profile_done,
                "fit_screen": fit_screen_done - search_done,
                "certification": certification_done - fit_screen_done,
                "ensemble_and_test": ensemble_done - certification_done,
                "total": elapsed,
            },
        }


def save_result(result: dict, path: str | Path) -> None:
    def json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        json_safe(result), indent=2, ensure_ascii=False, allow_nan=False
    )
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
