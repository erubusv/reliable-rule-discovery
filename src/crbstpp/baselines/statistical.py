from __future__ import annotations

import importlib.metadata
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..data import Dataset
from ..response import Context, _temporal_baseline_layout
from .config import BaselineConfig
from .data import LandmarkSplit
from .metrics import classification_metrics
from .seed import set_reproducible_seed


def _standardize(
    fit: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    center = np.asarray(np.mean(fit, axis=0), dtype=np.float64)
    scale = np.asarray(np.std(fit, axis=0), dtype=np.float64)
    scale[scale < 1.0e-8] = 1.0
    return tuple(
        np.asarray((array - center) / scale, dtype=np.float32)
        for array in (fit, *others)
    )


def fit_logistic(
    landmarks: dict[int, LandmarkSplit],
    config: BaselineConfig,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Official scikit-learn L2 logistic regression with cert-only tuning."""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise RuntimeError(
            "logistic baseline requires `pip install -e '.[baselines]'`"
        ) from error
    set_reproducible_seed(seed)
    fit, cert, test = landmarks[0], landmarks[1], landmarks[2]
    x_fit, x_cert, x_test = _standardize(
        fit.features, cert.features, test.features
    )
    candidates = []
    for value in config.logistic_c:
        model = LogisticRegression(
            C=float(value),
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        model.fit(x_fit, fit.outcomes)
        probability = model.predict_proba(x_cert)[:, 1]
        metrics = classification_metrics(cert.outcomes, probability)
        candidates.append((float(metrics["binary_nll"]), float(value), model))
    _, best_c, _ = min(candidates, key=lambda item: (item[0], item[1]))
    x_train = np.concatenate((x_fit, x_cert), axis=0)
    y_train = np.concatenate((fit.outcomes, cert.outcomes), axis=0)
    final = LogisticRegression(
        C=best_c,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    final.fit(x_train, y_train)
    probability = final.predict_proba(x_test)[:, 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "implementation": "scikit-learn.LogisticRegression",
        "implementation_version": importlib.metadata.version("scikit-learn"),
        "selected_C": best_c,
        "selection_metric": "cert_binary_nll",
        "test": classification_metrics(test.outcomes, probability),
    }


def fit_xgboost(
    landmarks: dict[int, LandmarkSplit],
    config: BaselineConfig,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Official XGBoost histogram classifier with cert early stopping."""

    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "XGBoost baseline requires `pip install -e '.[baselines]'`"
        ) from error
    set_reproducible_seed(seed)
    fit, cert, test = landmarks[0], landmarks[1], landmarks[2]
    candidates = []
    for depth in config.xgboost_depths:
        for rate in config.xgboost_learning_rates:
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device=config.device,
                max_depth=int(depth),
                learning_rate=float(rate),
                n_estimators=2000,
                min_child_weight=1.0,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=1.0,
                random_state=seed,
                seed=seed,
                nthread=config.num_workers,
                early_stopping_rounds=50,
            )
            model.fit(
                fit.features,
                fit.outcomes,
                eval_set=[(cert.features, cert.outcomes)],
                verbose=False,
            )
            probability = model.predict_proba(cert.features)[:, 1]
            metrics = classification_metrics(cert.outcomes, probability)
            candidates.append(
                (
                    float(metrics["binary_nll"]),
                    int(depth),
                    float(rate),
                    int(model.best_iteration + 1),
                )
            )
    _, depth, rate, rounds = min(candidates)
    final = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=config.device,
        max_depth=depth,
        learning_rate=rate,
        n_estimators=rounds,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=seed,
        seed=seed,
        nthread=config.num_workers,
    )
    final.fit(
        np.concatenate((fit.features, cert.features), axis=0),
        np.concatenate((fit.outcomes, cert.outcomes), axis=0),
        verbose=False,
    )
    probability = final.predict_proba(test.features)[:, 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    final.save_model(output_dir / "model.ubj")
    return {
        "implementation": "xgboost.XGBClassifier",
        "implementation_version": importlib.metadata.version("xgboost"),
        "selected_max_depth": depth,
        "selected_learning_rate": rate,
        "selected_rounds": rounds,
        "selection_metric": "cert_binary_nll",
        "test": classification_metrics(test.outcomes, probability),
    }


@dataclass(frozen=True)
class HawkesStatistics:
    baseline_exposure: np.ndarray
    kernel_exposure: np.ndarray
    target_groups: np.ndarray
    target_features: np.ndarray
    target_counts: np.ndarray
    n_entities: int

    def __add__(self, other: "HawkesStatistics") -> "HawkesStatistics":
        if self.baseline_exposure.shape != other.baseline_exposure.shape:
            raise ValueError("Hawkes baseline layouts differ")
        return HawkesStatistics(
            self.baseline_exposure + other.baseline_exposure,
            self.kernel_exposure + other.kernel_exposure,
            np.concatenate((self.target_groups, other.target_groups)),
            np.concatenate((self.target_features, other.target_features), axis=0),
            np.concatenate((self.target_counts, other.target_counts)),
            self.n_entities + other.n_entities,
        )


def _partition_context(
    dataset: Dataset, split: int | tuple[int, ...]
) -> Context:
    if dataset.partitions is None:
        raise ValueError("Hawkes baseline requires frozen partitions")
    values = (split,) if isinstance(split, int) else split
    entities = np.flatnonzero(np.isin(dataset.partitions, values)).astype(np.int32)
    return Context.make(dataset, entities)


def _source_offsets(dataset: Dataset) -> np.ndarray:
    counts = np.bincount(dataset.event_entities, minlength=dataset.n_entities)
    return np.r_[0, np.cumsum(counts, dtype=np.int64)]


def hawkes_statistics(
    dataset: Dataset,
    config: BaselineConfig,
    split: int | tuple[int, ...],
    *,
    half_life: float,
) -> HawkesStatistics:
    """Exact sufficient statistics for one target row of exponential MHP."""

    context = _partition_context(dataset, split)
    beta = math.log(2.0) / float(half_life)
    continuous = dataset.likelihood == "continuous_poisson"
    dimension = dataset.n_reported_predicates
    group_count = int(
        np.count_nonzero(
            _temporal_baseline_layout(dataset, config.baseline_time_bins) >= 0
        )
    )
    baseline_exposure = context.weighted_baseline_totals(
        group_count, time_bins=config.baseline_time_bins
    )
    target_groups = context.temporal_baseline_groups_at_rows(
        context.target_rows, time_bins=config.baseline_time_bins
    )
    source_offsets = _source_offsets(dataset)
    target_offsets = np.r_[
        0,
        np.cumsum(
            np.bincount(dataset.target_entities, minlength=dataset.n_entities),
            dtype=np.int64,
        ),
    ]
    features = np.zeros((len(context.target_rows), dimension), dtype=np.float64)
    kernel_exposure = np.zeros(dimension, dtype=np.float64)
    target_cursor = 0
    for local, entity in enumerate(context.entity_codes):
        left, right = int(source_offsets[entity]), int(source_offsets[entity + 1])
        times = dataset.event_times[left:right]
        predicates = dataset.event_predicates[left:right]
        keep = predicates < dimension
        times = times[keep]
        predicates = predicates[keep]
        end_units = (float(context.ends[local]) - float(context.starts[local])) / float(
            dataset.ticks_per_unit
        )
        source_units = (times.astype(np.float64) - context.starts[local]) / float(
            dataset.ticks_per_unit
        )
        remaining = np.maximum(0.0, end_units - source_units)
        if continuous:
            contributions = 1.0 - np.exp(-beta * remaining)
        else:
            steps = np.floor(remaining).astype(np.int64)
            ratio = math.exp(-beta)
            contributions = ratio * (1.0 - np.power(ratio, steps)) / max(
                1.0 - ratio, np.finfo(float).tiny
            )
        np.add.at(kernel_exposure, predicates, contributions)
        target_left = int(target_offsets[entity])
        target_right = int(target_offsets[entity + 1])
        target_times = dataset.target_times[target_left:target_right]
        if not len(target_times):
            continue
        state = np.zeros(dimension, dtype=np.float64)
        source_cursor = 0
        previous = float(context.starts[local])
        for target_time in target_times:
            while source_cursor < len(times) and times[source_cursor] < target_time:
                current = float(times[source_cursor])
                elapsed = (current - previous) / float(dataset.ticks_per_unit)
                state *= math.exp(-beta * elapsed)
                same = current
                while source_cursor < len(times) and times[source_cursor] == same:
                    state[int(predicates[source_cursor])] += (
                        beta if continuous else 1.0
                    )
                    source_cursor += 1
                previous = current
            elapsed = (float(target_time) - previous) / float(dataset.ticks_per_unit)
            state *= math.exp(-beta * elapsed)
            previous = float(target_time)
            features[target_cursor] = state
            target_cursor += 1
    if target_cursor != len(features):
        raise AssertionError("target Hawkes features are incomplete")
    return HawkesStatistics(
        baseline_exposure=np.asarray(baseline_exposure, dtype=np.float64),
        kernel_exposure=kernel_exposure,
        target_groups=np.asarray(target_groups, dtype=np.int32),
        target_features=features,
        target_counts=np.asarray(context.target_counts, dtype=np.float64),
        n_entities=len(context.entity_codes),
    )


def _hawkes_value_gradient(
    parameters: np.ndarray,
    statistics: HawkesStatistics,
    likelihood: str,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    group_count = len(statistics.baseline_exposure)
    baseline = parameters[:group_count]
    coefficient = parameters[group_count:]
    kernel_exposure = statistics.kernel_exposure[: len(coefficient)]
    target_features = statistics.target_features[:, : len(coefficient)]
    value = float(
        statistics.baseline_exposure @ baseline
        + kernel_exposure @ coefficient
        + 0.5 * l2 * (coefficient @ coefficient)
    )
    gradient = np.r_[
        statistics.baseline_exposure.copy(),
        kernel_exposure + l2 * coefficient,
    ]
    intensity = baseline[statistics.target_groups]
    if len(coefficient):
        intensity = intensity + target_features @ coefficient
    intensity = np.maximum(intensity, 1.0e-12)
    counts = statistics.target_counts
    if likelihood == "continuous_poisson" or likelihood == "poisson":
        value -= float(np.sum(counts * np.log(intensity)))
        derivative = -counts / intensity
    elif likelihood == "first_event_cloglog":
        event_loss = -np.log(-np.expm1(-intensity))
        value += float(np.sum(counts * (event_loss - intensity)))
        derivative = counts * (-1.0 / np.expm1(intensity) - 1.0)
    else:
        raise ValueError(f"unsupported Hawkes likelihood: {likelihood}")
    np.add.at(gradient[:group_count], statistics.target_groups, derivative)
    if len(coefficient):
        gradient[group_count:] += target_features.T @ derivative
    return value, gradient


@dataclass(frozen=True)
class HawkesFit:
    parameters: np.ndarray
    nll: float
    converged: bool


def fit_hawkes_parameters(
    statistics: HawkesStatistics,
    likelihood: str,
    *,
    source_dimension: int,
    warm_start: np.ndarray | None = None,
) -> HawkesFit:
    groups = len(statistics.baseline_exposure)
    target_total = float(np.sum(statistics.target_counts))
    exposure_total = float(np.sum(statistics.baseline_exposure))
    rate = max(1.0e-8, target_total / max(exposure_total, 1.0e-12))
    initial = (
        np.r_[
            np.full(groups, rate, dtype=np.float64),
            np.full(source_dimension, 1.0e-4, dtype=np.float64),
        ]
        if warm_start is None
        else np.asarray(warm_start, dtype=np.float64)
    )
    if initial.shape != (groups + source_dimension,) or np.any(initial <= 0.0):
        raise ValueError("invalid Hawkes warm start")
    # Continuous-time baselines are around 1e-6 while Hawkes amplitudes can be
    # several orders larger.  Optimizing both directly under box constraints
    # makes L-BFGS-B stop at its iteration limit even when the model is small.
    # The shifted softplus is a one-to-one reparameterization of exactly the
    # same positive parameter space.  Its chain-rule gradient removes that
    # scale mismatch without changing the likelihood or fitted model.
    lower = 1.0e-10
    shifted = np.maximum(initial - lower, np.finfo(np.float64).tiny)
    unconstrained = np.where(
        shifted > 30.0,
        shifted,
        np.log(np.expm1(shifted)),
    )

    def transformed_value_gradient(value: np.ndarray) -> tuple[float, np.ndarray]:
        positive = lower + np.logaddexp(0.0, value)
        objective, gradient = _hawkes_value_gradient(
            positive, statistics, likelihood, l2=1.0e-8
        )
        sigmoid = np.empty_like(value)
        nonnegative = value >= 0.0
        sigmoid[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
        exponential = np.exp(value[~nonnegative])
        sigmoid[~nonnegative] = exponential / (1.0 + exponential)
        return objective, gradient * sigmoid

    result = minimize(
        transformed_value_gradient,
        unconstrained,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1500, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    parameters = lower + np.logaddexp(0.0, result.x)
    return HawkesFit(
        np.asarray(parameters, dtype=np.float64),
        float(result.fun),
        bool(result.success and np.isfinite(result.fun)),
    )


def evaluate_hawkes(
    fit: HawkesFit,
    statistics: HawkesStatistics,
    likelihood: str,
) -> float:
    value, _ = _hawkes_value_gradient(
        fit.parameters, statistics, likelihood, l2=0.0
    )
    return value / max(1, statistics.n_entities)


def fit_point_process_baseline(
    dataset: Dataset,
    config: BaselineConfig,
    *,
    source_dimension: int,
    output_dir: Path,
) -> dict[str, object]:
    half_lives = (
        (math.inf,) if source_dimension == 0 else config.hawkes_half_lives
    )
    candidates = []
    for half_life in half_lives:
        effective = 1.0 if not math.isfinite(half_life) else float(half_life)
        fit_stats = hawkes_statistics(dataset, config, 0, half_life=effective)
        cert_stats = hawkes_statistics(dataset, config, 1, half_life=effective)
        fitted = fit_hawkes_parameters(
            fit_stats, dataset.likelihood, source_dimension=source_dimension
        )
        if not fitted.converged:
            continue
        candidates.append(
            (
                evaluate_hawkes(fitted, cert_stats, dataset.likelihood),
                effective,
                fitted.parameters,
            )
        )
    if not candidates:
        raise RuntimeError("all point-process baseline fits failed")
    cert_nll, half_life, warm_start = min(candidates, key=lambda item: item[0])
    train = hawkes_statistics(dataset, config, (0, 1), half_life=half_life)
    test = hawkes_statistics(dataset, config, 2, half_life=half_life)
    final = fit_hawkes_parameters(
        train,
        dataset.likelihood,
        source_dimension=source_dimension,
        warm_start=warm_start,
    )
    if not final.converged:
        raise RuntimeError("final point-process baseline fit failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "model.npz",
        parameters=final.parameters,
        half_life=np.asarray([half_life]),
    )
    return {
        "implementation": (
            "intercept-only target TPP"
            if source_dimension == 0
            else "positive exponential multivariate Hawkes target row"
        ),
        "selected_half_life": None if source_dimension == 0 else half_life,
        "cert_nll_per_entity": cert_nll,
        "test_nll_per_entity": evaluate_hawkes(
            final, test, dataset.likelihood
        ),
        "converged": final.converged,
        "source_types": source_dimension,
    }
