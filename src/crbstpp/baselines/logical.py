from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..data import Dataset
from ..likelihood import loss_value_rows
from ..response import Context, ResponseEngine
from ..rules import RuleIdentity, Support
from ..solver import fit_model_matrix_continued
from .config import BaselineConfig
from .data import LandmarkSplit
from .metrics import classification_metrics
from .seed import set_reproducible_seed


@dataclass(frozen=True, order=True)
class LogicRule:
    antecedent: tuple[int, ...]

    def activation(self, presence: np.ndarray) -> np.ndarray:
        return np.all(presence[:, self.antecedent], axis=1).astype(np.float64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    output = np.empty_like(value, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _rule_design(presence: np.ndarray, rules: tuple[LogicRule, ...]) -> np.ndarray:
    if not rules:
        return np.zeros((len(presence), 0), dtype=np.float64)
    return np.column_stack([rule.activation(presence) for rule in rules])


@dataclass(frozen=True)
class LogicFit:
    intercept: float
    coefficients: np.ndarray
    nll: float
    converged: bool


def _fit_rule_logistic(
    presence: np.ndarray,
    outcomes: np.ndarray,
    rules: tuple[LogicRule, ...],
    *,
    l1: float,
) -> LogicFit:
    design = _rule_design(presence, rules)
    rate = np.clip(np.mean(outcomes), 1.0e-8, 1.0 - 1.0e-8)
    initial = np.r_[math.log(rate / (1.0 - rate)), np.zeros(len(rules))]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = parameters[0] + design @ parameters[1:]
        probability = _sigmoid(eta)
        value = float(
            np.sum(np.logaddexp(0.0, eta) - outcomes * eta)
            + l1 * np.sum(np.sqrt(parameters[1:] ** 2 + 1.0e-10))
        )
        residual = probability - outcomes
        gradient = np.r_[np.sum(residual), design.T @ residual]
        if len(rules):
            gradient[1:] += l1 * parameters[1:] / np.sqrt(
                parameters[1:] ** 2 + 1.0e-10
            )
        return value, gradient

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    return LogicFit(
        float(result.x[0]),
        np.asarray(result.x[1:], dtype=np.float64),
        float(result.fun),
        bool(result.success and np.isfinite(result.fun)),
    )


def _probability(
    fit: LogicFit, presence: np.ndarray, rules: tuple[LogicRule, ...]
) -> np.ndarray:
    return _sigmoid(fit.intercept + _rule_design(presence, rules) @ fit.coefficients)


def _partition_context(dataset: Dataset, partitions: tuple[int, ...]) -> Context:
    if dataset.partitions is None:
        raise ValueError("logical TPP baselines require frozen partitions")
    entities = np.flatnonzero(np.isin(dataset.partitions, partitions)).astype(np.int32)
    return Context.make(dataset, entities)


def _tpp_rule_readout(
    dataset: Dataset,
    config: BaselineConfig,
    rules: tuple[LogicRule, ...],
    coefficients: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    """Refit discovered rules with the experiment's exact TPP likelihood.

    Branch pricing and differentiable covering discover the logical
    antecedents.  Their final amplitudes are then estimated on D_fit+D_cert
    using the same target process, baseline, strict-future response and
    effect horizon as CRBS-TPP.  The sign is frozen from the discovery model;
    its nonnegative one-amplitude kernel is optimized exactly.
    """

    identities = tuple(
        RuleIdentity(
            rule.antecedent,
            0 if len(rule.antecedent) == 1 else int(config.history_horizon),
            1 if float(coefficient) >= 0.0 else -1,
            kernel_rank=1,
            relation="atomic" if len(rule.antecedent) == 1 else "unordered",
            support_additive=True,
        )
        for rule, coefficient in zip(rules, coefficients, strict=True)
    )
    support = Support.of(identities)
    engine = ResponseEngine(
        dataset,
        lag=config.effect_horizon,
        knot_count=1,
        cache_bytes=config.cache_bytes,
        baseline_time_bins=config.baseline_time_bins,
        effect_model="support_additive",
    )
    train_context = _partition_context(dataset, (0, 1))
    test_context = _partition_context(dataset, (2,))
    train_matrix = engine.model_matrix(train_context, support)
    fit = fit_model_matrix_continued(
        train_matrix,
        likelihood=dataset.likelihood,
        tolerance=1.0e-7,
        max_iter=200,
        device=config.device,
    )
    if not fit.converged:
        raise RuntimeError(f"logical TPP readout did not converge: {fit.message}")
    test_matrix = engine.model_matrix(test_context, support)
    if test_matrix.dimension != len(fit.coefficients):
        raise AssertionError("logical TPP train/test column layouts differ")
    eta = test_matrix.x @ fit.coefficients
    test_nll = float(
        np.sum(
            loss_value_rows(
                eta,
                likelihood=dataset.likelihood,
                exposure_weight=test_matrix.exposure_weight,
                noevent_weight=test_matrix.noevent_weight,
                event_weight=test_matrix.event_weight,
            ),
            dtype=np.float64,
        )
    )
    return (
        {
            "likelihood": dataset.likelihood,
            "effect_horizon": config.effect_horizon,
            "formation_window": config.history_horizon,
            "test_nll_per_entity": test_nll / max(1, len(test_context.entity_codes)),
            "converged": True,
            "projected_kkt": fit.projected_kkt,
        },
        fit.coefficients,
    )


def _normalized_score(mask: np.ndarray, residual: np.ndarray) -> float:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0.0
    return abs(float(np.sum(residual[mask]))) / math.sqrt(float(count))


def _branch_price_candidates(
    dimension: int,
    selected: tuple[LogicRule, ...],
    maximum_order: int,
) -> tuple[LogicRule, ...]:
    selected_set = set(selected)
    candidates = {LogicRule((index,)) for index in range(dimension)}
    for rule in selected:
        if len(rule.antecedent) >= maximum_order:
            continue
        for predicate in range(dimension):
            if predicate not in rule.antecedent:
                candidates.add(LogicRule(tuple(sorted((*rule.antecedent, predicate)))))
    return tuple(sorted(candidates - selected_set))


def fit_branch_price(
    landmarks: dict[int, LandmarkSplit],
    dataset: Dataset,
    config: BaselineConfig,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Scalable port of Li et al.'s branch-and-price rule generation.

    The authors' public code is pinned in the result provenance, but it is
    fixed to four synthetic predicates, Python 3.7 and Torch 1.2.  This port
    retains its alternating master problem and gradient-priced rule extension
    while accepting the experiment's immutable predicate dictionary.
    """

    set_reproducible_seed(seed)
    fit_rows, cert_rows, test_rows = landmarks[0], landmarks[1], landmarks[2]
    fit_presence = fit_rows.features > 0.0
    cert_presence = cert_rows.features > 0.0
    test_presence = test_rows.features > 0.0
    rules: tuple[LogicRule, ...] = ()
    fitted = _fit_rule_logistic(fit_presence, fit_rows.outcomes, rules, l1=0.1)
    best_cert = float("inf")
    best: tuple[tuple[LogicRule, ...], LogicFit] = (rules, fitted)
    deadline = time.monotonic() + config.logical_time_limit_seconds
    audits = 0
    while len(rules) < config.logical_max_rules and time.monotonic() < deadline:
        residual = fit_rows.outcomes - _probability(fitted, fit_presence, rules)
        candidates = _branch_price_candidates(
            fit_presence.shape[1], rules, config.logical_max_order
        )
        ranked = []
        for candidate in candidates:
            activation = np.all(
                fit_presence[:, candidate.antecedent], axis=1
            )
            ranked.append((_normalized_score(activation, residual), candidate))
        audits += len(ranked)
        if not ranked:
            break
        _, candidate = max(ranked, key=lambda item: (item[0], item[1]))
        trial_rules = tuple(sorted((*rules, candidate)))
        trial = _fit_rule_logistic(
            fit_presence, fit_rows.outcomes, trial_rules, l1=0.1
        )
        if not trial.converged or trial.nll >= fitted.nll - 1.0e-8:
            break
        rules, fitted = trial_rules, trial
        cert_probability = _probability(fitted, cert_presence, rules)
        cert_nll = float(classification_metrics(
            cert_rows.outcomes, cert_probability
        )["binary_nll"])
        if cert_nll < best_cert:
            best_cert = cert_nll
            best = rules, fitted
    rules, _ = best
    train_presence = np.concatenate((fit_presence, cert_presence), axis=0)
    train_outcomes = np.concatenate((fit_rows.outcomes, cert_rows.outcomes))
    final = _fit_rule_logistic(train_presence, train_outcomes, rules, l1=0.1)
    probability = _probability(final, test_presence, rules)
    point_process, point_process_coefficients = _tpp_rule_readout(
        dataset, config, rules, final.coefficients
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "model.npz",
        intercept=np.asarray([final.intercept]),
        coefficients=final.coefficients,
        point_process_coefficients=point_process_coefficients,
        antecedents=np.asarray([rule.antecedent for rule in rules], dtype=object),
    )
    return {
        "implementation": "scalable port of authors' RAFS/REFS branch-and-price",
        "official_commit": config.branch_price_commit,
        "official_repository": (
            "https://github.com/FengMingquan-sjtu/Logic_Point_Processes_ICLR"
        ),
        "generalization_reason": (
            "official code is hard-coded to four synthetic predicates and Torch 1.2"
        ),
        "gradient_candidates_audited": audits,
        "rules": [
            {
                "antecedent": list(rule.antecedent),
                "names": [dataset.predicate_names[index] for index in rule.antecedent],
                "coefficient": float(final.coefficients[position]),
            }
            for position, rule in enumerate(rules)
        ],
        "test": classification_metrics(test_rows.outcomes, probability),
        "point_process": point_process,
    }


def _train_soft_rule(
    presence: np.ndarray,
    outcomes: np.ndarray,
    existing_eta: np.ndarray,
    *,
    order: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
) -> tuple[LogicRule, float]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Neuro-Symbolic TPP requires PyTorch") from error
    set_reproducible_seed(seed)
    if device_name == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    dimension = presence.shape[1]
    logits = torch.zeros((order, dimension), device=device, requires_grad=True)
    weight = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam((logits, weight), lr=learning_rate)
    generator = np.random.default_rng(seed)
    n = len(outcomes)
    for _ in range(epochs):
        permutation = generator.permutation(n)
        for left in range(0, n, batch_size):
            indices = permutation[left : left + batch_size]
            x = torch.as_tensor(
                presence[indices], dtype=torch.float32, device=device
            )
            y = torch.as_tensor(outcomes[indices], dtype=torch.float32, device=device)
            offset = torch.as_tensor(
                existing_eta[indices], dtype=torch.float32, device=device
            )
            selector = torch.softmax(logits, dim=1)
            activation = torch.prod(x @ selector.T, dim=1)
            eta = offset + weight * activation
            loss = torch.nn.functional.binary_cross_entropy_with_logits(eta, y)
            # Entropy annealing makes the final embedding map to a compact rule.
            entropy = -torch.sum(selector * torch.log(selector + 1.0e-12))
            loss = loss + 1.0e-4 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    scores = logits.detach().cpu().numpy()
    chosen: list[int] = []
    for slot in range(order):
        for predicate in np.argsort(-scores[slot], kind="stable"):
            if int(predicate) not in chosen:
                chosen.append(int(predicate))
                break
    return LogicRule(tuple(sorted(chosen))), float(abs(weight.detach().cpu().item()))


def fit_neurosymbolic_tpp(
    landmarks: dict[int, LandmarkSplit],
    dataset: Dataset,
    config: BaselineConfig,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Differentiable rule embeddings with sequential covering (ICML 2024)."""

    set_reproducible_seed(seed)
    fit_rows, cert_rows, test_rows = landmarks[0], landmarks[1], landmarks[2]
    fit_presence = fit_rows.features > 0.0
    cert_presence = cert_rows.features > 0.0
    test_presence = test_rows.features > 0.0
    rules: tuple[LogicRule, ...] = ()
    current = _fit_rule_logistic(fit_presence, fit_rows.outcomes, rules, l1=0.0)
    best_cert = float("inf")
    best_rules = rules
    deadline = time.monotonic() + config.logical_time_limit_seconds
    for round_index in range(config.logical_max_rules):
        if time.monotonic() >= deadline:
            break
        existing_eta = current.intercept + _rule_design(
            fit_presence, rules
        ) @ current.coefficients
        proposals = []
        for order in range(1, config.logical_max_order + 1):
            rule, soft_weight = _train_soft_rule(
                fit_presence,
                fit_rows.outcomes,
                existing_eta,
                order=order,
                seed=seed + 104729 * round_index + order,
                epochs=min(config.max_epochs, 20),
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                device_name=config.device,
            )
            if rule in rules:
                continue
            trial_rules = tuple(sorted((*rules, rule)))
            trial = _fit_rule_logistic(
                fit_presence, fit_rows.outcomes, trial_rules, l1=0.0
            )
            proposals.append((trial.nll, -soft_weight, trial_rules, trial))
        if not proposals:
            break
        _, _, trial_rules, trial = min(proposals, key=lambda item: item[:2])
        if not trial.converged or trial.nll >= current.nll - 1.0e-8:
            break
        rules, current = trial_rules, trial
        cert_probability = _probability(current, cert_presence, rules)
        cert_nll = float(classification_metrics(
            cert_rows.outcomes, cert_probability
        )["binary_nll"])
        if cert_nll < best_cert:
            best_cert = cert_nll
            best_rules = rules
    train_presence = np.concatenate((fit_presence, cert_presence), axis=0)
    train_outcomes = np.concatenate((fit_rows.outcomes, cert_rows.outcomes))
    final = _fit_rule_logistic(
        train_presence, train_outcomes, best_rules, l1=0.0
    )
    probability = _probability(final, test_presence, best_rules)
    point_process, point_process_coefficients = _tpp_rule_readout(
        dataset, config, best_rules, final.coefficients
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "model.npz",
        intercept=np.asarray([final.intercept]),
        coefficients=final.coefficients,
        point_process_coefficients=point_process_coefficients,
        antecedents=np.asarray(
            [rule.antecedent for rule in best_rules], dtype=object
        ),
    )
    return {
        "implementation": (
            "paper-based differentiable predicate embeddings with sequential covering"
        ),
        "paper": "Yang et al., Neuro-Symbolic Temporal Point Processes, ICML 2024",
        "official_code_available": False,
        "rules": [
            {
                "antecedent": list(rule.antecedent),
                "names": [dataset.predicate_names[index] for index in rule.antecedent],
                "coefficient": float(final.coefficients[position]),
            }
            for position, rule in enumerate(best_rules)
        ],
        "test": classification_metrics(test_rows.outcomes, probability),
        "point_process": point_process,
    }
