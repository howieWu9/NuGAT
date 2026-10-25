"""Explicit constrained alternatives to the Nash utility objective.

EGS and KSS are solved with an augmented Lagrangian, including per-example
dual variables and constraint checks. This is a numerical local solver for a
nonconvex neural model, not a guarantee of a globally optimal bargaining
solution. Solver settings and feasibility residuals are recorded with each run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


UtilityCallback = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class BargainingSettings:
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-4
    dual_update_every: int = 3
    initial_penalty: float = 1.0
    penalty_multiplier: float = 2.0
    maximum_penalty: float = 1e6
    equality_tolerance: float = 1e-3

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "dual_update_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("learning_rate", "initial_penalty", "maximum_penalty", "equality_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.penalty_multiplier <= 1 or not math.isfinite(self.penalty_multiplier):
            raise ValueError("penalty_multiplier must be finite and greater than one.")
        if self.maximum_penalty < self.initial_penalty:
            raise ValueError("maximum_penalty must be at least initial_penalty.")


@dataclass
class BargainingResult:
    solution: str
    feasible: bool
    maximum_log_equality_residual: float
    mean_objective: float
    history: list[dict[str, float | int]]
    dual_variables: Tensor
    settings: BargainingSettings

    def summary(self) -> dict:
        return {
            "solution": self.solution,
            "feasible": self.feasible,
            "maximum_log_equality_residual": self.maximum_log_equality_residual,
            "mean_objective": self.mean_objective,
            "history": self.history,
            "settings": asdict(self.settings),
            "constraint_scope": "per_training_example",
            "optimality": "not_certified; numerical local candidate",
        }


class BargainingConvergenceError(RuntimeError):
    def __init__(self, result: BargainingResult):
        self.result = result
        super().__init__(
            f"{result.solution.upper()} equality constraints are not satisfied: "
            f"maximum log residual {result.maximum_log_equality_residual:.6g} "
            f"> tolerance {result.settings.equality_tolerance:.6g}. "
            "Do not report this candidate as a converged constrained solution."
        )


def bargaining_terms(log_utilities: Tensor, solution: str, log_ideals: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return per-example negative min utility and exact-equivalent equalities.

    EGS: objective -min(U), with U_m = U_0.
    KSS: objective -min(U / ideal), with U_m / ideal_m = U_0 / ideal_0.
    Because utilities are positive, equality of their logarithms is equivalent
    to equality of the original values. Log residuals avoid treating two tiny
    but very unequal utilities as feasible under an absolute utility tolerance.
    """
    if log_utilities.ndim != 2 or log_utilities.shape[1] < 2:
        raise ValueError("Bargaining requires [batch, modalities] with at least two modalities.")
    if not bool(torch.isfinite(log_utilities).all()):
        raise FloatingPointError("Log utilities must be finite.")
    solution = solution.lower()
    if solution == "egs":
        if log_ideals is not None:
            raise ValueError("EGS does not use ideal utilities.")
        normalized_logs = log_utilities
    elif solution == "kss":
        if log_ideals is None or log_ideals.shape != log_utilities.shape:
            raise ValueError("KSS requires fixed per-example log ideals matching the utility shape.")
        if not bool(torch.isfinite(log_ideals).all()):
            raise ValueError("Estimated log ideal utilities must be finite.")
        if log_ideals.requires_grad:
            raise ValueError("KSS ideal utilities must remain fixed during constrained optimization.")
        normalized_logs = log_utilities - log_ideals.to(log_utilities)
    else:
        raise ValueError("The constrained solver supports only 'egs' or 'kss'.")
    objective = -normalized_logs.amin(dim=-1).exp()
    residual = normalized_logs[:, 1:] - normalized_logs[:, :1]
    if not bool(torch.isfinite(objective).all()):
        raise FloatingPointError("Bargaining objective overflows; inspect the estimated ideal utilities.")
    return objective, residual


def _checked_logs(evaluate: UtilityCallback, indices: Tensor, modalities: int) -> Tensor:
    logs = evaluate(indices)
    if not isinstance(logs, Tensor) or logs.shape != (len(indices), modalities):
        raise ValueError("Utility callback must return log utilities [requested examples, modalities].")
    if not bool(torch.isfinite(logs).all()):
        raise FloatingPointError("Utility callback returns nonfinite log utilities.")
    return logs


def evaluate_log_utilities(evaluate: UtilityCallback, examples: int, modalities: int, batch_size: int) -> Tensor:
    """Evaluate in stable ID order, returning a detached CPU tensor."""
    if min(examples, modalities, batch_size) < 1:
        raise ValueError("Dataset size, modalities, and batch_size must be positive.")
    pieces = []
    with torch.no_grad():
        for start in range(0, examples, batch_size):
            indices = torch.arange(start, min(start + batch_size, examples))
            pieces.append(_checked_logs(evaluate, indices, modalities).detach().cpu())
    return torch.cat(pieces, dim=0)


def solve_bargaining(
    parameters: Iterable[nn.Parameter],
    evaluate: UtilityCallback,
    examples: int,
    modalities: int,
    *,
    solution: str,
    log_ideals: Tensor | None = None,
    settings: BargainingSettings | None = None,
    allow_infeasible: bool = False,
) -> BargainingResult:
    """Fit a constrained neural candidate with per-query multiplier updates.

    ``evaluate(indices)`` must map stable integer training IDs to log utilities
    and be deterministic for a fixed parameter state. For stochastic gates use
    a fixed noise realization (common random numbers) and separately audit the
    chosen evaluation gate rule. Neither test nor validation data enter this
    optimization. Equality feasibility, not global optimality, is certified by
    the returned numerical residual.
    """
    settings = settings or BargainingSettings()
    settings.validate()
    if examples < 1 or modalities < 2:
        raise ValueError("Use a nonempty training set and at least two modalities.")
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters.")
    solution = solution.lower()
    if solution == "kss":
        if log_ideals is None or log_ideals.shape != (examples, modalities):
            raise ValueError("KSS requires log ideals for every training example and modality.")
        log_ideals = log_ideals.detach().cpu().clone()
    elif solution != "egs" or log_ideals is not None:
        raise ValueError("Use EGS without ideals or KSS with fixed ideals.")
    optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
    duals = torch.zeros(examples, modalities - 1, dtype=torch.float64)
    penalty = settings.initial_penalty
    previous_residual = math.inf
    history = []
    for epoch in range(1, settings.epochs + 1):
        for start in range(0, examples, settings.batch_size):
            indices = torch.arange(start, min(start + settings.batch_size, examples))
            optimizer.zero_grad(set_to_none=True)
            logs = _checked_logs(evaluate, indices, modalities)
            ideals = None if log_ideals is None else log_ideals[indices].to(logs)
            objective, residual = bargaining_terms(logs, solution, ideals)
            multiplier = duals[indices].to(logs)
            augmented = (
                objective + (multiplier * residual).sum(dim=-1)
                + 0.5 * penalty * residual.square().sum(dim=-1)
            ).mean()
            if not bool(torch.isfinite(augmented)):
                raise FloatingPointError("Nonfinite augmented Lagrangian.")
            augmented.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Nonfinite constrained-training gradient.")
            optimizer.step()
        all_logs = evaluate_log_utilities(evaluate, examples, modalities, settings.batch_size)
        all_objectives, all_residuals = bargaining_terms(all_logs, solution, log_ideals)
        maximum = float(all_residuals.abs().max())
        record = {
            "epoch": epoch,
            "mean_objective": float(all_objectives.mean()),
            "maximum_log_equality_residual": maximum,
            "penalty": penalty,
        }
        history.append(record)
        if epoch % settings.dual_update_every == 0 or epoch == settings.epochs:
            duals += penalty * all_residuals.double()
            if maximum > 0.75 * previous_residual:
                penalty = min(settings.maximum_penalty, penalty * settings.penalty_multiplier)
            previous_residual = maximum
    result = BargainingResult(
        solution=solution,
        feasible=maximum <= settings.equality_tolerance,
        maximum_log_equality_residual=maximum,
        mean_objective=float(all_objectives.mean()),
        history=history,
        dual_variables=duals,
        settings=settings,
    )
    if not result.feasible and not allow_infeasible:
        raise BargainingConvergenceError(result)
    return result


def estimate_log_ideals(
    candidate_factory: Callable[[], tuple[Iterable[nn.Parameter], UtilityCallback]],
    examples: int,
    modalities: int,
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
) -> tuple[Tensor, dict]:
    """Estimate attainable ideal utilities by separate modality maximizations.

    Each factory call must create the same initial model, retaining *all*
    modalities. Run m minimizes -U_m; the largest observed U_m for each example
    over initialization and epoch checkpoints is stored. These are attained
    estimates, not proven upper bounds or globally maximized utility values.
    """
    if min(examples, modalities, epochs, batch_size) < 1:
        raise ValueError("Sizes and the ideal-point epoch budget must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    columns = []
    records = []
    for modality in range(modalities):
        parameters, evaluate = candidate_factory()
        parameters = [p for p in parameters if p.requires_grad]
        if not parameters:
            raise ValueError("Candidate factory returns no trainable parameters.")
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        best = evaluate_log_utilities(evaluate, examples, modalities, batch_size)[:, modality]
        initial_mean = float(best.exp().mean())
        for _ in range(epochs):
            for start in range(0, examples, batch_size):
                indices = torch.arange(start, min(start + batch_size, examples))
                optimizer.zero_grad(set_to_none=True)
                logs = _checked_logs(evaluate, indices, modalities)
                loss = -logs[:, modality].exp().mean()
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise FloatingPointError("Nonfinite ideal-point training gradient.")
                optimizer.step()
            observed = evaluate_log_utilities(evaluate, examples, modalities, batch_size)[:, modality]
            best = torch.maximum(best, observed)
        columns.append(best)
        records.append({"modality_index": modality, "epochs": epochs, "initial_mean_utility": initial_mean,
                        "mean_attained_ideal_utility": float(best.exp().mean())})
    return torch.stack(columns, dim=-1).detach(), {
        "runs": records,
        "all_modalities_present": True,
        "scope": "per_training_example",
        "definition": "largest_attained_utility_over_separate_modality_run_checkpoints",
        "global_upper_bound_certified": False,
    }
