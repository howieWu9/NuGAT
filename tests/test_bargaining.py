import math
import json
import sys

import pytest
import torch
import numpy as np
import yaml
from torch.nn import functional as F

from nugat.bargaining import (
    BargainingConvergenceError, BargainingSettings, bargaining_terms,
    estimate_log_ideals, solve_bargaining,
)


def test_egs_and_kss_use_declared_objectives_and_positive_utility_equalities():
    utilities = torch.tensor([[0.2, 0.4], [0.3, 0.3]], dtype=torch.float64)
    objective, residual = bargaining_terms(utilities.log(), "egs")
    torch.testing.assert_close(objective, -utilities.min(dim=1).values)
    torch.testing.assert_close(residual[:, 0], torch.tensor([math.log(2), 0.0], dtype=torch.float64))
    ideals = torch.tensor([[0.4, 0.8], [0.6, 0.6]], dtype=torch.float64)
    objective, residual = bargaining_terms(utilities.log(), "kss", ideals.log())
    torch.testing.assert_close(objective, torch.full((2,), -0.5, dtype=torch.float64))
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-14, rtol=0)


@pytest.mark.parametrize("solution", ["egs", "kss"])
def test_augmented_lagrangian_finds_known_feasible_toy_solution(solution):
    value = torch.nn.Parameter(torch.tensor([0.7, -0.2], dtype=torch.float64))
    scales = torch.tensor([0.8, 0.4], dtype=torch.float64) if solution == "kss" else torch.ones(2, dtype=torch.float64)
    def evaluate(indices):
        x = value[indices]
        return torch.stack([F.logsigmoid(x), F.logsigmoid(-x)], dim=-1) + scales.log()
    ideals = scales.log().expand(2, 2).clone() if solution == "kss" else None
    result = solve_bargaining(
        [value], evaluate, 2, 2, solution=solution, log_ideals=ideals,
        settings=BargainingSettings(epochs=350, batch_size=2, learning_rate=0.02,
                                  dual_update_every=25, equality_tolerance=0.02),
    )
    assert result.feasible
    assert result.maximum_log_equality_residual <= 0.02
    assert value.detach().abs().max() < 0.02
    assert result.dual_variables.shape == (2, 1)
    assert result.dual_variables.abs().sum() > 0


def test_infeasible_candidates_raise_instead_of_claiming_convergence():
    value = torch.nn.Parameter(torch.zeros(1))
    def evaluate(indices):
        return torch.stack([value[indices] * 0 - 1, value[indices] * 0 - 3], dim=-1)
    with pytest.raises(BargainingConvergenceError) as failure:
        solve_bargaining([value], evaluate, 1, 2, solution="egs",
                         settings=BargainingSettings(epochs=2, batch_size=1))
    assert not failure.value.result.feasible
    assert failure.value.result.maximum_log_equality_residual == 2


def test_ideal_utilities_are_attained_estimates_from_separate_fresh_runs():
    starts = []
    def factory():
        value = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        starts.append(value.detach().clone())
        def evaluate(indices):
            x = value[indices]
            return torch.stack([F.logsigmoid(x), F.logsigmoid(-x)], dim=-1)
        return [value], evaluate
    ideals, report = estimate_log_ideals(factory, 2, 2, epochs=5, batch_size=2, learning_rate=0.1)
    assert len(starts) == 2
    assert all(torch.equal(s, torch.zeros_like(s)) for s in starts)
    assert torch.all(ideals.exp() > 0.5)
    assert torch.all(ideals.exp() < 1)
    assert not ideals.requires_grad
    assert report["all_modalities_present"]
    assert not report["global_upper_bound_certified"]


def test_kss_requires_fixed_positive_ideal_utilities():
    logs = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="fixed per-example"):
        bargaining_terms(logs, "kss")
    with pytest.raises(ValueError, match="remain fixed"):
        bargaining_terms(logs, "kss", torch.zeros_like(logs, requires_grad=True))


@pytest.mark.parametrize("solution", ["nash", "egs", "kss", "uts"])
def test_comparison_cli_runs_on_explicit_numerical_fixtures(tmp_path, monkeypatch, capsys, solution):
    """Synthetic fixtures validate plumbing only, never manuscript results."""
    from scripts.train_bargaining import main
    from nugat.training import load_checkpoint

    rng = np.random.default_rng(3)
    records = []
    for index in range(2):
        np.save(tmp_path / f"q{index}.npy", rng.normal(size=4).astype("float32"))
        paths = {}
        for modality in ("audio", "video"):
            path = f"{modality}{index}.npy"
            np.save(tmp_path / path, rng.normal(size=(4, 4)).astype("float32"))
            paths[modality] = path
        records.append({"id": f"fixture-{index}", "split": "train", "query": "fixture query",
                        "query_features": f"q{index}.npy", "modalities": paths})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "dataset": "synthetic_mechanics_fixture_only",
        "model": {"num_evidence_units": 2, "evidence_dim": 4, "iterations": 1},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-3,
                     "seeds": [7], "device": "cpu"},
    }), encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["train_bargaining.py", "--config", str(config_path),
                        "--manifest", str(manifest), "--output", str(output), "--solution", solution,
                        "--ideal-epochs", "1", "--allow-infeasible"])
    main()
    capsys.readouterr()
    report = json.loads((output / "seed_7" / "result.json").read_text())
    assert report["solution"] == solution and report["epochs"] == 1
    assert not report["reported_paper_results_reproduced"]
    checkpoint_name = "checkpoint.pt" if report["feasible"] else "candidate_infeasible.pt"
    _, checkpoint = load_checkpoint(output / "seed_7" / checkpoint_name)
    assert checkpoint["bargaining_feasible"] == report["feasible"]
    assert checkpoint["noise_protocol"] == "fixed_per_minibatch_common_random_numbers"
    assert checkpoint["model_kwargs"]["objective"] == ("utilitarian" if solution == "uts" else "nash")
    if solution == "kss":
        ideals = torch.load(output / "seed_7" / "estimated_ideals.pt", weights_only=True)
        assert ideals["ids"] == ["fixture-0", "fixture-1"]
        assert ideals["log_ideals"].shape == (2, 2)
