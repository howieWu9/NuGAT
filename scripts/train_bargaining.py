from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from nugat.bargaining import (
    BargainingSettings, bargaining_terms, estimate_log_ideals,
    evaluate_log_utilities, solve_bargaining,
)
from nugat.data import ManifestDataset, collate_evidence
from nugat.fusion import utility_objective
from nugat.training import TrainingSettings, build_model, resolve_device, seed_everything


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solution", choices=("nash", "egs", "kss", "uts"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ideal-epochs", type=int)
    parser.add_argument("--dual-update-every", type=int, default=3)
    parser.add_argument("--initial-penalty", type=float, default=1.0)
    parser.add_argument("--equality-tolerance", type=float, default=1e-3)
    parser.add_argument("--allow-infeasible", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("Config must be a YAML mapping")
    if args.ideal_epochs is not None and args.ideal_epochs < 1:
        parser.error("ideal-epochs must be positive")
    if config.get("model", {}).get("stochastic_eval", False):
        parser.error("This comparison runner audits deterministic evaluation; set stochastic_eval: false")
    training = TrainingSettings.from_mapping(config.get("training", {}))
    dataset = ManifestDataset(args.manifest, "train")
    examples, modalities = len(dataset), len(dataset.schema.modality_dims)
    if modalities < 2:
        parser.error("Comparison training requires at least two modalities")
    device = resolve_device(training.device)
    seeds = (args.seed,) if args.seed is not None else training.seeds
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = BargainingSettings(
        epochs=training.epochs, batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        dual_update_every=args.dual_update_every,
        initial_penalty=args.initial_penalty,
        equality_tolerance=args.equality_tolerance,
    )
    settings.validate()
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    model_objective = "utilitarian" if args.solution == "uts" else "nash"
    effective_config = {**config, "model": {**config.get("model", {}), "objective": model_objective}}
    summaries = []
    any_infeasible = False
    for seed in seeds:
        seed_everything(seed)
        run_dir = args.output / f"seed_{seed}"
        run_dir.mkdir()
        initial_model, kwargs = build_model(effective_config, dataset, allow_alternative_objective=True)
        initial_state = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}

        def evaluator(model: torch.nn.Module):
            def evaluate(indices: torch.Tensor) -> torch.Tensor:
                batch = collate_evidence([dataset[int(i)] for i in indices]).to(device)


                seed_bytes = f"{seed}:".encode() + indices.numpy().tobytes()
                batch_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "little")
                generator = torch.Generator(device=device).manual_seed(batch_seed)
                return model(batch.features, batch.query, masks=batch.masks, generator=generator).log_utilities
            return evaluate

        def candidate_factory():
            model, _ = build_model(effective_config, dataset, allow_alternative_objective=True)
            model.load_state_dict(initial_state)
            model.to(device).train()
            return model.parameters(), evaluator(model)

        log_ideals = None
        ideal_report = None
        if args.solution == "kss":
            log_ideals, ideal_report = estimate_log_ideals(
                candidate_factory, examples, modalities,
                epochs=args.ideal_epochs or training.epochs,
                batch_size=training.batch_size, learning_rate=training.learning_rate,
            )
            torch.save({"ids": [r["id"] for r in dataset.records],
                        "modalities": list(dataset.schema.modality_dims),
                        "log_ideals": log_ideals,
                        "manifest_sha256": manifest_sha}, run_dir / "estimated_ideals.pt")
            write_json(run_dir / "ideal_search.json", ideal_report)
        model, kwargs = build_model(effective_config, dataset, allow_alternative_objective=True)
        model.load_state_dict(initial_state)
        model.to(device).train()
        evaluate = evaluator(model)
        if args.solution in {"nash", "uts"}:
            optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
            history = []
            for epoch in range(1, training.epochs + 1):
                total, count = 0.0, 0
                for start in range(0, examples, training.batch_size):
                    indices = torch.arange(start, min(start + training.batch_size, examples))
                    optimizer.zero_grad(set_to_none=True)
                    loss = utility_objective(evaluate(indices), objective=model_objective,
                                             epsilon=kwargs.get("epsilon", 1e-8))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite comparison objective")
                    loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError("Nonfinite comparison gradient")
                    optimizer.step()
                    total += float(loss.detach()) * len(indices)
                    count += len(indices)
                history.append({"epoch": epoch, "mean_objective": total / count})
            report = {"solution": args.solution, "feasible": True, "history": history,
                      "constraint_scope": "none", "optimality": "not_certified; numerical local candidate"}
            duals = None
        else:
            result = solve_bargaining(
                model.parameters(), evaluate, examples, modalities,
                solution=args.solution, log_ideals=log_ideals,
                settings=settings, allow_infeasible=True,
            )
            report = result.summary()
            duals = result.dual_variables
            model.eval()
            eval_logs = evaluate_log_utilities(evaluate, examples, modalities, training.batch_size)
            _, eval_residual = bargaining_terms(eval_logs, args.solution, log_ideals)
            evaluation_residual = float(eval_residual.abs().max())
            report["deterministic_eval_maximum_log_equality_residual"] = evaluation_residual
            if log_ideals is not None:
                report["deterministic_eval_maximum_estimated_ideal_fraction"] = float((eval_logs - log_ideals).exp().max())
            report["training_noise_feasible"] = report["feasible"]
            report["feasible"] = report["feasible"] and evaluation_residual <= settings.equality_tolerance
        report.update({"seed": seed, "training_examples": examples,
                       "epochs": training.epochs, "manifest_sha256": manifest_sha,
                       "noise_rule": "fixed_per_minibatch_common_random_numbers",
                       "utility_column_order": list(dataset.schema.modality_dims),
                       "reported_paper_results_reproduced": False})
        feasible = bool(report["feasible"])
        any_infeasible |= not feasible
        checkpoint_name = "checkpoint.pt" if feasible else "candidate_infeasible.pt"
        torch.save({"format_version": 1, "model_state": model.state_dict(),
                    "model_kwargs": kwargs, "config": effective_config,
                    "seed": seed, "epoch": training.epochs,
                    "manifest_sha256": manifest_sha, "torch_version": str(torch.__version__),
                    "bargaining_solution": args.solution,
                    "bargaining_feasible": feasible,
                    "optimization_objective": (model_objective if args.solution in {"nash", "uts"}
                                               else f"externally_constrained_{args.solution}"),
                    "noise_protocol": "fixed_per_minibatch_common_random_numbers",
                    "batch_order": "fixed_manifest_order",
                    "bargaining_settings": asdict(settings),
                    "dual_variables": duals}, run_dir / checkpoint_name)
        write_json(run_dir / "result.json", report)
        summaries.append(report)
    summary = {"solution": args.solution, "runs": summaries,
               "all_candidates_feasible": not any_infeasible}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if any_infeasible and not args.allow_infeasible:
        raise SystemExit("Equality tolerance is not met; failed candidates are explicitly marked and are not converged EGS/KSS results.")


if __name__ == "__main__":
    main()
