from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nugat.evaluation import evaluate_predictions, load_adapter, predictor_example, read_jsonl


def _json(path: str | None) -> dict:
    if not path:
        return {}
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Adapter options must be a JSON object")
    return result


def _score(args: argparse.Namespace) -> None:
    records = read_jsonl(args.predictions)
    expected_ids = None
    if args.expected_ids:
        expected_ids = [r["id"] for r in read_jsonl(args.expected_ids)]
    result = evaluate_predictions(
        records, task=args.task, dataset=args.dataset, seed=args.seed,
        experiment_id=args.experiment_id, answerer=args.answerer,
        judge_protocol=args.judge_protocol, score_max=args.score_max,
        threshold=args.threshold, expected_ids=expected_ids,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


def _infer(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from nugat.data import ManifestDataset, FeatureSchema, collate_evidence
    from nugat.training import load_checkpoint, seed_everything

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.judge and (not args.predictor or args.task != "mqa"):
        raise ValueError("A judge requires an MQA predictor")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    seed_everything(checkpoint["seed"])
    model.requires_grad_(False).eval()
    kwargs = checkpoint["model_kwargs"]
    schema = FeatureSchema(query_dim=kwargs["query_dim"], modality_dims=kwargs["modality_dims"])
    dataset = ManifestDataset(args.manifest, args.split, schema=schema)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_evidence)
    predictor_options, judge_options = _json(args.predictor_options), _json(args.judge_options)
    predictor = load_adapter(args.predictor, predictor_options) if args.predictor else None
    judge = load_adapter(args.judge, judge_options) if args.judge else None
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "grounding"
    artifacts.mkdir(exist_ok=True)


    predictions_path = output / "predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"Prediction output already exists: {predictions_path}")
    with predictions_path.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            result = model(batch.features, batch.query, masks=batch.masks)
            for i, sample_id in enumerate(batch.ids):
                example = dict(batch.metadata[i])
                arrays = {"fused": result.fused[i].detach().float().cpu().numpy(),
                          "retention": result.retention[i].detach().float().cpu().numpy(),
                          "next_retention": result.next_retention[i].detach().float().cpu().numpy()}
                transport, rankings = {}, {}
                for modality, plan in result.transport.items():
                    valid = batch.masks[modality][i]
                    matrix = plan[i][valid].detach().float().cpu().numpy()
                    transport[modality] = matrix
                    arrays[f"transport_{modality}"] = matrix


                    mass = matrix.mean(axis=1)
                    order = np.argsort(-mass, kind="stable")
                    rankings[modality] = [{"source_index": int(j), "mass": float(mass[j])} for j in order]
                file_name = hashlib.sha256(sample_id.encode()).hexdigest() + ".npz"
                np.savez_compressed(artifacts / file_name, **arrays)
                grounding = {"fused": arrays["fused"], "transport": transport,
                             "retention": arrays["retention"], "rankings": rankings,
                             "next_retention": arrays["next_retention"],
                             "modality_order": list(kwargs["modality_dims"]),
                             "manifest_directory": str(Path(args.manifest).resolve().parent)}
                record = {"id": sample_id, "split": args.split, "query": example["query"],
                          "grounding_file": str(Path("grounding") / file_name), "rankings": rankings}
                if predictor is not None:
                    prediction = predictor(example=predictor_example(example), grounding=grounding)
                    if args.task == "mqa":
                        if not isinstance(prediction, str):
                            raise TypeError("MQA predictor must return an answer string")
                        record["prediction"] = prediction
                        if "answer" in example:
                            record["answer"] = example["answer"]
                        if judge is not None:
                            if "answer" not in example:
                                raise ValueError("A reference answer is required for MQA judging")
                            value = judge(query=example["query"], reference=example["answer"], prediction=prediction)
                            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                                raise TypeError("Judge must return a finite numerical score")
                            record["judge_score"] = float(value)
                    else:
                        if isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not 0 <= prediction <= 1:
                            raise TypeError("MCV predictor must return a probability in [0, 1]")
                        if example.get("label") not in (0, 1):
                            raise ValueError("MCV records require binary label")
                        record.update(score=float(prediction), label=example["label"])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    provenance = {"checkpoint": str(Path(args.checkpoint).resolve()), "seed": checkpoint["seed"],
                  "manifest": str(Path(args.manifest).resolve()), "split": args.split,
                  "task": args.task, "predictor": args.predictor, "predictor_options": predictor_options,
                  "judge": args.judge, "judge_options": judge_options, "num_examples": len(dataset),
                  "model_kwargs": kwargs, "export_only": predictor is None}

    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(dataset)} examples to {predictions_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--predictions", required=True)
    score.add_argument("--task", choices=("mqa", "mcv"), required=True)
    score.add_argument("--dataset", required=True)
    score.add_argument("--seed", type=int, required=True)
    score.add_argument("--experiment-id", required=True)
    score.add_argument("--answerer")
    score.add_argument("--judge-protocol")
    score.add_argument("--score-max", type=float)
    score.add_argument("--threshold", type=float, default=0.5)
    score.add_argument("--expected-ids")
    score.add_argument("--output", required=True)
    score.set_defaults(function=_score)
    infer = sub.add_parser("infer")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--split", choices=("validation", "test"), default="test")
    infer.add_argument("--task", choices=("mqa", "mcv"), default="mqa")
    infer.add_argument("--batch-size", type=int, default=16)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--predictor")
    infer.add_argument("--predictor-options")
    infer.add_argument("--judge")
    infer.add_argument("--judge-options")
    infer.add_argument("--output", required=True)
    infer.set_defaults(function=_infer)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
