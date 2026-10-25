from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            records.append(value)
    return records


def _validated_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    if not records:
        raise ValueError("At least one recorded prediction is required")
    ids = []
    for record in records:
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Every prediction needs a nonempty string id")
        ids.append(sample_id)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction ids are not permitted")
    return ids


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict[str, float | None]:

    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be nonempty and equally sized")
    if any(isinstance(y, bool) or not isinstance(y, (int, np.integer)) or y not in (0, 1) for y in labels):
        raise ValueError("MCV labels must be integer 0 or 1")
    threshold = _finite_number(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Probability threshold must lie in [0, 1]")
    values = np.asarray([_finite_number(s, "score") for s in scores], dtype=np.float64)
    if bool(np.any((values < 0) | (values > 1))):
        raise ValueError("MCV scores must be probabilities in [0, 1]")
    truth = np.asarray(labels, dtype=np.int64)
    prediction = (values >= threshold).astype(np.int64)
    per_class = []
    for label in (0, 1):
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append((precision, recall, f1))
    macro = np.mean(per_class, axis=0)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    positives, negatives = int(truth.sum()), int((truth == 0).sum())
    auc = None
    if positives and negatives:


        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        auc = float((ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    return {"precision_macro": float(macro[0]), "recall_macro": float(macro[1]),
            "f1_macro": float(macro[2]), "accuracy": float(np.mean(prediction == truth)),
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0, "auc": auc}


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]], *, task: str, dataset: str,
    seed: int, experiment_id: str, answerer: str | None = None,
    judge_protocol: str | None = None, score_max: float | None = None,
    threshold: float = 0.5, expected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:

    ids = _validated_ids(records)
    if expected_ids is not None and set(ids) != set(expected_ids):
        raise ValueError("Prediction ids do not exactly match the evaluation set")
    if not dataset or not experiment_id:
        raise ValueError("Dataset and experiment_id are required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    if task == "mqa":
        if not answerer or not judge_protocol:
            raise ValueError("MQA requires explicit answerer and judge_protocol identifiers")
        maximum = _finite_number(score_max, "score_max")
        if maximum <= 0:
            raise ValueError("score_max must be positive")
        scores = [_finite_number(r.get("judge_score"), "judge_score") for r in records]
        if any(s < 0 or s > maximum for s in scores):
            raise ValueError("Judge scores must lie between zero and score_max")
        metrics: dict[str, float | None] = {"mj": 100.0 * float(np.mean(scores)) / maximum}
        protocol = {"judge": judge_protocol, "score_max": maximum,
                    "aggregation": "100 * mean(judge_score) / score_max"}
    elif task == "mcv":
        metrics = binary_metrics([r.get("label") for r in records], [r.get("score") for r in records], threshold)
        protocol = {"threshold": threshold, "averaging": "macro over labels [0, 1]", "undefined_class_metric": 0.0}
    else:
        raise ValueError("task must be 'mqa' or 'mcv'")
    id_digest = hashlib.sha256(json.dumps(sorted(ids), ensure_ascii=False).encode()).hexdigest()
    return {"dataset": dataset, "task": task, "answerer": answerer, "seed": seed,
            "experiment_id": experiment_id, "protocol": protocol, "n_examples": len(ids),
            "example_ids_sha256": id_digest, "metrics": metrics}


def summarize_runs(runs: Sequence[Mapping[str, Any]], expected_runs: int = 5) -> dict[str, Any]:

    if expected_runs < 2 or len(runs) != expected_runs:
        raise ValueError(f"Exactly {expected_runs} independent runs are required")
    signature_keys = ("dataset", "task", "answerer", "experiment_id", "protocol", "n_examples", "example_ids_sha256")
    first = runs[0]
    for key in signature_keys:
        if key not in first:
            raise ValueError(f"Missing run metadata: {key}")
    seeds = []
    keys = set(first.get("metrics", {}))
    if not keys:
        raise ValueError("Run metrics are missing")
    for run in runs:
        if any(run.get(key) != first[key] for key in signature_keys):
            raise ValueError("Runs must use matching dataset, examples, model/config and evaluation protocol")
        if set(run.get("metrics", {})) != keys:
            raise ValueError("Run metric names do not match")
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Each run requires its actual integer seed")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Independent runs must have distinct seeds")
    summary = {}
    for key in sorted(keys):
        values = [r["metrics"][key] for r in runs]
        if any(value is None for value in values):
            summary[key] = {"mean": None, "std": None, "runs": values}
            continue
        numeric = [_finite_number(value, key) for value in values]
        summary[key] = {"mean": float(np.mean(numeric)), "std": float(np.std(numeric, ddof=1)), "runs": numeric}
    return {**{key: first[key] for key in signature_keys}, "seeds": seeds,
            "num_runs": len(runs), "std_ddof": 1, "metrics": summary}


def load_adapter(spec: str, options: Mapping[str, Any] | None = None) -> Callable[..., Any]:

    if ":" not in spec:
        raise ValueError("Adapter must use module:factory syntax")
    module_name, factory_name = spec.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    adapter = factory(dict(options or {}))
    if not callable(adapter):
        raise TypeError("Adapter factory must return a callable")
    if hasattr(adapter, "requires_grad_") and hasattr(adapter, "eval"):
        adapter.requires_grad_(False)
        adapter.eval()
    return adapter


def predictor_example(record: Mapping[str, Any]) -> dict[str, Any]:

    allowed = {"id", "query", "query_features", "modalities", "evidence_text"}
    result = {key: value for key, value in record.items() if key in allowed}
    for key in ("id", "query", "query_features"):
        if key in result and not isinstance(result[key], str):
            raise ValueError(f"Predictor {key} must be a string")
    if "modalities" in result and (not isinstance(result["modalities"], dict) or
            any(not isinstance(key, str) or not isinstance(value, str)
                for key, value in result["modalities"].items())):
        raise ValueError("Predictor modalities must map names to cached feature paths")
    if "evidence_text" in result:
        text = result["evidence_text"]
        def plain_text(value: Any) -> bool:
            return isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        if not (plain_text(text) or (isinstance(text, dict) and
                all(isinstance(key, str) and plain_text(value) for key, value in text.items()))):
            raise ValueError("evidence_text must contain only strings, not annotation objects")
    if "raw_modalities" in record:
        raw = record["raw_modalities"]
        if not isinstance(raw, dict):
            raise ValueError("raw_modalities must map modality names to source lists")
        sanitized = {}
        for modality, sources in raw.items():
            if not isinstance(sources, list):
                raise ValueError("Each raw modality must contain a list of sources")
            sanitized[modality] = []
            for source in sources:
                if isinstance(source, str):
                    sanitized[modality].append(source)
                elif isinstance(source, dict) and isinstance(source.get("path"), str):
                    if "sampling_rate" in source and (isinstance(source["sampling_rate"], bool) or
                            not isinstance(source["sampling_rate"], int) or source["sampling_rate"] <= 0):
                        raise ValueError("Native source sampling_rate must be a positive integer")
                    sanitized[modality].append({key: source[key] for key in ("path", "sampling_rate") if key in source})
                else:
                    raise ValueError("Raw sources must be text/path strings or path/sampling_rate objects")
        result["raw_modalities"] = sanitized
    if "candidates" in record:
        candidates = record["candidates"]
        if not isinstance(candidates, list) or any(not isinstance(value, str) for value in candidates):
            raise ValueError("Predictor candidate choices must be plain strings without annotations")
        result["candidates"] = list(candidates)
    return result
