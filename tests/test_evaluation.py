import copy
import math

import pytest

from nugat.evaluation import binary_metrics, evaluate_predictions, load_adapter, predictor_example, summarize_runs


def mqa_run(seed=42, scores=(4.0, 2.0)):
    return evaluate_predictions(
        [{"id": "first", "judge_score": scores[0]}, {"id": "second", "judge_score": scores[1]}],
        task="mqa", dataset="test-fixture", seed=seed, experiment_id="fixed-config",
        answerer="actual-checkpoint-id", judge_protocol="checkpoint+prompt-hash", score_max=5.0,
    )


def test_mqa_explicit_scale_and_complete_set():
    result = mqa_run()
    assert result["metrics"]["mj"] == pytest.approx(60.0)
    assert result["n_examples"] == 2
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_predictions([{"id": "first", "judge_score": 2}], task="mqa", dataset="fixture",
                             seed=42, experiment_id="fixture", answerer="real", judge_protocol="real",
                             score_max=5, expected_ids=["first", "second"])


@pytest.mark.parametrize("records", [
    [{"id": "same", "judge_score": 1}, {"id": "same", "judge_score": 2}],
    [{"id": "one"}], [{"id": "one", "judge_score": float("nan")}],
    [{"id": "one", "judge_score": 6}], [{"id": "one", "judge_score": "4"}],
])
def test_mqa_rejects_incomplete_or_invalid_scores(records):
    with pytest.raises(ValueError):
        evaluate_predictions(records, task="mqa", dataset="fixture", seed=42,
                             experiment_id="fixed", answerer="real", judge_protocol="real", score_max=5)


def test_mqa_does_not_guess_judge_protocol_or_scale():
    for kwargs in ({"answerer": "real", "judge_protocol": "real"},
                   {"answerer": "real", "score_max": 5}):
        with pytest.raises(ValueError):
            evaluate_predictions([{"id": "a", "judge_score": 3}], task="mqa", dataset="fixture",
                                 seed=42, experiment_id="fixed", **kwargs)


def test_mcv_macro_metrics_and_mann_whitney_auc():
    result = binary_metrics([0, 0, 0, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["precision_macro"] == pytest.approx(0.75)
    assert result["recall_macro"] == pytest.approx(5 / 6)
    assert result["f1_macro"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["mcc"] == pytest.approx(1 / math.sqrt(3))
    assert result["auc"] == pytest.approx(1.0)


def test_mcv_auc_ties_and_single_class():
    assert binary_metrics([0, 1, 0, 1], [0.5] * 4)["auc"] == pytest.approx(0.5)
    assert binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])["auc"] == pytest.approx(0.75)
    assert binary_metrics([1, 1], [0.8, 0.9])["auc"] is None


@pytest.mark.parametrize("labels,scores", [([0, 2], [.1, .5]), ([0.0, 1.0], [.1, .8]),
                                           ([0, 1], [.1, float("inf")]), ([0, 1], [.1, 1.01])])
def test_mcv_rejects_nonbinary_labels_or_nonprobabilities(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)


def test_summary_is_sample_std_over_runs():
    runs = [mqa_run(42 + i, scores=(i + 1, i + 1)) for i in range(5)]
    result = summarize_runs(runs)
    assert result["metrics"]["mj"]["mean"] == pytest.approx(60)
    assert result["metrics"]["mj"]["std"] == pytest.approx(math.sqrt(1000))
    assert result["std_ddof"] == 1
    assert result["seeds"] == [42, 43, 44, 45, 46]


def test_summary_rejects_changed_protocol_examples_or_reused_seed():
    runs = [mqa_run(42 + i) for i in range(5)]
    for key, value in (("seed", 42), ("example_ids_sha256", "different"),
                       ("experiment_id", "different"), ("protocol", {"judge": "different"})):
        bad = copy.deepcopy(runs)
        bad[1][key] = value
        with pytest.raises(ValueError):
            summarize_runs(bad)
    with pytest.raises(ValueError, match="Exactly 5"):
        summarize_runs(runs[:4])


def test_answerer_does_not_receive_ground_truth():
    record = {"id": "x", "query": "What?", "answer": "secret", "label": 1,
              "reference_answer": "secret", "ground_truth": "secret", "correct_answer": "secret",
              "annotations": {"label": 1}, "evidence_text": ["actual observation"]}
    context = predictor_example(record)
    assert context == {"id": "x", "query": "What?", "evidence_text": ["actual observation"]}
    assert record["answer"] == "secret"


def test_predictor_raw_sources_exclude_annotations():
    example = predictor_example({"id": "x", "query": "What?", "raw_modalities": {
        "audio": [{"path": "actual.wav", "sampling_rate": 16000, "correct_answer": "secret"}]}})
    assert example["raw_modalities"]["audio"] == [{"path": "actual.wav", "sampling_rate": 16000}]
    with pytest.raises(ValueError, match="plain strings"):
        predictor_example({"candidates": [{"text": "one", "correct": True}]})
    with pytest.raises(ValueError, match="annotation objects"):
        predictor_example({"evidence_text": [{"text": "observation", "gold_label": 1}]})


def test_adapter_factory_freezes_actual_torch_module(monkeypatch):
    import sys
    import types
    import torch
    module = types.ModuleType("nugat_test_adapter")
    module.factory = lambda options: torch.nn.Linear(options["width"], 1)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = load_adapter("nugat_test_adapter:factory", {"width": 3})
    assert not adapter.training
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert adapter(torch.ones(2, 3)).shape == (2, 1)
