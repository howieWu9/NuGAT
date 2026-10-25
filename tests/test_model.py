import torch

from nugat import NuGAT


def example():
    torch.manual_seed(7)
    model = NuGAT({"text": 7, "audio": 5}, query_dim=6,
                  num_evidence_units=3, evidence_dim=4, iterations=3)
    features = {"text": torch.randn(2, 5, 7), "audio": torch.randn(2, 4, 5)}
    query = torch.randn(2, 6)
    return model, features, query


def test_exact_three_updates_and_trainable_initial_evidence():
    model, features, query = example()
    out = model(features, query)
    assert out.fused.shape == (2, 3, 4)
    assert len(out.history) == 3
    out.loss.backward()
    for parameter in model.initial_evidence.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for parameter in model.qat.compatibility_mlp.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_each_iteration_uses_previous_grounded_state_and_utility_feedback():
    model, features, query = example()
    model.eval()
    out = model(features, query)
    expected = model.qat(features, query, out.history[0].grounding.grounded)
    for name in features:
        torch.testing.assert_close(out.history[1].grounding.transport[name], expected.transport[name])
    expected_fusion = model.fusion(expected.grounded, out.history[0].fusion.fused,
                                 out.history[0].fusion.next_retention, 1)
    torch.testing.assert_close(out.history[1].fusion.fused, expected_fusion.fused)
    torch.testing.assert_close(out.retention, out.history[-2].fusion.next_retention)
    torch.testing.assert_close(out.next_retention, out.history[-1].fusion.next_retention)


def test_eval_reproducible_and_checkpoint_roundtrip(tmp_path):
    model, features, query = example()
    model.eval()
    first = model(features, query)
    torch.testing.assert_close(first.fused, model(features, query).fused)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    replacement, _, _ = example()
    replacement.load_state_dict(torch.load(path, weights_only=True))
    replacement.eval()
    torch.testing.assert_close(first.fused, replacement(features, query).fused)


def test_frozen_input_features_are_not_changed():
    model, features, query = example()
    original = {name: value.clone() for name, value in features.items()}
    model(features, query).loss.backward()
    for name, value in features.items():
        torch.testing.assert_close(value, original[name])
        assert value.grad is None
