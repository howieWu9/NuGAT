import math

import pytest
import torch

from nugat.qat import QueryConditionedAttentionTransport, balanced_transport


def example(variant="qat"):
    torch.manual_seed(71)
    model = QueryConditionedAttentionTransport({"text": 5, "audio": 3}, 6, evidence_dim=4, num_evidence_units=2, temperature=0.8, variant=variant)
    features = {"text": torch.randn(2, 4, 5), "audio": torch.randn(2, 3, 3)}
    query = torch.randn(2, 6)
    previous = {m: torch.randn(2, 2, 4) for m in features}
    return model, features, query, previous


def test_qat_column_normalization_without_fixed_source_mass():
    model, features, query, previous = example()
    output = model(features, query, previous)
    for name, plan in output.transport.items():
        torch.testing.assert_close(plan.sum(dim=1), torch.ones(2, 2))
        assert not torch.allclose(plan.sum(dim=2), torch.full_like(plan.sum(dim=2), 2 / plan.shape[1]), atol=1e-4)
        assert output.grounded[name].shape == (2, 2, 4)


def test_no_prior_is_exact_dot_product_cross_attention():
    model, features, query, previous = example("without_prior")
    output = model(features, query, previous)
    for name, x in features.items():
        keys = model.key_projections[name](x)
        eq = model.evidence_projection(previous[name])
        expected = torch.softmax(torch.bmm(keys, eq.transpose(1, 2)) / math.sqrt(4) / 0.8, dim=1)
        torch.testing.assert_close(output.transport[name], expected)
        torch.testing.assert_close(output.grounded[name], expected.transpose(1, 2) @ model.value_projections[name](x))


def test_compatibility_block_affine_matches_explicit_concat():
    model, features, query, previous = example()
    projected_query = model.query_projection(query)
    keys = model.key_projections["text"](features["text"])
    y = previous["text"]
    concatenated = torch.cat([projected_query[:, None, None, :].expand(-1, 4, 2, -1), keys[:, :, None, :].expand(-1, -1, 2, -1), y[:, None, :, :].expand(-1, 4, -1, -1)], dim=-1)
    expected = model.compatibility_mlp(concatenated).squeeze(-1)
    torch.testing.assert_close(model._compatibility_scores(projected_query, keys, y), expected)


def test_prior_depends_on_query_and_receives_gradients_from_both_modalities():
    model, features, query, previous = example()
    output = model(features, query, previous)
    changed = model(features, query + torch.tensor([1., -2., 0., 3., 1., -1.]), previous)
    assert any(not torch.allclose(output.log_prior[m], changed.log_prior[m]) for m in features)
    for name in features:
        model.zero_grad(set_to_none=True)
        loss = model(features, query, previous).grounded[name].square().mean()
        loss.backward()
        for parameter in model.compatibility_mlp.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert model.compatibility_mlp[0].weight.grad.abs().sum() > 0
        assert model.query_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["qat", "without_prior", "eot", "cot"])
def test_padded_samples_have_zero_mass_and_no_gradient(variant):
    model, features, query, previous = example(variant)
    valid = torch.tensor([[True, True, False, False], [True, False, True, False]])
    text = features["text"].detach().clone().requires_grad_(True)
    features["text"] = text
    output = model(features, query, previous, {"text": valid})
    assert torch.count_nonzero(output.transport["text"].masked_select(~valid.unsqueeze(-1))) == 0
    output.grounded["text"].square().sum().backward()
    assert torch.count_nonzero(text.grad.masked_select(~valid.unsqueeze(-1))) == 0
    assert torch.isfinite(text.grad).all()


def test_nan_padding_is_ignored_but_all_invalid_sources_raise():
    model, features, query, previous = example()
    valid = torch.ones(2, 4, dtype=torch.bool)
    valid[:, -1] = False
    features["text"][:, -1] = float("nan")
    output = model(features, query, previous, {"text": valid})
    assert torch.isfinite(output.grounded["text"]).all()
    valid[0] = False
    with pytest.raises(ValueError, match="at least one valid"):
        model(features, query, previous, {"text": valid})


def test_balanced_ot_enforces_uniform_source_and_target_marginals():
    log_kernel = torch.tensor([[[1., -1.], [0.5, 0.2], [-2., 1.], [9., 9.]]], dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    plan = balanced_transport(log_kernel, valid, iterations=200, tolerance=1e-10)
    torch.testing.assert_close(plan.sum(1), torch.ones(1, 2, dtype=torch.float64))
    torch.testing.assert_close(plan.sum(2), torch.tensor([[2 / 3, 2 / 3, 2 / 3, 0.]], dtype=torch.float64))
    plan.square().sum().backward()
    assert torch.isfinite(log_kernel.grad).all()


def test_balanced_variants_use_specified_costs():
    for variant in ("eot", "cot"):
        model, features, query, previous = example(variant)
        output = model(features, query, previous)
        keys = model.key_projections["text"](features["text"])
        targets = model.evidence_projection(previous["text"])
        log_kernel = -(keys[:, :, None, :] - targets[:, None, :, :]).square().sum(-1) / 0.8 if variant == "eot" else keys @ targets.transpose(1, 2) / math.sqrt(4) / 0.8
        expected = balanced_transport(log_kernel, torch.ones(2, 4, dtype=torch.bool), 1000)
        torch.testing.assert_close(output.transport["text"], expected)


def test_underconverged_balanced_ot_raises_instead_of_returning_unbalanced_plan():
    log_kernel = torch.tensor([[[8., 8.], [8., 8.], [-8., 8.]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="does not converge"):
        balanced_transport(log_kernel, valid, iterations=1)
    plan = balanced_transport(log_kernel, valid, iterations=1000)
    torch.testing.assert_close(plan.sum(1), torch.ones(1, 2), atol=1e-3, rtol=0)
    torch.testing.assert_close(plan.sum(2), torch.full((1, 3), 2 / 3), atol=1e-3, rtol=0)


def test_balanced_ot_is_invariant_to_large_row_and_column_cost_offsets():
    kernel = torch.tensor([[[1., -1.], [0.5, 0.2], [-2., 1.]]], dtype=torch.float64)
    shifted = kernel + torch.tensor([[[1000.], [-1000.], [500.]]], dtype=torch.float64) + torch.tensor([[[800., -700.]]], dtype=torch.float64)
    valid = torch.ones(1, 3, dtype=torch.bool)
    expected = balanced_transport(kernel, valid, iterations=1000, tolerance=1e-10)
    actual = balanced_transport(shifted, valid, iterations=1000, tolerance=1e-10)
    torch.testing.assert_close(actual, expected)
