import math

import pytest
import torch
from torch.nn import functional as F

from nugat.fusion import NashUtilityFusion, utility_objective


def make_inputs(batch=2, units=3, dim=5, requires_grad=False):
    generator = torch.Generator().manual_seed(21)
    grounded = {
        m: torch.randn(batch, units, dim, generator=generator, dtype=torch.float64).requires_grad_(requires_grad)
        for m in ("audio", "video")
    }
    previous = torch.randn(batch, units, dim, generator=generator, dtype=torch.float64)
    retention = torch.full((batch, 2), 0.5, dtype=torch.float64)
    return grounded, previous, retention


def test_uniform_fusion_and_equations_14_through_17():
    grounded, previous, retention = make_inputs()
    model = NashUtilityFusion(["audio", "video"], 3, 5, variant="uniform").double()
    output = model(grounded, previous, retention, 0)
    expected_fused = (grounded["audio"] + grounded["video"]) / 2
    torch.testing.assert_close(output.fused, expected_fused)
    cosines = torch.stack([F.cosine_similarity(v, expected_fused, dim=-1) for v in grounded.values()], dim=1)
    probabilities = cosines.exp() / cosines.exp().sum(dim=1, keepdim=True)
    expected_j = -probabilities.log().mean(dim=-1)
    expected_u = torch.exp(-expected_j / 0.15)
    torch.testing.assert_close(output.contrastive, expected_j)
    torch.testing.assert_close(output.utilities, expected_u)
    torch.testing.assert_close(output.next_retention, expected_u / expected_u.sum(dim=-1, keepdim=True))
    torch.testing.assert_close(output.loss, -torch.log(expected_u + 1e-8).sum(dim=-1).mean())


def test_nash_objective_remains_finite_when_utilities_underflow():
    log_utilities = torch.tensor([[-1000.0, -1001.0]], requires_grad=True)
    loss = utility_objective(log_utilities)
    torch.testing.assert_close(loss, loss.new_tensor(-2 * math.log(1e-8)))
    loss.backward()
    assert torch.isfinite(log_utilities.grad).all()
    assert torch.isfinite(loss)


def test_utilitarian_is_negative_sum_of_utilities():
    log_utilities = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    torch.testing.assert_close(
        utility_objective(log_utilities, "utilitarian"),
        -log_utilities.exp().sum(dim=-1).mean(),
    )


@pytest.mark.parametrize("residual_weight", [0.0, 1.0])
def test_residual_endpoints(residual_weight):
    grounded, previous, retention = make_inputs()
    model = NashUtilityFusion(["audio", "video"], 3, 5, residual_weight=residual_weight).double().eval()
    with torch.no_grad():
        model.threshold_logits.fill_(-20)
    output = model(grounded, previous, retention, 0)
    expected = previous if residual_weight else (grounded["audio"] + grounded["video"]) / 2
    torch.testing.assert_close(output.fused, expected)


def test_hard_gaussian_masks_have_finite_training_gradients():
    grounded, previous, retention = make_inputs(requires_grad=True)
    model = NashUtilityFusion(["audio", "video"], 3, 5).double().train()
    output = model(grounded, previous, retention, 0, generator=torch.Generator().manual_seed(7))
    assert output.fused.shape == previous.shape
    for gate in output.gates.values():
        assert torch.all((gate == 0) | (gate == 1))
    output.loss.backward()
    for values in grounded.values():
        assert values.grad is not None and torch.isfinite(values.grad).all()
    assert model.threshold_logits.grad is not None
    assert torch.isfinite(model.threshold_logits.grad).all()
    assert model.threshold_logits.grad[0].abs().sum() > 0


def test_deterministic_eval_preserves_or_removes_whole_singular_directions():
    grounded, previous, retention = make_inputs()
    model = NashUtilityFusion(["audio", "video"], 3, 5).double().eval()
    with torch.no_grad():
        model.threshold_logits[0, 0].fill_(-20)
        model.threshold_logits[0, 1].fill_(20)
    one = model(grounded, previous, retention, 0)
    two = model(grounded, previous, retention, 0)
    torch.testing.assert_close(one.retained["audio"], grounded["audio"])
    torch.testing.assert_close(one.retained["video"], torch.zeros_like(grounded["video"]))
    torch.testing.assert_close(one.fused, two.fused)


def test_degenerate_spectra_have_finite_surrogate_gradients():
    repeated = torch.eye(3, 5, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    grounded = {"audio": repeated.clone().requires_grad_(), "video": torch.zeros_like(repeated).requires_grad_()}
    model = NashUtilityFusion(["audio", "video"], 3, 5).double().train()
    output = model(grounded, torch.ones_like(repeated), torch.full((2, 2), 0.5, dtype=torch.float64), 0)
    output.loss.backward()
    for values in grounded.values():
        assert torch.isfinite(values.grad).all()
    assert torch.isfinite(model.threshold_logits.grad).all()


def test_swf_requires_rank_and_uses_top_k_reconstruction_error_weights():
    with pytest.raises(ValueError, match="subspace_rank"):
        NashUtilityFusion(["audio", "video"], 3, 5, variant="swf")
    grounded, previous, retention = make_inputs()
    model = NashUtilityFusion(["audio", "video"], 3, 5, variant="swf", subspace_rank=1).double()
    output = model(grounded, previous, retention, 0)
    reference = (grounded["audio"] + grounded["video"]) / 2
    errors = []
    for values in grounded.values():
        _, _, right_t = torch.linalg.svd(values, full_matrices=False)
        basis = right_t[:, :1].transpose(-2, -1)
        errors.append(torch.linalg.vector_norm(reference - reference @ basis @ basis.transpose(-2, -1), dim=(-2, -1)))
    expected_weights = F.softmax(-torch.stack(errors, dim=-1), dim=-1)
    torch.testing.assert_close(output.fusion_weights, expected_weights)
    aggregate = sum(expected_weights[:, i, None, None] * grounded[m] for i, m in enumerate(grounded))
    torch.testing.assert_close(output.fused, aggregate)


def test_invalid_retention_and_iteration_fail_explicitly():
    grounded, previous, retention = make_inputs()
    model = NashUtilityFusion(["audio", "video"], 3, 5).double()
    with pytest.raises(ValueError, match="sum to one"):
        model(grounded, previous, retention * 2, 0)
    with pytest.raises(ValueError, match="zero-based"):
        model(grounded, previous, retention, 3)
