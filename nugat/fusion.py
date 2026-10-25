from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class FusionOutput:
    fused: Tensor
    retained: dict[str, Tensor]
    contrastive: Tensor
    utilities: Tensor
    next_retention: Tensor
    gates: dict[str, Tensor]
    loss: Tensor
    log_utilities: Tensor
    fusion_weights: Tensor


def utility_objective(
    log_utilities: Tensor,
    objective: str = "nash",
    epsilon: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:

    if log_utilities.ndim != 2 or log_utilities.shape[1] == 0:
        raise ValueError("log_utilities must have shape [batch, modalities].")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if objective == "nash":
        log_epsilon = log_utilities.new_tensor(math.log(epsilon))
        per_example = -torch.logaddexp(log_utilities, log_epsilon).sum(dim=-1)
    elif objective == "utilitarian":
        per_example = -log_utilities.exp().sum(dim=-1)
    else:
        raise ValueError(f"Unsupported utility objective: {objective!r}")
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def _masked_svd(values: Tensor, gates: Tensor) -> Tensor:

    left, singular, right_t = torch.linalg.svd(values, full_matrices=False)
    if not torch.is_grad_enabled() or not (values.requires_grad or gates.requires_grad):
        return (left * (singular * gates).unsqueeze(-2)) @ right_t

    scale = singular.detach().amax(dim=-1).clamp_min(torch.finfo(values.dtype).tiny)
    tolerance = 10 * torch.finfo(values.dtype).eps * scale
    degenerate = singular.detach().amin(dim=-1) <= tolerance
    if singular.shape[-1] > 1:
        gaps = singular.detach()[..., :-1] - singular.detach()[..., 1:]
        degenerate = degenerate | (gaps <= tolerance.unsqueeze(-1)).any(dim=-1)
    if not bool(degenerate.any()):
        return (left * (singular * gates).unsqueeze(-2)) @ right_t


    result = torch.empty_like(values)
    valid = ~degenerate
    if bool(valid.any()):
        left_valid, singular_valid, right_valid = torch.linalg.svd(
            values[valid], full_matrices=False
        )
        result[valid] = (
            left_valid * (singular_valid * gates[valid]).unsqueeze(-2)
        ) @ right_valid
    basis_t = right_t[degenerate].detach()
    coordinates = values[degenerate] @ basis_t.transpose(-2, -1)
    result[degenerate] = (coordinates * gates[degenerate].unsqueeze(-2)) @ basis_t
    return result


class NashUtilityFusion(nn.Module):


    def __init__(
        self,
        modalities: list[str],
        num_evidence_units: int = 16,
        evidence_dim: int = 256,
        iterations: int = 3,
        utility_temperature: float = 0.15,
        residual_weight: float = 0.20,
        gate_temperature: float = 0.05,
        epsilon: float = 1e-8,
        variant: str = "nuf",
        objective: str = "nash",
        subspace_rank: int | None = None,
        stochastic_eval: bool = False,
    ) -> None:
        super().__init__()
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError("modalities must be a nonempty list of unique names.")
        if any(not isinstance(m, str) or not m for m in modalities):
            raise ValueError("Each modality must have a nonempty string name.")
        if min(num_evidence_units, evidence_dim, iterations) < 1:
            raise ValueError("Evidence dimensions and iterations must be positive.")
        for name, value in (
            ("utility_temperature", utility_temperature),
            ("gate_temperature", gate_temperature),
            ("epsilon", epsilon),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= residual_weight <= 1:
            raise ValueError("residual_weight must lie in [0, 1].")
        variant = {"uwf": "uniform", "swf": "subspace_weighted"}.get(variant, variant)
        if variant not in {"nuf", "uniform", "subspace_weighted", "unweighted_subspace"}:
            raise ValueError(f"Unsupported fusion variant: {variant!r}")
        if objective not in {"nash", "utilitarian"}:
            raise ValueError(f"Unsupported utility objective: {objective!r}")
        self.modalities = tuple(modalities)
        self.num_evidence_units = num_evidence_units
        self.evidence_dim = evidence_dim
        self.iterations = iterations
        self.rank = min(num_evidence_units, evidence_dim)
        self.utility_temperature = float(utility_temperature)
        self.residual_weight = float(residual_weight)
        self.gate_temperature = float(gate_temperature)
        self.epsilon = float(epsilon)
        self.variant = variant
        self.objective = objective
        self.subspace_rank = subspace_rank
        self.stochastic_eval = stochastic_eval
        if variant == "subspace_weighted" and (
            subspace_rank is None or not 1 <= subspace_rank <= self.rank
        ):
            raise ValueError(f"SWF requires an explicit subspace_rank in [1, {self.rank}].")


        beta_initial = min(1.0 / len(self.modalities), 0.5)
        beta_logit = math.log(beta_initial / (1 - beta_initial))
        self.threshold_logits = nn.Parameter(
            torch.full((iterations, len(self.modalities), self.rank), beta_logit)
        )

    @property
    def thresholds(self) -> Tensor:

        return self.threshold_logits.sigmoid()

    def _gate(
        self,
        alpha: Tensor,
        beta: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        mean = alpha.unsqueeze(-1) - beta.unsqueeze(0)
        if self.training or self.stochastic_eval:
            noise = torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
            )
            latent = mean + self.gate_temperature * noise
        else:
            latent = mean
        hard = (latent >= 0).to(mean.dtype)
        if not self.training:
            return hard
        probability = 0.5 * (
            1 + torch.erf(mean / (self.gate_temperature * math.sqrt(2.0)))
        )
        return hard + (probability - probability.detach())

    def forward(
        self,
        grounded: Mapping[str, Tensor],
        previous_shared: Tensor,
        retention: Tensor,
        iteration: int,
        generator: torch.Generator | None = None,
    ) -> FusionOutput:
        if set(grounded) != set(self.modalities):
            raise ValueError("grounded must contain exactly the configured modalities.")
        if not 0 <= iteration < self.iterations:
            raise ValueError("iteration is a zero-based index within the configured iterations.")
        if previous_shared.ndim != 3:
            raise ValueError("previous_shared must have shape [batch, evidence units, dim].")
        batch = previous_shared.shape[0]
        shape = (batch, self.num_evidence_units, self.evidence_dim)
        if tuple(previous_shared.shape) != shape or batch < 1:
            raise ValueError(f"previous_shared must have nonempty shape {shape}.")
        if tuple(retention.shape) != (batch, len(self.modalities)):
            raise ValueError("retention must have shape [batch, modalities].")
        if not torch.is_floating_point(previous_shared):
            raise TypeError("Evidence representations must be floating-point tensors.")
        for name, values in grounded.items():
            if tuple(values.shape) != shape:
                raise ValueError(f"grounded[{name!r}] must have shape {shape}.")
            if values.device != previous_shared.device or values.dtype != previous_shared.dtype:
                raise ValueError("All evidence tensors must have the same device and dtype.")
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"grounded[{name!r}] contains a nonfinite value.")
        if retention.device != previous_shared.device:
            raise ValueError("retention and evidence must be on the same device.")
        if not bool(torch.isfinite(previous_shared).all() and torch.isfinite(retention).all()):
            raise ValueError("previous_shared and retention must be finite.")
        if bool((retention < 0).any()) or not torch.allclose(
            retention.sum(dim=-1), torch.ones_like(retention[:, 0]), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("retention must be nonnegative and sum to one per example.")


        computation_dtype = (
            torch.float32 if previous_shared.dtype in {torch.float16, torch.bfloat16}
            else previous_shared.dtype
        )
        grounded_work = {m: grounded[m].to(computation_dtype) for m in self.modalities}
        previous_work = previous_shared.to(computation_dtype)
        alpha = retention.to(computation_dtype)
        uniform_weights = alpha.new_full(alpha.shape, 1.0 / len(self.modalities))
        retained: dict[str, Tensor] = {}
        gates: dict[str, Tensor] = {}
        fusion_weights = uniform_weights

        if self.variant in {"nuf", "unweighted_subspace"}:
            if self.variant == "unweighted_subspace":
                alpha = uniform_weights
            beta = self.thresholds[iteration].to(computation_dtype)
            for index, modality in enumerate(self.modalities):
                gates[modality] = self._gate(alpha[:, index], beta[index], generator)
                retained[modality] = _masked_svd(grounded_work[modality], gates[modality])
        else:
            retained = dict(grounded_work)
            gates = {
                m: previous_work.new_ones((batch, self.rank)) for m in self.modalities
            }
            if self.variant == "subspace_weighted":
                reference = torch.stack(list(grounded_work.values()), dim=1).mean(dim=1)
                errors = []
                for modality in self.modalities:


                    values = grounded_work[modality]
                    _, singular, right_t = torch.linalg.svd(values, full_matrices=False)
                    scale = singular.detach().amax(dim=-1).clamp_min(torch.finfo(values.dtype).tiny)
                    tolerance = 10 * torch.finfo(values.dtype).eps * scale
                    gaps = singular.detach()[..., :-1] - singular.detach()[..., 1:]
                    bad = (singular.detach().amin(dim=-1) <= tolerance) | (
                        gaps <= tolerance.unsqueeze(-1)
                    ).any(dim=-1)


                    basis_t = right_t.detach().clone()
                    if bool((~bad).any()) and torch.is_grad_enabled():
                        _, _, differentiable_basis = torch.linalg.svd(values[~bad], full_matrices=False)
                        basis_t[~bad] = differentiable_basis
                    basis_t = basis_t[:, : self.subspace_rank, :]
                    reconstruction = (reference @ basis_t.transpose(-2, -1)) @ basis_t
                    errors.append(torch.linalg.vector_norm(reference - reconstruction, dim=(-2, -1)))
                fusion_weights = F.softmax(-torch.stack(errors, dim=-1), dim=-1)

        stacked_retained = torch.stack([retained[m] for m in self.modalities], dim=1)
        aggregate = (stacked_retained * fusion_weights[:, :, None, None]).sum(dim=1)
        if self.variant in {"uniform", "subspace_weighted"}:
            fused = aggregate
        else:
            fused = (1 - self.residual_weight) * aggregate + self.residual_weight * previous_work


        stacked_grounded = torch.stack([grounded_work[m] for m in self.modalities], dim=1)
        similarities = F.cosine_similarity(stacked_grounded, fused[:, None, :, :], dim=-1)
        contrastive = -F.log_softmax(similarities, dim=1).mean(dim=-1)
        log_utilities = -contrastive / self.utility_temperature
        utilities = log_utilities.exp()
        next_retention = F.softmax(log_utilities, dim=-1)
        loss = utility_objective(log_utilities, self.objective, self.epsilon)
        return FusionOutput(
            fused=fused,
            retained=retained,
            contrastive=contrastive,
            utilities=utilities,
            next_retention=next_retention,
            gates=gates,
            loss=loss,
            log_utilities=log_utilities,
            fusion_weights=fusion_weights,
        )
