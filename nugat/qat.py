"""Query-conditioned attention transport, corresponding to Eqs. (3)--(8).

The source axis is dimension 1 throughout. QAT normalizes only that axis;
it does not prescribe how much total mass a source sample must receive.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class QATOutput:
    grounded: dict[str, Tensor]
    transport: dict[str, Tensor]
    attention: dict[str, Tensor]
    log_prior: dict[str, Tensor]


def balanced_transport(log_kernel: Tensor, valid: Tensor, iterations: int,
                       tolerance: float = 1e-3) -> Tensor:
    """Compute the EOT/COT ablation with uniform two-sided marginals.

    Sinkhorn operates on a unit-total plan with source marginal 1/s and
    target marginal 1/n. Its output is multiplied by n for barycentric
    aggregation, so every returned column sums to one and every valid row
    approaches n/s. This rescaling does not remove the source constraint.
    Residuals are checked every 20 iterations and at the supplied iteration
    budget. Nonconverged plans raise rather than silently violating marginals.
    Masked source positions have exactly zero mass.
    """
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Sinkhorn tolerance must be finite and positive")
    if log_kernel.ndim != 3 or valid.shape != log_kernel.shape[:2]:
        raise ValueError("Expected log_kernel [batch, samples, units] and mask [batch, samples]")
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Every example requires at least one valid source sample")
    if not bool(torch.isfinite(log_kernel.masked_select(valid.unsqueeze(-1))).all()):
        raise ValueError("Valid Sinkhorn log-kernel entries must be finite")
    # Float32 accumulation is necessary when a caller uses mixed precision.
    work_dtype = torch.float64 if log_kernel.dtype == torch.float64 else torch.float32
    log_kernel = log_kernel.to(work_dtype).masked_fill(~valid.unsqueeze(-1), -torch.inf)
    batch, samples, units = log_kernel.shape
    # Adding any row/column constant to the cost leaves balanced OT's optimum
    # unchanged because BOTH marginals are fixed. Remove such constants for
    # numerical stability without rescaling the cost or temperature.
    row_shift = log_kernel.amax(dim=2, keepdim=True).masked_fill(~valid.unsqueeze(-1), 0.0)
    log_kernel = log_kernel - row_shift
    log_kernel = log_kernel - log_kernel.amax(dim=1, keepdim=True)
    log_source = -valid.sum(dim=1).to(work_dtype).log()
    log_target = -torch.log(torch.tensor(float(units), device=log_kernel.device, dtype=work_dtype))
    log_v = log_kernel.new_zeros(batch, units)
    log_u = log_kernel.new_zeros(batch, samples)
    max_row_error = max_column_error = float("inf")
    for step in range(iterations):
        # Mask *before* logsumexp: an all-minus-infinity row has undefined
        # derivatives even if its result is subsequently masked away.
        row_inputs = (log_kernel + log_v.unsqueeze(1)).masked_fill(~valid.unsqueeze(-1), 0.0)
        row_sum = torch.logsumexp(row_inputs, dim=2).masked_fill(~valid, 0.0)
        log_u = (log_source.unsqueeze(1) - row_sum).masked_fill(~valid, -torch.inf)
        log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=1)
        if (step + 1) % 20 == 0 or step + 1 == iterations:
            with torch.no_grad():
                check_plan = torch.exp(log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(1)) * units
                target_row = units / valid.sum(dim=1).to(work_dtype)
                relative_row_error = (check_plan.sum(dim=2) - target_row.unsqueeze(1)).abs() / target_row.unsqueeze(1)
                max_row_error = float(relative_row_error.masked_fill(~valid, 0.0).amax())
                max_column_error = float((check_plan.sum(dim=1) - 1).abs().amax())
            if max_row_error <= tolerance and max_column_error <= tolerance:
                break
    if not isfinite(max_row_error) or not isfinite(max_column_error) or max_row_error > tolerance or max_column_error > tolerance:
        raise RuntimeError(
            f"Balanced OT does not converge within {iterations} Sinkhorn iterations "
            f"(relative row residual={max_row_error:.3g}, column residual={max_column_error:.3g}, "
            f"tolerance={tolerance:.3g}). Increase sinkhorn_iterations. Changing the "
            "transport temperature changes the OT problem and must be explicit."
        )
    plan = torch.exp(log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(1))
    return (plan * units).masked_fill(~valid.unsqueeze(-1), 0.0)


class QueryConditionedAttentionTransport(nn.Module):
    """QAT and its Euclidean-OT, cross-attention-OT, and no-prior ablations.

    All modalities and refinement iterations share the evidence projection,
    query projection, and compatibility MLP. Modality keys and values have
    separate projections. The MLP uses projected query, sample key, and
    previous evidence features in that order: 3d -> d -> 1, with GELU.
    Initial evidence parameters belong to the enclosing NuGAT model.
    """

    VARIANTS = frozenset({"qat", "without_prior", "eot", "cot"})

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        query_dim: int,
        evidence_dim: int = 256,
        num_evidence_units: int = 16,
        temperature: float = 0.10,
        variant: str = "qat",
        sinkhorn_iterations: int = 1000,
        compatibility_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if not modality_dims or any(not isinstance(v, int) or v < 1 for v in modality_dims.values()):
            raise ValueError("modality_dims must contain positive feature dimensions")
        if any(not isinstance(k, str) or not k or "." in k for k in modality_dims):
            raise ValueError("Modality names must be nonempty strings without periods")
        if query_dim < 1 or evidence_dim < 1 or num_evidence_units < 1:
            raise ValueError("Query/evidence dimensions and evidence unit count must be positive")
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("Transport temperature must be finite and positive")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown transport variant {variant!r}; use {sorted(self.VARIANTS)}")
        if sinkhorn_iterations < 1 or compatibility_chunk_size < 1:
            raise ValueError("Iteration and chunk sizes must be positive")
        self.modality_dims = dict(modality_dims)
        self.query_dim = query_dim
        self.evidence_dim = evidence_dim
        self.num_evidence_units = num_evidence_units
        self.temperature = float(temperature)
        self.variant = variant
        self.sinkhorn_iterations = sinkhorn_iterations
        self.compatibility_chunk_size = compatibility_chunk_size
        self.evidence_projection = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.key_projections = nn.ModuleDict({m: nn.Linear(dim, evidence_dim, bias=False) for m, dim in modality_dims.items()})
        self.value_projections = nn.ModuleDict({m: nn.Linear(dim, evidence_dim, bias=False) for m, dim in modality_dims.items()})
        self.query_projection = nn.Linear(query_dim, evidence_dim, bias=False)
        self.compatibility_mlp = nn.Sequential(nn.Linear(3 * evidence_dim, evidence_dim), nn.GELU(), nn.Linear(evidence_dim, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compatibility_scores(self, query: Tensor, keys: Tensor, previous: Tensor) -> Tensor:
        """Evaluate the concatenation MLP without materializing a 3d broadcast.

        Splitting the first affine map by its three input blocks is exactly
        equivalent to applying the MLP to cat(query, key, previous). Chunking
        the batch limits temporary memory for large training batches.
        """
        first = self.compatibility_mlp[0]
        last = self.compatibility_mlp[2]
        wq, wk, wy = first.weight.split(self.evidence_dim, dim=1)
        q_hidden = F.linear(query, wq, first.bias)
        k_hidden = F.linear(keys, wk)
        y_hidden = F.linear(previous, wy)
        chunks = []
        for start in range(0, query.shape[0], self.compatibility_chunk_size):
            stop = start + self.compatibility_chunk_size
            hidden = (q_hidden[start:stop, None, None, :] + k_hidden[start:stop, :, None, :] + y_hidden[start:stop, None, :, :])
            chunks.append(last(self.compatibility_mlp[1](hidden)).squeeze(-1))
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        features: Mapping[str, Tensor],
        query: Tensor,
        previous: Mapping[str, Tensor],
        masks: Mapping[str, Tensor] | None = None,
    ) -> QATOutput:
        expected = set(self.modality_dims)
        if set(features) != expected or set(previous) != expected:
            raise ValueError("Feature and previous-evidence keys must match the configured modalities")
        if masks is not None and set(masks) - expected:
            raise ValueError("Masks contain an unknown modality")
        if query.ndim != 2 or query.shape[1] != self.query_dim or query.shape[0] < 1:
            raise ValueError(f"query must have shape [batch, {self.query_dim}]")
        if not query.is_floating_point() or not bool(torch.isfinite(query).all()):
            raise ValueError("Query features must be finite floating-point values")
        projected_query = self.query_projection(query) if self.variant == "qat" else None
        grounded, transports, scores, priors = {}, {}, {}, {}
        for modality, original_dim in self.modality_dims.items():
            x, y = features[modality], previous[modality]
            if x.ndim != 3 or x.shape[0] != query.shape[0] or x.shape[1] < 1 or x.shape[2] != original_dim:
                raise ValueError(f"{modality}: features must have shape [batch, samples, {original_dim}]")
            if y.shape != (query.shape[0], self.num_evidence_units, self.evidence_dim):
                raise ValueError(f"{modality}: previous evidence has an incorrect shape")
            if x.device != query.device or y.device != query.device:
                raise ValueError("Query, features, and previous evidence must share a device")
            valid = masks.get(modality) if masks is not None else None
            if valid is None:
                valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            if valid.shape != x.shape[:2] or valid.dtype != torch.bool or valid.device != x.device:
                raise ValueError(f"{modality}: sample mask must be boolean [batch, samples] on the feature device")
            if not bool(valid.any(dim=1).all()):
                raise ValueError(f"{modality}: every example requires at least one valid sample")
            # Padded values are ignored even if a feature exporter uses NaN padding.
            x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
            if not x.is_floating_point() or not y.is_floating_point() or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
                raise ValueError(f"{modality}: valid samples and previous evidence must be finite floating-point values")
            keys = self.key_projections[modality](x)
            values = self.value_projections[modality](x)
            evidence_queries = self.evidence_projection(y)
            attention = torch.bmm(keys, evidence_queries.transpose(1, 2)) / sqrt(self.evidence_dim)
            if self.variant == "qat":
                compatibility = self._compatibility_scores(projected_query, keys, y)
                compatibility = compatibility.masked_fill(~valid.unsqueeze(-1), -torch.inf)
                log_prior = F.log_softmax(compatibility, dim=1)
                logits = attention / self.temperature + log_prior
                transport = F.softmax(logits, dim=1)
            elif self.variant == "without_prior":
                log_prior = torch.zeros_like(attention).masked_fill(~valid.unsqueeze(-1), -torch.inf)
                transport = F.softmax((attention / self.temperature).masked_fill(~valid.unsqueeze(-1), -torch.inf), dim=1)
            else:
                log_prior = torch.zeros_like(attention).masked_fill(~valid.unsqueeze(-1), -torch.inf)
                if self.variant == "eot":
                    # Squared Euclidean cost uses the projected key and evidence.
                    cost = (keys.square().sum(-1, keepdim=True) + evidence_queries.square().sum(-1).unsqueeze(1) - 2 * torch.bmm(keys, evidence_queries.transpose(1, 2))).clamp_min(0)
                    log_kernel = -cost / self.temperature
                else:
                    log_kernel = attention / self.temperature
                transport = balanced_transport(log_kernel, valid, self.sinkhorn_iterations).to(values.dtype)
            grounded[modality] = torch.bmm(transport.transpose(1, 2), values)
            transports[modality] = transport
            scores[modality] = attention
            priors[modality] = log_prior
        return QATOutput(grounded, transports, scores, priors)
