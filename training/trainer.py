from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fotnuf.config import ExperimentConfig
from fotnuf.data.features import FeatureStore
from fotnuf.data.mmkg import MMKGData, TripleDataset
from fotnuf.models.fotnuf import FoTNuFModel
from fotnuf.training.checkpoint import save_checkpoint
from fotnuf.training.metrics import evaluate_ranking
from fotnuf.training.negative_sampling import NegativeSampler
from fotnuf.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    last_epoch: int
    checkpoint_path: Path


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(config: ExperimentConfig, data: MMKGData, feature_store: FeatureStore) -> FoTNuFModel:
    model = FoTNuFModel(
        num_relations=data.num_relations,
        modality_dims={m: feature_store.dims[m] for m in config.modality_order},
        modality_order=config.modality_order,
        relation_dim=config.model.relation_dim,
        support_units=config.model.support_units,
        cost_hidden_dim=config.model.cost_hidden_dim,
        scorer_hidden_dim=config.model.scorer_hidden_dim,
        dropout=config.model.dropout,
        eta=config.model.eta,
        gamma=config.model.gamma,
        epsilon=config.model.epsilon,
        sinkhorn_iterations=config.model.sinkhorn_iterations,
        fusion_iterations=config.model.fusion_iterations,
    )
    return model


def kgc_loss(outputs: Dict[str, torch.Tensor], lambda_nash: float) -> torch.Tensor:
    pos = outputs["positive_scores"]
    neg = outputs["negative_scores"]
    pos_loss = F.softplus(-pos).mean()
    neg_loss = F.softplus(neg).mean()
    return pos_loss + neg_loss + lambda_nash * outputs["nash_loss"]


def l2_regularization(model: torch.nn.Module) -> torch.Tensor:
    total = next(model.parameters()).new_tensor(0.0)
    for parameter in model.parameters():
        if parameter.requires_grad:
            total = total + parameter.pow(2).sum()
    return total


def train(
    config: ExperimentConfig,
    data: MMKGData,
    feature_store: FeatureStore,
    limit_train: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> TrainResult:
    device = resolve_device(config.device)
    ensure_dir(config.run_dir)
    model = build_model(config, data, feature_store).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    train_dataset = TripleDataset(data.train, limit=limit_train)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size or config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    sampler = NegativeSampler(
        num_entities=data.num_entities,
        true_triples=data.true_triples,
        negative_samples=config.training.negative_samples,
        seed=config.seed,
    )
    max_epochs = int(epochs or config.training.max_epochs)
    best_metric = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = config.run_dir / "best.pt"
    latest_path = config.run_dir / "latest.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        iterator = loader
        if show_progress:
            iterator = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for positives in iterator:
            negatives = sampler.sample(positives)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(positives, negatives, feature_store)
            loss = kgc_loss(outputs, config.training.lambda_nash)
            if config.training.lambda_reg > 0:
                loss = loss + config.training.lambda_reg * l2_regularization(model)
            loss.backward()
            if config.training.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=sum(losses) / len(losses))
        train_loss = sum(losses) / max(len(losses), 1)
        metrics = {"loss": train_loss}
        if epoch % config.training.eval_every == 0:
            ranking = evaluate_ranking(
                model=model,
                data=data,
                feature_store=feature_store,
                triples=data.valid,
                config=config.evaluation,
                batch_size=batch_size or config.training.batch_size,
                seed=config.seed + epoch,
                show_progress=False,
            )
            metrics.update(ranking.as_dict())
            metric_value = metrics.get(config.training.early_stopping_metric, ranking.mrr)
            LOGGER.info("epoch=%d loss=%.6f mrr=%.6f", epoch, train_loss, ranking.mrr)
            if metric_value > best_metric:
                best_metric = float(metric_value)
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config.as_dict(),
                    metrics=metrics,
                )
            else:
                stale_epochs += 1
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config.as_dict(),
            metrics=metrics,
        )
        if stale_epochs >= config.training.early_stopping_patience:
            LOGGER.info("Early stopping at epoch %d", epoch)
            break
    return TrainResult(
        best_metric=best_metric if best_metric != float("-inf") else 0.0,
        best_epoch=best_epoch,
        last_epoch=epoch,
        checkpoint_path=checkpoint_path if checkpoint_path.exists() else latest_path,
    )

