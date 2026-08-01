from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mel_net.constants import MANIPULATION_OPERATIONS


DEFAULT_DIRECTION_LOSS_WEIGHTS = {
    "image_swap": 1.15,
    "entity_shift": 1.45,
    "relation_misbind": 1.0,
    "context_drop": 1.0,
    "text_overclaim": 1.2,
}


def direction_separation_loss(prototypes: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    if prototypes.size(0) <= 1:
        return prototypes.sum() * 0.0
    dirs = F.normalize(prototypes, dim=-1)
    sim = dirs @ dirs.t()
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    return torch.relu(off_diag - float(margin)).mean()


def sparse_activation_loss(activation: torch.Tensor) -> torch.Tensor:
    if activation.size(-1) <= 1:
        return activation.sum() * 0.0
    probs = activation.float().clamp(min=1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    return (entropy / torch.log(torch.tensor(float(activation.size(-1)), device=activation.device))).mean()


def contrastive_invariance_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    target = batch["edit_target"].float()
    active = target.sum(dim=-1) > 0
    if active.any():
        return F.smooth_l1_loss(outputs["edit_state"][active], outputs["state"][active])
    return outputs["state"].sum() * 0.0


def manipulation_margin_alignment_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    positive_margin: float = 0.35,
    negative_margin: float = 0.05,
) -> torch.Tensor:
    target = target.float()
    positive = target > 0
    negative = ~positive
    losses = []
    if positive.any():
        losses.append(torch.relu(float(positive_margin) - scores[positive]).mean())
    if negative.any():
        losses.append(torch.relu(scores[negative] - float(negative_margin)).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def _direction_weight_tensor(
    device: torch.device,
    dtype: torch.dtype,
    direction_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    if direction_weights is None:
        values = [DEFAULT_DIRECTION_LOSS_WEIGHTS.get(op, 1.0) for op in MANIPULATION_OPERATIONS]
        return torch.tensor(values, device=device, dtype=dtype)
    if isinstance(direction_weights, torch.Tensor):
        return direction_weights.to(device=device, dtype=dtype)
    return torch.tensor([float(item) for item in direction_weights], device=device, dtype=dtype)


def balanced_multilabel_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    direction_weights: Sequence[float] | torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    target = target.float()
    if target.numel() == 0:
        return logits.sum() * 0.0
    positive = target.sum(dim=0)
    negative = target.size(0) - positive
    pos_weight = (negative / positive.clamp(min=1.0)).clamp(min=1.0, max=8.0)
    raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    if float(focal_gamma) > 0.0:
        prob = torch.sigmoid(logits.detach())
        pt = torch.where(target > 0, prob, 1.0 - prob)
        raw = raw * torch.pow((1.0 - pt).clamp(min=1e-6), float(focal_gamma))
    weights = _direction_weight_tensor(logits.device, logits.dtype, direction_weights).view(1, -1)
    return (raw * weights).mean()


def prototype_alignment_bce_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    scale: float = 8.0,
    direction_weights: Sequence[float] | torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    target = target.float()
    return balanced_multilabel_bce_loss(
        scores * float(scale),
        target,
        direction_weights=direction_weights,
        focal_gamma=focal_gamma,
    )


def uncertainty_direction_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    activation = outputs.get("edit_direction_activation", outputs["direction_activation"])
    uncertainty = outputs.get("edit_eq_uncertainty", outputs["eq_uncertainty"])
    if activation.size(-1) <= 1:
        return activation.sum() * 0.0
    if batch is not None and "edit_target" in batch:
        target = batch["edit_target"].float()
        active = target.sum(dim=-1) > 0
        if active.any():
            multi_direction = (target[active].sum(dim=-1) > 1).float()
            return F.smooth_l1_loss(uncertainty[active], multi_direction * 0.5)
        return uncertainty.sum() * 0.0
    return uncertainty.sum() * 0.0


def real_prototype_anchor_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    fake_margin: float = 0.2,
) -> torch.Tensor:
    state = outputs["state"].float()
    real_prototype = outputs["real_prototype"].view(1, -1).float()
    centered = F.normalize(state, dim=-1)
    proto = F.normalize(real_prototype, dim=-1, eps=1e-8)
    sim = (centered * proto).sum(dim=-1)
    labels = labels.long()
    real_mask = labels == 0
    fake_mask = labels == 1
    losses = []
    if real_mask.any():
        losses.append((1.0 - sim[real_mask]).mean())
    if fake_mask.any():
        losses.append(torch.relu(sim[fake_mask] - float(fake_margin)).pow(2).mean())
    if not losses:
        return state.sum() * 0.0
    return torch.stack(losses).mean()


def compute_mel_pretrain_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    lambda_eq: float = 1.0,
    lambda_sep: float = 0.1,
    lambda_sparse: float = 0.02,
    lambda_unc: float = 0.05,
    equivariance_objective: str = "equivariance",
    direction_weights: Sequence[float] | torch.Tensor | None = None,
    direction_focal_gamma: float = 1.5,
) -> Dict[str, torch.Tensor]:
    target = batch["edit_target"].float()
    active = target.sum(dim=-1) > 0
    if str(equivariance_objective) == "invariance":
        loss_eq = contrastive_invariance_loss(outputs, batch)
        loss_eq_head = outputs["edit_direction_logits"].sum() * 0.0
        loss_eq_proto = outputs["delta_alignment_scores"].sum() * 0.0
        loss_align = outputs["delta_alignment_scores"].sum() * 0.0
    elif active.any():
        loss_eq_head = balanced_multilabel_bce_loss(
            outputs["edit_direction_logits"][active],
            target[active],
            direction_weights=direction_weights,
            focal_gamma=direction_focal_gamma,
        )
        loss_eq_proto = prototype_alignment_bce_loss(
            outputs["delta_alignment_scores"][active],
            target[active],
            direction_weights=direction_weights,
            focal_gamma=direction_focal_gamma,
        )
        loss_eq = loss_eq_head + loss_eq_proto
        loss_align = manipulation_margin_alignment_loss(outputs["delta_alignment_scores"][active], target[active])
    else:
        loss_eq_head = outputs["edit_direction_logits"].sum() * 0.0
        loss_eq_proto = outputs["delta_alignment_scores"].sum() * 0.0
        loss_eq = loss_eq_head + loss_eq_proto
        loss_align = outputs["delta_alignment_scores"].sum() * 0.0
    loss_sep = direction_separation_loss(outputs["direction_prototypes"])
    loss_sparse = sparse_activation_loss(outputs["direction_activation"])
    loss_unc = uncertainty_direction_loss(outputs, batch)
    loss = (
        float(lambda_eq) * (loss_eq + loss_align)
        + float(lambda_sep) * loss_sep
        + float(lambda_sparse) * loss_sparse
        + float(lambda_unc) * loss_unc
    )
    return {
        "loss": loss,
        "loss_eq": loss_eq.detach(),
        "loss_eq_head": loss_eq_head.detach(),
        "loss_eq_proto": loss_eq_proto.detach(),
        "loss_align": loss_align.detach(),
        "loss_sep": loss_sep.detach(),
        "loss_sparse": loss_sparse.detach(),
        "loss_unc": loss_unc.detach(),
    }


def compute_mel_joint_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    lambda_eq: float = 0.5,
    lambda_sep: float = 0.1,
    lambda_sparse: float = 0.02,
    lambda_unc: float = 0.05,
    lambda_real_anchor: float = 0.01,
    equivariance_objective: str = "equivariance",
    direction_weights: Sequence[float] | torch.Tensor | None = None,
    direction_focal_gamma: float = 1.5,
) -> Dict[str, torch.Tensor]:
    labels = batch["label"].long()
    loss_cls = criterion(outputs["logits"], labels)
    target = batch["edit_target"].float()
    active = target.sum(dim=-1) > 0
    if str(equivariance_objective) == "invariance":
        loss_eq = contrastive_invariance_loss(outputs, batch)
        loss_eq_head = outputs["edit_direction_logits"].sum() * 0.0
        loss_eq_proto = outputs["delta_alignment_scores"].sum() * 0.0
        loss_align = outputs["delta_alignment_scores"].sum() * 0.0
    elif active.any():
        loss_eq_head = balanced_multilabel_bce_loss(
            outputs["edit_direction_logits"][active],
            target[active],
            direction_weights=direction_weights,
            focal_gamma=direction_focal_gamma,
        )
        loss_eq_proto = prototype_alignment_bce_loss(
            outputs["delta_alignment_scores"][active],
            target[active],
            direction_weights=direction_weights,
            focal_gamma=direction_focal_gamma,
        )
        loss_eq = loss_eq_head + loss_eq_proto
        loss_align = manipulation_margin_alignment_loss(outputs["delta_alignment_scores"][active], target[active])
    else:
        loss_eq_head = outputs["edit_direction_logits"].sum() * 0.0
        loss_eq_proto = outputs["delta_alignment_scores"].sum() * 0.0
        loss_eq = loss_eq_head + loss_eq_proto
        loss_align = outputs["delta_alignment_scores"].sum() * 0.0
    loss_sep = direction_separation_loss(outputs["direction_prototypes"])
    loss_sparse = sparse_activation_loss(outputs["direction_activation"])
    loss_unc = uncertainty_direction_loss(outputs, batch)
    loss_anchor = real_prototype_anchor_loss(outputs, labels)
    loss = (
        loss_cls
        + float(lambda_eq) * (loss_eq + loss_align)
        + float(lambda_sep) * loss_sep
        + float(lambda_sparse) * loss_sparse
        + float(lambda_unc) * loss_unc
        + float(lambda_real_anchor) * loss_anchor
    )
    return {
        "loss": loss,
        "loss_cls": loss_cls.detach(),
        "loss_eq": loss_eq.detach(),
        "loss_eq_head": loss_eq_head.detach(),
        "loss_eq_proto": loss_eq_proto.detach(),
        "loss_align": loss_align.detach(),
        "loss_sep": loss_sep.detach(),
        "loss_sparse": loss_sparse.detach(),
        "loss_unc": loss_unc.detach(),
        "loss_anchor": loss_anchor.detach(),
    }


def compute_mel_finetune_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    lambda_sep: float = 0.05,
    lambda_sparse: float = 0.02,
    lambda_unc: float = 0.05,
    lambda_real_anchor: float = 0.01,
) -> Dict[str, torch.Tensor]:
    labels = batch["label"].long()
    loss_cls = criterion(outputs["logits"], labels)
    loss_sep = direction_separation_loss(outputs["direction_prototypes"])
    loss_sparse = sparse_activation_loss(outputs["direction_activation"])
    loss_unc = uncertainty_direction_loss(outputs, None)
    loss_anchor = real_prototype_anchor_loss(outputs, labels)
    loss = loss_cls + float(lambda_sep) * loss_sep + float(lambda_sparse) * loss_sparse + float(lambda_real_anchor) * loss_anchor
    return {
        "loss": loss,
        "loss_cls": loss_cls.detach(),
        "loss_sep": loss_sep.detach(),
        "loss_sparse": loss_sparse.detach(),
        "loss_unc": loss_unc.detach(),
        "loss_anchor": loss_anchor.detach(),
    }
