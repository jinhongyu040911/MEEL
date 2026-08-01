from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mel_net.constants import MANIPULATION_OPERATIONS


def _orthogonal_directions(num_directions: int, state_dim: int, scale: float = 0.05) -> torch.Tensor:
    if num_directions <= 1:
        return torch.randn(num_directions, state_dim) * scale
    base = torch.randn(state_dim, num_directions)
    q, _ = torch.linalg.qr(base, mode="reduced")
    return q.t().contiguous() * scale


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)
    return (features.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(shifted, dim=dim, descending=True).values
    range_shape = [1] * shifted.dim()
    range_shape[dim] = shifted.size(dim)
    k = torch.arange(1, shifted.size(dim) + 1, device=shifted.device, dtype=shifted.dtype).view(range_shape)
    bound = 1 + k * zs
    cumsum = torch.cumsum(zs, dim=dim)
    is_gt = bound > cumsum
    k_z = is_gt.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, k_z.long() - 1) - 1) / k_z.to(shifted.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class EvidenceStateEncoder(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        entity_dim: int,
        state_dim: int,
        dropout: float,
        use_scalar_cues: bool = True,
    ) -> None:
        super().__init__()
        self.use_scalar_cues = bool(use_scalar_cues)
        input_dim = text_dim + image_dim + entity_dim + 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, batch: Dict[str, torch.Tensor], prefix: str = "") -> torch.Tensor:
        text = batch[f"{prefix}text_global"].float()
        image = batch[f"{prefix}image_global"].float()
        text_ent = _masked_mean(batch[f"{prefix}text_entities"], batch[f"{prefix}text_entity_mask"])
        image_ent = _masked_mean(batch[f"{prefix}image_entities"], batch[f"{prefix}image_entity_mask"])
        ent_abs = torch.abs(text_ent - image_ent)
        global_cos = F.cosine_similarity(text, image, dim=-1, eps=1e-8).unsqueeze(-1)
        ent_cos = F.cosine_similarity(text_ent, image_ent, dim=-1, eps=1e-8).unsqueeze(-1)
        relation_gap = torch.abs(text - image).mean(dim=-1, keepdim=True)
        text_count = batch[f"{prefix}text_entity_mask"].float().sum(dim=1, keepdim=True)
        image_count = batch[f"{prefix}image_entity_mask"].float().sum(dim=1, keepdim=True)
        coverage_gap = torch.relu(text_count - image_count) / text_count.clamp(min=1.0)
        claim_gap = torch.relu(1.0 - ent_cos) * (text_count / (image_count + 1.0)).clamp(max=5.0) / 5.0
        scalars = torch.cat(
            [
                global_cos,
                1.0 - global_cos,
                ent_cos,
                1.0 - ent_cos,
                relation_gap,
                (text * image).mean(dim=-1, keepdim=True),
                text_count / 30.0,
                image_count / 10.0,
                coverage_gap,
                claim_gap,
            ],
            dim=-1,
        )
        if not self.use_scalar_cues:
            scalars = torch.zeros_like(scalars)
        return self.encoder(torch.cat([text, image, ent_abs, scalars], dim=-1))


class MELNet(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        entity_dim: int,
        hidden_dim: int = 256,
        state_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_direction_features: bool = True,
        use_sparse_activation: bool = True,
        use_uncertainty: bool = True,
        single_shared_direction: bool = False,
        uncertainty_scale: float = 5.0,
        use_alignment_calibration: bool = False,
        use_scalar_cues: bool = True,
        alignment_offset: list[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.operations = list(MANIPULATION_OPERATIONS)
        self.num_ops = len(self.operations)
        self.num_direction_prototypes = 1 if single_shared_direction else self.num_ops
        self.use_direction_features = bool(use_direction_features)
        self.use_sparse_activation = bool(use_sparse_activation)
        self.use_uncertainty = bool(use_uncertainty)
        self.single_shared_direction = bool(single_shared_direction)
        self.uncertainty_scale = float(uncertainty_scale)
        self.use_alignment_calibration = bool(use_alignment_calibration)
        self.use_scalar_cues = bool(use_scalar_cues)
        self.state_encoder = EvidenceStateEncoder(
            text_dim,
            image_dim,
            entity_dim,
            state_dim,
            dropout,
            use_scalar_cues=self.use_scalar_cues,
        )
        self.real_prototype = nn.Parameter(torch.randn(state_dim) * 0.05)
        self.direction_prototypes = nn.Parameter(_orthogonal_directions(self.num_direction_prototypes, state_dim))
        if alignment_offset is None:
            offset = torch.zeros(self.num_ops)
        else:
            offset = torch.as_tensor(alignment_offset, dtype=torch.float32)
            if offset.numel() != self.num_ops:
                offset = torch.zeros(self.num_ops)
        self.register_buffer("alignment_offset", offset.view(self.num_ops), persistent=True)
        # Kept for backward checkpoint compatibility; baseline pi(x) is computed
        # directly from alignment scores to match the locked MEL formulation.
        self.activation_head = nn.Linear(self.num_ops, self.num_ops)
        classifier_extra_dim = self.num_ops + self.num_ops + 1
        self.classifier = nn.Sequential(
            nn.Linear(state_dim + classifier_extra_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_ops),
        )

    def _expanded_directions(self) -> torch.Tensor:
        if self.single_shared_direction:
            return self.direction_prototypes.expand(self.num_ops, -1)
        return self.direction_prototypes

    def direction_bank(self) -> torch.Tensor:
        return self._expanded_directions()

    def set_alignment_offset(self, offset: torch.Tensor | list[float] | None) -> None:
        if offset is None:
            self.alignment_offset.zero_()
            return
        value = torch.as_tensor(offset, device=self.alignment_offset.device, dtype=self.alignment_offset.dtype).view(-1)
        if value.numel() != self.num_ops:
            raise ValueError(f"alignment_offset must contain {self.num_ops} values.")
        self.alignment_offset.copy_(value)

    def _direction_alignment(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = state - self.real_prototype.view(1, -1)
        dirs = F.normalize(self._expanded_directions(), dim=-1)
        raw_scores = F.normalize(centered, dim=-1) @ dirs.t()
        scores = raw_scores - self.alignment_offset.view(1, -1) if self.use_alignment_calibration else raw_scores
        max_entropy = torch.log(torch.tensor(float(self.num_ops), device=state.device))
        activation_logits = scores * self.uncertainty_scale
        sparse_activation = _sparsemax(activation_logits, dim=-1)
        activation_probs = sparse_activation / sparse_activation.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        activation_entropy = -(activation_probs * torch.log(activation_probs.clamp(min=1e-8))).sum(dim=-1)
        activation_entropy = activation_entropy / max_entropy.clamp(min=1e-8)
        max_alignment = scores.max(dim=-1).values
        low_alignment_penalty = 1.0 - ((max_alignment + 1.0) * 0.5).clamp(min=0.0, max=1.0)
        uncertainty = 0.5 * activation_entropy + 0.5 * low_alignment_penalty
        if not self.use_direction_features:
            scores = torch.zeros_like(scores)
        if not self.use_sparse_activation:
            sparse_activation = torch.zeros_like(sparse_activation)
        if not self.use_uncertainty:
            uncertainty = torch.zeros_like(uncertainty)
            activation_entropy = torch.zeros_like(activation_entropy)
            low_alignment_penalty = torch.zeros_like(low_alignment_penalty)
        return raw_scores, scores, sparse_activation, uncertainty, activation_entropy, low_alignment_penalty

    def forward(self, batch: Dict[str, torch.Tensor], include_edit: bool = True) -> Dict[str, torch.Tensor]:
        state = self.state_encoder(batch, prefix="")
        raw_scores, scores, sparse_activation, uncertainty, activation_entropy, low_alignment_penalty = self._direction_alignment(state)
        logits = self.classifier(torch.cat([state, scores, sparse_activation, uncertainty.unsqueeze(-1)], dim=-1))
        outputs = {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "state": state,
            "raw_alignment_scores": raw_scores,
            "alignment_scores": scores,
            "direction_activation": sparse_activation,
            "eq_uncertainty": uncertainty,
            "direction_activation_entropy": activation_entropy,
            "low_alignment_penalty": low_alignment_penalty,
            "direction_logits": self.direction_head(state),
            "direction_prototypes": self.direction_prototypes,
            "direction_bank": self.direction_bank(),
            "real_prototype": self.real_prototype,
            "alignment_offset": self.alignment_offset,
        }
        if include_edit and "edit_text_global" in batch:
            edit_state = self.state_encoder(batch, prefix="edit_")
            (
                edit_raw_scores,
                edit_scores,
                edit_sparse_activation,
                edit_uncertainty,
                edit_activation_entropy,
                edit_low_alignment_penalty,
            ) = self._direction_alignment(edit_state)
            delta = edit_state - state
            delta_scores = F.normalize(delta, dim=-1) @ F.normalize(self._expanded_directions(), dim=-1).t()
            outputs.update(
                {
                    "edit_state": edit_state,
                    "edit_raw_alignment_scores": edit_raw_scores,
                    "edit_alignment_scores": edit_scores,
                    "edit_direction_activation": edit_sparse_activation,
                    "edit_eq_uncertainty": edit_uncertainty,
                    "edit_direction_activation_entropy": edit_activation_entropy,
                    "edit_low_alignment_penalty": edit_low_alignment_penalty,
                    "edit_delta": delta,
                    "delta_alignment_scores": delta_scores,
                    "edit_direction_logits": self.direction_head(delta),
                }
            )
        return outputs
