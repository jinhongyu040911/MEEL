from __future__ import annotations

import math
from typing import Dict

import torch


def direction_similarity_metrics(prototypes: torch.Tensor) -> Dict[str, float]:
    prototypes = prototypes.detach().float().cpu()
    if prototypes.size(0) <= 1:
        return {
            "direction_max_cosine": 0.0,
            "direction_mean_abs_cosine": 0.0,
        }
    dirs = torch.nn.functional.normalize(prototypes, dim=-1)
    sim = dirs @ dirs.t()
    eye = torch.eye(sim.size(0), dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    return {
        "direction_max_cosine": float(off_diag.max().item()),
        "direction_mean_abs_cosine": float(off_diag.abs().mean().item()),
    }


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=-1)
    labels = labels.long()
    tp = torch.sum((preds == 1) & (labels == 1)).item()
    tn = torch.sum((preds == 0) & (labels == 0)).item()
    fp = torch.sum((preds == 1) & (labels == 0)).item()
    fn = torch.sum((preds == 0) & (labels == 1)).item()
    total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return {
        "accuracy": float((tp + tn) / max(total, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(((tp * tn) - (fp * fn)) / denom),
    }


def multilabel_direction_metrics(logits: torch.Tensor, target: torch.Tensor, names: list[str]) -> Dict[str, float]:
    pred = (torch.sigmoid(logits.detach().cpu()) >= 0.5).float()
    target = target.detach().cpu().float()
    metrics: Dict[str, float] = {}
    total_tp = total_fp = total_fn = 0
    for idx, name in enumerate(names):
        tp = int(((pred[:, idx] == 1) & (target[:, idx] == 1)).sum().item())
        fp = int(((pred[:, idx] == 1) & (target[:, idx] == 0)).sum().item())
        fn = int(((pred[:, idx] == 0) & (target[:, idx] == 1)).sum().item())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics[f"dir_f1_{name}"] = 0.0 if precision + recall <= 0 else float(2 * precision * recall / (precision + recall))
    metrics["dir_f1_macro"] = float(sum(metrics[f"dir_f1_{name}"] for name in names) / max(len(names), 1))
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    metrics["dir_f1_micro"] = 0.0 if micro_p + micro_r <= 0 else float(2 * micro_p * micro_r / (micro_p + micro_r))
    metrics["dir_exact_match"] = float(torch.all(pred == target, dim=1).float().mean().item()) if pred.numel() else 0.0
    return metrics


def prefixed_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}
