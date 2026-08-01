from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.global_config import FEATURE_CACHE_ROOT
from src.mel_net.analysis_utils import checkpoint_state, load_run_config, model_from_config, to_device
from src.mel_net.constants import MANIPULATION_OPERATIONS
from src.mel_net.data import MELEditViewDataset, collate_mel, default_holdout_compositions
from src.mel_net.metrics import compute_binary_metrics, direction_similarity_metrics, multilabel_direction_metrics, prefixed_metrics


def _infer_dims(loader: DataLoader) -> tuple[int, int, int]:
    batch = next(iter(loader))
    return int(batch["text_global"].size(-1)), int(batch["image_global"].size(-1)), int(batch["text_entities"].size(-1))


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() <= 1 or float(x.std(unbiased=False).item()) <= 1e-8 or float(y.std(unbiased=False).item()) <= 1e-8:
        return 0.0
    return float(torch.corrcoef(torch.stack([x.float(), y.float()]))[0, 1].item())


@torch.no_grad()
def _evaluate(model, loader, device: torch.device, direction: bool) -> Dict[str, float]:
    logits_list = []
    labels_list = []
    dir_logits = []
    proto_dir_logits = []
    dir_targets = []
    uncertainties = []
    confidences = []
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(batch, include_edit=direction)
        logits_list.append(outputs["logits"].detach().cpu())
        labels_list.append(batch["label"].detach().cpu())
        uncertainties.append(outputs["eq_uncertainty"].detach().cpu())
        confidences.append(outputs["probabilities"].max(dim=-1).values.detach().cpu())
        if direction:
            dir_logits.append(outputs["edit_direction_logits"].detach().cpu())
            proto_dir_logits.append((outputs["delta_alignment_scores"] * 8.0).detach().cpu())
            dir_targets.append(batch["edit_target"].detach().cpu())
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    metrics = compute_binary_metrics(logits, labels)
    unc = torch.cat(uncertainties)
    conf = torch.cat(confidences)
    metrics["eq_uncertainty_mean"] = float(unc.mean().item())
    metrics["eq_uncertainty_confidence_corr"] = _corr(unc, conf)
    metrics.update(direction_similarity_metrics(model.direction_bank()))
    if direction and dir_logits:
        targets = torch.cat(dir_targets)
        metrics.update(multilabel_direction_metrics(torch.cat(dir_logits), targets, MANIPULATION_OPERATIONS))
        proto_metrics = multilabel_direction_metrics(torch.cat(proto_dir_logits), targets, MANIPULATION_OPERATIONS)
        metrics.update(prefixed_metrics(proto_metrics, "proto_"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether learned MEL directions transfer across datasets.")
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_ds = MELEditViewDataset(FEATURE_CACHE_ROOT / args.target_dataset / args.split, mode="real")
    seen_ds = MELEditViewDataset(
        FEATURE_CACHE_ROOT / args.target_dataset / args.split,
        mode="pair",
        holdout_compositions=default_holdout_compositions(),
        composition_mode="seen",
    )
    holdout_ds = MELEditViewDataset(
        FEATURE_CACHE_ROOT / args.target_dataset / args.split,
        mode="pair",
        holdout_compositions=default_holdout_compositions(),
        composition_mode="holdout",
    )
    real_loader = DataLoader(real_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mel)
    seen_loader = DataLoader(seen_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mel)
    holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mel)

    payload = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint_state(payload)
    config = load_run_config(args.checkpoint, args.config or None)
    model = model_from_config(config, state, _infer_dims(real_loader))
    model.load_state_dict(state)
    model.to(device).eval()

    out = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": config,
        "target_real": _evaluate(model, real_loader, device, direction=False),
        "target_seen_edit": _evaluate(model, seen_loader, device, direction=True),
        "target_holdout_edit": _evaluate(model, holdout_loader, device, direction=True),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "target_real": out["target_real"], "target_holdout_edit": out["target_holdout_edit"]}, indent=2))


if __name__ == "__main__":
    main()
