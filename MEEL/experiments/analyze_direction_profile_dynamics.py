from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.global_config import DATASETS, FEATURE_CACHE_ROOT
from scripts.train_mel_net import _infer_dims, _load_checkpoint, _to_device, evaluate
from src.mel_net.constants import MANIPULATION_OPERATIONS
from src.mel_net.data import MELEditViewDataset, collate_mel
from src.mel_net.model import MELNet


def _loader(dataset: str, split: str, batch_size: int) -> DataLoader:
    ds = MELEditViewDataset(cache_path=FEATURE_CACHE_ROOT / dataset / split, mode="real")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_mel)


@torch.no_grad()
def analyze(dataset: str, checkpoint: Path, args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    loader = _loader(dataset, args.split, args.batch_size)
    text_dim, image_dim, entity_dim = _infer_dims(loader)
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("args", {})
    model = MELNet(
        text_dim,
        image_dim,
        entity_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        state_dim=int(cfg.get("state_dim", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_direction_features=bool(cfg.get("use_direction_features", True)),
        use_sparse_activation=bool(cfg.get("use_sparse_activation", True)),
        use_uncertainty=bool(cfg.get("use_uncertainty", True)),
        single_shared_direction=bool(cfg.get("single_shared_direction", False)),
        uncertainty_scale=float(cfg.get("uncertainty_scale", 5.0)),
        use_alignment_calibration=bool(cfg.get("use_alignment_calibration", False)),
        use_scalar_cues=bool(cfg.get("use_scalar_cues", True)),
        alignment_offset=cfg.get("alignment_offset"),
    ).to(device)
    state = ckpt.get("best_model_state") or ckpt.get("model_state")
    if state is None and all(torch.is_tensor(value) for value in ckpt.values()):
        state = ckpt
    if state is None:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    model.load_state_dict(state)
    model.eval()

    activations = []
    uncertainties = []
    max_alignments = []
    labels = []
    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(batch, include_edit=False)
        activations.append(outputs["direction_activation"].detach().cpu())
        uncertainties.append(outputs["eq_uncertainty"].detach().cpu())
        max_alignments.append(outputs["alignment_scores"].max(dim=-1).values.detach().cpu())
        labels.append(batch["label"].detach().cpu())
    pi = torch.cat(activations).float()
    unc = torch.cat(uncertainties).float()
    max_align = torch.cat(max_alignments).float()
    y = torch.cat(labels).long()
    probs = pi / pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
    if pi.size(1) > 1:
        entropy = entropy / torch.log(torch.tensor(float(pi.size(1))))
    top = pi.argmax(dim=-1)
    top_counts = {MANIPULATION_OPERATIONS[i]: int((top == i).sum().item()) for i in range(len(MANIPULATION_OPERATIONS))}
    per_direction = {}
    for i, name in enumerate(MANIPULATION_OPERATIONS):
        values = pi[:, i]
        per_direction[name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    return {
        "dataset": dataset,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "n": int(pi.size(0)),
        "activation_mean_std_across_directions": float(pi.std(dim=1, unbiased=False).mean().item()),
        "activation_entropy_mean": float(entropy.mean().item()),
        "activation_entropy_std": float(entropy.std(unbiased=False).item()),
        "eq_uncertainty_mean": float(unc.mean().item()),
        "eq_uncertainty_std": float(unc.std(unbiased=False).item()),
        "max_alignment_mean": float(max_align.mean().item()),
        "top_direction_counts": top_counts,
        "per_direction": per_direction,
        "label_counts": {"real": int((y == 0).sum().item()), "fake": int((y == 1).sum().item())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sparse direction-profile dynamics for trained MEEL checkpoints.")
    parser.add_argument("--checkpoint-map", required=True, help="JSON mapping dataset names to checkpoint paths.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=str(PROJECT_ROOT / "experiments" / "direction_profile_dynamics.json"))
    parser.add_argument("--csv-output", default=str(PROJECT_ROOT / "experiments" / "direction_profile_dynamics.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    mapping = json.loads(Path(args.checkpoint_map).read_text(encoding="utf-8"))
    results = []
    for dataset, checkpoint in mapping.items():
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        results.append(analyze(dataset, Path(checkpoint), args))
    Path(args.output).write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    with Path(args.csv_output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "n", "activation_mean_std_across_directions", "activation_entropy_mean", "eq_uncertainty_mean", "max_alignment_mean"])
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps({"output": args.output, "csv": args.csv_output}, indent=2))


if __name__ == "__main__":
    main()
