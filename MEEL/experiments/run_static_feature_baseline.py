from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.global_config import DATASETS, FEATURE_CACHE_ROOT, SEED
from src.mel_net.metrics import compute_binary_metrics
from src.mel_net.seed import set_seed


def _load_split(dataset: str, split: str, max_samples: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    root = FEATURE_CACHE_ROOT / dataset / split
    files = sorted(root.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No feature files found under {root}")
    if max_samples > 0:
        files = files[:max_samples]
    xs = []
    ys = []
    for path in files:
        sample = torch.load(path, map_location="cpu")
        text = sample.get("文本", sample.get("text", sample))
        image = sample.get("图像", sample.get("image", sample))
        meta = sample.get("元数据", sample.get("metadata", sample))
        t = text["clip_text_global"].float()
        v = image["clip_image_global"].float()
        cos = torch.nn.functional.cosine_similarity(t.view(1, -1), v.view(1, -1), dim=-1).view(1)
        xs.append(torch.cat([t, v, torch.abs(t - v), t * v, cos], dim=0))
        label = meta.get("label", sample.get("label"))
        ys.append(int(label.item() if torch.is_tensor(label) else label))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class LinearStaticClassifier(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def _standardize(train_x: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return tuple((x - mean) / std for x in (train_x, *others))


@torch.no_grad()
def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> Dict[str, float]:
    model.eval()
    logits = model(x.to(device)).cpu()
    return compute_binary_metrics(logits, y)


def run_dataset(dataset: str, args: argparse.Namespace) -> Dict[str, object]:
    train_x, train_y = _load_split(dataset, "train", args.max_train)
    val_x, val_y = _load_split(dataset, "val", args.max_val)
    test_x, test_y = _load_split(dataset, "test", args.max_test)
    train_x, val_x, test_x = _standardize(train_x, val_x, test_x)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model = LinearStaticClassifier(train_x.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    best_state = None
    best_val = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            steps += 1
        val = _evaluate(model, val_x, val_y, device)
        row = {"epoch": epoch, "loss": total / max(steps, 1), "val_f1": val["f1"], "val_mcc": val["mcc"]}
        history.append(row)
        if val["f1"] > best_val:
            best_val = val["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test = _evaluate(model, test_x, test_y, device)
    return {
        "dataset": dataset,
        "model": "CLIP-static linear classifier",
        "feature_dim": int(train_x.size(1)),
        "train_size": int(train_y.numel()),
        "val_size": int(val_y.numel()),
        "test_size": int(test_y.numel()),
        "best_val_f1": float(best_val),
        "test": test,
        "history_tail": history[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen CLIP static linear baseline.")
    parser.add_argument("--datasets", default="MR2_Chinese,MR2_English,weibo")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "experiments" / "static_clip_baseline_summary.json"))
    parser.add_argument("--csv-output", default=str(PROJECT_ROOT / "experiments" / "static_clip_baseline_summary.csv"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--max-train", type=int, default=0, help="Use only the first N train samples; 0 means all.")
    parser.add_argument("--max-val", type=int, default=0, help="Use only the first N validation samples; 0 means all.")
    parser.add_argument("--max-test", type=int, default=0, help="Use only the first N test samples; 0 means all.")
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    results = [run_dataset(dataset, args) for dataset in datasets]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    with Path(args.csv_output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "feature_dim", "test_accuracy", "test_f1", "test_mcc", "best_val_f1"])
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "feature_dim": row["feature_dim"],
                    "test_accuracy": row["test"]["accuracy"],
                    "test_f1": row["test"]["f1"],
                    "test_mcc": row["test"]["mcc"],
                    "best_val_f1": row["best_val_f1"],
                }
            )
    print(json.dumps({"output": str(output), "csv": args.csv_output, "datasets": datasets}, indent=2))


if __name__ == "__main__":
    main()
