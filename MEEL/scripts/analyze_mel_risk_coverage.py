from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mel_net.analysis_utils import load_mel_model, to_device
from src.mel_net.metrics import compute_binary_metrics, direction_similarity_metrics


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    if x.numel() <= 1 or float(x.std(unbiased=False).item()) <= 1e-8 or float(y.std(unbiased=False).item()) <= 1e-8:
        return 0.0
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1].item())


def _risk_curve(rows: List[Dict], sort_key: str, coverage_points: List[float]) -> List[Dict]:
    rows_sorted = sorted(rows, key=lambda x: x[sort_key])
    curve = []
    for coverage in coverage_points:
        keep = max(1, int(len(rows_sorted) * coverage))
        subset = rows_sorted[:keep]
        logits = torch.tensor([[1.0 - r["prob_fake"], r["prob_fake"]] for r in subset])
        labels = torch.tensor([r["label"] for r in subset])
        metric = compute_binary_metrics(logits, labels)
        metric["coverage"] = coverage
        metric["kept"] = keep
        metric["sort_key"] = sort_key
        curve.append(metric)
    return curve


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MEL-Net risk-coverage by equivariance uncertainty.")
    parser.add_argument("--dataset", required=True)
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

    checkpoint = Path(args.checkpoint)
    model, config, loader = load_mel_model(
        checkpoint=checkpoint,
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
        config_path=args.config or None,
    )

    rows: List[Dict] = []
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(batch, include_edit=False)
        probs = outputs["probabilities"].detach().cpu()
        preds = probs.argmax(dim=-1)
        labels = batch["label"].detach().cpu()
        uncertainty = outputs["eq_uncertainty"].detach().cpu()
        max_align = outputs["alignment_scores"].max(dim=-1).values.detach().cpu()
        confidence = probs.max(dim=-1).values
        for i, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": int(labels[i].item()),
                    "pred": int(preds[i].item()),
                    "prob_fake": float(probs[i, 1].item()),
                    "confidence": float(confidence[i].item()),
                    "confidence_risk": float(1.0 - confidence[i].item()),
                    "eq_uncertainty": float(uncertainty[i].item()),
                    "max_alignment": float(max_align[i].item()),
                    "correct": int(preds[i].item() == labels[i].item()),
                }
            )

    coverage_points = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    eq_values = torch.tensor([r["eq_uncertainty"] for r in rows])
    confidence_values = torch.tensor([r["confidence"] for r in rows])
    confidence_risk_values = torch.tensor([r["confidence_risk"] for r in rows])
    out = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "config": config,
        "summary": {
            "num_samples": len(rows),
            "eq_uncertainty_confidence_corr": _corr(eq_values, confidence_values),
            "eq_uncertainty_confidence_risk_corr": _corr(eq_values, confidence_risk_values),
            **direction_similarity_metrics(model.direction_bank()),
        },
        "rows": rows,
        "risk_coverage_eq_uncertainty": _risk_curve(rows, "eq_uncertainty", coverage_points),
        "risk_coverage_confidence": _risk_curve(rows, "confidence_risk", coverage_points),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": out["summary"]}, indent=2))


if __name__ == "__main__":
    main()
