from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mel_net.analysis_utils import load_mel_model, to_device
from src.mel_net.constants import MANIPULATION_OPERATIONS
from src.mel_net.metrics import direction_similarity_metrics


def _pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.size(0) == 0:
        return torch.empty(0, 2)
    centered = x - x.mean(dim=0, keepdim=True)
    if centered.size(0) < 2:
        return torch.zeros(centered.size(0), 2)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[: min(2, vh.size(0))]
    coords = centered @ basis.t()
    if coords.size(1) == 1:
        coords = torch.cat([coords, torch.zeros(coords.size(0), 1)], dim=1)
    return coords[:, :2]


def _top_ops(values: torch.Tensor, k: int = 2) -> List[Dict]:
    top = torch.topk(values, k=min(k, values.numel()))
    return [{"op": MANIPULATION_OPERATIONS[int(i.item())], "score": float(v.item())} for v, i in zip(top.values, top.indices)]


def _calibration_baseline(rows: List[Dict]) -> Dict[str, float]:
    real_rows = [row for row in rows if int(row["label"]) == 0]
    source = real_rows or rows
    if not source:
        return {op: 0.0 for op in MANIPULATION_OPERATIONS}
    return {
        op: float(sum(row["alignment_scores"][op] for row in source) / max(len(source), 1))
        for op in MANIPULATION_OPERATIONS
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export MEL direction diagnostics and PCA coordinates.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config, loader = load_mel_model(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
        config_path=args.config or None,
    )
    states = []
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(batch, include_edit=False)
        probs = outputs["probabilities"].detach().cpu()
        state = outputs["state"].detach().cpu()
        align = outputs["alignment_scores"].detach().cpu()
        act = outputs["direction_activation"].detach().cpu()
        unc = outputs["eq_uncertainty"].detach().cpu()
        states.append(state)
        for i, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": int(batch["label"][i].detach().cpu().item()),
                    "pred": int(probs[i].argmax().item()),
                    "prob_fake": float(probs[i, 1].item()),
                    "eq_uncertainty": float(unc[i].item()),
                    "top_alignment": _top_ops(align[i]),
                    "top_activation": _top_ops(act[i]),
                    "alignment_scores": {op: float(align[i, j].item()) for j, op in enumerate(MANIPULATION_OPERATIONS)},
                    "direction_activation": {op: float(act[i, j].item()) for j, op in enumerate(MANIPULATION_OPERATIONS)},
                }
            )
            if len(rows) >= args.max_samples:
                break
        if len(rows) >= args.max_samples:
            break

    state_mat = torch.cat(states, dim=0)[: len(rows)]
    coords = _pca_2d(state_mat)
    for row, coord in zip(rows, coords):
        row["pca_x"] = float(coord[0].item())
        row["pca_y"] = float(coord[1].item())

    alignment_baseline = _calibration_baseline(rows)
    for row in rows:
        calibrated = {
            op: float(row["alignment_scores"][op] - alignment_baseline[op])
            for op in MANIPULATION_OPERATIONS
        }
        row["calibrated_alignment_scores"] = calibrated
        row["top_calibrated_alignment"] = sorted(
            [{"op": op, "score": score} for op, score in calibrated.items()],
            key=lambda item: item["score"],
            reverse=True,
        )[:2]

    bank = model.direction_bank().detach().cpu().float()
    dirs = torch.nn.functional.normalize(bank, dim=-1)
    cosine = dirs @ dirs.t()
    active_counts = [sum(1 for value in row["direction_activation"].values() if value > 1e-4) for row in rows]
    top_counter = Counter(row["top_activation"][0]["op"] for row in rows if row["top_activation"])
    calibrated_top_counter = Counter(row["top_calibrated_alignment"][0]["op"] for row in rows if row["top_calibrated_alignment"])
    high_uncertainty_rate = sum(1 for row in rows if row["eq_uncertainty"] >= 0.75) / max(len(rows), 1)
    out = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": config,
        "summary": {
            "num_samples": len(rows),
            **direction_similarity_metrics(bank),
            "prototype_cosine_matrix": cosine.tolist(),
            "active_direction_count_mean": float(sum(active_counts) / max(len(active_counts), 1)),
            "active_direction_count_histogram": {str(k): active_counts.count(k) for k in sorted(set(active_counts))},
            "top_activation_distribution": dict(top_counter),
            "alignment_calibration_baseline": alignment_baseline,
            "top_calibrated_alignment_distribution": dict(calibrated_top_counter),
            "high_eq_uncertainty_rate": float(high_uncertainty_rate),
        },
        "operations": MANIPULATION_OPERATIONS,
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": out["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
