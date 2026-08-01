from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.global_config import FEATURE_CACHE_ROOT
from src.mel_net.constants import MANIPULATION_OPERATIONS
from src.mel_net.data import MELEditViewDataset, default_holdout_compositions


def _l2_delta(item: Dict, key: str) -> float:
    return float(torch.norm(item[f"edit_{key}"].float() - item[key].float()).item())


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _summarize(rows: List[Dict]) -> Dict:
    by_op: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        for op in row["edit_ops"]:
            if op != "identity":
                by_op[op].append(row)
    summary = {"num_rows": len(rows), "operations": {}}
    for op in MANIPULATION_OPERATIONS:
        op_rows = by_op.get(op, [])
        sims = [float(rec.get("semantic_similarity", 0.0)) for row in op_rows for rec in row["edit_records"] if rec.get("op") == op]
        fallback = [bool(rec.get("fallback_self", False)) for row in op_rows for rec in row["edit_records"] if rec.get("op") == op]
        summary["operations"][op] = {
            "count": len(op_rows),
            "semantic_similarity_mean": _mean(sims),
            "semantic_similarity_min": float(min(sims)) if sims else 0.0,
            "fallback_self_rate": _mean([1.0 if item else 0.0 for item in fallback]),
            "text_global_delta_mean": _mean([row["text_global_delta"] for row in op_rows]),
            "image_global_delta_mean": _mean([row["image_global_delta"] for row in op_rows]),
            "text_entity_delta_mean": _mean([row["text_entity_delta"] for row in op_rows]),
            "image_entity_delta_mean": _mean([row["image_entity_delta"] for row in op_rows]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MEL edit-view construction before formal experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--composition-mode", default="seen", choices=["seen", "holdout"])
    parser.add_argument("--transform-mode", default="semantic", choices=["semantic", "random"])
    parser.add_argument("--semantic-top-k", type=int, default=32)
    args = parser.parse_args()

    ds = MELEditViewDataset(
        FEATURE_CACHE_ROOT / args.dataset / args.split,
        mode="pair",
        seed=args.seed,
        holdout_compositions=default_holdout_compositions(),
        composition_mode=args.composition_mode,
        transform_mode=args.transform_mode,
        semantic_top_k=args.semantic_top_k,
    )
    rows = []
    for idx in range(min(args.samples, len(ds))):
        item = ds[idx]
        rows.append(
            {
                "index": idx,
                "base_index": item["base_index"],
                "sample_id": item["sample_id"],
                "label": int(item["label"]),
                "edit_ops": item["edit_ops"],
                "edit_records": item["edit_records"],
                "text_global_delta": _l2_delta(item, "text_global"),
                "image_global_delta": _l2_delta(item, "image_global"),
                "text_entity_delta": _l2_delta(item, "text_entities"),
                "image_entity_delta": _l2_delta(item, "image_entities"),
            }
        )

    out = {
        "dataset": args.dataset,
        "split": args.split,
        "composition_mode": args.composition_mode,
        "transform_mode": args.transform_mode,
        "semantic_top_k": args.semantic_top_k,
        "summary": _summarize(rows),
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": out["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
