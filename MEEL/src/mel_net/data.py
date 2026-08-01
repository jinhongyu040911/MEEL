from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.mel_net.constants import DEFAULT_HOLDOUT_COMPOSITIONS, EDIT_OPERATIONS, MANIPULATION_OPERATIONS


def parse_composition(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split("+") if part.strip())


def parse_compositions(values: Sequence[str] | str | None) -> set[tuple[str, ...]]:
    if values is None:
        return set()
    if isinstance(values, str):
        if not values.strip():
            return set()
        values = [item.strip() for item in values.split(",") if item.strip()]
    return {tuple(sorted(parse_composition(item))) for item in values if parse_composition(item)}


def default_holdout_compositions() -> set[tuple[str, ...]]:
    return parse_compositions(DEFAULT_HOLDOUT_COMPOSITIONS)


def _clone_item(item: Dict) -> Dict:
    out = {}
    for key, value in item.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def _valid_indices(mask: torch.Tensor) -> torch.Tensor:
    idx = torch.nonzero(mask.float() > 0, as_tuple=False).view(-1)
    return idx if idx.numel() > 0 else torch.arange(mask.numel())


def _first_existing_key(sample: Dict, candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in sample:
            return key
    return None


def _first_key_with_fields(sample: Dict, required_fields: Sequence[str]) -> str | None:
    for key, value in sample.items():
        if isinstance(value, dict) and all(field in value for field in required_fields):
            return key
    return None


def _metadata_key(sample: Dict) -> str | None:
    for key, value in sample.items():
        if isinstance(value, dict) and "label" in value:
            return key
    return None


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path: str | Path, lazy_load: bool = False) -> None:
        self.cache_path = Path(cache_path)
        self.dataset_name = self.cache_path.parent.name if self.cache_path.parent else ""
        self.split_name = self.cache_path.name if self.cache_path.name else ""
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Feature cache path does not exist: {self.cache_path}")
        if self.cache_path.is_file():
            self.features = torch.load(self.cache_path, map_location="cpu")
            self.feature_files = None
        else:
            self.feature_files = sorted(self.cache_path.glob("*.pt"))
            if not self.feature_files:
                raise ValueError(f"No .pt files found in {self.cache_path}")
            self.features = None if lazy_load else [torch.load(p, map_location="cpu") for p in self.feature_files]

    def __len__(self) -> int:
        return len(self.features) if self.features is not None else len(self.feature_files)

    def _flatten(self, sample: Dict) -> Dict:
        # Supports nested feature dictionaries without depending on source-specific key spellings.
        text_key = _first_existing_key(sample, ["文本", "text"]) or _first_key_with_fields(sample, ["clip_text_global"])
        image_key = _first_existing_key(sample, ["图像", "image"]) or _first_key_with_fields(sample, ["clip_image_global"])
        entity_key = _first_existing_key(sample, ["实体", "entity", "entities"]) or _first_key_with_fields(
            sample,
            ["text_entity_features", "image_entity_features", "text_entity_mask", "image_entity_mask"],
        )
        meta_key = _first_existing_key(sample, ["元数据", "metadata", "meta"]) or _metadata_key(sample)
        if isinstance(sample, dict) and text_key and image_key and entity_key and meta_key:
            meta = sample.get(meta_key, {})
            return {
                "sample_id": str(meta.get("sample_id", meta.get("id", ""))),
                "label": int(meta["label"]),
                "text_global": sample[text_key]["clip_text_global"].float(),
                "image_global": sample[image_key]["clip_image_global"].float(),
                "text_entities": sample[entity_key]["text_entity_features"].float(),
                "image_entities": sample[entity_key]["image_entity_features"].float(),
                "text_entity_mask": sample[entity_key]["text_entity_mask"].float(),
                "image_entity_mask": sample[entity_key]["image_entity_mask"].float(),
                "raw_text": meta.get("text", meta.get("caption", "")),
                "dataset": meta.get("dataset", self.dataset_name),
                "split": self.split_name,
            }
        return {
            "sample_id": str(sample.get("sample_id", sample.get("id", ""))),
            "label": int(sample["label"]),
            "text_global": sample["clip_text_global"].float(),
            "image_global": sample["clip_image_global"].float(),
            "text_entities": sample["text_entity_features"].float(),
            "image_entities": sample["image_entity_features"].float(),
            "text_entity_mask": sample["text_entity_mask"].float(),
            "image_entity_mask": sample["image_entity_mask"].float(),
            "raw_text": sample.get("text", sample.get("caption", "")),
            "dataset": sample.get("dataset", self.dataset_name),
            "split": self.split_name,
        }

    def __getitem__(self, index: int) -> Dict:
        sample = self.features[index] if self.features is not None else torch.load(self.feature_files[index], map_location="cpu")
        return self._flatten(sample)


class MELEditViewDataset(Dataset):
    """Builds original/edited pairs for self-supervised manipulation equivariance."""

    def __init__(
        self,
        cache_path: str | Path,
        mode: str = "pair",
        samples_per_item: int = 1,
        seed: int = 42,
        holdout_compositions: set[tuple[str, ...]] | None = None,
        composition_mode: str = "seen",
        transform_mode: str = "semantic",
        semantic_top_k: int = 32,
    ) -> None:
        self.base = CachedFeatureDataset(cache_path=cache_path, lazy_load=False)
        self.mode = str(mode)
        self.samples_per_item = max(1, int(samples_per_item))
        self.seed = int(seed)
        self.holdout_compositions = holdout_compositions or set()
        self.composition_mode = str(composition_mode)
        self.transform_mode = str(transform_mode)
        self.semantic_top_k = max(1, int(semantic_top_k))
        self.indices_by_label: Dict[int, List[int]] = {0: [], 1: []}
        self.text_index = []
        for idx in range(len(self.base)):
            sample = self.base[idx]
            self.indices_by_label.setdefault(int(sample["label"]), []).append(idx)
            self.text_index.append(sample["text_global"].float())
        self.text_index = torch.stack(self.text_index) if self.text_index else torch.empty(0)

    def __len__(self) -> int:
        return len(self.base) * (self.samples_per_item if self.mode in {"pair", "holdout", "pseudo"} else 1)

    def _rng_for_index(self, index: int) -> tuple[random.Random, torch.Generator]:
        seed = self.seed + int(index) * 1_000_003
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        return random.Random(seed), torch_gen

    def _candidate_pool(self, base_index: int, same_label: bool | None = None) -> List[int]:
        current = self.base[base_index]
        if same_label is None:
            pool = list(range(len(self.base)))
        else:
            label = int(current["label"]) if same_label else 1 - int(current["label"])
            pool = list(self.indices_by_label.get(label, []))
        return [idx for idx in pool if idx != base_index]

    def _semantic_scores(self, base_index: int, pool: List[int]) -> tuple[List[int], torch.Tensor]:
        if not pool or self.text_index.numel() == 0:
            return pool, torch.empty(len(pool))
        pool_tensor = torch.tensor(pool, dtype=torch.long)
        query = self.text_index[base_index].view(1, -1)
        sims = F.cosine_similarity(query, self.text_index[pool_tensor], dim=-1, eps=1e-8)
        return pool, sims

    def _semantic_pool(self, base_index: int, pool: List[int]) -> List[int]:
        pool, sims = self._semantic_scores(base_index, pool)
        if sims.numel() == 0:
            return pool
        pool_tensor = torch.tensor(pool, dtype=torch.long)
        k = min(self.semantic_top_k, int(pool_tensor.numel()))
        top_local = torch.topk(sims, k=k).indices
        return [int(pool_tensor[i].item()) for i in top_local]

    def _semantic_similarity(self, base_index: int, other_index: int) -> float:
        if self.text_index.numel() == 0:
            return 0.0
        query = self.text_index[base_index].view(1, -1)
        other = self.text_index[other_index].view(1, -1)
        return float(F.cosine_similarity(query, other, dim=-1, eps=1e-8).item())

    def _sample_other(
        self,
        base_index: int,
        rng: random.Random,
        same_label: bool | None = None,
        semantic: bool = True,
    ) -> tuple[Dict, Dict]:
        current = self.base[base_index]
        pool = self._candidate_pool(base_index, same_label=same_label)
        if not pool:
            return _clone_item(current), {
                "source_index": int(base_index),
                "source_sample_id": str(current.get("sample_id", "")),
                "source_label": int(current["label"]),
                "semantic_similarity": 1.0,
                "fallback_self": True,
            }
        if semantic:
            pool = self._semantic_pool(base_index, pool)
        other_index = int(rng.choice(pool))
        other = self.base[other_index]
        return _clone_item(other), {
            "source_index": other_index,
            "source_sample_id": str(other.get("sample_id", "")),
            "source_label": int(other["label"]),
            "semantic_similarity": self._semantic_similarity(base_index, other_index),
            "fallback_self": False,
        }

    def _target(self, ops: List[str]) -> torch.Tensor:
        target = torch.zeros(len(MANIPULATION_OPERATIONS), dtype=torch.float32)
        for op in ops:
            if op in MANIPULATION_OPERATIONS:
                target[MANIPULATION_OPERATIONS.index(op)] = 1.0
        return target

    def _choose_ops(self, rng: random.Random) -> List[str]:
        if self.composition_mode == "holdout":
            if self.holdout_compositions:
                comp = rng.choice(sorted(self.holdout_compositions))
            else:
                comp = ("image_swap", "text_overclaim")
            return list(comp)
        op_pool = list(MANIPULATION_OPERATIONS)
        for _ in range(50):
            op_count = 1 if rng.random() < 0.75 else 2
            ops = rng.sample(op_pool, k=op_count)
            if tuple(sorted(ops)) not in self.holdout_compositions:
                return ops
        return [rng.choice(op_pool)]

    def _image_swap(self, item: Dict, other: Dict) -> None:
        item["image_global"] = other["image_global"].clone()
        item["image_entities"] = other["image_entities"].clone()
        item["image_entity_mask"] = other["image_entity_mask"].clone()

    def _entity_shift(self, item: Dict, other: Dict, torch_gen: torch.Generator) -> None:
        text_idx = _valid_indices(item["text_entity_mask"])
        other_idx = _valid_indices(other["text_entity_mask"])
        # Use salient slots so the entity-level manipulation survives mean pooling.
        # This keeps the edit boundary entity-only while making the direction observable.
        shift_count = max(1, int(round(float(text_idx.numel()) * 0.4)))
        n = max(1, min(6, shift_count, int(other_idx.numel())))
        salience = item["text_entities"][text_idx].float().norm(dim=-1)
        top_k = min(int(text_idx.numel()), max(n * 2, n))
        candidate_local = torch.topk(salience, k=top_k).indices
        candidate_idx = text_idx[candidate_local]
        chosen = candidate_idx[torch.randperm(candidate_idx.numel(), generator=torch_gen)[:n]]
        repl = other_idx[torch.randint(0, other_idx.numel(), (n,), generator=torch_gen)]
        item["text_entities"][chosen] = other["text_entities"][repl]

    def _relation_misbind(self, item: Dict, other: Dict) -> None:
        item["text_global"] = other["text_global"].clone()
        item["text_entities"] = other["text_entities"].clone()
        item["text_entity_mask"] = other["text_entity_mask"].clone()
        item["raw_text"] = other.get("raw_text", item.get("raw_text", ""))

    def _context_drop(self, item: Dict, torch_gen: torch.Generator) -> None:
        valid = _valid_indices(item["text_entity_mask"])
        if valid.numel() > 1:
            drop_count = max(1, int(valid.numel() * 0.25))
            drop = valid[torch.randperm(valid.numel(), generator=torch_gen)[:drop_count]]
            item["text_entity_mask"][drop] = 0
            item["text_entities"][drop] = 0
        item["text_global"] = item["text_global"] * 0.85

    def _text_overclaim(self, item: Dict, other: Dict, torch_gen: torch.Generator) -> None:
        item["text_global"] = 0.65 * item["text_global"] + 0.35 * other["text_global"]
        text_slots = torch.arange(item["text_entity_mask"].numel())
        inactive = text_slots[item["text_entity_mask"].float() <= 0]
        other_idx = _valid_indices(other["text_entity_mask"])
        if inactive.numel() > 0 and other_idx.numel() > 0:
            n = max(1, min(3, int(inactive.numel()), int(other_idx.numel())))
            chosen = inactive[torch.randperm(inactive.numel(), generator=torch_gen)[:n]]
            repl = other_idx[torch.randint(0, other_idx.numel(), (n,), generator=torch_gen)]
            item["text_entities"][chosen] = other["text_entities"][repl]
            item["text_entity_mask"][chosen] = 1
            return
        text_idx = _valid_indices(item["text_entity_mask"])
        if text_idx.numel() > 0 and other_idx.numel() > 0:
            n = max(1, min(3, int(text_idx.numel()), int(other_idx.numel())))
            chosen = text_idx[torch.randperm(text_idx.numel(), generator=torch_gen)[:n]]
            repl = other_idx[torch.randint(0, other_idx.numel(), (n,), generator=torch_gen)]
            item["text_entities"][chosen] = 0.5 * item["text_entities"][chosen] + 0.5 * other["text_entities"][repl]

    def _random_transform(self, item: Dict, rng: random.Random, torch_gen: torch.Generator) -> None:
        choice = rng.choice(["text_noise", "image_noise", "entity_roll", "mask_drop"])
        if choice == "text_noise":
            noise = torch.randn(item["text_global"].shape, dtype=item["text_global"].dtype, generator=torch_gen)
            item["text_global"] = item["text_global"] + 0.08 * noise
        elif choice == "image_noise":
            noise = torch.randn(item["image_global"].shape, dtype=item["image_global"].dtype, generator=torch_gen)
            item["image_global"] = item["image_global"] + 0.08 * noise
        elif choice == "entity_roll":
            if item["text_entities"].size(0) > 1:
                item["text_entities"] = torch.roll(item["text_entities"], shifts=1, dims=0)
            if item["image_entities"].size(0) > 1:
                item["image_entities"] = torch.roll(item["image_entities"], shifts=1, dims=0)
        else:
            text_valid = _valid_indices(item["text_entity_mask"])
            image_valid = _valid_indices(item["image_entity_mask"])
            if text_valid.numel() > 1:
                item["text_entity_mask"][text_valid[torch.randperm(text_valid.numel(), generator=torch_gen)[:1]]] = 0
            if image_valid.numel() > 1:
                item["image_entity_mask"][image_valid[torch.randperm(image_valid.numel(), generator=torch_gen)[:1]]] = 0

    def _source_record(self, op: str, meta: Dict | None = None) -> Dict:
        record = {"op": op}
        if meta:
            record.update(meta)
        return record

    def _apply_ops(self, item: Dict, base_index: int, ops: List[str], rng: random.Random, torch_gen: torch.Generator) -> List[Dict]:
        records: List[Dict] = []
        if self.transform_mode == "random":
            for _ in ops:
                self._random_transform(item, rng, torch_gen)
                records.append(self._source_record("random_transform", {"source_index": int(base_index), "semantic_similarity": 1.0}))
            return records
        for op in ops:
            if op == "image_swap":
                other, meta = self._sample_other(base_index, rng, same_label=None, semantic=True)
                self._image_swap(item, other)
                records.append(self._source_record(op, meta))
            elif op == "entity_shift":
                other, meta = self._sample_other(base_index, rng, same_label=None, semantic=True)
                self._entity_shift(item, other, torch_gen)
                records.append(self._source_record(op, meta))
            elif op == "relation_misbind":
                other, meta = self._sample_other(base_index, rng, same_label=None, semantic=True)
                self._relation_misbind(item, other)
                records.append(self._source_record(op, meta))
            elif op == "context_drop":
                self._context_drop(item, torch_gen)
                records.append(self._source_record(op, {"source_index": int(base_index), "semantic_similarity": 1.0}))
            elif op == "text_overclaim":
                other, meta = self._sample_other(base_index, rng, same_label=True, semantic=True)
                self._text_overclaim(item, other, torch_gen)
                records.append(self._source_record(op, meta))
        return records

    def __getitem__(self, index: int) -> Dict:
        base_index = index % len(self.base)
        rng, torch_gen = self._rng_for_index(index)
        original = _clone_item(self.base[base_index])
        item = _clone_item(original)
        edit_records: List[Dict] = []
        if self.mode == "real" or (self.mode == "pseudo" and rng.random() < 0.5):
            ops: List[str] = []
        else:
            ops = self._choose_ops(rng)
            edit_records = self._apply_ops(item, base_index, ops, rng, torch_gen)
        out = dict(original)
        out["base_index"] = int(base_index)
        out["edit_ops"] = ops if ops else ["identity"]
        out["edit_records"] = edit_records
        out["edit_target"] = self._target(ops)
        out["pseudo_label"] = int(bool(ops))
        for key in ["text_global", "image_global", "text_entities", "image_entities", "text_entity_mask", "image_entity_mask"]:
            out[f"edit_{key}"] = item[key]
        return out


def _pad_2d(values: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(v.size(0) for v in values)
    dim = values[0].size(1)
    rows = []
    for v in values:
        if v.size(0) < max_len:
            v = torch.cat([v, torch.zeros(max_len - v.size(0), dim, dtype=v.dtype)], dim=0)
        rows.append(v)
    return torch.stack(rows)


def _pad_1d(values: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(v.size(0) for v in values)
    rows = []
    for v in values:
        if v.size(0) < max_len:
            v = torch.cat([v, torch.zeros(max_len - v.size(0), dtype=v.dtype)], dim=0)
        rows.append(v)
    return torch.stack(rows)


def collate_mel(batch: List[Dict]) -> Dict:
    out = {
        "sample_id": [item["sample_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "split": [item["split"] for item in batch],
        "raw_text": [item["raw_text"] for item in batch],
        "base_index": [int(item.get("base_index", -1)) for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "pseudo_label": torch.tensor([item.get("pseudo_label", 0) for item in batch], dtype=torch.long),
        "edit_target": torch.stack([item["edit_target"] for item in batch]),
        "edit_ops": [item.get("edit_ops", []) for item in batch],
        "edit_records": [item.get("edit_records", []) for item in batch],
    }
    for prefix in ["", "edit_"]:
        out[f"{prefix}text_global"] = torch.stack([item[f"{prefix}text_global"] for item in batch])
        out[f"{prefix}image_global"] = torch.stack([item[f"{prefix}image_global"] for item in batch])
        out[f"{prefix}text_entities"] = _pad_2d([item[f"{prefix}text_entities"] for item in batch])
        out[f"{prefix}image_entities"] = _pad_2d([item[f"{prefix}image_entities"] for item in batch])
        out[f"{prefix}text_entity_mask"] = _pad_1d([item[f"{prefix}text_entity_mask"] for item in batch])
        out[f"{prefix}image_entity_mask"] = _pad_1d([item[f"{prefix}image_entity_mask"] for item in batch])
    return out
