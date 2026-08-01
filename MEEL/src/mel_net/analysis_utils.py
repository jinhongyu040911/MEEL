from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from configs.global_config import FEATURE_CACHE_ROOT
from src.mel_net.data import MELEditViewDataset, collate_mel
from src.mel_net.model import MELNet


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def infer_dims(loader: DataLoader) -> tuple[int, int, int]:
    batch = next(iter(loader))
    return int(batch["text_global"].size(-1)), int(batch["image_global"].size(-1)), int(batch["text_entities"].size(-1))


def checkpoint_state(payload: Dict) -> Dict[str, torch.Tensor]:
    if "model_state" in payload:
        return payload["model_state"]
    if "best_model_state" in payload:
        return payload["best_model_state"]
    return payload


def load_run_config(checkpoint: str | Path, explicit_config: str | Path | None = None) -> Dict:
    checkpoint = Path(checkpoint)
    config_path = Path(explicit_config) if explicit_config else checkpoint.parent / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def model_from_config(config: Dict, state: Dict[str, torch.Tensor], dims: tuple[int, int, int]) -> MELNet:
    text_dim, image_dim, entity_dim = dims
    direction_shape = state["direction_prototypes"].shape
    return MELNet(
        text_dim,
        image_dim,
        entity_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        state_dim=int(config.get("state_dim", direction_shape[1])),
        dropout=float(config.get("dropout", 0.1)),
        use_direction_features=bool(config.get("use_direction_features", True)),
        use_sparse_activation=bool(config.get("use_sparse_activation", True)),
        use_uncertainty=bool(config.get("use_uncertainty", True)),
        single_shared_direction=bool(config.get("single_shared_direction", direction_shape[0] == 1)),
        uncertainty_scale=float(config.get("uncertainty_scale", 5.0)),
        use_alignment_calibration=bool(config.get("use_alignment_calibration", True)),
        use_scalar_cues=bool(config.get("use_scalar_cues", True)),
        alignment_offset=config.get("alignment_offset"),
    )


def load_mel_model(
    checkpoint: str | Path,
    dataset: str,
    split: str = "test",
    batch_size: int = 64,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
) -> tuple[MELNet, Dict, DataLoader]:
    device = device or torch.device("cpu")
    ds = MELEditViewDataset(FEATURE_CACHE_ROOT / dataset / split, mode="real")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mel)
    dims = infer_dims(loader)
    payload = torch.load(checkpoint, map_location="cpu")
    state = checkpoint_state(payload)
    config = load_run_config(checkpoint, config_path)
    model = model_from_config(config, state, dims)
    if "alignment_offset" not in state:
        state = dict(state)
        state["alignment_offset"] = torch.zeros(len(model.operations))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config, loader
