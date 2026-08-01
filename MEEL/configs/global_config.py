from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("MEEL_RUNS_ROOT", PROJECT_ROOT / "outputs"))
FEATURE_CACHE_ROOT = Path(
    os.environ.get("MEEL_FEATURE_ROOT", PROJECT_ROOT / "data" / "features")
)

DATASETS = {
    "MR2_Chinese": {"language": "zh"},
    "MR2_English": {"language": "en"},
    "weibo": {"language": "zh"},
}

SEED = 42
NUM_CLASSES = 2

TRAIN_BATCH_SIZE = 24
TRAIN_LR = 1e-4
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_DROPOUT = 0.1
TRAIN_HIDDEN_DIM = 256
TRAIN_STATE_DIM = 256
