# MEEL

Official implementation of Manipulation-Equivariant Evidence Learning (MEEL)
for multimodal fake news detection. MEEL learns structured evidence-state
displacements from semantic image-text edits and uses their alignment with
learned manipulation directions for veracity prediction.

## Requirements

- Python 3.9 or later
- PyTorch
- NumPy
- scikit-learn

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Data

The benchmark datasets and pretrained feature files are not redistributed.
Prepare frozen multimodal representations in the format described in
`DATA_FORMAT.md`, using the following directory layout:

```text
data/features/
  MR2_Chinese/{train,val,test}/
  MR2_English/{train,val,test}/
  weibo/{train,val,test}/
```

Each split may be a directory of `.pt` samples or a single `.pt` file
containing a list of samples. To keep features outside this repository, set
`MEEL_FEATURE_ROOT` to the feature root. Output location can similarly be set
with `MEEL_RUNS_ROOT`.

## Training

```bash
python scripts/train_mel_net.py \
  --dataset MR2_Chinese \
  --variant baseline \
  --device cuda \
  --seed 42
```

The training program supports the ablation variants used in the paper. Run
`python scripts/train_mel_net.py --help` for all optimization, loss, edit, and
evaluation options.

## Evaluation and Analysis

```bash
python scripts/evaluate_mel_direction_transfer.py --help
python scripts/analyze_mel_directions.py --help
python scripts/analyze_mel_risk_coverage.py --help
python scripts/audit_mel_edit_views.py --help
```

The `experiments/` directory contains the static frozen-feature baseline,
one-variable parameter sensitivity launcher, and direction-profile analysis
used in the revision experiments. Generated checkpoints, logs, cached
features, and result files are intentionally excluded from version control.
