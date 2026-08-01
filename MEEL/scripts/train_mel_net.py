from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.global_config import DATASETS, FEATURE_CACHE_ROOT, RUNS_ROOT, SEED
from src.mel_net.constants import MANIPULATION_OPERATIONS
from src.mel_net.data import MELEditViewDataset, collate_mel, default_holdout_compositions, parse_compositions
from src.mel_net.losses import compute_mel_finetune_loss, compute_mel_joint_loss, compute_mel_pretrain_loss
from src.mel_net.metrics import compute_binary_metrics, direction_similarity_metrics, multilabel_direction_metrics, prefixed_metrics
from src.mel_net.model import MELNet
from src.mel_net.seed import set_seed


VARIANTS = [
    "baseline",
    "wo_scalar_cues",
    "wo_equivariance_learning",
    "wo_manipulation_directions",
    "wo_direction_separation",
    "wo_sparse_activation",
    "wo_equivariance_uncertainty",
    "pseudo_news_classification_only",
    "contrastive_invariance_substitute",
    "random_edit_transformations",
    "single_shared_direction",
    "no_pretraining_real_label_only",
]


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


class TeeLogger:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _resolve_holdouts(text: str) -> set[tuple[str, ...]]:
    if str(text).strip().lower() in {"default", "defaults"}:
        return default_holdout_compositions()
    return parse_compositions(text)


def _parse_direction_weights(text: str) -> list[float] | None:
    value = str(text).strip()
    if not value or value.lower() in {"default", "defaults", "auto"}:
        return None
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != len(MANIPULATION_OPERATIONS):
        raise ValueError(
            "--direction-loss-weights must provide "
            f"{len(MANIPULATION_OPERATIONS)} comma-separated values ordered as {MANIPULATION_OPERATIONS}."
        )
    return parts


def _loader(args, split: str, mode: str, shuffle: bool, composition_mode: str = "seen") -> DataLoader:
    ds = MELEditViewDataset(
        cache_path=FEATURE_CACHE_ROOT / args.dataset / split,
        mode=mode,
        samples_per_item=args.synthetic_multiplier if mode in {"pair", "holdout", "pseudo"} else 1,
        seed=args.seed + (0 if composition_mode == "seen" else 1009),
        holdout_compositions=args.holdout_set,
        composition_mode=composition_mode,
        transform_mode=args.transform_mode,
        semantic_top_k=args.semantic_top_k,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mel,
        generator=torch.Generator().manual_seed(args.seed + {"train": 11, "val": 23, "test": 37}.get(split, 0)),
    )


def _infer_dims(loader: DataLoader) -> tuple[int, int, int]:
    batch = next(iter(loader))
    return int(batch["text_global"].size(-1)), int(batch["image_global"].size(-1)), int(batch["text_entities"].size(-1))


@torch.no_grad()
def evaluate(model: MELNet, loader: DataLoader, device: torch.device, direction: bool = False) -> Dict[str, float]:
    model.eval()
    logits_list = []
    labels_list = []
    direction_logits = []
    prototype_direction_logits = []
    direction_targets = []
    uncertainties = []
    max_alignments = []
    confidences = []
    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(batch, include_edit=direction)
        logits_list.append(outputs["logits"].detach().cpu())
        labels_list.append(batch["label"].detach().cpu())
        uncertainties.append(outputs["eq_uncertainty"].detach().cpu())
        max_alignments.append(outputs["alignment_scores"].max(dim=-1).values.detach().cpu())
        confidences.append(outputs["probabilities"].max(dim=-1).values.detach().cpu())
        if direction:
            direction_logits.append(outputs["edit_direction_logits"].detach().cpu())
            prototype_direction_logits.append((outputs["delta_alignment_scores"] * 8.0).detach().cpu())
            direction_targets.append(batch["edit_target"].detach().cpu())
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    metrics = compute_binary_metrics(logits, labels)
    metrics["eq_uncertainty_mean"] = float(torch.cat(uncertainties).mean().item())
    metrics["max_alignment_mean"] = float(torch.cat(max_alignments).mean().item())
    unc = torch.cat(uncertainties).float()
    conf = torch.cat(confidences).float()
    if unc.numel() > 1 and float(unc.std(unbiased=False).item()) > 1e-8 and float(conf.std(unbiased=False).item()) > 1e-8:
        metrics["eq_uncertainty_confidence_corr"] = float(torch.corrcoef(torch.stack([unc, conf]))[0, 1].item())
    else:
        metrics["eq_uncertainty_confidence_corr"] = 0.0
    metrics.update(direction_similarity_metrics(model.direction_bank()))
    if direction and direction_logits:
        targets = torch.cat(direction_targets)
        metrics.update(multilabel_direction_metrics(torch.cat(direction_logits), targets, MANIPULATION_OPERATIONS))
        proto_metrics = multilabel_direction_metrics(torch.cat(prototype_direction_logits), targets, MANIPULATION_OPERATIONS)
        metrics.update(prefixed_metrics(proto_metrics, "proto_"))
    return metrics


@torch.no_grad()
def _estimate_alignment_offset(model: MELNet, loader: DataLoader, device: torch.device) -> list[float]:
    model.eval()
    chunks = []
    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(batch, include_edit=False)
        labels = batch["label"].long()
        real_mask = labels == 0
        if real_mask.any():
            chunks.append(outputs["raw_alignment_scores"][real_mask].detach())
    if not chunks:
        return [0.0 for _ in MANIPULATION_OPERATIONS]
    offset = torch.cat(chunks, dim=0).mean(dim=0)
    return [float(v) for v in offset.detach().cpu().tolist()]


def _refresh_alignment_offset(model: MELNet, loader: DataLoader, device: torch.device, args) -> list[float]:
    if not getattr(args, "use_alignment_calibration", True):
        model.set_alignment_offset(None)
        return [0.0 for _ in MANIPULATION_OPERATIONS]
    offset = _estimate_alignment_offset(model, loader, device)
    model.set_alignment_offset(offset)
    args.alignment_offset = offset
    return offset


def _train_epoch(model, loader, optimizer, criterion, device, args, stage: str) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    steps = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch, include_edit=(stage != "finetune"))
        if stage == "pretrain":
            loss_dict = compute_mel_pretrain_loss(
                outputs,
                batch,
                lambda_eq=args.lambda_eq,
                lambda_sep=args.lambda_sep,
                lambda_sparse=args.lambda_sparse,
                lambda_unc=args.lambda_unc,
                equivariance_objective=args.equivariance_objective,
                direction_weights=args.direction_loss_weights,
                direction_focal_gamma=args.direction_focal_gamma,
            )
        elif stage == "joint":
            if args.variant == "pseudo_news_classification_only":
                batch = dict(batch)
                batch["label"] = batch["pseudo_label"]
            loss_dict = compute_mel_joint_loss(
                outputs,
                batch,
                criterion,
                lambda_eq=args.lambda_eq,
                lambda_sep=args.lambda_sep,
                lambda_sparse=args.lambda_sparse,
                lambda_unc=args.lambda_unc,
                lambda_real_anchor=args.lambda_real_anchor,
                equivariance_objective=args.equivariance_objective,
                direction_weights=args.direction_loss_weights,
                direction_focal_gamma=args.direction_focal_gamma,
            )
        else:
            loss_dict = compute_mel_finetune_loss(
                outputs,
                batch,
                criterion,
                lambda_sep=args.lambda_sep,
                lambda_sparse=args.lambda_sparse,
                lambda_unc=args.lambda_unc,
                lambda_real_anchor=args.lambda_real_anchor,
            )
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
        steps += 1
        if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
            break
    return {key: value / max(steps, 1) for key, value in totals.items()}


def _configure_variant(args) -> None:
    args.use_direction_features = True
    args.use_sparse_activation = True
    args.use_uncertainty = True
    args.use_alignment_calibration = False
    args.use_scalar_cues = True
    args.single_shared_direction = False
    args.transform_mode = "semantic"
    args.equivariance_objective = "equivariance"
    args.train_loader_mode = "real"
    args.joint_loader_mode = "pair"
    args.effective_epochs_pretrain = args.epochs_pretrain
    args.effective_epochs_joint = args.epochs_joint
    args.effective_epochs_finetune = args.epochs_finetune
    args.effective_lambda_eq = args.lambda_eq
    args.effective_lambda_sep = args.lambda_sep
    args.effective_lambda_sparse = args.lambda_sparse
    args.effective_lambda_unc = args.lambda_unc
    args.effective_lambda_real_anchor = args.lambda_real_anchor

    if args.variant == "wo_scalar_cues":
        args.use_scalar_cues = False
    elif args.variant == "wo_equivariance_learning":
        args.effective_epochs_pretrain = 0
        args.effective_lambda_eq = 0.0
    elif args.variant == "wo_manipulation_directions":
        args.use_direction_features = False
        args.use_sparse_activation = False
        args.use_uncertainty = False
        args.use_alignment_calibration = False
        args.effective_epochs_pretrain = 0
        args.effective_lambda_eq = 0.0
        args.effective_lambda_sep = 0.0
        args.effective_lambda_sparse = 0.0
        args.effective_lambda_unc = 0.0
        args.effective_lambda_real_anchor = 0.0
    elif args.variant == "wo_direction_separation":
        args.effective_lambda_sep = 0.0
    elif args.variant == "wo_sparse_activation":
        args.use_sparse_activation = False
        args.effective_lambda_sparse = 0.0
    elif args.variant == "wo_equivariance_uncertainty":
        args.use_uncertainty = False
        args.effective_lambda_unc = 0.0
    elif args.variant == "pseudo_news_classification_only":
        args.use_direction_features = False
        args.use_sparse_activation = False
        args.use_uncertainty = False
        args.use_alignment_calibration = False
        args.train_loader_mode = "pseudo"
        args.joint_loader_mode = "pseudo"
        args.effective_epochs_pretrain = 0
        args.effective_epochs_finetune = 0
        args.effective_lambda_eq = 0.0
        args.effective_lambda_sep = 0.0
        args.effective_lambda_sparse = 0.0
        args.effective_lambda_unc = 0.0
        args.effective_lambda_real_anchor = 0.0
    elif args.variant == "contrastive_invariance_substitute":
        args.equivariance_objective = "invariance"
    elif args.variant == "random_edit_transformations":
        args.transform_mode = "random"
    elif args.variant == "single_shared_direction":
        args.single_shared_direction = True
    elif args.variant == "no_pretraining_real_label_only":
        args.effective_epochs_pretrain = 0
        args.effective_epochs_joint = 0
        args.effective_lambda_eq = 0.0
        args.effective_lambda_unc = 0.0

    args.lambda_eq = args.effective_lambda_eq
    args.lambda_sep = args.effective_lambda_sep
    args.lambda_sparse = args.effective_lambda_sparse
    args.lambda_unc = args.effective_lambda_unc
    args.lambda_real_anchor = args.effective_lambda_real_anchor


def _json_dump(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _serializable_args(args) -> Dict:
    payload = vars(args).copy()
    payload["holdout_set"] = ["+".join(item) for item in sorted(args.holdout_set)]
    return payload


def _write_run_config(run_dir: Path, args) -> None:
    _json_dump(run_dir / "config.json", _serializable_args(args))


def _make_run_dir(args) -> Path:
    root = Path(args.output_root) / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name.strip() if args.run_name.strip() else f"{time.strftime('%Y%m%d_%H%M%S')}_{args.variant}_seed{args.seed}"
    run_dir = root / run_name
    if run_dir.exists():
        if args.resume:
            return run_dir
        if args.force:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f"Run directory already exists: {run_dir}. Use --resume or --force.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_checkpoint(path: Path, model, optimizer, scheduler, args, history, epoch_state, best_val_f1: float, best_state) -> None:
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args).copy(),
        "history": history,
        "epoch_state": epoch_state,
        "best_val_f1": best_val_f1,
        "best_model_state": best_state,
    }
    payload["args"] = _serializable_args(args)
    torch.save(payload, path)


def _load_checkpoint(path: Path, model, optimizer, scheduler, device):
    payload = torch.load(path, map_location=device)
    model_state = payload["model_state"]
    if "alignment_offset" not in model_state:
        model_state = dict(model_state)
        model_state["alignment_offset"] = torch.zeros_like(model.alignment_offset.detach().cpu())
        payload["model_state"] = model_state
    best_state = payload.get("best_model_state")
    if isinstance(best_state, dict) and "alignment_offset" not in best_state:
        best_state = dict(best_state)
        best_state["alignment_offset"] = torch.zeros_like(model.alignment_offset.detach().cpu())
        payload["best_model_state"] = best_state
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MEL-Net.")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--variant", default="baseline", choices=VARIANTS)
    parser.add_argument("--output-root", default=str(RUNS_ROOT / "mel_net"))
    parser.add_argument("--epochs-pretrain", type=int, default=8)
    parser.add_argument("--epochs-joint", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--synthetic-multiplier", type=int, default=1)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--lambda-eq", type=float, default=1.0)
    parser.add_argument("--lambda-sep", type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=0.02)
    parser.add_argument("--lambda-unc", type=float, default=0.05)
    parser.add_argument("--lambda-real-anchor", type=float, default=0.01)
    parser.add_argument("--holdout-compositions", default="default")
    parser.add_argument("--semantic-top-k", type=int, default=32)
    parser.add_argument("--uncertainty-scale", type=float, default=5.0)
    parser.add_argument("--direction-focal-gamma", type=float, default=1.5)
    parser.add_argument(
        "--direction-loss-weights",
        default="default",
        help=(
            "Comma-separated operation weights ordered as "
            "image_swap,entity_shift,relation_misbind,context_drop,text_overclaim. "
            "Use default for the v4 balanced-boundary prior."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.holdout_set = _resolve_holdouts(args.holdout_compositions)
    args.direction_loss_weights = _parse_direction_weights(args.direction_loss_weights)
    _configure_variant(args)
    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = _make_run_dir(args)
    lock_path = run_dir / "RUNNING.lock"
    if lock_path.exists() and not args.resume:
        raise RuntimeError(f"Run appears active or interrupted: {lock_path}. Use --resume or --force intentionally.")
    lock_path.write_text(str(time.time()), encoding="utf-8")
    log_file = (run_dir / "train.log").open("a", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeLogger(original_stdout, log_file)
    sys.stderr = TeeLogger(original_stderr, log_file)

    _write_run_config(run_dir, args)
    manifest = {
        "command": [sys.executable, *sys.argv],
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "feature_cache_root": str(FEATURE_CACHE_ROOT),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if torch.cuda.is_available():
        manifest["cuda_device_0"] = torch.cuda.get_device_name(0)
        manifest["cuda_capability_0"] = list(torch.cuda.get_device_capability(0))
    _json_dump(run_dir / "manifest.json", manifest)

    try:
        pretrain_loader = _loader(args, "train", "pair", True, "seen")
        direction_val_loader = _loader(args, "val", "pair", False, "seen")
        heldout_val_loader = _loader(args, "val", "pair", False, "holdout")
        joint_loader = _loader(args, "train", args.joint_loader_mode, True, "seen")
        train_loader = _loader(args, "train", args.train_loader_mode, True)
        val_loader = _loader(args, "val", "real", False)
        test_loader = _loader(args, "test", "real", False)
        text_dim, image_dim, entity_dim = _infer_dims(train_loader)

        model = MELNet(
            text_dim,
            image_dim,
            entity_dim,
            hidden_dim=args.hidden_dim,
            state_dim=args.state_dim,
            dropout=args.dropout,
            use_direction_features=args.use_direction_features,
            use_sparse_activation=args.use_sparse_activation,
            use_uncertainty=args.use_uncertainty,
            single_shared_direction=args.single_shared_direction,
            uncertainty_scale=args.uncertainty_scale,
            use_alignment_calibration=args.use_alignment_calibration,
            use_scalar_cues=args.use_scalar_cues,
            alignment_offset=getattr(args, "alignment_offset", None),
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_epochs = max(args.effective_epochs_pretrain + args.effective_epochs_joint + args.effective_epochs_finetune, 1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        history = []
        epoch_state = {"pretrain": 0, "joint": 0, "finetune": 0}
        best_val_f1 = -1.0
        best_state = None
        latest_checkpoint = run_dir / "latest_checkpoint.pt"
        if args.resume and latest_checkpoint.exists():
            payload = _load_checkpoint(latest_checkpoint, model, optimizer, scheduler, device)
            history = payload.get("history", [])
            epoch_state = payload.get("epoch_state", epoch_state)
            best_val_f1 = float(payload.get("best_val_f1", -1.0))
            best_state = payload.get("best_model_state")
            saved_args = payload.get("args", {})
            if "alignment_offset" in saved_args:
                model.set_alignment_offset(saved_args.get("alignment_offset"))
                args.alignment_offset = saved_args.get("alignment_offset")
            print(json.dumps({"resumed_from": str(latest_checkpoint), "epoch_state": epoch_state}, ensure_ascii=False))

        for epoch in range(epoch_state.get("pretrain", 0) + 1, args.effective_epochs_pretrain + 1):
            train_loss = _train_epoch(model, pretrain_loader, optimizer, criterion, device, args, "pretrain")
            seen_metrics = evaluate(model, direction_val_loader, device, direction=True)
            heldout_metrics = evaluate(model, heldout_val_loader, device, direction=True)
            scheduler.step()
            row = {"stage": "pretrain", "epoch": epoch, "train": train_loss, "seen_val": seen_metrics, "heldout_val": heldout_metrics}
            history.append(row)
            epoch_state["pretrain"] = epoch
            print(json.dumps(row, ensure_ascii=False))
            _save_checkpoint(latest_checkpoint, model, optimizer, scheduler, args, history, epoch_state, best_val_f1, best_state)

        offset = _refresh_alignment_offset(model, train_loader, device, args)
        _write_run_config(run_dir, args)
        if args.use_alignment_calibration:
            print(json.dumps({"stage": "calibration", "after": "pretrain", "alignment_offset": offset}, ensure_ascii=False))

        for stage, epochs, loader in [("joint", args.effective_epochs_joint, joint_loader), ("finetune", args.effective_epochs_finetune, train_loader)]:
            for epoch in range(epoch_state.get(stage, 0) + 1, epochs + 1):
                train_loss = _train_epoch(model, loader, optimizer, criterion, device, args, stage)
                offset = _refresh_alignment_offset(model, train_loader, device, args)
                _write_run_config(run_dir, args)
                val_metrics = evaluate(model, val_loader, device, direction=False)
                direction_metrics = evaluate(model, direction_val_loader, device, direction=True)
                scheduler.step()
                row = {
                    "stage": stage,
                    "epoch": epoch,
                    "train": train_loss,
                    "val": val_metrics,
                    "direction_val": direction_metrics,
                    "alignment_offset": offset,
                }
                history.append(row)
                epoch_state[stage] = epoch
                print(json.dumps(row, ensure_ascii=False))
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    torch.save(best_state, run_dir / "best_model.pt")
                    _save_checkpoint(run_dir / "best_checkpoint.pt", model, optimizer, scheduler, args, history, epoch_state, best_val_f1, best_state)
                _save_checkpoint(latest_checkpoint, model, optimizer, scheduler, args, history, epoch_state, best_val_f1, best_state)

        if best_state is None:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_loader, device, direction=False)
        seen_test = evaluate(model, _loader(args, "test", "pair", False, "seen"), device, direction=True)
        heldout_test = evaluate(model, _loader(args, "test", "pair", False, "holdout"), device, direction=True)
        result = {
            "variant": args.variant,
            "best_val_f1": best_val_f1,
            "test": test_metrics,
            "direction_seen_test": seen_test,
            "direction_heldout_test": heldout_test,
            "history": history,
            "operations": MANIPULATION_OPERATIONS,
            "holdout_compositions": ["+".join(item) for item in sorted(args.holdout_set)],
        }
        _json_dump(run_dir / "result.json", result)
        _json_dump(
            run_dir / "summary.json",
            {
                "run_dir": str(run_dir),
                "dataset": args.dataset,
                "variant": args.variant,
                "best_val_f1": best_val_f1,
                "test": test_metrics,
                "direction_seen_test": seen_test,
                "direction_heldout_test": heldout_test,
                "epoch_state": epoch_state,
            },
        )
        print(json.dumps({"run_dir": str(run_dir), "test": test_metrics, "heldout_test": heldout_test}, indent=2))
    finally:
        if lock_path.exists():
            lock_path.unlink()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
