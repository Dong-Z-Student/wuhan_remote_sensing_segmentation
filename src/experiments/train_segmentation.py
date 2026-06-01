from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.wuhan_dataset import WuhanPatchDataset
from src.models.optical_unet import OpticalUNet
from src.models.dual_branch_unet import DualBranchUNet
from src.models.sar_unet import SARUNet
from src.utils.metrics import (
    fast_confusion_matrix,
    compute_metrics_from_confusion_matrix,
    format_metrics,
)
from src.utils.seed import set_seed


CLASS_NAMES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built_area",
    "bare_ground",
]


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_output_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "predictions": output_dir / "predictions",
    }

    for path in dirs.values():
        ensure_dir(path)

    return dirs


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    mode = cfg["experiment"]["mode"]

    if mode not in ["sar", "optical", "dual"]:
        raise ValueError(
            f"当前 train_segmentation.py 支持 mode='sar', 'optical' 或 'dual'，当前为: {mode}"
        )

    train_list = PROJECT_ROOT / cfg["data"]["train_list"]
    val_list = PROJECT_ROOT / cfg["data"]["val_list"]

    sar_indices = cfg["bands"].get("sar_indices", [0, 1])
    optical_indices = cfg["bands"]["optical_indices"]
    label_index = cfg["bands"]["label_index"]

    ignore_label_values = cfg["data"].get("ignore_label_values", [])
    ignore_index = int(cfg["data"].get("ignore_index", 255))

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))

    train_dataset = WuhanPatchDataset(
        split_txt=train_list,
        project_root=PROJECT_ROOT,
        mode=mode,
        sar_indices=sar_indices,
        optical_indices=optical_indices,
        label_index=label_index,
        ignore_label_values=ignore_label_values,
        ignore_index=ignore_index,
        normalize=False,
        check_nan=True,
    )

    val_dataset = WuhanPatchDataset(
        split_txt=val_list,
        project_root=PROJECT_ROOT,
        mode=mode,
        sar_indices=sar_indices,
        optical_indices=optical_indices,
        label_index=label_index,
        ignore_label_values=ignore_label_values,
        ignore_index=ignore_index,
        normalize=False,
        check_nan=True,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print("Dataset 构建完成")
    print(f"  mode          : {mode}")
    print(f"  train samples : {len(train_dataset)}")
    print(f"  val samples   : {len(val_dataset)}")
    print(f"  batch_size    : {batch_size}")
    print(f"  sar_indices   : {sar_indices}")
    print(f"  optical_indices: {optical_indices}")
    print(f"  ignore_label_values: {ignore_label_values}")
    print(f"  ignore_index  : {ignore_index}")

    return train_loader, val_loader


def build_model(cfg: Dict) -> nn.Module:
    model_name = cfg["experiment"]["model"]
    model_cfg = cfg["model"]

    # 类别数以 data.num_classes 为准，避免 model.num_classes 出错
    num_classes = int(cfg["data"]["num_classes"])

    if model_name == "sar_unet":
        model = SARUNet(
            in_channels=int(model_cfg["in_channels"]),
            num_classes=num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
            bilinear=bool(model_cfg.get("bilinear", True)),
        )

        print("\n模型构建完成:")
        print(f"  model        : {model_name}")
        print(f"  in_channels  : {int(model_cfg['in_channels'])}")
        print(f"  num_classes  : {num_classes}")
        print(f"  base_channels: {int(model_cfg.get('base_channels', 32))}")

        return model

    if model_name == "optical_unet":
        model = OpticalUNet(
            in_channels=int(model_cfg["in_channels"]),
            num_classes=num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
            bilinear=bool(model_cfg.get("bilinear", True)),
        )

        print("\n模型构建完成:")
        print(f"  model        : {model_name}")
        print(f"  in_channels  : {int(model_cfg['in_channels'])}")
        print(f"  num_classes  : {num_classes}")
        print(f"  base_channels: {int(model_cfg.get('base_channels', 32))}")

        return model

    if model_name == "dual_branch_unet":
        model = DualBranchUNet(
            sar_in_channels=int(model_cfg["sar_in_channels"]),
            optical_in_channels=int(model_cfg["optical_in_channels"]),
            num_classes=num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
            bilinear=bool(model_cfg.get("bilinear", True)),
        )

        print("\n模型构建完成:")
        print(f"  model              : {model_name}")
        print(f"  sar_in_channels    : {int(model_cfg['sar_in_channels'])}")
        print(f"  optical_in_channels: {int(model_cfg['optical_in_channels'])}")
        print(f"  num_classes        : {num_classes}")
        print(f"  base_channels      : {int(model_cfg.get('base_channels', 32))}")
        print(f"  fusion             : {model_cfg.get('fusion', 'multiscale_concat')}")

        return model

    raise ValueError(
        f"不支持的 model: {model_name}. "
        f"当前支持: sar_unet, optical_unet, dual_branch_unet"
    )


def compute_class_weights_from_split_distribution(
    cfg: Dict,
    device: torch.device,
) -> torch.Tensor | None:
    """
    根据 data/wuhan/meta/split_class_distribution.csv 中的 train split 类别分布
    计算 class weights。
    """
    loss_cfg = cfg.get("loss", {})
    use_class_weight = bool(loss_cfg.get("use_class_weight", False))

    if not use_class_weight:
        print("class weight: disabled")
        return None

    num_classes = int(cfg["data"]["num_classes"])

    source_rel = loss_cfg.get(
        "class_weight_source",
        "data/wuhan/meta/split_class_distribution.csv",
    )
    source_path = PROJECT_ROOT / source_rel

    if not source_path.exists():
        raise FileNotFoundError(
            f"class_weight_source 不存在: {source_path}\n"
            f"请确认已经运行 scripts/03_prepare_dataset.py"
        )

    method = str(loss_cfg.get("class_weight_method", "inverse_sqrt"))
    normalize = bool(loss_cfg.get("normalize_class_weight", True))
    min_weight = float(loss_cfg.get("min_weight", 0.25))
    max_weight = float(loss_cfg.get("max_weight", 5.0))

    counts = torch.zeros(num_classes, dtype=torch.float64)

    with open(source_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            split = row["split"]
            class_id = int(row["class_id"])
            pixel_count = int(row["pixel_count"])

            # 只使用 train split，并且只使用 0 到 num_classes-1
            # 当前 num_classes=8，因此会自动忽略原始 class 8
            if split == "train" and 0 <= class_id < num_classes:
                counts[class_id] = pixel_count

    if torch.any(counts <= 0):
        zero_classes = torch.where(counts <= 0)[0].tolist()
        raise ValueError(
            f"以下类别在 train split 中像元数为 0，无法计算类别权重: {zero_classes}"
        )

    freq = counts / counts.sum()
    eps = 1e-12

    if method == "inverse":
        weights = 1.0 / (freq + eps)

    elif method == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(freq + eps)

    elif method == "median_frequency":
        median_freq = torch.median(freq)
        weights = median_freq / (freq + eps)

    else:
        raise ValueError(
            f"不支持的 class_weight_method: {method}. "
            f"可选: inverse, inverse_sqrt, median_frequency"
        )

    if normalize:
        weights = weights / weights.mean()

    weights = torch.clamp(
        weights,
        min=min_weight,
        max=max_weight,
    )

    weights = weights.float().to(device)

    print("\n类别权重设置:")
    print(f"  source : {source_path}")
    print(f"  method : {method}")
    print(f"  normalize: {normalize}")
    print(f"  min_weight: {min_weight}")
    print(f"  max_weight: {max_weight}")

    for class_id, weight in enumerate(weights.detach().cpu().tolist()):
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        print(f"  class {class_id} {class_name:20s}: weight={weight:.4f}, count={int(counts[class_id].item())}")

    return weights


def build_criterion(
    cfg: Dict,
    device: torch.device,
) -> nn.Module:
    """
    构建损失函数。
    如果 loss.use_class_weight=true，则加入类别权重。
    """
    ignore_index = int(cfg["data"].get("ignore_index", 255))

    class_weights = compute_class_weights_from_split_distribution(
        cfg=cfg,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=ignore_index,
    )

    print("\nLoss 构建完成:")
    print(f"  type: CrossEntropyLoss")
    print(f"  ignore_index: {ignore_index}")
    print(f"  use_class_weight: {class_weights is not None}")

    return criterion


def move_batch_to_device(
    batch,
    mode: str,
    device: torch.device,
):
    """
    将 DataLoader 返回的 batch 移动到 GPU。
    """
    if mode == "sar":
        sar, masks = batch
        sar = sar.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        return sar, masks

    if mode == "optical":
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        return images, masks

    if mode == "dual":
        sar, optical, masks = batch
        sar = sar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        return (sar, optical), masks

    raise ValueError(f"不支持的 mode: {mode}")


def forward_model(
    model: nn.Module,
    inputs,
    mode: str,
) -> torch.Tensor:
    """
    根据 mode 调用模型。
    """
    if mode == "sar":
        return model(inputs)

    if mode == "optical":
        return model(inputs)

    if mode == "dual":
        sar, optical = inputs
        return model(sar, optical)

    raise ValueError(f"不支持的 mode: {mode}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(loader, start=1):
        inputs, masks = move_batch_to_device(
            batch=batch,
            mode=mode,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = forward_model(
            model=model,
            inputs=inputs,
            mode=mode,
        )

        if step == 1:
            print(f"    debug logits shape: {tuple(logits.shape)}")
            print(f"    debug masks unique : {torch.unique(masks)}")

        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = masks.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        if step % 20 == 0 or step == len(loader):
            print(
                f"    train step {step:04d}/{len(loader):04d}, "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    mode: str,
) -> Tuple[float, Dict]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    cm_total = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )

    for batch in loader:
        inputs, masks = move_batch_to_device(
            batch=batch,
            mode=mode,
            device=device,
        )

        logits = forward_model(
            model=model,
            inputs=inputs,
            mode=mode,
        )

        loss = criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)

        cm = fast_confusion_matrix(
            preds=preds,
            targets=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        cm_total += cm.to(device)

        batch_size = masks.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    metrics = compute_metrics_from_confusion_matrix(
        cm_total,
        class_names=CLASS_NAMES,
    )

    return avg_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_miou: float,
    cfg: Dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_miou": best_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }

    torch.save(checkpoint, path)


def append_train_log(
    log_path: Path,
    row: Dict,
    write_header: bool,
) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_pixel_acc",
        "val_miou",
        "val_mean_class_acc",
        "learning_rate",
        "is_best",
    ]

    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_metrics_json(metrics: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train optical-only U-Net for Wuhan land-cover segmentation."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径，例如 configs/optical_only.yaml",
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)

    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    output_dir = PROJECT_ROOT / cfg["output"]["dir"]
    output_dirs = build_output_dirs(output_dir)

    # 保存一份本次使用的配置
    used_config_path = output_dirs["root"] / "used_config.yaml"
    with open(used_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Train {cfg['experiment']['name']}")
    print("=" * 80)
    print(f"project root : {PROJECT_ROOT}")
    print(f"config       : {config_path}")
    print(f"output dir   : {output_dir}")
    print(f"device       : {device}")
    print(f"seed         : {seed}")
    print("=" * 80)

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    mode = cfg["experiment"]["mode"]

    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"].get("ignore_index", 255))

    learning_rate = float(cfg["train"]["learning_rate"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    epochs = int(cfg["train"]["epochs"])

    criterion = build_criterion(
        cfg=cfg,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_miou = -1.0
    best_epoch = -1

    log_path = output_dirs["logs"] / "train_log.csv"
    best_ckpt_path = output_dirs["checkpoints"] / "best_model.pth"
    last_ckpt_path = output_dirs["checkpoints"] / "last_model.pth"
    best_metrics_path = output_dirs["metrics"] / "best_val_metrics.json"
    last_metrics_path = output_dirs["metrics"] / "last_val_metrics.json"

    # 如果已有旧日志，删除，避免接着写造成混乱
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 80)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mode=mode,
        )

        val_loss, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            mode=mode,
        )

        val_miou = float(val_metrics["miou"])
        is_best = val_miou > best_miou

        if is_best:
            best_miou = val_miou
            best_epoch = epoch

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_miou=best_miou,
                cfg=cfg,
            )

            save_metrics_json(val_metrics, best_metrics_path)

        # 每个 epoch 都保存 last
        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_miou=best_miou,
            cfg=cfg,
        )

        save_metrics_json(val_metrics, last_metrics_path)

        current_lr = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_miou": val_metrics["miou"],
            "val_mean_class_acc": val_metrics["mean_class_acc"],
            "learning_rate": current_lr,
            "is_best": int(is_best),
        }

        append_train_log(
            log_path=log_path,
            row=log_row,
            write_header=(epoch == 1),
        )

        print("\nEpoch 结果:")
        print(f"  train_loss: {train_loss:.4f}")
        print(f"  val_loss  : {val_loss:.4f}")
        print(f"  {format_metrics(val_metrics)}")
        print(f"  is_best   : {is_best}")
        print(f"  best_mIoU : {best_miou:.4f} at epoch {best_epoch}")

    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"best epoch : {best_epoch}")
    print(f"best mIoU  : {best_miou:.4f}")
    print(f"best model : {best_ckpt_path}")
    print(f"last model : {last_ckpt_path}")
    print(f"train log  : {log_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()