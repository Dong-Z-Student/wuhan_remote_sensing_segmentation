from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def fast_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    快速计算混淆矩阵。
    参数：
        preds:[B, H, W]，预测类别，范围应为 0 到 num_classes-1
        targets:[B, H, W]，真实类别。
        num_classes:有效类别数。例如忽略 snow 后，num_classes=8。
        ignore_index:需要忽略的 label 值，例如 255。
    返回：
        cm:[num_classes, num_classes]，行是真实类别，列是预测类别。
    """
    preds = preds.reshape(-1).long()
    targets = targets.reshape(-1).long()

    valid = (targets >= 0) & (targets < num_classes)

    if ignore_index is not None:
        valid = valid & (targets != int(ignore_index))

    preds = preds[valid]
    targets = targets[valid]

    pred_valid = (preds >= 0) & (preds < num_classes)
    preds = preds[pred_valid]
    targets = targets[pred_valid]

    if targets.numel() == 0:
        return torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
            device=preds.device,
        )

    inds = targets * num_classes + preds

    cm = torch.bincount(
        inds,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    return cm


def compute_metrics_from_confusion_matrix(
    cm: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    根据混淆矩阵计算：
    - pixel accuracy
    - per-class IoU
    - mean IoU
    - per-class accuracy
    - mean class accuracy
    """
    cm_np = cm.detach().cpu().numpy().astype(np.float64)

    true_positive = np.diag(cm_np)
    row_sum = cm_np.sum(axis=1)
    col_sum = cm_np.sum(axis=0)

    total = cm_np.sum()
    pixel_acc = true_positive.sum() / total if total > 0 else 0.0

    union = row_sum + col_sum - true_positive

    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=union > 0,
    )

    class_acc = np.divide(
        true_positive,
        row_sum,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=row_sum > 0,
    )

    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0

    mean_class_acc = (
        float(np.nanmean(class_acc))
        if np.any(~np.isnan(class_acc))
        else 0.0
    )

    num_classes = cm_np.shape[0]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    per_class = []

    for i in range(num_classes):
        per_class.append(
            {
                "class_id": i,
                "class_name": class_names[i] if i < len(class_names) else f"class_{i}",
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
                "accuracy": None if np.isnan(class_acc[i]) else float(class_acc[i]),
                "support": int(row_sum[i]),
            }
        )

    return {
        "pixel_acc": float(pixel_acc),
        "miou": float(miou),
        "mean_class_acc": float(mean_class_acc),
        "per_class": per_class,
        "confusion_matrix": cm_np.astype(int).tolist(),
    }


def format_metrics(metrics: Dict) -> str:
    return (
        f"pixel_acc={metrics['pixel_acc']:.4f}, "
        f"mIoU={metrics['miou']:.4f}, "
        f"mean_class_acc={metrics['mean_class_acc']:.4f}"
    )