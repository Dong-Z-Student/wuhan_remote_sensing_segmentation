from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile


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


def read_txt_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_patch_array(path: Path) -> np.ndarray:
    """
    使用 tifffile 读取 patch，并统一为 [bands, height, width]。
    """
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"patch 维度异常: {path}, shape={arr.shape}")

    if arr.shape[0] == 15:
        return arr
    if arr.shape[-1] == 15:
        return np.transpose(arr, (2, 0, 1))

    raise ValueError(
        f"无法判断波段维度位置: {path}, shape={arr.shape}. "
        f"期望 [15,H,W] 或 [H,W,15]."
    )


def extract_valid_pixels_from_patch(
    patch_path: Path,
    feature_indices: List[int],
    label_index: int,
    ignore_label_values: List[int],
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从单个 patch 中提取有效像元特征和标签。

    返回:
        X: [n_pixels, n_features]
        y: [n_pixels]
    """
    arr = read_patch_array(patch_path)

    features = arr[feature_indices, :, :]
    label = arr[label_index, :, :]

    # [C,H,W] -> [H,W,C] -> [N,C]
    X = np.transpose(features, (1, 2, 0)).reshape(-1, len(feature_indices))
    y = np.rint(label.reshape(-1)).astype(np.int64)

    valid = np.ones(y.shape[0], dtype=bool)

    # 过滤 NaN / Inf 特征、label
    valid &= np.all(np.isfinite(X), axis=1)
    valid &= np.isfinite(label.reshape(-1))

    # 过滤 ignore 类别
    for ignore_value in ignore_label_values:
        valid &= y != int(ignore_value)

    # 只保留 0 到 num_classes-1
    valid &= y >= 0
    valid &= y < num_classes

    X = X[valid]
    y = y[valid]

    return X, y


def count_labels_in_paths(
    relative_paths: List[str],
    project_root: Path,
    label_index: int,
    ignore_label_values: List[int],
    num_classes: int,
) -> Dict[int, int]:
    """
    统计一组 patch 中各类别有效像元数量。
    """
    counts = {i: 0 for i in range(num_classes)}

    for idx, rel_path in enumerate(relative_paths, start=1):
        patch_path = project_root / rel_path
        arr = read_patch_array(patch_path)
        label = np.rint(arr[label_index].reshape(-1)).astype(np.int64)

        valid = np.ones(label.shape[0], dtype=bool)

        for ignore_value in ignore_label_values:
            valid &= label != int(ignore_value)

        valid &= label >= 0
        valid &= label < num_classes

        label = label[valid]

        unique, nums = np.unique(label, return_counts=True)
        for class_id, n in zip(unique.tolist(), nums.tolist()):
            counts[int(class_id)] += int(n)

        if idx % 200 == 0 or idx == len(relative_paths):
            print(f"  已统计类别分布 {idx}/{len(relative_paths)} 个 patch")

    return counts


def build_balanced_training_samples(
    relative_paths: List[str],
    project_root: Path,
    feature_indices: List[int],
    label_index: int,
    ignore_label_values: List[int],
    num_classes: int,
    max_pixels_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    构建随机森林训练样本。
    返回:
        X_train: [n_samples, n_features]
        y_train: [n_samples]
        sampling_info: 采样统计信息
    """
    rng = np.random.default_rng(seed)

    print("\n第一遍：统计训练集中各类别像元数量")
    class_counts = count_labels_in_paths(
        relative_paths=relative_paths,
        project_root=project_root,
        label_index=label_index,
        ignore_label_values=ignore_label_values,
        num_classes=num_classes,
    )

    target_per_class = {
        class_id: min(int(count), int(max_pixels_per_class))
        for class_id, count in class_counts.items()
    }

    print("\n训练集类别像元数量与目标采样量:")
    for class_id in range(num_classes):
        print(
            f"  class {class_id} {CLASS_NAMES[class_id]:20s}: "
            f"count={class_counts[class_id]}, "
            f"target={target_per_class[class_id]}"
        )

    samples_by_class = {i: [] for i in range(num_classes)}
    sampled_counts = {i: 0 for i in range(num_classes)}

    print("\n第二遍：按类别均衡采样训练像元")

    for idx, rel_path in enumerate(relative_paths, start=1):
        patch_path = project_root / rel_path

        X_patch, y_patch = extract_valid_pixels_from_patch(
            patch_path=patch_path,
            feature_indices=feature_indices,
            label_index=label_index,
            ignore_label_values=ignore_label_values,
            num_classes=num_classes,
        )

        if X_patch.shape[0] == 0:
            continue

        for class_id in range(num_classes):
            target = target_per_class[class_id]
            total_count = class_counts[class_id]

            if target <= 0 or total_count <= 0:
                continue

            remaining_need = target - sampled_counts[class_id]
            if remaining_need <= 0:
                continue

            class_idx = np.where(y_patch == class_id)[0]
            n_class_patch = class_idx.size

            if n_class_patch == 0:
                continue

            # 某类总数小于目标数，直接全部使用
            if total_count <= max_pixels_per_class:
                take_idx = class_idx
            else:
                # 按当前 patch 中该类像元占全局该类的比例分配采样量
                expected_take = int(
                    np.ceil(target * n_class_patch / total_count)
                )
                expected_take = max(1, expected_take)
                expected_take = min(expected_take, n_class_patch, remaining_need)

                take_idx = rng.choice(
                    class_idx,
                    size=expected_take,
                    replace=False,
                )

            samples_by_class[class_id].append(X_patch[take_idx])
            sampled_counts[class_id] += int(take_idx.size)

        if idx % 100 == 0 or idx == len(relative_paths):
            print(f"  已采样 {idx}/{len(relative_paths)} 个 patch")

    X_list = []
    y_list = []

    for class_id in range(num_classes):
        if len(samples_by_class[class_id]) == 0:
            continue

        X_c = np.concatenate(samples_by_class[class_id], axis=0)

        # 防止因为 ceil 导致超过目标量
        target = target_per_class[class_id]
        if X_c.shape[0] > target:
            keep_idx = rng.choice(
                np.arange(X_c.shape[0]),
                size=target,
                replace=False,
            )
            X_c = X_c[keep_idx]

        y_c = np.full(X_c.shape[0], class_id, dtype=np.int64)

        X_list.append(X_c)
        y_list.append(y_c)

    if len(X_list) == 0:
        raise RuntimeError("没有采样到任何训练样本，请检查数据路径和过滤规则。")

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    # 整体打乱
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    final_counts = {
        int(class_id): int((y_train == class_id).sum())
        for class_id in range(num_classes)
    }

    sampling_info = {
        "class_pixel_counts_in_train_split": {
            str(k): int(v) for k, v in class_counts.items()
        },
        "target_pixels_per_class": {
            str(k): int(v) for k, v in target_per_class.items()
        },
        "actual_sampled_pixels_per_class": {
            str(k): int(v) for k, v in final_counts.items()
        },
        "total_sampled_pixels": int(X_train.shape[0]),
        "num_features": int(X_train.shape[1]),
    }

    print("\n最终采样结果:")
    for class_id in range(num_classes):
        print(
            f"  class {class_id} {CLASS_NAMES[class_id]:20s}: "
            f"{final_counts[class_id]}"
        )
    print(f"  total samples: {X_train.shape[0]}")

    return X_train, y_train, sampling_info


def update_confusion_matrix_np(
    cm: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    累加 numpy 混淆矩阵。
    行是真实类别，列是预测类别。
    """
    valid = (y_true >= 0) & (y_true < num_classes)
    valid &= (y_pred >= 0) & (y_pred < num_classes)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    inds = y_true * num_classes + y_pred
    bincount = np.bincount(
        inds,
        minlength=num_classes ** 2,
    )

    cm += bincount.reshape(num_classes, num_classes)
    return cm


def compute_metrics_from_confusion_matrix_np(
    cm: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    根据混淆矩阵计算 pixel_acc、mIoU、mean_class_acc、per-class metrics。
    """
    cm = cm.astype(np.float64)

    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    total = cm.sum()

    pixel_acc = float(tp.sum() / total) if total > 0 else 0.0

    union = row_sum + col_sum - tp

    iou = np.divide(
        tp,
        union,
        out=np.full_like(tp, np.nan, dtype=np.float64),
        where=union > 0,
    )

    class_acc = np.divide(
        tp,
        row_sum,
        out=np.full_like(tp, np.nan, dtype=np.float64),
        where=row_sum > 0,
    )

    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
    mean_class_acc = (
        float(np.nanmean(class_acc))
        if np.any(~np.isnan(class_acc))
        else 0.0
    )

    per_class = []
    for class_id in range(cm.shape[0]):
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "iou": None if np.isnan(iou[class_id]) else float(iou[class_id]),
                "accuracy": None if np.isnan(class_acc[class_id]) else float(class_acc[class_id]),
                "support": int(row_sum[class_id]),
            }
        )

    return {
        "pixel_acc": pixel_acc,
        "miou": miou,
        "mean_class_acc": mean_class_acc,
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def save_confusion_matrix_csv(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    df = pd.DataFrame(
        cm,
        index=[f"true_{i}_{name}" for i, name in enumerate(class_names)],
        columns=[f"pred_{i}_{name}" for i, name in enumerate(class_names)],
    )
    df.to_csv(output_path, encoding="utf-8-sig")


def save_per_class_metrics_csv(
    metrics: Dict,
    output_path: Path,
) -> None:
    df = pd.DataFrame(metrics["per_class"])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")