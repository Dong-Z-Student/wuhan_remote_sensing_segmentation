from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ml.ml_dataset import (
    CLASS_NAMES,
    compute_metrics_from_confusion_matrix_np,
    ensure_dir,
    extract_valid_pixels_from_patch,
    read_txt_lines,
    save_confusion_matrix_csv,
    save_json,
    save_per_class_metrics_csv,
    update_confusion_matrix_np,
)


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def build_output_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "model": output_dir / "model",
        "metrics": output_dir / "metrics",
        "logs": output_dir / "logs",
    }

    for path in dirs.values():
        ensure_dir(path)

    return dirs


def count_valid_pixels_per_patch(
    relative_paths: List[str],
    project_root: Path,
    feature_indices: List[int],
    label_index: int,
    ignore_label_values: List[int],
    num_classes: int,
) -> Tuple[List[int], int]:
    """
    第一遍扫描 train patch，统计每个 patch 中可用非 snow 像元数量。
    """
    counts = []
    total_valid = 0

    for idx, rel_path in enumerate(relative_paths, start=1):
        patch_path = project_root / rel_path

        X_patch, y_patch = extract_valid_pixels_from_patch(
            patch_path=patch_path,
            feature_indices=feature_indices,
            label_index=label_index,
            ignore_label_values=ignore_label_values,
            num_classes=num_classes,
        )

        n_valid = int(y_patch.shape[0])
        counts.append(n_valid)
        total_valid += n_valid

        if idx % 200 == 0 or idx == len(relative_paths):
            print(
                f"  已统计 {idx}/{len(relative_paths)} 个 patch, "
                f"累计有效像元 {total_valid}"
            )

    return counts, total_valid


def build_random_training_samples_for_kmeans(
    relative_paths: List[str],
    project_root: Path,
    feature_indices: List[int],
    label_index: int,
    ignore_label_values: List[int],
    num_classes: int,
    num_train_pixels: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    从 train split 中随机采样非 snow 像元，用于 K-means 训练和 cluster-class 映射。
    保留 y_train_sample，用于训练后建立 cluster -> class 映射。
    """
    rng = np.random.default_rng(seed)

    print("\n第一遍：统计每个 train patch 中的有效非 snow 像元")
    patch_counts, total_valid_pixels = count_valid_pixels_per_patch(
        relative_paths=relative_paths,
        project_root=project_root,
        feature_indices=feature_indices,
        label_index=label_index,
        ignore_label_values=ignore_label_values,
        num_classes=num_classes,
    )

    if total_valid_pixels == 0:
        raise RuntimeError("train split 中没有有效像元，请检查数据和过滤规则。")

    actual_sample_size = min(int(num_train_pixels), int(total_valid_pixels))

    print("\nK-means 训练采样设置:")
    print(f"  total valid train pixels: {total_valid_pixels}")
    print(f"  requested train pixels  : {num_train_pixels}")
    print(f"  actual sample size      : {actual_sample_size}")

    # 按 patch 有效像元数量比例分配采样量
    target_per_patch = []
    accumulated = 0

    for n_valid in patch_counts:
        if n_valid <= 0:
            target_per_patch.append(0)
            continue

        n_take = int(np.floor(actual_sample_size * n_valid / total_valid_pixels))
        target_per_patch.append(n_take)
        accumulated += n_take

    # 把剩余数量按有效像元数随机分给 patch
    remaining = actual_sample_size - accumulated

    if remaining > 0:
        valid_patch_indices = np.array(
            [i for i, n in enumerate(patch_counts) if n > 0],
            dtype=np.int64,
        )

        patch_weights = np.array(
            [patch_counts[i] for i in valid_patch_indices],
            dtype=np.float64,
        )
        patch_weights = patch_weights / patch_weights.sum()

        extra_indices = rng.choice(
            valid_patch_indices,
            size=remaining,
            replace=True,
            p=patch_weights,
        )

        for patch_idx in extra_indices.tolist():
            target_per_patch[patch_idx] += 1

    X_list = []
    y_list = []

    print("\n第二遍：从 train patch 中随机采样像元")

    for idx, rel_path in enumerate(relative_paths, start=1):
        n_take = target_per_patch[idx - 1]

        if n_take <= 0:
            continue

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

        n_take = min(n_take, X_patch.shape[0])

        take_idx = rng.choice(
            np.arange(X_patch.shape[0]),
            size=n_take,
            replace=False,
        )

        X_list.append(X_patch[take_idx])
        y_list.append(y_patch[take_idx])

        if idx % 100 == 0 or idx == len(relative_paths):
            current_n = sum(x.shape[0] for x in X_list)
            print(
                f"  已处理 {idx}/{len(relative_paths)} 个 patch, "
                f"当前采样像元 {current_n}"
            )

    if len(X_list) == 0:
        raise RuntimeError("没有采样到任何 K-means 训练样本。")

    X_sample = np.concatenate(X_list, axis=0)
    y_sample = np.concatenate(y_list, axis=0)

    if X_sample.shape[0] > actual_sample_size:
        keep_idx = rng.choice(
            np.arange(X_sample.shape[0]),
            size=actual_sample_size,
            replace=False,
        )
        X_sample = X_sample[keep_idx]
        y_sample = y_sample[keep_idx]

    perm = rng.permutation(X_sample.shape[0])
    X_sample = X_sample[perm]
    y_sample = y_sample[perm]

    class_counts = {
        int(class_id): int((y_sample == class_id).sum())
        for class_id in range(num_classes)
    }

    sampling_info = {
        "total_valid_train_pixels": int(total_valid_pixels),
        "requested_train_pixels": int(num_train_pixels),
        "actual_sampled_pixels": int(X_sample.shape[0]),
        "num_features": int(X_sample.shape[1]),
        "sampled_class_counts": {
            str(k): int(v) for k, v in class_counts.items()
        },
    }

    print("\nK-means 训练采样完成:")
    print(f"  X_sample shape: {X_sample.shape}")
    print(f"  y_sample shape: {y_sample.shape}")
    print("  sampled class counts:")
    for class_id in range(num_classes):
        print(
            f"    class {class_id} {CLASS_NAMES[class_id]:20s}: "
            f"{class_counts[class_id]}"
        )

    return X_sample, y_sample, sampling_info


def build_kmeans_model(cfg: Dict):
    model_cfg = cfg["model"]

    model_type = str(model_cfg.get("type", "MiniBatchKMeans"))
    n_clusters = int(model_cfg.get("n_clusters", 8))
    random_state = int(model_cfg.get("random_state", 42))
    n_init = int(model_cfg.get("n_init", 10))
    max_iter = int(model_cfg.get("max_iter", 300))
    verbose = int(model_cfg.get("verbose", 0))

    if model_type == "MiniBatchKMeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=int(model_cfg.get("batch_size", 8192)),
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            reassignment_ratio=float(model_cfg.get("reassignment_ratio", 0.01)),
            verbose=verbose,
        )
        return model

    if model_type == "KMeans":
        model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        return model

    raise ValueError(
        f"不支持的 K-means 类型: {model_type}. "
        f"可选: MiniBatchKMeans, KMeans"
    )


def build_cluster_to_class_mapping(
    clusters: np.ndarray,
    y_true: np.ndarray,
    num_clusters: int,
    num_classes: int,
) -> Tuple[Dict[int, int], np.ndarray]:
    """
    用训练采样数据建立 cluster -> class 映射。
    步骤：
        1. 统计 contingency matrix:
           行为 cluster，列为真实 class
        2. 使用 Hungarian algorithm 寻找最大匹配
        3. 返回 cluster_to_class 字典
    """
    contingency = np.zeros(
        (num_clusters, num_classes),
        dtype=np.int64,
    )

    valid = (clusters >= 0) & (clusters < num_clusters)
    valid &= (y_true >= 0) & (y_true < num_classes)

    clusters = clusters[valid]
    y_true = y_true[valid]

    for c, y in zip(clusters.tolist(), y_true.tolist()):
        contingency[int(c), int(y)] += 1

    # Hungarian 是最小化问题，因此取负号实现最大匹配
    cost = -contingency
    row_ind, col_ind = linear_sum_assignment(cost)

    cluster_to_class = {}

    for cluster_id, class_id in zip(row_ind.tolist(), col_ind.tolist()):
        cluster_to_class[int(cluster_id)] = int(class_id)

    for cluster_id in range(num_clusters):
        if cluster_id not in cluster_to_class:
            # fallback: 映射到该 cluster 中多数类别
            majority_class = int(np.argmax(contingency[cluster_id]))
            cluster_to_class[int(cluster_id)] = majority_class

    return cluster_to_class, contingency


def apply_cluster_mapping(
    clusters: np.ndarray,
    cluster_to_class: Dict[int, int],
) -> np.ndarray:
    """
    将 cluster id 转成 land-cover class id。
    """
    y_pred = np.full(clusters.shape, -1, dtype=np.int64)

    for cluster_id, class_id in cluster_to_class.items():
        y_pred[clusters == int(cluster_id)] = int(class_id)

    return y_pred


def save_mapping_csv(
    cluster_to_class: Dict[int, int],
    contingency: np.ndarray,
    output_path: Path,
) -> None:
    rows = []

    for cluster_id in range(contingency.shape[0]):
        mapped_class = int(cluster_to_class[int(cluster_id)])

        row = {
            "cluster_id": int(cluster_id),
            "mapped_class_id": mapped_class,
            "mapped_class_name": CLASS_NAMES[mapped_class],
            "cluster_total": int(contingency[cluster_id].sum()),
        }

        for class_id in range(contingency.shape[1]):
            row[f"class_{class_id}_{CLASS_NAMES[class_id]}"] = int(
                contingency[cluster_id, class_id]
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def evaluate_kmeans_on_split(
    kmeans_model,
    scaler: StandardScaler | None,
    cluster_to_class: Dict[int, int],
    relative_paths: List[str],
    project_root: Path,
    feature_indices: List[int],
    label_index: int,
    ignore_label_values: List[int],
    num_classes: int,
) -> Dict:
    """
    在 test split 上逐 patch 评价 K-means 聚类结果。
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_valid_pixels = 0

    for idx, rel_path in enumerate(relative_paths, start=1):
        patch_path = project_root / rel_path

        X, y_true = extract_valid_pixels_from_patch(
            patch_path=patch_path,
            feature_indices=feature_indices,
            label_index=label_index,
            ignore_label_values=ignore_label_values,
            num_classes=num_classes,
        )

        if X.shape[0] == 0:
            continue

        if scaler is not None:
            X_input = scaler.transform(X)
        else:
            X_input = X

        clusters = kmeans_model.predict(X_input)
        y_pred = apply_cluster_mapping(
            clusters=clusters,
            cluster_to_class=cluster_to_class,
        )

        cm = update_confusion_matrix_np(
            cm=cm,
            y_true=y_true,
            y_pred=y_pred,
            num_classes=num_classes,
        )

        total_valid_pixels += int(y_true.shape[0])

        if idx % 50 == 0 or idx == len(relative_paths):
            print(
                f"  已评价 {idx}/{len(relative_paths)} 个 patch, "
                f"累计有效像元 {total_valid_pixels}"
            )

    metrics = compute_metrics_from_confusion_matrix_np(
        cm=cm,
        class_names=CLASS_NAMES[:num_classes],
    )

    return metrics


def save_cluster_centers_csv(
    kmeans_model,
    scaler: StandardScaler | None,
    feature_names: List[str],
    output_path: Path,
) -> None:
    """
    保存聚类中心。
    """
    centers_scaled = kmeans_model.cluster_centers_

    df = pd.DataFrame(
        centers_scaled,
        columns=[f"scaled_{name}" for name in feature_names],
    )
    df.insert(0, "cluster_id", np.arange(centers_scaled.shape[0]))

    if scaler is not None:
        centers_original = scaler.inverse_transform(centers_scaled)
        df_original = pd.DataFrame(
            centers_original,
            columns=[f"original_{name}" for name in feature_names],
        )
        df = pd.concat([df, df_original], axis=1)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate K-means for Wuhan pixel-wise land-cover clustering."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径，例如 configs/kmeans.yaml",
    )

    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)

    output_dir = PROJECT_ROOT / cfg["output"]["dir"]
    output_dirs = build_output_dirs(output_dir)

    used_config_path = output_dirs["root"] / "used_config.yaml"
    with open(used_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_list = PROJECT_ROOT / cfg["data"]["train_list"]
    test_list = PROJECT_ROOT / cfg["data"]["test_list"]

    train_paths = read_txt_lines(train_list)
    test_paths = read_txt_lines(test_list)

    num_classes = int(cfg["data"]["num_classes"])
    ignore_label_values = cfg["data"].get("ignore_label_values", [])

    feature_indices = cfg["features"]["feature_indices"]
    label_index = int(cfg["features"]["label_index"])
    feature_names = cfg["features"].get(
        "feature_names",
        [f"band_{idx + 1}" for idx in feature_indices],
    )

    num_train_pixels = int(cfg["sampling"]["num_train_pixels"])
    seed = int(cfg["sampling"].get("seed", 42))
    standardize = bool(cfg.get("preprocess", {}).get("standardize", True))

    print("=" * 80)
    print("Train K-means")
    print("=" * 80)
    print(f"project root : {PROJECT_ROOT}")
    print(f"config       : {config_path}")
    print(f"output dir   : {output_dir}")
    print(f"train patches: {len(train_paths)}")
    print(f"test patches : {len(test_paths)}")
    print(f"num_classes  : {num_classes}")
    print(f"ignore labels: {ignore_label_values}")
    print(f"feature_indices: {feature_indices}")
    print(f"num_train_pixels: {num_train_pixels}")
    print(f"standardize  : {standardize}")
    print("=" * 80)

    # 从 train split 随机采样非 snow 像元
    X_sample, y_sample, sampling_info = build_random_training_samples_for_kmeans(
        relative_paths=train_paths,
        project_root=PROJECT_ROOT,
        feature_indices=feature_indices,
        label_index=label_index,
        ignore_label_values=ignore_label_values,
        num_classes=num_classes,
        num_train_pixels=num_train_pixels,
        seed=seed,
    )

    # 标准化
    scaler = None

    if standardize:
        print("\n开始 StandardScaler 标准化...")
        scaler = StandardScaler()
        X_for_kmeans = scaler.fit_transform(X_sample)
        print("StandardScaler 完成")
    else:
        X_for_kmeans = X_sample

    # 训练 K-means
    print("\n开始训练 K-means...")
    kmeans_model = build_kmeans_model(cfg)
    print(kmeans_model)

    kmeans_model.fit(X_for_kmeans)

    print("K-means 训练完成")

    # 用训练采样数据建立 cluster -> class 映射
    print("\n开始建立 cluster -> class 映射...")
    train_clusters = kmeans_model.predict(X_for_kmeans)

    n_clusters = int(cfg["model"]["n_clusters"])

    cluster_to_class, contingency = build_cluster_to_class_mapping(
        clusters=train_clusters,
        y_true=y_sample,
        num_clusters=n_clusters,
        num_classes=num_classes,
    )

    print("cluster -> class 映射:")
    for cluster_id in sorted(cluster_to_class.keys()):
        class_id = cluster_to_class[cluster_id]
        print(
            f"  cluster {cluster_id} -> "
            f"class {class_id} {CLASS_NAMES[class_id]}"
        )

    # 保存模型、scaler、映射
    model_path = output_dirs["model"] / "kmeans.joblib"
    scaler_path = output_dirs["model"] / "standard_scaler.joblib"
    mapping_json_path = output_dirs["model"] / "cluster_to_class_mapping.json"

    joblib.dump(kmeans_model, model_path)

    if scaler is not None:
        joblib.dump(scaler, scaler_path)

    save_json(
        {str(k): int(v) for k, v in cluster_to_class.items()},
        mapping_json_path,
    )

    # 保存聚类中心和映射表
    cluster_centers_path = output_dirs["metrics"] / "cluster_centers.csv"
    mapping_csv_path = output_dirs["metrics"] / "cluster_mapping_contingency.csv"

    save_cluster_centers_csv(
        kmeans_model=kmeans_model,
        scaler=scaler,
        feature_names=feature_names,
        output_path=cluster_centers_path,
    )

    save_mapping_csv(
        cluster_to_class=cluster_to_class,
        contingency=contingency,
        output_path=mapping_csv_path,
    )

    # 在 test split 上评价
    print("\n开始在 test split 上评价 K-means...")
    test_metrics = evaluate_kmeans_on_split(
        kmeans_model=kmeans_model,
        scaler=scaler,
        cluster_to_class=cluster_to_class,
        relative_paths=test_paths,
        project_root=PROJECT_ROOT,
        feature_indices=feature_indices,
        label_index=label_index,
        ignore_label_values=ignore_label_values,
        num_classes=num_classes,
    )

    # 保存评价结果
    metrics_json_path = output_dirs["metrics"] / "test_metrics.json"
    confusion_csv_path = output_dirs["metrics"] / "confusion_matrix.csv"
    per_class_csv_path = output_dirs["metrics"] / "per_class_metrics.csv"
    sampling_json_path = output_dirs["logs"] / "sampling_info.json"
    run_summary_path = output_dirs["logs"] / "run_summary.json"

    save_json(test_metrics, metrics_json_path)
    save_confusion_matrix_csv(
        cm=np.array(test_metrics["confusion_matrix"], dtype=np.int64),
        class_names=CLASS_NAMES[:num_classes],
        output_path=confusion_csv_path,
    )
    save_per_class_metrics_csv(
        metrics=test_metrics,
        output_path=per_class_csv_path,
    )
    save_json(sampling_info, sampling_json_path)

    run_summary = {
        "experiment": cfg["experiment"],
        "num_classes": num_classes,
        "ignore_label_values": ignore_label_values,
        "feature_indices": feature_indices,
        "feature_names": feature_names,
        "train_patches": len(train_paths),
        "test_patches": len(test_paths),
        "sampling_info": sampling_info,
        "preprocess": {
            "standardize": standardize,
        },
        "model": cfg["model"],
        "cluster_to_class": {
            str(k): int(v) for k, v in cluster_to_class.items()
        },
        "metrics": {
            "pixel_acc": test_metrics["pixel_acc"],
            "miou": test_metrics["miou"],
            "mean_class_acc": test_metrics["mean_class_acc"],
        },
        "outputs": {
            "model": str(model_path),
            "scaler": str(scaler_path) if scaler is not None else None,
            "cluster_to_class_mapping": str(mapping_json_path),
            "test_metrics": str(metrics_json_path),
            "confusion_matrix": str(confusion_csv_path),
            "per_class_metrics": str(per_class_csv_path),
            "cluster_centers": str(cluster_centers_path),
            "cluster_mapping_contingency": str(mapping_csv_path),
            "sampling_info": str(sampling_json_path),
        },
    }

    save_json(run_summary, run_summary_path)

    print("\n" + "=" * 80)
    print("K-means 测试集结果")
    print("=" * 80)
    print(f"pixel_acc      : {test_metrics['pixel_acc']:.4f}")
    print(f"mIoU           : {test_metrics['miou']:.4f}")
    print(f"mean_class_acc : {test_metrics['mean_class_acc']:.4f}")

    print("\nper-class metrics:")
    for item in test_metrics["per_class"]:
        print(
            f"  class {item['class_id']} {item['class_name']:20s}: "
            f"IoU={item['iou']:.4f}, "
            f"Acc={item['accuracy']:.4f}, "
            f"support={item['support']}"
        )

    print("\n输出文件:")
    print(f"  model                    : {model_path}")
    if scaler is not None:
        print(f"  scaler                   : {scaler_path}")
    print(f"  cluster mapping          : {mapping_json_path}")
    print(f"  metrics                  : {metrics_json_path}")
    print(f"  confusion matrix         : {confusion_csv_path}")
    print(f"  per-class metrics        : {per_class_csv_path}")
    print(f"  cluster centers          : {cluster_centers_path}")
    print(f"  mapping contingency      : {mapping_csv_path}")
    print(f"  sampling info            : {sampling_json_path}")
    print(f"  run summary              : {run_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()