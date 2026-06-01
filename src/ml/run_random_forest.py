from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ml.ml_dataset import (
    CLASS_NAMES,
    build_balanced_training_samples,
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


def build_random_forest(cfg: Dict) -> RandomForestClassifier:
    model_cfg = cfg["model"]

    max_depth = model_cfg.get("max_depth", None)
    if max_depth in ["null", "None"]:
        max_depth = None

    clf = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=max_depth,
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
        max_features=model_cfg.get("max_features", "sqrt"),
        class_weight=model_cfg.get("class_weight", "balanced_subsample"),
        n_jobs=int(model_cfg.get("n_jobs", -1)),
        random_state=int(model_cfg.get("random_state", 42)),
        verbose=int(model_cfg.get("verbose", 0)),
    )

    return clf


def evaluate_random_forest_on_split(
    clf: RandomForestClassifier,
    relative_paths,
    project_root: Path,
    feature_indices,
    label_index: int,
    ignore_label_values,
    num_classes: int,
) -> Dict:
    """
    在 test split 上逐 patch 评价，不一次性把全部 test 像元放进内存。
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

        y_pred = clf.predict(X)

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


def save_feature_importance(
    clf: RandomForestClassifier,
    feature_names,
    output_path: Path,
) -> None:
    import pandas as pd

    importance = clf.feature_importances_

    df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest for Wuhan pixel-wise land-cover classification."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径，例如 configs/random_forest.yaml",
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

    max_pixels_per_class = int(cfg["sampling"]["max_pixels_per_class"])
    seed = int(cfg["sampling"].get("seed", 42))

    print("=" * 80)
    print("Train Random Forest")
    print("=" * 80)
    print(f"project root : {PROJECT_ROOT}")
    print(f"config       : {config_path}")
    print(f"output dir   : {output_dir}")
    print(f"train patches: {len(train_paths)}")
    print(f"test patches : {len(test_paths)}")
    print(f"num_classes  : {num_classes}")
    print(f"ignore labels: {ignore_label_values}")
    print(f"feature_indices: {feature_indices}")
    print(f"max_pixels_per_class: {max_pixels_per_class}")
    print("=" * 80)

    # 构建训练样本
    X_train, y_train, sampling_info = build_balanced_training_samples(
        relative_paths=train_paths,
        project_root=PROJECT_ROOT,
        feature_indices=feature_indices,
        label_index=label_index,
        ignore_label_values=ignore_label_values,
        num_classes=num_classes,
        max_pixels_per_class=max_pixels_per_class,
        seed=seed,
    )

    print("\n训练样本矩阵:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X dtype      : {X_train.dtype}")
    print(f"  y dtype      : {y_train.dtype}")

    # 训练随机森林
    clf = build_random_forest(cfg)

    print("\n开始训练 Random Forest...")
    print(clf)

    clf.fit(X_train, y_train)

    print("Random Forest 训练完成")

    # 保存模型
    model_path = output_dirs["model"] / "random_forest.joblib"
    joblib.dump(clf, model_path)

    print(f"模型已保存: {model_path}")

    # 保存特征重要性
    feature_importance_path = output_dirs["metrics"] / "feature_importance.csv"
    save_feature_importance(
        clf=clf,
        feature_names=feature_names,
        output_path=feature_importance_path,
    )

    # 在 test split 上评价
    print("\n开始在 test split 上评价...")
    test_metrics = evaluate_random_forest_on_split(
        clf=clf,
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
        "metrics": {
            "pixel_acc": test_metrics["pixel_acc"],
            "miou": test_metrics["miou"],
            "mean_class_acc": test_metrics["mean_class_acc"],
        },
        "outputs": {
            "model": str(model_path),
            "test_metrics": str(metrics_json_path),
            "confusion_matrix": str(confusion_csv_path),
            "per_class_metrics": str(per_class_csv_path),
            "feature_importance": str(feature_importance_path),
            "sampling_info": str(sampling_json_path),
        },
    }

    save_json(run_summary, run_summary_path)

    print("\n" + "=" * 80)
    print("Random Forest 测试集结果")
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
    print(f"  model             : {model_path}")
    print(f"  metrics           : {metrics_json_path}")
    print(f"  confusion matrix  : {confusion_csv_path}")
    print(f"  per-class metrics : {per_class_csv_path}")
    print(f"  feature importance: {feature_importance_path}")
    print(f"  sampling info     : {sampling_json_path}")
    print(f"  run summary       : {run_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()