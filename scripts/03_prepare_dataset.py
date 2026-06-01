"""
功能：
1. 读取 data/wuhan/meta/valid_patches.txt
2. 按 region 分层随机划分 train / val / test
3. 划分比例为 8 : 1 : 1
4. 固定随机种子 42
5. 输出：
   - data/wuhan/splits/train.txt
   - data/wuhan/splits/val.txt
   - data/wuhan/splits/test.txt
   - data/wuhan/meta/split_summary.json
   - data/wuhan/meta/split_class_distribution.csv
   - data/wuhan/meta/split_region_distribution.csv
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio


# 1. 基本配置
SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

EXPECTED_SUM = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

LABEL_BAND_INDEX = 14  # Python 0-based index, Band 15
VALID_LABELS = list(range(9))

DW_CLASSES: Dict[int, str] = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built_area",
    7: "bare_ground",
    8: "snow_and_ice",
}


# 2. 路径函数
def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_txt_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def write_txt_lines(lines: List[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# 3. split 工具函数
def get_region_name(relative_path: str) -> str:
    """
    从路径中提取 region 名称。
    """
    return Path(relative_path).parent.name


def group_paths_by_region(paths: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)

    for path in paths:
        region = get_region_name(path)
        groups[region].append(path)

    return dict(groups)


def split_one_region(
    paths: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    对单个 region 内部的 patch 做随机划分。
    """
    paths = list(paths)

    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]

    assert len(train_paths) + len(val_paths) + len(test_paths) == n
    assert len(test_paths) == n_test

    return train_paths, val_paths, test_paths


def stratified_split_by_region(
    valid_paths: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
    """
    按 region 分层随机划分。
    """
    groups = group_paths_by_region(valid_paths)

    all_train = []
    all_val = []
    all_test = []

    region_rows = []

    for region_name, region_paths in sorted(groups.items()):
        # 每个 region 使用不同 seed，避免四个 region 内部排序完全相同
        region_seed = seed + abs(hash(region_name)) % 100000

        train_paths, val_paths, test_paths = split_one_region(
            paths=region_paths,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=region_seed,
        )

        all_train.extend(train_paths)
        all_val.extend(val_paths)
        all_test.extend(test_paths)

        region_rows.append(
            {
                "region": region_name,
                "total": len(region_paths),
                "train": len(train_paths),
                "val": len(val_paths),
                "test": len(test_paths),
                "train_ratio": len(train_paths) / len(region_paths) if region_paths else 0,
                "val_ratio": len(val_paths) / len(region_paths) if region_paths else 0,
                "test_ratio": len(test_paths) / len(region_paths) if region_paths else 0,
            }
        )

    # 整体打乱一次
    rng = random.Random(seed)
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    region_df = pd.DataFrame(region_rows)

    return all_train, all_val, all_test, region_df


# 4. 类别分布统计
def init_counts() -> Dict[int, int]:
    return {class_id: 0 for class_id in VALID_LABELS}


def count_labels_in_patch(patch_path: Path) -> Dict[int, int]:
    """
    统计单个 patch 中 label 类别数量。
    """
    counts = init_counts()

    with rasterio.open(patch_path) as src:
        label = src.read(LABEL_BAND_INDEX + 1, masked=False).astype(np.float32)

    finite_mask = np.isfinite(label)
    label = label[finite_mask]

    if label.size == 0:
        return counts

    label_int = np.rint(label).astype(np.int16)
    valid_mask = np.isin(label_int, np.array(VALID_LABELS, dtype=np.int16))
    label_int = label_int[valid_mask]

    if label_int.size == 0:
        return counts

    unique, nums = np.unique(label_int, return_counts=True)

    for class_id, n in zip(unique.tolist(), nums.tolist()):
        counts[int(class_id)] += int(n)

    return counts


def count_labels_in_split(
    relative_paths: List[str],
    project_root: Path,
    split_name: str,
) -> Dict[int, int]:
    """
    统计一个 split 中所有 patch 的类别数量。
    """
    split_counts = init_counts()

    total = len(relative_paths)

    for idx, rel_path in enumerate(relative_paths, start=1):
        patch_path = project_root / rel_path
        patch_counts = count_labels_in_patch(patch_path)

        for class_id in VALID_LABELS:
            split_counts[class_id] += patch_counts[class_id]

        if idx % 200 == 0 or idx == total:
            print(f"  [{split_name}] 已统计 {idx}/{total} 个 patch")

    return split_counts


def build_class_distribution_dataframe(
    split_counts_dict: Dict[str, Dict[int, int]]
) -> pd.DataFrame:
    """
    构建 train / val / test 类别分布表。
    """
    rows = []

    for split_name, counts in split_counts_dict.items():
        total_pixels = int(sum(counts.values()))

        for class_id in VALID_LABELS:
            pixel_count = int(counts[class_id])
            percentage = pixel_count / total_pixels * 100 if total_pixels > 0 else 0.0

            rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": DW_CLASSES[class_id],
                    "pixel_count": pixel_count,
                    "percentage": percentage,
                    "present": pixel_count > 0,
                    "total_pixels_in_split": total_pixels,
                }
            )

    return pd.DataFrame(rows)


# 5. 主函数
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test split for Wuhan SAR-optical patches."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="项目根目录。默认使用脚本所在目录的上一级。"
    )

    args = parser.parse_args()

    if args.project_root is None:
        project_root = get_default_project_root()
    else:
        project_root = Path(args.project_root).resolve()

    if not np.isclose(EXPECTED_SUM, 1.0):
        raise ValueError(
            f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO 必须为 1，当前为 {EXPECTED_SUM}"
        )

    meta_dir = project_root / "data" / "wuhan" / "meta"
    splits_dir = project_root / "data" / "wuhan" / "splits"
    ensure_dir(meta_dir)
    ensure_dir(splits_dir)

    valid_txt_path = meta_dir / "valid_patches.txt"

    train_txt_path = splits_dir / "train.txt"
    val_txt_path = splits_dir / "val.txt"
    test_txt_path = splits_dir / "test.txt"

    split_summary_path = meta_dir / "split_summary.json"
    split_class_csv_path = meta_dir / "split_class_distribution.csv"
    split_region_csv_path = meta_dir / "split_region_distribution.csv"

    print("=" * 80)
    print("03_prepare_dataset.py")
    print("=" * 80)
    print(f"项目根目录: {project_root}")
    print(f"有效 patch 清单: {valid_txt_path}")
    print(f"划分比例: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"随机种子: {SEED}")
    print("划分方式: 按 region 分层随机划分")
    print("=" * 80)

    if not valid_txt_path.exists():
        raise FileNotFoundError(f"valid_patches.txt 不存在: {valid_txt_path}")

    valid_paths = read_txt_lines(valid_txt_path)

    if len(valid_paths) == 0:
        raise ValueError(f"valid_patches.txt 为空: {valid_txt_path}")

    print(f"有效 patch 数量: {len(valid_paths)}")

    # 检查文件是否都存在
    missing_paths = [p for p in valid_paths if not (project_root / p).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "valid_patches.txt 中存在找不到的 patch，例如：\n"
            + "\n".join(missing_paths[:10])
        )

    # 划分
    train_paths, val_paths, test_paths, region_df = stratified_split_by_region(
        valid_paths=valid_paths,
        seed=SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # 保存 split txt
    write_txt_lines(train_paths, train_txt_path)
    write_txt_lines(val_paths, val_txt_path)
    write_txt_lines(test_paths, test_txt_path)

    # 保存 region 分布
    region_df.to_csv(split_region_csv_path, index=False, encoding="utf-8-sig")

    print("\n划分完成：")
    print(f"  train: {len(train_paths)}")
    print(f"  val  : {len(val_paths)}")
    print(f"  test : {len(test_paths)}")

    print("\n按 region 划分统计：")
    print(region_df.to_string(index=False))

    # 统计各 split 类别分布
    print("\n开始统计 train / val / test 类别分布...")
    train_counts = count_labels_in_split(train_paths, project_root, "train")
    val_counts = count_labels_in_split(val_paths, project_root, "val")
    test_counts = count_labels_in_split(test_paths, project_root, "test")

    split_counts_dict = {
        "train": train_counts,
        "val": val_counts,
        "test": test_counts,
    }

    class_df = build_class_distribution_dataframe(split_counts_dict)
    class_df.to_csv(split_class_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "seed": SEED,
        "split_method": "stratified_random_split_by_region",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "total_valid_patches": len(valid_paths),
        "train_patches": len(train_paths),
        "val_patches": len(val_paths),
        "test_patches": len(test_paths),
        "num_classes": 9,
        "class_names": {str(k): v for k, v in DW_CLASSES.items()},
        "label_band_index_python": LABEL_BAND_INDEX,
        "label_band_number_raster": LABEL_BAND_INDEX + 1,
        "splits": {
            "train": str(train_txt_path),
            "val": str(val_txt_path),
            "test": str(test_txt_path),
        },
        "outputs": {
            "split_class_distribution": str(split_class_csv_path),
            "split_region_distribution": str(split_region_csv_path),
        },
        "class_pixel_counts": {
            split_name: {str(k): int(v) for k, v in counts.items()}
            for split_name, counts in split_counts_dict.items()
        },
    }

    save_json(summary, split_summary_path)

    print("\n类别分布统计：")
    print(class_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("输出文件")
    print("=" * 80)
    print(f"  {train_txt_path}")
    print(f"  {val_txt_path}")
    print(f"  {test_txt_path}")
    print(f"  {split_summary_path}")
    print(f"  {split_class_csv_path}")
    print(f"  {split_region_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()