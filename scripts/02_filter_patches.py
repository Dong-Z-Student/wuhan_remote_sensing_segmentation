"""
功能：
1. 递归读取 data/wuhan/patches/ 下所有 patch
2. 检查 patch 是否为 256×256
3. 检查 patch 是否为 15 个波段
4. 检查 label 波段是否存在 NaN
   - 只要 label 中存在 NaN，就认为该 patch 存在黑边或无效区域，直接剔除
5. 检查 label 是否属于 Dynamic World 0-8 类别
6. 不删除单一类别 patch，只记录 dominant_class 和 dominant_ratio
7. 输出：
   - data/wuhan/meta/valid_patches.txt
   - data/wuhan/meta/filter_report.csv
   - data/wuhan/meta/filter_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio


# 1. 基本配置
EXPECTED_HEIGHT = 256
EXPECTED_WIDTH = 256
EXPECTED_BAND_COUNT = 15

LABEL_BAND_INDEX = 14

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


# 2. 路径与文件工具函数
def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_posix_relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def collect_patch_paths(patch_root: Path) -> List[Path]:
    """
    递归收集所有 .tif patch。
    """
    patch_paths = sorted(patch_root.rglob("*.tif"))
    return patch_paths


# 3. label 检查与统计函数
def get_label_stats(label: np.ndarray) -> Dict:
    """
    统计单个 patch 的 label 信息。
    label 是 float 类型也可以处理：
    - 先判断 NaN
    - 对非 NaN 部分执行 np.rint 后转 int
    """
    total_pixels = label.size

    label_nan_mask = np.isnan(label)
    label_nan_count = int(label_nan_mask.sum())
    label_nan_ratio = label_nan_count / total_pixels if total_pixels > 0 else 0.0

    finite_mask = np.isfinite(label)
    finite_label = label[finite_mask]

    if finite_label.size == 0:
        return {
            "label_nan_count": label_nan_count,
            "label_nan_ratio": label_nan_ratio,
            "label_inf_count": int(np.isinf(label).sum()),
            "label_non_integer_count": 0,
            "label_non_integer_ratio": 0.0,
            "invalid_label_count": 0,
            "invalid_label_ratio": 0.0,
            "dominant_class": None,
            "dominant_class_name": None,
            "dominant_ratio": None,
            "class_counts": {str(k): 0 for k in VALID_LABELS},
        }

    label_rounded = np.rint(finite_label)

    # 检查 label 是否接近整数
    non_integer_mask = ~np.isclose(finite_label, label_rounded, atol=1e-6)
    label_non_integer_count = int(non_integer_mask.sum())
    label_non_integer_ratio = (
        label_non_integer_count / finite_label.size
        if finite_label.size > 0
        else 0.0
    )

    label_int = label_rounded.astype(np.int16)

    valid_label_mask = np.isin(label_int, np.array(VALID_LABELS, dtype=np.int16))
    invalid_label_count = int((~valid_label_mask).sum())
    invalid_label_ratio = (
        invalid_label_count / finite_label.size
        if finite_label.size > 0
        else 0.0
    )

    valid_label_int = label_int[valid_label_mask]

    class_counts = {str(k): 0 for k in VALID_LABELS}

    if valid_label_int.size > 0:
        unique, counts = np.unique(valid_label_int, return_counts=True)
        for class_id, count in zip(unique.tolist(), counts.tolist()):
            class_counts[str(int(class_id))] = int(count)

        dominant_class = int(unique[np.argmax(counts)])
        dominant_count = int(counts.max())
        dominant_ratio = dominant_count / int(valid_label_int.size)
        dominant_class_name = DW_CLASSES.get(dominant_class, "unknown")
    else:
        dominant_class = None
        dominant_class_name = None
        dominant_ratio = None

    return {
        "label_nan_count": label_nan_count,
        "label_nan_ratio": label_nan_ratio,
        "label_inf_count": int(np.isinf(label).sum()),
        "label_non_integer_count": label_non_integer_count,
        "label_non_integer_ratio": label_non_integer_ratio,
        "invalid_label_count": invalid_label_count,
        "invalid_label_ratio": invalid_label_ratio,
        "dominant_class": dominant_class,
        "dominant_class_name": dominant_class_name,
        "dominant_ratio": dominant_ratio,
        "class_counts": class_counts,
    }

def analyze_patch(path: Path, project_root: Path) -> Dict:
    """
    检查单个 patch 是否有效。

    剔除规则：
    1. 尺寸不是 256×256
    2. 波段数不是 15
    3. 输入特征 Band 1-14 中存在任意 NaN / Inf
    4. label Band 15 中存在任意 NaN / Inf
    5. label 有非整数值
    6. label 中存在非 0-8 类别
    """
    record = {
        "path": str(path),
        "relative_path": None,
        "region": path.parent.name,
        "file_name": path.name,
        "is_valid": False,
        "reject_reason": "",
        "width": None,
        "height": None,
        "band_count": None,
        "dtype_label": None,
        "nodata_label": None,

        # 输入特征无效统计
        "feature_nan_count": None,
        "feature_nan_ratio": None,
        "feature_inf_count": None,
        "feature_inf_ratio": None,

        # label 无效统计
        "label_nan_count": None,
        "label_nan_ratio": None,
        "label_inf_count": None,
        "label_non_integer_count": None,
        "label_non_integer_ratio": None,
        "invalid_label_count": None,
        "invalid_label_ratio": None,

        # 类别主导情况
        "dominant_class": None,
        "dominant_class_name": None,
        "dominant_ratio": None,
    }

    for class_id in VALID_LABELS:
        record[f"class_{class_id}_count"] = 0

    try:
        relative_path = to_posix_relative_path(path, project_root)
        record["relative_path"] = relative_path

        with rasterio.open(path) as src:
            record["width"] = int(src.width)
            record["height"] = int(src.height)
            record["band_count"] = int(src.count)

            if src.count >= LABEL_BAND_INDEX + 1:
                record["dtype_label"] = src.dtypes[LABEL_BAND_INDEX]
                record["nodata_label"] = src.nodatavals[LABEL_BAND_INDEX]

            # 检查尺寸
            if src.width != EXPECTED_WIDTH or src.height != EXPECTED_HEIGHT:
                record["reject_reason"] = (
                    f"invalid_size: expected "
                    f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, "
                    f"got {src.width}x{src.height}"
                )
                return record

            # 检查波段数
            if src.count != EXPECTED_BAND_COUNT:
                record["reject_reason"] = (
                    f"invalid_band_count: expected "
                    f"{EXPECTED_BAND_COUNT}, got {src.count}"
                )
                return record

            # 读取所有波段
            arr = src.read(masked=False).astype(np.float32)

        # 输入特征：Band 1-14，Python index 0-13
        features = arr[0:14, :, :]

        # label：Band 15，Python index 14
        label = arr[LABEL_BAND_INDEX, :, :]

        # 检查输入特征是否存在 NaN / Inf
        feature_nan_count = int(np.isnan(features).sum())
        feature_inf_count = int(np.isinf(features).sum())
        feature_total_count = int(features.size)

        record["feature_nan_count"] = feature_nan_count
        record["feature_inf_count"] = feature_inf_count
        record["feature_nan_ratio"] = (
            feature_nan_count / feature_total_count
            if feature_total_count > 0
            else 0.0
        )
        record["feature_inf_ratio"] = (
            feature_inf_count / feature_total_count
            if feature_total_count > 0
            else 0.0
        )

        if feature_nan_count > 0:
            record["reject_reason"] = "features_contain_nan"
            return record

        if feature_inf_count > 0:
            record["reject_reason"] = "features_contain_inf"
            return record

        # 检查 label
        stats = get_label_stats(label)

        record["label_nan_count"] = stats["label_nan_count"]
        record["label_nan_ratio"] = stats["label_nan_ratio"]
        record["label_inf_count"] = stats["label_inf_count"]
        record["label_non_integer_count"] = stats["label_non_integer_count"]
        record["label_non_integer_ratio"] = stats["label_non_integer_ratio"]
        record["invalid_label_count"] = stats["invalid_label_count"]
        record["invalid_label_ratio"] = stats["invalid_label_ratio"]
        record["dominant_class"] = stats["dominant_class"]
        record["dominant_class_name"] = stats["dominant_class_name"]
        record["dominant_ratio"] = stats["dominant_ratio"]

        for class_id in VALID_LABELS:
            record[f"class_{class_id}_count"] = stats["class_counts"][str(class_id)]

        # 只要 label 存在 NaN，就删除该 patch
        if stats["label_nan_count"] > 0:
            record["reject_reason"] = "label_contains_nan"
            return record

        # label 中存在 inf，也删除
        if stats["label_inf_count"] > 0:
            record["reject_reason"] = "label_contains_inf"
            return record

        # label 出现非整数值，删除
        if stats["label_non_integer_count"] > 0:
            record["reject_reason"] = "label_contains_non_integer_values"
            return record

        # label 中存在非 0-8 类别，删除
        if stats["invalid_label_count"] > 0:
            record["reject_reason"] = "label_contains_values_outside_0_8"
            return record

        # 通过全部检查
        record["is_valid"] = True
        record["reject_reason"] = "valid"
        return record

    except Exception as e:
        record["is_valid"] = False
        record["reject_reason"] = f"read_error: {repr(e)}"
        return record


# 4. 输出函数
def save_valid_patches(valid_relative_paths: List[str], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for path in valid_relative_paths:
            f.write(path + "\n")


def save_json(obj: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# 5. 主函数
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Wuhan 256x256 patches by size, band count, and label NaN."
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

    patch_root = project_root / "data" / "wuhan" / "patches"
    meta_dir = project_root / "data" / "wuhan" / "meta"
    ensure_dir(meta_dir)

    valid_txt_path = meta_dir / "valid_patches.txt"
    report_csv_path = meta_dir / "filter_report.csv"
    summary_json_path = meta_dir / "filter_summary.json"

    print("=" * 80)
    print("02_filter_patches.py")
    print("=" * 80)
    print(f"项目根目录: {project_root}")
    print(f"patch 根目录: {patch_root}")
    print(f"输出 meta 目录: {meta_dir}")
    print(f"期望尺寸: {EXPECTED_WIDTH} x {EXPECTED_HEIGHT}")
    print(f"期望波段数: {EXPECTED_BAND_COUNT}")
    print(f"label 波段索引: Band {LABEL_BAND_INDEX + 1}")
    print("过滤规则: 只要输入特征或 label 中存在 NaN / Inf，即删除该 patch")
    print("=" * 80)

    if not patch_root.exists():
        raise FileNotFoundError(f"patch 根目录不存在: {patch_root}")

    patch_paths = collect_patch_paths(patch_root)

    if len(patch_paths) == 0:
        raise FileNotFoundError(f"未在目录中找到 tif 文件: {patch_root}")

    print(f"发现 patch 数量: {len(patch_paths)}")

    records: List[Dict] = []

    for idx, path in enumerate(patch_paths, start=1):
        record = analyze_patch(path, project_root)
        records.append(record)

        if idx % 100 == 0 or idx == len(patch_paths):
            print(f"已检查 {idx}/{len(patch_paths)} 个 patch")

    report_df = pd.DataFrame(records)

    valid_df = report_df[report_df["is_valid"] == True].copy()
    invalid_df = report_df[report_df["is_valid"] == False].copy()

    valid_relative_paths = valid_df["relative_path"].tolist()

    save_valid_patches(valid_relative_paths, valid_txt_path)
    report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")

    # 按 reject_reason 汇总
    reason_counts = (
        report_df["reject_reason"]
        .value_counts(dropna=False)
        .to_dict()
    )

    # 汇总有效 patch 的类别像元数量
    valid_class_counts = {}
    for class_id in VALID_LABELS:
        col = f"class_{class_id}_count"
        valid_class_counts[str(class_id)] = int(valid_df[col].sum()) if col in valid_df else 0

    total_valid_label_pixels = int(sum(valid_class_counts.values()))

    valid_class_percentages = {}
    for class_id in VALID_LABELS:
        count = valid_class_counts[str(class_id)]
        if total_valid_label_pixels > 0:
            valid_class_percentages[str(class_id)] = count / total_valid_label_pixels * 100
        else:
            valid_class_percentages[str(class_id)] = 0.0

    summary = {
        "patch_root": str(patch_root),
        "total_patches": int(len(report_df)),
        "valid_patches": int(len(valid_df)),
        "invalid_patches": int(len(invalid_df)),
        "valid_ratio": float(len(valid_df) / len(report_df)) if len(report_df) > 0 else 0.0,
        "reject_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "expected_width": EXPECTED_WIDTH,
        "expected_height": EXPECTED_HEIGHT,
        "expected_band_count": EXPECTED_BAND_COUNT,
        "label_band": LABEL_BAND_INDEX + 1,
        "filter_rule": "A patch is rejected if any input feature band or label band contains NaN/Inf.",
        "class_names": {str(k): v for k, v in DW_CLASSES.items()},
        "valid_patch_class_pixel_counts": valid_class_counts,
        "valid_patch_class_percentages": valid_class_percentages,
    }

    save_json(summary, summary_json_path)

    print("\n" + "=" * 80)
    print("过滤结果")
    print("=" * 80)
    print(f"总 patch 数量: {len(report_df)}")
    print(f"有效 patch 数量: {len(valid_df)}")
    print(f"无效 patch 数量: {len(invalid_df)}")
    print(f"有效比例: {len(valid_df) / len(report_df) * 100:.2f}%")

    print("\n拒绝原因统计:")
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")

    print("\n有效 patch 类别统计:")
    for class_id in VALID_LABELS:
        count = valid_class_counts[str(class_id)]
        percent = valid_class_percentages[str(class_id)]
        class_name = DW_CLASSES[class_id]
        print(f"  {class_id} {class_name:20s}: {count:12d}  {percent:8.4f}%")

    print("\n" + "=" * 80)
    print("输出文件")
    print("=" * 80)
    print(f"  {valid_txt_path}")
    print(f"  {report_csv_path}")
    print(f"  {summary_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()