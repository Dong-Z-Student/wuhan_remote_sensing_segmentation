"""
功能：
1. 读取武汉区域 4 个未切片的完整 GeoTIFF 影像
2. 读取最后一个波段 label，也就是第 15 个波段
3. 统计 Dynamic World 0-8 类别的像元数量和占比
4. 找出武汉区域中像元数量为 0 的类别
5. 输出：
   - label_stats_full_regions.csv
   - label_stats_by_region.csv
   - label_stats_full_regions.json
   - ignore_classes.json
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
RAW_REGION_FILENAMES = [
    "Wuhan_S1S2_DW_stack_20251001-0000000000-0000000000-002.tif",
    "Wuhan_S1S2_DW_stack_20251001-0000000000-0000008704.tif",
    "Wuhan_S1S2_DW_stack_20251001-0000008704-0000000000-004.tif",
    "Wuhan_S1S2_DW_stack_20251001-0000008704-0000008704.tif",
]


LABEL_BAND = 15
FEATURE_BANDS = list(range(1, 15))

# Dynamic World 类别体系
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

#避免黑边 label=0 被误统计为 water
USE_FEATURE_ZERO_MASK = True
# 判断输入特征是否接近 0 的阈值
ZERO_ATOL = 1e-8

# 2. 工具函数
def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_count_dict() -> Dict[int, int]:
    return {class_id: 0 for class_id in DW_CLASSES.keys()}


def update_counts(counts: Dict[int, int], label_values: np.ndarray) -> int:
    """
    更新类别计数。

    参数：
        counts: 类别计数字典
        label_values: 一维 label 数组，已经过有效像元过滤

    返回：
        unknown_count: 不属于 0-8 的像元数量
    """
    if label_values.size == 0:
        return 0

    # label 以 float 保存，四舍五入为整数类别
    label_int = np.rint(label_values).astype(np.int16)

    class_ids = np.array(list(DW_CLASSES.keys()), dtype=np.int16)
    known_mask = np.isin(label_int, class_ids)

    known_labels = label_int[known_mask]
    unknown_count = int((~known_mask).sum())

    unique, cnt = np.unique(known_labels, return_counts=True)

    for class_id, n in zip(unique.tolist(), cnt.tolist()):
        counts[int(class_id)] += int(n)

    return unknown_count


def build_valid_mask(
    src: rasterio.DatasetReader,
    window,
    label_arr: np.ndarray,
    nodata_value,
) -> np.ndarray:
    """
    构建有效像元掩膜。

    过滤规则：
    1. label 不是 NaN / inf
    2. label 不是 raster 自带 nodata
    """
    valid = np.isfinite(label_arr)

    if nodata_value is not None:
        if isinstance(nodata_value, float) and np.isnan(nodata_value):
            valid &= ~np.isnan(label_arr)
        else:
            valid &= label_arr != nodata_value

    if USE_FEATURE_ZERO_MASK:
        feature_arr = src.read(FEATURE_BANDS, window=window, masked=False)

        # shape: [14, h, w]
        feature_arr = feature_arr.astype(np.float32, copy=False)
        finite_feature = np.isfinite(feature_arr)

        all_not_finite = ~np.any(finite_feature, axis=0)
        feature_filled = np.where(finite_feature, feature_arr, 0.0)
        all_zero = np.all(np.isclose(feature_filled, 0.0, atol=ZERO_ATOL), axis=0)

        # 黑边或无效区域
        invalid_by_feature = all_not_finite | all_zero

        valid &= ~invalid_by_feature

    return valid


def analyze_one_region(region_path: Path) -> Tuple[Dict[int, int], int, int]:
    """
    统计单个完整影像中的 label 类别。

    返回：
        region_counts: 0-8 类别计数
        valid_pixel_count: 有效 label 像元数量
        unknown_pixel_count: 不属于 0-8 的像元数量
    """
    region_counts = init_count_dict()
    valid_pixel_count = 0
    unknown_pixel_count = 0

    print(f"\n正在读取: {region_path.name}")

    with rasterio.open(region_path) as src:
        if src.count < LABEL_BAND:
            raise ValueError(
                f"{region_path.name} 的波段数为 {src.count}，不足以读取第 {LABEL_BAND} 个 label 波段。"
            )

        print(f"  波段数: {src.count}")
        print(f"  尺寸: {src.width} x {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  label band dtype: {src.dtypes[LABEL_BAND - 1]}")
        print(f"  label band nodata: {src.nodatavals[LABEL_BAND - 1]}")

        nodata_value = src.nodatavals[LABEL_BAND - 1]

        # 使用 label 波段的 block window 分块读取，避免一次性读入大影像
        for _, window in src.block_windows(LABEL_BAND):
            label_arr = src.read(LABEL_BAND, window=window, masked=False)

            valid_mask = build_valid_mask(
                src=src,
                window=window,
                label_arr=label_arr,
                nodata_value=nodata_value,
            )

            valid_labels = label_arr[valid_mask]

            valid_pixel_count += int(valid_labels.size)
            unknown_pixel_count += update_counts(region_counts, valid_labels)

    return region_counts, valid_pixel_count, unknown_pixel_count


def counts_to_dataframe(
    counts: Dict[int, int],
    total_valid_pixels: int,
) -> pd.DataFrame:
    """
    将类别计数字典转换为 DataFrame。
    """
    rows = []

    for class_id, class_name in DW_CLASSES.items():
        pixel_count = int(counts.get(class_id, 0))

        if total_valid_pixels > 0:
            percentage = pixel_count / total_valid_pixels * 100
        else:
            percentage = 0.0

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixel_count,
                "percentage": percentage,
                "present": pixel_count > 0,
            }
        )

    return pd.DataFrame(rows)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# 3. 主函数
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Dynamic World label distribution in Wuhan full-region GeoTIFFs."
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

    raw_dir = project_root / "data" / "wuhan" / "raw_regions"
    meta_dir = project_root / "data" / "wuhan" / "meta"
    ensure_dir(meta_dir)

    print("=" * 80)
    print("01_analyze_labels.py")
    print("=" * 80)
    print(f"项目根目录: {project_root}")
    print(f"原始影像目录: {raw_dir}")
    print(f"输出 meta 目录: {meta_dir}")
    print(f"label 波段: Band {LABEL_BAND}")
    print(f"是否启用输入特征全 0 黑边过滤: {USE_FEATURE_ZERO_MASK}")
    print("=" * 80)

    region_paths: List[Path] = [raw_dir / name for name in RAW_REGION_FILENAMES]

    missing_files = [str(path) for path in region_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "以下原始影像不存在，请检查文件名或路径：\n" + "\n".join(missing_files)
        )

    total_counts = init_count_dict()
    total_valid_pixels = 0
    total_unknown_pixels = 0

    by_region_rows = []

    for region_path in region_paths:
        region_counts, valid_pixels, unknown_pixels = analyze_one_region(region_path)

        total_valid_pixels += valid_pixels
        total_unknown_pixels += unknown_pixels

        for class_id in DW_CLASSES.keys():
            total_counts[class_id] += region_counts[class_id]

        region_df = counts_to_dataframe(region_counts, valid_pixels)

        for _, row in region_df.iterrows():
            by_region_rows.append(
                {
                    "region_name": region_path.name,
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "pixel_count": int(row["pixel_count"]),
                    "percentage": float(row["percentage"]),
                    "present": bool(row["present"]),
                    "valid_pixel_count_in_region": int(valid_pixels),
                    "unknown_pixel_count_in_region": int(unknown_pixels),
                }
            )

        print(f"  有效 label 像元数: {valid_pixels}")
        print(f"  非 0-8 类别像元数: {unknown_pixels}")

    # 总体统计
    full_df = counts_to_dataframe(total_counts, total_valid_pixels)
    by_region_df = pd.DataFrame(by_region_rows)

    missing_classes = [
        int(class_id)
        for class_id, count in total_counts.items()
        if count == 0
    ]

    present_classes = [
        int(class_id)
        for class_id, count in total_counts.items()
        if count > 0
    ]

    # 输出 CSV
    full_csv_path = meta_dir / "label_stats_full_regions.csv"
    by_region_csv_path = meta_dir / "label_stats_by_region.csv"

    full_df.to_csv(full_csv_path, index=False, encoding="utf-8-sig")
    by_region_df.to_csv(by_region_csv_path, index=False, encoding="utf-8-sig")

    # 输出 JSON
    full_json_path = meta_dir / "label_stats_full_regions.json"
    ignore_json_path = meta_dir / "ignore_classes.json"

    full_json = {
        "label_source": "Dynamic World",
        "label_band": LABEL_BAND,
        "class_names": {str(k): v for k, v in DW_CLASSES.items()},
        "total_valid_pixels": int(total_valid_pixels),
        "total_unknown_pixels": int(total_unknown_pixels),
        "class_pixel_counts": {str(k): int(v) for k, v in total_counts.items()},
        "present_classes": present_classes,
        "missing_classes": missing_classes,
        "missing_rule": "pixel_count == 0",
        "use_feature_zero_mask": USE_FEATURE_ZERO_MASK,
    }

    ignore_json = {
        "ignore_classes": missing_classes,
        "present_classes": present_classes,
        "reason": "Classes whose pixel_count equals 0 in the four full Wuhan region GeoTIFFs.",
        "recommendation": (
            "Keep num_classes=9 during training. "
            "Use ignore_classes mainly when computing evaluation metrics such as mIoU."
        ),
    }

    save_json(full_json, full_json_path)
    save_json(ignore_json, ignore_json_path)

    # 控制台输出结果
    print("\n" + "=" * 80)
    print("总体类别统计")
    print("=" * 80)
    print(full_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("武汉区域不存在的类别")
    print("=" * 80)

    if missing_classes:
        for class_id in missing_classes:
            print(f"  {class_id}: {DW_CLASSES[class_id]}")
    else:
        print("  没有发现像元数量为 0 的类别。")

    print("\n" + "=" * 80)
    print("输出文件")
    print("=" * 80)
    print(f"  {full_csv_path}")
    print(f"  {by_region_csv_path}")
    print(f"  {full_json_path}")
    print(f"  {ignore_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()