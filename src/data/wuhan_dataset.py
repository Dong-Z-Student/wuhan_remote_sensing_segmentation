from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class WuhanPatchDataset(Dataset):
    """
    武汉 SAR-Optical 融合 patch 数据集。

    支持三种模式：
        mode="sar"     -> 返回 s1, mask
        mode="optical" -> 返回 s2, mask
        mode="dual"    -> 返回 s1, s2, mask

    当前波段顺序：
        Band 1-5    SAR + radar indices
        Band 6-14   Optical + optical indices
        Band 15     label
    """

    def __init__(
        self,
        split_txt: Union[str, Path],
        project_root: Optional[Union[str, Path]] = None,
        mode: str = "dual",
        sar_indices: Optional[List[int]] = None,
        optical_indices: Optional[List[int]] = None,
        label_index: int = 14,
        ignore_label_values: Optional[List[int]] = None,
        ignore_index: int = 255,
        return_path: bool = False,
        check_nan: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.split_txt = Path(split_txt)

        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

        self.mode = mode.lower()

        if self.mode not in ["sar", "optical", "dual"]:
            raise ValueError(
                f"mode 必须是 'sar', 'optical', 'dual' 之一，当前为: {mode}"
            )

        self.sar_indices = sar_indices if sar_indices is not None else [0, 1, 2, 3, 4]

        self.optical_indices = (
            optical_indices
            if optical_indices is not None
            else [5, 6, 7, 8, 9, 10, 11, 12, 13]
        )

        self.label_index = label_index

        self.ignore_label_values = (
            ignore_label_values if ignore_label_values is not None else []
        )
        self.ignore_index = ignore_index

        self.return_path = return_path
        self.check_nan = check_nan
        self.normalize = normalize

        self.patch_paths = self._read_split_file(self.split_txt)

        if len(self.patch_paths) == 0:
            raise ValueError(f"split 文件为空: {self.split_txt}")

    def _read_split_file(self, split_txt: Path) -> List[Path]:
        if not split_txt.exists():
            raise FileNotFoundError(f"split 文件不存在: {split_txt}")

        paths: List[Path] = []

        with open(split_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                path = Path(line)

                if not path.is_absolute():
                    path = self.project_root / path

                paths.append(path)

        return paths

    def __len__(self) -> int:
        return len(self.patch_paths)

    @staticmethod
    def _ensure_band_first(arr: np.ndarray, path: Path) -> np.ndarray:
        """
        将 tifffile 读取结果统一成 [bands, height, width]。
        """
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

    def _read_patch(self, path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"patch 文件不存在: {path}")

        arr = tifffile.imread(str(path))
        arr = np.asarray(arr, dtype=np.float32)
        arr = self._ensure_band_first(arr, path)

        if arr.shape[0] <= self.label_index:
            raise ValueError(
                f"patch 波段数不足: {path}, shape={arr.shape}, "
                f"label_index={self.label_index}"
            )

        if arr.shape[1] != 256 or arr.shape[2] != 256:
            raise ValueError(
                f"patch 尺寸不是 256x256: {path}, shape={arr.shape}"
            )

        s1 = arr[self.sar_indices, :, :]
        s2 = arr[self.optical_indices, :, :]
        label = arr[self.label_index, :, :]

        if self.check_nan:
            if np.isnan(s1).any():
                raise ValueError(f"SAR 输入存在 NaN: {path}")

            if np.isnan(s2).any():
                raise ValueError(f"Optical 输入存在 NaN: {path}")

            if np.isnan(label).any():
                raise ValueError(f"label 存在 NaN: {path}")

        # label 从 float 转成整数类别
        mask = np.rint(label).astype(np.int64)

        # 原始标签必须在 0-8 范围内
        raw_min = int(mask.min())
        raw_max = int(mask.max())

        if raw_min < 0 or raw_max > 8:
            raise ValueError(
                f"原始 label 类别超出 0-8 范围: {path}, "
                f"min={raw_min}, max={raw_max}"
            )

        # 将需要忽略的原始类别改成 ignore_index
        for ignore_value in self.ignore_label_values:
            mask[mask == int(ignore_value)] = int(self.ignore_index)

        if self.normalize:
            s1 = self._simple_normalize(s1)
            s2 = self._simple_normalize(s2)

        return s1, s2, mask

    @staticmethod
    def _simple_normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mean = np.mean(x, axis=(1, 2), keepdims=True)
        std = np.std(x, axis=(1, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (x - mean) / std

    def __getitem__(self, idx: int):
        path = self.patch_paths[idx]

        s1, s2, mask = self._read_patch(path)

        s1_tensor = torch.from_numpy(s1).float()
        s2_tensor = torch.from_numpy(s2).float()
        mask_tensor = torch.from_numpy(mask).long()

        if self.mode == "sar":
            if self.return_path:
                return s1_tensor, mask_tensor, str(path)
            return s1_tensor, mask_tensor

        if self.mode == "optical":
            if self.return_path:
                return s2_tensor, mask_tensor, str(path)
            return s2_tensor, mask_tensor

        if self.mode == "dual":
            if self.return_path:
                return s1_tensor, s2_tensor, mask_tensor, str(path)
            return s1_tensor, s2_tensor, mask_tensor

        raise RuntimeError(f"未知 mode: {self.mode}")


def test_dataset_loading() -> None:
    project_root = Path(__file__).resolve().parents[2]
    train_txt = project_root / "data" / "wuhan" / "splits" / "train.txt"

    dataset = WuhanPatchDataset(
        split_txt=train_txt,
        project_root=project_root,
        mode="optical",
        optical_indices=[5, 6, 7, 8, 9, 10],
        label_index=14,
        ignore_label_values=[8],
        ignore_index=255,
        return_path=True,
        check_nan=True,
        normalize=False,
    )

    s2, mask, path = dataset[0]

    print("Dataset 测试成功")
    print(f"path: {path}")
    print(f"s2 shape: {s2.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"mask dtype: {mask.dtype}")
    print(f"mask unique: {torch.unique(mask)}")


if __name__ == "__main__":
    test_dataset_loading()