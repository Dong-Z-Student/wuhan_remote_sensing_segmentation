# Wuhan SAR-Optical Remote Sensing Segmentation

本项目面向武汉地区遥感影像像元级地物识别任务，围绕光学遥感、SAR 雷达数据及二者融合开展实验。项目同时包含传统机器学习方法和深度学习方法，用于完成监督分类、非监督聚类以及语义分割实验。

项目仓库：

```bash
git clone https://github.com/Dong-Z-Student/wuhan_remote_sensing_segmentation.git
cd wuhan_remote_sensing_segmentation
```

## 1. Project Overview

本项目主要包含三类实验流程：

| 方法类型 | 具体方法 | 任务形式 | 主要脚本 |
|---|---|---|---|
| 传统监督学习 | Random Forest | 像元级监督分类 | `src/ml/run_random_forest.py` |
| 传统非监督学习 | K-Means | 像元级聚类 | `src/ml/run_kmeans.py` |
| 深度学习 | U-Net 及双分支 U-Net | 遥感影像语义分割 | `src/experiments/train_segmentation.py` |

深度学习部分进一步包含以下对比实验：

| 实验名称 | 输入数据 | 配置文件 |
|---|---|---|
| SAR-only U-Net | Sentinel-1 SAR 波段 | `configs/sar_only.yaml` |
| Optical-only U-Net | Sentinel-2 光学波段 | `configs/optical_only.yaml` |
| Dual-branch U-Net | 光学分支 + SAR 分支 | `configs/dual_branch.yaml` |

项目目标不是只训练单一模型，而是通过传统机器学习、非监督聚类和深度学习语义分割的对照，观察不同数据源和不同方法在武汉地区地物识别任务中的表现差异。

## 2. Repository Structure

```text
wuhan_sar_optical_fusion/
├── data/
│   └── wuhan/
│       ├── raw_regions/
│       ├── patches/
│       ├── meta/
│       └── splits/
│
├── scripts/
│   ├── 01_analyze_labels.py
│   ├── 02_filter_patches.py
│   └── 03_prepare_dataset.py
│
├── src/
│   ├── data/
│   │   └── wuhan_dataset.py
│   ├── models/
│   │   ├── blocks.py
│   │   ├── sar_unet.py
│   │   ├── optical_unet.py
│   │   └── dual_branch_unet.py
│   ├── experiments/
│   │   ├── train_segmentation.py
│   │   ├── evaluate_segmentation.py
│   │   ├── predict_patches.py
│   │   └── compare_ablation.py
│   ├── ml/
│   │   ├── run_random_forest.py
│   │   └── run_kmeans.py
│   └── utils/
│       ├── raster_io.py
│       ├── metrics.py
│       ├── visualization.py
│       └── seed.py
│
├── configs/
│   ├── sar_only.yaml
│   ├── optical_only.yaml
│   └── dual_branch.yaml
│
├── outputs/
│   ├── ablation/
│   │   ├── sar_only/
│   │   ├── optical_only/
│   │   └── dual_branch/
│   ├── random_forest/
│   └── kmeans/
│
├── requirements.txt
└── README.md
```

## 3. Directory Description

### 3.1 `data/`

`data/wuhan/` 用于存放武汉研究区的原始影像、切片数据、元数据和训练验证划分文件。

| 子目录 | 作用 |
|---|---|
| `raw_regions/` | 存放原始研究区影像数据 |
| `patches/` | 存放切片后的 patch 数据 |
| `meta/` | 存放类别统计、波段信息、数据说明等元数据 |
| `splits/` | 存放训练集、验证集或测试集划分文件 |

### 3.2 `scripts/`

`scripts/` 中的脚本主要用于数据准备阶段。

| 文件 | 作用 |
|---|---|
| `01_analyze_labels.py` | 统计标签类别、类别像元数量和类别比例 |
| `02_filter_patches.py` | 过滤无效 patch，例如标签缺失比例过高的切片 |
| `03_prepare_dataset.py` | 根据有效 patch 生成训练、验证或测试数据划分 |

### 3.3 `src/data/`

| 文件 | 作用 |
|---|---|
| `wuhan_dataset.py` | 定义 PyTorch 数据集类，负责读取 patch、选择输入波段、读取标签并返回模型训练所需的数据格式 |

### 3.4 `src/models/`

| 文件 | 作用 |
|---|---|
| `blocks.py` | 定义 U-Net 中可复用的基础卷积模块 |
| `sar_unet.py` | 定义仅使用 SAR 数据的 U-Net 模型 |
| `optical_unet.py` | 定义仅使用光学数据的 U-Net 模型 |
| `dual_branch_unet.py` | 定义光学分支与 SAR 分支融合的双分支 U-Net 模型 |

### 3.5 `src/experiments/`

| 文件 | 作用 |
|---|---|
| `train_segmentation.py` | 训练深度学习语义分割模型 |

### 3.6 `src/ml/`

| 文件                     | 作用                 |
|------------------------|--------------------|
| `ml_dataset.py`        | 返回传统机器学习训练所需的数据格式  |
| `run_random_forest.py` | 运行随机森林监督分类实验       |
| `run_kmeans.py`        | 运行 K-Means 非监督聚类实验 |

### 3.7 `src/utils/`

| 文件 | 作用 |
|---|---|
| `metrics.py` | 精度评价指标计算，包括 pixel accuracy、mean class accuracy、mIoU 等 |
| `seed.py` | 设置随机种子，增强实验可复现性 |

## 4. Environment Setup

建议使用 Conda 创建或管理 Python 环境。若已有本地环境，可直接激活。

Windows 示例：

```bash
conda activate D:\conda_envs\torch_gpu
```

这里的环境路径需要根据本机实际情况修改。

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 GPU 训练，需要确保本机 PyTorch、CUDA、显卡驱动版本匹配。

## 5. Data Preparation

### 5.1 Patch Filtering

过滤无效 patch：

```bash
python scripts\02_filter_patches.py
```

该步骤通常用于剔除标签缺失、黑边比例过高或无效像元比例过高的 patch。过滤规则由脚本内部参数或相关配置控制。

### 5.2 Dataset Preparation

生成训练、验证或测试划分文件：

```bash
python scripts\03_prepare_dataset.py
```

执行后，划分结果通常保存在：

```text
data/wuhan/splits/
```

### 5.3 Dataset Check

可以直接运行数据集文件，检查数据是否能够被正常读取：

```bash
python src\data\wuhan_dataset.py
```

该步骤可用于确认：

- patch 路径是否正确；
- 输入波段索引是否正确；
- 标签是否能正常读取；
- batch 数据维度是否符合模型输入要求。

## 6. Deep Learning Experiments

深度学习实验统一通过 `src/experiments/train_segmentation.py` 启动，不同模型和输入数据由 `configs/` 下的配置文件控制。

### 6.1 Optical-only U-Net

先检查模型结构：

```bash
python -m src.models.optical_unet
```

启动训练：

```bash
python src\experiments\train_segmentation.py --config configs\optical_only.yaml
```

该实验仅使用光学波段作为输入，用于评估光学遥感数据在地物语义分割任务中的表现。

### 6.2 SAR-only U-Net

启动训练：

```bash
python src\experiments\train_segmentation.py --config configs\sar_only.yaml
```

该实验仅使用 SAR 数据作为输入，用于评估雷达影像对不同地物类别的区分能力。

### 6.3 Dual-branch U-Net

先检查双分支模型结构：

```bash
python -m src.models.dual_branch_unet
```

启动训练：

```bash
python src\experiments\train_segmentation.py --config configs\dual_branch.yaml
```

双分支模型通常包含：

- 光学编码分支；
- SAR 编码分支；
- 融合模块；
- 共享解码器；
- 像元级分类输出层。

该结构用于比较单源遥感数据和 SAR-Optical 融合数据在语义分割任务中的差异。

## 7. Traditional Machine Learning Experiments

### 7.1 Random Forest Supervised Classification

运行随机森林监督分类：

```bash
python src\ml\run_random_forest.py --config configs\random_forest.yaml
```

该方法将遥感影像像元或采样像元作为特征向量，使用已有标签进行监督训练。实验结果可用于与深度学习语义分割模型进行对比。

输出目录通常为：

```text
outputs/random_forest/
```

### 7.2 K-Means Unsupervised Clustering

运行 K-Means 非监督聚类：

```bash
python src\ml\run_kmeans.py --config configs\kmeans.yaml
```

该方法不直接使用标签进行训练，而是根据输入光谱或雷达特征对像元进行聚类。聚类结果可与真实标签进行后验对比，用于观察不同地物类别在特征空间中的可分性。

输出目录通常为：

```text
outputs/kmeans/
```

## 8. Evaluation Metrics

项目中常用的语义分割评价指标包括：

| 指标 | 含义 |
|---|---|
| `pixel_acc` | 所有有效像元中预测正确的比例 |
| `mean_class_acc` | 各类别精度的平均值，可减弱大类像元数量过多带来的影响 |
| `mIoU` | 各类别 Intersection over Union 的平均值，是语义分割中常用的综合指标 |
| `per_class_acc` | 每个类别单独计算的分类精度 |
| `per_class_iou` | 每个类别单独计算的 IoU |
| `confusion_matrix` | 混淆矩阵，用于分析类别之间的误分关系 |

在类别样本数量不均衡的情况下，不能只看 `pixel_acc`。应同时查看 `mean_class_acc`、`mIoU`、单类别精度和混淆矩阵。

## 9. Output Files

深度学习实验结果通常保存在：

```text
outputs/ablation/
├── sar_only/
├── optical_only/
└── dual_branch/
```

每个实验目录下可能包含：

```text
best_model.pth
last_model.pth
best_val_metrics.json
train_log.csv
```

具体文件名称以训练脚本实际保存结果为准。

传统机器学习结果通常保存在：

```text
outputs/random_forest/
outputs/kmeans/
```

## 10. Suggested Workflow

推荐按照以下顺序运行整个项目：

```bash
# 1. 激活环境
conda activate D:\conda_envs\torch_gpu

# 2. 安装依赖
pip install -r requirements.txt

# 3. 过滤无效 patch
python scripts\02_filter_patches.py

# 4. 准备训练和验证数据
python scripts\03_prepare_dataset.py

# 5. 检查数据集读取是否正常
python src\data\wuhan_dataset.py

# 6. 检查光学 U-Net 模型
python -m src.models.optical_unet

# 7. 训练 Optical-only U-Net
python src\experiments\train_segmentation.py --config configs\optical_only.yaml

# 8. 检查双分支 U-Net 模型
python -m src.models.dual_branch_unet

# 9. 训练 Dual-branch U-Net
python src\experiments\train_segmentation.py --config configs\dual_branch.yaml

# 10. 训练 SAR-only U-Net
python src\experiments\train_segmentation.py --config configs\sar_only.yaml

# 11. 运行随机森林监督分类
python src\ml\run_random_forest.py --config configs\random_forest.yaml

# 12. 运行 K-Means 非监督聚类
python src\ml\run_kmeans.py --config configs\kmeans.yaml
```

## 11. Configuration Files

`configs/` 中的 YAML 文件用于控制实验参数。常见配置项包括：

```yaml
data:
  root_dir: data/wuhan
  split_dir: data/wuhan/splits
  patch_dir: data/wuhan/patches

model:
  name: optical_unet
  in_channels: 10
  num_classes: 8
  base_channels: 32

training:
  batch_size: 8
  epochs: 50
  learning_rate: 0.001
  seed: 42

loss:
  ignore_index: null
  class_weight: true

output:
  output_dir: outputs/ablation/optical_only
```

不同实验需要重点检查以下参数：

| 参数 | 说明 |
|---|---|
| `model.name` | 使用的模型类型 |
| `model.in_channels` | 输入通道数 |
| `model.num_classes` | 分类类别数 |
| `data.optical_indices` | 光学输入波段索引 |
| `data.sar_indices` | SAR 输入波段索引 |
| `training.batch_size` | 批大小 |
| `training.epochs` | 训练轮数 |
| `training.learning_rate` | 学习率 |
| `loss.ignore_index` | 是否忽略某些标签类别 |
| `loss.class_weight` | 是否使用类别权重 |
| `output.output_dir` | 实验输出路径 |

具体字段名称应以项目中的实际 YAML 文件为准。

## 12. Notes

1. 运行脚本前应确认当前工作目录位于项目根目录。
2. Windows 环境下可以使用反斜杠路径，例如 `src\experiments\train_segmentation.py`。
3. Linux 或 macOS 环境下需要改为斜杠路径，例如 `src/experiments/train_segmentation.py`。
4. 如果新增配置文件，例如 `random_forest.yaml`、`kmeans.yaml` 或 `dual_branch_with_sar_indices.yaml`，需要确认它们已经放入 `configs/` 目录。
5. 如果修改输入波段数量，需要同步修改配置文件中的 `in_channels`、数据集读取逻辑和模型输入设置。
6. 如果忽略某一类别，需要确保训练、评价和可视化阶段使用一致的类别映射关系。
7. 类别极度不均衡时，建议同时查看类别权重、单类别精度、mIoU 和混淆矩阵，而不只依据总体像元精度判断模型效果。

## 13. Reproducibility

项目通过 `src/utils/seed.py` 设置随机种子，以减少实验结果的随机波动。由于 GPU 并行计算、数据划分、随机采样和深度学习训练过程本身存在不确定性，不同机器上的结果可能存在小幅差异。

建议在报告实验结果时记录：

- 使用的配置文件；
- 输入数据版本；
- 训练集和验证集划分；
- batch size；
- epoch 数；
- learning rate；
- 是否使用类别权重；
- 是否忽略某些类别；
- 最优模型对应的验证集指标。

## 14. Project Status

当前项目已包含：

- 武汉研究区遥感数据处理流程；
- patch 过滤与数据集准备流程；
- 随机森林监督分类实验；
- K-Means 非监督聚类实验；
- SAR-only、Optical-only 和 Dual-branch U-Net 模型；
- 精度评价与消融实验对比流程。

后续可继续扩展：

- 更复杂的融合模块；
- 类别重采样或难例采样策略；
- 更完整的预测结果可视化与制图输出。
