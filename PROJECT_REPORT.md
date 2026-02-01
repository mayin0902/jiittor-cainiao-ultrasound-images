# 项目解析报告（jiittor-cainiao-image）

## 1. 项目概览
本项目面向「第五届 Jittor 算法挑战赛」赛道一：超声图像智能筛查与分级。核心流程是基于乳腺癌超声影像数据训练多分类模型，使用 Jittor 框架与基于 jimm（timm 早期版本改造）的模型库作为 backbone，最终采用 `EfficientNetV2-S` 的多尺度特征融合 + 多个 dropout 的分类头。

## 2. 目录结构速览
```
/home/mayin/jiittor-cainiao-image
├─ README.md
├─ requirements.txt
├─ config/
│  └─ config.yaml
├─ data/
│  ├─ dataset.py
│  └─ transform.py
├─ models/
│  ├─ jit_model.py
│  └─ jimm/               # 迁移/改造的模型库（timm早期版本）
├─ utils/
│  ├─ comm.py
│  └─ trainer.py
├─ train.py
├─ test.py
├─ test.sh
└─ CGAN.py                # 独立的条件GAN示例，和主训练/推理流程无耦合
```

## 3. 依赖与环境
`requirements.txt` 主要依赖：
- Jittor 1.3.9.14
- Albumentations / Pillow / numpy / pandas / tqdm
- 额外包含 torch / torchvision（主流程未直接使用，但可能用于 jimm 或本地实验）

README 中建议环境：Ubuntu 18.04、CUDA 12.1、Python 3.12.2、4090D。

## 4. 配置（config/config.yaml）
关键配置项：
- 训练控制：`epochs: 50`、`train_bs: 16`、`valid_bs: 16`、`seed: 42`
- 交叉验证：`N_FOLDS: 5`、`folds: [0,1,2,3,4]`
- 数据增强：`img_size: 512x512`、`crop_size: 448x448`
- 优化器：AdamW + CosineAnnealingLR
- 模型：`pretrained: true`
- 分类数：`num_class: 6`

注意：`early_stop` 配置存在，但训练逻辑未使用。

## 5. 数据格式与读取流程
### 5.1 训练数据
- `train.py` 默认读取 `./data_jittor/train.csv`。
- CSV 需要包含至少三列：
  - `image_name`: 图像文件名
  - `label`: 类别（0~5）
  - `fold`: 折号（用于交叉验证分割）

### 5.2 数据集实现（data/dataset.py）
- 读取灰度图（L），复制为 3 通道。
- 训练/验证使用不同增广 pipeline。
- 标签使用 one-hot（维度 = `total_classes`）。

### 5.3 增强与预处理（data/transform.py）
- 训练：Resize → RandomCrop → Flip/Rotate/Blur/Contrast → Normalize
- 验证：Resize → CenterCrop → Normalize
- 均值/方差为 ImageNet 风格 `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

## 6. 模型结构与核心逻辑
### 6.1 主模型（models/jit_model.py）
- Backbone: `tf_efficientnetv2_s_in21k`
- 特征融合：
  - 取最后一个 block 输出 + 倒数第二个 block 输出
  - 全局池化后拼接（`mid_features = 1280 + 160`）
- 分类头：
  - `BatchNorm1d` → 多个 Dropout → Linear
  - 多 dropout 输出求平均（ensemble-style）

### 6.2 训练与评估（utils/trainer.py）
- Loss: `BCEWithLogitsLoss`
- 预测采用 `argmax`，以 accuracy 为验证指标。
- 每个 epoch 都保存模型：`output/<train_version>/fold{idx}_epoch{epoch}.pkl`

### 6.3 推理（test.py）
- 支持多模型融合：多个 ckpt + 权重融合
- 输出格式：每行 `image_name pred`，写入 `result.txt`

## 7. 训练流程概述
`python train.py` 执行流程：
1. 加载 YAML 配置并解析 CLI 覆盖参数。
2. 读取 CSV，按 fold 划分训练/验证。
3. 构建数据加载器 + 模型 + AdamW + CosineAnnealingLR。
4. 迭代训练，逐 epoch 验证与保存模型。

## 8. 推理流程概述
`bash test.sh` 运行：
1. 读取测试目录（`--test_dir`）
2. 加载多权重并融合
3. 逐批预测，输出 `result.txt`

## 9. 关键问题与改进建议
### 9.1 可用性与健壮性
- `folds` 配置未生效：`train.py` 实际循环使用 `N_FOLDS`，忽略 `config['folds']`。
- `early_stop` 未实现。
- `jt.flags.use_cuda = 1` 强制使用 GPU，缺少 CPU fallback。

### 9.2 损失函数与指标一致性
- 当前使用 `BCEWithLogitsLoss` + one-hot + argmax，适合多标签，但此处是单标签多类任务，通常应使用 `CrossEntropyLoss`（无需 one-hot）。
- 若保持 `BCEWithLogitsLoss`，建议明确是否是多标签设定。

### 9.3 模型保存策略
- 每个 epoch 都保存，不区分最佳模型与最终模型。
- 建议保存 best acc 的 checkpoint，或按固定间隔保存。

### 9.4 代码资产
- 仓库包含大量 `__pycache__/*.pyc`，建议忽略或清理。
- `CGAN.py` 为独立示例，与主流程无耦合，可能干扰维护。

## 10. 复现与运行指南（总结版）
```
# 安装依赖
pip install -r requirements.txt

# 训练
python train.py --cfg ./config/config.yaml

# 推理
bash test.sh
```

## 11. 产出物说明
- 训练输出：`output/<train_version>/foldX_epochY.pkl`
- 推理输出：`result.txt`，每行 `image_name pred`

---

如需我继续：
- 输出更详细的代码级流程图
- 给出训练/推理参数配置模板
- 针对单卡/多卡或更大数据集进行优化建议
