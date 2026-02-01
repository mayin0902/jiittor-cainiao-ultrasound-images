# 训练修改方案与实验计划

## 背景与目标
- 任务：单标签 6 类 BI-RADS 分级
- 指标：Top‑1 分类准确率
- 目标：在不改变主干结构的前提下，优化损失函数与训练策略；保留 BCE 作为对照实验。

## 方案总览
### 方案 A（主方案）：交叉熵 + 类权重
- 损失：`CrossEntropyLoss(weight=class_weights)`
- 目的：匹配单标签互斥假设，同时缓解类别不均衡
- 标签格式：整数类标（不再 one‑hot）

### 方案 B（主方案增强）：交叉熵 + 软标签
- 损失：交叉熵（支持软目标分布）或 KL/软交叉熵
- 软标签策略：
  - Label Smoothing：`1-ε` / `ε/(K-1)`
  - 等级软分布：以真实类别为峰，邻近类别衰减
- 目的：缓解过度自信，提高泛化能力

### 方案 C（对照实验）：BCEWithLogitsLoss + One‑Hot
- 损失：`BCEWithLogitsLoss`
- 标签格式：one‑hot
- 推理：仍用 `argmax` 产生单标签输出（与 Top‑1 指标一致）
- 目的：对照验证 BCE 是否在该任务下带来收益

## 具体修改点（代码层面）
### 1) 数据与标签
- 当前 `CustomImageDataset` 直接返回 one‑hot
- 修改方向：
  - 交叉熵方案：返回整型标签
  - BCE 对照：保留 one‑hot

### 2) 训练损失
- 当前：`BCEWithLogitsLoss`
- 修改为：
  - 主方案：`CrossEntropyLoss(weight=class_weights)`
  - 软标签方案：`CrossEntropyLoss` 或 `KLDivLoss`（配合 softmax / log‑softmax）
  - 对照：维持 BCE

### 3) 推理与评估
- 统一评估指标：Top‑1 accuracy
- 推理：`argmax`（softmax 或 logits 均可）
- BCE 方案不做阈值策略，以保持评测一致

## 实验设计（建议顺序）
1. **Baseline**：现有 BCE 版本（不改动）
2. **CE + 类权重**：替换损失与标签
3. **CE + Label Smoothing**：基于 CE 版本做平滑
4. **CE + 等级软分布**：构造相邻等级软目标
5. **BCE 对照**：保留 BCE 作为对比，严格对照 Top‑1

## 需要的配置与脚本支持
- 添加参数开关（loss_type / smoothing / class_weights / soft_label_mode）
- 记录实验配置与结果（建议在输出目录写 `config.json`）

## 输出与评估
- 每次实验保存：
  - 最佳 Top‑1 结果
  - 训练日志（loss/acc）
  - 对应配置
- 最终对比表：
  - BCE baseline vs CE + 权重 vs CE + 软标签

---

（如需我直接改代码，请确认：
- 是否接受新增配置项与 CLI 参数
- 是否已有类别频次统计（用于 class_weights））
