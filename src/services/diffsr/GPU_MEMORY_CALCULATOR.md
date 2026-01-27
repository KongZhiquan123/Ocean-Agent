# GPU 显存计算器使用指南

## 概述

`gpu_memory_calculator.py` 是一个用于验证 DiffSR 训练配置是否会导致 GPU 显存溢出的工具。

## 为什么需要显存计算？

在深度学习训练中，显存占用主要包括：
- **激活值** (~70%): 前向传播的中间结果
- **模型权重** (~10%): 模型参数
- **梯度** (~10%): 反向传播的梯度
- **优化器状态** (~10%): Adam 等优化器的动量和方差

如果配置不当，容易出现 **OOM (Out of Memory)** 错误，导致训练崩溃。

## 激活值显存计算公式

```
单层激活显存 = batch × seq_len × H × W × hidden_dim × 4 bytes
总激活显存 = 单层激活显存 × 网络层数
```

**安全阈值**: 总激活显存 < GPU显存 × 0.7

## 使用方法

### 方法 1: 从配置文件读取（推荐）

```bash
python src/services/diffsr/gpu_memory_calculator.py --config configs/fno_ocean.yaml
```

**优点**:
- 自动从 YAML 配置读取所有参数
- 无需手动输入参数
- 与实际训练配置完全一致

### 方法 2: 手动指定参数

```bash
python src/services/diffsr/gpu_memory_calculator.py \
  --batch 8 \
  --height 128 \
  --width 128 \
  --hidden_dim 256 \
  --num_layers 10 \
  --gpu_memory 24
```

**使用场景**:
- 快速验证不同参数组合
- 配置文件尚未创建时
- 探索最优配置

## 输出示例

### ✅ 安全配置

```
============================================================
🧮 GPU 显存计算器
============================================================

📂 读取配置文件: configs/fno_ocean.yaml
✓ 模型: fno

✓ GPU 总显存: 24.00 GB

💡 可用显存 (70%): 16.80 GB
   (预留 30% 给模型权重、梯度、优化器)

📊 配置参数:
   - Batch Size: 8
   - Spatial Resolution: 128 × 128
   - Hidden Dim: 256
   - Num Layers: 10
   - Seq Length: 1

🔢 显存占用:
   - 单层激活: 1.0737 GB
   - 总激活显存: 10.74 GB

✅ 安全
   剩余显存: 6.06 GB (36.1%)

💚 配置安全，可以开始训练！
============================================================
```

### ❌ 超限配置

```
============================================================
🧮 GPU 显存计算器
============================================================

✓ GPU 总显存: 24.00 GB

💡 可用显存 (70%): 16.80 GB
   (预留 30% 给模型权重、梯度、优化器)

📊 配置参数:
   - Batch Size: 16
   - Spatial Resolution: 256 × 256
   - Hidden Dim: 512
   - Num Layers: 10
   - Seq Length: 1

🔢 显存占用:
   - 单层激活: 8.5899 GB
   - 总激活显存: 85.90 GB

❌ 超限
   超出显存: 69.10 GB

⚠️  警告: 配置可能导致 OOM (Out of Memory)

🔧 建议调整:
   1. 降低 batch size: 16 → 3
   2. 降低分辨率: [256, 256] → [106, 106]
   3. 使用梯度累积模拟大 batch

============================================================
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | YAML 配置文件路径 | `configs/fno_ocean.yaml` |
| `--batch` | 批次大小 | `8` |
| `--height` | 输入高度（像素） | `128` |
| `--width` | 输入宽度（像素） | `128` |
| `--hidden_dim` | 隐藏层维度 | `256` |
| `--num_layers` | 网络层数 | `10` |
| `--gpu_memory` | GPU 显存（GB） | `24.0` |

## 配置调整策略

如果显存超限，按以下优先级调整：

### 1. 降低 Batch Size（首选）

- **优点**: 不影响模型质量和分辨率
- **缺点**: 训练速度变慢
- **建议**: 从 8 → 4 → 2 → 1

### 2. 降低空间分辨率

- **优点**: 显著降低显存占用
- **缺点**: 可能影响模型性能
- **建议**: 按比例缩小，如 [128,128] → [96,96] → [64,64]

### 3. 使用梯度累积

```yaml
train:
  batch_size: 2  # 实际 batch
  accumulation_steps: 4  # 模拟 batch=8
```

- **优点**: 模拟大 batch，不增加显存
- **缺点**: 训练速度慢 4 倍

### 4. 使用混合精度训练

```yaml
train:
  use_amp: true  # 使用 FP16
```

- **优点**: 显存占用减半
- **缺点**: 可能影响训练稳定性

## 不同 GPU 的推荐配置

### RTX 3090 / RTX 4090 (24GB)

| 分辨率 | Hidden Dim | 推荐 Batch | 预期占用 |
|--------|-----------|-----------|---------|
| 64×64  | 256       | 16        | ~5.4 GB |
| 128×128| 256       | 8         | ~10.7 GB|
| 256×256| 256       | 2         | ~10.7 GB|

### A100 (40GB)

| 分辨率 | Hidden Dim | 推荐 Batch | 预期占用 |
|--------|-----------|-----------|---------|
| 128×128| 512       | 8         | ~21.5 GB|
| 256×256| 512       | 4         | ~21.5 GB|
| 512×512| 256       | 2         | ~21.5 GB|

### V100 (16GB)

| 分辨率 | Hidden Dim | 推荐 Batch | 预期占用 |
|--------|-----------|-----------|---------|
| 64×64  | 256       | 8         | ~2.7 GB |
| 128×128| 256       | 4         | ~5.4 GB |
| 256×256| 256       | 1         | ~5.4 GB |

## 工作流示例

### 完整训练流程

```bash
# 1. 检查 GPU 状态
nvidia-smi

# 2. 验证配置显存
python src/services/diffsr/gpu_memory_calculator.py --config configs/my_config.yaml

# 3. 如果安全，开始训练
python src/services/diffsr/main.py --config configs/my_config.yaml

# 4. 训练中监控 GPU
watch -n 1 nvidia-smi
```

### 参数调优流程

```bash
# 尝试不同配置
python src/services/diffsr/gpu_memory_calculator.py \
  --batch 16 --height 256 --width 256 --hidden_dim 512 --num_layers 10 --gpu_memory 24

# 如果超限，逐步降低
python src/services/diffsr/gpu_memory_calculator.py \
  --batch 8 --height 256 --width 256 --hidden_dim 512 --num_layers 10 --gpu_memory 24

# 继续调整直到安全
python src/services/diffsr/gpu_memory_calculator.py \
  --batch 4 --height 256 --width 256 --hidden_dim 512 --num_layers 10 --gpu_memory 24
```

## 常见问题

### Q1: 计算器显示安全，但训练仍然 OOM？

**A**: 可能原因：
1. **多个模型同时运行**: 其他进程占用显存
2. **数据加载占用**: DataLoader 的 `num_workers` 过多
3. **模型估算偏差**: 某些模型结构复杂度超出估算

**解决方案**:
- 运行 `nvidia-smi` 检查其他占用
- 降低 `num_workers` (如 8 → 4)
- 进一步降低 batch size (如 8 → 6)

### Q2: 如何查看实际训练显存占用？

**A**: 使用实时监控：
```bash
watch -n 1 nvidia-smi
```

或在训练日志中查看 PyTorch 的显存报告。

### Q3: 不同模型的层数如何确定？

**A**: 常见模型层数参考：
- **FNO**: 4-8 层
- **U-Net**: 4 个尺度 × 2-4 个 ResBlock = 8-16 层
- **DDPM**: U-Net 结构，通常 12-20 层
- **Transformer**: 取决于 `num_encoder_layers`

可以从模板配置文件中读取准确值。

### Q4: 为什么安全阈值是 70%？

**A**: 显存分配策略：
- **70%**: 激活值（前向传播中间结果）
- **10%**: 模型权重
- **10%**: 梯度（反向传播）
- **10%**: 优化器状态（Adam 的动量和方差）

这是经验值，保守策略可用 60%。

## 集成到 AI 工作流

### 在 Kode Agent 中使用

AI Agent 应在创建配置文件后，训练前执行：

```python
# 1. 生成配置文件
config_path = "configs/my_training.yaml"

# 2. 验证显存
result = await bash(f"python src/services/diffsr/gpu_memory_calculator.py --config {config_path}")

# 3. 解析结果
if "✅ 安全" in result:
    # 开始训练
    await DiffSRPipeline({ operation: "train", config_path })
else:
    # 提示用户调整配置
    ask_user_to_adjust_config()
```

## 脚本返回值

- **Exit Code 0**: 配置安全，可以训练
- **Exit Code 1**: 配置超限，需要调整

可以在脚本中判断：

```bash
python src/services/diffsr/gpu_memory_calculator.py --config configs/my_config.yaml

if [ $? -eq 0 ]; then
    echo "开始训练..."
    python src/services/diffsr/main.py --config configs/my_config.yaml
else
    echo "配置不安全，请调整参数"
    exit 1
fi
```

## 贡献

如果发现计算不准确或有改进建议，请提交 Issue 或 PR。

## 许可证

MIT License
