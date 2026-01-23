# Kode 快速部署指南

## 🚀 快速部署步骤

### 1. 克隆代码库

```bash
# clone代码库
git clone https://github.com/KongZhiquan123/Ocean-Agent

# 进入kode目录
cd kode
```

### 2. 安装依赖

**推荐使用Bun（更快）:**
```bash
bun install
```

**或使用npm:**
```bash
npm install
```

### 3. 构建项目

```bash
# 使用Bun
bun run build

# 或使用npm
npm run build
```

构建成功后会看到：
```
✅ DiffSR-main copied to dist/services/diffsr
✅ Prediction service copied to dist/services/prediction
✅ Preprocessing service copied to dist/services/preprocessing
✅ cli.js made executable
✅ Build completed for cross-platform compatibility!
```

### 4. 全局安装（可选）

**如需在任意目录使用 `kode` 命令:**
```bash
# 使用npm
npm link

# 或使用bun
bun link
```

### 5. 验证安装

```bash
kode --version
kode --help
```

---

## ✅ 本次更新重点

### 修复: Prediction自动生成MD格式报告

**问题**: 训练完成后只生成`.npz`文件，未生成MD报告

**现在**: 训练/测试完成后自动生成：
- ✅ `training_report.md` / `test_report.md` - 完整MD格式报告
- ✅ `report_config.json` - 配置数据
- ✅ `report_metrics.json` - 指标数据

**报告内容包括:**
- 执行摘要（核心成果、关键指标）
- 训练配置（模型架构、数据配置、超参数、硬件）
- 训练过程（时间线、性能演进）
- 最终评估（验证/测试指标：R²、RMSE、MAE等）
- 总结（核心成就、关键数据）

---

## 💻 使用示例

### 示例1: 训练Prediction模型

```bash
# 启动kode
kode

# 在kode中使用PredictionPipeline工具
# 训练完成后自动生成报告:
# ✅ Training report generated: outputs/training_report.md
```

### 示例2: 直接运行Python训练

```bash
cd kode/src/services/prediction

python main.py --mode train --config configs/surface_config.yaml

# 训练完成后查看输出目录:
ls outputs/
# training_report.md          ← 新增: MD格式报告
# report_config.json          ← 新增: 配置JSON
# report_metrics.json         ← 新增: 指标JSON
# final_metrics.npz           ← 原有: npz文件
# best_model.pth              ← 模型检查点
```

### 示例3: 测试模式

```bash
python main.py --mode test \
  --config configs/surface_config.yaml \
  --model_path outputs/best_model.pth

# 测试完成后生成:
# test_report.md              ← 测试报告
```

---

## 📊 报告示例

生成的 `training_report.md` 包含：

```markdown
# 海洋预测模型训练完整报告

**生成时间**: 2025-12-04 15:30:25
**模型**: Fuxi
**数据集**: ocean
**训练时长**: 2小时15分30秒

## 执行摘要

### 核心成果
- ✅ **模型训练**: 成功完成 100 个 epochs
- ✅ **测试性能**: R² 0.9234, RMSE 0.0567, MAE 0.0423
- ✅ **模型检查点**: /path/to/best_model.pth
- ✅ **训练稳定性**: 训练过程稳定，损失函数收敛良好

### 关键指标
- **参数量**: 45,678,901
- **训练模式**: 单GPU
- **最终测试集 R²**: 0.9234
- **最终测试集 RMSE**: 0.0567
- **最终测试集 MAE**: 0.0423

## 1. 训练配置
### 1.1 模型架构
| 配置项 | 值 |
|--------|-----|
| **模型名称** | Fuxi |
| **模型类型** | Transformer |
| **参数量** | 45,678,901 |
...（更多详细内容）
```

---

## 🔍 目录结构

```
kode/
├── cli.js                      # 跨平台CLI入口
├── dist/                       # 构建输出目录
├── src/                        # 源代码
│   ├── services/
│   │   ├── prediction/         # Prediction服务
│   │   │   ├── report_generator.py        ← 报告生成器
│   │   │   ├── report_templates/          ← MD报告模板
│   │   │   ├── trainers/
│   │   │   │   └── ocean_trainer.py       ← 已修复
│   │   │   └── main.py
│   │   └── diffsr/             # DiffSR服务
│   └── tools/                  # 工具集
├── KODE_v1.1.25_更新说明.md    # 详细更新说明
├── package.json
└── bun.lock
```

---

## ⚙️ 系统要求

### 前端部署
- Node.js >= 18.0.0 或 Bun >= 1.0.0
- 操作系统: Windows / Linux / macOS

### 后端Python环境（用于Prediction/DiffSR）
- Python >= 3.8
- PyTorch >= 1.10
- CUDA (可选，用于GPU加速)

---

## 🔧 配置说明

### Prediction配置文件

配置文件位于: `src/services/prediction/configs/`

可用配置:
- `surface_config.yaml` - 海洋表层数据
- `mid_config.yaml` - 海洋中层数据
- `pearl_river_config.yaml` - 珠江口数据

### 报告模板

模板位于: `src/services/prediction/report_templates/`
- `predict_training_report.md` - 训练报告模板
- `predict_data_report.md` - 数据报告模板

可根据需要自定义模板格式。

---

## 📞 技术支持

### 常见问题

**Q: 报告没有生成？**
A: 检查训练日志，确认 `_generate_training_report` 方法被调用。如有错误会打印警告信息。

**Q: 报告格式不对？**
A: 确认模板文件存在于 `report_templates/` 目录。

**Q: 构建失败？**
A: 尝试清理后重新构建:
```bash
bun run clean
bun install
bun run build
```

### 修改的核心文件
- `src/services/prediction/trainers/ocean_trainer.py`
  - 新增 `_generate_training_report()` 方法
  - 新增 `_generate_test_report()` 方法
  - 修改 `process()` 和 `test()` 方法

---

## 📝 更新记录

### v1.1.25 (2025-12-04)
- ✅ 修复Prediction报告生成问题
- ✅ 训练完成自动生成MD格式报告
- ✅ 测试完成自动生成MD格式报告
- ✅ 报告包含完整的配置、指标和性能数据

### v1.1.24 及之前
- 基础Prediction训练和测试功能
- DiffSR超分辨率功能
- OceanPreprocess数据预处理功能

---

## 📄 相关文档

详细更新说明请查看: `KODE_v1.1.25_更新说明.md`

---
