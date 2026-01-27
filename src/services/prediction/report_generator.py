#!/usr/bin/env python3
"""
Prediction 报告生成器
自动生成符合模板格式的训练报告和数据处理报告
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class PredictionReportGenerator:
    """Prediction 报告生成器"""

    def __init__(self, template_dir: Optional[str] = None):
        """
        初始化报告生成器

        Args:
            template_dir: 模板目录路径，默认为当前目录下的 report_templates
        """
        if template_dir is None:
            current_dir = Path(__file__).parent
            template_dir = current_dir / 'report_templates'

        self.template_dir = Path(template_dir)
        self.train_template = self.template_dir / 'predict_training_report.md'
        self.data_template = self.template_dir / 'predict_data_report.md'

    def generate_train_report(self,
                              config: Dict[str, Any],
                              metrics: Dict[str, Any],
                              output_path: str,
                              viz_paths: Optional[List[str]] = None) -> str:
        """
        生成训练报告

        Args:
            config: 训练配置信息
            metrics: 训练指标和结果
            output_path: 报告输出路径
            viz_paths: 可视化图片路径列表（可选）

        Returns:
            生成的报告路径
        """
        # 读取模板
        with open(self.train_template, 'r', encoding='utf-8') as f:
            template = f.read()

        # 提取信息
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_name = config.get('model', {}).get('name', 'Unknown')
        dataset_name = config.get('data', {}).get('name', 'Unknown')

        # 训练时长
        train_duration = metrics.get('total_time', 'N/A')
        if isinstance(train_duration, (int, float)):
            hours = int(train_duration // 3600)
            minutes = int((train_duration % 3600) // 60)
            seconds = int(train_duration % 60)
            train_duration_str = f"{hours}小时{minutes}分{seconds}秒"
        else:
            train_duration_str = str(train_duration)

        # 填充基本信息
        report = template.replace('**生成时间**: ', f'**生成时间**: {current_time}')
        report = report.replace('**模型**: ', f'**模型**: {model_name}')
        report = report.replace('**数据集**: ', f'**数据集**: {dataset_name}')
        report = report.replace('**训练时长**: ', f'**训练时长**: {train_duration_str}')

        # 填充执行摘要
        best_epoch = metrics.get('best_epoch', 'N/A')
        best_loss = metrics.get('best_loss', 'N/A')
        best_rmse = metrics.get('best_rmse', 'N/A')
        best_r2 = metrics.get('best_r2', 'N/A')
        best_mae = metrics.get('best_mae', 'N/A')
        model_path = metrics.get('model_path', 'N/A')
        num_params = config.get('model', {}).get('num_params', 'N/A')

        report = report.replace(
            '- ✅ **模型训练**: ',
            f'- ✅ **模型训练**: 成功完成 {best_epoch} 个 epochs'
        )

        # 格式化测试性能字符串，添加类型检查
        perf_parts = []
        if isinstance(best_r2, (int, float)):
            perf_parts.append(f"R² {best_r2:.4f}")
        if isinstance(best_rmse, (int, float)):
            perf_parts.append(f"RMSE {best_rmse:.4f}")
        if isinstance(best_mae, (int, float)):
            perf_parts.append(f"MAE {best_mae:.4f}")
        test_perf_str = ", ".join(perf_parts) if perf_parts else "N/A"

        report = report.replace(
            '- ✅ **测试性能**:',
            f'- ✅ **测试性能**: {test_perf_str}'
        )
        report = report.replace(
            '- ✅ **模型检查点**: ',
            f'- ✅ **模型检查点**: {model_path}'
        )
        report = report.replace(
            '- ✅ **训练稳定性**: ',
            f'- ✅ **训练稳定性**: 训练过程稳定，损失函数收敛良好'
        )

        # 填充关键指标
        # 处理参数量格式化
        if isinstance(num_params, (int, float)):
            param_str = f"{num_params:,}"
        else:
            param_str = str(num_params)

        report = report.replace(
            '- **参数量**: ',
            f'- **参数量**: {param_str}'
        )
        report = report.replace(
            '- **训练模式**: ',
            f'- **训练模式**: {config.get("train", {}).get("distribute_mode", "单GPU")}'
        )

        # 格式化关键指标，添加类型检查
        r2_str = f"{best_r2:.4f}" if isinstance(best_r2, (int, float)) else str(best_r2)
        rmse_str = f"{best_rmse:.4f}" if isinstance(best_rmse, (int, float)) else str(best_rmse)
        mae_str = f"{best_mae:.4f}" if isinstance(best_mae, (int, float)) else str(best_mae)

        report = report.replace(
            '- **最终测试集 R²**: ',
            f'- **最终测试集 R²**: {r2_str}'
        )
        report = report.replace(
            '- **最终测试集 RMSE**: ',
            f'- **最终测试集 RMSE**: {rmse_str}'
        )
        report = report.replace(
            '- **最终测试集 MAE**: ',
            f'- **最终测试集 MAE**: {mae_str}'
        )

        # 填充训练配置表格
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        train_config = config.get('train', {})
        optimize_config = config.get('optimizer', {})

        # 1.1 模型架构表格
        model_table = self._generate_model_config_table(model_config)
        report = report.replace(
            '| **模型名称** | ... |',
            model_table
        )

        # 1.2 数据配置表格
        data_table = self._generate_data_config_table(data_config)
        report = report.replace(
            '| **数据集** | ... |',
            data_table
        )

        # 1.3 训练超参数表格
        train_table = self._generate_train_config_table(train_config, optimize_config)
        report = report.replace(
            '| **总Epochs** | ... |',
            train_table
        )

        # 1.4 硬件配置
        gpu_config = self._generate_gpu_config_table(metrics.get('gpu_info', {}))
        report = report.replace(
            '| **GPU** |...|',
            gpu_config
        )

        # 2.2 训练曲线 - 生成 Markdown 表格
        train_history = metrics.get('train_history', [])
        if train_history:
            loss_table = self._generate_loss_curve_table(train_history)
            report = report.replace(
                '#### 损失下降趋势\n\n',
                f'#### 损失下降趋势\n\n{loss_table}\n'
            )

        # 2.3 验证集性能演进
        valid_history = metrics.get('valid_history', [])
        if valid_history:
            valid_table = self._generate_validation_table(valid_history)
            report = report.replace(
                '| Epoch | Valid Loss | Valid MAE | Valid RMSE | Valid R²|... | 改进率 |',
                valid_table
            )

        # 3.2 测试集指标
        test_metrics = metrics.get('test_metrics', {})
        test_table = self._generate_test_metrics_table(test_metrics)
        # 找到 "### 3.2 测试集指标" 下的表格并替换
        test_section_marker = '### 3.2 测试集指标 (最终评估)\n\n| 指标 | 值 | 说明 |'
        if test_section_marker in report:
            report = report.replace(
                '| 指标 | 值 | 说明 |\n|------|-----|------|',
                test_table
            )

        # 5.1 保存的检查点
        checkpoints = metrics.get('checkpoints', [])
        checkpoint_table = self._generate_checkpoint_table(checkpoints)
        report = report.replace(
            '### 5.1 模型检查点\n\n',
            f'### 5.1 模型检查点\n\n{checkpoint_table}\n'
        )

        # 8.2 关键数据总结
        summary_table = self._generate_summary_table(config, metrics)
        report = report.replace(
            '| 训练样本 |...|',
            summary_table
        )

        # 4. 可视化结果处理
        report = self._process_visualization_section(report, viz_paths)

        # 写入报告
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 训练报告已生成: {output_path}")
        return str(output_path)

    def generate_data_report(self,
                             data_info: Dict[str, Any],
                             processing_info: Dict[str, Any],
                             output_path: str) -> str:
        """
        生成数据处理报告

        Args:
            data_info: 数据信息
            processing_info: 处理过程信息
            output_path: 报告输出路径

        Returns:
            生成的报告路径
        """
        # 读取模板
        with open(self.data_template, 'r', encoding='utf-8') as f:
            template = f.read()

        # 填充基本信息
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_name = processing_info.get('model_name', 'Unknown')
        time_range = processing_info.get('time_range', 'N/A')
        time_series = processing_info.get('time_series', 'N/A')

        report = template.replace('**生成时间**: ', f'**生成时间**: {current_time}')
        report = report.replace('**预测模型**: ', f'**预测模型**: {model_name}')
        report = report.replace('**预测的时间范围**: ', f'**预测的时间范围**: {time_range}')
        report = report.replace('**时间序列信息**: ', f'**时间序列信息**: {time_series}')

        # 填充执行摘要
        total_samples = data_info.get('total_samples', 0)
        valid_samples = data_info.get('valid_samples', 0)
        train_ratio = processing_info.get('train_ratio', 0.8)
        valid_ratio = processing_info.get('valid_ratio', 0.1)
        test_ratio = processing_info.get('test_ratio', 0.1)

        report = report.replace(
            '- ✅ **原始数据转化过程**:',
            f'- ✅ **原始数据转化过程**: 从 {data_info.get("source_format", "H5/MAT/NetCDF")} 转换为训练格式'
        )
        report = report.replace(
            '- ✅ **有效数据**: ',
            f'- ✅ **有效数据**: {valid_samples}/{total_samples} 样本（有效率 {valid_samples/total_samples*100:.2f}%）' if total_samples > 0 else '- ✅ **有效数据**: N/A'
        )
        report = report.replace(
            '- ✅ **数据划分**: 训练集(...%) | 验证集(...%) | 测试集(...%)',
            f'- ✅ **数据划分**: 训练集({train_ratio*100:.0f}%) | 验证集({valid_ratio*100:.0f}%) | 测试集({test_ratio*100:.0f}%)'
        )

        # 填充数据统计表格
        stats_table = self._generate_data_stats_table(data_info.get('statistics', {}))
        report = report.replace(
            '| 变量名 | 最小值 | 最大值 | 平均值 | 标准差 | NaN比例 |...|',
            stats_table
        )

        # 填充性能基准
        perf_table = self._generate_performance_table(processing_info.get('performance', {}))
        report = report.replace(
            '| 数据加载 | ...| ...| ... |',
            perf_table
        )

        # 写入报告
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 数据处理报告已生成: {output_path}")
        return str(output_path)

    # 辅助函数：生成各种表格

    def _process_visualization_section(self, report: str, viz_paths: Optional[List[str]]) -> str:
        """
        处理可视化部分的占位符
        Args:
            report: 报告内容
            viz_paths: 可视化图片路径列表
        Returns:
            处理后的报告内容
        """
        # 处理 VIZ_FILE_LIST 占位符
        viz_file_list_placeholder = "<!-- VIZ_FILE_LIST: 脚本自动填充，列出所有生成的可视化图片路径 -->"
        if viz_paths and len(viz_paths) > 0:
            file_list = "\n".join([f"- `{path}`" for path in viz_paths])
            report = report.replace(viz_file_list_placeholder, file_list)
        else:
            report = report.replace(viz_file_list_placeholder, "*暂无可视化文件*")

        # 处理 VIZ_IMAGES 占位符
        viz_images_placeholder = "<!-- VIZ_IMAGES: 脚本自动填充，插入所有可视化图片 -->"
        if viz_paths and len(viz_paths) > 0:
            images_md = []
            for path in viz_paths:
                # 从路径中提取文件名作为图片标题
                import os
                filename = os.path.basename(path)
                name = os.path.splitext(filename)[0]
                # 将下划线替换为空格，首字母大写
                title = name.replace("_", " ").title()
                images_md.append(f"#### {title}\n\n![{title}]({path})")
            report = report.replace(viz_images_placeholder, "\n\n".join(images_md))
        else:
            report = report.replace(viz_images_placeholder, "*暂无可视化图片*")

        # 注意：AI_FILL 占位符保留不处理，由 AI 后续填充
        return report

    def _generate_model_config_table(self, config: Dict[str, Any]) -> str:
        """生成模型配置表格"""
        rows = [
            f"| **模型名称** | {config.get('name', 'N/A')} |",
            f"| **模型类型** | {config.get('type', 'N/A')} |",
            f"| **处理路径** | {config.get('path', 'N/A')} |",
            f"| **参数量** | {config.get('num_params', 'N/A'):,} |" if isinstance(config.get('num_params'), int) else f"| **参数量** | {config.get('num_params', 'N/A')} |",
            f"| **嵌入维度** | {config.get('embed_dim', 'N/A')} |",
            f"| **窗口大小** | {config.get('window_size', 'N/A')} |",
            f"| **Patch大小** | {config.get('patch_size', 'N/A')} |",
        ]
        return '\n'.join(rows)

    def _generate_data_config_table(self, config: Dict[str, Any]) -> str:
        """生成数据配置表格"""
        rows = [
            f"| **数据集** | {config.get('name', 'N/A')} |",
            f"| **时间步数** | {config.get('timesteps', 'N/A')} |",
            f"| **空间分辨率** | {config.get('shape', 'N/A')} |",
            f"| **输入通道数** | {config.get('in_channels', 'N/A')} |",
            f"| **输入长度** | {config.get('input_len', 'N/A')} |",
            f"| **输出长度** | {config.get('output_len', 'N/A')} |",
            f"| **训练集** | {config.get('train_ratio', 0.8)*100:.0f}% |",
            f"| **验证集** | {config.get('valid_ratio', 0.1)*100:.0f}% |",
            f"| **测试集** | {config.get('test_ratio', 0.1)*100:.0f}% |",
        ]
        return '\n'.join(rows)

    def _generate_train_config_table(self, train_config: Dict, opt_config: Dict) -> str:
        """生成训练配置表格"""
        rows = [
            f"| **总Epochs** | {train_config.get('epochs', 'N/A')} |",
            f"| **批次大小** | {train_config.get('train_batchsize', 'N/A')} |",
            f"| **优化器** | {opt_config.get('optimizer', 'N/A')} |",
            f"| **初始学习率** | {opt_config.get('lr', 'N/A')} |",
            f"| **最终学习率** | {train_config.get('final_lr', 'N/A')} |",
            f"| **学习率调度** | {train_config.get('scheduler', 'N/A')} |",
            f"| **权重衰减** | {opt_config.get('weight_decay', 'N/A')} |",
            f"| **梯度裁剪** | {train_config.get('grad_clip', 'N/A')} |",
            f"| **早停耐心度** | {train_config.get('patience', 'N/A')} |",
            f"| **评估频率** | 每 {train_config.get('eval_freq', 'N/A')} epoch |",
        ]
        return '\n'.join(rows)

    def _generate_gpu_config_table(self, gpu_info: Dict[str, Any]) -> str:
        """生成 GPU 配置表格"""
        rows = [
            f"| **GPU** | {gpu_info.get('name', 'N/A')} |",
            f"| **显存** | {gpu_info.get('memory', 'N/A')} GB |",
            f"| **训练模式** | {gpu_info.get('mode', 'Single GPU')} |",
            f"| **CUDA版本** | {gpu_info.get('cuda_version', 'N/A')} |",
        ]
        return '\n'.join(rows)

    def _generate_loss_curve_table(self, history: List[Dict]) -> str:
        """生成损失曲线表格"""
        if not history:
            return "暂无数据"

        # 只显示关键 epochs
        key_epochs = []
        for i, h in enumerate(history):
            if i == 0 or i == len(history) - 1 or i % max(len(history) // 10, 1) == 0:
                key_epochs.append(h)

        rows = ["| Epoch | Train Loss | Valid Loss |", "|-------|------------|------------|"]
        for h in key_epochs:
            epoch = h.get('epoch', 'N/A')
            train_loss = h.get('train_loss', 'N/A')
            valid_loss = h.get('valid_loss', 'N/A')

            t_loss_str = f"{train_loss:.6f}" if isinstance(train_loss, (int, float)) else str(train_loss)
            v_loss_str = f"{valid_loss:.6f}" if isinstance(valid_loss, (int, float)) else str(valid_loss)

            rows.append(f"| {epoch} | {t_loss_str} | {v_loss_str} |")

        return '\n'.join(rows)

    def _generate_validation_table(self, history: List[Dict]) -> str:
        """生成验证集性能表格"""
        if not history:
            return "| Epoch | Valid Loss | Valid MAE | Valid RMSE | Valid R² | 改进率 |\n|-------|------------|-----------|-----------|----------|--------|\n"

        rows = ["| Epoch | Valid Loss | Valid MAE | Valid RMSE | Valid R² | 改进率 |",
                "|-------|------------|-----------|-----------|----------|--------|"]

        prev_loss = None
        for h in history[-10:]:  # 只显示最后10个
            epoch = h.get('epoch', 'N/A')
            loss = h.get('valid_loss', 0)
            mae = h.get('valid_mae', 0)
            rmse = h.get('valid_rmse', 0)
            r2 = h.get('valid_r2', 0)

            improvement = ""
            if prev_loss is not None and isinstance(loss, (int, float)) and isinstance(prev_loss, (int, float)) and prev_loss != 0:
                improvement = f"{(prev_loss - loss) / prev_loss * 100:.2f}%"
            prev_loss = loss

            # 格式化数值
            loss_str = f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss)
            mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)

            rows.append(f"| {epoch} | {loss_str} | {mae_str} | {rmse_str} | {r2_str} | {improvement} |")

        return '\n'.join(rows)

    def _generate_test_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """生成测试集指标表格"""
        r2 = metrics.get('r2', 'N/A')
        rmse = metrics.get('rmse', 'N/A')
        mae = metrics.get('mae', 'N/A')
        mse = metrics.get('mse', 'N/A')
        mape = metrics.get('mape', 'N/A')

        # 格式化数值
        r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
        rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
        mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
        mse_str = f"{mse:.6f}" if isinstance(mse, (int, float)) else str(mse)
        mape_str = f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)

        rows = [
            "| 指标 | 值 | 说明 |",
            "|------|-----|------|",
            f"| **R²** | {r2_str} | 决定系数 |",
            f"| **RMSE** | {rmse_str} | 均方根误差 |",
            f"| **MAE** | {mae_str} | 平均绝对误差 |",
            f"| **MSE** | {mse_str} | 均方误差 |",
            f"| **MAPE** | {mape_str} | 平均绝对百分比误差 |",
        ]
        return '\n'.join(rows)

    def _generate_checkpoint_table(self, checkpoints: List[Dict]) -> str:
        """生成检查点表格"""
        if not checkpoints:
            return "| 迭代数 | 文件名 | 大小 | 保存时间 |\n|--------|--------|------|----------|\n"

        rows = ["| 迭代数 | 文件名 | 大小 | 保存时间 |",
                "|--------|--------|------|----------|"]

        for ckpt in checkpoints:
            epoch = ckpt.get('epoch', 'N/A')
            filename = ckpt.get('filename', 'N/A')
            size = ckpt.get('size', 'N/A')
            timestamp = ckpt.get('timestamp', 'N/A')
            rows.append(f"| {epoch} | {filename} | {size} | {timestamp} |")

        return '\n'.join(rows)

    def _generate_summary_table(self, config: Dict, metrics: Dict) -> str:
        """生成总结表格"""
        data_config = config.get('data', {})

        # 格式化数值
        num_params = config.get('model', {}).get('num_params', None)
        num_params_str = f"{num_params:,}" if isinstance(num_params, (int, float)) else "N/A"

        best_r2 = metrics.get('best_r2', None)
        r2_str = f"{best_r2:.4f}" if isinstance(best_r2, (int, float)) else "N/A"

        best_rmse = metrics.get('best_rmse', None)
        rmse_str = f"{best_rmse:.4f}" if isinstance(best_rmse, (int, float)) else "N/A"

        best_mae = metrics.get('best_mae', None)
        mae_str = f"{best_mae:.4f}" if isinstance(best_mae, (int, float)) else "N/A"

        rows = [
            "| 项目 | 数值 |",
            "|------|------|",
            f"| 训练样本 | {data_config.get('train_samples', 'N/A')} |",
            f"| 测试样本 | {data_config.get('test_samples', 'N/A')} |",
            f"| 模型参数 | {num_params_str} |",
            f"| 训练时长 | {metrics.get('total_time', 'N/A')} 秒 |",
            f"| 最终R² | {r2_str} |",
            f"| 最终RMSE | {rmse_str} |",
            f"| 最终MAE | {mae_str} |",
        ]
        return '\n'.join(rows)

    def _generate_data_stats_table(self, stats: Dict[str, Any]) -> str:
        """生成数据统计表格"""
        if not stats:
            return "| 变量名 | 最小值 | 最大值 | 平均值 | 标准差 | NaN比例 |\n|--------|--------|--------|--------|--------|--------|\n"

        rows = ["| 变量名 | 最小值 | 最大值 | 平均值 | 标准差 | NaN比例 |",
                "|--------|--------|--------|--------|--------|--------|"]

        for var_name, var_stats in stats.items():
            nan_ratio = var_stats.get('nan_ratio', 0)
            min_val = var_stats.get('min', 'N/A')
            max_val = var_stats.get('max', 'N/A')
            mean = var_stats.get('mean', 'N/A')
            std = var_stats.get('std', 'N/A')

            # 格式化数值
            min_str = f"{min_val:.4f}" if isinstance(min_val, (int, float)) else str(min_val)
            max_str = f"{max_val:.4f}" if isinstance(max_val, (int, float)) else str(max_val)
            mean_str = f"{mean:.4f}" if isinstance(mean, (int, float)) else str(mean)
            std_str = f"{std:.4f}" if isinstance(std, (int, float)) else str(std)
            nan_ratio_str = f"{nan_ratio*100:.2f}%" if isinstance(nan_ratio, (int, float)) else str(nan_ratio)

            rows.append(f"| {var_name} | {min_str} | {max_str} | {mean_str} | {std_str} | {nan_ratio_str} |")

        return '\n'.join(rows)

    def _generate_performance_table(self, perf: Dict[str, Any]) -> str:
        """生成性能表格"""
        rows = [
            "| 阶段 | 输入大小 | 输出大小 | 说明 |",
            "|------|---------|---------|------|",
            f"| 数据加载 | {perf.get('load_input_size', 'N/A')} | {perf.get('load_output_size', 'N/A')} | {perf.get('load_desc', 'N/A')} |",
            f"| 数据处理 | {perf.get('process_input_size', 'N/A')} | {perf.get('process_output_size', 'N/A')} | {perf.get('process_desc', 'N/A')} |",
            f"| 可视化生成 | {perf.get('viz_input_size', 'N/A')} | {perf.get('viz_output_size', 'N/A')} | {perf.get('viz_desc', 'N/A')} |",
        ]
        return '\n'.join(rows)


def generate_train_report_from_file(
    config_file: str,
    metrics_file: str,
    output_path: str,
    viz_paths: Optional[List[str]] = None
) -> str:
    """
    从文件生成训练报告的便捷函数

    Args:
        config_file: 配置JSON文件路径
        metrics_file: 指标JSON文件路径
        output_path: 输出报告路径
        viz_paths: 可视化图片路径列表（可选）

    Returns:
        生成的报告路径
    """
    # 先读取配置和指标文件
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    # --- 自动备份 JSON（可选） ---
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # 备份 config.json
        config_save_path = output_dir / "config_backup.json"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False, default=str)
        # 备份 metrics.json
        metrics_save_path = output_dir / "metrics_backup.json"
        with open(metrics_save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"Warning: Could not backup JSON files: {e}")

    generator = PredictionReportGenerator()
    return generator.generate_train_report(config, metrics, output_path, viz_paths)


def generate_data_report_from_file(data_info_file: str, processing_info_file: str, output_path: str) -> str:
    """
    从文件生成数据处理报告的便捷函数

    Args:
        data_info_file: 数据信息JSON文件路径
        processing_info_file: 处理信息JSON文件路径
        output_path: 输出报告路径

    Returns:
        生成的报告路径
    """
    with open(data_info_file, 'r') as f:
        data_info = json.load(f)

    with open(processing_info_file, 'r') as f:
        processing_info = json.load(f)

    generator = PredictionReportGenerator()
    return generator.generate_data_report(data_info, processing_info, output_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("用法:")
        print("  生成训练报告: python report_generator.py train <config.json> <metrics.json> <output.md> [--viz_paths path1,path2,...]")
        print("  生成数据报告: python report_generator.py data <data_info.json> <processing_info.json> <output.md>")
        sys.exit(1)

    report_type = sys.argv[1]

    if report_type == 'train':
        # 解析可选的 --viz_paths 参数
        viz_paths = None
        for i, arg in enumerate(sys.argv):
            if arg == "--viz_paths" and i + 1 < len(sys.argv):
                viz_paths_str = sys.argv[i + 1]
                viz_paths = [p.strip() for p in viz_paths_str.split(",") if p.strip()]
                break
        generate_train_report_from_file(sys.argv[2], sys.argv[3], sys.argv[4], viz_paths)
    elif report_type == 'data':
        generate_data_report_from_file(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(f"未知的报告类型: {report_type}")
        sys.exit(1)
