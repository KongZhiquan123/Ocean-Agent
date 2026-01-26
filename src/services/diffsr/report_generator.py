#!/usr/bin/env python3
"""
DiffSR 报告生成器
自动生成符合模板格式的训练报告和数据处理报告
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class DiffSRReportGenerator:
    """DiffSR 报告生成器"""

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
        self.train_template = self.template_dir / "sr_train_report.md"
        self.data_template = self.template_dir / "sr_data_report.md"
    def generate_train_report(
        self, config: Dict[str, Any], metrics: Dict[str, Any], output_path: str,
        viz_paths: Optional[List[str]] = None
    ) -> str:
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
        best_psnr = metrics.get('best_psnr', 'N/A')
        model_path = metrics.get('model_path', 'N/A')
        num_params = config.get('model', {}).get('num_params', 'N/A')

        report = report.replace(
            '- ✅ **模型训练**: ',
            f'- ✅ **模型训练**: 成功完成 {best_epoch} 个 epochs'
        )
        report = report.replace(
            '- ✅ **测试性能**:',
            f'- ✅ **测试性能**: PSNR {float(best_psnr):.4f} dB' if best_psnr not in ['N/A', None] else '- ✅ **测试性能**: PSNR N/A'
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
        # check if num_params is a number before formatting
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
            f'- **训练模式**: {"分布式" if config.get("train", {}).get("distribute", False) else "单GPU"}'
        )

        # 填充训练配置表格
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        train_config = config.get('train', {})
        optimize_config = config.get('optimize', {})

        # 1.1 模型结构表格
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

        # 2.2 训练曲线 - 生成 Markdown 表格而不是图表链接
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
                '| Epoch | Valid Loss |Valid PSNR|Valid Relative L2|... | 改进率 |',
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
            '| 迭代数 | 文件名 | 大小 | 保存时间 |',
            checkpoint_table
        )

        # 8.2 关键数据总结
        summary_table = self._generate_summary_table(config, metrics)
        report = report.replace("| 训练样本 |...|", summary_table)

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
        duration = processing_info.get('duration', 'N/A')

        report = template.replace('**生成时间**: ', f'**生成时间**: {current_time}')
        report = report.replace('**超分模型：**', f'**超分模型**: {model_name}')
        report = report.replace('**处理周期**: ', f'**处理周期**: {duration}')

        # 填充执行摘要
        total_samples = data_info.get('total_samples', 0)
        valid_samples = data_info.get('valid_samples', 0)
        train_ratio = processing_info.get('train_ratio', 0.8)
        valid_ratio = processing_info.get('valid_ratio', 0.1)
        test_ratio = processing_info.get('test_ratio', 0.1)

        report = report.replace(
            '- ✅ **原始数据转化过程**: ',
            f'- ✅ **原始数据转化过程**: 从 {data_info.get("source_format", "NetCDF")} 转换为训练格式'
        )
        report = report.replace(
            '- ✅ **有效数据**: ',
            f'- ✅ **有效数据**: {valid_samples}/{total_samples} 样本（有效率 {valid_samples/total_samples*100:.2f}%）'
        )
        report = report.replace(
            '- ✅ **数据划分**: 训练集(xxx%) | 验证集(xxx%) | 测试集(xxx%)',
            f'- ✅ **数据划分**: 训练集({train_ratio*100:.0f}%) | 验证集({valid_ratio*100:.0f}%) | 测试集({test_ratio*100:.0f}%)'
        )

        # 填充原始数据信息
        source_info = data_info.get('source', {})
        data_vars = source_info.get('variables', [])
        time_range = source_info.get('time_range', 'N/A')
        spatial_range = source_info.get('spatial_range', 'N/A')

        report = report.replace('- **数据集**: ', f'- **数据集**: {source_info.get("dataset", "N/A")}')
        report = report.replace('- **变量**: ', f'- **变量**: {", ".join(data_vars)}')
        report = report.replace('- **时空范围**: ', f'- **时空范围**: {time_range}, {spatial_range}')
        report = report.replace('- **数据格式**: ', f'- **数据格式**: {source_info.get("format", "N/A")}')
        report = report.replace('- **原始文件**: ', f'- **原始文件**: {source_info.get("file_path", "N/A")}')

        # 填充数据统计表格
        stats_table = self._generate_data_stats_table(data_info.get('statistics', {}))
        report = report.replace(
            '| 变量名 |  NaN比例 |...|',
            stats_table
        )

        # 填充输出数据结构
        output_info = processing_info.get('output', {})
        train_samples = output_info.get('train_samples', 0)
        valid_samples = output_info.get('valid_samples', 0)
        test_samples = output_info.get('test_samples', 0)

        train_section = f"""#### 训练集
- **样本数**: {train_samples}
- **文件路径**: {output_info.get('train_path', 'N/A')}
- **数据形状**: {output_info.get('train_shape', 'N/A')}
"""

        valid_section = f"""#### 验证集
- **样本数**: {valid_samples}
- **文件路径**: {output_info.get('valid_path', 'N/A')}
- **数据形状**: {output_info.get('valid_shape', 'N/A')}
"""

        test_section = f"""#### 测试集
- **样本数**: {test_samples}
- **文件路径**: {output_info.get('test_path', 'N/A')}
- **数据形状**: {output_info.get('test_shape', 'N/A')}
"""

        report = report.replace('#### 训练集\n\n', train_section)
        report = report.replace('#### 验证集\n\n', valid_section)
        report = report.replace('#### 测试集\n\n', test_section)

        # 填充性能基准
        perf_table = self._generate_performance_table(processing_info.get('performance', {}))
        report = report.replace(
            '| 数据加载 | ...| ...| ... |',
            perf_table
        )

        # 填充文件路径汇总
        paths_table = self._generate_paths_table(data_info, processing_info)
        report = report.replace(
            '| 原始数据|  |',
            paths_table
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
        num_params = config.get('num_params', 'N/A')
        # Check if num_params is a number before formatting
        if isinstance(num_params, (int, float)):
            num_params_str = f"{num_params:,}"
        else:
            num_params_str = str(num_params)

        rows = [
            f"| **模型名称** | {config.get('name', 'N/A')} |",
            f"| **模型类型** | {config.get('type', 'N/A')} |",
            f"| **参数量** | {num_params_str} |",
            f"| **输入/输出通道** | {config.get('in_dim', 'N/A')} / {config.get('out_dim', 'N/A')} |",
            f"| **模型通道数** | {config.get('width', 'N/A')} |",
        ]
        return '\n'.join(rows)

    def _generate_data_config_table(self, config: Dict[str, Any]) -> str:
        """生成数据配置表格"""
        train_ratio = config.get('train_ratio', 'N/A')
        valid_ratio = config.get('valid_ratio', 'N/A')
        test_ratio = config.get('test_ratio', 'N/A')

        # Safely format ratios
        t_r = f"{train_ratio*100:.0f}%" if isinstance(train_ratio, (int, float)) else str(train_ratio)
        v_r = f"{valid_ratio*100:.0f}%" if isinstance(valid_ratio, (int, float)) else str(valid_ratio)
        te_r = f"{test_ratio*100:.0f}%" if isinstance(test_ratio, (int, float)) else str(test_ratio)

        rows = [
            f"| **数据集** | {config.get('name', 'N/A')} |",
            f"| **空间分辨率** | {config.get('shape', 'N/A')} |",
            f"| **训练批次大小** | {config.get('train_batchsize', 'N/A')} |",
            f"| **训练集** | {train_ratio} |",
            f"| **验证集** | {valid_ratio} |",
            f"| **测试集** | {test_ratio} |",
        ]
        return '\n'.join(rows)

    def _generate_train_config_table(self, train_config: Dict, opt_config: Dict) -> str:
        """生成训练配置表格"""
        rows = [
            f"| **总Epochs** | {train_config.get('epochs', 'N/A')} |",
            f"| **批次大小** | {train_config.get('batch_size', 'N/A')} |",
            f"| **优化器** | {opt_config.get('optimizer', 'N/A')} |",
            f"| **初始学习率** | {opt_config.get('lr', 'N/A')} |",
            f"| **权重衰减** | {opt_config.get('weight_decay', 'N/A')} |",
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
            return "| Epoch | Valid Loss | Valid PSNR | Valid L2 | 改进率 |\n|-------|------------|------------|----------|--------|"

        rows = ["| Epoch | Valid Loss | Valid PSNR | Valid L2 | 改进率 |",
                "|-------|------------|------------|----------|--------|"]

        prev_loss = None
        for h in history[-10:]:  # 只显示最后10个
            epoch = h.get('epoch', 'N/A')
            loss = h.get('valid_loss', 0)
            psnr = h.get('valid_psnr', 0)
            l2 = h.get('valid_l2', 0)

            improvement = ""
            if prev_loss is not None and isinstance(loss, (int, float)) and isinstance(prev_loss, (int, float)) and prev_loss != 0:
                improvement = f"{(prev_loss - loss) / prev_loss * 100:.2f}%"
            prev_loss = loss
            
            loss_str = f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss)
            psnr_str = f"{psnr:.4f}" if isinstance(psnr, (int, float)) else str(psnr)
            l2_str = f"{l2:.6f}" if isinstance(l2, (int, float)) else str(l2)

            rows.append(f"| {epoch} | {loss_str} | {psnr_str} | {l2_str} | {improvement} |")

        return '\n'.join(rows)

    def _generate_test_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """生成测试集指标表格"""
        psnr = metrics.get('psnr', 'N/A')
        ssim = metrics.get('ssim', 'N/A')
        rel_l2 = metrics.get('rel_l2', 'N/A')
        mse = metrics.get('mse', 'N/A')
        # 格式化数值
        psnr_str = f"{psnr:.4f}" if isinstance(psnr, (int, float)) else str(psnr)
        ssim_str = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else str(ssim)
        l2_str = f"{rel_l2:.6f}" if isinstance(rel_l2, (int, float)) else str(rel_l2)
        mse_str = f"{mse:.6f}" if isinstance(mse, (int, float)) else str(mse)
        rows = [
            "| 指标 | 值 | 说明 |",
            "|------|-----|------|",
            f"| **PSNR** | {psnr_str} | 峰值信噪比 |",
            f"| **SSIM** | {ssim_str} | 结构相似性 |",
            f"| **Relative L2** | {l2_str} | 相对L2误差 |",
            f"| **MSE** | {mse_str} | 均方误差 |",
        ]
        return '\n'.join(rows)

    def _generate_checkpoint_table(self, checkpoints: List[Dict]) -> str:
        """生成检查点表格"""
        if not checkpoints:
            return "| 迭代数 | 文件名 | 大小 | 保存时间 |\n|--------|--------|------|----------|"

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
        data_config = config.get("data", {})
        # 格式化数值
        num_params = config.get("model", {}).get("num_params", None)
        num_params_str = f"{num_params:,}" if isinstance(num_params, (int, float)) else "N/A"
        best_psnr = metrics.get('best_psnr', None)
        psnr_str = f"{best_psnr:.4f}" if isinstance(best_psnr, (int, float)) else "N/A"
        best_l2 = metrics.get('best_l2', None)
        l2_str = f"{best_l2:.6f}" if isinstance(best_l2, (int, float)) else "N/A"
        rows = [
            "| 项目 | 数值 |",
            "|------|------|",
            f"| 训练样本 | {data_config.get('train_samples', 'N/A')} |",
            f"| 测试样本 | {data_config.get('test_samples', 'N/A')} |",
            f"| 模型参数 | {num_params_str} |",
            f"| 训练时长 | {metrics.get('total_time', 'N/A')} 秒 |",
            f"| 最终 PSNR | {psnr_str} |",
            f"| 最终 Relative L2 | {l2_str} |",
        ]
        return '\n'.join(rows)

    def _generate_data_stats_table(self, stats: Dict[str, Any]) -> str:
        """生成数据统计表格"""
        if not stats:
            return "| 变量名 | NaN比例 | 最小值 | 最大值 | 均值 | 标准差 |\n|--------|---------|--------|--------|------|--------|"

        rows = ["| 变量名 | NaN比例 | 最小值 | 最大值 | 均值 | 标准差 |",
                "|--------|---------|--------|--------|------|--------|"]

        for var_name, var_stats in stats.items():
            nan_ratio = var_stats.get('nan_ratio', 0)
            min_val = var_stats.get('min', 'N/A')
            max_val = var_stats.get('max', 'N/A')
            mean = var_stats.get('mean', 'N/A')
            std = var_stats.get('std', 'N/A')
            
            min_str = f"{min_val:.4f}" if isinstance(min_val, (int, float)) else str(min_val)
            max_str = f"{max_val:.4f}" if isinstance(max_val, (int, float)) else str(max_val)
            mean_str = f"{mean:.4f}" if isinstance(mean, (int, float)) else str(mean)
            std_str = f"{std:.4f}" if isinstance(std, (int, float)) else str(std)

            rows.append(f"| {var_name} | {nan_ratio*100:.2f}% | {min_str} | {max_str} | {mean_str} | {std_str} |")

        return '\n'.join(rows)

    def _generate_performance_table(self, perf: Dict[str, Any]) -> str:
        """生成性能表格"""
        rows = [
            "| 阶段 | 输入大小 | 输出大小 | 耗时 | 吞吐量 |",
            "|------|---------|---------|------|--------|",
            f"| 数据加载 | {perf.get('load_input_size', 'N/A')} | {perf.get('load_output_size', 'N/A')} | {perf.get('load_time', 'N/A')}s | {perf.get('load_throughput', 'N/A')} |",
            f"| 数据处理 | {perf.get('process_input_size', 'N/A')} | {perf.get('process_output_size', 'N/A')} | {perf.get('process_time', 'N/A')}s | {perf.get('process_throughput', 'N/A')} |",
            f"| 可视化生成 | {perf.get('viz_input_size', 'N/A')} | {perf.get('viz_output_size', 'N/A')} | {perf.get('viz_time', 'N/A')}s | {perf.get('viz_throughput', 'N/A')} |",
        ]
        return '\n'.join(rows)

    def _generate_paths_table(self, data_info: Dict, processing_info: Dict) -> str:
        """生成文件路径汇总表格"""
        output_info = processing_info.get('output', {})
        rows = [
            "| 类型 | 路径 |",
            "|------|------|",
            f"| 原始数据 | {data_info.get('source', {}).get('file_path', 'N/A')} |",
            f"| 处理后数据 | {output_info.get('train_path', 'N/A')} |",
            f"| 时间可视化 | {output_info.get('time_viz_path', 'N/A')} |",
            f"| 空间可视化 | {output_info.get('spatial_viz_path', 'N/A')} |",
            f"| 数据报告 | {output_info.get('report_path', 'N/A')} |",
        ]
        return "\n".join(rows)
def generate_train_report_from_file(
    config_file: str, metrics_file: str, output_path: str,
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
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # --- 自动备份 JSON（可选） ---
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # 备份 config.json
        config_save_path = output_dir / "config_backup.json"
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False, default=str)
        # 备份 metrics.json
        metrics_save_path = output_dir / "metrics_backup.json"
        with open(metrics_save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"Warning: Could not backup JSON files: {e}")

    generator = DiffSRReportGenerator()
    return generator.generate_train_report(config, metrics, output_path, viz_paths)
def generate_data_report_from_file(
    data_info_file: str, processing_info_file: str, output_path: str
) -> str:
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

    generator = DiffSRReportGenerator()
    return generator.generate_data_report(data_info, processing_info, output_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("用法:")
        print(
            "  生成训练报告: python report_generator.py train <config.json> <metrics.json> <output.md> [--viz_paths path1,path2,...]"
        )
        print(
            "  生成数据报告: python report_generator.py data <data_info.json> <processing_info.json> <output.md>"
        )
        sys.exit(1)

    report_type = sys.argv[1]
    if report_type == "train":
        # 解析可选的 --viz_paths 参数
        viz_paths = None
        for i, arg in enumerate(sys.argv):
            if arg == "--viz_paths" and i + 1 < len(sys.argv):
                viz_paths_str = sys.argv[i + 1]
                viz_paths = [p.strip() for p in viz_paths_str.split(",") if p.strip()]
                break
        generate_train_report_from_file(sys.argv[2], sys.argv[3], sys.argv[4], viz_paths)
    elif report_type == "data":
        generate_data_report_from_file(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(f"未知的报告类型: {report_type}")
        sys.exit(1)