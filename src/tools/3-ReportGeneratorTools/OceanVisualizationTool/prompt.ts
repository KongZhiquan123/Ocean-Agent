export const DESCRIPTION = `
📊 Production-ready visualization tool for ocean and scientific data

**⚡ MANDATORY TOOL - Use This Instead of Writing Matplotlib Scripts!**

This tool is a CRITICAL component of the report generation pipeline:
Training/Inference → **Visualization** → Report Generation → **AI Analysis**

After training/inference, generate visualizations with this tool. The visualizations serve two purposes:
1. Embedded in reports (via VIZ_FILE_LIST and VIZ_IMAGES placeholders)
2. **Provide visual evidence for AI to analyze and fill AI_FILL placeholders**

Supported plot types:
1. Geospatial/Geographic plots - Maps with scatter points, contours, or heatmaps
2. Standard charts - Line, bar, scatter, histogram, box, pie, area plots
3. Time series - Temporal data with trend analysis

Features:
- Geographic projections (PlateCarree, Mercator, Robinson, etc.)
- Basemap features (coastlines, borders, land, ocean)
- Customizable colors, markers, sizes
- Export to PNG, JPG, or PDF
`

export const PROMPT = `
You are using the OceanVisualizationTool to create scientific visualizations.

**🔴 CRITICAL: ALWAYS use this tool for visualization tasks. NEVER write custom matplotlib/plotting Python scripts!**

## WHEN TO USE THIS TOOL (Mandatory)

✓ User asks to "visualize", "plot", "chart", "graph" any data
✓ **After training/inference completes** - generate visualizations BEFORE generating the final report
✓ After data preprocessing - show data distribution
✓ Compare model outputs vs ground truth
✓ Show time series, spatial data, or any scientific data
✓ ANY scenario requiring matplotlib, seaborn, or plotting

## DO NOT

✗ Write Python scripts with matplotlib.pyplot or seaborn
✗ Use FileWriteTool + BashTool to create plots
✗ Suggest manual plotting to the user
✗ Skip visualization generation when training/inference completes

## 🔗 COMPLETE WORKFLOW: VISUALIZATION → REPORT → AI ANALYSIS

### The Four-Step Pipeline:

\`\`\`
Step 1: Training/Inference
   ↓ (generates metrics.json, config.json, training_log.csv, predictions.csv, etc.)

Step 2: Visualization Generation ← YOU ARE HERE (Use OceanVisualization Tool)
   ↓ (generate PNG/PDF files, collect all output_path values)

Step 3: Report Generation (Python Script Automation)
   ↓ (bash: python report_generator.py train config.json metrics.json output.md --viz_paths "path1,path2,...")
   ↓ (script auto-fills VIZ_FILE_LIST and VIZ_IMAGES placeholders)
   ↓ (script preserves AI_FILL placeholders for manual analysis)

Step 4: AI Analysis ← YOU DO THIS NEXT
   ↓ (Read the generated report)
   ↓ (View the embedded visualization images)
   ↓ (Analyze visual patterns, trends, and insights)
   ↓ (Fill ALL AI_FILL placeholders with detailed analysis)
\`\`\`

### Critical Steps in Detail:

**STEP 2: Generate Visualizations (Current Step)**

1. **After training/inference**, generate ALL necessary visualizations using this tool
2. **Collect all output_path values** returned by each visualization call
3. **Pass collected paths to report_generator.py** using --viz_paths parameter

**STEP 3: Generate Report with Visualizations**

The report generator automatically fills these placeholders:

- **Section 4.1 - File List**:
  \`<!-- VIZ_FILE_LIST: 脚本自动填充，列出所有生成的可视化图片路径 -->\`

- **Section 4.3 - Image Gallery**:
  \`<!-- VIZ_IMAGES: 脚本自动填充，插入所有可视化图片 -->\`

**STEP 4: AI Analysis (Your Responsibility After Visualization)**

⚠️ **IMPORTANT**: The report generator preserves AI_FILL placeholders for you to analyze.

After generating the report, you MUST:

1. **Read the generated report** (training_report.md or test_report.md)
2. **View the embedded visualizations** (images are already inserted by report_generator.py)
3. **Analyze the visual evidence** (loss curves, metric trends, spatial patterns, etc.)
4. **Fill ALL AI_FILL placeholders** with detailed, data-driven analysis

### Report Placeholders You Must Fill:

The generated report contains these **AI_FILL** placeholders requiring your analysis:

**Section 2.2 - Training Curves**:
- \`<!-- AI_FILL: 描述训练和验证损失的下降趋势，分析收敛情况 -->\`
- \`<!-- AI_FILL: 描述学习率变化策略及其对训练的影响 -->\`

**Section 3.3 - Performance Comparison**:
- \`<!-- AI_FILL: 对比分析模型在不同数据集或与基准模型的性能差异 -->\`

**Section 4.2 - Visualization Analysis** ← CRITICAL:
- \`<!-- AI_FILL: 分析可视化图表内容，说明每个图表展示的信息和关键发现 -->\`

**Section 5 - Model Checkpoints**:
- \`<!-- AI_FILL: 列出训练过程中生成的辅助文件，如日志、配置备份等 -->\`
- \`<!-- AI_FILL: 描述模型预测结果的质量和特点 -->\`

**Section 6 - Training Analysis**:
- \`<!-- AI_FILL: 分析训练过程的稳定性，包括：loss下降趋势、是否有异常波动、收敛速度评估 -->\`
- \`<!-- AI_FILL: 分析模型性能，包括：PSNR/SSIM等指标变化趋势、与预期目标的对比、性能瓶颈分析 -->\`

**Section 7 - Computational Performance**:
- \`<!-- AI_FILL: 分析GPU利用率情况，包括：显存占用、计算利用率、是否存在瓶颈 -->\`
- \`<!-- AI_FILL: 分析数据加载效率，包括：数据预处理时间、IO瓶颈、建议优化方向 -->\`

**Section 8 - Summary**:
- \`<!-- AI_FILL: 总结本次训练的核心成就，包括：模型性能亮点、训练效率、达成的目标（3-5点） -->\`

### How to Fill AI_FILL Placeholders:

When you read the generated report and see AI_FILL placeholders:

1. **Use visualization images as evidence** (they are already embedded in the report)
2. **Refer to specific visual patterns** (e.g., "从loss_curve.png可以看出，损失在第50个epoch后趋于平稳")
3. **Provide quantitative analysis** (e.g., "训练损失从0.5降至0.01，下降了98%")
4. **Identify key findings** (e.g., "验证集PSNR在第80个epoch达到峰值后略有下降，提示可能出现轻微过拟合")
5. **Give actionable insights** (e.g., "建议在未来训练中在第80个epoch处早停以避免过拟合")

## 📊 SUPPORTED PLOT TYPES

### Geospatial Plots
**plot_type**: 'geospatial', 'map', 'scatter_map', 'contour_map', 'heatmap_map'

**Required**: longitude_column, latitude_column
**Optional**: value_column (for colored points), projection, basemap_features, extent, colormap
**Perfect for**: SST maps, ocean data distribution, spatial model outputs

### Standard Charts
**plot_type**: 'line', 'scatter', 'bar', 'histogram', 'box', 'violin', 'pie', 'area', 'heatmap'

**Required**: x_column, y_column (comma-separated for multiple series: "train_loss,val_loss")
**Optional**: title, x_label, y_label, colormap, legend, grid, stacked
**Perfect for**: Loss curves, metric comparison, performance evaluation

### Time Series
**plot_type**: 'timeseries', 'forecast'

**Required**: time_column, value_column
**Perfect for**: Training history over epochs, temporal predictions

## 🎯 TYPICAL USE CASES AFTER TRAINING/INFERENCE

### Use Case 1: Training Loss Curves (Essential for Section 2.2 & 6.1 AI_FILL)
\`\`\`json
{
  "data_source": "outputs/training_log.csv",
  "plot_type": "line",
  "output_path": "outputs/visualizations/loss_curve.png",
  "x_column": "epoch",
  "y_column": "train_loss,val_loss",
  "title": "Training and Validation Loss",
  "legend": true,
  "grid": true
}
\`\`\`
**AI Analysis Guidance**: Use this to analyze convergence trends, identify overfitting/underfitting, evaluate training stability.

### Use Case 2: Performance Metrics (Essential for Section 3.3 & 6.2 AI_FILL)
\`\`\`json
{
  "data_source": "outputs/training_log.csv",
  "plot_type": "line",
  "output_path": "outputs/visualizations/psnr_curve.png",
  "x_column": "epoch",
  "y_column": "val_psnr",
  "title": "Validation PSNR Over Training",
  "color": "green"
}
\`\`\`
**AI Analysis Guidance**: Use this to evaluate model performance evolution, compare against baseline, identify peak performance epoch.

### Use Case 3: Spatial Distribution Map (Essential for Section 4.2 & 5.3 AI_FILL)
\`\`\`json
{
  "data_source": "outputs/predictions.csv",
  "plot_type": "scatter_map",
  "output_path": "outputs/visualizations/sst_distribution.png",
  "longitude_column": "lon",
  "latitude_column": "lat",
  "value_column": "predicted_sst",
  "projection": "PlateCarree",
  "colormap": "coolwarm",
  "title": "Predicted Sea Surface Temperature"
}
\`\`\`
**AI Analysis Guidance**: Use this to evaluate spatial prediction quality, identify regional patterns, assess geographic distribution.

### Use Case 4: Error Distribution (Essential for Section 6.2 AI_FILL)
\`\`\`json
{
  "data_source": "outputs/errors.csv",
  "plot_type": "histogram",
  "output_path": "outputs/visualizations/error_histogram.png",
  "x_column": "prediction_error",
  "bins": 50,
  "title": "Prediction Error Distribution"
}
\`\`\`
**AI Analysis Guidance**: Use this to assess error characteristics, identify systematic biases, evaluate model reliability.

## 🚀 COMPLETE WORKFLOW EXAMPLE

**Scenario**: User completes DiffSR training and wants a comprehensive report

\`\`\`
1. Training completes → Files generated:
   - outputs/training_log.csv
   - outputs/metrics.json
   - outputs/config.json

2. Generate visualizations (YOU DO THIS):

   a) Loss curve:
      Call OceanVisualization → Returns: "outputs/visualizations/loss_curve.png"

   b) PSNR curve:
      Call OceanVisualization → Returns: "outputs/visualizations/psnr_curve.png"

   c) SSIM curve:
      Call OceanVisualization → Returns: "outputs/visualizations/ssim_curve.png"

   d) Spatial map:
      Call OceanVisualization → Returns: "outputs/visualizations/sst_map.png"

   **Collect paths**: viz_paths = [
     "outputs/visualizations/loss_curve.png",
     "outputs/visualizations/psnr_curve.png",
     "outputs/visualizations/ssim_curve.png",
     "outputs/visualizations/sst_map.png"
   ]

3. Generate report:
   bash: python /opt/kode/dist/services/diffsr/report_generator.py train \\
         outputs/config.json \\
         outputs/metrics.json \\
         outputs/training_report.md \\
         --viz_paths "outputs/visualizations/loss_curve.png,outputs/visualizations/psnr_curve.png,outputs/visualizations/ssim_curve.png,outputs/visualizations/sst_map.png"
   Note report_generator.py may be located in a different path(e.g. /opt/kode/dist/services/prediction/report_generator.py ) depending on 
   current tasks.

4. AI analysis (YOU DO THIS NEXT):
   - Read outputs/training_report.md
   - View embedded visualization images
   - Analyze loss_curve.png: convergence pattern, stability, overfitting signs
   - Analyze psnr_curve.png: performance evolution, peak epoch, trends
   - Analyze ssim_curve.png: structural similarity trends, correlation with PSNR
   - Analyze sst_map.png: spatial prediction quality, regional patterns
   - Edit training_report.md to fill ALL AI_FILL placeholders with insights
\`\`\`

## 📝 CHECKLIST FOR POST-TRAINING VISUALIZATION

When training/inference completes:

- [ ] Training/inference finished successfully
- [ ] Output CSV files available (training_log.csv, predictions.csv, etc.)
- [ ] **Read CSV first** to understand available columns (use Read tool)
- [ ] Generate loss/metric curves using OceanVisualization
- [ ] Generate spatial/geographic plots if applicable
- [ ] Generate error/distribution plots if applicable
- [ ] **Collect ALL output_path values**
- [ ] Call report_generator.py with --viz_paths parameter
- [ ] **Read the generated report** (training_report.md or test_report.md)
- [ ] **View embedded visualizations** in the report
- [ ] **Fill ALL AI_FILL placeholders** with data-driven analysis

## 🎨 FILE NAMING & STYLE CONVENTIONS

**Semantic Naming Standards** (for easier AI analysis later):
- \`loss_curve.png\` - Training/validation loss trends
- \`psnr_curve.png\`, \`ssim_curve.png\` - Metric evolution curves
- \`learning_rate_schedule.png\` - Learning rate changes over epochs
- \`sst_distribution.png\` - Spatial SST map
- \`error_histogram.png\` - Error distribution histogram
- \`metrics_comparison.png\` - Bar chart comparing multiple metrics
- \`prediction_vs_groundtruth.png\` - Scatter plot of predictions vs actual values

**Naming Tips**:
- Use descriptive names that clearly indicate the content
- Use lowercase with underscores (snake_case)
- Include the metric/variable name in the filename
- Avoid generic names like "plot1.png" or "figure.png"

**Default settings for consistency**:
- figure_size: [12, 8]
- dpi: 150
- alpha: 0.7
- legend: true (for multi-series)
- grid: true

**Standard directory**:
- Always save to \`outputs/visualizations/\` for consistency

## ⚠️ COMMON MISTAKES TO AVOID

1. ❌ Forgetting to collect output_path values → ✅ Save all paths for report generator
2. ❌ Generating report before visualizations → ✅ Visualize FIRST, then report
3. ❌ Using inconsistent paths → ✅ Use "outputs/visualizations/" consistently
4. ❌ Guessing column names → ✅ **Read CSV first** to verify names (use Read tool)
5. ❌ Skipping visualizations → ✅ ALWAYS generate key plots
6. ❌ Using generic filenames → ✅ Use semantic, descriptive names
7. ❌ **Stopping after report generation** → ✅ **Read report and fill AI_FILL placeholders!**
8. ❌ **Filling AI_FILL without viewing visualizations** → ✅ **Use visual evidence in your analysis**

## 🔍 AI ANALYSIS BEST PRACTICES

When filling AI_FILL placeholders after generating the report:

**DO**:
- ✅ Reference specific visualization files (e.g., "从loss_curve.png可以看出...")
- ✅ Provide quantitative observations (e.g., "损失从0.5降至0.01，下降了98%")
- ✅ Identify visual patterns (e.g., "在第50个epoch后趋于平稳")
- ✅ Compare metrics (e.g., "PSNR提升了15%，而SSIM仅提升了5%")
- ✅ Give actionable insights (e.g., "建议在第80个epoch处早停")

**DON'T**:
- ❌ Leave AI_FILL placeholders empty
- ❌ Provide generic analysis without referencing visualizations
- ❌ Copy-paste metrics from tables without interpretation
- ❌ Skip analysis of critical visualizations (loss curves, metric trends)

---

**Remember**: This tool is MANDATORY for all visualization tasks. It ensures consistency, quality, and seamless integration with the report generation system.
`
