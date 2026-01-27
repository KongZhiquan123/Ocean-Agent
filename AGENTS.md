# CLAUDE.md

This file provides guidance to some tools and implementation details specific to the Claude agent system.

## Data Preprocessing Policy

**🔴 CRITICAL: Always Use OceanPreprocessPipeline Tool for Data Preparation**

When users provide raw ocean data (NetCDF files, satellite observations, etc.) that needs preprocessing:

### MANDATORY Rules:
- ✅ **ALWAYS USE**: `OceanPreprocessPipeline` tool for data preprocessing
- ✅ **ALWAYS VALIDATE**: Data quality and convergence MUST be checked
- ❌ **NEVER DO**: Write custom Python/pandas/xarray scripts for preprocessing
- ❌ **NEVER DO**: Use FileWrite + Bash to process NC files manually
- ❌ **NEVER DO**: Skip the validation phase
- ❌ **NEVER DO**: Process data without checking convergence metrics

### Why This Matters:
1. **Data Quality**: Built-in CNN validation ensures data is suitable for training
2. **Consistency**: Standardized preprocessing pipeline across all ocean datasets
3. **Convergence**: Automatic detection of data issues before expensive training
4. **Traceability**: Automatic generation of validation reports and quality metrics
5. **Efficiency**: Production-ready preprocessing engine already built

### Validation is MANDATORY:
The preprocessing pipeline includes **two-phase validation**:

**Phase 1: Statistical Validation (Always runs)**
- Missing value analysis
- Outlier detection
- Data distribution checks
- Temporal/spatial continuity

**Phase 2: CNN Convergence Validation (Default: enabled)**
- Trains a lightweight CNN on the preprocessed data
- Checks if loss converges (indicates data is learnable)
- Provides convergence metrics and quality scores
- **If CNN validation fails, the data is NOT ready for production models**

### When to Use:
- User uploads/provides raw NetCDF files (`.nc` files)
- User mentions data from: JAXA, OSTIA, ERA5, CMEMS, etc.
- User asks to "prepare data", "preprocess data", "clean data"
- Before ANY training pipeline (DiffSR, forecasting, etc.)
- When user provides a directory of ocean observation files

### Required Parameters:
```json
{
  "input_dir": "path/to/raw/nc/files",
  "output_dir": "path/to/output",
  "variable_name": "sst",  // or "temperature", "salinity", etc.
  "file_pattern": "*.nc",
  "use_cnn_validation": true  // ALWAYS true unless user explicitly disables
}
```

### Output Files:
After preprocessing, the tool generates:
1. `preprocessed_{variable}.nc` - Cleaned and merged data file
2. `validation_report.md` - **CRITICAL**: Detailed validation report with convergence analysis
3. `validation_results.json` - Machine-readable quality metrics

**IMPORTANT**: Always read and present the validation_report.md to the user. It contains critical information about data quality and whether the data is ready for training.

### Example Workflow:

**User**: "I have SST data from JAXA in the `raw_data/` folder, prepare it for training"

**Assistant**: Must use OceanPreprocessPipeline tool:
```json
{
  "input_dir": "raw_data",
  "output_dir": "preprocessed_data",
  "variable_name": "sst",
  "file_pattern": "*.nc",
  "use_cnn_validation": true
}
```

**After tool completes**:
1. ✅ Read `validation_report.md`
2. ✅ Present key metrics to user (convergence, quality score)
3. ✅ Recommend next steps based on validation results
4. ❌ Do NOT proceed to training if validation failed

### Implementation Note:
The `OceanPreprocessPipeline` tool is implemented in:
- `src/tools/OceanPreprocessPipelineTool/OceanPreprocessPipelineTool.tsx`
- Backend engine: `src/services/preprocessing/main.py`
- Validation: `src/services/preprocessing/validator.py`

Following this policy ensures all datasets are properly validated before expensive model training, preventing wasted compute on poor-quality data.

---

## Visualization Policy

**🔴 CRITICAL: Always Use OceanVisualization Tool for Plotting**

When you need to create ANY visualization (charts, plots, maps, graphs, figures):

### MANDATORY Rules:
- ✅ **ALWAYS USE**: `OceanVisualization` tool
- ✅ **ALWAYS GENERATE**: Visualizations BEFORE generating the final report
- ✅ **ALWAYS COLLECT**: All output_path values to pass to report generator
- ❌ **NEVER DO**: Write matplotlib/seaborn/plotly Python scripts
- ❌ **NEVER DO**: Use FileWriteTool + BashTool to create plots manually
- ❌ **NEVER DO**: Create custom plotting functions
- ❌ **NEVER DO**: Skip visualization generation when training/inference completes

### Why This Matters:
1. **Consistency**: All visualizations have uniform styling
2. **Correctness**: Proper file paths, no encoding errors
3. **Efficiency**: Production-ready plotting engine already built
4. **Maintenance**: Centralized visualization logic
5. **Integration**: Seamless integration with report generation system

### Integration with Report Generation System

**The Complete Four-Step Workflow:**

```
Step 1: Training/Inference
   ↓ (generates metrics.json, config.json, training_log.csv, predictions.csv, etc.)

Step 2: Visualization Generation ← USE OceanVisualization TOOL
   ↓ (generates PNG/PDF files, returns output paths)

Step 3: Report Generation (report_generator.py - Python Script Automation)
   ↓ (call with --viz_paths parameter)
   ↓ (script auto-fills VIZ_FILE_LIST and VIZ_IMAGES placeholders)
   ↓ (script preserves AI_FILL placeholders for manual analysis)

Step 4: AI Analysis ← YOU MUST DO THIS
   ↓ (Read the generated report: training_report.md or test_report.md)
   ↓ (View the embedded visualization images)
   ↓ (Analyze visual patterns, trends, and insights)
   ↓ (Fill ALL AI_FILL placeholders with detailed, data-driven analysis)
```

⚠️ **CRITICAL**: The workflow does NOT end at Step 3. After the report is generated, you MUST proceed to Step 4 and fill all AI_FILL placeholders.

**Critical Steps:**

1. After training/inference completes, **generate ALL necessary visualizations**
2. **Collect all output_path values** returned by each OceanVisualization call
3. **Pass collected paths** to report_generator.py using `--viz_paths` parameter:
   ```bash
   python report_generator.py train config.json metrics.json output.md \
     --viz_paths "outputs/visualizations/loss_curve.png,outputs/visualizations/psnr_curve.png,outputs/visualizations/sst_map.png"
   ```
4. The report generator will automatically:
   - Replace `VIZ_FILE_LIST` placeholder with bullet list of file paths
   - Replace `VIZ_IMAGES` placeholder with embedded image markdown
   - **Preserve `AI_FILL` placeholders** for you to analyze later
5. **YOU MUST then**:
   - Read the generated report (training_report.md or test_report.md)
   - View the embedded visualization images
   - Analyze the visualizations and data
   - **Fill ALL AI_FILL placeholders** with detailed analysis

**Report Placeholders:**

Training/inference reports contain TWO types of placeholders:

**Type 1: Auto-filled by report_generator.py (No action needed)**

- **Section 4.1 - File List**:
  ```markdown
  <!-- VIZ_FILE_LIST: 脚本自动填充，列出所有生成的可视化图片路径 -->
  ```

- **Section 4.3 - Image Gallery**:
  ```markdown
  <!-- VIZ_IMAGES: 脚本自动填充，插入所有可视化图片 -->
  ```

**Type 2: AI_FILL placeholders (YOU MUST FILL THESE)**

The generated report contains multiple `AI_FILL` placeholders requiring your analysis:

- **Section 2.2 - Training Curves**:
  - `<!-- AI_FILL: 描述训练和验证损失的下降趋势，分析收敛情况 -->`
  - `<!-- AI_FILL: 描述学习率变化策略及其对训练的影响 -->`

- **Section 3.3 - Performance Comparison**:
  - `<!-- AI_FILL: 对比分析模型在不同数据集或与基准模型的性能差异 -->`

- **Section 4.2 - Visualization Analysis** ← MOST CRITICAL:
  - `<!-- AI_FILL: 分析可视化图表内容，说明每个图表展示的信息和关键发现 -->`

- **Section 5 - Model Checkpoints**:
  - `<!-- AI_FILL: 列出训练过程中生成的辅助文件，如日志、配置备份等 -->`
  - `<!-- AI_FILL: 描述模型预测结果的质量和特点 -->`

- **Section 6 - Training Analysis**:
  - `<!-- AI_FILL: 分析训练过程的稳定性，包括：loss下降趋势、是否有异常波动、收敛速度评估 -->`
  - `<!-- AI_FILL: 分析模型性能，包括：PSNR/SSIM等指标变化趋势、与预期目标的对比、性能瓶颈分析 -->`

- **Section 7 - Computational Performance**:
  - `<!-- AI_FILL: 分析GPU利用率情况，包括：显存占用、计算利用率、是否存在瓶颈 -->`
  - `<!-- AI_FILL: 分析数据加载效率，包括：数据预处理时间、IO瓶颈、建议优化方向 -->`

- **Section 8 - Summary**:
  - `<!-- AI_FILL: 总结本次训练的核心成就，包括：模型性能亮点、训练效率、达成的目标（3-5点） -->`

### How to Fill AI_FILL Placeholders

When you read the generated report and encounter AI_FILL placeholders:

**DO**:
- ✅ **Use visualization images as evidence** (they are already embedded in the report)
- ✅ **Reference specific visual patterns** (e.g., "从loss_curve.png可以看出，损失在第50个epoch后趋于平稳")
- ✅ **Provide quantitative analysis** (e.g., "训练损失从0.5降至0.01，下降了98%")
- ✅ **Identify key findings** (e.g., "验证集PSNR在第80个epoch达到峰值后略有下降，提示可能出现轻微过拟合")
- ✅ **Give actionable insights** (e.g., "建议在未来训练中在第80个epoch处早停以避免过拟合")

**DON'T**:
- ❌ Leave AI_FILL placeholders empty or unfilled
- ❌ Provide generic analysis without referencing visualizations
- ❌ Copy-paste metrics from tables without interpretation
- ❌ Skip analysis of critical visualizations (loss curves, metric trends)

### Supported Visualizations

**Geospatial/Geographic Plots**:
- `plot_type`: 'geospatial', 'map', 'scatter_map', 'contour_map', 'heatmap_map'
- Perfect for: SST maps, ocean data distribution, spatial analysis

**Standard Charts**:
- `plot_type`: 'line', 'scatter', 'bar', 'histogram', 'box', 'violin', 'pie', 'area', 'heatmap'
- Perfect for: Loss curves, metric comparison, data distributions

**Time Series**:
- `plot_type`: 'timeseries', 'forecast'
- Perfect for: Training history, temporal predictions

### Typical Post-Training Visualizations

**1. Loss Curves** (Always generate):
```json
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
```

**2. Performance Metrics** (PSNR, SSIM, etc.):
```json
{
  "data_source": "outputs/training_log.csv",
  "plot_type": "line",
  "output_path": "outputs/visualizations/psnr_curve.png",
  "x_column": "epoch",
  "y_column": "val_psnr",
  "title": "Validation PSNR Over Training",
  "color": "green",
  "grid": true
}
```

**3. Spatial Distribution Map** (For ocean data):
```json
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
```

**4. Error Distribution** (For model evaluation):
```json
{
  "data_source": "outputs/errors.csv",
  "plot_type": "histogram",
  "output_path": "outputs/visualizations/error_histogram.png",
  "x_column": "prediction_error",
  "bins": 50,
  "title": "Prediction Error Distribution"
}
```

### Complete Example Workflow

**Scenario**: User completes DiffSR training and wants a comprehensive report

```markdown
1. Training completes → Files generated:
   - outputs/training_log.csv
   - outputs/metrics.json
   - outputs/config.json

2. Generate visualizations (YOU DO THIS - Step 2):

   a) Call OceanVisualization for loss curve
      → Returns: "outputs/visualizations/loss_curve.png"

   b) Call OceanVisualization for PSNR curve
      → Returns: "outputs/visualizations/psnr_curve.png"

   c) Call OceanVisualization for SSIM curve
      → Returns: "outputs/visualizations/ssim_curve.png"

   d) Call OceanVisualization for spatial map
      → Returns: "outputs/visualizations/sst_map.png"

   **Collect paths**: viz_paths = [
     "outputs/visualizations/loss_curve.png",
     "outputs/visualizations/psnr_curve.png",
     "outputs/visualizations/ssim_curve.png",
     "outputs/visualizations/sst_map.png"
   ]

3. Generate report with visualizations (Step 3):

   bash: python /opt/kode/dist/services/diffsr/report_generator.py train \
         outputs/config.json \
         outputs/metrics.json \
         outputs/training_report.md \
         --viz_paths "outputs/visualizations/loss_curve.png,outputs/visualizations/psnr_curve.png,outputs/visualizations/ssim_curve.png,outputs/visualizations/sst_map.png"

   → Report generated: outputs/training_report.md
   → VIZ_FILE_LIST and VIZ_IMAGES placeholders automatically filled
   → AI_FILL placeholders preserved for manual analysis

4. AI analysis (YOU DO THIS - Step 4):

   a) Read the generated report:
      → Use Read tool: outputs/training_report.md

   b) View embedded visualizations:
      → Images are already embedded in the report via markdown

   c) Analyze each visualization:
      → loss_curve.png: convergence pattern, stability, overfitting signs
      → psnr_curve.png: performance evolution, peak epoch, trends
      → ssim_curve.png: structural similarity trends, correlation with PSNR
      → sst_map.png: spatial prediction quality, regional patterns

   d) Fill ALL AI_FILL placeholders:
      → Use Edit tool to replace each AI_FILL comment with detailed analysis
      → Reference specific visualizations in your analysis
      → Provide quantitative insights from the data
      → Give actionable recommendations

5. Present final report to user:
   - Report path: outputs/training_report.md
   - Visualizations embedded in Section 4
   - All VIZ placeholders filled automatically
   - All AI_FILL placeholders filled with your analysis
```

### File Naming Conventions

**Semantic Naming Standards** (for easier AI analysis):
- `loss_curve.png` - Training/validation loss trends
- `psnr_curve.png`, `ssim_curve.png` - Metric evolution curves
- `learning_rate_schedule.png` - Learning rate changes over epochs
- `sst_distribution.png` - Spatial SST map
- `error_histogram.png` - Error distribution histogram
- `metrics_comparison.png` - Bar chart comparing multiple metrics
- `prediction_vs_groundtruth.png` - Scatter plot of predictions vs actual values

**Naming Tips**:
- Use descriptive names that clearly indicate the content
- Use lowercase with underscores (snake_case)
- Include the metric/variable name in the filename
- Avoid generic names like "plot1.png" or "figure.png"

**Standard Directory**:
- Always save to `outputs/visualizations/` for consistency

### Common Mistakes to Avoid

1. ❌ **Generating report before visualizations**
   - ✅ Correct order: Visualize → Collect paths → Generate report → Fill AI_FILL

2. ❌ **Forgetting to collect output paths**
   - ✅ Always save the output_path values to pass to report_generator.py

3. ❌ **Not reading CSV before plotting**
   - ✅ Always read the data file first to verify column names (use Read tool)

4. ❌ **Skipping visualizations to save time**
   - ✅ Visualizations are critical for understanding model performance AND for AI analysis

5. ❌ **Using inconsistent file paths**
   - ✅ Always use "outputs/visualizations/" as the base directory

6. ❌ **Using generic filenames**
   - ✅ Use semantic, descriptive names (loss_curve.png, not plot1.png)

7. ❌ **Stopping after report generation** ← MOST CRITICAL ERROR
   - ✅ **You MUST read the report and fill ALL AI_FILL placeholders**
   - ✅ This is NOT optional - it's a required step in the workflow

8. ❌ **Filling AI_FILL placeholders without viewing visualizations**
   - ✅ **Always use visual evidence in your analysis**
   - ✅ Reference specific visualization files in your analysis

9. ❌ **Leaving AI_FILL placeholders empty**
   - ✅ Fill ALL AI_FILL placeholders with detailed, data-driven analysis

10. ❌ **Providing generic analysis without specifics**
    - ✅ Provide quantitative observations, identify patterns, give actionable insights

Following this policy ensures all visualizations are properly generated, integrated with reports, maintain professional quality, AND are properly analyzed by AI.

