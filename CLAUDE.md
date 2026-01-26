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
- ❌ **NEVER DO**: Write matplotlib/seaborn/plotly Python scripts
- ❌ **NEVER DO**: Use FileWriteTool + BashTool to create plots manually
- ❌ **NEVER DO**: Create custom plotting functions

### Why This Matters:
1. **Consistency**: All visualizations have uniform styling
2. **Correctness**: Proper file paths, no encoding errors
3. **Efficiency**: Production-ready plotting engine already built
4. **Maintenance**: Centralized visualization logic

### Supported Visualizations:

**Geospatial/Geographic Plots**:
- `plot_type`: 'geospatial', 'map', 'scatter_map', 'contour_map', 'heatmap_map'
- Perfect for: SST maps, ocean data distribution, spatial analysis

**Standard Charts**:
- `plot_type`: 'line', 'scatter', 'bar', 'histogram', 'box', 'violin', 'pie', 'area', 'heatmap'
- Perfect for: Loss curves, metric comparison, data distributions

**Time Series**:
- `plot_type`: 'timeseries', 'forecast'
- Perfect for: Training history, temporal predictions

### Common Use Cases:

**After Super-Resolution Training**:
```json
{
  "data_source": "outputs/training_log.csv",
  "plot_type": "line",
  "output_path": "outputs/visualizations/loss_curve.png",
  "x_column": "epoch",
  "y_column": "train_loss,val_loss",
  "title": "Training Loss Curve",
  "legend": true,
  "grid": true
}
```

**Geographic Ocean Data**:
```json
{
  "data_source": "outputs/predictions.csv",
  "plot_type": "scatter_map",
  "output_path": "outputs/visualizations/sst_map.png",
  "longitude_column": "lon",
  "latitude_column": "lat",
  "value_column": "sst",
  "projection": "PlateCarree",
  "colormap": "coolwarm",
  "title": "Sea Surface Temperature Distribution"
}
```

**Model Evaluation Metrics**:
```json
{
  "data_source": "outputs/metrics.csv",
  "plot_type": "bar",
  "output_path": "outputs/visualizations/metrics_comparison.png",
  "x_column": "model",
  "y_column": "rmse",
  "title": "Model Performance Comparison"
}
```

Following this policy ensures all visualizations across the project maintain professional quality and consistency.

