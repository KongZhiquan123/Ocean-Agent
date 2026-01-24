# CLAUDE.md

This file provides guidance to some tools and implementation details specific to the Claude agent system.

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

