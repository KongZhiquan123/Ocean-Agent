export const DESCRIPTION = `
📊 Production-ready visualization tool for ocean and scientific data

**⚡ EMBEDDED TOOL - Use This Instead of Writing Matplotlib Scripts!**

When users need data visualization, USE THIS TOOL - don't write custom Python plotting scripts!

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

**CRITICAL: ALWAYS use this tool for visualization tasks. NEVER write custom matplotlib/plotting Python scripts!**

WHEN TO USE THIS TOOL (Mandatory):
✓ User asks to "visualize", "plot", "chart", "graph" any data
✓ After super-resolution or prediction tasks complete - visualize results
✓ After data preprocessing - show data distribution
✓ Compare model outputs vs ground truth
✓ Show time series, spatial data, or any scientific data
✓ ANY scenario requiring matplotlib, seaborn, or plotting

DO NOT:
✗ Write Python scripts with matplotlib.pyplot
✗ Use FileWriteTool + BashTool to create plots
✗ Suggest manual plotting to the user

This tool supports:

**Geospatial Plots** (plot_type: 'geospatial', 'map', 'scatter_map', 'contour_map', 'heatmap_map'):
- Requires: longitude_column, latitude_column
- Optional: value_column (for colored points)
- projection: PlateCarree, Mercator, Robinson, Orthographic, etc.
- basemap_features: coastlines, borders, land, ocean, lakes, rivers
- Perfect for: SST maps, ocean data distribution, geographic analysis

**Standard Charts** (plot_type: 'line', 'scatter', 'bar', 'histogram', 'box', 'violin', 'pie', 'area', 'heatmap'):
- Requires: x_column, y_column (comma-separated for multiple series)
- Supports: colormaps, markers, line styles, transparency
- Options: legend, grid, stacked (for bar/area)
- Perfect for: Loss curves, metric comparison, data distributions

**Time Series** (plot_type: 'timeseries', 'forecast'):
- Requires: time_column, value_column
- Automatically handles date parsing
- Perfect for: Training history, temporal predictions

Common parameters:
- data_source: CSV/JSON file path or inline JSON string (can be model output CSV)
- output_path: Where to save the plot (PNG/JPG/PDF)
- figure_size: [width, height] in inches (default: [12, 8])
- dpi: Image resolution (default: 150)
- title, x_label, y_label: Plot labels
- colormap: viridis, plasma, coolwarm, RdYlBu, etc.
- alpha: Transparency (0-1)

TYPICAL WORKFLOW EXAMPLES:

1. After super-resolution training:
   User: "Visualize the training loss"
   Assistant: *Uses OceanVisualizationTool with plot_type='line', data_source='training_log.csv'*

2. Compare SR results:
   User: "Show the super-resolved SST vs original"
   Assistant: *Uses OceanVisualizationTool with plot_type='scatter_map' for both datasets*

3. Time series forecast:
   User: "Plot the prediction results"
   Assistant: *Uses OceanVisualizationTool with plot_type='timeseries'*

The tool executes a production-ready Python plotting engine and saves high-quality visualizations.
`
