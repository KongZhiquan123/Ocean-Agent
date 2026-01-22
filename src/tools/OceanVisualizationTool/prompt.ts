export const DESCRIPTION = `
Create various types of visualizations for ocean and general scientific data

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

This tool supports:

**Geospatial Plots** (plot_type: 'geospatial', 'map', 'scatter_map'):
- Requires: longitude_column, latitude_column
- Optional: value_column (for colored points)
- projection: PlateCarree, Mercator, Robinson, Orthographic, etc.
- basemap_features: coastlines, borders, land, ocean, lakes, rivers

**Standard Charts** (plot_type: 'line', 'scatter', 'bar', 'histogram', 'box', 'pie', 'area'):
- Requires: x_column, y_column (comma-separated for multiple series)
- Supports: colormaps, markers, line styles, transparency
- Options: legend, grid, stacked (for bar/area)

**Time Series** (plot_type: 'timeseries'):
- Requires: time_column, value_column
- Automatically handles date parsing

Common parameters:
- data_source: CSV/JSON file path or inline JSON string
- output_path: Where to save the plot (PNG/JPG/PDF)
- figure_size: [width, height] in inches
- dpi: Image resolution (default: 150)
- title, x_label, y_label: Plot labels
- colormap: viridis, plasma, coolwarm, RdYlBu, etc.
- alpha: Transparency (0-1)

The tool will execute a Python script and save the visualization to the specified path.
`
