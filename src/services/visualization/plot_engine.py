#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal plotting engine for ocean data visualization

Usage:
    Set environment variables and run:
    PLOT_TYPE=<type> PLOT_CONFIG=<json_config> python plot_engine.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available, geographic projections disabled")


class PlotEngine:
    """Unified plotting engine"""

    def __init__(self, config: dict):
        self.config = config
        self.plot_type = config.get('plot_type', 'line')
        self.data_source = config.get('data_source')
        self.output_path = config.get('output_path', 'output.png')

    def load_data(self):
        """Load data"""
        if self.data_source.endswith('.csv'):
            return pd.read_csv(self.data_source)
        elif self.data_source.endswith('.json'):
            with open(self.data_source, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif self.data_source.endswith('.nc'):
            import xarray as xr
            ds = xr.open_dataset(self.data_source)
            return ds
        else:
            # Try parse as inline JSON
            try:
                data = json.loads(self.data_source)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                else:
                    return data
            except:
                raise ValueError(f"Unsupported data source: {self.data_source}")

    def plot(self):
        """Call appropriate plotting method based on type"""
        # Geospatial types
        if self.plot_type in ['geospatial', 'map', 'scatter_map', 'contour_map', 'heatmap_map']:
            return self.plot_geospatial()
        # Standard chart types
        elif self.plot_type in ['line', 'scatter', 'bar', 'histogram', 'box', 'violin', 'pie', 'area', 'heatmap']:
            return self.plot_chart()
        # Time series types
        elif self.plot_type in ['timeseries', 'forecast']:
            return self.plot_timeseries()
        else:
            raise ValueError(f"Unknown plot type: {self.plot_type}")

    def plot_geospatial(self):
        """Geospatial plotting"""
        data = self.load_data()

        lon_col = self.config.get('longitude_column', 'lon')
        lat_col = self.config.get('latitude_column', 'lat')
        value_col = self.config.get('value_column')

        # Extract data
        lons = data[lon_col].values
        lats = data[lat_col].values
        values = data[value_col].values if value_col else None

        # Create figure
        projection = self.config.get('projection', 'PlateCarree')
        fig_size = self.config.get('figure_size', [12, 8])

        if HAS_CARTOPY and projection != 'None':
            proj_obj = getattr(ccrs, projection)()
            fig = plt.figure(figsize=fig_size)
            ax = plt.axes(projection=proj_obj)

            # Plot data
            if self.plot_type == 'scatter_map' or not self.plot_type.endswith('_map'):
                scatter = ax.scatter(
                    lons, lats,
                    c=values if values is not None else 'blue',
                    cmap=self.config.get('colormap', 'viridis'),
                    s=self.config.get('marker_size', 50),
                    alpha=self.config.get('alpha', 0.7),
                    transform=ccrs.PlateCarree()
                )
                if values is not None and self.config.get('add_colorbar', True):
                    plt.colorbar(scatter, ax=ax, label=value_col or 'Value')

            # Add map features
            features = self.config.get('basemap_features', ['coastlines', 'borders'])
            if 'coastlines' in features:
                ax.coastlines()
            if 'borders' in features:
                ax.add_feature(cfeature.BORDERS, linestyle=':')
            if 'land' in features:
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
            if 'ocean' in features:
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

            # Gridlines
            if self.config.get('add_gridlines', True):
                ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')

            # Extent
            extent = self.config.get('extent')
            if extent:
                ax.set_extent(extent, crs=ccrs.PlateCarree())

        else:
            # Without cartopy
            fig, ax = plt.subplots(figsize=fig_size)

            scatter = ax.scatter(
                lons, lats,
                c=values if values is not None else 'blue',
                cmap=self.config.get('colormap', 'viridis'),
                s=self.config.get('marker_size', 50),
                alpha=self.config.get('alpha', 0.7)
            )

            if values is not None and self.config.get('add_colorbar', True):
                plt.colorbar(scatter, ax=ax, label=value_col or 'Value')

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)

        # Title
        title = self.config.get('title', 'Geospatial Plot')
        plt.title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_path, dpi=self.config.get('dpi', 150), bbox_inches='tight')
        plt.close()

        print(f"Geospatial plot saved to: {self.output_path}")
        return self.output_path

    def plot_chart(self):
        """Standard chart plotting"""
        data = self.load_data()

        x_col = self.config.get('x_column')
        y_col = self.config.get('y_column')

        fig_size = self.config.get('figure_size', [10, 6])
        fig, ax = plt.subplots(figsize=fig_size)

        # Plot based on type
        if self.plot_type == 'line':
            if y_col:
                y_columns = [c.strip() for c in y_col.split(',')]
                for y in y_columns:
                    ax.plot(
                        data[x_col] if x_col else range(len(data)),
                        data[y],
                        label=y,
                        linewidth=self.config.get('line_width', 2),
                        marker=self.config.get('marker_style', 'o') if len(data) < 50 else None,
                        alpha=self.config.get('alpha', 0.8)
                    )

        elif self.plot_type == 'scatter':
            ax.scatter(
                data[x_col],
                data[y_col],
                s=self.config.get('marker_size', 50),
                marker=self.config.get('marker_style', 'o'),
                alpha=self.config.get('alpha', 0.7),
                c=self.config.get('color', 'blue')
            )

        elif self.plot_type == 'bar':
            data.plot.bar(
                x=x_col,
                y=y_col,
                ax=ax,
                alpha=self.config.get('alpha', 0.8)
            )

        elif self.plot_type == 'histogram':
            data[x_col or y_col].plot.hist(
                bins=self.config.get('bins', 30),
                ax=ax,
                alpha=self.config.get('alpha', 0.7),
                edgecolor='black'
            )

        elif self.plot_type == 'box':
            data.boxplot(column=y_col, by=x_col, ax=ax)

        elif self.plot_type == 'pie':
            data.set_index(x_col)[y_col].plot.pie(
                ax=ax,
                autopct='%1.1f%%',
                startangle=90
            )

        elif self.plot_type == 'area':
            y_columns = [c.strip() for c in y_col.split(',')]
            data.plot.area(
                x=x_col,
                y=y_columns,
                ax=ax,
                alpha=self.config.get('alpha', 0.6),
                stacked=self.config.get('stacked', False)
            )

        elif self.plot_type == 'heatmap':
            # Assume data is 2D matrix form
            if isinstance(data, pd.DataFrame):
                import seaborn as sns
                sns.heatmap(data, ax=ax, cmap=self.config.get('colormap', 'viridis'))

        # Labels and title
        if self.plot_type != 'pie':
            ax.set_xlabel(self.config.get('x_label', x_col or 'X'))
            ax.set_ylabel(self.config.get('y_label', y_col or 'Y'))

        title = self.config.get('title', f'{self.plot_type.title()} Chart')
        plt.title(title, fontsize=14, fontweight='bold')

        # Legend
        if self.config.get('legend', True) and self.plot_type not in ['pie', 'histogram']:
            ax.legend()

        # Grid
        if self.config.get('grid', True) and self.plot_type != 'pie':
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_path, dpi=self.config.get('dpi', 150), bbox_inches='tight')
        plt.close()

        print(f"Chart saved to: {self.output_path}")
        return self.output_path

    def plot_timeseries(self):
        """Time series plotting"""
        data = self.load_data()

        time_col = self.config.get('time_column', 'time')
        value_col = self.config.get('value_column', 'value')

        fig_size = self.config.get('figure_size', [12, 6])
        fig, ax = plt.subplots(figsize=fig_size)

        # Convert time column
        if time_col in data.columns:
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.sort_values(time_col)

        ax.plot(
            data[time_col] if time_col in data.columns else range(len(data)),
            data[value_col],
            linewidth=self.config.get('line_width', 2),
            marker=self.config.get('marker_style') if len(data) < 100 else None,
            alpha=self.config.get('alpha', 0.8),
            label=value_col
        )

        ax.set_xlabel(self.config.get('x_label', 'Time'))
        ax.set_ylabel(self.config.get('y_label', value_col))
        ax.set_title(self.config.get('title', 'Time Series Plot'), fontsize=14, fontweight='bold')

        if self.config.get('legend', True):
            ax.legend()

        if self.config.get('grid', True):
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=self.config.get('dpi', 150), bbox_inches='tight')
        plt.close()

        print(f"Time series plot saved to: {self.output_path}")
        return self.output_path


def main():
    """Read config from environment variable and execute plotting"""
    try:
        # Read config from environment variable
        config_str = os.environ.get('PLOT_CONFIG')
        if not config_str:
            print("Error: PLOT_CONFIG environment variable not set")
            sys.exit(1)

        config = json.loads(config_str)

        print(f"Starting plot generation...")
        print(f"   Type: {config.get('plot_type', 'unknown')}")
        print(f"   Output: {config.get('output_path', 'unknown')}")

        engine = PlotEngine(config)
        result = engine.plot()

        print(f"Plot generation completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
