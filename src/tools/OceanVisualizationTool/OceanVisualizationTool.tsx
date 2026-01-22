import { Box, Text } from 'ink'
import * as React from 'react'
import { z } from 'zod'
import type { Tool } from '@tool'
import { getCwd } from '@utils/state'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import * as path from 'node:path'
import { DESCRIPTION, PROMPT } from './prompt'

const PLOT_TYPES = [
  'geospatial', 'map', 'scatter_map', 'contour_map', 'heatmap_map',
  'line', 'scatter', 'bar', 'histogram', 'box', 'violin', 'pie', 'area', 'heatmap',
  'timeseries', 'forecast'
] as const

const PROJECTIONS = [
  'None', 'PlateCarree', 'Mercator', 'Robinson', 'Orthographic',
  'LambertConformal', 'Stereographic', 'Mollweide'
] as const

const BASEMAP_FEATURES = ['coastlines', 'borders', 'land', 'ocean', 'lakes', 'rivers'] as const

const COLORMAPS = [
  'viridis', 'plasma', 'inferno', 'magma', 'cividis',
  'coolwarm', 'RdYlBu', 'RdBu', 'seismic', 'jet', 'rainbow'
] as const

const inputSchema = z.strictObject({
  data_source: z.string().describe('Path to CSV/JSON file or inline JSON array'),
  plot_type: z.enum(PLOT_TYPES).default('line').describe('Type of plot to create'),
  output_path: z.string().describe('Output file path for the generated image (PNG/JPG/PDF)'),

  // Data columns
  x_column: z.string().optional().describe('Column name for x-axis'),
  y_column: z.string().optional().describe('Column name for y-axis (comma-separated for multiple series)'),
  time_column: z.string().optional().describe('Column name for time data (for time series)'),
  value_column: z.string().optional().describe('Column name for values'),
  longitude_column: z.string().optional().describe('Column name for longitude'),
  latitude_column: z.string().optional().describe('Column name for latitude'),

  // Geographic options
  projection: z.enum(PROJECTIONS).default('PlateCarree').describe('Map projection for geospatial plots'),
  basemap_features: z.array(z.enum(BASEMAP_FEATURES)).default(['coastlines', 'borders']).describe('Basemap features to display'),
  extent: z.tuple([z.number(), z.number(), z.number(), z.number()]).optional().describe('Map extent [lon_min, lon_max, lat_min, lat_max]'),

  // Style options
  title: z.string().optional().describe('Plot title'),
  x_label: z.string().optional().describe('X-axis label'),
  y_label: z.string().optional().describe('Y-axis label'),
  colormap: z.enum(COLORMAPS).default('viridis').describe('Colormap'),
  marker_style: z.string().default('o').describe('Marker style'),
  marker_size: z.number().default(50).describe('Marker size'),
  line_width: z.number().default(2).describe('Line width'),
  alpha: z.number().min(0).max(1).default(0.7).describe('Transparency (0=transparent, 1=opaque)'),

  // Layout options
  figure_size: z.tuple([z.number(), z.number()]).default([12, 8]).describe('Figure size [width, height] in inches'),
  dpi: z.number().default(150).describe('Image resolution (DPI)'),
  legend: z.boolean().default(true).describe('Show legend'),
  grid: z.boolean().default(true).describe('Show grid'),
  add_colorbar: z.boolean().default(true).describe('Add colorbar for geospatial plots'),
  add_gridlines: z.boolean().default(true).describe('Add gridlines for geospatial plots'),

  // Chart-specific options
  bins: z.number().default(30).describe('Number of bins for histogram'),
  stacked: z.boolean().default(false).describe('Stack multiple series (for bar/area charts)'),
  color: z.string().optional().describe('Color for single-series plots'),
})

type Input = z.infer<typeof inputSchema>

export const OceanVisualizationTool: Tool = {
  name: 'OceanVisualization',
  description: DESCRIPTION,
  schema: inputSchema,
  alwaysShowTabs: true,

  execute: async (input: Input) => {
    const { data_source, plot_type, output_path, ...config } = input
    const cwd = getCwd()

    // 验证数据源
    if (!data_source.trim().startsWith('[') && !data_source.trim().startsWith('{')) {
      // File path
      const fullPath = path.isAbsolute(data_source) ? data_source : path.join(cwd, data_source)
      if (!existsSync(fullPath)) {
        throw new Error(`Data file not found: ${fullPath}`)
      }
    }

    // 验证输出路径
    const outputExt = output_path.toLowerCase()
    if (!outputExt.endsWith('.png') && !outputExt.endsWith('.jpg') && !outputExt.endsWith('.pdf')) {
      throw new Error('Output file must be PNG, JPG, or PDF format')
    }

    // 准备完整配置
    const plotConfig = {
      data_source: data_source.trim().startsWith('[') || data_source.trim().startsWith('{')
        ? data_source
        : path.isAbsolute(data_source) ? data_source : path.join(cwd, data_source),
      plot_type,
      output_path: path.isAbsolute(output_path) ? output_path : path.join(cwd, output_path),
      ...config
    }

    // Python 脚本路径
    const projectRoot = path.join(cwd, 'kode')
    const scriptPath = path.join(projectRoot, '..', 'src', 'services', 'visualization', 'plot_engine.py')

    // 查找 Python 可执行文件
    const pythonCmd = process.platform === 'win32'
      ? 'C:\\ProgramData\\anaconda3\\python.exe'
      : 'python3'

    return {
      Component: () => {
        const [output, setOutput] = React.useState<string[]>([])
        const [error, setError] = React.useState<string | null>(null)
        const [exitCode, setExitCode] = React.useState<number | null>(null)

        React.useEffect(() => {
          const env = {
            ...process.env,
            PLOT_CONFIG: JSON.stringify(plotConfig),
          }

          const proc = spawn(pythonCmd, [scriptPath], {
            cwd: projectRoot,
            env,
          })

          proc.stdout.on('data', (data: Buffer) => {
            const lines = data.toString().split('\n').filter(l => l.trim())
            setOutput(prev => [...prev, ...lines])
          })

          proc.stderr.on('data', (data: Buffer) => {
            const errorMsg = data.toString()
            setError(errorMsg)
            setOutput(prev => [...prev, `[ERROR] ${errorMsg}`])
          })

          proc.on('close', (code) => {
            setExitCode(code)
          })

          return () => {
            proc.kill()
          }
        }, [])

        return (
          <Box flexDirection="column">
            <Box marginBottom={1}>
              <Text bold>📊 Ocean Visualization Tool</Text>
            </Box>

            <Box flexDirection="column" marginBottom={1}>
              <Text dimColor>Plot Type: {plot_type}</Text>
              <Text dimColor>Data Source: {data_source.length > 50 ? data_source.substring(0, 50) + '...' : data_source}</Text>
              <Text dimColor>Output: {output_path}</Text>
            </Box>

            <Box flexDirection="column" borderStyle="single" paddingX={1}>
              {output.map((line, i) => (
                <Text key={i}>{line}</Text>
              ))}
            </Box>

            {exitCode !== null && (
              <Box marginTop={1}>
                {exitCode === 0 ? (
                  <Text color="green" bold>
                    ✅ Visualization created: {output_path}
                  </Text>
                ) : (
                  <Text color="red" bold>
                    ❌ Visualization failed (exit code: {exitCode})
                  </Text>
                )}
              </Box>
            )}

            {error && (
              <Box marginTop={1}>
                <Text color="yellow">⚠️  {error}</Text>
              </Box>
            )}
          </Box>
        )
      },
      meta: {
        plot_type,
        data_source,
        output_path,
      },
    }
  },
}
