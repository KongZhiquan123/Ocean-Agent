import * as React from 'react'
import { z } from 'zod'
import { FallbackToolUseRejectedMessage } from '@components/FallbackToolUseRejectedMessage'
import { Tool, ValidationResult } from '@tool'
import { getCwd } from '@utils/state'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import * as path from 'node:path'
import OceanVisualizationToolResultMessage from './OceanVisualizationToolResultMessage'
import { PROMPT } from './prompt'

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

export const inputSchema = z.strictObject({
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

type In = typeof inputSchema
export type Out = {
  success: boolean
  output_path: string
  stdout: string
  stderr: string
  exit_code: number | null
}

export const OceanVisualizationTool = {
  name: 'OceanVisualization',
  async description() {
    return 'Creates oceanographic and scientific visualizations from data'
  },
  async prompt() {
    return PROMPT
  },
  isReadOnly() {
    return false
  },
  isConcurrencySafe() {
    return false
  },
  inputSchema,
  userFacingName() {
    return 'Ocean Visualization'
  },
  async isEnabled() {
    return true
  },
  needsPermissions(): boolean {
    return true
  },
  async validateInput({ data_source, output_path }): Promise<ValidationResult> {
    const cwd = getCwd()

    // Validate data source
    if (!data_source.trim().startsWith('[') && !data_source.trim().startsWith('{')) {
      const fullPath = path.isAbsolute(data_source) ? data_source : path.join(cwd, data_source)
      if (!existsSync(fullPath)) {
        return {
          result: false,
          message: `Data file not found: ${fullPath}`,
        }
      }
    }

    // Validate output path extension
    const outputExt = output_path.toLowerCase()
    if (!outputExt.endsWith('.png') && !outputExt.endsWith('.jpg') && !outputExt.endsWith('.pdf')) {
      return {
        result: false,
        message: 'Output file must be PNG, JPG, or PDF format',
      }
    }

    return { result: true }
  },
  renderToolUseMessage({ plot_type, data_source, output_path }) {
    const displaySource = data_source.length > 50 ? data_source.substring(0, 50) + '...' : data_source
    return `Creating ${plot_type} visualization from ${displaySource} → ${output_path}`
  },
  renderToolUseRejectedMessage() {
    return <FallbackToolUseRejectedMessage />
  },
  renderToolResultMessage(content) {
    return <OceanVisualizationToolResultMessage content={content} />
  },
  renderResultForAssistant({ success, output_path, stdout, stderr, exit_code }) {
    if (success && exit_code === 0) {
      return `Visualization successfully created: ${output_path}\n${stdout}`
    } else {
      return `Visualization failed (exit code: ${exit_code})\nStdout: ${stdout}\nStderr: ${stderr}`
    }
  },
  async *call(
    { data_source, plot_type, output_path, ...config },
    { abortController },
  ) {
    const cwd = getCwd()

    // Prepare plot configuration
    const plotConfig = {
      data_source: data_source.trim().startsWith('[') || data_source.trim().startsWith('{')
        ? data_source
        : path.isAbsolute(data_source) ? data_source : path.join(cwd, data_source),
      plot_type,
      output_path: path.isAbsolute(output_path) ? output_path : path.join(cwd, output_path),
      ...config
    }

    // Python script path
    const projectRoot = path.join(cwd, 'kode')
    const scriptPath = path.join(projectRoot, '..', 'src', 'services', 'visualization', 'plot_engine.py')

    // Find Python executable
    const pythonCmd = process.platform === 'win32'
      ? 'C:\\ProgramData\\anaconda3\\python.exe'
      : 'python3'

    let stdout = ''
    let stderr = ''
    let exitCode: number | null = null

    // Check if already cancelled
    if (abortController.signal.aborted) {
      const data: Out = {
        success: false,
        output_path,
        stdout: '',
        stderr: 'Command cancelled before execution',
        exit_code: null,
      }

      yield {
        type: 'result',
        resultForAssistant: this.renderResultForAssistant(data),
        data,
      }
      return
    }

    try {
      // Execute Python visualization script
      await new Promise<void>((resolve, reject) => {
        const env = {
          ...process.env,
          PLOT_CONFIG: JSON.stringify(plotConfig),
        }

        const proc = spawn(pythonCmd, [scriptPath], {
          cwd: projectRoot,
          env,
        })

        // Handle abort signal
        const abortHandler = () => {
          proc.kill()
          reject(new Error('Visualization cancelled by user'))
        }
        abortController.signal.addEventListener('abort', abortHandler)

        proc.stdout.on('data', (data: Buffer) => {
          stdout += data.toString()
        })

        proc.stderr.on('data', (data: Buffer) => {
          stderr += data.toString()
        })

        proc.on('close', (code) => {
          exitCode = code
          abortController.signal.removeEventListener('abort', abortHandler)
          if (code === 0) {
            resolve()
          } else {
            reject(new Error(`Python script exited with code ${code}`))
          }
        })

        proc.on('error', (err) => {
          abortController.signal.removeEventListener('abort', abortHandler)
          reject(err)
        })
      })

      const data: Out = {
        success: exitCode === 0,
        output_path,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        exit_code: exitCode,
      }

      yield {
        type: 'result',
        resultForAssistant: this.renderResultForAssistant(data),
        data,
      }
    } catch (error) {
      const isAborted = abortController.signal.aborted
      const errorMessage = isAborted
        ? 'Visualization was cancelled by user'
        : `Visualization failed: ${error instanceof Error ? error.message : String(error)}`

      const data: Out = {
        success: false,
        output_path,
        stdout: stdout.trim(),
        stderr: stderr.trim() + '\n' + errorMessage,
        exit_code: exitCode,
      }

      yield {
        type: 'result',
        resultForAssistant: this.renderResultForAssistant(data),
        data,
      }
    }
  },
} satisfies Tool<In, Out>
