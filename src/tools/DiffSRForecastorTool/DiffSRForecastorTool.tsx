import { Box, Text } from 'ink'
import { z } from 'zod'
import { Tool } from '../../Tool.js'
import path from 'path'
import { getCwd } from '@utils/state'
import { OceanDepsManager } from '@utils/oceanDepsManager'
import { createAssistantMessage } from '@utils/messages'

const inputSchema = z.strictObject({
	forecastor_type: z.enum(['ddpm', 'resshift', 'base']).describe(
		'Forecastor type: ddpm (Denoising Diffusion), resshift (Residual Shift), base (Basic forecasting)'
	),
	model_dir: z.string().describe('Path to trained model directory (contains best_model.pth and config.yaml)'),
	output_dir: z.string().describe('Output directory for inference results'),
	split: z.enum(['train', 'valid', 'test']).default('test').describe('Dataset split to evaluate on'),
})

type Input = z.infer<typeof inputSchema>

type Output = {
	result: string
	durationMs: number
}

const DESCRIPTION = `🔮 Execute super-resolution inference using trained DiffSR models.

**Real Inference - Calls Python Forecasters!**

This tool runs ACTUAL inference using the DiffSR forecaster classes:
- BaseForecaster: Deterministic single-step inference
- DDPMForecaster: Denoising diffusion probabilistic model
- ResshiftForecaster: Residual shift diffusion model

## What This Tool Does

1. **Loads trained model** from model directory (best_model.pth + config.yaml)
2. **Builds data loaders** from the training configuration
3. **Executes inference** using the appropriate forecaster
4. **Computes metrics**: PSNR, SSIM, MSE, RMSE
5. **Saves results**: metrics.json to output_dir
6. **Generates report**: Automatic inference report with all metrics

## Forecastor Types

- **base**: Standard deterministic inference (fastest)
- **ddpm**: Diffusion model with iterative refinement (highest quality)
- **resshift**: Residual shift diffusion (balanced speed/quality)

## Parameters

- **model_dir**: Training output directory containing:
  - best_model.pth (model weights)
  - config.yaml (training configuration)

- **output_dir**: Where to save inference results:
  - metrics.json (computed metrics)
  - inference_report.md (auto-generated report)

- **split**: Which dataset split to evaluate:
  - test (default): Final evaluation
  - valid: Validation set
  - train: Training set (for debugging)

## Example Usage

{
  "forecastor_type": "base",
  "model_dir": "outputs/ocean_fno",
  "output_dir": "results/inference_test",
  "split": "test"
}

## Important Notes

- Model directory must contain both best_model.pth and config.yaml
- Data loaders are built from the config (uses same data as training)
- Inference automatically generates a complete report
- All metrics (PSNR, SSIM, etc.) are computed and saved`

export const DiffSRForecastorTool = {
	name: 'DiffSRForecastor',
	async description() {
		return DESCRIPTION
	},
	userFacingName() {
		return 'DiffSR Forecastor'
	},
	inputSchema,
	async isEnabled() {
		return true
	},
	isReadOnly() {
		return false
	},
	isConcurrencySafe() {
		return false
	},
	needsPermissions() {
		return true
	},
	async prompt() {
		return DESCRIPTION
	},
	renderToolUseMessage(input: Input, { verbose }: { verbose: boolean }) {
		return `${input.forecastor_type} forecaster${verbose ? ` on ${input.split} set` : ''}`
	},
	renderToolResultMessage(output: Output) {
		return (
			<Box flexDirection="column">
				<Text color="green">Forecastor operation complete ({output.durationMs}ms)</Text>
			</Box>
		)
	},
	renderResultForAssistant(output: Output) {
		return output.result
	},
	async *call(params: Input, { abortController }: { abortController: AbortController }) {
		const start = Date.now()

		try {
			// Resolve absolute paths
			for (const key of ['model_dir', 'output_dir'] as const) {
				if (params[key]) {
					params[key] = path.isAbsolute(params[key]) ? path.resolve(params[key]) : path.resolve(getCwd(), params[key])
				}
			}
			yield {
				type: 'progress' as const,
				content: createAssistantMessage('🔧 Initializing DiffSR inference...\n')
			}

			// Get runtime configuration
			const runtime = await OceanDepsManager.getRuntimeConfig()
			const diffsr_path = runtime.diffsr_path
			const python_path = runtime.python_path

			yield {
				type: 'progress' as const,
				content: createAssistantMessage(`✓ DiffSR path: ${diffsr_path}\n✓ Python: ${python_path}\n\n`)
			}

			// Verify files exist
			const fs = await import('fs/promises')
			const inferenceScript = path.join(diffsr_path, 'inference.py')
			const modelDir = path.resolve(getCwd(), params.model_dir)
			const modelPath = path.join(modelDir, 'best_model.pth')
			const configPath = path.join(modelDir, 'config.yaml')

			try {
				await fs.access(inferenceScript)
				await fs.access(modelPath)
				await fs.access(configPath)
			} catch (e) {
				throw new Error(`Required file not found: ${e}`)
			}

			yield {
				type: 'progress' as const,
				content: createAssistantMessage(
					`✓ Inference script: ${inferenceScript}\n` +
					`✓ Model directory: ${modelDir}\n` +
					`✓ Model checkpoint: best_model.pth\n` +
					`✓ Config file: config.yaml\n\n`
				)
			}

			// Prepare output directory
			const outputDir = path.resolve(getCwd(), params.output_dir)
			await fs.mkdir(outputDir, { recursive: true })

			yield {
				type: 'progress' as const,
				content: createAssistantMessage(`✓ Output directory: ${outputDir}\n\n`)
			}

			const inferenceCommand = `"${python_path}" "${inferenceScript}" --model_dir "${modelDir}" --forecastor_type ${params.forecastor_type} --output_dir "${outputDir}" --split ${params.split}`

			yield {
				type: 'progress' as const,
				content: createAssistantMessage(
					`🚀 Starting inference...\n` +
					`📝 Forecaster: ${params.forecastor_type}\n` +
					`📊 Split: ${params.split}\n\n`
				)
			}

			yield {
				type: 'progress' as const,
				content: createAssistantMessage('=' .repeat(60) + '\n')
			}
			yield {
				type: 'progress' as const,
				content: createAssistantMessage('INFERENCE OUTPUT:\n')
			}
			yield {
				type: 'progress' as const,
				content: createAssistantMessage('=' .repeat(60) + '\n\n')
			}

			// Execute inference using spawn for streaming output
			const { spawn } = await import('child_process')

			const inferenceProcess = spawn('sh', ['-c', inferenceCommand], {
				cwd: getCwd(),
				stdio: ['ignore', 'pipe', 'pipe'],
				env: {
					...process.env,
					PYTHONUNBUFFERED: '1',
				}
			})

			let allStdout = ''
			let allStderr = ''

			// Collect stdout
			inferenceProcess.stdout?.on('data', (data: Buffer) => {
				const text = data.toString()
				allStdout += text
			})

			// Collect stderr
			inferenceProcess.stderr?.on('data', (data: Buffer) => {
				const text = data.toString()
				allStderr += text
			})

			// Wait for process to complete
			const exitCode = await new Promise<number>((resolve, reject) => {
				inferenceProcess.on('exit', (code) => {
					resolve(code || 0)
				})
				inferenceProcess.on('error', (err) => {
					reject(err)
				})

				// Handle abort
				if (abortController.signal.aborted) {
					inferenceProcess.kill('SIGTERM')
					reject(new Error('Inference aborted by user'))
				}
				abortController.signal.addEventListener('abort', () => {
					inferenceProcess.kill('SIGTERM')
					reject(new Error('Inference aborted by user'))
				})
			})

			// Output complete logs
			if (allStdout) {
				yield {
					type: 'progress' as const,
					content: createAssistantMessage(allStdout + '\n')
				}
			}

			if (allStderr) {
				yield {
					type: 'progress' as const,
					content: createAssistantMessage(`\n⚠️  Warnings:\n${allStderr}\n`)
				}
			}

			yield {
				type: 'progress' as const,
				content: createAssistantMessage('\n' + '=' .repeat(60) + '\n')
			}

			// Check exit code
			if (exitCode !== 0) {
				throw new Error(`Inference failed with error ${allStderr || 'Unknown error'}`)
			}

			// Read metrics.json
			const metricsFile = path.join(outputDir, 'metrics.json')
			let metricsData: any = {}

			try {
				const metricsContent = await fs.readFile(metricsFile, 'utf-8')
				metricsData = JSON.parse(metricsContent)
			} catch (e) {
				yield {
					type: 'progress' as const,
					content: createAssistantMessage(`⚠️  Could not read metrics.json: ${e}\n`)
				}
			}

			// Format result message
			const resultMsg = `✅ Inference completed successfully!\n\n` +
				`📊 Results:\n` +
				`  - PSNR: ${metricsData.best_psnr?.toFixed(4) || 'N/A'} dB\n` +
				`  - SSIM: ${metricsData.best_ssim?.toFixed(4) || 'N/A'}\n` +
				`  - MSE: ${metricsData.test_metrics?.mse?.toFixed(6) || 'N/A'}\n` +
				`  - RMSE: ${metricsData.test_metrics?.rmse?.toFixed(6) || 'N/A'}\n\n` +
				`💾 Output:\n` +
				`  - Metrics: ${metricsFile}\n` +
				`  - Report: ${path.join(outputDir, 'inference_report.md')}\n`

			const output: Output = {
				result: resultMsg,
				durationMs: Date.now() - start,
			}

			yield {
				type: 'result' as const,
				resultForAssistant: this.renderResultForAssistant(output),
				data: output,
			}

		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : String(error)
			const output: Output = {
				result: `❌ Inference error: ${errorMessage}`,
				durationMs: Date.now() - start,
			}
			yield {
				type: 'result' as const,
				resultForAssistant: this.renderResultForAssistant(output),
				data: output,
			}
		}
	},
} satisfies Tool<typeof inputSchema, Output>
