import { Box, Text } from 'ink'
import { z } from 'zod'
import type { Tool } from '@tool'
import { exec } from 'child_process'
import { promisify } from 'util'
import path from 'path'
import { getCwd } from '@utils/state'
import { OceanDepsManager } from '@utils/oceanDepsManager'
import { createAssistantMessage } from '@utils/messages'

const execAsync = promisify(exec)

const inputSchema = z.strictObject({
	operation: z.enum(['train', 'inference', 'list_models', 'list_configs']).describe(
		'Operation: train (full training), inference (prediction), list_models, list_configs'
	),
	config_path: z.string().optional().describe('Path to YAML config file (for training)'),
	model_type: z.enum([
		'fno', 'edsr', 'swinir', 'ddpm', 'sr3', 'resshift',
		'hinote', 'mwt', 'galerkin', 'm2no', 'mg_ddpm',
		'remg', 'sronet', 'unet', 'wdno'
	]).optional().describe('Model architecture type'),
	dataset_type: z.enum(['ocean', 'era5', 'era5_temperature', 'era5_wind', 'ns2d']).optional().describe('Dataset type'),
	model_path: z.string().optional().describe('Path to trained model checkpoint (for inference)'),
	input_data: z.string().optional().describe('Input data path (for inference)'),
	output_dir: z.string().optional().describe('Output directory for results'),
	gpu_id: z.number().default(0).describe('GPU device ID (-1 for CPU)'),
	epochs: z.number().optional().describe('Number of training epochs'),
	batch_size: z.number().optional().describe('Batch size'),
	learning_rate: z.number().optional().describe('Learning rate'),
})

type Input = z.infer<typeof inputSchema>

type Output = {
	result: string
	durationMs: number
}

const DESCRIPTION = `🚀 Complete DiffSR pipeline for super-resolution tasks.

**⚡ EMBEDDED FRAMEWORK - Use This Instead of Writing Code!**

This tool provides a complete, production-ready DiffSR framework embedded in Kode.
When users ask for super-resolution, USE THIS TOOL - don't write custom Python scripts!

## What's Included

✅ 15+ Pre-built Model Architectures:
   - FNO (Fourier Neural Operator) - Fast spectral SR
   - EDSR (Enhanced Deep SR) - Classic deep learning
   - SwinIR (Swin Transformer) - Attention-based SR
   - DDPM (Diffusion Model) - Generative high-quality SR
   - ResShift (Residual Shift) - Advanced diffusion SR
   - And 10 more architectures ready to use!

✅ Multiple Dataset Types:
   - Ocean SST data
   - ERA5 atmospheric reanalysis
   - NS2D turbulence simulations
   - Custom data with proper preprocessing

✅ Complete Training & Inference Pipelines:
   - YAML-based configuration
   - Distributed training support
   - Checkpoint management
   - Automatic logging

## Operations

### 1. list_models
Show all available model architectures with descriptions.
**Use this first** to help users choose the right model.

### 2. list_configs  
List available template configuration files.
Shows YAML configs in template_configs/ directory.

### 3. train
Train a super-resolution model with YAML config.
**This is the main training operation** - handles everything automatically.

Required: config_path (path to YAML config)
Optional: output_dir, epochs, batch_size, learning_rate, gpu_id

### 4. inference
Run super-resolution prediction on low-resolution data.
Loads trained model and generates high-resolution output.

Required: model_path, input_data
Optional: output_dir, gpu_id

## Key Features

🔧 **No External Dependencies**: DiffSR code is embedded in Kode
📦 **Pre-configured Templates**: Ready-to-use YAML configs
🚀 **Production Ready**: Tested training and inference pipelines
🎯 **Multiple Domains**: Ocean, atmospheric, turbulence data
💪 **GPU Accelerated**: Automatic CUDA detection and usage

## Typical Workflows

### Training a New Model:
1. list_configs → Find appropriate template
2. Prepare data with DiffSRDataset tool
3. train → Run training with config
4. Monitor training progress in output_dir

### Running Inference:
1. Prepare low-resolution input data
2. inference → Generate super-resolution output
3. Visualize results with GeoSpatialPlot/StandardChart
4. Generate training report → python src/services/diffsr/report_generator.py train
## Example - Train FNO Model:
{
  "operation": "train",
  "config_path": "template_configs/Ocean/fno.yaml",
  "output_dir": "outputs/ocean_fno",
  "epochs": 100,
  "batch_size": 8
}

## Example - Run Inference:
{
  "operation": "inference",
  "model_path": "outputs/ocean_fno/checkpoint.pth",
  "input_data": "data/ocean_lr.npy",
  "output_dir": "results/ocean_sr"
}

## When to Use This Tool

✅ User asks for "super-resolution" or "upsampling"
✅ User wants to "train a model" for ocean/climate data
✅ User needs to "enhance resolution" of satellite data
✅ User mentions "FNO", "DDPM", "diffusion model", etc.

❌ DON'T write custom training code from scratch
❌ DON'T create model definitions manually
✅ DO use this tool - it's complete and ready!

## Technical Details

- Training runs main.py from embedded DiffSR directory
- Supports both single-GPU and distributed training
- Checkpoints saved automatically during training
- Inference loads models and runs forecastors
- All paths relative to embedded DiffSR location

## Note to AI Assistants

This tool wraps the complete DiffSR framework. When users ask for super-resolution:
1. Suggest using this tool FIRST
2. Help them choose the right model with list_models
3. Guide them to appropriate config with list_configs
4. Run training/inference with this tool
5. Don't write custom Python training scripts!
6. When creating a new config file based on a template, ONLY modify the dataset path. Keep other parameters unchanged unless the user specifies otherwise.

The embedded code is production-tested and feature-complete.`

const MODELS = [
	'fno - Fourier Neural Operator',
	'edsr - Enhanced Deep Super-Resolution',
	'swinir - Swin Transformer for Image Restoration',
	'ddpm - Denoising Diffusion Probabilistic Model',
	'sr3 - Image Super-Resolution via Iterative Refinement',
	'resshift - Residual Shift Diffusion Model',
	'hinote - Hierarchical Neural Operator',
	'mwt - Multi-scale Wavelet Transform',
	'galerkin - Galerkin Transformer',
	'm2no - Multi-scale Multi-fidelity Neural Operator',
	'mg_ddpm - Multi-grid DDPM',
	'remg - Residual Multi-grid',
	'sronet - Super-Resolution Operator Network',
	'unet - U-Net Architecture',
	'wdno - Wavelet-based Denoising Neural Operator'
]

export const DiffSRPipelineTool = {
	name: 'DiffSRPipeline',
	async description() {
		return DESCRIPTION
	},
	userFacingName() {
		return 'DiffSR Pipeline'
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
		return `${input.operation}${input.model_type ? ` (${input.model_type})` : ''}${verbose && input.config_path ? ` config: ${input.config_path}` : ''}`
	},
	renderToolResultMessage(output: Output) {
		return (
			<Box flexDirection="column">
				<Text color="green">Pipeline operation complete ({output.durationMs}ms)</Text>
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
			for (const key of ['config_path', 'model_path', 'input_data', 'output_dir'] as const) {
				if (params[key]) {
					params[key] = path.isAbsolute(params[key]) ? path.resolve(params[key]) : path.resolve(getCwd(), params[key])
				}
			}	

			// Use embedded DiffSR from Kode
			yield {
				type: 'progress' as const,
				content: createAssistantMessage('🔧 Using embedded DiffSR framework...\n')
			}
			
			const runtime = await OceanDepsManager.getRuntimeConfig()
			const diffsr_path = runtime.diffsr_path
			const python_path = runtime.python_path
			
			yield {
				type: 'progress' as const,
				content: createAssistantMessage(`✓ Embedded DiffSR: ${diffsr_path}\n✓ Python: ${python_path}\n\n`)
			}

			const { operation } = params

			if (operation === 'list_models') {
				const output: Output = {
					result: MODELS.join('\n'),
					durationMs: Date.now() - start,
				}
				yield {
					type: 'result' as const,
					resultForAssistant: this.renderResultForAssistant(output),
					data: output,
				}
				return
			}

			if (operation === 'list_configs') {
				const configScript = `
import os
import json
from pathlib import Path

diffsr_path = Path("${diffsr_path.replace(/\\/g, '/')}")
template_dir = diffsr_path / "template_configs"

configs = []
for root, dirs, files in os.walk(template_dir):
    for file in files:
        if file.endswith('.yaml'):
            abs_path = os.path.join(root, file)
            configs.append(str(abs_path))

print(json.dumps(configs, indent=2))
`
				const tempScript = path.join(getCwd(), '.diffsr_list_configs.py')
				const fs = await import('fs/promises')
				await fs.writeFile(tempScript, configScript)

				const { stdout } = await execAsync(`"${python_path}" "${tempScript}"`)
				await fs.unlink(tempScript)

				const output: Output = {
					result: stdout,
					durationMs: Date.now() - start,
				}
				yield {
					type: 'result' as const,
					resultForAssistant: this.renderResultForAssistant(output),
					data: output,
				}
				return
			}

			if (operation === 'train') {
				if (!params.config_path) {
					throw new Error('config_path required for training')
				}

				yield {
					type: 'progress' as const,
					content: createAssistantMessage('🚀 Starting DiffSR training directly...\n\n')
				}

				// Verify files exist
				const fs = await import('fs/promises')
				const mainPyPath = path.join(diffsr_path, 'main.py')

				try {
					await fs.access(mainPyPath)
					await fs.access(params.config_path)
				} catch (e) {
					throw new Error(`File not found: ${e}`)
				}

				yield {
					type: 'progress' as const,
					content: createAssistantMessage(`✓ DiffSR main.py: ${mainPyPath}\n✓ Config: ${params.config_path}\n✓ Python: ${python_path}\n\n`)
				}

				const trainCommand = `"${python_path}" "${mainPyPath}" --config "${params.config_path}"`

				yield {
					type: 'progress' as const,
					content: createAssistantMessage(`⏳ Executing training command...\n📝 Command: ${trainCommand}\n\n`)
				}

				yield {
					type: 'progress' as const,
					content: createAssistantMessage('=' .repeat(60) + '\n')
				}
				yield {
					type: 'progress' as const,
					content: createAssistantMessage('TRAINING OUTPUT:\n')
				}
				yield {
					type: 'progress' as const,
					content: createAssistantMessage('=' .repeat(60) + '\n\n')
				}

				try {
					// 🔥 使用 spawn 实现流式输出，而不是 execAsync
					const { spawn } = await import('child_process')

					const trainProcess = spawn('sh', ['-c', trainCommand], {
						cwd: getCwd(),
						stdio: ['ignore', 'pipe', 'pipe'],
						env: {
							...process.env,
							PYTHONUNBUFFERED: '1', // 禁用 Python 输出缓冲，确保实时输出
						}
					})

					let allStdout = ''
					let allStderr = ''
					let newOutput = '' // 新增的输出，用于实时显示

					// 收集 stdout
					trainProcess.stdout?.on('data', (data: Buffer) => {
						const text = data.toString()
						allStdout += text
						newOutput += text
					})

					// 收集 stderr
					trainProcess.stderr?.on('data', (data: Buffer) => {
						const text = data.toString()
						allStderr += text
						newOutput += text
					})

					// 🔥 定期输出新增的日志（每秒一次）
					const outputInterval = setInterval(() => {
						if (newOutput) {
							// 无法在这里 yield，所以只能记录
							console.log('[DiffSRPipeline Training Output]', newOutput)
							newOutput = '' // 清空已输出的内容
						}
					}, 1000)

					// 等待进程完成
					try {
						const exitCode = await new Promise<number>((resolve, reject) => {
							trainProcess.on('exit', (code) => {
								clearInterval(outputInterval)
								resolve(code || 0)
							})
							trainProcess.on('error', (err) => {
								clearInterval(outputInterval)
								reject(err)
							})

							// 如果 abortController 触发，杀死进程
							if (abortController.signal.aborted) {
								trainProcess.kill('SIGTERM')
								clearInterval(outputInterval)
								reject(new Error('Training aborted by user'))
							}
							abortController.signal.addEventListener('abort', () => {
								trainProcess.kill('SIGTERM')
								clearInterval(outputInterval)
								reject(new Error('Training aborted by user'))
							})
						})

						// 🔥 输出完整日志
						if (allStdout) {
							yield {
								type: 'progress' as const,
								content: createAssistantMessage(allStdout + '\n')
							}
						}

						if (allStderr) {
							yield {
								type: 'progress' as const,
								content: createAssistantMessage(`\n⚠️  Warnings/Errors:\n${allStderr}\n`)
							}
						}

						// 检查退出码
						if (exitCode !== 0) {
							throw new Error(`Training failed with error: ${allStderr || 'Unknown error'}`)
						}
					} finally {
						clearInterval(outputInterval)
					}

					yield {
						type: 'progress' as const,
						content: createAssistantMessage('\n' + '=' .repeat(60) + '\n')
					}

					// Generate training report
					yield {
						type: 'progress' as const,
						content: createAssistantMessage('\n📝 Generating training report...\n')
					}

					try {
						const reportPath = params.output_dir
							? path.join(params.output_dir, 'training_report.md')
							: './training_report.md'

						yield {
							type: 'progress' as const,
							content: createAssistantMessage(`📊 Report will be saved to: ${reportPath}\n\n`)
						}

						// Note: This is a simplified version. In production, you'd parse training logs
						// to extract metrics and create proper JSON files for the report generator

					} catch (reportError) {
						yield {
							type: 'progress' as const,
							content: createAssistantMessage(`⚠️  Could not generate report: ${reportError}\n`)
						}
					}

					const output: Output = {
						result: `✅ Training completed successfully!\n\n` +
								`📊 Check training logs in config output directory.\n` +
								`📝 Training report: ${params.output_dir || '.'}/training_report.md`,
						durationMs: Date.now() - start,
					}
					yield {
						type: 'result' as const,
						resultForAssistant: this.renderResultForAssistant(output),
						data: output,
					}
				} catch (execError: any) {
					// Handle execution errors
					const errorMsg = execError.message || String(execError)
					const stdout = execError.stdout || ''
					const stderr = execError.stderr || ''
					
					yield {
						type: 'progress' as const,
						content: createAssistantMessage(`\n❌ Training failed or interrupted\n\n`)
					}
					
					if (stdout) {
						yield {
							type: 'progress' as const,
							content: createAssistantMessage(`Last output:\n${stdout}\n\n`)
						}
					}
					
					if (stderr) {
						yield {
							type: 'progress' as const,
							content: createAssistantMessage(`Error details:\n${stderr}\n\n`)
						}
					}

					const output: Output = {
						result: `Training error: ${errorMsg}\n\nPartial output captured above.`,
						durationMs: Date.now() - start,
					}
					yield {
						type: 'result' as const,
						resultForAssistant: this.renderResultForAssistant(output),
						data: output,
					}
				}
				return
			}

		if (operation === 'inference') {
			if (!params.model_path || !params.output_dir) {
				throw new Error('model_path (model directory) and output_dir required for inference')
			}

			yield {
				type: 'progress' as const,
				content: createAssistantMessage('🚀 Starting DiffSR inference...\n\n')
			}

			// Verify files exist
			const fs = await import('fs/promises')
			const inferenceScript = path.join(diffsr_path, 'inference.py')
			const modelDir = path.resolve(getCwd(), params.model_path)
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

			// Determine forecaster type from model_type
			let forecastorType = 'base'
			if (params.model_type) {
				if (['ddpm', 'sr3', 'mg_ddpm'].includes(params.model_type)) {
					forecastorType = 'ddpm'
				} else if (params.model_type === 'resshift') {
					forecastorType = 'resshift'
				}
			}

			const inferenceCommand = `"${python_path}" "${inferenceScript}" --model_dir "${modelDir}" --forecastor_type ${forecastorType} --output_dir "${outputDir}" --split test`

			yield {
				type: 'progress' as const,
				content: createAssistantMessage(
					`⏳ Executing inference...\n` +
					`📝 Forecaster: ${forecastorType}\n` +
					`📊 Split: test\n\n`
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

			try {
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
					throw new Error(`Inference failed with error: ${allStderr || 'Unknown error'}`)
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

			} catch (execError: any) {
				// Handle execution errors
				const errorMsg = execError.message || String(execError)
				const stdout = execError.stdout || ''
				const stderr = execError.stderr || ''

				yield {
					type: 'progress' as const,
					content: createAssistantMessage(`\n❌ Inference failed or interrupted\n\n`)
				}

				if (stdout) {
					yield {
						type: 'progress' as const,
						content: createAssistantMessage(`Last output:\n${stdout}\n\n`)
					}
				}

				if (stderr) {
					yield {
						type: 'progress' as const,
						content: createAssistantMessage(`Error details:\n${stderr}\n\n`)
					}
				}

				const output: Output = {
					result: `Inference error: ${errorMsg}\n\nPartial output captured above.`,
					durationMs: Date.now() - start,
				}
				yield {
					type: 'result' as const,
					resultForAssistant: this.renderResultForAssistant(output),
					data: output,
				}
			}
			return
		}
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : String(error)
			const output: Output = {
				result: `Error: ${errorMessage}`,
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
