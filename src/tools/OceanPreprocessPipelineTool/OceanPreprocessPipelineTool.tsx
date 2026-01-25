import { Box, Text } from 'ink'
import { z } from 'zod'
import type { Tool } from '@tool'
import { DESCRIPTION, PROMPT } from './prompt'
import { getCwd } from '@utils/state'
import {
  hasReadPermission,
  hasWritePermission,
} from '@utils/permissions/filesystem'
import { OceanDepsManager } from '@utils/oceanDepsManager'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'

const inputSchema = z.strictObject({
  input_dir: z.string().describe('input data directory path'),
  output_dir: z.string().describe('output data directory path'),
  file_pattern: z
    .string()
    .optional()
    .default('*.nc')
    .describe('file matching pattern, default *.nc'),
  variable_name: z
    .string()
    .optional()
    .default('sst')
    .describe('variable name, default sst'),
  use_cnn_validation: z
    .boolean()
    .optional()
    .default(true)
    .describe('whether to use CNN validation (requires PyTorch)'),
})

type Input = z.infer<typeof inputSchema>

type Output = {
  durationMs: number
  exitCode: number | null
  stdout: string
  stderr: string
  reportPath: string
  scriptPath: string
}

export const OceanPreprocessPipelineTool = {
  name: 'OceanPreprocessPipeline',
  async description() {
    return DESCRIPTION
  },
  inputSchema,
  userFacingName() {
    return 'Ocean Preprocess Pipeline'
  },
  async isEnabled() {
    return true
  },
  isReadOnly() {
    return false
  },
  isConcurrencySafe() {
    return false
  },
  needsPermissions(input?: Input) {
    if (!input) return true
    return (
      !hasReadPermission(input.input_dir) ||
      !hasWritePermission(input.output_dir)
    )
  },
  async prompt() {
    return PROMPT
  },
  renderToolUseMessage(
    {
      input_dir,
      output_dir,
      variable_name,
      file_pattern,
      use_cnn_validation,
    }: Input,
    { verbose }: { verbose: boolean },
  ) {
    const parts = [
      `var=${variable_name}`,
      `src=${input_dir}`,
      `dst=${output_dir}`,
    ]

    if (verbose) {
      parts.push(
        `pattern=${file_pattern}`,
        `cnn=${use_cnn_validation ? 'on' : 'off'}`,
      )
    }

    return parts.join(' | ')
  },
  renderToolResultMessage(output: Output) {
    const success = output.exitCode === 0
    return (
      <Box flexDirection="column">
        <Text color={success ? 'green' : 'red'}>
          {success ? '✅ Preprocessing completed' : '❌ Preprocessing failed'}{' '}
          (duration {output.durationMs}ms)
        </Text>
        <Text dimColor>Script: {output.scriptPath}</Text>
        <Text dimColor>Report: {output.reportPath}</Text>
        {output.stderr && (
          <Text color="yellow">Warning: {output.stderr.trim()}</Text>
        )}
      </Box>
    )
  },
  renderResultForAssistant(output: Output) {
    const status = output.exitCode === 0 ? 'success' : 'failed'
    const summary = [
      `status: ${status}`,
      `report: ${output.reportPath}`,
      `output: ${output.stdout.trim()}`
    ]
      .filter(Boolean)
      .join('\n')
    return summary
  },
  async *call(
    input: Input,
    { abortController }: { abortController: AbortController },
  ) {
    const {
      input_dir,
      output_dir,
      file_pattern,
      variable_name,
      use_cnn_validation,
    } = input
    const start = Date.now()
    const cwd = getCwd()

    try {
      if (!existsSync(input_dir)) {
        throw new Error(`Input directory does not exist: ${input_dir}`)
      }
      const env = {
        ...process.env,
        PREPROCESS_INPUT_DIR: input_dir,
        PREPROCESS_OUTPUT_DIR: output_dir,
        PREPROCESS_FILE_PATTERN: file_pattern,
        PREPROCESS_VARIABLE: variable_name,
      }
      const stdoutChunks: string[] = []
      const stderrChunks: string[] = []

      // Find Python executable and script path
      const pythonCmd = await OceanDepsManager.findPython()
      const preprocessingDir = await OceanDepsManager.ensurePreprocessing()
      const scriptPath = path.join(preprocessingDir, 'main.py')

      // 因为python脚本中使用了相对导入，所以不要求工作目录必须是脚本所在目录(即不需要cd命令)，但需要确保脚本路径正确
      const executeCommand = `${pythonCmd} ${scriptPath} ${use_cnn_validation ? '' : '--simple'}`.trim()
      const exitCode = await new Promise<number | null>((resolve, reject) => {
        const proc = spawn('sh', ['-c', executeCommand], {
          cwd,
          stdio: ['ignore', 'pipe', 'pipe'],
          env,
        })

        const abort = () => {
          proc.kill('SIGTERM')
          reject(new Error('Process aborted by user'))
        }

        abortController.signal.addEventListener('abort', abort)

        proc.stdout.on('data', (data: Buffer) => {
          stdoutChunks.push(data.toString())
        })

        proc.stderr.on('data', (data: Buffer) => {
          stderrChunks.push(data.toString())
        })

        proc.on('error', err => {
          abortController.signal.removeEventListener('abort', abort)
          reject(err)
        })

        proc.on('close', code => {
          abortController.signal.removeEventListener('abort', abort)
          resolve(code)
        })
      })
      if (exitCode !== 0) {
        throw new Error(stderrChunks.join('').trim() || 'Preprocessing failed')
      }
      const output: Output = {
        durationMs: Date.now() - start,
        exitCode,
        stdout: stdoutChunks.join('').trim(),
        stderr: stderrChunks.join('').trim(),
        reportPath: path.join(output_dir, 'validation_report.md'),
        scriptPath,
      }

      yield {
        type: 'result' as const,
        resultForAssistant: this.renderResultForAssistant(output),
        data: output,
      }
    } catch (err) {
      yield {
        type: 'result' as const,
        resultForAssistant: `status: failed\nerror: ${(err as Error).message}`,
        data: {
          durationMs: Date.now() - start,
          exitCode: 1,
          stdout: '',
          stderr: (err as Error).message,
          reportPath: '',
          scriptPath: '',
        },
      }
    }
  },
} satisfies Tool<typeof inputSchema, Output>
