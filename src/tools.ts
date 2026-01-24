import { Tool } from './Tool'
import { TaskTool } from './tools/4-SystemTools/TaskTool/TaskTool'
import { ArchitectTool } from './tools/4-SystemTools/ArchitectTool/ArchitectTool'
import { BashTool } from './tools/4-SystemTools/BashTool/BashTool'
import { AskExpertModelTool } from './tools/4-SystemTools/AskExpertModelTool/AskExpertModelTool'
import { FileEditTool } from './tools/4-SystemTools/FileEditTool/FileEditTool'
import { FileReadTool } from './tools/4-SystemTools/FileReadTool/FileReadTool'
import { FileWriteTool } from './tools/4-SystemTools/FileWriteTool/FileWriteTool'
import { GlobTool } from './tools/4-SystemTools/GlobTool/GlobTool'
import { GrepTool } from './tools/4-SystemTools/GrepTool/GrepTool'
import { LSTool } from './tools/4-SystemTools/lsTool/lsTool'
import { MemoryReadTool } from './tools/4-SystemTools/MemoryReadTool/MemoryReadTool'
import { MemoryWriteTool } from './tools/4-SystemTools/MemoryWriteTool/MemoryWriteTool'
import { MultiEditTool } from './tools/4-SystemTools/MultiEditTool/MultiEditTool'
import { NotebookEditTool } from './tools/4-SystemTools/NotebookEditTool/NotebookEditTool'
import { NotebookReadTool } from './tools/4-SystemTools/NotebookReadTool/NotebookReadTool'
import { ThinkTool } from './tools/4-SystemTools/ThinkTool/ThinkTool'
import { TodoWriteTool } from './tools/4-SystemTools/TodoWriteTool/TodoWriteTool'
import { WebSearchTool } from './tools/4-SystemTools/WebSearchTool/WebSearchTool'
import { URLFetcherTool } from './tools/4-SystemTools/URLFetcherTool/URLFetcherTool'
import { getMCPTools } from './services/mcpClient'
import { memoize } from 'lodash-es'
import { OceanPreprocessPipelineTool } from './tools/1-DatasetProcessTools/OceanPreprocessPipelineTool/OceanPreprocessPipelineTool'
import { OceanVisualizationTool } from './tools/3-ReportGeneratorTools/OceanVisualizationTool/OceanVisualizationTool'

// DiffSR tools (complete DiffSR-main integration)
import { DiffSRDatasetTool } from './tools/1-DatasetProcessTools/DiffSRDatasetTool/DiffSRDatasetTool'
import { DiffSRModelTool } from './tools/2-ModelsForTasksTools/3-SuperResolutionTools/DiffSRModelTool/DiffSRModelTool'
import { DiffSRForecastorTool } from './tools/2-ModelsForTasksTools/3-SuperResolutionTools/DiffSRForecastorTool/DiffSRForecastorTool'
import { DiffSRPipelineTool } from './tools/2-ModelsForTasksTools/3-SuperResolutionTools/DiffSRPipelineTool/DiffSRPipelineTool'
import { PredictionPipelineTool } from './tools/2-ModelsForTasksTools/2-PredictionTools/PredictionPipelineTool/PredictionPipelineTool'

const ANT_ONLY_TOOLS = [MemoryReadTool as unknown as Tool, MemoryWriteTool as unknown as Tool]

// Function to avoid circular dependencies that break bun
export const getAllTools = (): Tool[] => {
  return [
    TaskTool as unknown as Tool,
    AskExpertModelTool as unknown as Tool,
    BashTool as unknown as Tool,
    GlobTool as unknown as Tool,
    GrepTool as unknown as Tool,
    LSTool as unknown as Tool,
    FileReadTool as unknown as Tool,
    FileEditTool as unknown as Tool,
    MultiEditTool as unknown as Tool,
    FileWriteTool as unknown as Tool,
    NotebookReadTool as unknown as Tool,
    NotebookEditTool as unknown as Tool,
    ThinkTool as unknown as Tool,
    TodoWriteTool as unknown as Tool,
    WebSearchTool as unknown as Tool,
    URLFetcherTool as unknown as Tool,
    OceanPreprocessPipelineTool as unknown as Tool,
    OceanVisualizationTool as unknown as Tool,
    // DiffSR tools
    DiffSRDatasetTool as unknown as Tool,
    DiffSRModelTool as unknown as Tool,
    DiffSRForecastorTool as unknown as Tool,
    DiffSRPipelineTool as unknown as Tool,
    PredictionPipelineTool as unknown as Tool,
    ...ANT_ONLY_TOOLS,
  ]
}

export const getTools = memoize(
  async (enableArchitect?: boolean): Promise<Tool[]> => {
    const tools = [...getAllTools(), ...(await getMCPTools())]

    // Only include Architect tool if enabled via config or CLI flag
    if (enableArchitect) {
      tools.push(ArchitectTool as unknown as Tool)
    }

    const isEnabled = await Promise.all(tools.map(tool => tool.isEnabled()))
    return tools.filter((_, i) => isEnabled[i])
  },
)

export const getReadOnlyTools = memoize(async (): Promise<Tool[]> => {
  const tools = getAllTools().filter(tool => tool.isReadOnly())
  const isEnabled = await Promise.all(tools.map(tool => tool.isEnabled()))
  return tools.filter((_, index) => isEnabled[index])
})

export const getEditModeTools = memoize(async (): Promise<Tool[]> => {
  const tools = getAllTools()
  const isEnabled = await Promise.all(tools.map(tool => tool.isEnabled()))
  return tools.filter((_, index) => isEnabled[index])
})

export const getAskModeTools = memoize(async (): Promise<Tool[]> => {
  const tools = await getReadOnlyTools()
  // Exclude TaskTool in Ask mode
  return tools.filter(tool => tool.name !== TaskTool.name)
})