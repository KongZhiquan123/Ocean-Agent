/**
 * Kode CLI 核心 API 导出
 * 供后端 Web API 调用
 *
 * 使用方式:
 * import { query, getAllTools, getContext } from '@shareai-lab/kode/api'
 */

// ============ 必需的 SDK 初始化 ============
// Anthropic SDK 要求先导入 shims
import '@anthropic-ai/sdk/shims/node'

// ============ 核心查询函数 ============
/**
 * 发送消息到 Claude 并获取流式响应
 * 这是最核心的函数，后端主要调用这个
 */
export { query } from './query'
export type { Message, UserMessage, AssistantMessage } from './query'
/**
 * Claude API 直接调用函数
 */
export { queryLLM, queryQuick } from './services/claude'

// ============ 工具系统 ============
/**
 * 获取所有可用工具（包括自定义工具）
 */
export { getAllTools } from './tools'

// ============ 上下文管理 ============
/**
 * 获取项目上下文（目录结构、Git 状态等）
 */
export { getContext } from './context'

// ============ 模型管理 ============
/**
 * 模型管理器，用于切换和管理不同的 AI 模型
 */
export { ModelManager } from './utils/model'

// ============ 配置管理 ============
/**
 * 配置文件读写
 */
export {
  getGlobalConfig,
  getCurrentProjectConfig,
  saveGlobalConfig,
  saveCurrentProjectConfig,
} from './utils/config'

// ============ 工具和工具使用上下文类型 ============
export type { Tool, ToolUseContext, ExtendedToolUseContext } from './Tool'
