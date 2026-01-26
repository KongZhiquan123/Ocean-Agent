# 与后端的联调文档

本文件详细记录了Kode Agent与后端服务联调时需要遵循的接口规范、数据格式和调试注意事项。当前版本中，由于后端系统要求原始JSON数据，Kode Agent在服务模式下将直接返回未经JSX包装的工具调用数据，以便后端进行直接处理。

## 基础响应格式

所有工具调用的响应都遵循以下基础格式：

```json
{
  "type": "backend_only",
  "tool_name": "string",
  "tool_use_id": "string",
  "uuid": "UUID",
  "data": "具体工具的返回数据，详见下方各工具说明"
}
```

---

## 1. FileWriteTool (工具名: Replace)

### 功能说明
用于创建新文件或完全替换现有文件内容。

### data 字段格式

#### 创建文件时 (type: 'create')

| 字段名 | 类型 | 说明 |
|--------|------|------|
| type | `'create'` | 操作类型：创建 |
| filePath | `string` | 文件的绝对路径 |
| content | `string` | 文件的完整内容 |
| structuredPatch | `[]` | 空数组（新建文件无补丁） |

**示例：**
```json
{
  "type": "backend_only",
  "tool_name": "Replace",
  "tool_use_id": "toolu_01ABC123",
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "type": "create",
    "filePath": "/home/user/project/new_file.py",
    "content": "def hello():\n    print('Hello, World!')\n",
    "structuredPatch": []
  }
}
```

#### 更新文件时 (type: 'update')

| 字段名 | 类型 | 说明 |
|--------|------|------|
| type | `'update'` | 操作类型：更新 |
| filePath | `string` | 文件的绝对路径 |
| content | `string` | 文件更新后的完整内容 |
| structuredPatch | `Hunk[]` | 结构化的差异补丁数组 |

**示例：**
```json
{
  "type": "backend_only",
  "tool_name": "Replace",
  "tool_use_id": "toolu_01ABC124",
  "uuid": "550e8400-e29b-41d4-a716-446655440001",
  "data": {
    "type": "update",
    "filePath": "/home/user/project/existing_file.py",
    "content": "def hello():\n    print('Hello, Updated World!')\n",
    "structuredPatch": [
      {
        "oldStart": 1,
        "oldLines": 2,
        "newStart": 1,
        "newLines": 2,
        "lines": [
          " def hello():",
          "-    print('Hello, World!')",
          "+    print('Hello, Updated World!')"
        ]
      }
    ]
  }
}
```

---

## 2. FileEditTool (工具名: Edit)

### 功能说明
用于精确替换文件中的指定字符串，适合小范围修改。

### data 字段格式

| 字段名 | 类型 | 说明 |
|--------|------|------|
| filePath | `string` | 文件的绝对路径 |
| oldString | `string` | 被替换的原始字符串 |
| newString | `string` | 替换后的新字符串 |
| originalFile | `string` | 文件修改前的完整内容 |
| structuredPatch | `Hunk[]` | 结构化的差异补丁数组 |

**示例：**
```json
{
  "type": "backend_only",
  "tool_name": "Edit",
  "tool_use_id": "toolu_01ABC125",
  "uuid": "550e8400-e29b-41d4-a716-446655440002",
  "data": {
    "filePath": "/home/user/project/config.py",
    "oldString": "DEBUG = False",
    "newString": "DEBUG = True",
    "originalFile": "# Configuration\nDEBUG = False\nPORT = 8000\n",
    "structuredPatch": [
      {
        "oldStart": 2,
        "oldLines": 1,
        "newStart": 2,
        "newLines": 1,
        "lines": [
          "-DEBUG = False",
          "+DEBUG = True"
        ]
      }
    ]
  }
}
```

---

## 3. MultiEditTool (工具名: MultiEdit)

### 功能说明
用于在单个文件上原子性地执行多个编辑操作。

### data 字段格式

#### 成功时的返回格式

| 字段名 | 类型 | 说明 |
|--------|------|------|
| filePath | `string` | 文件的绝对路径 |
| wasNewFile | `boolean` | 是否是新创建的文件 |
| editsApplied | `EditResult[]` | 已应用的编辑操作数组 |
| totalEdits | `number` | 总编辑操作数 |
| summary | `string` | 操作摘要信息 |
| structuredPatch | `Hunk[]` | 结构化的差异补丁数组 |

**EditResult 对象结构：**

| 字段名 | 类型 | 说明 |
|--------|------|------|
| editIndex | `number` | 编辑操作的索引（从1开始） |
| success | `boolean` | 操作是否成功 |
| old_string | `string` | 被替换的字符串（截取前100字符） |
| new_string | `string` | 替换后的字符串（截取前100字符） |
| occurrences | `number` | 替换的次数 |

**示例：**
```json
{
  "type": "backend_only",
  "tool_name": "MultiEdit",
  "tool_use_id": "toolu_01ABC126",
  "uuid": "550e8400-e29b-41d4-a716-446655440003",
  "data": {
    "filePath": "/home/user/project/utils.py",
    "wasNewFile": false,
    "editsApplied": [
      {
        "editIndex": 1,
        "success": true,
        "old_string": "import os",
        "new_string": "import os\nimport sys",
        "occurrences": 1
      },
      {
        "editIndex": 2,
        "success": true,
        "old_string": "def process():",
        "new_string": "def process_data():",
        "occurrences": 1
      }
    ],
    "totalEdits": 2,
    "summary": "Successfully applied 2 edits to utils.py",
    "structuredPatch": [
      {
        "oldStart": 1,
        "oldLines": 1,
        "newStart": 1,
        "newLines": 2,
        "lines": [
          " import os",
          "+import sys"
        ]
      },
      {
        "oldStart": 5,
        "oldLines": 1,
        "newStart": 6,
        "newLines": 1,
        "lines": [
          "-def process():",
          "+def process_data():"
        ]
      }
    ]
  }
}
```

#### 失败时的返回格式

当编辑操作失败时，data 字段为错误消息字符串：

```json
{
  "type": "backend_only",
  "tool_name": "MultiEdit",
  "tool_use_id": "toolu_01ABC127",
  "uuid": "550e8400-e29b-41d4-a716-446655440004",
  "data": "Error in edit 2: String not found: def process()..."
}
```

---

## 4. NotebookEditTool (工具名: NotebookEditCell)

### 功能说明
用于编辑 Jupyter Notebook (.ipynb) 文件的单元格。

### data 字段格式

| 字段名 | 类型 | 说明 |
|--------|------|------|
| cell_number | `number` | 单元格索引（从0开始） |
| new_source | `string` | 单元格的新内容 |
| cell_type | `'code' \| 'markdown'` | 单元格类型 |
| language | `string` | 编程语言（如 'python', 'julia' 等） |
| edit_mode | `'replace' \| 'insert' \| 'delete'` | 编辑模式 |
| error | `string` (可选) | 错误信息（如有） |

**成功示例 - 替换模式 (replace)：**
```json
{
  "type": "backend_only",
  "tool_name": "NotebookEditCell",
  "tool_use_id": "toolu_01ABC128",
  "uuid": "550e8400-e29b-41d4-a716-446655440005",
  "data": {
    "cell_number": 3,
    "new_source": "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.head()",
    "cell_type": "code",
    "language": "python",
    "edit_mode": "replace"
  }
}
```

**成功示例 - 插入模式 (insert)：**
```json
{
  "type": "backend_only",
  "tool_name": "NotebookEditCell",
  "tool_use_id": "toolu_01ABC129",
  "uuid": "550e8400-e29b-41d4-a716-446655440006",
  "data": {
    "cell_number": 5,
    "new_source": "# Data Visualization\nThis section contains plots and charts.",
    "cell_type": "markdown",
    "language": "python",
    "edit_mode": "insert"
  }
}
```

**成功示例 - 删除模式 (delete)：**
```json
{
  "type": "backend_only",
  "tool_name": "NotebookEditCell",
  "tool_use_id": "toolu_01ABC130",
  "uuid": "550e8400-e29b-41d4-a716-446655440007",
  "data": {
    "cell_number": 2,
    "new_source": "",
    "cell_type": "code",
    "language": "python",
    "edit_mode": "delete"
  }
}
```

**失败示例：**
```json
{
  "type": "backend_only",
  "tool_name": "NotebookEditCell",
  "tool_use_id": "toolu_01ABC131",
  "uuid": "550e8400-e29b-41d4-a716-446655440008",
  "data": {
    "cell_number": 10,
    "new_source": "print('test')",
    "cell_type": "code",
    "language": "python",
    "edit_mode": "replace",
    "error": "Cell number is out of bounds. Notebook has 8 cells."
  }
}
```

---

## 附录：Hunk 类型说明

`Hunk` 类型表示文件差异的一个片段，遵循统一差异格式（unified diff format）：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| oldStart | `number` | 原文件中的起始行号 |
| oldLines | `number` | 原文件中的行数 |
| newStart | `number` | 新文件中的起始行号 |
| newLines | `number` | 新文件中的行数 |
| lines | `string[]` | 差异行数组，以 ' '(不变)、'-'(删除)、'+'(添加) 开头 |

**Hunk 示例：**
```json
{
  "oldStart": 10,
  "oldLines": 3,
  "newStart": 10,
  "newLines": 4,
  "lines": [
    " def calculate(x):",
    "-    return x * 2",
    "+    # Updated calculation",
    "+    return x * 3",
    " "
  ]
}
```
