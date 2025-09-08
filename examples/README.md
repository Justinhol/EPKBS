# 📝 EPKBS 示例代码

这里包含了 EPKBS 系统的各种使用示例，从基础功能到高级特性。

## 📂 示例分类

### 🔰 基础示例 (`basic/`)

适合初学者，展示核心功能的基本用法：

- **[document_parsing_demo.py](basic/document_parsing_demo.py)** - 文档解析基础示例

  - 支持多种文档格式
  - 展示解析结果处理
  - 错误处理最佳实践

- **[simple_rag.py](basic/simple_rag.py)** - RAG 系统基础使用

  - 文档索引和检索
  - 简单问答功能
  - 结果展示

- **[agent_chat.py](basic/agent_chat.py)** - 统一 Agent 对话示例
  - 智能任务分析
  - 自动工具选择
  - 统一执行接口

### 🚀 高级示例 (`advanced/`)

展示系统的高级特性和复杂用法：

- **[langgraph_showcase.py](advanced/langgraph_showcase.py)** - 🆕 LangGraph 功能展示

  - 智能工作流编排
  - 状态管理和可视化
  - 并行处理优化
  - 性能对比分析

- **[mcp_usage.py](advanced/mcp_usage.py)** - MCP 系统完整示例

  - MCP 服务器管理
  - 智能工具选择
  - 批量操作演示
  - 性能监控

- **[batch_processing.py](advanced/batch_processing.py)** - 批量文档处理

  - 大规模文档处理
  - 并行处理优化
  - 进度监控

- **[custom_tools.py](advanced/custom_tools.py)** - 自定义工具开发
  - MCP 工具扩展
  - 统一 Agent 集成
  - 工具调用示例

### 📓 Jupyter 示例 (`notebooks/`)

交互式教程和演示：

- **[rag_demo.ipynb](notebooks/rag_demo.ipynb)** - RAG 系统交互式演示
- **[mcp_tutorial.ipynb](notebooks/mcp_tutorial.ipynb)** - MCP 功能教程
- **[document_analysis.ipynb](notebooks/document_analysis.ipynb)** - 文档分析工作流

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活epkbs conda环境
conda activate epkbs

# 验证环境配置
source scripts/setup/activate_environment.sh
python scripts/setup/check_environment.py

# 初始化系统
python scripts/setup/init_system.py
```

### 2. 运行基础示例

```bash
# 文档解析示例
python examples/basic/document_parsing_demo.py

# RAG问答示例
python examples/basic/simple_rag.py

# 统一Agent对话示例
python examples/basic/agent_chat.py
```

### 3. 体验高级功能

```bash
# 🆕 LangGraph功能展示 (推荐)
python examples/advanced/langgraph_showcase.py

# MCP系统演示
python examples/advanced/mcp_usage.py

# 批量处理示例
python examples/advanced/batch_processing.py
```

## 📋 示例说明

### 运行要求

- Python 3.8+
- 已安装项目依赖
- 系统已完成初始化

### 数据准备

部分示例需要测试数据：

```bash
# 下载示例文档
python scripts/download_sample_data.py

# 或使用自己的文档
cp your_document.pdf data/uploads/
```

### 配置调整

根据需要修改配置文件：

```python
# config/settings.py
MODEL_PATH = "your/model/path"
VECTOR_STORE_PATH = "your/vector/store"
```

## 🎯 学习路径

### 新手推荐

1. **document_parsing_demo.py** - 了解文档处理
2. **simple_rag.py** - 掌握 RAG 基础
3. **agent_chat.py** - 体验统一 Agent 功能

### 进阶用户

1. **mcp_usage.py** - 深入 MCP 系统
2. **batch_processing.py** - 优化处理流程
3. **custom_tools.py** - 扩展系统功能

### 开发者

1. 阅读源码注释
2. 修改示例参数
3. 开发自定义功能

## 🤝 贡献示例

欢迎贡献新的示例代码！

### 贡献指南

1. 在对应目录创建示例文件
2. 添加详细的代码注释
3. 更新本 README 文档
4. 提交 Pull Request

### 示例规范

- 代码清晰易懂
- 包含错误处理
- 添加使用说明
- 提供预期输出

## 📞 获取帮助

遇到问题？

- 查看 [文档中心](../docs/README.md)
- 提交 [GitHub Issue](https://github.com/your-org/epkbs/issues)
- 参与 [社区讨论](https://github.com/your-org/epkbs/discussions)

---

_让我们一起探索 EPKBS 的强大功能！_ 🚀
