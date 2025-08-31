# 📁 EPKBS 项目结构总览

## 🎯 项目重组完成

项目已完成结构重组，现在拥有清晰、规范的目录结构和文件组织。

## 📂 新的项目结构

```
EPKBS/
├── 📖 README.md                        # 项目主要说明
├── 📦 requirements.txt                 # 统一依赖管理
├── ⚙️ pyproject.toml                   # 项目配置
├── 🐳 docker-compose.yml              # 容器编排
├── 🐳 Dockerfile                       # 容器构建
├── 🚀 main.py                          # 主入口文件
│
├── ⚙️ config/                          # 配置文件目录
│   ├── __init__.py
│   ├── settings.py                     # 主配置文件
│   ├── mcp_settings.py                 # MCP专用配置
│   └── nginx.conf                      # Nginx配置
│
├── 💻 src/                             # 源代码目录
│   ├── __init__.py
│   ├── 🤖 agent/                       # Agent模块
│   │   ├── core.py                     # Agent核心管理器
│   │   ├── llm.py                      # LLM管理器
│   │   ├── tools.py                    # 传统工具系统
│   │   ├── react_agent.py              # ReAct Agent
│   │   └── mcp_agent.py                # MCP Agent
│   │
│   ├── 🔌 mcp/                         # MCP实现模块
│   │   ├── types/                      # MCP类型定义
│   │   ├── servers/                    # MCP服务器集群
│   │   │   ├── rag_server.py           # RAG服务器
│   │   │   ├── document_parsing_server.py  # 文档解析服务器
│   │   │   └── table_extraction_server.py  # 表格提取服务器
│   │   ├── clients/                    # MCP客户端
│   │   │   └── mcp_client.py           # MCP客户端管理器
│   │   └── server_manager.py           # 服务器管理器
│   │
│   ├── 🔍 rag/                         # RAG核心模块
│   ├── 🌐 api/                         # API接口模块
│   ├── 📊 data/                        # 数据处理模块
│   ├── 🎨 frontend/                    # 前端界面模块
│   └── 🛠️ utils/                       # 工具函数模块
│
├── 📚 docs/                            # 文档中心
│   ├── README.md                       # 文档索引
│   ├── 🏗️ architecture/                # 架构文档
│   │   ├── mcp_implementation.md       # MCP实现详解
│   │   ├── document_processing.md      # 文档处理流程
│   │   └── vector_storage.md           # 向量存储策略
│   ├── 📋 guides/                      # 使用指南
│   │   ├── quick_start.md              # 快速开始
│   │   ├── configuration.md            # 配置管理
│   │   └── document_parsing.md         # 文档解析指南
│   ├── 🌐 api/                         # API文档
│   └── 👨‍💻 development/                 # 开发文档
│
├── 📝 examples/                        # 示例代码
│   ├── README.md                       # 示例索引
│   ├── 🔰 basic/                       # 基础示例
│   │   └── document_parsing_demo.py    # 文档解析演示
│   ├── 🚀 advanced/                    # 高级示例
│   │   └── mcp_usage.py                # MCP使用示例
│   └── 📓 notebooks/                   # Jupyter示例
│
├── 🧪 tests/                           # 测试代码
│   ├── conftest.py                     # pytest配置
│   ├── 🔬 unit/                        # 单元测试
│   │   ├── test_document_parsing.py    # 文档解析测试
│   │   └── test_data_pipeline.py       # 数据管道测试
│   ├── 🔗 integration/                 # 集成测试
│   │   └── test_mcp_implementation.py  # MCP集成测试
│   └── 🎯 e2e/                         # 端到端测试
│
├── 🔧 scripts/                         # 脚本工具
│   ├── 🛠️ setup/                       # 安装脚本
│   │   ├── init_db.sql                 # 数据库初始化
│   │   └── start_database.sh           # 数据库启动
│   ├── 🚀 deployment/                  # 部署脚本
│   │   ├── deploy.sh                   # 部署脚本
│   │   ├── start_services.py           # 服务启动
│   │   └── verify_deployment.sh        # 部署验证
│   └── 👨‍💻 development/                 # 开发脚本
│       ├── dev_start.sh                # 开发环境启动
│       └── test_system.py              # 系统测试
│
├── 📁 data/                            # 数据目录
│   ├── uploads/                        # 上传文件
│   ├── vector_store/                   # 向量存储
│   ├── models/                         # 模型文件
│   └── embeddings_cache/               # 嵌入缓存
│
└── 📋 logs/                            # 日志文件
    ├── app.log                         # 应用日志
    ├── error.log                       # 错误日志
    └── rag.log                         # RAG日志
```

## 🔄 重组改进

### ✅ 已完成的改进

1. **📚 文档整理**
   - 将散落的文档移动到 `docs/` 下分类管理
   - 创建了完整的文档索引和导航
   - 按功能分类：架构、指南、API、开发

2. **📝 示例重组**
   - 将根目录的示例文件移动到 `examples/` 下
   - 按难度分类：basic、advanced、notebooks
   - 创建了示例索引和使用说明

3. **🧪 测试规范**
   - 按测试类型分类：unit、integration、e2e
   - 移动现有测试文件到对应目录
   - 为MCP功能创建专门的集成测试

4. **🔧 脚本整理**
   - 按功能分类：setup、deployment、development
   - 重组现有脚本文件
   - 便于维护和使用

5. **⚙️ 配置优化**
   - 合并重复的依赖文件
   - 创建MCP专用配置文件
   - 统一配置管理

### 🎯 重组收益

#### 🔍 清晰性提升
- 根目录简洁，只保留核心文件
- 功能模块清晰分离
- 文档结构化，易于查找

#### 🛠️ 维护性改善
- 统一的依赖管理
- 规范的测试结构
- 清晰的开发流程

#### 🚀 扩展性增强
- 模块化的代码组织
- 标准化的项目结构
- 便于新功能添加

#### 👥 协作友好
- 清晰的项目导航
- 完整的文档体系
- 标准的开发规范

## 📋 使用指南

### 🚀 快速开始
```bash
# 1. 查看项目文档
cat docs/README.md

# 2. 运行基础示例
python examples/basic/document_parsing_demo.py

# 3. 体验MCP功能
python examples/advanced/mcp_usage.py
```

### 🧪 运行测试
```bash
# 单元测试
python -m pytest tests/unit/

# 集成测试
python -m pytest tests/integration/

# MCP功能测试
python tests/integration/test_mcp_implementation.py
```

### 📚 查看文档
```bash
# 架构文档
ls docs/architecture/

# 使用指南
ls docs/guides/

# 开发文档
ls docs/development/
```

## 🎉 项目现状

✅ **结构清晰** - 文件组织规范，易于导航  
✅ **文档完整** - 覆盖架构、使用、开发各方面  
✅ **示例丰富** - 从基础到高级的完整示例  
✅ **测试规范** - 分层测试，覆盖全面  
✅ **配置统一** - 集中管理，便于维护  

现在的EPKBS项目拥有了企业级的项目结构和管理规范！🚀
