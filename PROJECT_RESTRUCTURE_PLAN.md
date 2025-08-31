# 📁 项目结构重组方案

## 🎯 目标
- 清理根目录，只保留核心配置文件
- 统一文档管理
- 规范测试和示例结构
- 优化依赖管理

## 📋 重组前后对比

### 当前结构问题
```
EPKBS/
├── DOCUMENT_PARSING_README.md          # ❌ 文档散乱
├── DOCUMENT_PARSING_SUMMARY.md         # ❌ 文档散乱
├── VECTOR_STORAGE_STRATEGY.md          # ❌ 文档散乱
├── demo_document_parsing.py            # ❌ 示例文件在根目录
├── test_mcp_implementation.py          # ❌ 测试文件在根目录
├── requirements_document_parsing.txt   # ❌ 依赖文件重复
├── docs/                               # ✅ 但内容不全
├── examples/                           # ✅ 但内容不全
├── tests/                              # ✅ 但内容不全
└── ...
```

### 重组后结构
```
EPKBS/
├── README.md                           # 📖 主要项目说明
├── requirements.txt                    # 📦 统一依赖管理
├── pyproject.toml                      # 🔧 项目配置
├── docker-compose.yml                 # 🐳 容器编排
├── Dockerfile                          # 🐳 容器构建
├── main.py                             # 🚀 主入口
│
├── config/                             # ⚙️ 配置文件
│   ├── __init__.py
│   ├── settings.py                     # 主配置
│   ├── mcp_settings.py                 # MCP配置
│   └── nginx.conf                      # Nginx配置
│
├── src/                                # 💻 源代码
│   ├── __init__.py
│   ├── agent/                          # 🤖 Agent模块
│   ├── api/                            # 🌐 API模块
│   ├── data/                           # 📊 数据处理
│   ├── frontend/                       # 🎨 前端界面
│   ├── mcp/                            # 🔌 MCP实现
│   ├── rag/                            # 🔍 RAG核心
│   └── utils/                          # 🛠️ 工具函数
│
├── docs/                               # 📚 文档中心
│   ├── README.md                       # 文档索引
│   ├── architecture/                   # 架构文档
│   │   ├── system_overview.md
│   │   ├── mcp_implementation.md
│   │   └── rag_architecture.md
│   ├── guides/                         # 使用指南
│   │   ├── quick_start.md
│   │   ├── installation.md
│   │   ├── configuration.md
│   │   └── deployment.md
│   ├── api/                            # API文档
│   │   ├── rest_api.md
│   │   └── mcp_tools.md
│   └── development/                    # 开发文档
│       ├── contributing.md
│       ├── testing.md
│       └── extending.md
│
├── examples/                           # 📝 示例代码
│   ├── README.md                       # 示例索引
│   ├── basic/                          # 基础示例
│   │   ├── simple_rag.py
│   │   ├── document_parsing.py
│   │   └── agent_chat.py
│   ├── advanced/                       # 高级示例
│   │   ├── mcp_usage.py
│   │   ├── batch_processing.py
│   │   └── custom_tools.py
│   └── notebooks/                      # Jupyter示例
│       ├── rag_demo.ipynb
│       └── mcp_tutorial.ipynb
│
├── tests/                              # 🧪 测试代码
│   ├── __init__.py
│   ├── conftest.py                     # pytest配置
│   ├── unit/                           # 单元测试
│   │   ├── test_rag_core.py
│   │   ├── test_mcp_servers.py
│   │   └── test_agents.py
│   ├── integration/                    # 集成测试
│   │   ├── test_mcp_integration.py
│   │   ├── test_document_pipeline.py
│   │   └── test_api_endpoints.py
│   └── e2e/                            # 端到端测试
│       ├── test_full_workflow.py
│       └── test_deployment.py
│
├── scripts/                            # 🔧 脚本工具
│   ├── setup/                          # 安装脚本
│   │   ├── install_dependencies.sh
│   │   ├── init_database.sh
│   │   └── setup_environment.sh
│   ├── deployment/                     # 部署脚本
│   │   ├── deploy.sh
│   │   ├── start_services.py
│   │   └── health_check.py
│   └── development/                    # 开发脚本
│       ├── run_tests.sh
│       ├── format_code.sh
│       └── generate_docs.sh
│
├── data/                               # 📁 数据目录
│   ├── uploads/                        # 上传文件
│   ├── vector_store/                   # 向量存储
│   ├── models/                         # 模型文件
│   └── cache/                          # 缓存数据
│
└── logs/                               # 📋 日志文件
    ├── app.log
    ├── error.log
    └── system.log
```

## 🔄 重组步骤

### 第一步：整理文档
1. 将根目录的文档移动到 `docs/` 下合适的分类
2. 创建文档索引和导航
3. 统一文档格式和风格

### 第二步：重组示例和测试
1. 将根目录的示例文件移动到 `examples/` 下分类
2. 将根目录的测试文件移动到 `tests/` 下分类
3. 创建完整的测试套件

### 第三步：优化配置管理
1. 合并重复的依赖文件
2. 创建分层配置系统
3. 优化Docker配置

### 第四步：脚本工具整理
1. 按功能分类脚本
2. 创建统一的工具入口
3. 添加使用说明

## 📊 重组收益

### 🎯 清晰性
- 根目录简洁，只有核心文件
- 功能模块清晰分离
- 文档结构化管理

### 🔧 可维护性
- 统一的配置管理
- 规范的测试结构
- 清晰的开发流程

### 🚀 可扩展性
- 模块化的代码组织
- 标准化的项目结构
- 便于新功能添加

### 👥 协作友好
- 清晰的项目导航
- 完整的文档体系
- 标准的开发规范

## 🛠️ 实施计划

1. **Phase 1**: 文档重组 (30分钟)
2. **Phase 2**: 代码文件整理 (20分钟)  
3. **Phase 3**: 配置优化 (15分钟)
4. **Phase 4**: 测试验证 (15分钟)

总计约 1.5 小时完成项目结构重组。
