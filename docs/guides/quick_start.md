# 🚀 快速开始指南

本指南将帮助您在5分钟内启动企业私有知识库系统。

## 📋 前置要求

- **Docker**: 20.10+ 和 Docker Compose
- **Python**: 3.10+ (如果选择本地部署)
- **内存**: 8GB+ (推荐 16GB)
- **存储**: 20GB+ 可用空间

## ⚡ 方式一：Docker一键部署 (推荐)

### 1. 克隆项目
```bash
git clone <repository-url>
cd EPKBS
```

### 2. 启动数据库服务
```bash
# 启动所有数据库服务
./scripts/start_database.sh start

# 检查服务状态
./scripts/start_database.sh status
```

### 3. 初始化数据库
```bash
# 初始化数据库表和数据
./scripts/start_database.sh init
```

### 4. 启动应用
```bash
# 配置环境变量
cp .env.example .env

# 启动应用服务
docker-compose up -d epkbs-app
```

### 5. 访问系统
- 🌐 **Web界面**: http://localhost:8501
- 📊 **API服务**: http://localhost:8000
- 📖 **API文档**: http://localhost:8000/docs

## 💻 方式二：本地开发部署

### 1. 环境准备
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动数据库
```bash
# 使用Docker启动数据库服务
./scripts/start_database.sh start
```

### 3. 配置环境
```bash
# 复制配置文件
cp .env.example .env

# 编辑配置 (可选)
nano .env
```

### 4. 启动应用
```bash
# 启动所有服务
python scripts/start_services.py

# 或分别启动
# 终端1: API服务
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 终端2: Web界面
streamlit run src.frontend.app:main --server.port 8501
```

## 🎯 方式三：演示模式 (快速体验)

如果您只想快速体验界面，无需数据库：

```bash
# 安装基础依赖
pip install streamlit pandas numpy pydantic python-dotenv loguru

# 启动演示模式
DEMO_MODE=true streamlit run src/frontend/app.py
```

## 🔧 常用命令

### 数据库管理
```bash
# 启动数据库服务
./scripts/start_database.sh start

# 停止数据库服务
./scripts/start_database.sh stop

# 重启数据库服务
./scripts/start_database.sh restart

# 查看服务状态
./scripts/start_database.sh status

# 查看服务日志
./scripts/start_database.sh logs

# 初始化数据库
./scripts/start_database.sh init
```

### 系统检查
```bash
# 检查系统状态
python scripts/check_system.py

# 检查服务状态
./scripts/deploy.sh -s
```

### 应用管理
```bash
# 查看所有容器状态
docker-compose ps

# 查看应用日志
docker-compose logs -f epkbs-app

# 重启应用
docker-compose restart epkbs-app
```

## 🎉 验证安装

### 1. 检查服务状态
```bash
# 检查所有服务
python scripts/check_system.py

# 或手动检查
curl http://localhost:8000/health
curl http://localhost:8501
```

### 2. 登录系统
- 访问: http://localhost:8501
- 默认管理员账户: `admin` / `admin123`
- 演示用户账户: `demo_user` / `user123`

### 3. 测试功能
1. **文档上传**: 上传一个PDF文档测试
2. **智能搜索**: 搜索相关内容
3. **智能对话**: 与AI助手对话

## ❗ 常见问题

### 端口占用
```bash
# 查看端口占用
lsof -i :8000  # API端口
lsof -i :8501  # Streamlit端口
lsof -i :5432  # PostgreSQL端口

# 杀死占用进程
kill -9 <PID>
```

### 数据库连接失败
```bash
# 检查数据库服务
docker-compose ps postgres

# 重启数据库
docker-compose restart postgres

# 查看数据库日志
docker-compose logs postgres
```

### 内存不足
```bash
# 限制Docker内存使用
docker-compose up -d --memory=4g

# 或修改 .env 文件
EMBEDDING_BATCH_SIZE=8
RERANKER_BATCH_SIZE=4
```

## 🔄 更新系统

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose down
docker-compose up -d
```

## 🛑 停止系统

```bash
# 停止所有服务
docker-compose down

# 停止并删除数据卷 (谨慎使用)
docker-compose down -v

# 仅停止数据库服务
./scripts/start_database.sh stop
```

## 📞 获取帮助

如果遇到问题：

1. 查看 [README.md](../README.md) 详细文档
2. 运行系统检查: `python scripts/check_system.py`
3. 查看日志: `docker-compose logs -f`
4. 提交Issue: [GitHub Issues](https://github.com/your-repo/EPKBS/issues)

---

**🎉 恭喜！您已成功启动企业私有知识库系统！**
