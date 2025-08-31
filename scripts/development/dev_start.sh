#!/bin/bash

# 开发环境快速启动脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "企业私有知识库系统 - 开发环境启动"
echo -e "========================================${NC}"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3未安装，请先安装Python3${NC}"
    exit 1
fi

# 检查并创建虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${BLUE}创建Python虚拟环境...${NC}"
    python3 -m venv venv
fi

# 激活虚拟环境
echo -e "${BLUE}激活虚拟环境...${NC}"
source venv/bin/activate

# 安装依赖
echo -e "${BLUE}安装/更新依赖...${NC}"
pip install -r requirements.txt

# 创建必要目录
echo -e "${BLUE}创建必要目录...${NC}"
mkdir -p data/uploads data/models data/vector_store logs

# 检查环境配置
if [ ! -f .env ]; then
    echo -e "${YELLOW}创建环境配置文件...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}请编辑.env文件配置您的环境变量${NC}"
fi

# 启动开发服务器
echo -e "${BLUE}启动开发服务器...${NC}"
echo -e "${GREEN}API服务器: http://localhost:8000${NC}"
echo -e "${GREEN}Web界面: http://localhost:8501${NC}"
echo -e "${GREEN}API文档: http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"
echo ""

# 启动服务
python scripts/start_services.py
