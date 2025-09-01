#!/bin/bash
"""
环境激活脚本
确保使用正确的epkbs conda环境
"""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 EPKBS环境管理脚本${NC}"
echo "=================================="

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda未找到，请先安装Anaconda或Miniconda${NC}"
    exit 1
fi

# 初始化conda（如果需要）
if [ ! -f ~/.conda_initialized ]; then
    echo -e "${YELLOW}🔄 初始化conda...${NC}"
    conda init bash
    touch ~/.conda_initialized
    echo -e "${GREEN}✅ Conda初始化完成${NC}"
fi

# 激活conda
source /opt/anaconda3/etc/profile.d/conda.sh

# 检查epkbs环境是否存在
if conda env list | grep -q "epkbs"; then
    echo -e "${GREEN}✅ 找到epkbs环境${NC}"
    
    # 激活epkbs环境
    echo -e "${YELLOW}🔄 激活epkbs环境...${NC}"
    conda activate epkbs
    
    # 验证环境
    if [ "$CONDA_DEFAULT_ENV" = "epkbs" ]; then
        echo -e "${GREEN}✅ epkbs环境激活成功${NC}"
        echo "当前Python路径: $(which python)"
        echo "Python版本: $(python --version)"
        
        # 检查关键包
        echo -e "\n${BLUE}📦 检查关键依赖包...${NC}"
        
        key_packages=("langchain" "langgraph" "fastapi" "streamlit" "milvus" "unstructured")
        
        for package in "${key_packages[@]}"; do
            if python -c "import $package" 2>/dev/null; then
                version=$(python -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
                echo -e "${GREEN}✅ $package ($version)${NC}"
            else
                echo -e "${RED}❌ $package (未安装)${NC}"
            fi
        done
        
        echo -e "\n${GREEN}🎉 epkbs环境准备就绪！${NC}"
        
    else
        echo -e "${RED}❌ epkbs环境激活失败${NC}"
        exit 1
    fi
    
else
    echo -e "${RED}❌ epkbs环境不存在${NC}"
    echo -e "${YELLOW}💡 请先创建epkbs环境:${NC}"
    echo "   conda create -n epkbs python=3.10"
    echo "   conda activate epkbs"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export EPKBS_ENV="epkbs"
export EPKBS_ROOT="${PWD}"

echo -e "\n${BLUE}🌍 环境变量设置:${NC}"
echo "PYTHONPATH: $PYTHONPATH"
echo "EPKBS_ENV: $EPKBS_ENV"
echo "EPKBS_ROOT: $EPKBS_ROOT"

echo -e "\n${GREEN}🚀 环境准备完成，可以开始使用EPKBS！${NC}"
