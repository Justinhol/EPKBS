#!/bin/bash

# 部署验证脚本
# 验证系统是否正确部署和运行

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "🔍 企业私有知识库系统 - 部署验证"
echo -e "========================================${NC}"

# 验证计数器
TOTAL_CHECKS=0
PASSED_CHECKS=0

# 检查函数
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "检查 $service_name ... "
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✅ 通过${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${RED}❌ 失败${NC}"
        return 1
    fi
}

check_port() {
    local service_name=$1
    local host=$2
    local port=$3
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "检查 $service_name 端口 $port ... "
    
    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✅ 开放${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${RED}❌ 关闭${NC}"
        return 1
    fi
}

check_docker_container() {
    local container_name=$1
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "检查容器 $container_name ... "
    
    if docker-compose ps "$container_name" | grep -q "Up"; then
        echo -e "${GREEN}✅ 运行中${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${RED}❌ 未运行${NC}"
        return 1
    fi
}

# 1. 检查Docker服务
echo -e "\n${BLUE}1. 检查Docker服务${NC}"
echo "----------------------------------------"

if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✅ Docker Compose 已安装${NC}"
    
    # 检查关键容器
    if [ -f "docker-compose.yml" ]; then
        check_docker_container "postgres"
        check_docker_container "redis" 
        check_docker_container "milvus"
        check_docker_container "epkbs-app"
    else
        echo -e "${YELLOW}⚠️ docker-compose.yml 不存在，跳过容器检查${NC}"
    fi
else
    echo -e "${RED}❌ Docker Compose 未安装${NC}"
fi

# 2. 检查网络端口
echo -e "\n${BLUE}2. 检查网络端口${NC}"
echo "----------------------------------------"

check_port "PostgreSQL" "localhost" 5432
check_port "Redis" "localhost" 6379
check_port "Milvus" "localhost" 19530
check_port "API服务" "localhost" 8000
check_port "Streamlit" "localhost" 8501

# 3. 检查HTTP服务
echo -e "\n${BLUE}3. 检查HTTP服务${NC}"
echo "----------------------------------------"

check_service "API健康检查" "http://localhost:8000/health"
check_service "API系统信息" "http://localhost:8000/info"
check_service "Streamlit应用" "http://localhost:8501"

# 4. 检查API端点
echo -e "\n${BLUE}4. 检查API端点${NC}"
echo "----------------------------------------"

check_service "API文档" "http://localhost:8000/docs"
check_service "ReDoc文档" "http://localhost:8000/redoc"
check_service "OpenAPI规范" "http://localhost:8000/openapi.json"

# 5. 检查文件系统
echo -e "\n${BLUE}5. 检查文件系统${NC}"
echo "----------------------------------------"

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ -f ".env" ]; then
    echo -e "环境配置文件: ${GREEN}✅ 存在${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "环境配置文件: ${RED}❌ 不存在${NC}"
fi

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ -d "data/uploads" ]; then
    echo -e "上传目录: ${GREEN}✅ 存在${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "上传目录: ${RED}❌ 不存在${NC}"
fi

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ -d "logs" ]; then
    echo -e "日志目录: ${GREEN}✅ 存在${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "日志目录: ${RED}❌ 不存在${NC}"
fi

# 6. 运行系统检查
echo -e "\n${BLUE}6. 运行系统检查${NC}"
echo "----------------------------------------"

if [ -f "scripts/check_system.py" ]; then
    echo "运行详细系统检查..."
    python scripts/check_system.py > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "系统检查: ${GREEN}✅ 通过${NC}"
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "系统检查: ${YELLOW}⚠️ 部分失败${NC}"
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    fi
else
    echo -e "系统检查脚本: ${RED}❌ 不存在${NC}"
fi

# 7. 测试基本功能
echo -e "\n${BLUE}7. 测试基本功能${NC}"
echo "----------------------------------------"

# 测试API响应
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if curl -s "http://localhost:8000/health" | grep -q "healthy\|status"; then
    echo -e "API响应测试: ${GREEN}✅ 通过${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "API响应测试: ${RED}❌ 失败${NC}"
fi

# 测试Streamlit响应
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if curl -s "http://localhost:8501" | grep -q "Streamlit\|企业私有知识库"; then
    echo -e "Streamlit响应测试: ${GREEN}✅ 通过${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "Streamlit响应测试: ${RED}❌ 失败${NC}"
fi

# 显示结果
echo -e "\n${BLUE}========================================"
echo "📊 验证结果汇总"
echo -e "========================================${NC}"

echo "总检查项: $TOTAL_CHECKS"
echo -e "通过检查: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "失败检查: ${RED}$((TOTAL_CHECKS - PASSED_CHECKS))${NC}"

SUCCESS_RATE=$(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))
echo "成功率: $SUCCESS_RATE%"

echo -e "\n${BLUE}访问地址:${NC}"
echo "🌐 Web界面: http://localhost:8501"
echo "📊 API服务: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"

if [ $SUCCESS_RATE -ge 80 ]; then
    echo -e "\n${GREEN}🎉 部署验证通过！系统运行正常。${NC}"
    exit 0
elif [ $SUCCESS_RATE -ge 60 ]; then
    echo -e "\n${YELLOW}⚠️ 部分功能可能不可用，但基本功能正常。${NC}"
    exit 0
else
    echo -e "\n${RED}❌ 部署验证失败，请检查系统配置。${NC}"
    echo -e "\n${YELLOW}建议操作:${NC}"
    echo "1. 检查Docker服务是否启动"
    echo "2. 运行: ./scripts/start_database.sh start"
    echo "3. 检查端口是否被占用"
    echo "4. 查看日志: docker-compose logs -f"
    exit 1
fi
