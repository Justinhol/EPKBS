#!/bin/bash

# 企业私有知识库系统部署脚本
# 支持开发环境和生产环境部署

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "企业私有知识库系统部署脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -e, --env ENV        部署环境 (dev|prod) [默认: dev]"
    echo "  -m, --mode MODE      部署模式 (docker|local) [默认: docker]"
    echo "  -p, --pull           拉取最新镜像"
    echo "  -b, --build          重新构建镜像"
    echo "  -d, --down           停止并删除容器"
    echo "  -l, --logs           查看日志"
    echo "  -s, --status         查看服务状态"
    echo "  -h, --help           显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -e dev -m docker    # 开发环境Docker部署"
    echo "  $0 -e prod -m local    # 生产环境本地部署"
    echo "  $0 -d                  # 停止所有服务"
    echo "  $0 -l                  # 查看日志"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装，请先安装Python3"
        exit 1
    fi
    
    log_success "系统依赖检查通过"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    mkdir -p data/uploads
    mkdir -p data/models
    mkdir -p data/vector_store
    mkdir -p logs
    mkdir -p config/ssl
    
    log_success "目录创建完成"
}

# 检查环境配置
check_env_config() {
    log_info "检查环境配置..."
    
    if [ ! -f .env ]; then
        log_warning ".env文件不存在，从模板创建..."
        cp .env.example .env
        log_warning "请编辑.env文件配置您的环境变量"
    fi
    
    log_success "环境配置检查完成"
}

# Docker部署
deploy_docker() {
    local env=$1
    local pull_images=$2
    local build_images=$3
    
    log_info "开始Docker部署 (环境: $env)..."
    
    # 设置Docker Compose文件
    local compose_file="docker-compose.yml"
    if [ "$env" = "prod" ]; then
        compose_file="docker-compose.prod.yml"
        if [ ! -f "$compose_file" ]; then
            compose_file="docker-compose.yml"
            log_warning "生产环境配置文件不存在，使用默认配置"
        fi
    fi
    
    # 拉取镜像
    if [ "$pull_images" = true ]; then
        log_info "拉取最新镜像..."
        docker-compose -f $compose_file pull
    fi
    
    # 构建镜像
    if [ "$build_images" = true ]; then
        log_info "构建应用镜像..."
        docker-compose -f $compose_file build --no-cache
    fi
    
    # 启动服务
    log_info "启动服务..."
    docker-compose -f $compose_file up -d
    
    # 等待服务启动
    log_info "等待服务启动..."
    sleep 30
    
    # 检查服务状态
    check_services_docker $compose_file
    
    log_success "Docker部署完成"
}

# 本地部署
deploy_local() {
    local env=$1
    
    log_info "开始本地部署 (环境: $env)..."
    
    # 检查Python虚拟环境
    if [ ! -d "venv" ]; then
        log_info "创建Python虚拟环境..."
        python3 -m venv venv
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 安装依赖
    log_info "安装Python依赖..."
    pip install -r requirements.txt
    
    # 启动服务
    log_info "启动服务..."
    if [ "$env" = "prod" ]; then
        python scripts/start_services.py --prod
    else
        python scripts/start_services.py --dev
    fi
    
    log_success "本地部署完成"
}

# 检查Docker服务状态
check_services_docker() {
    local compose_file=$1
    
    log_info "检查服务状态..."
    
    # 检查容器状态
    docker-compose -f $compose_file ps
    
    # 检查健康状态
    local services=("postgres" "redis" "milvus" "epkbs-app")
    
    for service in "${services[@]}"; do
        local status=$(docker-compose -f $compose_file ps -q $service | xargs docker inspect --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
        
        if [ "$status" = "healthy" ]; then
            log_success "$service: 健康"
        elif [ "$status" = "starting" ]; then
            log_warning "$service: 启动中"
        else
            log_error "$service: 不健康 ($status)"
        fi
    done
}

# 停止服务
stop_services() {
    log_info "停止服务..."
    
    if [ -f docker-compose.yml ]; then
        docker-compose down
        log_success "Docker服务已停止"
    fi
    
    # 停止本地服务
    pkill -f "uvicorn" || true
    pkill -f "streamlit" || true
    log_success "本地服务已停止"
}

# 查看日志
show_logs() {
    log_info "显示服务日志..."
    
    if [ -f docker-compose.yml ]; then
        docker-compose logs -f --tail=100
    else
        log_warning "Docker Compose文件不存在"
        if [ -f logs/epkbs.log ]; then
            tail -f logs/epkbs.log
        else
            log_error "日志文件不存在"
        fi
    fi
}

# 显示服务状态
show_status() {
    log_info "服务状态:"
    
    # Docker服务状态
    if command -v docker-compose &> /dev/null && [ -f docker-compose.yml ]; then
        echo ""
        echo "Docker服务:"
        docker-compose ps
    fi
    
    # 本地服务状态
    echo ""
    echo "本地服务:"
    
    # 检查API服务
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "API服务: 运行中 (http://localhost:8000)"
    else
        log_error "API服务: 未运行"
    fi
    
    # 检查Streamlit服务
    if curl -s http://localhost:8501 > /dev/null; then
        log_success "Web界面: 运行中 (http://localhost:8501)"
    else
        log_error "Web界面: 未运行"
    fi
    
    # 检查数据库连接
    echo ""
    echo "数据库连接:"
    python3 -c "
import asyncio
import sys
sys.path.append('.')
from src.api.database import check_database_health

async def check():
    health = await check_database_health()
    for service, status in health.items():
        if 'healthy' in status:
            print(f'✅ {service}: {status}')
        else:
            print(f'❌ {service}: {status}')

asyncio.run(check())
" 2>/dev/null || log_warning "无法检查数据库状态"
}

# 主函数
main() {
    local env="dev"
    local mode="docker"
    local pull_images=false
    local build_images=false
    local stop_services_flag=false
    local show_logs_flag=false
    local show_status_flag=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                env="$2"
                shift 2
                ;;
            -m|--mode)
                mode="$2"
                shift 2
                ;;
            -p|--pull)
                pull_images=true
                shift
                ;;
            -b|--build)
                build_images=true
                shift
                ;;
            -d|--down)
                stop_services_flag=true
                shift
                ;;
            -l|--logs)
                show_logs_flag=true
                shift
                ;;
            -s|--status)
                show_status_flag=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证参数
    if [[ "$env" != "dev" && "$env" != "prod" ]]; then
        log_error "无效的环境: $env (支持: dev, prod)"
        exit 1
    fi
    
    if [[ "$mode" != "docker" && "$mode" != "local" ]]; then
        log_error "无效的模式: $mode (支持: docker, local)"
        exit 1
    fi
    
    # 执行操作
    if [ "$stop_services_flag" = true ]; then
        stop_services
        exit 0
    fi
    
    if [ "$show_logs_flag" = true ]; then
        show_logs
        exit 0
    fi
    
    if [ "$show_status_flag" = true ]; then
        show_status
        exit 0
    fi
    
    # 部署流程
    echo "========================================"
    echo "企业私有知识库系统部署"
    echo "========================================"
    echo "环境: $env"
    echo "模式: $mode"
    echo "========================================"
    
    check_dependencies
    create_directories
    check_env_config
    
    if [ "$mode" = "docker" ]; then
        deploy_docker $env $pull_images $build_images
    else
        deploy_local $env
    fi
    
    echo ""
    echo "========================================"
    log_success "部署完成！"
    echo "========================================"
    echo "🌐 Web界面: http://localhost:8501"
    echo "📊 API服务: http://localhost:8000"
    echo "📖 API文档: http://localhost:8000/docs"
    echo "========================================"
    echo ""
    echo "常用命令:"
    echo "  查看状态: $0 -s"
    echo "  查看日志: $0 -l"
    echo "  停止服务: $0 -d"
    echo "========================================"
}

# 执行主函数
main "$@"
