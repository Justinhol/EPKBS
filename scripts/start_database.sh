#!/bin/bash

# 数据库服务启动脚本
# 用于启动PostgreSQL、Redis和Milvus服务

set -e

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
    echo "数据库服务启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start     启动所有数据库服务"
    echo "  stop      停止所有数据库服务"
    echo "  restart   重启所有数据库服务"
    echo "  status    查看服务状态"
    echo "  logs      查看服务日志"
    echo "  init      初始化数据库"
    echo "  help      显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start    # 启动数据库服务"
    echo "  $0 status   # 查看服务状态"
    echo "  $0 init     # 初始化数据库"
}

# 检查Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker服务未启动，请先启动Docker"
        exit 1
    fi
}

# 启动数据库服务
start_services() {
    log_info "启动数据库服务..."
    
    # 检查docker-compose.yml文件
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml文件不存在"
        exit 1
    fi
    
    # 启动数据库服务
    log_info "启动PostgreSQL..."
    docker-compose up -d postgres
    
    log_info "启动Redis..."
    docker-compose up -d redis
    
    log_info "启动Milvus依赖服务..."
    docker-compose up -d etcd minio
    
    # 等待依赖服务启动
    log_info "等待依赖服务启动..."
    sleep 10
    
    log_info "启动Milvus..."
    docker-compose up -d milvus
    
    # 等待所有服务启动
    log_info "等待所有服务启动完成..."
    sleep 30
    
    # 检查服务状态
    check_services_status
}

# 停止数据库服务
stop_services() {
    log_info "停止数据库服务..."
    
    docker-compose stop milvus
    docker-compose stop postgres
    docker-compose stop redis
    docker-compose stop etcd
    docker-compose stop minio
    
    log_success "数据库服务已停止"
}

# 重启数据库服务
restart_services() {
    log_info "重启数据库服务..."
    stop_services
    sleep 5
    start_services
}

# 检查服务状态
check_services_status() {
    log_info "检查服务状态..."
    
    echo ""
    echo "Docker容器状态:"
    docker-compose ps postgres redis milvus etcd minio
    
    echo ""
    echo "服务健康检查:"
    
    # 检查PostgreSQL
    if docker-compose exec -T postgres pg_isready -U epkbs_user &> /dev/null; then
        log_success "PostgreSQL: 运行正常"
    else
        log_error "PostgreSQL: 连接失败"
    fi
    
    # 检查Redis
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        log_success "Redis: 运行正常"
    else
        log_error "Redis: 连接失败"
    fi
    
    # 检查Milvus (需要等待更长时间)
    if curl -s http://localhost:19530/health &> /dev/null; then
        log_success "Milvus: 运行正常"
    else
        log_warning "Milvus: 可能还在启动中，请稍后再试"
    fi
    
    echo ""
    echo "端口监听状态:"
    echo "PostgreSQL (5432): $(netstat -an 2>/dev/null | grep :5432 | wc -l) 个连接"
    echo "Redis (6379): $(netstat -an 2>/dev/null | grep :6379 | wc -l) 个连接"
    echo "Milvus (19530): $(netstat -an 2>/dev/null | grep :19530 | wc -l) 个连接"
}

# 查看服务日志
show_logs() {
    echo "选择要查看的服务日志:"
    echo "1) PostgreSQL"
    echo "2) Redis"
    echo "3) Milvus"
    echo "4) 所有服务"
    echo "5) 退出"
    
    read -p "请选择 (1-5): " choice
    
    case $choice in
        1)
            docker-compose logs -f postgres
            ;;
        2)
            docker-compose logs -f redis
            ;;
        3)
            docker-compose logs -f milvus
            ;;
        4)
            docker-compose logs -f postgres redis milvus etcd minio
            ;;
        5)
            exit 0
            ;;
        *)
            log_error "无效选择"
            exit 1
            ;;
    esac
}

# 初始化数据库
init_database() {
    log_info "初始化数据库..."
    
    # 等待PostgreSQL启动
    log_info "等待PostgreSQL启动..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U epkbs_user &> /dev/null; then
            break
        fi
        sleep 2
    done
    
    # 执行初始化脚本
    if [ -f "scripts/init_db.sql" ]; then
        log_info "执行数据库初始化脚本..."
        docker-compose exec -T postgres psql -U epkbs_user -d epkbs -f /docker-entrypoint-initdb.d/init_db.sql
        log_success "数据库初始化完成"
    else
        log_warning "初始化脚本不存在，跳过数据库初始化"
    fi
}

# 主函数
main() {
    case "${1:-help}" in
        start)
            check_docker
            start_services
            ;;
        stop)
            check_docker
            stop_services
            ;;
        restart)
            check_docker
            restart_services
            ;;
        status)
            check_docker
            check_services_status
            ;;
        logs)
            check_docker
            show_logs
            ;;
        init)
            check_docker
            init_database
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
