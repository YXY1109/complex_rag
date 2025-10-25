#!/bin/bash

# 复杂RAG服务部署脚本
# 使用方法: ./deploy.sh [环境] [选项]
# 环境: dev, staging, prod
# 选项: --build, --migrate, --seed, --clean

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
    cat << EOF
复杂RAG服务部署脚本

使用方法:
    $0 [环境] [选项]

环境:
    dev         开发环境
    staging     测试环境
    prod        生产环境

选项:
    --build     重新构建镜像
    --migrate   执行数据库迁移
    --seed      执行数据种子
    --clean     清理旧容器和镜像
    --logs      显示日志
    --stop      停止服务
    --restart   重启服务

示例:
    $0 dev --build --seed
    $0 prod --migrate
    $0 --clean
    $0 staging --logs
    $0 prod --restart

EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装"
        exit 1
    fi

    log_success "依赖检查通过"
}

# 设置环境变量
setup_environment() {
    local env=$1

    log_info "设置环境: $env"

    # 复制环境配置文件
    if [ -f ".env.$env" ]; then
        cp ".env.$env" .env
        log_success "已加载环境配置: .env.$env"
    elif [ -f ".env" ]; then
        log_warning "使用现有.env文件"
    else
        log_warning "创建默认.env文件"
        cat > .env << EOF
# RAG服务环境配置
COMPOSE_PROJECT_NAME=rag-$env
RAG_ENVIRONMENT=$env

# API密钥
OPENAI_API_KEY=your-openai-key

# 数据库配置
MYSQL_ROOT_PASSWORD=rootpass
MYSQL_PASSWORD=ragpass

# MinIO配置
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# 其他配置...
EOF
    fi

    # 导出环境变量
    export COMPOSE_PROJECT_NAME=rag-$env
    export RAG_ENVIRONMENT=$env

    log_success "环境设置完成"
}

# 清理旧容器和镜像
cleanup() {
    log_info "清理旧容器和镜像..."

    # 停止并删除容器
    docker-compose down --remove-orphans 2>/dev/null || true

    # 删除未使用的镜像
    docker image prune -f 2>/dev/null || true

    # 删除未使用的卷
    docker volume prune -f 2>/dev/null || true

    log_success "清理完成"
}

# 构建镜像
build_images() {
    log_info "构建Docker镜像..."

    if [ "$RAG_ENVIRONMENT" = "dev" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
    else
        docker-compose build
    fi

    log_success "镜像构建完成"
}

# 启动数据库服务
start_databases() {
    log_info "启动数据库服务..."

    docker-compose up -d mysql redis milvus elasticsearch minio-storage

    log_info "等待数据库服务启动..."
    sleep 30

    log_success "数据库服务启动完成"
}

# 执行数据库迁移
migrate_database() {
    log_info "执行数据库迁移..."

    # 这里应该执行实际的迁移命令
    # docker-compose exec rag-api uv run alembic upgrade head

    log_success "数据库迁移完成"
}

# 执行数据种子
seed_database() {
    log_info "执行数据种子..."

    # 这里应该执行实际的数据种子命令
    # docker-compose exec rag-api uv run python seed_data.py

    log_success "数据种子执行完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."

    if [ "$RAG_ENVIRONMENT" = "dev" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    else
        docker-compose up -d
    fi

    log_info "等待服务启动..."
    sleep 20

    log_success "服务启动完成"
}

# 检查服务状态
check_services() {
    log_info "检查服务状态..."

    # 检查API服务
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API服务正常"
    else
        log_warning "API服务可能未就绪"
    fi

    # 显示服务状态
    docker-compose ps
}

# 显示日志
show_logs() {
    log_info "显示服务日志..."

    if [ -n "$2" ]; then
        docker-compose logs -f "$2"
    else
        docker-compose logs -f
    fi
}

# 停止服务
stop_services() {
    log_info "停止服务..."

    docker-compose down

    log_success "服务已停止"
}

# 重启服务
restart_services() {
    log_info "重启服务..."

    stop_services
    start_services
    check_services

    log_success "服务重启完成"
}

# 主函数
main() {
    local env=${1:-dev}
    local build=false
    local migrate=false
    local seed=false
    local clean=false
    local logs=false
    local stop=false
    local restart=false

    # 解析参数
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                build=true
                shift
                ;;
            --migrate)
                migrate=true
                shift
                ;;
            --seed)
                seed=true
                shift
                ;;
            --clean)
                clean=true
                shift
                ;;
            --logs)
                logs=true
                shift
                ;;
            --stop)
                stop=true
                shift
                ;;
            --restart)
                restart=true
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

    # 检查依赖
    check_dependencies

    # 设置环境
    setup_environment "$env"

    # 执行操作
    if [ "$clean" = true ]; then
        cleanup
        exit 0
    fi

    if [ "$stop" = true ]; then
        stop_services
        exit 0
    fi

    if [ "$restart" = true ]; then
        restart_services
        exit 0
    fi

    if [ "$logs" = true ]; then
        show_logs "$@"
        exit 0
    fi

    # 部署流程
    if [ "$build" = true ]; then
        cleanup
        build_images
    fi

    start_databases

    if [ "$migrate" = true ]; then
        migrate_database
    fi

    if [ "$seed" = true ]; then
        seed_database
    fi

    start_services
    check_services

    log_success "部署完成！"
    log_info "API地址: http://localhost:8000"
    log_info "API文档: http://localhost:8000/docs"
    log_info "Grafana: http://localhost:3000 (admin/admin)"
    log_info "Prometheus: http://localhost:9090"
}

# 执行主函数
main "$@"