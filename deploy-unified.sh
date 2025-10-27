#!/bin/bash

# 统一RAG服务部署脚本
# 用于部署整合后的FastAPI服务

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

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
}

# 检查环境变量文件
check_env_file() {
    if [ ! -f ".env" ]; then
        log_warning ".env文件不存在，从模板复制..."
        if [ -f ".env.unified" ]; then
            cp .env.unified .env
            log_warning "请编辑 .env 文件配置您的环境变量"
            read -p "配置完成后按回车继续..." -r
        else
            log_error "未找到 .env.unified 模板文件"
            exit 1
        fi
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    mkdir -p logs
    mkdir -p data
    mkdir -p uploads
    mkdir -p temp
    mkdir -p config/nginx/conf.d
    mkdir -p ssl
    log_success "目录创建完成"
}

# 构建Docker镜像
build_images() {
    log_info "构建统一RAG服务Docker镜像..."
    docker compose -f docker-compose.unified.yml build
    log_success "Docker镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动统一RAG服务..."

    # 首先启动基础服务
    log_info "启动基础服务（MySQL, Redis, Milvus等）..."
    docker compose -f docker-compose.unified.yml up -d mysql redis etcd minio elasticsearch

    # 等待基础服务启动
    log_info "等待基础服务启动..."
    sleep 30

    # 启动Milvus
    log_info "启动Milvus向量数据库..."
    docker compose -f docker-compose.unified.yml up -d milvus
    sleep 20

    # 启动主应用
    log_info "启动统一RAG API服务..."
    docker compose -f docker-compose.unified.yml up -d unified-rag-api

    log_success "所有服务启动完成"
}

# 检查服务状态
check_services() {
    log_info "检查服务状态..."
    docker compose -f docker-compose.unified.yml ps

    # 检查API服务健康状态
    log_info "检查API服务健康状态..."
    for i in {1..10}; do
        if curl -f http://localhost:${API_PORT:-8000}/health/ping &> /dev/null; then
            log_success "API服务健康检查通过"
            break
        fi
        if [ $i -eq 10 ]; then
            log_error "API服务健康检查失败"
            exit 1
        fi
        log_info "等待API服务启动... ($i/10)"
        sleep 10
    done
}

# 显示服务信息
show_service_info() {
    echo ""
    echo "============================================="
    echo "🚀 统一RAG服务部署成功！"
    echo "============================================="
    echo ""
    echo "📍 服务地址:"
    echo "   主API服务: http://localhost:${API_PORT:-8000}"
    echo "   API文档: http://localhost:${API_PORT:-8000}/docs"
    echo "   健康检查: http://localhost:${API_PORT:-8000}/health/ping"
    echo ""
    echo "🔧 管理命令:"
    echo "   查看日志: docker compose -f docker-compose.unified.yml logs -f unified-rag-api"
    echo "   停止服务: docker compose -f docker-compose.unified.yml down"
    echo "   重启服务: docker compose -f docker-compose.unified.yml restart unified-rag-api"
    echo ""
    echo "📊 外部服务地址:"
    echo "   Kibana: http://localhost:${KIBANA_PORT:-5601}"
    echo "   MinIO: http://localhost:${MINIO_PORT:-9000}"
    echo ""
}

# 清理函数
cleanup() {
    log_info "清理部署..."
    docker compose -f docker-compose.unified.yml down -v
    docker system prune -f
    log_success "清理完成"
}

# 主函数
main() {
    case "${1:-deploy}" in
        "deploy")
            log_info "开始部署统一RAG服务..."
            check_docker
            check_env_file
            create_directories
            build_images
            start_services
            check_services
            show_service_info
            ;;
        "update")
            log_info "更新统一RAG服务..."
            build_images
            docker compose -f docker-compose.unified.yml up -d --force-recreate unified-rag-api
            check_services
            show_service_info
            ;;
        "stop")
            log_info "停止统一RAG服务..."
            docker compose -f docker-compose.unified.yml down
            log_success "服务已停止"
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            docker compose -f docker-compose.unified.yml logs -f unified-rag-api
            ;;
        "status")
            docker compose -f docker-compose.unified.yml ps
            ;;
        "health")
            curl -f http://localhost:${API_PORT:-8000}/health/detailed | jq '.' 2>/dev/null || curl http://localhost:${API_PORT:-8000}/health/detailed
            ;;
        *)
            echo "用法: $0 {deploy|update|stop|cleanup|logs|status|health}"
            echo ""
            echo "命令说明:"
            echo "  deploy  - 完整部署服务"
            echo "  update  - 更新服务（不重启基础服务）"
            echo "  stop    - 停止所有服务"
            echo "  cleanup - 清理所有容器和数据"
            echo "  logs    - 查看API服务日志"
            echo "  status  - 查看服务状态"
            echo "  health  - 检查服务健康状态"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"