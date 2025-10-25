#!/bin/bash

# RAG服务备份脚本
# 使用方法: ./backup.sh [环境] [类型]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
RAG服务备份脚本

使用方法:
    $0 [环境] [备份类型]

环境:
    dev         开发环境
    staging     测试环境
    prod        生产环境

备份类型:
    all         备份所有数据 (默认)
    database    备份数据库
    vectors     备份向量数据
    files       备份文件存储
    config      备份配置文件

示例:
    $0 prod all
    $0 dev database
    $0 staging vectors

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

# 设置环境
setup_environment() {
    local env=$1

    log_info "设置环境: $env"

    export COMPOSE_PROJECT_NAME=rag-$env
    export RAG_ENVIRONMENT=$env

    # 创建备份目录
    BACKUP_DIR="./backups/$env"
    mkdir -p "$BACKUP_DIR"

    # 创建时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_PREFIX="${env}_${TIMESTAMP}"

    log_success "环境设置完成"
}

# 备份数据库
backup_database() {
    log_info "备份数据库..."

    local backup_file="$BACKUP_DIR/${BACKUP_PREFIX}_database.sql"

    # 备份MySQL
    docker-compose exec -T mysql mysqldump \
        -u raguser \
        -pragpass \
        --single-transaction \
        --routines \
        --triggers \
        ragdb > "$backup_file"

    # 压缩备份文件
    gzip "$backup_file"

    log_success "数据库备份完成: ${backup_file}.gz"
}

# 备份向量数据
backup_vectors() {
    log_info "备份向量数据..."

    local backup_dir="$BACKUP_DIR/${BACKUP_PREFIX}_vectors"
    mkdir -p "$backup_dir"

    # 备份Milvus数据
    docker cp $(docker-compose ps -q milvus):/var/lib/milvus "$backup_dir/milvus"

    # 压缩备份
    tar -czf "${backup_dir}.tar.gz" -C "$BACKUP_DIR" "$(basename $backup_dir)"
    rm -rf "$backup_dir"

    log_success "向量数据备份完成: ${backup_dir}.tar.gz"
}

# 备份文件存储
backup_files() {
    log_info "备份文件存储..."

    local backup_dir="$BACKUP_DIR/${BACKUP_PREFIX}_files"
    mkdir -p "$backup_dir"

    # 备份MinIO数据
    docker cp $(docker-compose ps -q minio-storage):/data "$backup_dir/minio"

    # 备份应用数据
    if [ -d "./data" ]; then
        cp -r ./data "$backup_dir/app_data"
    fi

    # 压缩备份
    tar -czf "${backup_dir}.tar.gz" -C "$BACKUP_DIR" "$(basename $backup_dir)"
    rm -rf "$backup_dir"

    log_success "文件存储备份完成: ${backup_dir}.tar.gz"
}

# 备份配置文件
backup_config() {
    log_info "备份配置文件..."

    local backup_dir="$BACKUP_DIR/${BACKUP_PREFIX}_config"
    mkdir -p "$backup_dir"

    # 备份Docker配置
    cp docker-compose*.yml "$backup_dir/" 2>/dev/null || true
    cp Dockerfile* "$backup_dir/" 2>/dev/null || true

    # 备份应用配置
    if [ -f ".env" ]; then
        cp .env "$backup_dir/"
    fi

    if [ -f "config.json" ]; then
        cp config.json "$backup_dir/"
    fi

    # 备份部署配置
    cp -r deployment "$backup_dir/"

    # 压缩备份
    tar -czf "${backup_dir}.tar.gz" -C "$BACKUP_DIR" "$(basename $backup_dir)"
    rm -rf "$backup_dir"

    log_success "配置文件备份完成: ${backup_dir}.tar.gz"
}

# 清理旧备份
cleanup_old_backups() {
    log_info "清理旧备份..."

    local keep_days=${BACKUP_RETENTION_DAYS:-7}

    # 删除超过保留天数的备份
    find "$BACKUP_DIR" -name "*.gz" -mtime +$keep_days -delete 2>/dev/null || true
    find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$keep_days -delete 2>/dev/null || true

    log_success "旧备份清理完成"
}

# 生成备份报告
generate_backup_report() {
    local backup_type=$1
    local backup_file=$2

    log_info "生成备份报告..."

    local report_file="$BACKUP_DIR/${BACKUP_PREFIX}_report.txt"

    cat > "$report_file" << EOF
RAG服务备份报告
================

备份环境: $RAG_ENVIRONMENT
备份类型: $backup_type
备份时间: $(date)
备份文件: $backup_file

备份内容:
EOF

    case $backup_type in
        "database")
            echo "- MySQL数据库" >> "$report_file"
            ;;
        "vectors")
            echo "- Milvus向量数据" >> "$report_file"
            ;;
        "files")
            echo "- MinIO对象存储" >> "$report_file"
            echo "- 应用文件数据" >> "$report_file"
            ;;
        "config")
            echo "- Docker配置文件" >> "$report_file"
            echo "- 应用配置文件" >> "$report_file"
            echo "- 部署配置文件" >> "$report_file"
            ;;
        "all")
            echo "- MySQL数据库" >> "$report_file"
            echo "- Milvus向量数据" >> "$report_file"
            echo "- MinIO对象存储" >> "$report_file"
            echo "- 应用文件数据" >> "$report_file"
            echo "- Docker配置文件" >> "$report_file"
            echo "- 应用配置文件" >> "$report_file"
            echo "- 部署配置文件" >> "$report_file"
            ;;
    esac

    cat >> "$report_file" << EOF

备份统计:
- 备份目录大小: $(du -sh "$BACKUP_DIR" | cut -f1)
- 备份文件数量: $(find "$BACKUP_DIR" -name "${BACKUP_PREFIX}*" | wc -l)

注意事项:
1. 请定期验证备份文件的完整性
2. 建议将备份文件存储到不同的物理位置
3. 重要数据建议使用云存储服务进行异地备份
4. 定期测试恢复流程确保备份可用

EOF

    log_success "备份报告生成完成: $report_file"
}

# 主函数
main() {
    local env=${1:-dev}
    local backup_type=${2:-all}

    # 解析参数
    case $backup_type in
        -h|--help)
            show_help
            exit 0
            ;;
        all|database|vectors|files|config)
            ;;
        *)
            log_error "无效的备份类型: $backup_type"
            show_help
            exit 1
            ;;
    esac

    log_info "开始备份 - 环境: $env, 类型: $backup_type"

    # 检查依赖
    check_dependencies

    # 设置环境
    setup_environment "$env"

    # 执行备份
    case $backup_type in
        "all")
            backup_database
            backup_vectors
            backup_files
            backup_config
            ;;
        "database")
            backup_database
            ;;
        "vectors")
            backup_vectors
            ;;
        "files")
            backup_files
            ;;
        "config")
            backup_config
            ;;
    esac

    # 清理旧备份
    cleanup_old_backups

    # 生成备份报告
    generate_backup_report "$backup_type" "$BACKUP_DIR"

    log_success "备份完成！"
    log_info "备份目录: $BACKUP_DIR"
    log_info "备份文件: ${BACKUP_PREFIX}_*"
}

# 执行主函数
main "$@"