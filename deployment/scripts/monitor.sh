#!/bin/bash

# RAG服务监控脚本
# 使用方法: ./monitor.sh [环境] [检查类型]

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
RAG服务监控脚本

使用方法:
    $0 [环境] [检查类型]

环境:
    dev         开发环境
    staging     测试环境
    prod        生产环境

检查类型:
    all         检查所有服务 (默认)
    health      健康检查
    resources   资源使用情况
    logs        日志分析
    performance 性能指标
    security    安全检查

示例:
    $0 prod all
    $0 dev health
    $0 staging resources

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

    if ! command -v curl &> /dev/null; then
        log_error "curl未安装"
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

    log_success "环境设置完成"
}

# 检查服务健康状态
check_health() {
    log_info "检查服务健康状态..."

    local unhealthy_services=()

    # 检查Docker容器状态
    log_info "检查Docker容器状态..."
    docker-compose ps

    # 检查API服务
    log_info "检查API服务..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API服务正常"
    else
        log_error "API服务异常"
        unhealthy_services+=("rag-api")
    fi

    # 检查数据库连接
    log_info "检查数据库连接..."
    if docker-compose exec -T mysql mysql -u raguser -pragpass -e "SELECT 1;" > /dev/null 2>&1; then
        log_success "MySQL数据库正常"
    else
        log_error "MySQL数据库异常"
        unhealthy_services+=("mysql")
    fi

    # 检查Redis连接
    log_info "检查Redis连接..."
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis正常"
    else
        log_error "Redis异常"
        unhealthy_services+=("redis")
    fi

    # 检查Elasticsearch
    log_info "检查Elasticsearch..."
    if curl -f http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        log_success "Elasticsearch正常"
    else
        log_error "Elasticsearch异常"
        unhealthy_services+=("elasticsearch")
    fi

    # 检查Milvus
    log_info "检查Milvus..."
    if curl -f http://localhost:9091/healthz > /dev/null 2>&1; then
        log_success "Milvus正常"
    else
        log_error "Milvus异常"
        unhealthy_services+=("milvus")
    fi

    # 汇总结果
    if [ ${#unhealthy_services[@]} -eq 0 ]; then
        log_success "所有服务运行正常"
        return 0
    else
        log_error "异常服务: ${unhealthy_services[*]}"
        return 1
    fi
}

# 检查资源使用情况
check_resources() {
    log_info "检查资源使用情况..."

    # Docker容器资源使用
    log_info "Docker容器资源使用:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

    # 系统资源使用
    log_info "系统资源使用:"
    echo "CPU使用率: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)"
    echo "内存使用: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    echo "磁盘使用: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"

    # Docker磁盘使用
    log_info "Docker磁盘使用:"
    docker system df

    # 检查资源瓶颈
    log_info "检查资源瓶颈..."

    # CPU检查
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        log_warning "CPU使用率过高: ${cpu_usage}%"
    fi

    # 内存检查
    mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$mem_usage" -gt 80 ]; then
        log_warning "内存使用率过高: ${mem_usage}%"
    fi

    # 磁盘检查
    disk_usage=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    if [ "$disk_usage" -gt 80 ]; then
        log_warning "磁盘使用率过高: ${disk_usage}%"
    fi
}

# 检查日志
check_logs() {
    log_info "检查服务日志..."

    local error_count=0
    local warning_count=0

    # 检查API服务错误日志
    log_info "检查API服务错误日志..."
    api_errors=$(docker-compose logs --tail=100 rag-api 2>&1 | grep -i "error\|exception\|failed" | wc -l)
    if [ "$api_errors" -gt 0 ]; then
        log_warning "API服务发现 $api_errors 个错误"
        error_count=$((error_count + api_errors))
        docker-compose logs --tail=10 rag-api | grep -i "error\|exception\|failed"
    fi

    # 检查数据库错误日志
    log_info "检查数据库错误日志..."
    db_errors=$(docker-compose logs --tail=100 mysql 2>&1 | grep -i "error\|failed" | wc -l)
    if [ "$db_errors" -gt 0 ]; then
        log_warning "数据库发现 $db_errors 个错误"
        error_count=$((error_count + db_errors))
    fi

    # 检查最近的警告
    log_info "检查最近的警告..."
    warnings=$(docker-compose logs --tail=50 2>&1 | grep -i "warning" | wc -l)
    if [ "$warnings" -gt 0 ]; then
        log_warning "发现 $warnings 个警告"
        warning_count=$warnings
    fi

    # 汇总结果
    if [ "$error_count" -eq 0 ] && [ "$warning_count" -eq 0 ]; then
        log_success "日志检查正常"
    else
        log_warning "发现问题: $error_count 个错误, $warning_count 个警告"
    fi
}

# 检查性能指标
check_performance() {
    log_info "检查性能指标..."

    # API响应时间
    log_info "检查API响应时间..."
    if command -v curl &> /dev/null; then
        response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/health)
        if (( $(echo "$response_time > 2.0" | bc -l) )); then
            log_warning "API响应时间较慢: ${response_time}s"
        else
            log_success "API响应时间正常: ${response_time}s"
        fi
    fi

    # 数据库查询性能
    log_info "检查数据库查询性能..."
    db_query_time=$(docker-compose exec -T mysql mysql -u raguser -pragpass -e "SELECT SLEEP(0.1);" 2>/dev/null | wc -l)
    if [ "$db_query_time" -gt 0 ]; then
        log_success "数据库查询响应正常"
    else
        log_warning "数据库查询可能存在问题"
    fi

    # 检查慢查询
    log_info "检查慢查询..."
    if docker-compose exec -T mysql mysql -u raguser -pragpass ragdb -e "SHOW VARIABLES LIKE 'slow_query_log';" | grep -q "ON"; then
        slow_queries=$(docker-compose exec -T mysql mysql -u raguser -pragpass ragdb -e "SHOW STATUS LIKE 'Slow_queries';" | tail -1 | awk '{print $2}')
        if [ "$slow_queries" -gt 0 ]; then
            log_warning "发现 $slow_queries 个慢查询"
        else
            log_success "没有慢查询"
        fi
    fi
}

# 检查安全配置
check_security() {
    log_info "检查安全配置..."

    # 检查默认密码
    log_info "检查默认密码..."
    if grep -q "your-openai-key" .env 2>/dev/null; then
        log_error "发现默认API密钥，请更换"
    fi

    if grep -q "minioadmin" .env 2>/dev/null; then
        log_warning "发现默认MinIO密码，建议更换"
    fi

    # 检查端口暴露
    log_info "检查端口暴露..."
    exposed_ports=$(docker-compose ps | grep "0.0.0.0" | wc -l)
    if [ "$exposed_ports" -gt 0 ]; then
        log_warning "发现 $exposed_ports 个端口暴露到所有接口"
    fi

    # 检查SSL/TLS配置
    log_info "检查SSL/TLS配置..."
    if ! curl -k https://localhost:8000/health > /dev/null 2>&1; then
        log_warning "HTTPS未配置或证书有问题"
    fi

    # 检查防火墙状态
    if command -v ufw &> /dev/null; then
        firewall_status=$(ufw status | grep "Status:" | awk '{print $2}')
        log_info "防火墙状态: $firewall_status"
        if [ "$firewall_status" = "inactive" ]; then
            log_warning "防火墙未启用"
        fi
    fi
}

# 生成监控报告
generate_monitoring_report() {
    local check_type=$1
    local exit_code=$2

    log_info "生成监控报告..."

    local report_file="./monitoring_reports/${RAG_ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p "$(dirname "$report_file")"

    cat > "$report_file" << EOF
RAG服务监控报告
===============

检查环境: $RAG_ENVIRONMENT
检查类型: $check_type
检查时间: $(date)
检查结果: $([ $exit_code -eq 0 ] && echo "正常" || echo "发现问题")

服务状态:
EOF

    docker-compose ps >> "$report_file"

    cat >> "$report_file" << EOF

资源使用:
- CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
- 内存: $(free -h | grep Mem | awk '{print $3 "/" $2}')
- 磁盘: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')

最近日志:
$(docker-compose logs --tail=20 2>&1 | tail -20)

建议:
1. 定期执行监控检查
2. 设置告警通知
3. 备份重要数据
4. 更新安全配置
5. 优化性能瓶颈

EOF

    log_success "监控报告生成完成: $report_file"
}

# 主函数
main() {
    local env=${1:-dev}
    local check_type=${2:-all}

    # 解析参数
    case $check_type in
        -h|--help)
            show_help
            exit 0
            ;;
        all|health|resources|logs|performance|security)
            ;;
        *)
            log_error "无效的检查类型: $check_type"
            show_help
            exit 1
            ;;
    esac

    log_info "开始监控 - 环境: $env, 类型: $check_type"

    # 检查依赖
    check_dependencies

    # 设置环境
    setup_environment "$env"

    local exit_code=0

    # 执行检查
    case $check_type in
        "all")
            check_health || exit_code=1
            check_resources
            check_logs
            check_performance
            check_security
            ;;
        "health")
            check_health || exit_code=1
            ;;
        "resources")
            check_resources
            ;;
        "logs")
            check_logs
            ;;
        "performance")
            check_performance
            ;;
        "security")
            check_security
            ;;
    esac

    # 生成监控报告
    generate_monitoring_report "$check_type" "$exit_code"

    if [ $exit_code -eq 0 ]; then
        log_success "监控检查完成 - 未发现问题"
    else
        log_error "监控检查完成 - 发现问题，请查看报告"
        exit 1
    fi
}

# 执行主函数
main "$@"