@echo off
REM 统一RAG服务部署脚本 (Windows版本)
REM 用于部署整合后的FastAPI服务

setlocal enabledelayedexpansion

echo [INFO] 开始部署统一RAG服务...

REM 检查Docker是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Docker Compose未安装，请先安装Docker Compose
        pause
        exit /b 1
    )
)

REM 检查环境变量文件
if not exist ".env" (
    echo [WARNING] .env文件不存在，从模板复制...
    if exist ".env.unified" (
        copy .env.unified .env >nul
        echo [WARNING] 请编辑 .env 文件配置您的环境变量
        pause
    ) else (
        echo [ERROR] 未找到 .env.unified 模板文件
        pause
        exit /b 1
    )
)

REM 创建必要的目录
echo [INFO] 创建必要的目录...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "uploads" mkdir uploads
if not exist "temp" mkdir temp
if not exist "config\nginx\conf.d" mkdir config\nginx\conf.d
if not exist "ssl" mkdir ssl
echo [SUCCESS] 目录创建完成

REM 构建Docker镜像
echo [INFO] 构建统一RAG服务Docker镜像...
docker compose -f docker-compose.unified.yml build
if errorlevel 1 (
    echo [ERROR] Docker镜像构建失败
    pause
    exit /b 1
)
echo [SUCCESS] Docker镜像构建完成

REM 启动服务
echo [INFO] 启动统一RAG服务...

REM 首先启动基础服务
echo [INFO] 启动基础服务（MySQL, Redis, Milvus等）...
docker compose -f docker-compose.unified.yml up -d mysql redis etcd minio elasticsearch

REM 等待基础服务启动
echo [INFO] 等待基础服务启动...
timeout /t 30 /nobreak >nul

REM 启动Milvus
echo [INFO] 启动Milvus向量数据库...
docker compose -f docker-compose.unified.yml up -d milvus
timeout /t 20 /nobreak >nul

REM 启动主应用
echo [INFO] 启动统一RAG API服务...
docker compose -f docker-compose.unified.yml up -d unified-rag-api

echo [SUCCESS] 所有服务启动完成

REM 检查服务状态
echo [INFO] 检查服务状态...
docker compose -f docker-compose.unified.yml ps

REM 检查API服务健康状态
echo [INFO] 检查API服务健康状态...
set /a count=0
:health_check
set /a count+=1
curl -f http://localhost:8000/health/ping >nul 2>&1
if errorlevel 1 (
    if !count! geq 10 (
        echo [INFO] 等待API服务启动... (!count!/10)
        timeout /t 10 /nobreak >nul
        goto health_check
    ) else (
        echo [ERROR] API服务健康检查失败
        pause
        exit /b 1
    )
) else (
    echo [SUCCESS] API服务健康检查通过
)

REM 显示服务信息
echo.
echo =============================================
echo 🚀 统一RAG服务部署成功！
echo =============================================
echo.
echo 📍 服务地址:
echo    主API服务: http://localhost:8000
echo    API文档: http://localhost:8000/docs
echo    健康检查: http://localhost:8000/health/ping
echo.
echo 🔧 管理命令:
echo    查看日志: docker compose -f docker-compose.unified.yml logs -f unified-rag-api
echo    停止服务: docker compose -f docker-compose.unified.yml down
echo    重启服务: docker compose -f docker-compose.unified.yml restart unified-rag-api
echo.
echo 📊 外部服务地址:
echo    Kibana: http://localhost:5601
echo    MinIO: http://localhost:9000
echo.

pause