#!/usr/bin/env python3
"""
统一FastAPI服务启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ["PYTHONPATH"] = str(project_root)

if __name__ == "__main__":
    import uvicorn
    from api.unified_main import app
    from config.settings_simple import get_settings

    settings = get_settings()

    print(f"🚀 启动统一RAG FastAPI服务...")
    print(f"📍 服务地址: http://{settings.api_host}:{settings.api_port}")
    print(f"📚 API文档: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🔍 详细健康检查: http://{settings.api_host}:{settings.api_port}/health/")
    print(f"🏥 简单健康检查: http://{settings.api_host}:{settings.api_port}/health/ping")
    print(f"💬 聊天API: http://{settings.api_host}:{settings.api_port}/v1/chat/completions")
    print(f"🔤 嵌入API: http://{settings.api_host}:{settings.api_port}/v1/embeddings/")
    print(f"📄 重排序API: http://{settings.api_host}:{settings.api_port}/v1/rerank/")
    print(f"🧠 记忆API: http://{settings.api_host}:{settings.api_port}/v1/memory/")
    print(f"🌍 环境: {settings.environment}")
    print("=" * 60)

    # 启动统一FastAPI应用
    uvicorn.run(
        "start_unified_api:app",  # 使用字符串以支持reload
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level="info",
        access_log=True,
        workers=1,  # 单进程模式，符合要求
    )