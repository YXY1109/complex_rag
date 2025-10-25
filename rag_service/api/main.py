"""
API服务主入口

启动FastAPI应用的主入口文件。
"""

import uvicorn
import logging
from typing import Dict, Any
import json
import os

from .app import create_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """
    加载配置文件。

    Returns:
        Dict[str, Any]: 配置字典
    """
    # 默认配置
    default_config = {
        "rag": {
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "rag_engine": {
            "enable_reranking": True,
            "enable_context_optimization": True,
            "parallel_processing": True
        },
        "vector_store": {
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "rag_vectors"
            },
            "elasticsearch": {
                "hosts": ["localhost:9200"],
                "index_name": "rag_documents"
            }
        },
        "embedding": {
            "models": {
                "text-embedding-ada-002": {
                    "model_name": "text-embedding-ada-002",
                    "model_type": "openai",
                    "api_key": os.getenv("OPENAI_API_KEY", "your-openai-key"),
                    "default": True
                }
            }
        },
        "generation": {
            "models": {
                "gpt-3.5-turbo": {
                    "model_name": "gpt-3.5-turbo",
                    "model_type": "openai",
                    "api_key": os.getenv("OPENAI_API_KEY", "your-openai-key"),
                    "default": True
                }
            }
        },
        "knowledge_manager": {
            "max_kb_per_tenant": 10,
            "max_documents_per_kb": 1000
        },
        "chat": {
            "max_messages_per_session": 50,
            "max_conversation_history": 10
        },
        "cache": {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None
            },
            "memory": {
                "max_size": 1000,
                "ttl": 3600
            }
        },
        "monitoring": {
            "enable_metrics": True,
            "enable_tracing": True,
            "enable_alerting": True
        }
    }

    # 尝试从配置文件加载
    config_file = os.getenv("RAG_CONFIG_FILE", "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            # 合并配置
            default_config.update(file_config)
            logger.info(f"已加载配置文件: {config_file}")
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")

    return default_config


def main():
    """主函数。"""
    # 加载配置
    config = load_config()

    # 创建应用
    app = create_app(config)

    # 获取运行参数
    host = os.getenv("RAG_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_PORT", "8000"))
    workers = int(os.getenv("RAG_WORKERS", "1"))
    reload = os.getenv("RAG_RELOAD", "false").lower() == "true"
    log_level = os.getenv("RAG_LOG_LEVEL", "info")

    logger.info(
        f"启动RAG API服务 - "
        f"地址: {host}:{port}, "
        f"工作进程: {workers}, "
        f"重载: {reload}, "
        f"日志级别: {log_level}"
    )

    # 启动服务
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers if not reload else 1,  # reload模式下只能使用1个worker
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()