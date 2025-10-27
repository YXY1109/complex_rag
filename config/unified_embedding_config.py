"""
统一嵌入服务配置
"""

from typing import Dict, Any

def get_unified_embedding_config() -> Dict[str, Any]:
    """获取统一嵌入服务配置"""
    return {
        # 缓存配置
        "cache": {
            "enabled": True,
            "ttl": 3600,  # 1小时
            "max_size": 100000  # 最大缓存条目数
        },

        # 性能配置
        "max_concurrent_requests": 10,
        "default_batch_size": 32,
        "request_timeout": 30,

        # 模型配置
        "models": {
            "bce-base": {
                "model_type": "bce",
                "model_path": None,  # 将使用默认路径
                "device": "cpu",
                "use_gpu": False,
                "dimension": 768,
                "max_length": 512,
                "batch_size": 32,
                "cache_enabled": True,
                "priority": 1,
                "model_params": {
                    "normalize": True
                }
            },

            "qwen3-embedding": {
                "model_type": "qwen3",
                "model_path": None,  # 将使用默认路径
                "device": "cpu",
                "use_gpu": False,
                "dimension": 1536,
                "max_length": 512,
                "batch_size": 16,
                "cache_enabled": True,
                "priority": 2,
                "model_params": {
                    "normalize": True
                }
            },

            "openai-text-embedding-ada-002": {
                "model_type": "openai",
                "model_name": "text-embedding-ada-002",
                "dimension": 1536,
                "max_length": 8191,
                "batch_size": 100,
                "cache_enabled": True,
                "priority": 3,
                "api_key": None,  # 从环境变量获取
                "api_base": None,
                "model_params": {}
            },

            "openai-text-embedding-3-small": {
                "model_type": "openai",
                "model_name": "text-embedding-3-small",
                "dimension": 1536,
                "max_length": 8191,
                "batch_size": 100,
                "cache_enabled": True,
                "priority": 3,
                "api_key": None,  # 从环境变量获取
                "api_base": None,
                "model_params": {}
            },

            "openai-text-embedding-3-large": {
                "model_type": "openai",
                "model_name": "text-embedding-3-large",
                "dimension": 3072,
                "max_length": 8191,
                "batch_size": 100,
                "cache_enabled": True,
                "priority": 3,
                "api_key": None,  # 从环境变量获取
                "api_base": None,
                "model_params": {}
            }
        },

        # 默认模型
        "default_model": "bce-base",

        # 模型路径配置
        "model_paths": {
            "bce": "models/bce-embedding-base_v1",
            "qwen3": "models/Qwen/Qwen3-Embedding-0.6B",
            "generic": "models/generic-embedding"
        }
    }