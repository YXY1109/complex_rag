"""
统一向量存储配置
"""

from typing import Dict, Any, List, Optional


def get_unified_vector_store_config() -> Dict[str, Any]:
    """获取统一向量存储配置"""
    return {
        # 默认后端配置
        "default_backend": "milvus",
        "max_concurrent_requests": 10,
        "request_timeout": 30,

        # 后端配置
        "backends": {
            "milvus": {
                "type": "milvus",
                "host": "localhost",
                "port": 19530,
                "user": None,
                "password": None,
                "db_name": "default",
                "timeout": 30,
                "connection_pool_size": 10,
                "enable_tls": False,
                "consistency_level": "Strong"
            },
            "elasticsearch": {
                "type": "elasticsearch",
                "hosts": ["localhost:9200"],
                "username": None,
                "password": None,
                "scheme": "http",
                "verify_certs": False,
                "timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            },
            "faiss": {
                "type": "faiss",
                "index_type": "hnsw",
                "dimension": 768,
                "metric": "cosine",
                "nlist": 128,
                "nprobe": 8,
                "M": 16,
                "efConstruction": 200,
                "efSearch": 50
            },
            "qdrant": {
                "type": "qdrant",
                "host": "localhost",
                "port": 6333,
                "api_key": None,
                "timeout": 30,
                "prefer_grpc": False,
                "https": False
            },
            "chroma": {
                "type": "chroma",
                "persist_directory": "./chroma_db",
                "host": "localhost",
                "port": 8000,
                "auth_method": "basic",
                "username": None,
                "password": None,
                "tenant": None",
                "database": None
            }
        },

        # 集合配置
        "collections": {
            "documents": {
                "name": "documents",
                "dimension": 1536,
                "description": "文档向量存储",
                "max_capacity": 1000000,
                "shard_num": 1,
                "consistency_level": "Strong",
                "index_config": {
                    "index_type": "hnsw",
                    "metric_type": "cosine",
                    "dimension": 1536,
                    "ef_construction": 200,
                    "M": 16,
                    "nlist": 128,
                    "nprobe": 8
                },
                "metadata": {
                    "schema_version": "1.0",
                    "created_by": "vector_store_migration",
                    "created_at": "2025-01-27T00:00:00Z"
                }
            },
            "knowledge_base": {
                "name": "knowledge_base",
                "dimension": 1536,
                "description": "知识库向量存储",
                "max_capacity": 500000,
                "shard_num": 1,
                "consistency_level": "Strong",
                "index_config": {
                    "index_type": "hnsw",
                    "metric_type": "cosine",
                    "dimension": 1536,
                    "ef_construction": 200,
                    "M": 16,
                    "nlist": 128,
                    "nprobe": 8
                },
                "metadata": {
                    "schema_version": "1.0",
                    "created_by": "vector_store_migration",
                    "created_at": "2025-01-27T00:00:00Z"
                }
            },
            "chunks": {
                "name": "chunks",
                "dimension": 1536,
                "description": "文档块向量存储",
                "max_capacity": 2000000,
                "shard_num": 1,
                "consistency_level": "Strong",
                "index_config": {
                    "index_type": "hnsw",
                    "metric_type": "cosine",
                    "dimension": 1536,
                    "ef_construction": 200,
                    "M": 16,
                    "nlist": 128,
                    "nprobe": 8
                },
                "metadata": {
                    "schema_version": "1.0",
                    "created_by": "vector_store_migration",
                    "created_at": "2025-01-27T00:00:00Z"
                }
            },
            "conversations": {
                "name": "conversations",
                "dimension": 1536,
                "description": "对话历史向量存储",
                "max_capacity": 100000,
                "shard_num": 1,
                "consistency_level": "Strong",
                "index_config": {
                    "index_type": "hnsw",
                    "metric_type": "cosine",
                    "dimension": 1536,
                    "ef_construction": 200,
                    "M": 16,
                    "nlist": 128,
                    "nprobe": 8
                },
                "metadata": {
                    "schema_version": "1.0",
                    "created_by": "vector_store_migration",
                    "created_at": "2025-01-27T00:00:00Z"
                }
            }
        },

        # 性能配置
        "performance": {
            "batch_size": 1000,
            "parallel_search": True,
            "cache_enabled": True,
            "cache_ttl": 3600,
            "index_cache_size": 1000,
            "search_timeout": 30,
            "write_timeout": 60,
            "connection_timeout": 10
        },

        # 索引配置模板
        "index_templates": {
            "high_precision": {
                "index_type": "hnsw",
                "metric_type": "cosine",
                "ef_construction": 400,
                "M": 48,
                "nlist": 256,
                "nprobe": 16,
                "description": "高精度搜索，适用于精确度要求高的场景"
            },
            "high_speed": {
                "index_type": "hnsw",
                "metric_type": "cosine",
                "ef_construction": 200,
                "M": 16,
                "nlist": 128,
                "nprobe": 8,
                "description": "高速度搜索，适用于大规模数据集"
            },
            "balanced": {
                "index_type": "hnsw",
                "metric_type": "cosine",
                "ef_construction": 300,
                "M": 32,
                "nlist": 192,
                "nprobe": 12,
                "description": "平衡精度与速度的通用配置"
            },
            "memory_optimized": {
                "index_type": "hnsw",
                "metric_type": "cosine",
                "ef_construction": 128,
                "M": 24,
                "nlist": 96,
                "nprobe": 6,
                "description": "内存优化配置，适用于内存受限环境"
            }
        },

        # 过滤器配置
        "filters": {
            "supported_operators": [
                "eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in",
                "contains", "regex", "exists"
            ],
            "supported_field_types": [
                "string", "integer", "float", "boolean", "array", "object"
            ]
        },

        # 监控配置
        "monitoring": {
            "enable_metrics": True,
            "health_check_interval": 60,
            "performance_logging": True,
            "slow_query_threshold": 1.0,
            "stats_retention_days": 30
        },

        # 安全配置
        "security": {
            "enable_authentication": False,
            "api_keys": [],
            "allowed_origins": ["*"],
            "rate_limit": {
                "requests_per_minute": 1000,
                "burst_size": 100
            },
            "encryption": {
                "enabled": False,
                "algorithm": "aes256",
                "key_rotation_days": 90
            }
        },

        # 维护配置
        "maintenance": {
            "auto_compaction": True,
            "compaction_interval": 86400,  # 24小时
            "auto_reindex": True,
            "reindex_interval": 604800,  # 7天
            "backup_enabled": True,
            "backup_interval": 86400,  # 24小时
            "backup_retention_days": 30
        }
    }


def get_vector_store_config_for_environment(env: str) -> Dict[str, Any]:
    """根据环境获取向量存储配置"""
    config = get_unified_vector_store_config()

    if env == "development":
        # 开发环境配置
        config["backends"]["milvus"]["host"] = "localhost"
        config["performance"]["cache_enabled"] = True
        config["monitoring"]["performance_logging"] = True
        config["maintenance"]["auto_compaction"] = True

    elif env == "testing":
        # 测试环境配置
        config["backends"]["milvus"]["host"] = "localhost"
        config["collections"]["documents"]["max_capacity"] = 100000
        config["performance"]["batch_size"] = 100
        config["monitoring"]["health_check_interval"] = 30

    elif env == "production":
        # 生产环境配置
        config["backends"]["milvus"]["host"] = "milvus-cluster.local"
        config["security"]["enable_authentication"] = True
        config["monitoring"]["slow_query_threshold"] = 0.5
        config["maintenance"]["auto_compaction"] = True
        config["maintenance"]["auto_reindex"] = True

    return config


def get_collection_config(collection_name: str) -> Optional[Dict[str, Any]]:
    """获取指定集合的配置"""
    config = get_unified_vector_store_config()
    collections = config.get("collections", {})

    if collection_name in collections:
        collection_config = collections[collection_name]
        # 设置默认值
        collection_config.setdefault("name", collection_name)
        collection_config.setdefault("dimension", 1536)
        collection_config.setdefault("description", f"{collection_name}向量存储")
        collection_config.setdefault("max_capacity", 1000000)
        collection_config.setdefault("shard_num", 1)
        collection_config.setdefault("consistency_level", "Strong")

        # 设置索引配置
        if "index_config" not in collection_config:
            collection_config["index_config"] = {
                "index_type": "hnsw",
                "metric_type": "cosine",
                "dimension": collection_config["dimension"],
                "ef_construction": 200,
                "M": 16,
                "nlist": 128,
                "nprobe": 8
            }

        return collection_config

    return None


# 支持的后端类型
SUPPORTED_BACKENDS = ["milvus", "elasticsearch", "faiss", "qdrant", "chroma"]

# 支持的索引类型
SUPPORTED_INDEX_TYPES = ["flat", "ivf_flat", "ivf_sq", "ivf_pq", "hnsw", "annoy", "scann"]

# 支持的距离度量
SUPPORTED_DISTANCE_METRICS = ["cosine", "euclidean", "manhattan", "dot_product", "hamming", "jaccard"]