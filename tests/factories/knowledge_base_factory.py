"""
知识库数据工厂
生成测试用的知识库数据
"""
from factory import Factory, Faker, SubFactory, lazy_attribute, List
from typing import Dict, Any, List as ListType
import uuid
from datetime import datetime


class KnowledgeBaseFactory(Factory):
    """知识库工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    name = Faker("sentence", nb_words=3)
    description = Faker("text", max_nb_chars=500)
    owner_id = Faker("uuid4")
    created_at = Faker("date_time_this_year")
    updated_at = Faker("date_time_this_month")

    @lazy_attribute
    def settings(self):
        """生成知识库设置"""
        return {
            "embedding_model": Faker("random_element", elements=[
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]),
            "reranking_model": "bge-reranker-base",
            "chunk_size": Faker("random_int", min=500, max=2000),
            "chunk_overlap": Faker("random_int", min=50, max=200),
            "search_top_k": Faker("random_int", min=3, max=20),
            "similarity_threshold": round(Faker("random.uniform", min=0.5, max=0.9), 3),
            "language": Faker("random_element", elements=["zh-CN", "en-US", "ja-JP"]),
            "auto_update": Faker("boolean"),
            "enable_rerank": Faker("boolean"),
            "enable_hybrid_search": Faker("boolean")
        }

    @lazy_attribute
    def metadata(self):
        """生成知识库元数据"""
        return {
            "tags": Faker("words", nb=5),
            "category": Faker("word"),
            "domain": Faker("random_element", elements=["general", "technical", "medical", "legal", "education"]),
            "access_level": Faker("random_element", elements=["public", "private", "team", "organization"]),
            "license": Faker("random_element", elements=["MIT", "CC-BY", "CC-BY-SA", "proprietary"]),
            "version": "1.0.0",
            "contact_email": Faker("email"),
            "documentation_url": Faker("url")
        }

    @lazy_attribute
    def statistics(self):
        """生成统计信息"""
        return {
            "document_count": Faker("random_int", min=0, max=10000),
            "total_chunks": Faker("random_int", min=0, max=100000),
            "total_tokens": Faker("random_int", min=0, max=10000000),
            "index_size_mb": round(Faker("random.uniform", min=0, max=10000), 2),
            "last_updated": Faker("date_time_this_month"),
            "last_searched": Faker("date_time_this_week"),
            "search_count": Faker("random_int", min=0, max=100000),
            "average_search_time": round(Faker("random.uniform", min=10, max=500), 2)  # 毫秒
        }

    @lazy_attribute
    def status(self):
        """生成状态信息"""
        return {
            "state": Faker("random_element", elements=["active", "indexing", "error", "maintenance"]),
            "is_ready": Faker("boolean", truth_probability=0.8),
            "error_message": None,
            "health_score": round(Faker("random.uniform", min=0.7, max=1.0), 3),
            "last_health_check": Faker("date_time_this_day")
        }


class KnowledgeBaseDocumentFactory(Factory):
    """知识库文档工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    knowledge_base_id = SubFactory(KnowledgeBaseFactory).id
    document_id = Faker("uuid4")
    title = Faker("sentence", nb_words=4)
    content = Faker("text", max_nb_chars=2000)
    file_name = Faker("file_name", extension="txt")
    file_size = Faker("random_int", min=1024, max=10485760)
    added_at = Faker("date_time_this_month")

    @lazy_attribute
    def processing_status(self):
        """生成处理状态"""
        return {
            "state": Faker("random_element", elements=["pending", "processing", "completed", "failed"]),
            "progress": Faker("random_int", min=0, max=100),
            "chunks_created": Faker("random_int", min=0, max=100),
            "embeddings_generated": Faker("random_int", min=0, max=100),
            "error_count": Faker("random_int", min=0, max=5),
            "processing_time": Faker("random_int", min=1, max=300),  # 秒
            "last_processed": Faker("date_time_this_month")
        }

    @lazy_attribute
    def metadata(self):
        """生成文档元数据"""
        return {
            "original_source": Faker("url"),
            "content_type": Faker("random_element", elements=["text", "pdf", "word", "web", "api"]),
            "language": Faker("random_element", elements=["zh-CN", "en-US"]),
            "quality_score": round(Faker("random.uniform", min=0.7, max=1.0), 3),
            "relevance_score": round(Faker("random.uniform", min=0.5, max=1.0), 3),
            "tags": Faker("words", nb=3),
            "author": Faker("name"),
            "created_date": Faker("date_between", start_date="-2y", end_date="today")
        }


class KnowledgeBaseChunkFactory(Factory):
    """知识库文档块工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    knowledge_base_id = SubFactory(KnowledgeBaseFactory).id
    document_id = SubFactory(KnowledgeBaseDocumentFactory).document_id
    content = Faker("text", max_nb_chars=500)
    chunk_index = Faker("random_int", min=0, max=1000)
    token_count = Faker("random_int", min=50, max=500)

    @lazy_attribute
    def metadata(self):
        """生成块元数据"""
        return {
            "start_position": Faker("random_int", min=0, max=10000),
            "end_position": Faker("random_int", min=100, max=15000),
            "page_number": Faker("random_int", min=1, max=100),
            "section_title": Faker("sentence", nb_words=3),
            "chunk_type": Faker("random_element", elements=["text", "heading", "table", "list", "code"]),
            "importance_score": round(Faker("random.uniform", min=0.3, max=1.0), 3),
            "density_score": round(Faker("random.uniform", min=0.1, max=1.0), 3)
        }

    @lazy_attribute
    def embedding_info(self):
        """生成嵌入向量信息"""
        return {
            "model": "text-embedding-ada-002",
            "dimension": 1536,
            "embedding_id": Faker("uuid4"),
            "indexed_at": Faker("date_time_this_month"),
            "index_version": "1.0"
        }


class KnowledgeBaseSearchFactory(Factory):
    """知识库搜索工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    knowledge_base_id = SubFactory(KnowledgeBaseFactory).id
    query = Faker("text", max_nb_chars=200)
    filters = {}
    search_settings = {}
    results = None  # 将在lazy_attribute中生成
    search_time = Faker("random_int", min=10, max=1000)  # 毫秒
    created_at = Faker("date_time_this_month")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result_count = kwargs.get("result_count", 5)

    @lazy_attribute
    def filters(self):
        """生成搜索过滤条件"""
        return {
            "document_ids": [Faker("uuid4") for _ in range(Faker("random_int", min=0, max=3))],
            "tags": Faker("words", nb=3),
            "date_range": {
                "start": Faker("date_between", start_date="-1y", end_date="-1m"),
                "end": Faker("date_between", start_date="-1m", end_date="today")
            },
            "content_types": Faker("random_elements", elements=["text", "pdf", "word"], length=2),
            "min_relevance": round(Faker("random.uniform", min=0.5, max=0.8), 3)
        }

    @lazy_attribute
    def search_settings(self):
        """生成搜索设置"""
        return {
            "top_k": Faker("random_int", min=3, max=20),
            "similarity_threshold": round(Faker("random.uniform", min=0.5, max=0.9), 3),
            "rerank": Faker("boolean"),
            "hybrid_search": Faker("boolean"),
            "include_metadata": Faker("boolean"),
            "highlight_results": Faker("boolean")
        }

    @lazy_attribute
    def results(self):
        """生成搜索结果"""
        results = []
        for i in range(self.result_count):
            result = {
                "id": str(uuid.uuid4()),
                "document_id": str(uuid.uuid4()),
                "chunk_id": str(uuid.uuid4()),
                "content": Faker("text", max_nb_chars=300),
                "score": round(Faker("random.uniform", min=0.7, max=1.0), 6),
                "metadata": {
                    "title": Faker("sentence", nb_words=3),
                    "source": Faker("url"),
                    "author": Faker("name"),
                    "created_at": Faker("date_time_this_year"),
                    "tags": Faker("words", nb=2)
                },
                "highlights": [
                    {
                        "text": Faker("sentence", nb_words=5),
                        "score": round(Faker("random.uniform", min=0.8, max=1.0), 3)
                    }
                ]
            }
            results.append(result)
        return results

    @lazy_attribute
    def statistics(self):
        """生成搜索统计"""
        return {
            "total_results": len(self.results),
            "search_strategy": Faker("random_element", elements=["vector", "keyword", "hybrid"]),
            "candidates_evaluated": Faker("random_int", min=self.result_count, max=self.result_count * 10),
            "reranked_results": Faker("random_int", min=0, max=self.result_count),
            "average_score": round(sum(r["score"] for r in self.results) / len(self.results), 6) if self.results else 0.0
        }


class KnowledgeBaseChatFactory(Factory):
    """知识库对话工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    knowledge_base_id = SubFactory(KnowledgeBaseFactory).id
    user_id = Faker("uuid4")
    messages = None  # 将在lazy_attribute中生成
    model = Faker("random_element", elements=["gpt-3.5-turbo", "gpt-4"])
    created_at = Faker("date_time_this_month")
    updated_at = Faker("date_time_this_month")

    @lazy_attribute
    def messages(self):
        """生成对话消息"""
        from .llm_factory import LLMMessageFactory
        return [
            LLMMessageFactory(role="system", content=f"你是一个基于知识库 {self.knowledge_base_id} 的AI助手。"),
            LLMMessageFactory(role="user", content="请介绍一下相关知识。"),
            LLMMessageFactory(role="assistant", content="根据知识库中的信息，我可以为您提供以下介绍...")
        ]

    @lazy_attribute
    def context(self):
        """生成上下文信息"""
        return {
            "retrieved_chunks": Faker("random_int", min=1, max=10),
            "total_tokens": Faker("random_int", min=500, max=5000),
            "citations": [Faker("uuid4") for _ in range(Faker("random_int", min=1, max=5))],
            "search_time": Faker("random_int", min=50, max=500),
            "generation_time": Faker("random_int", min=100, max=3000)
        }

    @lazy_attribute
    def metadata(self):
        """生成对话元数据"""
        return {
            "title": Faker("sentence", nb_words=4),
            "tags": Faker("words", nb=3),
            "feedback_score": round(Faker("random.uniform", min=1, max=5), 1),
            "is_helpful": Faker("boolean", truth_probability=0.8),
            "resolved": Faker("boolean", truth_probability=0.7)
        }


# 便捷函数
def create_test_knowledge_base(**overrides) -> Dict[str, Any]:
    """创建测试知识库"""
    return KnowledgeBaseFactory(**overrides)


def create_test_knowledge_base_document(knowledge_base_id: str, **overrides) -> Dict[str, Any]:
    """创建测试知识库文档"""
    return KnowledgeBaseDocumentFactory(knowledge_base_id=knowledge_base_id, **overrides)


def create_test_knowledge_base_chunk(knowledge_base_id: str, document_id: str, **overrides) -> Dict[str, Any]:
    """创建测试知识库文档块"""
    return KnowledgeBaseChunkFactory(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        **overrides
    )


def create_test_knowledge_base_search(knowledge_base_id: str, result_count: int = 5, **overrides) -> Dict[str, Any]:
    """创建测试知识库搜索"""
    return KnowledgeBaseSearchFactory(
        knowledge_base_id=knowledge_base_id,
        result_count=result_count,
        **overrides
    )


def create_test_knowledge_base_chat(knowledge_base_id: str, user_id: str, **overrides) -> Dict[str, Any]:
    """创建测试知识库对话"""
    return KnowledgeBaseChatFactory(
        knowledge_base_id=knowledge_base_id,
        user_id=user_id,
        **overrides
    )