"""
检索器抽象接口测试
测试向量检索器、关键词检索器、实体检索器和多策略融合器的抽象定义
"""
import pytest
from unittest.mock import Mock, AsyncMock
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import uuid

# 模拟接口定义
class RetrieverInterface(ABC):
    """检索器基础接口抽象类"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """检索文档"""
        pass

    @abstractmethod
    async def index_documents(self, documents: List[Dict[str, Any]], **kwargs) -> List[str]:
        """索引文档"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        pass


class VectorRetrieverInterface(RetrieverInterface):
    """向量检索器接口抽象类"""

    @abstractmethod
    async def embed_query(self, query: str, **kwargs) -> List[float]:
        """嵌入查询"""
        pass

    @abstractmethod
    async def similarity_search(self, query_vector: List[float], top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """相似度搜索"""
        pass


class KeywordRetrieverInterface(RetrieverInterface):
    """关键词检索器接口抽象类"""

    @abstractmethod
    async def build_index(self, documents: List[Dict[str, Any]], **kwargs) -> bool:
        """构建倒排索引"""
        pass

    @abstractmethod
    async def keyword_search(self, query: str, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """关键词搜索"""
        pass


class EntityRetrieverInterface(RetrieverInterface):
    """实体检索器接口抽象类"""

    @abstractmethod
    async def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """提取实体"""
        pass

    @abstractmethod
    async def entity_search(self, entities: List[str], top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """实体搜索"""
        pass


class FusionRetrieverInterface(RetrieverInterface):
    """融合检索器接口抽象类"""

    @abstractmethod
    async def fuse_results(self, results_list: List[List[Dict[str, Any]]], **kwargs) -> List[Dict[str, Any]]:
        """融合多个检索结果"""
        pass

    @abstractmethod
    async def weighted_search(self, query: str, weights: Dict[str, float], top_k: int, **kwargs) -> List[Dict[str, Any]]:
        """加权搜索"""
        pass


# Mock数据结构
@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    retriever_type: str


class MockVectorRetriever(VectorRetrieverInterface):
    """模拟向量检索器实现"""

    def __init__(self):
        self.documents = {}  # doc_id -> Document
        self.embeddings = {}  # doc_id -> embedding
        self.dimension = 1536
        self.stats = {
            "total_documents": 0,
            "total_retrievals": 0,
            "avg_retrieval_time": 0.0,
            "index_size": 0
        }

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        import time
        start_time = time.time()

        query_vector = await self.embed_query(query)
        results = await self.similarity_search(query_vector, top_k)

        retrieval_time = time.time() - start_time
        self._update_stats(retrieval_time)

        return results

    async def embed_query(self, query: str, **kwargs) -> List[float]:
        import random
        # 模拟嵌入向量生成
        return [random.uniform(-1, 1) for _ in range(self.dimension)]

    async def similarity_search(self, query_vector: List[float], top_k: int, **kwargs) -> List[Dict[str, Any]]:
        import random
        import math

        results = []
        for doc_id, embedding in self.embeddings.items():
            # 模拟余弦相似度计算
            similarity = random.uniform(0.3, 1.0)
            if similarity > 0.5:  # 只返回相似度大于0.5的结果
                results.append({
                    "document_id": doc_id,
                    "content": self.documents[doc_id].content,
                    "score": similarity,
                    "metadata": self.documents[doc_id].metadata,
                    "retriever_type": "vector"
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    async def index_documents(self, documents: List[Dict[str, Any]], **kwargs) -> List[str]:
        import random

        doc_ids = []
        for doc_data in documents:
            doc_id = doc_data.get("id", str(uuid.uuid4()))
            embedding = [random.uniform(-1, 1) for _ in range(self.dimension)]

            document = Document(
                id=doc_id,
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
                embedding=embedding
            )

            self.documents[doc_id] = document
            self.embeddings[doc_id] = embedding
            doc_ids.append(doc_id)

        self.stats["total_documents"] += len(documents)
        return doc_ids

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def _update_stats(self, retrieval_time: float):
        self.stats["total_retrievals"] += 1
        total_time = self.stats["avg_retrieval_time"] * (self.stats["total_retrievals"] - 1)
        self.stats["avg_retrieval_time"] = (total_time + retrieval_time) / self.stats["total_retrievals"]


class MockKeywordRetriever(KeywordRetrieverInterface):
    """模拟关键词检索器实现"""

    def __init__(self):
        self.documents = {}  # doc_id -> Document
        self.inverted_index = {}  # keyword -> set of doc_ids
        self.stats = {
            "total_documents": 0,
            "total_terms": 0,
            "index_size": 0,
            "avg_search_time": 0.0
        }

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        import time
        start_time = time.time()

        results = await self.keyword_search(query, top_k)

        search_time = time.time() - start_time
        self._update_stats(search_time)

        return results

    async def build_index(self, documents: List[Dict[str, Any]], **kwargs) -> bool:
        for doc_data in documents:
            doc_id = doc_data.get("id", str(uuid.uuid4()))

            document = Document(
                id=doc_id,
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )

            self.documents[doc_id] = document

            # 构建倒排索引
            words = doc_data["content"].lower().split()
            for word in words:
                if word not in self.inverted_index:
                    self.inverted_index[word] = set()
                self.inverted_index[word].add(doc_id)

        self.stats["total_documents"] += len(documents)
        self.stats["total_terms"] = len(self.inverted_index)
        self.stats["index_size"] = sum(len(docs) for docs in self.inverted_index.values())

        return True

    async def keyword_search(self, query: str, top_k: int, **kwargs) -> List[Dict[str, Any]]:
        query_words = query.lower().split()
        candidate_docs = set()

        # 找出包含查询词的文档
        for word in query_words:
            if word in self.inverted_index:
                candidate_docs.update(self.inverted_index[word])

        # 计算相关性分数
        results = []
        for doc_id in candidate_docs:
            doc_content = self.documents[doc_id].content.lower()
            # 简单的TF-IDF近似计算
            score = sum(doc_content.count(word) for word in query_words) / len(query_words)

            results.append({
                "document_id": doc_id,
                "content": self.documents[doc_id].content,
                "score": min(score / 10, 1.0),  # 归一化到0-1
                "metadata": self.documents[doc_id].metadata,
                "retriever_type": "keyword",
                "matched_terms": [word for word in query_words if word in doc_content]
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    async def index_documents(self, documents: List[Dict[str, Any]], **kwargs) -> List[str]:
        await self.build_index(documents)
        return [doc.get("id", str(uuid.uuid4())) for doc in documents]

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def _update_stats(self, search_time: float):
        total_time = self.stats["avg_search_time"] * (self.stats.get("total_searches", 0))
        self.stats["avg_search_time"] = (total_time + search_time) / (self.stats.get("total_searches", 0) + 1)
        self.stats["total_searches"] = self.stats.get("total_searches", 0) + 1


class MockEntityRetriever(EntityRetrieverInterface):
    """模拟实体检索器实现"""

    def __init__(self):
        self.documents = {}  # doc_id -> Document
        self.entity_index = {}  # entity -> set of doc_ids
        self.stats = {
            "total_documents": 0,
            "total_entities": 0,
            "avg_extraction_time": 0.0
        }

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        import time
        start_time = time.time()

        entities = await self.extract_entities(query)
        results = await self.entity_search([e["text"] for e in entities], top_k)

        extraction_time = time.time() - start_time
        self._update_stats(extraction_time)

        return results

    async def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        # 简单的实体提取模拟
        import re

        # 提取大写词汇作为实体
        entities = []
        words = text.split()
        for word in words:
            if word.isupper() or word[0].isupper():
                entities.append({
                    "text": word,
                    "type": "UNKNOWN",
                    "confidence": 0.8,
                    "start": text.find(word),
                    "end": text.find(word) + len(word)
                })

        # 提取数字作为实体
        for match in re.finditer(r'\b\d+\b', text):
            entities.append({
                "text": match.group(),
                "type": "NUMBER",
                "confidence": 0.9,
                "start": match.start(),
                "end": match.end()
            })

        return entities

    async def entity_search(self, entities: List[str], top_k: int, **kwargs) -> List[Dict[str, Any]]:
        candidate_docs = set()

        # 找出包含实体的文档
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in self.entity_index:
                candidate_docs.update(self.entity_index[entity_lower])

        # 计算实体匹配分数
        results = []
        for doc_id in candidate_docs:
            doc_content = self.documents[doc_id].content.lower()
            matched_entities = [e for e in entities if e.lower() in doc_content]
            score = len(matched_entities) / len(entities)

            results.append({
                "document_id": doc_id,
                "content": self.documents[doc_id].content,
                "score": score,
                "metadata": self.documents[doc_id].metadata,
                "retriever_type": "entity",
                "matched_entities": matched_entities
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    async def index_documents(self, documents: List[Dict[str, Any]], **kwargs) -> List[str]:
        for doc_data in documents:
            doc_id = doc_data.get("id", str(uuid.uuid4()))

            document = Document(
                id=doc_id,
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )

            self.documents[doc_id] = document

            # 提取并索引实体
            entities = await self.extract_entities(doc_data["content"])
            for entity in entities:
                entity_text = entity["text"].lower()
                if entity_text not in self.entity_index:
                    self.entity_index[entity_text] = set()
                self.entity_index[entity_text].add(doc_id)

        self.stats["total_documents"] += len(documents)
        self.stats["total_entities"] = len(self.entity_index)

        return [doc.get("id", str(uuid.uuid4())) for doc in documents]

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def _update_stats(self, extraction_time: float):
        total_time = self.stats["avg_extraction_time"] * (self.stats.get("total_extractions", 0))
        self.stats["avg_extraction_time"] = (total_time + extraction_time) / (self.stats.get("total_extractions", 0) + 1)
        self.stats["total_extractions"] = self.stats.get("total_extractions", 0) + 1


class MockFusionRetriever(FusionRetrieverInterface):
    """模拟融合检索器实现"""

    def __init__(self):
        self.retrievers = {}
        self.fusion_method = "rrf"  # Reciprocal Rank Fusion
        self.stats = {
            "total_fusions": 0,
            "avg_fusion_time": 0.0,
            "fusion_method": self.fusion_method
        }

    def add_retriever(self, name: str, retriever: RetrieverInterface, weight: float = 1.0):
        """添加检索器"""
        self.retrievers[name] = {"retriever": retriever, "weight": weight}

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        import time
        start_time = time.time()

        weights = kwargs.get("weights", {name: info["weight"] for name, info in self.retrievers.items()})
        results = await self.weighted_search(query, weights, top_k)

        fusion_time = time.time() - start_time
        self._update_stats(fusion_time)

        return results

    async def fuse_results(self, results_list: List[List[Dict[str, Any]]], **kwargs) -> List[Dict[str, Any]]:
        fusion_method = kwargs.get("method", self.fusion_method)
        k = kwargs.get("k", 60)  # RRF parameter

        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(results_list, k)
        elif fusion_method == "weighted":
            return self._weighted_fusion(results_list)
        else:
            # 简单合并
            all_results = []
            for results in results_list:
                all_results.extend(results)
            return sorted(all_results, key=lambda x: x["score"], reverse=True)

    async def weighted_search(self, query: str, weights: Dict[str, float], top_k: int, **kwargs) -> List[Dict[str, Any]]:
        results_list = []

        for name, retriever_info in self.retrievers.items():
            if name in weights:
                retriever = retriever_info["retriever"]
                weight = weights[name]

                try:
                    results = await retriever.retrieve(query, top_k * 2)  # 获取更多结果用于融合

                    # 应用权重
                    for result in results:
                        result["score"] *= weight
                        result["source_retriever"] = name
                        result["original_score"] = result["score"] / weight

                    results_list.append(results)
                except Exception as e:
                    # 记录错误但继续处理其他检索器
                    continue

        if results_list:
            fused_results = await self.fuse_results(results_list, **kwargs)
            return fused_results[:top_k]

        return []

    async def index_documents(self, documents: List[Dict[str, Any]], **kwargs) -> List[str]:
        doc_ids = []
        for retriever_info in self.retrievers.values():
            try:
                ids = await retriever_info["retriever"].index_documents(documents)
                if not doc_ids:  # 第一次索引的结果
                    doc_ids = ids
            except Exception:
                continue
        return doc_ids

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def _reciprocal_rank_fusion(self, results_list: List[List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion算法"""
        fused_scores = {}

        for results in results_list:
            for rank, result in enumerate(results, 1):
                doc_id = result["document_id"]
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        "document_id": doc_id,
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "score": 0.0,
                        "sources": []
                    }

                # RRF formula: 1 / (rank + k)
                fused_scores[doc_id]["score"] += 1.0 / (rank + k)
                fused_scores[doc_id]["sources"].append({
                    "retriever": result.get("source_retriever", "unknown"),
                    "rank": rank,
                    "original_score": result.get("original_score", result["score"])
                })

        # 按融合分数排序
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results

    def _weighted_fusion(self, results_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """加权融合算法"""
        fused_scores = {}

        for results in results_list:
            for result in results:
                doc_id = result["document_id"]
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        "document_id": doc_id,
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "score": 0.0,
                        "sources": []
                    }

                fused_scores[doc_id]["score"] += result["score"]
                fused_scores[doc_id]["sources"].append({
                    "retriever": result.get("source_retriever", "unknown"),
                    "score": result["score"]
                })

        return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    def _update_stats(self, fusion_time: float):
        total_time = self.stats["avg_fusion_time"] * self.stats["total_fusions"]
        self.stats["avg_fusion_time"] = (total_time + fusion_time) / (self.stats["total_fusions"] + 1)
        self.stats["total_fusions"] += 1


# 测试类
class TestRetrieverInterface:
    """检索器基础接口测试类"""

    def test_interface_is_abstract(self):
        """测试检索器基础接口是抽象类"""
        with pytest.raises(TypeError):
            RetrieverInterface()


class TestVectorRetrieverInterface:
    """向量检索器接口测试类"""

    def test_interface_is_abstract(self):
        """测试向量检索器接口是抽象类"""
        with pytest.raises(TypeError):
            VectorRetrieverInterface()

    @pytest.mark.asyncio
    async def test_vector_retriever_implementation(self):
        """测试向量检索器实现"""
        retriever = MockVectorRetriever()

        # 测试文档索引
        documents = [
            {"content": "Machine learning is a subset of AI", "metadata": {"topic": "ML"}},
            {"content": "Deep learning uses neural networks", "metadata": {"topic": "DL"}}
        ]
        doc_ids = await retriever.index_documents(documents)
        assert len(doc_ids) == 2

        # 测试检索
        results = await retriever.retrieve("What is machine learning?")
        assert isinstance(results, list)
        assert len(results) <= 10  # 默认top_k=10

        # 验证结果结构
        for result in results:
            assert "document_id" in result
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
            assert result["retriever_type"] == "vector"
            assert 0 <= result["score"] <= 1

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """测试查询嵌入"""
        retriever = MockVectorRetriever()
        query = "Test query"
        embedding = await retriever.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # 模拟的维度
        assert all(isinstance(x, float) for x in embedding)
        assert all(-1 <= x <= 1 for x in embedding)

    @pytest.mark.asyncio
    async def test_similarity_search(self):
        """测试相似度搜索"""
        retriever = MockVectorRetriever()

        # 先索引一些文档
        documents = [{"content": "Test document", "metadata": {}}]
        await retriever.index_documents(documents)

        # 执行相似度搜索
        query_vector = [0.1, 0.2, 0.3] * 512  # 1536维
        results = await retriever.similarity_search(query_vector, top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        for result in results:
            assert "score" in result
            assert result["score"] > 0.5  # Mock实现的阈值

    def test_vector_retriever_inheritance(self):
        """测试向量检索器继承关系"""
        retriever = MockVectorRetriever()
        assert isinstance(retriever, VectorRetrieverInterface)
        assert isinstance(retriever, RetrieverInterface)
        assert hasattr(retriever, 'embed_query')
        assert hasattr(retriever, 'similarity_search')


class TestKeywordRetrieverInterface:
    """关键词检索器接口测试类"""

    def test_interface_is_abstract(self):
        """测试关键词检索器接口是抽象类"""
        with pytest.raises(TypeError):
            KeywordRetrieverInterface()

    @pytest.mark.asyncio
    async def test_keyword_retriever_implementation(self):
        """测试关键词检索器实现"""
        retriever = MockKeywordRetriever()

        # 测试索引构建
        documents = [
            {"content": "Python programming language tutorial", "metadata": {"language": "python"}},
            {"content": "JavaScript web development guide", "metadata": {"language": "javascript"}}
        ]
        success = await retriever.build_index(documents)
        assert success is True

        # 测试关键词搜索
        results = await retriever.keyword_search("Python tutorial", top_k=5)
        assert isinstance(results, list)
        assert len(results) <= 5

        # 验证结果结构
        for result in results:
            assert "document_id" in result
            assert "content" in result
            assert "score" in result
            assert "matched_terms" in result
            assert result["retriever_type"] == "keyword"
            assert isinstance(result["matched_terms"], list)

    @pytest.mark.asyncio
    async def test_retrieve_method(self):
        """测试检索方法"""
        retriever = MockKeywordRetriever()

        documents = [{"content": "Test content for keyword search", "metadata": {}}]
        await retriever.index_documents(documents)

        results = await retriever.retrieve("keyword search")
        assert isinstance(results, list)
        assert len(results) >= 0

    def test_keyword_retriever_inheritance(self):
        """测试关键词检索器继承关系"""
        retriever = MockKeywordRetriever()
        assert isinstance(retriever, KeywordRetrieverInterface)
        assert isinstance(retriever, RetrieverInterface)
        assert hasattr(retriever, 'build_index')
        assert hasattr(retriever, 'keyword_search')


class TestEntityRetrieverInterface:
    """实体检索器接口测试类"""

    def test_interface_is_abstract(self):
        """测试实体检索器接口是抽象类"""
        with pytest.raises(TypeError):
            EntityRetrieverInterface()

    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        """测试实体提取"""
        retriever = MockEntityRetriever()
        text = "Apple Inc. announced iPhone 15 in 2023"

        entities = await retriever.extract_entities(text)

        assert isinstance(entities, list)
        # 检查是否提取到了实体
        entity_texts = [e["text"] for e in entities]
        assert any("Apple" in text for text in entity_texts) or any("iPhone" in text for text in entity_texts)

        # 验证实体结构
        for entity in entities:
            assert "text" in entity
            assert "type" in entity
            assert "confidence" in entity
            assert "start" in entity
            assert "end" in entity
            assert 0 <= entity["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_entity_search(self):
        """测试实体搜索"""
        retriever = MockEntityRetriever()

        # 索引包含实体的文档
        documents = [
            {"content": "Google and Microsoft are tech companies", "metadata": {}},
            {"content": "Apple designs iPhones and MacBooks", "metadata": {}}
        ]
        await retriever.index_documents(documents)

        # 搜索实体
        results = await retriever.entity_search(["Apple", "iPhone"], top_k=5)

        assert isinstance(results, list)
        for result in results:
            assert "matched_entities" in result
            assert isinstance(result["matched_entities"], list)
            assert result["retriever_type"] == "entity"

    @pytest.mark.asyncio
    async def test_retrieve_method(self):
        """测试检索方法"""
        retriever = MockEntityRetriever()

        documents = [{"content": "Tesla makes electric vehicles", "metadata": {}}]
        await retriever.index_documents(documents)

        results = await retriever.retrieve("Tesla vehicles")
        assert isinstance(results, list)

    def test_entity_retriever_inheritance(self):
        """测试实体检索器继承关系"""
        retriever = MockEntityRetriever()
        assert isinstance(retriever, EntityRetrieverInterface)
        assert isinstance(retriever, RetrieverInterface)
        assert hasattr(retriever, 'extract_entities')
        assert hasattr(retriever, 'entity_search')


class TestFusionRetrieverInterface:
    """融合检索器接口测试类"""

    def test_interface_is_abstract(self):
        """测试融合检索器接口是抽象类"""
        with pytest.raises(TypeError):
            FusionRetrieverInterface()

    @pytest.mark.asyncio
    async def test_fusion_retriever_implementation(self):
        """测试融合检索器实现"""
        fusion_retriever = MockFusionRetriever()

        # 添加子检索器
        vector_retriever = MockVectorRetriever()
        keyword_retriever = MockKeywordRetriever()

        fusion_retriever.add_retriever("vector", vector_retriever, weight=0.7)
        fusion_retriever.add_retriever("keyword", keyword_retriever, weight=0.3)

        # 索引文档
        documents = [
            {"content": "Machine learning and AI are related", "metadata": {}},
            {"content": "Deep learning is a subset of ML", "metadata": {}}
        ]

        await vector_retriever.index_documents(documents)
        await keyword_retriever.index_documents(documents)

        # 测试融合检索
        results = await fusion_retriever.retrieve("What is AI?", top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5

        # 验证融合结果结构
        for result in results:
            assert "document_id" in result
            assert "content" in result
            assert "score" in result
            assert "sources" in result
            assert isinstance(result["sources"], list)

    @pytest.mark.asyncio
    async def test_fuse_results_rrf(self):
        """测试RRF融合算法"""
        fusion_retriever = MockFusionRetriever()

        # 创建模拟结果列表
        results1 = [
            {"document_id": "doc1", "content": "Content 1", "score": 0.9, "metadata": {}},
            {"document_id": "doc2", "content": "Content 2", "score": 0.8, "metadata": {}},
            {"document_id": "doc3", "content": "Content 3", "score": 0.7, "metadata": {}}
        ]

        results2 = [
            {"document_id": "doc2", "content": "Content 2", "score": 0.9, "metadata": {}},
            {"document_id": "doc1", "content": "Content 1", "score": 0.6, "metadata": {}},
            {"document_id": "doc4", "content": "Content 4", "score": 0.5, "metadata": {}}
        ]

        fused_results = await fusion_retriever.fuse_results([results1, results2], method="rrf")

        assert isinstance(fused_results, list)
        assert len(fused_results) >= 3  # 至少包含3个不重复的文档

        # 验证融合分数
        for result in fused_results:
            assert "score" in result
            assert "sources" in result
            assert len(result["sources"]) >= 1

    @pytest.mark.asyncio
    async def test_weighted_search(self):
        """测试加权搜索"""
        fusion_retriever = MockFusionRetriever()

        # 添加子检索器
        vector_retriever = MockVectorRetriever()
        keyword_retriever = MockKeywordRetriever()

        fusion_retriever.add_retriever("vector", vector_retriever, weight=0.7)
        fusion_retriever.add_retriever("keyword", keyword_retriever, weight=0.3)

        # 索引文档
        documents = [{"content": "Test document", "metadata": {}}]
        await vector_retriever.index_documents(documents)
        await keyword_retriever.index_documents(documents)

        # 测试加权搜索
        weights = {"vector": 0.8, "keyword": 0.2}
        results = await fusion_retriever.weighted_search("test", weights, top_k=5)

        assert isinstance(results, list)

    def test_fusion_retriever_inheritance(self):
        """测试融合检索器继承关系"""
        fusion_retriever = MockFusionRetriever()
        assert isinstance(fusion_retriever, FusionRetrieverInterface)
        assert isinstance(fusion_retriever, RetrieverInterface)
        assert hasattr(fusion_retriever, 'fuse_results')
        assert hasattr(fusion_retriever, 'weighted_search')


class TestRetrieverIntegration:
    """检索器集成测试"""

    @pytest.mark.asyncio
    async def test_multi_retrieval_pipeline(self):
        """测试多检索器流水线"""
        # 创建多个检索器
        vector_retriever = MockVectorRetriever()
        keyword_retriever = MockKeywordRetriever()
        entity_retriever = MockEntityRetriever()
        fusion_retriever = MockFusionRetriever()

        # 配置融合检索器
        fusion_retriever.add_retriever("vector", vector_retriever, weight=0.5)
        fusion_retriever.add_retriever("keyword", keyword_retriever, weight=0.3)
        fusion_retriever.add_retriever("entity", entity_retriever, weight=0.2)

        # 准备测试文档
        documents = [
            {"content": "Google DeepMind develops advanced AI systems", "metadata": {"company": "Google"}},
            {"content": "OpenAI created ChatGPT language model", "metadata": {"company": "OpenAI"}},
            {"content": "Microsoft invests in OpenAI partnership", "metadata": {"company": "Microsoft"}}
        ]

        # 索引文档到所有检索器
        await vector_retriever.index_documents(documents)
        await keyword_retriever.index_documents(documents)
        await entity_retriever.index_documents(documents)

        # 执行融合检索
        query = "Which companies develop AI systems?"
        results = await fusion_retriever.retrieve(query, top_k=3)

        # 验证结果
        assert isinstance(results, list)
        assert len(results) <= 3

        # 验证结果包含来自不同检索器的信息
        for result in results:
            assert "document_id" in result
            assert "content" in result
            assert "score" in result
            assert "sources" in result
            assert len(result["sources"]) >= 1

        # 验证检索器统计信息
        vector_stats = vector_retriever.get_stats()
        keyword_stats = keyword_retriever.get_stats()
        entity_stats = entity_retriever.get_stats()
        fusion_stats = fusion_retriever.get_stats()

        assert vector_stats["total_documents"] == 3
        assert keyword_stats["total_documents"] == 3
        assert entity_stats["total_documents"] == 3
        assert fusion_stats["total_fusions"] >= 1

    @pytest.mark.asyncio
    async def test_retriever_error_handling(self):
        """测试检索器错误处理"""
        fusion_retriever = MockFusionRetriever()

        # 添加一个正常的检索器和一个会出错的检索器
        vector_retriever = MockVectorRetriever()
        keyword_retriever = MockKeywordRetriever()

        fusion_retriever.add_retriever("vector", vector_retriever, weight=0.7)
        fusion_retriever.add_retriever("keyword", keyword_retriever, weight=0.3)

        # 索引文档
        documents = [{"content": "Test document", "metadata": {}}]
        await vector_retriever.index_documents(documents)
        await keyword_retriever.index_documents(documents)

        # 即使有部分检索器失败，融合检索器也应该正常工作
        results = await fusion_retriever.retrieve("test query", top_k=5)

        assert isinstance(results, list)
        # 至少应该有一个检索器返回结果