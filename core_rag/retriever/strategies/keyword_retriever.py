"""
关键词检索器

基于关键词匹配的文档检索实现。
"""

import asyncio
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from ..interfaces.retriever_interface import (
    RetrieverInterface,
    RetrievalQuery,
    RetrievalResult,
    DocumentChunk,
    RetrievalStrategy,
    RetrieverConfig,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.retriever.strategies.keyword_retriever")


@dataclass
class KeywordIndexConfig:
    """关键词索引配置"""
    enable_stemming: bool = True
    enable_stopwords: bool = True
    min_term_length: int = 2
    max_term_length: int = 50
    enable_phrase_search: bool = True
    enable_fuzzy_search: bool = True
    fuzzy_threshold: float = 0.8


class BM25Retriever(RetrieverInterface):
    """
    BM25关键词检索器

    基于BM25算法的关键词检索，支持多种文本预处理和匹配策略。
    """

    def __init__(self, config: RetrieverConfig):
        """
        初始化BM25检索器

        Args:
            config: 检索器配置
        """
        self.config = config
        self.index_config = KeywordIndexConfig()

        # BM25参数
        self.k1 = 1.2  # 控制词频饱和度
        self.b = 0.75  # 控制文档长度归一化程度

        # 索引数据结构
        self.documents: Dict[str, DocumentChunk] = {}
        self.document_lengths: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # term -> [(doc_id, freq)]
        self.document_frequencies: Dict[str, int] = {}  # term -> doc_count
        self.total_documents = 0
        self.avg_document_length = 0.0

        # 文本预处理
        self.stopwords = self._load_stopwords()
        self.stemmer = None

        # 缓存
        self._query_cache = {} if config.enable_caching else None
        self._cache_timestamps = {}

        self._initialized = False

    def _load_stopwords(self) -> Set[str]:
        """加载停用词列表"""
        # 常见英文停用词
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
            "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
            "that", "the", "their", "then", "there", "these", "they", "this",
            "to", "was", "will", "with", "about", "after", "all", "also", "an",
            "any", "can", "do", "has", "had", "how", "his", "her", "its", "may",
            "our", "she", "should", "so", "than", "them", "upon", "us", "was",
            "we", "what", "when", "where", "which", "who", "why", "your",
        }

        # 常见中文停用词
        chinese_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
            "看", "好", "自己", "这", "那", "里", "就是", "还", "把", "比", "或者",
            "什么", "可以", "这个", "那个", "这样", "那样", "因为", "所以", "但是",
        }

        stopwords.update(chinese_stopwords)
        return stopwords

    async def initialize(self) -> bool:
        """
        初始化BM25检索器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化BM25检索器",
                extra={
                    "k1": self.k1,
                    "b": self.b,
                    "enable_stemming": self.index_config.enable_stemming,
                    "enable_stopwords": self.index_config.enable_stopwords,
                }
            )

            # 初始化词干提取器（如果启用）
            if self.index_config.enable_stemming:
                try:
                    # 尝试导入NLTK
                    import nltk
                    from nltk.stem import PorterStemmer
                    self.stemmer = PorterStemmer()
                    structured_logger.info("使用NLTK词干提取器")
                except ImportError:
                    # 简单的词干提取实现
                    self.stemmer = SimpleStemmer()
                    structured_logger.info("使用简单词干提取器")

            self._initialized = True
            structured_logger.info("BM25检索器初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"BM25检索器初始化失败: {e}")
            return False

    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        添加文档到BM25检索器

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档ID列表
        """
        if not self._initialized:
            raise RuntimeError("BM25检索器未初始化")

        document_ids = []

        try:
            structured_logger.info(f"开始添加 {len(documents)} 个文档到BM25检索器")

            # 处理每个文档
            for doc in documents:
                doc_id = doc.get("id") or f"doc_{len(self.documents)}_{int(time.time())}"
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                # 创建文档片段
                chunk = DocumentChunk(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    source=doc.get("source"),
                    chunk_index=doc.get("chunk_index", 0),
                    start_pos=doc.get("start_pos", 0),
                    end_pos=doc.get("end_pos", len(content)),
                    created_at=datetime.utcnow().isoformat(),
                )

                # 文本预处理和分词
                tokens = self._preprocess_text(content)
                document_length = len(tokens)

                # 存储文档
                self.documents[doc_id] = chunk
                self.document_lengths[doc_id] = document_length
                document_ids.append(doc_id)

                # 构建倒排索引
                term_counts = Counter(tokens)
                for term, count in term_counts.items():
                    self.inverted_index[term].append((doc_id, count))

            # 更新统计信息
            self._update_statistics()

            structured_logger.info(f"成功添加 {len(document_ids)} 个文档到BM25检索器")
            return document_ids

        except Exception as e:
            structured_logger.error(f"添加文档失败: {e}")
            raise Exception(f"Failed to add documents: {e}")

    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理和分词"""
        # 转换为小写
        text = text.lower()

        # 提取单词和中文
        # 匹配英文单词和中文汉字
        tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+', text)

        processed_tokens = []

        for token in tokens:
            # 过滤长度
            if len(token) < self.index_config.min_term_length:
                continue
            if len(token) > self.index_config.max_term_length:
                continue

            # 移除停用词
            if self.index_config.enable_stopwords and token in self.stopwords:
                continue

            # 词干提取
            if self.index_config.enable_stemming and self.stemmer:
                if re.match(r'^[a-zA-Z]+$', token):  # 只对英文单词进行词干提取
                    token = self.stemmer.stem(token)

            processed_tokens.append(token)

        return processed_tokens

    def _update_statistics(self) -> None:
        """更新统计信息"""
        self.total_documents = len(self.documents)

        if self.total_documents > 0:
            total_length = sum(self.document_lengths.values())
            self.avg_document_length = total_length / self.total_documents

        # 更新文档频率
        self.document_frequencies.clear()
        for term, postings in self.inverted_index.items():
            self.document_frequencies[term] = len(postings)

        structured_logger.debug(
            f"更新统计信息完成: 文档数={self.total_documents}, "
            f"平均长度={self.avg_document_length:.2f}, 词汇数={len(self.inverted_index)}"
        )

    async def retrieve(
        self,
        query: RetrievalQuery
    ) -> RetrievalResult:
        """
        执行BM25检索

        Args:
            query: 检索查询

        Returns:
            RetrievalResult: 检索结果
        """
        if not self._initialized:
            raise RuntimeError("BM25检索器未初始化")

        start_time = time.time()

        try:
            # 检查缓存
            cache_key = f"{query.text}_{query.top_k}_{query.min_score}"
            if self._query_cache and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                structured_logger.debug(f"使用缓存结果: {query.text[:50]}...")
                return cached_result

            structured_logger.info(
                f"开始BM25检索",
                extra={
                    "query_length": len(query.text),
                    "top_k": query.top_k,
                    "min_score": query.min_score,
                }
            )

            # 查询预处理
            query_tokens = self._preprocess_text(query.text)
            if not query_tokens:
                return self._create_empty_result(query, start_time)

            # 计算BM25分数
            doc_scores = self._calculate_bm25_scores(query_tokens)

            # 排序和过滤
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

            # 构建结果
            chunks = []
            chunk_scores = []
            explanations = []

            for doc_id, score in sorted_docs:
                if score <= query.min_score:
                    break

                if doc_id in self.documents:
                    chunk = self.documents[doc_id]
                    chunk.score = score
                    chunks.append(chunk)
                    chunk_scores.append(score)

                    # 生成解释
                    matched_terms = self._get_matched_terms(doc_id, query_tokens)
                    explanation = f"BM25分数: {score:.3f}, 匹配词: {', '.join(matched_terms[:5])}"
                    explanations.append(explanation)

                if len(chunks) >= query.max_results:
                    break

            processing_time = (time.time() - start_time) * 1000

            result = RetrievalResult(
                chunks=chunks,
                query=query.text,
                strategy=RetrievalStrategy.BM25,
                total_found=len(chunks),
                search_time_ms=processing_time,
                scores=chunk_scores,
                explanations=explanations,
                metadata={
                    "total_documents": self.total_documents,
                    "query_tokens": query_tokens,
                    "avg_document_length": self.avg_document_length,
                    "vocabulary_size": len(self.inverted_index),
                },
                created_at=datetime.utcnow().isoformat(),
            )

            # 缓存结果
            if self._query_cache:
                self._query_cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()

            structured_logger.info(
                f"BM25检索完成",
                extra={
                    "results_count": len(chunks),
                    "processing_time_ms": processing_time,
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"BM25检索失败: {e}")
            raise Exception(f"BM25 retrieval failed: {e}")

    def _calculate_bm25_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        """计算BM25分数"""
        doc_scores = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            # 词项在文档中的频率
            postings = self.inverted_index[term]
            df = self.document_frequencies[term]  # 文档频率
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))

            for doc_id, tf in postings:
                # BM25公式
                doc_length = self.document_lengths[doc_id]
                normalized_tf = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_document_length)
                )

                score = idf * normalized_tf
                doc_scores[doc_id] += score

        return dict(doc_scores)

    def _get_matched_terms(self, doc_id: str, query_tokens: List[str]) -> List[str]:
        """获取文档中匹配的查询词"""
        matched_terms = []
        doc_content = self.documents[doc_id].content.lower()
        doc_tokens = self._preprocess_text(doc_content)
        doc_token_set = set(doc_tokens)

        for term in query_tokens:
            if term in doc_token_set:
                matched_terms.append(term)

        return matched_terms

    def _create_empty_result(self, query: RetrievalQuery, start_time: float) -> RetrievalResult:
        """创建空结果"""
        processing_time = (time.time() - start_time) * 1000
        return RetrievalResult(
            chunks=[],
            query=query.text,
            strategy=RetrievalStrategy.BM25,
            total_found=0,
            search_time_ms=processing_time,
            created_at=datetime.utcnow().isoformat(),
        )

    async def batch_retrieve(
        self,
        queries: List[RetrievalQuery]
    ) -> List[RetrievalResult]:
        """批量BM25检索"""
        if not self._initialized:
            raise RuntimeError("BM25检索器未初始化")

        try:
            structured_logger.info(f"开始批量BM25检索，查询数量: {len(queries)}")

            # 并行处理查询
            tasks = [self.retrieve(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    structured_logger.error(f"查询 {i} 处理失败: {result}")
                    valid_results.append(self._create_empty_result(queries[i], time.time()))
                else:
                    valid_results.append(result)

            structured_logger.info(f"批量BM25检索完成，成功处理 {len(valid_results)} 个查询")
            return valid_results

        except Exception as e:
            structured_logger.error(f"批量BM25检索失败: {e}")
            raise Exception(f"Batch BM25 retrieval failed: {e}")

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        try:
            deleted_count = 0

            for doc_id in document_ids:
                if doc_id in self.documents:
                    # 从文档集合中删除
                    del self.documents[doc_id]
                    del self.document_lengths[doc_id]
                    deleted_count += 1

                    # 从倒排索引中删除
                    terms_to_remove = []
                    for term, postings in self.inverted_index.items():
                        self.inverted_index[term] = [
                            (doc, freq) for doc, freq in postings
                            if doc != doc_id
                        ]
                        if not self.inverted_index[term]:
                            terms_to_remove.append(term)

                    for term in terms_to_remove:
                        del self.inverted_index[term]

            if deleted_count > 0:
                self._update_statistics()

            structured_logger.info(f"删除了 {deleted_count} 个文档")
            return True

        except Exception as e:
            structured_logger.error(f"删除文档失败: {e}")
            return False

    async def update_document(self, document_id: str, document: Dict[str, Any]) -> bool:
        """更新文档"""
        try:
            if document_id not in self.documents:
                return False

            # 先删除旧文档
            await self.delete_documents([document_id])

            # 添加新文档
            document["id"] = document_id
            await self.add_documents([document])

            return True

        except Exception as e:
            structured_logger.error(f"更新文档失败: {e}")
            return False

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        if document_id in self.documents:
            chunk = self.documents[document_id]
            return {
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "score": chunk.score,
                "created_at": chunk.created_at,
            }
        return None

    async def search_similar(self, document_id: str, top_k: int = 10) -> List[DocumentChunk]:
        """搜索相似文档"""
        if document_id not in self.documents:
            return []

        try:
            # 获取文档内容作为查询
            doc_content = self.documents[document_id].content

            # 创建查询
            query = RetrievalQuery(text=doc_content, top_k=top_k + 1)  # +1 因为包含自己

            # 执行检索
            result = await self.retrieve(query)

            # 排除自己
            similar_chunks = [
                chunk for chunk in result.chunks
                if chunk.id != document_id
            ]

            return similar_chunks[:top_k]

        except Exception as e:
            structured_logger.error(f"搜索相似文档失败: {e}")
            return []

    async def expand_query(self, query: str, max_terms: int = 5) -> List[str]:
        """查询扩展"""
        expanded_terms = [query]

        # 基于词频的扩展
        query_tokens = self._preprocess_text(query)

        # 找到包含查询词的文档，提取高频词
        related_terms = defaultdict(int)
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_id, _ in self.inverted_index[token]:
                    if doc_id in self.documents:
                        doc_tokens = self._preprocess_text(self.documents[doc_id].content)
                        for doc_token in doc_tokens:
                            if doc_token not in query_tokens:
                                related_terms[doc_token] += 1

        # 选择最相关的词
        top_related = sorted(related_terms.items(), key=lambda x: x[1], reverse=True)
        for term, count in top_related[:max_terms - 1]:
            expanded_terms.append(term)

        return expanded_terms[:max_terms]

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": self.total_documents,
            "vocabulary_size": len(self.inverted_index),
            "avg_document_length": self.avg_document_length,
            "cache_size": len(self._query_cache) if self._query_cache else 0,
            "k1_parameter": self.k1,
            "b_parameter": self.b,
            "stemming_enabled": self.index_config.enable_stemming,
            "stopwords_enabled": self.index_config.enable_stopwords,
            "initialized": self._initialized,
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 执行简单检索测试
            test_query = RetrievalQuery(text="test", top_k=1)
            test_result = await self.retrieve(test_query)

            return {
                "status": "healthy",
                "initialized": self._initialized,
                "total_documents": self.total_documents,
                "vocabulary_size": len(self.inverted_index),
                "cache_enabled": self._query_cache is not None,
                "test_retrieval_time_ms": test_result.search_time_ms,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "initialized": self._initialized,
            }

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            self.documents.clear()
            self.document_lengths.clear()
            self.inverted_index.clear()
            self.document_frequencies.clear()

            if self._query_cache:
                self._query_cache.clear()
            self._cache_timestamps.clear()

            self.total_documents = 0
            self.avg_document_length = 0.0
            self._initialized = False

            structured_logger.info("BM25检索器清理完成")

        except Exception as e:
            structured_logger.error(f"BM25检索器清理失败: {e}")


class SimpleStemmer:
    """简单词干提取器"""

    def stem(self, word: str) -> str:
        """简单的词干提取"""
        word = word.lower()

        # 简单的词尾规则
        suffixes = ['ing', 'ed', 'ly', 'es', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break

        return word