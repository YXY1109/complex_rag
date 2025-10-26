"""
查询理解和预处理阶段

实现查询分析、重写、扩展等功能。
"""

import asyncio
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import asyncio

from ..interfaces.pipeline_interface import (
    QueryUnderstanding,
    QueryType,
    QueryRequest,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.pipeline.stages.query_understanding")


class QueryUnderstandingEngine:
    """
    查询理解引擎

    负责查询分析、意图识别、实体提取、查询重写和扩展。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化查询理解引擎

        Args:
            config: 配置参数
        """
        self.config = config
        self.enable_query_rewrite = config.get("enable_query_rewrite", True)
        self.enable_query_expansion = config.get("enable_query_expansion", True)
        self.enable_intent_detection = config.get("enable_intent_detection", True)
        self.max_rewrite_suggestions = config.get("max_rewrite_suggestions", 3)

        # 查询类型识别规则
        self.query_type_patterns = self._load_query_type_patterns()

        # 常见查询模式
        self.query_patterns = {
            "what": r"\b(what|什么是|什么)\b",
            "how": r"\b(how|如何|怎样|怎么)\b",
            "why": r"\b(why|为什么|为何)\b",
            "when": r"\b(when|何时|什么时候)\b",
            "where": r"\b(where|哪里|何处)\b",
            "who": r"\b(who|谁|什么人)\b",
            "compare": r"\b(compare|对比|比较|区别|差异)\b",
            "list": r"\b(list|列举|列出|总结)\b",
            "explain": r"\b(explain|解释|说明)\b",
            "define": r"\b(define|定义)\b",
        }

        # 停用词
        self.stopwords = self._load_stopwords()

        # 同义词词典（简化版）
        self.synonyms = self._load_synonyms()

        self._initialized = False

    def _load_query_type_patterns(self) -> Dict[QueryType, List[str]]:
        """加载查询类型模式"""
        return {
            QueryType.FACTUAL: [
                r"\b(what|who|when|where|什么|谁|何时|哪里)\b",
                r"\b(is|are|是|为)\b",
                r"\b(define|定义)\b",
            ],
            QueryType.PROCEDURAL: [
                r"\b(how|如何|怎样|怎么)\b",
                r"\b(to|步骤|方法|流程)\b",
                r"\b(process|过程)\b",
            ],
            QueryType.EXPLANATORY: [
                r"\b(why|为什么|为何)\b",
                r"\b(explain|解释|说明)\b",
                r"\b(reason|原因|理由)\b",
            ],
            QueryType.COMPARATIVE: [
                r"\b(compare|对比|比较|区别|差异)\b",
                r"\b(vs|versus|和|对比)\b",
                r"\b(better|更好|优势|劣势)\b",
            ],
            QueryType.CREATIVE: [
                r"\b(create|创建|设计|想象)\b",
                r"\b(invent|发明)\b",
                r"\b(suggest|建议)\b",
            ],
            QueryType.CONVERSATIONAL: [
                r"\b(hello|hi|你好|嗨)\b",
                r"\b(thanks|谢谢|感谢)\b",
                r"\b(goodbye|bye|再见)\b",
            ],
        }

    def _load_stopwords(self) -> Set[str]:
        """加载停用词"""
        return {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "自己", "这", "那", "里", "就是", "还", "把", "比", "或者", "什么", "可以",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "up", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "among", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
        }

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """加载同义词词典"""
        return {
            "AI": ["artificial intelligence", "人工智能", "机器智能"],
            "机器学习": ["machine learning", "ML"],
            "深度学习": ["deep learning", "DL", "神经网络"],
            "自然语言处理": ["natural language processing", "NLP"],
            "计算机视觉": ["computer vision", "CV"],
            "数据科学": ["data science", "DS"],
            "算法": ["algorithm", "algo"],
            "模型": ["model", "网络"],
            "训练": ["training", "学习"],
            "预测": ["prediction", "forecast"],
        }

    async def initialize(self) -> bool:
        """
        初始化查询理解引擎

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化查询理解引擎",
                extra={
                    "enable_query_rewrite": self.enable_query_rewrite,
                    "enable_query_expansion": self.enable_query_expansion,
                    "enable_intent_detection": self.enable_intent_detection,
                }
            )

            # 这里可以加载更多资源和模型
            # 例如：预训练的意图分类模型、实体识别模型等

            self._initialized = True
            structured_logger.info("查询理解引擎初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"查询理解引擎初始化失败: {e}")
            return False

    async def understand_query(
        self,
        request: QueryRequest
    ) -> QueryUnderstanding:
        """
        理解查询

        Args:
            request: 查询请求

        Returns:
            QueryUnderstanding: 查询理解结果
        """
        if not self._initialized:
            raise RuntimeError("查询理解引擎未初始化")

        start_time = time.time()

        try:
            structured_logger.info(
                "开始查询理解",
                extra={
                    "query_length": len(request.query),
                    "has_context": request.context is not None,
                    "has_history": len(request.conversation_history or []),
                }
            )

            # 预处理查询
            processed_query = self._preprocess_query(request.query)

            # 检测查询类型
            query_type = await self._detect_query_type(processed_query)

            # 提取查询意图
            query_intent = await self._extract_query_intent(processed_query, query_type)

            # 提取关键实体
            key_entities = await self._extract_key_entities(processed_query)

            # 提取关键概念
            key_concepts = await self._extract_key_concepts(processed_query)

            # 查询重写建议
            rewrite_suggestions = []
            if self.enable_query_rewrite:
                rewrite_suggestions = await self._generate_rewrite_suggestions(
                    processed_query, query_type, key_entities
                )

            # 查询扩展
            expansion_terms = []
            if self.enable_query_expansion:
                expansion_terms = await self._expand_query(processed_query, key_entities, key_concepts)

            # 计算置信度
            confidence = self._calculate_confidence(
                processed_query, query_type, key_entities, key_concepts
            )

            processing_time = (time.time() - start_time) * 1000

            result = QueryUnderstanding(
                original_query=request.query,
                processed_query=processed_query,
                query_type=query_type,
                query_intent=query_intent,
                key_entities=key_entities,
                key_concepts=key_concepts,
                query_rewrite_suggestions=rewrite_suggestions[:self.max_rewrite_suggestions],
                expansion_terms=expansion_terms,
                confidence=confidence,
                processing_time_ms=processing_time,
                metadata={
                    "preprocessing_steps": ["lowercase", "punctuation_removal", "normalization"],
                    "detection_methods": ["pattern_matching", "keyword_analysis"],
                },
            )

            structured_logger.info(
                "查询理解完成",
                extra={
                    "query_type": query_type.value,
                    "key_entities_count": len(key_entities),
                    "key_concepts_count": len(key_concepts),
                    "rewrite_suggestions_count": len(rewrite_suggestions),
                    "expansion_terms_count": len(expansion_terms),
                    "confidence": confidence,
                    "processing_time_ms": processing_time,
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"查询理解失败: {e}")
            raise Exception(f"Query understanding failed: {e}")

    def _preprocess_query(self, query: str) -> str:
        """预处理查询"""
        # 转换为小写
        query = query.lower()

        # 移除多余空格
        query = re.sub(r'\s+', ' ', query.strip())

        # 移除标点符号（保留必要的）
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)

        # 移除停用词
        words = query.split()
        words = [word for word in words if word not in self.stopwords]

        return ' '.join(words)

    async def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        if not self.enable_intent_detection:
            return QueryType.FACTUAL

        type_scores = {}

        # 基于模式匹配打分
        for query_type, patterns in self.query_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                score += len(matches)
            type_scores[query_type] = score

        # 基于关键词模式打分
        for pattern_name, pattern in self.query_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if pattern_name in ["what", "who", "when", "where", "define"]:
                    type_scores[QueryType.FACTUAL] = type_scores.get(QueryType.FACTUAL, 0) + len(matches)
                elif pattern_name == "how":
                    type_scores[QueryType.PROCEDURAL] = type_scores.get(QueryType.PROCEDURAL, 0) + len(matches)
                elif pattern_name == "why":
                    type_scores[QueryType.EXPLANATORY] = type_scores.get(QueryType.EXPLANATORY, 0) + len(matches)
                elif pattern_name == "compare":
                    type_scores[QueryType.COMPARATIVE] = type_scores.get(QueryType.COMPARATIVE, 0) + len(matches)
                elif pattern_name in ["create", "invent", "suggest"]:
                    type_scores[QueryType.CREATIVE] = type_scores.get(QueryType.CREATIVE, 0) + len(matches)
                elif pattern_name in ["hello", "thanks", "goodbye"]:
                    type_scores[QueryType.CONVERSATIONAL] = type_scores.get(QueryType.CONVERSATIONAL, 0) + len(matches)

        # 选择得分最高的类型
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type

        # 默认返回事实查询
        return QueryType.FACTUAL

    async def _extract_query_intent(self, query: str, query_type: QueryType) -> str:
        """提取查询意图"""
        # 基于查询类型生成意图描述
        intent_descriptions = {
            QueryType.FACTUAL: "寻求事实信息或定义",
            QueryType.PROCEDURAL: "寻求操作步骤或方法",
            QueryType.EXPLANATORY: "寻求解释或原因",
            QueryType.COMPARATIVE: "寻求对比或差异信息",
            QueryType.CREATIVE: "寻求创意或建议",
            QueryType.CONVERSATIONAL: "进行对话交流",
        }

        base_intent = intent_descriptions.get(query_type, "未知查询意图")

        # 基于查询内容细化意图
        if "如何" in query or "怎么" in query:
            return f"寻求操作方法：{base_intent}"
        elif "为什么" in query or "为何" in query:
            return f"寻求原因解释：{base_intent}"
        elif "对比" in query or "区别" in query:
            return f"寻求对比分析：{base_intent}"
        else:
            return base_intent

    async def _extract_key_entities(self, query: str) -> List[str]:
        """提取关键实体"""
        # 简单的实体提取：基于大写字母和常见词汇
        entities = []

        # 提取大写开头的词组（可能是实体名）
        entity_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(entity_pattern, query)
        entities.extend(matches)

        # 提取常见技术词汇
        tech_terms = [
            "AI", "ML", "DL", "NLP", "CV", "API", "SQL", "JSON", "XML", "HTML",
            "Python", "Java", "JavaScript", "React", "Vue", "Angular", "Django",
            "Flask", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas",
            "NumPy", "Matplotlib", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
        ]

        for term in tech_terms:
            if term.lower() in query.lower():
                entities.append(term)

        # 去重并过滤
        entities = list(set(entities))
        entities = [entity for entity in entities if len(entity) > 1]

        return entities

    async def _extract_key_concepts(self, query: str) -> List[str]:
        """提取关键概念"""
        # 基于词频提取关键词
        words = query.split()
        word_freq = Counter(words)

        # 过滤停用词和短词
        key_concepts = [
            word for word, freq in word_freq.items()
            if len(word) > 2 and word not in self.stopwords and freq >= 1
        ]

        # 限制数量
        return key_concepts[:10]

    async def _generate_rewrite_suggestions(
        self,
        query: str,
        query_type: QueryType,
        key_entities: List[str]
    ) -> List[str]:
        """生成查询重写建议"""
        suggestions = []

        # 基于查询类型的重写
        if query_type == QueryType.FACTUAL:
            # 添加疑问词
            if not any(word in query for word in ["what", "什么", "什么是"]):
                suggestions.append(f"什么是 {query}")
                suggestions.append(f"{query} 是什么")

        elif query_type == QueryType.PROCEDURAL:
            # 添加操作词
            if not any(word in query for word in ["how", "如何", "怎么"]):
                suggestions.append(f"如何 {query}")
                suggestions.append(f"{query} 的步骤")

        elif query_type == QueryType.COMPARATIVE:
            # 确保有对比词
            if "对比" not in query and "比较" not in query:
                suggestions.append(f"对比 {query}")

        # 基于实体的重写
        for entity in key_entities[:3]:  # 限制数量
            suggestions.append(f"{entity} {query}")

        # 去重并限制数量
        suggestions = list(set(suggestions))
        return suggestions[:self.max_rewrite_suggestions]

    async def _expand_query(
        self,
        query: str,
        key_entities: List[str],
        key_concepts: List[str]
    ) -> List[str]:
        """查询扩展"""
        expansion_terms = []

        # 基于同义词扩展
        for concept in key_concepts:
            if concept in self.synonyms:
                expansion_terms.extend(self.synonyms[concept])

        # 基于实体扩展
        for entity in key_entities:
            if entity in self.synonyms:
                expansion_terms.extend(self.synonyms[entity])

        # 基于查询内容的扩展
        if "AI" in query or "人工智能" in query:
            expansion_terms.extend(["机器学习", "深度学习", "神经网络"])
        elif "机器学习" in query:
            expansion_terms.extend(["AI", "人工智能", "算法", "模型"])

        # 去重并限制数量
        expansion_terms = list(set(expansion_terms))
        return [term for term in expansion_terms if term.lower() not in query.lower()][:10]

    def _calculate_confidence(
        self,
        query: str,
        query_type: QueryType,
        key_entities: List[str],
        key_concepts: List[str]
    ) -> float:
        """计算置信度"""
        confidence = 0.5  # 基础置信度

        # 基于查询长度
        if len(query.split()) >= 3:
            confidence += 0.1

        # 基于实体数量
        confidence += min(0.2, len(key_entities) * 0.05)

        # 基于概念数量
        confidence += min(0.2, len(key_concepts) * 0.02)

        # 基于查询类型匹配度
        if query_type != QueryType.FACTUAL:  # 非默认类型
            confidence += 0.1

        return min(1.0, confidence)

    async def batch_understand_queries(
        self,
        requests: List[QueryRequest]
    ) -> List[QueryUnderstanding]:
        """批量查询理解"""
        if not self._initialized:
            raise RuntimeError("查询理解引擎未初始化")

        try:
            structured_logger.info(f"开始批量查询理解，请求数量: {len(requests)}")

            # 并行处理
            tasks = [self.understand_query(request) for request in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    structured_logger.error(f"查询 {i} 理解失败: {result}")
                    # 创建默认结果
                    valid_results.append(QueryUnderstanding(
                        original_query=requests[i].query,
                        processed_query=requests[i].query,
                        query_type=QueryType.FACTUAL,
                        query_intent="理解失败",
                        confidence=0.0,
                        processing_time_ms=0.0,
                    ))
                else:
                    valid_results.append(result)

            structured_logger.info(f"批量查询理解完成，成功处理 {len(valid_results)} 个查询")
            return valid_results

        except Exception as e:
            structured_logger.error(f"批量查询理解失败: {e}")
            raise Exception(f"Batch query understanding failed: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "initialized": self._initialized,
            "enable_query_rewrite": self.enable_query_rewrite,
            "enable_query_expansion": self.enable_query_expansion,
            "enable_intent_detection": self.enable_intent_detection,
            "max_rewrite_suggestions": self.max_rewrite_suggestions,
            "synonyms_count": len(self.synonyms),
            "stopwords_count": len(self.stopwords),
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 执行简单测试
            test_request = QueryRequest(query="什么是人工智能？")
            result = await self.understand_query(test_request)

            return {
                "status": "healthy",
                "initialized": self._initialized,
                "test_processing_time_ms": result.processing_time_ms,
                "test_confidence": result.confidence,
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
            self._initialized = False
            structured_logger.info("查询理解引擎清理完成")

        except Exception as e:
            structured_logger.error(f"查询理解引擎清理失败: {e}")