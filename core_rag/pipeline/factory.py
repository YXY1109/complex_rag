"""
RAG流水线工厂类

提供不同类型RAG流水线的创建和配置。
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .rag_pipeline import RAGPipeline
from .interfaces.pipeline_interface import (
    PipelineConfig,
    GenerationStrategy,
    QueryType,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.pipeline.factory")


@dataclass
class PipelineTemplate:
    """流水线模板"""
    name: str
    description: str
    config: Dict[str, Any]
    use_cases: List[str]


class PipelineFactory:
    """RAG流水线工厂类"""

    # 预定义的流水线模板
    TEMPLATES = {
        "standard": PipelineTemplate(
            name="标准RAG流水线",
            description="平衡性能和准确性的标准RAG处理流水线",
            use_cases=["通用问答", "知识检索", "文档查询"],
            config={
                "enable_query_rewrite": True,
                "enable_query_expansion": True,
                "enable_intent_detection": True,
                "retrieval_strategies": ["vector", "bm25"],
                "max_retrieval_results": 15,
                "min_relevance_score": 0.3,
                "enable_reranking": True,
                "max_context_tokens": 4000,
                "enable_context_compression": True,
                "enable_context_ranking": True,
                "generation_model": "default",
                "temperature": 0.7,
                "max_tokens": 1000,
                "enable_citation_generation": True,
                "enable_source_attribution": True,
                "enable_caching": True,
                "enable_parallel_processing": True,
                "timeout_seconds": 30,
            }
        ),
        "fast": PipelineTemplate(
            name="快速RAG流水线",
            description="优化响应速度的快速RAG处理流水线",
            use_cases=["实时问答", "高频查询", "快速响应"],
            config={
                "enable_query_rewrite": False,
                "enable_query_expansion": False,
                "enable_intent_detection": True,
                "retrieval_strategies": ["vector"],
                "max_retrieval_results": 8,
                "min_relevance_score": 0.4,
                "enable_reranking": False,
                "max_context_tokens": 2000,
                "enable_context_compression": False,
                "enable_context_ranking": True,
                "generation_model": "fast",
                "temperature": 0.5,
                "max_tokens": 500,
                "enable_citation_generation": False,
                "enable_source_attribution": True,
                "enable_caching": True,
                "enable_parallel_processing": False,
                "timeout_seconds": 15,
            }
        ),
        "comprehensive": PipelineTemplate(
            name="全面RAG流水线",
            description="追求最高准确性的全面RAG处理流水线",
            use_cases=["专业问答", "研究分析", "详细解释"],
            config={
                "enable_query_rewrite": True,
                "enable_query_expansion": True,
                "enable_intent_detection": True,
                "retrieval_strategies": ["vector", "bm25", "graph"],
                "max_retrieval_results": 25,
                "min_relevance_score": 0.2,
                "enable_reranking": True,
                "max_context_tokens": 6000,
                "enable_context_compression": True,
                "enable_context_ranking": True,
                "generation_model": "advanced",
                "temperature": 0.8,
                "max_tokens": 2000,
                "enable_citation_generation": True,
                "enable_source_attribution": True,
                "enable_caching": True,
                "enable_parallel_processing": True,
                "timeout_seconds": 60,
            }
        ),
        "conversational": PipelineTemplate(
            name="对话式RAG流水线",
            description="优化对话体验的RAG处理流水线",
            use_cases=["客服对话", "聊天机器人", "交互式问答"],
            config={
                "enable_query_rewrite": True,
                "enable_query_expansion": True,
                "enable_intent_detection": True,
                "retrieval_strategies": ["vector", "bm25"],
                "max_retrieval_results": 12,
                "min_relevance_score": 0.3,
                "enable_reranking": True,
                "max_context_tokens": 3000,
                "enable_context_compression": True,
                "enable_context_ranking": True,
                "generation_model": "conversational",
                "temperature": 0.9,
                "max_tokens": 800,
                "enable_citation_generation": True,
                "enable_source_attribution": True,
                "enable_caching": True,
                "enable_parallel_processing": True,
                "timeout_seconds": 25,
            }
        ),
        "research": PipelineTemplate(
            name="研究型RAG流水线",
            description="适合学术研究和深度分析的RAG处理流水线",
            use_cases=["学术研究", "深度分析", "专业咨询"],
            config={
                "enable_query_rewrite": True,
                "enable_query_expansion": True,
                "enable_intent_detection": True,
                "retrieval_strategies": ["vector", "bm25", "graph"],
                "max_retrieval_results": 30,
                "min_relevance_score": 0.15,
                "enable_reranking": True,
                "max_context_tokens": 8000,
                "enable_context_compression": True,
                "enable_context_ranking": True,
                "generation_model": "research",
                "temperature": 0.6,
                "max_tokens": 3000,
                "enable_citation_generation": True,
                "enable_source_attribution": True,
                "enable_caching": True,
                "enable_parallel_processing": True,
                "timeout_seconds": 90,
            }
        ),
    }

    @classmethod
    def create_pipeline(
        cls,
        config: Optional[Dict[str, Any]] = None,
        llm_service=None,
        embedding_service=None,
        graph_service=None
    ) -> RAGPipeline:
        """
        创建RAG流水线

        Args:
            config: 流水线配置
            llm_service: LLM服务实例
            embedding_service: 向量化服务实例
            graph_service: 图服务实例

        Returns:
            RAGPipeline: 流水线实例
        """
        if config is None:
            config = cls.TEMPLATES["standard"].config.copy()

        try:
            structured_logger.info(
                "创建RAG流水线",
                extra={
                    "config_keys": list(config.keys()),
                    "has_llm_service": llm_service is not None,
                    "has_embedding_service": embedding_service is not None,
                    "has_graph_service": graph_service is not None,
                }
            )

            # 创建流水线实例
            pipeline = RAGPipeline(config)

            # 设置服务实例（如果提供）
            if hasattr(pipeline, 'answer_generator') and pipeline.answer_generator and llm_service:
                pipeline.answer_generator.llm_service = llm_service

            if hasattr(pipeline, 'retriever') and pipeline.retriever:
                # 为不同策略设置服务
                for strategy, retriever in pipeline.retriever.retrievers.items():
                    if strategy.value == "vector" and embedding_service:
                        if hasattr(retriever, 'embedding_service'):
                            retriever.embedding_service = embedding_service
                    elif strategy.value == "graph" and graph_service:
                        if hasattr(retriever, 'graph_service'):
                            retriever.graph_service = graph_service

            return pipeline

        except Exception as e:
            structured_logger.error(f"创建RAG流水线失败: {e}")
            raise Exception(f"Failed to create RAG pipeline: {e}")

    @classmethod
    def create_from_template(
        cls,
        template_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        **services
    ) -> RAGPipeline:
        """
        从模板创建RAG流水线

        Args:
            template_name: 模板名称
            config_overrides: 配置覆盖
            **services: 服务实例

        Returns:
            RAGPipeline: 流水线实例
        """
        if template_name not in cls.TEMPLATES:
            available_templates = list(cls.TEMPLATES.keys())
            raise ValueError(f"未知的模板名称: {template_name}。可用模板: {available_templates}")

        template = cls.TEMPLATES[template_name]
        config = template.config.copy()

        # 应用配置覆盖
        if config_overrides:
            cls._deep_update(config, config_overrides)

        structured_logger.info(
            f"从模板创建RAG流水线: {template_name}",
            extra={
                "template_description": template.description,
                "use_cases": template.use_cases,
            }
        )

        return cls.create_pipeline(config, **services)

    @classmethod
    def create_standard_pipeline(cls, **services) -> RAGPipeline:
        """创建标准RAG流水线"""
        return cls.create_from_template("standard", **services)

    @classmethod
    def create_fast_pipeline(cls, **services) -> RAGPipeline:
        """创建快速RAG流水线"""
        return cls.create_from_template("fast", **services)

    @classmethod
    def create_comprehensive_pipeline(cls, **services) -> RAGPipeline:
        """创建全面RAG流水线"""
        return cls.create_from_template("comprehensive", **services)

    @classmethod
    def create_conversational_pipeline(cls, **services) -> RAGPipeline:
        """创建对话式RAG流水线"""
        return cls.create_from_template("conversational", **services)

    @classmethod
    def create_research_pipeline(cls, **services) -> RAGPipeline:
        """创建研究型RAG流水线"""
        return cls.create_from_template("research", **services)

    @classmethod
    def create_custom_pipeline(
        cls,
        requirements: Dict[str, Any],
        **services
    ) -> RAGPipeline:
        """
        根据需求创建自定义RAG流水线

        Args:
            requirements: 需求配置
            **services: 服务实例

        Returns:
            RAGPipeline: 自定义流水线实例
        """
        # 分析需求并选择合适的基础配置
        base_config = cls._analyze_requirements(requirements)

        # 应用特定需求
        if "performance_priority" in requirements:
            if requirements["performance_priority"] == "speed":
                # 优化速度
                base_config.update({
                    "enable_query_rewrite": False,
                    "enable_query_expansion": False,
                    "retrieval_strategies": ["vector"],
                    "max_retrieval_results": 8,
                    "max_context_tokens": 2000,
                    "max_tokens": 500,
                })
            elif requirements["performance_priority"] == "accuracy":
                # 优化准确性
                base_config.update({
                    "enable_query_rewrite": True,
                    "enable_query_expansion": True,
                    "retrieval_strategies": ["vector", "bm25", "graph"],
                    "max_retrieval_results": 25,
                    "max_context_tokens": 6000,
                    "max_tokens": 2000,
                })

        if "domain" in requirements:
            domain = requirements["domain"]
            if domain == "medical":
                base_config.update({
                    "temperature": 0.3,  # 更保守
                    "max_tokens": 1500,
                    "enable_citation_generation": True,
                })
            elif domain == "creative":
                base_config.update({
                    "temperature": 0.9,  # 更有创造性
                    "max_tokens": 1200,
                    "enable_query_expansion": True,
                })
            elif domain == "technical":
                base_config.update({
                    "temperature": 0.5,
                    "max_tokens": 2000,
                    "enable_context_compression": False,  # 保持技术细节
                })

        if "constraints" in requirements:
            constraints = requirements["constraints"]
            if "max_latency_ms" in constraints:
                # 根据延迟约束调整配置
                max_latency = constraints["max_latency_ms"]
                if max_latency < 2000:  # 2秒内
                    base_config.update({
                        "enable_query_rewrite": False,
                        "max_retrieval_results": 5,
                        "max_context_tokens": 1500,
                        "max_tokens": 300,
                    })
                elif max_latency < 5000:  # 5秒内
                    base_config.update({
                        "max_retrieval_results": 10,
                        "max_context_tokens": 2500,
                        "max_tokens": 600,
                    })

            if "max_memory_mb" in constraints:
                # 根据内存约束调整配置
                max_memory = constraints["max_memory_mb"]
                if max_memory < 512:  # 低内存
                    base_config.update({
                        "enable_caching": False,
                        "max_context_tokens": 2000,
                        "max_retrieval_results": 8,
                    })

        structured_logger.info(
            "根据需求创建自定义RAG流水线",
            extra={
                "requirements": requirements,
                "selected_config": base_config,
            }
        )

        return cls.create_pipeline(base_config, **services)

    @classmethod
    def _analyze_requirements(cls, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """分析需求并生成基础配置"""
        # 默认使用标准配置
        base_config = cls.TEMPLATES["standard"].config.copy()

        # 根据使用场景调整
        use_cases = requirements.get("use_cases", [])
        if "real_time" in use_cases:
            base_config.update(cls.TEMPLATES["fast"].config)
        elif "research" in use_cases:
            base_config.update(cls.TEMPLATES["research"].config)
        elif "conversation" in use_cases:
            base_config.update(cls.TEMPLATES["conversational"].config)

        return base_config

    @classmethod
    def _deep_update(cls, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                cls._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    @classmethod
    def get_available_templates(cls) -> List[Dict[str, Any]]:
        """获取可用的流水线模板"""
        return [
            {
                "name": name,
                "description": template.description,
                "use_cases": template.use_cases,
            }
            for name, template in cls.TEMPLATES.items()
        ]

    @classmethod
    def get_template_info(cls, template_name: str) -> Optional[Dict[str, Any]]:
        """获取模板详细信息"""
        if template_name not in cls.TEMPLATES:
            return None

        template = cls.TEMPLATES[template_name]
        return {
            "name": template.name,
            "description": template.description,
            "use_cases": template.use_cases,
            "config": template.config,
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        验证流水线配置

        Args:
            config: 配置字典

        Returns:
            List[str]: 验证错误列表
        """
        errors = []

        # 验证必需字段
        required_fields = [
            "retrieval_strategies",
            "max_retrieval_results",
            "max_context_tokens",
            "generation_model",
            "temperature",
            "max_tokens",
        ]

        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")

        # 验证字段类型和值
        if "retrieval_strategies" in config:
            strategies = config["retrieval_strategies"]
            if not isinstance(strategies, list) or not strategies:
                errors.append("retrieval_strategies 必须是非空列表")

            valid_strategies = ["vector", "bm25", "graph"]
            for strategy in strategies:
                if strategy not in valid_strategies:
                    errors.append(f"无效的检索策略: {strategy}")

        numeric_fields = {
            "max_retrieval_results": (1, 100),
            "min_relevance_score": (0.0, 1.0),
            "max_context_tokens": (100, 20000),
            "temperature": (0.0, 2.0),
            "max_tokens": (50, 10000),
            "timeout_seconds": (5, 300),
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} 必须是数字")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{field} 必须在 [{min_val}, {max_val}] 范围内")

        boolean_fields = [
            "enable_query_rewrite",
            "enable_query_expansion",
            "enable_intent_detection",
            "enable_reranking",
            "enable_context_compression",
            "enable_context_ranking",
            "enable_citation_generation",
            "enable_source_attribution",
            "enable_caching",
            "enable_parallel_processing",
        ]

        for field in boolean_fields:
            if field in config and not isinstance(config[field], bool):
                errors.append(f"{field} 必须是布尔值")

        return errors

    @classmethod
    def get_optimization_suggestions(cls, config: Dict[str, Any]) -> List[str]:
        """
        获取优化建议

        Args:
            config: 当前配置

        Returns:
            List[str]: 优化建议列表
        """
        suggestions = []

        # 性能优化建议
        if config.get("max_retrieval_results", 15) > 20:
            suggestions.append("考虑减少max_retrieval_results以提高检索速度")

        if config.get("max_context_tokens", 4000) > 6000:
            suggestions.append("大量上下文可能影响生成速度和答案质量")

        if not config.get("enable_caching", True):
            suggestions.append("启用缓存可以显著提高重复查询的性能")

        if config.get("temperature", 0.7) > 1.0:
            suggestions.append("高温度可能导致答案不稳定，建议降低到0.7-0.9之间")

        # 质量优化建议
        if not config.get("enable_query_rewrite", True):
            suggestions.append("启用查询重写可以提高查询理解质量")

        if not config.get("enable_context_compression", True):
            suggestions.append("启用上下文压缩可以提高信息密度")

        if len(config.get("retrieval_strategies", [])) == 1:
            suggestions.append("考虑使用多种检索策略以提高召回率")

        # 成本优化建议
        if config.get("max_tokens", 1000) > 2000:
            suggestions.append("减少max_tokens可以降低API调用成本")

        if config.get("enable_citation_generation", True):
            suggestions.append("如果不需要引用，可以禁用cite_generation以降低复杂度")

        return suggestions