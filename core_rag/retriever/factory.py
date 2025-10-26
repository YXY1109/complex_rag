"""
检索器工厂类

提供不同类型检索器的创建和配置。
"""

from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass

from .interfaces.retriever_interface import (
    RetrieverInterface,
    MultiStrategyRetrieverInterface,
    RetrievalStrategy,
    RetrieverConfig,
)
from .strategies.vector_retriever import VectorRetriever
from .strategies.keyword_retriever import BM25Retriever
from .strategies.graph_retriever import GraphRetriever
from .multi_strategy_retriever import MultiStrategyRetriever
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.retriever.factory")


@dataclass
class RetrieverRegistration:
    """检索器注册信息"""
    strategy: RetrievalStrategy
    retriever_class: Type[RetrieverInterface]
    default_config: Dict[str, Any]
    description: str


class RetrieverFactory:
    """检索器工厂类"""

    # 注册的检索器类型
    _registered_retrievers: Dict[RetrievalStrategy, RetrieverRegistration] = {
        RetrievalStrategy.VECTOR: RetrieverRegistration(
            strategy=RetrievalStrategy.VECTOR,
            retriever_class=VectorRetriever,
            default_config={
                "strategy": "vector",
                "mode": "single",
                "top_k": 10,
                "min_score": 0.0,
                "max_results": 50,
                "enable_caching": True,
                "cache_ttl": 3600,
            },
            description="向量检索器，基于向量相似度进行文档检索"
        ),
        RetrievalStrategy.BM25: RetrieverRegistration(
            strategy=RetrievalStrategy.BM25,
            retriever_class=BM25Retriever,
            default_config={
                "strategy": "bm25",
                "mode": "single",
                "top_k": 10,
                "min_score": 0.0,
                "max_results": 50,
                "enable_caching": True,
                "cache_ttl": 3600,
            },
            description="BM25检索器，基于BM25算法进行关键词检索"
        ),
        RetrievalStrategy.GRAPH: RetrieverRegistration(
            strategy=RetrievalStrategy.GRAPH,
            retriever_class=GraphRetriever,
            default_config={
                "strategy": "graph",
                "mode": "single",
                "top_k": 10,
                "min_score": 0.0,
                "max_results": 50,
                "enable_caching": True,
                "cache_ttl": 3600,
            },
            description="图检索器，基于图结构进行文档检索"
        ),
    }

    @classmethod
    def create_retriever(
        cls,
        strategy: RetrievalStrategy,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrieverInterface:
        """
        创建单个检索器

        Args:
            strategy: 检索策略
            config: 配置参数
            **kwargs: 其他参数

        Returns:
            RetrieverInterface: 检索器实例
        """
        if strategy not in cls._registered_retrievers:
            raise ValueError(f"不支持的检索策略: {strategy}")

        registration = cls._registered_retrievers[strategy]

        # 合并配置
        final_config = registration.default_config.copy()
        if config:
            final_config.update(config)

        # 创建检索器配置
        retriever_config = RetrieverConfig(**final_config)

        # 创建检索器实例
        try:
            if strategy == RetrievalStrategy.VECTOR:
                embedding_service = kwargs.get("embedding_service")
                return registration.retriever_class(retriever_config, embedding_service)
            elif strategy == RetrievalStrategy.GRAPH:
                graph_service = kwargs.get("graph_service")
                return registration.retriever_class(retriever_config, graph_service)
            else:
                return registration.retriever_class(retriever_config)

        except Exception as e:
            structured_logger.error(f"创建检索器失败: {strategy}, 错误: {e}")
            raise Exception(f"Failed to create retriever {strategy}: {e}")

    @classmethod
    def create_multi_strategy_retriever(
        cls,
        config: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[RetrievalStrategy]] = None,
        **kwargs
    ) -> MultiStrategyRetrieverInterface:
        """
        创建多策略检索器

        Args:
            config: 多策略检索器配置
            strategies: 要启用的策略列表
            **kwargs: 其他参数

        Returns:
            MultiStrategyRetrieverInterface: 多策略检索器实例
        """
        # 默认配置
        default_config = {
            "enable_adaptive": True,
            "enable_caching": True,
            "cache_ttl": 3600,
            "max_parallel_strategies": 3,
            "fusion": {
                "method": "weighted_sum",
                "normalize_scores": True,
                "top_k": 10,
                "diversity_threshold": 0.3,
                "max_results_per_strategy": 50,
            },
            "strategies": {},
        }

        # 合并配置
        final_config = default_config.copy()
        if config:
            final_config.update(config)
            # 深度合并fusion配置
            if "fusion" in config:
                final_config["fusion"].update(config["fusion"])

        # 配置策略
        if strategies is None:
            strategies = [RetrievalStrategy.VECTOR, RetrievalStrategy.BM25]

        for strategy in strategies:
            if strategy not in cls._registered_retrievers:
                structured_logger.warning(f"跳过未知策略: {strategy}")
                continue

            registration = cls._registered_retrievers[strategy]

            # 策略配置
            strategy_config = {
                "enabled": True,
                "weight": 1.0,
                "config": registration.default_config,
            }

            # 从主配置中获取策略特定配置
            strategy_key = strategy.value
            if "strategies" in final_config and strategy_key in final_config["strategies"]:
                strategy_config.update(final_config["strategies"][strategy_key])

            final_config["strategies"][strategy_key] = strategy_config

        # 创建多策略检索器
        try:
            multi_retriever = MultiStrategyRetriever(final_config)

            # 预创建并注册策略检索器
            for strategy in strategies:
                if strategy in cls._registered_retrievers:
                    strategy_config = final_config["strategies"][strategy.value]
                    if strategy_config.get("enabled", True):
                        retriever = cls.create_retriever(
                            strategy,
                            strategy_config.get("config", {}),
                            **kwargs
                        )
                        await multi_retriever.add_strategy(
                            strategy,
                            retriever,
                            strategy_config.get("weight", 1.0)
                        )

            return multi_retriever

        except Exception as e:
            structured_logger.error(f"创建多策略检索器失败: {e}")
            raise Exception(f"Failed to create multi-strategy retriever: {e}")

    @classmethod
    def create_hybrid_retriever(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> MultiStrategyRetrieverInterface:
        """
        创建混合检索器（向量+关键词）

        Args:
            config: 配置参数
            **kwargs: 其他参数

        Returns:
            MultiStrategyRetrieverInterface: 混合检索器实例
        """
        hybrid_config = {
            "fusion": {
                "method": "weighted_sum",
                "top_k": 10,
            },
            "strategies": {
                "vector": {
                    "enabled": True,
                    "weight": 0.7,
                },
                "bm25": {
                    "enabled": True,
                    "weight": 0.3,
                },
            },
        }

        if config:
            hybrid_config.update(config)

        return cls.create_multi_strategy_retriever(
            hybrid_config,
            [RetrievalStrategy.VECTOR, RetrievalStrategy.BM25],
            **kwargs
        )

    @classmethod
    def create_graph_enhanced_retriever(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> MultiStrategyRetrieverInterface:
        """
        创建图增强检索器

        Args:
            config: 配置参数
            **kwargs: 其他参数

        Returns:
            MultiStrategyRetrieverInterface: 图增强检索器实例
        """
        graph_config = {
            "fusion": {
                "method": "adaptive",
                "top_k": 10,
            },
            "strategies": {
                "vector": {
                    "enabled": True,
                    "weight": 0.5,
                },
                "bm25": {
                    "enabled": True,
                    "weight": 0.3,
                },
                "graph": {
                    "enabled": True,
                    "weight": 0.2,
                },
            },
        }

        if config:
            graph_config.update(config)

        return cls.create_multi_strategy_retriever(
            graph_config,
            [RetrievalStrategy.VECTOR, RetrievalStrategy.BM25, RetrievalStrategy.GRAPH],
            **kwargs
        )

    @classmethod
    def register_retriever(
        cls,
        strategy: RetrievalStrategy,
        retriever_class: Type[RetrieverInterface],
        default_config: Dict[str, Any],
        description: str = ""
    ) -> None:
        """
        注册新的检索器类型

        Args:
            strategy: 检索策略
            retriever_class: 检索器类
            default_config: 默认配置
            description: 描述
        """
        registration = RetrieverRegistration(
            strategy=strategy,
            retriever_class=retriever_class,
            default_config=default_config,
            description=description
        )

        cls._registered_retrievers[strategy] = registration
        structured_logger.info(f"注册检索器: {strategy.value} - {description}")

    @classmethod
    def get_available_strategies(cls) -> List[RetrievalStrategy]:
        """获取可用的检索策略"""
        return list(cls._registered_retrievers.keys())

    @classmethod
    def get_strategy_info(cls, strategy: RetrievalStrategy) -> Optional[Dict[str, Any]]:
        """获取策略信息"""
        if strategy not in cls._registered_retrievers:
            return None

        registration = cls._registered_retrievers[strategy]
        return {
            "strategy": strategy.value,
            "description": registration.description,
            "default_config": registration.default_config,
            "class_name": registration.retriever_class.__name__,
        }

    @classmethod
    def get_all_strategies_info(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有策略信息"""
        return {
            strategy.value: cls.get_strategy_info(strategy)
            for strategy in cls._registered_retrievers.keys()
        }

    @classmethod
    def create_retriever_from_template(
        cls,
        template_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrieverInterface:
        """
        从模板创建检索器

        Args:
            template_name: 模板名称
            config_overrides: 配置覆盖
            **kwargs: 其他参数

        Returns:
            RetrieverInterface: 检索器实例
        """
        templates = cls._get_retriever_templates()

        if template_name not in templates:
            raise ValueError(f"未知的模板名称: {template_name}")

        template = templates[template_name]
        final_config = template["config"].copy()
        if config_overrides:
            final_config.update(config_overrides)

        return cls.create_retriever(template["strategy"], final_config, **kwargs)

    @classmethod
    def _get_retriever_templates(cls) -> Dict[str, Dict[str, Any]]:
        """获取检索器模板"""
        return {
            "basic_vector": {
                "strategy": RetrievalStrategy.VECTOR,
                "config": {
                    "top_k": 5,
                    "min_score": 0.3,
                    "enable_caching": True,
                },
                "description": "基础向量检索器"
            },
            "fast_keyword": {
                "strategy": RetrievalStrategy.BM25,
                "config": {
                    "top_k": 10,
                    "min_score": 0.1,
                    "enable_caching": True,
                },
                "description": "快速关键词检索器"
            },
            "semantic_search": {
                "strategy": RetrievalStrategy.VECTOR,
                "config": {
                    "top_k": 20,
                    "min_score": 0.5,
                    "enable_caching": True,
                },
                "description": "语义搜索检索器"
            },
            "knowledge_graph": {
                "strategy": RetrievalStrategy.GRAPH,
                "config": {
                    "top_k": 15,
                    "min_score": 0.2,
                    "enable_caching": True,
                },
                "description": "知识图检索器"
            },
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        验证配置

        Args:
            config: 配置字典

        Returns:
            List[str]: 验证错误列表
        """
        errors = []

        # 验证必需字段
        required_fields = ["strategy", "top_k"]
        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")

        # 验证策略
        if "strategy" in config:
            strategy_str = config["strategy"]
            try:
                strategy = RetrievalStrategy(strategy_str)
                if strategy not in cls._registered_retrievers:
                    errors.append(f"不支持的检索策略: {strategy_str}")
            except ValueError:
                errors.append(f"无效的检索策略: {strategy_str}")

        # 验证数值字段
        numeric_fields = {
            "top_k": (1, 1000),
            "min_score": (0.0, 1.0),
            "max_results": (1, 10000),
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} 必须是数字")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{field} 必须在 [{min_val}, {max_val}] 范围内")

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
        if config.get("top_k", 10) > 50:
            suggestions.append("考虑减少top_k值以提高检索速度")

        if not config.get("enable_caching", True):
            suggestions.append("启用缓存可以提高重复查询的性能")

        if config.get("min_score", 0.0) < 0.1:
            suggestions.append("提高min_score阈值可以过滤低质量结果")

        # 策略组合建议
        strategy = config.get("strategy")
        if strategy == "vector":
            suggestions.append("考虑与BM25检索器组合使用以获得更好的召回率")
        elif strategy == "bm25":
            suggestions.append("考虑与向量检索器组合使用以获得更好的语义理解")

        return suggestions