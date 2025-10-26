"""
统一实体抽取服务

提供GraphRAG的实体抽取和关系解析功能，支持General和Light两种模式。
"""

import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum

from .interfaces.entity_interface import (
    EntityInterface,
    EntityExtractionRequest,
    EntityExtractionResponse,
    EntityResolutionRequest,
    EntityResolutionResponse,
    CommunityDetectionRequest,
    CommunityDetectionResponse,
    EntityType,
)
from .general.extraction import EntityExtractor
from .general.resolution import EntityResolver
from .general.community import CommunityDetector
from .light.extraction import LightEntityExtractor
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.graph_rag.entity_extraction_service")


class GraphRAGMode(str, Enum):
    """GraphRAG模式"""
    GENERAL = "general"
    LIGHT = "light"


class EntityExtractionService:
    """
    统一实体抽取服务

    支持General和Light两种模式的实体抽取、解析和社区发现。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化实体抽取服务

        Args:
            config: 配置参数
        """
        self.config = config
        self.default_mode = config.get("default_mode", GraphRAGMode.GENERAL)
        self.enable_caching = config.get("enable_caching", True)
        self.enable_parallel_processing = config.get("enable_parallel_processing", True)

        # 初始化不同模式的抽取器
        self.general_extractor = None
        self.general_resolver = None
        self.general_community_detector = None
        self.light_extractor = None

        # 缓存
        self._entity_cache = {} if self.enable_caching else None
        self._resolution_cache = {} if self.enable_caching else None

    async def initialize(self) -> bool:
        """
        初始化实体抽取服务

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化统一实体抽取服务",
                extra={
                    "default_mode": self.default_mode,
                    "enable_caching": self.enable_caching,
                    "enable_parallel_processing": self.enable_parallel_processing,
                }
            )

            # 初始化General模式组件
            general_config = self.config.get("general", {})
            self.general_extractor = EntityExtractor(general_config)
            self.general_resolver = EntityResolver(general_config)
            self.general_community_detector = CommunityDetector(general_config)

            # 初始化Light模式组件
            light_config = self.config.get("light", {})
            self.light_extractor = LightEntityExtractor(light_config)

            # 并行初始化所有组件
            init_tasks = [
                self.general_extractor.initialize(),
                self.general_resolver.initialize(),
                self.general_community_detector.initialize(),
                self.light_extractor.initialize(),
            ]

            results = await asyncio.gather(*init_tasks, return_exceptions=True)

            # 检查初始化结果
            success_count = sum(1 for result in results if result is True)
            total_count = len(results)

            if success_count == total_count:
                structured_logger.info("所有组件初始化成功")
                return True
            else:
                failed_components = [
                    name for name, result in zip(
                        ["general_extractor", "general_resolver", "general_community_detector", "light_extractor"],
                        results
                    )
                    if result is not True
                ]
                structured_logger.warning(f"部分组件初始化失败: {failed_components}")
                return True  # 部分失败仍然可以继续工作

        except Exception as e:
            structured_logger.error(f"实体抽取服务初始化失败: {e}")
            return False

    async def extract_entities(
        self,
        request: EntityExtractionRequest,
        mode: Optional[GraphRAGMode] = None
    ) -> EntityExtractionResponse:
        """
        抽取实体

        Args:
            request: 实体抽取请求
            mode: GraphRAG模式，如果为None则使用默认模式

        Returns:
            EntityExtractionResponse: 抽取结果
        """
        mode = mode or self.default_mode

        try:
            structured_logger.info(
                f"开始实体抽取",
                extra={
                    "mode": mode,
                    "text_length": len(request.text),
                    "entity_types": [t.value for t in request.entity_types] if request.entity_types else [],
                }
            )

            # 检查缓存
            cache_key = self._get_extraction_cache_key(request, mode)
            if self.enable_caching and cache_key in self._entity_cache:
                structured_logger.debug("使用缓存的实体抽取结果")
                return self._entity_cache[cache_key]

            # 选择抽取器
            if mode == GraphRAGMode.GENERAL:
                extractor = self.general_extractor
            elif mode == GraphRAGMode.LIGHT:
                extractor = self.light_extractor
            else:
                raise ValueError(f"不支持的GraphRAG模式: {mode}")

            # 执行抽取
            response = await extractor.extract_entities(request)

            # 缓存结果
            if self.enable_caching:
                self._entity_cache[cache_key] = response

            structured_logger.info(
                f"实体抽取完成",
                extra={
                    "mode": mode,
                    "entities_count": response.total_entities,
                    "relationships_count": response.total_relationships,
                    "processing_time_ms": response.processing_time_ms,
                }
            )

            return response

        except Exception as e:
            structured_logger.error(f"实体抽取失败: {e}")
            raise Exception(f"Entity extraction failed: {e}")

    async def resolve_entities(
        self,
        request: EntityResolutionRequest,
        mode: Optional[GraphRAGMode] = None
    ) -> EntityResolutionResponse:
        """
        解析实体

        Args:
            request: 实体解析请求
            mode: GraphRAG模式

        Returns:
            EntityResolutionResponse: 解析结果
        """
        mode = mode or self.default_mode

        try:
            structured_logger.info(
                f"开始实体解析",
                extra={
                    "mode": mode,
                    "input_entities_count": len(request.entities),
                    "existing_entities_count": len(request.existing_entities),
                }
            )

            # 检查缓存
            cache_key = self._get_resolution_cache_key(request, mode)
            if self.enable_caching and cache_key in self._resolution_cache:
                structured_logger.debug("使用缓存的实体解析结果")
                return self._resolution_cache[cache_key]

            # 选择解析器
            if mode == GraphRAGMode.GENERAL:
                resolver = self.general_resolver
            elif mode == GraphRAGMode.LIGHT:
                resolver = self.light_extractor  # Light模式包含解析功能
            else:
                raise ValueError(f"不支持的GraphRAG模式: {mode}")

            # 执行解析
            response = await resolver.resolve_entities(request)

            # 缓存结果
            if self.enable_caching:
                self._resolution_cache[cache_key] = response

            structured_logger.info(
                f"实体解析完成",
                extra={
                    "mode": mode,
                    "resolved_entities_count": len(response.resolved_entities),
                    "merged_entities_count": len(response.merged_entities),
                    "processing_time_ms": response.processing_time_ms,
                }
            )

            return response

        except Exception as e:
            structured_logger.error(f"实体解析失败: {e}")
            raise Exception(f"Entity resolution failed: {e}")

    async def detect_communities(
        self,
        request: CommunityDetectionRequest,
        mode: Optional[GraphRAGMode] = None
    ) -> CommunityDetectionResponse:
        """
        检测社区

        Args:
            request: 社区发现请求
            mode: GraphRAG模式

        Returns:
            CommunityDetectionResponse: 社区检测结果
        """
        mode = mode or self.default_mode

        try:
            structured_logger.info(
                f"开始社区发现",
                extra={
                    "mode": mode,
                    "entities_count": len(request.entities),
                    "relationships_count": len(request.relationships),
                    "algorithm": request.algorithm,
                }
            )

            # 选择社区发现器
            if mode == GraphRAGMode.GENERAL:
                detector = self.general_community_detector
            elif mode == GraphRAGMode.LIGHT:
                detector = self.light_extractor  # Light模式包含社区发现功能
            else:
                raise ValueError(f"不支持的GraphRAG模式: {mode}")

            # 执行社区发现
            response = await detector.detect_communities(request)

            structured_logger.info(
                f"社区发现完成",
                extra={
                    "mode": mode,
                    "communities_count": response.total_communities,
                    "total_entities_assigned": response.total_entities_assigned,
                    "processing_time_ms": response.processing_time_ms,
                }
            )

            return response

        except Exception as e:
            structured_logger.error(f"社区发现失败: {e}")
            raise Exception(f"Community detection failed: {e}")

    async def extract_and_resolve(
        self,
        extraction_request: EntityExtractionRequest,
        resolution_request: Optional[EntityResolutionRequest] = None,
        community_request: Optional[CommunityDetectionRequest] = None,
        mode: Optional[GraphRAGMode] = None
    ) -> Dict[str, Any]:
        """
        一站式实体抽取、解析和社区发现

        Args:
            extraction_request: 实体抽取请求
            resolution_request: 实体解析请求（可选）
            community_request: 社区发现请求（可选）
            mode: GraphRAG模式

        Returns:
            Dict[str, Any]: 包含所有结果的字典
        """
        mode = mode or self.default_mode

        try:
            structured_logger.info(
                f"开始一站式实体处理",
                extra={"mode": mode}
            )

            # 执行实体抽取
            extraction_response = await self.extract_entities(extraction_request, mode)

            results = {
                "mode": mode,
                "extraction": extraction_response,
            }

            # 如果有解析请求，执行实体解析
            if resolution_request:
                # 如果解析请求中没有现有实体，使用抽取结果
                if not resolution_request.existing_entities:
                    resolution_request.existing_entities = extraction_response.entities

                resolution_response = await self.resolve_entities(resolution_request, mode)
                results["resolution"] = resolution_response

            # 如果有社区发现请求，执行社区发现
            if community_request:
                # 如果社区发现请求中没有实体和关系，使用解析或抽取结果
                if not community_request.entities:
                    if "resolution" in results:
                        community_request.entities = results["resolution"].resolved_entities
                    else:
                        community_request.entities = extraction_response.entities

                if not community_request.relationships:
                    community_request.relationships = extraction_response.relationships

                community_response = await self.detect_communities(community_request, mode)
                results["community"] = community_response

            structured_logger.info(
                f"一站式实体处理完成",
                extra={
                    "mode": mode,
                    "steps_completed": len(results) - 1,  # 减去mode字段
                }
            )

            return results

        except Exception as e:
            structured_logger.error(f"一站式实体处理失败: {e}")
            raise Exception(f"Entity processing pipeline failed: {e}")

    def _get_extraction_cache_key(self, request: EntityExtractionRequest, mode: GraphRAGMode) -> str:
        """生成抽取缓存键"""
        # 简化的缓存键生成
        content = f"{mode}_{request.text[:100]}_{request.confidence_threshold}"
        if request.entity_types:
            content += f"_{'_'.join(sorted(t.value for t in request.entity_types))}"
        return str(hash(content))

    def _get_resolution_cache_key(self, request: EntityResolutionRequest, mode: GraphRAGMode) -> str:
        """生成解析缓存键"""
        # 简化的缓存键生成
        entity_names = '_'.join(sorted(e.name for e in request.entities[:10]))  # 限制前10个实体
        existing_names = '_'.join(sorted(e.name for e in request.existing_entities[:10]))
        content = f"{mode}_resolution_{entity_names}_{existing_names}_{request.similarity_threshold}"
        return str(hash(content))

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            # 并行获取各组件统计信息
            stats_tasks = [
                self.general_extractor.get_statistics(),
                self.general_resolver.get_statistics(),
                self.general_community_detector.get_statistics(),
                self.light_extractor.get_statistics(),
            ]

            results = await asyncio.gather(*stats_tasks, return_exceptions=True)

            return {
                "service_mode": self.default_mode,
                "general_extractor_stats": results[0] if not isinstance(results[0], Exception) else {},
                "general_resolver_stats": results[1] if not isinstance(results[1], Exception) else {},
                "general_community_detector_stats": results[2] if not isinstance(results[2], Exception) else {},
                "light_extractor_stats": results[3] if not isinstance(results[3], Exception) else {},
                "cache_stats": {
                    "entity_cache_size": len(self._entity_cache) if self._entity_cache else 0,
                    "resolution_cache_size": len(self._resolution_cache) if self._resolution_cache else 0,
                    "caching_enabled": self.enable_caching,
                },
                "config": {
                    "default_mode": self.default_mode,
                    "enable_caching": self.enable_caching,
                    "enable_parallel_processing": self.enable_parallel_processing,
                },
            }

        except Exception as e:
            structured_logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 并行健康检查
            health_tasks = [
                self.general_extractor.health_check(),
                self.general_resolver.health_check(),
                self.general_community_detector.health_check(),
                self.light_extractor.health_check(),
            ]

            results = await asyncio.gather(*health_tasks, return_exceptions=True)

            # 检查整体健康状态
            healthy_components = sum(1 for result in results if isinstance(result, dict) and result.get("status") == "healthy")
            total_components = len(results)
            overall_status = "healthy" if healthy_components == total_components else "degraded"

            return {
                "status": overall_status,
                "service_mode": self.default_mode,
                "components": {
                    "general_extractor": results[0] if not isinstance(results[0], Exception) else {"status": "error", "error": str(results[0])},
                    "general_resolver": results[1] if not isinstance(results[1], Exception) else {"status": "error", "error": str(results[1])},
                    "general_community_detector": results[2] if not isinstance(results[2], Exception) else {"status": "error", "error": str(results[2])},
                    "light_extractor": results[3] if not isinstance(results[3], Exception) else {"status": "error", "error": str(results[3])},
                },
                "healthy_components": healthy_components,
                "total_components": total_components,
                "config": {
                    "default_mode": self.default_mode,
                    "enable_caching": self.enable_caching,
                    "enable_parallel_processing": self.enable_parallel_processing,
                },
            }

        except Exception as e:
            structured_logger.error(f"健康检查失败: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            cleanup_tasks = []
            if self.general_extractor:
                cleanup_tasks.append(self.general_extractor.cleanup())
            if self.general_resolver:
                cleanup_tasks.append(self.general_resolver.cleanup())
            if self.general_community_detector:
                cleanup_tasks.append(self.general_community_detector.cleanup())
            if self.light_extractor:
                cleanup_tasks.append(self.light_extractor.cleanup())

            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            # 清理缓存
            if self._entity_cache:
                self._entity_cache.clear()
            if self._resolution_cache:
                self._resolution_cache.clear()

            structured_logger.info("实体抽取服务清理完成")

        except Exception as e:
            structured_logger.error(f"实体抽取服务清理失败: {e}")