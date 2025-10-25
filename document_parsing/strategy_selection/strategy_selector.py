"""
处理策略选择器

此模块提供智能的文档处理策略选择功能，
基于文档来源、类型、内容特征等因素选择最优处理策略。
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from ..interfaces.parser_interface import DocumentType, ProcessingStrategy
from ..interfaces.source_processor_interface import DocumentSource
from ..source_detection import SourceDetectionResult, ConfidenceLevel


class ProcessingComplexity(Enum):
    """处理复杂度。"""
    SIMPLE = "simple"        # 简单处理（纯文本提取）
    MODERATE = "moderate"    # 中等处理（保留布局、表格提取）
    COMPLEX = "complex"      # 复杂处理（多模态、OCR、结构保持）
    ADVANCED = "advanced"    # 高级处理（全面分析、深度理解）


class QualityRequirement(Enum):
    """质量要求。"""
    SPEED = "speed"          # 速度优先
    BALANCED = "balanced"    # 平衡模式
    QUALITY = "quality"      # 质量优先
    COMPREHENSIVE = "comprehensive"  # 全面处理


@dataclass
class StrategyWeights:
    """策略权重配置。"""
    accuracy_weight: float = 0.4      # 准确性权重
    speed_weight: float = 0.3         # 速度权重
    cost_weight: float = 0.2          # 成本权重
    resource_weight: float = 0.1      # 资源消耗权重


@dataclass
class ProcessingProfile:
    """处理配置文件。"""
    profile_name: str
    complexity: ProcessingComplexity
    quality_requirement: QualityRequirement
    weights: StrategyWeights
    max_processing_time: Optional[int] = None  # 最大处理时间（秒）
    max_resource_usage: Optional[float] = None  # 最大资源使用率
    preferred_strategies: List[ProcessingStrategy] = field(default_factory=list)
    excluded_strategies: List[ProcessingStrategy] = field(default_factory=list)


@dataclass
class StrategyRecommendation:
    """策略推荐结果。"""
    recommended_strategy: ProcessingStrategy
    confidence: float
    reasoning: List[str]
    estimated_processing_time: Optional[int] = None
    estimated_resource_usage: Optional[float] = None
    alternative_strategies: List[Tuple[ProcessingStrategy, float]] = field(default_factory=list)
    profile_used: Optional[str] = None


class StrategyScorer:
    """策略评分器。"""

    def __init__(self):
        """初始化策略评分器。"""
        self.strategy_scores = {
            ProcessingStrategy.EXTRACT_TEXT: {
                "accuracy": 0.6,
                "speed": 0.9,
                "cost": 0.9,
                "resource": 0.9,
                "complexity": ProcessingComplexity.SIMPLE
            },
            ProcessingStrategy.PRESERVE_LAYOUT: {
                "accuracy": 0.8,
                "speed": 0.6,
                "cost": 0.7,
                "resource": 0.7,
                "complexity": ProcessingComplexity.MODERATE
            },
            ProcessingStrategy.MULTIMODAL_ANALYSIS: {
                "accuracy": 0.9,
                "speed": 0.3,
                "cost": 0.4,
                "resource": 0.3,
                "complexity": ProcessingComplexity.COMPLEX
            },
            ProcessingStrategy.TABLE_EXTRACTION: {
                "accuracy": 0.85,
                "speed": 0.5,
                "cost": 0.6,
                "resource": 0.6,
                "complexity": ProcessingComplexity.MODERATE
            },
            ProcessingStrategy.IMAGE_ANALYSIS: {
                "accuracy": 0.8,
                "speed": 0.4,
                "cost": 0.5,
                "resource": 0.5,
                "complexity": ProcessingComplexity.COMPLEX
            },
            ProcessingStrategy.CODE_EXTRACTION: {
                "accuracy": 0.9,
                "speed": 0.8,
                "cost": 0.8,
                "resource": 0.8,
                "complexity": ProcessingComplexity.MODERATE
            },
            ProcessingStrategy.STRUCTURED_DATA: {
                "accuracy": 0.95,
                "speed": 0.9,
                "cost": 0.9,
                "resource": 0.9,
                "complexity": ProcessingComplexity.SIMPLE
            },
            ProcessingStrategy.FULL_CONTENT: {
                "accuracy": 0.95,
                "speed": 0.2,
                "cost": 0.3,
                "resource": 0.2,
                "complexity": ProcessingComplexity.ADVANCED
            }
        }

    def score_strategy(
        self,
        strategy: ProcessingStrategy,
        profile: ProcessingProfile,
        source_result: SourceDetectionResult
    ) -> float:
        """
        为策略评分。

        Args:
            strategy: 处理策略
            profile: 处理配置文件
            source_result: 来源检测结果

        Returns:
            float: 策略评分
        """
        if strategy not in self.strategy_scores:
            return 0.0

        # 检查排除的策略
        if strategy in profile.excluded_strategies:
            return 0.0

        # 获取策略基础评分
        base_scores = self.strategy_scores[strategy]

        # 根据配置文件权重计算加权分数
        weighted_score = (
            base_scores["accuracy"] * profile.weights.accuracy_weight +
            base_scores["speed"] * profile.weights.speed_weight +
            base_scores["cost"] * profile.weights.cost_weight +
            base_scores["resource"] * profile.weights.resource_weight
        )

        # 根据来源检测结果调整分数
        adjusted_score = self._adjust_score_by_source(
            weighted_score, strategy, source_result
        )

        # 根据质量要求调整分数
        quality_adjusted_score = self._adjust_score_by_quality(
            adjusted_score, strategy, profile.quality_requirement
        )

        return min(quality_adjusted_score, 1.0)

    def _adjust_score_by_source(
        self,
        base_score: float,
        strategy: ProcessingStrategy,
        source_result: SourceDetectionResult
    ) -> float:
        """根据来源检测结果调整分数。"""
        adjustment = 0.0
        source_type = source_result.source_type
        confidence = source_result.confidence

        # 来源类型与策略匹配度
        source_strategy_match = {
            DocumentSource.WEB_DOCUMENTS: {
                ProcessingStrategy.EXTRACT_TEXT: 0.1,
                ProcessingStrategy.PRESERVE_LAYOUT: 0.15,
                ProcessingStrategy.MULTIMODAL_ANALYSIS: 0.05,
            },
            DocumentSource.OFFICE_DOCUMENTS: {
                ProcessingStrategy.EXTRACT_TEXT: 0.05,
                ProcessingStrategy.PRESERVE_LAYOUT: 0.2,
                ProcessingStrategy.TABLE_EXTRACTION: 0.25,
                ProcessingStrategy.MULTIMODAL_ANALYSIS: 0.15,
                ProcessingStrategy.FULL_CONTENT: 0.2,
            },
            DocumentSource.SCANNED_DOCUMENTS: {
                ProcessingStrategy.EXTRACT_TEXT: -0.1,
                ProcessingStrategy.MULTIMODAL_ANALYSIS: 0.3,
                ProcessingStrategy.IMAGE_ANALYSIS: 0.25,
                ProcessingStrategy.FULL_CONTENT: 0.2,
            },
            DocumentSource.CODE_REPOSITORIES: {
                ProcessingStrategy.CODE_EXTRACTION: 0.4,
                ProcessingStrategy.EXTRACT_TEXT: 0.1,
                ProcessingStrategy.STRUCTURED_DATA: 0.05,
            },
            DocumentSource.STRUCTURED_DATA: {
                ProcessingStrategy.STRUCTURED_DATA: 0.4,
                ProcessingStrategy.EXTRACT_TEXT: 0.1,
                ProcessingStrategy.FULL_CONTENT: 0.05,
            }
        }

        if source_type in source_strategy_match:
            adjustment += source_strategy_match[source_type].get(strategy, 0.0)

        # 置信度调整
        if confidence > 0.8:
            adjustment += 0.05  # 高置信度略微提升
        elif confidence < 0.5:
            adjustment -= 0.05  # 低置信度略微降低

        return max(0.0, base_score + adjustment)

    def _adjust_score_by_quality(
        self,
        base_score: float,
        strategy: ProcessingStrategy,
        quality_requirement: QualityRequirement
    ) -> float:
        """根据质量要求调整分数。"""
        strategy_complexity = self.strategy_scores[strategy]["complexity"]

        if quality_requirement == QualityRequirement.SPEED:
            # 速度优先：简单策略加分，复杂策略减分
            if strategy_complexity == ProcessingComplexity.SIMPLE:
                return base_score + 0.2
            elif strategy_complexity == ProcessingComplexity.ADVANCED:
                return base_score - 0.3

        elif quality_requirement == QualityRequirement.QUALITY:
            # 质量优先：复杂策略加分，简单策略减分
            if strategy_complexity == ProcessingComplexity.ADVANCED:
                return base_score + 0.2
            elif strategy_complexity == ProcessingComplexity.SIMPLE:
                return base_score - 0.2

        elif quality_requirement == QualityRequirement.COMPREHENSIVE:
            # 全面处理：只有最复杂的策略得分高
            if strategy_complexity == ProcessingComplexity.ADVANCED:
                return base_score + 0.3
            elif strategy_complexity == ProcessingComplexity.COMPLEX:
                return base_score + 0.1
            else:
                return base_score - 0.2

        return base_score


class StrategySelector:
    """
    处理策略选择器。

    提供智能的文档处理策略选择功能。
    """

    def __init__(self):
        """初始化策略选择器。"""
        self.scorer = StrategyScorer()
        self.profiles = self._initialize_profiles()
        self.source_strategy_mapping = self._initialize_source_mapping()

    def _initialize_profiles(self) -> Dict[str, ProcessingProfile]:
        """初始化处理配置文件。"""
        return {
            "speed_optimized": ProcessingProfile(
                profile_name="speed_optimized",
                complexity=ProcessingComplexity.SIMPLE,
                quality_requirement=QualityRequirement.SPEED,
                weights=StrategyWeights(
                    accuracy_weight=0.2,
                    speed_weight=0.5,
                    cost_weight=0.2,
                    resource_weight=0.1
                ),
                max_processing_time=30,
                preferred_strategies=[ProcessingStrategy.EXTRACT_TEXT],
                excluded_strategies=[ProcessingStrategy.FULL_CONTENT, ProcessingStrategy.MULTIMODAL_ANALYSIS]
            ),

            "balanced": ProcessingProfile(
                profile_name="balanced",
                complexity=ProcessingComplexity.MODERATE,
                quality_requirement=QualityRequirement.BALANCED,
                weights=StrategyWeights(
                    accuracy_weight=0.4,
                    speed_weight=0.3,
                    cost_weight=0.2,
                    resource_weight=0.1
                ),
                max_processing_time=120,
                preferred_strategies=[
                    ProcessingStrategy.EXTRACT_TEXT,
                    ProcessingStrategy.PRESERVE_LAYOUT,
                    ProcessingStrategy.TABLE_EXTRACTION
                ]
            ),

            "quality_optimized": ProcessingProfile(
                profile_name="quality_optimized",
                complexity=ProcessingComplexity.COMPLEX,
                quality_requirement=QualityRequirement.QUALITY,
                weights=StrategyWeights(
                    accuracy_weight=0.6,
                    speed_weight=0.1,
                    cost_weight=0.2,
                    resource_weight=0.1
                ),
                max_processing_time=300,
                preferred_strategies=[
                    ProcessingStrategy.PRESERVE_LAYOUT,
                    ProcessingStrategy.MULTIMODAL_ANALYSIS,
                    ProcessingStrategy.TABLE_EXTRACTION
                ],
                excluded_strategies=[ProcessingStrategy.EXTRACT_TEXT]
            ),

            "comprehensive": ProcessingProfile(
                profile_name="comprehensive",
                complexity=ProcessingComplexity.ADVANCED,
                quality_requirement=QualityRequirement.COMPREHENSIVE,
                weights=StrategyWeights(
                    accuracy_weight=0.7,
                    speed_weight=0.05,
                    cost_weight=0.15,
                    resource_weight=0.1
                ),
                max_processing_time=600,
                preferred_strategies=[ProcessingStrategy.FULL_CONTENT],
                max_resource_usage=0.8
            ),

            "code_focused": ProcessingProfile(
                profile_name="code_focused",
                complexity=ProcessingComplexity.MODERATE,
                quality_requirement=QualityRequirement.QUALITY,
                weights=StrategyWeights(
                    accuracy_weight=0.6,
                    speed_weight=0.2,
                    cost_weight=0.15,
                    resource_weight=0.05
                ),
                preferred_strategies=[ProcessingStrategy.CODE_EXTRACTION],
                excluded_strategies=[ProcessingStrategy.MULTIMODAL_ANALYSIS, ProcessingStrategy.IMAGE_ANALYSIS]
            ),

            "data_focused": ProcessingProfile(
                profile_name="data_focused",
                complexity=ProcessingComplexity.SIMPLE,
                quality_requirement=QualityRequirement.QUALITY,
                weights=StrategyWeights(
                    accuracy_weight=0.7,
                    speed_weight=0.2,
                    cost_weight=0.05,
                    resource_weight=0.05
                ),
                preferred_strategies=[ProcessingStrategy.STRUCTURED_DATA],
                excluded_strategies=[ProcessingStrategy.MULTIMODAL_ANALYSIS, ProcessingStrategy.IMAGE_ANALYSIS]
            )
        }

    def _initialize_source_mapping(self) -> Dict[DocumentSource, str]:
        """初始化来源类型到配置文件的映射。"""
        return {
            DocumentSource.WEB_DOCUMENTS: "balanced",
            DocumentSource.OFFICE_DOCUMENTS: "quality_optimized",
            DocumentSource.SCANNED_DOCUMENTS: "comprehensive",
            DocumentSource.CODE_REPOSITORIES: "code_focused",
            DocumentSource.STRUCTURED_DATA: "data_focused",
            DocumentSource.LOCAL_FILES: "balanced",
            DocumentSource.REMOTE_STORAGE: "balanced",
            DocumentSource.DATABASE: "speed_optimized",
            DocumentSource.EMAIL: "balanced",
            DocumentSource.CHAT: "speed_optimized"
        }

    async def select_strategy(
        self,
        source_result: SourceDetectionResult,
        profile_name: Optional[str] = None,
        requirements: Optional[Dict[str, Any]] = None
    ) -> StrategyRecommendation:
        """
        选择最优处理策略。

        Args:
            source_result: 来源检测结果
            profile_name: 配置文件名称（可选）
            requirements: 特殊要求（可选）

        Returns:
            StrategyRecommendation: 策略推荐结果
        """
        # 选择处理配置文件
        profile = self._select_profile(source_result, profile_name, requirements)

        # 评分所有可用策略
        strategy_scores = []
        for strategy in ProcessingStrategy:
            score = self.scorer.score_strategy(strategy, profile, source_result)
            if score > 0:
                strategy_scores.append((strategy, score))

        # 按分数排序
        strategy_scores.sort(key=lambda x: x[1], reverse=True)

        if not strategy_scores:
            # 如果没有可用策略，返回默认策略
            return StrategyRecommendation(
                recommended_strategy=ProcessingStrategy.EXTRACT_TEXT,
                confidence=0.5,
                reasoning=["使用默认策略，因为没有匹配的策略"],
                profile_used=profile.profile_name
            )

        # 获取推荐策略
        recommended_strategy, confidence = strategy_scores[0]

        # 生成推理说明
        reasoning = self._generate_reasoning(
            recommended_strategy, confidence, source_result, profile, strategy_scores
        )

        # 获取备选策略
        alternatives = strategy_scores[1:3]  # 取前2个备选

        # 估算处理时间和资源使用
        estimated_time = self._estimate_processing_time(recommended_strategy, source_result)
        estimated_resource = self._estimate_resource_usage(recommended_strategy, source_result)

        return StrategyRecommendation(
            recommended_strategy=recommended_strategy,
            confidence=confidence,
            reasoning=reasoning,
            estimated_processing_time=estimated_time,
            estimated_resource_usage=estimated_resource,
            alternative_strategies=alternatives,
            profile_used=profile.profile_name
        )

    def _select_profile(
        self,
        source_result: SourceDetectionResult,
        profile_name: Optional[str],
        requirements: Optional[Dict[str, Any]]
    ) -> ProcessingProfile:
        """选择处理配置文件。"""
        # 如果明确指定了配置文件
        if profile_name and profile_name in self.profiles:
            return self.profiles[profile_name]

        # 根据要求动态选择配置文件
        if requirements:
            if requirements.get("speed_priority", False):
                return self.profiles["speed_optimized"]
            elif requirements.get("quality_priority", False):
                return self.profiles["quality_optimized"]
            elif requirements.get("comprehensive", False):
                return self.profiles["comprehensive"]

        # 根据来源类型选择默认配置文件
        default_profile_name = self.source_strategy_mapping.get(
            source_result.source_type, "balanced"
        )

        # 根据置信度调整
        if source_result.confidence_level == ConfidenceLevel.LOW:
            # 低置信度时选择更保守的策略
            return self.profiles["balanced"]
        elif source_result.confidence_level == ConfidenceLevel.HIGH:
            # 高置信度时可以使用更激进的策略
            return self.profiles[default_profile_name]

        return self.profiles[default_profile_name]

    def _generate_reasoning(
        self,
        strategy: ProcessingStrategy,
        confidence: float,
        source_result: SourceDetectionResult,
        profile: ProcessingProfile,
        all_scores: List[Tuple[ProcessingStrategy, float]]
    ) -> List[str]:
        """生成策略选择推理说明。"""
        reasoning = []

        # 基础推理
        reasoning.append(f"来源类型：{source_result.source_type.value}")
        reasoning.append(f"检测置信度：{source_result.confidence:.2f}")

        # 策略匹配推理
        source_type = source_result.source_type
        if source_type == DocumentSource.SCANNED_DOCUMENTS:
            reasoning.append("检测到扫描文档，推荐使用多模态分析策略")
        elif source_type == DocumentSource.CODE_REPOSITORIES:
            reasoning.append("检测到代码仓库，推荐使用代码提取策略")
        elif source_type == DocumentSource.STRUCTURED_DATA:
            reasoning.append("检测到结构化数据，推荐使用结构化数据处理策略")
        elif source_type == DocumentSource.OFFICE_DOCUMENTS:
            reasoning.append("检测到办公文档，推荐使用布局保持策略")

        # 配置文件推理
        reasoning.append(f"使用配置文件：{profile.profile_name}")
        reasoning.append(f"质量要求：{profile.quality_requirement.value}")

        # 置信度推理
        if confidence >= 0.8:
            reasoning.append("策略匹配置信度高")
        elif confidence >= 0.6:
            reasoning.append("策略匹配置信度中等")
        else:
            reasoning.append("策略匹配置信度较低，建议考虑备选策略")

        # 策略特性推理
        strategy_complexity = self.scorer.strategy_scores[strategy]["complexity"]
        reasoning.append(f"策略复杂度：{strategy_complexity.value}")

        return reasoning

    def _estimate_processing_time(
        self,
        strategy: ProcessingStrategy,
        source_result: SourceDetectionResult
    ) -> Optional[int]:
        """估算处理时间（秒）。"""
        base_times = {
            ProcessingStrategy.EXTRACT_TEXT: 10,
            ProcessingStrategy.PRESERVE_LAYOUT: 30,
            ProcessingStrategy.MULTIMODAL_ANALYSIS: 120,
            ProcessingStrategy.TABLE_EXTRACTION: 45,
            ProcessingStrategy.IMAGE_ANALYSIS: 90,
            ProcessingStrategy.CODE_EXTRACTION: 15,
            ProcessingStrategy.STRUCTURED_DATA: 5,
            ProcessingStrategy.FULL_CONTENT: 180
        }

        base_time = base_times.get(strategy, 60)

        # 根据来源调整
        if source_result.source_type == DocumentSource.SCANNED_DOCUMENTS:
            base_time *= 2  # 扫描文档处理时间更长
        elif source_result.source_type == DocumentSource.STRUCTURED_DATA:
            base_time *= 0.5  # 结构化数据处理更快

        # 根据置信度调整
        if source_result.confidence_level == ConfidenceLevel.LOW:
            base_time *= 1.5  # 低置信度可能需要更多处理时间

        return int(base_time)

    def _estimate_resource_usage(
        self,
        strategy: ProcessingStrategy,
        source_result: SourceDetectionResult
    ) -> Optional[float]:
        """估算资源使用率（0-1）。"""
        resource_usage = {
            ProcessingStrategy.EXTRACT_TEXT: 0.2,
            ProcessingStrategy.PRESERVE_LAYOUT: 0.4,
            ProcessingStrategy.MULTIMODAL_ANALYSIS: 0.8,
            ProcessingStrategy.TABLE_EXTRACTION: 0.5,
            ProcessingStrategy.IMAGE_ANALYSIS: 0.7,
            ProcessingStrategy.CODE_EXTRACTION: 0.3,
            ProcessingStrategy.STRUCTURED_DATA: 0.1,
            ProcessingStrategy.FULL_CONTENT: 0.9
        }

        base_usage = resource_usage.get(strategy, 0.5)

        # 根据来源调整
        if source_result.source_type == DocumentSource.SCANNED_DOCUMENTS:
            base_usage = min(base_usage + 0.2, 1.0)

        return base_usage

    def get_available_profiles(self) -> List[str]:
        """获取可用的配置文件列表。"""
        return list(self.profiles.keys())

    def get_profile(self, profile_name: str) -> Optional[ProcessingProfile]:
        """获取指定配置文件。"""
        return self.profiles.get(profile_name)

    def add_custom_profile(self, profile: ProcessingProfile) -> None:
        """添加自定义配置文件。"""
        self.profiles[profile.profile_name] = profile

    def update_source_mapping(self, mapping: Dict[DocumentSource, str]) -> None:
        """更新来源类型映射。"""
        self.source_strategy_mapping.update(mapping)


# 导出
__all__ = [
    'StrategySelector',
    'StrategyScorer',
    'ProcessingComplexity',
    'QualityRequirement',
    'StrategyWeights',
    'ProcessingProfile',
    'StrategyRecommendation'
]