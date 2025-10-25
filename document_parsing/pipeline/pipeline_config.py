"""
处理流水线配置

定义流水线处理阶段、配置参数和处理策略。
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..interfaces.parser_interface import ProcessingStrategy
from ..strategy_config import ProcessingStrategyConfig


class StageType(Enum):
    """处理阶段类型。"""

    DETECTION = "detection"
    STRATEGY_SELECTION = "strategy_selection"
    PARSING = "parsing"
    CONTENT_EXTRACTION = "content_extraction"
    STRUCTURED_PRESERVATION = "structured_preservation"
    MULTIMODAL_FUSION = "multimodal_fusion"
    QUALITY_VALIDATION = "quality_validation"
    POST_PROCESSING = "post_processing"
    OUTPUT_GENERATION = "output_generation"


class StageStatus(Enum):
    """阶段状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingStage:
    """处理阶段定义。"""

    stage_id: str
    stage_type: StageType
    name: str
    description: str
    enabled: bool = True
    timeout: Optional[int] = None  # 超时时间（秒）
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.stage_id:
            self.stage_id = str(uuid.uuid4())


@dataclass
class PipelineConfig:
    """流水线配置。"""

    pipeline_id: str
    name: str
    description: str = ""
    stages: List[ProcessingStage] = field(default_factory=list)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    parallel_execution: bool = True
    max_concurrent_stages: int = 4
    error_handling_strategy: str = "fail_fast"  # fail_fast, continue, retry
    output_format: str = "standard"  # standard, enhanced, custom

    def __post_init__(self):
        if not self.pipeline_id:
            self.pipeline_id = str(uuid.uuid4())

    def add_stage(self, stage: ProcessingStage) -> None:
        """添加处理阶段。"""
        self.stages.append(stage)

    def remove_stage(self, stage_id: str) -> None:
        """移除处理阶段。"""
        self.stages = [s for s in self.stages if s.stage_id != stage_id]

    def get_stage(self, stage_id: str) -> Optional[ProcessingStage]:
        """获取处理阶段。"""
        return next((s for s in self.stages if s.stage_id == stage_id), None)

    def get_enabled_stages(self) -> List[ProcessingStage]:
        """获取启用的处理阶段。"""
        return [s for s in self.stages if s.enabled]

    def validate_dependencies(self) -> List[str]:
        """验证阶段依赖关系。"""
        errors = []
        stage_ids = {s.stage_id for s in self.stages}

        for stage in self.stages:
            for dep in stage.dependencies:
                if dep not in stage_ids:
                    errors.append(f"阶段 {stage.name} 依赖的阶段 {dep} 不存在")

        return errors

    def create_execution_plan(self) -> List[List[str]]:
        """创建执行计划（按层级分组）。"""
        plan = []
        remaining_stages = self.get_enabled_stages().copy()
        processed = set()

        while remaining_stages:
            # 找出当前可以执行的阶段
            ready_stages = []
            for stage in remaining_stages:
                if all(dep in processed for dep in stage.dependencies):
                    ready_stages.append(stage.stage_id)

            if not ready_stages:
                raise ValueError("发现循环依赖或无法满足的依赖")

            plan.append(ready_stages)
            processed.update(ready_stages)
            remaining_stages = [s for s in remaining_stages if s.stage_id not in ready_stages]

        return plan


@dataclass
class PipelineContext:
    """流水线执行上下文。"""

    pipeline_id: str
    execution_id: str
    tenant_id: str
    user_id: str
    source_type: str
    processing_strategy: ProcessingStrategy
    strategy_config: ProcessingStrategyConfig
    input_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    execution_times: Dict[str, float] = field(default_factory=dict)

    def get_stage_result(self, stage_id: str) -> Any:
        """获取阶段结果。"""
        return self.stage_results.get(stage_id)

    def set_stage_result(self, stage_id: str, result: Any, execution_time: float) -> None:
        """设置阶段结果。"""
        self.stage_results[stage_id] = result
        self.execution_times[stage_id] = execution_time

    def get_input_data(self, key: str, default: Any = None) -> Any:
        """获取输入数据。"""
        return self.input_data.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据。"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据。"""
        return self.metadata.get(key, default)


@dataclass
class StageExecutionResult:
    """阶段执行结果。"""

    stage_id: str
    stage_type: StageType
    status: StageStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """是否执行成功。"""
        return self.status == StageStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """是否执行失败。"""
        return self.status == StageStatus.FAILED


# 预定义的流水线配置模板
class PipelineTemplates:
    """流水线配置模板。"""

    @staticmethod
    def create_standard_pipeline() -> PipelineConfig:
        """创建标准流水线配置。"""
        stages = [
            ProcessingStage(
                stage_id="detection_001",
                stage_type=StageType.DETECTION,
                name="文件来源检测",
                description="检测文档来源类型和特征",
                timeout=30,
                retry_count=2
            ),
            ProcessingStage(
                stage_id="strategy_001",
                stage_type=StageType.STRATEGY_SELECTION,
                name="处理策略选择",
                description="选择最优的处理策略",
                dependencies=["detection_001"],
                timeout=60,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="parsing_001",
                stage_type=StageType.PARSING,
                name="文档解析",
                description="执行文档解析处理",
                dependencies=["strategy_001"],
                timeout=300,
                retry_count=2
            ),
            ProcessingStage(
                stage_id="quality_001",
                stage_type=StageType.QUALITY_VALIDATION,
                name="质量验证",
                description="验证解析质量",
                dependencies=["parsing_001"],
                timeout=60,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="output_001",
                stage_type=StageType.OUTPUT_GENERATION,
                name="输出生成",
                description="生成最终输出结果",
                dependencies=["quality_001"],
                timeout=30,
                retry_count=1
            )
        ]

        return PipelineConfig(
            pipeline_id="standard_001",
            name="标准文档处理流水线",
            description="适用于大多数文档的标准处理流程",
            stages=stages,
            parallel_execution=False,
            error_handling_strategy="fail_fast"
        )

    @staticmethod
    def create_multimodal_pipeline() -> PipelineConfig:
        """创建多模态流水线配置。"""
        stages = [
            ProcessingStage(
                stage_id="detection_001",
                stage_type=StageType.DETECTION,
                name="文件来源检测",
                description="检测文档来源类型和特征",
                timeout=30,
                retry_count=2
            ),
            ProcessingStage(
                stage_id="strategy_001",
                stage_type=StageType.STRATEGY_SELECTION,
                name="处理策略选择",
                description="选择最优的处理策略",
                dependencies=["detection_001"],
                timeout=60,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="parsing_001",
                stage_type=StageType.PARSING,
                name="文档解析",
                description="执行文档解析处理",
                dependencies=["strategy_001"],
                timeout=300,
                retry_count=2
            ),
            ProcessingStage(
                stage_id="content_001",
                stage_type=StageType.CONTENT_EXTRACTION,
                name="内容提取",
                description="提取文本、图像、表格等内容",
                dependencies=["parsing_001"],
                timeout=120,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="structured_001",
                stage_type=StageType.STRUCTURED_PRESERVATION,
                name="结构化信息保存",
                description="保存文档结构信息",
                dependencies=["content_001"],
                timeout=90,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="multimodal_001",
                stage_type=StageType.MULTIMODAL_FUSION,
                name="多模态融合",
                description="融合多模态内容",
                dependencies=["structured_001"],
                timeout=150,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="quality_001",
                stage_type=StageType.QUALITY_VALIDATION,
                name="质量验证",
                description="验证解析质量",
                dependencies=["multimodal_001"],
                timeout=60,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="post_001",
                stage_type=StageType.POST_PROCESSING,
                name="后处理",
                description="执行后处理优化",
                dependencies=["quality_001"],
                timeout=60,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="output_001",
                stage_type=StageType.OUTPUT_GENERATION,
                name="输出生成",
                description="生成最终输出结果",
                dependencies=["post_001"],
                timeout=30,
                retry_count=1
            )
        ]

        return PipelineConfig(
            pipeline_id="multimodal_001",
            name="多模态文档处理流水线",
            description="适用于复杂多模态文档的处理流程",
            stages=stages,
            parallel_execution=True,
            max_concurrent_stages=3,
            error_handling_strategy="continue"
        )

    @staticmethod
    def create_fast_pipeline() -> PipelineConfig:
        """创建快速流水线配置。"""
        stages = [
            ProcessingStage(
                stage_id="detection_001",
                stage_type=StageType.DETECTION,
                name="快速检测",
                description="快速检测文档类型",
                timeout=10,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="parsing_001",
                stage_type=StageType.PARSING,
                name="快速解析",
                description="快速文档解析",
                dependencies=["detection_001"],
                timeout=60,
                retry_count=1
            ),
            ProcessingStage(
                stage_id="output_001",
                stage_type=StageType.OUTPUT_GENERATION,
                name="快速输出",
                description="生成快速输出结果",
                dependencies=["parsing_001"],
                timeout=10,
                retry_count=1
            )
        ]

        return PipelineConfig(
            pipeline_id="fast_001",
            name="快速文档处理流水线",
            description="适用于简单文档的快速处理流程",
            stages=stages,
            parallel_execution=False,
            error_handling_strategy="continue"
        )