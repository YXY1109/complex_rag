"""
处理流水线编排器

负责协调和编排文档处理的各个阶段，确保处理流程的正确执行。
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
import logging
from enum import Enum

from .pipeline_config import (
    PipelineConfig, PipelineContext, ProcessingStage, StageType, StageStatus,
    StageExecutionResult, PipelineTemplates
)
from ..source_detection.source_detector import SourceDetector
from ..strategy_selection.strategy_selector import StrategySelector
from ..processors.base_processor import BaseProcessor


class OrchestratorState(Enum):
    """编排器状态。"""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class PipelineOrchestrator:
    """流水线编排器。"""

    def __init__(self, config: PipelineConfig):
        """
        初始化编排器。

        Args:
            config: 流水线配置
        """
        self.config = config
        self.state = OrchestratorState.IDLE
        self.logger = logging.getLogger(__name__)

        # 组件注册
        self.source_detector: Optional[SourceDetector] = None
        self.strategy_selector: Optional[StrategySelector] = None
        self.processors: Dict[str, BaseProcessor] = {}

        # 执行状态
        self.active_executions: Dict[str, PipelineContext] = {}
        self.stage_handlers: Dict[StageType, Callable] = {}
        self.execution_history: List[StageExecutionResult] = []

        # 注册默认阶段处理器
        self._register_default_handlers()

    def register_source_detector(self, detector: SourceDetector) -> None:
        """注册来源检测器。"""
        self.source_detector = detector

    def register_strategy_selector(self, selector: StrategySelector) -> None:
        """注册策略选择器。"""
        self.strategy_selector = selector

    def register_processor(self, processor: BaseProcessor) -> None:
        """注册文档处理器。"""
        self.processors[processor.parser_name] = processor

    def register_stage_handler(self, stage_type: StageType, handler: Callable) -> None:
        """注册阶段处理器。"""
        self.stage_handlers[stage_type] = handler

    def _register_default_handlers(self) -> None:
        """注册默认阶段处理器。"""
        self.stage_handlers[StageType.DETECTION] = self._handle_detection
        self.stage_handlers[StageType.STRATEGY_SELECTION] = self._handle_strategy_selection
        self.stage_handlers[StageType.PARSING] = self._handle_parsing
        self.stage_handlers[StageType.QUALITY_VALIDATION] = self._handle_quality_validation
        self.stage_handlers[StageType.OUTPUT_GENERATION] = self._handle_output_generation

    async def execute_pipeline(
        self,
        context: PipelineContext,
        progress_callback: Optional[Callable] = None
    ) -> List[StageExecutionResult]:
        """
        执行流水线。

        Args:
            context: 流水线执行上下文
            progress_callback: 进度回调函数

        Returns:
            List[StageExecutionResult]: 执行结果列表
        """
        if self.state != OrchestratorState.IDLE:
            raise RuntimeError(f"编排器当前状态为 {self.state.value}，无法执行新的流水线")

        try:
            self.state = OrchestratorState.RUNNING
            self.active_executions[context.execution_id] = context

            # 验证配置
            validation_errors = self.config.validate_dependencies()
            if validation_errors:
                raise ValueError(f"流水线配置验证失败: {validation_errors}")

            # 创建执行计划
            execution_plan = self.config.create_execution_plan()
            self.logger.info(f"开始执行流水线 {self.config.name}，计划层级: {len(execution_plan)}")

            results = []
            completed_stages: Set[str] = set()

            # 按层级执行阶段
            for level, stage_ids in enumerate(execution_plan):
                self.logger.info(f"执行第 {level + 1} 层，包含阶段: {stage_ids}")

                if self.config.parallel_execution and len(stage_ids) > 1:
                    # 并行执行
                    level_results = await self._execute_stage_group_parallel(
                        stage_ids, context, completed_stages, progress_callback
                    )
                else:
                    # 串行执行
                    level_results = await self._execute_stage_group_sequential(
                        stage_ids, context, completed_stages, progress_callback
                    )

                results.extend(level_results)
                completed_stages.update(stage_ids)

                # 检查是否需要停止
                if self.state == OrchestratorState.STOPPING:
                    self.logger.info("流水线执行被停止")
                    break

                # 错误处理策略
                failed_results = [r for r in level_results if r.failed]
                if failed_results:
                    if self.config.error_handling_strategy == "fail_fast":
                        self.logger.error(f"阶段执行失败，快速停止: {[r.stage_id for r in failed_results]}")
                        break
                    elif self.config.error_handling_strategy == "retry":
                        # 这里可以实现重试逻辑
                        pass

            return results

        except Exception as e:
            self.logger.error(f"流水线执行异常: {e}")
            self.state = OrchestratorState.ERROR
            raise
        finally:
            self.active_executions.pop(context.execution_id, None)
            if self.state == OrchestratorState.RUNNING:
                self.state = OrchestratorState.IDLE

    async def _execute_stage_group_parallel(
        self,
        stage_ids: List[str],
        context: PipelineContext,
        completed_stages: Set[str],
        progress_callback: Optional[Callable]
    ) -> List[StageExecutionResult]:
        """并行执行一组阶段。"""
        tasks = []
        for stage_id in stage_ids:
            task = asyncio.create_task(
                self._execute_stage(stage_id, context, completed_stages, progress_callback)
            )
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    stage_result = StageExecutionResult(
                        stage_id=stage_ids[i],
                        stage_type=StageType.PARSING,  # 默认类型
                        status=StageStatus.FAILED,
                        error=str(result)
                    )
                else:
                    stage_result = result

                processed_results.append(stage_result)
                self.execution_history.append(stage_result)

            return processed_results

        except Exception as e:
            self.logger.error(f"并行执行阶段组失败: {e}")
            raise

    async def _execute_stage_group_sequential(
        self,
        stage_ids: List[str],
        context: PipelineContext,
        completed_stages: Set[str],
        progress_callback: Optional[Callable]
    ) -> List[StageExecutionResult]:
        """串行执行一组阶段。"""
        results = []
        for stage_id in stage_ids:
            result = await self._execute_stage(
                stage_id, context, completed_stages, progress_callback
            )
            results.append(result)
            self.execution_history.append(result)

            # 如果阶段失败且策略为快速失败，则停止
            if result.failed and self.config.error_handling_strategy == "fail_fast":
                break

        return results

    async def _execute_stage(
        self,
        stage_id: str,
        context: PipelineContext,
        completed_stages: Set[str],
        progress_callback: Optional[Callable]
    ) -> StageExecutionResult:
        """执行单个阶段。"""
        stage = self.config.get_stage(stage_id)
        if not stage:
            raise ValueError(f"阶段 {stage_id} 不存在")

        if not stage.enabled:
            return StageExecutionResult(
                stage_id=stage_id,
                stage_type=stage.stage_type,
                status=StageStatus.SKIPPED,
                result=None,
                execution_time=0.0
            )

        start_time = datetime.now()
        self.logger.info(f"开始执行阶段: {stage.name} ({stage_id})")

        try:
            # 检查超时
            timeout = stage.timeout or self.config.global_settings.get("default_timeout", 300)

            # 执行阶段处理器
            handler = self.stage_handlers.get(stage.stage_type)
            if not handler:
                raise ValueError(f"未找到阶段类型 {stage.stage_type} 的处理器")

            result = await asyncio.wait_for(
                handler(stage, context),
                timeout=timeout
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # 保存结果到上下文
            context.set_stage_result(stage_id, result, execution_time)

            stage_result = StageExecutionResult(
                stage_id=stage_id,
                stage_type=stage.stage_type,
                status=StageStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                start_time=start_time,
                end_time=datetime.now()
            )

            self.logger.info(f"阶段 {stage.name} 执行完成，耗时: {execution_time:.2f}秒")

            # 调用进度回调
            if progress_callback:
                try:
                    await progress_callback(stage_id, stage_result, len(completed_stages) + 1)
                except Exception as e:
                    self.logger.warning(f"进度回调执行失败: {e}")

            return stage_result

        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"阶段 {stage.name} 执行超时 ({timeout}秒)"
            self.logger.error(error_msg)

            return StageExecutionResult(
                stage_id=stage_id,
                stage_type=stage.stage_type,
                status=StageStatus.FAILED,
                error=error_msg,
                execution_time=execution_time,
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"阶段 {stage.name} 执行失败: {str(e)}"
            self.logger.error(error_msg)

            return StageExecutionResult(
                stage_id=stage_id,
                stage_type=stage.stage_type,
                status=StageStatus.FAILED,
                error=error_msg,
                execution_time=execution_time,
                start_time=start_time,
                end_time=datetime.now()
            )

    async def _handle_detection(
        self,
        stage: ProcessingStage,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """处理检测阶段。"""
        if not self.source_detector:
            raise RuntimeError("来源检测器未注册")

        file_path = context.get_input_data("file_path")
        if not file_path:
            raise ValueError("缺少文件路径输入")

        detection_result = await self.source_detector.detect_source(file_path)

        # 更新上下文中的源类型
        context.source_type = detection_result.source_type.value

        return {
            "detection_result": detection_result,
            "source_type": detection_result.source_type.value,
            "confidence": detection_result.confidence,
            "features": detection_result.features
        }

    async def _handle_strategy_selection(
        self,
        stage: ProcessingStage,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """处理策略选择阶段。"""
        if not self.strategy_selector:
            raise RuntimeError("策略选择器未注册")

        # 获取检测结果
        detection_result = context.get_stage_result(stage.dependencies[0])
        if not detection_result:
            raise ValueError("缺少检测结果")

        # 选择策略
        strategy_recommendation = await self.strategy_selector.select_strategy(
            detection_result["detection_result"],
            context.processing_strategy
        )

        return {
            "selected_strategy": strategy_recommendation.strategy,
            "confidence": strategy_recommendation.confidence,
            "reasoning": strategy_recommendation.reasoning,
            "alternative_strategies": strategy_recommendation.alternative_strategies
        }

    async def _handle_parsing(
        self,
        stage: ProcessingStage,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """处理解析阶段。"""
        file_path = context.get_input_data("file_path")
        if not file_path:
            raise ValueError("缺少文件路径输入")

        # 获取选中的策略
        strategy_result = context.get_stage_result(stage.dependencies[0])
        selected_strategy = strategy_result.get("selected_strategy", context.processing_strategy)

        # 选择合适的处理器
        processor = self._get_processor_for_source(context.source_type)
        if not processor:
            raise ValueError(f"未找到适合 {context.source_type} 的处理器")

        # 执行解析
        parse_result = await processor.parse_document(file_path, selected_strategy)

        return {
            "parse_result": parse_result,
            "metadata": parse_result.metadata,
            "text_chunks": parse_result.text_chunks,
            "full_text": parse_result.full_text,
            "structured_data": parse_result.structured_data,
            "tables": parse_result.tables,
            "images": parse_result.images
        }

    async def _handle_quality_validation(
        self,
        stage: ProcessingStage,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """处理质量验证阶段。"""
        # 获取解析结果
        parse_result = context.get_stage_result(stage.dependencies[0])
        if not parse_result:
            raise ValueError("缺少解析结果")

        # 简单的质量验证逻辑
        parse_data = parse_result["parse_result"]

        quality_metrics = {
            "text_quality": self._assess_text_quality(parse_data.full_text),
            "structure_quality": self._assess_structure_quality(parse_data.structured_data),
            "completeness": self._assess_completeness(parse_data),
            "confidence": parse_data.metadata.metadata.get("overall_confidence", 0.8)
        }

        # 计算总体质量分数
        overall_quality = sum(quality_metrics.values()) / len(quality_metrics)

        return {
            "quality_metrics": quality_metrics,
            "overall_quality": overall_quality,
            "validation_passed": overall_quality >= 0.7,
            "recommendations": self._generate_quality_recommendations(quality_metrics)
        }

    async def _handle_output_generation(
        self,
        stage: ProcessingStage,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """处理输出生成阶段。"""
        # 收集所有阶段结果
        all_results = context.stage_results

        # 根据输出格式生成结果
        if self.config.output_format == "standard":
            output = self._generate_standard_output(all_results)
        elif self.config.output_format == "enhanced":
            output = self._generate_enhanced_output(all_results)
        else:
            output = self._generate_custom_output(all_results, stage.parameters)

        return {
            "output_format": self.config.output_format,
            "output_data": output,
            "execution_summary": self._generate_execution_summary(context),
            "pipeline_metadata": {
                "pipeline_id": self.config.pipeline_id,
                "pipeline_name": self.config.name,
                "execution_id": context.execution_id,
                "execution_time": sum(context.execution_times.values())
            }
        }

    def _get_processor_for_source(self, source_type: str) -> Optional[BaseProcessor]:
        """根据源类型获取处理器。"""
        # 这里可以根据源类型映射到合适的处理器
        source_processor_mapping = {
            "web_documents": "web_processor",
            "office_documents": "office_processor",
            "scanned_documents": "scanned_processor",
            "structured_data": "structured_processor",
            "code_repositories": "code_processor"
        }

        processor_name = source_processor_mapping.get(source_type)
        return self.processors.get(processor_name) if processor_name else None

    def _assess_text_quality(self, text: str) -> float:
        """评估文本质量。"""
        if not text:
            return 0.0

        # 简单的文本质量评估
        word_count = len(text.split())
        char_count = len(text)

        # 基础分数
        base_score = min(1.0, word_count / 100.0)  # 100词以上得满分

        # 长度惩罚（过长或过短的文本质量较低）
        if char_count < 50:
            length_penalty = 0.5
        elif char_count > 100000:
            length_penalty = 0.8
        else:
            length_penalty = 1.0

        return base_score * length_penalty

    def _assess_structure_quality(self, structured_data: Dict[str, Any]) -> float:
        """评估结构化数据质量。"""
        if not structured_data:
            return 0.0

        # 基于结构化数据的丰富程度评估
        structure_types = len([k for k, v in structured_data.items() if v])
        max_types = 5  # 假设最多5种结构化数据类型

        return min(1.0, structure_types / max_types)

    def _assess_completeness(self, parse_result) -> float:
        """评估解析完整性。"""
        completeness_factors = []

        # 文本完整性
        text_completeness = 1.0 if parse_result.full_text.strip() else 0.0
        completeness_factors.append(text_completeness)

        # 块完整性
        chunk_completeness = min(1.0, len(parse_result.text_chunks) / 5.0)
        completeness_factors.append(chunk_completeness)

        # 元数据完整性
        metadata_completeness = min(1.0, len(parse_result.metadata.metadata) / 10.0)
        completeness_factors.append(metadata_completeness)

        return sum(completeness_factors) / len(completeness_factors)

    def _generate_quality_recommendations(self, quality_metrics: Dict[str, float]) -> List[str]:
        """生成质量改进建议。"""
        recommendations = []

        if quality_metrics["text_quality"] < 0.7:
            recommendations.append("考虑提高OCR质量或使用更好的文本提取方法")

        if quality_metrics["structure_quality"] < 0.7:
            recommendations.append("启用更详细的结构分析或表格提取")

        if quality_metrics["completeness"] < 0.7:
            recommendations.append("检查文档是否完整或启用多模态处理")

        return recommendations

    def _generate_standard_output(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成标准格式输出。"""
        # 找到解析结果
        parse_result = None
        for result in all_results.values():
            if isinstance(result, dict) and "parse_result" in result:
                parse_result = result["parse_result"]
                break

        return {
            "text": parse_result.full_text if parse_result else "",
            "chunks": parse_result.text_chunks if parse_result else [],
            "metadata": parse_result.metadata if parse_result else None
        }

    def _generate_enhanced_output(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成增强格式输出。"""
        output = self._generate_standard_output(all_results)

        # 添加更多详细信息
        output.update({
            "processing_stages": list(all_results.keys()),
            "quality_metrics": all_results.get("quality_001", {}).get("quality_metrics", {}),
            "execution_times": {k: v for k, v in all_results.items() if isinstance(v, dict)},
            "structured_data": {},
            "tables": [],
            "images": []
        })

        return output

    def _generate_custom_output(self, all_results: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """生成自定义格式输出。"""
        # 根据参数自定义输出格式
        output_format = parameters.get("format", "standard")

        if output_format == "json":
            return {"data": all_results, "format": "json"}
        else:
            return self._generate_standard_output(all_results)

    def _generate_execution_summary(self, context: PipelineContext) -> Dict[str, Any]:
        """生成执行摘要。"""
        return {
            "total_stages": len(context.stage_results),
            "successful_stages": len([r for r in context.execution_history if r.status == StageStatus.COMPLETED]),
            "failed_stages": len([r for r in context.execution_history if r.status == StageStatus.FAILED]),
            "total_execution_time": sum(context.execution_times.values()),
            "average_stage_time": sum(context.execution_times.values()) / len(context.execution_times) if context.execution_times else 0
        }

    def stop_execution(self, execution_id: str) -> bool:
        """停止执行。"""
        if execution_id in self.active_executions:
            self.state = OrchestratorState.STOPPING
            return True
        return False

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态。"""
        context = self.active_executions.get(execution_id)
        if not context:
            return None

        return {
            "execution_id": execution_id,
            "state": self.state.value,
            "completed_stages": len(context.stage_results),
            "total_execution_time": sum(context.execution_times.values()),
            "current_stage": "idle" if self.state == OrchestratorState.IDLE else "running"
        }

    @staticmethod
    def create_standard_orchestrator() -> "PipelineOrchestrator":
        """创建标准编排器。"""
        config = PipelineTemplates.create_standard_pipeline()
        return PipelineOrchestrator(config)

    @staticmethod
    def create_multimodal_orchestrator() -> "PipelineOrchestrator":
        """创建多模态编排器。"""
        config = PipelineTemplates.create_multimodal_pipeline()
        return PipelineOrchestrator(config)

    @staticmethod
    def create_fast_orchestrator() -> "PipelineOrchestrator":
        """创建快速编排器。"""
        config = PipelineTemplates.create_fast_pipeline()
        return PipelineOrchestrator(config)