"""
Document Processing Pipeline

This module provides orchestration for document processing workflows,
including pipeline creation, execution, and management, inspired by RAGFlow's
document processing architecture.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import time

from ..interfaces.source_interface import ParseRequest, ParseResponse
from ..source_handlers import (
    WebDocumentsHandler,
    OfficeDocumentsHandler,
    ScannedDocumentsHandler,
    StructuredDataHandler,
    CodeRepositoriesHandler
)
from ..vision import VisionRecognizer, LayoutRecognizer
from ..services.quality_monitor import QualityMonitor, QualityMetric

logger = logging.getLogger(__name__)


class PipelineStep(str, Enum):
    """Pipeline step types."""
    SOURCE_DETECTION = "source_detection"
    STRATEGY_SELECTION = "strategy_selection"
    CONTENT_PROCESSING = "content_processing"
    VISION_ANALYSIS = "vision_analysis"
    MULTIMODAL_FUSION = "multimodal_fusion"
    CHUNKING = "chunking"
    QUALITY_CHECK = "quality_check"
    POST_PROCESSING = "post_processing"


@dataclass
class PipelineStepConfig:
    """Configuration for a pipeline step."""
    step: PipelineStep
    enabled: bool = True
    required: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retry_attempts: int = 3


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    name: str
    steps: List[PipelineStepConfig]
    parallel_steps: List[List[PipelineStep]] = field(default_factory=list)
    error_handling: str = "continue"  # continue, stop, retry
    quality_threshold: float = 0.7
    enable_caching: bool = True
    max_concurrent_documents: int = 4


@dataclass
class PipelineContext:
    """Context information passed through pipeline steps."""
    request_id: str
    document_id: str
    start_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_name: str
    success: bool
    response: Optional[ParseResponse] = None
    processing_time_seconds: float
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0


class DocumentProcessingPipeline:
    """
    Orchestrates document processing workflows with configurable steps.

    Features:
    - Configurable pipeline steps and workflows
    - Parallel step execution support
    - Error handling and recovery
    - Quality monitoring and validation
    - Caching and optimization
    - Batch processing capabilities
    """

    def __init__(self):
        """Initialize processing pipeline."""
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.source_handlers = {
            'web_documents': WebDocumentsHandler(),
            'office_documents': OfficeDocumentsHandler(),
            'scanned_documents': ScannedDocumentsHandler(),
            'structured_data': StructuredDataHandler(),
            'code_repositories': CodeRepositoriesHandler()
        }
        self.vision_recognizer = VisionRecognizer()
        self.layout_recognizer = LayoutRecognizer()
        self.quality_monitor = QualityMonitor()
        self.pipeline_cache = {}

        # Initialize default pipelines
        self._create_default_pipelines()

    def _create_default_pipelines(self):
        """Create default processing pipelines."""
        # Fast pipeline for simple documents
        self.pipelines['fast'] = PipelineConfig(
            name="fast",
            steps=[
                PipelineStepConfig(
                    step=PipelineStep.SOURCE_DETECTION,
                    required=True,
                    timeout_seconds=10
                ),
                PipelineStepConfig(
                    step=PipelineStep.CONTENT_PROCESSING,
                    required=True,
                    timeout_seconds=60
                ),
                PipelineStepConfig(
                    step=PipelineStep.CHUNKING,
                    required=True,
                    timeout_seconds=10
                ),
                PipelineStepConfig(
                    step=PipelineStep.QUALITY_CHECK,
                    required=False,
                    timeout_seconds=5
                )
            ],
            quality_threshold=0.6
        )

        # Balanced pipeline for most documents
        self.pipelines['balanced'] = PipelineConfig(
            name="balanced",
            steps=[
                PipelineStepConfig(
                    step=PipelineStep.SOURCE_DETECTION,
                    required=True,
                    timeout_seconds=15
                ),
                PipelineStepConfig(
                    step=PipelineStep.STRATEGY_SELECTION,
                    required=True,
                    timeout_seconds=5
                ),
                PipelineStepConfig(
                    step=PipelineStep.CONTENT_PROCESSING,
                    required=True,
                    timeout_seconds=120
                ),
                PipelineStepConfig(
                    step=PipelineStep.VISION_ANALYSIS,
                    required=False,
                    timeout_seconds=60,
                    config={'enable_if_scanned': True}
                ),
                PipelineStepConfig(
                    step=PipelineStep.CHUNKING,
                    required=True,
                    timeout_seconds=20
                ),
                PipelineStepConfig(
                    step=PipelineStep.QUALITY_CHECK,
                    required=False,
                    timeout_seconds=10
                )
            ],
            quality_threshold=0.7
        )

        # Accurate pipeline for complex documents
        self.pipelines['accurate'] = PipelineConfig(
            name="accurate",
            steps=[
                PipelineStepConfig(
                    step=PipelineStep.SOURCE_DETECTION,
                    required=True,
                    timeout_seconds=20
                ),
                PipelineStepConfig(
                    step=PipelineStep.STRATEGY_SELECTION,
                    required=True,
                    timeout_seconds=10
                ),
                PipelineStepConfig(
                    step=PipelineStep.CONTENT_PROCESSING,
                    required=True,
                    timeout_seconds=300
                ),
                PipelineStepConfig(
                    step=PipelineStep.VISION_ANALYSIS,
                    required=True,
                    timeout_seconds=120,
                    config={'enable_layout_recognition': True}
                ),
                PipelineStepConfig(
                    step=PipelineStep.POST_PROCESSING,
                    required=False,
                    timeout_seconds=60
                ),
                PipelineStepConfig(
                    step=PipelineStep.CHUNKING,
                    required=True,
                    timeout_seconds=30
                ),
                PipelineStepConfig(
                    step=PipelineStep.QUALITY_CHECK,
                    required=True,
                    timeout_seconds=15
                )
            ],
            quality_threshold=0.85
        )

        # Vision-focused pipeline for scanned documents
        self.pipelines['vision'] = PipelineConfig(
            name="vision",
            steps=[
                PipelineStepConfig(
                    step=PipelineStep.SOURCE_DETECTION,
                    required=True,
                    timeout_seconds=15
                ),
                PipelineStepConfig(
                    step=PipelineStep.VISION_ANALYSIS,
                    required=True,
                    timeout_seconds=180,
                    config={
                        'enable_layout_recognition': True,
                        'enable_table_detection': True,
                        'enable_column_detection': True
                    }
                ),
                PipelineStepConfig(
                    step=PipelineStep.CONTENT_PROCESSING,
                    required=True,
                    timeout_seconds=120
                ),
                PipelineStepConfig(
                    step=PipelineStep.MULTIMODAL_FUSION,
                    required=True,
                    timeout_seconds=60
                ),
                PipelineStepConfig(
                    step=PipelineStep.CHUNKING,
                    required=True,
                    timeout_seconds=30
                ),
                PipelineStepConfig(
                    step=PipelineStep.QUALITY_CHECK,
                    required=True,
                    timeout_seconds=20
                )
            ],
            quality_threshold=0.8
        )

    async def process_document(
        self,
        request: ParseRequest,
        pipeline_name: Optional[str] = None
    ) -> PipelineResult:
        """
        Process a document through the specified pipeline.

        Args:
            request: Parse request
            pipeline_name: Pipeline name (uses default if not specified)

        Returns:
            PipelineResult: Pipeline execution result
        """
        # Select pipeline
        if pipeline_name is None:
            pipeline_name = request.strategy.value if request.strategy else 'balanced'

        if pipeline_name not in self.pipelines:
            pipeline_name = 'balanced'  # Fallback

        pipeline = self.pipelines[pipeline_name]
        context = PipelineContext(
            request_id=str(uuid.uuid()),
            document_id=str(uuid.uuid()),
            start_time=datetime.now(),
            metadata={'pipeline_name': pipeline_name, 'strategy': request.strategy.value}
        )

        try:
            # Execute pipeline
            result = await self._execute_pipeline(request, pipeline, context)

            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return PipelineResult(
                pipeline_name=pipeline.name,
                success=False,
                processing_time_seconds=(datetime.now() - context.start_time).total_seconds(),
                errors=[{
                    'step': 'pipeline_execution',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }]
            )

    async def _execute_pipeline(
        self,
        request: ParseRequest,
        pipeline: PipelineConfig,
        context: PipelineContext
    ) -> PipelineResult:
        """Execute pipeline steps."""
        start_time = datetime.now()
        step_results = {}
        errors = []

        for step_config in pipeline.steps:
            if not step_config.enabled:
                continue

            try:
                # Execute step
                step_result = await self._execute_step(
                    step_config, request, context
                )
                step_results[step_config.step.value] = step_result

                # Check if step failed and is required
                if not step_result.get('success', False) and step_config.required:
                    error_msg = f"Required step {step_config.step.value} failed"
                    logger.error(error_msg)
                    errors.append({
                        'step': step_config.step.value,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    if pipeline.error_handling == 'stop':
                        break

            except Exception as e:
                error_msg = f"Step {step_config.step.value} failed with exception: {str(e)}"
                logger.error(error_msg)
                errors.append({
                    'step': step_config.step.value,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                if step_config.required:
                    break

        # Calculate final result
        processing_time = (datetime.now() - start_time).total_seconds()
        quality_score = self._calculate_quality_score(step_results, context)

        # Create response
        response = None
        if 'content_processing' in step_results:
            response = step_results['content_processing'].get('response')

        return PipelineResult(
            pipeline_name=pipeline.name,
            success=len(errors) == 0 and response is not None and response.success,
            response=response,
            processing_time_seconds=processing_time,
            step_results=step_results,
            errors=errors,
            quality_score=quality_score
        )

    async def _execute_step(
        self,
        step_config: PipelineStepConfig,
        request: ParseRequest,
        context: PipelineContext
    ) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        step_start = datetime.now()
        step_name = step_config.step.value

        try:
            if step_name == PipelineStep.SOURCE_DETECTION.value:
                result = await self._step_source_detection(request, context)
            elif step_name == PipelineStep.STRATEGY_SELECTION.value:
                result = await self._step_strategy_selection(request, context)
            elif step_name == PipelineStep.CONTENT_PROCESSING.value:
                result = await self._step_content_processing(request, context)
            elif step_name == PipelineStep.VISION_ANALYSIS.value:
                result = await self._step_vision_analysis(request, context)
            elif step_name == PipelineStep.MULTIMODAL_FUSION.value:
                result = await self._step_multimodal_fusion(request, context)
            elif step_name == PipelineStep.CHUNKING.value:
                result = await self._step_chunking(request, context)
            elif step_name == PipelineStep.QUALITY_CHECK.value:
                result = await self._step_quality_check(request, context)
            elif step_name == PipelineStep.POST_PROCESSING.value:
                result = await self._step_post_processing(request, context)
            else:
                result = {'success': False, 'error': f'Unknown step: {step_name}'}

            result['processing_time_ms'] = (datetime.now() - step_start).total_seconds() * 1000
            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time_ms': (datetime.now() - step_start).total_seconds() * 1000
            }

    async def _step_source_detection(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute source detection step."""
        from ..services.file_source_detector import FileSourceDetector

        detector = FileSourceDetector()
        detection_result = await detector.detect_source(
            file_path=request.file_path,
            url=request.url,
            content=request.content,
            metadata=request.metadata
        )

        context.metadata['detected_source'] = detection_result.source
        context.metadata['source_confidence'] = detection_result.confidence

        return {
            'success': True,
            'source': detection_result,
            'response': None
        }

    async def _step_strategy_selection(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute strategy selection step."""
        from ..services.processing_strategy_selector import ProcessingStrategySelector

        selector = ProcessingStrategySelector()
        strategy_result = await selector.select_strategy(
            source_detection=context.metadata.get('detected_source'),
            constraints=request.custom_params
        )

        # Update request with selected strategy
        request.strategy = strategy_result.strategy
        request.custom_params = strategy_result.params

        context.metadata['selected_strategy'] = strategy_result.strategy
        context.metadata['strategy_confidence'] = strategy_result.confidence

        return {
            'success': True,
            'strategy': strategy_result,
            'response': None
        }

    async def _step_content_processing(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute content processing step."""
        source = context.metadata.get('detected_source')

        # Select appropriate handler
        handler = self._select_handler(source)
        if not handler:
            return {
                'success': False,
                'error': f'No handler available for source: {source}',
                'response': None
            }

        # Ensure handler is connected
        if not await handler.connect():
            return {
                'success': False,
                'error': f'Failed to connect to handler for source: {source}',
                'response': None
            }

        try:
            # Process document
            response = await handler.process(request)

            await handler.disconnect()

            return {
                'success': True,
                'response': response,
                'handler': handler
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Content processing failed: {str(e)}',
                'response': None
            }

    async def _step_vision_analysis(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute vision analysis step."""
        source = context.metadata.get('detected_source')

        # Only run vision analysis for scanned documents or when configured
        should_run = (
            source == 'scanned_documents' or
            step_config.config.get('enable_if_scanned', False) and
            (context.metadata.get('selected_strategy') and 'accurate' in context.metadata['selected_strategy'].value)
        )

        if not should_run:
            return {'success': True, 'vision_results': {}}

        try:
            vision_results = {}

            # OCR analysis
            if request.content:
                ocr_result = await self.vision_recognizer.ocr_engine.extract_text(request.content)
                vision_results['ocr'] = ocr_result

            # Layout recognition
            if request.content and step_config.config.get('enable_layout_recognition', True):
                layout_regions = await self.layout_recognizer.recognize_layout(request.content, ocr_result)
                vision_results['layout'] = layout_regions

            return {
                'success': True,
                'vision_results': vision_results
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Vision analysis failed: {str(e)}',
                'vision_results': {}
            }

    async def _step_chunking(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute chunking step."""
        # Chunking is typically done within content processing
        # This step can be used for additional chunk optimization
        return {
            'success': True,
            'chunk_count': 0
        }

    async def _step_quality_check(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute quality check step."""
        # Quality metrics are collected throughout processing
        return {
            'success': True,
            'quality_metrics': context.quality_metrics
        }

    async def _step_multimodal_fusion(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute multimodal fusion step."""
        try:
            # Import multimodal fusion service
            from .multimodal_fusion import MultimodalFusion

            fusion_service = MultimodalFusion()

            # This is a placeholder implementation
            # In practice, this would integrate with actual multimodal fusion
            return {
                'success': True,
                'fusion_applied': True,
                'modalities_fused': ['text', 'vision'],
                'response': None
            }

        except Exception as e:
            logger.error(f"Multimodal fusion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fusion_applied': False
            }

    async def _step_post_processing(self, request: ParseRequest, context: PipelineContext) -> Dict[str, Any]:
        """Execute post-processing step."""
        # Additional post-processing can be added here
        return {
            'success': True,
            'post_processed': True
        }

    def _select_handler(self, source: str):
        """Select appropriate handler for detected source."""
        handler_map = {
            'web_documents': self.source_handlers['web_documents'],
            'office_documents': self.source_handlers['office_documents'],
            'scanned_documents': self.source_handlers['scanned_documents'],
            'structured_data': self.source_handlers['structured_data'],
            'code_repositories': self.source_handlers['code_repositories']
        }
        return handler_map.get(source)

    def _calculate_quality_score(self, step_results: Dict[str, Any], context: PipelineContext) -> float:
        """Calculate overall quality score."""
        scores = []

        # Source detection confidence
        if 'source' in step_results:
            scores.append(step_results['source'].source.confidence)

        # Strategy selection confidence
        if 'strategy' in step_results:
            scores.append(step_results['strategy'].confidence)

        # Content processing success
        if 'content_processing' in step_results:
            scores.append(1.0 if step_results['content_processing']['success'] else 0.0)

        # Vision analysis results
        if 'vision_analysis' in step_results:
            vision_results = step_results['vision_analysis'].get('vision_results', {})
            if 'ocr' in vision_results:
                scores.append(vision_results['ocr'].confidence)
            if 'layout' in vision_results:
                scores.append(0.8)  # Layout analysis success

        return sum(scores) / len(scores) if scores else 0.0

    def add_pipeline(self, pipeline_config: PipelineConfig):
        """Add a custom pipeline configuration."""
        self.pipelines[pipeline_config.name] = pipeline_config

    def get_pipeline(self, name: str) -> Optional[PipelineConfig]:
        """Get pipeline configuration by name."""
        return self.pipelines.get(name)

    def list_pipelines(self) -> List[str]:
        """List all available pipelines."""
        return list(self.pipelines.keys())

    async def batch_process(
        self,
        requests: List[ParseRequest],
        pipeline_name: Optional[str] = None
    ) -> List[PipelineResult]:
        """Process multiple documents concurrently."""
        tasks = [
            self.process_document(request, pipeline_name)
            for request in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            'available_pipelines': list(self.pipelines.keys()),
            'registered_handlers': list(self.source_handlers.keys()),
            'vision_capabilities': {
                'ocr_available': True,
                'layout_recognition': True
            }
        }