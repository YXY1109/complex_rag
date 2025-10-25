"""
Processing Strategy Selector

This module provides intelligent processing strategy selection based on
file characteristics, source type, and processing requirements.
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ..interfaces.source_interface import (
    FileSource,
    ProcessingStrategy,
    SourceDetectionResult,
    SourceConfig
)


class ProcessingRequirement(str, Enum):
    """Processing requirement types."""
    SPEED = "speed"  # Prioritize processing speed
    ACCURACY = "accuracy"  # Prioritize accuracy
    MEMORY = "memory"  # Optimize for memory usage
    COMPLETENESS = "completeness"  # Extract all possible content
    STRUCTURE = "structure"  # Preserve document structure


@dataclass
class StrategyParams:
    """Parameters for processing strategy."""
    chunk_size: int
    overlap_size: int
    use_ocr: bool
    use_vision: bool
    extract_images: bool
    extract_tables: bool
    preserve_formatting: bool
    parallel_processing: bool
    quality_threshold: float
    timeout_seconds: int


@dataclass
class StrategyRecommendation:
    """Strategy recommendation with rationale."""
    strategy: ProcessingStrategy
    params: StrategyParams
    confidence: float
    rationale: List[str]
    alternatives: List[Tuple[ProcessingStrategy, float]]


class ProcessingStrategySelector:
    """
    Intelligent processing strategy selector.

    Analyzes file characteristics and requirements to recommend
    the optimal processing strategy and parameters.
    """

    def __init__(self, config: Optional[SourceConfig] = None):
        """Initialize the strategy selector."""
        self.config = config or SourceConfig()
        self._setup_strategy_templates()

    def _setup_strategy_templates(self):
        """Setup parameter templates for different strategies."""

        self.strategy_templates = {
            ProcessingStrategy.FAST: StrategyParams(
                chunk_size=1000,
                overlap_size=100,
                use_ocr=False,
                use_vision=False,
                extract_images=False,
                extract_tables=False,
                preserve_formatting=False,
                parallel_processing=True,
                quality_threshold=0.7,
                timeout_seconds=30
            ),
            ProcessingStrategy.BALANCED: StrategyParams(
                chunk_size=800,
                overlap_size=200,
                use_ocr=True,
                use_vision=False,
                extract_images=True,
                extract_tables=True,
                preserve_formatting=True,
                parallel_processing=True,
                quality_threshold=0.8,
                timeout_seconds=60
            ),
            ProcessingStrategy.ACCURATE: StrategyParams(
                chunk_size=500,
                overlap_size=100,
                use_ocr=True,
                use_vision=True,
                extract_images=True,
                extract_tables=True,
                preserve_formatting=True,
                parallel_processing=False,
                quality_threshold=0.95,
                timeout_seconds=120
            ),
            ProcessingStrategy.AUTO: StrategyParams(
                chunk_size=800,
                overlap_size=200,
                use_ocr=True,
                use_vision=True,
                extract_images=True,
                extract_tables=True,
                preserve_formatting=True,
                parallel_processing=True,
                quality_threshold=0.85,
                timeout_seconds=90
            )
        }

        # Source-specific strategy preferences
        self.source_preferences = {
            FileSource.WEB_DOCUMENTS: {
                ProcessingRequirement.SPEED: ProcessingStrategy.FAST,
                ProcessingRequirement.ACCURACY: ProcessingStrategy.BALANCED,
                ProcessingRequirement.STRUCTURE: ProcessingStrategy.ACCURATE,
                'default': ProcessingStrategy.BALANCED
            },
            FileSource.OFFICE_DOCUMENTS: {
                ProcessingRequirement.SPEED: ProcessingStrategy.BALANCED,
                ProcessingRequirement.ACCURACY: ProcessingStrategy.ACCURATE,
                ProcessingRequirement.COMPLETENESS: ProcessingStrategy.ACCURATE,
                ProcessingRequirement.STRUCTURE: ProcessingStrategy.ACCURATE,
                'default': ProcessingStrategy.BALANCED
            },
            FileSource.SCANNED_DOCUMENTS: {
                ProcessingRequirement.SPEED: ProcessingStrategy.BALANCED,
                ProcessingRequirement.ACCURACY: ProcessingStrategy.ACCURATE,
                ProcessingRequirement.COMPLETENESS: ProcessingStrategy.ACCURATE,
                'default': ProcessingStrategy.ACCURATE
            },
            FileSource.STRUCTURED_DATA: {
                ProcessingRequirement.SPEED: ProcessingStrategy.FAST,
                ProcessingRequirement.STRUCTURE: ProcessingStrategy.BALANCED,
                'default': ProcessingStrategy.FAST
            },
            FileSource.CODE_REPOSITORIES: {
                ProcessingRequirement.SPEED: ProcessingStrategy.FAST,
                ProcessingRequirement.STRUCTURE: ProcessingStrategy.FAST,
                'default': ProcessingStrategy.FAST
            },
            FileSource.CUSTOM_SOURCES: {
                'default': ProcessingStrategy.AUTO
            },
            FileSource.UNKNOWN: {
                'default': ProcessingStrategy.AUTO
            }
        }

    async def select_strategy(
        self,
        source_detection: SourceDetectionResult,
        requirements: Optional[List[ProcessingRequirement]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> StrategyRecommendation:
        """
        Select the optimal processing strategy.

        Args:
            source_detection: Source detection result
            requirements: Processing requirements
            constraints: Processing constraints (memory, time, etc.)

        Returns:
            StrategyRecommendation: Recommended strategy with parameters
        """
        requirements = requirements or []
        constraints = constraints or {}

        # Analyze file characteristics
        file_analysis = await self._analyze_file_characteristics(source_detection)

        # Score strategies based on multiple factors
        strategy_scores = await self._score_strategies(
            source_detection,
            file_analysis,
            requirements,
            constraints
        )

        # Select best strategy
        best_strategy, best_score = max(strategy_scores.items(), key=lambda x: x[1])

        # Generate parameters
        params = await self._generate_strategy_parameters(
            best_strategy,
            source_detection,
            file_analysis,
            constraints
        )

        # Generate rationale
        rationale = await self._generate_rationale(
            best_strategy,
            source_detection,
            file_analysis,
            requirements,
            constraints
        )

        # Generate alternatives
        alternatives = [
            (strategy, score)
            for strategy, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
            if score > 0.3  # Only include reasonable alternatives
        ]

        return StrategyRecommendation(
            strategy=best_strategy,
            params=params,
            confidence=best_score,
            rationale=rationale,
            alternatives=alternatives
        )

    async def _analyze_file_characteristics(
        self,
        source_detection: SourceDetectionResult
    ) -> Dict[str, Any]:
        """Analyze file characteristics for strategy selection."""

        characteristics = {
            'file_size': 'unknown',
            'content_type': 'unknown',
            'complexity': 'medium',
            'processing_intensity': 'medium'
        }

        # Extract from metadata
        if source_detection.metadata:
            features = source_detection.metadata.get('features', {})

            # File size estimation
            file_size = source_detection.metadata_features.get('file_size', 0)
            if file_size:
                if file_size < 1024 * 1024:  # < 1MB
                    characteristics['file_size'] = 'small'
                elif file_size < 10 * 1024 * 1024:  # < 10MB
                    characteristics['file_size'] = 'medium'
                else:
                    characteristics['file_size'] = 'large'

            # Content type
            mime_type = features.get('mime_type', '')
            if 'text' in mime_type:
                characteristics['content_type'] = 'text'
            elif 'image' in mime_type:
                characteristics['content_type'] = 'image'
            elif 'pdf' in mime_type or 'office' in mime_type:
                characteristics['content_type'] = 'document'
            elif 'json' in mime_type or 'xml' in mime_type or 'csv' in mime_type:
                characteristics['content_type'] = 'structured'

        # Source-based complexity estimation
        source_complexity = {
            FileSource.WEB_DOCUMENTS: 'low',
            FileSource.STRUCTURED_DATA: 'low',
            FileSource.CODE_REPOSITORIES: 'low',
            FileSource.OFFICE_DOCUMENTS: 'medium',
            FileSource.SCANNED_DOCUMENTS: 'high',
            FileSource.CUSTOM_SOURCES: 'medium',
            FileSource.UNKNOWN: 'medium'
        }

        characteristics['complexity'] = source_complexity.get(
            source_detection.source, 'medium'
        )

        # Processing intensity based on confidence
        if source_detection.confidence < 0.5:
            characteristics['processing_intensity'] = 'high'
        elif source_detection.confidence < 0.8:
            characteristics['processing_intensity'] = 'medium'
        else:
            characteristics['processing_intensity'] = 'low'

        return characteristics

    async def _score_strategies(
        self,
        source_detection: SourceDetectionResult,
        file_analysis: Dict[str, Any],
        requirements: List[ProcessingRequirement],
        constraints: Dict[str, Any]
    ) -> Dict[ProcessingStrategy, float]:
        """Score strategies based on multiple factors."""

        scores = {
            ProcessingStrategy.FAST: 0.0,
            ProcessingStrategy.BALANCED: 0.0,
            ProcessingStrategy.ACCURATE: 0.0,
            ProcessingStrategy.AUTO: 0.0
        }

        # Base scores from source preferences
        source_prefs = self.source_preferences.get(source_detection.source, {})

        if requirements:
            # Use requirements to determine base preference
            for req in requirements:
                if req in source_prefs:
                    preferred_strategy = source_prefs[req]
                    scores[preferred_strategy] += 0.4
        else:
            # Use default preference
            default_strategy = source_prefs.get('default', ProcessingStrategy.AUTO)
            scores[default_strategy] += 0.3

        # Adjust based on file characteristics
        if file_analysis['file_size'] == 'large':
            scores[ProcessingStrategy.FAST] += 0.2
            scores[ProcessingStrategy.BALANCED] += 0.1
        elif file_analysis['file_size'] == 'small':
            scores[ProcessingStrategy.ACCURATE] += 0.2
            scores[ProcessingStrategy.BALANCED] += 0.1

        if file_analysis['complexity'] == 'high':
            scores[ProcessingStrategy.ACCURATE] += 0.3
            scores[ProcessingStrategy.BALANCED] += 0.2
        elif file_analysis['complexity'] == 'low':
            scores[ProcessingStrategy.FAST] += 0.2
            scores[ProcessingStrategy.BALANCED] += 0.1

        # Adjust based on confidence
        if source_detection.confidence < 0.5:
            scores[ProcessingStrategy.ACCURATE] += 0.2
            scores[ProcessingStrategy.AUTO] += 0.1
        elif source_detection.confidence > 0.9:
            scores[ProcessingStrategy.FAST] += 0.2

        # Apply constraints
        memory_constraint = constraints.get('max_memory_mb')
        if memory_constraint and memory_constraint < 1024:  # < 1GB
            scores[ProcessingStrategy.FAST] += 0.2
            scores[ProcessingStrategy.BALANCED] += 0.1
            scores[ProcessingStrategy.ACCURATE] -= 0.1

        time_constraint = constraints.get('max_time_seconds')
        if time_constraint and time_constraint < 60:  # < 1 minute
            scores[ProcessingStrategy.FAST] += 0.3
            scores[ProcessingStrategy.BALANCED] += 0.1
            scores[ProcessingStrategy.ACCURATE] -= 0.2

        # Normalize scores
        max_score = max(scores.values()) if max(scores.values()) > 0 else 1.0
        for strategy in scores:
            scores[strategy] = min(scores[strategy] / max_score, 1.0)

        return scores

    async def _generate_strategy_parameters(
        self,
        strategy: ProcessingStrategy,
        source_detection: SourceDetectionResult,
        file_analysis: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> StrategyParams:
        """Generate specific parameters for the selected strategy."""

        # Start with template
        base_params = self.strategy_templates[strategy]

        # Adjust based on file size
        if file_analysis['file_size'] == 'large':
            base_params.chunk_size = min(base_params.chunk_size * 2, 2000)
            base_params.overlap_size = min(base_params.overlap_size * 2, 400)
        elif file_analysis['file_size'] == 'small':
            base_params.chunk_size = max(base_params.chunk_size // 2, 200)
            base_params.overlap_size = max(base_params.overlap_size // 2, 50)

        # Adjust based on content type
        if file_analysis['content_type'] == 'structured':
            base_params.chunk_size = 2000  # Larger chunks for structured data
            base_params.preserve_formatting = False
        elif file_analysis['content_type'] == 'code':
            base_params.chunk_size = 1500  # Larger chunks for code
            base_params.preserve_formatting = True

        # Apply constraints
        if constraints.get('disable_ocr'):
            base_params.use_ocr = False

        if constraints.get('disable_vision'):
            base_params.use_vision = False

        if constraints.get('max_time_seconds'):
            base_params.timeout_seconds = min(
                base_params.timeout_seconds,
                constraints['max_time_seconds']
            )

        # Adjust for source type
        if source_detection.source == FileSource.SCANNED_DOCUMENTS:
            base_params.use_ocr = True
            base_params.use_vision = True
            base_params.extract_images = True
        elif source_detection.source == FileSource.CODE_REPOSITORIES:
            base_params.use_ocr = False
            base_params.use_vision = False
            base_params.extract_images = False
            base_params.preserve_formatting = True

        return base_params

    async def _generate_rationale(
        self,
        strategy: ProcessingStrategy,
        source_detection: SourceDetectionResult,
        file_analysis: Dict[str, Any],
        requirements: List[ProcessingRequirement],
        constraints: Dict[str, Any]
    ) -> List[str]:
        """Generate rationale for strategy selection."""

        rationale = []

        # Source-based rationale
        rationale.append(f"Selected {strategy.value} strategy based on {source_detection.source.value} source type")

        # Confidence-based rationale
        if source_detection.confidence > 0.9:
            rationale.append("High source detection confidence supports efficient processing")
        elif source_detection.confidence < 0.5:
            rationale.append("Low source detection confidence requires more careful processing")

        # File size rationale
        if file_analysis['file_size'] == 'large':
            rationale.append("Large file size favors faster processing with larger chunks")
        elif file_analysis['file_size'] == 'small':
            rationale.append("Small file size allows for more accurate processing")

        # Complexity rationale
        if file_analysis['complexity'] == 'high':
            rationale.append("High content complexity requires comprehensive processing approach")
        elif file_analysis['complexity'] == 'low':
            rationale.append("Low content complexity allows for streamlined processing")

        # Requirements rationale
        if requirements:
            req_names = [req.value for req in requirements]
            rationale.append(f"Processing requirements: {', '.join(req_names)}")

        # Constraints rationale
        if constraints:
            constraint_list = []
            if 'max_memory_mb' in constraints:
                constraint_list.append(f"memory limit: {constraints['max_memory_mb']}MB")
            if 'max_time_seconds' in constraints:
                constraint_list.append(f"time limit: {constraints['max_time_seconds']}s")
            if constraint_list:
                rationale.append(f"Applied constraints: {', '.join(constraint_list)}")

        return rationale

    async def batch_select_strategies(
        self,
        source_detections: List[SourceDetectionResult],
        requirements: Optional[List[ProcessingRequirement]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[StrategyRecommendation]:
        """Select strategies for multiple files concurrently."""

        tasks = [
            self.select_strategy(detection, requirements, constraints)
            for detection in source_detections
        ]

        return await asyncio.gather(*tasks)

    def get_strategy_template(self, strategy: ProcessingStrategy) -> StrategyParams:
        """Get the parameter template for a strategy."""
        return self.strategy_templates[strategy]