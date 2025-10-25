"""
Custom Processor Plugin Interface

This module defines the plugin interface for extending document processing
capabilities, inspired by RAGFlow's extensible architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import logging

from ..interfaces.source_interface import ParseRequest, ParseResponse, Content

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Plugin type categories."""
    SOURCE_HANDLER = "source_handler"
    CONTENT_PROCESSOR = "content_processor"
    VISION_ANALYZER = "vision_analyzer"
    QUALITY_CHECKER = "quality_checker"
    OUTPUT_FORMATTER = "output_formatter"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    ENRICHER = "enricher"


class PluginStatus(str, Enum):
    """Plugin status states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    configuration_schema: Optional[Dict[str, Any]] = None
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PluginContext:
    """Plugin execution context."""
    request_id: str
    plugin_id: str
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_stats: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class PluginResult:
    """Plugin execution result."""
    success: bool
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: Optional[float] = None
    confidence: float = 0.0


class BasePlugin(ABC):
    """
    Base class for all document processing plugins.

    All plugins must inherit from this class and implement the required methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration."""
        self.config = config or {}
        self.metadata = self.get_metadata()
        self.status = PluginStatus.INACTIVE
        self._initialized = False
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.

        Returns:
            PluginMetadata: Plugin metadata information
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources.

        Returns:
            bool: True if cleanup successful
        """
        pass

    @abstractmethod
    async def process(
        self,
        data: Any,
        context: PluginContext
    ) -> PluginResult:
        """
        Process data with the plugin.

        Args:
            data: Input data to process
            context: Plugin execution context

        Returns:
            PluginResult: Processing result
        """
        pass

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        # Default validation - can be overridden
        return True

    async def get_supported_formats(self) -> List[str]:
        """
        Get list of supported formats.

        Returns:
            List[str]: Supported format list
        """
        return self.metadata.supported_formats

    async def get_capabilities(self) -> List[str]:
        """
        Get plugin capabilities.

        Returns:
            List[str]: Plugin capabilities list
        """
        return self.metadata.capabilities

    def get_status(self) -> PluginStatus:
        """Get current plugin status."""
        return self.status

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin execution statistics."""
        avg_execution_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0 else 0.0
        )

        return {
            'execution_count': self._execution_count,
            'total_execution_time': self._total_execution_time,
            'average_execution_time': avg_execution_time,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._execution_count, 1),
            'status': self.status.value
        }

    async def health_check(self) -> bool:
        """
        Perform plugin health check.

        Returns:
            bool: True if plugin is healthy
        """
        return self.status == PluginStatus.ACTIVE

    def _update_stats(self, execution_time: float, success: bool):
        """Update internal statistics."""
        self._execution_count += 1
        self._total_execution_time += execution_time
        if not success:
            self._error_count += 1


class SourceHandlerPlugin(BasePlugin):
    """Base class for source handler plugins."""

    @abstractmethod
    async def can_handle(self, request: ParseRequest) -> bool:
        """
        Check if plugin can handle the given request.

        Args:
            request: Parse request to evaluate

        Returns:
            bool: True if plugin can handle the request
        """
        pass

    @abstractmethod
    async def process_document(self, request: ParseRequest) -> ParseResponse:
        """
        Process document from source.

        Args:
            request: Parse request

        Returns:
            ParseResponse: Processing response
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        if isinstance(data, ParseRequest):
            try:
                response = await self.process_document(data)
                return PluginResult(
                    success=response.success,
                    data=response,
                    confidence=response.confidence or 0.8,
                    metadata={'response_type': 'ParseResponse'}
                )
            except Exception as e:
                return PluginResult(
                    success=False,
                    data=None,
                    errors=[{'error': str(e), 'type': 'processing_error'}]
                )
        else:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': 'Expected ParseRequest', 'type': 'invalid_input'}]
            )


class ContentProcessorPlugin(BasePlugin):
    """Base class for content processor plugins."""

    @abstractmethod
    async def process_content(
        self,
        content: List[Content],
        context: PluginContext
    ) -> List[Content]:
        """
        Process content items.

        Args:
            content: List of content items to process
            context: Processing context

        Returns:
            List[Content]: Processed content items
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        if isinstance(data, list) and all(isinstance(item, Content) for item in data):
            try:
                processed_content = await self.process_content(data, context)
                return PluginResult(
                    success=True,
                    data=processed_content,
                    confidence=0.8,
                    metadata={'processed_count': len(processed_content)}
                )
            except Exception as e:
                return PluginResult(
                    success=False,
                    data=None,
                    errors=[{'error': str(e), 'type': 'processing_error'}]
                )
        else:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': 'Expected List[Content]', 'type': 'invalid_input'}]
            )


class VisionAnalyzerPlugin(BasePlugin):
    """Base class for vision analyzer plugins."""

    @abstractmethod
    async def analyze_image(
        self,
        image_data: Union[bytes, str],
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Analyze image data.

        Args:
            image_data: Image data (bytes or file path)
            context: Analysis context

        Returns:
            Dict[str, Any]: Analysis results
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        if isinstance(data, (bytes, str)):
            try:
                analysis_result = await self.analyze_image(data, context)
                return PluginResult(
                    success=True,
                    data=analysis_result,
                    confidence=analysis_result.get('confidence', 0.7),
                    metadata={'analysis_type': 'vision'}
                )
            except Exception as e:
                return PluginResult(
                    success=False,
                    data=None,
                    errors=[{'error': str(e), 'type': 'analysis_error'}]
                )
        else:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': 'Expected bytes or file path', 'type': 'invalid_input'}]
            )


class QualityCheckerPlugin(BasePlugin):
    """Base class for quality checker plugins."""

    @abstractmethod
    async def check_quality(
        self,
        content: Any,
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Check content quality.

        Args:
            content: Content to check
            context: Quality check context

        Returns:
            Dict[str, Any]: Quality assessment results
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        try:
            quality_result = await self.check_quality(data, context)
            quality_score = quality_result.get('overall_score', 0.5)

            return PluginResult(
                success=True,
                data=quality_result,
                confidence=quality_score,
                metadata={
                    'quality_score': quality_score,
                    'quality_metrics': quality_result.get('metrics', {})
                }
            )
        except Exception as e:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': str(e), 'type': 'quality_check_error'}]
            )


class OutputFormatterPlugin(BasePlugin):
    """Base class for output formatter plugins."""

    @abstractmethod
    async def format_output(
        self,
        data: Any,
        format_type: str,
        context: PluginContext
    ) -> str:
        """
        Format data as output string.

        Args:
            data: Data to format
            format_type: Target format type
            context: Formatting context

        Returns:
            str: Formatted output
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        format_type = context.config.get('format_type', 'json')

        try:
            formatted_output = await self.format_output(data, format_type, context)
            return PluginResult(
                success=True,
                data=formatted_output,
                confidence=0.9,
                metadata={
                    'format_type': format_type,
                    'output_length': len(formatted_output)
                }
            )
        except Exception as e:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': str(e), 'type': 'formatting_error'}]
            )


class PreprocessorPlugin(BasePlugin):
    """Base class for preprocessor plugins."""

    @abstractmethod
    async def preprocess(
        self,
        data: Any,
        context: PluginContext
    ) -> Any:
        """
        Preprocess data before main processing.

        Args:
            data: Input data to preprocess
            context: Preprocessing context

        Returns:
            Any: Preprocessed data
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        try:
            preprocessed_data = await self.preprocess(data, context)
            return PluginResult(
                success=True,
                data=preprocessed_data,
                confidence=0.8,
                metadata={'preprocessing_applied': True}
            )
        except Exception as e:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': str(e), 'type': 'preprocessing_error'}]
            )


class PostprocessorPlugin(BasePlugin):
    """Base class for postprocessor plugins."""

    @abstractmethod
    async def postprocess(
        self,
        data: Any,
        context: PluginContext
    ) -> Any:
        """
        Postprocess data after main processing.

        Args:
            data: Processed data to postprocess
            context: Postprocessing context

        Returns:
            Any: Postprocessed data
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        try:
            postprocessed_data = await self.postprocess(data, context)
            return PluginResult(
                success=True,
                data=postprocessed_data,
                confidence=0.8,
                metadata={'postprocessing_applied': True}
            )
        except Exception as e:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': str(e), 'type': 'postprocessing_error'}]
            )


class TransformerPlugin(BasePlugin):
    """Base class for transformer plugins."""

    @abstractmethod
    async def transform(
        self,
        data: Any,
        transformation_type: str,
        context: PluginContext
    ) -> Any:
        """
        Transform data using specified transformation.

        Args:
            data: Data to transform
            transformation_type: Type of transformation to apply
            context: Transformation context

        Returns:
            Any: Transformed data
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        transformation_type = context.config.get('transformation_type', 'default')

        try:
            transformed_data = await self.transform(data, transformation_type, context)
            return PluginResult(
                success=True,
                data=transformed_data,
                confidence=0.8,
                metadata={
                    'transformation_type': transformation_type,
                    'transformation_applied': True
                }
            )
        except Exception as e:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': str(e), 'type': 'transformation_error'}]
            )


class ValidatorPlugin(BasePlugin):
    """Base class for validator plugins."""

    @abstractmethod
    async def validate(
        self,
        data: Any,
        validation_rules: Dict[str, Any],
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Validate data against specified rules.

        Args:
            data: Data to validate
            validation_rules: Validation rules to apply
            context: Validation context

        Returns:
            Dict[str, Any]: Validation results
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        validation_rules = context.config.get('validation_rules', {})

        try:
            validation_result = await self.validate(data, validation_rules, context)
            is_valid = validation_result.get('valid', False)

            return PluginResult(
                success=is_valid,
                data=validation_result,
                confidence=1.0 if is_valid else 0.0,
                metadata={
                    'validation_applied': True,
                    'rules_count': len(validation_rules)
                }
            )
        except Exception as e:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': str(e), 'type': 'validation_error'}]
            )


class EnricherPlugin(BasePlugin):
    """Base class for enricher plugins."""

    @abstractmethod
    async def enrich(
        self,
        data: Any,
        enrichment_type: str,
        context: PluginContext
    ) -> Any:
        """
        Enrich data with additional information.

        Args:
            data: Data to enrich
            enrichment_type: Type of enrichment to apply
            context: Enrichment context

        Returns:
            Any: Enriched data
        """
        pass

    async def process(self, data: Any, context: PluginContext) -> PluginResult:
        """Process method for base plugin interface."""
        enrichment_type = context.config.get('enrichment_type', 'default')

        try:
            enriched_data = await self.enrich(data, enrichment_type, context)
            return PluginResult(
                success=True,
                data=enriched_data,
                confidence=0.8,
                metadata={
                    'enrichment_type': enrichment_type,
                    'enrichment_applied': True
                }
            )
        except Exception as e:
            return PluginResult(
                success=False,
                data=None,
                errors=[{'error': str(e), 'type': 'enrichment_error'}]
            )


# Plugin factory for creating plugin instances
class PluginFactory:
    """Factory class for creating plugin instances."""

    @staticmethod
    def create_plugin(
        plugin_class: type,
        config: Optional[Dict[str, Any]] = None
    ) -> BasePlugin:
        """
        Create plugin instance from class.

        Args:
            plugin_class: Plugin class to instantiate
            config: Plugin configuration

        Returns:
            BasePlugin: Plugin instance
        """
        return plugin_class(config)

    @staticmethod
    def get_plugin_classes() -> Dict[str, type]:
        """
        Get available plugin classes.

        Returns:
            Dict[str, type]: Mapping of plugin names to classes
        """
        # This would typically use reflection or registry
        # For now, return empty dict - would be populated by plugin discovery
        return {}


# Plugin decorator for registration
def register_plugin(
    name: str,
    plugin_type: PluginType,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    dependencies: Optional[List[str]] = None,
    supported_formats: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator for registering plugins.

    Args:
        name: Plugin name
        plugin_type: Plugin type
        version: Plugin version
        description: Plugin description
        author: Plugin author
        dependencies: Plugin dependencies
        supported_formats: Supported formats
        capabilities: Plugin capabilities
        tags: Plugin tags
    """
    def decorator(cls):
        # Add metadata to class
        cls._plugin_metadata = PluginMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            plugin_type=plugin_type,
            dependencies=dependencies or [],
            supported_formats=supported_formats or [],
            capabilities=capabilities or [],
            tags=tags or []
        )
        return cls

    return decorator