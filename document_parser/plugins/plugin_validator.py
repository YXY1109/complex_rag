"""
Plugin Testing and Validation System

This module provides comprehensive testing and validation capabilities for plugins,
including unit tests, integration tests, performance benchmarks, and compliance checks.
"""

import asyncio
import logging
import time
import inspect
import traceback
from typing import Dict, Any, Optional, List, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import statistics
from pathlib import Path

from .plugin_interface import (
    BasePlugin, PluginType, PluginMetadata, PluginContext, PluginResult,
    SourceHandlerPlugin, ContentProcessorPlugin, VisionAnalyzerPlugin,
    QualityCheckerPlugin, OutputFormatterPlugin, PreprocessorPlugin,
    PostprocessorPlugin, TransformerPlugin, ValidatorPlugin, EnricherPlugin
)
from .plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    status: TestStatus
    duration_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Plugin validation report."""
    plugin_id: str
    plugin_type: PluginType
    validation_level: ValidationLevel
    overall_status: TestStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration_ms: float
    test_results: List[TestResult] = field(default_factory=list)
    compliance_score: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationConfig:
    """Validation configuration."""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_performance_tests: bool = True
    enable_compliance_tests: bool = True
    enable_integration_tests: bool = True
    enable_security_tests: bool = True
    test_timeout_seconds: float = 30.0
    max_memory_mb: int = 512
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_init_time_ms': 5000.0,
        'max_process_time_ms': 1000.0,
        'max_memory_mb': 256.0,
        'min_throughput_ops_per_sec': 10.0
    })
    custom_test_data: Dict[str, Any] = field(default_factory=dict)
    generate_test_report: bool = True
    report_format: str = "json"  # json, html, xml


class PluginTestData:
    """Test data provider for plugin validation."""

    def __init__(self):
        """Initialize test data provider."""
        self._test_data = self._generate_test_data()

    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate comprehensive test data."""
        return {
            'text_samples': [
                "This is a simple test text.",
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "Python is a great programming language.",
                "The quick brown fox jumps over the lazy dog.",
                "In a hole in the ground there lived a hobbit."
            ],
            'document_samples': {
                'simple_text': {
                    'content': "This is a simple document for testing.",
                    'type': 'text'
                },
                'structured': {
                    'content': {
                        'title': 'Test Document',
                        'sections': [
                            {'heading': 'Introduction', 'content': 'This is the introduction.'},
                            {'heading': 'Conclusion', 'content': 'This is the conclusion.'}
                        ]
                    },
                    'type': 'structured'
                },
                'code': {
                    'content': "def hello_world():\n    print('Hello, World!')",
                    'type': 'code',
                    'language': 'python'
                }
            },
            'image_samples': [
                # Placeholder for image test data
                {'format': 'png', 'size': 'small', 'description': 'test image'},
                {'format': 'jpg', 'size': 'medium', 'description': 'test photo'}
            ],
            'configuration_samples': {
                'basic': {'debug': True, 'timeout': 30},
                'advanced': {
                    'debug': True,
                    'timeout': 60,
                    'cache_enabled': True,
                    'max_retries': 3
                }
            }
        }

    def get_test_data(self, data_type: str, category: str = None) -> Any:
        """Get test data by type and category."""
        if data_type not in self._test_data:
            return None

        data = self._test_data[data_type]
        if category and isinstance(data, dict):
            return data.get(category)
        return data


class PluginValidator:
    """
    Comprehensive plugin validation and testing system.

    Features:
    - Multi-level validation (basic, standard, comprehensive, strict)
    - Unit and integration testing
    - Performance benchmarking
    - Compliance checking
    - Security validation
    - Memory usage monitoring
    - Custom test scenarios
    - Detailed reporting
    """

    def __init__(
        self,
        registry: PluginRegistry,
        config: Optional[ValidationConfig] = None
    ):
        """Initialize plugin validator."""
        self.registry = registry
        self.config = config or ValidationConfig()
        self.test_data = PluginTestData()
        self._active_tests: Dict[str, asyncio.Task] = {}
        self._test_history: List[ValidationReport] = []

        # Initialize test methods
        self._test_methods = self._initialize_test_methods()

    def _initialize_test_methods(self) -> Dict[ValidationLevel, List[Callable]]:
        """Initialize test methods by validation level."""
        return {
            ValidationLevel.BASIC: [
                self._test_plugin_metadata,
                self._test_basic_functionality,
                self._test_error_handling
            ],
            ValidationLevel.STANDARD: [
                self._test_plugin_metadata,
                self._test_basic_functionality,
                self._test_error_handling,
                self._test_configuration_handling,
                self._test_resource_cleanup
            ],
            ValidationLevel.COMPREHENSIVE: [
                self._test_plugin_metadata,
                self._test_basic_functionality,
                self._test_error_handling,
                self._test_configuration_handling,
                self._test_resource_cleanup,
                self._test_performance_characteristics,
                self._test_concurrent_usage,
                self._test_edge_cases
            ],
            ValidationLevel.STRICT: [
                self._test_plugin_metadata,
                self._test_basic_functionality,
                self._test_error_handling,
                self._test_configuration_handling,
                self._test_resource_cleanup,
                self._test_performance_characteristics,
                self._test_concurrent_usage,
                self._test_edge_cases,
                self._test_memory_usage,
                self._test_security_compliance,
                self._test_api_compliance
            ]
        }

    async def validate_plugin(
        self,
        plugin_id: str,
        validation_level: Optional[ValidationLevel] = None
    ) -> ValidationReport:
        """
        Validate a plugin.

        Args:
            plugin_id: Plugin ID to validate
            validation_level: Validation level to use

        Returns:
            ValidationReport: Validation results
        """
        level = validation_level or self.config.validation_level
        start_time = time.time()

        # Create validation report
        report = ValidationReport(
            plugin_id=plugin_id,
            plugin_type=self._get_plugin_type(plugin_id),
            validation_level=level,
            overall_status=TestStatus.RUNNING,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            total_duration_ms=0.0
        )

        try:
            # Get plugin
            plugin = self.registry.get_plugin(plugin_id)
            if not plugin:
                report.test_results.append(TestResult(
                    test_name="plugin_availability",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Plugin not found in registry",
                    error="Plugin not available"
                ))
                report.overall_status = TestStatus.FAILED
                return report

            # Get test methods for validation level
            test_methods = self._test_methods.get(level, self._test_methods[ValidationLevel.STANDARD])

            # Execute tests
            for test_method in test_methods:
                test_result = await self._execute_test(test_method, plugin, plugin_id)
                report.test_results.append(test_result)
                report.total_tests += 1

                if test_result.status == TestStatus.PASSED:
                    report.passed_tests += 1
                elif test_result.status == TestStatus.FAILED:
                    report.failed_tests += 1
                elif test_result.status == TestStatus.SKIPPED:
                    report.skipped_tests += 1

            # Calculate overall status
            if report.failed_tests == 0:
                report.overall_status = TestStatus.PASSED
            elif report.passed_tests > 0:
                report.overall_status = TestStatus.FAILED
            else:
                report.overall_status = TestStatus.ERROR

            # Calculate compliance score
            report.compliance_score = self._calculate_compliance_score(report)

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            # Store in history
            self._test_history.append(report)

        except Exception as e:
            logger.error(f"Validation failed for plugin {plugin_id}: {e}")
            report.overall_status = TestStatus.ERROR
            report.test_results.append(TestResult(
                test_name="validation_execution",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Validation execution failed",
                error=str(e)
            ))

        finally:
            report.total_duration_ms = (time.time() - start_time) * 1000

        return report

    async def validate_all_plugins(
        self,
        validation_level: Optional[ValidationLevel] = None
    ) -> Dict[str, ValidationReport]:
        """
        Validate all registered plugins.

        Args:
            validation_level: Validation level to use

        Returns:
            Dict[str, ValidationReport]: Mapping of plugin IDs to validation reports
        """
        plugins = self.registry.list_plugins()
        reports = {}

        for plugin_info in plugins:
            plugin_id = plugin_info['plugin_id']
            try:
                report = await self.validate_plugin(plugin_id, validation_level)
                reports[plugin_id] = report
            except Exception as e:
                logger.error(f"Failed to validate plugin {plugin_id}: {e}")
                # Create error report
                error_report = ValidationReport(
                    plugin_id=plugin_id,
                    plugin_type=PluginType.SOURCE_HANDLER,  # Default
                    validation_level=validation_level or self.config.validation_level,
                    overall_status=TestStatus.ERROR,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    skipped_tests=0,
                    total_duration_ms=0.0
                )
                error_report.test_results.append(TestResult(
                    test_name="validation_error",
                    status=TestStatus.ERROR,
                    duration_ms=0.0,
                    message="Validation execution error",
                    error=str(e)
                ))
                reports[plugin_id] = error_report

        return reports

    async def _execute_test(
        self,
        test_method: Callable,
        plugin: BasePlugin,
        plugin_id: str
    ) -> TestResult:
        """Execute a single test method."""
        test_name = test_method.__name__.replace('_test_', '')
        start_time = time.time()

        try:
            # Execute test with timeout
            result = await asyncio.wait_for(
                test_method(plugin, plugin_id),
                timeout=self.config.test_timeout_seconds
            )

            duration_ms = (time.time() - start_time) * 1000

            if isinstance(result, TestResult):
                return result
            elif isinstance(result, bool):
                return TestResult(
                    test_name=test_name,
                    status=TestStatus.PASSED if result else TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message="Test executed" if result else "Test failed"
                )
            else:
                return TestResult(
                    test_name=test_name,
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message="Test executed successfully",
                    details={'result': result}
                )

        except asyncio.TimeoutError:
            return TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                duration_ms=self.config.test_timeout_seconds * 1000,
                message="Test timed out",
                error=f"Test exceeded timeout of {self.config.test_timeout_seconds}s"
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message="Test execution error",
                error=str(e),
                details={'traceback': traceback.format_exc()}
            )

    async def _test_plugin_metadata(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin metadata completeness and validity."""
        try:
            metadata = plugin.get_metadata()

            # Required fields check
            required_fields = ['name', 'version', 'description', 'author', 'plugin_type']
            missing_fields = [field for field in required_fields if not getattr(metadata, field)]

            if missing_fields:
                return TestResult(
                    test_name="metadata_completeness",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message=f"Missing required metadata fields: {missing_fields}",
                    details={'missing_fields': missing_fields}
                )

            # Version format check
            version = metadata.version
            if not self._is_valid_version(version):
                return TestResult(
                    test_name="metadata_version",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message=f"Invalid version format: {version}",
                    details={'version': version}
                )

            # Plugin type check
            if not isinstance(metadata.plugin_type, PluginType):
                return TestResult(
                    test_name="metadata_plugin_type",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message=f"Invalid plugin type: {metadata.plugin_type}",
                    details={'plugin_type': str(metadata.plugin_type)}
                )

            return TestResult(
                test_name="plugin_metadata",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Plugin metadata is valid",
                details={'metadata': metadata.__dict__}
            )

        except Exception as e:
            return TestResult(
                test_name="plugin_metadata",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Failed to validate plugin metadata",
                error=str(e)
            )

    async def _test_basic_functionality(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test basic plugin functionality."""
        try:
            # Test plugin initialization
            if plugin.get_status() != PluginStatus.ACTIVE:
                init_success = await plugin.initialize()
                if not init_success:
                    return TestResult(
                        test_name="initialization",
                        status=TestStatus.FAILED,
                        duration_ms=0.0,
                        message="Plugin initialization failed"
                    )

            # Test plugin processing
            test_data = self._get_test_data_for_plugin(plugin)
            if test_data:
                context = PluginContext(
                    request_id="test_validation",
                    plugin_id=plugin_id,
                    config={}
                )

                process_result = await plugin.process(test_data, context)
                if not process_result.success:
                    return TestResult(
                        test_name="processing",
                        status=TestStatus.FAILED,
                        duration_ms=0.0,
                        message="Plugin processing failed",
                        details={'errors': process_result.errors}
                    )

            return TestResult(
                test_name="basic_functionality",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Basic functionality test passed",
                details={'test_data_used': str(type(test_data))}
            )

        except Exception as e:
            return TestResult(
                test_name="basic_functionality",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Basic functionality test failed",
                error=str(e)
            )

    async def _test_error_handling(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin error handling capabilities."""
        try:
            # Test with invalid data
            context = PluginContext(
                request_id="test_error",
                plugin_id=plugin_id,
                config={}
            )

            # Process with None data
            result = await plugin.process(None, context)

            if result.success:
                return TestResult(
                    test_name="error_handling",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Plugin should handle invalid data gracefully"
                )

            if not result.errors:
                return TestResult(
                    test_name="error_handling",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Plugin should provide error information"
                )

            return TestResult(
                test_name="error_handling",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Error handling test passed",
                details={'error_count': len(result.errors)}
            )

        except Exception as e:
            return TestResult(
                test_name="error_handling",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Error handling test failed",
                error=str(e)
            )

    async def _test_configuration_handling(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin configuration handling."""
        try:
            # Test with valid configuration
            valid_config = self.test_data.get_test_data('configuration_samples', 'basic')
            config_valid = await plugin.validate_config(valid_config)

            if not config_valid:
                return TestResult(
                    test_name="configuration_handling",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Plugin rejected valid configuration"
                )

            # Test with invalid configuration
            invalid_config = {'invalid_key': 'invalid_value'}
            config_invalid = await plugin.validate_config(invalid_config)

            # Plugin might accept invalid config or reject it - both are acceptable
            # as long as it doesn't crash
            return TestResult(
                test_name="configuration_handling",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Configuration handling test passed",
                details={
                    'valid_config_accepted': config_valid,
                    'invalid_config_accepted': config_invalid
                }
            )

        except Exception as e:
            return TestResult(
                test_name="configuration_handling",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Configuration handling test failed",
                error=str(e)
            )

    async def _test_resource_cleanup(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin resource cleanup."""
        try:
            # Test cleanup method
            cleanup_success = await plugin.cleanup()

            if not cleanup_success:
                return TestResult(
                    test_name="resource_cleanup",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Plugin cleanup failed"
                )

            # Re-initialize for subsequent tests
            await plugin.initialize()

            return TestResult(
                test_name="resource_cleanup",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Resource cleanup test passed"
            )

        except Exception as e:
            return TestResult(
                test_name="resource_cleanup",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Resource cleanup test failed",
                error=str(e)
            )

    async def _test_performance_characteristics(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin performance characteristics."""
        try:
            performance_metrics = {}

            # Test initialization performance
            init_start = time.time()
            await plugin.cleanup()
            await plugin.initialize()
            init_time = (time.time() - init_start) * 1000
            performance_metrics['init_time_ms'] = init_time

            # Test processing performance
            test_data = self._get_test_data_for_plugin(plugin)
            if test_data:
                context = PluginContext(
                    request_id="perf_test",
                    plugin_id=plugin_id,
                    config={}
                )

                process_times = []
                for _ in range(5):  # Run 5 times for average
                    process_start = time.time()
                    await plugin.process(test_data, context)
                    process_times.append((time.time() - process_start) * 1000)

                performance_metrics['avg_process_time_ms'] = statistics.mean(process_times)
                performance_metrics['min_process_time_ms'] = min(process_times)
                performance_metrics['max_process_time_ms'] = max(process_times)

            # Check against thresholds
            threshold_violations = []
            thresholds = self.config.performance_thresholds

            if 'max_init_time_ms' in thresholds and init_time > thresholds['max_init_time_ms']:
                threshold_violations.append(f"Init time {init_time:.2f}ms > {thresholds['max_init_time_ms']}ms")

            if 'avg_process_time_ms' in performance_metrics:
                avg_time = performance_metrics['avg_process_time_ms']
                if 'max_process_time_ms' in thresholds and avg_time > thresholds['max_process_time_ms']:
                    threshold_violations.append(f"Avg process time {avg_time:.2f}ms > {thresholds['max_process_time_ms']}ms")

            if threshold_violations:
                return TestResult(
                    test_name="performance_characteristics",
                    status=TestStatus.FAILED,
                    duration_ms=init_time,
                    message="Performance thresholds exceeded",
                    details={
                        'metrics': performance_metrics,
                        'violations': threshold_violations
                    }
                )

            return TestResult(
                test_name="performance_characteristics",
                status=TestStatus.PASSED,
                duration_ms=init_time,
                message="Performance characteristics test passed",
                details={'metrics': performance_metrics}
            )

        except Exception as e:
            return TestResult(
                test_name="performance_characteristics",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Performance characteristics test failed",
                error=str(e)
            )

    async def _test_concurrent_usage(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin concurrent usage."""
        try:
            test_data = self._get_test_data_for_plugin(plugin)
            if not test_data:
                return TestResult(
                    test_name="concurrent_usage",
                    status=TestStatus.SKIPPED,
                    duration_ms=0.0,
                    message="No suitable test data for concurrent testing"
                )

            # Run concurrent processing tasks
            async def process_task():
                context = PluginContext(
                    request_id=f"concurrent_test_{time.time()}",
                    plugin_id=plugin_id,
                    config={}
                )
                return await plugin.process(test_data, context)

            # Execute 10 concurrent tasks
            tasks = [process_task() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check results
            successful_results = [r for r in results if isinstance(r, PluginResult) and r.success]
            failed_results = [r for r in results if isinstance(r, Exception) or (isinstance(r, PluginResult) and not r.success)]

            if len(failed_results) > len(successful_results):
                return TestResult(
                    test_name="concurrent_usage",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Too many failures in concurrent processing",
                    details={
                        'total_tasks': len(tasks),
                        'successful': len(successful_results),
                        'failed': len(failed_results)
                    }
                )

            return TestResult(
                test_name="concurrent_usage",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Concurrent usage test passed",
                details={
                    'total_tasks': len(tasks),
                    'successful': len(successful_results),
                    'failed': len(failed_results)
                }
            )

        except Exception as e:
            return TestResult(
                test_name="concurrent_usage",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Concurrent usage test failed",
                error=str(e)
            )

    async def _test_edge_cases(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin edge cases."""
        try:
            edge_cases = [
                ("empty_string", ""),
                ("empty_list", []),
                ("empty_dict", {}),
                ("large_string", "x" * 10000),
                ("unicode_text", "Hello 世界 🌍"),
                ("special_chars", "!@#$%^&*()_+-=[]{}|;':\",./<>?")
            ]

            failed_cases = []

            for case_name, test_data in edge_cases:
                try:
                    context = PluginContext(
                        request_id=f"edge_case_{case_name}",
                        plugin_id=plugin_id,
                        config={}
                    )

                    result = await plugin.process(test_data, context)

                    # Edge case processing can fail, but shouldn't crash
                    if isinstance(result, Exception):
                        failed_cases.append(f"{case_name}: crashed with exception")
                        logger.warning(f"Edge case {case_name} caused exception: {result}")

                except Exception as e:
                    # Plugin should handle edge cases gracefully
                    failed_cases.append(f"{case_name}: unhandled exception")
                    logger.warning(f"Edge case {case_name} unhandled exception: {e}")

            if len(failed_cases) > len(edge_cases) / 2:
                return TestResult(
                    test_name="edge_cases",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Too many edge cases failed",
                    details={
                        'total_cases': len(edge_cases),
                        'failed_cases': failed_cases
                    }
                )

            return TestResult(
                test_name="edge_cases",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Edge cases test passed",
                details={
                    'total_cases': len(edge_cases),
                    'failed_cases': failed_cases
                }
            )

        except Exception as e:
            return TestResult(
                test_name="edge_cases",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Edge cases test failed",
                error=str(e)
            )

    async def _test_memory_usage(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin memory usage."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform multiple processing operations
            test_data = self._get_test_data_for_plugin(plugin)
            if test_data:
                context = PluginContext(
                    request_id="memory_test",
                    plugin_id=plugin_id,
                    config={}
                )

                for _ in range(100):
                    await plugin.process(test_data, context)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Check memory threshold
            max_memory = self.config.performance_thresholds.get('max_memory_mb', 256)
            if memory_increase > max_memory:
                return TestResult(
                    test_name="memory_usage",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Memory usage exceeded threshold",
                    details={
                        'initial_memory_mb': initial_memory,
                        'final_memory_mb': final_memory,
                        'memory_increase_mb': memory_increase,
                        'threshold_mb': max_memory
                    }
                )

            return TestResult(
                test_name="memory_usage",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Memory usage test passed",
                details={
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_increase_mb': memory_increase
                }
            )

        except ImportError:
            return TestResult(
                test_name="memory_usage",
                status=TestStatus.SKIPPED,
                duration_ms=0.0,
                message="psutil not available for memory testing"
            )
        except Exception as e:
            return TestResult(
                test_name="memory_usage",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Memory usage test failed",
                error=str(e)
            )

    async def _test_security_compliance(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin security compliance."""
        try:
            security_checks = []

            # Check for dangerous imports (basic check)
            plugin_source = inspect.getsource(plugin.__class__)
            dangerous_imports = ['os.system', 'subprocess.call', 'eval(', 'exec(']
            found_dangerous = [imp for imp in dangerous_imports if imp in plugin_source]

            if found_dangerous:
                security_checks.append(f"Found potentially dangerous imports: {found_dangerous}")

            # Check file system access (basic check)
            file_operations = ['open(', 'file(', 'Path(']
            found_file_ops = [op for op in file_operations if op in plugin_source]

            # File operations might be legitimate, so just log them
            if found_file_ops:
                security_checks.append(f"Found file operations: {found_file_ops}")

            # Test with malicious input
            malicious_inputs = [
                "../../etc/passwd",
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --"
            ]

            for malicious_input in malicious_inputs:
                try:
                    context = PluginContext(
                        request_id="security_test",
                        plugin_id=plugin_id,
                        config={}
                    )

                    result = await plugin.process(malicious_input, context)

                    # Plugin should handle malicious input safely
                    if result.success and isinstance(result.data, str) and len(str(result.data)) > 1000:
                        security_checks.append(f"Potential injection vulnerability with input: {malicious_input}")

                except Exception:
                    # Exception is acceptable for malicious input
                    pass

            if security_checks:
                return TestResult(
                    test_name="security_compliance",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="Security compliance issues found",
                    details={'security_issues': security_checks}
                )

            return TestResult(
                test_name="security_compliance",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="Security compliance test passed",
                details={'security_checks': len(security_checks)}
            )

        except Exception as e:
            return TestResult(
                test_name="security_compliance",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="Security compliance test failed",
                error=str(e)
            )

    async def _test_api_compliance(self, plugin: BasePlugin, plugin_id: str) -> TestResult:
        """Test plugin API compliance."""
        try:
            compliance_issues = []

            # Check required methods
            required_methods = ['get_metadata', 'initialize', 'cleanup', 'process']
            for method in required_methods:
                if not hasattr(plugin, method):
                    compliance_issues.append(f"Missing required method: {method}")

            # Check method signatures
            if hasattr(plugin, 'get_metadata'):
                sig = inspect.signature(plugin.get_metadata)
                if len(sig.parameters) > 0:
                    compliance_issues.append("get_metadata should take no parameters")

            if hasattr(plugin, 'process'):
                sig = inspect.signature(plugin.process)
                if len(sig.parameters) < 2:
                    compliance_issues.append("process should take at least 2 parameters (data, context)")

            # Check return types
            if hasattr(plugin, 'get_metadata'):
                try:
                    metadata = plugin.get_metadata()
                    if not isinstance(metadata, PluginMetadata):
                        compliance_issues.append("get_metadata should return PluginMetadata instance")
                except Exception:
                    compliance_issues.append("get_metadata raised an exception")

            if compliance_issues:
                return TestResult(
                    test_name="api_compliance",
                    status=TestStatus.FAILED,
                    duration_ms=0.0,
                    message="API compliance issues found",
                    details={'compliance_issues': compliance_issues}
                )

            return TestResult(
                test_name="api_compliance",
                status=TestStatus.PASSED,
                duration_ms=0.0,
                message="API compliance test passed"
            )

        except Exception as e:
            return TestResult(
                test_name="api_compliance",
                status=TestStatus.ERROR,
                duration_ms=0.0,
                message="API compliance test failed",
                error=str(e)
            )

    def _get_plugin_type(self, plugin_id: str) -> PluginType:
        """Get plugin type from registry."""
        plugin_info = self.registry.get_plugin_info(plugin_id)
        if plugin_info and 'metadata' in plugin_info:
            return PluginType(plugin_info['metadata']['plugin_type'])
        return PluginType.SOURCE_HANDLER  # Default

    def _get_test_data_for_plugin(self, plugin: BasePlugin) -> Any:
        """Get appropriate test data for plugin type."""
        if isinstance(plugin, SourceHandlerPlugin):
            return self.test_data.get_test_data('document_samples', 'simple_text')
        elif isinstance(plugin, ContentProcessorPlugin):
            return self.test_data.get_test_data('text_samples', 0)
        elif isinstance(plugin, VisionAnalyzerPlugin):
            return b"test_image_data"  # Placeholder
        elif isinstance(plugin, QualityCheckerPlugin):
            return "test content for quality checking"
        elif isinstance(plugin, OutputFormatterPlugin):
            return {"data": "test data", "format": "json"}
        elif isinstance(plugin, (PreprocessorPlugin, PostprocessorPlugin)):
            return "test data for processing"
        elif isinstance(plugin, TransformerPlugin):
            return "test data for transformation"
        elif isinstance(plugin, ValidatorPlugin):
            return "test data for validation"
        elif isinstance(plugin, EnricherPlugin):
            return "test data for enrichment"
        else:
            return "generic test data"

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid."""
        import re
        # Simple semantic version check
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
        return bool(re.match(pattern, version))

    def _calculate_compliance_score(self, report: ValidationReport) -> float:
        """Calculate compliance score from validation report."""
        if report.total_tests == 0:
            return 0.0

        passed_ratio = report.passed_tests / report.total_tests

        # Bonus points for comprehensive testing
        level_bonus = {
            ValidationLevel.BASIC: 0.0,
            ValidationLevel.STANDARD: 0.1,
            ValidationLevel.COMPREHENSIVE: 0.15,
            ValidationLevel.STRICT: 0.2
        }.get(report.validation_level, 0.0)

        score = min(1.0, passed_ratio + level_bonus)
        return round(score, 2)

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        for test_result in report.test_results:
            if test_result.status == TestStatus.FAILED:
                if "metadata" in test_result.test_name:
                    recommendations.append("Complete plugin metadata with all required fields")
                elif "performance" in test_result.test_name:
                    recommendations.append("Optimize plugin performance to meet threshold requirements")
                elif "error_handling" in test_result.test_name:
                    recommendations.append("Improve error handling and provide meaningful error messages")
                elif "configuration" in test_result.test_name:
                    recommendations.append("Enhance configuration validation and handling")
                elif "security" in test_result.test_name:
                    recommendations.append("Address security compliance issues")
                elif "api_compliance" in test_result.test_name:
                    recommendations.append("Fix API compliance issues to match plugin interface")

        if report.compliance_score < 0.8:
            recommendations.append("Consider additional testing to improve compliance score")

        if not recommendations:
            recommendations.append("Plugin validation passed successfully")

        return recommendations

    def get_test_history(
        self,
        plugin_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ValidationReport]:
        """Get test history."""
        history = self._test_history

        if plugin_id:
            history = [r for r in history if r.plugin_id == plugin_id]

        return history[-limit:]

    def generate_report(self, report: ValidationReport, format: str = "json") -> str:
        """Generate formatted validation report."""
        if format == "json":
            return self._generate_json_report(report)
        elif format == "html":
            return self._generate_html_report(report)
        else:
            return str(report)

    def _generate_json_report(self, report: ValidationReport) -> str:
        """Generate JSON format report."""
        report_dict = {
            'plugin_id': report.plugin_id,
            'plugin_type': report.plugin_type.value,
            'validation_level': report.validation_level.value,
            'overall_status': report.overall_status.value,
            'compliance_score': report.compliance_score,
            'total_tests': report.total_tests,
            'passed_tests': report.passed_tests,
            'failed_tests': report.failed_tests,
            'skipped_tests': report.skipped_tests,
            'total_duration_ms': report.total_duration_ms,
            'timestamp': report.timestamp.isoformat(),
            'performance_metrics': report.performance_metrics,
            'recommendations': report.recommendations,
            'test_results': []
        }

        for test_result in report.test_results:
            test_dict = {
                'test_name': test_result.test_name,
                'status': test_result.status.value,
                'duration_ms': test_result.duration_ms,
                'message': test_result.message,
                'timestamp': test_result.timestamp.isoformat()
            }

            if test_result.details:
                test_dict['details'] = test_result.details

            if test_result.error:
                test_dict['error'] = test_result.error

            report_dict['test_results'].append(test_dict)

        return json.dumps(report_dict, indent=2, ensure_ascii=False)

    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML format report."""
        # Simple HTML report template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plugin Validation Report - {report.plugin_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .status-passed {{ color: green; font-weight: bold; }}
                .status-failed {{ color: red; font-weight: bold; }}
                .status-error {{ color: orange; font-weight: bold; }}
                .status-skipped {{ color: blue; font-weight: bold; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .test-passed {{ border-left-color: green; }}
                .test-failed {{ border-left-color: red; }}
                .test-error {{ border-left-color: orange; }}
                .test-skipped {{ border-left-color: blue; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Plugin Validation Report</h1>
                <p><strong>Plugin ID:</strong> {report.plugin_id}</p>
                <p><strong>Plugin Type:</strong> {report.plugin_type.value}</p>
                <p><strong>Validation Level:</strong> {report.validation_level.value}</p>
                <p><strong>Overall Status:</strong> <span class="status-{report.overall_status.value}">{report.overall_status.value.upper()}</span></p>
                <p><strong>Compliance Score:</strong> {report.compliance_score:.2f}</p>
                <p><strong>Total Tests:</strong> {report.total_tests}</p>
                <p><strong>Passed:</strong> {report.passed_tests}</p>
                <p><strong>Failed:</strong> {report.failed_tests}</p>
                <p><strong>Skipped:</strong> {report.skipped_tests}</p>
                <p><strong>Duration:</strong> {report.total_duration_ms:.2f}ms</p>
                <p><strong>Timestamp:</strong> {report.timestamp.isoformat()}</p>
            </div>

            <h2>Test Results</h2>
        """

        for test_result in report.test_results:
            html += f"""
            <div class="test-result test-{test_result.status.value}">
                <h3>{test_result.test_name.replace('_', ' ').title()}</h3>
                <p><strong>Status:</strong> <span class="status-{test_result.status.value}">{test_result.status.value.upper()}</span></p>
                <p><strong>Duration:</strong> {test_result.duration_ms:.2f}ms</p>
                <p><strong>Message:</strong> {test_result.message}</p>
            """

            if test_result.error:
                html += f"<p><strong>Error:</strong> {test_result.error}</p>"

            if test_result.details:
                html += f"<p><strong>Details:</strong> {test_result.details}</p>"

            html += "</div>"

        if report.recommendations:
            html += "<h2>Recommendations</h2><ul>"
            for rec in report.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"

        html += """
        </body>
        </html>
        """

        return html

    async def cleanup(self):
        """Cleanup validator resources."""
        # Cancel any active tests
        for test_id, task in self._active_tests.items():
            if not task.done():
                task.cancel()

        self._active_tests.clear()