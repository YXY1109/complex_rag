"""
Structured Logging Configuration

This module provides comprehensive logging configuration using Loguru
with structured JSON output, multiple handlers, and log rotation.
"""

import sys
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import asyncio
from dataclasses import dataclass, asdict

try:
    from loguru import logger
except ImportError:
    logger = None


@dataclass
class LoggingConfig:
    """Logging configuration model."""
    # Basic settings
    level: str = "INFO"
    format: str = "json"  # "json" or "text"

    # File settings
    log_dir: str = "logs"
    log_file: str = "app.log"
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "gz"

    # Console settings
    console_format: str = "text"
    console_colors: bool = True

    # Structured logging
    include_extra: bool = True
    extra_fields: Dict[str, Any] = None

    # Performance
    enqueue: bool = True
    catch: bool = True

    # Filter settings
    filter_modules: List[str] = None
    exclude_modules: List[str] = None

    # Custom handlers
    custom_handlers: List[Dict[str, Any]] = None


class StructuredLogger:
    """
    Structured logger wrapper using Loguru.

    Provides consistent logging format with structured JSON output,
    automatic correlation IDs, and performance monitoring.
    """

    def __init__(self, config: LoggingConfig):
        """
        Initialize structured logger.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.logger = logger
        self._handlers = []

        if self.logger is None:
            raise ImportError("loguru is required for structured logging")

        # Configure logger
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure Loguru logger with custom handlers."""
        # Remove default handler
        self.logger.remove()

        # Add console handler
        self._add_console_handler()

        # Add file handler
        self._add_file_handler()

        # Add custom handlers
        if self.config.custom_handlers:
            self._add_custom_handlers()

    def _add_console_handler(self) -> None:
        """Add console handler."""
        if self.config.console_format == "json":
            format_str = self._get_json_format()
        else:
            format_str = self._get_text_format(colors=self.config.console_colors)

        handler_id = self.logger.add(
            sys.stdout,
            level=self.config.level,
            format=format_str,
            enqueue=self.config.enqueue,
            catch=self.config.catch,
            filter=self._create_filter()
        )

        self._handlers.append(handler_id)

    def _add_file_handler(self) -> None:
        """Add file handler with rotation."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / self.config.log_file

        handler_id = self.logger.add(
            str(log_path),
            level=self.config.level,
            format=self._get_json_format(),
            rotation=self.config.rotation,
            retention=self.config.retention,
            compression=self.config.compression,
            enqueue=self.config.enqueue,
            catch=self.config.catch,
            filter=self._create_filter(),
            serialize=True  # Always JSON for files
        )

        self._handlers.append(handler_id)

    def _add_custom_handlers(self) -> None:
        """Add custom handlers from configuration."""
        for handler_config in self.config.custom_handlers:
            try:
                handler_id = self.logger.add(**handler_config)
                self._handlers.append(handler_id)
            except Exception as e:
                self.logger.error(f"Failed to add custom handler: {e}")

    def _get_json_format(self) -> str:
        """Get JSON log format."""
        return (
            "{{"
            '"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"logger": "{name}", '
            '"message": "{message}", '
            '"module": "{module}", '
            '"function": "{function}", '
            '"line": {line}'
            "}}"
        )

    def _get_text_format(self, colors: bool = True) -> str:
        """Get text log format."""
        if colors:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        else:
            return (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )

    def _create_filter(self) -> Optional[callable]:
        """Create log filter based on configuration."""
        if not self.config.filter_modules and not self.config.exclude_modules:
            return None

        def log_filter(record):
            # Check include modules
            if self.config.filter_modules:
                if not any(record["name"].startswith(module) for module in self.config.filter_modules):
                    return False

            # Check exclude modules
            if self.config.exclude_modules:
                if any(record["name"].startswith(module) for module in self.config.exclude_modules):
                    return False

            return True

        return log_filter

    def bind(self, **kwargs) -> Any:
        """
        Bind context to logger.

        Args:
            **kwargs: Context key-value pairs

        Returns:
            Bound logger
        """
        return self.logger.bind(**kwargs)

    def patch(self, **kwargs) -> Any:
        """
        Patch logger with additional context.

        Args:
            **kwargs: Context key-value pairs

        Returns:
            Patched logger
        """
        return self.logger.patch(lambda record: record["extra"].update(kwargs))

    def get_logger(self, name: Optional[str] = None) -> Any:
        """
        Get logger with optional name.

        Args:
            name: Logger name (optional)

        Returns:
            Logger instance
        """
        if name:
            return self.logger.bind(name=name)
        return self.logger

    def configure_context(self, **context) -> Any:
        """
        Configure global context for all logs.

        Args:
            **context: Global context key-value pairs

        Returns:
            Configured logger
        """
        # Add global extra fields
        if self.config.extra_fields:
            context.update(self.config.extra_fields)

        return self.logger.configure(extra=context)

    def remove_handlers(self) -> None:
        """Remove all handlers."""
        for handler_id in self._handlers:
            self.logger.remove(handler_id)
        self._handlers.clear()

    def add_handler(self, handler_config: Dict[str, Any]) -> int:
        """
        Add a new handler.

        Args:
            handler_config: Handler configuration

        Returns:
            Handler ID
        """
        handler_id = self.logger.add(**handler_config)
        self._handlers.append(handler_id)
        return handler_id


class RequestLogger:
    """
    Request logger for HTTP/API requests with structured logging.
    """

    def __init__(self, base_logger: StructuredLogger):
        """
        Initialize request logger.

        Args:
            base_logger: Base structured logger
        """
        self.logger = base_logger.bind(component="request")

    def log_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Log incoming request.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            body: Request body (optional)
            request_id: Request ID (optional)
            user_id: User ID (optional)
        """
        extra = {
            "event_type": "request_start",
            "method": method,
            "path": path,
            "headers": dict(headers),
            "request_id": request_id,
            "user_id": user_id
        }

        if body:
            extra["body_size"] = len(body)

        self.logger.info("Request started", **extra)

    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time_ms: float,
        response_size: Optional[int] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Log response.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            response_time_ms: Response time in milliseconds
            response_size: Response size in bytes (optional)
            request_id: Request ID (optional)
            user_id: User ID (optional)
        """
        extra = {
            "event_type": "request_complete",
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "request_id": request_id,
            "user_id": user_id
        }

        if response_size:
            extra["response_size"] = response_size

        level = "info" if status_code < 400 else "warning" if status_code < 500 else "error"
        getattr(self.logger, level)(f"Request completed with status {status_code}", **extra)

    def log_error(
        self,
        method: str,
        path: str,
        error: Exception,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Log request error.

        Args:
            method: HTTP method
            path: Request path
            error: Exception that occurred
            request_id: Request ID (optional)
            user_id: User ID (optional)
        """
        extra = {
            "event_type": "request_error",
            "method": method,
            "path": path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_id": request_id,
            "user_id": user_id
        }

        self.logger.error(f"Request failed: {error}", **extra)


class PerformanceLogger:
    """
    Performance logger for monitoring application performance.
    """

    def __init__(self, base_logger: StructuredLogger):
        """
        Initialize performance logger.

        Args:
            base_logger: Base structured logger
        """
        self.logger = base_logger.bind(component="performance")
        self._timers = {}

    def start_timer(self, operation: str, **context) -> str:
        """
        Start timing an operation.

        Args:
            operation: Operation name
            **context: Additional context

        Returns:
            Timer ID
        """
        import uuid
        timer_id = str(uuid.uuid4())

        self._timers[timer_id] = {
            "operation": operation,
            "start_time": asyncio.get_event_loop().time(),
            "context": context
        }

        self.logger.debug(f"Started timing operation: {operation}", timer_id=timer_id, **context)

        return timer_id

    def end_timer(self, timer_id: str, **context) -> float:
        """
        End timing an operation and log the duration.

        Args:
            timer_id: Timer ID
            **context: Additional context

        Returns:
            Duration in milliseconds
        """
        if timer_id not in self._timers:
            self.logger.warning(f"Timer not found: {timer_id}")
            return 0.0

        timer_info = self._timers.pop(timer_id)
        duration_ms = (asyncio.get_event_loop().time() - timer_info["start_time"]) * 1000

        log_context = {
            "operation": timer_info["operation"],
            "duration_ms": duration_ms,
            **timer_info["context"],
            **context
        }

        if duration_ms > 1000:  # Slow operation
            self.logger.warning(f"Slow operation detected: {timer_info['operation']}", **log_context)
        else:
            self.logger.debug(f"Operation completed: {timer_info['operation']}", **log_context)

        return duration_ms

    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "count",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Log a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            tags: Additional tags (optional)
        """
        extra = {
            "event_type": "metric",
            "metric_name": metric_name,
            "value": value,
            "unit": unit
        }

        if tags:
            extra["tags"] = tags

        self.logger.info(f"Metric: {metric_name} = {value} {unit}", **extra)

    def log_memory_usage(self, process_id: Optional[int] = None) -> None:
        """
        Log current memory usage.

        Args:
            process_id: Process ID (optional)
        """
        try:
            import psutil
            process = psutil.Process(process_id) if process_id else psutil.Process()
            memory_info = process.memory_info()

            extra = {
                "event_type": "memory_usage",
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "process_id": process.pid
            }

            self.logger.info("Memory usage", **extra)

        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")


# Default configuration
DEFAULT_LOGGING_CONFIG = LoggingConfig(
    level="INFO",
    format="json",
    log_dir="logs",
    log_file="app.log",
    rotation="10 MB",
    retention="30 days",
    console_colors=True,
    include_extra=True,
    extra_fields={
        "service": "complex_rag",
        "version": "1.0.0"
    }
)


def configure_logging(config: Optional[LoggingConfig] = None) -> StructuredLogger:
    """
    Configure structured logging.

    Args:
        config: Logging configuration (optional)

    Returns:
        Configured structured logger
    """
    if config is None:
        config = DEFAULT_LOGGING_CONFIG

    return StructuredLogger(config)


# Export
__all__ = [
    'LoggingConfig',
    'StructuredLogger',
    'RequestLogger',
    'PerformanceLogger',
    'DEFAULT_LOGGING_CONFIG',
    'configure_logging'
]