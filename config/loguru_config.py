"""
Loguru Logging Configuration

This module provides centralized logging configuration for the Complex RAG system.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class LoguruConfig:
    """Loguru configuration manager."""

    def __init__(self):
        self._configured = False

    def remove_default_handlers(self):
        """Remove default loguru handlers."""
        logger.remove()
        self._configured = False

    def configure_console_logging(
        self,
        level: str = "INFO",
        format_string: Optional[str] = None,
        colorize: Optional[bool] = None,
        backtrace: bool = True,
        diagnose: bool = True,
        catch: bool = True,
    ):
        """Configure console logging."""
        if format_string is None:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )

        if colorize is None:
            colorize = sys.stdout.isatty()

        logger.add(
            sys.stdout,
            format=format_string,
            level=level,
            colorize=colorize,
            backtrace=backtrace,
            diagnose=diagnose,
            catch=catch,
            enqueue=True,
        )
        self._configured = True

    def configure_file_logging(
        self,
        log_dir: Union[str, Path],
        level: str = "INFO",
        format_string: Optional[str] = None,
        rotation: str = "1 day",
        retention: str = "30 days",
        compression: str = "zip",
        backtrace: bool = True,
        diagnose: bool = True,
        catch: bool = True,
        encoding: str = "utf-8",
        enqueue: bool = True,
    ):
        """Configure file logging."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if format_string is None:
            format_string = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )

        # General log file
        logger.add(
            log_dir / "app.log",
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=backtrace,
            diagnose=diagnose,
            catch=catch,
            encoding=encoding,
            enqueue=enqueue,
        )

        # Error log file
        logger.add(
            log_dir / "error.log",
            format=format_string,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=backtrace,
            diagnose=diagnose,
            catch=catch,
            encoding=encoding,
            enqueue=enqueue,
        )

        # Debug log file (only in debug mode)
        if level.upper() == "DEBUG":
            logger.add(
                log_dir / "debug.log",
                format=format_string,
                level="DEBUG",
                rotation=rotation,
                retention=retention,
                compression=compression,
                backtrace=backtrace,
                diagnose=diagnose,
                catch=catch,
                encoding=encoding,
                enqueue=enqueue,
            )

        self._configured = True

    def configure_module_logging(
        self,
        log_dir: Union[str, Path],
        modules: List[str],
        level: str = "INFO",
        rotation: str = "1 day",
        retention: str = "30 days",
        compression: str = "zip",
    ):
        """Configure separate log files for specific modules."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        format_string = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        for module in modules:
            module_log_file = log_dir / f"{module}.log"
            logger.add(
                module_log_file,
                format=format_string,
                level=level,
                rotation=rotation,
                retention=retention,
                compression=compression,
                filter=lambda record: record["extra"].get("module") == module,
                enqueue=True,
            )

    def configure_json_logging(
        self,
        log_dir: Union[str, Path],
        level: str = "INFO",
        rotation: str = "1 day",
        retention: str = "30 days",
        compression: str = "zip",
    ):
        """Configure JSON structured logging."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # JSON format for structured logging
        def json_formatter(record: Dict[str, Any]) -> str:
            import json
            import datetime

            log_entry = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "logger": record["name"],
                "module": record["module"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
                "extra": record.get("extra", {}),
            }

            # Add exception info if present
            if record["exception"]:
                log_entry["exception"] = {
                    "type": record["exception"].type.__name__,
                    "value": str(record["exception"].value),
                    "traceback": record["exception"].traceback,
                }

            return json.dumps(log_entry, ensure_ascii=False)

        logger.add(
            log_dir / "structured.json",
            format=json_formatter,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
        )

    def configure_request_logging(
        self,
        log_dir: Union[str, Path],
        level: str = "INFO",
        rotation: str = "1 day",
        retention: str = "7 days",
        compression: str = "zip",
    ):
        """Configure request logging for API endpoints."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        def request_formatter(record: Dict[str, Any]) -> str:
            import json

            extra = record["extra"]
            request_data = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "method": extra.get("method"),
                "url": extra.get("url"),
                "status_code": extra.get("status_code"),
                "duration_ms": extra.get("duration_ms"),
                "user_agent": extra.get("user_agent"),
                "client_ip": extra.get("client_ip"),
                "request_id": extra.get("request_id"),
                "message": record["message"],
            }

            return json.dumps(request_data, ensure_ascii=False)

        logger.add(
            log_dir / "requests.json",
            format=request_formatter,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            filter=lambda record: record["extra"].get("type") == "request",
            enqueue=True,
        )

    def configure_performance_logging(
        self,
        log_dir: Union[str, Path],
        level: str = "INFO",
        rotation: str = "1 day",
        retention: str = "7 days",
        compression: str = "zip",
    ):
        """Configure performance logging."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        def performance_formatter(record: Dict[str, Any]) -> str:
            import json

            extra = record["extra"]
            perf_data = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "operation": extra.get("operation"),
                "duration_ms": extra.get("duration_ms"),
                "memory_mb": extra.get("memory_mb"),
                "cpu_percent": extra.get("cpu_percent"),
                "success": extra.get("success", True),
                "error": extra.get("error"),
                "message": record["message"],
            }

            return json.dumps(perf_data, ensure_ascii=False)

        logger.add(
            log_dir / "performance.json",
            format=performance_formatter,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            filter=lambda record: record["extra"].get("type") == "performance",
            enqueue=True,
        )

    def configure_development_logging(self, log_dir: Union[str, Path] = "logs"):
        """Configure development environment logging."""
        self.remove_default_handlers()

        # Console logging with colors
        self.configure_console_logging(
            level="DEBUG",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # File logging
        self.configure_file_logging(
            log_dir=log_dir,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

        # JSON structured logging
        self.configure_json_logging(
            log_dir=log_dir,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    def configure_production_logging(self, log_dir: Union[str, Path] = "logs"):
        """Configure production environment logging."""
        self.remove_default_handlers()

        # Console logging (errors only)
        self.configure_console_logging(
            level="ERROR",
            colorize=False,
            backtrace=False,
            diagnose=False,
        )

        # File logging
        self.configure_file_logging(
            log_dir=log_dir,
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

        # JSON structured logging
        self.configure_json_logging(
            log_dir=log_dir,
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

        # Request logging
        self.configure_request_logging(
            log_dir=log_dir,
            level="INFO",
            rotation="50 MB",
            retention="7 days",
            compression="zip",
        )

        # Performance logging
        self.configure_performance_logging(
            log_dir=log_dir,
            level="INFO",
            rotation="50 MB",
            retention="7 days",
            compression="zip",
        )

    def configure_testing_logging(self, log_dir: Union[str, Path] = "logs"):
        """Configure testing environment logging."""
        self.remove_default_handlers()

        # Console logging (capture only important messages)
        self.configure_console_logging(
            level="WARNING",
            colorize=True,
            backtrace=False,
            diagnose=False,
        )

        # File logging for debugging tests
        self.configure_file_logging(
            log_dir=log_dir,
            level="DEBUG",
            rotation="10 MB",
            retention="3 days",
            compression="zip",
        )

    def configure_staging_logging(self, log_dir: Union[str, Path] = "logs"):
        """Configure staging environment logging."""
        self.remove_default_handlers()

        # Console logging
        self.configure_console_logging(
            level="INFO",
            colorize=False,
            backtrace=True,
            diagnose=True,
        )

        # File logging
        self.configure_file_logging(
            log_dir=log_dir,
            level="INFO",
            rotation="50 MB",
            retention="14 days",
            compression="zip",
        )

        # JSON structured logging
        self.configure_json_logging(
            log_dir=log_dir,
            level="INFO",
            rotation="50 MB",
            retention="14 days",
            compression="zip",
        )

        # Request logging
        self.configure_request_logging(
            log_dir=log_dir,
            level="INFO",
            rotation="25 MB",
            retention="7 days",
            compression="zip",
        )

    def get_logger(self, name: Optional[str] = None, module: Optional[str] = None):
        """Get a configured logger instance."""
        if name:
            log = logger.bind(name=name)
        else:
            log = logger

        if module:
            log = log.bind(module=module)

        return log

    def is_configured(self) -> bool:
        """Check if logging has been configured."""
        return self._configured


# Global loguru configuration instance
loguru_config = LoguruConfig()


def setup_logging(
    environment: str = "development",
    log_dir: Union[str, Path] = "logs",
    custom_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Setup logging based on environment.

    Args:
        environment: Environment name (development, production, testing, staging)
        log_dir: Directory for log files
        custom_config: Custom configuration overrides
    """
    if custom_config is None:
        custom_config = {}

    environment = environment.lower()

    if environment == "development":
        loguru_config.configure_development_logging(log_dir)
    elif environment == "production":
        loguru_config.configure_production_logging(log_dir)
    elif environment == "testing":
        loguru_config.configure_testing_logging(log_dir)
    elif environment == "staging":
        loguru_config.configure_staging_logging(log_dir)
    else:
        # Default to development
        loguru_config.configure_development_logging(log_dir)

    logger.info(f"Logging configured for environment: {environment}")


def get_logger(name: Optional[str] = None, module: Optional[str] = None):
    """Get a configured logger instance."""
    return loguru_config.get_logger(name=name, module=module)


# Module-level logger
module_logger = get_logger(__name__)