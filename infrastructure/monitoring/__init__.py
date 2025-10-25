"""
Monitoring Module

This module provides comprehensive monitoring and observability capabilities
including structured logging, metrics collection, distributed tracing,
and alerting systems.
"""

from .logging_config import (
    # Logging configuration
    LoggingConfig,
    StructuredLogger,
    RequestLogger,
    PerformanceLogger,
    DEFAULT_LOGGING_CONFIG,
    configure_logging,
)

from .metrics import (
    # Metrics collection
    MetricType,
    MetricValue,
    HistogramBucket,
    MetricCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
    MetricsRegistry,
    SystemMetrics,
    get_registry,
    counter,
    gauge,
    histogram,
    timer,
)

from .tracing import (
    # Distributed tracing
    SpanKind,
    SpanStatus,
    SpanContext,
    Span,
    Tracer,
    TraceSampler,
    SpanExporter,
    ConsoleSpanExporter,
    JSONFileSpanExporter,
    TraceManager,
    trace_span,
    trace_async_span,
    get_default_tracer,
    set_default_tracer,
)

from .alerting import (
    # Alerting and notifications
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    Alert,
    AlertRule,
    Silence,
    NotificationProvider,
    EmailNotificationProvider,
    WebhookNotificationProvider,
    SlackNotificationProvider,
    ConsoleNotificationProvider,
    AlertRuleEvaluator,
    AlertManager,
    get_alert_manager,
    set_alert_manager,
)

__all__ = [
    # Logging configuration
    "LoggingConfig",
    "StructuredLogger",
    "RequestLogger",
    "PerformanceLogger",
    "DEFAULT_LOGGING_CONFIG",
    "configure_logging",

    # Metrics collection
    "MetricType",
    "MetricValue",
    "HistogramBucket",
    "MetricCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "MetricsRegistry",
    "SystemMetrics",
    "get_registry",
    "counter",
    "gauge",
    "histogram",
    "timer",

    # Distributed tracing
    "SpanKind",
    "SpanStatus",
    "SpanContext",
    "Span",
    "Tracer",
    "TraceSampler",
    "SpanExporter",
    "ConsoleSpanExporter",
    "JSONFileSpanExporter",
    "TraceManager",
    "trace_span",
    "trace_async_span",
    "get_default_tracer",
    "set_default_tracer",

    # Alerting and notifications
    "AlertSeverity",
    "AlertStatus",
    "NotificationChannel",
    "Alert",
    "AlertRule",
    "Silence",
    "NotificationProvider",
    "EmailNotificationProvider",
    "WebhookNotificationProvider",
    "SlackNotificationProvider",
    "ConsoleNotificationProvider",
    "AlertRuleEvaluator",
    "AlertManager",
    "get_alert_manager",
    "set_alert_manager",
]
