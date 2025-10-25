"""
Distributed Tracing

This module provides distributed tracing capabilities with support for
span creation, context propagation, and trace sampling.
"""

import time
import uuid
import json
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
import functools


class SpanKind(Enum):
    """Span kinds."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span status codes."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class SpanContext:
    """Span context for trace propagation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for propagation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id or "",
            "baggage": json.dumps(self.baggage)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'SpanContext':
        """Create from dictionary."""
        baggage = {}
        try:
            baggage = json.loads(data.get("baggage", "{}"))
        except:
            pass

        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id") or None,
            baggage=baggage
        )


@dataclass
class Span:
    """Trace span."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    status_message: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        self.events.append(event)

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def finish(self, end_time: Optional[float] = None) -> None:
        """Finish span."""
        self.end_time = end_time or time.time()

    def duration(self) -> float:
        """Get span duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
            "baggage": self.baggage
        }


class Tracer:
    """
    Tracer for creating and managing spans.
    """

    def __init__(self, name: str):
        """
        Initialize tracer.

        Args:
            name: Tracer name (usually service name)
        """
        self.name = name
        self._active_span: Optional[Span] = None
        self._spans: List[Span] = []
        self._lock = threading.Lock()

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[Union[Span, SpanContext]] = None
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Span name
            kind: Span kind
            parent: Parent span or context

        Returns:
            Started span
        """
        trace_id = None
        parent_span_id = None

        if parent:
            if isinstance(parent, Span):
                trace_id = parent.trace_id
                parent_span_id = parent.span_id
            else:
                trace_id = parent.trace_id
                parent_span_id = parent.span_id
        elif self._active_span:
            trace_id = self._active_span.trace_id
            parent_span_id = self._active_span.span_id

        if trace_id is None:
            trace_id = str(uuid.uuid4())

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            kind=kind
        )

        # Copy baggage from parent
        if parent:
            if isinstance(parent, Span):
                span.baggage = parent.baggage.copy()
            else:
                span.baggage = parent.baggage.copy()

        with self._lock:
            self._spans.append(span)
            self._active_span = span

        return span

    def end_span(self, span: Span) -> None:
        """
        End a span.

        Args:
            span: Span to end
        """
        span.finish()

        with self._lock:
            if self._active_span == span:
                # Find the parent span
                if span.parent_span_id:
                    for parent in reversed(self._spans):
                        if parent.span_id == span.parent_span_id and parent.end_time is None:
                            self._active_span = parent
                            break
                    else:
                        self._active_span = None
                else:
                    self._active_span = None

    def get_active_span(self) -> Optional[Span]:
        """Get currently active span."""
        return self._active_span

    def get_span_context(self) -> Optional[SpanContext]:
        """Get context of active span."""
        if self._active_span:
            return SpanContext(
                trace_id=self._active_span.trace_id,
                span_id=self._active_span.span_id,
                parent_span_id=self._active_span.parent_span_id,
                baggage=self._active_span.baggage.copy()
            )
        return None

    def set_span_context(self, context: SpanContext) -> Optional[Span]:
        """
        Set active span from context.

        Args:
            context: Span context to set

        Returns:
            Active span if found
        """
        with self._lock:
            for span in reversed(self._spans):
                if span.trace_id == context.trace_id and span.span_id == context.span_id and span.end_time is None:
                    self._active_span = span
                    return span
        return None

    def get_finished_spans(self) -> List[Span]:
        """Get all finished spans."""
        with self._lock:
            return [span for span in self._spans if span.end_time is not None]

    def get_all_spans(self) -> List[Span]:
        """Get all spans."""
        with self._lock:
            return self._spans.copy()

    def clear(self) -> None:
        """Clear all spans."""
        with self._lock:
            self._spans.clear()
            self._active_span = None

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for creating a span.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes

        Yields:
            Created span
        """
        span = self.start_span(name, kind)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            self.end_span(span)

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Async context manager for creating a span.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes

        Yields:
            Created span
        """
        span = self.start_span(name, kind)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            self.end_span(span)


class TraceSampler:
    """
    Trace sampler for deciding which traces to sample.
    """

    def __init__(self, sample_rate: float = 0.1):
        """
        Initialize sampler.

        Args:
            sample_rate: Sample rate between 0.0 and 1.0
        """
        self.sample_rate = max(0.0, min(1.0, sample_rate))

    def should_sample(self, trace_id: str) -> bool:
        """
        Decide whether to sample a trace.

        Args:
            trace_id: Trace ID

        Returns:
            True if trace should be sampled
        """
        import random
        return random.random() < self.sample_rate

    def get_sample_decision(self, trace_id: str) -> Dict[str, Any]:
        """
        Get sampling decision with metadata.

        Args:
            trace_id: Trace ID

        Returns:
            Sampling decision dictionary
        """
        sampled = self.should_sample(trace_id)
        return {
            "trace_id": trace_id,
            "sampled": sampled,
            "sample_rate": self.sample_rate,
            "decision_time": time.time()
        }


class SpanExporter:
    """
    Base class for span exporters.
    """

    def export(self, spans: List[Span]) -> None:
        """
        Export spans.

        Args:
            spans: Spans to export
        """
        raise NotImplementedError


class ConsoleSpanExporter(SpanExporter):
    """
    Console span exporter for debugging.
    """

    def export(self, spans: List[Span]) -> None:
        """Export spans to console."""
        for span in spans:
            print(f"Span: {span.name} ({span.duration():.3f}s) - {span.status.value}")
            if span.attributes:
                print(f"  Attributes: {span.attributes}")
            if span.events:
                print(f"  Events: {len(span.events)}")


class JSONFileSpanExporter(SpanExporter):
    """
    JSON file span exporter.
    """

    def __init__(self, file_path: str):
        """
        Initialize exporter.

        Args:
            file_path: File path to export spans to
        """
        self.file_path = file_path

    def export(self, spans: List[Span]) -> None:
        """Export spans to JSON file."""
        try:
            with open(self.file_path, 'a') as f:
                for span in spans:
                    span_data = span.to_dict()
                    f.write(json.dumps(span_data) + '\n')
        except Exception as e:
            print(f"Failed to export spans to {self.file_path}: {e}")


class TraceManager:
    """
    Central trace manager.
    """

    def __init__(
        self,
        service_name: str,
        sample_rate: float = 0.1,
        exporters: Optional[List[SpanExporter]] = None
    ):
        """
        Initialize trace manager.

        Args:
            service_name: Service name
            sample_rate: Sample rate
            exporters: List of span exporters
        """
        self.service_name = service_name
        self.tracer = Tracer(service_name)
        self.sampler = TraceSampler(sample_rate)
        self.exporters = exporters or []
        self._export_buffer: List[Span] = []
        self._buffer_lock = threading.Lock()
        self._export_task: Optional[asyncio.Task] = None

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[Union[Span, SpanContext]] = None
    ) -> Span:
        """
        Start a new span with sampling.

        Args:
            name: Span name
            kind: Span kind
            parent: Parent span or context

        Returns:
            Started span
        """
        span = self.tracer.start_span(name, kind, parent)

        # Set sampling decision
        sample_decision = self.sampler.get_sample_decision(span.trace_id)
        span.set_attribute("sampled", sample_decision["sampled"])
        span.set_attribute("sample_rate", sample_decision["sample_rate"])

        return span

    def end_span(self, span: Span) -> None:
        """
        End a span and queue for export if sampled.

        Args:
            span: Span to end
        """
        self.tracer.end_span(span)

        # Queue for export if sampled
        if span.attributes.get("sampled", False):
            with self._buffer_lock:
                self._export_buffer.append(span)

    def export_spans(self) -> None:
        """Export buffered spans."""
        with self._buffer_lock:
            spans_to_export = self._export_buffer.copy()
            self._export_buffer.clear()

        if spans_to_export:
            for exporter in self.exporters:
                try:
                    exporter.export(spans_to_export)
                except Exception as e:
                    print(f"Failed to export spans: {e}")

    def start_background_export(self, interval: float = 5.0) -> None:
        """
        Start background export task.

        Args:
            interval: Export interval in seconds
        """
        if self._export_task and not self._export_task.done():
            return

        async def export_loop():
            while True:
                await asyncio.sleep(interval)
                self.export_spans()

        self._export_task = asyncio.create_task(export_loop())

    def stop_background_export(self) -> None:
        """Stop background export task."""
        if self._export_task and not self._export_task.done():
            self._export_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._export_task)
            except asyncio.CancelledError:
                pass

    def get_trace_headers(self) -> Dict[str, str]:
        """
        Get trace headers for propagation.

        Returns:
            Trace headers
        """
        context = self.tracer.get_span_context()
        if context:
            return {
                "x-trace-id": context.trace_id,
                "x-span-id": context.span_id,
                "x-parent-span-id": context.parent_span_id or "",
                "x-trace-baggage": json.dumps(context.baggage)
            }
        return {}

    def set_trace_from_headers(self, headers: Dict[str, str]) -> Optional[Span]:
        """
        Set active span from trace headers.

        Args:
            headers: HTTP headers with trace information

        Returns:
            Active span if found
        """
        trace_id = headers.get("x-trace-id")
        span_id = headers.get("x-span-id")
        parent_span_id = headers.get("x-parent-span-id")
        baggage_str = headers.get("x-trace-baggage", "{}")

        if not trace_id or not span_id:
            return None

        try:
            baggage = json.loads(baggage_str)
        except:
            baggage = {}

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id if parent_span_id else None,
            baggage=baggage
        )

        return self.tracer.set_span_context(context)


# Decorators for automatic tracing
def trace_span(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    tracer: Optional[Tracer] = None
):
    """
    Decorator for automatic span creation.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        tracer: Tracer instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            tracer_instance = tracer or get_default_tracer()

            with tracer_instance.span(span_name, kind):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_span(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    tracer: Optional[Tracer] = None
):
    """
    Decorator for automatic async span creation.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        tracer: Tracer instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            tracer_instance = tracer or get_default_tracer()

            async with tracer_instance.async_span(span_name, kind):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Global tracer instance
_default_tracer: Optional[Tracer] = None


def get_default_tracer() -> Tracer:
    """Get default tracer instance."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer("default")
    return _default_tracer


def set_default_tracer(tracer: Tracer) -> None:
    """Set default tracer instance."""
    global _default_tracer
    _default_tracer = tracer


# Export
__all__ = [
    'SpanKind',
    'SpanStatus',
    'SpanContext',
    'Span',
    'Tracer',
    'TraceSampler',
    'SpanExporter',
    'ConsoleSpanExporter',
    'JSONFileSpanExporter',
    'TraceManager',
    'trace_span',
    'trace_async_span',
    'get_default_tracer',
    'set_default_tracer'
]