"""
Performance Metrics Collection

This module provides comprehensive metrics collection and monitoring
with support for counters, gauges, histograms, and timers.
"""

import time
import asyncio
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timezone
import threading
import statistics
from enum import Enum

try:
    import psutil
except ImportError:
    psutil = None


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str]
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class HistogramBucket:
    """Histogram bucket for value distribution."""
    upper_bound: float
    count: int


class MetricCollector:
    """
    Base class for metric collectors.
    """

    def __init__(self, name: str, description: str = "", unit: str = ""):
        """
        Initialize metric collector.

        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.tags = {}
        self.created_at = time.time()

    def with_tags(self, **tags) -> 'MetricCollector':
        """
        Add tags to metric.

        Args:
            **tags: Tag key-value pairs

        Returns:
            Self for chaining
        """
        new_tags = self.tags.copy()
        new_tags.update(tags)
        new_collector = self.__class__(self.name, self.description, self.unit)
        new_collector.tags = new_tags
        return new_collector

    def get_value(self) -> Union[int, float, List[float]]:
        """Get current metric value."""
        raise NotImplementedError


class Counter(MetricCollector):
    """
    Counter metric that can only increase.
    """

    def __init__(self, name: str, description: str = "", unit: str = "count"):
        """
        Initialize counter.

        Args:
            name: Counter name
            description: Counter description
            unit: Unit of measurement
        """
        super().__init__(name, description, unit)
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        """
        Increment counter.

        Args:
            amount: Amount to increment
        """
        with self._lock:
            self._value += amount

    def get_value(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Gauge(MetricCollector):
    """
    Gauge metric that can increase or decrease.
    """

    def __init__(self, name: str, description: str = "", unit: str = ""):
        """
        Initialize gauge.

        Args:
            name: Gauge name
            description: Gauge description
            unit: Unit of measurement
        """
        super().__init__(name, description, unit)
        self._value = 0
        self._lock = threading.Lock()

    def set(self, value: Union[int, float]) -> None:
        """
        Set gauge value.

        Args:
            value: New value
        """
        with self._lock:
            self._value = value

    def inc(self, amount: Union[int, float] = 1) -> None:
        """
        Increment gauge.

        Args:
            amount: Amount to increment
        """
        with self._lock:
            self._value += amount

    def dec(self, amount: Union[int, float] = 1) -> None:
        """
        Decrement gauge.

        Args:
            amount: Amount to decrement
        """
        with self._lock:
            self._value -= amount

    def get_value(self) -> Union[int, float]:
        """Get current gauge value."""
        with self._lock:
            return self._value


class Histogram(MetricCollector):
    """
    Histogram metric for value distribution.
    """

    def __init__(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        description: str = "",
        unit: str = ""
    ):
        """
        Initialize histogram.

        Args:
            name: Histogram name
            buckets: Bucket boundaries
            description: Histogram description
            unit: Unit of measurement
        """
        super().__init__(name, description, unit)
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._observations = []
        self._lock = threading.Lock()

    def observe(self, value: Union[int, float]) -> None:
        """
        Observe a value.

        Args:
            value: Value to observe
        """
        with self._lock:
            self._observations.append(float(value))

    def get_value(self) -> List[float]:
        """Get all observations."""
        with self._lock:
            return self._observations.copy()

    def get_buckets(self) -> List[HistogramBucket]:
        """Get histogram buckets with counts."""
        with self._lock:
            counts = [0] * (len(self.buckets) + 1)

            for value in self._observations:
                for i, bucket in enumerate(self.buckets):
                    if value <= bucket:
                        counts[i] += 1
                        break
                else:
                    counts[-1] += 1  # +Inf bucket

            result = []
            for i, bound in enumerate(self.buckets):
                result.append(HistogramBucket(upper_bound=bound, count=counts[i]))
            result.append(HistogramBucket(upper_bound=float('inf'), count=counts[-1]))

            return result

    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            if not self._observations:
                return {}

            values = self._observations
            return {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p50": statistics.quantiles(values, n=2)[0] if len(values) > 1 else values[0],
                "p90": statistics.quantiles(values, n=10)[8] if len(values) > 9 else values[-1],
                "p95": statistics.quantiles(values, n=20)[18] if len(values) > 19 else values[-1],
                "p99": statistics.quantiles(values, n=100)[98] if len(values) > 99 else values[-1]
            }

    def reset(self) -> None:
        """Reset histogram observations."""
        with self._lock:
            self._observations.clear()


class Timer:
    """
    Timer for measuring duration.
    """

    def __init__(self, histogram: Histogram):
        """
        Initialize timer.

        Args:
            histogram: Histogram to record timings
        """
        self.histogram = histogram
        self._start_time = None

    def start(self) -> 'Timer':
        """Start timer."""
        self._start_time = time.time()
        return self

    def stop(self) -> float:
        """
        Stop timer and record duration.

        Returns:
            Duration in seconds
        """
        if self._start_time is None:
            return 0.0

        duration = time.time() - self._start_time
        self.histogram.observe(duration)
        self._start_time = None
        return duration

    def __enter__(self) -> 'Timer':
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class MetricsRegistry:
    """
    Central registry for all metrics.
    """

    def __init__(self):
        """Initialize metrics registry."""
        self._metrics: Dict[str, MetricCollector] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "", unit: str = "count") -> Counter:
        """
        Get or create a counter.

        Args:
            name: Counter name
            description: Counter description
            unit: Unit of measurement

        Returns:
            Counter instance
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description, unit)
            return self._metrics[name]

    def gauge(self, name: str, description: str = "", unit: str = "") -> Gauge:
        """
        Get or create a gauge.

        Args:
            name: Gauge name
            description: Gauge description
            unit: Unit of measurement

        Returns:
            Gauge instance
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description, unit)
            return self._metrics[name]

    def histogram(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        description: str = "",
        unit: str = ""
    ) -> Histogram:
        """
        Get or create a histogram.

        Args:
            name: Histogram name
            buckets: Bucket boundaries
            description: Histogram description
            unit: Unit of measurement

        Returns:
            Histogram instance
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, buckets, description, unit)
            return self._metrics[name]

    def timer(self, name: str, description: str = "", unit: str = "seconds") -> Timer:
        """
        Create a timer.

        Args:
            name: Timer name
            description: Timer description
            unit: Unit of measurement

        Returns:
            Timer instance
        """
        histogram = self.histogram(name, description=description, unit=unit)
        return Timer(histogram)

    def get_metric(self, name: str) -> Optional[MetricCollector]:
        """
        Get a metric by name.

        Args:
            name: Metric name

        Returns:
            Metric instance or None
        """
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricCollector]:
        """
        Get all registered metrics.

        Returns:
            Dictionary of all metrics
        """
        with self._lock:
            return self._metrics.copy()

    def collect(self) -> List[MetricValue]:
        """
        Collect all metric values.

        Returns:
            List of metric values
        """
        timestamp = time.time()
        values = []

        with self._lock:
            for metric in self._metrics.values():
                value = MetricValue(
                    name=metric.name,
                    value=metric.get_value(),
                    metric_type=MetricType(metric.__class__.name.lower()),
                    timestamp=timestamp,
                    tags=metric.tags.copy(),
                    unit=metric.unit,
                    description=metric.description
                )
                values.append(value)

        return values

    def reset_all(self) -> None:
        """Reset all metrics that support resetting."""
        with self._lock:
            for metric in self._metrics.values():
                if hasattr(metric, 'reset'):
                    metric.reset()

    def remove_metric(self, name: str) -> bool:
        """
        Remove a metric by name.

        Args:
            name: Metric name

        Returns:
            True if metric was removed
        """
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                return True
            return False


class SystemMetrics:
    """
    System-level metrics collector.
    """

    def __init__(self, registry: MetricsRegistry):
        """
        Initialize system metrics.

        Args:
            registry: Metrics registry to register metrics to
        """
        self.registry = registry
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Setup system metrics."""
        if psutil is None:
            return

        # CPU metrics
        self.cpu_usage = self.registry.gauge(
            "system_cpu_usage_percent",
            "CPU usage percentage",
            "percent"
        )

        # Memory metrics
        self.memory_usage = self.registry.gauge(
            "system_memory_usage_bytes",
            "Memory usage in bytes",
            "bytes"
        )
        self.memory_available = self.registry.gauge(
            "system_memory_available_bytes",
            "Available memory in bytes",
            "bytes"
        )
        self.memory_percent = self.registry.gauge(
            "system_memory_usage_percent",
            "Memory usage percentage",
            "percent"
        )

        # Disk metrics
        self.disk_usage = self.registry.gauge(
            "system_disk_usage_bytes",
            "Disk usage in bytes",
            "bytes"
        )
        self.disk_free = self.registry.gauge(
            "system_disk_free_bytes",
            "Free disk space in bytes",
            "bytes"
        )
        self.disk_percent = self.registry.gauge(
            "system_disk_usage_percent",
            "Disk usage percentage",
            "percent"
        )

        # Process metrics
        self.process_cpu = self.registry.gauge(
            "process_cpu_percent",
            "Process CPU usage percentage",
            "percent"
        )
        self.process_memory = self.registry.gauge(
            "process_memory_rss_bytes",
            "Process RSS memory in bytes",
            "bytes"
        )
        self.process_memory_vms = self.registry.gauge(
            "process_memory_vms_bytes",
            "Process VMS memory in bytes",
            "bytes"
        )

    def collect(self) -> None:
        """Collect system metrics."""
        if psutil is None:
            return

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.set(cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_available.set(memory.available)
            self.memory_percent.set(memory.percent)

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.disk_usage.set(disk.used)
            self.disk_free.set(disk.free)
            self.disk_percent.set(disk.percent)

            # Process metrics
            process = psutil.Process()
            self.process_cpu.set(process.cpu_percent())
            memory_info = process.memory_info()
            self.process_memory.set(memory_info.rss)
            self.process_memory_vms.set(memory_info.vms)

        except Exception:
            pass  # Ignore collection errors


# Global registry instance
_global_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """
    Get global metrics registry.

    Returns:
        Global metrics registry
    """
    return _global_registry


def counter(name: str, description: str = "", unit: str = "count") -> Counter:
    """
    Get or create a counter from global registry.

    Args:
        name: Counter name
        description: Counter description
        unit: Unit of measurement

    Returns:
        Counter instance
    """
    return _global_registry.counter(name, description, unit)


def gauge(name: str, description: str = "", unit: str = "") -> Gauge:
    """
    Get or create a gauge from global registry.

    Args:
        name: Gauge name
        description: Gauge description
        unit: Unit of measurement

    Returns:
        Gauge instance
    """
    return _global_registry.gauge(name, description, unit)


def histogram(
    name: str,
    buckets: Optional[List[float]] = None,
    description: str = "",
    unit: str = ""
) -> Histogram:
    """
    Get or create a histogram from global registry.

    Args:
        name: Histogram name
        buckets: Bucket boundaries
        description: Histogram description
        unit: Unit of measurement

    Returns:
        Histogram instance
    """
    return _global_registry.histogram(name, buckets, description, unit)


def timer(name: str, description: str = "", unit: str = "seconds") -> Timer:
    """
    Create a timer from global registry.

    Args:
        name: Timer name
        description: Timer description
        unit: Unit of measurement

    Returns:
        Timer instance
    """
    return _global_registry.timer(name, description, unit)


# Export
__all__ = [
    'MetricType',
    'MetricValue',
    'HistogramBucket',
    'MetricCollector',
    'Counter',
    'Gauge',
    'Histogram',
    'Timer',
    'MetricsRegistry',
    'SystemMetrics',
    'get_registry',
    'counter',
    'gauge',
    'histogram',
    'timer'
]