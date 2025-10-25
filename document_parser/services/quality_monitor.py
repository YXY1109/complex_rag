"""
Processing Quality Monitor

This module provides comprehensive quality monitoring for document processing
including quality metrics, performance tracking, and optimization recommendations.
"""

import time
import asyncio
import statistics
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
from pathlib import Path

from ..interfaces.source_interface import (
    FileSource,
    ProcessingStrategy,
    SourceDetectionResult
)
from .processing_strategy_selector import StrategyParams
from ..interfaces.parser_interface import ParseResponse


class QualityMetric(str, Enum):
    """Quality metric types."""
    TEXT_EXTRACTION = "text_extraction"
    STRUCTURE_PRESERVATION = "structure_preservation"
    CONTENT_COMPLETENESS = "content_completeness"
    ACCURACY = "accuracy"
    PROCESSING_SPEED = "processing_speed"
    MEMORY_EFFICIENCY = "memory_efficiency"
    ERROR_RATE = "error_rate"


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityMeasurement:
    """Single quality measurement."""
    metric: QualityMetric
    value: float
    threshold: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingSession:
    """Processing session with quality metrics."""
    session_id: str
    file_source: FileSource
    strategy: ProcessingStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    file_size: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    measurements: List[QualityMeasurement] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    success: Optional[bool] = None


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    session_id: str
    overall_quality_level: QualityLevel
    overall_score: float
    metric_scores: Dict[QualityMetric, float]
    processing_stats: Dict[str, Any]
    recommendations: List[str]
    quality_trends: Dict[QualityMetric, List[float]]
    benchmark_comparison: Optional[Dict[str, float]] = None


class QualityMonitor:
    """
    Comprehensive quality monitoring system for document processing.

    Tracks quality metrics, analyzes performance, and provides
    recommendations for optimization.
    """

    def __init__(self, history_size: int = 1000):
        """Initialize quality monitor."""
        self.history_size = history_size
        self.sessions: List[ProcessingSession] = []
        self.active_sessions: Dict[str, ProcessingSession] = {}
        self.quality_thresholds = self._setup_quality_thresholds()
        self.benchmarks = self._setup_benchmarks()

    def _setup_quality_thresholds(self) -> Dict[QualityMetric, Dict[str, float]]:
        """Setup quality thresholds for different metrics."""
        return {
            QualityMetric.TEXT_EXTRACTION: {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            },
            QualityMetric.STRUCTURE_PRESERVATION: {
                'excellent': 0.90,
                'good': 0.80,
                'acceptable': 0.65,
                'poor': 0.45
            },
            QualityMetric.CONTENT_COMPLETENESS: {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            },
            QualityMetric.ACCURACY: {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            },
            QualityMetric.PROCESSING_SPEED: {
                'excellent': 0.9,  # MB/s
                'good': 0.5,
                'acceptable': 0.2,
                'poor': 0.1
            },
            QualityMetric.MEMORY_EFFICIENCY: {
                'excellent': 0.9,
                'good': 0.7,
                'acceptable': 0.5,
                'poor': 0.3
            },
            QualityMetric.ERROR_RATE: {
                'excellent': 0.01,
                'good': 0.05,
                'acceptable': 0.15,
                'poor': 0.30
            }
        }

    def _setup_benchmarks(self) -> Dict[FileSource, Dict[str, float]]:
        """Setup performance benchmarks for different sources."""
        return {
            FileSource.WEB_DOCUMENTS: {
                'processing_speed': 1.0,  # MB/s
                'memory_efficiency': 0.8,
                'success_rate': 0.95
            },
            FileSource.OFFICE_DOCUMENTS: {
                'processing_speed': 0.5,
                'memory_efficiency': 0.7,
                'success_rate': 0.90
            },
            FileSource.SCANNED_DOCUMENTS: {
                'processing_speed': 0.2,
                'memory_efficiency': 0.6,
                'success_rate': 0.85
            },
            FileSource.STRUCTURED_DATA: {
                'processing_speed': 2.0,
                'memory_efficiency': 0.9,
                'success_rate': 0.98
            },
            FileSource.CODE_REPOSITORIES: {
                'processing_speed': 3.0,
                'memory_efficiency': 0.85,
                'success_rate': 0.92
            }
        }

    def start_session(
        self,
        session_id: str,
        file_source: FileSource,
        strategy: ProcessingStrategy,
        file_size: Optional[int] = None,
        input_tokens: Optional[int] = None
    ) -> ProcessingSession:
        """Start a new processing session."""
        session = ProcessingSession(
            session_id=session_id,
            file_source=file_source,
            strategy=strategy,
            start_time=datetime.now(timezone.utc),
            file_size=file_size,
            input_tokens=input_tokens
        )

        self.active_sessions[session_id] = session
        return session

    def end_session(
        self,
        session_id: str,
        success: bool,
        output_tokens: Optional[int] = None,
        error: Optional[Exception] = None
    ) -> Optional[ProcessingSession]:
        """End a processing session and record final metrics."""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        session.end_time = datetime.now(timezone.utc)
        session.success = success
        session.output_tokens = output_tokens

        if error:
            session.errors.append({
                'type': type(error).__name__,
                'message': str(error),
                'timestamp': session.end_time.isoformat()
            })

        # Calculate final metrics
        self._calculate_final_metrics(session)

        # Move to completed sessions
        self.sessions.append(session)
        del self.active_sessions[session_id]

        # Trim history if needed
        if len(self.sessions) > self.history_size:
            self.sessions = self.sessions[-self.history_size:]

        return session

    def add_measurement(
        self,
        session_id: str,
        metric: QualityMetric,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Add a quality measurement to a session."""
        if session_id not in self.active_sessions:
            return

        threshold = self.quality_thresholds[metric].get('acceptable', 0.7)
        measurement = QualityMeasurement(
            metric=metric,
            value=value,
            threshold=threshold,
            timestamp=datetime.now(timezone.utc),
            context=context or {}
        )

        self.active_sessions[session_id].measurements.append(measurement)

    def _calculate_final_metrics(self, session: ProcessingSession):
        """Calculate final metrics for a completed session."""
        if not session.end_time:
            return

        duration = (session.end_time - session.start_time).total_seconds()

        # Processing speed (MB/s)
        if session.file_size:
            speed_mb_per_sec = (session.file_size / (1024 * 1024)) / duration
            self.add_measurement(
                session.session_id,
                QualityMetric.PROCESSING_SPEED,
                speed_mb_per_sec,
                {'file_size_mb': session.file_size / (1024 * 1024), 'duration_sec': duration}
            )

        # Error rate (errors per MB)
        if session.file_size and session.errors:
            error_rate = len(session.errors) / (session.file_size / (1024 * 1024))
            self.add_measurement(
                session.session_id,
                QualityMetric.ERROR_RATE,
                error_rate,
                {'error_count': len(session.errors)}
            )

    def get_quality_report(self, session_id: str) -> Optional[QualityReport]:
        """Generate a comprehensive quality report for a session."""
        # Find the session
        session = None
        for s in self.sessions:
            if s.session_id == session_id:
                session = s
                break

        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]

        if not session:
            return None

        # Calculate metric scores
        metric_scores = {}
        for metric in QualityMetric:
            measurements = [m for m in session.measurements if m.metric == metric]
            if measurements:
                metric_scores[metric] = statistics.mean([m.value for m in measurements])
            else:
                metric_scores[metric] = 0.0

        # Determine overall quality level
        overall_score = statistics.mean(list(metric_scores.values()))
        overall_quality_level = self._determine_quality_level(overall_score)

        # Generate processing stats
        processing_stats = self._calculate_processing_stats(session)

        # Generate recommendations
        recommendations = self._generate_recommendations(session, metric_scores)

        # Calculate quality trends
        quality_trends = self._calculate_quality_trends(session.file_source, metric_scores)

        # Benchmark comparison
        benchmark_comparison = self._compare_with_benchmarks(session, metric_scores)

        return QualityReport(
            session_id=session_id,
            overall_quality_level=overall_quality_level,
            overall_score=overall_score,
            metric_scores=metric_scores,
            processing_stats=processing_stats,
            recommendations=recommendations,
            quality_trends=quality_trends,
            benchmark_comparison=benchmark_comparison
        )

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.GOOD
        elif score >= 0.6:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.4:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _calculate_processing_stats(self, session: ProcessingSession) -> Dict[str, Any]:
        """Calculate processing statistics."""
        stats = {
            'duration_seconds': None,
            'success': session.success,
            'error_count': len(session.errors),
            'measurement_count': len(session.measurements)
        }

        if session.end_time:
            stats['duration_seconds'] = (session.end_time - session.start_time).total_seconds()

        if session.file_size:
            stats['file_size_mb'] = session.file_size / (1024 * 1024)

        if session.input_tokens:
            stats['input_tokens'] = session.input_tokens

        if session.output_tokens:
            stats['output_tokens'] = session.output_tokens
            if session.input_tokens:
                stats['compression_ratio'] = session.output_tokens / session.input_tokens

        return stats

    def _generate_recommendations(
        self,
        session: ProcessingSession,
        metric_scores: Dict[QualityMetric, float]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Analyze each metric
        for metric, score in metric_scores.items():
            if score < 0.5:
                if metric == QualityMetric.TEXT_EXTRACTION:
                    recommendations.append(
                        "Consider using ACCURATE processing strategy for better text extraction"
                    )
                elif metric == QualityMetric.PROCESSING_SPEED:
                    recommendations.append(
                        "Processing speed is low, consider using FAST strategy or optimizing chunk sizes"
                    )
                elif metric == QualityMetric.ERROR_RATE:
                    recommendations.append(
                        "High error rate detected, check input file integrity and processing parameters"
                    )
                elif metric == QualityMetric.STRUCTURE_PRESERVATION:
                    recommendations.append(
                        "Structure preservation is poor, enable formatting preservation options"
                    )

        # Strategy-specific recommendations
        if session.strategy == ProcessingStrategy.FAST:
            low_quality_metrics = [m for m, s in metric_scores.items() if s < 0.6]
            if len(low_quality_metrics) >= 2:
                recommendations.append(
                    "Multiple quality metrics are low with FAST strategy, consider BALANCED strategy"
                )

        # Error-based recommendations
        if session.errors:
            error_types = set(error['type'] for error in session.errors)
            if 'TimeoutError' in error_types:
                recommendations.append(
                    "Processing timeouts detected, consider increasing timeout or using smaller chunks"
                )
            if 'MemoryError' in error_types:
                recommendations.append(
                    "Memory issues detected, consider reducing chunk sizes or disabling memory-intensive features"
                )

        return recommendations

    def _calculate_quality_trends(
        self,
        source: FileSource,
        current_scores: Dict[QualityMetric, float]
    ) -> Dict[QualityMetric, List[float]]:
        """Calculate quality trends for the source type."""
        trends = {}

        # Get recent sessions for the same source
        recent_sessions = [
            s for s in self.sessions[-50:]  # Last 50 sessions
            if s.file_source == source and s.measurements
        ]

        for metric in QualityMetric:
            trend_values = []
            for session in recent_sessions:
                measurements = [m for m in session.measurements if m.metric == metric]
                if measurements:
                    trend_values.append(statistics.mean([m.value for m in measurements]))

            # Add current score
            trend_values.append(current_scores.get(metric, 0.0))
            trends[metric] = trend_values

        return trends

    def _compare_with_benchmarks(
        self,
        session: ProcessingSession,
        metric_scores: Dict[QualityMetric, float]
    ) -> Optional[Dict[str, float]]:
        """Compare performance with benchmarks."""
        if session.file_source not in self.benchmarks:
            return None

        benchmark = self.benchmarks[session.file_source]
        comparison = {}

        # Compare processing speed
        if QualityMetric.PROCESSING_SPEED in metric_scores:
            current_speed = metric_scores[QualityMetric.PROCESSING_SPEED]
            benchmark_speed = benchmark.get('processing_speed', 1.0)
            comparison['processing_speed_ratio'] = current_speed / benchmark_speed

        # Compare memory efficiency
        if QualityMetric.MEMORY_EFFICIENCY in metric_scores:
            current_memory = metric_scores[QualityMetric.MEMORY_EFFICIENCY]
            benchmark_memory = benchmark.get('memory_efficiency', 0.8)
            comparison['memory_efficiency_ratio'] = current_memory / benchmark_memory

        return comparison

    def get_aggregated_stats(
        self,
        source: Optional[FileSource] = None,
        strategy: Optional[ProcessingStrategy] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get aggregated statistics for sessions."""
        sessions = self.sessions.copy()

        # Filter by source
        if source:
            sessions = [s for s in sessions if s.file_source == source]

        # Filter by strategy
        if strategy:
            sessions = [s for s in sessions if s.strategy == strategy]

        # Filter by time window
        if time_window:
            cutoff_time = datetime.now(timezone.utc) - time_window
            sessions = [s for s in sessions if s.start_time >= cutoff_time]

        if not sessions:
            return {'session_count': 0}

        # Calculate aggregated stats
        successful_sessions = [s for s in sessions if s.success]
        total_duration = sum(
            (s.end_time - s.start_time).total_seconds()
            for s in sessions if s.end_time
        )

        return {
            'session_count': len(sessions),
            'success_rate': len(successful_sessions) / len(sessions),
            'average_duration': total_duration / len(sessions) if sessions else 0,
            'total_errors': sum(len(s.errors) for s in sessions),
            'most_common_source': max(set(s.file_source for s in sessions), key=lambda x: sum(1 for s in sessions if s.file_source == x)),
            'most_common_strategy': max(set(s.strategy for s in sessions), key=lambda x: sum(1 for s in sessions if s.strategy == x))
        }

    async def monitor_processing_async(
        self,
        session_id: str,
        processing_coroutine,
        *args,
        **kwargs
    ) -> Any:
        """Async wrapper to monitor processing coroutine."""
        try:
            result = await processing_coroutine(*args, **kwargs)
            self.end_session(session_id, success=True)
            return result
        except Exception as e:
            self.end_session(session_id, success=False, error=e)
            raise

    def export_metrics(self, file_path: Union[str, Path]):
        """Export quality metrics to file."""
        export_data = {
            'sessions': [
                {
                    'session_id': s.session_id,
                    'file_source': s.file_source.value,
                    'strategy': s.strategy.value,
                    'start_time': s.start_time.isoformat(),
                    'end_time': s.end_time.isoformat() if s.end_time else None,
                    'success': s.success,
                    'file_size': s.file_size,
                    'measurements': [
                        {
                            'metric': m.metric.value,
                            'value': m.value,
                            'threshold': m.threshold,
                            'timestamp': m.timestamp.isoformat(),
                            'context': m.context
                        }
                        for m in s.measurements
                    ],
                    'errors': s.errors
                }
                for s in self.sessions
            ],
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)