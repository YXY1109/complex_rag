"""
Async Batch Processing Service

This module provides high-performance batch processing capabilities for
document processing, including concurrent execution, resource management,
and progress tracking.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import time
import traceback
from collections import defaultdict

from .processing_pipeline import DocumentProcessingPipeline, PipelineResult
from ..interfaces.source_interface import ParseRequest, ParseResponse

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Batch processing status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Individual item in a batch."""
    id: str
    request: ParseRequest
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[PipelineResult] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: float = 0.0
    retry_count: int = 0


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    max_concurrent: int = 4
    timeout_per_item: float = 300.0  # 5 minutes per item
    batch_timeout: float = 3600.0  # 1 hour per batch
    retry_failed_items: bool = True
    max_retries: int = 3
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    enable_progress_tracking: bool = True


@dataclass
class BatchJob:
    """Batch processing job."""
    id: str
    name: str
    items: List[BatchItem]
    config: BatchConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: BatchStatus = BatchStatus.PENDING
    pipeline_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Batch processing progress information."""
    batch_id: str
    total_items: int
    completed_items: int
    failed_items: int
    processing_items: int
    success_rate: float
    average_processing_time: float
    estimated_completion_time: Optional[datetime]
    items_per_second: float


class BatchProcessor:
    """
    High-performance async batch processor for document processing.

    Features:
    - Concurrent processing with configurable limits
    - Progress tracking and callbacks
    - Error handling and retry logic
    - Resource management and optimization
    - Priority-based processing
    - Streaming batch processing
    """

    def __init__(self, pipeline: Optional[DocumentProcessingPipeline] = None):
        """Initialize batch processor."""
        self.pipeline = pipeline or DocumentProcessingPipeline()
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.processing_semaphore = asyncio.Semaphore(10)  # Overall limit
        self.job_queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []

        # Statistics
        self.total_processed = 0
        self.total_failed = 0
        self.total_success = 0

    async def create_batch(
        self,
        requests: List[ParseRequest],
        name: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        config: Optional[BatchConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchJob:
        """
        Create a new batch job.

        Args:
            requests: List of parse requests
            name: Optional batch name
            pipeline_name: Pipeline to use
            config: Batch configuration
            metadata: Additional metadata

        Returns:
            BatchJob: Created batch job
        """
        batch_id = str(uuid.uuid())
        batch_name = name or f"batch_{batch_id[:8]}"

        # Create batch items
        items = []
        for request in requests:
            item = BatchItem(
                id=str(uuid.uuid()),
                request=request,
                metadata={'original_index': len(items)}
            )
            items.append(item)

        # Create batch job
        job = BatchJob(
            id=batch_id,
            name=batch_name,
            items=items,
            config=config or BatchConfig(),
            created_at=datetime.now(),
            pipeline_name=pipeline_name,
            metadata=metadata or {}
        )

        self.active_jobs[batch_id] = job
        logger.info(f"Created batch job {batch_id} with {len(items)} items")

        return job

    async def process_batch(self, batch_id: str) -> BatchJob:
        """
        Process a batch job asynchronously.

        Args:
            batch_id: Batch job ID

        Returns:
            BatchJob: Completed batch job
        """
        if batch_id not in self.active_jobs:
            raise ValueError(f"Batch job {batch_id} not found")

        job = self.active_jobs[batch_id]
        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now()

        try:
            # Process items concurrently
            await self._process_items_concurrent(job)

            # Mark as completed
            job.status = BatchStatus.COMPLETED
            job.completed_at = datetime.now()

            # Move to completed jobs
            self.completed_jobs[batch_id] = job
            del self.active_jobs[batch_id]

            # Update statistics
            self._update_statistics(job)

            logger.info(f"Completed batch job {batch_id} with {len(job.items)} items")

        except Exception as e:
            logger.error(f"Batch job {batch_id} failed: {e}")
            job.status = BatchStatus.FAILED
            job.completed_at = datetime.now()

            # Move to completed jobs even on failure
            self.completed_jobs[batch_id] = job
            del self.active_jobs[batch_id]

            # Update statistics
            self._update_statistics(job)

        return job

    async def _process_items_concurrent(self, job: BatchJob):
        """Process batch items concurrently."""
        # Create semaphore for this job
        semaphore = asyncio.Semaphore(job.config.max_concurrent)

        # Create processing tasks
        tasks = [
            self._process_item_with_semaphore(item, job, semaphore)
            for item in job.items
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        for i, result in enumerate(results):
            item = job.items[i]
            if isinstance(result, Exception):
                # Handle error
                item.status = BatchStatus.FAILED
                item.error = str(result)
                logger.error(f"Item {item.id} failed: {result}")
            else:
                # Handle success
                item.result = result
                item.status = BatchStatus.COMPLETED if result.success else BatchStatus.FAILED
                if result.error:
                    item.error = result.error

                # Retry logic for failed items
                if item.status == BatchStatus.FAILED and job.config.retry_failed_items:
                    if item.retry_count < job.config.max_retries:
                        logger.info(f"Retrying item {item.id} (attempt {item.retry_count + 1})")
                        await asyncio.sleep(1)  # Brief delay before retry
                        retry_result = await self._process_item_with_semaphore(item, job, semaphore)
                        item.result = retry_result
                        item.status = BatchStatus.COMPLETED if retry_result.success else BatchStatus.FAILED
                        item.retry_count += 1
                        item.completed_at = datetime.now()

            item.processing_time_seconds = (
                (item.completed_at - item.started_at).total_seconds()
                if item.completed_at else 0
            )

    async def _process_item_with_semaphore(
        self,
        item: BatchItem,
        job: BatchJob,
        semaphore: asyncio.Semaphore
    ) -> PipelineResult:
        """Process a single item with semaphore control."""
        async with semaphore:
            return await self._process_single_item(item, job)

    async def _process_single_item(self, item: BatchItem, job: BatchJob) -> PipelineResult:
        """Process a single document."""
        item.status = BatchStatus.RUNNING
        item.started_at = datetime.now()

        try:
            # Process through pipeline
            result = await self.pipeline.process_document(
                item.request,
                job.pipeline_name
            )

            return result

        except Exception as e:
            logger.error(f"Item {item.id} processing failed: {e}")
            # Return a failure result
            return PipelineResult(
                pipeline_name=job.pipeline_name or "unknown",
                success=False,
                processing_time_seconds=(datetime.now() - item.started_at).total_seconds(),
                errors=[{
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }]
            )

    async def stream_process(
        self,
        requests: AsyncGenerator[ParseRequest, None],
        pipeline_name: Optional[str] = None,
        config: Optional[BatchConfig] = None
    ) -> AsyncGenerator[BatchItem, None]:
        """
        Stream process documents as they become available.

        Args:
            requests: Async generator of parse requests
            pipeline_name: Pipeline to use
            config: Batch configuration

        Yields:
            BatchItem: Processing results as they complete
        """
        batch_id = str(uuid.uuid())
        batch_name = f"stream_batch_{batch_id[:8]}"
        config = config or BatchConfig()

        # Create processing tasks as requests come in
        active_tasks = {}

        try:
            async for request in requests:
                # Create batch item
                item = BatchItem(
                    id=str(uuid.uuid()),
                    request=request,
                    metadata={'stream_index': len(active_tasks)}
                )

                # Start processing task
                task = asyncio.create_task(
                    self._process_single_item_with_config(item, pipeline_name, config)
                )
                active_tasks[item.id] = (item, task)

                # Yield results from completed tasks
                await self._yield_completed_items(active_tasks)

        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
        finally:
            # Wait for remaining tasks
            for item_id, (item, task) in active_tasks.items():
                try:
                    result = await task
                    item.result = result
                    item.status = BatchStatus.COMPLETED if result.success else BatchStatus.FAILED
                    item.completed_at = datetime.now()
                    yield item
                except Exception as e:
                    logger.error(f"Stream item {item_id} completion failed: {e}")

    async def _process_single_item_with_config(
        self,
        item: BatchItem,
        pipeline_name: Optional[str],
        config: BatchConfig
    ) -> PipelineResult:
        """Process a single item with specific config."""
        item.status = BatchStatus.RUNNING
        item.started_at = datetime.now()

        try:
            # Apply timeout
            result = await asyncio.wait_for(
                self.pipeline.process_document(item.request, pipeline_name),
                timeout=config.timeout_per_item
            )

            return result

        except asyncio.TimeoutError:
            logger.warning(f"Item {item.id} processing timed out")
            return PipelineResult(
                pipeline_name=pipeline_name or "unknown",
                success=False,
                processing_time_seconds=config.timeout_per_item,
                errors=[{'error': 'Processing timeout', 'timestamp': datetime.now().isoformat()}]
            )

    async def _yield_completed_items(self, active_tasks: Dict[str, Tuple[BatchItem, asyncio.Task]]):
        """Yield results from completed tasks."""
        completed_ids = []

        while active_tasks:
            # Check for completed tasks
            for item_id, (item, task) in list(active_tasks.items()):
                if item_id not in completed_ids and task.done():
                    # Get result
                    try:
                        result = await task
                        item.result = result
                        item.status = BatchStatus.COMPLETED if result.success else BatchStatus.FAILED
                        item.completed_at = datetime.now()
                        yield item
                        completed_ids.append(item_id)
                    except Exception as e:
                        logger.error(f"Task completion failed for {item_id}: {e}")
                        item.status = BatchStatus.FAILED
                        item.completed_at = datetime.now()
                        item.error = str(e)
                        yield item
                        completed_ids.append(item_id)

            # Remove completed items from active tasks
            for item_id in completed_ids:
                del active_tasks[item_id]

            # Brief pause to prevent busy waiting
            if active_tasks:
                await asyncio.sleep(0.1)

    def _update_statistics(self, job: BatchJob):
        """Update processing statistics."""
        total_items = len(job.items)
        completed_items = sum(1 for item in job.items if item.status == BatchStatus.COMPLETED)
        failed_items = sum(1 for item in job.items if item.status == BatchStatus.FAILED)

        self.total_processed += total_items
        self.total_success += completed_items
        self.total_failed += failed_items

    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get progress information for a batch job."""
        job = self.active_jobs.get(batch_id) or self.completed_jobs.get(batch_id)
        if not job:
            return None

        total_items = len(job.items)
        completed_items = sum(1 for item in job.items if item.status == BatchStatus.COMPLETED)
        failed_items = sum(1 for item in job.items if item.status == BatchStatus.FAILED)
        processing_items = sum(1 for item in job.items if item.status == BatchStatus.RUNNING)

        success_rate = completed_items / total_items if total_items > 0 else 0.0

        # Calculate average processing time
        completed_processing_times = [
            item.processing_time_seconds for item in job.items
            if item.processing_time_seconds > 0
        ]
        avg_processing_time = sum(completed_processing_times) / len(completed_processing_times) if completed_processing_times else 0.0

        # Calculate items per second
        if job.started_at:
            elapsed_time = (datetime.now() - job.started_at).total_seconds()
            items_per_second = completed_items / elapsed_time if elapsed_time > 0 else 0.0
        else:
            items_per_second = 0.0

        # Estimate completion time
        estimated_completion = None
        if processing_items > 0 and avg_processing_time > 0:
            remaining_items = total_items - completed_items - processing_items
            if remaining_items > 0:
                estimated_time = datetime.now() + timedelta(seconds=remaining_items * avg_processing_time)
                estimated_completion = estimated_time

        return BatchProgress(
            batch_id=batch_id,
            total_items=total_items,
            completed_items=completed_items,
            failed_items=failed_items,
            processing_items=processing_items,
            success_rate=success_rate,
            average_processing_time=avg_processing_time,
            estimated_completion_time=estimated_completion,
            items_per_second=items_per_second
        )

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch job."""
        if batch_id not in self.active_jobs:
            return False

        job = self.active_jobs[batch_id]
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.now()

        # Mark items as cancelled
        for item in job.items:
            if item.status in [BatchStatus.PENDING, BatchStatus.RUNNING]:
                item.status = BatchStatus.CANCELLED
                item.completed_at = datetime.now()

        # Move to completed jobs
        self.completed_jobs[batch_id] = job
        del self.active_jobs[batch_id]

        logger.info(f"Cancelled batch job {batch_id}")
        return True

    def get_job(self, batch_id: str) -> Optional[BatchJob]:
        """Get batch job by ID."""
        return self.active_jobs.get(batch_id) or self.completed_jobs.get(batch_id)

    def list_active_jobs(self) -> List[BatchJob]:
        """List all active batch jobs."""
        return list(self.active_jobs.values())

    def list_completed_jobs(self) -> List[BatchJob]:
        """List all completed batch jobs."""
        return list(self.completed_jobs.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_processed': self.total_processed,
            'total_success': self.total_success,
            'total_failed': self.total_failed,
            'success_rate': self.total_success / self.total_processed if self.total_processed > 0 else 0.0,
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'pipeline_stats': self.pipeline.get_pipeline_stats() if self.pipeline else {}
        }

    async def cleanup_completed_jobs(self, older_than_hours: int = 24):
        """Clean up completed jobs older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        jobs_to_remove = [
            batch_id for batch_id, job in self.completed_jobs.items()
            if job.completed_at and job.completed_at < cutoff_time
        ]

        for batch_id in jobs_to_remove:
            del self.completed_jobs[batch_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} completed batch jobs")