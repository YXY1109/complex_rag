"""
异步任务性能测试
测试异步任务处理性能
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from .framework import PerformanceTestFramework, BenchmarkSuite


@dataclass
class AsyncTaskResult:
    """异步任务结果"""
    task_id: str
    start_time: float
    end_time: float
    duration: float  # 毫秒
    success: bool
    result: Any = None
    error: str = None
    worker_id: int = None


class AsyncTaskPerformanceTester:
    """异步任务性能测试器"""

    def __init__(self):
        self.framework = PerformanceTestFramework()
        self.thread_pool = ThreadPoolExecutor(max_workers=100)
        self.results: List[AsyncResult] = []

    async def create_test_task(self, task_id: str, duration: float = 0.1, cpu_intensive: bool = False) -> AsyncTaskResult:
        """创建测试任务"""
        start_time = time.time()

        try:
            if cpu_intensive:
                # CPU密集型任务
                result = await self._cpu_intensive_task(duration)
            else:
                # IO密集型任务
                result = await self._io_intensive_task(duration)

            end_time = time.time()
            return AsyncTaskResult(
                task_id=task_id,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time) * 1000,
                success=True,
                result=result
            )
        except Exception as e:
            end_time = time.time()
            return AsyncTaskResult(
                task_id=task_id,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time) * 1000,
                success=False,
                error=str(e)
            )

    async def _io_intensive_task(self, duration: float) -> str:
        """IO密集型任务模拟"""
        await asyncio.sleep(duration)
        return f"IO task completed in {duration}s"

    async def _cpu_intensive_task(self, duration: float) -> int:
        """CPU密集型任务模拟"""
        # 计算密集型操作
        result = 0
        start = time.time()
        while time.time() - start < duration:
            result += sum(i * i for i in range(1000))
        return result

    async def test_async_task_creation(self) -> Dict[str, Any]:
        """测试异步任务创建性能"""
        task_count = 1000
        tasks = []

        start_time = time.time()

        # 创建大量异步任务
        for i in range(task_count):
            task = asyncio.create_task(
                self.create_test_task(f"task_{i}", 0.01)
            )
            tasks.append(task)

        creation_time = time.time() - start_time

        # 等待所有任务完成
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        completion_time = time.time() - start_time

        # 分析结果
        successful_tasks = [r for r in results if isinstance(r, AsyncTaskResult) and r.success]
        failed_tasks = [r for r in results if isinstance(r, AsyncTaskResult) and not r.success]

        return {
            "task_count": task_count,
            "creation_time": creation_time,
            "creation_rate": task_count / creation_time,
            "completion_time": completion_time,
            "completion_rate": len(successful_tasks) / completion_time,
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(successful_tasks) / task_count,
            "avg_task_duration": sum(r.duration for r in successful_tasks) / len(successful_tasks) if successful_tasks else 0
        }

    async def test_async_task_concurrency(self) -> Dict[str, Any]:
        """测试异步任务并发性能"""
        concurrency_levels = [10, 50, 100, 500, 1000]
        results = {}

        for concurrency in concurrency_levels:
            print(f"Testing with {concurrency} concurrent tasks...")

            # 创建并发任务
            tasks = [
                self.create_test_task(f"task_{i}", 0.1)
                for i in range(concurrency)
            ]

            start_time = time.time()
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # 分析结果
            successful = [r for r in task_results if isinstance(r, AsyncTaskResult) and r.success]
            failed = [r for r in task_results if isinstance(r, AsyncTaskResult) and not r.success]

            if successful:
                durations = [r.duration for r in successful]
                results[concurrency] = {
                    "total_time": end_time - start_time,
                    "successful_tasks": len(successful),
                    "failed_tasks": len(failed),
                    "success_rate": len(successful) / concurrency,
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "throughput": len(successful) / (end_time - start_time)
                }
            else:
                results[concurrency] = {
                    "total_time": end_time - start_time,
                    "successful_tasks": 0,
                    "failed_tasks": len(failed),
                    "success_rate": 0.0,
                    "throughput": 0.0
                }

        return results

    async def test_async_queue_processing(self) -> Dict[str, Any]:
        """测试异步队列处理性能"""
        queue_size = 1000
        task_duration = 0.01
        worker_count = 10

        # 创建异步队列
        queue = asyncio.Queue(maxsize=queue_size * 2)

        # 生产者任务
        async def producer():
            for i in range(queue_size):
                await queue.put(f"task_{i}")

        # 消费者任务
        async def consumer(worker_id: int):
            processed = 0
            while True:
                try:
                    task_id = queue.get_nowait()
                    # 处理任务
                    await asyncio.sleep(task_duration)
                    processed += 1
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
            return processed

        start_time = time.time()

        # 启动生产者
        producer_task = asyncio.create_task(producer())

        # 等待生产者完成
        await producer_task

        # 启动消费者
        consumer_tasks = [
            asyncio.create_task(consumer(i))
            for i in range(worker_count)
        ]

        # 等待所有任务被处理
        await queue.join()

        # 获取消费者结果
        consumer_results = await asyncio.gather(*consumer_tasks)

        end_time = time.time()
        total_processed = sum(consumer_results)

        return {
            "queue_size": queue_size,
            "worker_count": worker_count,
            "total_time": end_time - start_time,
            "total_processed": total_processed,
            "processing_rate": total_processed / (end_time - start_time),
            "avg_tasks_per_worker": total_processed / worker_count,
            "success_rate": total_processed / queue_size
        }

    async def test_async_semaphore_limiting(self) -> Dict[str, Any]:
        """测试异步信号量限制性能"""
        total_tasks = 500
        semaphore_limits = [10, 25, 50, 100, 200]
        results = {}

        for limit in semaphore_limits:
            print(f"Testing semaphore limit: {limit}")

            semaphore = asyncio.Semaphore(limit)

            async def limited_task(task_id: int):
                async with semaphore:
                    return await self.create_test_task(f"task_{task_id}", 0.05)

            start_time = time.time()

            tasks = [limited_task(i) for i in range(total_tasks)]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()

            # 分析结果
            successful = [r for r in task_results if isinstance(r, AsyncTaskResult) and r.success]
            failed = [r for r in task_results if isinstance(r, AsyncTaskResult) and not r.success]

            if successful:
                durations = [r.duration for r in successful]
                results[limit] = {
                    "total_time": end_time - start_time,
                    "successful_tasks": len(successful),
                    "failed_tasks": len(failed),
                    "success_rate": len(successful) / total_tasks,
                    "avg_duration": sum(durations) / len(durations),
                    "throughput": len(successful) / (end_time - start_time)
                }
            else:
                results[limit] = {
                    "total_time": end_time - start_time,
                    "successful_tasks": 0,
                    "failed_tasks": len(failed),
                    "success_rate": 0.0,
                    "throughput": 0.0
                }

        return results

    async def test_async_event_driven_processing(self) -> Dict[str, Any]:
        """测试事件驱动异步处理性能"""
        event_count = 1000
        events_processed = 0

        # 创建事件队列
        event_queue = asyncio.Queue()

        # 事件处理器
        async def event_handler():
            nonlocal events_processed
            while True:
                try:
                    event = event_queue.get_nowait()
                    # 模拟事件处理
                    await asyncio.sleep(0.001)  # 1ms处理时间
                    events_processed += 1
                    event_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        start_time = time.time()

        # 生产事件
        for i in range(event_count):
            await event_queue.put({
                "id": i,
                "type": "test_event",
                "data": f"event_data_{i}",
                "timestamp": time.time()
            })

        # 启动事件处理器
        handler_task = asyncio.create_task(event_handler())

        # 等待所有事件被处理
        await event_queue.join()
        await handler_task

        end_time = time.time()

        return {
            "event_count": event_count,
            "events_processed": events_processed,
            "processing_time": end_time - start_time,
            "processing_rate": events_processed / (end_time - start_time),
            "success_rate": events_processed / event_count
        }

    async def test_async_batch_processing(self) -> Dict[str, Any]:
        """测试异步批处理性能"""
        total_items = 1000
        batch_sizes = [10, 50, 100, 200]
        results = {}

        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")

            # 分批处理
            batches = [
                list(range(i, min(i + batch_size, total_items)))
                for i in range(0, total_items, batch_size)
            ]

            start_time = time.time()

            batch_results = []
            for batch in batches:
                # 并发处理每个批次
                batch_tasks = [
                    self.create_test_task(f"item_{item}", 0.001)
                    for item in batch
                ]
                batch_result = await asyncio.gather(*batch_tasks, return_exceptions=True)
                batch_results.extend(batch_result)

            end_time = time.time()

            # 分析结果
            successful = [r for r in batch_results if isinstance(r, AsyncTaskResult) and r.success]
            failed = [r for r in batch_results if isinstance(r, AsyncTaskResult) and not r.success]

            if successful:
                durations = [r.duration for r in successful]
                results[batch_size] = {
                    "batch_count": len(batches),
                    "total_time": end_time - start_time,
                    "successful_items": len(successful),
                    "failed_items": len(failed),
                    "success_rate": len(successful) / total_items,
                    "avg_duration": sum(durations) / len(durations),
                    "throughput": len(successful) / (end_time - start_time)
                }
            else:
                results[batch_size] = {
                    "batch_count": len(batches),
                    "total_time": end_time - start_time,
                    "successful_items": 0,
                    "failed_items": len(failed),
                    "success_rate": 0.0,
                    "throughput": 0.0
                }

        return results

    async def test_async_memory_efficiency(self) -> Dict[str, Any]:
        """测试异步任务内存效率"""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建大量异步任务但延迟执行
        task_count = 10000
        tasks = []

        start_time = time.time()

        # 创建任务但不立即执行
        for i in range(task_count):
            task = asyncio.create_task(
                self.create_test_task(f"memory_task_{i}", 0.001)
            )
            tasks.append(task)

            # 每1000个任务检查一次内存
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"  Created {i} tasks, memory: {current_memory:.2f}MB")

        creation_memory = process.memory_info().rss / 1024 / 1024

        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 等待垃圾回收
        await asyncio.sleep(1)

        final_memory = process.memory_info().rss / 1024 / 1024

        end_time = time.time()

        # 分析结果
        successful = [r for r in results if isinstance(r, AsyncTaskResult) and r.success]

        return {
            "task_count": task_count,
            "initial_memory_mb": initial_memory,
            "creation_memory_mb": creation_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": max(creation_memory, final_memory),
            "memory_growth_mb": final_memory - initial_memory,
            "memory_per_task_kb": (final_memory - initial_memory) * 1024 / task_count,
            "total_time": end_time - start_time,
            "successful_tasks": len(successful),
            "success_rate": len(successful) / task_count
        }

    async def run_async_performance_suite(self) -> Dict[str, Any]:
        """运行异步性能测试套件"""
        print("🚀 Starting Async Task Performance Tests")

        # 1. 任务创建性能
        print("\n📝 Testing task creation performance...")
        creation_result = await self.test_async_task_creation()

        # 2. 并发性能
        print("\n🔀 Testing concurrency performance...")
        concurrency_result = await self.test_async_task_concurrency()

        # 3. 队列处理性能
        print("\n📋 Testing queue processing performance...")
        queue_result = await self.test_async_queue_processing()

        # 4. 信号量限制性能
        print("\n🚦 Testing semaphore limiting performance...")
        semaphore_result = await self.test_async_semaphore_limiting()

        # 5. 事件驱动处理性能
        print("\n⚡ Testing event-driven processing performance...")
        event_result = await self.test_async_event_driven_processing()

        # 6. 批处理性能
        print("\n📦 Testing batch processing performance...")
        batch_result = await self.test_async_batch_processing()

        # 7. 内存效率测试
        print("\n💾 Testing memory efficiency...")
        memory_result = await self.test_async_memory_efficiency()

        return {
            "task_creation": creation_result,
            "concurrency": concurrency_result,
            "queue_processing": queue_result,
            "semaphore_limiting": semaphore_result,
            "event_driven": event_result,
            "batch_processing": batch_result,
            "memory_efficiency": memory_result
        }

    def generate_async_performance_report(self, results: Dict[str, Any]):
        """生成异步性能报告"""
        print("\n📊 Generating Async Performance Report...")

        report = {
            "test_summary": {
                "timestamp": time.time(),
                "total_tests": len(results)
            },
            "results": results
        }

        # 保存报告
        import json
        from pathlib import Path

        report_path = f"tests/performance/reports/async_performance_{int(time.time())}.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Async performance report saved to: {report_path}")

        # 打印摘要
        self._print_async_performance_summary(results)

        return report_path

    def _print_async_performance_summary(self, results: Dict):
        """打印异步性能摘要"""
        print(f"\n{'='*80}")
        print("📊 Async Task Performance Summary")
        print(f"{'='*80}")

        # 任务创建摘要
        if "task_creation" in results:
            creation = results["task_creation"]
            print(f"\n📝 Task Creation:")
            print(f"  Creation Rate: {creation.get('creation_rate', 0):.2f} tasks/sec")
            print(f"  Completion Rate: {creation.get('completion_rate', 0):.2f} tasks/sec")
            print(f"  Success Rate: {creation.get('success_rate', 0):.2%}")

        # 并发性能摘要
        if "concurrency" in results:
            concurrency = results["concurrency"]
            print(f"\n🔀 Concurrency Performance:")
            print(f"{'Concurrency':<12} {'Throughput':<12} {'Success Rate':<12} {'Avg Duration':<12}")
            print(f"{'-'*50}")
            for level, metrics in concurrency.items():
                print(f"{level:<12} "
                      f"{metrics.get('throughput', 0):<12.1f} "
                      f"{metrics.get('success_rate', 0):<12.1%} "
                      f"{metrics.get('avg_duration', 0):<12.1f}")

        # 队列处理摘要
        if "queue_processing" in results:
            queue = results["queue_processing"]
            print(f"\n📋 Queue Processing:")
            print(f"  Processing Rate: {queue.get('processing_rate', 0):.2f} tasks/sec")
            print(f"  Success Rate: {queue.get('success_rate', 0):.2%}")
            print(f"  Avg Tasks per Worker: {queue.get('avg_tasks_per_worker', 0):.1f}")

        # 内存效率摘要
        if "memory_efficiency" in results:
            memory = results["memory_efficiency"]
            print(f"\n💾 Memory Efficiency:")
            print(f"  Memory Growth: {memory.get('memory_growth_mb', 0):.2f}MB")
            print(f"  Memory per Task: {memory.get('memory_per_task_kb', 0):.2f}KB")
            print(f"  Success Rate: {memory.get('success_rate', 0):.2%}")

        print(f"\n{'='*80}")


# 便捷函数
async def run_async_performance_tests():
    """运行异步性能测试的主函数"""
    tester = AsyncTaskPerformanceTester()

    print("🎯 Starting Comprehensive Async Performance Tests")

    # 运行完整的性能测试套件
    results = await tester.run_async_performance_suite()

    # 生成报告
    report_path = tester.generate_async_performance_report(results)

    return results, report_path