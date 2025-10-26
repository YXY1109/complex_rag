"""
性能测试框架
提供性能测试的基础设施和工具
"""
import time
import asyncio
import statistics
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    name: str
    duration: float  # 毫秒
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time: float  # 毫秒
    min_response_time: float  # 毫秒
    max_response_time: float  # 毫秒
    p50_response_time: float  # 毫秒
    p95_response_time: float  # 毫秒
    p99_response_time: float  # 毫秒
    requests_per_second: float
    total_duration: float  # 秒
    errors: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResourceMonitor:
    """系统资源监控器"""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = []
        self.start_time = None
        self.stop_time = None

    def start_monitoring(self, interval: float = 0.5):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = []

        def monitor_loop():
            while self.monitoring:
                try:
                    # CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=None)

                    # 内存使用情况
                    memory = psutil.virtual_memory()

                    # 磁盘使用情况
                    disk = psutil.disk_usage('/')

                    # 网络IO
                    net_io = psutil.net_io_counters()

                    # 进程信息
                    process = psutil.Process()

                    metric = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_mb': memory.used / 1024 / 1024,
                        'disk_percent': disk.percent,
                        'disk_used_gb': disk.used / 1024 / 1024 / 1024,
                        'network_bytes_sent': net_io.bytes_sent,
                        'network_bytes_recv': net_io.bytes_recv,
                        'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                        'process_cpu_percent': process.cpu_percent(),
                        'process_threads': process.num_threads(),
                        'process_open_files': len(process.open_files())
                    }

                    self.metrics.append(metric)
                    time.sleep(interval)

                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    break

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.stop_time = time.time()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.metrics:
            return {}

        # 计算统计信息
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        process_memory_values = [m['process_memory_mb'] for m in self.metrics]

        return {
            'duration': self.stop_time - self.start_time if self.stop_time else time.time() - self.start_time,
            'sample_count': len(self.metrics),
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'process_memory': {
                'avg': statistics.mean(process_memory_values),
                'max': max(process_memory_values),
                'min': min(process_memory_values),
                'std': statistics.stdev(process_memory_values) if len(process_memory_values) > 1 else 0
            },
            'start_time': self.start_time,
            'stop_time': self.stop_time
        }


class PerformanceTestFramework:
    """性能测试框架"""

    def __init__(self):
        self.results: List[PerformanceMetrics] = []
        self.resource_monitor = ResourceMonitor()
        self.test_metadata = {}

    async def run_concurrent_test(
        self,
        test_func: Callable,
        concurrency: int,
        total_requests: int,
        **kwargs
    ) -> BenchmarkResult:
        """运行并发测试"""
        print(f"Running concurrent test: {concurrency} concurrent connections, {total_requests} total requests")

        # 开始资源监控
        self.resource_monitor.start_monitoring()

        # 准备测试
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        start_time = time.time()

        async def bounded_test():
            async with semaphore:
                return await self._run_single_test(test_func, **kwargs)

        # 执行并发请求
        tasks = [bounded_test() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # 停止资源监控
        self.resource_monitor.stop_monitoring()

        # 处理结果
        successful_results = []
        errors = []

        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif isinstance(result, PerformanceMetrics):
                successful_results.append(result)
                if not result.success:
                    errors.append(result.error_message or "Unknown error")
            else:
                errors.append(f"Unexpected result type: {type(result)}")

        # 计算性能指标
        if successful_results:
            durations = [r.duration for r in successful_results if r.success]
            if durations:
                durations.sort()
                n = len(durations)

                benchmark = BenchmarkResult(
                    test_name=test_func.__name__,
                    total_requests=total_requests,
                    successful_requests=len([r for r in successful_results if r.success]),
                    failed_requests=total_requests - len([r for r in successful_results if r.success]),
                    success_rate=len([r for r in successful_results if r.success]) / total_requests,
                    avg_response_time=statistics.mean(durations),
                    min_response_time=min(durations),
                    max_response_time=max(durations),
                    p50_response_time=durations[int(n * 0.5)],
                    p95_response_time=durations[int(n * 0.95)],
                    p99_response_time=durations[int(n * 0.99)],
                    requests_per_second=total_requests / (end_time - start_time),
                    total_duration=end_time - start_time,
                    errors=errors[:10],  # 只保留前10个错误
                    metadata={
                        'concurrency': concurrency,
                        'resource_usage': self.resource_monitor.get_summary(),
                        'test_metadata': self.test_metadata
                    }
                )
            else:
                benchmark = BenchmarkResult(
                    test_name=test_func.__name__,
                    total_requests=total_requests,
                    successful_requests=0,
                    failed_requests=total_requests,
                    success_rate=0.0,
                    avg_response_time=0.0,
                    min_response_time=0.0,
                    max_response_time=0.0,
                    p50_response_time=0.0,
                    p95_response_time=0.0,
                    p99_response_time=0.0,
                    requests_per_second=0.0,
                    total_duration=end_time - start_time,
                    errors=errors[:10],
                    metadata={
                        'concurrency': concurrency,
                        'resource_usage': self.resource_monitor.get_summary(),
                        'test_metadata': self.test_metadata
                    }
                )
        else:
            benchmark = BenchmarkResult(
                test_name=test_func.__name__,
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=total_requests,
                success_rate=0.0,
                avg_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                p50_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                requests_per_second=0.0,
                total_duration=end_time - start_time,
                errors=errors[:10],
                metadata={
                    'concurrency': concurrency,
                    'resource_usage': self.resource_monitor.get_summary(),
                    'test_metadata': self.test_metadata
                }
            )

        # 打印结果摘要
        self._print_benchmark_summary(benchmark)

        return benchmark

    async def _run_single_test(self, test_func: Callable, **kwargs) -> PerformanceMetrics:
        """运行单个测试"""
        start_time = time.time()
        success = False
        error_message = None
        result_data = None

        try:
            result = await test_func(**kwargs)
            success = True
            result_data = result
        except Exception as e:
            error_message = str(e)
            success = False

        end_time = time.time()
        duration = (end_time - start_time) * 1000  # 转换为毫秒

        metric = PerformanceMetrics(
            name=test_func.__name__,
            duration=duration,
            success=success,
            error_message=error_message,
            metadata={
                'result': result_data,
                'kwargs': kwargs
            }
        )

        self.results.append(metric)
        return metric

    def _print_benchmark_summary(self, result: BenchmarkResult):
        """打印基准测试摘要"""
        print(f"\n{'='*60}")
        print(f"🚀 Performance Test Results: {result.test_name}")
        print(f"{'='*60}")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Success Rate: {result.success_rate:.2%}")
        print(f"Requests/sec: {result.requests_per_second:.2f}")
        print(f"Total Duration: {result.total_duration:.2f}s")
        print(f"\nResponse Times (ms):")
        print(f"  Average: {result.avg_response_time:.2f}")
        print(f"  Min: {result.min_response_time:.2f}")
        print(f"  Max: {result.max_response_time:.2f}")
        print(f"  50th percentile: {result.p50_response_time:.2f}")
        print(f"  95th percentile: {result.p95_response_time:.2f}")
        print(f"  99th percentile: {result.p99_response_time:.2f}")

        # 资源使用情况
        if result.metadata.get('resource_usage'):
            resource = result.metadata['resource_usage']
            print(f"\nResource Usage:")
            print(f"  CPU Avg: {resource.get('cpu', {}).get('avg', 0):.1f}%")
            print(f"  Memory Avg: {resource.get('memory', {}).get('avg', 0):.1f}%")
            print(f"  Process Memory Avg: {resource.get('process_memory', {}).get('avg', 0):.1f}MB")

        if result.errors:
            print(f"\nErrors (showing first 5):")
            for error in result.errors[:5]:
                print(f"  - {error}")

        print(f"{'='*60}\n")

    def save_results(self, filepath: str):
        """保存测试结果"""
        results_data = {
            'test_metadata': self.test_metadata,
            'results': [asdict(r) for r in self.results],
            'resource_monitoring': self.resource_monitor.metrics
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"Test results saved to: {filepath}")


class BenchmarkSuite:
    """基准测试套件"""

    def __init__(self, name: str):
        self.name = name
        self.framework = PerformanceTestFramework()
        self.benchmarks = []

    def add_benchmark(self, test_func: Callable, name: str = None, **kwargs):
        """添加基准测试"""
        self.benchmarks.append({
            'func': test_func,
            'name': name or test_func.__name__,
            'kwargs': kwargs
        })

    async def run_all_benchmarks(self, concurrency_levels: List[int] = [1, 10, 50, 100]):
        """运行所有基准测试"""
        print(f"\n🎯 Running Benchmark Suite: {self.name}")
        print(f"Concurrency levels: {concurrency_levels}")
        print(f"Total benchmarks: {len(self.benchmarks)}")

        all_results = []

        for benchmark in self.benchmarks:
            print(f"\n📊 Running benchmark: {benchmark['name']}")

            benchmark_results = []
            for concurrency in concurrency_levels:
                result = await self.framework.run_concurrent_test(
                    benchmark['func'],
                    concurrency=concurrency,
                    total_requests=min(concurrency * 10, 1000),  # 根据并发数调整请求数
                    **benchmark['kwargs']
                )

                benchmark_results.append({
                    'concurrency': concurrency,
                    'result': result
                })

            all_results.append({
                'benchmark_name': benchmark['name'],
                'results': benchmark_results
            })

        # 生成报告
        self._generate_suite_report(all_results)

        return all_results

    def _generate_suite_report(self, all_results: List[Dict]):
        """生成测试套件报告"""
        report = {
            'suite_name': self.name,
            'timestamp': time.time(),
            'benchmarks': []
        }

        for benchmark_data in all_results:
            benchmark_report = {
                'name': benchmark_data['benchmark_name'],
                'results': []
            }

            for result_data in benchmark_data['results']:
                result = result_data['result']
                benchmark_report['results'].append({
                    'concurrency': result_data['concurrency'],
                    'requests_per_second': result.requests_per_second,
                    'avg_response_time': result.avg_response_time,
                    'p95_response_time': result.p95_response_time,
                    'success_rate': result.success_rate,
                    'max_cpu_usage': result.metadata.get('resource_usage', {}).get('cpu', {}).get('max', 0),
                    'max_memory_usage': result.metadata.get('resource_usage', {}).get('memory', {}).get('max', 0)
                })

            report['benchmarks'].append(benchmark_report)

        # 保存报告
        report_path = f"tests/performance/reports/{self.name}_{int(time.time())}.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📋 Benchmark suite report saved to: {report_path}")

        # 打印摘要
        self._print_suite_summary(report)

    def _print_suite_summary(self, report: Dict):
        """打印测试套件摘要"""
        print(f"\n{'='*80}")
        print(f"📊 Benchmark Suite Summary: {report['suite_name']}")
        print(f"{'='*80}")

        for benchmark in report['benchmarks']:
            print(f"\n🎯 {benchmark['name']}")
            print(f"{'-'*40}")
            print(f"{'Concurrency':<12} {'RPS':<8} {'Avg RT':<8} {'P95 RT':<8} {'Success':<8} {'Max CPU':<8} {'Max Mem':<8}")

            for result in benchmark['results']:
                print(f"{result['concurrency']:<12} "
                      f"{result['requests_per_second']:<8.1f} "
                      f"{result['avg_response_time']:<8.1f} "
                      f"{result['p95_response_time']:<8.1f} "
                      f"{result['success_rate']:<8.1%} "
                      f"{result['max_cpu_usage']:<8.1f} "
                      f"{result['max_memory_usage']:<8.1f}")

        print(f"\n{'='*80}")