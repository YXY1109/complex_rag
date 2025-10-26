"""
系统资源监控测试
验证系统资源使用情况
"""
import asyncio
import time
import psutil
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path
from .framework import PerformanceTestFramework


@dataclass
class ResourceSnapshot:
    """资源快照"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    process_memory_mb: float
    process_cpu_percent: float
    process_threads: int
    process_open_files: int


class ResourceMonitor:
    """系统资源监控器"""

    def __init__(self):
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_task = None

    async def start_monitoring(self, interval: float = 1.0, duration: float = None):
        """开始资源监控"""
        self.monitoring = True
        self.snapshots = []

        async def monitor_loop():
            start_time = time.time()

            while self.monitoring:
                # 检查是否达到指定持续时间
                if duration and (time.time() - start_time) >= duration:
                    break

                snapshot = await self._collect_snapshot()
                self.snapshots.append(snapshot)

                await asyncio.sleep(interval)

        self.monitor_task = asyncio.create_task(monitor_loop())

    async def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring = False
        if self.monitor_task:
            await self.monitor_task

    async def _collect_snapshot(self) -> ResourceSnapshot:
        """收集资源快照"""
        # 系统级资源
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        # 进程级资源
        process = psutil.Process()

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            disk_usage_percent=disk.percent,
            disk_used_gb=disk.used / 1024 / 1024 / 1024,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_count=len(psutil.pids()),
            process_memory_mb=process.memory_info().rss / 1024 / 1024,
            process_cpu_percent=process.cpu_percent(),
            process_threads=process.num_threads(),
            process_open_files=len(process.open_files())
        )

    def get_summary(self) -> Dict[str, Any]:
        """获取资源使用摘要"""
        if not self.snapshots:
            return {}

        import statistics

        # 提取各项指标
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_percent for s in self.snapshots]
        disk_values = [s.disk_usage_percent for s in self.snapshots]
        process_memory_values = [s.process_memory_mb for s in self.snapshots]
        process_cpu_values = [s.process_cpu_percent for s in self.snapshots]
        thread_values = [s.process_threads for s in self.snapshots]

        def calc_stats(values):
            if not values:
                return {}
            return {
                "avg": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "median": statistics.median(values),
                "p95": values[int(len(values) * 0.95)] if len(values) > 20 else max(values)
            }

        return {
            "monitoring_duration": self.snapshots[-1].timestamp - self.snapshots[0].timestamp if len(self.snapshots) > 1 else 0,
            "sample_count": len(self.snapshots),
            "cpu": calc_stats(cpu_values),
            "memory": calc_stats(memory_values),
            "disk": calc_stats(disk_values),
            "process_memory": calc_stats(process_memory_values),
            "process_cpu": calc_stats(process_cpu_values),
            "threads": calc_stats(thread_values),
            "peak_resource_usage": {
                "cpu_peak": max(cpu_values) if cpu_values else 0,
                "memory_peak": max(memory_values) if memory_values else 0,
                "process_memory_peak": max(process_memory_values) if process_memory_values else 0,
                "threads_peak": max(thread_values) if thread_values else 0
            }
        }


class SystemResourceTester:
    """系统资源测试器"""

    def __init__(self):
        self.monitor = ResourceMonitor()
        self.framework = PerformanceTestFramework()

    async def test_cpu_intensive_load(self) -> Dict[str, Any]:
        """测试CPU密集型负载下的资源使用"""
        print("🔥 Testing CPU-intensive load...")

        # 开始监控
        await self.monitor.start_monitoring(interval=0.5, duration=30)

        # 创建CPU密集型任务
        async def cpu_task():
            result = 0
            start_time = time.time()
            while time.time() - start_time < 25:  # 运行25秒
                result += sum(i * i for i in range(10000))
            return result

        # 启动多个CPU密集型任务
        cpu_tasks = [cpu_task() for _ in range(4)]  # 4个并发任务
        await asyncio.gather(*cpu_tasks)

        # 停止监控
        await self.monitor.stop_monitoring()

        # 分析结果
        summary = self.monitor.get_summary()
        cpu_usage = summary.get("cpu", {})

        return {
            "test_type": "cpu_intensive",
            "duration": 30,
            "concurrent_tasks": 4,
            "cpu_usage": {
                "average": cpu_usage.get("avg", 0),
                "peak": cpu_usage.get("max", 0),
                "std_dev": cpu_usage.get("std", 0)
            },
            "resource_summary": summary
        }

    async def test_memory_intensive_load(self) -> Dict[str, Any]:
        """测试内存密集型负载下的资源使用"""
        print("💾 Testing memory-intensive load...")

        # 开始监控
        await self.monitor.start_monitoring(interval=0.5, duration=30)

        # 创建内存密集型任务
        memory_blocks = []

        async def memory_task():
            # 分配大量内存
            for i in range(100):
                # 分配1MB内存块
                block = bytearray(1024 * 1024)
                memory_blocks.append(block)
                await asyncio.sleep(0.1)  # 短暂延迟

            # 保持内存占用一段时间
            await asyncio.sleep(10)

            # 释放内存
            memory_blocks.clear()

        # 运行内存任务
        await memory_task()

        # 停止监控
        await self.monitor.stop_monitoring()

        # 分析结果
        summary = self.monitor.get_summary()
        memory_usage = summary.get("memory", {})
        process_memory = summary.get("process_memory", {})

        return {
            "test_type": "memory_intensive",
            "duration": 30,
            "memory_blocks_allocated": 100,
            "memory_usage": {
                "system_average": memory_usage.get("avg", 0),
                "system_peak": memory_usage.get("max", 0),
                "process_average": process_memory.get("avg", 0),
                "process_peak": process_memory.get("max", 0),
                "process_growth": process_memory.get("max", 0) - process_memory.get("min", 0)
            },
            "resource_summary": summary
        }

    async def test_io_intensive_load(self) -> Dict[str, Any]:
        """测试IO密集型负载下的资源使用"""
        print("💽 Testing I/O-intensive load...")

        # 开始监控
        await self.monitor.start_monitoring(interval=0.5, duration=30)

        # 创建IO密集型任务
        async def io_task():
            # 模拟文件IO操作
            for i in range(50):
                # 写入临时文件
                temp_file = f"temp_test_{i}.txt"
                try:
                    with open(temp_file, 'w') as f:
                        f.write("Test data " * 1000)

                    # 读取文件
                    with open(temp_file, 'r') as f:
                        content = f.read()

                    # 删除文件
                    import os
                    os.remove(temp_file)

                except Exception:
                    pass  # 忽略文件操作错误

                await asyncio.sleep(0.1)  # 模拟IO等待

        # 启动多个IO任务
        io_tasks = [io_task() for _ in range(5)]
        await asyncio.gather(*io_tasks)

        # 停止监控
        await self.monitor.stop_monitoring()

        # 分析结果
        summary = self.monitor.get_summary()

        return {
            "test_type": "io_intensive",
            "duration": 30,
            "concurrent_tasks": 5,
            "io_operations": 250,  # 5 tasks * 50 operations each
            "resource_summary": summary
        }

    async def test_mixed_workload(self) -> Dict[str, Any]:
        """测试混合工作负载下的资源使用"""
        print("🔀 Testing mixed workload...")

        # 开始监控
        await self.monitor.start_monitoring(interval=0.5, duration=30)

        # 创建混合任务
        async def mixed_task(task_id: int):
            # CPU任务
            for _ in range(1000):
                _ = sum(i * i for i in range(100))

            # IO任务
            await asyncio.sleep(0.1)

            # 内存任务
            _ = [0] * 10000

            await asyncio.sleep(0.05)

        # 启动混合任务
        mixed_tasks = [mixed_task(i) for i in range(10)]
        await asyncio.gather(*mixed_tasks)

        # 停止监控
        await self.monitor.stop_monitoring()

        # 分析结果
        summary = self.monitor.get_summary()

        return {
            "test_type": "mixed_workload",
            "duration": 30,
            "concurrent_tasks": 10,
            "resource_summary": summary
        }

    async def test_resource_leaks(self) -> Dict[str, Any]:
        """测试资源泄漏"""
        print("🔍 Testing for resource leaks...")

        leak_test_results = []

        # 进行多轮测试，观察资源是否持续增长
        for round_num in range(5):
            print(f"  Round {round_num + 1}/5")

            # 开始监控
            await self.monitor.start_monitoring(interval=0.2, duration=10)

            # 创建和销毁大量任务
            tasks = []
            for i in range(100):
                task = asyncio.create_task(
                    self._leak_test_task(i)
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            # 停止监控
            await self.monitor.stop_monitoring()

            # 记录每轮的资源使用情况
            summary = self.monitor.get_summary()
            process_memory = summary.get("process_memory", {})

            leak_test_results.append({
                "round": round_num + 1,
                "peak_memory_mb": process_memory.get("max", 0),
                "avg_memory_mb": process_memory.get("avg", 0),
                "final_memory_mb": self.monitor.snapshots[-1].process_memory_mb if self.monitor.snapshots else 0
            })

            # 清理
            await asyncio.sleep(2)

        # 分析泄漏情况
        memory_peaks = [r["peak_memory_mb"] for r in leak_test_results]
        memory_finals = [r["final_memory_mb"] for r in leak_test_results]

        memory_growth = memory_finals[-1] - memory_finals[0] if len(memory_finals) > 1 else 0
        peak_growth = memory_peaks[-1] - memory_peaks[0] if len(memory_peaks) > 1 else 0

        return {
            "test_type": "resource_leaks",
            "rounds": 5,
            "tasks_per_round": 100,
            "memory_growth_mb": memory_growth,
            "peak_growth_mb": peak_growth,
            "leak_detected": memory_growth > 50 or peak_growth > 100,  # 50MB增长认为可能有泄漏
            "round_results": leak_test_results
        }

    async def _leak_test_task(self, task_id: int):
        """泄漏测试任务"""
        # 创建一些临时对象
        temp_objects = []
        for i in range(100):
            temp_objects.append({
                "id": task_id * 1000 + i,
                "data": "x" * 100,
                "nested": {"value": i, "list": [j for j in range(10)]}
            })

        # 模拟一些计算
        result = sum(i * i for i in range(1000))

        # 短暂延迟
        await asyncio.sleep(0.01)

        # 临时对象应该在这里被垃圾回收
        return result

    async def test_stress_limits(self) -> Dict[str, Any]:
        """测试系统压力极限"""
        print("⚡ Testing system stress limits...")

        # 逐步增加负载直到系统达到极限
        stress_results = []
        max_concurrency = 1

        for concurrency in [10, 50, 100, 200, 500, 1000]:
            print(f"  Testing with {concurrency} concurrent tasks...")

            try:
                # 开始监控
                await self.monitor.start_monitoring(interval=0.5, duration=15)

                # 创建压力测试任务
                async def stress_task():
                    # 混合CPU、内存、IO操作
                    for _ in range(10):
                        # CPU操作
                        _ = sum(i * i for i in range(1000))

                        # 内存操作
                        _ = [0] * 1000

                        # 短暂延迟
                        await asyncio.sleep(0.01)

                # 启动压力测试
                start_time = time.time()
                stress_tasks = [stress_task() for _ in range(concurrency)]

                try:
                    await asyncio.wait_for(
                        asyncio.gather(*stress_tasks),
                        timeout=20  # 20秒超时
                    )
                    end_time = time.time()

                    # 停止监控
                    await self.monitor.stop_monitoring()

                    # 分析结果
                    summary = self.monitor.get_summary()
                    cpu_usage = summary.get("cpu", {})
                    memory_usage = summary.get("memory", {})

                    stress_results.append({
                        "concurrency": concurrency,
                        "success": True,
                        "duration": end_time - start_time,
                        "cpu_peak": cpu_usage.get("max", 0),
                        "memory_peak": memory_usage.get("max", 0),
                        "timeout": False
                    })

                    max_concurrency = concurrency
                    print(f"    ✅ Success - CPU: {cpu_usage.get('max', 0):.1f}%, Memory: {memory_usage.get('max', 0):.1f}%")

                except asyncio.TimeoutError:
                    await self.monitor.stop_monitoring()
                    stress_results.append({
                        "concurrency": concurrency,
                        "success": False,
                        "timeout": True,
                        "reason": "Task timeout"
                    })
                    print(f"    ⏰ Timeout at concurrency {concurrency}")
                    break

            except Exception as e:
                await self.monitor.stop_monitoring()
                stress_results.append({
                    "concurrency": concurrency,
                    "success": False,
                    "error": str(e)
                })
                print(f"    ❌ Error at concurrency {concurrency}: {e}")
                break

        return {
            "test_type": "stress_limits",
            "max_successful_concurrency": max_concurrency,
            "stress_results": stress_results
        }

    async def run_resource_test_suite(self) -> Dict[str, Any]:
        """运行资源测试套件"""
        print("🎯 Starting System Resource Performance Tests")

        # 1. CPU密集型测试
        cpu_result = await self.test_cpu_intensive_load()

        # 2. 内存密集型测试
        memory_result = await self.test_memory_intensive_load()

        # 3. IO密集型测试
        io_result = await self.test_io_intensive_load()

        # 4. 混合工作负载测试
        mixed_result = await self.test_mixed_workload()

        # 5. 资源泄漏测试
        leak_result = await self.test_resource_leaks()

        # 6. 压力极限测试
        stress_result = await self.test_stress_limits()

        return {
            "cpu_intensive": cpu_result,
            "memory_intensive": memory_result,
            "io_intensive": io_result,
            "mixed_workload": mixed_result,
            "resource_leaks": leak_result,
            "stress_limits": stress_result
        }

    def generate_resource_report(self, results: Dict[str, Any]):
        """生成资源测试报告"""
        print("\n📊 Generating Resource Performance Report...")

        report = {
            "test_summary": {
                "timestamp": time.time(),
                "total_tests": len(results),
                "system_info": self._get_system_info()
            },
            "results": results
        }

        # 保存报告
        report_path = f"tests/performance/reports/resource_performance_{int(time.time())}.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Resource performance report saved to: {report_path}")

        # 打印摘要
        self._print_resource_summary(results)

        return report_path

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_total_gb": psutil.disk_usage('/').total / 1024 / 1024 / 1024,
            "platform": psutil.platform.platform(),
            "python_version": psutil.sys.version
        }

    def _print_resource_summary(self, results: Dict):
        """打印资源测试摘要"""
        print(f"\n{'='*80}")
        print("📊 System Resource Performance Summary")
        print(f"{'='*80}")

        # CPU测试摘要
        if "cpu_intensive" in results:
            cpu = results["cpu_intensive"]["cpu_usage"]
            print(f"\n🔥 CPU-Intensive Load:")
            print(f"  Average CPU Usage: {cpu.get('average', 0):.1f}%")
            print(f"  Peak CPU Usage: {cpu.get('peak', 0):.1f}%")
            print(f"  CPU Usage Std Dev: {cpu.get('std_dev', 0):.1f}%")

        # 内存测试摘要
        if "memory_intensive" in results:
            memory = results["memory_intensive"]["memory_usage"]
            print(f"\n💾 Memory-Intensive Load:")
            print(f"  Peak System Memory: {memory.get('system_peak', 0):.1f}%")
            print(f"  Peak Process Memory: {memory.get('process_peak', 0):.1f}MB")
            print(f"  Process Memory Growth: {memory.get('process_growth', 0):.1f}MB")

        # 资源泄漏测试摘要
        if "resource_leaks" in results:
            leak = results["resource_leaks"]
            print(f"\n🔍 Resource Leak Detection:")
            print(f"  Memory Growth: {leak.get('memory_growth_mb', 0):.1f}MB")
            print(f"  Peak Growth: {leak.get('peak_growth_mb', 0):.1f}MB")
            print(f"  Leak Detected: {'⚠️ Yes' if leak.get('leak_detected') else '✅ No'}")

        # 压力测试摘要
        if "stress_limits" in results:
            stress = results["stress_limits"]
            print(f"\n⚡ Stress Test Results:")
            print(f"  Max Successful Concurrency: {stress.get('max_successful_concurrency', 0)}")

            successful_tests = [r for r in stress.get('stress_results', []) if r.get('success')]
            if successful_tests:
                max_cpu = max(r.get('cpu_peak', 0) for r in successful_tests)
                max_memory = max(r.get('memory_peak', 0) for r in successful_tests)
                print(f"  Peak CPU Under Stress: {max_cpu:.1f}%")
                print(f"  Peak Memory Under Stress: {max_memory:.1f}%")

        print(f"\n{'='*80}")


# 便捷函数
async def run_resource_performance_tests():
    """运行资源性能测试的主函数"""
    tester = SystemResourceTester()

    print("🎯 Starting Comprehensive Resource Performance Tests")

    # 运行完整的资源测试套件
    results = await tester.run_resource_test_suite()

    # 生成报告
    report_path = tester.generate_resource_report(results)

    return results, report_path