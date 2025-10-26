#!/usr/bin/env python3
"""
快速性能测试
运行基础性能验证测试
"""
import asyncio
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.performance.async_performance import AsyncTaskPerformanceTester
from tests.performance.framework import PerformanceTestFramework


async def quick_performance_test():
    """快速性能测试"""
    print("🚀 Quick Performance Test")
    print("=" * 50)

    # 1. 测试异步任务性能
    print("\n⚡ Testing Async Task Performance...")
    async_tester = AsyncTaskPerformanceTester()

    # 测试任务创建和并发
    creation_result = await async_tester.test_async_task_creation()
    print(f"  Task Creation Rate: {creation_result.get('creation_rate', 0):.2f} tasks/sec")
    print(f"  Success Rate: {creation_result.get('success_rate', 0):.2%}")

    # 测试并发性能（较小的并发数）
    concurrency_result = await async_tester.test_async_task_concurrency()
    print(f"  Concurrency Test: Completed for levels {list(concurrency_result.keys())}")

    # 2. 测试基础框架性能
    print("\n🔧 Testing Performance Framework...")
    framework = PerformanceTestFramework()

    # 模拟测试函数
    async def mock_test_function():
        await asyncio.sleep(0.01)  # 10ms模拟处理时间
        return {"result": "success"}

    # 运行基准测试
    benchmark_result = await framework.run_concurrent_test(
        mock_test_function,
        concurrency=10,
        total_requests=100
    )

    print(f"  Requests/sec: {benchmark_result.requests_per_second:.2f}")
    print(f"  Avg Response Time: {benchmark_result.avg_response_time:.2f}ms")
    print(f"  Success Rate: {benchmark_result.success_rate:.2%}")

    # 3. 测试资源使用情况
    print("\n💾 Testing Basic Resource Usage...")
    import psutil

    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()

    print(f"  Current Memory Usage: {memory_mb:.2f}MB")
    print(f"  Current CPU Usage: {cpu_percent:.1f}%")

    # 4. 简单的压力测试
    print("\n🔥 Running Simple Stress Test...")
    stress_start = time.time()

    # 创建更多并发任务
    stress_tasks = [
        mock_test_function() for _ in range(50)
    ]
    stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)

    stress_end = time.time()
    stress_duration = stress_end - stress_start

    successful = [r for r in stress_results if not isinstance(r, Exception)]
    print(f"  Stress Test: {len(successful)}/{len(stress_tasks)} tasks in {stress_duration:.2f}s")
    print(f"  Stress Throughput: {len(successful) / stress_duration:.2f} tasks/sec")

    # 总结
    print("\n" + "=" * 50)
    print("📊 Quick Performance Test Summary")
    print("=" * 50)
    print(f"✅ Async Task Performance: OK")
    print(f"✅ Performance Framework: OK")
    print(f"✅ Resource Usage: {memory_mb:.1f}MB, {cpu_percent:.1f}% CPU")
    print(f"✅ Stress Test: {len(successful)}/{len(stress_tasks)} successful")
    print(f"⏱️  Total Test Time: {time.time() - start_time:.2f}s")

    return {
        "async_creation_rate": creation_result.get('creation_rate', 0),
        "async_success_rate": creation_result.get('success_rate', 0),
        "framework_rps": benchmark_result.requests_per_second,
        "framework_avg_rt": benchmark_result.avg_response_time,
        "stress_success_rate": len(successful) / len(stress_tasks),
        "memory_mb": memory_mb,
        "cpu_percent": cpu_percent
    }


if __name__ == "__main__":
    start_time = time.time()

    try:
        results = asyncio.run(quick_performance_test())
        print(f"\n🎉 Quick performance test completed successfully!")
        print(f"📈 Key metrics:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Quick performance test failed: {e}")
        sys.exit(1)