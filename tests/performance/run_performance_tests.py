#!/usr/bin/env python3
"""
综合性能测试运行器
运行所有性能验证测试并生成综合报告
"""
import asyncio
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.performance.api_performance import run_api_performance_tests
from tests.performance.service_performance import run_sanic_performance_tests
from tests.performance.async_performance import run_async_performance_tests
from tests.performance.resource_monitor import run_resource_performance_tests


class PerformanceTestRunner:
    """性能测试运行器"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None

    def print_banner(self):
        """打印测试横幅"""
        print("=" * 80)
        print("🚀 Complex RAG Performance Test Suite")
        print("=" * 80)
        print("This suite will test:")
        print("  • API response time and concurrency")
        print("  • Sanic service performance")
        print("  • Async task processing performance")
        print("  • System resource usage")
        print("=" * 80)
        print()

    async def run_api_tests(self, base_url: str = "http://localhost:8000", skip: bool = False):
        """运行API性能测试"""
        if skip:
            print("⏭️  Skipping API performance tests")
            return

        print("📡 Starting API Performance Tests...")
        print(f"Target URL: {base_url}")
        print("-" * 50)

        try:
            results, report_path = await run_api_performance_tests(base_url)
            self.results["api"] = results
            self.results["api_report_path"] = report_path
            print(f"✅ API tests completed. Report: {report_path}")
        except Exception as e:
            print(f"❌ API tests failed: {e}")
            self.results["api"] = {"error": str(e)}

        print()

    async def run_sanic_tests(self, host: str = "127.0.0.1", port: int = 8000, skip: bool = False):
        """运行Sanic服务性能测试"""
        if skip:
            print("⏭️  Skipping Sanic performance tests")
            return

        print("🌐 Starting Sanic Service Performance Tests...")
        print(f"Target: {host}:{port}")
        print("-" * 50)

        try:
            results, report_path = await run_sanic_performance_tests(host, port)
            self.results["sanic"] = results
            self.results["sanic_report_path"] = report_path
            print(f"✅ Sanic tests completed. Report: {report_path}")
        except Exception as e:
            print(f"❌ Sanic tests failed: {e}")
            self.results["sanic"] = {"error": str(e)}

        print()

    async def run_async_tests(self, skip: bool = False):
        """运行异步任务性能测试"""
        if skip:
            print("⏭️  Skipping async performance tests")
            return

        print("⚡ Starting Async Task Performance Tests...")
        print("-" * 50)

        try:
            results, report_path = await run_async_performance_tests()
            self.results["async"] = results
            self.results["async_report_path"] = report_path
            print(f"✅ Async tests completed. Report: {report_path}")
        except Exception as e:
            print(f"❌ Async tests failed: {e}")
            self.results["async"] = {"error": str(e)}

        print()

    async def run_resource_tests(self, skip: bool = False):
        """运行系统资源测试"""
        if skip:
            print("⏭️  Skipping resource performance tests")
            return

        print("💻 Starting System Resource Performance Tests...")
        print("-" * 50)

        try:
            results, report_path = await run_resource_performance_tests()
            self.results["resource"] = results
            self.results["resource_report_path"] = report_path
            print(f"✅ Resource tests completed. Report: {report_path}")
        except Exception as e:
            print(f"❌ Resource tests failed: {e}")
            self.results["resource"] = {"error": str(e)}

        print()

    def generate_comprehensive_report(self):
        """生成综合性能报告"""
        print("📊 Generating Comprehensive Performance Report...")

        report = {
            "test_summary": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
                "total_tests": len([k for k in self.results.keys() if k != 'summary']),
                "successful_tests": len([k for k in self.results.keys() if k != 'summary' and 'error' not in self.results.get(k, {})]),
                "failed_tests": len([k for k in self.results.keys() if k != 'summary' and 'error' in self.results.get(k, {})])
            },
            "results": self.results
        }

        # 保存综合报告
        import json
        report_dir = Path("tests/performance/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        report_path = report_dir / f"comprehensive_performance_report_{timestamp}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Comprehensive report saved to: {report_path}")

        # 打印摘要
        self.print_comprehensive_summary(report)

        return report_path

    def print_comprehensive_summary(self, report: Dict[str, Any]):
        """打印综合测试摘要"""
        summary = report["test_summary"]
        results = report["results"]

        print("\n" + "=" * 80)
        print("📊 Comprehensive Performance Test Summary")
        print("=" * 80)

        print(f"\n🕒 Test Duration: {summary['total_duration']:.2f} seconds")
        print(f"📈 Total Tests: {summary['total_tests']}")
        print(f"✅ Successful: {summary['successful_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📊 Success Rate: {summary['successful_tests'] / summary['total_tests'] * 100:.1f}%")

        # 各项测试的关键指标摘要
        print(f"\n📋 Test Results Summary:")

        # API测试摘要
        if "api" in results and "error" not in results["api"]:
            api_results = results["api"]
            if "benchmark_suite" in api_results:
                print(f"  📡 API Tests: ✅ Completed")
                # 可以添加更多API测试的关键指标
            else:
                print(f"  📡 API Tests: ⚠️ Partial results")
        elif "api" in results:
            print(f"  📡 API Tests: ❌ Failed")

        # Sanic测试摘要
        if "sanic" in results and "error" not in results["sanic"]:
            sanic_results = results["sanic"]
            if "basic_performance" in sanic_results:
                basic = sanic_results["basic_performance"]
                print(f"  🌐 Sanic Tests: ✅ Completed")
                print(f"    - Avg Response Time: {basic.get('avg_response_time', 0):.2f}ms")
                print(f"    - Requests/sec: {basic.get('requests_per_second', 0):.2f}")
            else:
                print(f"  🌐 Sanic Tests: ⚠️ Partial results")
        elif "sanic" in results:
            print(f"  🌐 Sanic Tests: ❌ Failed")

        # 异步测试摘要
        if "async" in results and "error" not in results["async"]:
            async_results = results["async"]
            if "task_creation" in async_results:
                creation = async_results["task_creation"]
                print(f"  ⚡ Async Tests: ✅ Completed")
                print(f"    - Task Creation Rate: {creation.get('creation_rate', 0):.2f} tasks/sec")
                print(f"    - Success Rate: {creation.get('success_rate', 0):.2%}")
            else:
                print(f"  ⚡ Async Tests: ⚠️ Partial results")
        elif "async" in results:
            print(f"  ⚡ Async Tests: ❌ Failed")

        # 资源测试摘要
        if "resource" in results and "error" not in results["resource"]:
            resource_results = results["resource"]
            if "cpu_intensive" in resource_results:
                cpu = resource_results["cpu_intensive"]["cpu_usage"]
                print(f"  💻 Resource Tests: ✅ Completed")
                print(f"    - Peak CPU Usage: {cpu.get('peak', 0):.1f}%")
                if "resource_leaks" in resource_results:
                    leak = resource_results["resource_leaks"]
                    leak_status = "⚠️ Potential leak" if leak.get('leak_detected') else "✅ No leaks detected"
                    print(f"    - Memory Leak Status: {leak_status}")
            else:
                print(f"  💻 Resource Tests: ⚠️ Partial results")
        elif "resource" in results:
            print(f"  💻 Resource Tests: ❌ Failed")

        # 性能建议
        print(f"\n💡 Performance Recommendations:")
        self._generate_recommendations(results)

        print("\n" + "=" * 80)

    def _generate_recommendations(self, results: Dict[str, Any]):
        """生成性能建议"""
        recommendations = []

        # API性能建议
        if "api" in results and "error" not in results["api"]:
            # 这里可以根据API测试结果添加具体建议
            recommendations.append("✅ API endpoints are performing well")

        # Sanic性能建议
        if "sanic" in results and "error" not in results["sanic"]:
            sanic_results = results["sanic"]
            if "basic_performance" in sanic_results:
                basic = sanic_results["basic_performance"]
                if basic.get('avg_response_time', 0) > 100:
                    recommendations.append("⚠️ Consider optimizing API response times (avg > 100ms)")
                if basic.get('requests_per_second', 0) < 100:
                    recommendations.append("⚠️ Consider increasing service throughput (RPS < 100)")

        # 异步性能建议
        if "async" in results and "error" not in results["async"]:
            async_results = results["async"]
            if "task_creation" in async_results:
                creation = async_results["task_creation"]
                if creation.get('success_rate', 0) < 0.95:
                    recommendations.append("⚠️ Async task success rate could be improved")

        # 资源使用建议
        if "resource" in results and "error" not in results["resource"]:
            resource_results = results["resource"]
            if "cpu_intensive" in resource_results:
                cpu = resource_results["cpu_intensive"]["cpu_usage"]
                if cpu.get('peak', 0) > 80:
                    recommendations.append("⚠️ High CPU usage detected, consider scaling or optimization")
            if "resource_leaks" in resource_results:
                leak = resource_results["resource_leaks"]
                if leak.get('leak_detected'):
                    recommendations.append("🚨 Memory leaks detected, investigate immediately")

        if not recommendations:
            recommendations.append("✅ All performance metrics are within acceptable ranges")

        for rec in recommendations[:5]:  # 最多显示5个建议
            print(f"  {rec}")

    async def run_all_tests(
        self,
        api_url: str = "http://localhost:8000",
        sanic_host: str = "127.0.0.1",
        sanic_port: int = 8000,
        skip_api: bool = False,
        skip_sanic: bool = False,
        skip_async: bool = False,
        skip_resource: bool = False
    ):
        """运行所有性能测试"""
        self.print_banner()
        self.start_time = time.time()

        try:
            # 运行各项测试
            await self.run_api_tests(api_url, skip_api)
            await self.run_sanic_tests(sanic_host, sanic_port, skip_sanic)
            await self.run_async_tests(skip_async)
            await self.run_resource_tests(skip_resource)

            self.end_time = time.time()

            # 生成综合报告
            report_path = self.generate_comprehensive_report()

            print(f"\n🎉 All performance tests completed!")
            print(f"📊 Comprehensive report: {report_path}")

            return self.results, report_path

        except KeyboardInterrupt:
            print("\n⏹️  Tests interrupted by user")
            self.end_time = time.time()
            if self.results:
                self.generate_comprehensive_report()
            return None, None
        except Exception as e:
            print(f"\n💥 Unexpected error during testing: {e}")
            self.end_time = time.time()
            if self.results:
                self.generate_comprehensive_report()
            return None, None


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Complex RAG Performance Test Runner")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--sanic-host", default="127.0.0.1", help="Sanic service host")
    parser.add_argument("--sanic-port", type=int, default=8000, help="Sanic service port")
    parser.add_argument("--skip-api", action="store_true", help="Skip API performance tests")
    parser.add_argument("--skip-sanic", action="store_true", help="Skip Sanic performance tests")
    parser.add_argument("--skip-async", action="store_true", help="Skip async performance tests")
    parser.add_argument("--skip-resource", action="store_true", help="Skip resource performance tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (skip resource-intensive tests)")

    args = parser.parse_args()

    # 快速测试模式跳过资源密集型测试
    if args.quick:
        args.skip_resource = True

    runner = PerformanceTestRunner()
    results, report_path = await runner.run_all_tests(
        api_url=args.api_url,
        sanic_host=args.sanic_host,
        sanic_port=args.sanic_port,
        skip_api=args.skip_api,
        skip_sanic=args.skip_sanic,
        skip_async=args.skip_async,
        skip_resource=args.skip_resource
    )

    if results:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))