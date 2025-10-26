"""
API性能测试
测试API响应时间和并发能力
"""
import asyncio
import aiohttp
import time
from typing import Dict, Any, List
from .framework import PerformanceTestFramework, BenchmarkSuite


class APIPerformanceTester:
    """API性能测试器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.framework = PerformanceTestFramework()

    async def test_health_endpoint(self) -> Dict[str, Any]:
        """测试健康检查端点"""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.get(f"{self.base_url}/api/health") as response:
                    data = await response.json()
                    end_time = time.time()
                    return {
                        "success": response.status == 200,
                        "response_time": (end_time - start_time) * 1000,
                        "status_code": response.status,
                        "data": data
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "response_time": (end_time - start_time) * 1000,
                    "error": str(e)
                }

    async def test_chat_completion(self, **kwargs) -> Dict[str, Any]:
        """测试聊天完成端点"""
        default_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }
        payload = {**default_payload, **kwargs}

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/api/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.content_type == "application/json":
                        data = await response.json()
                    else:
                        data = await response.text()

                    end_time = time.time()
                    return {
                        "success": response.status == 200,
                        "response_time": (end_time - start_time) * 1000,
                        "status_code": response.status,
                        "data": data,
                        "payload_size": len(str(payload))
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "response_time": (end_time - start_time) * 1000,
                    "error": str(e),
                    "payload_size": len(str(payload))
                }

    async def test_document_search(self, query: str = "test", **kwargs) -> Dict[str, Any]:
        """测试文档搜索端点"""
        params = {"query": query, "limit": 10}
        params.update(kwargs)

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.get(
                    f"{self.base_url}/api/documents/search",
                    params=params
                ) as response:
                    if response.content_type == "application/json":
                        data = await response.json()
                    else:
                        data = await response.text()

                    end_time = time.time()
                    return {
                        "success": response.status == 200,
                        "response_time": (end_time - start_time) * 1000,
                        "status_code": response.status,
                        "data": data,
                        "query": query
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "response_time": (end_time - start_time) * 1000,
                    "error": str(e),
                    "query": query
                }

    async def test_knowledge_base_search(self, kb_id: str = "test-kb", query: str = "test", **kwargs) -> Dict[str, Any]:
        """测试知识库搜索端点"""
        payload = {
            "query": query,
            "top_k": 5,
            "filters": {}
        }
        payload.update(kwargs)

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/api/knowledge/bases/{kb_id}/search",
                    json=payload
                ) as response:
                    if response.content_type == "application/json":
                        data = await response.json()
                    else:
                        data = await response.text()

                    end_time = time.time()
                    return {
                        "success": response.status == 200,
                        "response_time": (end_time - start_time) * 1000,
                        "status_code": response.status,
                        "data": data,
                        "kb_id": kb_id,
                        "query": query
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "response_time": (end_time - start_time) * 1000,
                    "error": str(e),
                    "kb_id": kb_id,
                    "query": query
                }

    async def run_api_benchmark_suite(self) -> Dict[str, Any]:
        """运行API基准测试套件"""
        print("🚀 Starting API Performance Benchmark Suite")
        print(f"Target URL: {self.base_url}")

        suite = BenchmarkSuite("API_Performance_Tests")

        # 添加各种基准测试
        suite.add_benchmark(self.test_health_endpoint, "Health_Check")
        suite.add_benchmark(self.test_chat_completion, "Chat_Completion")
        suite.add_benchmark(
            lambda: self.test_document_search("machine learning"),
            "Document_Search_ML"
        )
        suite.add_benchmark(
            lambda: self.test_knowledge_base_search("test-kb", "what is AI"),
            "Knowledge_Base_Search"
        )

        # 运行测试
        results = await suite.run_all_benchmarks(
            concurrency_levels=[1, 5, 10, 25, 50]
        )

        return results

    async def test_load_patterns(self):
        """测试不同的负载模式"""
        print("\n🎯 Testing Different Load Patterns")

        # 突发负载测试
        print("\n📈 Testing Burst Load...")
        burst_result = await self.framework.run_concurrent_test(
            self.test_health_endpoint,
            concurrency=100,
            total_requests=500
        )

        # 持续负载测试
        print("\n📊 Testing Sustained Load...")
        sustained_result = await self.framework.run_concurrent_test(
            self.test_health_endpoint,
            concurrency=20,
            total_requests=1000
        )

        # 渐进负载测试
        print("\n⬆️ Testing Ramp-up Load...")
        rampup_results = []
        for concurrency in [5, 10, 20, 40, 80]:
            result = await self.framework.run_concurrent_test(
                self.test_health_endpoint,
                concurrency=concurrency,
                total_requests=concurrency * 10
            )
            rampup_results.append({
                'concurrency': concurrency,
                'result': result
            })

        return {
            'burst_load': burst_result,
            'sustained_load': sustained_result,
            'rampup_load': rampup_results
        }

    async def test_api_limits(self):
        """测试API限制和边界条件"""
        print("\n🚧 Testing API Limits and Boundaries")

        # 测试大请求负载
        print("\n📝 Testing Large Payload...")
        large_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "A" * 10000}  # 10KB content
            ],
            "max_tokens": 100
        }

        large_payload_result = await self.framework.run_concurrent_test(
            lambda: self.test_chat_completion(**large_payload),
            concurrency=5,
            total_requests=50
        )

        # 测试快速连续请求
        print("\n⚡ Testing Rapid Fire Requests...")
        rapid_result = await self.framework.run_concurrent_test(
            self.test_health_endpoint,
            concurrency=1,
            total_requests=1000
        )

        # 测试并发限制
        print("\n🔒 Testing Concurrency Limits...")
        concurrency_result = await self.framework.run_concurrent_test(
            self.test_health_endpoint,
            concurrency=200,
            total_requests=1000
        )

        return {
            'large_payload': large_payload_result,
            'rapid_fire': rapid_result,
            'concurrency_limit': concurrency_result
        }

    async def test_error_handling(self):
        """测试错误处理性能"""
        print("\n❌ Testing Error Handling Performance")

        # 测试无效端点
        async def test_invalid_endpoint():
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}/api/invalid-endpoint") as response:
                        end_time = time.time()
                        return {
                            "success": False,
                            "response_time": (end_time - start_time) * 1000,
                            "status_code": response.status
                        }
                except Exception as e:
                    end_time = time.time()
                    return {
                        "success": False,
                        "response_time": (end_time - start_time) * 1000,
                        "error": str(e)
                    }

        # 测试无效负载
        async def test_invalid_payload():
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/api/chat/completions",
                        json={"invalid": "payload"},
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        end_time = time.time()
                        return {
                            "success": False,
                            "response_time": (end_time - start_time) * 1000,
                            "status_code": response.status
                        }
                except Exception as e:
                    end_time = time.time()
                    return {
                        "success": False,
                        "response_time": (end_time - start_time) * 1000,
                        "error": str(e)
                    }

        # 运行错误处理测试
        invalid_endpoint_result = await self.framework.run_concurrent_test(
            test_invalid_endpoint,
            concurrency=20,
            total_requests=100
        )

        invalid_payload_result = await self.framework.run_concurrent_test(
            test_invalid_payload,
            concurrency=20,
            total_requests=100
        )

        return {
            'invalid_endpoint': invalid_endpoint_result,
            'invalid_payload': invalid_payload_result
        }

    def generate_performance_report(self, all_results: Dict[str, Any]):
        """生成性能报告"""
        print("\n📊 Generating Performance Report...")

        report = {
            "test_summary": {
                "total_tests": len(all_results),
                "timestamp": time.time(),
                "base_url": self.base_url
            },
            "results": {}
        }

        # 处理各种测试结果
        for test_name, result in all_results.items():
            if hasattr(result, 'test_name'):  # BenchmarkResult
                report["results"][test_name] = {
                    "avg_response_time": result.avg_response_time,
                    "p95_response_time": result.p95_response_time,
                    "requests_per_second": result.requests_per_second,
                    "success_rate": result.success_rate,
                    "max_cpu": result.metadata.get('resource_usage', {}).get('cpu', {}).get('max', 0),
                    "max_memory": result.metadata.get('resource_usage', {}).get('memory', {}).get('max', 0)
                }
            elif isinstance(result, dict):  # 复合结果
                report["results"][test_name] = {}
                for sub_test, sub_result in result.items():
                    if hasattr(sub_result, 'test_name'):
                        report["results"][test_name][sub_test] = {
                            "avg_response_time": sub_result.avg_response_time,
                            "requests_per_second": sub_result.requests_per_second,
                            "success_rate": sub_result.success_rate
                        }

        # 保存报告
        import json
        from pathlib import Path

        report_path = f"tests/performance/reports/api_performance_{int(time.time())}.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Performance report saved to: {report_path}")

        # 打印摘要
        self._print_performance_summary(report)

        return report_path

    def _print_performance_summary(self, report: Dict):
        """打印性能摘要"""
        print(f"\n{'='*80}")
        print("📊 API Performance Summary")
        print(f"{'='*80}")

        results = report.get("results", {})
        if not results:
            print("No results to display")
            return

        print(f"{'Test Name':<30} {'RPS':<8} {'Avg RT':<8} {'P95 RT':<8} {'Success':<8} {'Max CPU':<8} {'Max Mem':<8}")
        print(f"{'-'*80}")

        for test_name, metrics in results.items():
            if "avg_response_time" in metrics:
                print(f"{test_name[:30]:<30} "
                      f"{metrics.get('requests_per_second', 0):<8.1f} "
                      f"{metrics.get('avg_response_time', 0):<8.1f} "
                      f"{metrics.get('p95_response_time', 0):<8.1f} "
                      f"{metrics.get('success_rate', 0):<8.1%} "
                      f"{metrics.get('max_cpu', 0):<8.1f} "
                      f"{metrics.get('max_memory', 0):<8.1f}")

        print(f"\n{'='*80}")


# 便捷函数
async def run_api_performance_tests(base_url: str = "http://localhost:8000"):
    """运行API性能测试的主函数"""
    tester = APIPerformanceTester(base_url)

    print("🎯 Starting Comprehensive API Performance Tests")
    print(f"Target: {base_url}")

    # 1. 基准测试套件
    benchmark_results = await tester.run_api_benchmark_suite()

    # 2. 负载模式测试
    load_results = await tester.test_load_patterns()

    # 3. 限制测试
    limits_results = await tester.test_api_limits()

    # 4. 错误处理测试
    error_results = await tester.test_error_handling()

    # 5. 生成综合报告
    all_results = {
        "benchmark_suite": benchmark_results,
        "load_patterns": load_results,
        "api_limits": limits_results,
        "error_handling": error_results
    }

    report_path = tester.generate_performance_report(all_results)

    return all_results, report_path