"""
服务性能测试
测试Sanic服务的性能表现
"""
import asyncio
import aiohttp
import time
import json
from typing import Dict, Any, List
from pathlib import Path
from .framework import PerformanceTestFramework, BenchmarkSuite


class ServicePerformanceTester:
    """服务性能测试器"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.framework = PerformanceTestFramework()

    async def check_service_availability(self) -> bool:
        """检查服务是否可用"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/ping", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False

    async def test_sanic_request_handling(self) -> Dict[str, Any]:
        """测试Sanic请求处理性能"""
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
                        "server_version": data.get("version", "unknown"),
                        "server_headers": dict(response.headers)
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "response_time": (end_time - start_time) * 1000,
                    "error": str(e)
                }

    async def test_sanic_static_file_serving(self) -> Dict[str, Any]:
        """测试Sanic静态文件服务性能"""
        # 假设有一个静态文件端点
        static_paths = [
            "/assets/static_resources/test.txt",
            "/favicon.ico",
            "/robots.txt"
        ]

        results = {}
        for path in static_paths:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}{path}") as response:
                        content = await response.read()
                        end_time = time.time()

                        results[path] = {
                            "success": response.status in [200, 404],  # 404也算成功（文件不存在）
                            "response_time": (end_time - start_time) * 1000,
                            "status_code": response.status,
                            "content_length": len(content),
                            "content_type": response.headers.get("content-type", "unknown")
                        }
                except Exception as e:
                    end_time = time.time()
                    results[path] = {
                        "success": False,
                        "response_time": (end_time - start_time) * 1000,
                        "error": str(e)
                    }

        return results

    async def test_sanic_middleware_performance(self) -> Dict[str, Any]:
        """测试Sanic中间件性能"""
        # 测试带有不同中间件的请求
        test_requests = [
            {"path": "/api/health", "description": "Basic health check"},
            {"path": "/api/chat/models", "description": "With authentication middleware"},
            {"path": "/api/documents", "description": "With caching middleware"},
            {"path": "/api/analytics/dashboard", "description": "With monitoring middleware"}
        ]

        results = {}
        for req in test_requests:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}{req['path']}") as response:
                        end_time = time.time()

                        # 检查响应头中的中间件信息
                        middleware_headers = {}
                        for header, value in response.headers.items():
                            if header.startswith('X-'):
                                middleware_headers[header] = value

                        results[req['description']] = {
                            "success": response.status in [200, 404],
                            "response_time": (end_time - start_time) * 1000,
                            "status_code": response.status,
                            "middleware_headers": middleware_headers,
                            "total_headers": len(response.headers)
                        }
                except Exception as e:
                    end_time = time.time()
                    results[req['description']] = {
                        "success": False,
                        "response_time": (end_time - start_time) * 1000,
                        "error": str(e)
                    }

        return results

    async def test_sanic_websocket_performance(self) -> Dict[str, Any]:
        """测试Sanic WebSocket性能（如果支持）"""
        try:
            import websockets
        except ImportError:
            return {"error": "websockets library not installed"}

        websocket_url = f"ws://{self.host}:{port}/ws"

        try:
            # 测试WebSocket连接
            start_time = time.time()
            async with websockets.connect(websocket_url) as websocket:
                # 发送测试消息
                test_message = json.dumps({"type": "ping", "data": "test"})
                await websocket.send(test_message)

                # 接收响应
                response = await websocket.recv()
                end_time = time.time()

                return {
                    "success": True,
                    "connection_time": (end_time - start_time) * 1000,
                    "response": response,
                    "message_size": len(test_message)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def test_sanic_error_handling(self) -> Dict[str, Any]:
        """测试Sanic错误处理性能"""
        error_scenarios = [
            {"path": "/api/nonexistent", "description": "404 Not Found"},
            {"path": "/api/chat/completions", "method": "POST", "payload": {}, "description": "400 Bad Request"},
            {"path": "/api/trigger-error", "description": "500 Server Error"}  # 如果存在错误触发端点
        ]

        results = {}
        for scenario in error_scenarios:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                try:
                    if scenario.get("method") == "POST":
                        async with session.post(
                            f"{self.base_url}{scenario['path']}",
                            json=scenario.get("payload", {})
                        ) as response:
                            end_time = time.time()
                            content = await response.text()
                    else:
                        async with session.get(f"{self.base_url}{scenario['path']}") as response:
                            end_time = time.time()
                            content = await response.text()

                    results[scenario['description']] = {
                        "expected_error": True,
                        "response_time": (end_time - start_time) * 1000,
                        "status_code": response.status,
                        "error_handling_time": (end_time - start_time) * 1000,
                        "response_size": len(content)
                    }
                except Exception as e:
                    end_time = time.time()
                    results[scenario['description']] = {
                        "success": False,
                        "response_time": (end_time - start_time) * 1000,
                        "error": str(e)
                    }

        return results

    async def test_sanic_memory_usage(self) -> Dict[str, Any]:
        """测试Sanic内存使用情况"""
        import psutil

        # 获取当前进程信息
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行一系列请求来测试内存增长
        memory_samples = [initial_memory]

        for i in range(100):
            await self.test_sanic_request_handling()

            if i % 10 == 0:  # 每10个请求采样一次
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)

        final_memory = process.memory_info().rss / 1024 / 1024

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": final_memory - initial_memory,
            "memory_samples": memory_samples,
            "peak_memory_mb": max(memory_samples),
            "avg_memory_mb": sum(memory_samples) / len(memory_samples)
        }

    async def test_sanic_concurrent_connections(self) -> Dict[str, Any]:
        """测试Sanic并发连接处理"""
        concurrent_levels = [10, 50, 100, 200, 500]
        results = {}

        for concurrency in concurrent_levels:
            print(f"Testing with {concurrency} concurrent connections...")

            start_time = time.time()
            success_count = 0
            error_count = 0
            response_times = []

            async def single_request():
                nonlocal success_count, error_count
                try:
                    req_start = time.time()
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.base_url}/api/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                            req_end = time.time()
                            if response.status == 200:
                                success_count += 1
                                response_times.append((req_end - req_start) * 1000)
                            else:
                                error_count += 1
                except Exception:
                    error_count += 1

            # 并发执行请求
            tasks = [single_request() for _ in range(concurrency)]
            await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            total_time = end_time - start_time

            results[concurrency] = {
                "total_requests": concurrency,
                "successful_requests": success_count,
                "failed_requests": error_count,
                "success_rate": success_count / concurrency,
                "total_time": total_time,
                "requests_per_second": concurrency / total_time,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0
            }

        return results

    async def run_sanic_performance_suite(self) -> Dict[str, Any]:
        """运行Sanic性能测试套件"""
        print("🚀 Starting Sanic Service Performance Tests")
        print(f"Target: {self.base_url}")

        # 检查服务可用性
        if not await self.check_service_availability():
            print("❌ Service is not available. Please start the Sanic service first.")
            return {"error": "Service not available"}

        # 1. 基础请求处理性能
        print("\n📊 Testing basic request handling...")
        basic_result = await self.framework.run_concurrent_test(
            self.test_sanic_request_handling,
            concurrency=50,
            total_requests=1000
        )

        # 2. 中间件性能测试
        print("\n🔧 Testing middleware performance...")
        middleware_result = await self.test_sanic_middleware_performance()

        # 3. 静态文件服务性能
        print("\n📁 Testing static file serving...")
        static_result = await self.test_sanic_static_file_serving()

        # 4. 错误处理性能
        print("\n❌ Testing error handling...")
        error_result = await self.test_sanic_error_handling()

        # 5. 并发连接测试
        print("\n🔗 Testing concurrent connections...")
        concurrent_result = await self.test_sanic_concurrent_connections()

        # 6. 内存使用测试
        print("\n💾 Testing memory usage...")
        memory_result = await self.test_sanic_memory_usage()

        # 7. WebSocket测试（如果支持）
        print("\n🌐 Testing WebSocket performance...")
        websocket_result = await self.test_sanic_websocket_performance()

        return {
            "basic_performance": basic_result,
            "middleware_performance": middleware_result,
            "static_file_performance": static_result,
            "error_handling_performance": error_result,
            "concurrent_connections": concurrent_result,
            "memory_usage": memory_result,
            "websocket_performance": websocket_result
        }

    def generate_sanic_performance_report(self, results: Dict[str, Any]):
        """生成Sanic性能报告"""
        print("\n📊 Generating Sanic Performance Report...")

        report = {
            "test_summary": {
                "timestamp": time.time(),
                "service_url": self.base_url,
                "total_tests": len(results)
            },
            "results": {}
        }

        # 处理基础性能测试
        if "basic_performance" in results:
            basic = results["basic_performance"]
            report["results"]["basic_performance"] = {
                "avg_response_time": basic.avg_response_time,
                "p95_response_time": basic.p95_response_time,
                "requests_per_second": basic.requests_per_second,
                "success_rate": basic.success_rate,
                "max_cpu": basic.metadata.get('resource_usage', {}).get('cpu', {}).get('max', 0),
                "max_memory": basic.metadata.get('resource_usage', {}).get('memory', {}).get('max', 0)
            }

        # 处理并发连接测试
        if "concurrent_connections" in results:
            concurrent = results["concurrent_connections"]
            report["results"]["concurrent_connections"] = concurrent

        # 处理内存使用测试
        if "memory_usage" in results:
            memory = results["memory_usage"]
            report["results"]["memory_usage"] = memory

        # 处理其他测试结果
        for test_name, result in results.items():
            if test_name not in ["basic_performance", "concurrent_connections", "memory_usage"]:
                report["results"][test_name] = result

        # 保存报告
        report_path = f"tests/performance/reports/sanic_performance_{int(time.time())}.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Sanic performance report saved to: {report_path}")

        # 打印摘要
        self._print_sanic_performance_summary(report)

        return report_path

    def _print_sanic_performance_summary(self, report: Dict):
        """打印Sanic性能摘要"""
        print(f"\n{'='*80}")
        print("📊 Sanic Service Performance Summary")
        print(f"{'='*80}")

        results = report.get("results", {})

        # 基础性能摘要
        if "basic_performance" in results:
            basic = results["basic_performance"]
            print(f"\n🎯 Basic Performance:")
            print(f"  Average Response Time: {basic.get('avg_response_time', 0):.2f}ms")
            print(f"  95th Percentile: {basic.get('p95_response_time', 0):.2f}ms")
            print(f"  Requests/Second: {basic.get('requests_per_second', 0):.2f}")
            print(f"  Success Rate: {basic.get('success_rate', 0):.2%}")

        # 并发连接摘要
        if "concurrent_connections" in results:
            concurrent = results["concurrent_connections"]
            print(f"\n🔗 Concurrent Connections:")
            print(f"{'Concurrency':<12} {'RPS':<8} {'Success':<8} {'Avg RT':<8}")
            print(f"{'-'*40}")
            for concurrency, metrics in concurrent.items():
                print(f"{concurrency:<12} "
                      f"{metrics.get('requests_per_second', 0):<8.1f} "
                      f"{metrics.get('success_rate', 0):<8.1%} "
                      f"{metrics.get('avg_response_time', 0):<8.1f}")

        # 内存使用摘要
        if "memory_usage" in results:
            memory = results["memory_usage"]
            print(f"\n💾 Memory Usage:")
            print(f"  Initial: {memory.get('initial_memory_mb', 0):.2f}MB")
            print(f"  Final: {memory.get('final_memory_mb', 0):.2f}MB")
            print(f"  Growth: {memory.get('memory_growth_mb', 0):.2f}MB")
            print(f"  Peak: {memory.get('peak_memory_mb', 0):.2f}MB")

        print(f"\n{'='*80}")


# 便捷函数
async def run_sanic_performance_tests(host: str = "127.0.0.1", port: int = 8000):
    """运行Sanic性能测试的主函数"""
    tester = ServicePerformanceTester(host, port)

    print("🎯 Starting Comprehensive Sanic Performance Tests")
    print(f"Target: {tester.base_url}")

    # 运行完整的性能测试套件
    results = await tester.run_sanic_performance_suite()

    # 生成报告
    report_path = tester.generate_sanic_performance_report(results)

    return results, report_path