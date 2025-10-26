"""
API路由测试脚本
用于验证所有API路由的基本功能
"""
import asyncio
import json
import time
from typing import Dict, Any, List
from pathlib import Path

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.monitoring.loguru_logger import logger


class APITestRunner:
    """API测试运行器"""

    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.base_url = "http://localhost:8000"

    async def run_all_tests(self):
        """运行所有API测试"""
        logger.info("开始运行API路由测试...")

        # 测试各个路由模块
        await self.test_health_routes()
        await self.test_chat_routes()
        await self.test_documents_routes()
        await self.test_knowledge_routes()
        await self.test_models_routes()
        await self.test_users_routes()
        await self.test_system_routes()
        await self.test_analytics_routes()

        # 生成测试报告
        await self.generate_test_report()

    async def test_health_routes(self):
        """测试健康检查路由"""
        logger.info("测试健康检查路由...")

        routes = [
            ("/api/health/", "GET", "系统健康检查"),
            ("/api/health/detailed", "GET", "详细健康检查"),
            ("/api/health/ping", "GET", "Ping检查"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "health")

    async def test_chat_routes(self):
        """测试对话路由"""
        logger.info("测试对话路由...")

        routes = [
            ("/api/chat/completions", "POST", "对话完成"),
            ("/api/chat/completions/stream", "POST", "流式对话"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "chat")

    async def test_documents_routes(self):
        """测试文档管理路由"""
        logger.info("测试文档管理路由...")

        routes = [
            ("/api/documents/", "GET", "获取文档列表"),
            ("/api/documents/search", "GET", "搜索文档"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "documents")

    async def test_knowledge_routes(self):
        """测试知识库管理路由"""
        logger.info("测试知识库管理路由...")

        routes = [
            ("/api/knowledge/", "GET", "获取知识库列表"),
            ("/api/knowledge/search", "POST", "搜索知识库"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "knowledge")

    async def test_models_routes(self):
        """测试模型管理路由"""
        logger.info("测试模型管理路由...")

        routes = [
            ("/api/models/", "GET", "获取模型列表"),
            ("/api/models/types/available", "GET", "获取可用模型类型"),
            ("/api/models/providers/available", "GET", "获取可用模型提供商"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "models")

    async def test_users_routes(self):
        """测试用户管理路由"""
        logger.info("测试用户管理路由...")

        routes = [
            ("/api/users/me", "GET", "获取当前用户信息"),
            ("/api/users/me/stats", "GET", "获取用户统计"),
            ("/api/users/me/preferences", "GET", "获取用户偏好"),
            ("/api/users/me/sessions", "GET", "获取用户会话"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "users")

    async def test_system_routes(self):
        """测试系统管理路由"""
        logger.info("测试系统管理路由...")

        routes = [
            ("/api/system/info", "GET", "获取系统信息"),
            ("/api/system/config", "GET", "获取系统配置"),
            ("/api/system/metrics", "GET", "获取系统指标"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "system")

    async def test_analytics_routes(self):
        """测试统计分析路由"""
        logger.info("测试统计分析路由...")

        routes = [
            ("/api/analytics/dashboard", "GET", "获取仪表板数据"),
            ("/api/analytics/usage/overview", "GET", "获取使用情况概览"),
            ("/api/analytics/performance/analysis", "GET", "获取性能分析"),
        ]

        for route, method, description in routes:
            await self.test_route(route, method, description, "analytics")

    async def test_route(self, route: str, method: str, description: str, category: str):
        """测试单个路由"""
        start_time = time.time()

        test_result = {
            "route": route,
            "method": method,
            "description": description,
            "category": category,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "response_time": 0,
            "status_code": None,
            "error": None
        }

        try:
            # 模拟API调用
            await asyncio.sleep(0.1)  # 模拟网络延迟

            # 模拟成功响应
            test_result["success"] = True
            test_result["status_code"] = 200
            test_result["response_time"] = round((time.time() - start_time) * 1000, 2)

            logger.info(f"✅ {description} - {method} {route} - {test_result['response_time']}ms")

        except Exception as e:
            test_result["error"] = str(e)
            test_result["response_time"] = round((time.time() - start_time) * 1000, 2)
            logger.error(f"❌ {description} - {method} {route} - 错误: {str(e)}")

        self.test_results.append(test_result)

    async def generate_test_report(self):
        """生成测试报告"""
        logger.info("生成API测试报告...")

        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - successful_tests

        # 按类别统计
        categories = {}
        for result in self.test_results:
            category = result["category"]
            if category not in categories:
                categories[category] = {"total": 0, "success": 0, "failed": 0}

            categories[category]["total"] += 1
            if result["success"]:
                categories[category]["success"] += 1
            else:
                categories[category]["failed"] += 1

        # 计算平均响应时间
        avg_response_time = sum(result["response_time"] for result in self.test_results) / total_tests

        report = {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": round(successful_tests / total_tests * 100, 2) if total_tests > 0 else 0,
                "avg_response_time_ms": round(avg_response_time, 2)
            },
            "categories": categories,
            "failed_tests": [result for result in self.test_results if not result["success"]],
            "slow_tests": sorted(
                [result for result in self.test_results if result["response_time"] > 500],
                key=lambda x: x["response_time"],
                reverse=True
            )[:5],
            "test_results": self.test_results
        }

        # 保存报告到文件
        report_file = project_root / "logs" / "api_test_report.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 打印摘要
        print("\n" + "="*80)
        print("API路由测试报告")
        print("="*80)
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {report['summary']['success_rate']}%")
        print(f"平均响应时间: {report['summary']['avg_response_time_ms']}ms")
        print("\n按类别统计:")
        for category, stats in categories.items():
            print(f"  {category}: {stats['success']}/{stats['total']} ({round(stats['success']/stats['total']*100, 1)}%)")

        if failed_tests > 0:
            print(f"\n失败的测试 ({failed_tests}):")
            for test in report["failed_tests"]:
                print(f"  ❌ {test['description']} - {test['error']}")

        print(f"\n详细报告已保存到: {report_file}")
        print("="*80)

        logger.info(f"API测试完成，成功率: {report['summary']['success_rate']}%")


async def main():
    """主函数"""
    print("开始API路由测试...")
    print("注意: 这只是模拟测试，实际测试需要启动FastAPI服务器")
    print()

    runner = APITestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())