#!/usr/bin/env python3
"""
重构前功能验证测试套件
用于验证重构前的系统功能，确保重构后功能完整性
"""

import asyncio
import httpx
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RefactorBaselineTester:
    """重构前功能基线测试器"""

    def __init__(self):
        self.test_results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }

    def record_test(self, test_name: str, passed: bool, details: str = ""):
        """记录测试结果"""
        self.test_results["total"] += 1
        if passed:
            self.test_results["passed"] += 1
            status = "✓ PASS"
        else:
            self.test_results["failed"] += 1
            status = "✗ FAIL"

        self.test_results["details"].append({
            "name": test_name,
            "status": status,
            "details": details
        })

        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  Details: {details}")

    async def test_file_structure(self) -> bool:
        """测试项目文件结构完整性"""
        logger.info("Testing project file structure...")

        required_files = [
            "rag_service/app.py",
            "api/main.py",
            "rag_service/bce/service.py",
            "rag_service/qwen3/service.py",
            "rag_service/ocr/service.py",
            "rag_service/llm/service.py",
        ]

        required_dirs = [
            "rag_service/",
            "api/",
            "config/",
            "tests/",
            "rag_service/providers/",
            "rag_service/services/",
        ]

        all_passed = True

        for file_path in required_files:
            if Path(file_path).exists():
                self.record_test(f"File exists: {file_path}", True)
            else:
                self.record_test(f"File exists: {file_path}", False, "File not found")
                all_passed = False

        for dir_path in required_dirs:
            if Path(dir_path).is_dir():
                self.record_test(f"Directory exists: {dir_path}", True)
            else:
                self.record_test(f"Directory exists: {dir_path}", False, "Directory not found")
                all_passed = False

        return all_passed

    async def test_api_imports(self) -> bool:
        """测试关键模块的导入能力"""
        logger.info("Testing module imports...")

        import_tests = [
            ("API main module", "api.main"),
            ("RAG Service app", "rag_service.app"),
            ("BCE Service", "rag_service.bce.service"),
            ("Qwen3 Service", "rag_service.qwen3.service"),
            ("OCR Service", "rag_service.ocr.service"),
            ("LLM Service", "rag_service.llm.service"),
        ]

        all_passed = True

        for test_name, module_name in import_tests:
            try:
                __import__(module_name)
                self.record_test(f"Import: {module_name}", True)
            except ImportError as e:
                self.record_test(f"Import: {module_name}", False, str(e))
                all_passed = False
            except Exception as e:
                self.record_test(f"Import: {module_name}", False, f"Unexpected error: {str(e)}")
                all_passed = False

        return all_passed

    async def test_service_configs(self) -> bool:
        """测试服务配置文件"""
        logger.info("Testing service configurations...")

        config_files = [
            "config/settings.py",
            "config/docker-compose.yml",
            "config/docker-compose.dev.yml",
        ]

        all_passed = True

        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    # 尝试读取配置文件内容
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content) > 0:
                        self.record_test(f"Config file readable: {config_file}", True)
                    else:
                        self.record_test(f"Config file readable: {config_file}", False, "Empty file")
                        all_passed = False
                except Exception as e:
                    self.record_test(f"Config file readable: {config_file}", False, str(e))
                    all_passed = False
            else:
                self.record_test(f"Config file exists: {config_file}", False, "File not found")
                all_passed = False

        return all_passed

    async def test_docker_com_structure(self) -> bool:
        """测试Docker相关配置"""
        logger.info("Testing Docker configuration...")

        docker_files = [
            "docker-compose.yml",
            "docker-compose.dev.yml",
            "docker-compose.milvus.yml",
        ]

        all_passed = True

        for docker_file in docker_files:
            docker_path = Path(docker_file)
            if docker_path.exists():
                try:
                    import yaml
                    with open(docker_path, 'r', encoding='utf-8') as f:
                        docker_config = yaml.safe_load(f)

                    if docker_config and 'services' in docker_config:
                        services = list(docker_config['services'].keys())
                        self.record_test(f"Docker compose valid: {docker_file}", True, f"Services: {services}")
                    else:
                        self.record_test(f"Docker compose valid: {docker_file}", False, "No services defined")
                        all_passed = False

                except ImportError:
                    self.record_test(f"Docker compose valid: {docker_file}", False, "PyYAML not available")
                    all_passed = False
                except Exception as e:
                    self.record_test(f"Docker compose valid: {docker_file}", False, str(e))
                    all_passed = False
            else:
                self.record_test(f"Docker compose exists: {docker_file}", False, "File not found")
                all_passed = False

        return all_passed

    async def test_provider_structure(self) -> bool:
        """测试提供者结构"""
        logger.info("Testing provider structure...")

        provider_dirs = [
            "rag_service/providers/openai/",
            "rag_service/providers/ollama/",
            "rag_service/providers/bce/",
            "rag_service/providers/qwen/",
        ]

        provider_files = [
            "rag_service/providers/openai/llm_provider.py",
            "rag_service/providers/openai/embedding_provider.py",
            "rag_service/providers/ollama/llm_provider.py",
            "rag_service/providers/bce/rerank_provider.py",
        ]

        all_passed = True

        for provider_dir in provider_dirs:
            if Path(provider_dir).is_dir():
                self.record_test(f"Provider directory exists: {provider_dir}", True)
            else:
                self.record_test(f"Provider directory exists: {provider_dir}", False, "Directory not found")
                all_passed = False

        for provider_file in provider_files:
            if Path(provider_file).exists():
                self.record_test(f"Provider file exists: {provider_file}", True)
            else:
                self.record_test(f"Provider file exists: {provider_file}", False, "File not found")
                all_passed = False

        return all_passed

    def generate_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 60)
        report.append("重构前功能基线测试报告")
        report.append("=" * 60)
        report.append(f"测试总数: {self.test_results['total']}")
        report.append(f"通过数量: {self.test_results['passed']}")
        report.append(f"失败数量: {self.test_results['failed']}")
        report.append(f"跳过数量: {self.test_results['skipped']}")
        report.append(f"通过率: {self.test_results['passed']/self.test_results['total']*100:.1f}%")
        report.append("")

        # 详细结果
        report.append("详细测试结果:")
        report.append("-" * 40)
        for detail in self.test_results["details"]:
            report.append(f"{detail['status']} {detail['name']}")
            if detail['details']:
                report.append(f"    └─ {detail['details']}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    async def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("Starting Refactor Baseline Tests...")

        test_methods = [
            self.test_file_structure,
            self.test_api_imports,
            self.test_service_configs,
            self.test_docker_com_structure,
            self.test_provider_structure,
        ]

        overall_passed = True

        for test_method in test_methods:
            try:
                passed = await test_method()
                if not passed:
                    overall_passed = False
                logger.info("-" * 40)
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} failed: {str(e)}")
                traceback.print_exc()
                overall_passed = False

        # 生成并打印报告
        report = self.generate_report()
        print(report)

        # 保存报告到文件
        with open("refactor_baseline_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        return overall_passed

async def main():
    """主函数"""
    tester = RefactorBaselineTester()

    try:
        success = await tester.run_all_tests()

        if success:
            print("✅ 所有基线测试通过 - 系统已准备好进行重构")
            sys.exit(0)
        else:
            print("❌ 部分基线测试失败 - 请在重构前解决这些问题")
            sys.exit(1)

    except Exception as e:
        logger.error(f"测试执行失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())