#!/usr/bin/env python3
"""
统一嵌入服务测试
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"

class UnifiedEmbeddingTester:
    """统一嵌入服务测试器"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def test_models_list(self) -> bool:
        """测试模型列表"""
        print("📋 测试模型列表...")
        try:
            response = await self.client.get(f"{self.base_url}/v1/embeddings/models")
            if response.status_code == 200:
                models_data = response.json()
                print(f"✅ 模型列表获取成功: {len(models_data.get('data', []))} 个模型")
                for model in models_data.get('data', []):
                    print(f"   - {model['name']} ({model['type']}) - 默认: {model['is_default']}")
                return True
            else:
                print(f"❌ 模型列表获取失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 模型列表测试失败: {str(e)}")
            return False

    async def test_embeddings(self) -> bool:
        """测试文本嵌入"""
        print("🔤 测试文本嵌入...")
        try:
            test_text = "这是一个测试文本，用于验证嵌入服务功能。"

            # 测试OpenAI兼容接口
            payload = {
                "input": test_text,
                "model": None  # 使用默认模型
            }

            response = await self.client.post(
                f"{self.base_url}/v1/embeddings/",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                embeddings = result.get('data', [])
                if embeddings:
                    embedding = embeddings[0]['embedding']
                    print(f"✅ 文本嵌入生成成功")
                    print(f"   模型: {result.get('model')}")
                    print(f"   维度: {len(embedding)}")
                    print(f"   使用量: {result.get('usage')}")
                    return True
                else:
                    print("❌ 嵌入结果为空")
                    return False
            else:
                print(f"❌ 文本嵌入生成失败: {response.status_code}")
                print(f"   错误详情: {response.text}")
                return False

        except Exception as e:
            print(f"❌ 文本嵌入测试失败: {str(e)}")
            return False

    async def test_batch_embeddings(self) -> bool:
        """测试批量嵌入"""
        print("📦 测试批量嵌入...")
        try:
            test_texts = [
                "第一个测试文本",
                "第二个测试文本",
                "第三个测试文本"
            ]

            payload = {
                "input": test_texts,
                "model": None  # 使用默认模型
            }

            response = await self.client.post(
                f"{self.base_url}/v1/embeddings/",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                embeddings = result.get('data', [])
                if len(embeddings) == len(test_texts):
                    print(f"✅ 批量嵌入生成成功")
                    print(f"   生成嵌入数量: {len(embeddings)}")
                    print(f"   每个嵌入维度: {len(embeddings[0]['embedding'])}")
                    return True
                else:
                    print(f"❌ 批量嵌入数量不匹配: 期望 {len(test_texts)}, 实际 {len(embeddings)}")
                    return False
            else:
                print(f"❌ 批量嵌入生成失败: {response.status_code}")
                print(f"   错误详情: {response.text}")
                return False

        except Exception as e:
            print(f"❌ 批量嵌入测试失败: {str(e)}")
            return False

    async def test_similarity(self) -> bool:
        """测试文本相似度"""
        print("📊 测试文本相似度...")
        try:
            payload = {
                "text1": "苹果是一种水果",
                "text2": "橙子也是一种水果",
                "model": None  # 使用默认模型
            }

            response = await self.client.post(
                f"{self.base_url}/v1/embeddings/similarity",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                similarity = result.get('similarity_score')
                if similarity is not None:
                    print(f"✅ 文本相似度计算成功")
                    print(f"   相似度分数: {similarity:.4f}")
                    print(f"   使用模型: {result.get('model')}")
                    return True
                else:
                    print("❌ 相似度分数为空")
                    return False
            else:
                print(f"❌ 文本相似度计算失败: {response.status_code}")
                print(f"   错误详情: {response.text}")
                return False

        except Exception as e:
            print(f"❌ 文本相似度测试失败: {str(e)}")
            return False

    async def test_health_check(self) -> bool:
        """测试健康检查"""
        print("🏥 测试健康检查...")
        try:
            response = await self.client.get(f"{self.base_url}/v1/embeddings/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ 健康检查成功")
                print(f"   服务状态: {health_data.get('data', {}).get('status')}")
                print(f"   已加载模型: {health_data.get('data', {}).get('loaded_models', [])}")
                print(f"   缓存大小: {health_data.get('data', {}).get('cache_size', 0)}")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ 健康检查测试失败: {str(e)}")
            return False

    async def run_all_tests(self) -> bool:
        """运行所有测试"""
        print("🚀 开始统一嵌入服务测试")
        print("=" * 50)

        tests = [
            ("健康检查", self.test_health_check),
            ("模型列表", self.test_models_list),
            ("文本嵌入", self.test_embeddings),
            ("批量嵌入", self.test_batch_embeddings),
            ("文本相似度", self.test_similarity),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n🧪 运行测试: {test_name}")
            try:
                if await test_func():
                    passed += 1
                    print(f"✅ {test_name} - 通过")
                else:
                    print(f"❌ {test_name} - 失败")
            except Exception as e:
                print(f"❌ {test_name} - 异常: {str(e)}")

            print("-" * 30)

        print(f"\n📊 测试结果: {passed}/{total} 通过")
        print("=" * 50)

        if passed == total:
            print("🎉 所有测试通过！统一嵌入服务工作正常。")
            return True
        else:
            print("⚠️  部分测试失败，请检查服务配置。")
            return False


async def main():
    """主函数"""
    async with UnifiedEmbeddingTester() as tester:
        success = await tester.run_all_tests()
        return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)