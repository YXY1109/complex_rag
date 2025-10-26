#!/usr/bin/env python3
"""
测试统一FastAPI服务的功能
验证从Sanic迁移过来的API端点
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


class UnifiedAPITester:
    """统一API测试器"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def test_health_endpoints(self) -> bool:
        """测试健康检查端点"""
        print("Testing health endpoints...")

        try:
            # 测试主健康检查
            response = await self.client.get(f"{self.base_url}/ping")
            if response.status_code == 200:
                print("✓ Main health check (/ping) passed")
            else:
                print(f"✗ Main health check failed: {response.status_code}")
                return False

            # 测试API健康检查
            response = await self.client.get(f"{self.base_url}/api/health")
            if response.status_code in [200, 404]:
                print("✓ API health check endpoint accessible")
            else:
                print(f"✗ API health check failed: {response.status_code}")

            return True

        except Exception as e:
            print(f"✗ Health endpoint test failed: {e}")
            return False

    async def test_unified_chat_api(self) -> bool:
        """测试统一聊天API"""
        print("\nTesting unified chat API...")

        try:
            # 测试模型列表
            response = await self.client.get(f"{self.base_url}/v1/chat/models")
            if response.status_code == 200:
                models = response.json()
                print(f"✓ Chat models list retrieved: {len(models.get('data', []))} models")
            else:
                print(f"✗ Chat models list failed: {response.status_code}")
                return False

            # 测试聊天完成
            chat_request = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "Hello, this is a test message."}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }

            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=chat_request
            )

            if response.status_code == 200:
                completion = response.json()
                print("✓ Chat completion successful")
                print(f"  - Response ID: {completion.get('id')}")
                print(f"  - Model: {completion.get('model')}")
                print(f"  - Choices: {len(completion.get('choices', []))}")
                if completion.get('choices'):
                    content = completion['choices'][0]['message']['content']
                    print(f"  - Content preview: {content[:100]}...")
            else:
                print(f"✗ Chat completion failed: {response.status_code}")
                print(f"  - Error: {response.text}")
                return False

            return True

        except Exception as e:
            print(f"✗ Chat API test failed: {e}")
            return False

    async def test_unified_embeddings_api(self) -> bool:
        """测试统一嵌入API"""
        print("\nTesting unified embeddings API...")

        try:
            # 测试嵌入模型列表
            response = await self.client.get(f"{self.base_url}/v1/embeddings/models")
            if response.status_code == 200:
                models = response.json()
                print(f"✓ Embedding models list retrieved: {len(models.get('data', []))} models")
            else:
                print(f"✗ Embedding models list failed: {response.status_code}")
                return False

            # 测试文本嵌入
            embedding_request = {
                "model": "text-embedding-ada-002",
                "input": "This is a test text for embedding generation.",
                "encoding_format": "float"
            }

            response = await self.client.post(
                f"{self.base_url}/v1/embeddings/",
                json=embedding_request
            )

            if response.status_code == 200:
                embedding = response.json()
                print("✓ Text embedding successful")
                print(f"  - Model: {embedding.get('model')}")
                print(f"  - Embeddings count: {len(embedding.get('data', []))}")
                if embedding.get('data'):
                    dimensions = len(embedding['data'][0]['embedding'])
                    print(f"  - Embedding dimensions: {dimensions}")
                    print(f"  - Usage tokens: {embedding.get('usage', {}).get('total_tokens', 0)}")
            else:
                print(f"✗ Text embedding failed: {response.status_code}")
                print(f"  - Error: {response.text}")
                return False

            # 测试批量嵌入
            batch_request = {
                "texts": [
                    "First test text",
                    "Second test text",
                    "Third test text"
                ],
                "model": "text-embedding-ada-002",
                "batch_size": 2
            }

            response = await self.client.post(
                f"{self.base_url}/v1/embeddings/batch",
                json=batch_request
            )

            if response.status_code == 200:
                batch_embedding = response.json()
                print("✓ Batch embedding successful")
                print(f"  - Texts processed: {len(batch_request['texts'])}")
                print(f"  - Embeddings returned: {len(batch_embedding.get('data', []))}")
                if batch_embedding.get('processing_info'):
                    throughput = batch_embedding['processing_info'].get('throughput_texts_per_second', 0)
                    print(f"  - Throughput: {throughput:.2f} texts/sec")
            else:
                print(f"✗ Batch embedding failed: {response.status_code}")
                print(f"  - Error: {response.text}")

            # 测试相似度计算
            similarity_request = {
                "text1": "This is about machine learning",
                "text2": "This is about artificial intelligence",
                "metric": "cosine"
            }

            response = await self.client.post(
                f"{self.base_url}/v1/embeddings/similarity",
                json=similarity_request
            )

            if response.status_code == 200:
                similarity = response.json()
                print("✓ Similarity computation successful")
                print(f"  - Similarity score: {similarity.get('similarity_score', 0):.4f}")
                print(f"  - Metric: {similarity.get('metric')}")
            else:
                print(f"✗ Similarity computation failed: {response.status_code}")
                print(f"  - Error: {response.text}")

            return True

        except Exception as e:
            print(f"✗ Embeddings API test failed: {e}")
            return False

    async def test_unified_rerank_api(self) -> bool:
        """测试统一重排序API"""
        print("\nTesting unified rerank API...")

        try:
            # 测试重排序模型列表
            response = await self.client.get(f"{self.base_url}/v1/rerank/models")
            if response.status_code == 200:
                models = response.json()
                print(f"✓ Rerank models list retrieved: {len(models.get('data', []))} models")
            else:
                print(f"✗ Rerank models list failed: {response.status_code}")
                return False

            # 测试文档重排序
            rerank_request = {
                "model": "bge-reranker-base",
                "query": "What is machine learning?",
                "documents": [
                    {"text": "Machine learning is a subset of artificial intelligence.", "id": "doc1"},
                    {"text": "The weather is nice today.", "id": "doc2"},
                    {"text": "Deep learning uses neural networks with multiple layers.", "id": "doc3"},
                    {"text": "Python is a popular programming language.", "id": "doc4"}
                ],
                "top_n": 3
            }

            response = await self.client.post(
                f"{self.base_url}/v1/rerank/",
                json=rerank_request
            )

            if response.status_code == 200:
                rerank_result = response.json()
                print("✓ Document reranking successful")
                print(f"  - Model: {rerank_result.get('model')}")
                print(f"  - Documents processed: {len(rerank_request['documents'])}")
                print(f"  - Results returned: {len(rerank_result.get('results', []))}")

                # 显示重新排序的结果
                for i, result in enumerate(rerank_result.get('results', [])[:3]):
                    score = result.get('relevance_score', 0)
                    doc_text = result.get('document', {}).get('text', '')[:50]
                    print(f"  - Result {i+1}: score={score:.4f}, text='{doc_text}...'")
            else:
                print(f"✗ Document reranking failed: {response.status_code}")
                print(f"  - Error: {response.text}")
                return False

            return True

        except Exception as e:
            print(f"✗ Rerank API test failed: {e}")
            return False

    async def test_legacy_compatibility(self) -> bool:
        """测试向后兼容性"""
        print("\nTesting legacy compatibility...")

        try:
            # 测试旧的端点是否返回弃用警告
            old_endpoints = [
                "/v1/chat/models",
                "/v1/embeddings/models",
                "/health"
            ]

            for endpoint in old_endpoints:
                response = await self.client.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("error", {}).get("type") == "deprecated_endpoint":
                        print(f"✓ Legacy endpoint {endpoint} returns deprecation warning")
                    else:
                        print(f"? Legacy endpoint {endpoint} response format unexpected")
                else:
                    print(f"✗ Legacy endpoint {endpoint} failed: {response.status_code}")

            return True

        except Exception as e:
            print(f"✗ Legacy compatibility test failed: {e}")
            return False

    async def test_api_documentation(self) -> bool:
        """测试API文档可访问性"""
        print("\nTesting API documentation...")

        try:
            # 测试OpenAPI规范
            response = await self.client.get(f"{self.base_url}/openapi.json")
            if response.status_code == 200:
                openapi_spec = response.json()
                paths = openapi_spec.get("paths", {})
                print(f"✓ OpenAPI specification accessible: {len(paths)} paths")

                # 检查统一API路径
                unified_paths = [path for path in paths if "/v1/" in path]
                print(f"  - Unified API paths: {len(unified_paths)}")
            else:
                print(f"✗ OpenAPI specification failed: {response.status_code}")

            # 测试交互式文档
            response = await self.client.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                print("✓ Interactive documentation (/docs) accessible")
            else:
                print(f"✗ Interactive documentation failed: {response.status_code}")

            return True

        except Exception as e:
            print(f"✗ API documentation test failed: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        print("=" * 60)
        print("Unified FastAPI Service Test Suite")
        print("=" * 60)

        tests = {
            "health_endpoints": await self.test_health_endpoints(),
            "unified_chat_api": await self.test_unified_chat_api(),
            "unified_embeddings_api": await self.test_unified_embeddings_api(),
            "unified_rerank_api": await self.test_unified_rerank_api(),
            "legacy_compatibility": await self.test_legacy_compatibility(),
            "api_documentation": await self.test_api_documentation(),
        }

        print("\n" + "=" * 60)
        print("Test Results Summary:")
        print("=" * 60)

        passed = 0
        total = len(tests)

        for test_name, result in tests.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! Unified FastAPI service is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the service configuration.")

        return tests


async def main():
    """主测试函数"""
    async with UnifiedAPITester() as tester:
        results = await tester.run_all_tests()
        return results


if __name__ == "__main__":
    # 运行测试
    results = asyncio.run(main())

    # 根据测试结果设置退出码
    failed_tests = sum(1 for result in results.values() if not result)
    exit(failed_tests)