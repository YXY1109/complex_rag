#!/usr/bin/env python3
"""
OpenAI兼容接口测试脚本
测试BCE服务的OpenAI格式向量化和重排序接口
"""

import json
import time
from typing import Dict, Any

import requests


class BCEOpenAITestClient:
    def __init__(self, base_url: str = "http://localhost:7001"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def test_health_check(self) -> Dict[str, Any]:
        """测试健康检查接口"""
        print("=== 测试健康检查接口 ===")
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.json()
        except Exception as e:
            print(f"健康检查失败: {e}")
            return {}

    def test_openai_embeddings(self, texts: list) -> Dict[str, Any]:
        """测试OpenAI兼容的向量化接口"""
        print(f"\n=== 测试OpenAI向量化接口 ===")
        print(f"输入文本: {texts}")

        payload = {
            "input": texts,
            "model": "bce-embedding-base_v1",
            "encoding_format": "float"
        }

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()

            print(f"状态码: {response.status_code}")
            print(f"请求耗时: {end_time - start_time:.2f}s")

            if response.status_code == 200:
                result = response.json()
                print(f"模型: {result.get('model')}")
                print(f"数据对象: {result.get('object')}")
                print(f"使用量: {result.get('usage')}")
                print(f"向量数量: {len(result.get('data', []))}")
                print(f"向量维度: {len(result.get('data', [{}])[0].get('embedding', [])) if result.get('data') else 0}")
                return result
            else:
                print(f"错误响应: {response.text}")
                return {}

        except Exception as e:
            print(f"向量化测试失败: {e}")
            return {}

    def test_openai_rerank(self, query: str, documents: list, top_n: int = None) -> Dict[str, Any]:
        """测试OpenAI兼容的重排序接口"""
        print(f"\n=== 测试OpenAI重排序接口 ===")
        print(f"查询: {query}")
        print(f"文档数量: {len(documents)}")
        print(f"文档列表: {documents}")

        payload = {
            "query": query,
            "documents": documents,
            "model": "bce-reranker-base_v1"
        }

        if top_n:
            payload["top_n"] = top_n

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()

            print(f"状态码: {response.status_code}")
            print(f"请求耗时: {end_time - start_time:.2f}s")

            if response.status_code == 200:
                result = response.json()
                print(f"任务ID: {result.get('id')}")
                print(f"模型: {result.get('model')}")
                print(f"使用量: {result.get('usage')}")
                print("重排序结果:")
                for i, item in enumerate(result.get('results', [])):
                    print(
                        f"  {i + 1}. [分数: {item.get('relevance_score', 0):.4f}] {item.get('document', {}).get('text', '')}")
                return result
            else:
                print(f"错误响应: {response.text}")
                return {}

        except Exception as e:
            print(f"重排序测试失败: {e}")
            return {}

    def run_all_tests(self):
        """运行所有测试"""
        print("BCE OpenAI兼容接口测试开始")
        print("=" * 50)

        # 健康检查
        health = self.test_health_check()
        if health.get("status") != "healthy":
            print("服务状态不健康，终止测试")
            return

        # OpenAI向量化测试
        self.test_openai_embeddings(["这是一个测试文本", "这是另一个测试文本"])

        # OpenAI重排序测试
        query = "什么是人工智能"
        documents = [
            "人工智能是指由人制造出来的机器所表现出来的智能",
            "机器学习是人工智能的一个子领域",
            "今天天气晴朗适合外出游玩",
            "深度学习使用神经网络来模拟人脑的工作方式"
        ]
        self.test_openai_rerank(query, documents, top_n=3)

        print("\n" + "=" * 50)
        print("所有测试完成")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="BCE OpenAI兼容接口测试")
    parser.add_argument("--url", default="http://localhost:7001", help="服务地址")
    parser.add_argument("--test", choices=["health", "embeddings", "rerank", "all"],
                        default="all", help="指定测试类型")

    args = parser.parse_args()

    client = BCEOpenAITestClient(args.url)

    if args.test == "all":
        client.run_all_tests()
    elif args.test == "health":
        client.test_health_check()
    elif args.test == "embeddings":
        client.test_openai_embeddings(["测试向量化功能"])
    elif args.test == "rerank":
        client.test_openai_rerank("测试查询", ["文档1", "文档2", "文档3"])


if __name__ == "__main__":
    main()
