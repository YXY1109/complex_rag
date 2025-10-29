import json
import time
from typing import Dict, Any

import requests


class QwenOpenAITestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
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
            "model": "qwen3-embedding",
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
            "model": "qwen3-reranker"
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
                print(f"对象类型: {result.get('object')}")
                print(f"模型: {result.get('model')}")
                print(f"使用量: {result.get('usage')}")
                print("重排序结果:")
                for i, item in enumerate(result.get('data', [])):
                    print(f"  {i + 1}. [分数: {item.get('relevance_score', 0):.4f}] {item.get('document', '')}")
                return result
            else:
                print(f"错误响应: {response.text}")
                return {}

        except Exception as e:
            print(f"重排序测试失败: {e}")
            return {}

    def test_embedding_dimensions(self) -> Dict[str, Any]:
        """测试指定嵌入维度的向量化接口"""
        print(f"\n=== 测试指定维度的向量化接口 ===")

        payload = {
            "input": "测试文本维度指定功能",
            "model": "qwen3-embedding",
            "dimensions": 768,
            "user": "test_user"
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
                embedding = result.get('data', [{}])[0].get('embedding', [])
                print(f"向量维度: {len(embedding)}")
                print(f"指定维度: 768")
                print(f"维度匹配: {'✓' if len(embedding) == 768 else '✗'}")
                return result
            else:
                print(f"错误响应: {response.text}")
                return {}

        except Exception as e:
            print(f"维度测试失败: {e}")
            return {}

    def test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        print(f"\n=== 测试错误处理 ===")

        # 测试1: 缺少input参数
        print("测试1: 嵌入接口缺少input参数")
        try:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": "qwen3-embedding"},
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            print(f"错误响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"测试1失败: {e}")

        # 测试2: 缺少query参数
        print("\n测试2: 重排接口缺少query参数")
        try:
            response = self.session.post(
                f"{self.base_url}/v1/rerank",
                json={"documents": ["测试文档"]},
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            print(f"错误响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"测试2失败: {e}")

        # 测试3: 无效JSON
        print("\n测试3: 无效JSON格式")
        try:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            print(f"错误响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"测试3失败: {e}")

        return {}

    def test_batch_embedding(self) -> Dict[str, Any]:
        """测试批量文本嵌入"""
        print(f"\n=== 测试批量文本嵌入 ===")

        texts = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的核心技术",
            "深度学习使用多层神经网络",
            "自然语言处理帮助计算机理解人类语言",
            "计算机视觉使机器能够理解和分析图像"
        ]

        payload = {
            "input": texts,
            "model": "qwen3-embedding",
            "user": "batch_test"
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
            print(f"输入文本数量: {len(texts)}")

            if response.status_code == 200:
                result = response.json()
                print(f"返回向量数量: {len(result.get('data', []))}")
                print(f"总token使用量: {result.get('usage', {}).get('total_tokens', 0)}")

                # 验证每个向量都有正确的索引
                indices = [item.get('index') for item in result.get('data', [])]
                print(f"向量索引: {indices}")
                print(f"索引连续性: {'✓' if indices == list(range(len(texts))) else '✗'}")

                return result
            else:
                print(f"错误响应: {response.text}")
                return {}

        except Exception as e:
            print(f"批量嵌入测试失败: {e}")
            return {}

    def test_large_rerank(self) -> Dict[str, Any]:
        """测试大量文档重排序"""
        print(f"\n=== 测试大量文档重排序 ===")

        query = "什么是机器学习算法"

        # 生成测试文档
        documents = [
            "机器学习是人工智能的一个重要分支",
            "监督学习使用标记的训练数据来训练模型",
            "无监督学习从未标记的数据中发现模式",
            "强化学习通过与环境交互来学习最优策略",
            "深度学习是机器学习的一个子领域",
            "神经网络是深度学习的基础",
            "决策树是一种常用的机器学习算法",
            "支持向量机在分类任务中表现良好",
            "随机森林是多个决策树的集成",
            "梯度提升是一种强大的集成学习方法",
            "线性回归是基本的回归算法",
            "逻辑回归用于二分类问题",
            "聚类算法将相似的数据分组",
            "主成分分析用于降维",
            "自然语言处理是AI的应用领域"
        ]

        payload = {
            "query": query,
            "documents": documents,
            "model": "qwen3-reranker",
            "top_n": 5,
            "user": "large_test"
        }

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
            print(f"文档数量: {len(documents)}")
            print(f"返回结果数量: 5")

            if response.status_code == 200:
                result = response.json()
                print(f"总token使用量: {result.get('usage', {}).get('total_tokens', 0)}")
                print("Top 5 结果:")
                for i, item in enumerate(result.get('data', [])):
                    original_idx = item.get('index')
                    score = item.get('relevance_score', 0)
                    doc = item.get('document', '')
                    print(f"  {i + 1}. [原索引:{original_idx}] [分数:{score:.4f}] {doc[:50]}...")

                return result
            else:
                print(f"错误响应: {response.text}")
                return {}

        except Exception as e:
            print(f"大量文档重排序测试失败: {e}")
            return {}

    def run_comprehensive_tests(self):
        """运行全面的接口测试"""
        print("Qwen3 OpenAI兼容接口全面测试开始")
        print("=" * 60)

        # 1. 健康检查
        health = self.test_health_check()
        if health.get("status") != "healthy":
            print("服务状态不健康，终止测试")
            return

        # 2. 基础功能测试
        print("\n" + "=" * 40)
        print("基础功能测试")
        print("=" * 40)

        # 基础嵌入测试
        self.test_openai_embeddings(["Hello World", "你好世界"])

        # 基础重排序测试
        query = "什么是人工智能"
        documents = [
            "人工智能是指由人制造出来的机器所表现出来的智能",
            "机器学习是人工智能的一个子领域",
            "今天天气晴朗适合外出游玩",
            "深度学习使用神经网络来模拟人脑的工作方式"
        ]
        self.test_openai_rerank(query, documents, top_n=3)

        # 3. 高级功能测试
        print("\n" + "=" * 40)
        print("高级功能测试")
        print("=" * 40)

        # 指定维度测试
        self.test_embedding_dimensions()

        # 批量嵌入测试
        self.test_batch_embedding()

        # 大量文档重排序测试
        self.test_large_rerank()

        # 4. 错误处理测试
        print("\n" + "=" * 40)
        print("错误处理测试")
        print("=" * 40)

        self.test_error_handling()

        print("\n" + "=" * 60)
        print("所有测试完成")

    def run_basic_tests(self):
        """运行基础测试"""
        print("Qwen3 OpenAI兼容接口基础测试开始")
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
        print("基础测试完成")


if __name__ == "__main__":
    """
    # 运行基础测试
    python test_api.py

    # 运行全面测试
    python test_api.py --comprehensive
    """

    # 创建测试客户端
    client = QwenOpenAITestClient("http://localhost:8000")

    # 选择测试模式
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        client.run_comprehensive_tests()
    else:
        client.run_basic_tests()
