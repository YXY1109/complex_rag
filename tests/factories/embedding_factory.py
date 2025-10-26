"""
嵌入向量数据工厂
生成测试用的嵌入向量数据
"""
from factory import Factory, Faker, SubFactory, lazy_attribute, List
from typing import Dict, Any, List as ListType
import random
import uuid


class EmbeddingRequestFactory(Factory):
    """嵌入向量请求工厂"""

    class Meta:
        model = dict

    input = List([Faker("sentence", nb_words=10)])
    model = Faker("random_element", elements=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"])
    encoding_format = "float"
    dimensions = Faker("random_int", min=128, max=3072)
    user = Faker("uuid4")

    @lazy_attribute
    def input(self):
        """生成输入文本"""
        input_count = Faker("random_int", min=1, max=5)
        return [Faker("text", max_nb_chars=8000) for _ in range(input_count)]

    @lazy_attribute
    def dimensions(self):
        """根据模型确定维度"""
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return model_dimensions.get(self.model, 1536)


class EmbeddingDataFactory(Factory):
    """嵌入向量数据工厂"""

    class Meta:
        model = dict

    object = "embedding"
    embedding = None  # 将在lazy_attribute中生成
    index = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dimension = kwargs.get("dimension", 1536)

    @lazy_attribute
    def embedding(self):
        """生成嵌入向量"""
        # 生成指定维度的向量
        dimension = getattr(self, "dimension", 1536)
        return [round(random.uniform(-1.0, 1.0), 6) for _ in range(dimension)]


class EmbeddingResponseFactory(Factory):
    """嵌入向量响应工厂"""

    class Meta:
        model = dict

    object = "list"
    data = None  # 将在lazy_attribute中生成
    model = Faker("random_element", elements=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"])
    usage = None  # 将在lazy_attribute中生成

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_count = kwargs.get("input_count", 1)
        self.dimension = kwargs.get("dimension", 1536)

    @lazy_attribute
    def data(self):
        """生成嵌入向量数据"""
        data = []
        for i in range(self.input_count):
            embedding_data = EmbeddingDataFactory(
                index=i,
                dimension=self.dimension
            )
            data.append(embedding_data)
        return data

    @lazy_attribute
    def usage(self):
        """生成使用情况"""
        prompt_tokens = sum(
            len(Faker("text", max_nb_chars=8000).split())
            for _ in range(self.input_count)
        )
        return {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }


class EmbeddingUsageFactory(Factory):
    """嵌入向量使用情况工厂"""

    class Meta:
        model = dict

    prompt_tokens = Faker("random_int", min=1, max=10000)
    total_tokens = Faker("random_int", min=1, max=10000)

    @lazy_attribute
    def cost(self):
        """计算费用（基于token数量）"""
        # 模拟定价：$0.0001 per 1K tokens for ada-002
        cost_per_1k_tokens = 0.0001
        return round((self.total_tokens / 1000) * cost_per_1k_tokens, 6)


class EmbeddingComparisonFactory(Factory):
    """嵌入向量比较工厂"""

    class Meta:
        model = dict

    text1 = Faker("sentence", nb_words=15)
    text2 = Faker("sentence", nb_words=15)
    embedding1 = None  # 将在lazy_attribute中生成
    embedding2 = None  # 将在lazy_attribute中生成
    similarity_score = None  # 将在lazy_attribute中生成

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dimension = kwargs.get("dimension", 1536)

    @lazy_attribute
    def embedding1(self):
        """生成第一个嵌入向量"""
        return [round(random.uniform(-1.0, 1.0), 6) for _ in range(self.dimension)]

    @lazy_attribute
    def embedding2(self):
        """生成第二个嵌入向量"""
        return [round(random.uniform(-1.0, 1.0), 6) for _ in range(self.dimension)]

    @lazy_attribute
    def similarity_score(self):
        """计算相似度分数"""
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(self.embedding1, self.embedding2))
        norm1 = sum(a * a for a in self.embedding1) ** 0.5
        norm2 = sum(b * b for b in self.embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return round(similarity, 6)


class EmbeddingBatchFactory(Factory):
    """批量嵌入向量工厂"""

    class Meta:
        model = dict

    batch_id = Faker("uuid4")
    texts = None  # 将在lazy_attribute中生成
    embeddings = None  # 将在lazy_attribute中生成
    model = Faker("random_element", elements=["text-embedding-ada-002", "text-embedding-3-small"])
    created_at = Faker("date_time_this_month")
    processing_time = Faker("random_int", min=100, max=5000)  # 毫秒

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_count = kwargs.get("text_count", 10)
        self.dimension = kwargs.get("dimension", 1536)

    @lazy_attribute
    def texts(self):
        """生成文本列表"""
        return [Faker("text", max_nb_chars=1000) for _ in range(self.text_count)]

    @lazy_attribute
    def embeddings(self):
        """生成嵌入向量列表"""
        return [
            [round(random.uniform(-1.0, 1.0), 6) for _ in range(self.dimension)]
            for _ in range(self.text_count)
        ]

    @lazy_attribute
    def metadata(self):
        """生成元数据"""
        return {
            "total_tokens": sum(len(text.split()) for text in self.texts),
            "average_tokens_per_text": sum(len(text.split()) for text in self.texts) / len(self.texts),
            "batch_size": len(self.texts),
            "model_version": "1.0",
            "quality_score": round(Faker("random.uniform", min=0.8, max=1.0), 3)
        }


class EmbeddingSearchResultFactory(Factory):
    """嵌入向量搜索结果工厂"""

    class Meta:
        model = dict

    query = Faker("sentence", nb_words=10)
    query_embedding = None  # 将在lazy_attribute中生成
    results = None  # 将在lazy_attribute中生成
    search_time = Faker("random_int", min=10, max=1000)  # 毫秒
    total_results = Faker("random_int", min=0, max=100)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dimension = kwargs.get("dimension", 1536)
        self.result_count = kwargs.get("result_count", 5)

    @lazy_attribute
    def query_embedding(self):
        """生成查询嵌入向量"""
        return [round(random.uniform(-1.0, 1.0), 6) for _ in range(self.dimension)]

    @lazy_attribute
    def results(self):
        """生成搜索结果"""
        results = []
        for i in range(self.result_count):
            result = {
                "id": str(uuid.uuid4()),
                "text": Faker("text", max_nb_chars=500),
                "embedding": [round(random.uniform(-1.0, 1.0), 6) for _ in range(self.dimension)],
                "score": round(Faker("random.uniform", min=0.7, max=1.0), 6),
                "metadata": {
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "source": Faker("word"),
                    "created_at": Faker("date_time_this_year")
                }
            }
            results.append(result)
        return results


# 便捷函数
def create_test_embedding_request(**overrides) -> Dict[str, Any]:
    """创建测试嵌入向量请求"""
    return EmbeddingRequestFactory(**overrides)


def create_test_embedding_response(input_count: int = 1, **overrides) -> Dict[str, Any]:
    """创建测试嵌入向量响应"""
    return EmbeddingResponseFactory(input_count=input_count, **overrides)


def create_test_embedding_dimension(dimension: int = 1536, **overrides) -> Dict[str, Any]:
    """创建测试嵌入向量数据"""
    return EmbeddingDataFactory(dimension=dimension, **overrides)


def create_test_embedding_comparison(**overrides) -> Dict[str, Any]:
    """创建测试嵌入向量比较"""
    return EmbeddingComparisonFactory(**overrides)


def create_test_embedding_batch(text_count: int = 10, **overrides) -> Dict[str, Any]:
    """创建测试批量嵌入向量"""
    return EmbeddingBatchFactory(text_count=text_count, **overrides)


def create_test_embedding_search_results(result_count: int = 5, **overrides) -> Dict[str, Any]:
    """创建测试嵌入向量搜索结果"""
    return EmbeddingSearchResultFactory(result_count=result_count, **overrides)


def normalize_vector(vector: ListType[float]) -> ListType[float]:
    """归一化向量"""
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def calculate_cosine_similarity(vec1: ListType[float], vec2: ListType[float]) -> float:
    """计算余弦相似度"""
    if len(vec1) != len(vec2):
        raise ValueError("向量维度不匹配")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)