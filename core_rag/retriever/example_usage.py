"""
多策略检索器使用示例

演示如何使用不同类型的检索器和多策略组合。
"""

import asyncio
from typing import Dict, Any, List

from .factory import RetrieverFactory
from .interfaces.retriever_interface import (
    RetrievalQuery,
    RetrievalStrategy,
    RetrievalMode,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.retriever.example_usage")


async def single_vector_retriever_example():
    """单一向量检索器示例"""
    print("=== 单一向量检索器示例 ===")

    # 创建向量检索器
    config = {
        "top_k": 5,
        "min_score": 0.3,
        "enable_caching": True,
    }

    retriever = RetrieverFactory.create_retriever(RetrievalStrategy.VECTOR, config)
    await retriever.initialize()

    # 添加示例文档
    documents = [
        {
            "id": "doc1",
            "content": "Apple Inc. is a technology company that designs and develops consumer electronics.",
            "metadata": {"source": "wiki", "category": "technology"},
        },
        {
            "id": "doc2",
            "content": "Google is a multinational technology company specializing in Internet-related services.",
            "metadata": {"source": "wiki", "category": "technology"},
        },
        {
            "id": "doc3",
            "content": "Microsoft Corporation is an American multinational technology company.",
            "metadata": {"source": "wiki", "category": "technology"},
        },
    ]

    doc_ids = await retriever.add_documents(documents)
    print(f"添加了 {len(doc_ids)} 个文档")

    # 执行检索
    query = RetrievalQuery(
        text="technology companies",
        top_k=3,
        strategy=RetrievalStrategy.VECTOR
    )

    result = await retriever.retrieve(query)
    print(f"检索到 {len(result.chunks)} 个结果")
    for i, chunk in enumerate(result.chunks):
        print(f"  {i+1}. {chunk.id} (分数: {chunk.score:.3f})")
        print(f"     内容: {chunk.content[:100]}...")

    await retriever.cleanup()


async def single_keyword_retriever_example():
    """单一关键词检索器示例"""
    print("\n=== 单一关键词检索器示例 ===")

    # 创建BM25检索器
    config = {
        "top_k": 5,
        "min_score": 0.1,
        "enable_caching": True,
    }

    retriever = RetrieverFactory.create_retriever(RetrievalStrategy.BM25, config)
    await retriever.initialize()

    # 添加示例文档
    documents = [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        },
        {
            "id": "doc2",
            "content": "Deep learning uses neural networks with multiple layers to learn from data.",
        },
        {
            "id": "doc3",
            "content": "Natural language processing helps computers understand human language.",
        },
        {
            "id": "doc4",
            "content": "Computer vision enables machines to interpret and understand visual information.",
        },
    ]

    doc_ids = await retriever.add_documents(documents)
    print(f"添加了 {len(doc_ids)} 个文档")

    # 执行检索
    query = RetrievalQuery(
        text="machine learning algorithms",
        top_k=3,
        strategy=RetrievalStrategy.BM25
    )

    result = await retriever.retrieve(query)
    print(f"检索到 {len(result.chunks)} 个结果")
    for i, chunk in enumerate(result.chunks):
        print(f"  {i+1}. {chunk.id} (分数: {chunk.score:.3f})")
        print(f"     内容: {chunk.content[:100]}...")

    await retriever.cleanup()


async def multi_strategy_retriever_example():
    """多策略检索器示例"""
    print("\n=== 多策略检索器示例 ===")

    # 创建多策略检索器配置
    config = {
        "enable_adaptive": True,
        "enable_caching": True,
        "fusion": {
            "method": "weighted_sum",
            "top_k": 5,
        },
        "strategies": {
            "vector": {
                "enabled": True,
                "weight": 0.7,
                "config": {
                    "top_k": 10,
                    "min_score": 0.2,
                }
            },
            "bm25": {
                "enabled": True,
                "weight": 0.3,
                "config": {
                    "top_k": 10,
                    "min_score": 0.1,
                }
            },
        },
    }

    multi_retriever = RetrieverFactory.create_multi_strategy_retriever(config)
    await multi_retriever.initialize()

    # 添加示例文档
    documents = [
        {
            "id": "doc1",
            "content": "Artificial intelligence and machine learning are transforming technology industries.",
            "metadata": {"topic": "AI", "year": 2023},
        },
        {
            "id": "doc2",
            "content": "Deep learning models require large datasets and computational power.",
            "metadata": {"topic": "Deep Learning", "year": 2023},
        },
        {
            "id": "doc3",
            "content": "Natural language processing applications include chatbots and translation.",
            "metadata": {"topic": "NLP", "year": 2023},
        },
        {
            "id": "doc4",
            "content": "Computer vision technology is used in autonomous vehicles and medical imaging.",
            "metadata": {"topic": "Computer Vision", "year": 2023},
        },
        {
            "id": "doc5",
            "content": "Quantum computing promises to solve complex problems beyond classical computers.",
            "metadata": {"topic": "Quantum", "year": 2023},
        },
    ]

    # 需要为每个策略添加文档
    for strategy in multi_retriever.retrievers.keys():
        retriever = multi_retriever.retrievers[strategy]
        doc_ids = await retriever.add_documents(documents)
        print(f"为策略 {strategy.value} 添加了 {len(doc_ids)} 个文档")

    # 执行多策略检索
    query = RetrievalQuery(
        text="AI and machine learning applications",
        top_k=3,
        strategy=RetrievalStrategy.HYBRID,
        mode=RetrievalMode.MULTI
    )

    result = await multi_retriever.retrieve_multi_strategy(query)
    print(f"多策略检索完成")
    print(f"使用的策略: {list(result.results.keys())}")
    print(f"最佳策略: {result.best_strategy.value if result.best_strategy else 'None'}")
    print(f"融合结果数量: {len(result.combined_chunks)}")

    # 显示各策略的结果
    for strategy, strategy_result in result.results.items():
        print(f"\n策略 {strategy.value} 结果:")
        for i, chunk in enumerate(strategy_result.chunks[:2]):
            print(f"  {i+1}. {chunk.id} (分数: {chunk.score:.3f})")

    # 显示融合结果
    print(f"\n融合结果 (前3个):")
    for i, chunk in enumerate(result.combined_chunks[:3]):
        print(f"  {i+1}. {chunk.id} (融合分数: {chunk.score:.3f})")
        print(f"     内容: {chunk.content[:80]}...")

    await multi_retriever.cleanup()


async def adaptive_retriever_example():
    """自适应检索器示例"""
    print("\n=== 自适应检索器示例 ===")

    # 创建图增强检索器
    config = {
        "enable_adaptive": True,
        "fusion": {
            "method": "adaptive",
            "top_k": 3,
        },
    }

    retriever = RetrieverFactory.create_graph_enhanced_retriever(config)
    await retriever.initialize()

    # 添加示例文档（包含实体信息）
    documents = [
        {
            "id": "doc1",
            "content": "Tim Cook is the CEO of Apple Inc., which produces iPhone and MacBook.",
            "entities": [
                {"id": "entity1", "name": "Tim Cook", "type": "PERSON"},
                {"id": "entity2", "name": "Apple Inc.", "type": "ORGANIZATION"},
                {"id": "entity3", "name": "iPhone", "type": "PRODUCT"},
            ],
            "relationships": [
                {"id": "rel1", "source_entity_id": "entity1", "target_entity_id": "entity2", "relationship_type": "CEO_OF"},
                {"id": "rel2", "source_entity_id": "entity2", "target_entity_id": "entity3", "relationship_type": "PRODUCES"},
            ],
        },
        {
            "id": "doc2",
            "content": "Satya Nadella leads Microsoft, the company behind Windows and Office.",
            "entities": [
                {"id": "entity4", "name": "Satya Nadella", "type": "PERSON"},
                {"id": "entity5", "name": "Microsoft", "type": "ORGANIZATION"},
                {"id": "entity6", "name": "Windows", "type": "PRODUCT"},
            ],
            "relationships": [
                {"id": "rel3", "source_entity_id": "entity4", "target_entity_id": "entity5", "relationship_type": "LEADS"},
                {"id": "rel4", "source_entity_id": "entity5", "target_entity_id": "entity6", "relationship_type": "PRODUCES"},
            ],
        },
    ]

    # 为每个策略添加文档
    for strategy in retriever.retrievers.keys():
        strategy_retriever = retriever.retrievers[strategy]
        doc_ids = await strategy_retriever.add_documents(documents)
        print(f"为策略 {strategy.value} 添加了 {len(doc_ids)} 个文档")

    # 执行自适应检索
    query = RetrievalQuery(
        text="Who is the CEO of Apple?",
        top_k=2,
        strategy=RetrievalStrategy.HYBRID,
        mode=RetrievalMode.ADAPTIVE
    )

    result = await retriever.adaptive_retrieve(query)
    print(f"自适应检索完成")
    print(f"选择的策略: {list(result.results.keys())}")
    print(f"最佳策略: {result.best_strategy.value if result.best_strategy else 'None'}")

    # 显示结果
    for i, chunk in enumerate(result.combined_chunks):
        print(f"  {i+1}. {chunk.id} (分数: {chunk.score:.3f})")
        if hasattr(chunk, 'metadata') and 'entities' in chunk.metadata:
            entities = chunk.metadata['entities']
            print(f"     相关实体: {entities}")

    await retriever.cleanup()


async def batch_retrieval_example():
    """批量检索示例"""
    print("\n=== 批量检索示例 ===")

    # 创建混合检索器
    retriever = RetrieverFactory.create_hybrid_retriever()
    await retriever.initialize()

    # 添加文档
    documents = [
        {"id": "doc1", "content": "Python is a popular programming language for data science."},
        {"id": "doc2", "content": "JavaScript is widely used for web development."},
        {"id": "doc3", "content": "Java is commonly used for enterprise applications."},
        {"id": "doc4", "content": "Go is gaining popularity for cloud-native applications."},
        {"id": "doc5", "content": "Rust is known for its memory safety and performance."},
    ]

    for strategy in retriever.retrievers.keys():
        strategy_retriever = retriever.retrievers[strategy]
        await strategy_retriever.add_documents(documents)

    # 批量检索
    queries = [
        RetrievalQuery(text="programming languages", top_k=3),
        RetrievalQuery(text="web development", top_k=2),
        RetrievalQuery(text="data science tools", top_k=2),
    ]

    results = await retriever.retrieve_multi_strategy(queries[0])  # 混合检索器没有批量接口
    # 这里简化为单个检索，实际实现中可以扩展

    print(f"批量检索完成，处理了 {len(queries)} 个查询")
    for i, query in enumerate(queries):
        print(f"查询 {i+1}: {query.text}")
        # 这里需要为每个查询单独检索
        result = await retriever.retrieve_multi_strategy(query)
        print(f"  结果数量: {len(result.combined_chunks)}")

    await retriever.cleanup()


async def performance_comparison_example():
    """性能比较示例"""
    print("\n=== 性能比较示例 ===")

    # 创建不同类型的检索器
    vector_retriever = RetrieverFactory.create_retriever(RetrievalStrategy.VECTOR)
    keyword_retriever = RetrieverFactory.create_retriever(RetrievalStrategy.BM25)
    hybrid_retriever = RetrieverFactory.create_hybrid_retriever()

    # 初始化
    await vector_retriever.initialize()
    await keyword_retriever.initialize()
    await hybrid_retriever.initialize()

    # 添加相同的文档
    documents = [
        {"id": f"doc{i}", "content": f"This is document {i} about various topics and subjects."}
        for i in range(100)
    ]

    await vector_retriever.add_documents(documents)
    await keyword_retriever.add_documents(documents)
    for strategy in hybrid_retriever.retrievers.keys():
        await hybrid_retriever.retrievers[strategy].add_documents(documents)

    # 测试查询
    test_query = RetrievalQuery(text="document topics and subjects", top_k=10)

    import time

    # 测试向量检索器
    start_time = time.time()
    vector_result = await vector_retriever.retrieve(test_query)
    vector_time = time.time() - start_time

    # 测试关键词检索器
    start_time = time.time()
    keyword_result = await keyword_retriever.retrieve(test_query)
    keyword_time = time.time() - start_time

    # 测试混合检索器
    start_time = time.time()
    hybrid_result = await hybrid_retriever.retrieve_multi_strategy(test_query)
    hybrid_time = time.time() - start_time

    # 比较结果
    print(f"向量检索器: {vector_time:.3f}s, 结果数量: {len(vector_result.chunks)}")
    print(f"关键词检索器: {keyword_time:.3f}s, 结果数量: {len(keyword_result.chunks)}")
    print(f"混合检索器: {hybrid_time:.3f}s, 结果数量: {len(hybrid_result.combined_chunks)}")

    # 清理
    await vector_retriever.cleanup()
    await keyword_retriever.cleanup()
    await hybrid_retriever.cleanup()


async def configuration_examples():
    """配置示例"""
    print("\n=== 配置示例 ===")

    # 获取可用策略
    strategies = RetrieverFactory.get_available_strategies()
    print(f"可用策略: {[s.value for s in strategies]}")

    # 获取策略信息
    for strategy in strategies:
        info = RetrieverFactory.get_strategy_info(strategy)
        print(f"策略 {info['strategy']}: {info['description']}")

    # 从模板创建检索器
    vector_retriever = RetrieverFactory.create_retriever_from_template(
        "semantic_search",
        {"top_k": 8}
    )
    await vector_retriever.initialize()
    print("从模板创建语义搜索检索器成功")

    # 验证配置
    config = {
        "strategy": "vector",
        "top_k": 15,
        "min_score": 0.5,
        "invalid_field": "value"
    }
    errors = RetrieverFactory.validate_config(config)
    if errors:
        print(f"配置验证错误: {errors}")
    else:
        print("配置验证通过")

    # 获取优化建议
    suggestions = RetrieverFactory.get_optimization_suggestions(config)
    print(f"优化建议: {suggestions}")

    await vector_retriever.cleanup()


async def main():
    """主函数，运行所有示例"""
    print("多策略检索器使用示例")
    print("=" * 50)

    try:
        await single_vector_retriever_example()
        await single_keyword_retriever_example()
        await multi_strategy_retriever_example()
        await adaptive_retriever_example()
        await batch_retrieval_example()
        await performance_comparison_example()
        await configuration_examples()

        print("\n所有示例运行完成！")

    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())