"""
RAG流水线使用示例

演示如何使用不同类型的RAG流水线进行查询处理。
"""

import asyncio
from typing import Dict, Any, List

from .factory import PipelineFactory
from .interfaces.pipeline_interface import (
    QueryRequest,
    GenerationStrategy,
    QueryType,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.pipeline.example_usage")


async def standard_pipeline_example():
    """标准流水线示例"""
    print("=== 标准RAG流水线示例 ===")

    # 创建标准流水线
    pipeline = PipelineFactory.create_standard_pipeline()
    await pipeline.initialize()

    # 添加示例文档
    sample_documents = [
        {
            "id": "doc1",
            "content": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "metadata": {"source": "技术文档", "category": "AI", "author": "技术专家"},
        },
        {
            "id": "doc2",
            "content": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习和改进的算法，而无需明确编程。",
            "metadata": {"source": "技术文档", "category": "ML", "author": "技术专家"},
        },
        {
            "id": "doc3",
            "content": "深度学习是机器学习的一个子集，使用具有多个层的神经网络来学习数据的复杂表示。",
            "metadata": {"source": "技术文档", "category": "DL", "author": "技术专家"},
        },
        {
            "id": "doc4",
            "content": "自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
            "metadata": {"source": "技术文档", "category": "NLP", "author": "技术专家"},
        },
    ]

    # 为检索器添加文档
    if pipeline.retriever:
        for strategy in pipeline.retriever.retrievers.keys():
            retriever = pipeline.retriever.retrievers[strategy]
            if hasattr(retriever, 'add_documents'):
                await retriever.add_documents(sample_documents)
                print(f"为策略 {strategy.value} 添加了 {len(sample_documents)} 个文档")

    # 处理查询
    query_request = QueryRequest(
        query="什么是机器学习和深度学习的关系？",
        generation_strategy=GenerationStrategy.DETAILED,
        max_context_length=2000,
        enable_citations=True,
    )

    result = await pipeline.process(query_request)

    print(f"查询: {result.query}")
    print(f"处理时间: {result.total_processing_time_ms:.2f}ms")
    print(f"查询类型: {result.query_understanding.query_type}")
    print(f"查询意图: {result.query_understanding.query_intent}")
    print(f"检索到文档数: {len(result.retrieval_results.chunks)}")
    print(f"上下文文档数: {len(result.context.documents)}")
    print(f"置信度: {result.answer.confidence:.2f}")
    print(f"答案长度: {len(result.answer.content)}")
    print(f"引用数量: {len(result.answer.citations)}")

    print(f"\n答案:")
    print(result.answer.content)

    if result.answer.citations:
        print(f"\n引用:")
        for citation in result.answer.citations:
            print(f"- {citation}")

    await pipeline.cleanup()


async def fast_pipeline_example():
    """快速流水线示例"""
    print("\n=== 快速RAG流水线示例 ===")

    # 创建快速流水线
    pipeline = PipelineFactory.create_fast_pipeline()
    await pipeline.initialize()

    # 添加测试文档
    test_documents = [
        {"id": "fast1", "content": "Python是一种高级编程语言，以其简洁易读的语法而闻名。"},
        {"id": "fast2", "content": "JavaScript是Web开发中最常用的编程语言之一。"},
        {"id": "fast3", "content": "Java是一种面向对象的编程语言，广泛用于企业级应用开发。"},
    ]

    if pipeline.retriever:
        for strategy in pipeline.retriever.retrievers.keys():
            retriever = pipeline.retriever.retrievers[strategy]
            if hasattr(retriever, 'add_documents'):
                await retriever.add_documents(test_documents)

    # 处理查询
    query_request = QueryRequest(
        query="Python有什么特点？",
        generation_strategy=GenerationStrategy.CONCISE,
    )

    start_time = asyncio.get_event_loop().time()
    result = await pipeline.process(query_request)
    end_time = asyncio.get_event_loop().time()

    print(f"响应时间: {(end_time - start_time) * 1000:.2f}ms")
    print(f"答案: {result.answer.content}")

    await pipeline.cleanup()


async def conversational_pipeline_example():
    """对话式流水线示例"""
    print("\n=== 对话式RAG流水线示例 ===")

    # 创建对话式流水线
    pipeline = PipelineFactory.create_conversational_pipeline()
    await pipeline.initialize()

    # 添加对话相关文档
    conversation_docs = [
        {
            "id": "conv1",
            "content": "ChatGPT是由OpenAI开发的大型语言模型，能够进行自然对话和回答各种问题。",
            "metadata": {"type": "AI助手", "company": "OpenAI"},
        },
        {
            "id": "conv2",
            "content": "Claude是Anthropic开发的AI助手，专注于有用、无害和诚实的对话。",
            "metadata": {"type": "AI助手", "company": "Anthropic"},
        },
        {
            "id": "conv3",
            "content": "Gemini是Google开发的多模态AI模型，能够处理文本、图像和音频。",
            "metadata": {"type": "AI助手", "company": "Google"},
        },
    ]

    if pipeline.retriever:
        for strategy in pipeline.retriever.retrievers.keys():
            retriever = pipeline.retriever.retrievers[strategy]
            if hasattr(retriever, 'add_documents'):
                await retriever.add_documents(conversation_docs)

    # 带对话历史的查询
    query_request = QueryRequest(
        query="哪个AI助手最适合学术研究？",
        generation_strategy=GenerationStrategy.CONVERSATIONAL,
        conversation_history=[
            {"role": "user", "content": "我想了解不同的AI助手"},
            {"role": "assistant", "content": "目前主要有ChatGPT、Claude和Gemini等AI助手"},
        ],
    )

    result = await pipeline.process(query_request)

    print(f"对话式回答:")
    print(result.answer.content)

    await pipeline.cleanup()


async def comprehensive_pipeline_example():
    """全面流水线示例"""
    print("\n=== 全面RAG流水线示例 ===")

    # 创建全面流水线
    pipeline = PipelineFactory.create_comprehensive_pipeline()
    await pipeline.initialize()

    # 添加研究相关文档
    research_docs = [
        {
            "id": "research1",
            "content": "Transformer架构是现代自然语言处理的基础，由Vaswani等人在2017年提出，完全基于注意力机制。",
            "metadata": {"type": "研究论文", "year": 2017, "citations": 15000},
        },
        {
            "id": "research2",
            "content": "BERT是Google提出的双向编码器表示Transformer，在多个NLP任务上取得了state-of-the-art的结果。",
            "metadata": {"type": "研究论文", "year": 2018, "citations": 12000},
        },
        {
            "id": "research3",
            "content": "GPT系列模型是OpenAI开发的生成式预训练Transformer，展示了强大的语言生成能力。",
            "metadata": {"type": "研究论文", "year": 2018, "citations": 8000},
        },
        {
            "id": "research4",
            "content": "T5模型是Google提出的Text-to-Text Transfer Transformer，将所有NLP任务统一为文本到文本的格式。",
            "metadata": {"type": "研究论文", "year": 2019, "citations": 5000},
        },
    ]

    if pipeline.retriever:
        for strategy in pipeline.retriever.retrievers.keys():
            retriever = pipeline.retriever.retrievers[strategy]
            if hasattr(retriever, 'add_documents'):
                await retriever.add_documents(research_docs)

    # 复杂查询
    query_request = QueryRequest(
        query="Transformer架构对现代NLP研究有什么影响？请详细分析其发展和应用。",
        generation_strategy=GenerationStrategy.STRUCTURED,
        max_context_length=4000,
    )

    result = await pipeline.process(query_request)

    print(f"结构化回答:")
    print(result.answer.content)

    if result.answer.sources:
        print(f"\n参考来源:")
        for source in result.answer.sources:
            print(f"- {source.get('title', '未知标题')} (相关度: {source.get('relevance_score', 0):.2f})")

    await pipeline.cleanup()


async def streaming_pipeline_example():
    """流式流水线示例"""
    print("\n=== 流式RAG流水线示例 ===")

    # 创建流水线
    pipeline = PipelineFactory.create_standard_pipeline()
    await pipeline.initialize()

    # 添加测试文档
    stream_docs = [
        {"id": "stream1", "content": "流式处理允许逐步生成答案，提供更好的用户体验。"},
        {"id": "stream2", "content": "实时响应在对话系统中非常重要，可以减少等待时间。"},
    ]

    if pipeline.retriever:
        for strategy in pipeline.retriever.retrievers.keys():
            retriever = pipeline.retriever.retrievers[strategy]
            if hasattr(retriever, 'add_documents'):
                await retriever.add_documents(stream_docs)

    # 流式查询
    query_request = QueryRequest(
        query="流式处理有什么优势？",
        generation_strategy=GenerationStrategy.CONVERSATIONAL,
    )

    print("流式回答:")
    async for chunk in pipeline.process_stream(query_request):
        print(chunk, end="", flush=True)
    print()  # 换行

    await pipeline.cleanup()


async def custom_pipeline_example():
    """自定义流水线示例"""
    print("\n=== 自定义RAG流水线示例 ===")

    # 根据需求创建自定义流水线
    requirements = {
        "use_cases": ["real_time", "conversation"],
        "performance_priority": "speed",
        "domain": "conversational",
        "constraints": {
            "max_latency_ms": 3000,  # 3秒内响应
            "max_memory_mb": 256,     # 低内存使用
        }
    }

    pipeline = PipelineFactory.create_custom_pipeline(requirements)
    await pipeline.initialize()

    print(f"自定义流水线配置:")
    stats = await pipeline.get_pipeline_statistics()
    print(f"- 成功率: {stats['success_rate']:.2%}")
    print(f"- 总请求数: {stats['total_requests']}")

    await pipeline.cleanup()


async def batch_processing_example():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")

    pipeline = PipelineFactory.create_standard_pipeline()
    await pipeline.initialize()

    # 添加测试文档
    batch_docs = [
        {"id": "batch1", "content": "异步编程允许同时处理多个任务。"},
        {"id": "batch2", "content": "批量处理可以提高系统效率。"},
        {"id": "batch3", "content": "并发控制是多线程编程的重要概念。"},
    ]

    if pipeline.retriever:
        for strategy in pipeline.retriever.retrievers.keys():
            retriever = pipeline.retriever.retrievers[strategy]
            if hasattr(retriever, 'add_documents'):
                await retriever.add_documents(batch_docs)

    # 批量查询
    queries = [
        QueryRequest(query="什么是异步编程？"),
        QueryRequest(query="批量处理有什么好处？"),
        QueryRequest(query="如何实现并发控制？"),
    ]

    start_time = asyncio.get_event_loop().time()
    results = await pipeline.batch_process(queries)
    end_time = asyncio.get_event_loop().time()

    print(f"批量处理完成，总时间: {(end_time - start_time) * 1000:.2f}ms")
    print(f"处理查询数: {len(results)}")

    for i, result in enumerate(results):
        print(f"\n查询 {i+1}: {result.query}")
        print(f"成功: {result.success}")
        if result.success:
            print(f"答案: {result.answer.content[:100]}...")

    await pipeline.cleanup()


async def template_comparison_example():
    """模板比较示例"""
    print("\n=== 流水线模板比较 ===")

    # 获取所有可用模板
    templates = PipelineFactory.get_available_templates()
    print("可用的流水线模板:")
    for template in templates:
        print(f"- {template['name']}: {template['description']}")
        print(f"  适用场景: {', '.join(template['use_cases'])}")

    # 比较不同模板的性能特点
    template_names = ["standard", "fast", "comprehensive", "conversational"]

    for template_name in template_names:
        print(f"\n测试模板: {template_name}")

        pipeline = PipelineFactory.create_from_template(template_name)
        await pipeline.initialize()

        # 简单测试
        test_query = QueryRequest(query="什么是AI？")
        start_time = asyncio.get_event_loop().time()
        result = await pipeline.process(test_query)
        end_time = asyncio.get_event_loop().time()

        print(f"- 响应时间: {(end_time - start_time) * 1000:.2f}ms")
        print(f"- 答案长度: {len(result.answer.content)}")
        print(f"- 检索文档数: {len(result.context.documents)}")

        await pipeline.cleanup()


async def configuration_validation_example():
    """配置验证示例"""
    print("\n=== 配置验证示例 ===")

    # 有效配置
    valid_config = {
        "retrieval_strategies": ["vector", "bm25"],
        "max_retrieval_results": 10,
        "min_relevance_score": 0.3,
        "max_context_tokens": 3000,
        "generation_model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "enable_caching": True,
    }

    errors = PipelineFactory.validate_config(valid_config)
    if errors:
        print(f"配置验证错误: {errors}")
    else:
        print("有效配置验证通过")

    # 无效配置
    invalid_config = {
        "retrieval_strategies": ["invalid_strategy"],
        "max_retrieval_results": -5,
        "temperature": 3.0,  # 超出范围
        "enable_caching": "yes",  # 应该是布尔值
    }

    errors = PipelineFactory.validate_config(invalid_config)
    print(f"无效配置错误: {errors}")

    # 获取优化建议
    suggestions = PipelineFactory.get_optimization_suggestions(valid_config)
    print(f"优化建议: {suggestions}")


async def main():
    """主函数，运行所有示例"""
    print("RAG流水线使用示例")
    print("=" * 50)

    try:
        await standard_pipeline_example()
        await fast_pipeline_example()
        await conversational_pipeline_example()
        await comprehensive_pipeline_example()
        await streaming_pipeline_example()
        await custom_pipeline_example()
        await batch_processing_example()
        await template_comparison_example()
        await configuration_validation_example()

        print("\n所有示例运行完成！")

    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())