"""
GraphRAG使用示例

演示如何使用GraphRAG组件进行实体抽取、解析和社区发现。
"""

import asyncio
from typing import Dict, Any

from .config import ConfigManager, GraphRAGMode, ProcessingLevel
from .graph_rag_integration import GraphRAGFactory
from .interfaces.entity_interface import EntityType, EntityExtractionRequest
from .infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.graph_rag.example_usage")


async def basic_entity_extraction_example():
    """基础实体抽取示例"""
    print("=== 基础实体抽取示例 ===")

    # 创建Light模式处理器
    config = {
        "mode": "light",
        "processing_level": "basic",
        "entity_extraction": {
            "confidence_threshold": 0.6,
            "max_entities": 20,
        },
    }

    processor = GraphRAGFactory.create_processor(config)
    await processor.initialize()

    # 示例文本
    text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    Tim Cook serves as the CEO of Apple, while Steve Jobs was the co-founder.
    The company was founded on April 1, 1976, and has a market capitalization of over $2 trillion.
    Microsoft and Google are major competitors in the technology sector.
    """

    # 处理文档
    result = await processor.process_document(text, document_id="example_doc_1")

    print(f"抽取的实体数量: {len(result['entities'])}")
    for entity in result['entities']:
        print(f"- {entity.name} ({entity.type.value}, 置信度: {entity.confidence:.2f})")

    await processor.cleanup()


async def enhanced_entity_processing_example():
    """增强实体处理示例（包含实体解析）"""
    print("\n=== 增强实体处理示例 ===")

    # 创建General模式处理器
    config = {
        "mode": "general",
        "processing_level": "enhanced",
        "entity_extraction": {
            "confidence_threshold": 0.5,
            "max_entities": 50,
        },
        "entity_resolution": {
            "similarity_threshold": 0.8,
        },
    }

    processor = GraphRAGFactory.create_processor(config)
    await processor.initialize()

    # 示例文本（包含重复和相似实体）
    text = """
    Apple Inc. is a technology company based in Cupertino, California.
    Apple designs and develops consumer electronics. Apple's main products include iPhone.
    Tim Cook is the CEO of Apple. Mr. Cook joined Apple in 1998.
    Google is another technology company. Google is headquartered in Mountain View, California.
    """

    # 处理文档
    result = await processor.process_document(text, document_id="example_doc_2")

    print(f"解析后的实体数量: {len(result['entities'])}")
    print(f"合并的实体数量: {len(result.get('merged_entities', []))}")

    for entity in result['entities']:
        print(f"- {entity.name} ({entity.type.value}, 置信度: {entity.confidence:.2f})")

    # 显示合并信息
    if result.get('resolution_mappings'):
        print(f"实体映射: {result['resolution_mappings']}")

    await processor.cleanup()


async def full_graph_processing_example():
    """完整图处理示例（包含社区发现）"""
    print("\n=== 完整图处理示例 ===")

    # 创建General模式处理器，完整处理
    config = {
        "mode": "general",
        "processing_level": "full",
        "entity_extraction": {
            "confidence_threshold": 0.5,
            "max_entities": 100,
            "enable_relationship_extraction": True,
        },
        "entity_resolution": {
            "similarity_threshold": 0.8,
        },
        "community_detection": {
            "algorithm": "leiden",
            "min_community_size": 2,
        },
    }

    processor = GraphRAGFactory.create_processor(config)
    await processor.initialize()

    # 更复杂的示例文本
    text = """
    Apple Inc. is an American technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs and Steve Wozniak in 1976.
    Tim Cook currently serves as the CEO of Apple, while Jeff Williams is the COO.
    Apple's main products include iPhone, iPad, and MacBook.

    Microsoft Corporation is another major technology company based in Redmond, Washington.
    Satya Nadella is the CEO of Microsoft, while Bill Gates was the co-founder.
    Microsoft's main products include Windows and Office.

    Google LLC is a technology company headquartered in Mountain View, California.
    Sundar Pichai serves as the CEO of Google, which was founded by Larry Page and Sergey Brin.
    Google's main products include Google Search and Android.

    These three companies are major competitors in the technology industry and are all based in the United States.
    """

    # 处理文档
    result = await processor.process_document(text, document_id="example_doc_3")

    print(f"实体数量: {len(result['entities'])}")
    print(f"关系数量: {len(result['relationships'])}")
    print(f"社区数量: {len(result.get('communities', []))}")

    # 显示社区信息
    if result.get('communities'):
        for community in result['communities']:
            print(f"\n社区: {community.name}")
            print(f"  大小: {community.size}")
            print(f"  描述: {community.description}")
            print(f"  实体: {[e.name for e in result['entities'] if e.id in community.entities]}")

    # 显示关系信息
    if result['relationships']:
        print(f"\n关系:")
        for rel in result['relationships'][:5]:  # 显示前5个关系
            source_name = next((e.name for e in result['entities'] if e.id == rel.source_entity_id), "Unknown")
            target_name = next((e.name for e in result['entities'] if e.id == rel.target_entity_id), "Unknown")
            print(f"- {source_name} -> {target_name} ({rel.relationship_type}, 置信度: {rel.confidence:.2f})")

    await processor.cleanup()


async def query_processing_example():
    """查询处理示例"""
    print("\n=== 查询处理示例 ===")

    # 创建处理器
    config = ConfigManager.get_default_config()
    processor = GraphRAGFactory.create_processor(config.__dict__)
    await processor.initialize()

    # 处理一些文档建立上下文
    context_text = """
    Apple Inc. is a technology company that produces iPhone and iPad.
    Tim Cook is the CEO of Apple.
    Microsoft produces Windows and Office.
    Satya Nadella is the CEO of Microsoft.
    """

    context_result = await processor.process_document(context_text, document_id="context_doc")
    context_entities = context_result['entities']

    # 处理用户查询
    query = "Who is the CEO of Apple and what products do they make?"
    query_result = await processor.process_query(
        query,
        context_entities=context_entities[:5],  # 限制上下文实体数量
        max_context_entities=10
    )

    print(f"查询: {query}")
    print(f"查询中的实体: {[e.name for e in query_result['query_entities']]}")
    print(f"总上下文实体: {[e.name for e in query_result['all_entities']]}")

    await processor.cleanup()


async def configuration_example():
    """配置示例"""
    print("\n=== 配置示例 ===")

    # 使用配置管理器创建不同环境的配置
    dev_config = ConfigManager.create_config(
        mode=GraphRAGMode.LIGHT,
        environment="development"
    )
    print(f"开发环境配置 - 模式: {dev_config.mode}, 缓存: {dev_config.enable_caching}")

    prod_config = ConfigManager.create_config(
        mode=GraphRAGMode.GENERAL,
        environment="production"
    )
    print(f"生产环境配置 - 模式: {prod_config.mode}, 缓存: {prod_config.enable_caching}")

    # 自定义配置
    custom_config = ConfigManager.create_config(
        mode=GraphRAGMode.GENERAL,
        custom_overrides={
            "entity_extraction": {
                "confidence_threshold": 0.8,
                "max_entities": 200,
            },
            "community_detection": {
                "algorithm": "louvain",
                "min_community_size": 5,
            },
        }
    )
    print(f"自定义配置 - 置信度阈值: {custom_config.entity_extraction.confidence_threshold}")

    # 验证配置
    errors = ConfigManager.validate_config(custom_config)
    if errors:
        print(f"配置验证错误: {errors}")
    else:
        print("配置验证通过")


async def performance_comparison_example():
    """性能比较示例"""
    print("\n=== 性能比较示例 ===")

    # 测试文本
    test_text = """
    """ * 100  # 重复文本以增加处理量
    Apple Inc. and Google are technology companies. Apple is led by Tim Cook and Google by Sundar Pichai.
    """ * 100

    # 测试Light模式
    light_config = {"mode": "light", "processing_level": "basic"}
    light_processor = GraphRAGFactory.create_processor(light_config)
    await light_processor.initialize()

    import time
    start_time = time.time()
    light_result = await light_processor.process_document(test_text, document_id="perf_test_light")
    light_time = time.time() - start_time

    print(f"Light模式处理时间: {light_time:.2f}秒, 实体数量: {len(light_result['entities'])}")

    await light_processor.cleanup()

    # 测试General模式
    general_config = {"mode": "general", "processing_level": "enhanced"}
    general_processor = GraphRAGFactory.create_processor(general_config)
    await general_processor.initialize()

    start_time = time.time()
    general_result = await general_processor.process_document(test_text, document_id="perf_test_general")
    general_time = time.time() - start_time

    print(f"General模式处理时间: {general_time:.2f}秒, 实体数量: {len(general_result['entities'])}")

    await general_processor.cleanup()

    # 性能比较
    if general_time > 0:
        speedup = general_time / light_time
        print(f"Light模式比General模式快 {speedup:.1f} 倍")


async def main():
    """主函数，运行所有示例"""
    print("GraphRAG 使用示例")
    print("=" * 50)

    try:
        await basic_entity_extraction_example()
        await enhanced_entity_processing_example()
        await full_graph_processing_example()
        await query_processing_example()
        await configuration_example()
        await performance_comparison_example()

        print("\n所有示例运行完成！")

    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())