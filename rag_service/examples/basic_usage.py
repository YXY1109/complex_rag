"""
RAG服务基本使用示例

演示如何使用统一RAG服务进行问答、聊天、知识库管理等操作。
"""

import asyncio
import json
import logging
from datetime import datetime

# 导入RAG服务
from ..services.unified_rag_service import UnifiedRAGService
from ..interfaces.rag_interface import RAGQuery, RAGConfig, RetrievalMode, GenerationMode


async def basic_qa_example():
    """基础问答示例。"""
    print("=== 基础问答示例 ===")

    # 配置
    config = {
        "vector_store": {
            "milvus": {
                "host": "localhost",
                "port": 19530
            },
            "elasticsearch": {
                "hosts": ["localhost:9200"]
            }
        },
        "embedding": {
            "models": {
                "text-embedding-ada-002": {
                    "model_name": "text-embedding-ada-002",
                    "model_type": "openai",
                    "api_key": "your-openai-key"
                }
            }
        },
        "generation": {
            "models": {
                "gpt-3.5-turbo": {
                    "model_name": "gpt-3.5-turbo",
                    "model_type": "openai",
                    "api_key": "your-openai-key"
                }
            }
        }
    }

    # 初始化服务
    rag_service = UnifiedRAGService(config)

    # 初始化
    success = await rag_service.initialize()
    if not success:
        print("RAG服务初始化失败")
        return

    print("RAG服务初始化成功")

    # 简单问答
    question = "什么是人工智能？"
    print(f"\n问题: {question}")

    result = await rag_service.simple_qa(
        question=question,
        user_id="demo_user",
        tenant_id="demo_tenant"
    )

    print(f"回答: {result['answer']}")
    print(f"来源数量: {len(result['sources'])}")

    # 清理
    await rag_service.cleanup()


async def knowledge_base_example():
    """知识库管理示例。"""
    print("\n=== 知识库管理示例 ===")

    # 配置（简化版本）
    config = {
        "knowledge_manager": {
            "max_kb_per_tenant": 10,
            "max_documents_per_kb": 1000
        }
    }

    # 初始化服务
    rag_service = UnifiedRAGService(config)
    success = await rag_service.initialize()
    if not success:
        print("RAG服务初始化失败")
        return

    # 创建知识库
    kb = await rag_service.create_knowledge_base(
        name="技术文档知识库",
        description="包含各种技术文档的知识库",
        tenant_id="demo_tenant",
        created_by="admin"
    )

    print(f"创建知识库成功: {kb.name} (ID: {kb.kb_id})")

    # 添加文档
    doc_id = await rag_service.add_document_to_kb(
        kb_id=kb.kb_id,
        title="Python编程基础",
        content="""
        Python是一种高级编程语言，由Guido van Rossum于1991年创建。
        Python具有简洁易读的语法，被广泛用于Web开发、数据科学、人工智能等领域。

        Python的主要特点包括：
        1. 简洁易读的语法
        2. 丰富的标准库
        3. 强大的社区支持
        4. 跨平台兼容性
        """,
        created_by="admin"
    )

    print(f"添加文档成功: {doc_id}")

    # 搜索文档
    search_results = await rag_service.search_documents(
        kb_id=kb.kb_id,
        query="Python的特点",
        top_k=5
    )

    print(f"搜索结果: {len(search_results)} 个相关文档")
    for result in search_results[:3]:
        print(f"  - {result['title']} (分数: {result['score']:.3f})")
        print(f"    {result['content'][:100]}...")

    # 清理
    await rag_service.cleanup()


async def chat_example():
    """聊天对话示例。"""
    print("\n=== 聊天对话示例 ===")

    # 配置
    config = {
        "chat": {
            "max_messages_per_session": 50,
            "max_conversation_history": 10
        }
    }

    # 初始化服务
    rag_service = UnifiedRAGService(config)
    success = await rag_service.initialize()
    if not success:
        print("RAG服务初始化失败")
        return

    # 创建会话
    session = await rag_service.create_session(
        user_id="demo_user",
        tenant_id="demo_tenant",
        title="Python学习对话"
    )

    print(f"创建聊天会话: {session.title}")

    # 多轮对话
    questions = [
        "Python是什么？",
        "Python有哪些主要应用领域？",
        "学习Python有什么建议？",
        "Python和Java有什么区别？"
    ]

    for question in questions:
        print(f"\n用户: {question}")

        result = await rag_service.chat(
            session_id=session.session_id,
            message=question
        )

        print(f"助手: {result.answer[:200]}...")
        print(f"引用来源: {len(result.retrieval_result.chunks)} 个")

    # 获取会话摘要
    summary = await rag_service.chat_service.get_session_summary(session.session_id)
    if summary:
        print(f"\n会话摘要: {summary.summary}")
        print(f"关键主题: {summary.key_topics}")

    # 清理
    await rag_service.cleanup()


async def advanced_rag_example():
    """高级RAG示例。"""
    print("\n=== 高级RAG示例 ===")

    # 配置
    config = {
        "rag": {
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "rag_engine": {
            "enable_reranking": True,
            "enable_context_optimization": True,
            "parallel_processing": True
        }
    }

    # 初始化服务
    rag_service = UnifiedRAGService(config)
    success = await rag_service.initialize()
    if not success:
        print("RAG服务初始化失败")
        return

    # 创建复杂的RAG查询
    rag_query = RAGQuery(
        query_id="advanced_example",
        query="请详细解释机器学习中的过拟合问题，包括产生原因、预防和解决方案",
        retrieval_mode=RetrievalMode.HYBRID,
        top_k=8,
        similarity_threshold=0.6,
        generation_mode=GenerationMode.CHAIN_OF_THOUGHT,
        temperature=0.8,
        max_tokens=1500,
        user_id="demo_user",
        tenant_id="demo_tenant"
    )

    print(f"查询: {rag_query.query}")
    print(f"检索模式: {rag_query.retrieval_mode.value}")
    print(f"生成模式: {rag_query.generation_mode.value}")

    # 执行查询
    result = await rag_service.query(rag_query)

    print(f"回答长度: {len(result.answer)} 字符")
    print(f"检索到文档: {len(result.retrieval_result.chunks)} 个")
    print(f"生成时间: {result.generation_result.generation_time:.3f}秒")
    print(f"检索时间: {result.retrieval_result.search_time:.3f}秒")
    print(f"总时间: {result.total_time:.3f}秒")

    # 显示详细元数据
    print("\n详细元数据:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")

    # 清理
    await rag_service.cleanup()


async def batch_query_example():
    """批量查询示例。"""
    print("\n=== 批量查询示例 ===")

    # 配置
    config = {
        "rag_engine": {
            "max_concurrent_queries": 3
        }
    }

    # 初始化服务
    rag_service = UnifiedRAGService(config)
    success = await rag_service.initialize()
    if not success:
        print("RAG服务初始化失败")
        return

    # 准备批量查询
    questions = [
        "什么是机器学习？",
        "深度学习和机器学习有什么区别？",
        "如何选择合适的机器学习算法？",
        "什么是神经网络？",
        "如何评估机器学习模型的性能？"
    ]

    # 创建RAG查询列表
    rag_queries = []
    for i, question in enumerate(questions):
        query = RAGQuery(
            query_id=f"batch_{i}",
            query=question,
            user_id="demo_user",
            tenant_id="demo_tenant"
        )
        rag_queries.append(query)

    print(f"准备执行 {len(rag_queries)} 个批量查询")

    # 执行批量查询
    start_time = datetime.now()
    results = await rag_service.batch_query(rag_queries, max_concurrent=3)
    end_time = datetime.now()

    total_time = (end_time - start_time).total_seconds()

    print(f"批量查询完成，耗时: {total_time:.3f}秒")
    print(f"成功: {sum(1 for r in results if r.success)} 个")
    print(f"失败: {sum(1 for r in results if not r.success)} 个")

    # 显示部分结果
    print("\n部分结果:")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. 问题: {rag_queries[i].query}")
        print(f"   回答: {result.answer[:100]}...")
        print(f"   状态: {'成功' if result.success else '失败'}")

    # 清理
    await rag_service.cleanup()


async def service_management_example():
    """服务管理示例。"""
    print("\n=== 服务管理示例 ===")

    # 配置
    config = {
        "generation": {
            "models": {
                "gpt-3.5-turbo": {
                    "model_name": "gpt-3.5-turbo",
                    "model_type": "openai",
                    "api_key": "your-openai-key",
                    "default": True
                }
            }
        }
    }

    # 初始化服务
    rag_service = UnifiedRAGService(config)
    success = await rag_service.initialize()
    if not success:
        print("RAG服务初始化失败")
        return

    # 获取服务状态
    status = await rag_service.get_service_status()
    print("服务状态:")
    print(f"  已初始化: {status['initialized']}")
    print(f"  组件状态: {status['components']}")

    # 获取统计信息
    stats = await rag_service.get_statistics()
    print("\n统计信息:")
    for component, component_stats in stats.items():
        if isinstance(component_stats, dict):
            print(f"  {component}:")
            for key, value in component_stats.items():
                print(f"    {key}: {value}")

    # 健康检查
    health = await rag_service.health_check()
    print(f"\n健康状态: {health['status']}")
    if health['issues']:
        print("发现的问题:")
        for issue in health['issues']:
            print(f"  - {issue}")

    # 清理
    await rag_service.cleanup()


async def main():
    """主函数。"""
    print("RAG服务使用示例")
    print("=" * 50)

    try:
        # 运行各种示例
        await basic_qa_example()
        await knowledge_base_example()
        await chat_example()
        await advanced_rag_example()
        await batch_query_example()
        await service_management_example()

    except Exception as e:
        print(f"示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行示例
    asyncio.run(main())