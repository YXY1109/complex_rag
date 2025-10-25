"""
RAG核心功能路由

提供RAG查询、生成、流式响应等核心功能的API接口。
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
import logging

from ..dependencies import get_rag_service, get_current_user, get_request_context
from ...interfaces.rag_interface import RAGQuery, RAGResult, RetrievalMode, GenerationMode
from ...services.unified_rag_service import UnifiedRAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=Dict[str, Any])
async def rag_query(
    query: RAGQuery = Body(...),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    执行RAG查询。

    Args:
        query: RAG查询参数
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 查询结果
    """
    try:
        # 设置用户信息
        query.user_id = query.user_id or current_user.get("user_id")
        query.tenant_id = query.tenant_id or current_user.get("tenant_id")

        logger.info(
            f"执行RAG查询 - 用户: {query.user_id}, "
            f"租户: {query.tenant_id}, "
            f"查询: {query.query[:100]}..."
        )

        # 执行查询
        result = await rag_service.query(query)

        return {
            "success": True,
            "data": {
                "query_id": result.query_id,
                "answer": result.answer,
                "retrieval_result": {
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "title": chunk.title,
                            "score": chunk.score,
                            "metadata": chunk.metadata
                        }
                        for chunk in result.retrieval_result.chunks
                    ],
                    "search_time": result.retrieval_result.search_time,
                    "total_found": result.retrieval_result.total_found
                },
                "generation_result": {
                    "model": result.generation_result.model,
                    "generation_time": result.generation_result.generation_time,
                    "token_count": result.generation_result.token_count
                },
                "total_time": result.total_time,
                "metadata": result.metadata
            },
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询执行失败: {str(e)}")


@router.post("/query/stream")
async def rag_query_stream(
    query: RAGQuery = Body(...),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    流式RAG查询。

    Args:
        query: RAG查询参数
        rag_service: RAG服务实例
        current_user: 当前用户信息

    Returns:
        StreamingResponse: 流式响应
    """
    try:
        # 设置用户信息
        query.user_id = query.user_id or current_user.get("user_id")
        query.tenant_id = query.tenant_id or current_user.get("tenant_id")

        async def generate():
            """生成流式响应。"""
            try:
                async for chunk in rag_service.query_stream(query):
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"流式查询错误: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )

    except Exception as e:
        logger.error(f"流式RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"流式查询失败: {str(e)}")


@router.post("/query/batch", response_model=Dict[str, Any])
async def rag_batch_query(
    queries: List[RAGQuery] = Body(...),
    max_concurrent: Optional[int] = Query(5, ge=1, le=10),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    批量RAG查询。

    Args:
        queries: RAG查询列表
        max_concurrent: 最大并发数
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 批量查询结果
    """
    try:
        # 设置用户信息
        for query in queries:
            query.user_id = query.user_id or current_user.get("user_id")
            query.tenant_id = query.tenant_id or current_user.get("tenant_id")

        logger.info(
            f"执行批量RAG查询 - 用户: {current_user.get('user_id')}, "
            f"查询数量: {len(queries)}"
        )

        # 执行批量查询
        results = await rag_service.batch_query(queries, max_concurrent)

        return {
            "success": True,
            "data": {
                "results": [
                    {
                        "query_id": result.query_id,
                        "success": result.success,
                        "answer": result.answer if result.success else None,
                        "error": result.error if not result.success else None,
                        "total_time": result.total_time
                    }
                    for result in results
                ],
                "total_queries": len(queries),
                "successful_queries": sum(1 for r in results if r.success),
                "failed_queries": sum(1 for r in results if not r.success)
            },
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"批量RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量查询失败: {str(e)}")


@router.post("/qa/simple", response_model=Dict[str, Any])
async def simple_qa(
    question: str = Body(..., embed=True),
    knowledge_bases: Optional[List[str]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    简单问答接口。

    Args:
        question: 问题
        knowledge_bases: 知识库列表
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 问答结果
    """
    try:
        logger.info(
            f"执行简单问答 - 用户: {current_user.get('user_id')}, "
            f"问题: {question[:100]}..."
        )

        # 执行简单问答
        result = await rag_service.simple_qa(
            question=question,
            knowledge_bases=knowledge_bases,
            user_id=current_user.get("user_id"),
            tenant_id=current_user.get("tenant_id")
        )

        return {
            "success": True,
            "data": result,
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"简单问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@router.post("/document/summary", response_model=Dict[str, Any])
async def document_summary(
    document_content: str = Body(..., embed=True),
    max_length: int = Body(500, ge=100, le=2000),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    文档摘要。

    Args:
        document_content: 文档内容
        max_length: 最大摘要长度
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 摘要结果
    """
    try:
        logger.info(
            f"执行文档摘要 - 用户: {current_user.get('user_id')}, "
            f"文档长度: {len(document_content)}"
        )

        # 执行文档摘要
        summary = await rag_service.document_summary(
            document_content=document_content,
            max_length=max_length
        )

        return {
            "success": True,
            "data": {
                "summary": summary,
                "original_length": len(document_content),
                "summary_length": len(summary)
            },
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"文档摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"摘要失败: {str(e)}")


@router.post("/documents/compare", response_model=Dict[str, Any])
async def compare_documents(
    documents: List[str] = Body(...),
    criteria: Optional[str] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    文档比较。

    Args:
        documents: 文档内容列表
        criteria: 比较标准
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 比较结果
    """
    try:
        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个文档进行比较")

        logger.info(
            f"执行文档比较 - 用户: {current_user.get('user_id')}, "
            f"文档数量: {len(documents)}"
        )

        # 执行文档比较
        results = await rag_service.compare_documents(
            documents=documents,
            criteria=criteria
        )

        return {
            "success": True,
            "data": {
                "comparison_results": results,
                "total_documents": len(documents)
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档比较失败: {e}")
        raise HTTPException(status_code=500, detail=f"比较失败: {str(e)}")