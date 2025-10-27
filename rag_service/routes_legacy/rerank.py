"""
Rerank路由

提供文档重排序接口。
"""

import time
from typing import Dict, Any, List

from sanic import Blueprint, Request, Response
from sanic.response import json

from ..interfaces.rerank_interface import (
    RerankRequest,
    RerankResponse,
    RerankDocument
)
from ..exceptions import ValidationError, ModelError, NotFoundError
from ..infrastructure.monitoring.loguru_logger import get_logger


# 创建Rerank蓝图
rerank_router = Blueprint("rerank", url_prefix="/rerank")
structured_logger = get_logger("rag_service.rerank")


@rerank_router.post("/")
async def rerank_documents(request: Request) -> Response:
    """
    文档重排序接口

    Args:
        request: Sanic请求对象

    Returns:
        Response: 重排序结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")
    start_time = time.time()

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "query" not in request_data:
            raise ValidationError("Missing required field: query")
        if "documents" not in request_data:
            raise ValidationError("Missing required field: documents")

        # 构造请求对象
        rerank_request = RerankRequest(
            model=request_data.get("model", "default"),
            query=request_data["query"],
            documents=request_data["documents"],
            top_k=request_data.get("top_k"),
            return_documents=request_data.get("return_documents", True),
            return_text=request_data.get("return_text", True),
            max_chunks_per_doc=request_data.get("max_chunks_per_doc"),
            overlap_tokens=request_data.get("overlap_tokens"),
            user=request_data.get("user"),
        )

        # 获取Rerank服务
        service = request.app.ctx.service
        rerank_service = service.get_rerank_service()
        if not rerank_service:
            raise NotFoundError("Rerank service not available")

        # 验证请求
        await rerank_service.validate_request(rerank_request)

        # 记录请求信息
        documents = rerank_request.documents
        query_length = len(rerank_request.query)
        total_doc_length = sum(len(doc) for doc in documents)

        structured_logger.info(
            "重排序请求",
            extra={
                "request_id": request_id,
                "model": rerank_request.model,
                "query_length": query_length,
                "documents_count": len(documents),
                "total_doc_length": total_doc_length,
                "top_k": rerank_request.top_k,
                "return_documents": rerank_request.return_documents,
                "max_chunks_per_doc": rerank_request.max_chunks_per_doc,
            }
        )

        # 调用重排序服务
        response: RerankResponse = await rerank_service.rerank(rerank_request)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 构造响应
        api_response = {
            "object": "rerank",
            "model": response.model,
            "results": [],
        }

        # 添加可选字段
        if response.id:
            api_response["id"] = response.id
        if response.created:
            api_response["created"] = response.created
        if response.usage:
            api_response["usage"] = response.usage

        # 转换结果
        for result in response.results:
            rerank_result = {
                "index": result.index,
                "relevance_score": result.relevance_score,
            }

            # 添加文档内容（如果存在）
            if result.document is not None:
                rerank_result["document"] = result.document
            if result.text is not None:
                rerank_result["text"] = result.text

            api_response["results"].append(rerank_result)

        structured_logger.info(
            "重排序完成",
            extra={
                "request_id": request_id,
                "model": response.model,
                "results_count": len(response.results),
                "top_score": response.results[0].relevance_score if response.results else 0,
                "processing_time_seconds": round(processing_time, 3),
                "throughput_docs_per_second": round(len(documents) / processing_time, 2),
            }
        )

        return json(api_response)

    except ValidationError as e:
        structured_logger.warning(
            f"重排序请求验证失败: {e.message}",
            extra={
                "request_id": request_id,
                "error_details": e.details,
            }
        )
        return json({
            "error": {
                "message": e.message,
                "type": "invalid_request_error",
                "code": "invalid_request",
                "param": None,
                "request_id": request_id,
            }
        }, status=400)

    except ModelError as e:
        structured_logger.error(
            f"重排序模型错误: {e.message}",
            extra={
                "request_id": request_id,
                "error_details": e.details,
            }
        )
        return json({
            "error": {
                "message": e.message,
                "type": "model_error",
                "code": e.error_code,
                "param": None,
                "request_id": request_id,
            }
        }, status=500)

    except Exception as e:
        structured_logger.error(
            f"重排序请求处理失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "An unexpected error occurred",
                "type": "internal_server_error",
                "code": "internal_error",
                "param": None,
                "request_id": request_id,
            }
        }, status=500)


@rerank_router.post("/scores")
async def rerank_with_scores(request: Request) -> Response:
    """
    返回文档和相似度分数的简化接口

    Args:
        request: Sanic请求对象

    Returns:
        Response: 重排序分数结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "query" not in request_data:
            raise ValidationError("Missing required field: query")
        if "documents" not in request_data:
            raise ValidationError("Missing required field: documents")

        query = request_data["query"]
        documents = request_data["documents"]
        model = request_data.get("model")
        top_k = request_data.get("top_k")

        # 获取Rerank服务
        service = request.app.ctx.service
        rerank_service = service.get_rerank_service()
        if not rerank_service:
            raise NotFoundError("Rerank service not available")

        structured_logger.info(
            "重排序分数请求",
            extra={
                "request_id": request_id,
                "model": model,
                "query_length": len(query),
                "documents_count": len(documents),
                "top_k": top_k,
            }
        )

        # 调用重排序服务
        scores = await rerank_service.rerank_with_scores(
            query=query,
            documents=documents,
            model=model,
            top_k=top_k
        )

        structured_logger.info(
            "重排序分数完成",
            extra={
                "request_id": request_id,
                "model": model or rerank_service.model,
                "results_count": len(scores),
                "top_score": scores[0][1] if scores else 0,
            }
        )

        # 构造简化响应
        results = []
        for i, (document, score) in enumerate(scores):
            results.append({
                "rank": i + 1,
                "document": document,
                "relevance_score": score,
            })

        return json({
            "object": "rerank_scores",
            "model": model or rerank_service.model,
            "query": query,
            "results": results,
            "total_results": len(results),
        })

    except ValidationError as e:
        structured_logger.warning(
            f"重排序分数请求验证失败: {e.message}",
            extra={
                "request_id": request_id,
                "error_details": e.details,
            }
        )
        return json({
            "error": {
                "message": e.message,
                "type": "invalid_request_error",
                "code": "invalid_request",
                "param": None,
                "request_id": request_id,
            }
        }, status=400)

    except Exception as e:
        structured_logger.error(
            f"重排序分数失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to compute rerank scores",
                "type": "internal_server_error",
                "code": "scores_error",
                "param": None,
                "request_id": request_id,
            }
        }, status=500)


@rerank_router.post("/chunks")
async def rerank_chunks(request: Request) -> Response:
    """
    文档分块重排序接口

    Args:
        request: Sanic请求对象

    Returns:
        Response: 分块重排序结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "query" not in request_data:
            raise ValidationError("Missing required field: query")
        if "documents" not in request_data:
            raise ValidationError("Missing required field: documents")

        # 获取Rerank服务
        service = request.app.ctx.service
        rerank_service = service.get_rerank_service()
        if not rerank_service:
            raise NotFoundError("Rerank service not available")

        # 检查是否支持分块
        if not rerank_service.supports_feature("chunking"):
            return json({
                "error": {
                    "message": "Chunking is not supported by this provider",
                    "type": "unsupported_feature",
                    "code": "chunking_not_supported",
                    "param": None,
                    "request_id": request_id,
                }
            }, status=400)

        query = request_data["query"]
        documents = request_data["documents"]
        model = request_data.get("model")
        max_chunks_per_doc = request_data.get("max_chunks_per_doc")
        overlap_tokens = request_data.get("overlap_tokens")
        top_k = request_data.get("top_k")

        structured_logger.info(
            "分块重排序请求",
            extra={
                "request_id": request_id,
                "model": model,
                "query_length": len(query),
                "documents_count": len(documents),
                "max_chunks_per_doc": max_chunks_per_doc,
                "overlap_tokens": overlap_tokens,
                "top_k": top_k,
            }
        )

        # 调用分块重排序服务
        response = await rerank_service.rerank_chunks(
            query=query,
            documents=documents,
            max_chunks_per_doc=max_chunks_per_doc,
            overlap_tokens=overlap_tokens,
            model=model,
            top_k=top_k
        )

        # 构造响应
        api_response = {
            "object": "rerank_chunks",
            "model": response.model,
            "results": [],
            "chunking_info": {
                "max_chunks_per_doc": max_chunks_per_doc,
                "overlap_tokens": overlap_tokens,
            },
        }

        # 添加可选字段
        if response.id:
            api_response["id"] = response.id
        if response.created:
            api_response["created"] = response.created
        if response.usage:
            api_response["usage"] = response.usage

        # 转换结果
        for result in response.results:
            chunk_result = {
                "index": result.index,
                "relevance_score": result.relevance_score,
            }

            # 添加文档内容
            if result.document is not None:
                chunk_result["document"] = result.document
            if result.text is not None:
                chunk_result["text"] = result.text

            api_response["results"].append(chunk_result)

        structured_logger.info(
            "分块重排序完成",
            extra={
                "request_id": request_id,
                "model": response.model,
                "chunks_count": len(response.results),
                "top_score": response.results[0].relevance_score if response.results else 0,
            }
        )

        return json(api_response)

    except ValidationError as e:
        structured_logger.warning(
            f"分块重排序请求验证失败: {e.message}",
            extra={
                "request_id": request_id,
                "error_details": e.details,
            }
        )
        return json({
            "error": {
                "message": e.message,
                "type": "invalid_request_error",
                "code": "invalid_request",
                "param": None,
                "request_id": request_id,
            }
        }, status=400)

    except Exception as e:
        structured_logger.error(
            f"分块重排序失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to rerank chunks",
                "type": "internal_server_error",
                "code": "chunks_error",
                "param": None,
                "request_id": request_id,
            }
        }, status=500)


@rerank_router.get("/models")
async def list_rerank_models(request: Request) -> Response:
    """
    列出可用的重排序模型

    Args:
        request: Sanic请求对象

    Returns:
        Response: 重排序模型列表响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        rerank_service = service.get_rerank_service()
        if not rerank_service:
            return json({
                "object": "list",
                "data": [],
            })

        # 获取模型能力信息
        capabilities = rerank_service.capabilities

        # 构造模型列表
        models = []
        for model_name in capabilities.supported_models:
            model_info = {
                "id": model_name,
                "object": "rerank_model",
                "created": int(time.time()),
                "owned_by": rerank_service.provider_name,
                "capabilities": {
                    "max_query_length": capabilities.max_query_length,
                    "max_document_length": capabilities.max_document_length,
                    "max_documents_per_request": capabilities.max_documents_per_request,
                    "max_top_k": capabilities.max_top_k,
                    "supports_chunking": capabilities.supports_chunking,
                    "supports_overlap": capabilities.supports_overlap,
                    "supports_custom_top_k": capabilities.supports_custom_top_k,
                    "supports_return_documents": capabilities.supports_return_documents,
                    "supports_scoring_only": capabilities.supports_scoring_only,
                }
            }
            models.append(model_info)

        structured_logger.info(
            "重排序模型列表请求",
            extra={
                "request_id": request_id,
                "models_count": len(models),
                "provider": rerank_service.provider_name,
            }
        )

        return json({
            "object": "list",
            "data": models,
        })

    except Exception as e:
        structured_logger.error(
            f"获取重排序模型列表失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to retrieve rerank models",
                "type": "internal_server_error",
                "code": "models_error",
                "param": None,
                "request_id": request_id,
            }
        }, status=500)


@rerank_router.post("/merge")
async def merge_reranked_results(request: Request) -> Response:
    """
    合并重排序结果与原始文档

    Args:
        request: Sanic请求对象

    Returns:
        Response: 合并后的结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "original_documents" not in request_data:
            raise ValidationError("Missing required field: original_documents")
        if "reranked_results" not in request_data:
            raise ValidationError("Missing required field: reranked_results")

        original_documents = request_data["original_documents"]
        reranked_results = request_data["reranked_results"]
        preserve_unranked = request_data.get("preserve_unranked", True)

        # 获取Rerank服务
        service = request.app.ctx.service
        rerank_service = service.get_rerank_service()
        if not rerank_service:
            raise NotFoundError("Rerank service not available")

        structured_logger.info(
            "合并重排序结果请求",
            extra={
                "request_id": request_id,
                "original_docs_count": len(original_documents),
                "reranked_results_count": len(reranked_results),
                "preserve_unranked": preserve_unranked,
            }
        )

        # 构造RerankResponse对象（从请求数据）
        rerank_response = RerankResponse(
            model=reranked_results.get("model", "unknown"),
            results=[
                RerankDocument(
                    index=result["index"],
                    relevance_score=result["relevance_score"],
                    document=result.get("document"),
                    text=result.get("text"),
                )
                for result in reranked_results.get("results", [])
            ]
        )

        # 合并结果
        merged_results = await rerank_service.merge_reranked_results(
            original_documents=original_documents,
            reranked_results=rerank_response,
            preserve_unranked=preserve_unranked
        )

        structured_logger.info(
            "合并重排序结果完成",
            extra={
                "request_id": request_id,
                "merged_results_count": len(merged_results),
                "ranked_results": sum(1 for r in merged_results if r.get("ranked", False)),
                "unranked_results": sum(1 for r in merged_results if not r.get("ranked", False)),
            }
        )

        return json({
            "object": "merged_rerank_results",
            "original_documents_count": len(original_documents),
            "merged_results": merged_results,
            "preserve_unranked": preserve_unranked,
            "statistics": {
                "total_results": len(merged_results),
                "ranked_results": sum(1 for r in merged_results if r.get("ranked", False)),
                "unranked_results": sum(1 for r in merged_results if not r.get("ranked", False)),
            }
        })

    except ValidationError as e:
        structured_logger.warning(
            f"合并重排序结果请求验证失败: {e.message}",
            extra={
                "request_id": request_id,
                "error_details": e.details,
            }
        )
        return json({
            "error": {
                "message": e.message,
                "type": "invalid_request_error",
                "code": "invalid_request",
                "param": None,
                "request_id": request_id,
            }
        }, status=400)

    except Exception as e:
        structured_logger.error(
            f"合并重排序结果失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to merge reranked results",
                "type": "internal_server_error",
                "code": "merge_error",
                "param": None,
                "request_id": request_id,
            }
        }, status=500)


structured_logger.info("Rerank路由加载完成")