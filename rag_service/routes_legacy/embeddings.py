"""
Embeddings路由

提供OpenAI兼容的文本向量化接口。
"""

import time
from typing import Dict, Any, List, Union

from sanic import Blueprint, Request, Response
from sanic.response import json

from ..interfaces.embedding_interface import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
    EmbeddingInputType
)
from ..exceptions import ValidationError, ModelError, NotFoundError
from ..infrastructure.monitoring.loguru_logger import get_logger


# 创建Embeddings蓝图
embeddings_router = Blueprint("embeddings", url_prefix="/embeddings")
structured_logger = get_logger("rag_service.embeddings")


@embeddings_router.post("/")
async def create_embeddings(request: Request) -> Response:
    """
    OpenAI兼容的文本向量化接口

    Args:
        request: Sanic请求对象

    Returns:
        Response: 向量化响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")
    start_time = time.time()

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "input" not in request_data:
            raise ValidationError("Missing required field: input")
        if "model" not in request_data:
            raise ValidationError("Missing required field: model")

        # 构造请求对象
        embedding_request = EmbeddingRequest(
            input=request_data["input"],
            model=request_data["model"],
            encoding_format=request_data.get("encoding_format", "float"),
            dimensions=request_data.get("dimensions"),
            user=request_data.get("user"),
            input_type=EmbeddingInputType(request_data["input_type"]) if "input_type" in request_data else None,
        )

        # 获取Embedding服务
        service = request.app.ctx.service
        embedding_service = service.get_embedding_service()
        if not embedding_service:
            raise NotFoundError("Embedding service not available")

        # 验证请求
        await embedding_service.validate_request(embedding_request)

        # 记录请求信息
        input_texts = embedding_request.input
        if isinstance(input_texts, str):
            input_length = len(input_texts)
            input_count = 1
        else:
            input_length = sum(len(text) for text in input_texts)
            input_count = len(input_texts)

        structured_logger.info(
            "向量化请求",
            extra={
                "request_id": request_id,
                "model": embedding_request.model,
                "input_count": input_count,
                "input_length": input_length,
                "encoding_format": embedding_request.encoding_format,
                "dimensions": embedding_request.dimensions,
                "input_type": embedding_request.input_type,
            }
        )

        # 调用向量化服务
        response: EmbeddingResponse = await embedding_service.create_embedding(embedding_request)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 构造OpenAI格式的响应
        openai_response = {
            "object": "list",
            "data": [],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }

        # 转换数据
        for data in response.data:
            embedding_data = {
                "object": "embedding",
                "embedding": data.embedding,
                "index": data.index,
            }
            openai_response["data"].append(embedding_data)

        structured_logger.info(
            "向量化完成",
            extra={
                "request_id": request_id,
                "model": response.model,
                "embeddings_count": len(response.data),
                "embedding_dimensions": len(response.data[0].embedding) if response.data else 0,
                "prompt_tokens": response.usage.prompt_tokens,
                "processing_time_seconds": round(processing_time, 3),
            }
        )

        return json(openai_response)

    except ValidationError as e:
        structured_logger.warning(
            f"向量化请求验证失败: {e.message}",
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
            f"向量化模型错误: {e.message}",
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
            f"向量化请求处理失败: {e}",
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


@embeddings_router.get("/models")
async def list_embedding_models(request: Request) -> Response:
    """
    列出可用的向量化模型

    Args:
        request: Sanic请求对象

    Returns:
        Response: 向量化模型列表响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        embedding_service = service.get_embedding_service()
        if not embedding_service:
            return json({
                "object": "list",
                "data": [],
            })

        # 获取模型能力信息
        capabilities = embedding_service.capabilities

        # 构造OpenAI格式的模型列表
        models = []
        for model_name in capabilities.supported_models:
            # 获取模型维度
            dimensions = capabilities.embedding_dimensions.get(model_name, 0)

            model_info = {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": embedding_service.provider_name,
                "permission": [],
                "root": model_name,
                "parent": None,
                # 添加向量化模型特有信息
                "embedding": {
                    "dimensions": dimensions,
                    "max_input_length": capabilities.max_input_length,
                    "max_batch_size": capabilities.max_batch_size,
                    "supports_batch": capabilities.supports_batch,
                    "supports_different_input_types": capabilities.supports_different_input_types,
                    "supports_custom_dimensions": capabilities.supports_custom_dimensions,
                }
            }
            models.append(model_info)

        structured_logger.info(
            "向量化模型列表请求",
            extra={
                "request_id": request_id,
                "models_count": len(models),
                "provider": embedding_service.provider_name,
            }
        )

        return json({
            "object": "list",
            "data": models,
        })

    except Exception as e:
        structured_logger.error(
            f"获取向量化模型列表失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to retrieve embedding models",
                "type": "internal_server_error",
                "code": "models_error",
                "request_id": request_id,
            }
        }, status=500)


@embeddings_router.post("/similarity")
async def compute_similarity(request: Request) -> Response:
    """
    计算两个文本的相似度（扩展接口）

    Args:
        request: Sanic请求对象

    Returns:
        Response: 相似度计算结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "text1" not in request_data or "text2" not in request_data:
            raise ValidationError("Missing required fields: text1, text2")

        text1 = request_data["text1"]
        text2 = request_data["text2"]
        model = request_data.get("model")
        metric = request_data.get("metric", "cosine")

        # 获取Embedding服务
        service = request.app.ctx.service
        embedding_service = service.get_embedding_service()
        if not embedding_service:
            raise NotFoundError("Embedding service not available")

        structured_logger.info(
            "相似度计算请求",
            extra={
                "request_id": request_id,
                "model": model,
                "metric": metric,
                "text1_length": len(text1),
                "text2_length": len(text2),
            }
        )

        # 向量化两个文本
        embedding1 = await embedding_service.embed_text(text1, model=model)
        embedding2 = await embedding_service.embed_text(text2, model=model)

        # 计算相似度
        similarity_score = await embedding_service.compute_similarity(
            embedding1, embedding2, metric=metric
        )

        structured_logger.info(
            "相似度计算完成",
            extra={
                "request_id": request_id,
                "model": model or embedding_service.model,
                "metric": metric,
                "similarity_score": similarity_score,
            }
        )

        return json({
            "similarity_score": similarity_score,
            "metric": metric,
            "model": model or embedding_service.model,
            "text1_length": len(text1),
            "text2_length": len(text2),
            "embedding_dimensions": len(embedding1) if embedding1 else 0,
        })

    except ValidationError as e:
        structured_logger.warning(
            f"相似度计算请求验证失败: {e.message}",
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
            f"相似度计算失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to compute similarity",
                "type": "internal_server_error",
                "code": "similarity_error",
                "param": None,
                "request_id": request_id,
            }
        }, status=500)


@embeddings_router.post("/batch")
async def batch_embeddings(request: Request) -> Response:
    """
    批量向量化接口（优化版）

    Args:
        request: Sanic请求对象

    Returns:
        Response: 批量向量化响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")
    start_time = time.time()

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "texts" not in request_data:
            raise ValidationError("Missing required field: texts")

        texts = request_data["texts"]
        if not isinstance(texts, list):
            raise ValidationError("texts must be a list")

        if len(texts) == 0:
            return json({
                "object": "list",
                "data": [],
                "model": request_data.get("model", "unknown"),
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                }
            })

        # 获取Embedding服务
        service = request.app.ctx.service
        embedding_service = service.get_embedding_service()
        if not embedding_service:
            raise NotFoundError("Embedding service not available")

        model = request_data.get("model")
        input_type = EmbeddingInputType(request_data["input_type"]) if "input_type" in request_data else None
        batch_size = request_data.get("batch_size", embedding_service.batch_size)

        structured_logger.info(
            "批量向量化请求",
            extra={
                "request_id": request_id,
                "model": model,
                "texts_count": len(texts),
                "batch_size": batch_size,
                "input_type": input_type,
                "total_chars": sum(len(text) for text in texts),
            }
        )

        # 批量处理
        embeddings = await embedding_service.embed_texts_batch(
            texts=texts,
            model=model,
            input_type=input_type,
            batch_size=batch_size
        )

        # 计算处理时间
        processing_time = time.time() - start_time

        # 构造响应
        data = []
        for i, embedding in enumerate(embeddings):
            embedding_data = {
                "object": "embedding",
                "embedding": embedding,
                "index": i,
            }
            data.append(embedding_data)

        # 估算token使用量
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars // 4  # 简单估算

        structured_logger.info(
            "批量向量化完成",
            extra={
                "request_id": request_id,
                "model": model or embedding_service.model,
                "embeddings_count": len(embeddings),
                "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
                "estimated_tokens": estimated_tokens,
                "processing_time_seconds": round(processing_time, 3),
                "throughput_texts_per_second": round(len(texts) / processing_time, 2),
            }
        )

        return json({
            "object": "list",
            "data": data,
            "model": model or embedding_service.model,
            "usage": {
                "prompt_tokens": estimated_tokens,
                "total_tokens": estimated_tokens,
            },
            "processing_info": {
                "batch_size": batch_size,
                "processing_time_seconds": round(processing_time, 3),
                "throughput_texts_per_second": round(len(texts) / processing_time, 2),
            }
        })

    except ValidationError as e:
        structured_logger.warning(
            f"批量向量化请求验证失败: {e.message}",
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
            f"批量向量化失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to process batch embeddings",
                "type": "internal_server_error",
                "code": "batch_error",
                "param": None,
                "request_id": request_id,
            }
        }, status=500)


structured_logger.info("Embeddings路由加载完成")