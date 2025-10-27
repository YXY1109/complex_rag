"""
Memory路由

提供对话记忆管理接口（Mem0集成）。
"""

from typing import Dict, Any

from sanic import Blueprint, Request, Response
from sanic.response import json

from ..exceptions import NotFoundError, ValidationError
from ..infrastructure.monitoring.loguru_logger import get_logger


# 创建Memory蓝图
memory_router = Blueprint("memory", url_prefix="/memory")
structured_logger = get_logger("rag_service.memory")


@memory_router.post("/search")
async def search_memories(request: Request) -> Response:
    """
    搜索相关记忆

    Args:
        request: Sanic请求对象

    Returns:
        Response: 搜索结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        memory_service = service.get_memory_service()
        if not memory_service:
            raise NotFoundError("Memory service not available")

        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "query" not in request_data:
            raise ValidationError("Missing required field: query")

        query = request_data["query"]
        user_id = request_data.get("user_id")
        limit = request_data.get("limit", 10)
        threshold = request_data.get("threshold")

        # 记录请求信息
        structured_logger.info(
            "记忆搜索请求",
            extra={
                "request_id": request_id,
                "query_length": len(query),
                "user_id": user_id,
                "limit": limit,
                "threshold": threshold,
            }
        )

        # 搜索记忆
        memories = await memory_service.search_memories(
            query=query,
            user_id=user_id,
            limit=limit,
            threshold=threshold
        )

        structured_logger.info(
            "记忆搜索完成",
            extra={
                "request_id": request_id,
                "results_count": len(memories),
                "user_id": user_id,
            }
        )

        return json({
            "object": "memory_search",
            "request_id": request_id,
            "query": query,
            "user_id": user_id,
            "memories": memories,
            "count": len(memories),
        })

    except ValidationError as e:
        structured_logger.warning(
            f"记忆搜索请求验证失败: {e.message}",
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
                "request_id": request_id,
            }
        }, status=400)

    except Exception as e:
        structured_logger.error(
            f"记忆搜索失败: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return json({
            "error": {
                "message": "Memory search failed",
                "type": "internal_server_error",
                "code": "search_error",
                "request_id": request_id,
            }
        }, status=500)


@memory_router.post("/add")
async def add_memory(request: Request) -> Response:
    """
    添加新记忆

    Args:
        request: Sanic请求对象

    Returns:
        Response: 添加结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        memory_service = service.get_memory_service()
        if not memory_service:
            raise NotFoundError("Memory service not available")

        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证必需字段
        if "content" not in request_data:
            raise ValidationError("Missing required field: content")

        content = request_data["content"]
        user_id = request_data.get("user_id")
        metadata = request_data.get("metadata")

        # 记录请求信息
        structured_logger.info(
            "添加记忆请求",
            extra={
                "request_id": request_id,
                "content_length": len(content),
                "user_id": user_id,
                "has_metadata": bool(metadata),
            }
        )

        # 添加记忆
        result = await memory_service.add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata
        )

        if result.get("success"):
            structured_logger.info(
                "添加记忆成功",
                extra={
                    "request_id": request_id,
                    "memory_id": result["memory_id"],
                    "user_id": user_id,
                }
            )
            return json({
                "object": "memory_add",
                "request_id": request_id,
                "success": True,
                "memory_id": result["memory_id"],
                "created_at": result["created_at"],
            })
        else:
            structured_logger.error(
                f"添加记忆失败: {result.get('message', 'Unknown error')}",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                }
            )
            return json({
                "error": {
                    "message": result.get("message", "Failed to add memory"),
                    "type": "add_memory_error",
                    "code": "add_failed",
                    "request_id": request_id,
                }
            }, status=500)

    except ValidationError as e:
        structured_logger.warning(
            f"添加记忆请求验证失败: {e.message}",
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
                "request_id": request_id,
            }
        }, status=400)

    except Exception as e:
        structured_logger.error(
            f"添加记忆失败: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return json({
            "error": {
                "message": "Add memory failed",
                "type": "internal_server_error",
                "code": "add_error",
                "request_id": request_id,
            }
        }, status=500)


@memory_router.put("/<memory_id>")
async def update_memory(request: Request, memory_id: str) -> Response:
    """
    更新记忆

    Args:
        request: Sanic请求对象
        memory_id: 记忆ID

    Returns:
        Response: 更新结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        memory_service = service.get_memory_service()
        if not memory_service:
            raise NotFoundError("Memory service not available")

        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        content = request_data.get("content")
        metadata = request_data.get("metadata")
        user_id = request_data.get("user_id")

        # 记录请求信息
        structured_logger.info(
            "更新记忆请求",
            extra={
                "request_id": request_id,
                "memory_id": memory_id,
                "user_id": user_id,
                "content_updated": content is not None,
                "metadata_updated": metadata is not None,
            }
        )

        # 更新记忆
        result = await memory_service.update_memory(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            user_id=user_id
        )

        if result.get("success"):
            structured_logger.info(
                "更新记忆成功",
                extra={
                    "request_id": request_id,
                    "memory_id": memory_id,
                    "user_id": user_id,
                }
            )
            return json({
                "object": "memory_update",
                "request_id": request_id,
                "success": True,
                "memory_id": memory_id,
                "updated_at": result["updated_at"],
            })
        else:
            structured_logger.error(
                f"更新记忆失败: {result.get('message', 'Unknown error')}",
                extra={
                    "request_id": request_id,
                    "memory_id": memory_id,
                    "user_id": user_id,
                }
            )
            return json({
                "error": {
                    "message": result.get("message", "Failed to update memory"),
                    "type": "update_memory_error",
                    "code": "update_failed",
                    "request_id": request_id,
                }
            }, status=500)

    except ValidationError as e:
        structured_logger.warning(
            f"更新记忆请求验证失败: {e.message}",
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
                "request_id": request_id,
            }
        }, status=400)

    except Exception as e:
        structured_logger.error(
            f"更新记忆失败: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return json({
            "error": {
                "message": "Update memory failed",
                "type": "internal_server_error",
                "code": "update_error",
                "request_id": request_id,
            }
        }, status=500)


@memory_router.delete("/<memory_id>")
async def delete_memory(request: Request, memory_id: str) -> Response:
    """
    删除记忆

    Args:
        request: Sanic请求对象
        memory_id: 记忆ID

    Returns:
        Response: 删除结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        memory_service = service.get_memory_service()
        if not memory_service:
            raise NotFoundError("Memory service not available")

        # 获取用户ID（从查询参数或请求体）
        user_id = request.args.get("user_id")
        if not user_id and request.json:
            user_id = request.json.get("user_id")

        # 记录请求信息
        structured_logger.info(
            "删除记忆请求",
            extra={
                "request_id": request_id,
                "memory_id": memory_id,
                "user_id": user_id,
            }
        )

        # 删除记忆
        success = await memory_service.delete_memory(
            memory_id=memory_id,
            user_id=user_id
        )

        if success:
            structured_logger.info(
                "删除记忆成功",
                extra={
                    "request_id": request_id,
                    "memory_id": memory_id,
                    "user_id": user_id,
                }
            )
            return json({
                "object": "memory_delete",
                "request_id": request_id,
                "success": True,
                "memory_id": memory_id,
            })
        else:
            structured_logger.warning(
                "记忆不存在或删除失败",
                extra={
                    "request_id": request_id,
                    "memory_id": memory_id,
                    "user_id": user_id,
                }
            )
            return json({
                "error": {
                    "message": "Memory not found or delete failed",
                    "type": "not_found",
                    "code": "delete_failed",
                    "request_id": request_id,
                }
            }, status=404)

    except Exception as e:
        structured_logger.error(
            f"删除记忆失败: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return json({
            "error": {
                "message": "Delete memory failed",
                "type": "internal_server_error",
                "code": "delete_error",
                "request_id": request_id,
            }
        }, status=500)


@memory_router.get("/history")
async def get_memory_history(request: Request) -> Response:
    """
    获取记忆历史

    Args:
        request: Sanic请求对象

    Returns:
        Response: 记忆历史
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        memory_service = service.get_memory_service()
        if not memory_service:
            raise NotFoundError("Memory service not available")

        # 获取查询参数
        user_id = request.args.get("user_id")
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        # 记录请求信息
        structured_logger.info(
            "获取记忆历史请求",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "limit": limit,
                "offset": offset,
            }
        )

        # 获取记忆历史
        memories = await memory_service.get_memory_history(
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        structured_logger.info(
            "获取记忆历史完成",
            extra={
                "request_id": request_id,
                "results_count": len(memories),
                "user_id": user_id,
            }
        )

        return json({
            "object": "memory_history",
            "request_id": request_id,
            "user_id": user_id,
            "memories": memories,
            "count": len(memories),
            "limit": limit,
            "offset": offset,
        })

    except Exception as e:
        structured_logger.error(
            f"获取记忆历史失败: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return json({
            "error": {
                "message": "Get memory history failed",
                "type": "internal_server_error",
                "code": "history_error",
                "request_id": request_id,
            }
        }, status=500)


@memory_router.get("/stats")
async def get_memory_stats(request: Request) -> Response:
    """
    获取记忆统计信息

    Args:
        request: Sanic请求对象

    Returns:
        Response: 统计信息
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        memory_service = service.get_memory_service()
        if not memory_service:
            raise NotFoundError("Memory service not available")

        # 获取查询参数
        user_id = request.args.get("user_id")

        # 记录请求信息
        structured_logger.info(
            "获取记忆统计请求",
            extra={
                "request_id": request_id,
                "user_id": user_id,
            }
        )

        # 获取统计信息
        stats = await memory_service.get_memory_stats(user_id=user_id)

        structured_logger.info(
            "获取记忆统计完成",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "total_memories": stats.get("total_memories", 0),
            }
        )

        return json({
            "object": "memory_stats",
            "request_id": request_id,
            "user_id": user_id,
            "stats": stats,
        })

    except Exception as e:
        structured_logger.error(
            f"获取记忆统计失败: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return json({
            "error": {
                "message": "Get memory stats failed",
                "type": "internal_server_error",
                "code": "stats_error",
                "request_id": request_id,
            }
        }, status=500)


@memory_router.delete("/clear")
async def clear_all_memories(request: Request) -> Response:
    """
    清除所有记忆

    Args:
        request: Sanic请求对象

    Returns:
        Response: 清除结果
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        memory_service = service.get_memory_service()
        if not memory_service:
            raise NotFoundError("Memory service not available")

        # 获取用户ID（从查询参数或请求体）
        user_id = request.args.get("user_id")
        if not user_id and request.json:
            user_id = request.json.get("user_id")

        # 记录请求信息
        structured_logger.info(
            "清除所有记忆请求",
            extra={
                "request_id": request_id,
                "user_id": user_id,
            }
        )

        # 清除记忆
        success = await memory_service.clear_all_memories(user_id=user_id)

        if success:
            structured_logger.info(
                "清除所有记忆成功",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                }
            )
            return json({
                "object": "memory_clear",
                "request_id": request_id,
                "success": True,
                "user_id": user_id,
            })
        else:
            structured_logger.warning(
                "清除记忆失败或无记忆可清除",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                }
            )
            return json({
                "error": {
                    "message": "No memories found or clear failed",
                    "type": "not_found",
                    "code": "clear_failed",
                    "request_id": request_id,
                }
            }, status=404)

    except Exception as e:
        structured_logger.error(
            f"清除记忆失败: {e}",
            extra={"request_id": request_id, "error": str(e)}
        )
        return json({
            "error": {
                "message": "Clear memories failed",
                "type": "internal_server_error",
                "code": "clear_error",
                "request_id": request_id,
            }
        }, status=500)


structured_logger.info("Memory路由加载完成")