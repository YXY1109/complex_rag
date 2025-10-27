"""
统一记忆路由
从Sanic RAG服务迁移的记忆管理功能到FastAPI
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse

from config.loguru_config import get_logger

# 创建路由器
router = APIRouter()
structured_logger = get_logger("api.unified_memory")


# Pydantic模型定义
class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    user_id: Optional[str] = Field(None, description="用户ID")
    limit: Optional[int] = Field(10, ge=1, le=100, description="返回结果数量限制")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")


class MemoryAddRequest(BaseModel):
    content: str = Field(..., description="记忆内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    user_id: Optional[str] = Field(None, description="用户ID")


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = Field(None, description="更新后的记忆内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="更新的元数据")


class MemoryResponse(BaseModel):
    memory_id: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: Optional[str] = None
    similarity_score: Optional[float] = None


class MemorySearchResponse(BaseModel):
    results: List[MemoryResponse]
    total: int
    query: str


class MemoryStatsResponse(BaseModel):
    total_memories: int
    user_memories: Dict[str, int]
    recent_memories: int  # 最近24小时内的记忆数量
    storage_usage_mb: float


@router.post("/search", response_model=MemorySearchResponse, summary="搜索记忆")
async def search_memories(request: Request, search_request: MemorySearchRequest):
    """
    搜索相关记忆

    Args:
        request: FastAPI请求对象
        search_request: 记忆搜索请求

    Returns:
        MemorySearchResponse: 搜索结果
    """
    try:
        # 获取全局服务实例（从unified_main注入）
        memory_service = getattr(request.app.state, 'memory_service', None)
        if not memory_service:
            raise HTTPException(status_code=503, detail="记忆服务不可用")

        # 记录请求信息
        structured_logger.info(
            "记忆搜索请求",
            extra={
                "query_length": len(search_request.query),
                "user_id": search_request.user_id,
                "limit": search_request.limit,
                "threshold": search_request.threshold,
            }
        )

        # 搜索记忆
        memories = await memory_service.search_memories(
            query=search_request.query,
            user_id=search_request.user_id,
            limit=search_request.limit,
            threshold=search_request.threshold
        )

        # 转换为响应模型
        results = []
        for memory in memories:
            memory_response = MemoryResponse(
                memory_id=memory.get('memory_id', ''),
                content=memory.get('content', ''),
                metadata=memory.get('metadata', {}),
                created_at=memory.get('created_at', ''),
                updated_at=memory.get('updated_at'),
                similarity_score=memory.get('similarity_score')
            )
            results.append(memory_response)

        structured_logger.info(
            "记忆搜索完成",
            extra={
                "results_count": len(results),
                "user_id": search_request.user_id,
            }
        )

        return MemorySearchResponse(
            results=results,
            total=len(results),
            query=search_request.query
        )

    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(f"搜索记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail="记忆搜索失败")


@router.post("/add", response_model=Dict[str, Any], summary="添加记忆")
async def add_memory(request: Request, add_request: MemoryAddRequest):
    """
    添加新记忆

    Args:
        request: FastAPI请求对象
        add_request: 记忆添加请求

    Returns:
        Dict: 包含memory_id的响应
    """
    try:
        # 获取全局服务实例
        memory_service = getattr(request.app.state, 'memory_service', None)
        if not memory_service:
            raise HTTPException(status_code=503, detail="记忆服务不可用")

        # 记录请求信息
        structured_logger.info(
            "添加记忆请求",
            extra={
                "content_length": len(add_request.content),
                "user_id": add_request.user_id,
                "has_metadata": add_request.metadata is not None,
            }
        )

        # 添加记忆
        memory_id = await memory_service.add_memory(
            content=add_request.content,
            metadata=add_request.metadata,
            user_id=add_request.user_id
        )

        structured_logger.info(
            "记忆添加完成",
            extra={
                "memory_id": memory_id,
                "user_id": add_request.user_id,
            }
        )

        return {"success": True, "memory_id": memory_id, "message": "记忆添加成功"}

    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(f"添加记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail="记忆添加失败")


@router.put("/{memory_id}", response_model=Dict[str, Any], summary="更新记忆")
async def update_memory(request: Request, memory_id: str, update_request: MemoryUpdateRequest):
    """
    更新记忆

    Args:
        request: FastAPI请求对象
        memory_id: 记忆ID
        update_request: 记忆更新请求

    Returns:
        Dict: 更新结果
    """
    try:
        # 获取全局服务实例
        memory_service = getattr(request.app.state, 'memory_service', None)
        if not memory_service:
            raise HTTPException(status_code=503, detail="记忆服务不可用")

        # 记录请求信息
        structured_logger.info(
            "更新记忆请求",
            extra={
                "memory_id": memory_id,
                "has_content": update_request.content is not None,
                "has_metadata": update_request.metadata is not None,
            }
        )

        # 更新记忆
        success = await memory_service.update_memory(
            memory_id=memory_id,
            content=update_request.content,
            metadata=update_request.metadata
        )

        if not success:
            raise HTTPException(status_code=404, detail="记忆不存在")

        structured_logger.info(
            "记忆更新完成",
            extra={"memory_id": memory_id}
        )

        return {"success": True, "message": "记忆更新成功"}

    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(f"更新记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail="记忆更新失败")


@router.delete("/{memory_id}", response_model=Dict[str, Any], summary="删除记忆")
async def delete_memory(request: Request, memory_id: str):
    """
    删除记忆

    Args:
        request: FastAPI请求对象
        memory_id: 记忆ID

    Returns:
        Dict: 删除结果
    """
    try:
        # 获取全局服务实例
        memory_service = getattr(request.app.state, 'memory_service', None)
        if not memory_service:
            raise HTTPException(status_code=503, detail="记忆服务不可用")

        # 记录请求信息
        structured_logger.info(
            "删除记忆请求",
            extra={"memory_id": memory_id}
        )

        # 删除记忆
        success = await memory_service.delete_memory(memory_id)

        if not success:
            raise HTTPException(status_code=404, detail="记忆不存在")

        structured_logger.info(
            "记忆删除完成",
            extra={"memory_id": memory_id}
        )

        return {"success": True, "message": "记忆删除成功"}

    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(f"删除记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail="记忆删除失败")


@router.get("/history", summary="获取记忆历史")
async def get_memory_history(
    request: Request,
    limit: int = Query(100, ge=1, le=1000, description="返回结果数量限制"),
    user_id: Optional[str] = Query(None, description="用户ID")
):
    """
    获取记忆历史

    Args:
        request: FastAPI请求对象
        limit: 返回结果数量限制
        user_id: 用户ID（可选）

    Returns:
        Dict: 记忆历史列表
    """
    try:
        # 获取全局服务实例
        memory_service = getattr(request.app.state, 'memory_service', None)
        if not memory_service:
            raise HTTPException(status_code=503, detail="记忆服务不可用")

        # 记录请求信息
        structured_logger.info(
            "获取记忆历史请求",
            extra={
                "limit": limit,
                "user_id": user_id,
            }
        )

        # 获取记忆历史
        memories = await memory_service.get_history(limit=limit, user_id=user_id)

        # 转换为响应模型
        results = []
        for memory in memories:
            memory_response = MemoryResponse(
                memory_id=memory.get('memory_id', ''),
                content=memory.get('content', ''),
                metadata=memory.get('metadata', {}),
                created_at=memory.get('created_at', ''),
                updated_at=memory.get('updated_at')
            )
            results.append(memory_response)

        structured_logger.info(
            "记忆历史获取完成",
            extra={
                "results_count": len(results),
                "user_id": user_id,
            }
        )

        return {"success": True, "data": results, "total": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(f"获取记忆历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取记忆历史失败")


@router.get("/stats", response_model=MemoryStatsResponse, summary="获取记忆统计")
async def get_memory_stats(request: Request, user_id: Optional[str] = Query(None, description="用户ID")):
    """
    获取记忆统计

    Args:
        request: FastAPI请求对象
        user_id: 用户ID（可选）

    Returns:
        MemoryStatsResponse: 记忆统计信息
    """
    try:
        # 获取全局服务实例
        memory_service = getattr(request.app.state, 'memory_service', None)
        if not memory_service:
            raise HTTPException(status_code=503, detail="记忆服务不可用")

        # 记录请求信息
        structured_logger.info(
            "获取记忆统计请求",
            extra={"user_id": user_id}
        )

        # 获取记忆统计
        stats = await memory_service.get_stats(user_id=user_id)

        structured_logger.info("记忆统计获取完成")

        return MemoryStatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(f"获取记忆统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取记忆统计失败")


@router.delete("/clear", response_model=Dict[str, Any], summary="清除所有记忆")
async def clear_all_memories(request: Request, user_id: Optional[str] = Query(None, description="用户ID")):
    """
    清除所有记忆

    Args:
        request: FastAPI请求对象
        user_id: 用户ID（可选，如果不提供则清除所有用户的记忆）

    Returns:
        Dict: 清除结果
    """
    try:
        # 获取全局服务实例
        memory_service = getattr(request.app.state, 'memory_service', None)
        if not memory_service:
            raise HTTPException(status_code=503, detail="记忆服务不可用")

        # 记录请求信息
        structured_logger.warning(
            "清除记忆请求",
            extra={
                "user_id": user_id,
                "operation": "clear_memories"
            }
        )

        # 清除记忆
        cleared_count = await memory_service.clear_memories(user_id=user_id)

        structured_logger.warning(
            "记忆清除完成",
            extra={
                "user_id": user_id,
                "cleared_count": cleared_count
            }
        )

        return {
            "success": True,
            "message": f"已清除 {cleared_count} 条记忆",
            "cleared_count": cleared_count
        }

    except HTTPException:
        raise
    except Exception as e:
        structured_logger.error(f"清除记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail="清除记忆失败")