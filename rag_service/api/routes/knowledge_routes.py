"""
知识库管理路由

提供知识库创建、查询、更新、删除等管理功能的API接口。
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
import logging

from ..dependencies import get_rag_service, get_current_user, get_request_context, get_tenant_context
from ...interfaces.rag_interface import KnowledgeBase
from ...services.unified_rag_service import UnifiedRAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=Dict[str, Any])
async def create_knowledge_base(
    name: str = Body(..., embed=True),
    description: str = Body(..., embed=True),
    config: Optional[Dict[str, Any]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    创建知识库。

    Args:
        name: 知识库名称
        description: 知识库描述
        config: 知识库配置
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 创建的知识库信息
    """
    try:
        tenant_id = current_user.get("tenant_id")
        created_by = current_user.get("user_id")

        logger.info(
            f"创建知识库 - 租户: {tenant_id}, "
            f"名称: {name}, 创建者: {created_by}"
        )

        # 创建知识库
        kb = await rag_service.create_knowledge_base(
            name=name,
            description=description,
            tenant_id=tenant_id,
            config=config,
            created_by=created_by
        )

        return {
            "success": True,
            "data": {
                "kb_id": kb.kb_id,
                "name": kb.name,
                "description": kb.description,
                "tenant_id": kb.tenant_id,
                "config": kb.config,
                "created_by": kb.created_by,
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat(),
                "status": kb.status
            },
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"创建知识库失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建知识库失败: {str(e)}")


@router.get("/", response_model=Dict[str, Any])
async def list_knowledge_bases(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_context),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取知识库列表。

    Args:
        limit: 返回数量限制
        offset: 偏移量
        search: 搜索关键词
        rag_service: RAG服务实例
        tenant_id: 租户ID
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 知识库列表
    """
    try:
        logger.info(
            f"获取知识库列表 - 租户: {tenant_id}, "
            f"搜索: {search or '无'}"
        )

        # 获取知识库列表
        kbs = await rag_service.list_knowledge_bases(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset
        )

        # 如果有搜索关键词，进行过滤
        if search:
            kbs = [kb for kb in kbs if search.lower() in kb.name.lower() or search.lower() in kb.description.lower()]

        return {
            "success": True,
            "data": {
                "knowledge_bases": [
                    {
                        "kb_id": kb.kb_id,
                        "name": kb.name,
                        "description": kb.description,
                        "tenant_id": kb.tenant_id,
                        "created_by": kb.created_by,
                        "created_at": kb.created_at.isoformat(),
                        "updated_at": kb.updated_at.isoformat(),
                        "status": kb.status,
                        "document_count": kb.document_count if hasattr(kb, 'document_count') else 0
                    }
                    for kb in kbs
                ],
                "total": len(kbs),
                "limit": limit,
                "offset": offset
            },
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"获取知识库列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取知识库列表失败: {str(e)}")


@router.get("/{kb_id}", response_model=Dict[str, Any])
async def get_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取指定知识库信息。

    Args:
        kb_id: 知识库ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 知识库信息
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(f"获取知识库信息 - ID: {kb_id}, 租户: {tenant_id}")

        # 获取知识库信息
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        # 验证权限
        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权访问此知识库")

        return {
            "success": True,
            "data": {
                "kb_id": kb.kb_id,
                "name": kb.name,
                "description": kb.description,
                "tenant_id": kb.tenant_id,
                "config": kb.config,
                "created_by": kb.created_by,
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat(),
                "status": kb.status,
                "document_count": kb.document_count if hasattr(kb, 'document_count') else 0
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取知识库信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取知识库信息失败: {str(e)}")


@router.put("/{kb_id}", response_model=Dict[str, Any])
async def update_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    name: Optional[str] = Body(None),
    description: Optional[str] = Body(None),
    config: Optional[Dict[str, Any]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    更新知识库信息。

    Args:
        kb_id: 知识库ID
        name: 新名称
        description: 新描述
        config: 新配置
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 更新结果
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(f"更新知识库 - ID: {kb_id}, 租户: {tenant_id}")

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权修改此知识库")

        # 准备更新数据
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if config is not None:
            updates["config"] = config

        if not updates:
            raise HTTPException(status_code=400, detail="没有提供更新数据")

        # 更新知识库
        success = await rag_service.update_knowledge_base(kb_id, updates)

        return {
            "success": success,
            "message": "知识库更新成功" if success else "知识库更新失败",
            "updated_fields": list(updates.keys()),
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新知识库失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新知识库失败: {str(e)}")


@router.delete("/{kb_id}", response_model=Dict[str, Any])
async def delete_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    删除知识库。

    Args:
        kb_id: 知识库ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 删除结果
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(f"删除知识库 - ID: {kb_id}, 租户: {tenant_id}")

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权删除此知识库")

        # 删除知识库
        success = await rag_service.delete_knowledge_base(kb_id)

        return {
            "success": success,
            "message": "知识库删除成功" if success else "知识库删除失败",
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识库失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除知识库失败: {str(e)}")


@router.get("/{kb_id}/statistics", response_model=Dict[str, Any])
async def get_knowledge_base_statistics(
    kb_id: str = Path(..., description="知识库ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取知识库统计信息。

    Args:
        kb_id: 知识库ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 统计信息
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(f"获取知识库统计 - ID: {kb_id}, 租户: {tenant_id}")

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权访问此知识库")

        # 获取统计信息（这里需要实现相应的方法）
        # 暂时返回模拟数据
        statistics = {
            "document_count": 0,
            "chunk_count": 0,
            "total_size": 0,
            "last_updated": kb.updated_at.isoformat(),
            "embedding_model": kb.config.get("embedding_model", "unknown") if kb.config else "unknown"
        }

        return {
            "success": True,
            "data": statistics,
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取知识库统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.post("/{kb_id}/search", response_model=Dict[str, Any])
async def search_knowledge_base(
    kb_id: str = Path(..., description="知识库ID"),
    query: str = Body(..., embed=True),
    top_k: int = Body(10, ge=1, le=50),
    filters: Optional[Dict[str, Any]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    在知识库中搜索文档。

    Args:
        kb_id: 知识库ID
        query: 搜索查询
        top_k: 返回结果数量
        filters: 过滤条件
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 搜索结果
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(
            f"搜索知识库 - ID: {kb_id}, 租户: {tenant_id}, "
            f"查询: {query[:50]}..."
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权搜索此知识库")

        # 执行搜索
        results = await rag_service.search_documents(
            kb_id=kb_id,
            query=query,
            top_k=top_k,
            filters=filters
        )

        return {
            "success": True,
            "data": {
                "kb_id": kb_id,
                "query": query,
                "results": results,
                "total_found": len(results),
                "top_k": top_k
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索知识库失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")