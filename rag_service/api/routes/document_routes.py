"""
文档管理路由

提供文档上传、处理、搜索、删除等管理功能的API接口。
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path, File, UploadFile
from fastapi.responses import StreamingResponse
import logging
import json

from ..dependencies import get_rag_service, get_current_user, get_request_context, validate_request_size
from ...services.unified_rag_service import UnifiedRAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{kb_id}/documents", response_model=Dict[str, Any])
async def upload_document(
    kb_id: str = Path(..., description="知识库ID"),
    title: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
    file_path: Optional[str] = Body(None),
    file_type: Optional[str] = Body(None),
    metadata: Optional[Dict[str, Any]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    上传文档到知识库。

    Args:
        kb_id: 知识库ID
        title: 文档标题
        content: 文档内容
        file_path: 文件路径
        file_type: 文件类型
        metadata: 元数据
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 上传结果
    """
    try:
        tenant_id = current_user.get("tenant_id")
        created_by = current_user.get("user_id")

        logger.info(
            f"上传文档 - 知识库: {kb_id}, "
            f"标题: {title}, 用户: {created_by}"
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权向此知识库添加文档")

        # 添加文档
        doc_id = await rag_service.add_document_to_kb(
            kb_id=kb_id,
            title=title,
            content=content,
            file_path=file_path,
            file_type=file_type,
            metadata=metadata,
            created_by=created_by
        )

        return {
            "success": True,
            "data": {
                "document_id": doc_id,
                "kb_id": kb_id,
                "title": title,
                "file_type": file_type,
                "created_by": created_by
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传文档失败: {str(e)}")


@router.post("/{kb_id}/documents/file", response_model=Dict[str, Any])
async def upload_file_document(
    kb_id: str = Path(..., description="知识库ID"),
    title: Optional[str] = Body(None),
    file: UploadFile = File(...),
    metadata: Optional[str] = Body(None, description="JSON格式的元数据"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    上传文件文档到知识库。

    Args:
        kb_id: 知识库ID
        title: 文档标题（可选，默认使用文件名）
        file: 上传的文件
        metadata: JSON格式的元数据
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 上传结果
    """
    try:
        tenant_id = current_user.get("tenant_id")
        created_by = current_user.get("user_id")

        # 验证文件大小
        await validate_request_size(file, max_size=50 * 1024 * 1024)  # 50MB

        logger.info(
            f"上传文件文档 - 知识库: {kb_id}, "
            f"文件: {file.filename}, 用户: {created_by}"
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权向此知识库添加文档")

        # 读取文件内容
        content = await file.read()
        file_content = content.decode('utf-8', errors='ignore')

        # 解析元数据
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("元数据JSON解析失败，使用空元数据")

        # 添加文件相关元数据
        doc_metadata.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content)
        })

        # 使用文件名作为默认标题
        doc_title = title or file.filename or "未命名文档"

        # 添加文档
        doc_id = await rag_service.add_document_to_kb(
            kb_id=kb_id,
            title=doc_title,
            content=file_content,
            file_path=file.filename,
            file_type=file.content_type,
            metadata=doc_metadata,
            created_by=created_by
        )

        return {
            "success": True,
            "data": {
                "document_id": doc_id,
                "kb_id": kb_id,
                "title": doc_title,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "created_by": created_by
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传文件文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传文件文档失败: {str(e)}")


@router.get("/{kb_id}/documents", response_model=Dict[str, Any])
async def list_documents(
    kb_id: str = Path(..., description="知识库ID"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    file_type: Optional[str] = Query(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取知识库中的文档列表。

    Args:
        kb_id: 知识库ID
        limit: 返回数量限制
        offset: 偏移量
        search: 搜索关键词
        file_type: 文件类型过滤
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 文档列表
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(
            f"获取文档列表 - 知识库: {kb_id}, "
            f"搜索: {search or '无'}, 类型: {file_type or '无'}"
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权访问此知识库")

        # 搜索文档（这里需要实现相应的方法）
        # 暂时返回空列表
        documents = []

        # 如果有搜索条件，执行搜索
        if search:
            search_results = await rag_service.search_documents(
                kb_id=kb_id,
                query=search,
                top_k=limit,
                filters={"file_type": file_type} if file_type else None
            )
            documents = search_results
        else:
            # 否则返回文档列表（这里需要实现列表方法）
            documents = []

        return {
            "success": True,
            "data": {
                "kb_id": kb_id,
                "documents": documents,
                "total": len(documents),
                "limit": limit,
                "offset": offset,
                "filters": {
                    "search": search,
                    "file_type": file_type
                }
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.get("/{kb_id}/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(
    kb_id: str = Path(..., description="知识库ID"),
    document_id: str = Path(..., description="文档ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取指定文档信息。

    Args:
        kb_id: 知识库ID
        document_id: 文档ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 文档信息
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(
            f"获取文档信息 - 知识库: {kb_id}, "
            f"文档: {document_id}, 租户: {tenant_id}"
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权访问此知识库")

        # 获取文档信息（这里需要实现相应的方法）
        # 暂时返回模拟数据
        document = {
            "document_id": document_id,
            "kb_id": kb_id,
            "title": "示例文档",
            "content": "文档内容...",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }

        return {
            "success": True,
            "data": document,
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档信息失败: {str(e)}")


@router.delete("/{kb_id}/documents/{document_id}", response_model=Dict[str, Any])
async def delete_document(
    kb_id: str = Path(..., description="知识库ID"),
    document_id: str = Path(..., description="文档ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    删除文档。

    Args:
        kb_id: 知识库ID
        document_id: 文档ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 删除结果
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(
            f"删除文档 - 知识库: {kb_id}, "
            f"文档: {document_id}, 租户: {tenant_id}"
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权删除此知识库的文档")

        # 删除文档（这里需要实现相应的方法）
        # 暂时返回成功
        success = True

        return {
            "success": success,
            "message": "文档删除成功" if success else "文档删除失败",
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@router.post("/{kb_id}/documents/{document_id}/process", response_model=Dict[str, Any])
async def process_document(
    kb_id: str = Path(..., description="知识库ID"),
    document_id: str = Path(..., description="文档ID"),
    force_reprocess: bool = Body(False, embed=True),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    重新处理文档。

    Args:
        kb_id: 知识库ID
        document_id: 文档ID
        force_reprocess: 是否强制重新处理
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(
            f"处理文档 - 知识库: {kb_id}, "
            f"文档: {document_id}, 强制: {force_reprocess}"
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权处理此知识库的文档")

        # 处理文档（这里需要实现相应的方法）
        # 暂时返回模拟结果
        process_result = {
            "document_id": document_id,
            "status": "processing",
            "chunks_created": 0,
            "processing_time": 0.0
        }

        return {
            "success": True,
            "data": process_result,
            "message": "文档处理已启动",
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理文档失败: {str(e)}")


@router.get("/{kb_id}/documents/{document_id}/chunks", response_model=Dict[str, Any])
async def get_document_chunks(
    kb_id: str = Path(..., description="知识库ID"),
    document_id: str = Path(..., description="文档ID"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取文档的分块信息。

    Args:
        kb_id: 知识库ID
        document_id: 文档ID
        limit: 返回数量限制
        offset: 偏移量
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 文档分块列表
    """
    try:
        tenant_id = current_user.get("tenant_id")

        logger.info(
            f"获取文档分块 - 知识库: {kb_id}, "
            f"文档: {document_id}, 租户: {tenant_id}"
        )

        # 验证知识库存在和权限
        kb = await rag_service.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")

        if kb.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="无权访问此知识库")

        # 获取文档分块（这里需要实现相应的方法）
        # 暂时返回空列表
        chunks = []

        return {
            "success": True,
            "data": {
                "document_id": document_id,
                "kb_id": kb_id,
                "chunks": chunks,
                "total": len(chunks),
                "limit": limit,
                "offset": offset
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档分块失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档分块失败: {str(e)}")