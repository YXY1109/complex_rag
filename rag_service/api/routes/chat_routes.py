"""
聊天功能路由

提供聊天会话管理、消息发送、对话历史等功能的API接口。
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
import logging

from ..dependencies import get_rag_service, get_current_user, get_request_context
from ...interfaces.rag_interface import ChatSession
from ...services.unified_rag_service import UnifiedRAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sessions", response_model=Dict[str, Any])
async def create_chat_session(
    title: Optional[str] = Body(None),
    knowledge_bases: Optional[List[str]] = Body(None),
    config: Optional[Dict[str, Any]] = Body(None),
    metadata: Optional[Dict[str, Any]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    创建聊天会话。

    Args:
        title: 会话标题
        knowledge_bases: 关联的知识库列表
        config: 会话配置
        metadata: 元数据
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 创建的会话信息
    """
    try:
        user_id = current_user.get("user_id")
        tenant_id = current_user.get("tenant_id")

        logger.info(
            f"创建聊天会话 - 用户: {user_id}, "
            f"标题: {title or '未命名会话'}"
        )

        # 创建会话
        session = await rag_service.create_session(
            user_id=user_id,
            tenant_id=tenant_id,
            title=title,
            knowledge_bases=knowledge_bases,
            config=config,
            metadata=metadata
        )

        return {
            "success": True,
            "data": {
                "session_id": session.session_id,
                "title": session.title,
                "user_id": session.user_id,
                "tenant_id": session.tenant_id,
                "knowledge_bases": session.knowledge_bases,
                "config": session.config,
                "metadata": session.metadata,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            },
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"创建聊天会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@router.get("/sessions", response_model=Dict[str, Any])
async def list_chat_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取用户的聊天会话列表。

    Args:
        limit: 返回数量限制
        offset: 偏移量
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 会话列表
    """
    try:
        user_id = current_user.get("user_id")
        tenant_id = current_user.get("tenant_id")

        logger.info(f"获取聊天会话列表 - 用户: {user_id}")

        # 获取会话列表（这里需要实现相应的方法）
        # 暂时返回模拟数据
        sessions = []

        return {
            "success": True,
            "data": {
                "sessions": sessions,
                "total": len(sessions),
                "limit": limit,
                "offset": offset
            },
            "request_id": request_context.get("request_id")
        }

    except Exception as e:
        logger.error(f"获取聊天会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_chat_session(
    session_id: str = Path(..., description="会话ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取指定聊天会话信息。

    Args:
        session_id: 会话ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 会话信息
    """
    try:
        user_id = current_user.get("user_id")

        logger.info(f"获取聊天会话信息 - 会话ID: {session_id}, 用户: {user_id}")

        # 获取会话信息
        session = await rag_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 验证权限
        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此会话")

        return {
            "success": True,
            "data": {
                "session_id": session.session_id,
                "title": session.title,
                "user_id": session.user_id,
                "tenant_id": session.tenant_id,
                "knowledge_bases": session.knowledge_bases,
                "config": session.config,
                "metadata": session.metadata,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取聊天会话信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话信息失败: {str(e)}")


@router.post("/sessions/{session_id}/chat", response_model=Dict[str, Any])
async def send_chat_message(
    session_id: str = Path(..., description="会话ID"),
    message: str = Body(..., embed=True),
    config: Optional[Dict[str, Any]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    发送聊天消息。

    Args:
        session_id: 会话ID
        message: 消息内容
        config: 消息配置
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 聊天响应
    """
    try:
        user_id = current_user.get("user_id")

        logger.info(
            f"发送聊天消息 - 会话ID: {session_id}, "
            f"用户: {user_id}, 消息: {message[:100]}..."
        )

        # 验证会话存在和权限
        session = await rag_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此会话")

        # 发送消息
        result = await rag_service.chat(
            session_id=session_id,
            message=message,
            config=config
        )

        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "message": message,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"发送聊天消息失败: {e}")
        raise HTTPException(status_code=500, detail=f"发送消息失败: {str(e)}")


@router.post("/sessions/{session_id}/chat/stream")
async def send_chat_message_stream(
    session_id: str = Path(..., description="会话ID"),
    message: str = Body(..., embed=True),
    config: Optional[Dict[str, Any]] = Body(None),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    流式聊天消息。

    Args:
        session_id: 会话ID
        message: 消息内容
        config: 消息配置
        rag_service: RAG服务实例
        current_user: 当前用户信息

    Returns:
        StreamingResponse: 流式响应
    """
    try:
        user_id = current_user.get("user_id")

        # 验证会话存在和权限
        session = await rag_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此会话")

        async def generate():
            """生成流式响应。"""
            try:
                # 这里需要实现流式聊天方法
                # 暂时使用普通聊天方法
                result = await rag_service.chat(
                    session_id=session_id,
                    message=message,
                    config=config
                )

                yield f"data: {{\"type\": \"answer\", \"content\": {result.answer.json()}}}\n\n"
                yield f"data: {{\"type\": \"done\", \"total_time\": {result.total_time}}}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"流式聊天错误: {e}")
                yield f"data: {{\"type\": \"error\", \"error\": \"{str(e)}\"}}\n\n"

        from fastapi.responses import StreamingResponse

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流式聊天消息失败: {e}")
        raise HTTPException(status_code=500, detail=f"流式聊天失败: {str(e)}")


@router.delete("/sessions/{session_id}", response_model=Dict[str, Any])
async def delete_chat_session(
    session_id: str = Path(..., description="会话ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    删除聊天会话。

    Args:
        session_id: 会话ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 删除结果
    """
    try:
        user_id = current_user.get("user_id")

        logger.info(f"删除聊天会话 - 会话ID: {session_id}, 用户: {user_id}")

        # 验证会话存在和权限
        session = await rag_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权删除此会话")

        # 删除会话
        success = await rag_service.delete_session(session_id)

        return {
            "success": success,
            "message": "会话删除成功" if success else "会话删除失败",
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除聊天会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")


@router.get("/sessions/{session_id}/history", response_model=Dict[str, Any])
async def get_chat_history(
    session_id: str = Path(..., description="会话ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取聊天历史记录。

    Args:
        session_id: 会话ID
        limit: 返回数量限制
        offset: 偏移量
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 聊天历史记录
    """
    try:
        user_id = current_user.get("user_id")

        logger.info(f"获取聊天历史 - 会话ID: {session_id}, 用户: {user_id}")

        # 验证会话存在和权限
        session = await rag_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此会话")

        # 获取历史记录（这里需要实现相应的方法）
        # 暂时返回空列表
        messages = []

        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "messages": messages,
                "total": len(messages),
                "limit": limit,
                "offset": offset
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取聊天历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取聊天历史失败: {str(e)}")


@router.get("/sessions/{session_id}/summary", response_model=Dict[str, Any])
async def get_chat_summary(
    session_id: str = Path(..., description="会话ID"),
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    获取会话摘要。

    Args:
        session_id: 会话ID
        rag_service: RAG服务实例
        current_user: 当前用户信息
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 会话摘要
    """
    try:
        user_id = current_user.get("user_id")

        logger.info(f"获取会话摘要 - 会话ID: {session_id}, 用户: {user_id}")

        # 验证会话存在和权限
        session = await rag_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问此会话")

        # 获取会话摘要
        summary = await rag_service.chat_service.get_session_summary(session_id)

        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "summary": summary.summary if summary else None,
                "key_topics": summary.key_topics if summary else [],
                "message_count": summary.message_count if summary else 0,
                "created_at": summary.created_at.isoformat() if summary else None
            },
            "request_id": request_context.get("request_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话摘要失败: {str(e)}")