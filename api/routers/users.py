"""
用户管理API路由
提供用户信息和会话管理功能
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query

from infrastructure.monitoring.loguru_logger import logger
from api.exceptions import ValidationError, NotFoundError, ServiceUnavailableError

router = APIRouter()


class UserInfo(BaseModel):
    """用户信息模型"""
    id: str
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    status: str
    created_at: str
    last_login: Optional[str] = None
    preferences: Dict[str, Any] = {}


class UserCreate(BaseModel):
    """用户创建请求模型"""
    username: str = Field(..., description="用户名", min_length=3, max_length=50)
    email: Optional[str] = Field(None, description="邮箱地址")
    display_name: Optional[str] = Field(None, description="显示名称")
    password: str = Field(..., description="密码", min_length=6)


class UserUpdate(BaseModel):
    """用户更新请求模型"""
    email: Optional[str] = Field(None, description="邮箱地址")
    display_name: Optional[str] = Field(None, description="显示名称")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    preferences: Optional[Dict[str, Any]] = Field(None, description="用户偏好设置")


class UserSession(BaseModel):
    """用户会话模型"""
    id: str
    user_id: str
    title: str
    knowledge_base_id: Optional[str] = None
    created_at: str
    updated_at: str
    message_count: int
    status: str


class UserStats(BaseModel):
    """用户统计信息模型"""
    user_id: str
    total_conversations: int
    total_messages: int
    total_documents: int
    knowledge_bases_count: int
    last_activity: Optional[str] = None


@router.get("/me", response_model=UserInfo, summary="获取当前用户信息")
async def get_current_user():
    """
    获取当前用户的基本信息
    在无需认证的环境中，返回默认用户信息

    Returns:
        UserInfo: 当前用户信息
    """
    logger.info("获取当前用户信息")

    # 在无需认证的环境中，返回默认用户
    user_info = UserInfo(
        id="default-user",
        username="default_user",
        email="default@example.com",
        display_name="默认用户",
        status="active",
        created_at="2024-01-01T10:00:00Z",
        last_login="2024-01-01T10:00:00Z",
        preferences={
            "theme": "light",
            "language": "zh-CN",
            "default_model": "gpt-3.5-turbo"
        }
    )

    return user_info


@router.put("/me", response_model=UserInfo, summary="更新当前用户信息")
async def update_current_user(request: UserUpdate):
    """
    更新当前用户的信息

    Args:
        request: 用户更新请求

    Returns:
        UserInfo: 更新后的用户信息
    """
    logger.info("更新当前用户信息")

    # 模拟更新操作
    updated_user = UserInfo(
        id="default-user",
        username="default_user",
        email=request.email or "default@example.com",
        display_name=request.display_name or "默认用户",
        avatar_url=request.avatar_url,
        status="active",
        created_at="2024-01-01T10:00:00Z",
        last_login="2024-01-01T10:00:00Z",
        preferences=request.preferences or {
            "theme": "light",
            "language": "zh-CN",
            "default_model": "gpt-3.5-turbo"
        }
    )

    return updated_user


@router.get("/me/sessions", response_model=List[UserSession], summary="获取用户会话列表")
async def get_user_sessions(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="会话状态过滤")
):
    """
    获取当前用户的会话列表

    Args:
        page: 页码
        page_size: 每页数量
        status: 会话状态过滤

    Returns:
        List[UserSession]: 用户会话列表
    """
    logger.info(f"获取用户会话列表，页码: {page}")

    # 模拟返回会话列表
    sessions = []
    for i in range(min(page_size, 5)):
        session = UserSession(
            id=f"session_{i+1}",
            user_id="default-user",
            title=f"对话会话 {i+1}",
            knowledge_base_id=f"kb_{i+1}" if i % 2 == 0 else None,
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:30:00Z",
            message_count=10 + i * 2,
            status="active" if i % 2 == 0 else "archived"
        )
        sessions.append(session)

    return sessions


@router.get("/me/stats", response_model=UserStats, summary="获取用户统计信息")
async def get_user_stats():
    """
    获取当前用户的统计信息

    Returns:
        UserStats: 用户统计信息
    """
    logger.info("获取用户统计信息")

    stats = UserStats(
        user_id="default-user",
        total_conversations=15,
        total_messages=234,
        total_documents=45,
        knowledge_bases_count=8,
        last_activity="2024-01-01T10:00:00Z"
    )

    return stats


@router.post("/me/sessions", response_model=UserSession, summary="创建新会话")
async def create_user_session(
    title: str = Query(..., description="会话标题"),
    knowledge_base_id: Optional[str] = Query(None, description="关联的知识库ID")
):
    """
    创建新的用户会话

    Args:
        title: 会话标题
        knowledge_base_id: 关联的知识库ID

    Returns:
        UserSession: 创建的会话信息
    """
    logger.info(f"创建用户会话: {title}")

    import uuid
    from datetime import datetime

    session = UserSession(
        id=str(uuid.uuid4()),
        user_id="default-user",
        title=title,
        knowledge_base_id=knowledge_base_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        updated_at=datetime.utcnow().isoformat() + "Z",
        message_count=0,
        status="active"
    )

    return session


@router.delete("/me/sessions/{session_id}", summary="删除用户会话")
async def delete_user_session(session_id: str):
    """
    删除指定的用户会话

    Args:
        session_id: 会话ID

    Returns:
        Dict: 删除结果
    """
    logger.info(f"删除用户会话: {session_id}")

    # 模拟删除操作
    return {
        "success": True,
        "message": "会话删除成功",
        "session_id": session_id
    }


@router.get("/me/preferences", summary="获取用户偏好设置")
async def get_user_preferences():
    """
    获取当前用户的偏好设置

    Returns:
        Dict: 用户偏好设置
    """
    logger.info("获取用户偏好设置")

    preferences = {
        "theme": "light",
        "language": "zh-CN",
        "default_model": "gpt-3.5-turbo",
        "max_tokens": 2000,
        "temperature": 0.7,
        "auto_save_conversations": True,
        "show_timestamps": True,
        "enable_suggestions": True,
        "notification_settings": {
            "email_notifications": False,
            "browser_notifications": True
        }
    }

    return preferences


@router.put("/me/preferences", summary="更新用户偏好设置")
async def update_user_preferences(preferences: Dict[str, Any]):
    """
    更新当前用户的偏好设置

    Args:
        preferences: 新的偏好设置

    Returns:
        Dict: 更新后的偏好设置
    """
    logger.info("更新用户偏好设置")

    # 模拟更新操作
    updated_preferences = {
        "theme": preferences.get("theme", "light"),
        "language": preferences.get("language", "zh-CN"),
        "default_model": preferences.get("default_model", "gpt-3.5-turbo"),
        "max_tokens": preferences.get("max_tokens", 2000),
        "temperature": preferences.get("temperature", 0.7),
        "auto_save_conversations": preferences.get("auto_save_conversations", True),
        "show_timestamps": preferences.get("show_timestamps", True),
        "enable_suggestions": preferences.get("enable_suggestions", True),
        "notification_settings": preferences.get("notification_settings", {
            "email_notifications": False,
            "browser_notifications": True
        })
    }

    return updated_preferences


@router.get("/me/activity", summary="获取用户活动历史")
async def get_user_activity(
    limit: int = Query(50, ge=1, le=200, description="返回数量限制"),
    activity_type: Optional[str] = Query(None, description="活动类型过滤")
):
    """
    获取用户的活动历史记录

    Args:
        limit: 返回数量限制
        activity_type: 活动类型过滤

    Returns:
        Dict: 活动历史记录
    """
    logger.info(f"获取用户活动历史，限制: {limit}")

    # 模拟活动记录
    activities = []
    activity_types = ["chat", "document_upload", "knowledge_base_create", "model_test"]

    for i in range(min(limit, 20)):
        import uuid
        from datetime import datetime, timedelta

        activity = {
            "id": str(uuid.uuid4()),
            "type": activity_types[i % len(activity_types)],
            "description": f"用户活动 {i+1}",
            "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat() + "Z",
            "metadata": {
                "session_id": f"session_{i+1}" if i % 2 == 0 else None,
                "document_id": f"doc_{i+1}" if i % 3 == 0 else None
            }
        }
        activities.append(activity)

    return {
        "activities": activities,
        "total": len(activities),
        "has_more": len(activities) == limit
    }