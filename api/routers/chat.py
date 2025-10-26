"""
对话API路由
提供智能问答和对话管理功能
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from infrastructure.monitoring.loguru_logger import logger
from rag_service.services.chat_service_adapter import ChatService
from api.exceptions import ValidationError, ServiceUnavailableError

router = APIRouter()


class ChatMessage(BaseModel):
    """对话消息模型"""
    role: str = Field(..., description="消息角色：system/user/assistant")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[str] = Field(None, description="消息时间戳")


class ChatRequest(BaseModel):
    """对话请求模型"""
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    model: Optional[str] = Field("default", description="使用的模型名称")
    temperature: Optional[float] = Field(0.7, description="生成温度", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2000, description="最大token数", ge=1, le=8000)
    stream: Optional[bool] = Field(False, description="是否流式返回")
    knowledge_base_id: Optional[str] = Field(None, description="知识库ID")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="检索配置")


class ChatResponse(BaseModel):
    """对话响应模型"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class StreamChatChunk(BaseModel):
    """流式对话响应块模型"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


@router.post("/completions", response_model=ChatResponse, summary="对话完成")
async def chat_completions(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    生成对话回复，兼容OpenAI chat/completions接口

    Args:
        request: 对话请求
        background_tasks: 后台任务

    Returns:
        ChatResponse: 对话响应
    """
    logger.info(f"收到对话请求，消息数量: {len(request.messages)}")

    try:
        # 验证请求参数
        if not request.messages:
            raise ValidationError("消息列表不能为空")

        # 获取最后一条用户消息
        user_message = None
        for message in reversed(request.messages):
            if message.role == "user":
                user_message = message.content
                break

        if not user_message:
            raise ValidationError("未找到用户消息")

        # 创建对话服务
        chat_service = ChatService()

        # 处理对话
        if request.stream:
            # 流式响应
            return await _handle_stream_chat(request, chat_service)
        else:
            # 非流式响应
            return await _handle_normal_chat(request, chat_service, background_tasks)

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"对话处理失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"对话服务暂时不可用: {str(e)}")


@router.post("/completions/stream", summary="流式对话完成")
async def chat_completions_stream(request: ChatRequest):
    """
    生成流式对话回复

    Args:
        request: 对话请求

    Returns:
        StreamingResponse: 流式响应
    """
    logger.info(f"收到流式对话请求，消息数量: {len(request.messages)}")

    try:
        # 强制设置为流式模式
        request.stream = True

        # 验证请求参数
        if not request.messages:
            raise ValidationError("消息列表不能为空")

        # 获取最后一条用户消息
        user_message = None
        for message in reversed(request.messages):
            if message.role == "user":
                user_message = message.content
                break

        if not user_message:
            raise ValidationError("未找到用户消息")

        # 创建对话服务
        chat_service = ChatService()

        # 返回流式响应
        return await _handle_stream_chat(request, chat_service)

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"流式对话处理失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"流式对话服务暂时不可用: {str(e)}")


@router.get("/conversations/{conversation_id}", summary="获取对话历史")
async def get_conversation_history(conversation_id: str):
    """
    获取指定对话的历史记录

    Args:
        conversation_id: 对话ID

    Returns:
        Dict: 对话历史信息
    """
    logger.info(f"获取对话历史: {conversation_id}")

    try:
        chat_service = ChatService()
        history = await chat_service.get_conversation_history(conversation_id)

        if not history:
            raise HTTPException(status_code=404, detail="对话不存在")

        return {
            "conversation_id": conversation_id,
            "messages": history,
            "total_messages": len(history)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取对话历史失败: {str(e)}")


@router.delete("/conversations/{conversation_id}", summary="删除对话")
async def delete_conversation(conversation_id: str):
    """
    删除指定对话及其历史记录

    Args:
        conversation_id: 对话ID

    Returns:
        Dict: 删除结果
    """
    logger.info(f"删除对话: {conversation_id}")

    try:
        chat_service = ChatService()
        success = await chat_service.delete_conversation(conversation_id)

        if not success:
            raise HTTPException(status_code=404, detail="对话不存在")

        return {
            "success": True,
            "message": "对话删除成功",
            "conversation_id": conversation_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"删除对话失败: {str(e)}")


async def _handle_normal_chat(
    request: ChatRequest,
    chat_service: ChatService,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """处理非流式对话"""
    import time
    import uuid

    start_time = time.time()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    try:
        # 调用对话服务
        response_content, usage = await chat_service.generate_response(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=conversation_id,
            retrieval_config=request.retrieval_config
        )

        # 添加后台任务记录对话日志
        background_tasks.add_task(
            _log_chat_completion,
            conversation_id,
            request.messages[-1].content if request.messages else "",
            response_content,
            usage,
            time.time() - start_time
        )

        # 构建响应
        return ChatResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }],
            usage=usage
        )

    except Exception as e:
        logger.error(f"非流式对话处理失败: {str(e)}", exc_info=True)
        raise


async def _handle_stream_chat(request: ChatRequest, chat_service: ChatService) -> StreamingResponse:
    """处理流式对话"""
    import uuid

    conversation_id = request.conversation_id or str(uuid.uuid4())

    async def generate_stream():
        """生成流式响应"""
        try:
            # 调用流式对话服务
            async for chunk in chat_service.generate_stream_response(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                knowledge_base_id=request.knowledge_base_id,
                conversation_id=conversation_id,
                retrieval_config=request.retrieval_config
            ):
                yield f"data: {chunk}\n\n"

            # 发送结束标记
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"流式对话生成失败: {str(e)}", exc_info=True)
            error_chunk = {
                "error": {
                    "message": f"流式对话生成失败: {str(e)}",
                    "type": "service_error"
                }
            }
            yield f"data: {error_chunk}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


async def _log_chat_completion(
    conversation_id: str,
    user_message: str,
    assistant_response: str,
    usage: Dict[str, int],
    process_time: float
):
    """记录对话完成日志（后台任务）"""
    try:
        logger.info(
            "对话完成",
            extra={
                "conversation_id": conversation_id,
                "user_message_length": len(user_message),
                "assistant_response_length": len(assistant_response),
                "usage": usage,
                "process_time": round(process_time, 2)
            }
        )
    except Exception as e:
        logger.error(f"记录对话日志失败: {str(e)}")