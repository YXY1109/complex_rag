"""
统一聊天路由
迁移Sanic RAG服务的LLM功能到FastAPI
"""

import time
from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.loguru_config import get_logger

# 创建路由器
router = APIRouter()
structured_logger = get_logger("api.unified_chat")


# Pydantic模型定义
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    name: Optional[str] = Field(None, description="发送者名称")
    function_call: Optional[Dict[str, Any]] = Field(None, description="函数调用")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="工具调用")
    tool_call_id: Optional[str] = Field(None, description="工具调用ID")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="使用的模型")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    temperature: Optional[float] = Field(1.0, ge=0, le=2, description="温度参数")
    max_tokens: Optional[int] = Field(None, ge=1, description="最大生成token数")
    top_p: Optional[float] = Field(1.0, ge=0, le=1, description="top-p采样")
    stream: Optional[bool] = Field(False, description="是否流式返回")
    stop: Optional[Union[str, List[str]]] = Field(None, description="停止词")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="存在惩罚")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="频率惩罚")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    completion_request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    OpenAI兼容的聊天完成接口

    Args:
        request: FastAPI请求对象
        completion_request: 聊天完成请求

    Returns:
        ChatCompletionResponse: 聊天完成响应
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    try:
        structured_logger.info(
            "聊天完成请求",
            extra={
                "request_id": request_id,
                "model": completion_request.model,
                "messages_count": len(completion_request.messages),
                "stream": completion_request.stream,
                "max_tokens": completion_request.max_tokens,
                "temperature": completion_request.temperature,
            }
        )

        # TODO: 集成实际的LLM服务
        # 这里暂时返回模拟响应
        response = await _create_mock_completion(completion_request, request_id, start_time)

        return response

    except Exception as e:
        structured_logger.error(
            f"聊天完成请求处理失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred",
                "type": "internal_server_error",
                "code": "internal_error",
                "request_id": request_id,
            }
        )


@router.post("/completions/stream")
async def chat_completions_stream(
    request: Request,
    completion_request: ChatCompletionRequest
) -> StreamingResponse:
    """
    流式聊天完成接口

    Args:
        request: FastAPI请求对象
        completion_request: 聊天完成请求

    Returns:
        StreamingResponse: 流式响应
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        structured_logger.info(
            "流式聊天完成请求",
            extra={
                "request_id": request_id,
                "model": completion_request.model,
                "messages_count": len(completion_request.messages),
            }
        )

        # TODO: 集成实际的LLM流式服务
        async def generate_mock_stream():
            """生成模拟流式响应"""
            response_id = f"chatcmpl-{completion_request.model}-{int(time.time())}"
            created = int(time.time())

            # 模拟流式响应块
            mock_content = "This is a mock streaming response from the unified FastAPI service."

            for i, char in enumerate(mock_content):
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": completion_request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": char},
                            "finish_reason": None,
                        }
                    ]
                }

                yield f"data: {__import__('json').dumps(chunk_data, ensure_ascii=False)}\n\n"
                await __import__('asyncio').sleep(0.05)  # 模拟延迟

            # 发送结束标记
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_mock_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )

    except Exception as e:
        structured_logger.error(
            f"流式聊天完成错误: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "type": "internal_server_error",
                "code": "stream_error",
                "request_id": request_id,
            }
        )


@router.get("/models", response_model=ModelsListResponse)
async def list_models(request: Request) -> ModelsListResponse:
    """
    列出可用的模型

    Args:
        request: FastAPI请求对象

    Returns:
        ModelsListResponse: 模型列表响应
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        # TODO: 从实际LLM服务获取模型列表
        # 这里返回模拟模型列表
        mock_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "qwen2.5:7b",
            "qwen2.5:14b",
            "claude-3-sonnet",
        ]

        models = []
        for model_name in mock_models:
            model_info = ModelInfo(
                id=model_name,
                created=int(time.time()),
                owned_by="unified-rag-service"
            )
            models.append(model_info)

        structured_logger.info(
            "模型列表请求",
            extra={
                "request_id": request_id,
                "models_count": len(models),
            }
        )

        return ModelsListResponse(data=models)

    except Exception as e:
        structured_logger.error(
            f"获取模型列表失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to retrieve models",
                "type": "internal_server_error",
                "code": "models_error",
                "request_id": request_id,
            }
        )


async def _create_mock_completion(
    completion_request: ChatCompletionRequest,
    request_id: str,
    start_time: float
) -> ChatCompletionResponse:
    """
    创建模拟聊天完成响应

    Args:
        completion_request: 聊天完成请求
        request_id: 请求ID
        start_time: 开始时间

    Returns:
        ChatCompletionResponse: 模拟响应
    """
    processing_time = time.time() - start_time

    # 获取最后一条用户消息
    last_message = ""
    for message in reversed(completion_request.messages):
        if message.role == "user":
            last_message = message.content
            break

    # 生成模拟回复
    mock_response = f"This is a mock response from the unified FastAPI service. Your message was: '{last_message[:100]}{'...' if len(last_message) > 100 else ''}'."

    # 构造响应
    response = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=completion_request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=mock_response),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(last_message.split()),
            completion_tokens=len(mock_response.split()),
            total_tokens=len(last_message.split()) + len(mock_response.split())
        ),
        system_fingerprint="unified-service-1.0"
    )

    structured_logger.info(
        "聊天完成完成",
        extra={
            "request_id": request_id,
            "model": response.model,
            "choices_count": len(response.choices),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "processing_time_seconds": round(processing_time, 3),
        }
    )

    return response


structured_logger.info("统一聊天路由加载完成")