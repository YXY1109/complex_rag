"""
LLM路由

提供OpenAI兼容的聊天完成接口。
"""

import time
import json as json_module
from typing import Dict, Any, Optional

from sanic import Blueprint, Request, Response
from sanic.response import json, stream

from ..interfaces.llm_interface import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatMessage,
    ChatRole
)
from ..exceptions import ValidationError, ModelError, NotFoundError
from ..infrastructure.monitoring.loguru_logger import get_logger


# 创建LLM蓝图
llm_router = Blueprint("llm", url_prefix="/chat")
structured_logger = get_logger("rag_service.llm")


@llm_router.post("/completions")
async def chat_completions(request: Request) -> Response:
    """
    OpenAI兼容的聊天完成接口

    Args:
        request: Sanic请求对象

    Returns:
        Response: 聊天完成响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")
    start_time = time.time()

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 验证请求
        completion_request = ChatCompletionRequest(**request_data)

        # 获取LLM服务
        service = request.app.ctx.service
        llm_service = service.get_llm_service()
        if not llm_service:
            raise NotFoundError("LLM service not available")

        # 验证请求
        await llm_service.validate_request(completion_request)

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

        # 处理流式或非流式请求
        if completion_request.stream:
            return await _handle_streaming_completion(
                request, completion_request, llm_service, request_id
            )
        else:
            return await _handle_non_streaming_completion(
                request, completion_request, llm_service, request_id, start_time
            )

    except ValidationError as e:
        structured_logger.warning(
            f"聊天完成请求验证失败: {e.message}",
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
            f"聊天完成模型错误: {e.message}",
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
            f"聊天完成请求处理失败: {e}",
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


async def _handle_non_streaming_completion(
    request: Request,
    completion_request: ChatCompletionRequest,
    llm_service,
    request_id: str,
    start_time: float
) -> Response:
    """
    处理非流式聊天完成请求

    Args:
        request: Sanic请求对象
        completion_request: 聊天完成请求
        llm_service: LLM服务实例
        request_id: 请求ID
        start_time: 开始时间

    Returns:
        Response: 非流式响应
    """
    # 调用LLM服务
    response: ChatCompletionResponse = await llm_service.chat_completion(completion_request)

    # 计算处理时间
    processing_time = time.time() - start_time

    # 构造OpenAI格式的响应
    openai_response = {
        "id": f"chatcmpl-{response.id}",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model,
        "choices": [],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        } if response.usage else None,
        "system_fingerprint": response.system_fingerprint,
    }

    # 转换选择项
    for choice in response.choices:
        openai_choice = {
            "index": choice.index,
            "message": {
                "role": choice.message.role.value,
                "content": choice.message.content,
            },
            "finish_reason": choice.finish_reason,
        }

        # 添加可选字段
        if choice.message.name:
            openai_choice["message"]["name"] = choice.message.name
        if choice.message.function_call:
            openai_choice["message"]["function_call"] = choice.message.function_call
        if choice.message.tool_calls:
            openai_choice["message"]["tool_calls"] = choice.message.tool_calls
        if choice.message.tool_call_id:
            openai_choice["message"]["tool_call_id"] = choice.message.tool_call_id

        openai_response["choices"].append(openai_choice)

    structured_logger.info(
        "非流式聊天完成完成",
        extra={
            "request_id": request_id,
            "model": response.model,
            "choices_count": len(response.choices),
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "processing_time_seconds": round(processing_time, 3),
        }
    )

    return json(openai_response)


async def _handle_streaming_completion(
    request: Request,
    completion_request: ChatCompletionRequest,
    llm_service,
    request_id: str
) -> Response:
    """
    处理流式聊天完成请求

    Args:
        request: Sanic请求对象
        completion_request: 聊天完成请求
        llm_service: LLM服务实例
        request_id: 请求ID

    Returns:
        Response: 流式响应
    """
    async def streaming_response():
        """流式响应生成器"""
        try:
            # 创建流式响应ID
            response_id = f"chatcmpl-{completion_request.model}-{int(time.time())}"
            created = int(time.time())

            # 获取流式响应
            async for chunk in llm_service.chat_completion_stream(completion_request):
                # 构造OpenAI格式的流式响应
                openai_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": chunk.model,
                    "choices": chunk.choices,
                }

                # 添加使用信息（如果存在）
                if chunk.usage:
                    openai_chunk["usage"] = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

                # 发送数据块
                chunk_data = f"data: {json_module.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                yield chunk_data.encode('utf-8')

            # 发送结束标记
            yield "data: [DONE]\n\n".encode('utf-8')

            structured_logger.info(
                "流式聊天完成完成",
                extra={
                    "request_id": request_id,
                    "model": completion_request.model,
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
            # 发送错误信息
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": "stream_error",
                    "request_id": request_id,
                }
            }
            error_data = f"data: {json_module.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield error_data.encode('utf-8')

    return stream(
        streaming_response(),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@llm_router.get("/models")
async def list_models(request: Request) -> Response:
    """
    列出可用的模型

    Args:
        request: Sanic请求对象

    Returns:
        Response: 模型列表响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service
        llm_service = service.get_llm_service()
        if not llm_service:
            return json({
                "object": "list",
                "data": [],
            })

        # 获取模型能力信息
        capabilities = llm_service.capabilities

        # 构造OpenAI格式的模型列表
        models = []
        for model_name in capabilities.supported_models:
            model_info = {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": llm_service.provider_name,
                "permission": [],
                "root": model_name,
                "parent": None,
            }
            models.append(model_info)

        structured_logger.info(
            "模型列表请求",
            extra={
                "request_id": request_id,
                "models_count": len(models),
                "provider": llm_service.provider_name,
            }
        )

        return json({
            "object": "list",
            "data": models,
        })

    except Exception as e:
        structured_logger.error(
            f"获取模型列表失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to retrieve models",
                "type": "internal_server_error",
                "code": "models_error",
                "request_id": request_id,
            }
        }, status=500)


@llm_router.post("/completions")
async def legacy_completions(request: Request) -> Response:
    """
    兼容OpenAI的文本完成接口（legacy）

    Args:
        request: Sanic请求对象

    Returns:
        Response: 文本完成响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        # 解析请求体
        request_data = request.json
        if not request_data:
            raise ValidationError("Invalid JSON request body")

        # 转换为聊天完成格式
        prompt = request_data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)

        messages = [
            ChatMessage(role=ChatRole.USER, content=prompt)
        ]

        completion_request = ChatCompletionRequest(
            messages=messages,
            model=request_data.get("model", "gpt-3.5-turbo"),
            temperature=request_data.get("temperature", 1.0),
            max_tokens=request_data.get("max_tokens"),
            top_p=request_data.get("top_p", 1.0),
            stream=request_data.get("stream", False),
            stop=request_data.get("stop"),
            presence_penalty=request_data.get("presence_penalty", 0.0),
            frequency_penalty=request_data.get("frequency_penalty", 0.0),
        )

        # 获取LLM服务
        service = request.app.ctx.service
        llm_service = service.get_llm_service()
        if not llm_service:
            raise NotFoundError("LLM service not available")

        structured_logger.info(
            "Legacy文本完成请求",
            extra={
                "request_id": request_id,
                "model": completion_request.model,
                "prompt_length": len(prompt),
                "stream": completion_request.stream,
            }
        )

        # 处理请求
        if completion_request.stream:
            return await _handle_streaming_completion(
                request, completion_request, llm_service, request_id
            )
        else:
            response = await _handle_non_streaming_completion(
                request, completion_request, llm_service, request_id, time.time()
            )

            # 转换为legacy格式
            response_data = response.json
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                legacy_response = {
                    "id": response_data["id"],
                    "object": "text_completion",
                    "created": response_data["created"],
                    "model": response_data["model"],
                    "choices": [
                        {
                            "text": choice["message"]["content"],
                            "index": choice["index"],
                            "logprobs": choice.get("logprobs"),
                            "finish_reason": choice["finish_reason"],
                        }
                    ],
                    "usage": response_data.get("usage"),
                }
                return json(legacy_response)

            return response

    except Exception as e:
        structured_logger.error(
            f"Legacy文本完成请求失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        return json({
            "error": {
                "message": "Failed to process completion request",
                "type": "internal_server_error",
                "code": "completion_error",
                "request_id": request_id,
            }
        }, status=500)


structured_logger.info("LLM路由加载完成")