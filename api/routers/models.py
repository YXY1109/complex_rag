"""
模型管理API路由
提供AI模型的配置、状态查询和切换功能
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query

from infrastructure.monitoring.loguru_logger import logger
from rag_service.providers.model_manager_adapter import ModelManager
from api.exceptions import ValidationError, NotFoundError, ServiceUnavailableError

router = APIRouter()


class ModelInfo(BaseModel):
    """模型信息模型"""
    id: str
    name: str
    type: str  # llm, embedding, rerank
    provider: str
    status: str
    config: Dict[str, Any]
    capabilities: List[str]
    created_at: str
    updated_at: str


class ModelListResponse(BaseModel):
    """模型列表响应模型"""
    models: List[ModelInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


class ModelConfigUpdate(BaseModel):
    """模型配置更新请求模型"""
    config: Dict[str, Any] = Field(..., description="模型配置")
    enabled: Optional[bool] = Field(None, description="是否启用")


class ModelTestRequest(BaseModel):
    """模型测试请求模型"""
    input: str = Field(..., description="测试输入")
    config: Optional[Dict[str, Any]] = Field(None, description="临时配置覆盖")


class ModelTestResponse(BaseModel):
    """模型测试响应模型"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time: float


@router.get("/", response_model=ModelListResponse, summary="获取模型列表")
async def get_models(
    type: Optional[str] = Query(None, description="模型类型过滤：llm/embedding/rerank"),
    provider: Optional[str] = Query(None, description="提供商过滤"),
    status: Optional[str] = Query(None, description="状态过滤"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """
    获取可用的AI模型列表

    Args:
        type: 模型类型过滤
        provider: 提供商过滤
        status: 状态过滤
        page: 页码
        page_size: 每页数量

    Returns:
        ModelListResponse: 模型列表
    """
    logger.info(f"获取模型列表，类型: {type}, 提供商: {provider}")

    try:
        model_manager = ModelManager()

        # 获取模型列表
        models, total = await model_manager.get_models(
            model_type=type,
            provider=provider,
            status=status,
            page=page,
            page_size=page_size
        )

        # 计算总页数
        total_pages = (total + page_size - 1) // page_size

        return ModelListResponse(
            models=[ModelInfo(**model) for model in models],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取模型列表失败: {str(e)}")


@router.get("/{model_id}", response_model=ModelInfo, summary="获取模型详情")
async def get_model(model_id: str):
    """
    获取指定模型的详细信息

    Args:
        model_id: 模型ID

    Returns:
        ModelInfo: 模型详细信息
    """
    logger.info(f"获取模型详情: {model_id}")

    try:
        model_manager = ModelManager()
        model = await model_manager.get_model(model_id)

        if not model:
            raise NotFoundError(f"模型不存在: {model_id}")

        return ModelInfo(**model)

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取模型详情失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取模型详情失败: {str(e)}")


@router.put("/{model_id}", response_model=ModelInfo, summary="更新模型配置")
async def update_model_config(
    model_id: str,
    request: ModelConfigUpdate
):
    """
    更新指定模型的配置

    Args:
        model_id: 模型ID
        request: 配置更新请求

    Returns:
        ModelInfo: 更新后的模型信息
    """
    logger.info(f"更新模型配置: {model_id}")

    try:
        model_manager = ModelManager()

        # 检查模型是否存在
        existing = await model_manager.get_model(model_id)
        if not existing:
            raise NotFoundError(f"模型不存在: {model_id}")

        # 更新模型配置
        update_data = {"config": request.config}
        if request.enabled is not None:
            update_data["enabled"] = request.enabled

        updated_model = await model_manager.update_model(model_id, update_data)

        logger.info(f"模型配置更新成功: {model_id}")

        return ModelInfo(**updated_model)

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"更新模型配置失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"更新模型配置失败: {str(e)}")


@router.post("/{model_id}/test", response_model=ModelTestResponse, summary="测试模型")
async def test_model(model_id: str, request: ModelTestRequest):
    """
    测试指定模型的功能

    Args:
        model_id: 模型ID
        request: 测试请求

    Returns:
        ModelTestResponse: 测试结果
    """
    logger.info(f"测试模型: {model_id}")

    try:
        if not request.input.strip():
            raise ValidationError("测试输入不能为空")

        model_manager = ModelManager()

        # 检查模型是否存在
        model = await model_manager.get_model(model_id)
        if not model:
            raise NotFoundError(f"模型不存在: {model_id}")

        # 执行模型测试
        import time
        start_time = time.time()

        try:
            result = await model_manager.test_model(
                model_id=model_id,
                test_input=request.input,
                config_override=request.config
            )

            response_time = time.time() - start_time

            return ModelTestResponse(
                success=True,
                result=result,
                response_time=round(response_time, 3)
            )

        except Exception as test_error:
            response_time = time.time() - start_time

            return ModelTestResponse(
                success=False,
                error=str(test_error),
                response_time=round(response_time, 3)
            )

    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"测试模型失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"测试模型失败: {str(e)}")


@router.post("/{model_id}/enable", summary="启用模型")
async def enable_model(model_id: str):
    """
    启用指定模型

    Args:
        model_id: 模型ID

    Returns:
        Dict: 启用结果
    """
    logger.info(f"启用模型: {model_id}")

    try:
        model_manager = ModelManager()

        # 检查模型是否存在
        existing = await model_manager.get_model(model_id)
        if not existing:
            raise NotFoundError(f"模型不存在: {model_id}")

        # 启用模型
        success = await model_manager.enable_model(model_id)

        if not success:
            raise ServiceUnavailableError("启用模型失败")

        return {
            "success": True,
            "message": "模型启用成功",
            "model_id": model_id
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"启用模型失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"启用模型失败: {str(e)}")


@router.post("/{model_id}/disable", summary="禁用模型")
async def disable_model(model_id: str):
    """
    禁用指定模型

    Args:
        model_id: 模型ID

    Returns:
        Dict: 禁用结果
    """
    logger.info(f"禁用模型: {model_id}")

    try:
        model_manager = ModelManager()

        # 检查模型是否存在
        existing = await model_manager.get_model(model_id)
        if not existing:
            raise NotFoundError(f"模型不存在: {model_id}")

        # 禁用模型
        success = await model_manager.disable_model(model_id)

        if not success:
            raise ServiceUnavailableError("禁用模型失败")

        return {
            "success": True,
            "message": "模型禁用成功",
            "model_id": model_id
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"禁用模型失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"禁用模型失败: {str(e)}")


@router.get("/types/available", summary="获取可用的模型类型")
async def get_available_model_types():
    """
    获取系统中可用的模型类型

    Returns:
        Dict: 可用的模型类型
    """
    logger.info("获取可用的模型类型")

    try:
        model_manager = ModelManager()
        types = await model_manager.get_available_model_types()

        return {
            "model_types": types,
            "total": len(types)
        }

    except Exception as e:
        logger.error(f"获取可用模型类型失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取可用模型类型失败: {str(e)}")


@router.get("/providers/available", summary="获取可用的模型提供商")
async def get_available_providers():
    """
    获取系统中可用的模型提供商

    Returns:
        Dict: 可用的模型提供商
    """
    logger.info("获取可用的模型提供商")

    try:
        model_manager = ModelManager()
        providers = await model_manager.get_available_providers()

        return {
            "providers": providers,
            "total": len(providers)
        }

    except Exception as e:
        logger.error(f"获取可用模型提供商失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取可用模型提供商失败: {str(e)}")


@router.get("/{model_id}/status", summary="获取模型状态")
async def get_model_status(model_id: str):
    """
    获取指定模型的实时状态

    Args:
        model_id: 模型ID

    Returns:
        Dict: 模型状态
    """
    logger.info(f"获取模型状态: {model_id}")

    try:
        model_manager = ModelManager()

        # 检查模型是否存在
        existing = await model_manager.get_model(model_id)
        if not existing:
            raise NotFoundError(f"模型不存在: {model_id}")

        # 获取模型状态
        status = await model_manager.get_model_status(model_id)

        return {
            "model_id": model_id,
            "status": status
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取模型状态失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取模型状态失败: {str(e)}")


@router.get("/{model_id}/metrics", summary="获取模型性能指标")
async def get_model_metrics(
    model_id: str,
    time_range: str = Query("24h", description="时间范围：1h/24h/7d/30d")
):
    """
    获取指定模型的性能指标

    Args:
        model_id: 模型ID
        time_range: 时间范围

    Returns:
        Dict: 性能指标
    """
    logger.info(f"获取模型性能指标: {model_id}, 时间范围: {time_range}")

    try:
        model_manager = ModelManager()

        # 检查模型是否存在
        existing = await model_manager.get_model(model_id)
        if not existing:
            raise NotFoundError(f"模型不存在: {model_id}")

        # 获取性能指标
        metrics = await model_manager.get_model_metrics(model_id, time_range)

        return {
            "model_id": model_id,
            "time_range": time_range,
            "metrics": metrics
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取模型性能指标失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取模型性能指标失败: {str(e)}")