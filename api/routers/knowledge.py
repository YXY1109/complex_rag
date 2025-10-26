"""
知识库管理API路由
提供知识库的创建、配置、管理和检索功能
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query

from infrastructure.monitoring.loguru_logger import logger
from core_rag.services.knowledge_service_adapter import KnowledgeService
from api.exceptions import ValidationError, NotFoundError, ServiceUnavailableError

router = APIRouter()


class KnowledgeBaseCreate(BaseModel):
    """知识库创建请求模型"""
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    config: Optional[Dict[str, Any]] = Field(None, description="知识库配置")


class KnowledgeBaseUpdate(BaseModel):
    """知识库更新请求模型"""
    name: Optional[str] = Field(None, description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    config: Optional[Dict[str, Any]] = Field(None, description="知识库配置")


class KnowledgeBaseInfo(BaseModel):
    """知识库信息模型"""
    id: str
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    status: str
    document_count: int
    vector_count: int
    created_at: str
    updated_at: str


class KnowledgeBaseListResponse(BaseModel):
    """知识库列表响应模型"""
    knowledge_bases: List[KnowledgeBaseInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


class VectorIndexRequest(BaseModel):
    """向量索引请求模型"""
    document_ids: Optional[List[str]] = Field(None, description="指定文档ID列表，为空则索引所有文档")
    rebuild: bool = Field(False, description="是否重建索引")
    config: Optional[Dict[str, Any]] = Field(None, description="索引配置")


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    top_k: Optional[int] = Field(10, description="返回结果数量", ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(None, description="搜索过滤条件")
    search_config: Optional[Dict[str, Any]] = Field(None, description="搜索配置")


class SearchResult(BaseModel):
    """搜索结果模型"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: str
    document_name: str


class SearchResponse(BaseModel):
    """搜索响应模型"""
    query: str
    results: List[SearchResult]
    total: int
    search_time: float


@router.post("/", response_model=KnowledgeBaseInfo, summary="创建知识库")
async def create_knowledge_base(request: KnowledgeBaseCreate):
    """
    创建新的知识库

    Args:
        request: 知识库创建请求

    Returns:
        KnowledgeBaseInfo: 创建的知识库信息
    """
    logger.info(f"创建知识库: {request.name}")

    try:
        # 验证请求参数
        if not request.name.strip():
            raise ValidationError("知识库名称不能为空")

        knowledge_service = KnowledgeService()

        # 检查知识库名称是否已存在
        existing = await knowledge_service.get_knowledge_base_by_name(request.name)
        if existing:
            raise ValidationError(f"知识库名称已存在: {request.name}")

        # 创建知识库
        knowledge_base = await knowledge_service.create_knowledge_base(
            name=request.name,
            description=request.description,
            config=request.config or {}
        )

        logger.info(f"知识库创建成功: {knowledge_base['id']}")

        return KnowledgeBaseInfo(**knowledge_base)

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"创建知识库失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"创建知识库失败: {str(e)}")


@router.get("/", response_model=KnowledgeBaseListResponse, summary="获取知识库列表")
async def get_knowledge_bases(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="状态过滤")
):
    """
    获取知识库列表

    Args:
        page: 页码
        page_size: 每页数量
        status: 状态过滤

    Returns:
        KnowledgeBaseListResponse: 知识库列表
    """
    logger.info(f"获取知识库列表，页码: {page}")

    try:
        knowledge_service = KnowledgeService()

        # 获取知识库列表
        knowledge_bases, total = await knowledge_service.get_knowledge_bases(
            page=page,
            page_size=page_size,
            status=status
        )

        # 计算总页数
        total_pages = (total + page_size - 1) // page_size

        return KnowledgeBaseListResponse(
            knowledge_bases=[KnowledgeBaseInfo(**kb) for kb in knowledge_bases],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取知识库列表失败: {str(e)}")


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseInfo, summary="获取知识库详情")
async def get_knowledge_base(knowledge_base_id: str):
    """
    获取指定知识库的详细信息

    Args:
        knowledge_base_id: 知识库ID

    Returns:
        KnowledgeBaseInfo: 知识库详细信息
    """
    logger.info(f"获取知识库详情: {knowledge_base_id}")

    try:
        knowledge_service = KnowledgeService()
        knowledge_base = await knowledge_service.get_knowledge_base(knowledge_base_id)

        if not knowledge_base:
            raise NotFoundError(f"知识库不存在: {knowledge_base_id}")

        return KnowledgeBaseInfo(**knowledge_base)

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取知识库详情失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取知识库详情失败: {str(e)}")


@router.put("/{knowledge_base_id}", response_model=KnowledgeBaseInfo, summary="更新知识库")
async def update_knowledge_base(
    knowledge_base_id: str,
    request: KnowledgeBaseUpdate
):
    """
    更新指定知识库的信息

    Args:
        knowledge_base_id: 知识库ID
        request: 更新请求

    Returns:
        KnowledgeBaseInfo: 更新后的知识库信息
    """
    logger.info(f"更新知识库: {knowledge_base_id}")

    try:
        knowledge_service = KnowledgeService()

        # 检查知识库是否存在
        existing = await knowledge_service.get_knowledge_base(knowledge_base_id)
        if not existing:
            raise NotFoundError(f"知识库不存在: {knowledge_base_id}")

        # 如果更新名称，检查名称是否冲突
        if request.name and request.name != existing["name"]:
            name_conflict = await knowledge_service.get_knowledge_base_by_name(request.name)
            if name_conflict:
                raise ValidationError(f"知识库名称已存在: {request.name}")

        # 更新知识库
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.config is not None:
            update_data["config"] = request.config

        updated_kb = await knowledge_service.update_knowledge_base(
            knowledge_base_id,
            update_data
        )

        logger.info(f"知识库更新成功: {knowledge_base_id}")

        return KnowledgeBaseInfo(**updated_kb)

    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"更新知识库失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"更新知识库失败: {str(e)}")


@router.delete("/{knowledge_base_id}", summary="删除知识库")
async def delete_knowledge_base(knowledge_base_id: str):
    """
    删除指定知识库

    Args:
        knowledge_base_id: 知识库ID

    Returns:
        Dict: 删除结果
    """
    logger.info(f"删除知识库: {knowledge_base_id}")

    try:
        knowledge_service = KnowledgeService()

        # 检查知识库是否存在
        existing = await knowledge_service.get_knowledge_base(knowledge_base_id)
        if not existing:
            raise NotFoundError(f"知识库不存在: {knowledge_base_id}")

        # 删除知识库
        success = await knowledge_service.delete_knowledge_base(knowledge_base_id)

        if not success:
            raise ServiceUnavailableError("删除知识库失败")

        logger.info(f"知识库删除成功: {knowledge_base_id}")

        return {
            "success": True,
            "message": "知识库删除成功",
            "knowledge_base_id": knowledge_base_id
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"删除知识库失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"删除知识库失败: {str(e)}")


@router.post("/{knowledge_base_id}/index", summary="创建向量索引")
async def create_vector_index(
    knowledge_base_id: str,
    request: VectorIndexRequest,
    background_tasks: BackgroundTasks
):
    """
    为知识库创建或更新向量索引

    Args:
        knowledge_base_id: 知识库ID
        request: 索引请求
        background_tasks: 后台任务

    Returns:
        Dict: 索引创建结果
    """
    logger.info(f"创建知识库向量索引: {knowledge_base_id}")

    try:
        knowledge_service = KnowledgeService()

        # 检查知识库是否存在
        existing = await knowledge_service.get_knowledge_base(knowledge_base_id)
        if not existing:
            raise NotFoundError(f"知识库不存在: {knowledge_base_id}")

        # 添加后台索引任务
        background_tasks.add_task(
            _create_index_background,
            knowledge_base_id,
            request.document_ids,
            request.rebuild,
            request.config or {}
        )

        return {
            "success": True,
            "message": "向量索引创建任务已启动",
            "knowledge_base_id": knowledge_base_id,
            "rebuild": request.rebuild
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"启动向量索引创建失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"启动向量索引创建失败: {str(e)}")


@router.get("/{knowledge_base_id}/index/status", summary="获取索引状态")
async def get_index_status(knowledge_base_id: str):
    """
    获取知识库向量索引状态

    Args:
        knowledge_base_id: 知识库ID

    Returns:
        Dict: 索引状态
    """
    logger.info(f"获取知识库索引状态: {knowledge_base_id}")

    try:
        knowledge_service = KnowledgeService()
        status = await knowledge_service.get_index_status(knowledge_base_id)

        if not status:
            raise NotFoundError(f"知识库不存在: {knowledge_base_id}")

        return status

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取索引状态失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取索引状态失败: {str(e)}")


@router.post("/{knowledge_base_id}/search", response_model=SearchResponse, summary="搜索知识库")
async def search_knowledge_base(
    knowledge_base_id: str,
    request: SearchRequest
):
    """
    在知识库中搜索相关内容

    Args:
        knowledge_base_id: 知识库ID
        request: 搜索请求

    Returns:
        SearchResponse: 搜索结果
    """
    logger.info(f"搜索知识库: {knowledge_base_id}, 查询: {request.query}")

    try:
        if not request.query.strip():
            raise ValidationError("搜索查询不能为空")

        knowledge_service = KnowledgeService()

        # 检查知识库是否存在
        existing = await knowledge_service.get_knowledge_base(knowledge_base_id)
        if not existing:
            raise NotFoundError(f"知识库不存在: {knowledge_base_id}")

        # 执行搜索
        import time
        start_time = time.time()

        results = await knowledge_service.search_knowledge_base(
            knowledge_base_id=knowledge_base_id,
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            config=request.search_config or {}
        )

        search_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results=[SearchResult(**result) for result in results],
            total=len(results),
            search_time=round(search_time, 3)
        )

    except (NotFoundError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"知识库搜索失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"知识库搜索失败: {str(e)}")


@router.get("/{knowledge_base_id}/statistics", summary="获取知识库统计信息")
async def get_knowledge_base_statistics(knowledge_base_id: str):
    """
    获取知识库的统计信息

    Args:
        knowledge_base_id: 知识库ID

    Returns:
        Dict: 统计信息
    """
    logger.info(f"获取知识库统计信息: {knowledge_base_id}")

    try:
        knowledge_service = KnowledgeService()

        # 检查知识库是否存在
        existing = await knowledge_service.get_knowledge_base(knowledge_base_id)
        if not existing:
            raise NotFoundError(f"知识库不存在: {knowledge_base_id}")

        # 获取统计信息
        statistics = await knowledge_service.get_knowledge_base_statistics(knowledge_base_id)

        return statistics

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取知识库统计信息失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取知识库统计信息失败: {str(e)}")


async def _create_index_background(
    knowledge_base_id: str,
    document_ids: Optional[List[str]],
    rebuild: bool,
    config: Dict[str, Any]
):
    """
    后台创建向量索引任务

    Args:
        knowledge_base_id: 知识库ID
        document_ids: 指定文档ID列表
        rebuild: 是否重建索引
        config: 索引配置
    """
    try:
        logger.info(f"开始创建知识库向量索引: {knowledge_base_id}")

        knowledge_service = KnowledgeService()

        # 执行索引创建
        result = await knowledge_service.create_vector_index(
            knowledge_base_id=knowledge_base_id,
            document_ids=document_ids,
            rebuild=rebuild,
            config=config
        )

        logger.info(f"知识库向量索引创建完成: {knowledge_base_id}, 结果: {result}")

    except Exception as e:
        logger.error(f"知识库向量索引创建失败: {knowledge_base_id}, 错误: {str(e)}", exc_info=True)