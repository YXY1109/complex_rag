"""
文档管理API路由
提供文档上传、解析、管理和检索功能
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from infrastructure.monitoring.loguru_logger import logger
from document_parser.services.parser_service_adapter import DocumentParserService
from document_parser.services.pipeline_service_adapter import PipelineService
from api.exceptions import ValidationError, NotFoundError, ServiceUnavailableError

router = APIRouter()


class DocumentInfo(BaseModel):
    """文档信息模型"""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: str
    updated_at: str
    knowledge_base_id: str
    parsed_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """文档上传响应模型"""
    success: bool
    document_id: str
    filename: str
    status: str
    message: str


class DocumentListResponse(BaseModel):
    """文档列表响应模型"""
    documents: List[DocumentInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentParseRequest(BaseModel):
    """文档解析请求模型"""
    document_id: str
    parser_type: Optional[str] = Field("auto", description="解析器类型")
    parse_config: Optional[Dict[str, Any]] = Field(None, description="解析配置")


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传文档")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(...),
    auto_parse: bool = Form(True),
    parse_config: str = Form("{}")
):
    """
    上传文档到指定知识库

    Args:
        background_tasks: 后台任务
        file: 上传的文件
        knowledge_base_id: 知识库ID
        auto_parse: 是否自动解析
        parse_config: 解析配置（JSON字符串）

    Returns:
        DocumentUploadResponse: 上传结果
    """
    logger.info(f"上传文档: {file.filename} 到知识库: {knowledge_base_id}")

    try:
        # 验证文件
        if not file.filename:
            raise ValidationError("文件名不能为空")

        # 解析配置
        import json
        try:
            config = json.loads(parse_config)
        except json.JSONDecodeError:
            config = {}

        # 检查文件大小（100MB限制）
        file_size = 0
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # 重置文件指针

        if file_size > 100 * 1024 * 1024:  # 100MB
            raise ValidationError("文件大小不能超过100MB")

        # 创建文档解析服务
        parser_service = DocumentParserService()

        # 上传文档
        document_id = await parser_service.upload_document(
            file_content=content,
            filename=file.filename,
            knowledge_base_id=knowledge_base_id,
            file_content_type=file.content_type
        )

        # 如果设置了自动解析，添加后台解析任务
        if auto_parse:
            background_tasks.add_task(
                _parse_document_background,
                document_id,
                config
            )

        logger.info(f"文档上传成功: {file.filename}, ID: {document_id}")

        return DocumentUploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            status="uploaded" + ("_parsing" if auto_parse else ""),
            message="文档上传成功" + ("，正在解析中..." if auto_parse else "")
        )

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"文档上传失败: {str(e)}")


@router.get("/", response_model=DocumentListResponse, summary="获取文档列表")
async def get_documents(
    knowledge_base_id: Optional[str] = Query(None, description="知识库ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="文档状态"),
    file_type: Optional[str] = Query(None, description="文件类型")
):
    """
    获取文档列表

    Args:
        knowledge_base_id: 知识库ID
        page: 页码
        page_size: 每页数量
        status: 文档状态过滤
        file_type: 文件类型过滤

    Returns:
        DocumentListResponse: 文档列表
    """
    logger.info(f"获取文档列表，知识库: {knowledge_base_id}, 页码: {page}")

    try:
        parser_service = DocumentParserService()

        # 获取文档列表
        documents, total = await parser_service.get_documents(
            knowledge_base_id=knowledge_base_id,
            page=page,
            page_size=page_size,
            status=status,
            file_type=file_type
        )

        # 计算总页数
        total_pages = (total + page_size - 1) // page_size

        return DocumentListResponse(
            documents=documents,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取文档列表失败: {str(e)}")


@router.get("/{document_id}", response_model=DocumentInfo, summary="获取文档详情")
async def get_document(document_id: str):
    """
    获取指定文档的详细信息

    Args:
        document_id: 文档ID

    Returns:
        DocumentInfo: 文档详细信息
    """
    logger.info(f"获取文档详情: {document_id}")

    try:
        parser_service = DocumentParserService()
        document = await parser_service.get_document(document_id)

        if not document:
            raise NotFoundError(f"文档不存在: {document_id}")

        return DocumentInfo(**document)

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取文档详情失败: {str(e)}")


@router.post("/{document_id}/parse", summary="解析文档")
async def parse_document(
    document_id: str,
    request: DocumentParseRequest,
    background_tasks: BackgroundTasks
):
    """
    解析指定文档

    Args:
        document_id: 文档ID
        request: 解析请求
        background_tasks: 后台任务

    Returns:
        Dict: 解析结果
    """
    logger.info(f"解析文档: {document_id}")

    try:
        parser_service = DocumentParserService()

        # 检查文档是否存在
        document = await parser_service.get_document(document_id)
        if not document:
            raise NotFoundError(f"文档不存在: {document_id}")

        # 添加后台解析任务
        background_tasks.add_task(
            _parse_document_background,
            document_id,
            request.parse_config or {},
            request.parser_type
        )

        return {
            "success": True,
            "message": "文档解析任务已启动",
            "document_id": document_id
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"启动文档解析失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"启动文档解析失败: {str(e)}")


@router.get("/{document_id}/parse_status", summary="获取文档解析状态")
async def get_parse_status(document_id: str):
    """
    获取文档解析状态

    Args:
        document_id: 文档ID

    Returns:
        Dict: 解析状态
    """
    logger.info(f"获取文档解析状态: {document_id}")

    try:
        parser_service = DocumentParserService()
        status = await parser_service.get_parse_status(document_id)

        if not status:
            raise NotFoundError(f"文档不存在: {document_id}")

        return status

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取文档解析状态失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取文档解析状态失败: {str(e)}")


@router.delete("/{document_id}", summary="删除文档")
async def delete_document(document_id: str):
    """
    删除指定文档

    Args:
        document_id: 文档ID

    Returns:
        Dict: 删除结果
    """
    logger.info(f"删除文档: {document_id}")

    try:
        parser_service = DocumentParserService()

        # 检查文档是否存在
        document = await parser_service.get_document(document_id)
        if not document:
            raise NotFoundError(f"文档不存在: {document_id}")

        # 删除文档
        success = await parser_service.delete_document(document_id)

        if not success:
            raise ServiceUnavailableError("删除文档失败")

        return {
            "success": True,
            "message": "文档删除成功",
            "document_id": document_id
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"删除文档失败: {str(e)}")


@router.get("/{document_id}/content", summary="获取文档解析内容")
async def get_document_content(
    document_id: str,
    format: str = Query("text", description="内容格式：text/markdown/json")
):
    """
    获取文档解析后的内容

    Args:
        document_id: 文档ID
        format: 内容格式

    Returns:
        Dict: 文档内容
    """
    logger.info(f"获取文档内容: {document_id}, 格式: {format}")

    try:
        parser_service = DocumentParserService()

        # 检查文档是否存在
        document = await parser_service.get_document(document_id)
        if not document:
            raise NotFoundError(f"文档不存在: {document_id}")

        # 获取文档内容
        content = await parser_service.get_document_content(document_id, format)

        return {
            "document_id": document_id,
            "format": format,
            "content": content
        }

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"获取文档内容失败: {str(e)}", exc_info=True)
        raise ServiceUnavailableError(f"获取文档内容失败: {str(e)}")


async def _parse_document_background(
    document_id: str,
    parse_config: Dict[str, Any],
    parser_type: str = "auto"
):
    """
    后台文档解析任务

    Args:
        document_id: 文档ID
        parse_config: 解析配置
        parser_type: 解析器类型
    """
    try:
        logger.info(f"开始后台解析文档: {document_id}")

        pipeline_service = PipelineService()

        # 执行文档解析流水线
        result = await pipeline_service.parse_document(
            document_id=document_id,
            parser_type=parser_type,
            config=parse_config
        )

        logger.info(f"文档解析完成: {document_id}, 结果: {result}")

    except Exception as e:
        logger.error(f"后台文档解析失败: {document_id}, 错误: {str(e)}", exc_info=True)

        # 更新文档状态为解析失败
        try:
            parser_service = DocumentParserService()
            await parser_service.update_document_status(document_id, "parse_failed", str(e))
        except:
            pass