"""
基础处理器类

此模块提供所有专用处理器的基础类，
包含通用的处理逻辑和工具方法。
"""

import asyncio
import os
import tempfile
import time
from typing import Dict, Any, List, Optional, Union, BinaryIO
from abc import ABC, abstractmethod
from pathlib import Path
import uuid

from ..interfaces.parser_interface import (
    DocumentParserInterface,
    ParserConfig,
    ParseResult,
    DocumentMetadata,
    DocumentType,
    ProcessingStrategy,
    TextChunk,
    ImageInfo,
    TableInfo,
    ParseException,
    UnsupportedFormatError,
    CorruptedFileError,
    ProcessingError,
    ValidationError,
    TimeoutError
)
from ..source_detection import SourceDetector, SourceDetectionResult
from ..strategy_selection import StrategySelector, StrategyRecommendation
from ..strategy_config import get_config_manager
from ..quality_monitoring import get_quality_monitor, QualityMonitor


class BaseProcessor(DocumentParserInterface):
    """
    基础处理器类。

    为所有专用处理器提供通用功能和工具方法。
    """

    def __init__(self, config: ParserConfig):
        """
        初始化基础处理器。

        Args:
            config: 解析器配置
        """
        super().__init__(config)
        self.source_detector = SourceDetector()
        self.strategy_selector = StrategySelector()
        self.config_manager = get_config_manager()
        self.quality_monitor = get_quality_monitor()
        self._processing_sessions: Dict[str, str] = {}  # session_id -> file_path mapping

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化处理器（子类实现）。"""
        return True

    @abstractmethod
    async def cleanup(self) -> None:
        """清理处理器资源（子类实现）。"""
        pass

    async def parse_file(
        self,
        file_path: str,
        strategy: Optional[ProcessingStrategy] = None,
        **kwargs
    ) -> ParseResult:
        """
        解析文件的主入口方法。

        Args:
            file_path: 文件路径
            strategy: 处理策略（可选）
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果
        """
        start_time = time.time()
        session_id = None

        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 开始质量监控会话
            session_id = self.quality_monitor.start_session(file_path, strategy.value if strategy else "auto")

            # 记录会话映射
            self._processing_sessions[session_id] = file_path

            # 自动检测来源和选择策略
            if strategy is None:
                strategy = await self._auto_select_strategy(file_path)

            # 获取策略配置
            source_result = await self.source_detector.detect_source(file_path=file_path)
            strategy_config = self.config_manager.get_config(strategy, source_result.source_type)

            # 执行实际解析
            result = await self._parse_with_config(file_path, strategy, strategy_config, **kwargs)

            # 记录处理时间
            processing_time = time.time() - start_time
            result.processing_time_ms = processing_time * 1000

            # 更新统计信息
            self._update_statistics(result, processing_time)

            return result

        except Exception as e:
            # 创建错误结果
            error_result = ParseResult(
                success=False,
                metadata=self._create_error_metadata(file_path, str(e)),
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

            # 如果有会话，结束会话
            if session_id:
                try:
                    self.quality_monitor.end_session(session_id, error_result, time.time() - start_time)
                except Exception:
                    pass  # 忽略质量监控错误

            raise ParseException(
                f"文件解析失败: {str(e)}",
                parser=self.parser_name,
                file_path=file_path
            ) from e

        finally:
            # 清理会话映射
            if session_id and session_id in self._processing_sessions:
                del self._processing_sessions[session_id]

    async def parse_bytes(
        self,
        data: bytes,
        file_name: str,
        strategy: Optional[ProcessingStrategy] = None,
        **kwargs
    ) -> ParseResult:
        """
        解析字节数据。

        Args:
            data: 文件字节数据
            file_name: 文件名
            strategy: 处理策略（可选）
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
            temp_file.write(data)
            temp_file_path = temp_file.name

        try:
            # 使用临时文件解析
            result = await self.parse_file(temp_file_path, strategy, **kwargs)

            # 更新元数据中的文件名
            result.metadata.file_name = file_name

            return result

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    async def parse_stream(
        self,
        stream: BinaryIO,
        file_name: str,
        strategy: Optional[ProcessingStrategy] = None,
        **kwargs
    ) -> ParseResult:
        """
        解析流数据。

        Args:
            stream: 文件流
            file_name: 文件名
            strategy: 处理策略（可选）
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果
        """
        # 读取流数据
        data = stream.read()
        if isinstance(data, str):
            data = data.encode('utf-8')

        return await self.parse_bytes(data, file_name, strategy, **kwargs)

    async def _auto_select_strategy(self, file_path: str) -> ProcessingStrategy:
        """
        自动选择处理策略。

        Args:
            file_path: 文件路径

        Returns:
            ProcessingStrategy: 选择的策略
        """
        # 检测来源
        source_result = await self.source_detector.detect_source(file_path=file_path)

        # 选择策略
        recommendation = await self.strategy_selector.select_strategy(source_result)

        # 记录策略选择日志
        self._log_strategy_selection(file_path, source_result, recommendation)

        return recommendation.recommended_strategy

    @abstractmethod
    async def _parse_with_config(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: Any,
        **kwargs
    ) -> ParseResult:
        """
        使用配置执行解析（子类实现）。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 策略配置
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果
        """
        pass

    def _create_error_metadata(self, file_path: str, error_message: str) -> DocumentMetadata:
        """
        创建错误元数据。

        Args:
            file_path: 文件路径
            error_message: 错误消息

        Returns:
            DocumentMetadata: 错误元数据
        """
        try:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            file_ext = os.path.splitext(file_path)[1].lower()
        except:
            file_size = 0
            file_ext = ""

        return DocumentMetadata(
            file_name=os.path.basename(file_path),
            file_size=file_size,
            file_type=self._get_document_type_from_extension(file_ext),
            mime_type=self._get_mime_type_from_extension(file_ext),
            metadata={"error": error_message}
        )

    def _get_document_type_from_extension(self, file_ext: str) -> DocumentType:
        """从文件扩展名获取文档类型。"""
        ext_mapping = {
            '.pdf': DocumentType.PDF,
            '.doc': DocumentType.DOC,
            '.docx': DocumentType.DOCX,
            '.xls': DocumentType.XLS,
            '.xlsx': DocumentType.XLSX,
            '.ppt': DocumentType.PPT,
            '.pptx': DocumentType.PPTX,
            '.txt': DocumentType.TXT,
            '.md': DocumentType.MD,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.xml': DocumentType.XML,
            '.json': DocumentType.JSON,
            '.csv': DocumentType.CSV,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.png': DocumentType.IMAGE,
            '.tiff': DocumentType.IMAGE,
            '.tif': DocumentType.IMAGE,
        }
        return ext_mapping.get(file_ext, DocumentType.TXT)

    def _get_mime_type_from_extension(self, file_ext: str) -> str:
        """从文件扩展名获取MIME类型。"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_ext)
        return mime_type or "application/octet-stream"

    def _update_statistics(self, result: ParseResult, processing_time: float) -> None:
        """
        更新统计信息。

        Args:
            result: 解析结果
            processing_time: 处理时间
        """
        if not result.statistics:
            result.statistics = {}

        result.statistics.update({
            "processing_time": processing_time,
            "text_length": len(result.full_text) if result.full_text else 0,
            "chunk_count": len(result.text_chunks) if result.text_chunks else 0,
            "image_count": len(result.images) if result.images else 0,
            "table_count": len(result.tables) if result.tables else 0,
            "processor": self.parser_name,
            "timestamp": time.time()
        })

    def _log_strategy_selection(
        self,
        file_path: str,
        source_result: SourceDetectionResult,
        recommendation: StrategyRecommendation
    ) -> None:
        """
        记录策略选择日志。

        Args:
            file_path: 文件路径
            source_result: 来源检测结果
            recommendation: 策略推荐结果
        """
        # 这里可以集成日志系统
        print(f"文件: {file_path}")
        print(f"来源类型: {source_result.source_type.value}")
        print(f"置信度: {source_result.confidence:.2f}")
        print(f"推荐策略: {recommendation.recommended_strategy.value}")
        print(f"策略置信度: {recommendation.confidence:.2f}")
        print(f"推理: {'; '.join(recommendation.reasoning)}")
        print("-" * 50)

    def _create_text_chunks(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        page_numbers: Optional[List[int]] = None
    ) -> List[TextChunk]:
        """
        创建文本块。

        Args:
            text: 文本内容
            chunk_size: 块大小
            chunk_overlap: 块重叠
            page_numbers: 页码列表（可选）

        Returns:
            List[TextChunk]: 文本块列表
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # 避免在单词中间分割
            if end < len(text) and not chunk_text.endswith((' ', '\n', '\t')):
                last_space = chunk_text.rfind(' ')
                if last_space > 0:
                    end = start + last_space
                    chunk_text = text[start:end]

            if chunk_text.strip():
                chunk = TextChunk(
                    content=chunk_text.strip(),
                    chunk_id=str(chunk_id),
                    page_number=page_numbers[chunk_id] if page_numbers and chunk_id < len(page_numbers) else None
                )
                chunks.append(chunk)
                chunk_id += 1

            start = max(start + 1, end - chunk_overlap)

        return chunks

    def _extract_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        提取文本统计信息。

        Args:
            text: 文本内容

        Returns:
            Dict[str, Any]: 统计信息
        """
        if not text:
            return {}

        import re

        lines = text.split('\n')
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = re.split(r'[.!?]+', text)
        words = text.split()

        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }

    async def _process_with_timeout(
        self,
        coro,
        timeout_seconds: int,
        error_message: str = "处理超时"
    ):
        """
        带超时的协程执行。

        Args:
            coro: 协程
            timeout_seconds: 超时时间
            error_message: 错误消息

        Returns:
            协程结果
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(
                error_message,
                parser=self.parser_name,
                error_code="TIMEOUT"
            )

    def _validate_file(self, file_path: str) -> None:
        """
        验证文件。

        Args:
            file_path: 文件路径

        Raises:
            ValidationError: 文件验证失败
        """
        if not os.path.exists(file_path):
            raise ValidationError(f"文件不存在: {file_path}")

        if not os.path.isfile(file_path):
            raise ValidationError(f"路径不是文件: {file_path}")

        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValidationError(f"文件为空: {file_path}")

        # 检查文件权限
        if not os.access(file_path, os.R_OK):
            raise ValidationError(f"文件不可读: {file_path}")

    def _cleanup_session(self, session_id: str) -> None:
        """
        清理会话资源。

        Args:
            session_id: 会话ID
        """
        if session_id in self._processing_sessions:
            del self._processing_sessions[session_id]

    def get_active_sessions(self) -> List[str]:
        """获取活跃的会话列表。"""
        return list(self._processing_sessions.keys())

    def get_session_count(self) -> int:
        """获取活跃会话数量。"""
        return len(self._processing_sessions)


# 导出
__all__ = ['BaseProcessor']