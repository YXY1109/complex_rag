"""
格式转换器接口抽象类

此模块定义了格式转换服务的抽象接口。
所有格式转换实现都必须继承自这个基类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pydantic import BaseModel, Field
from enum import Enum
import time


class ConversionFormat(str, Enum):
    """转换格式。"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    HTML = "html"
    TXT = "txt"
    MD = "md"
    XML = "xml"
    JSON = "json"
    EPUB = "epub"
    RTF = "rtf"
    ODT = "odt"
    XLSX = "xlsx"
    CSV = "csv"


class ConversionConfig(BaseModel):
    """转换配置模型。"""
    converter_name: str
    source_format: ConversionFormat
    target_format: ConversionFormat

    # 质量设置
    quality: str = Field(default="high", description="转换质量：low, medium, high")
    preserve_formatting: bool = Field(default=True, description="保留格式")
    preserve_images: bool = Field(default=True, description="保留图像")
    preserve_tables: bool = Field(default=True, description="保留表格")

    # PDF特定设置
    pdf_dpi: int = Field(default=300, description="PDF分辨率")
    pdf_password: Optional[str] = Field(default=None, description="PDF密码")

    # 图像设置
    image_quality: int = Field(default=90, description="图像质量（1-100）")
    image_format: str = Field(default="jpeg", description="图像格式")

    # 文本设置
    encoding: str = Field(default="utf-8", description="文本编码")
    line_endings: str = Field(default="unix", description="行结尾：unix, windows, mac")

    # 性能设置
    timeout_seconds: int = Field(default=300, description="超时时间（秒）")
    max_file_size_mb: Optional[int] = Field(default=None, description="最大文件大小（MB）")

    # 自定义选项
    custom_options: Optional[Dict[str, Any]] = Field(default=None, description="自定义选项")


class ConversionResult:
    """转换结果。"""

    def __init__(
        self,
        success: bool,
        data: Optional[Union[bytes, str]] = None,
        file_name: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.file_name = file_name
        self.processing_time_ms = processing_time_ms
        self.error_message = error_message
        self.metadata = metadata or {}


class FormatConverterInterface(ABC):
    """
    格式转换器的抽象接口。

    此类定义了所有格式转换实现必须遵循的契约。
    """

    def __init__(self, config: ConversionConfig):
        """
        使用配置初始化转换器。

        Args:
            config: 转换配置
        """
        self.config = config
        self.converter_name = config.converter_name
        self.source_format = config.source_format
        self.target_format = config.target_format

    @abstractmethod
    async def convert_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> ConversionResult:
        """
        转换文件。

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（可选）

        Returns:
            ConversionResult: 转换结果

        Raises:
            ConversionError: 如果转换失败
        """
        pass

    @abstractmethod
    async def convert_bytes(
        self,
        data: bytes,
        file_name: Optional[str] = None
    ) -> ConversionResult:
        """
        转换字节数据。

        Args:
            data: 输入数据
            file_name: 文件名（可选）

        Returns:
            ConversionResult: 转换结果

        Raises:
            ConversionError: 如果转换失败
        """
        pass

    @abstractmethod
    async def convert_stream(
        self,
        stream: BinaryIO,
        file_name: Optional[str] = None
    ) -> ConversionResult:
        """
        转换流数据。

        Args:
            stream: 输入流
            file_name: 文件名（可选）

        Returns:
            ConversionResult: 转换结果

        Raises:
            ConversionError: 如果转换失败
        """
        pass

    def get_supported_conversions(self) -> List[tuple]:
        """
        获取支持的转换格式对。

        Returns:
            List[tuple]: 支持的（源格式，目标格式）对列表
        """
        return [(self.source_format, self.target_format)]

    def supports_conversion(self, source: ConversionFormat, target: ConversionFormat) -> bool:
        """
        检查是否支持特定的转换。

        Args:
            source: 源格式
            target: 目标格式

        Returns:
            bool: 如果支持转换则为True
        """
        return (source, target) in self.get_supported_conversions()


class ConversionError(Exception):
    """转换错误引发的异常。"""

    def __init__(
        self,
        message: str,
        converter: str = None,
        source_format: str = None,
        target_format: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.converter = converter
        self.source_format = source_format
        self.target_format = target_format
        self.error_code = error_code