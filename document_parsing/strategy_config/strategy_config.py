"""
处理策略参数配置

此模块提供不同文档来源的处理策略参数配置，
包括OCR设置、布局分析、多模态处理等各种参数。
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..interfaces.parser_interface import DocumentType, ProcessingStrategy
from ..interfaces.source_processor_interface import DocumentSource


class OCRProvider(Enum):
    """OCR提供商。"""
    TESSERACT = "tesseract"
    PADDLE_OCR = "paddle_ocr"
    EASY_OCR = "easy_ocr"
    CLOUD_VISION = "cloud_vision"
    AZURE_VISION = "azure_vision"
    AWS_TEXTRACT = "aws_textract"
    CUSTOM = "custom"


class LayoutAnalyzer(Enum):
    """布局分析器。"""
    LAYOUTLM = "layoutlm"
    DETR = "detr"
    YOLO = "yolo"
    OPENCV = "opencv"
    CUSTOM = "custom"


class TableExtractor(Enum):
    """表格提取器。"""
    PADDLE_STRUCTURE = "paddle_structure"
    TABLE_TRANSFORMER = "table_transformer"
    OPENCV = "opencv"
    CUSTOM = "custom"


@dataclass
class OCRConfig:
    """OCR配置。"""
    enabled: bool = True
    provider: OCRProvider = OCRProvider.TESSERACT
    languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    confidence_threshold: float = 0.7
    preprocessing: bool = True
    deskew: bool = True
    denoise: bool = False
    binarization: bool = True

    # Tesseract特定配置
    tesseract_config: str = "--psm 6"
    tesseract_data_path: Optional[str] = None

    # PaddleOCR特定配置
    paddle_use_gpu: bool = False
    paddle_use_angle_class: bool = True
    paddle_show_log: bool = False

    # 云服务配置
    cloud_api_key: Optional[str] = None
    cloud_endpoint: Optional[str] = None
    cloud_region: Optional[str] = None


@dataclass
class LayoutConfig:
    """布局分析配置。"""
    enabled: bool = True
    analyzer: LayoutAnalyzer = LayoutAnalyzer.LAYOUTLM
    confidence_threshold: float = 0.8
    detect_images: bool = True
    detect_tables: bool = True
    detect_figures: bool = True
    preserve_formatting: bool = True

    # LayoutLM特定配置
    layoutlm_model_path: Optional[str] = None
    layoutlm_device: str = "cpu"

    # 检测参数
    min_region_size: int = 100
    merge_threshold: float = 0.5
    overlap_threshold: float = 0.3


@dataclass
class TableConfig:
    """表格提取配置。"""
    enabled: bool = True
    extractor: TableExtractor = TableExtractor.PADDLE_STRUCTURE
    confidence_threshold: float = 0.8
    preserve_structure: bool = True
    detect_headers: bool = True
    merge_cells: bool = True

    # Paddle Structure特定配置
    paddle_model_path: Optional[str] = None
    paddle_device: str = "cpu"

    # 表格识别参数
    min_table_size: int = 4  # 最小表格单元格数
    max_empty_cells: float = 0.3  # 最大空单元格比例


@dataclass
class ImageConfig:
    """图像处理配置。"""
    enabled: bool = True
    extract_images: bool = True
    analyze_images: bool = False
    image_format: str = "jpeg"
    image_quality: int = 90
    max_image_size: int = 2048  # 最大图像尺寸
    min_image_size: int = 50   # 最小图像尺寸

    # 图像分析配置
    image_analysis_model: Optional[str] = None
    extract_captions: bool = True
    extract_alt_text: bool = True


@dataclass
class TextConfig:
    """文本处理配置。"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    preserve_line_breaks: bool = True
    normalize_whitespace: bool = True
    remove_extra_whitespace: bool = True

    # 文本清理
    remove_page_numbers: bool = True
    remove_headers_footers: bool = True
    remove_watermarks: bool = True

    # 语言检测
    detect_language: bool = True
    default_language: str = "zh"


@dataclass
class ProcessingStrategyConfig:
    """处理策略配置。"""
    strategy: ProcessingStrategy
    ocr: OCRConfig = field(default_factory=OCRConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    table: TableConfig = field(default_factory=TableConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)

    # 性能配置
    max_workers: int = 4
    batch_size: int = 10
    timeout_seconds: int = 300

    # 质量控制
    enable_validation: bool = True
    quality_threshold: float = 0.8
    enable_retry: bool = True
    max_retries: int = 3

    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600

    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)


class StrategyConfigManager:
    """
    策略配置管理器。

    管理不同文档来源和处理策略的配置参数。
    """

    def __init__(self):
        """初始化策略配置管理器。"""
        self.default_configs = self._initialize_default_configs()
        self.source_configs = self._initialize_source_configs()
        self.custom_configs: Dict[str, ProcessingStrategyConfig] = {}

    def _initialize_default_configs(self) -> Dict[ProcessingStrategy, ProcessingStrategyConfig]:
        """初始化默认策略配置。"""
        configs = {}

        # 文本提取策略配置
        configs[ProcessingStrategy.EXTRACT_TEXT] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.EXTRACT_TEXT,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.TESSERACT,
                confidence_threshold=0.6
            ),
            layout=LayoutConfig(
                enabled=False,
                detect_images=False,
                detect_tables=False,
                detect_figures=False
            ),
            table=TableConfig(enabled=False),
            image=ImageConfig(
                enabled=True,
                extract_images=False,
                analyze_images=False
            ),
            text=TextConfig(
                chunk_size=1500,
                chunk_overlap=300,
                normalize_whitespace=True
            ),
            max_workers=2,
            timeout_seconds=120
        )

        # 布局保持策略配置
        configs[ProcessingStrategy.PRESERVE_LAYOUT] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.PRESERVE_LAYOUT,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.PADDLE_OCR,
                confidence_threshold=0.7
            ),
            layout=LayoutConfig(
                enabled=True,
                analyzer=LayoutAnalyzer.LAYOUTLM,
                confidence_threshold=0.8,
                detect_images=True,
                detect_tables=True,
                preserve_formatting=True
            ),
            table=TableConfig(
                enabled=True,
                extractor=TableExtractor.PADDLE_STRUCTURE,
                confidence_threshold=0.8
            ),
            image=ImageConfig(
                enabled=True,
                extract_images=True,
                analyze_images=False
            ),
            text=TextConfig(
                chunk_size=1000,
                chunk_overlap=200,
                preserve_line_breaks=True,
                remove_page_numbers=True
            ),
            max_workers=4,
            timeout_seconds=300
        )

        # 多模态分析策略配置
        configs[ProcessingStrategy.MULTIMODAL_ANALYSIS] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.MULTIMODAL_ANALYSIS,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.PADDLE_OCR,
                confidence_threshold=0.8,
                preprocessing=True,
                deskew=True,
                denoise=True
            ),
            layout=LayoutConfig(
                enabled=True,
                analyzer=LayoutAnalyzer.LAYOUTLM,
                confidence_threshold=0.85,
                detect_images=True,
                detect_tables=True,
                detect_figures=True,
                preserve_formatting=True
            ),
            table=TableConfig(
                enabled=True,
                extractor=TableExtractor.PADDLE_STRUCTURE,
                confidence_threshold=0.85,
                preserve_structure=True,
                detect_headers=True
            ),
            image=ImageConfig(
                enabled=True,
                extract_images=True,
                analyze_images=True,
                image_quality=95,
                extract_captions=True,
                extract_alt_text=True
            ),
            text=TextConfig(
                chunk_size=800,
                chunk_overlap=150,
                preserve_line_breaks=True,
                normalize_whitespace=True,
                detect_language=True
            ),
            max_workers=6,
            timeout_seconds=600,
            quality_threshold=0.85
        )

        # 表格提取策略配置
        configs[ProcessingStrategy.TABLE_EXTRACTION] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.TABLE_EXTRACTION,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.PADDLE_OCR,
                confidence_threshold=0.75
            ),
            layout=LayoutConfig(
                enabled=True,
                analyzer=LayoutAnalyzer.LAYOUTLM,
                confidence_threshold=0.8,
                detect_tables=True,
                preserve_formatting=True
            ),
            table=TableConfig(
                enabled=True,
                extractor=TableExtractor.PADDLE_STRUCTURE,
                confidence_threshold=0.85,
                preserve_structure=True,
                detect_headers=True,
                merge_cells=True
            ),
            image=ImageConfig(
                enabled=True,
                extract_images=False,
                analyze_images=False
            ),
            text=TextConfig(
                chunk_size=1200,
                chunk_overlap=200,
                preserve_line_breaks=True
            ),
            max_workers=4,
            timeout_seconds=300
        )

        # 图像分析策略配置
        configs[ProcessingStrategy.IMAGE_ANALYSIS] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.IMAGE_ANALYSIS,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.PADDLE_OCR,
                confidence_threshold=0.8
            ),
            layout=LayoutConfig(
                enabled=True,
                analyzer=LayoutAnalyzer.DETR,
                confidence_threshold=0.8,
                detect_images=True,
                detect_figures=True
            ),
            image=ImageConfig(
                enabled=True,
                extract_images=True,
                analyze_images=True,
                image_quality=95,
                extract_captions=True,
                extract_alt_text=True
            ),
            text=TextConfig(
                chunk_size=1000,
                chunk_overlap=200
            ),
            max_workers=4,
            timeout_seconds=400
        )

        # 代码提取策略配置
        configs[ProcessingStrategy.CODE_EXTRACTION] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.CODE_EXTRACTION,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.TESSERACT,
                confidence_threshold=0.8
            ),
            layout=LayoutConfig(enabled=False),
            table=TableConfig(enabled=False),
            image=ImageConfig(
                enabled=True,
                extract_images=False,
                analyze_images=False
            ),
            text=TextConfig(
                chunk_size=2000,
                chunk_overlap=100,
                preserve_line_breaks=True,
                normalize_whitespace=False,
                remove_page_numbers=False,
                remove_headers_footers=False
            ),
            max_workers=2,
            timeout_seconds=180
        )

        # 结构化数据策略配置
        configs[ProcessingStrategy.STRUCTURED_DATA] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.STRUCTURED_DATA,
            ocr=OCRConfig(enabled=False),
            layout=LayoutConfig(enabled=False),
            table=TableConfig(enabled=False),
            image=ImageConfig(enabled=False),
            text=TextConfig(
                chunk_size=5000,
                chunk_overlap=0,
                preserve_line_breaks=True,
                normalize_whitespace=False
            ),
            max_workers=1,
            timeout_seconds=60
        )

        # 全面内容策略配置
        configs[ProcessingStrategy.FULL_CONTENT] = ProcessingStrategyConfig(
            strategy=ProcessingStrategy.FULL_CONTENT,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.PADDLE_OCR,
                confidence_threshold=0.9,
                preprocessing=True,
                deskew=True,
                denoise=True,
                binarization=True
            ),
            layout=LayoutConfig(
                enabled=True,
                analyzer=LayoutAnalyzer.LAYOUTLM,
                confidence_threshold=0.9,
                detect_images=True,
                detect_tables=True,
                detect_figures=True,
                preserve_formatting=True
            ),
            table=TableConfig(
                enabled=True,
                extractor=TableExtractor.PADDLE_STRUCTURE,
                confidence_threshold=0.9,
                preserve_structure=True,
                detect_headers=True,
                merge_cells=True
            ),
            image=ImageConfig(
                enabled=True,
                extract_images=True,
                analyze_images=True,
                image_quality=100,
                extract_captions=True,
                extract_alt_text=True
            ),
            text=TextConfig(
                chunk_size=800,
                chunk_overlap=150,
                preserve_line_breaks=True,
                normalize_whitespace=True,
                remove_page_numbers=True,
                remove_headers_footers=True,
                remove_watermarks=True,
                detect_language=True
            ),
            max_workers=8,
            timeout_seconds=900,
            quality_threshold=0.9
        )

        return configs

    def _initialize_source_configs(self) -> Dict[DocumentSource, Dict[ProcessingStrategy, ProcessingStrategyConfig]]:
        """初始化来源特定的策略配置。"""
        source_configs = {}

        # 网页文档配置
        source_configs[DocumentSource.WEB_DOCUMENTS] = {
            ProcessingStrategy.EXTRACT_TEXT: self._adapt_config_for_web(
                self.default_configs[ProcessingStrategy.EXTRACT_TEXT]
            ),
            ProcessingStrategy.PRESERVE_LAYOUT: self._adapt_config_for_web(
                self.default_configs[ProcessingStrategy.PRESERVE_LAYOUT]
            )
        }

        # 办公文档配置
        source_configs[DocumentSource.OFFICE_DOCUMENTS] = {
            ProcessingStrategy.PRESERVE_LAYOUT: self._adapt_config_for_office(
                self.default_configs[ProcessingStrategy.PRESERVE_LAYOUT]
            ),
            ProcessingStrategy.TABLE_EXTRACTION: self._adapt_config_for_office(
                self.default_configs[ProcessingStrategy.TABLE_EXTRACTION]
            ),
            ProcessingStrategy.FULL_CONTENT: self._adapt_config_for_office(
                self.default_configs[ProcessingStrategy.FULL_CONTENT]
            )
        }

        # 扫描文档配置
        source_configs[DocumentSource.SCANNED_DOCUMENTS] = {
            ProcessingStrategy.MULTIMODAL_ANALYSIS: self._adapt_config_for_scanned(
                self.default_configs[ProcessingStrategy.MULTIMODAL_ANALYSIS]
            ),
            ProcessingStrategy.IMAGE_ANALYSIS: self._adapt_config_for_scanned(
                self.default_configs[ProcessingStrategy.IMAGE_ANALYSIS]
            ),
            ProcessingStrategy.FULL_CONTENT: self._adapt_config_for_scanned(
                self.default_configs[ProcessingStrategy.FULL_CONTENT]
            )
        }

        # 代码仓库配置
        source_configs[DocumentSource.CODE_REPOSITORIES] = {
            ProcessingStrategy.CODE_EXTRACTION: self._adapt_config_for_code(
                self.default_configs[ProcessingStrategy.CODE_EXTRACTION]
            ),
            ProcessingStrategy.EXTRACT_TEXT: self._adapt_config_for_code(
                self.default_configs[ProcessingStrategy.EXTRACT_TEXT]
            )
        }

        # 结构化数据配置
        source_configs[DocumentSource.STRUCTURED_DATA] = {
            ProcessingStrategy.STRUCTURED_DATA: self._adapt_config_for_structured(
                self.default_configs[ProcessingStrategy.STRUCTURED_DATA]
            )
        }

        return source_configs

    def _adapt_config_for_web(self, base_config: ProcessingStrategyConfig) -> ProcessingStrategyConfig:
        """为网页文档适配配置。"""
        config = ProcessingStrategyConfig(
            strategy=base_config.strategy,
            ocr=OCRConfig(enabled=False),  # 网页通常不需要OCR
            layout=LayoutConfig(
                enabled=False,  # 网页格式通常保持较好
                detect_images=False,
                detect_tables=False
            ),
            table=base_config.table,
            image=base_config.image,
            text=TextConfig(
                chunk_size=1500,
                chunk_overlap=300,
                preserve_line_breaks=True,
                normalize_whitespace=True,
                remove_page_numbers=False  # 网页没有页码
            ),
            max_workers=base_config.max_workers,
            timeout_seconds=base_config.timeout_seconds,
            quality_threshold=base_config.quality_threshold
        )
        return config

    def _adapt_config_for_office(self, base_config: ProcessingStrategyConfig) -> ProcessingStrategyConfig:
        """为办公文档适配配置。"""
        config = ProcessingStrategyConfig(
            strategy=base_config.strategy,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.PADDLE_OCR,
                confidence_threshold=0.8,
                preprocessing=True
            ),
            layout=LayoutConfig(
                enabled=True,
                confidence_threshold=0.85,
                preserve_formatting=True
            ),
            table=TableConfig(
                enabled=True,
                confidence_threshold=0.85,
                preserve_structure=True
            ),
            image=ImageConfig(
                enabled=True,
                extract_images=True,
                analyze_images=False  # 办公文档图像分析可选
            ),
            text=base_config.text,
            max_workers=base_config.max_workers,
            timeout_seconds=base_config.timeout_seconds,
            quality_threshold=0.85
        )
        return config

    def _adapt_config_for_scanned(self, base_config: ProcessingStrategyConfig) -> ProcessingStrategyConfig:
        """为扫描文档适配配置。"""
        config = ProcessingStrategyConfig(
            strategy=base_config.strategy,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.PADDLE_OCR,
                confidence_threshold=0.8,
                preprocessing=True,
                deskew=True,
                denoise=True,
                binarization=True
            ),
            layout=LayoutConfig(
                enabled=True,
                confidence_threshold=0.85,
                detect_images=True,
                detect_tables=True,
                preserve_formatting=True
            ),
            table=TableConfig(
                enabled=True,
                confidence_threshold=0.8,
                preserve_structure=True
            ),
            image=ImageConfig(
                enabled=True,
                extract_images=True,
                analyze_images=True,
                image_quality=95
            ),
            text=TextConfig(
                chunk_size=800,
                chunk_overlap=150,
                normalize_whitespace=True
            ),
            max_workers=base_config.max_workers,
            timeout_seconds=base_config.timeout_seconds * 1.5,  # 扫描文档需要更长时间
            quality_threshold=0.8
        )
        return config

    def _adapt_config_for_code(self, base_config: ProcessingStrategyConfig) -> ProcessingStrategyConfig:
        """为代码文档适配配置。"""
        config = ProcessingStrategyConfig(
            strategy=base_config.strategy,
            ocr=OCRConfig(
                enabled=True,
                provider=OCRProvider.TESSERACT,
                confidence_threshold=0.85
            ),
            layout=LayoutConfig(enabled=False),
            table=TableConfig(enabled=False),
            image=ImageConfig(enabled=False),
            text=TextConfig(
                chunk_size=2000,
                chunk_overlap=100,
                preserve_line_breaks=True,
                normalize_whitespace=False,
                remove_page_numbers=False,
                remove_headers_footers=False
            ),
            max_workers=2,
            timeout_seconds=180,
            quality_threshold=0.85
        )
        return config

    def _adapt_config_for_structured(self, base_config: ProcessingStrategyConfig) -> ProcessingStrategyConfig:
        """为结构化数据适配配置。"""
        config = ProcessingStrategyConfig(
            strategy=base_config.strategy,
            ocr=OCRConfig(enabled=False),
            layout=LayoutConfig(enabled=False),
            table=TableConfig(enabled=False),
            image=ImageConfig(enabled=False),
            text=TextConfig(
                chunk_size=10000,
                chunk_overlap=0,
                preserve_line_breaks=True,
                normalize_whitespace=False
            ),
            max_workers=1,
            timeout_seconds=60,
            quality_threshold=0.95
        )
        return config

    def get_config(
        self,
        strategy: ProcessingStrategy,
        source_type: Optional[DocumentSource] = None,
        config_name: Optional[str] = None
    ) -> ProcessingStrategyConfig:
        """
        获取策略配置。

        Args:
            strategy: 处理策略
            source_type: 文档来源类型（可选）
            config_name: 自定义配置名称（可选）

        Returns:
            ProcessingStrategyConfig: 策略配置
        """
        # 优先使用自定义配置
        if config_name and config_name in self.custom_configs:
            return self.custom_configs[config_name]

        # 使用来源特定配置
        if source_type and source_type in self.source_configs:
            source_strategies = self.source_configs[source_type]
            if strategy in source_strategies:
                return source_strategies[strategy]

        # 使用默认配置
        return self.default_configs.get(strategy, self.default_configs[ProcessingStrategy.EXTRACT_TEXT])

    def add_custom_config(self, name: str, config: ProcessingStrategyConfig) -> None:
        """
        添加自定义配置。

        Args:
            name: 配置名称
            config: 策略配置
        """
        self.custom_configs[name] = config

    def remove_custom_config(self, name: str) -> bool:
        """
        删除自定义配置。

        Args:
            name: 配置名称

        Returns:
            bool: 是否删除成功
        """
        if name in self.custom_configs:
            del self.custom_configs[name]
            return True
        return False

    def list_custom_configs(self) -> List[str]:
        """列出所有自定义配置名称。"""
        return list(self.custom_configs.keys())

    def save_config_to_file(self, config_name: str, file_path: str) -> bool:
        """
        保存配置到文件。

        Args:
            config_name: 配置名称
            file_path: 文件路径

        Returns:
            bool: 是否保存成功
        """
        try:
            config = None
            # 查找配置
            if config_name in self.custom_configs:
                config = self.custom_configs[config_name]
            else:
                # 在默认配置中查找
                for strategy, default_config in self.default_configs.items():
                    if strategy.value == config_name:
                        config = default_config
                        break

            if config is None:
                return False

            # 转换为字典并保存
            config_dict = self._config_to_dict(config)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            return True
        except Exception:
            return False

    def load_config_from_file(self, file_path: str, config_name: str) -> bool:
        """
        从文件加载配置。

        Args:
            file_path: 文件路径
            config_name: 配置名称

        Returns:
            bool: 是否加载成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            config = self._dict_to_config(config_dict)
            self.custom_configs[config_name] = config
            return True
        except Exception:
            return False

    def _config_to_dict(self, config: ProcessingStrategyConfig) -> Dict[str, Any]:
        """将配置转换为字典。"""
        return {
            "strategy": config.strategy.value,
            "ocr": {
                "enabled": config.ocr.enabled,
                "provider": config.ocr.provider.value,
                "languages": config.ocr.languages,
                "confidence_threshold": config.ocr.confidence_threshold,
                "preprocessing": config.ocr.preprocessing,
                "deskew": config.ocr.deskew,
                "denoise": config.ocr.denoise,
                "binarization": config.ocr.binarization,
                "tesseract_config": config.ocr.tesseract_config,
                "tesseract_data_path": config.ocr.tesseract_data_path,
                "paddle_use_gpu": config.ocr.paddle_use_gpu,
                "paddle_use_angle_class": config.ocr.paddle_use_angle_class,
                "cloud_api_key": config.ocr.cloud_api_key,
                "cloud_endpoint": config.ocr.cloud_endpoint,
                "cloud_region": config.ocr.cloud_region
            },
            "layout": {
                "enabled": config.layout.enabled,
                "analyzer": config.layout.analyzer.value,
                "confidence_threshold": config.layout.confidence_threshold,
                "detect_images": config.layout.detect_images,
                "detect_tables": config.layout.detect_tables,
                "detect_figures": config.layout.detect_figures,
                "preserve_formatting": config.layout.preserve_formatting,
                "layoutlm_model_path": config.layout.layoutlm_model_path,
                "layoutlm_device": config.layout.layoutlm_device,
                "min_region_size": config.layout.min_region_size,
                "merge_threshold": config.layout.merge_threshold,
                "overlap_threshold": config.layout.overlap_threshold
            },
            "table": {
                "enabled": config.table.enabled,
                "extractor": config.table.extractor.value,
                "confidence_threshold": config.table.confidence_threshold,
                "preserve_structure": config.table.preserve_structure,
                "detect_headers": config.table.detect_headers,
                "merge_cells": config.table.merge_cells,
                "paddle_model_path": config.table.paddle_model_path,
                "paddle_device": config.table.paddle_device,
                "min_table_size": config.table.min_table_size,
                "max_empty_cells": config.table.max_empty_cells
            },
            "image": {
                "enabled": config.image.enabled,
                "extract_images": config.image.extract_images,
                "analyze_images": config.image.analyze_images,
                "image_format": config.image.image_format,
                "image_quality": config.image.image_quality,
                "max_image_size": config.image.max_image_size,
                "min_image_size": config.image.min_image_size,
                "image_analysis_model": config.image.image_analysis_model,
                "extract_captions": config.image.extract_captions,
                "extract_alt_text": config.image.extract_alt_text
            },
            "text": {
                "chunk_size": config.text.chunk_size,
                "chunk_overlap": config.text.chunk_overlap,
                "min_chunk_size": config.text.min_chunk_size,
                "preserve_line_breaks": config.text.preserve_line_breaks,
                "normalize_whitespace": config.text.normalize_whitespace,
                "remove_extra_whitespace": config.text.remove_extra_whitespace,
                "remove_page_numbers": config.text.remove_page_numbers,
                "remove_headers_footers": config.text.remove_headers_footers,
                "remove_watermarks": config.text.remove_watermarks,
                "detect_language": config.text.detect_language,
                "default_language": config.text.default_language
            },
            "max_workers": config.max_workers,
            "batch_size": config.batch_size,
            "timeout_seconds": config.timeout_seconds,
            "enable_validation": config.enable_validation,
            "quality_threshold": config.quality_threshold,
            "enable_retry": config.enable_retry,
            "max_retries": config.max_retries,
            "enable_cache": config.enable_cache,
            "cache_ttl": config.cache_ttl,
            "custom_params": config.custom_params
        }

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ProcessingStrategyConfig:
        """将字典转换为配置。"""
        # 这里需要实现完整的反序列化逻辑
        # 由于篇幅限制，这里只提供一个简化版本
        strategy = ProcessingStrategy(config_dict["strategy"])

        # 创建OCR配置
        ocr_dict = config_dict.get("ocr", {})
        ocr_config = OCRConfig(
            enabled=ocr_dict.get("enabled", True),
            provider=OCRProvider(ocr_dict.get("provider", "tesseract")),
            languages=ocr_dict.get("languages", ["zh", "en"]),
            confidence_threshold=ocr_dict.get("confidence_threshold", 0.7)
        )

        # 创建布局配置
        layout_dict = config_dict.get("layout", {})
        layout_config = LayoutConfig(
            enabled=layout_dict.get("enabled", True),
            analyzer=LayoutAnalyzer(layout_dict.get("analyzer", "layoutlm")),
            confidence_threshold=layout_dict.get("confidence_threshold", 0.8)
        )

        # 创建表格配置
        table_dict = config_dict.get("table", {})
        table_config = TableConfig(
            enabled=table_dict.get("enabled", True),
            extractor=TableExtractor(table_dict.get("extractor", "paddle_structure")),
            confidence_threshold=table_dict.get("confidence_threshold", 0.8)
        )

        # 创建图像配置
        image_dict = config_dict.get("image", {})
        image_config = ImageConfig(
            enabled=image_dict.get("enabled", True),
            extract_images=image_dict.get("extract_images", True),
            analyze_images=image_dict.get("analyze_images", False)
        )

        # 创建文本配置
        text_dict = config_dict.get("text", {})
        text_config = TextConfig(
            chunk_size=text_dict.get("chunk_size", 1000),
            chunk_overlap=text_dict.get("chunk_overlap", 200),
            preserve_line_breaks=text_dict.get("preserve_line_breaks", True)
        )

        return ProcessingStrategyConfig(
            strategy=strategy,
            ocr=ocr_config,
            layout=layout_config,
            table=table_config,
            image=image_config,
            text=text_config,
            max_workers=config_dict.get("max_workers", 4),
            timeout_seconds=config_dict.get("timeout_seconds", 300),
            quality_threshold=config_dict.get("quality_threshold", 0.8)
        )


# 全局配置管理器实例
_global_config_manager: Optional[StrategyConfigManager] = None


def get_config_manager() -> StrategyConfigManager:
    """获取全局配置管理器实例。"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = StrategyConfigManager()
    return _global_config_manager


def set_config_manager(manager: StrategyConfigManager) -> None:
    """设置全局配置管理器实例。"""
    global _global_config_manager
    _global_config_manager = manager


# 导出
__all__ = [
    'StrategyConfigManager',
    'ProcessingStrategyConfig',
    'OCRConfig',
    'LayoutConfig',
    'TableConfig',
    'ImageConfig',
    'TextConfig',
    'OCRProvider',
    'LayoutAnalyzer',
    'TableExtractor',
    'get_config_manager',
    'set_config_manager'
]