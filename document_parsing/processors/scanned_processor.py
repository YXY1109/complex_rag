"""
扫描文档处理器

此模块实现OCR、图片、多模态文档的专用处理器，
参考RAGFlow vision模块中的视觉识别和OCR处理逻辑。
"""

import asyncio
import os
import tempfile
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Any, List, Optional, Union, BinaryIO, Tuple
from pathlib import Path
import io
import uuid
import cv2
import base64

from .base_processor import BaseProcessor
from ..interfaces.parser_interface import (
    ParseResult,
    DocumentMetadata,
    DocumentType,
    ProcessingStrategy,
    TextChunk,
    ImageInfo,
    TableInfo,
    ParseException,
    UnsupportedFormatError
)
from ..strategy_config import ProcessingStrategyConfig


class ScannedDocumentProcessor(BaseProcessor):
    """
    扫描文档处理器。

    专门处理OCR图片、扫描文档、多模态内容等。
    """

    def __init__(self, config):
        """
        初始化扫描文档处理器。

        Args:
            config: 解析器配置
        """
        super().__init__(config)
        self.supported_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'
        }
        self.ocr_engines = {}
        self._init_ocr_engines()

    def _init_ocr_engines(self):
        """初始化OCR引擎。"""
        try:
            # 尝试导入Tesseract
            import pytesseract
            self.ocr_engines['tesseract'] = pytesseract
        except ImportError:
            print("Tesseract OCR不可用")

        try:
            # 尝试导入PaddleOCR
            from paddleocr import PaddleOCR
            self.ocr_engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='ch')
        except ImportError:
            print("PaddleOCR不可用")

        try:
            # 尝试导入EasyOCR
            import easyocr
            self.ocr_engines['easyocr'] = easyocr.Reader(['ch_sim', 'en'])
        except ImportError:
            print("EasyOCR不可用")

    async def initialize(self) -> bool:
        """
        初始化处理器。

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 测试依赖库
            import PIL
            import cv2
            import numpy
            return len(self.ocr_engines) > 0  # 至少需要一个OCR引擎
        except ImportError as e:
            print(f"扫描文档处理器初始化失败，缺少依赖: {e}")
            return False

    async def cleanup(self) -> None:
        """清理处理器资源。"""
        for engine in self.ocr_engines.values():
            if hasattr(engine, 'close'):
                try:
                    engine.close()
                except:
                    pass

    async def _parse_with_config(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        **kwargs
    ) -> ParseResult:
        """
        使用配置解析扫描文档。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 策略配置
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果
        """
        # 验证文件
        self._validate_file(file_path)

        # 预处理图像
        preprocessed_path = await self._preprocess_image(file_path, config)

        try:
            # 根据策略选择处理方式
            if strategy == ProcessingStrategy.MULTIMODAL_ANALYSIS:
                result = await self._parse_multimodal_image(preprocessed_path, config)
            elif strategy == ProcessingStrategy.IMAGE_ANALYSIS:
                result = await self._parse_image_analysis(preprocessed_path, config)
            else:
                result = await self._parse_basic_ocr(preprocessed_path, config)

            return result

        finally:
            # 清理预处理文件
            if preprocessed_path != file_path:
                try:
                    os.unlink(preprocessed_path)
                except OSError:
                    pass

    async def _preprocess_image(self, file_path: str, config: ProcessingStrategyConfig) -> str:
        """
        预处理图像。

        Args:
            file_path: 原始文件路径
            config: 配置

        Returns:
            str: 预处理后的文件路径
        """
        try:
            # 打开图像
            with Image.open(file_path) as img:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 图像增强
                if config.ocr.preprocessing:
                    # 调整对比度
                    if config.ocr.binarization:
                        img = ImageEnhance.Contrast(img, 1.5).convert('L')

                    # 降噪
                    if config.ocr.denoise:
                        img = img.filter(ImageFilter.MedianFilter(size=3))

                    # 纠正倾斜
                    if config.ocr.deskew:
                        img = await self._deskew_image(img)

                # 保存预处理后的图像
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    img.save(temp_file.name, 'PNG', quality=config.image.image_quality)
                    return temp_file.name

        except Exception as e:
            print(f"图像预处理失败: {e}")
            return file_path

    async def _deskew_image(self, img: Image.Image) -> Image.Image:
        """
        纠正图像倾斜。

        Args:
            img: PIL图像

        Returns:
            Image.Image: 纠正后的图像
        """
        try:
            # 转换为OpenCV格式
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 检测倾斜角度
            coords = np.column_stack(np.where(gray > 0))
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # 纠正图像
            (h, w) = cv_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            corrected = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # 转换回PIL格式
            corrected_pil = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            return corrected_pil

        except Exception:
            return img

    async def _parse_basic_ocr(self, image_path: str, config: ProcessingStrategyConfig) -> ParseResult:
        """
        基础OCR解析。

        Args:
            image_path: 图像文件路径
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        ocr_text = ""
        confidence = 0.0

        # 选择OCR引擎
        if config.ocr.provider.value in self.ocr_engines:
            engine = self.ocr_engines[config.ocr.provider.value]

            if config.ocr.provider.value == 'tesseract':
                ocr_text = engine.image_to_string(
                    image_path,
                    lang='+'.join(config.ocr.languages),
                    config=config.ocr.tesseract_config
                )
                confidence = 0.8  # Tesseract默认置信度

            elif config.ocr.provider.value == 'paddleocr':
                result = engine.ocr(image_path, cls=True)
                if result and len(result) > 0 and len(result[0]) > 1:
                    ocr_text = '\n'.join([line[1][0] for line in result[0]])
                    confidence = np.mean([line[1][1] for line in result[0]]) if result[0] else 0.8

            elif config.ocr.provider.value == 'easyocr':
                result = engine.readtext(image_path)
                if result and len(result) > 0:
                    ocr_text = '\n'.join([text[1] for text in result])
                    confidence = np.mean([text[2] for text in result]) if result else 0.8
        else:
            # 备用方案：使用Tesseract
            if 'tesseract' in self.ocr_engines:
                ocr_text = self.ocr_engines['tesseract'].image_to_string(image_path)
                confidence = 0.7
            else:
                raise ParseException("没有可用的OCR引擎", parser=self.parser_name, file_path=image_path)

        # 检查OCR质量
        if confidence < config.ocr.confidence_threshold:
            print(f"OCR置信度较低: {confidence:.2f}")

        # 创建元数据
        file_size = os.path.getsize(image_path)
        file_name = os.path.basename(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        metadata = DocumentMetadata(
            file_name=file_name,
            file_size=file_size,
            file_type=DocumentType.IMAGE,
            mime_type="image/jpeg",
            title=file_name,
            page_count=1,
            word_count=len(ocr_text.split()) if ocr_text else 0,
            character_count=len(ocr_text),
            metadata={
                "image_width": width,
                "image_height": height,
                "ocr_engine": config.ocr.provider.value,
                "ocr_confidence": confidence,
                "ocr_languages": config.ocr.languages
            }
        )

        # 创建文本块
        text_chunks = []
        if ocr_text.strip():
            chunk = TextChunk(
                content=ocr_text.strip(),
                page_number=1,
                chunk_id="ocr_1",
                confidence=confidence
            )
            text_chunks.append(chunk)

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=ocr_text.strip(),
            text_chunks=text_chunks,
            structured_data={
                "ocr_confidence": confidence,
                "ocr_engine": config.ocr.provider.value
            }
        )

    async def _parse_image_analysis(self, image_path: str, config: ProcessingStrategyConfig) -> ParseResult:
        """
        图像分析解析。

        Args:
            image_path: 图像文件路径
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        # 先进行基础OCR
        basic_result = await self._parse_basic_ocr(image_path, config)

        # 添加图像分析结果
        with Image.open(image_path) as img:
            width, height = img.size

            # 图像质量分析
            image_stats = await self._analyze_image_quality(img)

            # 检测图像中的元素
            elements = await self._detect_image_elements(img, config)

        # 更新结构化数据
        basic_result.structured_data.update({
            "image_width": width,
            "image_height": height,
            "image_quality": image_stats,
            "detected_elements": elements
        })

        return basic_result

    async def _parse_multimodal_image(self, image_path: str, config: ProcessingStrategyConfig) -> ParseResult:
        """
        多模态图像解析。

        Args:
            image_path: 图像文件路径
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        # 进行图像分析
        analysis_result = await self._parse_image_analysis(image_path, config)

        # 添加多模态特定功能
        with Image.open(image_path) as img:
            # 检测布局结构
            layout_info = await self._detect_layout_structure(img, config)

            # 提取文本区域（如果适用）
            text_regions = await self._extract_text_regions(img, config)

        # 更新结构化数据
        analysis_result.structured_data.update({
            "layout_structure": layout_info,
            "text_regions": text_regions
        })

        return analysis_result

    async def _analyze_image_quality(self, img: Image.Image) -> Dict[str, Any]:
        """
        分析图像质量。

        Args:
            img: PIL图像

        Returns:
            Dict[str, Any]: 图像质量信息
        """
        try:
            # 转换为OpenCV格式
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 计算图像质量指标
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)

            # 锐利度评分（0-1，越高越好）
            sharpness = min(blur_score / 1000.0, 1.0)

            # 亮度评分
            brightness_score = 1.0 - abs(brightness - 128) / 128.0

            # 对比度评分
            contrast_score = min(contrast / 128.0, 1.0)

            # 整体质量评分
            overall_quality = (sharpness + brightness_score + contrast_score) / 3.0

            return {
                "sharpness": sharpness,
                "brightness_score": brightness_score,
                "contrast_score": contrast_score,
                "overall_quality": overall_quality,
                "mean_brightness": brightness,
                "std_contrast": contrast
            }

        except Exception as e:
            print(f"图像质量分析失败: {e}")
            return {"error": str(e)}

    async def _detect_image_elements(self, img: Image.Image, config: ProcessingStrategyConfig) -> Dict[str, Any]:
        """
        检测图像中的元素。

        Args:
            img: PIL图像
            config: 配置

        Returns:
            Dict[str, Any]: 检测到的元素信息
        """
        try:
            # 转换为OpenCV格式
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            elements = {
                "text_blocks": 0,
                "tables": 0,
                "figures": 0,
                "charts": 0
            }

            # 简单的文本块检测（使用轮廓检测）
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_like_contours = [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 10000]
            elements["text_blocks"] = len(text_like_contours)

            # 检测表格（水平线和垂直线的交点）
            edges_h = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines_h = cv2.HoughLinesP(edges_h, 1, np.pi/180, 100, threshold=0, minLineLength=100)

            edges_v = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines_v = cv2.HoughLinesP(edges_v, 1, np.pi/2, 100, threshold=0, minLineLength=100)

            if lines_h is not None and lines_v is not None:
                elements["tables"] = min(len(lines_h), len(lines_v))

            return elements

        except Exception as e:
            print(f"图像元素检测失败: {e}")
            return {"error": str(e)}

    async def _detect_layout_structure(self, img: Image.Image, config: ProcessingStrategyConfig) -> Dict[str, Any]:
        """
        检测布局结构。

        Args:
            img: PIL图像
            config: 配置

        Returns:
            Dict[str, Any]: 布局结构信息
        """
        try:
            # 转换为OpenCV格式
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 使用投影分析检测列
            height, width = gray.shape
            horizontal_proj = np.sum(gray, axis=1)
            vertical_proj = np.sum(gray, axis=0)

            # 检测可能的列分隔（垂直投影的谷值）
            col_separators = []
            for i in range(1, len(vertical_proj) - 1):
                if (vertical_proj[i] < vertical_proj[i-1] and
                    vertical_proj[i] < vertical_proj[i+1] and
                    vertical_proj[i] < np.mean(vertical_proj) * 0.5):
                    col_separators.append(i)

            # 检测可能的行分隔（水平投影的谷值）
            row_separators = []
            for i in range(1, len(horizontal_proj) - 1):
                if (horizontal_proj[i] < horizontal_proj[i-1] and
                    horizontal_proj[i] < horizontal_proj[i+1] and
                    horizontal_proj[i] < np.mean(horizontal_proj) * 0.5):
                    row_separators.append(i)

            return {
                "columns": len(col_separators) + 1,
                "rows": len(row_separators) + 1,
                "col_separators": col_separators,
                "row_separators": row_separators,
                "image_dimensions": (width, height)
            }

        except Exception as e:
            print(f"布局结构检测失败: {e}")
            return {"error": str(e)}

    async def _extract_text_regions(self, img: Image.Image, config: ProcessingStrategyConfig) -> List[Dict[str, Any]]:
        """
        提取文本区域。

        Args:
            img: PIL图像
            config: 配置

        Returns:
            List[Dict[str, Any]]: 文本区域列表
        """
        try:
            # 转换为OpenCV格式
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 使用MSER检测文本区域
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray, None)

            text_regions = []
            for i, region in enumerate(regions[:20]):  # 限制数量
                if len(region) > 10:  # 过滤小区域
                    hull = cv2.convexHull(region.reshape(-1, 1, 2))
                    x, y, w, h = cv2.boundingRect(hull)

                    text_regions.append({
                        "id": i,
                        "bbox": [x, y, x + w, y + h],
                        "area": cv2.contourArea(hull),
                        "confidence": min(len(region) / 100.0, 1.0)
                    })

            return text_regions

        except Exception as e:
            print(f"文本区域提取失败: {e}")
            return []


# 导出
__all__ = ['ScannedDocumentProcessor']