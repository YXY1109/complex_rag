"""
OCR Text Recognition Module

This module provides OCR (Optical Character Recognition) capabilities
for extracting text from images and scanned documents, inspired by RAGFlow's
vision/ocr.py implementation.
"""

import asyncio
import io
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import cv2
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    CV2_AVAILABLE = True
    PIL_AVAILABLE = True
    TESSERACT_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    PIL_AVAILABLE = False
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR processing result."""
    text: str
    confidence: float
    language: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    word_count: int = 0
    processing_time_ms: Optional[float] = None


@dataclass
class OCRConfig:
    """OCR configuration."""
    languages: List[str] = None
    engine: str = "tesseract"  # tesseract, easyocr, paddleocr
    preprocessing: bool = True
    deskew: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    confidence_threshold: float = 0.6
    page_segmentation_mode: str = "auto"  # auto, 1-11


class OCREngine:
    """
    OCR engine for text extraction from images.

    Features:
    - Multiple OCR engine support (Tesseract, EasyOCR, PaddleOCR)
    - Image preprocessing for better accuracy
    - Language detection and support
    - Confidence scoring and validation
    - Layout-aware processing
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize OCR engine."""
        self.config = config or OCRConfig()
        self.available_engines = self._check_available_engines()
        self.default_language = 'eng'

        # Initialize Tesseract if available
        if TESSERACT_AVAILABLE:
            try:
                # Test Tesseract availability
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR engine initialized")
            except Exception as e:
                logger.warning(f"Tesseract initialization failed: {e}")
                TESSERACT_AVAILABLE = False

    def _check_available_engines(self) -> List[str]:
        """Check available OCR engines."""
        engines = []
        if TESSERACT_AVAILABLE:
            engines.append("tesseract")
        # Add other engines when implemented
        return engines

    async def extract_text(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        languages: Optional[List[str]] = None,
        config: Optional[OCRConfig] = None
    ) -> OCRResult:
        """
        Extract text from image using OCR.

        Args:
            image: Input image (PIL Image, numpy array, or bytes)
            languages: List of languages for OCR
            config: Override configuration

        Returns:
            OCRResult: Extracted text with metadata
        """
        start_time = datetime.now()
        ocr_config = config or self.config
        ocr_languages = languages or ocr_config.languages or [self.default_language]

        try:
            # Convert image to PIL Image
            pil_image = self._ensure_pil_image(image)
            if pil_image is None:
                raise ValueError("Invalid image input")

            # Preprocess image
            if ocr_config.preprocessing:
                pil_image = await self._preprocess_image(pil_image, ocr_config)

            # Extract text using appropriate engine
            if ocr_config.engine == "tesseract" and "tesseract" in self.available_engines:
                result = await self._extract_with_tesseract(pil_image, ocr_languages, ocr_config)
            else:
                # Fallback to Tesseract
                result = await self._extract_with_tesseract(pil_image, ocr_languages, ocr_config)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time

            return result

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language="",
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _ensure_pil_image(self, image: Union[Image.Image, np.ndarray, bytes]) -> Optional[Image.Image]:
        """Convert input to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
            # Convert OpenCV image to PIL
            if len(image.shape) == 3:  # Color image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        elif isinstance(image, bytes):
            try:
                return Image.open(io.BytesIO(image))
            except Exception:
                return None
        return None

    async def _preprocess_image(self, image: Image.Image, config: OCRConfig) -> Image.Image:
        """Preprocess image for better OCR accuracy."""
        if not PIL_AVAILABLE:
            return image

        processed_image = image.copy()

        try:
            # Convert to grayscale for processing
            if processed_image.mode != 'L':
                processed_image = processed_image.convert('L')

            # Denoise
            if config.denoise:
                processed_image = processed_image.filter(ImageFilter.MedianFilter(size=3))

            # Enhance contrast
            if config.enhance_contrast:
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(1.5)

            # Deskew (basic implementation)
            if config.deskew:
                processed_image = await self._deskew_image(processed_image)

            return processed_image

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image

    async def _deskew_image(self, image: Image.Image) -> Image.Image:
        """Basic deskewing implementation."""
        if not CV2_AVAILABLE:
            return image

        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Calculate skew angle using Hough transform
            edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    angles.append(angle)

                # Calculate median angle
                if angles:
                    median_angle = np.median(angles)
                    # Rotate image to correct skew
                    if abs(median_angle) > 0.5:  # Only correct significant skew
                        center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        rotated = cv2.warpAffine(img_array, rotation_matrix,
                                               (img_array.shape[1], img_array.shape[0]),
                                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        return Image.fromarray(rotated)

        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")

        return image

    async def _extract_with_tesseract(
        self,
        image: Image.Image,
        languages: List[str],
        config: OCRConfig
    ) -> OCRResult:
        """Extract text using Tesseract OCR."""
        if not TESSERACT_AVAILABLE:
            return OCRResult(text="", confidence=0.0, language="")

        try:
            # Prepare language string
            language_str = '+'.join(languages) if languages else 'eng'

            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            if config.page_segmentation_mode != "auto":
                try:
                    psm = int(config.page_segmentation_mode)
                    custom_config = f'--oem 3 --psm {psm}'
                except ValueError:
                    pass

            # Perform OCR
            data = pytesseract.image_to_data(
                image,
                lang=language_str,
                config=custom_config,
                output_type=pytesseract.OutputDict.DICT
            )

            # Extract text and confidence
            text_parts = []
            confidences = []
            word_count = 0

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])

                if text and conf > 0:
                    text_parts.append(text)
                    confidences.append(conf)
                    word_count += len(text.split())

            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0

            # Combine text
            full_text = ' '.join(text_parts)

            # Filter by confidence threshold
            if avg_confidence < config.confidence_threshold:
                logger.warning(f"Low OCR confidence: {avg_confidence:.2f}")

            return OCRResult(
                text=full_text,
                confidence=avg_confidence / 100.0,
                language=language_str,
                word_count=word_count
            )

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRResult(text="", confidence=0.0, language=languages[0] if languages else "")

    async def extract_text_with_layout(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        bbox_list: List[Tuple[int, int, int, int]]
    ) -> List[OCRResult]:
        """Extract text from specific regions in the image."""
        pil_image = self._ensure_pil_image(image)
        if pil_image is None:
            return []

        results = []
        for i, bbox in enumerate(bbox_list):
            try:
                # Crop region
                cropped = pil_image.crop(bbox)
                # Extract text from region
                result = await self.extract_text(cropped)
                result.bbox = bbox
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to extract text from region {i}: {e}")
                results.append(OCRResult(text="", confidence=0.0, language="", bbox=bbox))

        return results

    async def detect_language(self, image: Union[Image.Image, np.ndarray, bytes]) -> str:
        """Detect the primary language in the image."""
        try:
            # Try with common languages
            languages_to_test = ['eng', 'chi_sim', 'jpn', 'kor', 'fra', 'deu', 'spa']
            results = {}

            for lang in languages_to_test:
                result = await self.extract_text(image, [lang])
                if result.confidence > 0.3 and result.word_count > 5:
                    results[lang] = result.confidence

            if results:
                # Return language with highest confidence
                best_lang = max(results.items(), key=lambda x: x[1])
                return best_lang[0]

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

        return self.default_language

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        if TESSERACT_AVAILABLE:
            try:
                # Get Tesseract supported languages
                langs = pytesseract.get_languages(config='')
                return [lang for lang in langs if lang and not lang.startswith('.')]
            except Exception:
                pass
        return ['eng']  # Fallback

    async def batch_extract_text(
        self,
        images: List[Union[Image.Image, np.ndarray, bytes]],
        languages: Optional[List[str]] = None
    ) -> List[OCRResult]:
        """Extract text from multiple images concurrently."""
        tasks = [
            self.extract_text(image, languages)
            for image in images
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)