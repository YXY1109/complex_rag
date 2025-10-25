"""
Scanned Documents Handler

This module provides specialized processing for scanned documents and images
requiring OCR (Optical Character Recognition) and computer vision.
"""

import asyncio
import mimetypes
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import io
import base64

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    CV2_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from ...interfaces.source_interface import (
    SourceHandler,
    FileSource,
    ProcessingStrategy,
    ParseRequest,
    ParseResponse
)
from ...interfaces.parser_interface import (
    DocumentFormat,
    DocumentChunk,
    Metadata
)
from ...services.quality_monitor import QualityMonitor, QualityMetric


@dataclass
class ScannedDocumentFeatures:
    """Features extracted from scanned documents."""
    image_format: str
    resolution: Optional[int]
    is_color: bool
    is_textual: bool
    text_density: float
    noise_level: float
    skew_angle: float
    brightness: float
    contrast: float
    has_tables: bool
    has_columns: bool
    quality_score: float


class ScannedDocumentsHandler(SourceHandler):
    """
    Handler for scanned documents and images requiring OCR.

    Features:
    - Multi-format image support (JPEG, PNG, TIFF, BMP)
    - OCR text extraction with multiple engines
    - Image preprocessing (deskew, denoise, enhance)
    - Layout detection and analysis
    - Table and column detection
    - Quality assessment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize scanned documents handler."""
        super().__init__(FileSource.SCANNED_DOCUMENTS, config)
        self.quality_monitor = QualityMonitor()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'}

    async def connect(self) -> bool:
        """Initialize handler."""
        # Check dependencies
        if not PIL_AVAILABLE:
            print("Warning: PIL/Pillow not available, image processing limited")
        if not TESSERACT_AVAILABLE:
            print("Warning: Tesseract not available, OCR processing limited")
        return True

    async def disconnect(self) -> None:
        """Cleanup handler."""
        pass

    async def can_handle(self, request: ParseRequest) -> bool:
        """Check if this handler can process the request."""
        # Check file extension
        if request.file_path:
            file_ext = Path(request.file_path).suffix.lower()
            if file_ext in self.supported_formats:
                return True

        # Check MIME type
        if request.mime_type:
            image_types = {'image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'image/gif'}
            if request.mime_type in image_types:
                return True

        # Check if content looks like an image
        if request.content:
            try:
                Image.open(io.BytesIO(request.content))
                return True
            except:
                pass

        return False

    async def process(self, request: ParseRequest) -> ParseResponse:
        """Process scanned document."""
        session_id = f"scanned_{datetime.now().timestamp()}"
        processing_start = datetime.now()

        # Start quality monitoring
        quality_session = self.quality_monitor.start_session(
            session_id=session_id,
            file_source=FileSource.SCANNED_DOCUMENTS,
            strategy=request.strategy,
            file_size=len(request.content) if request.content else None
        )

        try:
            # Load and preprocess image
            image = await self._load_image(request.content)
            if image is None:
                raise ValueError("Failed to load image")

            # Analyze image features
            features = await self._analyze_image(image)

            # Preprocess image for better OCR
            processed_image = await self._preprocess_image(image, features, request.custom_params)

            # Extract text using OCR
            text = await self._extract_text(processed_image, request.custom_params)

            # Detect layout elements
            layout_info = await self._detect_layout(processed_image, text)

            # Create chunks
            chunks = await self._create_chunks(text, layout_info, request)

            # Generate metadata
            metadata = await self._extract_metadata(features, layout_info, request)

            # Calculate quality metrics
            await self._calculate_quality_metrics(quality_session, text, features, chunks)

            response = ParseResponse(
                content=text,
                chunks=chunks,
                metadata=metadata,
                format=DocumentFormat.IMAGE,
                processing_time=(datetime.now() - processing_start).total_seconds(),
                success=True
            )

            # End quality monitoring
            self.quality_monitor.end_session(session_id, success=True)

            return response

        except Exception as e:
            # End quality monitoring with error
            self.quality_monitor.end_session(session_id, success=False, error=e)

            return ParseResponse(
                content="",
                chunks=[],
                metadata=Metadata(),
                format=DocumentFormat.IMAGE,
                processing_time=(datetime.now() - processing_start).total_seconds(),
                success=False,
                error=str(e)
            )

    async def _load_image(self, content: bytes) -> Optional[Image.Image]:
        """Load image from content."""
        if not content or not PIL_AVAILABLE:
            return None

        try:
            return Image.open(io.BytesIO(content))
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    async def _analyze_image(self, image: Image.Image) -> ScannedDocumentFeatures:
        """Analyze image features."""
        if not PIL_AVAILABLE:
            return ScannedDocumentFeatures("unknown", None, False, False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0)

        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)

            # Basic features
            image_format = image.format or "unknown"
            is_color = image.mode in ['RGB', 'RGBA']
            resolution = None
            if hasattr(image, 'info') and 'dpi' in image.info:
                resolution = image.info['dpi'][0] if isinstance(image.info['dpi'], (tuple, list)) else image.info['dpi']

            # Calculate brightness and contrast
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray) / 255.0
                contrast = np.std(gray) / 255.0
            else:
                # Fallback calculation using PIL
                gray_image = image.convert('L')
                gray_array = np.array(gray_image)
                brightness = np.mean(gray_array) / 255.0
                contrast = np.std(gray_array) / 255.0

            # Calculate skew angle (basic estimation)
            skew_angle = 0.0  # Would need more sophisticated implementation

            # Calculate text density (placeholder)
            text_density = 0.5  # Would need OCR preview or edge detection

            # Calculate noise level (basic)
            if CV2_AVAILABLE:
                noise = cv2.Laplacian(gray, cv2.CV_64F).var()
                noise_level = min(1.0, noise / 1000)
            else:
                noise_level = 0.3

            # Quality score (basic)
            quality_score = min(1.0, (brightness * 0.3 + contrast * 0.3 + (1 - noise_level) * 0.4))

            return ScannedDocumentFeatures(
                image_format=image_format,
                resolution=resolution,
                is_color=is_color,
                is_textual=text_density > 0.3,
                text_density=text_density,
                noise_level=noise_level,
                skew_angle=skew_angle,
                brightness=brightness,
                contrast=contrast,
                has_tables=False,  # Would need layout detection
                has_columns=False,  # Would need layout detection
                quality_score=quality_score
            )

        except Exception as e:
            # Return default features on error
            return ScannedDocumentFeatures("unknown", None, False, False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0)

    async def _preprocess_image(
        self,
        image: Image.Image,
        features: ScannedDocumentFeatures,
        params: Optional[Dict[str, Any]]
    ) -> Image.Image:
        """Preprocess image for better OCR."""
        if not PIL_AVAILABLE:
            return image

        processed_image = image.copy()

        try:
            # Get preprocessing parameters
            enable_preprocessing = params.get('enable_preprocessing', True) if params else True
            if not enable_preprocessing:
                return processed_image

            # Convert to grayscale for OCR
            if processed_image.mode != 'L':
                processed_image = processed_image.convert('L')

            # Deskew if needed
            if abs(features.skew_angle) > 1.0 and CV2_AVAILABLE:
                # Would implement deskewing here
                pass

            # Enhance contrast
            if features.contrast < 0.3:
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(1.5)

            # Enhance brightness
            if features.brightness < 0.3:
                enhancer = ImageEnhance.Brightness(processed_image)
                processed_image = enhancer.enhance(1.2)
            elif features.brightness > 0.8:
                enhancer = ImageEnhance.Brightness(processed_image)
                processed_image = enhancer.enhance(0.8)

            # Reduce noise
            if features.noise_level > 0.5:
                processed_image = processed_image.filter(ImageFilter.MedianFilter(size=3))

            # Sharpen image
            enhancer = ImageEnhance.Sharpness(processed_image)
            processed_image = enhancer.enhance(1.2)

        except Exception as e:
            print(f"Warning: Image preprocessing failed: {e}")
            return image

        return processed_image

    async def _extract_text(self, image: Image.Image, params: Optional[Dict[str, Any]]) -> str:
        """Extract text from image using OCR."""
        if not TESSERACT_AVAILABLE:
            # Fallback: return placeholder text
            return "[OCR not available - install Tesseract for text extraction]"

        try:
            # Get OCR parameters
            ocr_languages = params.get('ocr_languages', ['eng']) if params else ['eng']
            config = '--oem 3 --psm 6'  # Default Tesseract config

            # Perform OCR
            language_str = '+'.join(ocr_languages)
            text = pytesseract.image_to_string(image, lang=language_str, config=config)

            return text.strip()

        except Exception as e:
            raise ValueError(f"OCR extraction failed: {e}")

    async def _detect_layout(self, image: Image.Image, text: str) -> Dict[str, Any]:
        """Detect layout elements in the document."""
        layout_info = {
            'has_tables': False,
            'has_columns': False,
            'paragraph_count': 0,
            'line_count': 0,
            'estimated_reading_time': 0.0
        }

        try:
            # Basic text analysis
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            layout_info['line_count'] = len(lines)

            # Count paragraphs (empty line separated)
            paragraphs = []
            current_paragraph = []
            for line in lines:
                if line == '':
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                else:
                    current_paragraph.append(line)
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))

            layout_info['paragraph_count'] = len(paragraphs)

            # Estimate reading time (250 words per minute)
            word_count = len(text.split())
            layout_info['estimated_reading_time'] = word_count / 250

            # Basic table detection (look for tab-separated content)
            for line in lines[:20]:  # Check first 20 lines
                if '\t' in line or line.count('|') >= 3:
                    layout_info['has_tables'] = True
                    break

            # Basic column detection (multiple short lines)
            short_lines = [line for line in lines if len(line) < 50 and len(line) > 10]
            if len(short_lines) > len(lines) * 0.6:
                layout_info['has_columns'] = True

        except Exception as e:
            print(f"Warning: Layout detection failed: {e}")

        return layout_info

    async def _create_chunks(
        self,
        text: str,
        layout_info: Dict[str, Any],
        request: ParseRequest
    ) -> List[DocumentChunk]:
        """Create document chunks respecting layout."""
        if not text.strip():
            return []

        # Get chunking parameters
        params = request.custom_params or {}
        chunk_size = params.get('chunk_size', 600)  # Smaller chunks for OCR text
        overlap = params.get('overlap_size', 150)

        chunks = []

        # Try to respect paragraph boundaries
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_length = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_length = len(paragraph)

            if current_length + paragraph_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=Metadata({
                        'chunk_index': len(chunks),
                        'word_count': len(current_chunk.split()),
                        'source': 'scanned_documents',
                        'has_layout_elements': layout_info['has_tables'] or layout_info['has_columns']
                    })
                ))

                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-overlap//6:] if len(words) > overlap//6 else words
                current_chunk = ' '.join(overlap_words) + "\n\n" + paragraph
                current_length = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_length += paragraph_length + 2

        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=Metadata({
                    'chunk_index': len(chunks),
                    'word_count': len(current_chunk.split()),
                    'source': 'scanned_documents',
                    'has_layout_elements': layout_info['has_tables'] or layout_info['has_columns']
                })
            ))

        return chunks

    async def _extract_metadata(
        self,
        features: ScannedDocumentFeatures,
        layout_info: Dict[str, Any],
        request: ParseRequest
    ) -> Metadata:
        """Extract metadata from processed content."""
        metadata_dict = {
            'source_type': 'scanned_documents',
            'processing_strategy': request.strategy.value,
            'image_format': features.image_format,
            'is_color': features.is_color,
            'resolution': features.resolution,
            'quality_score': features.quality_score,
            'brightness': features.brightness,
            'contrast': features.contrast,
            'noise_level': features.noise_level,
            'skew_angle': features.skew_angle,
            'has_tables': layout_info['has_tables'],
            'has_columns': layout_info['has_columns'],
            'paragraph_count': layout_info['paragraph_count'],
            'line_count': layout_info['line_count'],
            'estimated_reading_time': layout_info['estimated_reading_time']
        }

        # Add file path if available
        if request.file_path:
            metadata_dict['file_path'] = request.file_path

        return Metadata(metadata_dict)

    async def _calculate_quality_metrics(
        self,
        session_id: str,
        text: str,
        features: ScannedDocumentFeatures,
        chunks: List[DocumentChunk]
    ):
        """Calculate quality metrics for the processing session."""
        # OCR quality (based on text quality and image features)
        if text.strip():
            # Basic text quality indicators
            word_count = len(text.split())
            avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0

            # Quality score based on readability
            ocr_quality = min(1.0, (
                (features.quality_score * 0.4) +
                (min(1.0, avg_word_length / 6) * 0.3) +  # Reasonable word length
                (min(1.0, word_count / 100) * 0.3)  # Sufficient text
            ))
        else:
            ocr_quality = 0.0

        self.quality_monitor.add_measurement(
            session_id, QualityMetric.TEXT_EXTRACTION, ocr_quality
        )

        # Image quality
        image_quality = features.quality_score
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.ACCURACY, image_quality
        )

        # Processing speed (relative to image size)
        processing_speed = min(1.0, len(text) / (features.text_density * 1000 + 1))
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.PROCESSING_SPEED, processing_speed
        )