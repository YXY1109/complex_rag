"""
Layout Recognition Module

This module provides advanced document layout recognition capabilities,
inspired by RAGFlow's vision/layout_recognizer.py implementation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import re

try:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    CV2_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    PIL_AVAILABLE = False

from .ocr import OCREngine, OCRResult
from .recognizer import VisionRecognizer, RecognitionTask

logger = logging.getLogger(__name__)


class LayoutElementType(str, Enum):
    """Layout element types."""
    TEXT = "text"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    FIGURE = "figure"
    FOOTER = "footer"
    HEADER = "header"
    SIDEBAR = "sidebar"
    COLUMN = "column"
    BLOCKQUOTE = "blockquote"
    CODE = "code"
    FORM = "form"
    UNKNOWN = "unknown"


@dataclass
class LayoutRegion:
    """Layout region representation."""
    element_type: LayoutElementType
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    properties: Dict[str, Any]
    children: List['LayoutRegion'] = None


@dataclass
class LayoutConfig:
    """Layout recognition configuration."""
    enable_table_detection: bool = True
    enable_column_detection: bool = True
    enable_heading_detection: bool = True
    enable_list_detection: bool = True
    min_region_size: int = 10
    confidence_threshold: float = 0.6
    merge_similar_regions: bool = True


class LayoutRecognizer:
    """
    Advanced document layout recognizer.

    Features:
    - Multi-element type detection (text, tables, images, etc.)
    - Column and paragraph segmentation
    - Heading and title identification
    - Table structure analysis
    - Hierarchical layout parsing
    - Region merging and optimization
    """

    def __init__(self, config: Optional[LayoutConfig] = None):
        """Initialize layout recognizer."""
        self.config = config or LayoutConfig()
        self.ocr_engine = OCREngine()
        self.vision_recognizer = VisionRecognizer()

    async def recognize_layout(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        ocr_result: Optional[OCRResult] = None
    ) -> List[LayoutRegion]:
        """
        Recognize document layout.

        Args:
            image: Input image
            ocr_result: Optional pre-computed OCR result

        Returns:
            List[LayoutRegion]: Detected layout regions
        """
        try:
            # Get OCR result if not provided
            if ocr_result is None:
                ocr_result = await self.ocr_engine.extract_text(image)

            if not ocr_result.text.strip():
                return []

            # Perform layout analysis
            layout_regions = await self._analyze_layout(image, ocr_result)

            # Optimize layout
            if self.config.merge_similar_regions:
                layout_regions = await self._merge_similar_regions(layout_regions)

            return layout_regions

        except Exception as e:
            logger.error(f"Layout recognition failed: {e}")
            return []

    async def _analyze_layout(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        ocr_result: OCRResult
    ) -> List[LayoutRegion]:
        """Analyze document layout using multiple techniques."""
        regions = []

        # Convert to PIL Image if needed
        pil_image = self._ensure_pil_image(image)

        # Split text into lines with positions
        text_lines = self._split_text_with_positions(ocr_result.text)

        # Detect different layout elements
        heading_regions = await self._detect_headings(text_lines)
        table_regions = await self._detect_tables(text_lines, pil_image)
        list_regions = await self._detect_lists(text_lines)
        column_regions = await self._detect_columns(text_lines)

        # Identify remaining text as paragraphs
        paragraph_regions = await self._identify_paragraphs(
            text_lines, heading_regions, table_regions, list_regions, column_regions
        )

        # Combine all regions
        regions.extend(heading_regions)
        regions.extend(table_regions)
        regions.extend(list_regions)
        regions.extend(column_regions)
        regions.extend(paragraph_regions)

        # Sort regions by position
        regions.sort(key=lambda r: r.bbox[1])  # Sort by y-coordinate

        return regions

    def _ensure_pil_image(self, image: Union[Image.Image, np.ndarray, bytes]) -> Optional[Image.Image]:
        """Convert input to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        elif isinstance(image, bytes):
            try:
                return Image.open(image)
            except:
                return None
        return None

    def _split_text_with_positions(self, text: str) -> List[Dict[str, Any]]:
        """Split text into lines with position information."""
        lines = []
        y_position = 0
        line_height = 20  # Estimated line height

        for line_text in text.split('\n'):
            if line_text.strip():
                lines.append({
                    'text': line_text.strip(),
                    'y_start': y_position,
                    'y_end': y_position + line_height,
                    'x_start': 0,
                    'x_end': len(line_text) * 10,  # Estimated width
                    'length': len(line_text)
                })
            y_position += line_height

        return lines

    async def _detect_headings(self, text_lines: List[Dict[str, Any]]) -> List[LayoutRegion]:
        """Detect heading regions."""
        if not self.config.enable_heading_detection:
            return []

        heading_regions = []
        heading_patterns = [
            r'^#{1,6}\s+(.+)',  # Markdown headers
            r'^[A-Z][^.]*[.!?]$',  # Title case sentences
            r'^\d+\.\s+(.+)',  # Numbered headings
        ]

        for line_data in text_lines:
            text = line_data['text']
            line_length = line_data['length']

            # Skip very short lines
            if line_length < 3:
                continue

            # Check for heading patterns
            heading_level = None
            for i, pattern in enumerate(heading_patterns):
                match = re.match(pattern, text)
                if match:
                    if i == 0:  # Markdown headers
                        heading_level = len(match.group(1).split()[0])
                    else:
                        heading_level = 1
                    break

            if heading_level is not None:
                element_type = LayoutElementType.HEADING if heading_level <= 3 else LayoutElementType.TITLE

                region = LayoutRegion(
                    element_type=element_type,
                    bbox=(
                        line_data['x_start'],
                        line_data['y_start'],
                        line_data['x_end'],
                        line_data['y_end']
                    ),
                    text=text,
                    confidence=0.8,
                    properties={
                        'heading_level': heading_level,
                        'line_count': 1
                    }
                )
                heading_regions.append(region)

        return heading_regions

    async def _detect_tables(self, text_lines: List[Dict[str, Any]], image: Optional[Image.Image]) -> List[LayoutRegion]:
        """Detect table regions."""
        if not self.config.enable_table_detection:
            return []

        table_regions = []
        current_table_lines = []
        table_start = None

        for i, line_data in enumerate(text_lines):
            text = line_data['text']

            # Check for table indicators
            is_table_line = self._is_table_line(text)

            if is_table_line:
                if current_table_lines:  # Continuing table
                    current_table_lines.append(line_data)
                else:  # Starting new table
                    current_table_lines = [line_data]
                    table_start = i
            else:
                # End current table if exists
                if current_table_lines and len(current_table_lines) >= 2:
                    table_region = await self._create_table_region(
                        current_table_lines, table_start
                    )
                    if table_region:
                        table_regions.append(table_region)

                current_table_lines = []
                table_start = None

        # Handle table at end of text
        if current_table_lines and len(current_table_lines) >= 2:
            table_region = await self._create_table_region(
                current_table_lines, table_start
            )
            if table_region:
                table_regions.append(table_region)

        return table_regions

    def _is_table_line(self, text: str) -> bool:
        """Check if line represents table content."""
        # Tab-separated
        if '\t' in text and text.count('\t') >= 2:
            return True

        # Pipe-separated
        if '|' in text and text.count('|') >= 3:
            return True

        # Multiple spaces (column alignment)
        if text.count('  ') >= 3 and len(text.split()) >= 3:
            return True

        # Number patterns (common in tables)
        words = text.split()
        if len(words) >= 3:
            numeric_words = sum(1 for word in words if re.match(r'^[\d,.]+$', word))
            if numeric_words >= len(words) * 0.3:  # At least 30% numeric
                return True

        return False

    async def _create_table_region(self, table_lines: List[Dict[str, Any]], start_index: int) -> Optional[LayoutRegion]:
        """Create table region from table lines."""
        if not table_lines:
            return None

        # Calculate bounding box
        min_x = min(line['x_start'] for line in table_lines)
        max_x = max(line['x_end'] for line in table_lines)
        min_y = table_lines[0]['y_start']
        max_y = table_lines[-1]['y_end']

        # Combine table text
        table_text = '\n'.join(line['text'] for line in table_lines)

        # Analyze table structure
        cols = max(len(line['text'].split('\t')) if '\t' in line['text'] else
                   len(line['text'].split('|')) if '|' in line['text'] else
                   len(line['text'].split()) for line in table_lines)

        region = LayoutRegion(
            element_type=LayoutElementType.TABLE,
            bbox=(min_x, min_y, max_x, max_y),
            text=table_text,
            confidence=0.7,
            properties={
                'row_count': len(table_lines),
                'column_count': cols,
                'start_line': start_index
            }
        )

        return region

    async def _detect_lists(self, text_lines: List[Dict[str, Any]]) -> List[LayoutRegion]:
        """Detect list regions."""
        if not self.config.enable_list_detection:
            return []

        list_regions = []
        current_list_lines = []
        list_start = None
        list_pattern = re.compile(r'^\s*[-*+•]\s+(.+)')  # Bullet points
        numbered_pattern = re.compile(r'^\s*\d+[\.\)]\s+(.+)')  # Numbered lists

        for i, line_data in enumerate(text_lines):
            text = line_data['text']

            # Check for list patterns
            is_list_item = bool(list_pattern.match(text) or numbered_pattern.match(text))

            if is_list_item:
                if current_list_lines:  # Continuing list
                    current_list_lines.append(line_data)
                else:  # Starting new list
                    current_list_lines = [line_data]
                    list_start = i
            else:
                # End current list if exists
                if current_list_lines and len(current_list_lines) >= 2:
                    list_region = await self._create_list_region(
                        current_list_lines, list_start
                    )
                    if list_region:
                        list_regions.append(list_region)

                current_list_lines = []
                list_start = None

        # Handle list at end of text
        if current_list_lines and len(current_list_lines) >= 2:
            list_region = await self._create_list_region(
                current_list_lines, list_start
            )
            if list_region:
                list_regions.append(list_region)

        return list_regions

    async def _create_list_region(self, list_lines: List[Dict[str, Any]], start_index: int) -> Optional[LayoutRegion]:
        """Create list region from list lines."""
        if not list_lines:
            return None

        # Calculate bounding box
        min_x = min(line['x_start'] for line in list_lines)
        max_x = max(line['x_end'] for line in list_lines)
        min_y = list_lines[0]['y_start']
        max_y = list_lines[-1]['y_end']

        # Combine list text
        list_text = '\n'.join(line['text'] for line in list_lines)

        region = LayoutRegion(
            element_type=LayoutElementType.LIST,
            bbox=(min_x, min_y, max_x, max_y),
            text=list_text,
            confidence=0.7,
            properties={
                'item_count': len(list_lines),
                'start_line': start_index
            }
        )

        return region

    async def _detect_columns(self, text_lines: List[Dict[str, Any]]) -> List[LayoutRegion]:
        """Detect column regions."""
        if not self.config.enable_column_detection:
            return []

        # Simple column detection based on text distribution
        total_width = max(line['x_end'] for line in text_lines) if text_lines else 0

        if total_width == 0:
            return []

        # Check for potential column structure
        column_count = self._estimate_column_count(text_lines, total_width)

        if column_count <= 1:
            return []

        # Create column regions
        column_width = total_width // column_count
        column_regions = []

        for col in range(column_count):
            col_start_x = col * column_width
            col_end_x = (col + 1) * column_width

            # Find lines in this column
            col_lines = [
                line for line in text_lines
                if line['x_start'] >= col_start_x and line['x_end'] <= col_end_x
            ]

            if col_lines:
                col_min_y = min(line['y_start'] for line in col_lines)
                col_max_y = max(line['y_end'] for line in col_lines)
                col_text = '\n'.join(line['text'] for line in col_lines)

                region = LayoutRegion(
                    element_type=LayoutElementType.COLUMN,
                    bbox=(col_start_x, col_min_y, col_end_x, col_max_y),
                    text=col_text,
                    confidence=0.6,
                    properties={
                        'column_number': col + 1,
                        'line_count': len(col_lines),
                        'width': column_width
                    }
                )
                column_regions.append(region)

        return column_regions

    def _estimate_column_count(self, text_lines: List[Dict[str, Any]], total_width: int) -> int:
        """Estimate number of columns in the document."""
        if not text_lines or total_width == 0:
            return 1

        # Analyze text distribution
        x_positions = [line['x_start'] for line in text_lines]
        if len(x_positions) < 3:
            return 1

        # Look for gaps in x positions that might indicate column boundaries
        x_positions.sort()
        gaps = []

        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Significant gap
                gaps.append(gap)

        # Estimate columns based on gaps
        if len(gaps) >= 2:
            return min(len(gaps) + 1, 4)  # Max 4 columns
        elif len(gaps) == 1 and gaps[0] > 100:
            return 2
        else:
            return 1

    async def _identify_paragraphs(
        self,
        text_lines: List[Dict[str, Any]],
        heading_regions: List[LayoutRegion],
        table_regions: List[LayoutRegion],
        list_regions: List[LayoutRegion],
        column_regions: List[LayoutRegion]
    ) -> List[LayoutRegion]:
        """Identify paragraph regions from remaining text."""
        # Mark lines that are already part of other regions
        occupied_lines = set()

        for region in heading_regions + table_regions + list_regions + column_regions:
            if region.properties.get('start_line') is not None:
                start_line = region.properties['start_line']
                if 'line_count' in region.properties:
                    end_line = start_line + region.properties['line_count']
                    occupied_lines.update(range(start_line, end_line))
                else:
                    occupied_lines.add(start_line)

        paragraph_regions = []
        current_paragraph = []
        paragraph_start = None

        for i, line_data in enumerate(text_lines):
            if i in occupied_lines or not line_data['text'].strip():
                # End current paragraph if exists
                if current_paragraph and len(current_paragraph) >= 2:
                    paragraph_region = await self._create_paragraph_region(
                        current_paragraph, paragraph_start
                    )
                    if paragraph_region:
                        paragraph_regions.append(paragraph_region)

                current_paragraph = []
                paragraph_start = None
            else:
                if not current_paragraph:
                    current_paragraph = [line_data]
                    paragraph_start = i
                else:
                    current_paragraph.append(line_data)

        # Handle paragraph at end of text
        if current_paragraph and len(current_paragraph) >= 2:
            paragraph_region = await self._create_paragraph_region(
                current_paragraph, paragraph_start
            )
            if paragraph_region:
                paragraph_regions.append(paragraph_region)

        return paragraph_regions

    async def _create_paragraph_region(self, paragraph_lines: List[Dict[str, Any]], start_index: int) -> Optional[LayoutRegion]:
        """Create paragraph region from paragraph lines."""
        if not paragraph_lines:
            return None

        # Calculate bounding box
        min_x = min(line['x_start'] for line in paragraph_lines)
        max_x = max(line['x_end'] for line in paragraph_lines)
        min_y = paragraph_lines[0]['y_start']
        max_y = paragraph_lines[-1]['y_end']

        # Combine paragraph text
        paragraph_text = '\n'.join(line['text'] for line in paragraph_lines)

        region = LayoutRegion(
            element_type=LayoutElementType.PARAGRAPH,
            bbox=(min_x, min_y, max_x, max_y),
            text=paragraph_text,
            confidence=0.8,
            properties={
                'line_count': len(paragraph_lines),
                'start_line': start_index,
                'word_count': len(paragraph_text.split())
            }
        )

        return region

    async def _merge_similar_regions(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Merge similar adjacent regions."""
        if not regions:
            return []

        merged_regions = []
        current_region = regions[0]

        for next_region in regions[1:]:
            if self._should_merge_regions(current_region, next_region):
                # Merge regions
                current_region = await self._merge_two_regions(current_region, next_region)
            else:
                merged_regions.append(current_region)
                current_region = next_region

        merged_regions.append(current_region)
        return merged_regions

    def _should_merge_regions(self, region1: LayoutRegion, region2: LayoutRegion) -> bool:
        """Check if two regions should be merged."""
        # Merge if they are the same type and vertically adjacent
        if region1.element_type == region2.element_type:
            y_distance = abs(region1.bbox[3] - region2.bbox[1])
            if y_distance < 50:  # Close vertically
                return True

        return False

    async def _merge_two_regions(self, region1: LayoutRegion, region2: LayoutRegion) -> LayoutRegion:
        """Merge two layout regions."""
        merged_bbox = (
            min(region1.bbox[0], region2.bbox[0]),
            min(region1.bbox[1], region2.bbox[1]),
            max(region1.bbox[2], region2.bbox[2]),
            max(region1.bbox[3], region2.bbox[3])
        )

        merged_text = region1.text + '\n' + region2.text
        merged_confidence = (region1.confidence + region2.confidence) / 2

        merged_properties = region1.properties.copy()
        if 'line_count' in region1.properties and 'line_count' in region2.properties:
            merged_properties['line_count'] = (
                region1.properties['line_count'] + region2.properties['line_count']
            )

        return LayoutRegion(
            element_type=region1.element_type,
            bbox=merged_bbox,
            text=merged_text,
            confidence=merged_confidence,
            properties=merged_properties,
            children=[region1, region2]
        )

    def get_layout_summary(self, regions: List[LayoutRegion]) -> Dict[str, Any]:
        """Get summary statistics of detected layout."""
        if not regions:
            return {}

        element_counts = {}
        total_confidence = 0
        total_text_length = 0

        for region in regions:
            element_type = region.element_type.value
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
            total_confidence += region.confidence
            total_text_length += len(region.text)

        return {
            'total_regions': len(regions),
            'element_types': element_counts,
            'average_confidence': total_confidence / len(regions) if regions else 0,
            'total_text_length': total_text_length,
            'detected_elements': list(element_counts.keys())
        }