"""
Table Structure Recognition Module

This module provides advanced table structure recognition capabilities,
inspired by RAGFlow's vision/table_structure_recognizer.py implementation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
import numpy as np

try:
    import cv2
    from PIL import Image, ImageDraw
    CV2_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    PIL_AVAILABLE = False

from .ocr import OCREngine, OCRResult
from .layout_recognizer import LayoutRegion, LayoutElementType

logger = logging.getLogger(__name__)


class TableStructureType(str, Enum):
    """Table structure element types."""
    TABLE = "table"
    ROW = "row"
    COLUMN = "column"
    CELL = "cell"
    HEADER = "header"
    DATA_CELL = "data_cell"
    SPANNING_CELL = "spanning_cell"
    EMPTY_CELL = "empty_cell"
    BORDER = "border"
    GRID_LINE = "grid_line"


@dataclass
class TableCell:
    """Table cell representation."""
    id: str
    row_index: int
    col_index: int
    row_span: int
    col_span: int
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    cell_type: TableStructureType
    properties: Dict[str, Any]
    merged_cells: List[str] = None


@dataclass
class TableRow:
    """Table row representation."""
    id: str
    index: int
    cells: List[TableCell]
    bbox: Tuple[int, int, int, int]
    confidence: float
    is_header: bool = False
    properties: Dict[str, Any] = None


@dataclass
class TableStructure:
    """Complete table structure representation."""
    id: str
    bbox: Tuple[int, int, int, int]
    rows: List[TableRow]
    columns: int
    confidence: float
    detection_method: str
    properties: Dict[str, Any]
    ocr_result: Optional[OCRResult] = None


@dataclass
class TableRecognitionConfig:
    """Table recognition configuration."""
    enable_ocr_enhancement: bool = True
    enable_structure_analysis: bool = True
    enable_header_detection: bool = True
    enable_cell_merging: bool = True
    min_confidence: float = 0.6
    max_row_gap: int = 50
    max_col_gap: int = 30
    cell_padding: int = 5
    border_detection_threshold: float = 0.3
    ocr_language_hint: Optional[str] = None


class TableStructureRecognizer:
    """
    Advanced table structure recognizer.

    Features:
    - Multi-format table detection (lined, borderless, complex)
    - OCR-based cell content extraction
    - Header row detection
    - Cell merging and spanning detection
    - Structure validation and correction
    - Confidence scoring
    """

    def __init__(self, config: Optional[TableRecognitionConfig] = None):
        """Initialize table structure recognizer."""
        self.config = config or TableRecognitionConfig()
        self.ocr_engine = OCREngine()

    async def recognize_table_structure(
        self,
        image: Union[bytes, np.ndarray, Image.Image],
        ocr_result: Optional[OCRResult] = None
    ) -> Optional[TableStructure]:
        """
        Recognize table structure from image.

        Args:
            image: Input image
            ocr_result: Optional pre-computed OCR result

        Returns:
            Optional[TableStructure]: Recognized table structure
        """
        try:
            # Convert to PIL Image
            pil_image = self._ensure_pil_image(image)
            if not pil_image:
                return None

            # Get OCR result if not provided
            if ocr_result is None and self.config.enable_ocr_enhancement:
                ocr_result = await self.ocr_engine.extract_text(pil_image)

            # Detect table presence
            table_bbox = await self._detect_table_boundary(pil_image, ocr_result)
            if not table_bbox:
                logger.warning("No table structure detected")
                return None

            # Analyze table structure
            if self.config.enable_structure_analysis:
                table_structure = await self._analyze_table_structure(
                    pil_image, table_bbox, ocr_result
                )
            else:
                # Fallback to simple grid-based analysis
                table_structure = await self._simple_grid_analysis(
                    pil_image, table_bbox, ocr_result
                )

            # Post-processing and validation
            if table_structure:
                table_structure = await self._post_process_table(table_structure)

            return table_structure

        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            return None

    async def _detect_table_boundary(
        self,
        image: Image.Image,
        ocr_result: Optional[OCRResult]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect table boundary in image."""
        try:
            # Method 1: OCR-based detection
            if ocr_result:
                ocr_bbox = await self._detect_table_from_ocr(ocr_result)
                if ocr_bbox:
                    return ocr_bbox

            # Method 2: Image-based detection
            if CV2_AVAILABLE:
                image_bbox = await self._detect_table_from_image(image)
                if image_bbox:
                    return image_bbox

            # Method 3: Simple heuristic
            return (0, 0, image.width, image.height)

        except Exception as e:
            logger.error(f"Table boundary detection failed: {e}")
            return None

    async def _detect_table_from_ocr(self, ocr_result: OCRResult) -> Optional[Tuple[int, int, int, int]]:
        """Detect table from OCR text layout."""
        lines = ocr_result.text.split('\n')
        table_lines = []

        # Look for table-like patterns
        for line in lines:
            if self._is_table_like_line(line):
                table_lines.append(line)

        if len(table_lines) < 2:
            return None

        # Extract bounding box from OCR regions (simplified)
        # In practice, you would use OCR region information
        return (0, 0, 800, 400)  # Placeholder

    async def _detect_table_from_image(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Detect table from image features."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find rectangular contours that might be tables
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Table-like aspect ratio
                    if 0.5 < aspect_ratio < 5.0:
                        return (x, y, x + w, y + h)

            return None

        except Exception as e:
            logger.error(f"Image-based table detection failed: {e}")
            return None

    def _is_table_like_line(self, line: str) -> bool:
        """Check if line looks like part of a table."""
        # Tab-separated
        if '\t' in line and line.count('\t') >= 2:
            return True

        # Pipe-separated
        if '|' in line and line.count('|') >= 3:
            return True

        # Multiple spaces (column alignment)
        if line.count('  ') >= 3 and len(line.split()) >= 3:
            return True

        # Number patterns (common in tables)
        words = line.split()
        if len(words) >= 3:
            numeric_words = sum(1 for word in words if re.match(r'^[\d,.]+$', word))
            if numeric_words >= len(words) * 0.3:  # At least 30% numeric
                return True

        return False

    async def _analyze_table_structure(
        self,
        image: Image.Image,
        table_bbox: Tuple[int, int, int, int],
        ocr_result: Optional[OCRResult]
    ) -> Optional[TableStructure]:
        """Analyze detailed table structure."""
        try:
            # Crop table region
            table_image = image.crop(table_bbox)

            # Detect grid lines
            horizontal_lines = await self._detect_horizontal_lines(table_image)
            vertical_lines = await self._detect_vertical_lines(table_image)

            # Build cell structure
            cells = await self._build_cells_from_grid(
                horizontal_lines, vertical_lines, table_bbox
            )

            # Organize cells into rows
            rows = await self._organize_cells_into_rows(cells)

            # Detect header row
            if self.config.enable_header_detection:
                await self._detect_header_row(rows)

            # Extract cell content using OCR
            if ocr_result:
                await self._populate_cell_content(cells, ocr_result, table_bbox)

            # Create table structure
            table_structure = TableStructure(
                id=f"table_{hash(str(table_bbox)) % 10000:04d}",
                bbox=table_bbox,
                rows=rows,
                columns=max(len(row.cells) for row in rows) if rows else 0,
                confidence=0.8,  # Placeholder confidence calculation
                detection_method="grid_analysis",
                properties={
                    'horizontal_lines': len(horizontal_lines),
                    'vertical_lines': len(vertical_lines),
                    'total_cells': len(cells)
                },
                ocr_result=ocr_result
            )

            return table_structure

        except Exception as e:
            logger.error(f"Table structure analysis failed: {e}")
            return None

    async def _detect_horizontal_lines(self, image: Image.Image) -> List[int]:
        """Detect horizontal grid lines."""
        try:
            if not CV2_AVAILABLE:
                return []

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect horizontal lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=50, minLineLength=50)

            horizontal_positions = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is relatively horizontal
                    if abs(y1 - y2) < 10:  # Small vertical tolerance
                        horizontal_positions.append((y1 + y2) // 2)

            # Remove duplicates and sort
            horizontal_positions = list(set(horizontal_positions))
            horizontal_positions.sort()

            return horizontal_positions

        except Exception as e:
            logger.error(f"Horizontal line detection failed: {e}")
            return []

    async def _detect_vertical_lines(self, image: Image.Image) -> List[int]:
        """Detect vertical grid lines."""
        try:
            if not CV2_AVAILABLE:
                return []

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect vertical lines
            lines = cv2.HoughLinesP(edges, 1, 0, threshold=50, minLineLength=50)

            vertical_positions = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is relatively vertical
                    if abs(x1 - x2) < 10:  # Small horizontal tolerance
                        vertical_positions.append((x1 + x2) // 2)

            # Remove duplicates and sort
            vertical_positions = list(set(vertical_positions))
            vertical_positions.sort()

            return vertical_positions

        except Exception as e:
            logger.error(f"Vertical line detection failed: {e}")
            return []

    async def _build_cells_from_grid(
        self,
        horizontal_lines: List[int],
        vertical_lines: List[int],
        table_bbox: Tuple[int, int, int, int]
    ) -> List[TableCell]:
        """Build table cells from grid lines."""
        cells = []

        # Add table boundaries
        x1, y1, x2, y2 = table_bbox
        all_horizontal = [y1] + horizontal_lines + [y2]
        all_vertical = [x1] + vertical_lines + [x2]

        # Sort and deduplicate
        all_horizontal = sorted(set(all_horizontal))
        all_vertical = sorted(set(all_vertical))

        # Create cells
        for row_idx in range(len(all_horizontal) - 1):
            for col_idx in range(len(all_vertical) - 1):
                cell_bbox = (
                    all_vertical[col_idx],
                    all_horizontal[row_idx],
                    all_vertical[col_idx + 1],
                    all_horizontal[row_idx + 1]
                )

                cell = TableCell(
                    id=f"cell_{row_idx}_{col_idx}",
                    row_index=row_idx,
                    col_index=col_idx,
                    row_span=1,
                    col_span=1,
                    text="",
                    bbox=cell_bbox,
                    confidence=0.7,
                    cell_type=TableStructureType.DATA_CELL,
                    properties={}
                )
                cells.append(cell)

        return cells

    async def _organize_cells_into_rows(self, cells: List[TableCell]) -> List[TableRow]:
        """Organize cells into table rows."""
        rows_dict = {}

        # Group cells by row
        for cell in cells:
            if cell.row_index not in rows_dict:
                rows_dict[cell.row_index] = []
            rows_dict[cell.row_index].append(cell)

        # Create row objects
        rows = []
        for row_index in sorted(rows_dict.keys()):
            row_cells = sorted(rows_dict[row_index], key=lambda c: c.col_index)

            # Calculate row bounding box
            if row_cells:
                min_x = min(cell.bbox[0] for cell in row_cells)
                min_y = min(cell.bbox[1] for cell in row_cells)
                max_x = max(cell.bbox[2] for cell in row_cells)
                max_y = max(cell.bbox[3] for cell in row_cells)
                row_bbox = (min_x, min_y, max_x, max_y)
            else:
                row_bbox = (0, 0, 0, 0)

            row = TableRow(
                id=f"row_{row_index}",
                index=row_index,
                cells=row_cells,
                bbox=row_bbox,
                confidence=0.7,
                is_header=False,
                properties={}
            )
            rows.append(row)

        return rows

    async def _detect_header_row(self, rows: List[TableRow]) -> bool:
        """Detect which row is the header row."""
        if not rows or len(rows) < 2:
            return False

        # Simple heuristic: first row is header if it has different text patterns
        first_row = rows[0]
        second_row = rows[1]

        # Check for typical header patterns
        first_row_text = [cell.text for cell in first_row.cells if cell.text.strip()]
        second_row_text = [cell.text for cell in second_row.cells if cell.text.strip()]

        if not first_row_text:
            return False

        # Headers often have shorter text or different formatting
        first_row_avg_length = np.mean([len(text) for text in first_row_text]) if first_row_text else 0
        second_row_avg_length = np.mean([len(text) for text in second_row_text]) if second_row_text else 0

        # If first row has significantly shorter text, it might be a header
        if first_row_avg_length > 0 and first_row_avg_length < second_row_avg_length * 0.8:
            first_row.is_header = True
            for cell in first_row.cells:
                cell.cell_type = TableStructureType.HEADER
            return True

        return False

    async def _populate_cell_content(
        self,
        cells: List[TableCell],
        ocr_result: OCRResult,
        table_bbox: Tuple[int, int, int, int]
    ):
        """Populate cell content from OCR result."""
        try:
            # This is a simplified implementation
            # In practice, you would use OCR word/line position information
            # to map text to specific cells

            ocr_lines = ocr_result.text.split('\n')
            cell_idx = 0

            for line in ocr_lines:
                if cell_idx < len(cells) and line.strip():
                    cells[cell_idx].text = line.strip()
                    cells[cell_idx].confidence = ocr_result.confidence
                    cell_idx += 1

        except Exception as e:
            logger.error(f"Cell content population failed: {e}")

    async def _simple_grid_analysis(
        self,
        image: Image.Image,
        table_bbox: Tuple[int, int, int, int],
        ocr_result: Optional[OCRResult]
    ) -> Optional[TableStructure]:
        """Simple fallback grid analysis."""
        try:
            # Create a basic 2xN grid structure
            ocr_lines = ocr_result.text.split('\n') if ocr_result else ["Sample", "Data"]
            table_lines = [line for line in ocr_lines if line.strip()]

            if not table_lines:
                return None

            rows = []
            for i, line in enumerate(table_lines):
                # Simple split by tabs or multiple spaces
                cells_data = re.split(r'\t|\s{2,}', line)

                cells = []
                for j, cell_text in enumerate(cells_data):
                    cell = TableCell(
                        id=f"cell_{i}_{j}",
                        row_index=i,
                        col_index=j,
                        row_span=1,
                        col_span=1,
                        text=cell_text.strip(),
                        bbox=(j * 200, i * 50, (j + 1) * 200, (i + 1) * 50),  # Placeholder
                        confidence=0.6,
                        cell_type=TableStructureType.HEADER if i == 0 else TableStructureType.DATA_CELL,
                        properties={}
                    )
                    cells.append(cell)

                row = TableRow(
                    id=f"row_{i}",
                    index=i,
                    cells=cells,
                    bbox=(0, i * 50, 800, (i + 1) * 50),  # Placeholder
                    confidence=0.6,
                    is_header=(i == 0),
                    properties={}
                )
                rows.append(row)

            table_structure = TableStructure(
                id="simple_table",
                bbox=table_bbox,
                rows=rows,
                columns=max(len(row.cells) for row in rows) if rows else 0,
                confidence=0.6,
                detection_method="simple_grid",
                properties={'method': 'fallback_simple'},
                ocr_result=ocr_result
            )

            return table_structure

        except Exception as e:
            logger.error(f"Simple grid analysis failed: {e}")
            return None

    async def _post_process_table(self, table: TableStructure) -> TableStructure:
        """Post-process and validate table structure."""
        try:
            # Validate table consistency
            if not self._validate_table_structure(table):
                logger.warning("Table structure validation failed, applying corrections")
                table = await self._correct_table_structure(table)

            # Detect and handle merged cells
            if self.config.enable_cell_merging:
                table = await self._detect_merged_cells(table)

            # Calculate overall confidence
            table.confidence = self._calculate_table_confidence(table)

            return table

        except Exception as e:
            logger.error(f"Table post-processing failed: {e}")
            return table

    def _validate_table_structure(self, table: TableStructure) -> bool:
        """Validate table structure consistency."""
        if not table.rows:
            return False

        # Check if all rows have consistent structure
        column_counts = [len(row.cells) for row in table.rows]
        if len(set(column_counts)) > 1:
            return False

        return True

    async def _correct_table_structure(self, table: TableStructure) -> TableStructure:
        """Correct table structure inconsistencies."""
        try:
            if not table.rows:
                return table

            # Find maximum number of columns
            max_cols = max(len(row.cells) for row in table.rows)

            # Pad rows with missing cells
            for row in table.rows:
                while len(row.cells) < max_cols:
                    empty_cell = TableCell(
                        id=f"empty_cell_{row.index}_{len(row.cells)}",
                        row_index=row.index,
                        col_index=len(row.cells),
                        row_span=1,
                        col_span=1,
                        text="",
                        bbox=(0, 0, 0, 0),  # Placeholder
                        confidence=0.0,
                        cell_type=TableStructureType.EMPTY_CELL,
                        properties={}
                    )
                    row.cells.append(empty_cell)

            table.columns = max_cols
            return table

        except Exception as e:
            logger.error(f"Table structure correction failed: {e}")
            return table

    async def _detect_merged_cells(self, table: TableStructure) -> TableStructure:
        """Detect merged cells (placeholder implementation)."""
        # This is a simplified implementation
        # In practice, you would analyze cell positions and sizes
        # to detect actual merged cells
        return table

    def _calculate_table_confidence(self, table: TableStructure) -> float:
        """Calculate overall table confidence."""
        if not table.rows:
            return 0.0

        confidences = []
        for row in table.rows:
            for cell in row.cells:
                if cell.confidence > 0:
                    confidences.append(cell.confidence)

        return np.mean(confidences) if confidences else 0.0

    def _ensure_pil_image(self, image: Union[bytes, np.ndarray, Image.Image]) -> Optional[Image.Image]:
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

    def get_table_summary(self, table: TableStructure) -> Dict[str, Any]:
        """Get summary of recognized table."""
        if not table:
            return {}

        return {
            'table_id': table.id,
            'rows': len(table.rows),
            'columns': table.columns,
            'total_cells': sum(len(row.cells) for row in table.rows),
            'header_rows': sum(1 for row in table.rows if row.is_header),
            'confidence': table.confidence,
            'detection_method': table.detection_method,
            'has_content': any(cell.text.strip() for row in table.rows for cell in row.cells),
            'properties': table.properties
        }