"""
Structure Information Preservation Service

This module provides advanced document structure preservation capabilities,
maintaining the hierarchical organization, formatting, and semantic relationships
within documents during processing, inspired by RAGFlow's structure preservation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

from ..interfaces.source_interface import ParseRequest, ParseResponse
from ..vision.layout_recognizer import LayoutRegion, LayoutElementType
from ..vision.ocr import OCRResult
from .processing_pipeline import PipelineResult
from .multimodal_fusion import FusionResult, ModalityContent

logger = logging.getLogger(__name__)


class StructureType(str, Enum):
    """Document structure types."""
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TITLE = "title"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    QUOTE = "quote"
    CODE_BLOCK = "code_block"
    FORMULA = "formula"


class StructureFormat(str, Enum):
    """Structure output formats."""
    HIERARCHICAL_JSON = "hierarchical_json"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    YAML = "yaml"
    CUSTOM = "custom"


@dataclass
class StructureNode:
    """Document structure node."""
    id: str
    type: StructureType
    content: str
    level: int  # Hierarchical level (1 = highest)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureConfig:
    """Structure preservation configuration."""
    preserve_hierarchy: bool = True
    preserve_formatting: bool = True
    preserve_position: bool = True
    preserve_semantic_relationships: bool = True
    min_content_length: int = 5
    confidence_threshold: float = 0.5
    enable_structure_validation: bool = True
    output_format: StructureFormat = StructureFormat.HIERARCHICAL_JSON
    custom_format_template: Optional[str] = None
    preserve_whitespace: bool = False
    normalize_whitespace: bool = True


@dataclass
class StructureResult:
    """Result of structure preservation."""
    id: str
    structure_tree: Dict[str, Any]
    node_count: int
    max_depth: int
    format: StructureFormat
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructurePreservation:
    """
    Advanced document structure preservation service.

    Features:
    - Hierarchical structure extraction
    - Format and position preservation
    - Semantic relationship maintenance
    - Multiple output formats
    - Structure validation
    - Cross-referencing support
    """

    def __init__(self, config: Optional[StructureConfig] = None):
        """Initialize structure preservation service."""
        self.config = config or StructureConfig()
        self.structure_cache = {}
        self.validation_rules = self._init_validation_rules()

    async def preserve_structure(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]] = None,
        ocr_result: Optional[OCRResult] = None,
        fusion_result: Optional[FusionResult] = None
    ) -> StructureResult:
        """
        Preserve document structure from processing results.

        Args:
            pipeline_result: Pipeline processing result
            layout_regions: Detected layout regions
            ocr_result: OCR extraction result
            fusion_result: Multimodal fusion result

        Returns:
            StructureResult: Preserved document structure
        """
        structure_id = str(uuid.uuid())
        start_time = datetime.now()

        try:
            # Extract structure nodes from various sources
            structure_nodes = await self._extract_structure_nodes(
                pipeline_result, layout_regions, ocr_result, fusion_result
            )

            if not structure_nodes:
                return StructureResult(
                    id=structure_id,
                    structure_tree={},
                    node_count=0,
                    max_depth=0,
                    format=self.config.output_format,
                    confidence=0.0,
                    metadata={'error': 'No structure nodes found'}
                )

            # Build hierarchical structure
            structure_tree = await self._build_structure_hierarchy(structure_nodes)

            # Validate structure
            if self.config.enable_structure_validation:
                await self._validate_structure(structure_tree, structure_nodes)

            # Format output
            formatted_structure = await self._format_structure_output(
                structure_tree, structure_nodes
            )

            # Calculate metrics
            node_count = len(structure_nodes)
            max_depth = self._calculate_max_depth(structure_tree)
            confidence = self._calculate_structure_confidence(structure_nodes)

            processing_time = (datetime.now() - start_time).total_seconds()

            return StructureResult(
                id=structure_id,
                structure_tree=formatted_structure,
                node_count=node_count,
                max_depth=max_depth,
                format=self.config.output_format,
                confidence=confidence,
                metadata={
                    'processing_time_seconds': processing_time,
                    'config': self.config.__dict__,
                    'validation_passed': self.config.enable_structure_validation
                }
            )

        except Exception as e:
            logger.error(f"Structure preservation failed: {e}")
            return StructureResult(
                id=structure_id,
                structure_tree={},
                node_count=0,
                max_depth=0,
                format=self.config.output_format,
                confidence=0.0,
                metadata={'error': str(e)}
            )

    async def _extract_structure_nodes(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]],
        ocr_result: Optional[OCRResult],
        fusion_result: Optional[FusionResult]
    ) -> List[StructureNode]:
        """Extract structure nodes from processing results."""
        nodes = []

        # Extract from layout regions
        if layout_regions:
            layout_nodes = await self._extract_from_layout_regions(layout_regions)
            nodes.extend(layout_nodes)

        # Extract from OCR result
        if ocr_result:
            ocr_nodes = await self._extract_from_ocr_result(ocr_result)
            nodes.extend(ocr_nodes)

        # Extract from pipeline response
        if pipeline_result.response and pipeline_result.response.content:
            pipeline_nodes = await self._extract_from_pipeline_response(
                pipeline_result.response
            )
            nodes.extend(pipeline_nodes)

        # Extract from fusion result
        if fusion_result:
            fusion_nodes = await self._extract_from_fusion_result(fusion_result)
            nodes.extend(fusion_nodes)

        # Filter and clean nodes
        nodes = await self._filter_and_clean_nodes(nodes)

        return nodes

    async def _extract_from_layout_regions(
        self,
        layout_regions: List[LayoutRegion]
    ) -> List[StructureNode]:
        """Extract structure nodes from layout regions."""
        nodes = []

        for region in layout_regions:
            structure_type = self._map_layout_element_to_structure(region.element_type)
            if structure_type:
                level = self._determine_structure_level(region, structure_type)

                node = StructureNode(
                    id=str(uuid.uuid()),
                    type=structure_type,
                    content=region.text,
                    level=level,
                    bbox=region.bbox,
                    confidence=region.confidence,
                    attributes={
                        'source': 'layout',
                        'layout_type': region.element_type.value,
                        'region_properties': region.properties
                    },
                    metadata=region.properties
                )
                nodes.append(node)

        return nodes

    async def _extract_from_ocr_result(self, ocr_result: OCRResult) -> List[StructureNode]:
        """Extract structure nodes from OCR result."""
        nodes = []

        if ocr_result.text.strip():
            # Parse OCR text into structure
            text_lines = ocr_result.text.split('\n')
            current_paragraph_lines = []
            paragraph_start = 0

            for i, line in enumerate(text_lines):
                line = line.strip()
                if not line:
                    # End of paragraph
                    if current_paragraph_lines:
                        paragraph_text = '\n'.join(current_paragraph_lines)
                        node = StructureNode(
                            id=str(uuid.uuid()),
                            type=StructureType.PARAGRAPH,
                            content=paragraph_text,
                            level=3,  # Paragraph level
                            confidence=ocr_result.confidence,
                            attributes={
                                'source': 'ocr',
                                'start_line': paragraph_start,
                                'end_line': i - 1
                            }
                        )
                        nodes.append(node)
                        current_paragraph_lines = []
                else:
                    # Check if line is a heading
                    structure_type, level = self._classify_text_line(line)

                    if structure_type == StructureType.HEADING:
                        # Save any existing paragraph
                        if current_paragraph_lines:
                            paragraph_text = '\n'.join(current_paragraph_lines)
                            node = StructureNode(
                                id=str(uuid.uuid()),
                                type=StructureType.PARAGRAPH,
                                content=paragraph_text,
                                level=3,
                                confidence=ocr_result.confidence,
                                attributes={
                                    'source': 'ocr',
                                    'start_line': paragraph_start,
                                    'end_line': i - 1
                                }
                            )
                            nodes.append(node)
                            current_paragraph_lines = []

                        # Add heading
                        node = StructureNode(
                            id=str(uuid.uuid()),
                            type=structure_type,
                            content=line,
                            level=level,
                            confidence=ocr_result.confidence,
                            attributes={
                                'source': 'ocr',
                                'line_number': i
                            }
                        )
                        nodes.append(node)
                    else:
                        # Part of paragraph
                        if not current_paragraph_lines:
                            paragraph_start = i
                        current_paragraph_lines.append(line)

            # Handle final paragraph
            if current_paragraph_lines:
                paragraph_text = '\n'.join(current_paragraph_lines)
                node = StructureNode(
                    id=str(uuid.uuid()),
                    type=StructureType.PARAGRAPH,
                    content=paragraph_text,
                    level=3,
                    confidence=ocr_result.confidence,
                    attributes={
                        'source': 'ocr',
                        'start_line': paragraph_start,
                        'end_line': len(text_lines) - 1
                    }
                )
                nodes.append(node)

        return nodes

    async def _extract_from_pipeline_response(
        self,
        parse_response: ParseResponse
    ) -> List[StructureNode]:
        """Extract structure nodes from pipeline response."""
        nodes = []

        for content in parse_response.content:
            structure_type = self._map_content_type_to_structure(content.content_type)
            if structure_type:
                level = self._determine_content_level(content)

                node = StructureNode(
                    id=str(uuid.uuid()),
                    type=structure_type,
                    content=content.data if isinstance(content.data, str) else str(content.data),
                    level=level,
                    confidence=0.9,  # High confidence for pipeline content
                    attributes={
                        'source': 'pipeline',
                        'content_type': content.content_type,
                        'content_metadata': content.metadata
                    }
                )
                nodes.append(node)

        return nodes

    async def _extract_from_fusion_result(
        self,
        fusion_result: FusionResult
    ) -> List[StructureNode]:
        """Extract structure nodes from fusion result."""
        nodes = []

        # Create a document-level node
        document_node = StructureNode(
            id=str(uuid.uuid()),
            type=StructureType.DOCUMENT,
            content=fusion_result.semantic_summary or "Document",
            level=1,
            confidence=fusion_result.confidence,
            attributes={
                'source': 'fusion',
                'modality_weights': fusion_result.modality_weights,
                'fusion_metadata': fusion_result.metadata
            }
        )
        nodes.append(document_node)

        return nodes

    def _map_layout_element_to_structure(
        self,
        element_type: LayoutElementType
    ) -> Optional[StructureType]:
        """Map layout element types to structure types."""
        mapping = {
            LayoutElementType.TITLE: StructureType.TITLE,
            LayoutElementType.HEADING: StructureType.HEADING,
            LayoutElementType.PARAGRAPH: StructureType.PARAGRAPH,
            LayoutElementType.LIST: StructureType.LIST,
            LayoutElementType.TABLE: StructureType.TABLE,
            LayoutElementType.IMAGE: StructureType.FIGURE,
            LayoutElementType.FIGURE: StructureType.FIGURE,
            LayoutElementType.FOOTER: StructureType.FOOTER,
            LayoutElementType.HEADER: StructureType.HEADER,
            LayoutElementType.SIDEBAR: StructureType.SIDEBAR,
            LayoutElementType.BLOCKQUOTE: StructureType.QUOTE,
            LayoutElementType.CODE: StructureType.CODE_BLOCK,
            LayoutElementType.FORM: StructureType.FORMULA
        }
        return mapping.get(element_type)

    def _map_content_type_to_structure(
        self,
        content_type: str
    ) -> Optional[StructureType]:
        """Map content types to structure types."""
        mapping = {
            'text': StructureType.PARAGRAPH,
            'heading': StructureType.HEADING,
            'title': StructureType.TITLE,
            'list': StructureType.LIST,
            'table': StructureType.TABLE,
            'image': StructureType.FIGURE,
            'code': StructureType.CODE_BLOCK,
            'formula': StructureType.FORMULA
        }
        return mapping.get(content_type)

    def _determine_structure_level(
        self,
        region: LayoutRegion,
        structure_type: StructureType
    ) -> int:
        """Determine hierarchical level for structure node."""
        # Base levels by structure type
        base_levels = {
            StructureType.DOCUMENT: 1,
            StructureType.TITLE: 1,
            StructureType.HEADING: 2,
            StructureType.SECTION: 2,
            StructureType.SUBSECTION: 3,
            StructureType.PARAGRAPH: 3,
            StructureType.LIST: 3,
            StructureType.TABLE: 3,
            StructureType.FIGURE: 3,
            StructureType.LIST_ITEM: 4,
            StructureType.FOOTNOTE: 4,
            StructureType.HEADER: 5,
            StructureType.FOOTER: 5
        }

        base_level = base_levels.get(structure_type, 3)

        # Adjust based on properties
        if region.properties:
            if 'heading_level' in region.properties:
                return region.properties['heading_level']
            if structure_type == StructureType.HEADING:
                # Adjust based on text characteristics
                text_length = len(region.text)
                if text_length < 20:
                    return 2  # High-level heading
                elif text_length < 50:
                    return 3  # Mid-level heading
                else:
                    return 4  # Low-level heading

        return base_level

    def _determine_content_level(self, content) -> int:
        """Determine level for pipeline content."""
        # Use metadata if available
        if hasattr(content, 'metadata') and content.metadata:
            if 'level' in content.metadata:
                return content.metadata['level']
            if 'heading_level' in content.metadata:
                return content.metadata['heading_level']

        # Default levels by content type
        if hasattr(content, 'content_type'):
            type_levels = {
                'title': 1,
                'heading': 2,
                'text': 3,
                'paragraph': 3,
                'list': 3,
                'table': 3,
                'image': 3,
                'list_item': 4,
                'footnote': 4
            }
            return type_levels.get(content.content_type, 3)

        return 3

    def _classify_text_line(self, line: str) -> Tuple[StructureType, int]:
        """Classify a text line and determine its level."""
        # Heading patterns
        if re.match(r'^#{1,6}\s+', line):
            level = len(line.split()[0])
            return StructureType.HEADING, level + 1

        if re.match(r'^\d+\.\s+', line):
            return StructureType.HEADING, 2

        # All caps or title case
        if line.isupper() and len(line) > 10:
            return StructureType.HEADING, 1

        # Short, centered text (likely heading)
        if len(line) < 50 and line.strip() == line:
            return StructureType.HEADING, 2

        # Default to paragraph
        return StructureType.PARAGRAPH, 3

    async def _filter_and_clean_nodes(self, nodes: List[StructureNode]) -> List[StructureNode]:
        """Filter and clean structure nodes."""
        filtered_nodes = []

        for node in nodes:
            # Filter by content length
            if len(node.content.strip()) < self.config.min_content_length:
                continue

            # Filter by confidence
            if node.confidence < self.config.confidence_threshold:
                continue

            # Clean content
            if self.config.normalize_whitespace:
                node.content = self._normalize_whitespace(node.content)

            filtered_nodes.append(node)

        return filtered_nodes

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        if not self.config.preserve_whitespace:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            # Trim leading and trailing whitespace
            text = text.strip()
        return text

    async def _build_structure_hierarchy(
        self,
        nodes: List[StructureNode]
    ) -> Dict[str, Any]:
        """Build hierarchical structure from nodes."""
        # Sort nodes by position (if available) or by content
        if self.config.preserve_position and any(node.bbox for node in nodes):
            nodes.sort(key=lambda n: (n.bbox[1] if n.bbox else 0, n.level))
        else:
            nodes.sort(key=lambda n: (n.level, len(n.content)))

        # Build parent-child relationships
        root_nodes = []
        node_dict = {node.id: node for node in nodes}

        for i, node in enumerate(nodes):
            # Find potential parent
            parent = None
            for j in range(i - 1, -1, -1):
                candidate = nodes[j]
                if candidate.level < node.level:
                    parent = candidate
                    break

            if parent:
                node.parent_id = parent.id
                parent.children_ids.append(node.id)
            else:
                root_nodes.append(node.id)

        # Build hierarchical tree
        tree = {
            'document': {
                'id': 'root',
                'type': StructureType.DOCUMENT.value,
                'children': []
            }
        }

        for root_id in root_nodes:
            subtree = self._build_node_subtree(root_id, node_dict)
            tree['document']['children'].append(subtree)

        return tree

    def _build_node_subtree(
        self,
        node_id: str,
        node_dict: Dict[str, StructureNode]
    ) -> Dict[str, Any]:
        """Build subtree for a specific node."""
        node = node_dict[node_id]

        subtree = {
            'id': node.id,
            'type': node.type.value,
            'level': node.level,
            'content': node.content,
            'confidence': node.confidence,
            'attributes': node.attributes,
            'metadata': node.metadata,
            'children': []
        }

        if node.bbox:
            subtree['bbox'] = node.bbox

        for child_id in node.children_ids:
            child_subtree = self._build_node_subtree(child_id, node_dict)
            subtree['children'].append(child_subtree)

        return subtree

    async def _validate_structure(
        self,
        structure_tree: Dict[str, Any],
        nodes: List[StructureNode]
    ):
        """Validate document structure."""
        # Check for circular references
        await self._validate_no_circular_references(nodes)

        # Check level consistency
        await self._validate_level_consistency(nodes)

        # Check content completeness
        await self._validate_content_completeness(structure_tree)

        # Apply custom validation rules
        await self._apply_validation_rules(structure_tree, nodes)

    async def _validate_no_circular_references(self, nodes: List[StructureNode]):
        """Validate no circular references in structure."""
        visited = set()
        recursion_stack = set()

        def dfs(node_id: str) -> bool:
            if node_id in recursion_stack:
                raise ValueError(f"Circular reference detected at node {node_id}")
            if node_id in visited:
                return True

            visited.add(node_id)
            recursion_stack.add(node_id)

            node = next((n for n in nodes if n.id == node_id), None)
            if node:
                for child_id in node.children_ids:
                    dfs(child_id)

            recursion_stack.remove(node_id)
            return True

        for node in nodes:
            if node.id not in visited:
                dfs(node.id)

    async def _validate_level_consistency(self, nodes: List[StructureNode]):
        """Validate level consistency in structure."""
        for node in nodes:
            if node.parent_id:
                parent = next((n for n in nodes if n.id == node.parent_id), None)
                if parent and node.level <= parent.level:
                    logger.warning(
                        f"Level inconsistency: child {node.id} (level {node.level}) "
                        f"has lower or equal level than parent {parent.id} (level {parent.level})"
                    )

    async def _validate_content_completeness(self, structure_tree: Dict[str, Any]):
        """Validate content completeness in structure."""
        # Check if document has meaningful content
        def count_content_nodes(node):
            count = 1 if node.get('content', '').strip() else 0
            for child in node.get('children', []):
                count += count_content_nodes(child)
            return count

        document = structure_tree.get('document', {})
        content_nodes = count_content_nodes(document)

        if content_nodes < 2:
            logger.warning("Document appears to have very limited content structure")

    async def _apply_validation_rules(
        self,
        structure_tree: Dict[str, Any],
        nodes: List[StructureNode]
    ):
        """Apply custom validation rules."""
        for rule_name, rule_func in self.validation_rules.items():
            try:
                await rule_func(structure_tree, nodes)
            except Exception as e:
                logger.warning(f"Validation rule {rule_name} failed: {e}")

    def _init_validation_rules(self) -> Dict[str, callable]:
        """Initialize validation rules."""
        return {
            'max_depth_check': self._rule_max_depth_check,
            'heading_hierarchy_check': self._rule_heading_hierarchy_check,
            'content_density_check': self._rule_content_density_check
        }

    async def _rule_max_depth_check(
        self,
        structure_tree: Dict[str, Any],
        nodes: List[StructureNode]
    ):
        """Rule: Check maximum structure depth."""
        max_depth = max((node.level for node in nodes), default=0)
        if max_depth > 8:
            logger.warning(f"Document structure very deep: {max_depth} levels")

    async def _rule_heading_hierarchy_check(
        self,
        structure_tree: Dict[str, Any],
        nodes: List[StructureNode]
    ):
        """Rule: Check heading hierarchy consistency."""
        headings = [n for n in nodes if n.type == StructureType.HEADING]
        for i, heading in enumerate(headings):
            if i > 0:
                prev_heading = headings[i-1]
                if heading.level < prev_heading.level and abs(heading.level - prev_heading.level) > 1:
                    logger.warning(
                        f"Heading level jump detected: {prev_heading.level} -> {heading.level}"
                    )

    async def _rule_content_density_check(
        self,
        structure_tree: Dict[str, Any],
        nodes: List[StructureNode]
    ):
        """Rule: Check content density."""
        total_chars = sum(len(node.content) for node in nodes)
        node_count = len(nodes)

        if node_count > 0:
            avg_chars_per_node = total_chars / node_count
            if avg_chars_per_node < 10:
                logger.warning("Low content density detected")

    def _calculate_max_depth(self, structure_tree: Dict[str, Any]) -> int:
        """Calculate maximum depth of structure tree."""
        def get_depth(node, current_depth=1):
            max_child_depth = current_depth
            for child in node.get('children', []):
                child_depth = get_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            return max_child_depth

        document = structure_tree.get('document', {})
        return get_depth(document)

    def _calculate_structure_confidence(self, nodes: List[StructureNode]) -> float:
        """Calculate overall structure confidence."""
        if not nodes:
            return 0.0

        total_confidence = sum(node.confidence for node in nodes)
        return total_confidence / len(nodes)

    async def _format_structure_output(
        self,
        structure_tree: Dict[str, Any],
        nodes: List[StructureNode]
    ) -> Dict[str, Any]:
        """Format structure output based on configuration."""
        if self.config.output_format == StructureFormat.HIERARCHICAL_JSON:
            return structure_tree
        elif self.config.output_format == StructureFormat.MARKDOWN:
            return await self._format_as_markdown(structure_tree)
        elif self.config.output_format == StructureFormat.HTML:
            return await self._format_as_html(structure_tree)
        elif self.config.output_format == StructureFormat.XML:
            return await self._format_as_xml(structure_tree)
        elif self.config.output_format == StructureFormat.YAML:
            return await self._format_as_yaml(structure_tree)
        elif self.config.output_format == StructureFormat.CUSTOM:
            return await self._format_as_custom(structure_tree)
        else:
            return structure_tree

    async def _format_as_markdown(
        self,
        structure_tree: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format structure as Markdown."""
        def node_to_markdown(node, level=0):
            lines = []
            indent = "  " * level

            if node['type'] in ['heading', 'title']:
                header_level = node.get('level', 1)
                lines.append(f"{'#' * header_level} {node['content']}")
            elif node['type'] == 'paragraph':
                lines.append(f"{node['content']}")
            elif node['type'] == 'list':
                for child in node.get('children', []):
                    lines.append(f"- {child.get('content', '')}")
            elif node['type'] == 'code_block':
                lines.append(f"```\n{node['content']}\n```")
            else:
                lines.append(f"{indent}{node['content']}")

            for child in node.get('children', []):
                child_markdown = node_to_markdown(child, level + 1)
                lines.extend(child_markdown)

            return lines

        document = structure_tree.get('document', {})
        markdown_lines = []
        for child in document.get('children', []):
            markdown_lines.extend(node_to_markdown(child))

        return {
            'format': 'markdown',
            'content': '\n'.join(markdown_lines)
        }

    async def _format_as_html(
        self,
        structure_tree: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format structure as HTML."""
        def node_to_html(node):
            if node['type'] in ['heading', 'title']:
                tag = f"h{node.get('level', 1)}"
                content = node['content']
                children_html = ''.join(node_to_html(child) for child in node.get('children', []))
                return f"<{tag}>{content}</{tag}>\n{children_html}"
            elif node['type'] == 'paragraph':
                content = node['content']
                children_html = ''.join(node_to_html(child) for child in node.get('children', []))
                return f"<p>{content}</p>\n{children_html}"
            elif node['type'] == 'list':
                items = ''.join(
                    f"<li>{child.get('content', '')}</li>"
                    for child in node.get('children', [])
                )
                return f"<ul>{items}</ul>"
            elif node['type'] == 'code_block':
                content = node['content']
                return f"<pre><code>{content}</code></pre>"
            else:
                content = node.get('content', '')
                children_html = ''.join(node_to_html(child) for child in node.get('children', []))
                return f"<div>{content}</div>\n{children_html}"

        document = structure_tree.get('document', {})
        html_content = ''.join(node_to_html(child) for child in document.get('children', []))

        return {
            'format': 'html',
            'content': f"<html><body>{html_content}</body></html>"
        }

    async def _format_as_xml(
        self,
        structure_tree: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format structure as XML."""
        root = ET.Element('document')

        def node_to_xml(node, parent_elem):
            elem = ET.SubElement(parent_elem, node['type'])
            elem.set('level', str(node.get('level', 1)))
            elem.set('confidence', str(node.get('confidence', 0)))

            if node.get('content'):
                elem.text = node['content']

            for child in node.get('children', []):
                node_to_xml(child, elem)

        document = structure_tree.get('document', {})
        for child in document.get('children', []):
            node_to_xml(child, root)

        return {
            'format': 'xml',
            'content': ET.tostring(root, encoding='unicode')
        }

    async def _format_as_yaml(
        self,
        structure_tree: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format structure as YAML."""
        # Simplified YAML formatting
        yaml_content = self._dict_to_yaml(structure_tree)
        return {
            'format': 'yaml',
            'content': yaml_content
        }

    def _dict_to_yaml(self, d, indent=0):
        """Convert dictionary to YAML string."""
        lines = []
        spacing = "  " * indent

        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{spacing}{key}:")
                lines.append(self._dict_to_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{spacing}{key}:")
                for item in value:
                    lines.append(f"{spacing}  - {item}")
            else:
                lines.append(f"{spacing}{key}: {value}")

        return '\n'.join(lines)

    async def _format_as_custom(
        self,
        structure_tree: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format structure using custom template."""
        if self.config.custom_format_template:
            # Simple template substitution (placeholder)
            template = self.config.custom_format_template
            content = template.replace('{structure}', json.dumps(structure_tree, indent=2))
            return {
                'format': 'custom',
                'content': content
            }
        else:
            return structure_tree