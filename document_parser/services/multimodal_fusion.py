"""
Multimodal Content Fusion Service

This module provides advanced multimodal content fusion capabilities,
combining text, images, tables, and other modalities from document processing
results into coherent representations, inspired by RAGFlow's multimodal approach.
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
import numpy as np

from ..interfaces.source_interface import ParseRequest, ParseResponse
from ..vision.layout_recognizer import LayoutRegion, LayoutElementType
from ..vision.ocr import OCRResult
from .processing_pipeline import PipelineResult

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Content modality types."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    CHART = "chart"
    METADATA = "metadata"


@dataclass
class ModalityContent:
    """Individual modality content."""
    id: str
    type: ModalityType
    content: Any
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)  # IDs of related content
    semantic_embedding: Optional[np.ndarray] = None


@dataclass
class FusionConfig:
    """Multimodal fusion configuration."""
    enable_semantic_fusion: bool = True
    enable_spatial_fusion: bool = True
    enable_temporal_fusion: bool = True
    enable_cross_modal_attention: bool = True
    embedding_dimension: int = 768
    similarity_threshold: float = 0.7
    max_related_content: int = 5
    preserve_structure: bool = True
    fusion_strategy: str = "weighted_average"  # weighted_average, attention, hierarchical


@dataclass
class FusionResult:
    """Result of multimodal fusion."""
    id: str
    fused_content: Dict[str, Any]
    modality_weights: Dict[str, float]
    semantic_summary: str
    structure_preserved: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalFusion:
    """
    Advanced multimodal content fusion processor.

    Features:
    - Cross-modal content understanding
    - Semantic and spatial fusion
    - Structure preservation
    - Relationship extraction
    - Attention-based fusion
    - Hierarchical content organization
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize multimodal fusion service."""
        self.config = config or FusionConfig()
        self.embedding_cache = {}
        self.fusion_cache = {}

    async def fuse_content(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]] = None,
        ocr_result: Optional[OCRResult] = None
    ) -> FusionResult:
        """
        Fuse multimodal content from pipeline results.

        Args:
            pipeline_result: Pipeline processing result
            layout_regions: Detected layout regions
            ocr_result: OCR extraction result

        Returns:
            FusionResult: Fused multimodal content
        """
        fusion_id = str(uuid.uuid())
        start_time = datetime.now()

        try:
            # Extract modalities from pipeline result
            modalities = await self._extract_modalities(
                pipeline_result, layout_regions, ocr_result
            )

            if not modalities:
                return FusionResult(
                    id=fusion_id,
                    fused_content={},
                    modality_weights={},
                    semantic_summary="",
                    structure_preserved=False,
                    confidence=0.0,
                    metadata={'error': 'No modalities found for fusion'}
                )

            # Build relationships between modalities
            await self._build_relationships(modalities, layout_regions)

            # Generate semantic embeddings
            if self.config.enable_semantic_fusion:
                await self._generate_embeddings(modalities)

            # Perform fusion
            fused_content = await self._perform_fusion(modalities)

            # Calculate modality weights
            modality_weights = self._calculate_modality_weights(modalities)

            # Generate semantic summary
            semantic_summary = await self._generate_semantic_summary(
                modalities, fused_content
            )

            # Calculate overall confidence
            confidence = self._calculate_fusion_confidence(modalities, fused_content)

            processing_time = (datetime.now() - start_time).total_seconds()

            return FusionResult(
                id=fusion_id,
                fused_content=fused_content,
                modality_weights=modality_weights,
                semantic_summary=semantic_summary,
                structure_preserved=self.config.preserve_structure,
                confidence=confidence,
                metadata={
                    'modality_count': len(modalities),
                    'processing_time_seconds': processing_time,
                    'fusion_strategy': self.config.fusion_strategy,
                    'config': self.config.__dict__
                }
            )

        except Exception as e:
            logger.error(f"Multimodal fusion failed: {e}")
            return FusionResult(
                id=fusion_id,
                fused_content={},
                modality_weights={},
                semantic_summary="",
                structure_preserved=False,
                confidence=0.0,
                metadata={'error': str(e)}
            )

    async def _extract_modalities(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]],
        ocr_result: Optional[OCRResult]
    ) -> List[ModalityContent]:
        """Extract modalities from pipeline results."""
        modalities = []

        # Extract text content
        text_modalities = await self._extract_text_modalities(
            pipeline_result, layout_regions, ocr_result
        )
        modalities.extend(text_modalities)

        # Extract image content
        image_modalities = await self._extract_image_modalities(
            pipeline_result, layout_regions
        )
        modalities.extend(image_modalities)

        # Extract table content
        table_modalities = await self._extract_table_modalities(
            pipeline_result, layout_regions
        )
        modalities.extend(table_modalities)

        # Extract code content
        code_modalities = await self._extract_code_modalities(
            pipeline_result, layout_regions
        )
        modalities.extend(code_modalities)

        # Extract formula content
        formula_modalities = await self._extract_formula_modalities(
            pipeline_result, layout_regions
        )
        modalities.extend(formula_modalities)

        return modalities

    async def _extract_text_modalities(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]],
        ocr_result: Optional[OCRResult]
    ) -> List[ModalityContent]:
        """Extract text modalities."""
        text_modalities = []

        # From OCR result
        if ocr_result and ocr_result.text.strip():
            text_modalities.append(ModalityContent(
                id=str(uuid.uuid()),
                type=ModalityType.TEXT,
                content=ocr_result.text,
                confidence=ocr_result.confidence,
                properties={
                    'source': 'ocr',
                    'language': ocr_result.language,
                    'word_count': ocr_result.word_count
                }
            ))

        # From layout regions
        if layout_regions:
            for region in layout_regions:
                if region.element_type in [
                    LayoutElementType.TEXT,
                    LayoutElementType.PARAGRAPH,
                    LayoutElementType.HEADING,
                    LayoutElementType.TITLE
                ]:
                    text_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.TEXT,
                        content=region.text,
                        confidence=region.confidence,
                        bbox=region.bbox,
                        properties={
                            'source': 'layout',
                            'element_type': region.element_type.value,
                            'region_properties': region.properties
                        }
                    ))

        # From pipeline response
        if pipeline_result.response and pipeline_result.response.content:
            for content in pipeline_result.response.content:
                if content.content_type == 'text':
                    text_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.TEXT,
                        content=content.data,
                        confidence=0.9,
                        properties={
                            'source': 'pipeline',
                            'content_metadata': content.metadata
                        }
                    ))

        return text_modalities

    async def _extract_image_modalities(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]]
    ) -> List[ModalityContent]:
        """Extract image modalities."""
        image_modalities = []

        # From layout regions
        if layout_regions:
            for region in layout_regions:
                if region.element_type in [
                    LayoutElementType.IMAGE,
                    LayoutElementType.FIGURE
                ]:
                    image_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.IMAGE,
                        content=region.text,  # Image description or alt text
                        confidence=region.confidence,
                        bbox=region.bbox,
                        properties={
                            'source': 'layout',
                            'element_type': region.element_type.value,
                            'region_properties': region.properties
                        }
                    ))

        # From pipeline response
        if pipeline_result.response and pipeline_result.response.content:
            for content in pipeline_result.response.content:
                if content.content_type == 'image':
                    image_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.IMAGE,
                        content=content.data,  # Image data or description
                        confidence=0.8,
                        properties={
                            'source': 'pipeline',
                            'content_metadata': content.metadata
                        }
                    ))

        return image_modalities

    async def _extract_table_modalities(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]]
    ) -> List[ModalityContent]:
        """Extract table modalities."""
        table_modalities = []

        # From layout regions
        if layout_regions:
            for region in layout_regions:
                if region.element_type == LayoutElementType.TABLE:
                    table_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.TABLE,
                        content=region.text,
                        confidence=region.confidence,
                        bbox=region.bbox,
                        properties={
                            'source': 'layout',
                            'element_type': region.element_type.value,
                            'region_properties': region.properties
                        }
                    ))

        # From pipeline response
        if pipeline_result.response and pipeline_result.response.content:
            for content in pipeline_result.response.content:
                if content.content_type == 'table':
                    table_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.TABLE,
                        content=content.data,
                        confidence=0.9,
                        properties={
                            'source': 'pipeline',
                            'content_metadata': content.metadata
                        }
                    ))

        return table_modalities

    async def _extract_code_modalities(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]]
    ) -> List[ModalityContent]:
        """Extract code modalities."""
        code_modalities = []

        # From layout regions
        if layout_regions:
            for region in layout_regions:
                if region.element_type == LayoutElementType.CODE:
                    code_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.CODE,
                        content=region.text,
                        confidence=region.confidence,
                        bbox=region.bbox,
                        properties={
                            'source': 'layout',
                            'element_type': region.element_type.value,
                            'region_properties': region.properties
                        }
                    ))

        # From pipeline response
        if pipeline_result.response and pipeline_result.response.content:
            for content in pipeline_result.response.content:
                if content.content_type == 'code':
                    code_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.CODE,
                        content=content.data,
                        confidence=0.9,
                        properties={
                            'source': 'pipeline',
                            'content_metadata': content.metadata
                        }
                    ))

        return code_modalities

    async def _extract_formula_modalities(
        self,
        pipeline_result: PipelineResult,
        layout_regions: Optional[List[LayoutRegion]]
    ) -> List[ModalityContent]:
        """Extract formula modalities."""
        formula_modalities = []

        # From pipeline response
        if pipeline_result.response and pipeline_result.response.content:
            for content in pipeline_result.response.content:
                if content.content_type == 'formula':
                    formula_modalities.append(ModalityContent(
                        id=str(uuid.uuid()),
                        type=ModalityType.FORMULA,
                        content=content.data,
                        confidence=0.8,
                        properties={
                            'source': 'pipeline',
                            'content_metadata': content.metadata
                        }
                    ))

        return formula_modalities

    async def _build_relationships(
        self,
        modalities: List[ModalityContent],
        layout_regions: Optional[List[LayoutRegion]]
    ):
        """Build relationships between modalities."""
        # Spatial relationships based on bounding boxes
        if self.config.enable_spatial_fusion:
            await self._build_spatial_relationships(modalities)

        # Semantic relationships based on content similarity
        if self.config.enable_semantic_fusion:
            await self._build_semantic_relationships(modalities)

        # Structural relationships based on layout
        if layout_regions:
            await self._build_structural_relationships(modalities, layout_regions)

    async def _build_spatial_relationships(self, modalities: List[ModalityContent]):
        """Build spatial relationships between modalities."""
        for i, modality1 in enumerate(modalities):
            if not modality1.bbox:
                continue

            for j, modality2 in enumerate(modalities):
                if i >= j or not modality2.bbox:
                    continue

                # Calculate spatial distance
                distance = self._calculate_spatial_distance(
                    modality1.bbox, modality2.bbox
                )

                # If modalities are spatially close, create relationship
                if distance < 200:  # Threshold for spatial proximity
                    modality1.relationships.append(modality2.id)
                    modality2.relationships.append(modality1.id)

    async def _build_semantic_relationships(self, modalities: List[ModalityContent]):
        """Build semantic relationships between modalities."""
        # Generate embeddings for text-based modalities
        text_modalities = [
            m for m in modalities
            if m.type in [ModalityType.TEXT, ModalityType.CODE, ModalityType.FORMULA]
        ]

        # Generate embeddings
        await self._generate_embeddings(text_modalities)

        # Calculate semantic similarity
        for i, modality1 in enumerate(text_modalities):
            if not modality1.semantic_embedding is not None:
                continue

            for j, modality2 in enumerate(text_modalities):
                if i >= j or not modality2.semantic_embedding is not None:
                    continue

                similarity = self._calculate_cosine_similarity(
                    modality1.semantic_embedding,
                    modality2.semantic_embedding
                )

                # If modalities are semantically similar, create relationship
                if similarity > self.config.similarity_threshold:
                    modality1.relationships.append(modality2.id)
                    modality2.relationships.append(modality1.id)

    async def _build_structural_relationships(
        self,
        modalities: List[ModalityContent],
        layout_regions: List[LayoutRegion]
    ):
        """Build structural relationships based on layout."""
        # Group modalities by their layout regions
        region_modalities = {}
        for modality in modalities:
            if modality.bbox:
                # Find which region this modality belongs to
                for region in layout_regions:
                    if self._is_bbox_in_region(modality.bbox, region.bbox):
                        if region.element_type.value not in region_modalities:
                            region_modalities[region.element_type.value] = []
                        region_modalities[region.element_type.value].append(modality)
                        break

        # Create relationships within the same region
        for region_type, region_mods in region_modalities.items():
            if len(region_mods) > 1:
                for i, mod1 in enumerate(region_mods):
                    for mod2 in region_mods[i+1:]:
                        mod1.relationships.append(mod2.id)
                        mod2.relationships.append(mod1.id)

    async def _generate_embeddings(self, modalities: List[ModalityContent]):
        """Generate semantic embeddings for modalities."""
        for modality in modalities:
            if isinstance(modality.content, str) and modality.content.strip():
                # Simple embedding generation (placeholder)
                # In a real implementation, this would use a proper embedding model
                embedding = self._generate_simple_embedding(modality.content)
                modality.semantic_embedding = embedding

    def _generate_simple_embedding(self, text: str) -> np.ndarray:
        """Generate simple embedding from text (placeholder implementation)."""
        # Simple hash-based embedding (placeholder)
        # In a real implementation, use models like BERT, Sentence Transformers, etc.
        text_hash = hash(text.lower().strip())

        # Generate a fixed-size embedding
        embedding = np.zeros(self.config.embedding_dimension)
        for i, char in enumerate(text.lower()[:100]):  # Limit to first 100 chars
            idx = (ord(char) + i * 31) % self.config.embedding_dimension
            embedding[idx] += 1.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _calculate_spatial_distance(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Euclidean distance between bounding box centers."""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

        return np.sqrt(
            (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2
        )

    def _calculate_cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _is_bbox_in_region(
        self,
        bbox: Tuple[int, int, int, int],
        region_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Check if bounding box is within region."""
        return (
            bbox[0] >= region_bbox[0] and
            bbox[1] >= region_bbox[1] and
            bbox[2] <= region_bbox[2] and
            bbox[3] <= region_bbox[3]
        )

    async def _perform_fusion(self, modalities: List[ModalityContent]) -> Dict[str, Any]:
        """Perform multimodal fusion."""
        if self.config.fusion_strategy == "weighted_average":
            return await self._weighted_average_fusion(modalities)
        elif self.config.fusion_strategy == "attention":
            return await self._attention_fusion(modalities)
        elif self.config.fusion_strategy == "hierarchical":
            return await self._hierarchical_fusion(modalities)
        else:
            return await self._weighted_average_fusion(modalities)

    async def _weighted_average_fusion(
        self,
        modalities: List[ModalityContent]
    ) -> Dict[str, Any]:
        """Perform weighted average fusion."""
        fused_content = {
            'modalities': {},
            'relationships': [],
            'structure': {},
            'summary': {}
        }

        # Group modalities by type
        by_type = {}
        for modality in modalities:
            if modality.type.value not in by_type:
                by_type[modality.type.value] = []
            by_type[modality.type.value].append(modality)

        # Process each modality type
        for mod_type, mods in by_type.items():
            if mod_type == ModalityType.TEXT.value:
                fused_content['modalities']['text'] = {
                    'content': ' '.join([m.content for m in mods if isinstance(m.content, str)]),
                    'confidence': np.mean([m.confidence for m in mods]),
                    'count': len(mods)
                }
            elif mod_type == ModalityType.TABLE.value:
                fused_content['modalities']['tables'] = [
                    {
                        'content': m.content,
                        'confidence': m.confidence,
                        'properties': m.properties
                    }
                    for m in mods
                ]
            elif mod_type == ModalityType.IMAGE.value:
                fused_content['modalities']['images'] = [
                    {
                        'content': m.content,
                        'confidence': m.confidence,
                        'properties': m.properties
                    }
                    for m in mods
                ]
            elif mod_type == ModalityType.CODE.value:
                fused_content['modalities']['code'] = [
                    {
                        'content': m.content,
                        'confidence': m.confidence,
                        'properties': m.properties
                    }
                    for m in mods
                ]

        # Add relationships
        for modality in modalities:
            if modality.relationships:
                fused_content['relationships'].append({
                    'source': modality.id,
                    'targets': modality.relationships,
                    'type': modality.type.value
                })

        return fused_content

    async def _attention_fusion(self, modalities: List[ModalityContent]) -> Dict[str, Any]:
        """Perform attention-based fusion."""
        # Placeholder for attention-based fusion
        # In a real implementation, this would use cross-modal attention mechanisms
        return await self._weighted_average_fusion(modalities)

    async def _hierarchical_fusion(self, modalities: List[ModalityContent]) -> Dict[str, Any]:
        """Perform hierarchical fusion."""
        # Placeholder for hierarchical fusion
        # In a real implementation, this would use hierarchical content organization
        return await self._weighted_average_fusion(modalities)

    def _calculate_modality_weights(self, modalities: List[ModalityContent]) -> Dict[str, float]:
        """Calculate weights for each modality type."""
        weights = {}
        total_confidence = 0

        # Group by type and sum confidences
        type_confidences = {}
        for modality in modalities:
            mod_type = modality.type.value
            if mod_type not in type_confidences:
                type_confidences[mod_type] = 0
            type_confidences[mod_type] += modality.confidence
            total_confidence += modality.confidence

        # Normalize weights
        for mod_type, confidence in type_confidences.items():
            weights[mod_type] = confidence / total_confidence if total_confidence > 0 else 0

        return weights

    async def _generate_semantic_summary(
        self,
        modalities: List[ModalityContent],
        fused_content: Dict[str, Any]
    ) -> str:
        """Generate semantic summary of fused content."""
        summary_parts = []

        # Text summary
        text_content = fused_content.get('modalities', {}).get('text', {}).get('content', '')
        if text_content:
            # Truncate long text
            if len(text_content) > 500:
                text_content = text_content[:500] + "..."
            summary_parts.append(f"Text: {text_content}")

        # Other modalities summary
        for mod_type, content in fused_content.get('modalities', {}).items():
            if mod_type != 'text' and isinstance(content, list):
                summary_parts.append(f"{mod_type.title()}: {len(content)} items")

        return " | ".join(summary_parts)

    def _calculate_fusion_confidence(
        self,
        modalities: List[ModalityContent],
        fused_content: Dict[str, Any]
    ) -> float:
        """Calculate overall fusion confidence."""
        if not modalities:
            return 0.0

        # Weighted average of modality confidences
        total_weight = 0
        weighted_confidence = 0

        for modality in modalities:
            weight = len(modality.content) if isinstance(modality.content, str) else 1
            total_weight += weight
            weighted_confidence += modality.confidence * weight

        return weighted_confidence / total_weight if total_weight > 0 else 0.0