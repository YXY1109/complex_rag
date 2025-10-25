"""
多模态内容融合引擎

负责融合文档中的文本、图像、表格、音频、视频等多模态内容，
实现跨模态的语义关联和内容理解。
"""

import asyncio
import base64
import io
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import numpy as np
from PIL import Image
import json

from ..interfaces.parser_interface import TextChunk, ImageInfo, TableInfo, ParseResult


class ModalityType(Enum):
    """模态类型。"""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    AUDIO = "audio"
    VIDEO = "video"
    CHART = "chart"
    DIAGRAM = "diagram"


class FusionStrategy(Enum):
    """融合策略。"""

    SIMPLE_CONCAT = "simple_concat"           # 简单拼接
    SEMANTIC_ALIGNMENT = "semantic_alignment" # 语义对齐
    CROSS_MODAL_ATTENTION = "cross_attention" # 跨模态注意力
    HIERARCHICAL_FUSION = "hierarchical_fusion" # 层次化融合
    GRAPH_BASED = "graph_based"              # 基于图的融合


@dataclass
class ModalityContent:
    """模态内容。"""

    content_id: str
    modality_type: ModalityType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    features: Dict[str, Any] = field(default_factory=dict)
    spatial_info: Optional[Dict[str, Any]] = None  # 空间位置信息
    temporal_info: Optional[Dict[str, Any]] = None # 时间信息
    confidence: float = 1.0
    source_ref: Optional[str] = None  # 源引用

    def __post_init__(self):
        if not self.content_id:
            self.content_id = str(uuid.uuid4())


@dataclass
class CrossModalRelation:
    """跨模态关系。"""

    relation_id: str
    source_content_id: str
    target_content_id: str
    relation_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.relation_id:
            self.relation_id = str(uuid.uuid4())


@dataclass
class FusionResult:
    """融合结果。"""

    fusion_id: str
    unified_content: str
    modality_contents: List[ModalityContent]
    cross_modal_relations: List[CrossModalRelation]
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    processing_time: float = 0.0

    def __post_init__(self):
        if not self.fusion_id:
            self.fusion_id = str(uuid.uuid4())


class MultimodalFusionEngine:
    """多模态内容融合引擎。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化融合引擎。

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 融合策略配置
        self.default_strategy = FusionStrategy.SEMANTIC_ALIGNMENT
        self.embedding_models = {}
        self.feature_extractors = {}

        # 缓存
        self.content_cache: Dict[str, ModalityContent] = {}
        self.relation_cache: Dict[str, List[CrossModalRelation]] = {}

        # 初始化组件
        self._init_embedding_models()
        self._init_feature_extractors()

    def _init_embedding_models(self) -> None:
        """初始化嵌入模型。"""
        try:
            # 文本嵌入模型
            from sentence_transformers import SentenceTransformer
            self.embedding_models['text'] = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            self.logger.warning("sentence_transformers 未安装，文本嵌入功能不可用")

        try:
            # 图像嵌入模型
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            self.embedding_models['image'] = {
                'model': model,
                'preprocess': preprocess,
                'device': device
            }
        except ImportError:
            self.logger.warning("CLIP 未安装，图像嵌入功能不可用")

    def _init_feature_extractors(self) -> None:
        """初始化特征提取器。"""
        try:
            import cv2
            self.feature_extractors['opencv'] = cv2
        except ImportError:
            self.logger.warning("OpenCV 未安装，图像特征提取功能受限")

    async def fuse_content(
        self,
        parse_result: ParseResult,
        strategy: Optional[FusionStrategy] = None
    ) -> FusionResult:
        """
        融合多模态内容。

        Args:
            parse_result: 解析结果
            strategy: 融合策略

        Returns:
            FusionResult: 融合结果
        """
        start_time = datetime.now()

        # 使用默认策略
        fusion_strategy = strategy or self.default_strategy

        try:
            # 提取模态内容
            modality_contents = await self._extract_modalities(parse_result)

            if len(modality_contents) <= 1:
                # 单模态内容，直接返回
                return self._create_simple_fusion_result(modality_contents, start_time)

            # 执行融合
            if fusion_strategy == FusionStrategy.SIMPLE_CONCAT:
                result = await self._simple_concat_fusion(modality_contents)
            elif fusion_strategy == FusionStrategy.SEMANTIC_ALIGNMENT:
                result = await self._semantic_alignment_fusion(modality_contents)
            elif fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
                result = await self._cross_modal_attention_fusion(modality_contents)
            elif fusion_strategy == FusionStrategy.HIERARCHICAL_FUSION:
                result = await self._hierarchical_fusion(modality_contents)
            elif fusion_strategy == FusionStrategy.GRAPH_BASED:
                result = await self._graph_based_fusion(modality_contents)
            else:
                result = await self._semantic_alignment_fusion(modality_contents)

            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            # 评估融合质量
            result.quality_score = await self._evaluate_fusion_quality(result)

            self.logger.info(f"多模态内容融合完成，耗时: {processing_time:.2f}秒，质量分数: {result.quality_score:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"多模态内容融合失败: {e}")
            # 返回简单的文本内容作为后备
            return self._create_fallback_fusion_result(parse_result, start_time)

    async def _extract_modalities(self, parse_result: ParseResult) -> List[ModalityContent]:
        """提取模态内容。"""
        modalities = []

        # 提取文本内容
        if parse_result.text_chunks:
            for chunk in parse_result.text_chunks:
                text_content = ModalityContent(
                    content_id=f"text_{chunk.chunk_id}",
                    modality_type=ModalityType.TEXT,
                    content=chunk.content,
                    metadata={
                        "page_number": chunk.page_number,
                        "confidence": chunk.confidence,
                        "bbox": chunk.bbox
                    },
                    spatial_info={"bbox": chunk.bbox} if chunk.bbox else None,
                    confidence=chunk.confidence,
                    source_ref=chunk.chunk_id
                )
                modalities.append(text_content)

        # 提取图像内容
        if parse_result.images:
            for img in parse_result.images:
                image_content = ModalityContent(
                    content_id=img.image_id,
                    modality_type=ModalityType.IMAGE,
                    content=img,
                    metadata={
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "page_number": img.page_number
                    },
                    spatial_info={"bbox": img.bbox} if img.bbox else None,
                    confidence=0.9,
                    source_ref=img.image_id
                )
                modalities.append(image_content)

        # 提取表格内容
        if parse_result.tables:
            for table in parse_result.tables:
                table_content = ModalityContent(
                    content_id=table.table_id,
                    modality_type=ModalityType.TABLE,
                    content=table,
                    metadata={
                        "rows": table.rows,
                        "columns": table.columns,
                        "headers": table.headers,
                        "page_number": table.page_number
                    },
                    spatial_info={"bbox": table.bbox} if table.bbox else None,
                    confidence=table.confidence,
                    source_ref=table.table_id
                )
                modalities.append(table_content)

        # 提取结构化数据中的图表信息
        if parse_result.structured_data:
            charts = parse_result.structured_data.get("charts", [])
            for i, chart in enumerate(charts):
                chart_content = ModalityContent(
                    content_id=f"chart_{i}",
                    modality_type=ModalityType.CHART,
                    content=chart,
                    metadata={"chart_type": chart.get("type")},
                    confidence=0.8
                )
                modalities.append(chart_content)

        # 为每个模态内容生成嵌入
        for modality in modalities:
            await self._generate_embeddings(modality)

        return modalities

    async def _generate_embeddings(self, modality: ModalityContent) -> None:
        """生成嵌入向量。"""
        try:
            if modality.modality_type == ModalityType.TEXT:
                if 'text' in self.embedding_models:
                    embedding = self.embedding_models['text'].encode(modality.content)
                    modality.embeddings = embedding

            elif modality.modality_type == ModalityType.IMAGE:
                if 'image' in self.embedding_models:
                    # 这里需要实际的图像数据
                    # 简化处理，使用图像元数据生成嵌入
                    metadata_text = f"Image {modality.metadata.get('width', 0)}x{modality.metadata.get('height', 0)}"
                    if 'text' in self.embedding_models:
                        embedding = self.embedding_models['text'].encode(metadata_text)
                        modality.embeddings = embedding

        except Exception as e:
            self.logger.warning(f"生成嵌入失败: {e}")

    async def _simple_concat_fusion(self, modalities: List[ModalityContent]) -> FusionResult:
        """简单拼接融合。"""
        # 按页码和位置排序
        sorted_modalities = sorted(
            modalities,
            key=lambda m: (
                m.metadata.get("page_number", 0),
                m.spatial_info.get("bbox", [0, 0, 0, 0])[1] if m.spatial_info else 0
            )
        )

        # 拼接内容
        unified_content_parts = []
        for modality in sorted_modalities:
            if modality.modality_type == ModalityType.TEXT:
                unified_content_parts.append(modality.content)
            elif modality.modality_type == ModalityType.TABLE:
                table_text = self._table_to_text(modality.content)
                unified_content_parts.append(f"[表格]\n{table_text}")
            elif modality.modality_type == ModalityType.IMAGE:
                img_desc = self._image_to_description(modality.content)
                unified_content_parts.append(f"[图像] {img_desc}")
            elif modality.modality_type == ModalityType.CHART:
                chart_desc = self._chart_to_description(modality.content)
                unified_content_parts.append(f"[图表] {chart_desc}")

        unified_content = "\n\n".join(unified_content_parts)

        return FusionResult(
            fusion_id=str(uuid.uuid4()),
            unified_content=unified_content,
            modality_contents=modalities,
            cross_modal_relations=[],
            fusion_metadata={"strategy": FusionStrategy.SIMPLE_CONCAT.value}
        )

    async def _semantic_alignment_fusion(self, modalities: List[ModalityContent]) -> FusionResult:
        """语义对齐融合。"""
        # 计算模态间的语义相似度
        relations = await self._compute_cross_modal_relations(modalities)

        # 基于语义关系组织内容
        unified_content_parts = []
        processed_modalities = set()

        # 处理文本内容作为主干
        text_modalities = [m for m in modalities if m.modality_type == ModalityType.TEXT]
        for text_mod in text_modalities:
            unified_content_parts.append(text_mod.content)
            processed_modalities.add(text_mod.content_id)

            # 查找相关的非文本内容
            related_content = []
            for relation in relations:
                if relation.source_content_id == text_mod.content_id:
                    target_mod = next((m for m in modalities if m.content_id == relation.target_content_id), None)
                    if target_mod and target_mod.content_id not in processed_modalities:
                        if target_mod.modality_type == ModalityType.TABLE:
                            table_text = self._table_to_text(target_mod.content)
                            related_content.append(f"[相关表格]\n{table_text}")
                        elif target_mod.modality_type == ModalityType.IMAGE:
                            img_desc = self._image_to_description(target_mod.content)
                            related_content.append(f"[相关图像] {img_desc}")
                        elif target_mod.modality_type == ModalityType.CHART:
                            chart_desc = self._chart_to_description(target_mod.content)
                            related_content.append(f"[相关图表] {chart_desc}")
                        processed_modalities.add(target_mod.content_id)

            if related_content:
                unified_content_parts.extend(related_content)

        # 处理未关联的内容
        for modality in modalities:
            if modality.content_id not in processed_modalities:
                if modality.modality_type == ModalityType.TABLE:
                    table_text = self._table_to_text(modality.content)
                    unified_content_parts.append(f"[表格]\n{table_text}")
                elif modality.modality_type == ModalityType.IMAGE:
                    img_desc = self._image_to_description(modality.content)
                    unified_content_parts.append(f"[图像] {img_desc}")
                elif modality.modality_type == ModalityType.CHART:
                    chart_desc = self._chart_to_description(modality.content)
                    unified_content_parts.append(f"[图表] {chart_desc}")

        unified_content = "\n\n".join(unified_content_parts)

        return FusionResult(
            fusion_id=str(uuid.uuid4()),
            unified_content=unified_content,
            modality_contents=modalities,
            cross_modal_relations=relations,
            fusion_metadata={"strategy": FusionStrategy.SEMANTIC_ALIGNMENT.value}
        )

    async def _cross_modal_attention_fusion(self, modalities: List[ModalityContent]) -> FusionResult:
        """跨模态注意力融合。"""
        # 简化实现：基于空间位置计算注意力权重
        unified_content_parts = []
        spatial_groups = self._group_by_spatial_proximity(modalities)

        for group in spatial_groups:
            group_content = []
            for modality in group:
                if modality.modality_type == ModalityType.TEXT:
                    group_content.append(modality.content)
                elif modality.modality_type == ModalityType.TABLE:
                    table_text = self._table_to_text(modality.content)
                    group_content.append(f"[表格] {table_text}")
                elif modality.modality_type == ModalityType.IMAGE:
                    img_desc = self._image_to_description(modality.content)
                    group_content.append(f"[图像] {img_desc}")

            if group_content:
                unified_content_parts.append("\n".join(group_content))

        unified_content = "\n\n".join(unified_content_parts)

        return FusionResult(
            fusion_id=str(uuid.uuid4()),
            unified_content=unified_content,
            modality_contents=modalities,
            cross_modal_relations=[],
            fusion_metadata={"strategy": FusionStrategy.CROSS_MODAL_ATTENTION.value}
        )

    async def _hierarchical_fusion(self, modalities: List[ModalityContent]) -> FusionResult:
        """层次化融合。"""
        # 按页码分组
        page_groups = {}
        for modality in modalities:
            page_num = modality.metadata.get("page_number", 1)
            if page_num not in page_groups:
                page_groups[page_num] = []
            page_groups[page_num].append(modality)

        # 按页组织内容
        unified_content_parts = []
        for page_num in sorted(page_groups.keys()):
            page_content = [f"=== 第 {page_num} 页 ==="]
            page_modalities = page_groups[page_num]

            # 在页内按位置排序
            page_modalities.sort(key=lambda m: m.spatial_info.get("bbox", [0, 0, 0, 0])[1] if m.spatial_info else 0)

            for modality in page_modalities:
                if modality.modality_type == ModalityType.TEXT:
                    page_content.append(modality.content)
                elif modality.modality_type == ModalityType.TABLE:
                    table_text = self._table_to_text(modality.content)
                    page_content.append(f"[表格]\n{table_text}")
                elif modality.modality_type == ModalityType.IMAGE:
                    img_desc = self._image_to_description(modality.content)
                    page_content.append(f"[图像] {img_desc}")

            unified_content_parts.append("\n".join(page_content))

        unified_content = "\n\n".join(unified_content_parts)

        return FusionResult(
            fusion_id=str(uuid.uuid4()),
            unified_content=unified_content,
            modality_contents=modalities,
            cross_modal_relations=[],
            fusion_metadata={"strategy": FusionStrategy.HIERARCHICAL_FUSION.value}
        )

    async def _graph_based_fusion(self, modalities: List[ModalityContent]) -> FusionResult:
        """基于图的融合。"""
        # 构建模态关系图
        relations = await self._compute_cross_modal_relations(modalities)

        # 简化实现：基于关系图遍历生成内容
        visited = set()
        unified_content_parts = []

        def traverse_modality(modality_id: str, depth: int = 0):
            if modality_id in visited or depth > 3:
                return

            visited.add(modality_id)
            modality = next((m for m in modalities if m.content_id == modality_id), None)
            if not modality:
                return

            indent = "  " * depth
            if modality.modality_type == ModalityType.TEXT:
                unified_content_parts.append(f"{indent}{modality.content}")
            elif modality.modality_type == ModalityType.TABLE:
                table_text = self._table_to_text(modality.content)
                unified_content_parts.append(f"{indent}[表格]\n{indent}{table_text}")
            elif modality.modality_type == ModalityType.IMAGE:
                img_desc = self._image_to_description(modality.content)
                unified_content_parts.append(f"{indent}[图像] {img_desc}")

            # 遍历相关的模态
            for relation in relations:
                if relation.source_content_id == modality_id:
                    traverse_modality(relation.target_content_id, depth + 1)

        # 从文本模态开始遍历
        text_modalities = [m for m in modalities if m.modality_type == ModalityType.TEXT]
        for text_mod in text_modalities:
            traverse_modality(text_mod.content_id)

        # 处理未访问的内容
        for modality in modalities:
            if modality.content_id not in visited:
                if modality.modality_type == ModalityType.TABLE:
                    table_text = self._table_to_text(modality.content)
                    unified_content_parts.append(f"[表格]\n{table_text}")
                elif modality.modality_type == ModalityType.IMAGE:
                    img_desc = self._image_to_description(modality.content)
                    unified_content_parts.append(f"[图像] {img_desc}")

        unified_content = "\n\n".join(unified_content_parts)

        return FusionResult(
            fusion_id=str(uuid.uuid4()),
            unified_content=unified_content,
            modality_contents=modalities,
            cross_modal_relations=relations,
            fusion_metadata={"strategy": FusionStrategy.GRAPH_BASED.value}
        )

    async def _compute_cross_modal_relations(self, modalities: List[ModalityContent]) -> List[CrossModalRelation]:
        """计算跨模态关系。"""
        relations = []

        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i >= j:
                    continue

                # 基于空间位置计算关系
                if mod1.spatial_info and mod2.spatial_info:
                    spatial_relation = self._compute_spatial_relation(mod1, mod2)
                    if spatial_relation:
                        relations.append(spatial_relation)

                # 基于嵌入相似度计算关系
                if mod1.embeddings is not None and mod2.embeddings is not None:
                    semantic_relation = self._compute_semantic_relation(mod1, mod2)
                    if semantic_relation:
                        relations.append(semantic_relation)

        return relations

    def _compute_spatial_relation(self, mod1: ModalityContent, mod2: ModalityContent) -> Optional[CrossModalRelation]:
        """计算空间关系。"""
        bbox1 = mod1.spatial_info.get("bbox", [0, 0, 0, 0])
        bbox2 = mod2.spatial_info.get("bbox", [0, 0, 0, 0])

        # 计算距离
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

        if distance < 100:  # 阈值可配置
            return CrossModalRelation(
                relation_id=str(uuid.uuid4()),
                source_content_id=mod1.content_id,
                target_content_id=mod2.content_id,
                relation_type="spatial_proximity",
                confidence=max(0, 1 - distance / 100),
                metadata={"distance": distance}
            )

        return None

    def _compute_semantic_relation(self, mod1: ModalityContent, mod2: ModalityContent) -> Optional[CrossModalRelation]:
        """计算语义关系。"""
        try:
            # 计算余弦相似度
            similarity = np.dot(mod1.embeddings, mod2.embeddings) / (
                np.linalg.norm(mod1.embeddings) * np.linalg.norm(mod2.embeddings)
            )

            if similarity > 0.7:  # 阈值可配置
                return CrossModalRelation(
                    relation_id=str(uuid.uuid4()),
                    source_content_id=mod1.content_id,
                    target_content_id=mod2.content_id,
                    relation_type="semantic_similarity",
                    confidence=float(similarity),
                    metadata={"similarity": float(similarity)}
                )

        except Exception as e:
            self.logger.warning(f"计算语义关系失败: {e}")

        return None

    def _group_by_spatial_proximity(self, modalities: List[ModalityContent]) -> List[List[ModalityContent]]:
        """按空间邻近性分组。"""
        groups = []
        unassigned = modalities.copy()

        while unassigned:
            # 创建新组
            current_group = [unassigned.pop(0)]

            # 查找邻近的模态
            i = 0
            while i < len(unassigned):
                mod = unassigned[i]
                is_near = False

                for group_mod in current_group:
                    if (group_mod.spatial_info and mod.spatial_info and
                        self._is_spatially_near(group_mod, mod)):
                        is_near = True
                        break

                if is_near:
                    current_group.append(unassigned.pop(i))
                else:
                    i += 1

            groups.append(current_group)

        return groups

    def _is_spatially_near(self, mod1: ModalityContent, mod2: ModalityContent, threshold: float = 150) -> bool:
        """判断两个模态是否空间邻近。"""
        bbox1 = mod1.spatial_info.get("bbox", [0, 0, 0, 0])
        bbox2 = mod2.spatial_info.get("bbox", [0, 0, 0, 0])

        # 计算中心点距离
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        return distance < threshold

    def _table_to_text(self, table: TableInfo) -> str:
        """将表格转换为文本。"""
        if not table.data:
            return ""

        text_parts = []

        # 添加表头
        if table.headers:
            text_parts.append("表头: " + " | ".join(table.headers))

        # 添加数据行
        for row in table.data:
            if row:
                text_parts.append(" | ".join(str(cell) for cell in row))

        return "\n".join(text_parts)

    def _image_to_description(self, image: ImageInfo) -> str:
        """将图像转换为描述。"""
        description_parts = []

        if image.width and image.height:
            description_parts.append(f"尺寸: {image.width}x{image.height}")

        if image.format:
            description_parts.append(f"格式: {image.format}")

        if image.metadata:
            for key, value in image.metadata.items():
                if key not in ["xref", "colorspace"]:
                    description_parts.append(f"{key}: {value}")

        return ", ".join(description_parts) if description_parts else "图像"

    def _chart_to_description(self, chart: Dict[str, Any]) -> str:
        """将图表转换为描述。"""
        chart_type = chart.get("type", "未知图表")
        title = chart.get("title", "")
        description_parts = [f"类型: {chart_type}"]

        if title:
            description_parts.append(f"标题: {title}")

        return ", ".join(description_parts)

    def _create_simple_fusion_result(self, modalities: List[ModalityContent], start_time: datetime) -> FusionResult:
        """创建简单融合结果。"""
        if not modalities:
            unified_content = ""
        elif modalities[0].modality_type == ModalityType.TEXT:
            unified_content = modalities[0].content
        else:
            unified_content = f"[{modalities[0].modality_type.value}] 内容"

        processing_time = (datetime.now() - start_time).total_seconds()

        return FusionResult(
            fusion_id=str(uuid.uuid4()),
            unified_content=unified_content,
            modality_contents=modalities,
            cross_modal_relations=[],
            fusion_metadata={"strategy": "simple"},
            processing_time=processing_time
        )

    def _create_fallback_fusion_result(self, parse_result: ParseResult, start_time: datetime) -> FusionResult:
        """创建后备融合结果。"""
        processing_time = (datetime.now() - start_time).total_seconds()

        return FusionResult(
            fusion_id=str(uuid.uuid4()),
            unified_content=parse_result.full_text or "",
            modality_contents=[],
            cross_modal_relations=[],
            fusion_metadata={"strategy": "fallback", "error": "fusion_failed"},
            processing_time=processing_time
        )

    async def _evaluate_fusion_quality(self, result: FusionResult) -> float:
        """评估融合质量。"""
        quality_factors = []

        # 内容完整性
        content_completeness = len(result.unified_content) / 1000.0  # 基于内容长度
        quality_factors.append(min(1.0, content_completeness))

        # 模态覆盖率
        if result.modality_contents:
            modality_types = set(m.modality_type for m in result.modality_contents)
            modality_coverage = len(modality_types) / 4.0  # 假设最多4种模态
            quality_factors.append(min(1.0, modality_coverage))

        # 关系丰富度
        if result.modality_contents:
            relation_density = len(result.cross_modal_relations) / len(result.modality_contents)
            quality_factors.append(min(1.0, relation_density))

        # 平均置信度
        if result.modality_contents:
            avg_confidence = sum(m.confidence for m in result.modality_contents) / len(result.modality_contents)
            quality_factors.append(avg_confidence)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0

    async def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息。"""
        return {
            "supported_modalities": [m.value for m in ModalityType],
            "available_strategies": [s.value for s in FusionStrategy],
            "embedding_models": list(self.embedding_models.keys()),
            "feature_extractors": list(self.feature_extractors.keys()),
            "cache_size": len(self.content_cache)
        }