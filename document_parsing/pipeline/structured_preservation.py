"""
结构化信息保存引擎

负责保存和维护文档的结构化信息，包括标题层级、列表结构、
表格数据、图像位置、引用关系等，确保文档结构的完整性。
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
from collections import defaultdict

from ..interfaces.parser_interface import TextChunk, ImageInfo, TableInfo, ParseResult


class StructureType(Enum):
    """结构类型。"""

    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"
    CAPTION = "caption"
    QUOTE = "quote"
    CODE_BLOCK = "code_block"


class RelationType(Enum):
    """关系类型。"""

    PARENT_CHILD = "parent_child"
    SEQUENTIAL = "sequential"
    CROSS_REFERENCE = "cross_reference"
    CITATION = "citation"
    FIGURE_REFERENCE = "figure_reference"
    TABLE_REFERENCE = "table_reference"
    SECTION_REFERENCE = "section_reference"


@dataclass
class StructureNode:
    """结构节点。"""

    node_id: str
    node_type: StructureType
    content: str
    level: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    spatial_info: Optional[Dict[str, Any]] = None
    temporal_info: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())


@dataclass
class StructureRelation:
    """结构关系。"""

    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self):
        if not self.relation_id:
            self.relation_id = str(uuid.uuid4())


@dataclass
class StructureTree:
    """结构树。"""

    tree_id: str
    root_nodes: List[str] = field(default_factory=list)
    all_nodes: Dict[str, StructureNode] = field(default_factory=dict)
    relations: List[StructureRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tree_id:
            self.tree_id = str(uuid.uuid4())

    def add_node(self, node: StructureNode) -> None:
        """添加节点。"""
        self.all_nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[StructureNode]:
        """获取节点。"""
        return self.all_nodes.get(node_id)

    def add_relation(self, relation: StructureRelation) -> None:
        """添加关系。"""
        self.relations.append(relation)

    def get_children(self, node_id: str) -> List[StructureNode]:
        """获取子节点。"""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.get_node(child_id) for child_id in node.children_ids if self.get_node(child_id)]

    def get_parent(self, node_id: str) -> Optional[StructureNode]:
        """获取父节点。"""
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return None
        return self.get_node(node.parent_id)

    def get_descendants(self, node_id: str) -> List[StructureNode]:
        """获取所有后代节点。"""
        descendants = []
        children = self.get_children(node_id)
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_descendants(child.node_id))
        return descendants

    def get_ancestors(self, node_id: str) -> List[StructureNode]:
        """获取所有祖先节点。"""
        ancestors = []
        parent = self.get_parent(node_id)
        while parent:
            ancestors.append(parent)
            parent = self.get_parent(parent.node_id)
        return ancestors


class StructuredPreservationEngine:
    """结构化信息保存引擎。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化结构保存引擎。

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 模式配置
        self.preservation_modes = {
            "full": self._full_preservation,
            "semantic": self._semantic_preservation,
            "layout": self._layout_preservation,
            "minimal": self._minimal_preservation
        }

        # 模式识别模式
        self.heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown 标题
            r'^[A-Z][^.]*\.$',   # 编号标题 (1. A.)
            r'^第[一二三四五六七八九十\d]+[章节条款]',  # 中文章节
            r'^[IVX]+\.\s+.+$',  # 罗马数字标题
            r'^\d+\.\d+\s+.+$'   # 多级编号标题
        ]

        self.list_patterns = [
            r'^\s*[-*+]\s+(.+)$',      # 无序列表
            r'^\s*\d+\.\s+(.+)$',      # 有序列表
            r'^\s*[a-zA-Z]\.\s+(.+)$', # 字母列表
            r'^\s*[ivx]+\.\s+(.+)$'    # 罗马数字列表
        ]

        # 缓存
        self.structure_cache: Dict[str, StructureTree] = {}

    async def preserve_structure(
        self,
        parse_result: ParseResult,
        mode: str = "full"
    ) -> StructureTree:
        """
        保存文档结构。

        Args:
            parse_result: 解析结果
            mode: 保存模式 (full, semantic, layout, minimal)

        Returns:
            StructureTree: 结构树
        """
        try:
            # 选择保存模式
            preservation_func = self.preservation_modes.get(mode, self._full_preservation)

            # 执行结构保存
            structure_tree = await preservation_func(parse_result)

            # 后处理：优化和验证
            await self._optimize_structure(structure_tree)
            await self._validate_structure(structure_tree)

            self.logger.info(f"文档结构保存完成，模式: {mode}，节点数: {len(structure_tree.all_nodes)}")

            return structure_tree

        except Exception as e:
            self.logger.error(f"文档结构保存失败: {e}")
            # 返回最小结构作为后备
            return await self._create_minimal_structure(parse_result)

    async def _full_preservation(self, parse_result: ParseResult) -> StructureTree:
        """完整结构保存。"""
        structure_tree = StructureTree(
            tree_id=str(uuid.uuid4()),
            metadata={"preservation_mode": "full", "created_at": datetime.now().isoformat()}
        )

        # 创建文档根节点
        doc_node = StructureNode(
            node_id="doc_root",
            node_type=StructureType.DOCUMENT,
            content=parse_result.metadata.title or "文档",
            level=0,
            metadata=parse_result.metadata.metadata
        )
        structure_tree.add_node(doc_node)
        structure_tree.root_nodes.append(doc_node.node_id)

        # 处理文本块并构建层次结构
        text_nodes = await self._process_text_blocks(parse_result.text_chunks, structure_tree)

        # 处理表格结构
        table_nodes = await self._process_tables(parse_result.tables, structure_tree)

        # 处理图像结构
        image_nodes = await self._process_images(parse_result.images, structure_tree)

        # 建立节点间的关系
        await self._establish_relations(text_nodes + table_nodes + image_nodes, structure_tree)

        return structure_tree

    async def _semantic_preservation(self, parse_result: ParseResult) -> StructureTree:
        """语义结构保存。"""
        structure_tree = StructureTree(
            tree_id=str(uuid.uuid4()),
            metadata={"preservation_mode": "semantic", "created_at": datetime.now().isoformat()}
        )

        # 创建文档根节点
        doc_node = StructureNode(
            node_id="doc_root",
            node_type=StructureType.DOCUMENT,
            content=parse_result.metadata.title or "文档",
            level=0
        )
        structure_tree.add_node(doc_node)
        structure_tree.root_nodes.append(doc_node.node_id)

        # 识别语义块（段落、列表等）
        semantic_blocks = await self._identify_semantic_blocks(parse_result.text_chunks)

        # 构建语义结构
        for block in semantic_blocks:
            block_node = StructureNode(
                node_id=str(uuid.uuid4()),
                node_type=StructureType(block["type"]),
                content=block["content"],
                level=block.get("level", 1),
                parent_id=doc_node.node_id,
                metadata=block.get("metadata", {})
            )
            structure_tree.add_node(block_node)
            doc_node.children_ids.append(block_node.node_id)

        return structure_tree

    async def _layout_preservation(self, parse_result: ParseResult) -> StructureTree:
        """布局结构保存。"""
        structure_tree = StructureTree(
            tree_id=str(uuid.uuid4()),
            metadata={"preservation_mode": "layout", "created_at": datetime.now().isoformat()}
        )

        # 创建文档根节点
        doc_node = StructureNode(
            node_id="doc_root",
            node_type=StructureType.DOCUMENT,
            content=parse_result.metadata.title or "文档",
            level=0
        )
        structure_tree.add_node(doc_node)
        structure_tree.root_nodes.append(doc_node.node_id)

        # 按页面组织结构
        pages = defaultdict(list)
        for chunk in parse_result.text_chunks:
            page_num = chunk.page_number or 1
            pages[page_num].append(chunk)

        for page_num in sorted(pages.keys()):
            page_node = StructureNode(
                node_id=f"page_{page_num}",
                node_type=StructureType.SECTION,
                content=f"第 {page_num} 页",
                level=1,
                parent_id=doc_node.node_id,
                spatial_info={"page": page_num}
            )
            structure_tree.add_node(page_node)
            doc_node.children_ids.append(page_node.node_id)

            # 在页面内按位置组织内容
            page_chunks = sorted(pages[page_num], key=lambda c: c.bbox[1] if c.bbox else 0)
            for chunk in page_chunks:
                chunk_node = StructureNode(
                    node_id=f"chunk_{chunk.chunk_id}",
                    node_type=StructureType.PARAGRAPH,
                    content=chunk.content,
                    level=2,
                    parent_id=page_node.node_id,
                    spatial_info={"bbox": chunk.bbox}
                )
                structure_tree.add_node(chunk_node)
                page_node.children_ids.append(chunk_node.node_id)

        return structure_tree

    async def _minimal_preservation(self, parse_result: ParseResult) -> StructureTree:
        """最小结构保存。"""
        structure_tree = StructureTree(
            tree_id=str(uuid.uuid4()),
            metadata={"preservation_mode": "minimal", "created_at": datetime.now().isoformat()}
        )

        # 创建文档根节点
        doc_node = StructureNode(
            node_id="doc_root",
            node_type=StructureType.DOCUMENT,
            content=parse_result.metadata.title or "文档",
            level=0
        )
        structure_tree.add_node(doc_node)
        structure_tree.root_nodes.append(doc_node.node_id)

        # 只保留主要段落
        main_chunks = [chunk for chunk in parse_result.text_chunks if len(chunk.content.strip()) > 50]

        for chunk in main_chunks:
            chunk_node = StructureNode(
                node_id=f"chunk_{chunk.chunk_id}",
                node_type=StructureType.PARAGRAPH,
                content=chunk.content,
                level=1,
                parent_id=doc_node.node_id
            )
            structure_tree.add_node(chunk_node)
            doc_node.children_ids.append(chunk_node.node_id)

        return structure_tree

    async def _process_text_blocks(
        self,
        text_chunks: List[TextChunk],
        structure_tree: StructureTree
    ) -> List[StructureNode]:
        """处理文本块。"""
        nodes = []
        current_section = None
        current_level = 0

        for chunk in text_chunks:
            content = chunk.content.strip()
            if not content:
                continue

            # 识别标题
            heading_match = self._identify_heading(content)
            if heading_match:
                heading_level, heading_content = heading_match

                # 创建标题节点
                heading_node = StructureNode(
                    node_id=f"heading_{chunk.chunk_id}",
                    node_type=StructureType.HEADING,
                    content=heading_content,
                    level=heading_level,
                    spatial_info={"bbox": chunk.bbox},
                    metadata={"original_text": content, "page": chunk.page_number}
                )
                structure_tree.add_node(heading_node)
                nodes.append(heading_node)

                # 更新当前章节
                current_section = heading_node
                current_level = heading_level

                # 建立父子关系
                await self._attach_to_parent(heading_node, structure_tree)

            # 识别列表
            elif self._is_list_item(content):
                list_node = StructureNode(
                    node_id=f"list_{chunk.chunk_id}",
                    node_type=StructureType.LIST_ITEM,
                    content=content,
                    level=current_level + 1,
                    parent_id=current_section.node_id if current_section else "doc_root",
                    spatial_info={"bbox": chunk.bbox},
                    metadata={"list_type": self._get_list_type(content)}
                )
                structure_tree.add_node(list_node)
                nodes.append(list_node)

            # 普通段落
            else:
                para_node = StructureNode(
                    node_id=f"para_{chunk.chunk_id}",
                    node_type=StructureType.PARAGRAPH,
                    content=content,
                    level=current_level + 1,
                    parent_id=current_section.node_id if current_section else "doc_root",
                    spatial_info={"bbox": chunk.bbox},
                    metadata={"page": chunk.page_number}
                )
                structure_tree.add_node(para_node)
                nodes.append(para_node)

        return nodes

    async def _process_tables(
        self,
        tables: List[TableInfo],
        structure_tree: StructureTree
    ) -> List[StructureNode]:
        """处理表格。"""
        nodes = []

        for table in tables:
            table_node = StructureNode(
                node_id=table.table_id,
                node_type=StructureType.TABLE,
                content=f"表格 ({table.rows}行 x {table.columns}列)",
                level=2,
                spatial_info={"bbox": table.bbox},
                metadata={
                    "rows": table.rows,
                    "columns": table.columns,
                    "headers": table.headers,
                    "data": table.data,
                    "page": table.page_number,
                    "confidence": table.confidence
                }
            )
            structure_tree.add_node(table_node)
            nodes.append(table_node)

            # 为表格添加标题节点（如果有）
            if table.headers:
                caption_node = StructureNode(
                    node_id=f"caption_{table.table_id}",
                    node_type=StructureType.CAPTION,
                    content=" | ".join(table.headers),
                    level=3,
                    parent_id=table.table_id,
                    metadata={"type": "table_header"}
                )
                structure_tree.add_node(caption_node)
                table_node.children_ids.append(caption_node.node_id)

        return nodes

    async def _process_images(
        self,
        images: List[ImageInfo],
        structure_tree: StructureTree
    ) -> List[StructureNode]:
        """处理图像。"""
        nodes = []

        for image in images:
            image_node = StructureNode(
                node_id=image.image_id,
                node_type=StructureType.FIGURE,
                content=f"图像 ({image.width}x{image.height})",
                level=2,
                spatial_info={"bbox": image.bbox},
                metadata={
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                    "page": image.page_number
                }
            )
            structure_tree.add_node(image_node)
            nodes.append(image_node)

            # 为图像添加标题节点
            caption_node = StructureNode(
                node_id=f"caption_{image.image_id}",
                node_type=StructureType.CAPTION,
                content=f"图 {len(nodes)}",
                level=3,
                parent_id=image.image_id,
                metadata={"type": "figure_caption"}
            )
            structure_tree.add_node(caption_node)
            image_node.children_ids.append(caption_node.node_id)

        return nodes

    async def _establish_relations(
        self,
        nodes: List[StructureNode],
        structure_tree: StructureTree
    ) -> None:
        """建立节点间关系。"""
        # 建立父子关系
        for node in nodes:
            if node.parent_id and node.parent_id in structure_tree.all_nodes:
                parent = structure_tree.get_node(node.parent_id)
                if parent and node.node_id not in parent.children_ids:
                    parent.children_ids.append(node.node_id)

                    # 添加关系
                    relation = StructureRelation(
                        relation_id=str(uuid.uuid4()),
                        source_id=parent.node_id,
                        target_id=node.node_id,
                        relation_type=RelationType.PARENT_CHILD
                    )
                    structure_tree.add_relation(relation)

        # 建立顺序关系
        sorted_nodes = sorted(
            [n for n in nodes if n.spatial_info and n.spatial_info.get("bbox")],
            key=lambda n: (
                n.metadata.get("page", 1),
                n.spatial_info["bbox"][1]  # Y坐标
            )
        )

        for i in range(len(sorted_nodes) - 1):
            relation = StructureRelation(
                relation_id=str(uuid.uuid4()),
                source_id=sorted_nodes[i].node_id,
                target_id=sorted_nodes[i + 1].node_id,
                relation_type=RelationType.SEQUENTIAL
            )
            structure_tree.add_relation(relation)

        # 建立引用关系
        await self._establish_cross_references(nodes, structure_tree)

    async def _establish_cross_references(
        self,
        nodes: List[StructureNode],
        structure_tree: StructureTree
    ) -> None:
        """建立交叉引用关系。"""
        # 识别图表引用
        figure_ref_pattern = r'(?:图|表格|Fig\.|Table)\s*(\d+)'

        for node in nodes:
            if node.node_type in [StructureType.PARAGRAPH, StructureType.SECTION]:
                matches = re.findall(figure_ref_pattern, node.content)
                for match in matches:
                    # 查找对应的图表节点
                    for target_node in nodes:
                        if (target_node.node_type in [StructureType.FIGURE, StructureType.TABLE] and
                            match in target_node.content):
                            relation = StructureRelation(
                                relation_id=str(uuid.uuid4()),
                                source_id=node.node_id,
                                target_id=target_node.node_id,
                                relation_type=RelationType.FIGURE_REFERENCE,
                                metadata={"reference": match}
                            )
                            structure_tree.add_relation(relation)

    async def _identify_semantic_blocks(self, text_chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """识别语义块。"""
        blocks = []

        for chunk in text_chunks:
            content = chunk.content.strip()
            if not content:
                continue

            # 识别标题
            heading_match = self._identify_heading(content)
            if heading_match:
                level, heading_content = heading_match
                blocks.append({
                    "type": "heading",
                    "content": heading_content,
                    "level": level,
                    "metadata": {"original": content}
                })

            # 识别列表
            elif self._is_list_item(content):
                blocks.append({
                    "type": "list_item",
                    "content": content,
                    "level": 2,
                    "metadata": {"list_type": self._get_list_type(content)}
                })

            # 识别代码块
            elif self._is_code_block(content):
                blocks.append({
                    "type": "code_block",
                    "content": content,
                    "level": 2,
                    "metadata": {"language": self._detect_code_language(content)}
                })

            # 普通段落
            else:
                blocks.append({
                    "type": "paragraph",
                    "content": content,
                    "level": 2,
                    "metadata": {}
                })

        return blocks

    def _identify_heading(self, text: str) -> Optional[Tuple[int, str]]:
        """识别标题。"""
        for pattern in self.heading_patterns:
            match = re.match(pattern, text)
            if match:
                if pattern.startswith('^#{1,6}'):
                    level = len(re.match(r'^(#+)', text).group(1))
                    content = match.group(1).strip()
                elif '第' in pattern:
                    level = 1
                    content = match.group(0).strip()
                else:
                    level = 1
                    content = match.group(0).strip()

                return (level, content)

        return None

    def _is_list_item(self, text: str) -> bool:
        """判断是否为列表项。"""
        for pattern in self.list_patterns:
            if re.match(pattern, text):
                return True
        return False

    def _get_list_type(self, text: str) -> str:
        """获取列表类型。"""
        if re.match(r'^\s*[-*+]\s+', text):
            return "unordered"
        elif re.match(r'^\s*\d+\.\s+', text):
            return "ordered_numeric"
        elif re.match(r'^\s*[a-zA-Z]\.\s+', text):
            return "ordered_alpha"
        else:
            return "unknown"

    def _is_code_block(self, text: str) -> bool:
        """判断是否为代码块。"""
        # 简单的代码块识别
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'function()', 'var ', 'let ', 'const ',
            '<?php', '<!DOCTYPE', '<html>', '<?xml', '#!/bin/', 'SELECT ', 'INSERT '
        ]
        return any(indicator in text for indicator in code_indicators)

    def _detect_code_language(self, text: str) -> str:
        """检测代码语言。"""
        if 'def ' in text or 'class ' in text or 'import ' in text:
            return "python"
        elif 'function()' in text or 'var ' in text or 'let ' in text:
            return "javascript"
        elif '<?php' in text:
            return "php"
        elif 'SELECT ' in text or 'INSERT ' in text:
            return "sql"
        else:
            return "unknown"

    async def _attach_to_parent(self, node: StructureNode, structure_tree: StructureTree) -> None:
        """将节点附加到父节点。"""
        # 根据级别找到合适的父节点
        potential_parents = [
            n for n in structure_tree.all_nodes.values()
            if n.level < node.level and n.node_type != StructureType.DOCUMENT
        ]

        if potential_parents:
            # 选择级别最高的父节点
            parent = max(potential_parents, key=lambda n: n.level)
            node.parent_id = parent.node_id
            parent.children_ids.append(node.node_id)

            # 添加关系
            relation = StructureRelation(
                relation_id=str(uuid.uuid4()),
                source_id=parent.node_id,
                target_id=node.node_id,
                relation_type=RelationType.PARENT_CHILD
            )
            structure_tree.add_relation(relation)
        else:
            # 附加到根节点
            node.parent_id = "doc_root"
            root_node = structure_tree.get_node("doc_root")
            if root_node:
                root_node.children_ids.append(node.node_id)

    async def _optimize_structure(self, structure_tree: StructureTree) -> None:
        """优化结构。"""
        # 移除空节点
        empty_nodes = [
            node_id for node_id, node in structure_tree.all_nodes.items()
            if not node.content.strip()
        ]

        for node_id in empty_nodes:
            await self._remove_node(structure_tree, node_id)

        # 合并相邻的同级节点
        await self._merge_adjacent_nodes(structure_tree)

    async def _remove_node(self, structure_tree: StructureTree, node_id: str) -> None:
        """移除节点。"""
        node = structure_tree.get_node(node_id)
        if not node:
            return

        # 将子节点重新附加到父节点
        parent = structure_tree.get_parent(node_id)
        if parent:
            for child_id in node.children_ids:
                child = structure_tree.get_node(child_id)
                if child:
                    child.parent_id = parent.node_id
                    if child_id not in parent.children_ids:
                        parent.children_ids.append(child_id)

        # 从父节点中移除
        if parent and node_id in parent.children_ids:
            parent.children_ids.remove(node_id)

        # 从根节点中移除
        if node_id in structure_tree.root_nodes:
            structure_tree.root_nodes.remove(node_id)

        # 删除节点和相关关系
        del structure_tree.all_nodes[node_id]
        structure_tree.relations = [
            r for r in structure_tree.relations
            if r.source_id != node_id and r.target_id != node_id
        ]

    async def _merge_adjacent_nodes(self, structure_tree: StructureTree) -> None:
        """合并相邻的同级节点。"""
        # 简化实现：只合并相邻的段落
        for parent in structure_tree.all_nodes.values():
            if len(parent.children_ids) < 2:
                continue

            merged_pairs = []
            children = [structure_tree.get_node(child_id) for child_id in parent.children_ids]
            children = [c for c in children if c]  # 过滤空值

            for i in range(len(children) - 1):
                current = children[i]
                next_node = children[i + 1]

                # 合并条件：同类型且级别相同
                if (current.node_type == next_node.node_type and
                    current.level == next_node.level and
                    current.node_type == StructureType.PARAGRAPH):
                    merged_pairs.append((current.node_id, next_node.node_id))

            # 执行合并
            for source_id, target_id in merged_pairs:
                await self._merge_nodes(structure_tree, source_id, target_id)

    async def _merge_nodes(self, structure_tree: StructureTree, source_id: str, target_id: str) -> None:
        """合并两个节点。"""
        source = structure_tree.get_node(source_id)
        target = structure_tree.get_node(target_id)

        if not source or not target:
            return

        # 合并内容
        source.content += "\n" + target.content
        source.confidence = (source.confidence + target.confidence) / 2

        # 转移子节点
        for child_id in target.children_ids:
            child = structure_tree.get_node(child_id)
            if child:
                child.parent_id = source.node_id
                source.children_ids.append(child_id)

        # 移除目标节点
        await self._remove_node(structure_tree, target_id)

    async def _validate_structure(self, structure_tree: StructureTree) -> None:
        """验证结构。"""
        # 检查循环引用
        visited = set()
        for root_id in structure_tree.root_nodes:
            await self._check_circular_reference(structure_tree, root_id, visited)

        # 检查孤立节点
        await self._check_orphaned_nodes(structure_tree)

    async def _check_circular_reference(
        self,
        structure_tree: StructureTree,
        node_id: str,
        visited: Set[str],
        path: Set[str] = None
    ) -> None:
        """检查循环引用。"""
        if path is None:
            path = set()

        if node_id in path:
            self.logger.error(f"检测到循环引用: {node_id}")
            return

        if node_id in visited:
            return

        visited.add(node_id)
        path.add(node_id)

        node = structure_tree.get_node(node_id)
        if node:
            for child_id in node.children_ids:
                await self._check_circular_reference(structure_tree, child_id, visited, path.copy())

    async def _check_orphaned_nodes(self, structure_tree: StructureTree) -> None:
        """检查孤立节点。"""
        for node_id, node in structure_tree.all_nodes.items():
            if (node.node_id not in structure_tree.root_nodes and
                not node.parent_id and
                node.node_type != StructureType.DOCUMENT):
                self.logger.warning(f"发现孤立节点: {node_id}")

    async def _create_minimal_structure(self, parse_result: ParseResult) -> StructureTree:
        """创建最小结构（后备方案）。"""
        structure_tree = StructureTree(
            tree_id=str(uuid.uuid4()),
            metadata={"preservation_mode": "minimal_fallback", "created_at": datetime.now().isoformat()}
        )

        # 创建文档根节点
        doc_node = StructureNode(
            node_id="doc_root",
            node_type=StructureType.DOCUMENT,
            content=parse_result.metadata.title or "文档",
            level=0
        )
        structure_tree.add_node(doc_node)
        structure_tree.root_nodes.append(doc_node.node_id)

        # 创建单个内容节点
        content_node = StructureNode(
            node_id="content",
            node_type=StructureType.PARAGRAPH,
            content=parse_result.full_text or "",
            level=1,
            parent_id="doc_root"
        )
        structure_tree.add_node(content_node)
        doc_node.children_ids.append(content_node.node_id)

        return structure_tree

    async def export_structure(
        self,
        structure_tree: StructureTree,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """
        导出结构。

        Args:
            structure_tree: 结构树
            format: 导出格式 (json, yaml, xml)

        Returns:
            导出的结构数据
        """
        if format == "json":
            return await self._export_to_json(structure_tree)
        elif format == "yaml":
            return await self._export_to_yaml(structure_tree)
        elif format == "xml":
            return await self._export_to_xml(structure_tree)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    async def _export_to_json(self, structure_tree: StructureTree) -> str:
        """导出为JSON格式。"""
        def node_to_dict(node: StructureNode) -> Dict[str, Any]:
            return {
                "id": node.node_id,
                "type": node.node_type.value,
                "content": node.content,
                "level": node.level,
                "parent_id": node.parent_id,
                "children": node.children_ids,
                "metadata": node.metadata,
                "spatial_info": node.spatial_info,
                "attributes": node.attributes,
                "confidence": node.confidence
            }

        export_data = {
            "tree_id": structure_tree.tree_id,
            "metadata": structure_tree.metadata,
            "root_nodes": structure_tree.root_nodes,
            "nodes": {node_id: node_to_dict(node) for node_id, node in structure_tree.all_nodes.items()},
            "relations": [
                {
                    "id": rel.relation_id,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "type": rel.relation_type.value,
                    "metadata": rel.metadata,
                    "confidence": rel.confidence
                }
                for rel in structure_tree.relations
            ]
        }

        return json.dumps(export_data, ensure_ascii=False, indent=2)

    async def _export_to_yaml(self, structure_tree: StructureTree) -> str:
        """导出为YAML格式。"""
        try:
            import yaml
            json_data = await self._export_to_json(structure_tree)
            data = json.loads(json_data)
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        except ImportError:
            self.logger.warning("PyYAML 未安装，无法导出YAML格式")
            return await self._export_to_json(structure_tree)

    async def _export_to_xml(self, structure_tree: StructureTree) -> str:
        """导出为XML格式。"""
        # 简化的XML导出
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append(f'<document id="{structure_tree.tree_id}">')

        for node_id in structure_tree.root_nodes:
            node = structure_tree.get_node(node_id)
            if node:
                xml_parts.extend(await self._node_to_xml(node, structure_tree, 1))

        xml_parts.append('</document>')
        return '\n'.join(xml_parts)

    async def _node_to_xml(
        self,
        node: StructureNode,
        structure_tree: StructureTree,
        indent: int
    ) -> List[str]:
        """将节点转换为XML。"""
        indent_str = '  ' * indent
        tag_name = node.node_type.value

        xml_parts = [
            f'{indent_str}<{tag_name} id="{node.node_id}" level="{node.level}">'
        ]
        xml_parts.append(f'{indent_str}  <content><![CDATA[{node.content}]]></content>')

        # 递归处理子节点
        for child_id in node.children_ids:
            child = structure_tree.get_node(child_id)
            if child:
                xml_parts.extend(await self._node_to_xml(child, structure_tree, indent + 1))

        xml_parts.append(f'{indent_str}</{tag_name}>')
        return xml_parts