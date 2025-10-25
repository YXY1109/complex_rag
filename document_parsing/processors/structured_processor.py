"""
结构化数据处理器

此模块实现JSON、CSV、XML、YAML等结构化数据的专用处理器，
专门处理数据文件和结构化内容。
"""

import asyncio
import os
import json
import csv
import xml.etree.ElementTree as ET
import yaml
from typing import Dict, Any, List, Optional, Union, BinaryIO, Iterator
from pathlib import Path
import uuid
import io
import re

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


class StructuredDataProcessor(BaseProcessor):
    """
    结构化数据处理器。

    专门处理JSON、CSV、XML、YAML等结构化数据格式。
    """

    def __init__(self, config):
        """
        初始化结构化数据处理器。

        Args:
            config: 解析器配置
        """
        super().__init__(config)
        self.supported_extensions = {
            '.json', '.csv', '.xml', '.yml', '.yaml', '.txt'
        }

    async def initialize(self) -> bool:
        """
        初始化处理器。

        Returns:
            bool: 初始化是否成功
        """
        return True  # 结构化数据处理器不需要额外依赖

    async def cleanup(self) -> None:
        """清理处理器资源。"""
        pass

    async def _parse_with_config(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        **kwargs
    ) -> ParseResult:
        """
        使用配置解析结构化数据。

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

        # 确定文件类型
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.json':
                return await self._parse_json_file(file_path, strategy, config)
            elif file_ext == '.csv':
                return await self._parse_csv_file(file_path, strategy, config)
            elif file_ext in {'.xml', '.xsl', '.xslt'}:
                return await self._parse_xml_file(file_path, strategy, config)
            elif file_ext in {'.yml', '.yaml'}:
                return await self._parse_yaml_file(file_path, strategy, config)
            elif file_ext == '.txt':
                return await self._parse_text_file(file_path, strategy, config)
            else:
                raise UnsupportedFormatError(
                    f"不支持的结构化数据格式: {file_ext}",
                    parser=self.parser_name,
                    file_path=file_path
                )
        except Exception as e:
            raise ParseException(
                f"结构化数据解析失败: {str(e)}",
                parser=self.parser_name,
                file_path=file_path
            )

    async def _parse_json_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析JSON文件。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 创建格式化的JSON文本
            json_text = json.dumps(data, indent=2, ensure_ascii=False)

            # 创建元数据
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            metadata = DocumentMetadata(
                file_name=file_name,
                file_size=file_size,
                file_type=DocumentType.JSON,
                mime_type="application/json",
                title=file_name,
                page_count=1,
                character_count=len(json_text),
                metadata={
                    "json_type": type(data).__name__,
                    "keys_count": self._count_json_keys(data),
                    "nested_levels": self._get_json_nesting_level(data)
                }
            )

            # 创建文本块
            text_chunks = []
            if config.text.chunk_size < len(json_text):
                # 按指定大小分块
                chunks = self._create_text_chunks(
                    json_text,
                    config.text.chunk_size,
                    config.text.chunk_overlap
                )
                text_chunks = chunks
            else:
                # 整个JSON作为一个块
                chunk = TextChunk(
                    content=json_text,
                    chunk_id="json_1",
                    confidence=1.0
                )
                text_chunks.append(chunk)

            # 如果是数组或对象，创建表格表示
            tables = []
            if isinstance(data, (list, dict)):
                tables = await self._json_to_tables(data, file_name)

            return ParseResult(
                success=True,
                metadata=metadata,
                full_text=json_text,
                text_chunks=text_chunks,
                tables=tables,
                structured_data=data
            )

        except json.JSONDecodeError as e:
            raise ParseException(f"JSON格式错误: {str(e)}", parser=self.parser_name, file_path=file_path)
        except Exception as e:
            raise ParseException(f"JSON解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _parse_csv_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析CSV文件。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 检测CSV编码
            encoding = await self._detect_file_encoding(file_path)

            # 读取CSV文件
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                # 检测分隔符
                sample = f.read(1024)
                f.seek(0)
                delimiter = self._detect_csv_delimiter(sample)

                reader = csv.reader(f, delimiter=delimiter)
                data = list(reader)

            if not data:
                raise ParseException("CSV文件为空", parser=self.parser_name, file_path=file_path)

            # 创建表格信息
            headers = data[0] if data else []
            rows = data[1:] if len(data) > 1 else []

            table_info = TableInfo(
                table_id="csv_table_1",
                page_number=1,
                bbox=[0, 0, 0, 0],
                rows=len(rows),
                columns=len(headers),
                headers=headers,
                data=data,
                confidence=0.95
            )

            # 创建CSV文本
            csv_text = '\n'.join([','.join(row) for row in data])

            # 创建元数据
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            metadata = DocumentMetadata(
                file_name=file_name,
                file_size=file_size,
                file_type=DocumentType.CSV,
                mime_type="text/csv",
                title=file_name,
                page_count=1,
                character_count=len(csv_text),
                has_tables=True,
                metadata={
                    "encoding": encoding,
                    "delimiter": delimiter,
                    "rows_count": len(rows),
                    "columns_count": len(headers)
                }
            )

            # 创建文本块
            text_chunks = []
            chunk = TextChunk(
                content=csv_text,
                chunk_id="csv_1",
                confidence=0.95
            )
            text_chunks.append(chunk)

            return ParseResult(
                success=True,
                metadata=metadata,
                full_text=csv_text,
                text_chunks=text_chunks,
                tables=[table_info],
                structured_data={
                    "headers": headers,
                    "rows": rows,
                    "encoding": encoding,
                    "delimiter": delimiter
                }
            )

        except Exception as e:
            raise ParseException(f"CSV解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _parse_xml_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析XML文件。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 检测XML编码
            encoding = await self._detect_file_encoding(file_path)

            # 解析XML
            with open(file_path, 'r', encoding=encoding) as f:
                tree = ET.parse(f)
                root = tree.getroot()

            # 转换为格式化XML文本
            ET.indent(tree, space="  ", level=0)
            xml_text = ET.tostring(root, encoding='unicode')

            # 创建元数据
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            metadata = DocumentMetadata(
                file_name=file_name,
                file_size=file_size,
                file_type=DocumentType.XML,
                mime_type="application/xml",
                title=file_name,
                page_count=1,
                character_count=len(xml_text),
                metadata={
                    "root_tag": root.tag,
                    "encoding": encoding,
                    "namespaces": self._get_xml_namespaces(root),
                    "elements_count": len(list(root.iter()))
                }
            )

            # 创建文本块
            text_chunks = []
            if config.text.chunk_size < len(xml_text):
                chunks = self._create_text_chunks(
                    xml_text,
                    config.text.chunk_size,
                    config.text.chunk_overlap
                )
                text_chunks = chunks
            else:
                chunk = TextChunk(
                    content=xml_text,
                    chunk_id="xml_1",
                    confidence=1.0
                )
                text_chunks.append(chunk)

            # 创建表格（XML中的重复结构）
            tables = await self._xml_to_tables(root, file_name)

            return ParseResult(
                success=True,
                metadata=metadata,
                full_text=xml_text,
                text_chunks=text_chunks,
                tables=tables,
                structured_data=self._xml_to_dict(root)
            )

        except ET.ParseError as e:
            raise ParseException(f"XML格式错误: {str(e)}", parser=self.parser_name, file_path=file_path)
        except Exception as e:
            raise ParseException(f"XML解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _parse_yaml_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析YAML文件。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 读取YAML文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 转换为格式化YAML文本
            yaml_text = yaml.dump(data, default_flow_style=False, allow_unicode=True)

            # 创建元数据
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            metadata = DocumentMetadata(
                file_name=file_name,
                file_size=file_size,
                file_type=DocumentType.JSON,  # YAML通常转换为JSON处理
                mime_type="application/x-yaml",
                title=file_name,
                page_count=1,
                character_count=len(yaml_text),
                metadata={
                    "yaml_type": type(data).__name__,
                    "keys_count": self._count_json_keys(data) if isinstance(data, dict) else 0
                }
            )

            # 创建文本块
            text_chunks = []
            if config.text.chunk_size < len(yaml_text):
                chunks = self._create_text_chunks(
                    yaml_text,
                    config.text.chunk_size,
                    config.text.chunk_overlap
                )
                text_chunks = chunks
            else:
                chunk = TextChunk(
                    content=yaml_text,
                    chunk_id="yaml_1",
                    confidence=1.0
                )
                text_chunks.append(chunk)

            return ParseResult(
                success=True,
                metadata=metadata,
                full_text=yaml_text,
                text_chunks=text_chunks,
                structured_data=data
            )

        except yaml.YAMLError as e:
            raise ParseException(f"YAML格式错误: {str(e)}", parser=self.parser_name, file_path=file_path)
        except Exception as e:
            raise ParseException(f"YAML解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _parse_text_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析纯文本文件（尝试检测结构化内容）。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试检测文本中的结构化内容
            structured_data = await self._detect_structure_in_text(content)

            # 创建元数据
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            metadata = DocumentMetadata(
                file_name=file_name,
                file_size=file_size,
                file_type=DocumentType.TXT,
                mime_type="text/plain",
                title=file_name,
                page_count=1,
                word_count=len(content.split()),
                character_count=len(content),
                metadata={
                    "detected_structure": bool(structured_data)
                }
            )

            # 创建文本块
            text_chunks = self._create_text_chunks(
                content,
                config.text.chunk_size,
                config.text.chunk_overlap
            )

            # 如果检测到结构化数据，尝试解析
            tables = []
            if structured_data:
                tables = await self._parse_structured_text(content, file_name)

            return ParseResult(
                success=True,
                metadata=metadata,
                full_text=content,
                text_chunks=text_chunks,
                tables=tables,
                structured_data=structured_data
            )

        except Exception as e:
            raise ParseException(f"文本解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    def _count_json_keys(self, obj: Any, count: int = 0) -> int:
        """递归计算JSON键的数量。"""
        if isinstance(obj, dict):
            count += len(obj)
            for value in obj.values():
                count = self._count_json_keys(value, count)
        elif isinstance(obj, list):
            for item in obj:
                count = self._count_json_keys(item, count)
        return count

    def _get_json_nesting_level(self, obj: Any, level: int = 0) -> int:
        """获取JSON嵌套层级。"""
        if isinstance(obj, (dict, list)) and obj:
            if isinstance(obj, dict):
                return max([self._get_json_nesting_level(v, level + 1) for v in obj.values()] + [level])
            else:
                return max([self._get_json_nesting_level(item, level + 1) for item in obj] + [level])
        return level

    async def _json_to_tables(self, data: Any, table_name: str) -> List[TableInfo]:
        """将JSON数据转换为表格。"""
        tables = []

        if isinstance(data, list) and data:
            # 数组作为表格
            if all(isinstance(item, dict) for item in data):
                # 对象数组
                headers = list(data[0].keys()) if data else []
                table_data = []
                for item in data:
                    row = [item.get(key, "") for key in headers]
                    table_data.append(row)

                table_info = TableInfo(
                    table_id=f"json_table_1",
                    page_number=1,
                    bbox=[0, 0, 0, 0],
                    rows=len(table_data),
                    columns=len(headers),
                    headers=headers,
                    data=table_data,
                    confidence=0.9
                )
                tables.append(table_info)

        elif isinstance(data, dict) and data:
            # 对象作为表格（键值对）
                keys = list(data.keys())
                values = [str(data[key]) for key in keys]

                table_info = TableInfo(
                    table_id=f"json_table_1",
                    page_number=1,
                    bbox=[0, 0, 0, 0],
                    rows=1,
                    columns=len(keys),
                    headers=keys,
                    data=[values],
                    confidence=0.8
                )
                tables.append(table_info)

        return tables

    def _get_xml_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """获取XML命名空间。"""
        namespaces = {}
        for key, value in root.attrib.items():
            if key.startswith('xmlns'):
                ns_name = key.split(':')[-1] if ':' in key else 'default'
                namespaces[ns_name] = value
        return namespaces

    def _xml_to_dict(self, element: ET.Element) -> Any:
        """将XML元素转换为字典。"""
        result = {}

        # 处理属性
        if element.attrib:
            result['@attributes'] = element.attrib

        # 处理子元素
        children = {}
        for child in element:
            if child.tag in children:
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(self._xml_to_dict(child))
            else:
                children[child.tag] = self._xml_to_dict(child)

        # 处理文本内容
        if element.text and element.text.strip():
            if children:
                result['#text'] = element.text.strip()
            else:
                return element.text.strip()

        result.update(children)
        return result

    async def _xml_to_tables(self, root: ET.Element, table_name: str) -> List[TableInfo]:
        """将XML重复结构转换为表格。"""
        tables = []

        # 查找重复的子元素
        child_tags = {}
        for child in root:
            tag = child.tag
            if tag in child_tags:
                child_tags[tag] += 1
            else:
                child_tags[tag] = 1

        # 对于有多个相同标签的子元素，创建表格
        for tag, count in child_tags.items():
            if count > 1 and count <= 100:  # 限制表格大小
                elements = root.findall(tag)
                if elements:
                    # 提取表头（从第一个元素）
                    first_element = elements[0]
                    headers = list(first_element.keys()) if hasattr(first_element, 'keys') else []

                    # 如果元素有属性，使用属性作为表头
                    if first_element.attrib:
                        headers = list(first_element.attrib.keys())

                    # 提取数据
                    table_data = []
                    for element in elements:
                        if headers and hasattr(element, 'attrib'):
                            row = [element.get(attr, "") for attr in headers]
                        elif hasattr(element, 'attrib'):
                            row = [element.get(attr, "") for attr in element.attrib.keys()]
                        elif element.text:
                            row = [element.text]
                        else:
                            row = [str(self._xml_to_dict(element))]
                        table_data.append(row)

                    if table_data and headers:
                        table_info = TableInfo(
                            table_id=f"xml_table_{tag}",
                            page_number=1,
                            bbox=[0, 0, 0, 0],
                            rows=len(table_data),
                            columns=len(headers),
                            headers=headers,
                            data=table_data,
                            confidence=0.8
                        )
                        tables.append(table_info)

        return tables

    async def _detect_file_encoding(self, file_path: str) -> str:
        """检测文件编码。"""
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin1']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # 读取前1000字符测试
                return encoding
            except UnicodeDecodeError:
                continue

        return 'utf-8'  # 默认编码

    def _detect_csv_delimiter(self, sample: str) -> str:
        """检测CSV分隔符。"""
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}

        for delimiter in delimiters:
            count = sample.count(delimiter)
            delimiter_counts[delimiter] = count

        return max(delimiter_counts, key=delimiter_counts.get) if delimiter_counts else ','

    async def _detect_structure_in_text(self, content: str) -> Any:
        """检测文本中的结构化内容。"""
        # 尝试解析为JSON
        try:
            return json.loads(content)
        except:
            pass

        # 尝试解析为YAML
        try:
            return yaml.safe_load(content)
        except:
            pass

        # 检测是否是键值对格式
        if self._is_key_value_format(content):
            return self._parse_key_value_text(content)

        return None

    def _is_key_value_format(self, content: str) -> bool:
        """检查是否为键值对格式。"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False

        key_value_count = 0
        for line in lines[:10]:  # 检查前10行
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_value_count += 1

        return key_value_count / len(lines) > 0.5

    def _parse_key_value_text(self, content: str) -> Dict[str, str]:
        """解析键值对格式的文本。"""
        data = {}
        for line in content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
        return data

    async def _parse_structured_text(self, content: str, file_name: str) -> List[TableInfo]:
        """解析结构化文本中的表格。"""
        tables = []

        lines = content.strip().split('\n')
        if len(lines) < 2:
            return tables

        # 检测是否有表格结构（一致的列数）
        column_counts = [len(line.split(',')) for line in lines[:5]]
        if len(set(column_counts)) == 1 and column_counts[0] > 1:
            # 可能是CSV格式的表格
            headers = lines[0].split(',')
            data_rows = [line.split(',') for line in lines[1:] if line.strip()]

            table_info = TableInfo(
                table_id="detected_table_1",
                page_number=1,
                bbox=[0, 0, 0, 0],
                rows=len(data_rows),
                columns=len(headers),
                headers=headers,
                data=[headers] + data_rows,  # 包含表头
                confidence=0.7
            )
            tables.append(table_info)

        return tables


# 导出
__all__ = ['StructuredDataProcessor']