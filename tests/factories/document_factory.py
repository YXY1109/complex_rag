"""
文档数据工厂
生成测试用的文档数据
"""
from factory import Factory, Faker, SubFactory, SelfAttribute, lazy_attribute
from datetime import datetime
from typing import Dict, Any, List
import uuid


class DocumentDataFactory(Factory):
    """文档数据工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    title = Faker("sentence", nb_words=4)
    content = Faker("text", max_nb_chars=2000)
    file_name = Faker("file_name", extension="txt")
    file_type = "text/plain"
    file_size = Faker("random_int", min=1024, max=10485760)  # 1KB - 10MB
    created_at = Faker("date_time_this_year")
    updated_at = Faker("date_time_this_month")
    uploaded_by = Faker("uuid4")
    knowledge_base_id = Faker("uuid4")

    @lazy_attribute
    def metadata(self):
        """生成文档元数据"""
        return {
            "language": Faker("random_element", elements=["zh-CN", "en-US", "ja-JP"]),
            "encoding": "utf-8",
            "page_count": Faker("random_int", min=1, max=100),
            "word_count": Faker("random_int", min=100, max=50000),
            "character_count": Faker("random_int", min=500, max=200000),
            "line_count": Faker("random_int", min=10, max=5000),
            "paragraph_count": Faker("random_int", min=5, max=1000),
            "checksum": f"sha256:{Faker('sha256')}",
            "original_filename": self.file_name,
            "mime_type": self.file_type,
            "file_extension": self.file_name.split('.')[-1] if '.' in self.file_name else ''
        }

    @lazy_attribute
    def tags(self):
        """生成文档标签"""
        return Faker("words", nb=3)

    @lazy_attribute
    def status(self):
        """生成文档状态"""
        return Faker("random_element", elements=["processing", "completed", "failed", "pending"])

    @lazy_attribute
    def processing_status(self):
        """生成处理状态"""
        return {
            "extracted": True,
            "indexed": Faker("boolean"),
            "embeddings_generated": Faker("boolean"),
            "error_count": Faker("random_int", min=0, max=5),
            "last_processed": Faker("date_time_this_month"),
            "processing_time": Faker("random_int", min=1, max=300)  # 秒
        }


class PDFDocumentFactory(DocumentDataFactory):
    """PDF文档工厂"""

    file_name = Faker("file_name", extension="pdf")
    file_type = "application/pdf"

    @lazy_attribute
    def content(self):
        """PDF文档内容"""
        return f"""
        PDF文档标题: {self.title}

        第一章：引言
        {Faker("paragraph", nb_sentences=5)}

        第二章：主要内容
        {Faker("paragraph", nb_sentences=10)}

        第三章：结论
        {Faker("paragraph", nb_sentences=3)}
        """

    @lazy_attribute
    def metadata(self):
        """PDF文档元数据"""
        metadata = super().metadata
        metadata.update({
            "page_count": Faker("random_int", min=1, max=100),
            "has_images": Faker("boolean"),
            "has_tables": Faker("boolean"),
            "has_bookmarks": Faker("boolean"),
            "pdf_version": "1.7",
            "creator": Faker("company"),
            "producer": Faker("company")
        })
        return metadata


class WordDocumentFactory(DocumentDataFactory):
    """Word文档工厂"""

    file_name = Faker("file_name", extension="docx")
    file_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    @lazy_attribute
    def content(self):
        """Word文档内容"""
        return f"""
        文档标题: {self.title}

        摘要: {Faker("paragraph", nb_sentences=3)}

        1. 背景介绍
        {Faker("paragraph", nb_sentences=8)}

        2. 方法论
        {Faker("paragraph", nb_sentences=6)}

        3. 结果分析
        {Faker("paragraph", nb_sentences=7)}

        4. 结论与建议
        {Faker("paragraph", nb_sentences=4)}
        """


class StructuredDataDocumentFactory(DocumentDataFactory):
    """结构化数据文档工厂"""

    file_name = Faker("file_name", extension="json")
    file_type = "application/json"

    @lazy_attribute
    def content(self):
        """JSON内容"""
        import json
        data = {
            "title": self.title,
            "description": Faker("paragraph", nb_sentences=3),
            "metadata": {
                "version": "1.0",
                "created_by": Faker("name"),
                "category": Faker("word"),
                "tags": self.tags
            },
            "data": {
                "records": [
                    {
                        "id": i,
                        "name": Faker("name"),
                        "value": Faker("random_int", min=1, max=100),
                        "description": Faker("sentence")
                    } for i in range(Faker("random_int", min=5, max=20))
                ],
                "summary": {
                    "total_records": 0,
                    "average_value": 0.0,
                    "categories": []
                }
            }
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


class DocumentChunkFactory(Factory):
    """文档块工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    document_id = SubFactory(DocumentDataFactory).id
    chunk_index = Faker("random_int", min=0, max=1000)
    content = Faker("text", max_nb_chars=500)
    start_char = Faker("random_int", min=0, max=10000)
    end_char = Faker("random_int", min=100, max=15000)
    token_count = Faker("random_int", min=50, max=500)

    @lazy_attribute
    def metadata(self):
        """块元数据"""
        return {
            "page_number": Faker("random_int", min=1, max=50),
            "paragraph_index": Faker("random_int", min=0, max=100),
            "section_title": Faker("sentence", nb_words=3),
            "is_heading": Faker("boolean"),
            "is_table": Faker("boolean"),
            "is_list": Faker("boolean"),
            "chunk_type": Faker("random_element", elements=["text", "heading", "table", "list", "code"])
        }


class DocumentEmbeddingFactory(Factory):
    """文档嵌入向量工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    document_id = SubFactory(DocumentDataFactory).id
    chunk_id = SubFactory(DocumentChunkFactory).id
    embedding_model = "text-embedding-ada-002"
    dimension = 1536
    created_at = Faker("date_time_this_month")

    @lazy_attribute
    def embedding(self):
        """生成嵌入向量"""
        import random
        # 生成1536维的向量
        return [round(random.uniform(-1.0, 1.0), 6) for _ in range(self.dimension)]

    @lazy_attribute
    def metadata(self):
        """嵌入向量元数据"""
        return {
            "model_version": "1.0",
            "batch_id": Faker("uuid4"),
            "processing_time": Faker("random_int", min=10, max=1000),  # 毫秒
            "quality_score": round(Faker("random.uniform", min=0.7, max=1.0), 3)
        }


# 便捷函数
def create_test_document(**overrides) -> Dict[str, Any]:
    """创建测试文档"""
    return DocumentDataFactory(**overrides)


def create_test_pdf_document(**overrides) -> Dict[str, Any]:
    """创建测试PDF文档"""
    return PDFDocumentFactory(**overrides)


def create_test_word_document(**overrides) -> Dict[str, Any]:
    """创建测试Word文档"""
    return WordDocumentFactory(**overrides)


def create_test_document_chunks(document_id: str, count: int = 5, **overrides) -> List[Dict[str, Any]]:
    """创建文档块"""
    chunks = []
    for i in range(count):
        chunk = DocumentChunkFactory(
            document_id=document_id,
            chunk_index=i,
            **overrides
        )
        chunks.append(chunk)
    return chunks


def create_test_document_embedding(document_id: str, chunk_id: str, **overrides) -> Dict[str, Any]:
    """创建文档嵌入向量"""
    return DocumentEmbeddingFactory(
        document_id=document_id,
        chunk_id=chunk_id,
        **overrides
    )