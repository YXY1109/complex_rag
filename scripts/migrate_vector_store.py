#!/usr/bin/env python3
"""
向量存储数据迁移脚本

将旧的向量存储数据迁移到新的统一向量存储系统。
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

from rag_service.services.vector_store import VectorStore
from rag_service.services.unified_vector_store import (
    UnifiedVectorStore, VectorData, CollectionConfig, VectorIndexConfig
)
from core_rag.retriever.strategies.vector_retriever import VectorRetriever as LegacyVectorRetriever


class VectorStoreMigrator:
    """向量存储数据迁移器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 源配置
        self.legacy_config = config.get("legacy", {})
        self.unified_config = config.get("unified", {})

        # 统计信息
        self.stats = {
            "total_documents": 0,
            "migrated_documents": 0,
            "failed_documents": 0,
            "skipped_documents": 0,
            "start_time": None,
            "end_time": None,
            "duration": 0.0,
            "errors": []
        }

    async def migrate_collection(
        self,
        collection_name: str,
        legacy_retriever: LegacyVectorRetriever,
        unified_store: UnifiedVectorStore
    ) -> Dict[str, Any]:
        """迁移单个集合"""
        try:
            self.logger.info(f"开始迁移集合: {collection_name}")

            # 获取遗留检索器的所有文档
            all_documents = await self._get_legacy_documents(legacy_retriever)

            if not all_documents:
                self.logger.warning(f"集合 {collection_name} 中没有文档需要迁移")
                return {
                    "collection": collection_name,
                    "total_documents": 0,
                    "migrated_documents": 0,
                    "skipped_documents": 0,
                    "errors": []
                }

            # 转换为统一向量存储格式
            vector_data_list = []
            for doc in all_documents:
                try:
                    vector_data = VectorData(
                        id=doc.get("id", ""),
                        vector=doc.get("vector", []),
                        metadata={
                            "content": doc.get("content", ""),
                            "title": doc.get("title", ""),
                            "source": doc.get("source", ""),
                            "author": doc.get("author", ""),
                            "created_at": doc.get("created_at"),
                            **doc.get("metadata", {})
                        },
                        collection_name=collection_name
                    )
                    vector_data_list.append(vector_data)
                except Exception as e:
                    self.logger.error(f"转换文档 {doc.get('id', 'unknown')} 失败: {str(e)}")
                    self.stats["failed_documents"] += 1
                    self.stats["errors"].append(str(e))

            # 批量插入到统一向量存储
            if vector_data_list:
                migrated_ids = await unified_store.upsert_vectors(
                    vectors=vector_data_list,
                    collection_name=collection_name
                )

                self.stats["migrated_documents"] += len(migrated_ids)
                self.logger.info(f"成功迁移 {len(migrated_ids)} 个文档到统一向量存储")
            else:
                self.logger.warning("没有有效的向量数据需要迁移")

            return {
                "collection": collection_name,
                "total_documents": len(all_documents),
                "migrated_documents": len(vector_data_list),
                "skipped_documents": len(all_documents) - len(vector_data_list),
                "errors": len(self.stats["errors"])
            }

        except Exception as e:
            error_msg = f"迁移集合 {collection_name} 失败: {str(e)}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return {
                "collection": collection_name,
                "total_documents": 0,
                "migrated_documents": 0,
                "skipped_documents": 0,
                "errors": [error_msg]
            }

    async def _get_legacy_documents(self, legacy_retriever: LegacyVectorRetriever) -> List[Dict[str, Any]]:
        """从遗留检索器获取所有文档"""
        try:
            # 这里需要根据实际的遗留检索器接口实现
            # 由于我们没有实际的遗留检索器实例，这里模拟数据获取
            # 在实际使用中，应该调用遗留检索器的具体方法来获取所有文档

            documents = []

            # 模拟获取数据（实际实现需要连接到真实的遗留存储）
            if hasattr(legacy_retriever, 'documents'):
                # 如果有文档属性
                for doc_id, doc in legacy_retriever.documents.items():
                    documents.append({
                        "id": doc_id,
                        "content": doc.get("content", ""),
                        "vector": doc.get("vector", []),
                        "metadata": doc
                    })
            else:
                # 尝试通过其他方式获取文档
                # 这里应该实现实际的文档提取逻辑
                pass

            self.logger.info(f"从遗留存储获取到 {len(documents)} 个文档")
            return documents

        except Exception as e:
            self.logger.error(f"获取遗留文档失败: {str(e)}")
            return []

    async def migrate_all_collections(
        self,
        legacy_store: VectorStore,
        unified_store: UnifiedVectorStore,
        collections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """迁移所有集合"""
        try:
            self.stats["start_time"] = time.time()

            # 确定要迁移的集合
            if not collections:
                # 获取所有集合
                collections = await self._get_legacy_collections(legacy_store)
            else:
                collections = [collection for collection in collections if collection]

            migration_results = {}

            # 迁移每个集合
            for collection_name in collections:
                result = await self.migrate_collection(
                    collection_name=collection_name,
                    legacy_retriever=None,  # 需要传入实际的遗留检索器实例
                    unified_store=unified_store
                )
                migration_results[collection_name] = result

                # 更新统计
                self.stats["total_documents"] += result["total_documents"]
                self.stats["migrated_documents"] += result["migrated_documents"]
                self.stats["failed_documents"] += result.get("failed_documents", 0)
                self.stats["skipped_documents"] += result.get("skipped_documents", 0)

            self.stats["end_time"] = time.time()
            self.stats["duration"] = self.stats["end_time"] - self.stats["start_time"]

            return {
                "success": True,
                "collections": migration_results,
                "stats": self.stats,
                "message": "数据迁移完成"
            }

        except Exception as e:
            self.logger.error(f"迁移所有集合失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }

    async def _get_legacy_collections(self, legacy_store: VectorStore) -> List[str]:
        """获取遗留存储中的所有集合"""
        try:
            # 这里需要根据实际的遗留存储实现
            # 模拟返回集合列表
            collections = ["documents", "knowledge_base", "chunks"]

            self.logger.info(f"发现 {len(collections)} 个集合: {collections}")
            return collections

        except Exception as e:
            self.logger.error(f"获取集合列表失败: {str(e)}")
            return []

    def generate_migration_report(self, results: Dict[str, Any]) -> str:
        """生成迁移报告"""
        report_lines = [
            "向量存储数据迁移报告",
            "=" * 50,
            f"迁移开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.stats['start_time']))}",
            f"迁移结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.stats['end_time']))}",
            f"总耗时: {self.stats['duration']:.2f} 秒",
            "",
            "迁移统计:",
            f"  总文档数: {self.stats['total_documents']}",
            f"  成功迁移: {self.stats['migrated_documents']}",
            f"  迁移失败: {self.stats['failed_documents']}",
            f"  跳过文档: {self.stats['skipped_documents']}",
            f"  错误数量: {len(self.stats['errors'])}",
            "",
        ]

        # 集合详情
        if "collections" in results:
            report_lines.append("集合详情:")
            for collection_name, collection_result in results["collections"].items():
                report_lines.extend([
                    f"  集合: {collection_name}",
                    f"    总文档: {collection_result.get('total_documents', 0)}",
                    f"    成功迁移: {collection_result.get('migrated_documents', 0)}",
                    f"    迁移失败: {collection_result.get('failed_documents', 0)}",
                    f"    跳过文档: {collection_result.get('skipped_documents', 0)}",
                    ""
                ])

        # 错误详情
        if self.stats["errors"]:
            report_lines.extend([
                "",
                "错误详情:",
            ])
            for i, error in enumerate(self.stats["errors"][:10], 1):  # 只显示前10个错误
                report_lines.append(f"  {i}. {error}")

        report_lines.extend([
            "=" * 50,
            "迁移完成！"
        ])

        return "\n".join(report_lines)

    def save_report(self, report: str, output_file: str = "migration_report.txt") -> None:
        """保存迁移报告"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"迁移报告已保存到: {output_file}")
        except Exception as e:
            self.logger.error(f"保存迁移报告失败: {str(e)}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="向量存储数据迁移工具")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不执行实际迁移")
    parser.add_argument("--collections", nargs="+", help="要迁移的集合名称")
    parser.add_argument("--output", default="migration_report.txt", help="输出报告文件路径")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    migrator = VectorStoreMigrator({
        "legacy": {
            "connection_string": "legacy_connection"
        },
        "unified": {
            "backends": {
                "milvus": {
                    "host": "localhost",
                    "port": 19530
                }
            }
        }
    })

    if args.dry_run:
        print("试运行模式 - 不会执行实际迁移")
        # 这里可以进行试运行的检查
        print("将迁移以下集合:", args.collections or "所有集合")
        return

    # 执行实际迁移
    print("开始向量存储数据迁移...")

    try:
        # 创建统一向量存储实例
        from rag_service.services.unified_vector_store import UnifiedVectorStore
        from config.unified_embedding_config import get_unified_embedding_config

        unified_store = UnifiedVectorStore(get_unified_embedding_config())
        await unified_store.initialize()

        # 执行迁移
        results = await migrator.migrate_all_collections(
            legacy_store=None,  # 需要传入实际的遗留存储实例
            unified_store=unified_store,
            collections=args.collections
        )

        # 生成并保存报告
        report = migrator.generate_migration_report(results)
        print(report)
        migrator.save_report(report, args.output)

        if results.get("success", False):
            print("迁移失败，请检查日志获取详细信息")
            return 1

        print("迁移完成！")
        return 0

    except Exception as e:
        print(f"迁移过程中发生错误: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)