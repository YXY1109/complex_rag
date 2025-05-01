import time
from typing import List

from langchain_core.documents import Document
from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from pymilvus.orm import db, utility

from src.config.config import settings
from src.utils.common import chinese_to_pinyin, has_chinese
from src.utils.handler.bce_embedding import bce_model_encode


def init_milvus(collection_name: str, partition_name: str):
    """
    milvus初始化，创建向量数据库
    :param collection_name: 集合名称，用户区分
    :param partition_name: 分区名称，用户有多个知识库
    """

    partition_name = chinese_to_pinyin(partition_name) if has_chinese(partition_name) else partition_name

    try:
        database_name = settings.MILVUS_DATABASE_NAME
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, max_length=64, is_primary=True, auto_id=True),
            FieldSchema(name="file_id", dtype=DataType.INT64),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=640),
            FieldSchema(name="file_url", dtype=DataType.VARCHAR, max_length=640),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=640),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        connections.connect(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
            db_name=database_name,
        )
        logger.info("Milvus连接成功")
        if database_name not in db.list_database():
            db.create_database(database_name)
        db.using_database(database_name)

        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            logger.info(f"Collection {collection_name} exists")
        else:
            schema = CollectionSchema(fields, description="rag索引库")
            collection = Collection(collection_name, schema)
            logger.info(f"创建Milvus集合：{collection_name}成功")

            create_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 2048}}
            collection.create_index(field_name="embedding", index_params=create_params)
            logger.info("创建Milvus索引成功")
        # 创建分区
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name, description=partition_name)
            logger.info(f"Milvus分区partitions: {partition_name}，创建成功！")
        collection.load()
        logger.success("Milvus初始化完成！！！")
    except Exception as e:
        logger.error(f"创建Milvus集合：{collection_name}失败: {e}")


def insert_to_milvus(collection_name: str, partition_name: str, file_dict: dict):
    """
    milvus初始化，创建向量数据库
    :param collection_name: 集合名称，用户区分
    :param partition_name: 分区名称，用户有多个知识库
    :param file_dict: 待插入的数据
    """

    partition_name = chinese_to_pinyin(partition_name) if has_chinese(partition_name) else partition_name

    start = time.perf_counter()

    file_id = file_dict["file_id"]
    file_name = file_dict["file_name"]
    file_url = file_dict["file_url"]
    file_path = file_dict["file_path"]
    timestamp = time.strftime("%Y%m%d", time.localtime())
    documents: List[Document] = file_dict["documents"]
    milvus_len = len(documents)

    # file_id_list = [file_id] * milvus_len
    # file_name_list = [file_name] * milvus_len
    # file_url_list = [file_url] * milvus_len
    content_list = [doc.page_content for doc in documents]
    #
    # timestamp_list = [timestamp] * milvus_len
    embedding_list = bce_model_encode(content_list)

    insert_data = []
    # 将以上转为数组中是字典的形式
    for content, embedding in zip(content_list, embedding_list):
        new_dict = {
            "file_id": file_id,
            "file_name": file_name,
            "file_url": file_url,
            "file_path": file_url,
            "content": content,
            "timestamp": timestamp,
            "embedding": embedding,
        }
        insert_data.append(new_dict)

    # 往向量库中插入数据
    collection = Collection(collection_name)
    # insert_data = [
    #     file_id_list,
    #     file_name_list,
    #     file_url_list,
    #     content_list,
    #     timestamp_list,
    #     embedding_list,
    # ]
    mr = collection.insert(insert_data, partition_name)
    print(f"插入数据的结果：{mr}")
    # 插入的数据存储在内存，需要传输到磁盘
    collection.flush()
    print("插入完成，向量插入共耗时约 {:.2f} 秒".format(time.perf_counter() - start))


if __name__ == "__main__":
    collection_name1 = "knowledge_name_1"
    p_name1 = "民法典"  # 知识库id

    # 初始化
    init_milvus(collection_name1, p_name1)
    # 插入数据
    insert_to_milvus(collection_name1, p_name1)
