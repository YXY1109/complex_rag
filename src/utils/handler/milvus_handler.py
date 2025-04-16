import time

from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from pymilvus.orm import db, utility

from src.config.config import settings
from src.utils.common import chinese_to_pinyin, has_chinese


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
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=640),
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


def insert_to_milvus(self):
    start = time.perf_counter()

    chunk_id_list = ["3", "4"]
    file_id_list = ["file1", "file2"]
    file_name_list = ["yxy1", "yxy2"]
    file_path_list = ["/yxy/yxy1.txt", "/yxy/yxy2.txt"]
    content_list = ["你是谁", "你再干什么"]
    timestamp_list = ["20240906", "20240907"]
    embedding_list = bce_model_encode(content_list)

    # 往向量库中插入数据
    collection = Collection(self.collection_name)
    insert_data = [
        chunk_id_list,
        file_id_list,
        file_name_list,
        file_path_list,
        content_list,
        timestamp_list,
        embedding_list,
    ]
    mr = collection.insert(insert_data, self.partition_name)
    print(f"插入数据的结果：{mr}")
    # 插入的数据存储在内存，需要传输到磁盘
    collection.flush()
    print("插入完成，向量插入共耗时约 {:.2f} 秒".format(time.perf_counter() - start))


if __name__ == "__main__":
    user_id1 = 1  # 用户id
    p_name1 = "民法典"  # 知识库id

    # 初始化
    init_milvus("yxy_1", p_name1)
    # 插入数据
    # insert_files_to_milvus(c_id1, p_id1)
