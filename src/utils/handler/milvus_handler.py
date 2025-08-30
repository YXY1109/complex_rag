import time
from typing import List

from langchain_core.documents import Document
from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from pymilvus.orm import db, utility

from src.config.config import settings
from src.utils.common import chinese_to_pinyin, has_chinese
from src.utils.handler.bce_embedding import bce_model_encode


def init_milvus(collection_name: str, partition_name: str, focus_delete=False):
    """
    milvus初始化，创建向量数据库
    :param collection_name: 集合名称，用户区分
    :param partition_name: 分区名称，用户有多个知识库
    :param focus_delete:
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

        if focus_delete:
            utility.drop_collection(collection_name)
            logger.warning(f"强制删除Milvus集合：{collection_name}成功")

        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            logger.info(f"Collection {collection_name} exists")
        else:
            schema = CollectionSchema(fields, description="rag索引库")
            collection = Collection(collection_name, schema)
            logger.info(f"创建Milvus集合：{collection_name}成功")

            """
            metric_type的设置，距离计算：https://milvus.io/docs/zh/metric.md
            L2（欧氏距离）：图像像素比较、传感器数据、地理坐标等需要“绝对距离”的场景。
            IP（内积）：适合已归一化的特征向量匹配、神经网络输出
            COSINE（余弦相似度）：文本语义匹配、图像特征检索、推荐系统
            
            若向量已归一化，IP 和 COSINE 效果一致
            """

            """
            内存索引
            index_type的设置，向量索引：https://milvus.io/docs/zh/index.md
            
            索引类型	分类	核心特点	典型场景
            FLAT	无（穷举）	100% 召回率，无压缩，速度慢，适合小数据集（百万级）。	精准检索需求（如小规模测试数据集）。
            IVF_FLAT	基于量化	聚类分桶加速，通过nprobe平衡速度与召回率，内存占用高。	中等规模数据集（千万级），需高速查询和较高召回率（如推荐系统候选集生成）。
            IVF_SQ8	基于量化	标量量化压缩（FLOAT→UINT8），内存占用减少 70-75%，召回率略降。	内存资源有限的场景（如边缘计算设备）。
            IVF_PQ	基于量化	乘积量化压缩，索引文件更小，精度损失比 IVF_SQ8 大。	超大规模数据集（亿级），对内存敏感且可接受较低召回率（如日志分析）。
            HNSW	基于图	分层图结构，查询速度极快，内存占用高，需提前训练索引。	实时高并发场景（如聊天 APP 语义搜索）。
            SCANN	基于量化	利用 SIMD 优化计算，速度优于 IVF_PQ，支持原始数据嵌入索引。	大规模数据集且追求极致查询速度（如广告推荐实时检索）。
            """

            """
            建立索引，nlist的设置：https://milvus.io/docs/zh/ivf-flat.md
            nlist 值越大，通过创建更精细的簇来提高召回率，但会增加索引构建时间。请根据数据集大小和可用资源进行优化。大多数情况下，我们建议在此范围内设置值：[32, 4096].
            """
            create_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 2048}}
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
    # file_path = file_dict["file_path"]
    timestamp = time.strftime("%Y%m%d", time.localtime())
    documents: List[Document] = file_dict["documents"]
    # milvus_len = len(documents)

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
    return mr.primary_keys


def search_milvus(collection_name: str, partition_name: str, query_text: str, top_k: int = 3):
    """
    milvus搜索
    :param collection_name: 集合名称，用户区分
    :param partition_name: 分区名称，用户有多个知识库
    :param query_text: 搜索文本
    :param top_k: 搜索结果数量
    :return
    """
    partition_name = chinese_to_pinyin(partition_name) if has_chinese(partition_name) else partition_name
    collection = Collection(collection_name)

    """
    nprobe的设置：https://milvus.io/docs/zh/ivf-flat.md
    增加该值可提高召回率，但可能会减慢搜索速度。设置nprobe 与nlist 成比例，以平衡速度和准确性。
    在大多数情况下，我们建议您在此范围内设置一个值：[1，nlist]。
    """
    query_result = collection.search(
        [query_text],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        expr=f"file_name=='{p_name1}'",
        partition_names=[partition_name],
        output_fields=["file_name", "file_url", "file_path", "content", "timestamp"],
    )
    pass


if __name__ == "__main__":
    collection_name1 = "knowledge_name_1"
    p_name1 = "民法典"  # 知识库id

    # 初始化
    init_milvus(collection_name1, p_name1)
    # 插入数据
    insert_to_milvus(collection_name1, p_name1)
