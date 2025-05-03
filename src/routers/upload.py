import os
import re
from typing import List
import urllib.parse

import aiofiles
from fastapi import APIRouter, Depends, UploadFile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sqlalchemy.orm import Session

from src.model.sqlalchemy_m.model import File
from src.utils.common import chinese_to_pinyin, get_collection_name, get_now_time, has_chinese, truncate_filename
from src.utils.handler.milvus_handler import init_milvus, insert_to_milvus
from src.utils.handler.minio_handler import upload_to_minio
from src.utils.handler.mysql_handler import get_session

router = APIRouter(
    prefix="/upload",
    tags=["文件上传"],
)


@router.post("/upload_files", summary="上传文件")
async def upload_files(
    files: List[UploadFile], db: Session = Depends(get_session), user_id: int = 1, partition_name: str = "minfadian"
):
    logger.info(f"开始上传文件：{get_now_time()}")
    knowledge_name = get_collection_name(user_id)
    logger.info(f"集合名称：{knowledge_name}")
    partition_name = partition_name.strip()  # 去除前后空格
    partition_name = chinese_to_pinyin(partition_name) if has_chinese(partition_name) else partition_name
    logger.info(f"分区名：{partition_name}")

    # 保存文件名称的列表
    file_names_list = []
    for file in files:
        original_filename = file.filename
        logger.info(f"原始文件名称: {original_filename}")
        file_name = urllib.parse.unquote(original_filename, encoding="UTF-8")
        logger.info(f"编码后的名称：{file_name}")
        # 删除掉全角字符
        file_name = re.sub(r"[\uFF01-\uFF5E\u3000-\u303F]", "", file_name)
        file_name = file_name.replace("/", "_")
        logger.info(f"清理后的名称: {file_name}")
        file_name = truncate_filename(file_name)
        file_names_list.append(file_name)

    # 本地文件临时路径
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    upload_dir = os.path.join(project_dir, "static", "upload", knowledge_name, partition_name)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # 保存文件到本地
    file_paths_list = []
    for file, file_name in zip(files, file_names_list):
        save_file_path = os.path.join(str(upload_dir), file_name)
        # 文件上传
        try:
            # 异步：https://github.com/Tinche/aiofiles
            default_chunk_size = 1024 * 1024 * 50  # 50M
            async with aiofiles.open(save_file_path, "wb") as f:
                while chunk := await file.read(default_chunk_size):
                    await f.write(chunk)
            file_paths_list.append(save_file_path)
        except Exception as e:
            logger.error(f"{file_name}，上传异常：{e}")
            continue
        finally:
            file.file.close()

    logger.info(f"上传成功后的文件列表：{file_paths_list}")
    # 上传文件到minio
    file_url_list = upload_to_minio(user_id, file_paths_list)

    file_info_list = []
    # 文件信息存入mysql
    for file_name, file_url, file_path in zip(file_names_list, file_url_list, file_paths_list):
        upload_file = File(file_name=file_name, minio_path=file_url, knowledge_id=knowledge_name)
        db.add(upload_file)

        db.flush()
        file_id = upload_file.file_id
        logger.info(f"文件id：{file_id}")
        file_info_list.append(
            {"file_id": file_id, "file_name": file_name, "file_url": file_url, "file_path": file_path}
        )
    db.commit()
    logger.success("文件上传，保存mysql成功！")

    for file_dict in file_info_list:
        file_path = file_dict["file_path"]

        # todo 文档解析，先使用txt
        loder = TextLoader(file_path, encoding="utf-8")
        documents = loder.load()

        # todo 文档分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10, length_function=len)
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"split_docs:{split_docs}")
        file_dict["documents"] = split_docs

        # todo 文件的文本写入到milvus
        init_milvus(knowledge_name, partition_name)
        insert_to_milvus(knowledge_name, partition_name, file_dict)
        logger.info("文件写入milvus成功！")

        # todo 文件的文本写入elasticsearch

    return {"msg": "上传成功"}
