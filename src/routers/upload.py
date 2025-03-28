import os
import re
from typing import List
import urllib.parse

from fastapi import APIRouter, Depends, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from src.utils.handler.mysql_handler import get_session

router = APIRouter(
    prefix="/upload",
    tags=["文件上传"],
)


@router.post("/upload_files", summary="上传文件")
async def upload_files(files: List[UploadFile], user_id: int = 1, db: Session = Depends(get_session)):
    logger.info(f"开始上传文件。{get_now_time()}")
    c_id = get_collection_from_user_id(user_id)
    logger.info(f"集合id：{c_id}")
    p_id = get_partition_from_p_id(user_id, db)
    logger.info(f"分区id：{p_id}")

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
    upload_dir = os.path.join(project_dir, "static", "upload", c_id, p_id)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # 保存文件到本地
    file_paths_list = []
    for file, file_name in zip(files, file_names_list):
        # todo 生成file_id存入mysql
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
    # 文件信息存入mysql
    for file_name, file_url in zip(file_names_list, file_url_list):
        upload_file = UploadMinioFile(file_name=file_name, minio_path=file_url, knowledge_id=p_id)
        db.add(upload_file)
    db.commit()
    logger.success("文件上传，保存mysql成功！")

    # todo 文件的文本写入到milvus
    # todo 文件的文本写入elasticsearch

    return {"msg": "上传成功"}
