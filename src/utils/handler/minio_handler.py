from mimetypes import guess_type
import os

from minio import Minio

from src.config.config import settings
from src.utils.logger import logger


def upload_to_minio(user_id: int, source_file_list: list):
    """
    上传文件到Minio
    :param user_id: 用户id
    :param source_file_list: 用户上传的文件
    :return:
    """
    logger.info("Minio连接配置信息")

    host = settings.MINIO_HOST
    port = settings.MINIO_PORT

    client = Minio(
        endpoint=f"{host}:{port}",
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )
    logger.info(f"Minio客户端创建成功；{client}")

    bucket_name = f"complex-rag-{user_id}"
    bucket_exists = client.bucket_exists(bucket_name)
    if not bucket_exists:
        # 创建桶的名称有规则要求
        client.make_bucket(bucket_name=bucket_name)
        logger.info(f"创建桶：{bucket_name}，成功！")

    file_url_list = []
    for source_file in source_file_list:
        # 获取文件名称
        destination_file = os.path.basename(source_file)
        # 尝试猜测文件类型
        content_type, _ = guess_type(source_file)

        # 上传文件，并指定内容类型
        with open(source_file, "rb") as file_data:
            file_stat = os.stat(source_file)
            client.put_object(bucket_name, destination_file, file_data, file_stat.st_size)
        logger.success(f"File {destination_file} uploaded with content-type {content_type}.")

        file_url = client.presigned_get_object(bucket_name, destination_file)
        file_url = file_url.replace(" ", "%20")
        logger.info(f"文件下载地址：{file_url}")
        # file_url2 = f"http://{host}:{port}/{bucket_name}/{quote(destination_file)}"
        # logger.info(f"文件下载地址2：{file_url2}")
        file_url_list.append(f"{bucket_name}/{destination_file}")
    return file_url_list


if __name__ == "__main__":
    upload_to_minio(1, ["/Users/cj/Downloads/test.doc"])
