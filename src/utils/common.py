from datetime import datetime, timedelta
import functools
import os
import time
import uuid

from orjson import orjson
from pydantic import typing
from pypinyin import Style, pinyin
from starlette.responses import JSONResponse


class ORJSONResponse(JSONResponse):
    """
    解决fastapi：ValueError: Out of range float values are not JSON compliant
    """

    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content)


def timer(func):
    """
    装饰器，函数执行的时间计时器
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f"共耗时约 {time.perf_counter() - start:.2f} 秒")
        return res

    return wrapper


def get_now_time(format_str="%Y-%m-%d %H:%M:%S"):
    """
    获取当前时间
    :return:
    """
    return datetime.now().strftime(format_str)


def run_seconds(seconds=2):
    """
    延迟N秒执行
    :return:
    """
    return datetime.now() + timedelta(seconds=seconds)


def truncate_filename(filename: str, max_length=50):
    """
    截取文件名称，最长50
    :param filename:
    :param max_length:
    :return:
    """
    # 获取文件名后缀
    file_ext = os.path.splitext(filename)[1]
    # 获取不带后缀的文件名
    file_name_no_ext = os.path.splitext(filename)[0]
    # 计算文件名长度，注意中文字符
    filename_length = len(filename)
    # 当文件名长度超过最大长度限制时，进行截断
    if filename_length > max_length:
        # 使用UUID生成一个唯一且足够长的标识符来替代时间戳
        unique_identifier = str(uuid.uuid4())
        # 确定需要从文件名中移除的字符数量
        remove_length = filename_length - max_length - len(file_ext) - 1  # 减去后缀、下划线和剩余部分的长度
        # 从文件名主体中移除足够数量的字符
        file_name_no_ext = file_name_no_ext[:-remove_length]
        new_filename = file_name_no_ext + "_" + unique_identifier + file_ext
    else:
        new_filename = filename

    return new_filename


def get_collection_name(user_id: int):
    collection_name = "knowledge_name_" + str(user_id)  # 集合的id
    return collection_name


def chinese_to_pinyin(text):
    result = pinyin(text, style=Style.NORMAL)
    pinyin_str = "".join([item[0] for item in result])
    return pinyin_str


def has_chinese(text):
    """
    判断字符串中是否包含中文
    :param text: 输入的字符串
    :return: 如果包含中文返回 True，否则返回 False
    """
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


if __name__ == "__main__":
    # 测试示例
    chinese_text = "你好，世界！"
    pinyin_text = chinese_to_pinyin(chinese_text)
    print(f"中文: {chinese_text}")
    print(f"拼音: {pinyin_text}")
