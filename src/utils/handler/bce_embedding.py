import base64
import json

import numpy as np
import requests

from src.config.config import settings
from src.utils.logger import logger


def bce_model_encode(input_words):
    """
    调用bce模型
    """
    input_words_flag = input_words
    if isinstance(input_words, str):
        input_words = [input_words]
    payload = json.dumps({"sentences": input_words})
    try:
        response = requests.request("POST", settings.EMBEDDING_URL, data=payload, timeout=10)
        logger.info(f"请求状态：{response.status_code}")
        logger.info(f"请求结果：{response.content}")
        response_dict = json.loads(response.content)
        base64_data = response_dict.get("embeddings")
        original_shape = tuple(eval(response_dict.get("original_shape")))
        binary_data = base64.b64decode(base64_data)
        embeddings_array = np.frombuffer(binary_data, dtype=np.float32)
        embeddings_array = embeddings_array.reshape(original_shape)
        if isinstance(input_words_flag, str):
            embeddings_array = embeddings_array[0]
        return embeddings_array
    except Exception as e:
        logger.info(f"请求异常：{e}")


if __name__ == "__main__":
    result = bce_model_encode("你好")
    print(result)
