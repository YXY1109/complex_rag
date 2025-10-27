import argparse
import asyncio
import base64
from datetime import datetime
import functools
import json
import os
import traceback

import numpy as np
from sanic import Sanic, response
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
parser.add_argument("--port", type=int, default=7001, help="服务端口号")
parser.add_argument("--workers", type=int, default=1, help="服务工作进程数")
parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")
args = parser.parse_args()
print(f"bce_args：{args}")

app = Sanic("bce_service")

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mode_dir = os.path.join(root_dir, "models")

bce_embedding_base_v1 = os.path.join(mode_dir, "bce-embedding-base_v1")
print(f"bce向量目录：{bce_embedding_base_v1}")
bce_rerank_base_v1 = os.path.join(mode_dir, "bce-reranker-base_v1")
print(f"bce重排目录：{bce_rerank_base_v1}")
device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
print(f"device: {device}")
embeddings_model = SentenceTransformer(bce_embedding_base_v1).to(device)
rerank_model = CrossEncoder(bce_rerank_base_v1, max_length=512, device=device.type)
print("bce加载完成！！！")


@app.route("/test", methods=["GET"])
async def test(request):
    # 获取当前年月日时分秒的时间
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return response.json({"test": f"我是bce测试接口：{formatted_now}"}, status=200)


@app.route("/bce_embedding", methods=["POST"])
async def bce_embedding_service(request) -> response.HTTPResponse:
    """
    基于Sanic的bce的向量服务
    """
    try:
        sentences = request.json.get("sentences")
        print(f"sentences:{sentences}")
        loop = asyncio.get_event_loop()
        bce_embeddings = await asyncio.gather(
            loop.run_in_executor(None, functools.partial(embeddings_model.encode, normalize_embeddings=True), sentences)
        )
        embeddings_array = np.vstack(bce_embeddings)
        original_shape = json.dumps(embeddings_array.shape)

        binary_data = embeddings_array.tobytes()
        base64_data = base64.b64encode(binary_data).decode()
        return response.json({"embeddings": base64_data, "original_shape": original_shape}, status=200)
    except Exception as e:
        print("基于Sanic的bce向量服务异常", e)
        return response.json({"Error": traceback.format_exc()}, status=500)


@app.route("/bce_rerank", methods=["POST"])
async def bce_rerank_service(request) -> response.HTTPResponse:
    """
    基于Sanic的bce的重排服务
    """
    try:
        user_query = request.json.get("user_query")
        print(f"user_query:{user_query}")
        content_list = request.json.get("content_list")
        print(f"content_list:{content_list}")

        input_data_list = []
        for user_corpus in content_list:
            input_data_list.append([user_query, user_corpus])
        loop = asyncio.get_event_loop()
        score_nd_array = await asyncio.gather(
            loop.run_in_executor(
                None, functools.partial(rerank_model.predict, sentences=input_data_list, batch_size=len(content_list))
            )
        )
        score_list = list(score_nd_array[0])
        print(f"重排分数：{score_list}")
        # 排序后的分数
        score_sort_list = sorted(score_list, reverse=True)  # 降序，分数从高到底
        print(f"排序后的分数：{score_sort_list}")
        max_score = score_sort_list[0]  # 最相似问题的分数
        max_index = score_list.index(max_score)  # 最相似问题的索引
        # 猪八戒不是人
        similar_problem = input_data_list[max_index][1]  # 最相似问题的文本
        # 排序后的下标
        rank_index = [index for index, value in sorted(list(enumerate(score_list)), key=lambda x: x[1], reverse=True)]
        problem_list = []
        for r in range(len(score_list)):
            problem_list.append(content_list[rank_index[r]])
        # [('猪八戒不是人', 0.5200636), ('我喜欢孙悟空', 0.35967737), ('孙大哥要走了', 0.34095687), ('孙悟空是其他人', 0.33899534), ('我喜欢吃香蕉', 0.29164788)]
        problem_32_score = [x for x in zip(problem_list, score_sort_list)]
        problem_64_score = [(item[0], np.float64(item[1])) for item in problem_32_score]
        # {'problem_score': [('猪八戒不是人', 0.5200636), ('我喜欢孙悟空', 0.35967737), ('孙大哥要走了', 0.34095687), ('孙悟空是其他人', 0.33899534), ('我喜欢吃香蕉', 0.29164788)], 'similar_problem': '猪八戒不是人'}
        response_dict = {"similar_problem": similar_problem, "problem_score": problem_64_score}
        return response.json(response_dict, status=200)
    except Exception as e:
        print("基于Sanic的bce重排服务异常", e)
        return response.json({"Error": traceback.format_exc()}, status=500)


if __name__ == "__main__":
    """
    windows可能出现的问题：
    https://blog.csdn.net/m0_58461769/article/details/137154676
    """
    app.run(host=args.host, port=args.port, workers=args.workers)
