import argparse
import asyncio
import base64
import functools
import os
import time
from datetime import datetime

import numpy as np
import torch
from sanic import Sanic, response
from sentence_transformers import CrossEncoder, SentenceTransformer

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


@app.route("/v1/embeddings", methods=["POST"])
async def openai_embeddings_service(request) -> response.HTTPResponse:
    """
    OpenAI兼容的向量嵌入服务
    符合OpenAI Embeddings API规范
    """
    start_time = time.time()
    try:
        # 解析请求数据
        data = request.json
        if not data:
            return response.json({
                "error": {
                    "message": "Missing request body",
                    "type": "invalid_request_error"
                }
            }, status=400)

        # 获取输入文本 - 支持字符串或字符串数组
        input_data = data.get("input")
        if not input_data:
            return response.json({
                "error": {
                    "message": "Missing 'input' field",
                    "type": "invalid_request_error"
                }
            }, status=400)

        # 统一处理为列表格式
        if isinstance(input_data, str):
            sentences = [input_data]
        elif isinstance(input_data, list):
            sentences = input_data
        else:
            return response.json({
                "error": {
                    "message": "'input' must be a string or array of strings",
                    "type": "invalid_request_error"
                }
            }, status=400)

        # 获取模型参数（可选）
        model = data.get("model", "bce-embedding-base_v1")
        encoding_format = data.get("encoding_format", "float")
        dimensions = data.get("dimensions")

        print(f"OpenAI Embeddings Request - Model: {model}, Input count: {len(sentences)}")

        # 执行向量化
        loop = asyncio.get_event_loop()
        bce_embeddings = await asyncio.gather(
            loop.run_in_executor(
                None,
                functools.partial(embeddings_model.encode, normalize_embeddings=True),
                sentences
            )
        )
        embeddings_array = np.vstack(bce_embeddings)

        # 处理维度缩减（如果指定）
        if dimensions and dimensions < embeddings_array.shape[1]:
            embeddings_array = embeddings_array[:, :dimensions]

        # 构建OpenAI格式的响应
        embeddings_data = []
        for i, embedding in enumerate(embeddings_array):
            embedding_obj = {
                "object": "embedding",
                "embedding": embedding.tolist() if encoding_format == "float" else base64.b64encode(
                    embedding.tobytes()).decode(),
                "index": i
            }
            embeddings_data.append(embedding_obj)

        processing_time = time.time() - start_time

        response_data = {
            "object": "list",
            "data": embeddings_data,
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(sentence.split()) for sentence in sentences),
                "total_tokens": sum(len(sentence.split()) for sentence in sentences)
            }
        }

        print(f"OpenAI Embeddings Response - Processing time: {processing_time:.2f}s")
        return response.json(response_data, status=200)

    except Exception as e:
        print("OpenAI兼容向量服务异常", e)
        return response.json({
            "error": {
                "message": f"Internal server error: {str(e)}",
                "type": "internal_server_error"
            }
        }, status=500)


@app.route("/v1/rerank", methods=["POST"])
async def openai_rerank_service(request) -> response.HTTPResponse:
    """
    OpenAI兼容的重排序服务
    符合Cohere Rerank API规范（OpenAI重排服务格式）
    """
    start_time = time.time()
    try:
        # 解析请求数据
        data = request.json
        if not data:
            return response.json({
                "error": {
                    "message": "Missing request body",
                    "type": "invalid_request_error"
                }
            }, status=400)

        # 获取必要参数
        query = data.get("query")
        documents = data.get("documents")
        top_n = data.get("top_n", len(documents) if documents else 10)
        model = data.get("model", "bce-reranker-base_v1")

        if not query:
            return response.json({
                "error": {
                    "message": "Missing 'query' field",
                    "type": "invalid_request_error"
                }
            }, status=400)

        if not documents or not isinstance(documents, list):
            return response.json({
                "error": {
                    "message": "Missing 'documents' field or it's not an array",
                    "type": "invalid_request_error"
                }
            }, status=400)

        print(f"OpenAI Rerank Request - Model: {model}, Query: {query[:50]}..., Documents: {len(documents)}")

        # 构建输入数据对
        input_data_list = [[query, doc] for doc in documents]

        # 执行重排序
        loop = asyncio.get_event_loop()
        score_nd_array = await asyncio.gather(
            loop.run_in_executor(
                None,
                functools.partial(rerank_model.predict, sentences=input_data_list, batch_size=len(documents))
            )
        )
        score_list = list(score_nd_array[0])
        print(f"重排分数：{score_list}")

        # 按分数排序
        indexed_scores = list(enumerate(score_list))
        sorted_results = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

        # 构建OpenAI格式的响应
        results = []
        for idx, (original_idx, score) in enumerate(sorted_results[:top_n]):
            result = {
                "index": original_idx,
                "relevance_score": float(score),
                "document": {
                    "text": documents[original_idx]
                }
            }
            results.append(result)

        processing_time = time.time() - start_time

        response_data = {
            "id": f"rerank-{int(time.time() * 1000)}",
            "model": model,
            "results": results,
            "usage": {
                "total_tokens": len(query.split()) + sum(len(doc.split()) for doc in documents)
            }
        }

        print(f"OpenAI Rerank Response - Processing time: {processing_time:.2f}s, Top results: {len(results)}")
        return response.json(response_data, status=200)

    except Exception as e:
        print("OpenAI兼容重排服务异常", e)
        return response.json({
            "error": {
                "message": f"Internal server error: {str(e)}",
                "type": "internal_server_error"
            }
        }, status=500)


@app.route("/health", methods=["GET"])
async def health_check(request):
    """
    健康检查接口
    """
    try:
        # 检查模型是否加载
        embedding_loaded = embeddings_model is not None
        rerank_loaded = rerank_model is not None

        # 检查GPU状态
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = 0
        if gpu_available:
            gpu_memory_used = torch.cuda.memory_allocated() / 1024 ** 3  # GB

        health_status = {
            "status": "healthy" if embedding_loaded and rerank_loaded else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "embedding": {
                    "loaded": embedding_loaded,
                    "model": "bce-embedding-base_v1"
                },
                "rerank": {
                    "loaded": rerank_loaded,
                    "model": "bce-reranker-base_v1"
                }
            },
            "device": {
                "type": device.type,
                "gpu_available": gpu_available,
                "gpu_memory_used_gb": round(gpu_memory_used, 2)
            },
            "version": "1.0.0"
        }

        status_code = 200 if health_status["status"] == "healthy" else 503
        return response.json(health_status, status=status_code)

    except Exception as e:
        return response.json({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status=503)


if __name__ == "__main__":
    """
    windows可能出现的问题：
    https://blog.csdn.net/m0_58461769/article/details/137154676
    """
    app.run(host=args.host, port=args.port, workers=args.workers)
