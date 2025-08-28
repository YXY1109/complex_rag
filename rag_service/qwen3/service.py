import asyncio
import logging
import os
from textwrap import dedent
from typing import List, Dict, Type, Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from sanic import Sanic, response
from sanic.worker.manager import WorkerManager
from sanic_ext import openapi
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Sanic(__name__)
WorkerManager.THRESHOLD = 6000  # 默认30s，600=60s
# 接口文档：http://127.0.0.1:8000/docs/swagger
# https://sanic.dev/en/plugins/sanic-ext/openapi/decorators.html#getting-started
app.ext.openapi.describe("基于qwen3的向量服务和重排服务", version="1.0.0", description=dedent(
    """
    ### 向量模型：https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B/summary 
    ### 重排模型：https://modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B/summary 
    """
))

root = os.path.dirname(os.path.abspath(__file__))

# 向量模型
MODEL_NAME_EM = os.path.join(root, "models", "Qwen", "Qwen3-Embedding-0.6B")
# 重排模型
MODEL_NAME_RE = os.path.join(root, "models", "Qwen", "Qwen3-Reranker-0.6B")

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")

# 加载向量模型和分词器
try:
    logger.info(f"正在加载向量模型: {MODEL_NAME_EM}")
    tokenizer_em = AutoTokenizer.from_pretrained(MODEL_NAME_EM, trust_remote_code=True, padding_side='left')
    model_em = AutoModel.from_pretrained(MODEL_NAME_EM, trust_remote_code=True)
    model_em = model_em.to(device)
    model_em.eval()
    logger.info("向量模型加载完成")
except Exception as e:
    logger.error(f"向量模型加载失败: {str(e)}", exc_info=True)
    raise

# 加载重排模型和分词器
try:
    logger.info(f"正在加载重排模型: {MODEL_NAME_RE}")
    tokenizer_re = AutoTokenizer.from_pretrained(MODEL_NAME_RE, padding_side='left')
    model_re = AutoModelForCausalLM.from_pretrained(MODEL_NAME_RE)
    model_re = model_re.to(device)
    model_re.eval()

    # 重排模型特殊配置
    global token_false_id, token_true_id, max_length_re, prefix_tokens, suffix_tokens
    token_false_id = tokenizer_re.convert_tokens_to_ids("no")
    token_true_id = tokenizer_re.convert_tokens_to_ids("yes")
    max_length_re = 8192

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer_re.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer_re.encode(suffix, add_special_tokens=False)

    logger.info("重排模型加载完成")
except Exception as e:
    logger.error(f"重排模型加载失败: {str(e)}", exc_info=True)
    raise


# 池化处理函数 - 官方实现
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# 重排模型辅助函数
def format_instruction(instruction, query, doc):
    if not instruction:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def process_inputs(pairs):
    inputs = tokenizer_re(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length_re - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer_re.pad(inputs, padding=True, return_tensors="pt", max_length=max_length_re)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs


@torch.no_grad()
def compute_logits(inputs):
    batch_scores = model_re(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    return batch_scores[:, 1].exp().tolist()


def perform_reranking(query: str, documents: List[str], instruct="") -> (List[Dict], int):
    """执行重排的同步函数"""
    with torch.no_grad():
        # 格式化输入对
        pairs = [format_instruction(instruct, query, doc) for doc in documents]

        # 处理输入
        inputs = process_inputs(pairs)

        # 计算token数量
        prompt_tokens = sum(inputs.attention_mask.sum(dim=1).tolist())

        # 计算分数
        scores = compute_logits(inputs)

        # 整理结果，按分数降序排列
        results = [
            {
                "object": "rerank_result",
                "document": documents[i],
                "score": float(scores[i]),
                "index": i
            } for i in range(len(documents))
        ]

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)

        return results, prompt_tokens


def generate_embeddings(texts: List[str], instruct="", max_length=8192):
    """生成文本嵌入的同步函数"""
    with torch.no_grad():
        if instruct:
            input_texts = [
                f"Instruct: {instruct}\nQuery: {txt}"
                for txt in texts
            ]
        else:
            input_texts = texts

        # 处理文本
        inputs = tokenizer_em(input_texts, padding=True, truncation=True, max_length=max_length,
                              return_tensors="pt").to(device)

        # 计算token数量
        prompt_tokens = sum(inputs.attention_mask.sum(dim=1).tolist())

        # 获取模型输出
        outputs = model_em(**inputs)

        # 对于Qwen嵌入模型，通常使用最后一层隐藏状态的平均值作为嵌入向量
        embeddings = last_token_pool(outputs.last_hidden_state, inputs.attention_mask)

        # 标准化嵌入向量 - 官方实现
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 转换为列表返回
        return embeddings.cpu().numpy().tolist(), prompt_tokens


def model_to_schema_with_defaults(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """
    把 Pydantic 模型转成 sanic-ext 需要的 JSON Schema（含默认值）。
    """
    # 让 Pydantic 先帮我们生成 schema
    schema = model_cls.model_json_schema(
        by_alias=False,  # 不用 alias
        ref_template="#/components/schemas/{model}",  # 避免 $ref
    )

    # 把 default 字段从 Pydantic Field 的 json_schema_extra 里搬出来
    for name, field_info in model_cls.model_fields.items():
        if field_info.default is not None or field_info.default_factory is not None:
            # 优先用 default_factory（很少人用，但兼容一下）
            default = field_info.default if field_info.default is not None else field_info.default_factory()
            schema["properties"][name]["default"] = default

    # 移除 $ref，让 sanic-openapi 直接内联
    schema.pop("$defs", None)
    return schema


class EmbeddingModel(BaseModel):
    input: list = Field(default=["你是谁", "住在那里"], description="输入文本")
    instruct: str = Field(default="", description="指令")
    max_length: int = Field(default=8192, description="最大长度")


class RerankModel(BaseModel):
    query: list = Field(default="What is the capital of China?", description="输入文本")
    documents: list = Field(default=[
        "i live chongqing city",
        "The capital of China is Beijing.",
        "中国的首都是北京",
        "大家知道重庆是个好地方，重庆是中国的，重庆和北京一样的等级，都是直辖市，且北京还是首都哦",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "i love you",
        "beijing is a big city"
    ], description="输入文本")
    instruct: str = Field(default="", description="指令")


embedding_schema = model_to_schema_with_defaults(EmbeddingModel)
rerank_schema = model_to_schema_with_defaults(RerankModel)


@app.post("/v1/embeddings")
@openapi.summary("向量服务")
@openapi.description("向量接口，与vllm接口规范一致")
@openapi.tag("向量")
@openapi.body({"application/json": embedding_schema})
async def create_embedding(request):
    try:
        # 解析请求数据
        json_data = request.json or {}
    except Exception as e:
        return response.json({"error": f"Invalid JSON: {str(e)}"}, status=400)

    texts = json_data.get("input", "")
    if not texts:
        return response.json({"error": "Missing 'input' field in request"}, status=400)

    instruct = json_data.get("instruct", "")
    max_length = json_data.get("max_length", 8192)

    # 确保输入是列表形式
    if isinstance(texts, str):
        texts = [texts]

    try:
        # 在单独的线程中运行模型推理，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        embeddings, prompt_tokens = await loop.run_in_executor(
            None,
            generate_embeddings,
            texts,
            instruct,
            max_length
        )

        # 按照vllm的格式构造响应
        response_data = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                } for i, embedding in enumerate(embeddings)  # noqa
            ],
            "model": MODEL_NAME_EM,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens
            }
        }

        logger.info(f"成功处理 {len(texts)} 个文本，生成嵌入向量")
        return response.json(response_data)

    except Exception as e:
        return response.json({"error": f"Failed to generate embeddings: {str(e)}"}, status=500)


@app.post("/v1/rerank")
@openapi.summary("重排服务")
@openapi.description("重排接口，与vllm接口规范一致")
@openapi.tag("重排")
@openapi.body({"application/json": rerank_schema})
async def rerank(request):
    try:
        # 解析请求数据
        json_data = request.json or {}
    except Exception as e:
        return response.json({"error": f"Invalid JSON: {str(e)}"}, status=400)

    query = json_data.get("query", "")
    documents = json_data.get("documents", [])

    if not query:
        return response.json({"error": "Missing 'query' field in request"}, status=400)
    if not documents or not isinstance(documents, list):
        return response.json({"error": "Missing or invalid 'documents' field in request"}, status=400)

    instruct = json_data.get("instruct", "")

    try:
        # 在单独的线程中运行模型推理
        loop = asyncio.get_event_loop()
        results, prompt_tokens = await loop.run_in_executor(
            None,
            perform_reranking,
            query,
            documents,
            instruct
        )

        # 构造响应
        response_data = {
            "object": "list",
            "data": results,
            "model": MODEL_NAME_RE,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens
            }
        }

        logger.info(f"成功处理重排请求，查询: {query[:30]}..., 文档数量: {len(documents)}")
        return response.json(response_data)

    except Exception as e:
        return response.json({"error": f"Failed to perform reranking: {str(e)}"}, status=500)


@app.get("/health")
async def health_check(request):
    """健康检查接口"""
    return response.json({"status": "healthy", "embedding_model": MODEL_NAME_EM, "rerank_model": MODEL_NAME_RE})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=1, debug=False)
