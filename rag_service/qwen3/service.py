import asyncio
import logging
import os
from textwrap import dedent
from typing import List, Dict, Type, Any, Optional, Union

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

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 向量模型
MODEL_NAME_EM = os.path.join(root, "models", "Qwen", "Qwen3-Embedding-0___6B")
# 重排模型
MODEL_NAME_RE = os.path.join(root, "models", "Qwen", "Qwen3-Reranker-0___6B")

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


# OpenAI Compatible Request Models
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(description="输入文本，可以是字符串或字符串列表")
    model: Optional[str] = Field(default=None, description="模型名称")
    encoding_format: Optional[str] = Field(default="float", description="编码格式")
    dimensions: Optional[int] = Field(default=None, description="嵌入维度")
    user: Optional[str] = Field(default=None, description="用户标识")


class RerankRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="模型名称")
    query: str = Field(description="查询文本")
    documents: List[str] = Field(description="文档列表")
    top_n: Optional[int] = Field(default=None, description="返回前N个结果")
    user: Optional[str] = Field(default=None, description="用户标识")


# OpenAI Compatible Response Models
class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(description="提示词token数")
    total_tokens: int = Field(description="总token数")


class EmbeddingData(BaseModel):
    object: str = Field(default="embedding", description="对象类型")
    embedding: List[float] = Field(description="嵌入向量")
    index: int = Field(description="索引")


class EmbeddingResponse(BaseModel):
    object: str = Field(default="list", description="对象类型")
    data: List[EmbeddingData] = Field(description="嵌入数据")
    model: str = Field(description="模型名称")
    usage: EmbeddingUsage = Field(description="使用情况")


class RerankResult(BaseModel):
    index: int = Field(description="原始文档索引")
    relevance_score: float = Field(description="相关性分数")
    document: Optional[str] = Field(default=None, description="文档内容")


class RerankUsage(BaseModel):
    prompt_tokens: int = Field(description="提示词token数")
    total_tokens: int = Field(description="总token数")


class RerankResponse(BaseModel):
    object: str = Field(default="list", description="对象类型")
    data: List[RerankResult] = Field(description="重排结果")
    model: str = Field(description="模型名称")
    usage: RerankUsage = Field(description="使用情况")


embedding_schema = model_to_schema_with_defaults(EmbeddingRequest)
rerank_schema = model_to_schema_with_defaults(RerankRequest)


@app.post("/v1/embeddings")
@openapi.summary("Embedding Service")
@openapi.description("OpenAI compatible embedding API")
@openapi.tag("Embeddings")
@openapi.body({"application/json": embedding_schema})
async def create_embedding(request):
    try:
        # 解析请求数据
        json_data = request.json or {}

        # 使用OpenAI规范解析请求
        input_data = json_data.get("input")
        if not input_data:
            return response.json({
                "error": {
                    "message": "Missing 'input' field in request",
                    "type": "invalid_request_error",
                    "code": "missing_input"
                }
            }, status=400)

        model_name = json_data.get("model", "qwen3-embedding")
        encoding_format = json_data.get("encoding_format", "float")
        dimensions = json_data.get("dimensions")
        user = json_data.get("user")

        # 确保输入是列表形式
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = input_data

        try:
            # 在单独的线程中运行模型推理，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            embeddings, prompt_tokens = await loop.run_in_executor(
                None,
                generate_embeddings,
                texts,
                "",  # OpenAI规范不使用instruct参数
                8192
            )

            # 如果指定了维度，进行截断或填充
            if dimensions is not None:
                for i, embedding in enumerate(embeddings):
                    if len(embedding) > dimensions:
                        embeddings[i] = embedding[:dimensions]
                    elif len(embedding) < dimensions:
                        embeddings[i] = embedding + [0.0] * (dimensions - len(embedding))

            # 构造OpenAI兼容的响应
            embedding_data = [
                EmbeddingData(
                    object="embedding",
                    embedding=embedding,
                    index=i
                )
                for i, embedding in enumerate(embeddings)
            ]

            response_data = EmbeddingResponse(
                object="list",
                data=embedding_data,
                model=model_name,
                usage=EmbeddingUsage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens
                )
            )

            logger.info(f"成功处理 {len(texts)} 个文本，生成嵌入向量，用户: {user}")
            return response.json(response_data.model_dump())

        except Exception as e:
            return response.json({
                "error": {
                    "message": f"Failed to generate embeddings: {str(e)}",
                    "type": "internal_server_error",
                    "code": "embedding_generation_failed"
                }
            }, status=500)

    except Exception as e:
        return response.json({
            "error": {
                "message": f"Invalid JSON: {str(e)}",
                "type": "invalid_request_error",
                "code": "invalid_json"
            }
        }, status=400)


@app.post("/v1/rerank")
@openapi.summary("Rerank Service")
@openapi.description("OpenAI compatible rerank API")
@openapi.tag("Rerank")
@openapi.body({"application/json": rerank_schema})
async def rerank(request):
    try:
        # 解析请求数据
        json_data = request.json or {}

        # 使用OpenAI规范解析请求
        query = json_data.get("query")
        documents = json_data.get("documents")
        model_name = json_data.get("model", "qwen3-reranker")
        top_n = json_data.get("top_n", len(documents) if documents else 0)
        user = json_data.get("user")

        if not query:
            return response.json({
                "error": {
                    "message": "Missing 'query' field in request",
                    "type": "invalid_request_error",
                    "code": "missing_query"
                }
            }, status=400)

        if not documents or not isinstance(documents, list):
            return response.json({
                "error": {
                    "message": "Missing or invalid 'documents' field in request",
                    "type": "invalid_request_error",
                    "code": "invalid_documents"
                }
            }, status=400)

        try:
            # 在单独的线程中运行模型推理
            loop = asyncio.get_event_loop()
            results, prompt_tokens = await loop.run_in_executor(
                None,
                perform_reranking,
                query,
                documents,
                ""  # OpenAI规范不使用instruct参数
            )

            # 构造OpenAI兼容的重排结果
            rerank_results = []
            for result in results:
                rerank_result = RerankResult(
                    index=result["index"],
                    relevance_score=result["score"],
                    document=result["document"]
                )
                rerank_results.append(rerank_result)

            # 按分数排序
            rerank_results.sort(key=lambda x: x.relevance_score, reverse=True)

            # 如果指定了top_n，截取前N个结果
            if top_n and top_n < len(rerank_results):
                rerank_results = rerank_results[:top_n]

            # 构造OpenAI兼容的响应
            response_data = RerankResponse(
                object="list",
                data=rerank_results,
                model=model_name,
                usage=RerankUsage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens
                )
            )

            logger.info(f"成功处理重排请求，查询: {query[:30]}..., 文档数量: {len(documents)}, 用户: {user}")
            return response.json(response_data.model_dump())

        except Exception as e:
            return response.json({
                "error": {
                    "message": f"Failed to perform reranking: {str(e)}",
                    "type": "internal_server_error",
                    "code": "reranking_failed"
                }
            }, status=500)

    except Exception as e:
        return response.json({
            "error": {
                "message": f"Invalid JSON: {str(e)}",
                "type": "invalid_request_error",
                "code": "invalid_json"
            }
        }, status=400)


@app.get("/health")
@openapi.summary("Health Check")
@openapi.description("Service health status check")
@openapi.tag("System")
async def health_check(request):
    """健康检查接口"""
    try:
        # 检查模型状态
        embedding_available = model_em is not None
        rerank_available = model_re is not None

        # 获取GPU内存使用情况（如果使用GPU）
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "device": "cuda",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                "gpu_name": torch.cuda.get_device_name(0)
            }
        else:
            gpu_info = {"device": "cpu"}

        health_status = {
            "status": "healthy" if embedding_available and rerank_available else "unhealthy",
            "models": {
                "embedding": {
                    "name": os.path.basename(MODEL_NAME_EM),
                    "available": embedding_available
                },
                "rerank": {
                    "name": os.path.basename(MODEL_NAME_RE),
                    "available": rerank_available
                }
            },
            "system": gpu_info,
            "version": "1.0.0",
            "endpoints": [
                "/v1/embeddings",
                "/v1/rerank",
                "/health"
            ]
        }

        status_code = 200 if health_status["status"] == "healthy" else 503
        return response.json(health_status, status=status_code)

    except Exception as e:
        return response.json({
            "status": "unhealthy",
            "error": str(e)
        }, status=503)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=1, debug=False, single_process=True)
