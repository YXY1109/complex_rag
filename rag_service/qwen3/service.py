import torch
from transformers import AutoTokenizer, AutoModel
from sanic import Sanic, response
import json
import asyncio
from functools import partial

# 初始化Sanic应用
app = Sanic(__name__)


# 模型加载函数
def load_model():
    model_name = "qwen/Qwen3-7B-Instruct"  # 这里使用Qwen3的基础模型作为示例
    # 对于专门的embedding模型，请使用正确的模型名称
    # 注意：目前官方可能没有qwen3-embedding-0.6b，这里是假设存在该模型

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",  # 自动分配设备，会优先使用GPU
        torch_dtype=torch.float16,  # 使用float16节省显存
        trust_remote_code=True
    )

    # 如果有GPU，将模型移动到GPU并设置为评估模式
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    return tokenizer, model


# 加载模型和分词器
tokenizer, model = load_model()


# 生成嵌入的函数
def generate_embedding(text, tokenizer, model):
    # 对文本进行编码
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    # 如果有GPU，将输入移动到GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # 生成嵌入
    with torch.no_grad():  # 关闭梯度计算，节省内存和计算时间
        outputs = model(**inputs, return_dict=True)

    # 通常使用最后一层的平均池化作为嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embeddings.tolist()


# 异步处理函数，将同步的生成嵌入函数包装为异步
async def async_generate_embedding(text):
    loop = asyncio.get_event_loop()
    # 使用run_in_executor在线程池中运行同步函数，避免阻塞事件循环
    embedding = await loop.run_in_executor(
        None,
        partial(generate_embedding, text, tokenizer, model)
    )
    return embedding


# 定义API接口
@app.route("/embed", methods=["POST"])
async def embed(request):
    try:
        # 获取请求数据
        data = request.json

        # 检查文本是否存在
        if not data or "text" not in data:
            return response.json(
                {"error": "Missing 'text' in request data"},
                status=400
            )

        text = data["text"]

        # 生成嵌入
        embedding = await async_generate_embedding(text)

        # 返回结果
        return response.json({
            "embedding": embedding,
            "length": len(embedding)
        })

    except Exception as e:
        return response.json(
            {"error": str(e)},
            status=500
        )


# 健康检查接口
@app.route("/health", methods=["GET"])
async def health_check(request):
    return response.json({
        "status": "healthy",
        "model": "qwen3-embedding-0.6b",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })


if __name__ == "__main__":
    # 启动服务，默认监听0.0.0.0:8000
    app.run(
        host="0.0.0.0",
        port=8000,
        workers=1,  # 对于GPU模型，建议只使用1个worker
        debug=False  # 生产环境禁用debug模式
    )
