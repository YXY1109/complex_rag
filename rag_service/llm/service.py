import argparse
from datetime import datetime
import traceback

from openai import OpenAI
from sanic import Sanic, response

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
parser.add_argument("--port", type=int, default=7003, help="服务端口号")
parser.add_argument("--workers", type=int, default=1, help="服务工作进程数")
parser.add_argument("--base_url", type=str, default="http://127.0.0.1:11434/v1", help="Ollama base_url")
parser.add_argument("--api_key", type=str, default="na", help="Ollama api_key")
args = parser.parse_args()
print(f"llm_args：{args}")

app = Sanic("llm_service")

client = OpenAI(base_url=args.base_url, api_key=args.api_key)
print("qwen2加载完成！")


@app.route("/test", methods=["GET"])
async def test(request):
    # 获取当前年月日时分秒的时间
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return response.json({"test": f"我是llm测试接口：{formatted_now}"}, status=200)


@app.route("/llm_ollama", methods=["POST"])
async def llm_service(request) -> response.HTTPResponse:
    try:
        temperature = float(request.json.get("temperature", 0.001))
        print(f"temperature:{temperature}")
        is_stream = request.json.get("stream", False)
        print(f"是否流式返回:{is_stream}")
        content = request.json.get("content")
        print(f"content:{content}")
        model_name = request.json.get("model_name") or "qwen2.5:7b"
        print(f"model_name:{model_name}")

        completion = client.chat.completions.create(
            model=model_name, temperature=temperature, stream=is_stream, messages=[{"role": "user", "content": content}]
        )
        response_str = completion.choices[0].message.content
        return response.json({"response": response_str}, status=200)
    except Exception as e:
        print("基于Sanic的llm服务异常", e)
        return response.json({"Error": traceback.format_exc()}, status=500)


if __name__ == "__main__":
    """
    windows可能出现的问题：
    https://blog.csdn.net/m0_58461769/article/details/137154676
    """
    app.run(host=args.host, port=args.port, workers=args.workers)
