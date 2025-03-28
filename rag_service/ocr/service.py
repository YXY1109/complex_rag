import argparse
import base64
from datetime import datetime
import os
import sys

import cv2
import numpy as np
from sanic import Request, Sanic, json, response

rag_service_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(rag_service_dir)

from rag_service.ocr.src.ocr_main import OCRQAnything

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
parser.add_argument("--port", type=int, default=7004, help="服务端口号")
parser.add_argument("--workers", type=int, default=1, help="服务工作进程数")
args = parser.parse_args()
print(f"ocr_args：{args}")

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mode_dir = os.path.join(root_dir, "models")
ocr_model = os.path.join(mode_dir, "ocr_models")
print(f"ocr目录：{ocr_model}")

app = Sanic("ocr_service")


@app.before_server_start
async def setup_ocr(app, loop):
    app.ctx.ocr = OCRQAnything(model_dir=ocr_model, device="cpu")


@app.route("/test", methods=["GET"])
async def test(request):
    # 获取当前年月日时分秒的时间
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return response.json({"test": f"我是ocr测试接口：{formatted_now}"}, status=200)


@app.post("/ocr_file")
async def ocr_file(request: Request):
    # 检查请求中是否包含文件
    if not request.files:
        return json({"error": "请上传文件！"}, status=400)

    # 获取上传的第一个文件
    file = request.files.get("file")

    if file:
        # 获取文件名和文件内容
        file_name = file.name
        file_body = file.body
        img = cv2.imdecode(np.frombuffer(file_body, np.uint8), cv2.IMREAD_COLOR)
        result = app.ctx.ocr(img)
        return json({"result": result})
    else:
        return json({"error": "无效的文件！"}, status=400)


@app.post("/ocr_base64")
async def ocr_base64(request: Request):
    img64 = request.json.get("img64")
    try:
        img_data = base64.b64decode(img64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        result = app.ctx.ocr(img)
        return json({"result": result})
    except Exception as e:
        print(f"Error processing image: {e}")
        return json({"error": "Invalid image data"}, status=400)


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, workers=args.workers)
