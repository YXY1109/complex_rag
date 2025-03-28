import cv2
import numpy as np

from rag_service.ocr.src.ocr_main import OCRQAnything

ocr_path = r"D:\PycharmProjects\yxy_rag\rag_service\models\ocr_models"
ocr = OCRQAnything(model_dir=ocr_path, device="cpu")
image_path = r"D:\PycharmProjects\yxy_rag\rag_service\ocr\韦小宝.jpg"

img_np = open(image_path, "rb").read()

# 接口传值使用
# img64=base64.b64encode(img_np).decode("utf-8")
# img_data = base64.b64decode(img64)
img = cv2.imdecode(np.frombuffer(img_np, np.uint8), cv2.IMREAD_COLOR)

result = ocr(img)
print(result)
