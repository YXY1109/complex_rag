import base64

import requests


def test_ocr_base64():
    image_path = r"D:\PycharmProjects\yxy_rag\rag_service\ocr\韦小宝.jpg"

    img_np = open(image_path, "rb").read()
    img64 = base64.b64encode(img_np).decode("utf-8")
    result = requests.post("http://127.0.0.1:7004/ocr_base64", json={"img64": img64})
    print(result.json())


def test_ocr_file():
    image_path = r"D:\PycharmProjects\yxy_rag\rag_service\ocr\韦小宝.jpg"

    with open(image_path, "rb") as file:
        # 准备要上传的文件数据，键名 'file' 要和 Sanic 服务中接收文件时的键名一致
        files = {"file": file}
        result = requests.post("http://127.0.0.1:7004/ocr_file", files=files)
        print(result.json())


if __name__ == "__main__":
    # test_ocr_base64()
    test_ocr_file()
