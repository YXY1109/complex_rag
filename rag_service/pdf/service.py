import argparse
from datetime import datetime
import os

current_dir = os.getcwd()
os.environ["MINERU_TOOLS_CONFIG_JSON"] = os.path.join(current_dir, "magic-pdf-server.json")

import aiofiles
from celery_app import long_task
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from sanic import Request, Sanic, json, response

pdf_parser = argparse.ArgumentParser()
pdf_parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
pdf_parser.add_argument("--port", type=int, default=7002, help="服务端口号")
pdf_parser.add_argument("--workers", type=int, default=1, help="服务工作进程数")
pdf_args = pdf_parser.parse_args()
print(f"pdf_args：{pdf_args}")

app = Sanic("pdf_service")


@app.before_server_start
async def setup_pdf(app, loop):
    pass


@app.route("/test", methods=["GET"])
async def test(request):
    # 获取当前年月日时分秒的时间
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return response.json({"test": f"我是pdf测试接口：{formatted_now}"}, status=200)


@app.route("/celery_test", methods=["POST"])
async def celery_test(request):
    print("celery_test")
    # task = long_task.delay("test")
    task = long_task.apply_async(args=["yxy", 30], queue="yxy")
    print(f"任务id为：{task.id}")
    return response.json({"test": f"我是celery_test:{task.id}"}, status=200)


@app.route("/pdf_to_md", methods=["POST"])
async def pdf_to_md(request: Request):
    file = request.files.get("pdf_file")
    if not file:
        return json({"error": "请上传文件！"}, status=400)

    # 获取文件名
    file_name = file.name

    # 所有文件保存的父目录
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 构建上传文件的目录
    file_name_no_suff = os.path.basename(file_name).split(".")[0]
    save_file_dir = os.path.join(output_dir, file_name_no_suff)
    os.makedirs(save_file_dir, exist_ok=True)

    # 原始文件保存路径
    local_pdf_path = os.path.join(save_file_dir, file_name)
    async with aiofiles.open(local_pdf_path, mode="wb") as f:
        # 异步写入文本内容
        await f.write(file.body)

    # 图片保存路径
    image_dir = "pdf_images"
    local_image_dir = os.path.join(save_file_dir, image_dir)
    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(save_file_dir)
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(local_pdf_path)  # read the pdf content
    ds = PymuDocDataset(pdf_bytes)

    # 推理
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        ## pipeline
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        ## pipeline
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # 保存模型推理结果
    model_result_path = os.path.join(save_file_dir, f"{file_name_no_suff}_model.pdf")
    infer_result.draw_model(model_result_path)

    model_inference_result = infer_result.get_infer_res()
    print(f"模型的推理结果：{model_inference_result}")

    # 保存布局分类结果
    model_draw_path = os.path.join(save_file_dir, f"{file_name_no_suff}_layout.pdf")
    pipe_result.draw_layout(model_draw_path)

    # 保存span结果
    model_span_path = os.path.join(save_file_dir, f"{file_name_no_suff}_spans.pdf")
    pipe_result.draw_span(model_span_path)

    # 获取markdown的文本内容，并保存到本地
    md_content = pipe_result.get_markdown(image_dir)
    print(f"markdown的文本内容：{md_content[:10]}")
    pipe_result.dump_md(md_writer, os.path.join(save_file_dir, f"{file_name_no_suff}.md"), image_dir)

    # 内容列表的json
    content_list_content = pipe_result.get_content_list(image_dir)
    print(f"内容列表：{len(content_list_content)}")
    pipe_result.dump_content_list(
        md_writer, os.path.join(save_file_dir, f"{file_name_no_suff}_content_list.json"), image_dir
    )

    # 中间json结果
    middle_json_content = pipe_result.get_middle_json()
    print(f"中间json结果：{len(middle_json_content)}")
    pipe_result.dump_middle_json(md_writer, os.path.join(save_file_dir, f"{file_name_no_suff}_middle.json"))

    return json({"result": "success"})


if __name__ == "__main__":
    app.run(host=pdf_args.host, port=pdf_args.port, workers=pdf_args.workers)
