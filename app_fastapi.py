from contextlib import asynccontextmanager
import time
from urllib.request import Request

from fastapi import FastAPI, applications
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import uvicorn

from src.config.config import settings
from src.routers import chat, knowledge, tasks, upload, user
from src.utils.common import ORJSONResponse
from src.utils.logger import logger


# 使用本地静态文件
def swagger_monkey_patch(*args, **kwargs):
    return get_swagger_ui_html(
        *args, **kwargs, swagger_js_url="./static/swagger-ui-bundle.js", swagger_css_url="./static/swagger-ui.css"
    )


applications.get_swagger_ui_html = swagger_monkey_patch


@asynccontextmanager
async def lifespan(app_life: FastAPI):
    # 配置na_cos
    logger.info("服务启动")
    # https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html
    yield
    logger.info("服务关闭")


app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 子路由
app.include_router(user.router)
app.include_router(knowledge.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(tasks.router)

app.add_middleware(
    CORSMiddleware,
    # 这里配置允许跨域访问的前端地址
    allow_origins=["*"],
    # 跨域请求是否支持 cookie， 如果这里配置true，则allow_origins不能配置*
    allow_credentials=True,
    # 支持跨域的请求类型，可以单独配置get、post等，也可以直接使用通配符*表示支持所有
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """计算接口处理时间"""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = f"{time.perf_counter() - start_time:.2f}"
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    if settings.DEBUG:
        logger.info("启动本地调试模式")
        uvicorn.run(app="app_fastapi:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT)
    else:
        logger.info("启动生产模式")
        # 正式使用，后台服务
        # gunicorn app_fastapi:app -c gunicorn.py
        # 调试使用，前台服务
        # uvicorn.run(app="app_fastapi:app", host=run_host, port=run_port, reload=is_debug)
