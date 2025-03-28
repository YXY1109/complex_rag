from fastapi import APIRouter, Depends, Request
from loguru import logger
from starlette.responses import StreamingResponse

router = APIRouter(
    prefix="/chat",
    tags=["对话"],
)


@router.post("/chat", summary="知识库对话")
@logger.catch
async def knowledge_chat(request: Request, chat_param: ChatParam, db: Session = Depends(get_session)):
    logger.info("知识库对话")
    api_key = router.state.config.get("api_key")
    logger.info(f"api_key:{api_key}")

    question = chat_param.question

    stream = chat_param.stream
    if stream:
        # 流示返回，milvus+es混合检索
        streaming_response_content = chat_es_milvus_stream(question)

        return StreamingResponse(content=streaming_response_content, media_type="text/event-stream")
    else:
        # 一次性全部返回，只使用milvus向量相似度

        # todo rag的中文提示词
        rag_prompt = ""
        # 构建rag的链
        rag_chain = create_retriever_chain(rag_prompt)
        # 获取检索结果和上下文原始信息
        context, metadata = get_docs_context(question)

    return {"message": "知识库对话"}
