@echo off
chcp 65001 >nul
rem 并行执行5个批处理脚本
start "本地FRP服务" frp_start.bat
start "BCE-embedding和rerank" bce_start.bat
start "LLM-本地Ollama服务" llm_start.bat
start "OCR-图片识别" ocr_start.bat
start "PDF-未完成" pdf_start.bat
echo 所有脚本已启动。