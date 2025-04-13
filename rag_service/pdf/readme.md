```安装

conda activate rag_server_pdf
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U "magic-pdf[full]" -i https://mirrors.aliyun.com/pypi/simple
pip install -r .\rag_service\pdf\requirements.txt

```