1、先创建虚拟环境
conda create -n office-ai python=3.12 -y
conda activate office-ai

2、安装系统依赖以转换音频文件
sudo apt update
sudo apt install -y ffmpeg libsndfile1 libgl1 libglib2.0-0

3、安装环境库
python -m pip install -r requirements.txt

4、下载模型
安装huggingface下载工具
python -m pip install huggingface_hub

创建本地模型目录
mkdir -p ./models

下载qwen-2.5-vl-3b-AWQ模型指令
hf download Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
    --local-dir ./models/Qwen2.5-VL-3B-Instruct-AWQ

下载SenseVoiceSmall
hf download FunAudioLLM/SenseVoiceSmall \
    --local-dir ./models/SenseVoiceSmall

5、运行
需要启动两个终端，都是office-ai环境

启动环境指令
conda activate office-ai

启动vllm的终端指令
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve ./models/Qwen2.5-VL-3B-Instruct-AWQ \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8000 \
    --quantization awq \
    --gpu-memory-utilization 0.80 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --enforce-eager

启动网络的终端指令
uvicorn server:app --host 127.0.0.1 --port 8001

6、使用
在浏览器打开
http://localhost:8001