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

网页支持文字、图片、语音自由组合：
- 在文字框填写要求（最多 1000 字符）。
- 点击或拖入一张 JPG、PNG、WebP 图片（最大 10 MB），可以预览或移除。
- 点击“开始录音”，停止后可试听、重录或移除，最长录制 20 秒。
- 点击“发送给助手”，统一提交本次文字、图片和录音。每次请求独立，不保留对话上下文。

图片接入说明：
- `POST /recognize` 接收表单字段 `prompt`（文字）、`image`（图片）和 `file`（录音），均为可选，但至少提供一种。
- 后端用 SenseVoice 转写录音，将文字要求、转写文本、情绪标签和图片一起提交给 `http://127.0.0.1:8000/v1/chat/completions`，模型名仍为 `qwen`。
- 图片通过 Pillow 校验、纠正 EXIF 方向并等比缩小至最长边 672 像素，再编码成 JPEG data URL，放入同一条 user 消息的 `image_url` 内容块。
- 缩小图片是为了给当前 `--max-model-len 2048` 留出文字和生成空间；密集小字建议先裁剪关键区域。图片不能超过 2500 万像素。
- 返回值保留 `text`、`emotion`、`reply`；未上传录音时 `text` 为空，`emotion` 为“未提供语音”。
- 更新后安装最新 requirements.txt（新增 Pillow）并重启 uvicorn。vLLM 启动命令不变。

接口回归验证（替代模型推理，使用实际 FastAPI、OpenAI SDK、Pillow 和 FFmpeg）：
`python -B -m unittest discover -s tests -v`
