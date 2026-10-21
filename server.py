import asyncio
import base64
import io
import os
import re
import tempfile
import subprocess
from pathlib import Path
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from PIL import Image, ImageOps, UnidentifiedImageError

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "SenseVoiceSmall"
HTML_PATH = BASE_DIR / "index.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# 672 / 28 = 24：方形图片约占 576 个视觉 token，为文字和回复留出空间。
MAX_IMAGE_EDGE = 672
MAX_IMAGE_PIXELS = 25_000_000
asr_lock = asyncio.Lock()

# 加载SenseVoiceSmall,默认使用CPU
model = AutoModel(
    model=MODEL_PATH,
    device="cpu",
    disable_update=True
)

EMOTIONS = {
    "HAPPY": "高兴",
    "SAD": "悲伤",
    "ANGRY": "生气",
    "NEUTRAL": "中性",
    "FEARFUL": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}

#api链接本地vllm启动的模型
client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    timeout=120.0,
    max_retries=0,
)

# 提供浏览器页面
@app.get("/")
def home():
    return FileResponse(HTML_PATH)


async def read_upload(upload: UploadFile, label: str) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"{label}为空，请重新选择")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{label}不能超过 10 MB")
    return data


def image_data_url(data: bytes) -> str:
    """校验真实图片内容，纠正方向并缩放后发送给 VL，不信任文件名或 MIME。"""
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(status_code=415, detail="图片仅支持 JPG、PNG 或 WebP")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="图片像素过大，请缩小到 2500 万像素以内")
            if max(source.size) / min(source.size) > 200:
                raise HTTPException(status_code=422, detail="图片过长或过窄，请裁剪后重试")
            source.load()
            picture = ImageOps.exif_transpose(source)
            picture.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
            # 将透明区域铺成白色，避免截图在转换 JPEG 后出现黑底。
            rgba = picture.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            output = io.BytesIO()
            background.save(output, format="JPEG", quality=90)
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="图片像素过大，请缩小后重试") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="无法读取图片，请上传完整的 JPG、PNG 或 WebP") from exc
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def transcribe_audio(data: bytes) -> tuple[str, str]:
    # 临时目录自动清理文件
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_audio")
        wav_path = os.path.join(tmpdir, "audio.wav")

        with open(input_path, "wb") as f:
            f.write(data)

        # 转成 16 kHz 单声道 WAV
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", input_path,
            "-t", "30",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            wav_path
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=30
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail="语音服务缺少 FFmpeg，请先安装") from e
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as e:
            raise HTTPException(
                status_code=422,
                detail="音频转换失败，请检查录音格式"
            ) from e

        #使用sense模型识别语音情绪
        result = model.generate(
            input=wav_path,
            language="auto",
            use_itn=True,
            batch_size=1
        )

        if not result or not result[0].get("text"):
            raise HTTPException(status_code=422, detail="没有识别到语音，请重新录制")

        # 获取转写文本与情绪标签
        raw = result[0]["text"]
        text = rich_transcription_postprocess(raw)

        tags = re.findall(r"<\|([^|]+)\|>", raw)

        emotion = next(
            (EMOTIONS[tag.upper()] for tag in tags if tag.upper() in EMOTIONS),
            "未知",
        )

        return text, emotion


@app.post("/recognize")
async def recognize(
    file: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    prompt: str = Form(default="", max_length=1000),
):
    """文字、录音、单张图片可自由组合，保留原有音频字段 file。"""
    prompt = prompt.strip()
    if file is None and image is None and not prompt:
        raise HTTPException(status_code=400, detail="请填写文字、录制语音或选择一张图片")

    image_url = None
    if image is not None:
        image_bytes = await read_upload(image, "图片")
        image_url = await run_in_threadpool(image_data_url, image_bytes)

    text, emotion = "", "未提供语音"
    if file is not None:
        audio_bytes = await read_upload(file, "录音")
        # 同一份 CPU 语音模型串行使用，阻塞操作放到线程中。
        async with asr_lock:
            text, emotion = await run_in_threadpool(transcribe_audio, audio_bytes)

    instructions = []
    if prompt:
        instructions.append(f"用户文字要求：{prompt}")
    if file is not None:
        instructions.extend([f"语音文本：{text}", f"语音情绪标签：{emotion}"])
    if image_url:
        instructions.append("请结合附图与用户的文字、语音要求作答；如果没有具体要求，请描述图片的主要内容。")
    else:
        instructions.append("理解并满足用户需求。")

    # OpenAI 兼容的多模态消息：文字和图片在同一条 user 消息中。
    content = [{"type": "text", "text": "\n".join(instructions)}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    try:
        response = await client.chat.completions.create(
            model="qwen",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个本地办公助手。"
                        "根据用户提供的文字、语音转写和图片帮助处理办公任务。"
                        "语音情绪标签可能不准确，只能作为"
                        "调整回复语气的参考，不能当作事实。"
                        "不要声称已经执行尚未调用的工具。"
                        "遇到无法做到的事情应该及时拒绝"
                    )
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_tokens=256,
            temperature=0.9
        )

    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="模型响应超时，请稍后重试") from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=503, detail="无法连接本地模型，请确认 vLLM 已在 8000 端口启动") from exc
    except APIStatusError as exc:
        detail = "模型处理失败，请查看 vLLM 日志后重试"
        if exc.status_code == 400:
            detail = "模型无法处理本次输入，请缩短文字或裁剪图片后重试，并检查 vLLM 日志"
        raise HTTPException(status_code=502, detail=detail) from exc

    return {
        "text": text,
        "emotion": emotion,
        "reply": response.choices[0].message.content or "模型未返回文字，请重试。",
    }
