
import os
import re
import tempfile
import subprocess
from pathlib import Path
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "SenseVoiceSmall"
HTML_PATH = BASE_DIR / "index.html"

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


# ========== 2. 提供浏览器页面 ==========
@app.get("/")
def home():
    return FileResponse(HTML_PATH)


# ========== 3. 接收浏览器录音 ==========
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    # 限制音频上传大小：10MB
    data = await file.read(10 * 1024 * 1024 + 1)

    if not data or len(data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="录音为空或文件超过10MB"
        )

    # 临时目录自动清理文件
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_audio")
        wav_path = os.path.join(tmpdir, "audio.wav")

        with open(input_path, "wb") as f:
            f.write(data)

        # ========== 4. 转成16kHz单声道WAV ==========
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

       #获取输出
        raw = result[0]['text']
        text = rich_transcription_postprocess(raw)
       
        tags = re.findall(r"<\|([^|]+)\|>", raw)
       
        emotion = next(
            (EMOTIONS[tag.upper()] for tag in tags if tag.upper() in EMOTIONS),
            '未知'
           )

        voice_result = {
            "text": text,
            "emotion": emotion
        }

        #api链接本地vllm启动的模型
        client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key='EMPTY'
        )


        #将音频文件输入到模型中
        response = await client.chat.completions.create(
            model='qwen',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个本地办公助手。"
                        "根据用户指令帮助处理办公任务。"
                        "语音情绪标签可能不准确，只能作为"
                        "调整回复语气的参考，不能当作事实。"
                        "不要声称已经执行尚未调用的工具。"
                        "遇到无法做到的事情应该及时拒绝"
                    )
                },
                {
                    'role':'user',
                    'content':(
                        f"语音文本:{voice_result['text']}\n"
                        f"语音情绪标签:{voice_result['emotion']}\n"
                        '理解并满足用户需求'
                    )
                }
            ],
            max_tokens=256,
            temperature=0.9
        )

        model_result = response.choices[0].message.content
        
        return {
            "text": text,
            "emotion": emotion,
            "reply": model_result
        }
