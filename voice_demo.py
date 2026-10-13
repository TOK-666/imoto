import re
import os
import torch
import sounddevice as sd
import soundfile as sf
from fastapi import FastAPI
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

#获取模型地址
MODEL_PATH = os.path("./models/SenseVoiceSmall")
#限制采样和录音时长
SAMPLE_RATE = 16000
DURATION = 10
#限制cpu线程使用
torch.set_num_threads(4)
#录音文件输出文件夹路径
OUTPUT_PATH = os.path('./output')

#加载模型
model = AutoModel(
    model=MODEL_PATH,
    device='cpu',
    disable_update=True
)

#录音
def record():
    print('开始录音，最大持续时间10秒……')

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='flaot32'
    )

    sd.wait()

    output_path = OUTPUT_PATH / 'test.wav'

    sf.write('output_path', audio, SAMPLE_RATE)
    print('录音结束')

    return output_path

#使用sense模型识别语音情绪
def analyze_audio(output_path):
    result = model.generate(
        input=output_path,
        language='auto',
        use_itn=True,
        batch_size=1
    )

    #将情绪映射为中文
    emotions = {
    "HAPPY": "高兴",
    "SAD": "悲伤",
    "ANGRY": "生气",
    "NEUTRAL": "中性",
    "FEARFUL": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
    "EMO_UNKNOWN": "未知"
    }

    #获取输出
    raw = result[0]['text']
    text = rich_transcription_postprocess(raw)

    tags = re.findall(r"<\|([^|]+)\|>", raw)

    emotion = next(
        (emotions[tag.upper()] for tag in tags if tag.upper() in emotions),
        '未知'
    )

    output = {
        "text" : text,
        "emotion" : emotion
    }

    return output

#测试
if __name__ == '__main__':
    record()
    print(analyze_audio)