import voice_demo as vd
from openai import OpenAI

#api链接本地vllm启动的模型
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key='EMPTY'
)

#开始录音获取音频
wav_path = vd.record()
voice_result = vd.analyze_audio(wav_path)

#将音频文件输入到模型中
response = client.chat.completions.create(
    model='Qwen2.5-VL-3B-Instruct',
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
                f'语音文本:{voice_result['text']}\n'
                f'语音情绪标签:{voice_result['emotion']}\n'
                '理解并满足用户需求'
            )
        }
    ],
    max_tokens=256,
    temperature=0.9
)

print("办公助手：", response.choices[0].message.content)