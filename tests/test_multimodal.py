"""验证真实表单解析、音频转码和发往 vLLM 的 HTTP 请求；不加载模型权重。"""

import asyncio
import base64
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import Mock, patch
import wave

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from PIL import Image


def load_server():
    funasr = types.ModuleType("funasr")
    funasr.AutoModel = Mock(return_value=Mock())
    postprocess = types.ModuleType("funasr.utils.postprocess_utils")
    postprocess.rich_transcription_postprocess = lambda raw: re.sub(r"<\|[^|]+\|>", "", raw)
    spec = importlib.util.spec_from_file_location("server_under_test", Path(__file__).resolve().parents[1] / "server.py")
    server = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"funasr": funasr, "funasr.utils.postprocess_utils": postprocess}):
        spec.loader.exec_module(server)
    asyncio.run(server.client.close())
    return server


server = load_server()


def image_bytes(fmt="PNG", size=(1200, 800), mode="RGBA", color=(255, 0, 0, 0), **kwargs):
    data = io.BytesIO()
    Image.new(mode, size, color).save(data, format=fmt, **kwargs)
    return data.getvalue()


def audio_bytes():
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\0" * 44100 * 4)
    return data.getvalue()


class MultimodalTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.upstream_status = 200
        self.upstream_error = None
        self.audio_paths = []
        server.model.generate.reset_mock()
        server.model.generate.side_effect = self.recognize
        server.client = AsyncOpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.upstream)),
        )
        self.http = TestClient(server.app).__enter__()

    def tearDown(self):
        self.http.portal.call(server.client.close)
        self.http.__exit__(None, None, None)

    def recognize(self, input, **kwargs):
        # ASR 只替代推理，真正检查 FFmpeg 产生的临时 WAV。
        self.audio_paths.append(Path(input))
        with wave.open(input, "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
        return [{"text": "<|zh|><|HAPPY|>帮我总结这张图片"}]

    def upstream(self, request):
        self.requests.append(request)
        if self.upstream_error:
            raise self.upstream_error("unavailable", request=request)
        if self.upstream_status != 200:
            return httpx.Response(self.upstream_status, json={"error": {"message": "bad input"}})
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 1, "model": "qwen",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "这是一张示例图片。"}, "finish_reason": "stop"}],
        })

    def payload(self):
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(str(self.requests[0].url), "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(self.requests[0].method, "POST")
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload["model"], "qwen")
        self.assertEqual(payload["messages"][1]["role"], "user")
        return payload["messages"][1]["content"]

    def test_text_only(self):
        result = self.http.post("/recognize", data={"prompt": "请整理待办"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["text"], "")
        content = self.payload()
        self.assertEqual(len(content), 1)
        self.assertIn("请整理待办", content[0]["text"])
        server.model.generate.assert_not_called()

    def test_image_and_text_share_one_message(self):
        result = self.http.post("/recognize", data={"prompt": "提取图片中的文字"}, files={"image": ("screen.png", image_bytes(), "image/png")})
        self.assertEqual(result.status_code, 200, result.text)
        content = self.payload()
        self.assertEqual([part["type"] for part in content], ["text", "image_url"])
        self.assertIn("提取图片中的文字", content[0]["text"])
        url = content[1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))) as image:
            self.assertEqual(image.size, (672, 448))
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        server.model.generate.assert_not_called()

    def test_image_only_has_default_instruction(self):
        result = self.http.post("/recognize", files={"image": ("photo.webp", image_bytes(fmt="WEBP"), "image/webp")})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn("描述图片的主要内容", self.payload()[0]["text"])

    def test_exif_orientation_is_applied(self):
        exif = Image.Exif()
        exif[274] = 6
        picture = image_bytes(fmt="JPEG", size=(100, 200), mode="RGB", color="red", exif=exif)
        result = self.http.post("/recognize", files={"image": ("photo.jpg", picture, "image/jpeg")})
        self.assertEqual(result.status_code, 200, result.text)
        url = self.payload()[1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))) as image:
            self.assertEqual(image.size, (200, 100))

    def test_audio_image_and_text_are_combined(self):
        result = self.http.post("/recognize", data={"prompt": "请列成三条"}, files={
            "file": ("recording.wav", audio_bytes(), "audio/wav"),
            "image": ("photo.png", image_bytes(), "image/png"),
        })
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["emotion"], "高兴")
        self.assertEqual(result.json()["text"], "帮我总结这张图片")
        content = self.payload()
        self.assertIn("请列成三条", content[0]["text"])
        self.assertIn("帮我总结这张图片", content[0]["text"])
        self.assertIn("高兴", content[0]["text"])
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(self.audio_paths)
        self.assertFalse(self.audio_paths[0].exists(), "临时录音应该已清理")

    def test_original_audio_only_request_still_works(self):
        result = self.http.post("/recognize", files={"file": ("recording.wav", audio_bytes(), "audio/wav")})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(set(result.json()), {"text", "emotion", "reply"})
        self.assertEqual(len(self.payload()), 1)

    def test_rejects_empty_or_oversize_input(self):
        self.assertEqual(self.http.post("/recognize", data={"prompt": "  "}).status_code, 400)
        self.assertEqual(self.http.post("/recognize", data={"prompt": "字" * 1001}).status_code, 422)
        for field in ("image", "file"):
            with self.subTest(field=field):
                self.assertEqual(self.http.post("/recognize", files={field: ("empty", b"")}).status_code, 400)
                self.assertEqual(self.http.post("/recognize", files={field: ("large", b"x" * (server.MAX_UPLOAD_BYTES + 1))}).status_code, 413)
        self.assertEqual(self.requests, [])

    def test_invalid_image_and_unsupported_format(self):
        for picture, status in [(b"not a picture", 422), (image_bytes(fmt="GIF"), 415)]:
            with self.subTest(status=status):
                result = self.http.post("/recognize", files={"image": ("fake.png", picture, "image/png")})
                self.assertEqual(result.status_code, status, result.text)
        self.assertEqual(self.requests, [])

    def test_invalid_audio(self):
        result = self.http.post("/recognize", files={"file": ("fake.wav", b"invalid", "audio/wav")})
        self.assertEqual(result.status_code, 422)
        self.assertEqual(self.requests, [])

    def test_unavailable_vllm_has_readable_error(self):
        for error, status in [(httpx.ConnectError, 503), (httpx.ReadTimeout, 504)]:
            with self.subTest(error=error):
                self.upstream_error = error
                result = self.http.post("/recognize", data={"prompt": "你好"})
                self.assertEqual(result.status_code, status)
                self.assertIsInstance(result.json()["detail"], str)

    def test_vllm_rejecting_context_returns_readable_error(self):
        self.upstream_status = 400
        result = self.http.post("/recognize", data={"prompt": "你好"})
        self.assertEqual(result.status_code, 502)
        self.assertIn("缩短文字", result.json()["detail"])


if __name__ == "__main__":
    unittest.main()
