import os
import tempfile
import whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper ASR for Label Studio")

# 加载模型（使用 CPU）
model_path = os.getenv("WHISPER_MODEL_PATH", "/app/models/base.pt")
device = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
logger.info(f"Loading Whisper model from {model_path} on device: {device}")
model = whisper.load_model(model_path, device=device)

# Label Studio 配置
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")  # 可选，用于认证


def get_audio_url(raw_url: str) -> str:
    """将相对路径转换为完整 URL"""
    if raw_url.startswith("/"):
        return LABEL_STUDIO_URL.rstrip("/") + raw_url
    return raw_url


def download_audio_file(url: str) -> str:
    """下载音频文件到临时位置，返回本地路径"""
    headers = {}
    if LABEL_STUDIO_TOKEN:
        headers["Authorization"] = f"Token {LABEL_STUDIO_TOKEN}"

    logger.info(f"Downloading audio from: {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            logger.error("401 Unauthorized: Please check LABEL_STUDIO_TOKEN or enable LABEL_STUDIO_EXPOSE_DATA_SERVER=true")
        raise HTTPException(status_code=resp.status_code, detail=f"Failed to download audio: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(resp.content)
        return tmp.name


class PredictionRequest(BaseModel):
    tasks: List[dict]
    project: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "UP", "model": model_path, "device": device}


@app.post("/setup")
def setup(config: dict):
    # Label Studio 会调用此接口传递项目配置（可忽略）
    logger.info("Received /setup call")
    return {"status": "ok"}


@app.post("/webhook")
def webhook():
    # 可选：避免 404
    return {"status": "ignored"}


@app.post("/predict")
def predict(request: PredictionRequest):
    predictions = []
    for task in request.tasks:
        try:
            audio_url = task["data"].get("audio")
            if not audio_url:
                logger.warning("No 'audio' field in task data")
                predictions.append({"result": []})
                continue

            full_url = get_audio_url(audio_url)

            # 下载音频
            if full_url.startswith(("http://", "https://")):
                local_path = download_audio_file(full_url)
                need_cleanup = True
            else:
                # 假设是本地路径（仅在共享 volume 时使用）
                local_path = full_url
                need_cleanup = False

            # 转写
            logger.info(f"Transcribing {local_path}...")
            result = model.transcribe(local_path, language="zh", fp16=False)
            transcription = result["text"].strip()

            # 构造 Label Studio 兼容的结果
            predictions.append({
                "result": [{
                    "from_name": "transcription",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {"text": [transcription]}
                }],
                "score": 0.95
            })

            # 清理临时文件
            if need_cleanup:
                os.unlink(local_path)

        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)
            predictions.append({"result": []})

    return predictions
