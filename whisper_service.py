import os
import whisper   # noqa
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper ASR for Label Studio")
model = whisper.load_model("/app/models/base.pt", device="cpu")


class PredictionRequest(BaseModel):
    tasks: List[dict]
    project: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "UP"}


@app.post("/setup")
def setup(config: dict):
    return {"status": "ok"}


@app.post("/webhook")
def webhook():
    return {"status": "ignored"}


def resolve_local_audio_path(logical_path: str) -> str:
    if logical_path.startswith("/data/upload/"):
        candidates = [
            logical_path.replace("/data/upload/", "/data/media/upload/", 1),  # 新版
            logical_path,                                                      # 旧版（直接）
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        # 如果都不存在，返回第一个候选（用于报错）
        return candidates[0]
    return logical_path


@app.post("/predict")
def predict(request: PredictionRequest):
    predictions = []
    for task in request.tasks:
        try:
            logical_audio_path = task["data"]["audio"]  # e.g., "/data/upload/5/xxx.mp3"
            actual_audio_path = resolve_local_audio_path(logical_audio_path)

            if not os.path.exists(actual_audio_path):
                logger.error(f"Audio file not found at: {actual_audio_path}")
                predictions.append({"result": []})
                continue

            logger.info(f"Transcribing: {actual_audio_path}")
            result = model.transcribe(actual_audio_path, language="zh", fp16=False)
            transcription = result["text"].strip()

            predictions.append({
                "result": [{
                    "from_name": "transcription",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {"text": [transcription]}
                }],
                "score": 0.95
            })

        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)
            predictions.append({"result": []})
    print('[predictions]:', predictions)
    return predictions

