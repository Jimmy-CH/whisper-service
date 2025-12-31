import os
import tempfile
import whisper     # noqa
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import requests

app = FastAPI(title="Whisper ASR for Label Studio")
# 可改为 small/medium/large
model = whisper.load_model("/app/models/base.pt", device="cpu")


LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080")


def get_audio_url(raw_url):
    if raw_url.startswith("/"):
        return LABEL_STUDIO_URL + raw_url
    return raw_url  # 已是完整 URL


class PredictionRequest(BaseModel):
    tasks: List[dict]
    project: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "UP"}


@app.post("/setup")
def setup(config: dict):
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    predictions = []
    for task in request.tasks:
        # task 是 dict，可以 .get()
        audio_url = get_audio_url(task["data"]["audio"])
        if not audio_url:
            predictions.append({"result": []})
            continue

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name

            if audio_url.startswith(("http://", "https://")):
                resp = requests.get(audio_url, timeout=30)
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
            else:
                tmp_path = audio_url

            transcription = model.transcribe(tmp_path, language="zh")["text"].strip()

            # 构造符合规范的 prediction
            predictions.append({
                "result": [{
                    "from_name": "transcription",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {"text": [transcription]}  # 注意：是列表！
                }],
                "score": 0.95
            })

            if audio_url.startswith(("http://", "https://")):
                os.unlink(tmp_path)

        except Exception as e:
            print(f"Error: {e}")
            predictions.append({"result": []})

    # 直接返回 list，不要包装！
    return predictions
