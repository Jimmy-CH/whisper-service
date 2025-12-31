import os
import tempfile
import whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests

app = FastAPI(title="Whisper ASR for Label Studio")
# 可改为 small/medium/large
model = whisper.load_model("/app/models/base.pt")


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
    results = []
    for task in request.tasks:
        audio_url = task["data"]["audio"]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            if audio_url.startswith(("http://", "https://")):
                resp = requests.get(audio_url)
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
            else:
                tmp_path = audio_url

            result = model.transcribe(tmp_path, language="zh")
            transcription = result["text"].strip()

            results.append({
                "result": [{
                    "from_name": "transcription",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {"text": [transcription]}
                }],
                "score": 0.9
            })

            if audio_url.startswith(("http://", "https://")):
                os.unlink(tmp_path)

        except Exception as e:
            print(f"Error: {e}")
            results.append({"result": []})

    return results
