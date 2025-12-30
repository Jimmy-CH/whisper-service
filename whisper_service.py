import os
import tempfile
import whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import subprocess

app = FastAPI(title="Whisper ASR for Label Studio")

# 加载模型（启动时加载一次）
model = whisper.load_model("base")  # 可改为 small/medium/large


class TaskData(BaseModel):
    audio: str  # 音频 URL


class PredictionRequest(BaseModel):
    tasks: List[dict]
    project: Optional[str] = None


@app.post("/predict")
def predict(request: PredictionRequest):
    results = []
    for task in request.tasks:
        audio_url = task["data"]["audio"]
        try:
            # 下载音频到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            # 支持本地路径或 HTTP(S)
            if audio_url.startswith(("http://", "https://")):
                resp = requests.get(audio_url)
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
            else:
                tmp_path = audio_url  # 假设是本地路径（仅开发环境）

            # 转录
            result = model.transcribe(tmp_path, language="zh")  # 根据需要调整语言
            transcription = result["text"].strip()

            # 构造 Label Studio 预测格式
            results.append({
                "result": [{
                    "from_name": "transcription",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {"text": [transcription]}
                }],
                "score": 0.9
            })

            # 清理临时文件（如果不是本地路径）
            if audio_url.startswith(("http://", "https://")):
                os.unlink(tmp_path)

        except Exception as e:
            print(f"Error processing {audio_url}: {e}")
            results.append({"result": []})

    return {"results": results}


