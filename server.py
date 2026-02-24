"""OpenAI Whisper-compatible API server wrapping doubaoime-asr."""

import os
import tempfile
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
import uvicorn

from doubaoime_asr import ASRConfig, transcribe

app = FastAPI(title="doubaoime-asr API", version="0.1.0")

API_KEY = os.environ.get("ASR_API_KEY", "")
CREDENTIAL_PATH = os.environ.get("ASR_CREDENTIAL_PATH", "/opt/doubaoime-asr/data/credentials.json")


def _check_auth(authorization: str | None):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("doubao"),
    language: str = Form(None),
    response_format: str = Form("json"),
    authorization: str | None = Header(None),
):
    _check_auth(authorization)

    audio_data = await file.read()
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "wav")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        config = ASRConfig(credential_path=CREDENTIAL_PATH)
        text = await transcribe(tmp_path, config=config)
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return text
    return {"text": text}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(None)):
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {"id": "doubao", "object": "model", "created": int(time.time()), "owned_by": "doubaoime-asr"},
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("ASR_PORT", "7000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
