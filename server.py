"""OpenAI Whisper-compatible API server wrapping doubaoime-asr.

Provides:
  - POST /v1/audio/transcriptions  (batch, OpenAI Whisper-compatible)
  - WS   /v1/audio/stream          (realtime streaming via WebSocket)
"""

import asyncio
import json
import os
import tempfile
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Query
import uvicorn

from doubaoime_asr import ASRConfig, transcribe
from doubaoime_asr.asr import DoubaoASR, ResponseType

app = FastAPI(title="doubaoime-asr API", version="0.2.0")

API_KEY = os.environ.get("ASR_API_KEY", "")
CREDENTIAL_PATH = os.environ.get("ASR_CREDENTIAL_PATH", "/opt/doubaoime-asr/data/credentials.json")


def _check_auth(authorization: str | None):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if authorization[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _check_auth_value(token: str | None) -> bool:
    """Check a raw token value (without Bearer prefix)."""
    if not API_KEY:
        return True
    return token == API_KEY


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


@app.websocket("/v1/audio/stream")
async def audio_stream(
    ws: WebSocket,
    token: str = Query(default=None),
):
    """
    Realtime streaming ASR via WebSocket.

    Protocol:
      1. Connect with ?token=<api_key>
      2. Send binary frames: raw PCM 16-bit 16kHz mono audio chunks
      3. Receive JSON text frames:
         {"type": "session_started"}
         {"type": "vad_start"}
         {"type": "interim", "text": "..."}
         {"type": "final",   "text": "..."}
         {"type": "session_finished"}
         {"type": "error", "message": "..."}
      4. Close connection to end session
    """
    if not _check_auth_value(token):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()

    # Audio queue: client sends PCM chunks, we forward to ASR
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def pcm_generator():
        """Async generator that yields PCM chunks from the WebSocket."""
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                return
            yield chunk

    async def receive_audio():
        """Read binary frames from WebSocket and put into queue."""
        try:
            while True:
                data = await ws.receive()
                if data["type"] == "websocket.disconnect":
                    break
                if "bytes" in data and data["bytes"]:
                    await audio_queue.put(data["bytes"])
                elif "text" in data and data["text"]:
                    # Allow text commands
                    try:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "stop":
                            break
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            pass
        finally:
            await audio_queue.put(None)  # Signal end of audio

    async def process_asr():
        """Run ASR and send results back via WebSocket."""
        try:
            config = ASRConfig(credential_path=CREDENTIAL_PATH)
            asr = DoubaoASR(config)

            async for response in asr.transcribe_realtime(pcm_generator()):
                msg = None
                if response.type == ResponseType.SESSION_STARTED:
                    msg = {"type": "session_started"}
                elif response.type == ResponseType.VAD_START:
                    msg = {"type": "vad_start"}
                elif response.type == ResponseType.INTERIM_RESULT:
                    msg = {"type": "interim", "text": response.text}
                elif response.type == ResponseType.FINAL_RESULT:
                    msg = {"type": "final", "text": response.text}
                elif response.type == ResponseType.SESSION_FINISHED:
                    msg = {"type": "session_finished"}
                elif response.type == ResponseType.ERROR:
                    msg = {"type": "error", "message": response.error_msg}

                if msg:
                    await ws.send_json(msg)

        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    # Run both tasks concurrently
    recv_task = asyncio.create_task(receive_audio())
    asr_task = asyncio.create_task(process_asr())

    try:
        await asyncio.gather(recv_task, asr_task)
    except Exception:
        pass
    finally:
        recv_task.cancel()
        asr_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


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
