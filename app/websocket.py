"""VenoDec `/ws/analyze` WebSocket endpoint.

Protocol (the Android client implements exactly this):

  Mode A -- live microphone
    -> {"type":"start","mode":"live","format":"pcm16","sample_rate":16000}
    <- {"type":"ready", ...}
    -> <binary>  raw PCM16LE mono 16 kHz, ~3 s per frame
    <- {"type":"result", ...}          (one result per chunk)
    ... repeat ...
    -> {"type":"stop"}                 (optional)

  Mode B -- test audio (WAV file)
    -> {"type":"start","mode":"file","format":"wav"}
    <- {"type":"ready", ...}
    -> <binary> ... <binary>           WAV bytes, may be split across frames
    -> {"type":"end"}
    <- {"type":"result", ...}          (single aggregated result)

Errors are reported as {"type":"error","message":...} rather than silently dropped.

Privacy: audio lives only in the per-connection buffer below and is discarded as soon as it has
been analysed or the socket closes. Nothing is written to disk and no audio bytes are logged.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .analysis import analyze_live_chunk, analyze_wav_bytes
from .audio import AudioError, LiveAudioBuffer
from .config import MAX_UPLOAD_BYTES, PRIVACY_NOTICE
from .models.aasist.inference import get_detector

log = logging.getLogger("venodec.ws")

router = APIRouter()

VALID_MODES = {"live", "file"}


class Session:
    """Per-connection state. Holds audio in memory only, for the life of the socket."""

    def __init__(self) -> None:
        self.mode: str | None = None
        self.live_buffer: LiveAudioBuffer | None = None
        self.file_bytes = bytearray()
        self.chunks_received = 0

    def start(self, mode: str) -> None:
        self.mode = mode
        self.chunks_received = 0
        self.live_buffer = LiveAudioBuffer() if mode == "live" else None
        self.file_bytes = bytearray()

    def discard(self) -> None:
        """Drop every audio byte held by this session."""
        if self.live_buffer is not None:
            self.live_buffer.clear()
        self.file_bytes = bytearray()


@router.websocket("/ws/analyze")
async def analyze_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = Session()

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if (text := message.get("text")) is not None:
                if await _handle_text(websocket, session, text):
                    break
            elif (data := message.get("bytes")) is not None:
                await _handle_binary(websocket, session, data)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - never leave the client hanging without a reason
        log.exception("websocket failure")
        await _safe_send(websocket, {"type": "error", "message": f"internal error: {exc}"})
    finally:
        # Guarantee the audio buffer does not outlive the connection.
        session.discard()


async def _handle_text(websocket: WebSocket, session: Session, text: str) -> bool:
    """Handle a control message. Returns True if the connection should close."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        await _safe_send(websocket, {"type": "error", "message": "control message is not valid JSON"})
        return False

    kind = msg.get("type")

    if kind == "start":
        mode = msg.get("mode", "live")
        if mode not in VALID_MODES:
            await _safe_send(
                websocket,
                {"type": "error", "message": f"unknown mode {mode!r}; expected one of {sorted(VALID_MODES)}"},
            )
            return False
        session.start(mode)
        await _safe_send(
            websocket,
            {
                "type": "ready",
                "mode": mode,
                "model": get_detector().info(),
                "privacy_notice": PRIVACY_NOTICE,
            },
        )
        return False

    if kind == "end":
        await _finish_file(websocket, session)
        return False

    if kind == "stop":
        session.discard()
        await _safe_send(websocket, {"type": "stopped"})
        return True

    if kind == "ping":
        await _safe_send(websocket, {"type": "pong"})
        return False

    await _safe_send(websocket, {"type": "error", "message": f"unknown control message {kind!r}"})
    return False


async def _handle_binary(websocket: WebSocket, session: Session, data: bytes) -> None:
    if session.mode is None:
        await _safe_send(
            websocket,
            {"type": "error", "message": "send a {'type':'start'} control message before audio"},
        )
        return

    session.chunks_received += 1

    if session.mode == "file":
        if len(session.file_bytes) + len(data) > MAX_UPLOAD_BYTES:
            session.discard()
            await _safe_send(
                websocket,
                {"type": "error", "message": f"audio exceeds {MAX_UPLOAD_BYTES} byte limit"},
            )
            return
        session.file_bytes.extend(data)
        return

    # Live mode: analyse this chunk immediately.
    assert session.live_buffer is not None
    try:
        payload = analyze_live_chunk(session.live_buffer, data)
    except AudioError as exc:
        await _safe_send(websocket, {"type": "error", "message": str(exc)})
        return
    payload["type"] = "result"
    payload["chunk"] = session.chunks_received
    await _safe_send(websocket, payload)


async def _finish_file(websocket: WebSocket, session: Session) -> None:
    if session.mode != "file":
        await _safe_send(websocket, {"type": "error", "message": "'end' is only valid in file mode"})
        return
    if not session.file_bytes:
        await _safe_send(websocket, {"type": "error", "message": "no audio received"})
        return

    data = bytes(session.file_bytes)
    session.file_bytes = bytearray()  # release the upload buffer before inference
    try:
        payload = analyze_wav_bytes(data)
    except AudioError as exc:
        await _safe_send(websocket, {"type": "error", "message": str(exc)})
        return
    finally:
        del data

    payload["type"] = "result"
    await _safe_send(websocket, payload)


async def _safe_send(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_text(json.dumps(payload))
    except (WebSocketDisconnect, RuntimeError):
        pass
