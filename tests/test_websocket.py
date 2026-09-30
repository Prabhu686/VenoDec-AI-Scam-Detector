"""WebSocket `/ws/analyze` -- both analysis modes, exercised exactly as Android drives them."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

from .conftest import wav_to_pcm16_16k

#: ~3 s of 16 kHz mono PCM16 == what Android sends per chunk.
CHUNK_BYTES = 3 * 16000 * 2


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_file_mode_analyses_a_real_wav(client, real_wav_bytes):
    """Mode B -- Test Audio."""
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({"type": "start", "mode": "file", "format": "wav"}))
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready"
        assert ready["model"]["variant"] == "FULL"

        # Send the WAV split across frames, as a real client streaming a file would.
        for i in range(0, len(real_wav_bytes), 32768):
            ws.send_bytes(real_wav_bytes[i : i + 32768])
        ws.send_text(json.dumps({"type": "end"}))

        result = json.loads(ws.receive_text())

    assert result["type"] == "result"
    assert result["mode"] == "file"
    assert 0.0 <= result["synthetic_probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["recommended_action"] in {"ALLOW", "WARN", "BLOCK"}
    assert result["model"] == "AASIST (FULL)"
    assert result["windows_analyzed"] >= 1


def test_live_mode_returns_one_result_per_chunk(client, real_wav_path):
    """Mode A -- Live microphone, driven with real speech as raw PCM16 chunks."""
    pcm = wav_to_pcm16_16k(real_wav_path)
    chunks = [pcm[i : i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)]
    chunks = [c for c in chunks if len(c) >= 16000 * 2][:3]  # >= 1 s each
    assert chunks, "need at least one usable chunk of speech"

    results = []
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(
            json.dumps(
                {"type": "start", "mode": "live", "format": "pcm16", "sample_rate": 16000}
            )
        )
        assert json.loads(ws.receive_text())["type"] == "ready"

        for chunk in chunks:
            ws.send_bytes(chunk)
            results.append(json.loads(ws.receive_text()))

        ws.send_text(json.dumps({"type": "stop"}))

    assert len(results) == len(chunks)
    for i, r in enumerate(results, start=1):
        assert r["type"] == "result"
        assert r["mode"] == "live"
        assert r["chunk"] == i
        assert 0.0 <= r["synthetic_probability"] <= 1.0
        assert r["windows_analyzed"] == 1
        assert r["model"] == "AASIST (FULL)"


def test_live_results_include_all_ui_fields(client, real_wav_path):
    pcm = wav_to_pcm16_16k(real_wav_path)[:CHUNK_BYTES]
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({"type": "start", "mode": "live"}))
        ws.receive_text()
        ws.send_bytes(pcm)
        result = json.loads(ws.receive_text())

    for key in (
        "risk_score",
        "risk_level",
        "recommended_action",
        "voice_authenticity",
        "synthetic_probability",
        "confidence",
        "confidence_label",
        "summary",
        "reasons",
        "signals",
    ):
        assert key in result, f"missing {key}"

    inactive = [s for s in result["signals"] if not s["active"]]
    assert len(inactive) == 3
    assert all(s["display"] == "Not active in prototype" for s in inactive)


def test_audio_before_start_is_rejected(client):
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_bytes(b"\x00\x01" * 1000)
        err = json.loads(ws.receive_text())
    assert err["type"] == "error"
    assert "start" in err["message"]


def test_unknown_mode_is_rejected(client):
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({"type": "start", "mode": "telepathy"}))
        err = json.loads(ws.receive_text())
    assert err["type"] == "error"


def test_malformed_control_message_is_reported(client):
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text("{not json")
        err = json.loads(ws.receive_text())
    assert err["type"] == "error"


def test_end_without_audio_is_reported(client):
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({"type": "start", "mode": "file"}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "end"}))
        err = json.loads(ws.receive_text())
    assert err["type"] == "error"
    assert "no audio" in err["message"]


def test_undecodable_upload_is_reported(client):
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({"type": "start", "mode": "file"}))
        ws.receive_text()
        ws.send_bytes(b"definitely not a wav" * 100)
        ws.send_text(json.dumps({"type": "end"}))
        err = json.loads(ws.receive_text())
    assert err["type"] == "error"
    assert "decode" in err["message"]


def test_ping_pong(client):
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text())["type"] == "pong"
