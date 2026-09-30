"""HTTP surface: /health, /, and POST /analyze."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_full_aasist(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["service"] == "VenoDec"

    model = body["model"]
    assert model["name"] == "AASIST"
    assert model["variant"] == "FULL"
    assert model["full_aasist"] is True
    assert model["is_mock"] is False
    assert model["checkpoint_bytes"] == 1_281_532
    assert model["input_samples"] == 64600
    assert model["sample_rate"] == 16000
    assert 250_000 < model["num_parameters"] < 350_000


def test_health_declares_inactive_signals_honestly(client):
    signals = {s["key"]: s for s in client.get("/health").json()["signals"]}

    assert signals["voice_authenticity"]["active"] is True
    for key in ("speaker_similarity", "conversation_risk", "context"):
        assert signals[key]["active"] is False
        assert signals[key]["detail"] == "Not active in prototype"


def test_privacy_notice_present(client):
    assert (
        client.get("/health").json()["privacy_notice"]
        == "Audio is processed for analysis and is not permanently stored."
    )


def test_root(client):
    body = client.get("/").json()
    assert "/ws/analyze" in body["endpoints"]


def test_analyze_endpoint_scores_a_real_wav(client, real_wav_bytes, real_wav_path):
    resp = client.post(
        "/analyze", files={"file": (real_wav_path.name, real_wav_bytes, "audio/wav")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["synthetic_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["model"] == "AASIST (FULL)"


def test_analyze_endpoint_rejects_garbage(client):
    resp = client.post("/analyze", files={"file": ("bad.wav", b"not audio", "audio/wav")})
    assert resp.status_code == 400
