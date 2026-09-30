"""VenoDec FastAPI application.

Endpoints:
    GET  /health        service + FULL AASIST model status
    WS   /ws/analyze    live microphone and test-audio analysis (see websocket.py)
    POST /analyze       one-shot WAV analysis, for curl/debugging from a laptop
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analysis import analyze_wav_bytes
from .audio import AudioError
from .config import APP_NAME, APP_VERSION, MAX_UPLOAD_BYTES, PRIVACY_NOTICE
from .models.aasist.inference import get_detector
from .risk_engine import SIGNAL_SPECS
from .websocket import router as websocket_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("venodec")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load FULL AASIST up front so the first judge-facing request is not the one that pays for it,
    # and so a bad checkpoint fails loudly at startup instead of mid-demo.
    detector = get_detector()
    info = detector.info()
    log.info(
        "loaded FULL AASIST: %s params, checkpoint %s (%d bytes), device %s",
        info["num_parameters"],
        info["checkpoint_sha256"][:16],
        info["checkpoint_bytes"],
        info["device"],
    )
    yield


app = FastAPI(
    title=f"{APP_NAME} Backend",
    version=APP_VERSION,
    description="AI voice-cloning / scam-call detection using the FULL AASIST model.",
    lifespan=lifespan,
)

# Open CORS: this is a LAN demo backend, not a deployed service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router)


@app.get("/health")
async def health() -> dict:
    detector = get_detector()
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "model": detector.info(),
        "signals": [
            {"key": s.key, "label": s.label, "weight": s.weight, "active": s.active, "detail": s.detail}
            for s in SIGNAL_SPECS
        ],
        "privacy_notice": PRIVACY_NOTICE,
    }


@app.get("/")
async def root() -> dict:
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "endpoints": ["/health", "/ws/analyze", "/analyze"],
        "privacy_notice": PRIVACY_NOTICE,
    }


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)) -> dict:
    """One-shot WAV analysis through the same FULL AASIST pipeline as the WebSocket."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"audio exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        payload = analyze_wav_bytes(data)
    except AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        del data  # audio is not retained
    payload["privacy_notice"] = PRIVACY_NOTICE
    return payload
