"""End-to-end client for a *running* VenoDec backend, over a real network socket.

This speaks byte-for-byte the same protocol the Android app implements, so it validates the
whole chain independently of the phone:

    WAV / PCM16 chunks -> WebSocket -> FastAPI -> FULL AASIST -> risk engine -> result JSON

Usage:
    # terminal 1
    cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

    # terminal 2
    python backend/scripts/e2e_client.py
    python backend/scripts/e2e_client.py --url ws://192.168.1.5:8000 --file path/to/audio.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo_audio"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3.0
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_SECONDS) * 2


def load_pcm16_mono_16k(path: Path) -> bytes:
    """Decode any WAV to the exact byte layout Android's AudioRecord produces."""
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = np.gcd(int(rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, int(rate) // g)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def show(result: dict, prefix: str = "") -> None:
    print(
        f"{prefix}risk={result['risk_score']:>3} {result['risk_level']:<6} "
        f"{result['recommended_action']:<5} | synthetic={result['synthetic_probability'] * 100:5.1f}% "
        f"authentic={result['voice_authenticity'] * 100:5.1f}% "
        f"conf={result['confidence']:.2f} ({result['confidence_label']}) "
        f"| {result['summary']}"
    )


def check_health(http_url: str) -> bool:
    print(f"GET {http_url}/health")
    try:
        with urllib.request.urlopen(f"{http_url}/health", timeout=15) as resp:
            body = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}", file=sys.stderr)
        return False

    m = body["model"]
    print(f"  status={body['status']} model={m['name']} variant={m['variant']} "
          f"params={m['num_parameters']} checkpoint={m['checkpoint_bytes']}B device={m['device']}")
    print(f"  active signals: {[s['key'] for s in body['signals'] if s['active']]}")
    print(f"  inactive:       {[s['key'] for s in body['signals'] if not s['active']]}")
    if not m.get("full_aasist"):
        print("  WARNING: backend does not report FULL AASIST", file=sys.stderr)
        return False
    return True


def run_file_mode(ws_url: str, path: Path) -> dict | None:
    print(f"\n[MODE B / TEST AUDIO] {path.name}")
    data = path.read_bytes()
    with connect(f"{ws_url}/ws/analyze", max_size=None) as ws:
        ws.send(json.dumps({"type": "start", "mode": "file", "format": "wav"}))
        ready = json.loads(ws.recv())
        if ready.get("type") != "ready":
            print(f"  unexpected: {ready}", file=sys.stderr)
            return None
        for i in range(0, len(data), 32768):
            ws.send(data[i : i + 32768])
        ws.send(json.dumps({"type": "end"}))
        result = json.loads(ws.recv())

    if result.get("type") != "result":
        print(f"  ERROR: {result.get('message', result)}", file=sys.stderr)
        return None
    show(result, prefix="  ")
    print(f"  windows analysed: {result['windows_analyzed']} "
          f"({result['audio']['duration_seconds']}s @ {result['audio']['source_sample_rate']}Hz "
          f"x{result['audio']['source_channels']}ch)")
    return result


def run_live_mode(ws_url: str, path: Path, realtime: bool = False) -> list[dict]:
    """Stream a WAV as ~3 s PCM16 chunks, exactly as the phone streams the microphone."""
    print(f"\n[MODE A / LIVE MIC simulation] streaming {path.name} as {CHUNK_SECONDS}s PCM16 chunks")
    pcm = load_pcm16_mono_16k(path)
    chunks = [pcm[i : i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)]
    chunks = [c for c in chunks if len(c) >= SAMPLE_RATE * 2]  # >= 1 s

    results: list[dict] = []
    with connect(f"{ws_url}/ws/analyze", max_size=None) as ws:
        ws.send(json.dumps({"type": "start", "mode": "live", "format": "pcm16",
                            "sample_rate": SAMPLE_RATE}))
        ready = json.loads(ws.recv())
        if ready.get("type") != "ready":
            print(f"  unexpected: {ready}", file=sys.stderr)
            return results

        for n, chunk in enumerate(chunks, start=1):
            started = time.perf_counter()
            ws.send(chunk)
            result = json.loads(ws.recv())
            elapsed = (time.perf_counter() - started) * 1000
            if result.get("type") != "result":
                print(f"  ERROR: {result.get('message', result)}", file=sys.stderr)
                continue
            show(result, prefix=f"  chunk {n}: ")
            print(f"           round-trip {elapsed:.0f} ms")
            results.append(result)
            if realtime:
                time.sleep(CHUNK_SECONDS)

        ws.send(json.dumps({"type": "stop"}))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="VenoDec end-to-end WebSocket client")
    parser.add_argument("--url", default="ws://127.0.0.1:8000", help="backend ws:// base URL")
    parser.add_argument("--file", type=Path, help="analyse a single WAV instead of demo_audio/")
    parser.add_argument("--realtime", action="store_true", help="pace live chunks in real time")
    args = parser.parse_args()

    ws_url = args.url.rstrip("/")
    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")

    if not check_health(http_url):
        return 1

    if args.file:
        targets = [args.file]
    else:
        targets = sorted(DEMO_DIR.glob("genuine/*.wav")) + sorted(DEMO_DIR.glob("synthetic/*.wav"))
    if not targets:
        print("no audio to send; run: python backend/scripts/fetch_demo_audio.py", file=sys.stderr)
        return 1

    failures = 0
    for path in targets:
        if run_file_mode(ws_url, path) is None:
            failures += 1

    if not run_live_mode(ws_url, targets[-1], realtime=args.realtime):
        failures += 1

    print("\nEND-TO-END " + ("FAILED" if failures else "OK"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
