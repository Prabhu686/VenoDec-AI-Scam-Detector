"""The VenoDec analysis pipeline.

Both analysis modes converge here, so live microphone audio and uploaded WAV files are scored by
exactly the same FULL AASIST inference path:

    audio bytes -> preprocessing -> FULL AASIST -> spoof probability -> risk engine -> result

Filenames are never inspected and no value is ever synthesised: every number in the returned
payload derives from a real forward pass of the pretrained AASIST checkpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import risk_engine
from .audio import LiveAudioBuffer, prepare_wav
from .models.aasist.inference import get_detector


def analyze_windows(windows: np.ndarray, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run FULL AASIST over prepared windows and fuse the result into a risk verdict."""
    detector = get_detector()
    result = detector.analyze(windows)
    payload = risk_engine.evaluate(
        result["synthetic_probability"],
        windows_analyzed=result["windows_analyzed"],
        extra=extra,
    )
    payload["per_window_synthetic_probability"] = [
        round(v, 4) for v in result["per_window_synthetic_probability"]
    ]
    payload["model"] = detector.info()["name"] + " (FULL)"
    return payload


def analyze_wav_bytes(data: bytes) -> dict[str, Any]:
    """Analyse a complete WAV file (Mode B -- Test Audio)."""
    windows, meta = prepare_wav(data)
    payload = analyze_windows(windows, extra={"mode": "file", "audio": meta})
    # The decoded waveform and the caller's byte buffer go out of scope here; nothing is stored.
    return payload


def analyze_live_chunk(buffer: LiveAudioBuffer, chunk: bytes) -> dict[str, Any]:
    """Analyse the newest live microphone chunk (Mode A -- Live Analysis).

    The chunk is folded into a rolling 64600-sample window so each result reflects the most
    recent ~4 s of speech.
    """
    buffer.add_pcm16(chunk)
    windows = buffer.window()
    return analyze_windows(
        windows,
        extra={
            "mode": "live",
            "audio": {
                "source_sample_rate": 16000,
                "source_channels": 1,
                "seconds_buffered": round(buffer.seconds_buffered, 3),
                "resampled_to": 16000,
            },
        },
    )
