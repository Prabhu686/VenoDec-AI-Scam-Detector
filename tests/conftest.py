from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo_audio"
GENUINE_DIR = DEMO_DIR / "genuine"
SYNTHETIC_DIR = DEMO_DIR / "synthetic"


def _wavs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.wav")) if directory.is_dir() else []


@pytest.fixture(scope="session")
def genuine_wavs() -> list[Path]:
    return _wavs(GENUINE_DIR)


@pytest.fixture(scope="session")
def synthetic_wavs() -> list[Path]:
    return _wavs(SYNTHETIC_DIR)


@pytest.fixture(scope="session")
def real_wav_path(genuine_wavs) -> Path:
    """A real speech WAV from demo_audio/."""
    if not genuine_wavs:
        pytest.skip(
            "no demo audio present; run: python backend/scripts/fetch_demo_audio.py"
        )
    return genuine_wavs[0]


@pytest.fixture(scope="session")
def real_wav_bytes(real_wav_path) -> bytes:
    return real_wav_path.read_bytes()


@pytest.fixture(scope="session")
def stereo_44k_wav_bytes() -> bytes:
    """A synthesised stereo 44.1 kHz WAV, to exercise downmix + resample."""
    rate = 44100
    t = np.linspace(0, 5.0, int(rate * 5.0), endpoint=False, dtype=np.float32)
    left = 0.2 * np.sin(2 * np.pi * 220 * t)
    right = 0.2 * np.sin(2 * np.pi * 330 * t)
    stereo = np.stack([left, right], axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def wav_to_pcm16_16k(path: Path) -> bytes:
    """Decode a WAV to the exact byte format Android's AudioRecord emits (16 kHz mono PCM16)."""
    from scipy.signal import resample_poly

    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != 16000:
        g = np.gcd(int(rate), 16000)
        mono = resample_poly(mono, 16000 // g, int(rate) // g)
    pcm = np.clip(mono, -1.0, 1.0) * 32767.0
    return pcm.astype("<i2").tobytes()
