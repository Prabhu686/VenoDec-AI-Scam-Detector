"""Audio preprocessing for VenoDec.

One shared path feeds FULL AASIST for both analysis modes:

    bytes (WAV file or raw mic PCM16)
      -> decode to float32
      -> downmix to mono
      -> resample to 16 kHz
      -> window to exactly 64600 samples (AASIST's native input)
      -> (n_windows, 64600) float32 array

Amplitude is kept in the native [-1, 1] float range with no extra normalisation, matching
`soundfile.read()` as used by the official AASIST data pipeline.

Nothing here writes audio to disk. Buffers are in-memory and discarded by the caller.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .models.aasist.inference import NB_SAMP, SAMPLE_RATE

#: Shortest audio we will accept for analysis (0.5 s at 16 kHz).
MIN_SAMPLES = SAMPLE_RATE // 2

#: A trailing partial window is analysed only if it holds at least this much real audio (1 s).
MIN_TAIL_SAMPLES = SAMPLE_RATE

#: Safety cap on how many windows one request may analyse (~60 s of audio).
MAX_WINDOWS = 15


class AudioError(ValueError):
    """Raised when submitted audio cannot be decoded or is unusable."""


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Decode raw little-endian 16-bit PCM (what Android's AudioRecord produces) to float32."""
    if len(data) < 2:
        return np.zeros(0, dtype=np.float32)
    # Drop a trailing odd byte rather than failing on a split frame.
    usable = len(data) - (len(data) % 2)
    pcm = np.frombuffer(data[:usable], dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).copy()


def decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV/audio container to (float32 samples, sample_rate) via libsndfile."""
    try:
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - report the real decode failure to the client
        raise AudioError(f"could not decode audio: {exc}") from exc
    return samples, int(sample_rate)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix any channel layout to mono by averaging channels."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr.mean(axis=1).astype(np.float32)


def resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample to AASIST's required 16 kHz using polyphase filtering."""
    if sample_rate == SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)
    if sample_rate <= 0:
        raise AudioError(f"invalid sample rate: {sample_rate}")
    gcd = np.gcd(int(sample_rate), SAMPLE_RATE)
    up, down = SAMPLE_RATE // gcd, int(sample_rate) // gcd
    return resample_poly(samples, up, down).astype(np.float32)


def pad(x: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """Tile-repeat short audio up to `max_len`.

    This is a direct port of `pad()` in the official AASIST `data_utils.py`, so short
    utterances are handled exactly the way the model's own evaluation pipeline handles them.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def to_windows(samples: np.ndarray, max_windows: int = MAX_WINDOWS) -> np.ndarray:
    """Slice a mono 16 kHz waveform into `(n_windows, 64600)` for AASIST.

    Audio shorter than one window is tile-padded (official behaviour). Longer audio is split
    into consecutive non-overlapping windows; a trailing remainder is analysed only if it holds
    at least `MIN_TAIL_SAMPLES` of real audio, so we never score near-silent padding.
    """
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.shape[0] < MIN_SAMPLES:
        raise AudioError(
            f"audio too short to analyse: {samples.shape[0]} samples "
            f"({samples.shape[0] / SAMPLE_RATE:.2f}s), need >= {MIN_SAMPLES / SAMPLE_RATE:.2f}s"
        )
    if not np.all(np.isfinite(samples)):
        raise AudioError("audio contains non-finite samples")

    if samples.shape[0] <= NB_SAMP:
        return pad(samples, NB_SAMP)[None, :]

    windows: list[np.ndarray] = []
    for start in range(0, samples.shape[0], NB_SAMP):
        chunk = samples[start : start + NB_SAMP]
        if chunk.shape[0] < NB_SAMP:
            if chunk.shape[0] < MIN_TAIL_SAMPLES:
                break
            chunk = pad(chunk, NB_SAMP)
        windows.append(chunk)
        if len(windows) >= max_windows:
            break
    return np.stack(windows).astype(np.float32)


def prepare_wav(data: bytes) -> tuple[np.ndarray, dict]:
    """Full WAV-file path: bytes -> windows ready for FULL AASIST, plus preprocessing metadata."""
    raw, sample_rate = decode_wav_bytes(data)
    channels = 1 if raw.ndim == 1 else raw.shape[1]
    mono = to_mono(raw)
    resampled = resample_to_16k(mono, sample_rate)
    windows = to_windows(resampled)
    meta = {
        "source_sample_rate": sample_rate,
        "source_channels": channels,
        "duration_seconds": round(len(resampled) / SAMPLE_RATE, 3),
        "resampled_to": SAMPLE_RATE,
    }
    return windows, meta


class LiveAudioBuffer:
    """Rolling waveform buffer for the live-microphone mode.

    Android sends ~3 s PCM16 chunks, but AASIST's native input is 64600 samples (4.04 s).
    Keeping a rolling window means every inference after the first sees real contiguous audio
    instead of tile-padding, while still emitting one result per chunk received.

    The buffer holds at most one window of audio and is dropped when the connection closes --
    no audio is persisted.
    """

    def __init__(self, capacity: int = NB_SAMP):
        self.capacity = capacity
        self._buf = np.zeros(0, dtype=np.float32)
        self.total_samples_seen = 0

    def add_pcm16(self, data: bytes) -> np.ndarray:
        return self.add(pcm16_to_float(data))

    def add(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.total_samples_seen += samples.shape[0]
        self._buf = np.concatenate([self._buf, samples])[-self.capacity :]
        return self._buf

    @property
    def seconds_buffered(self) -> float:
        return len(self._buf) / SAMPLE_RATE

    def ready(self) -> bool:
        """True once we hold enough audio to be worth analysing."""
        return len(self._buf) >= MIN_SAMPLES

    def window(self) -> np.ndarray:
        """The current `(1, 64600)` analysis window, tile-padded only while still filling."""
        if not self.ready():
            raise AudioError("not enough audio buffered yet")
        return pad(self._buf, self.capacity)[None, :]

    def clear(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
