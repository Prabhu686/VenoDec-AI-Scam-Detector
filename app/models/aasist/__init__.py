"""FULL AASIST anti-spoofing model, vendored from https://github.com/clovaai/aasist (MIT)."""

from .inference import (
    NB_SAMP,
    SAMPLE_RATE,
    AasistDetector,
    get_detector,
)

__all__ = ["AasistDetector", "get_detector", "NB_SAMP", "SAMPLE_RATE"]
