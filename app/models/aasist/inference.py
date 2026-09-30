"""FULL AASIST inference wrapper.

Loads the genuine pretrained AASIST checkpoint (`weights/AASIST.pth`, 1,281,532 bytes) into the
unmodified upstream `Model` architecture and runs real PyTorch inference.

There is no mock, no heuristic and no fallback path in this module. If the checkpoint is missing
or does not match the FULL AASIST architecture, loading raises -- it never degrades to a fake score.

Output convention comes from upstream `main.py`:

    _, batch_out = model(batch_x)
    batch_score = (batch_out[:, 1])   # bonafide score

so logit index 1 == bonafide (genuine) and index 0 == spoof (synthetic).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .aasist_model import Model

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "aasist_config.json"
WEIGHTS_PATH = HERE / "weights" / "AASIST.pth"

#: Byte size of the FULL AASIST checkpoint. The light variant (AASIST-L.pth) is 426,428 bytes.
FULL_AASIST_BYTES = 1_281_532
AASIST_L_BYTES = 426_428

#: Logit indices, per upstream main.py.
SPOOF_INDEX = 0
BONAFIDE_INDEX = 1

#: AASIST's native input length: 64600 samples @ 16 kHz == 4.0375 s.
NB_SAMP = 64600
SAMPLE_RATE = 16000


class AasistDetector:
    """Thread-safe singleton wrapper around the pretrained FULL AASIST model."""

    def __init__(self, weights_path: Path = WEIGHTS_PATH, config_path: Path = CONFIG_PATH):
        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        self.model_config = self._load_config()
        self.checkpoint_sha256, self.checkpoint_bytes = self._verify_checkpoint()
        self.model = self._build_and_load()
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------ setup

    def _load_config(self) -> dict[str, Any]:
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        return {k: v for k, v in cfg.items() if not k.startswith("_")}

    def _verify_checkpoint(self) -> tuple[str, int]:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"AASIST checkpoint not found at {self.weights_path}. "
                "Run: python backend/scripts/download_aasist.py"
            )
        raw = self.weights_path.read_bytes()
        size = len(raw)
        if size == AASIST_L_BYTES:
            raise RuntimeError(
                "The checkpoint at %s is AASIST-L (the light variant). VenoDec requires FULL AASIST."
                % self.weights_path
            )
        if size != FULL_AASIST_BYTES:
            raise RuntimeError(
                f"Unexpected AASIST checkpoint size {size} bytes at {self.weights_path}; "
                f"FULL AASIST is {FULL_AASIST_BYTES} bytes."
            )
        return hashlib.sha256(raw).hexdigest(), size

    def _build_and_load(self) -> Model:
        model = Model(self.model_config).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Defensive: upstream saves a bare state_dict, but strip a DataParallel prefix if present.
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        # strict=True is load-bearing: it is what proves these weights match FULL AASIST.
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def predict_windows(self, windows: np.ndarray) -> np.ndarray:
        """Run FULL AASIST over a batch of waveform windows.

        Args:
            windows: float array of shape ``(n_windows, 64600)``, 16 kHz mono, range [-1, 1].

        Returns:
            float array of shape ``(n_windows, 2)`` -- softmax probabilities
            ``[:, 0]`` = spoof/synthetic, ``[:, 1]`` = bonafide/genuine.
        """
        if windows.ndim != 2 or windows.shape[1] != NB_SAMP:
            raise ValueError(
                f"expected windows of shape (n, {NB_SAMP}), got {windows.shape}"
            )
        x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).to(self.device)
        with self._lock:
            _, logits = self.model(x)
            probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def analyze(self, windows: np.ndarray) -> dict[str, Any]:
        """Aggregate FULL AASIST over one or more windows into a single verdict."""
        probs = self.predict_windows(windows)
        spoof = probs[:, SPOOF_INDEX]
        synthetic_probability = float(np.mean(spoof))
        return {
            "synthetic_probability": synthetic_probability,
            "voice_authenticity": float(1.0 - synthetic_probability),
            "windows_analyzed": int(probs.shape[0]),
            "per_window_synthetic_probability": [float(v) for v in spoof],
        }

    # -------------------------------------------------------------- metadata

    def info(self) -> dict[str, Any]:
        return {
            "name": "AASIST",
            "variant": "FULL",
            "full_aasist": True,
            "is_mock": False,
            "source": "https://github.com/clovaai/aasist",
            "checkpoint": self.weights_path.name,
            "checkpoint_bytes": self.checkpoint_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
            "num_parameters": self.num_parameters,
            "device": str(self.device),
            "sample_rate": SAMPLE_RATE,
            "input_samples": NB_SAMP,
        }


_detector: AasistDetector | None = None
_detector_lock = threading.Lock()


def get_detector() -> AasistDetector:
    """Return the process-wide FULL AASIST detector, loading it on first use."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = AasistDetector()
    return _detector
