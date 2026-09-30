"""VenoDec backend configuration."""

from __future__ import annotations

import os

APP_NAME = "VenoDec"
APP_VERSION = "0.1.0"

#: Bind address. 0.0.0.0 so a phone on the same Wi-Fi can reach the demo backend.
HOST = os.getenv("VENODEC_HOST", "0.0.0.0")
PORT = int(os.getenv("VENODEC_PORT", "8000"))

#: Hard ceiling on an uploaded WAV (bytes). Keeps a demo mistake from exhausting memory.
MAX_UPLOAD_BYTES = int(os.getenv("VENODEC_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

#: Privacy statement surfaced by the API and shown in the Android app.
PRIVACY_NOTICE = "Audio is processed for analysis and is not permanently stored."
