"""Vendor the FULL AASIST model code + pretrained checkpoint from the official repo.

Source: https://github.com/clovaai/aasist (NAVER Corp., MIT license)

This script deliberately hard-asserts the size of the downloaded checkpoint so that
AASIST-L (the *light* variant, 426,428 bytes) can never be silently substituted for
FULL AASIST (1,281,532 bytes). VenoDec requires FULL AASIST.

Run:  python backend/scripts/download_aasist.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "clovaai/aasist"
REF = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REF}/"
API_COMMIT = f"https://api.github.com/repos/{REPO}/commits/{REF}"

HERE = Path(__file__).resolve().parent
VENDOR_DIR = HERE.parent / "app" / "models" / "aasist"
WEIGHTS_DIR = VENDOR_DIR / "weights"

# The FULL AASIST checkpoint. AASIST-L.pth is 426_428 bytes -- explicitly NOT this file.
FULL_AASIST_BYTES = 1_281_532
AASIST_L_BYTES = 426_428

# (remote path, local path)
FILES = [
    ("models/AASIST.py", VENDOR_DIR / "aasist_model.py"),
    ("models/weights/AASIST.pth", WEIGHTS_DIR / "AASIST.pth"),
    ("LICENSE", VENDOR_DIR / "LICENSE"),
    ("NOTICE", VENDOR_DIR / "NOTICE"),
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VenoDec-vendor/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    digests: dict[str, str] = {}

    for remote, local in FILES:
        url = RAW + remote
        print(f"downloading {url}")
        try:
            data = fetch(url)
        except Exception as exc:  # noqa: BLE001 - surface the exact failure
            print(f"FAILED to download {url}: {exc}", file=sys.stderr)
            return 1

        if remote.endswith("AASIST.pth"):
            if len(data) == AASIST_L_BYTES:
                print(
                    "REFUSING: downloaded file is AASIST-L (the light variant). "
                    "VenoDec requires FULL AASIST.",
                    file=sys.stderr,
                )
                return 1
            if len(data) != FULL_AASIST_BYTES:
                print(
                    f"REFUSING: expected FULL AASIST checkpoint of {FULL_AASIST_BYTES} bytes, "
                    f"got {len(data)} bytes. Not writing the file.",
                    file=sys.stderr,
                )
                return 1

        local.write_bytes(data)
        digests[remote] = sha256(data)
        print(f"  -> {local}  ({len(data)} bytes, sha256={digests[remote][:16]}...)")

    try:
        import json

        commit = json.loads(fetch(API_COMMIT).decode())["sha"]
    except Exception:  # noqa: BLE001 - provenance is best-effort
        commit = "unknown"

    (VENDOR_DIR / "VENDORED.md").write_text(
        "\n".join(
            [
                "# Vendored AASIST",
                "",
                f"Source repository: https://github.com/{REPO}",
                f"Ref: `{REF}`  |  Commit at vendoring time: `{commit}`",
                f"Vendored on (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                "License: MIT (NAVER Corp.) -- see `LICENSE` and `NOTICE` in this directory.",
                "",
                "## Files",
                "",
                "| Local file | Upstream path | sha256 |",
                "| --- | --- | --- |",
                *[
                    f"| `{local.name}` | `{remote}` | `{digests[remote]}` |"
                    for remote, local in FILES
                ],
                "",
                "## FULL AASIST confirmation",
                "",
                f"`weights/AASIST.pth` is **{FULL_AASIST_BYTES} bytes** -- this is the FULL AASIST",
                f"checkpoint. The light variant `AASIST-L.pth` is {AASIST_L_BYTES} bytes and is",
                "deliberately NOT used by VenoDec. `download_aasist.py` refuses to write the file",
                "if the size does not match, and the model is loaded with `strict=True` against the",
                "FULL AASIST architecture config, which would raise on AASIST-L weights.",
                "",
                "`aasist_model.py` is an unmodified copy of upstream `models/AASIST.py`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {VENDOR_DIR / 'VENDORED.md'}")
    print("FULL AASIST vendored successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
