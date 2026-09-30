"""Populate `demo_audio/` with real audio for validating and demonstrating VenoDec.

Every file here is real audio and nothing is hand-labelled to force a result -- whatever FULL
AASIST says about these files is what `validate_samples.py` reports.

  demo_audio/genuine/            real human speech, clean studio recordings
  demo_audio/synthetic/          real synthesised speech, generated locally by a TTS engine
  demo_audio/known_limitations/  real human speech that AASIST *misclassifies* (see README)

The `known_limitations/` set exists because AASIST was trained on clean ASVspoof-2019-LA (VCTK)
audio and is measurably sensitive to channel/domain mismatch. Noisy, band-limited or heavily
compressed genuine recordings can score as synthetic. These files are kept deliberately so the
team knows the failure mode instead of discovering it in front of judges.

Run:  python backend/scripts/fetch_demo_audio.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import urllib.request
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo_audio"
GENUINE_DIR = DEMO_DIR / "genuine"
SYNTHETIC_DIR = DEMO_DIR / "synthetic"
LIMITS_DIR = DEMO_DIR / "known_limitations"

# (filename, url, description, destination directory)
DOWNLOADS = [
    (
        "timit_ldc93s1.wav",
        "https://github.com/mozilla/DeepSpeech/raw/master/data/smoke_test/LDC93S1.wav",
        "TIMIT LDC93S1, clean studio speech, 16 kHz mono",
        GENUINE_DIR,
    ),
    (
        "librispeech_clean.wav",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/2.flac",
        "LibriSpeech clean read speech (public domain audiobook), 16 kHz mono",
        GENUINE_DIR,
    ),
    (
        "jfk_inaugural_1961.wav",
        "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav",
        "J.F. Kennedy inaugural 1961 (public domain) -- genuine human, but a noisy, band-limited "
        "archival broadcast recording that AASIST scores as synthetic",
        LIMITS_DIR,
    ),
    (
        "ljspeech_lj001_0001.wav",
        "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav",
        "LJ Speech, genuine human reader (public domain) -- home-studio recording that AASIST "
        "scores as borderline/synthetic",
        LIMITS_DIR,
    ),
]

# Sentences long enough to fill AASIST's 4.04 s analysis window, in a scam-call register.
TTS_SCRIPTS = [
    (
        "tts_david_bank_scam",
        "Microsoft David Desktop",
        "Hello, this is an urgent message from your bank security department. "
        "We have detected unusual activity on your account and we need you to verify "
        "your identity immediately by confirming your one time password.",
    ),
    (
        "tts_zira_delivery_scam",
        "Microsoft Zira Desktop",
        "Good afternoon, I am calling about the parcel that could not be delivered to your "
        "address this morning. Please confirm your full name and date of birth so that we "
        "can reschedule the delivery for tomorrow.",
    ),
]

POWERSHELL_TEMPLATE = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{ $s.SelectVoice('{voice}') }} catch {{ }}
$s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{path}', $fmt)
$s.Speak('{text}')
$s.SetOutputToNull()
$s.Dispose()
"""


def _to_wav16(data: bytes, dest: Path) -> None:
    """Write audio out as 16-bit PCM WAV so the Android file picker always accepts it."""
    samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if samples.shape[1] > 1:
        samples = samples.mean(axis=1, keepdims=True)
    sf.write(dest, samples[:, 0], rate, subtype="PCM_16", format="WAV")


def download_all() -> int:
    count = 0
    for name, url, desc, dest_dir in DOWNLOADS:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VenoDec/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            _to_wav16(raw, dest)
        except Exception as exc:  # noqa: BLE001 - report, never substitute
            print(f"  FAILED {name}: {exc}", file=sys.stderr)
            continue
        print(f"  {dest_dir.name}/{name}  ({dest.stat().st_size} bytes)  -- {desc}")
        count += 1
    return count


def generate_synthetic() -> int:
    """Generate genuinely synthetic speech with the Windows SAPI TTS engine."""
    if sys.platform != "win32":
        print("  skipped: SAPI TTS generation requires Windows. Place your own AI/cloned "
              "samples in demo_audio/synthetic/.", file=sys.stderr)
        return 0

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, voice, text in TTS_SCRIPTS:
        dest = SYNTHETIC_DIR / f"{name}.wav"
        script = POWERSHELL_TEMPLATE.format(
            voice=voice,
            path=str(dest).replace("\\", "\\\\"),
            text=text.replace("'", "''"),
        )
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not dest.exists():
            print(f"  FAILED {name}: {proc.stderr.strip()[:300]}", file=sys.stderr)
            continue
        print(f"  synthetic/{name}.wav  ({dest.stat().st_size} bytes)  -- SAPI TTS voice '{voice}'")
        count += 1
    return count


def main() -> int:
    print("Downloading real speech recordings:")
    downloaded = download_all()
    print("\nGenerating synthetic speech with a real TTS engine:")
    generated = generate_synthetic()

    total = downloaded + generated
    print(f"\n{total} file(s) written under {DEMO_DIR}")
    if total == 0:
        print("No demo audio could be produced.", file=sys.stderr)
        return 1
    print("Next: python backend/scripts/validate_samples.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
