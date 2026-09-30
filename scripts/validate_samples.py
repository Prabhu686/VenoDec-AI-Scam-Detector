"""Run WAV files through the real VenoDec pipeline and print the measured scores.

Every number printed comes from an actual FULL AASIST forward pass. Filenames and directory
names are used only to group the report -- they never influence the score.

Usage:
    python backend/scripts/validate_samples.py                 # scores demo_audio/
    python backend/scripts/validate_samples.py <dir-or-file>   # scores your own files
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.analysis import analyze_wav_bytes  # noqa: E402
from app.audio import AudioError  # noqa: E402
from app.models.aasist.inference import get_detector  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def collect(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def main() -> int:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "demo_audio"
    if not target.exists():
        print(f"no such path: {target}", file=sys.stderr)
        return 1

    files = collect(target)
    if not files:
        print(f"no audio files found under {target}", file=sys.stderr)
        return 1

    info = get_detector().info()
    print(f"Model: {info['name']} ({info['variant']}) | params={info['num_parameters']} "
          f"| checkpoint={info['checkpoint_bytes']} bytes | sha256={info['checkpoint_sha256'][:16]}...")
    print(f"Scoring {len(files)} file(s) under {target}\n")

    header = f"{'file':<34} {'synth%':>7} {'risk':>5} {'level':>7} {'action':>7} {'conf':>6} {'win':>4}"
    print(header)
    print("-" * len(header))

    groups: dict[str, list[float]] = {}
    failures = 0

    for path in files:
        try:
            result = analyze_wav_bytes(path.read_bytes())
        except (AudioError, Exception) as exc:  # noqa: BLE001 - report real failures
            print(f"{path.name:<34} ERROR: {exc}")
            failures += 1
            continue

        group = path.parent.name if path.parent != target else "-"
        groups.setdefault(group, []).append(result["synthetic_probability"])
        print(
            f"{path.name:<34} "
            f"{result['synthetic_probability'] * 100:>6.1f}% "
            f"{result['risk_score']:>5} "
            f"{result['risk_level']:>7} "
            f"{result['recommended_action']:>7} "
            f"{result['confidence']:>6.2f} "
            f"{result['windows_analyzed']:>4}"
        )

    print("\nMean synthetic probability by folder:")
    for group, values in sorted(groups.items()):
        print(f"  {group:<12} {sum(values) / len(values) * 100:>6.1f}%   (n={len(values)})")

    if "genuine" in groups and "synthetic" in groups:
        g = sum(groups["genuine"]) / len(groups["genuine"])
        s = sum(groups["synthetic"]) / len(groups["synthetic"])
        print(f"\nSeparation (synthetic mean - genuine mean): {(s - g) * 100:+.1f} percentage points")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
