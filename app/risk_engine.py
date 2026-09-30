"""VenoDec risk engine.

Transparent, auditable fusion of detection signals. The architecture carries the full planned
signal set with its intended weights, but **only signals that are actually implemented contribute
to the score**. Inactive signals are reported with `active: false` and contribute nothing -- no
placeholder value is invented for a model that does not exist yet.

In this prototype the only real signal is FULL AASIST voice authenticity, so after renormalising
over active signals it carries the entire score.

    risk_score = round(100 * sum(weight_i * risk_i) / sum(weight_i))   over ACTIVE signals only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Risk band thresholds (inclusive upper bounds), per the VenoDec spec.
LOW_MAX = 30
MEDIUM_MAX = 60

NOT_ACTIVE_TEXT = "Not active in prototype"


@dataclass(frozen=True)
class SignalSpec:
    key: str
    label: str
    weight: float
    active: bool
    detail: str


#: The planned signal set. Weights are the design weights from the VenoDec specification.
SIGNAL_SPECS: tuple[SignalSpec, ...] = (
    SignalSpec(
        key="voice_authenticity",
        label="Voice Authenticity",
        weight=0.35,
        active=True,
        detail="FULL AASIST anti-spoofing model (pretrained, ASVspoof 2019 LA)",
    ),
    SignalSpec(
        key="speaker_similarity",
        label="Speaker Verification",
        weight=0.25,
        active=False,
        detail=NOT_ACTIVE_TEXT,
    ),
    SignalSpec(
        key="conversation_risk",
        label="Conversation Analysis",
        weight=0.25,
        active=False,
        detail=NOT_ACTIVE_TEXT,
    ),
    SignalSpec(
        key="context",
        label="Context",
        weight=0.15,
        active=False,
        detail=NOT_ACTIVE_TEXT,
    ),
)


def risk_level(score: int) -> str:
    if score <= LOW_MAX:
        return "LOW"
    if score <= MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def recommended_action(level: str) -> str:
    return {"LOW": "ALLOW", "MEDIUM": "WARN", "HIGH": "BLOCK"}[level]


def confidence_label(confidence: float) -> str:
    if confidence >= 0.6:
        return "High"
    if confidence >= 0.3:
        return "Medium"
    return "Low"


def _summary(level: str) -> str:
    return {
        "LOW": "Likely genuine",
        "MEDIUM": "Suspicious voice",
        "HIGH": "AI-generated voice detected",
    }[level]


def _reasons(level: str, synthetic_probability: float, confidence: float) -> list[str]:
    pct = round(synthetic_probability * 100)
    reasons: list[str] = []
    if level == "HIGH":
        reasons.append(f"High probability of synthetic speech ({pct}%)")
    elif level == "MEDIUM":
        reasons.append(f"Partial indicators of synthetic speech ({pct}%)")
    else:
        reasons.append(f"Speech is consistent with a genuine human voice ({100 - pct}% authentic)")

    if confidence < 0.3:
        reasons.append("Model output is near the decision boundary; treat as inconclusive")
    reasons.append("Speaker verification and conversation analysis are not active in this prototype")
    return reasons


def evaluate(
    synthetic_probability: float,
    *,
    windows_analyzed: int = 1,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn a real FULL AASIST spoof probability into a VenoDec risk verdict.

    Args:
        synthetic_probability: AASIST softmax probability of the *spoof* class, in [0, 1].
        windows_analyzed: how many 64600-sample windows produced this probability.
        extra: optional extra fields merged into the payload (preprocessing metadata etc).
    """
    p = float(min(max(synthetic_probability, 0.0), 1.0))

    # Renormalise the design weights over the signals that are actually implemented.
    active = [s for s in SIGNAL_SPECS if s.active]
    active_weight = sum(s.weight for s in active)
    if active_weight <= 0:
        raise RuntimeError("no active detection signals configured")
    weighted = (0.35 * p) / active_weight  # voice_authenticity is the only active contributor

    score = int(round(100 * weighted))
    score = min(max(score, 0), 100)
    level = risk_level(score)

    # Confidence is distance from the 0.5 decision boundary -- derived from the model output,
    # not a fabricated number.
    confidence = round(abs(p - 0.5) * 2.0, 4)

    signals = []
    for spec in SIGNAL_SPECS:
        entry: dict[str, Any] = {
            "key": spec.key,
            "label": spec.label,
            "weight": spec.weight,
            "active": spec.active,
            "detail": spec.detail,
        }
        if spec.key == "voice_authenticity":
            entry["value"] = round(1.0 - p, 4)
            entry["display"] = f"{round((1.0 - p) * 100)}%"
        else:
            entry["value"] = None
            entry["display"] = NOT_ACTIVE_TEXT
        signals.append(entry)

    payload: dict[str, Any] = {
        "synthetic_probability": round(p, 4),
        "voice_authenticity": round(1.0 - p, 4),
        "risk_score": score,
        "risk_level": level,
        "recommended_action": recommended_action(level),
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "summary": _summary(level),
        "reasons": _reasons(level, p, confidence),
        "signals": signals,
        "windows_analyzed": windows_analyzed,
    }
    if extra:
        payload.update(extra)
    return payload
