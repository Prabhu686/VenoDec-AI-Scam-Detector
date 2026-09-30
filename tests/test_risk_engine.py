"""Risk engine: thresholds, actions, and honest reporting of unimplemented signals."""

from __future__ import annotations

import pytest

from app.risk_engine import (
    NOT_ACTIVE_TEXT,
    SIGNAL_SPECS,
    evaluate,
    recommended_action,
    risk_level,
)


@pytest.mark.parametrize(
    "score,expected",
    [(0, "LOW"), (30, "LOW"), (31, "MEDIUM"), (60, "MEDIUM"), (61, "HIGH"), (100, "HIGH")],
)
def test_risk_bands(score, expected):
    assert risk_level(score) == expected


@pytest.mark.parametrize(
    "level,action", [("LOW", "ALLOW"), ("MEDIUM", "WARN"), ("HIGH", "BLOCK")]
)
def test_actions(level, action):
    assert recommended_action(level) == action


def test_score_derives_from_aasist_probability():
    """With AASIST the only active signal, the score tracks its probability directly."""
    assert evaluate(0.0)["risk_score"] == 0
    assert evaluate(0.5)["risk_score"] == 50
    assert evaluate(0.91)["risk_score"] == 91
    assert evaluate(1.0)["risk_score"] == 100


def test_voice_authenticity_is_the_complement():
    r = evaluate(0.92)
    assert r["voice_authenticity"] == pytest.approx(0.08)
    assert r["synthetic_probability"] == pytest.approx(0.92)


def test_confidence_is_distance_from_decision_boundary():
    assert evaluate(0.5)["confidence"] == pytest.approx(0.0)
    assert evaluate(1.0)["confidence"] == pytest.approx(1.0)
    assert evaluate(0.0)["confidence"] == pytest.approx(1.0)
    assert evaluate(0.5)["confidence_label"] == "Low"
    assert evaluate(0.95)["confidence_label"] == "High"


def test_unimplemented_signals_are_declared_not_faked():
    signals = {s["key"]: s for s in evaluate(0.5)["signals"]}

    assert signals["voice_authenticity"]["active"] is True
    assert signals["voice_authenticity"]["value"] is not None

    for key in ("speaker_similarity", "conversation_risk", "context"):
        assert signals[key]["active"] is False
        assert signals[key]["value"] is None, "inactive signals must not carry an invented value"
        assert signals[key]["display"] == NOT_ACTIVE_TEXT


def test_design_weights_are_preserved_for_future_signals():
    weights = {s.key: s.weight for s in SIGNAL_SPECS}
    assert weights == {
        "voice_authenticity": 0.35,
        "speaker_similarity": 0.25,
        "conversation_risk": 0.25,
        "context": 0.15,
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_summaries_match_the_spec_wording():
    assert evaluate(0.95)["summary"] == "AI-generated voice detected"
    assert evaluate(0.45)["summary"] == "Suspicious voice"
    assert evaluate(0.05)["summary"] == "Likely genuine"


def test_probability_is_clamped():
    assert evaluate(1.5)["risk_score"] == 100
    assert evaluate(-0.5)["risk_score"] == 0


def test_reasons_always_disclose_prototype_limits():
    reasons = evaluate(0.9)["reasons"]
    assert any("not active in this prototype" in r for r in reasons)
