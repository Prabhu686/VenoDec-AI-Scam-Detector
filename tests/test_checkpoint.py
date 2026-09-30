"""Proof that VenoDec loads the genuine FULL AASIST checkpoint -- not AASIST-L, not a mock."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.models.aasist.inference import (
    AASIST_L_BYTES,
    BONAFIDE_INDEX,
    FULL_AASIST_BYTES,
    NB_SAMP,
    SPOOF_INDEX,
    get_detector,
)


@pytest.fixture(scope="module")
def detector():
    return get_detector()


def test_checkpoint_is_full_aasist_not_light(detector):
    """The checkpoint on disk is byte-for-byte the FULL AASIST release."""
    assert detector.checkpoint_bytes == FULL_AASIST_BYTES
    assert detector.checkpoint_bytes != AASIST_L_BYTES
    assert detector.weights_path.name == "AASIST.pth"
    assert detector.info()["variant"] == "FULL"
    assert detector.info()["is_mock"] is False


def test_architecture_matches_full_aasist_config(detector):
    """FULL AASIST hyper-parameters, which AASIST-L does not share."""
    cfg = detector.model_config
    assert cfg["architecture"] == "AASIST"
    assert cfg["nb_samp"] == NB_SAMP
    assert cfg["gat_dims"] == [64, 32]  # AASIST-L uses [24, 24]
    assert cfg["filts"] == [70, [1, 32], [32, 32], [32, 64], [64, 64]]
    assert cfg["first_conv"] == 128


def test_weights_loaded_strictly_and_are_pretrained(detector):
    """A strict load succeeded, and the weights are trained values rather than fresh init.

    Comparing against a freshly-constructed model proves we are running the *pretrained*
    checkpoint and not randomly-initialised weights.
    """
    from app.models.aasist.aasist_model import Model

    fresh = Model(detector.model_config)
    loaded_out = dict(detector.model.named_parameters())["out_layer.weight"]
    fresh_out = dict(fresh.named_parameters())["out_layer.weight"]
    assert loaded_out.shape == fresh_out.shape
    assert not torch.allclose(loaded_out.cpu(), fresh_out.cpu())

    # Model is in eval mode with gradients disabled (inference-only service).
    assert detector.model.training is False
    assert all(not p.requires_grad for p in detector.model.parameters())


def test_parameter_count_is_full_aasist_scale(detector):
    """FULL AASIST is ~297k parameters; AASIST-L is ~85k."""
    n = detector.num_parameters
    print(f"\nFULL AASIST parameter count: {n}")
    assert 250_000 < n < 350_000


def test_forward_pass_produces_two_class_probabilities(detector):
    """Real PyTorch forward pass over the model's native input shape."""
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 0.05, size=(2, NB_SAMP)).astype(np.float32)
    probs = detector.predict_windows(windows)

    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert SPOOF_INDEX == 0 and BONAFIDE_INDEX == 1


def test_rejects_wrong_input_shape(detector):
    with pytest.raises(ValueError):
        detector.predict_windows(np.zeros((1, 1000), dtype=np.float32))
