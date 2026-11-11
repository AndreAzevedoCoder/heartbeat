"""Testa os métodos de extração num sinal sintético de 72 bpm."""
from __future__ import annotations

import numpy as np
import pytest

from rppg.methods import METHODS, chrom, green, pos
from rppg.signal_processing import bandpass, detrend, estimate_bpm
from tests.synthetic import HR_HZ, synthetic_rgb

FS = 30.0
TRUE_BPM = HR_HZ * 60.0  # 72


def _estimate(method_fn, rgb: np.ndarray) -> float:
    pulse = bandpass(detrend(method_fn(rgb, FS)), FS)
    return estimate_bpm(pulse, FS).bpm


@pytest.mark.parametrize("method_fn", [chrom, pos], ids=["chrom", "pos"])
def test_chrom_pos_within_2bpm(method_fn):
    _, rgb = synthetic_rgb(fs=FS)
    assert abs(_estimate(method_fn, rgb) - TRUE_BPM) <= 2.0


def test_green_recovers_bpm():
    _, rgb = synthetic_rgb(fs=FS)
    assert abs(_estimate(green, rgb) - TRUE_BPM) <= 5.0


@pytest.mark.parametrize("seed", range(5))
def test_pos_robust_across_noise_seeds(seed):
    _, rgb = synthetic_rgb(fs=FS, seed=seed)
    assert abs(_estimate(pos, rgb) - TRUE_BPM) <= 2.0


def test_methods_share_signature():
    _, rgb = synthetic_rgb(fs=FS, duration_s=10.0)
    for name, fn in METHODS.items():
        out = fn(rgb, FS)
        assert out.shape == (len(rgb),), name
        assert np.isfinite(out).all(), name
