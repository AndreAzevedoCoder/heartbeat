"""Gerador de sinal RGB sintético para os testes."""
from __future__ import annotations

import numpy as np

HR_HZ = 1.2  # 72 bpm


def synthetic_rgb(
    fs: float = 30.0,
    duration_s: float = 30.0,
    hr_hz: float = HR_HZ,
    noise_std: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """RGB sintético: senoide cardíaca + deriva lenta de iluminação + ruído.

    - O pulso entra com as amplitudes relativas típicas do PPG na pele
      (~0.33R : 0.77G : 0.53B, de Haan & Jeanne 2013).
    - A iluminação varia multiplicativamente (mesmo fator nos 3 canais),
      que é o que CHROM e POS foram desenhados para cancelar.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / fs)

    pulse = np.sin(2 * np.pi * hr_hz * t)
    illum = 1.0 + 0.08 * np.sin(2 * np.pi * 0.1 * t) + 0.04 * (t / duration_s)

    base = np.array([120.0, 100.0, 80.0])
    ppg_amp = np.array([0.33, 0.77, 0.53]) * 1.2

    rgb = (
        base[None, :] * illum[:, None]
        + ppg_amp[None, :] * pulse[:, None]
        + rng.normal(0.0, noise_std, (len(t), 3))
    )
    return t, rgb
