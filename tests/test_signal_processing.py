"""Testes de reamostragem, filtragem e estimativa de BPM/SNR."""
from __future__ import annotations

import numpy as np
import pytest

from rppg.methods import pos
from rppg.signal_processing import (
    bandpass,
    compute_snr,
    detrend,
    estimate_bpm,
    extract_pulse,
    process_window,
    pulse_phase,
    resample_uniform,
)
from tests.synthetic import HR_HZ, synthetic_rgb

FS = 30.0


def test_resample_uniform_grid():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.02, 0.05, 300))  # intervalos irregulares
    x = np.sin(2 * np.pi * 1.0 * t)
    t_uni, x_uni = resample_uniform(t, x, FS)
    assert np.allclose(np.diff(t_uni), 1.0 / FS)
    assert len(t_uni) == len(x_uni)


def test_resample_drops_nan_rows():
    t = np.linspace(0, 10, 300)
    x = np.ones((300, 3))
    x[50:60] = np.nan
    _, x_uni = resample_uniform(t, x, FS)
    assert np.isfinite(x_uni).all()


def test_bandpass_removes_drift_keeps_pulse():
    t = np.arange(0, 20, 1 / FS)
    pulse = np.sin(2 * np.pi * HR_HZ * t)
    drift = 5.0 * np.sin(2 * np.pi * 0.05 * t) + 0.3 * t
    filtered = bandpass(detrend(pulse + drift), FS)
    # correlação alta com o pulso original, e deriva eliminada
    corr = np.corrcoef(filtered, pulse)[0, 1]
    assert abs(corr) > 0.95
    assert abs(filtered.mean()) < 0.05


def test_estimate_bpm_pure_sine():
    t = np.arange(0, 15, 1 / FS)
    est = estimate_bpm(np.sin(2 * np.pi * HR_HZ * t), FS)
    assert abs(est.bpm - 72.0) <= 1.0
    assert est.snr_db > 5.0  # senoide limpa -> SNR alto


def test_snr_degrades_with_noise():
    rng = np.random.default_rng(0)
    t = np.arange(0, 15, 1 / FS)
    clean = np.sin(2 * np.pi * HR_HZ * t)
    noisy = clean + rng.normal(0, 3.0, len(t))
    snr_clean = estimate_bpm(bandpass(clean, FS), FS).snr_db
    snr_noisy = estimate_bpm(bandpass(noisy, FS), FS).snr_db
    assert snr_clean > snr_noisy


def test_compute_snr_counts_harmonic_as_signal():
    freqs = np.linspace(0.7, 4.0, 500)
    power = np.full_like(freqs, 0.01)
    power[np.abs(freqs - 1.2) <= 0.05] = 10.0   # fundamental
    power[np.abs(freqs - 2.4) <= 0.05] = 3.0    # 1º harmônico
    snr = compute_snr(freqs, power, peak_freq=1.2)
    assert snr > 10.0


def _cos_ending_at(shift: float, duration_s: float = 10.0) -> np.ndarray:
    """Cosseno de 1.2 Hz cuja última amostra está `shift` ciclos após um pico."""
    t = np.arange(0, duration_s, 1 / FS)
    return np.cos(2 * np.pi * HR_HZ * (t - t[-1]) - 2 * np.pi * shift)


@pytest.mark.parametrize("shift", [0.0, 0.25, 0.5, 0.75])
def test_pulse_phase_matches_known_shift(shift):
    """A fase medida na borda deve ser -shift (0 = pico na última amostra)."""
    phase = pulse_phase(_cos_ending_at(shift), FS, HR_HZ)
    err = abs((phase - (-shift)) % 1.0)
    assert min(err, 1.0 - err) < 0.03


def test_pulse_phase_survives_noise_and_drift():
    rng = np.random.default_rng(0)
    t = np.arange(0, 12, 1 / FS)
    x = (_cos_ending_at(0.0, 12.0) + rng.normal(0, 0.6, len(t))
         + 2.0 * np.sin(2 * np.pi * 0.1 * t))
    phase = pulse_phase(bandpass(detrend(x), FS), FS, HR_HZ)
    err = abs(phase % 1.0)
    assert min(err, 1.0 - err) < 0.1


def test_pulse_phase_tolerates_bpm_error():
    """3% de erro no BPM estimado quase não desloca a fase do coração."""
    phase = pulse_phase(_cos_ending_at(0.0), FS, HR_HZ * 1.03)
    err = abs(phase % 1.0)
    assert min(err, 1.0 - err) < 0.1


def test_pulse_phase_guards_bad_input():
    assert np.isnan(pulse_phase(np.ones(5), FS, HR_HZ))          # curto demais
    assert np.isnan(pulse_phase(np.zeros(300), FS, HR_HZ))       # sem sinal
    assert np.isnan(pulse_phase(_cos_ending_at(0.0), FS, 0.0))   # freq inválida


def test_extract_pulse_matches_process_window():
    from rppg.methods import chrom

    t, rgb = synthetic_rgb(fs=FS, duration_s=12.0)
    t_a, pulse_a = extract_pulse(t, rgb, chrom, FS)
    t_b, pulse_b, _ = process_window(t, rgb, chrom, FS)
    assert np.allclose(t_a, t_b)
    assert np.allclose(pulse_a, pulse_b)


def test_process_window_end_to_end():
    t, rgb = synthetic_rgb(fs=FS, duration_s=15.0)
    # simula jitter real nos timestamps da câmera
    rng = np.random.default_rng(2)
    t_jitter = t + rng.uniform(-0.005, 0.005, len(t))
    t_uni, pulse, est = process_window(t_jitter, rgb, pos, FS)
    assert abs(est.bpm - 72.0) <= 2.0
    assert len(t_uni) == len(pulse)
