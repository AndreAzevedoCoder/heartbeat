"""Processamento de sinal: reamostragem, filtragem e estimativa de BPM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

_EPS = 1e-12


def resample_uniform(
    t: np.ndarray, x: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reamostra (t, x) para uma grade de tempo uniforme a `fs` Hz.

    Os frames da câmera não chegam em intervalos exatos, então interpolamos
    linearmente para uma taxa fixa antes de qualquer FFT. `x` pode ser (N,)
    ou (N, C); linhas com NaN (ex.: rosto não detectado) são descartadas.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float).T).T  # garante (N, C)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(t)
    t, x = t[valid], x[valid]
    if len(t) < 2:
        raise ValueError("sinal curto demais para reamostrar")

    t_uni = np.arange(t[0], t[-1], 1.0 / fs)
    x_uni = np.column_stack([np.interp(t_uni, t, x[:, c]) for c in range(x.shape[1])])
    return t_uni, x_uni.squeeze()


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove tendência linear (deriva lenta de iluminação/exposição)."""
    return sp_signal.detrend(np.asarray(x, dtype=float), type="linear")


def bandpass(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = (0.7, 4.0),
    order: int = 3,
) -> np.ndarray:
    """Butterworth passa-banda aplicado com filtfilt (fase zero).

    A banda padrão 0.7–4.0 Hz cobre 42–240 bpm.
    """
    low, high = band
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)  # protege contra fs baixo
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, np.asarray(x, dtype=float))


@dataclass
class BpmEstimate:
    bpm: float
    snr_db: float
    freqs: np.ndarray   # eixo de frequência (Hz) restrito à banda
    power: np.ndarray   # espectro de potência na banda
    peak_freq: float


def estimate_bpm(
    pulse: np.ndarray,
    fs: float,
    band: tuple[float, float] = (0.7, 4.0),
    nfft_factor: int = 8,
) -> BpmEstimate:
    """Estima o BPM pelo pico da FFT (com zero-padding) dentro da banda válida.

    O zero-padding interpola o espectro, dando resolução melhor que 1/T.
    A qualidade vem de `snr` (ver `compute_snr`).
    """
    pulse = np.asarray(pulse, dtype=float)
    pulse = pulse - pulse.mean()
    n = len(pulse)
    nfft = int(2 ** np.ceil(np.log2(max(n * nfft_factor, 256))))

    windowed = pulse * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    f_band, p_band = freqs[in_band], spectrum[in_band]
    peak_freq = float(f_band[np.argmax(p_band)])

    snr = compute_snr(f_band, p_band, peak_freq)
    return BpmEstimate(
        bpm=peak_freq * 60.0,
        snr_db=snr,
        freqs=f_band,
        power=p_band,
        peak_freq=peak_freq,
    )


def compute_snr(
    freqs: np.ndarray,
    power: np.ndarray,
    peak_freq: float,
    tol_hz: float = 0.1,
) -> float:
    """SNR à la de Haan: energia no pico ± `tol_hz` e no 1º harmônico vs resto.

    Valores altos (> ~5 dB) indicam pulso limpo; perto de 0 ou negativo
    indica ruído dominando (movimento, pouca luz, ROI ruim).
    """
    sig_mask = (np.abs(freqs - peak_freq) <= tol_hz) | (
        np.abs(freqs - 2.0 * peak_freq) <= tol_hz
    )
    sig = power[sig_mask].sum()
    noise = power[~sig_mask].sum()
    return float(10.0 * np.log10((sig + _EPS) / (noise + _EPS)))


def extract_pulse(
    t: np.ndarray,
    rgb: np.ndarray,
    method_fn,
    resample_fs: float,
    band: tuple[float, float] = (0.7, 4.0),
    filter_order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Reamostra -> método -> detrend -> passa-banda. Retorna (t_uniforme, pulso)."""
    t_uni, rgb_uni = resample_uniform(t, rgb, resample_fs)
    raw_pulse = method_fn(np.atleast_2d(rgb_uni), resample_fs)
    filtered = bandpass(detrend(raw_pulse), resample_fs, band, filter_order)
    return t_uni, filtered


def process_window(
    t: np.ndarray,
    rgb: np.ndarray,
    method_fn,
    resample_fs: float,
    band: tuple[float, float] = (0.7, 4.0),
    filter_order: int = 3,
    nfft_factor: int = 8,
) -> tuple[np.ndarray, np.ndarray, BpmEstimate]:
    """Pipeline completo numa janela: reamostra -> método -> filtra -> BPM.

    Retorna (t_uniforme, pulso_filtrado, estimativa).
    """
    t_uni, filtered = extract_pulse(t, rgb, method_fn, resample_fs, band,
                                    filter_order)
    est = estimate_bpm(filtered, resample_fs, band, nfft_factor)
    return t_uni, filtered, est


def pulse_phase(
    pulse: np.ndarray,
    fs: float,
    freq_hz: float,
    tau: float = 1.5,
) -> float:
    """Fase do pulso no último instante, em [0, 1) — 0 no pico do sinal.

    Ajusta uma senoide de frequência `freq_hz` ao sinal (correlação com
    e^{-i2πft}) e avalia a fase desse ajuste na última amostra. O peso
    exponencial de constante `tau` favorece os segundos mais recentes, então
    a fase acompanha o batimento atual.

    Preferido à transformada de Hilbert aqui: o valor que interessa é
    justamente o da borda do sinal, onde a Hilbert erra bastante (~0.1 ciclo
    num seno limpo, muito mais com ruído). Serve para animar o coração em
    sincronia com o pulso — é cosmético, não uma medida de HRV.
    """
    pulse = np.asarray(pulse, dtype=float)
    if len(pulse) < 16 or not np.isfinite(pulse).all() or freq_hz <= 0:
        return float("nan")

    t = np.arange(len(pulse)) / fs
    weight = np.exp(-(t[-1] - t) / tau)
    z = np.sum((pulse - pulse.mean()) * weight * np.exp(-2j * np.pi * freq_hz * t))
    if abs(z) < 1e-12:
        return float("nan")
    angle = 2.0 * np.pi * freq_hz * t[-1] + np.angle(z)
    return float((angle / (2.0 * np.pi)) % 1.0)
