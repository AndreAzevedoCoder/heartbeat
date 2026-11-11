"""Métodos de extração do sinal de pulso a partir das médias RGB da pele.

Todos têm a mesma assinatura:

    metodo(rgb: np.ndarray de forma (N, 3), fs: float) -> np.ndarray de forma (N,)

onde `rgb` são as médias (R, G, B) da ROI ao longo do tempo, já reamostradas
a uma taxa uniforme `fs`. A saída é o sinal de pulso 1D (sinal e escala são
irrelevantes — o que importa é a frequência dominante).
"""
from __future__ import annotations

import numpy as np

from .signal_processing import bandpass

_EPS = 1e-9


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, variância unitária (só para deixar as saídas comparáveis)."""
    return (x - x.mean()) / (x.std() + _EPS)


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    """GREEN: só o canal verde, normalizado.

    Baseline clássico (Verkruysse et al., 2008): a hemoglobina absorve mais
    luz verde, então o pulso aparece mais forte no canal G. Sensível a
    movimento e variação de iluminação — bom ponto de comparação.
    """
    return _standardize(rgb[:, 1].astype(float))


def chrom(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013).

    Projeta o RGB normalizado em dois eixos de crominância
    (X = 3R - 2G, Y = 1.5R + G - 1.5B) que cancelam variação de intensidade
    luminosa; combina S = X - alpha*Y com alpha = std(X)/std(Y) para cancelar
    também distorções por movimento, assumindo um tom de pele padronizado.

    Ref: G. de Haan, V. Jeanne, "Robust pulse rate from chrominance-based
    rPPG", IEEE Trans. Biomed. Eng., 2013.
    """
    rgb = rgb.astype(float)
    mean = rgb.mean(axis=0)
    mean[mean == 0] = _EPS
    cn = rgb / mean  # normalização temporal por canal

    x = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
    y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]

    # O paper filtra X e Y na banda cardíaca antes de calcular alpha.
    xf = bandpass(x, fs)
    yf = bandpass(y, fs)

    alpha = xf.std() / (yf.std() + _EPS)
    return _standardize(xf - alpha * yf)


def pos(rgb: np.ndarray, fs: float, window_s: float = 1.6) -> np.ndarray:
    """POS — Plane-Orthogonal-to-Skin (Wang et al., 2017).

    Em janelas curtas (~1.6 s), projeta o RGB normalizado num plano ortogonal
    ao tom de pele (S1 = G - B, S2 = -2R + G + B), combina
    h = S1 + (std(S1)/std(S2)) * S2 e reconstrói o sinal por overlap-add.
    A janela curta torna a normalização local, o que dá mais robustez a
    movimento do que o CHROM.

    Ref: W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, "Algorithmic
    principles of remote PPG", IEEE Trans. Biomed. Eng., 2017.
    """
    rgb = rgb.astype(float)
    n = len(rgb)
    win = max(int(round(window_s * fs)), 2)
    if n < win:
        win = n

    proj = np.array([[0.0, 1.0, -1.0],
                     [-2.0, 1.0, 1.0]])  # matriz P do paper
    h = np.zeros(n)

    for start in range(n - win + 1):
        c = rgb[start:start + win]
        mu = c.mean(axis=0)
        mu[mu == 0] = _EPS
        cn = c / mu                       # normalização local da janela
        s = cn @ proj.T                   # (win, 2)
        sigma = s[:, 0].std() / (s[:, 1].std() + _EPS)
        hw = s[:, 0] + sigma * s[:, 1]
        h[start:start + win] += hw - hw.mean()   # overlap-add

    return _standardize(h)


METHODS = {"green": green, "chrom": chrom, "pos": pos}
