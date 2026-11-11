"""Análise offline de uma gravação .npz: compara GREEN, CHROM e POS.

Uso:
    python analyze.py rec_20260101_120000.npz
    python analyze.py rec.npz --rois forehead --ref oximetro.csv

O CSV de referência deve ter duas colunas: timestamp (em segundos, na mesma
base de tempo da gravação ou relativo ao início) e bpm. Com referência, imprime
MAE e RMSE de cada método.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT
from rppg.methods import METHODS
from rppg.signal_processing import (
    bandpass,
    detrend,
    estimate_bpm,
    resample_uniform,
)


def load_recording(path: str, rois: list[str] | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega o .npz e devolve (t, rgb combinado (N,3), rois usadas)."""
    data = np.load(path, allow_pickle=False)
    available = [k[len("rgb_"):] for k in data.files if k.startswith("rgb_")]
    use = rois or available
    missing = set(use) - set(available)
    if missing:
        raise SystemExit(f"ROIs {missing} não existem no arquivo (tem: {available})")

    t = data["t"]
    stacked = np.stack([data[f"rgb_{r}"] for r in use])  # (R, N, 3)
    rgb = np.nanmean(stacked, axis=0)
    return t, rgb, use


def bpm_over_time(
    t: np.ndarray,
    rgb: np.ndarray,
    method_fn,
    fs: float,
    window_s: float,
    step_s: float,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimativas de BPM e SNR em janelas deslizantes. Retorna (t, bpm, snr)."""
    t_uni, rgb_uni = resample_uniform(t, rgb, fs)
    win = int(window_s * fs)
    step = max(int(step_s * fs), 1)
    ts, bpms, snrs = [], [], []
    for start in range(0, len(t_uni) - win + 1, step):
        seg = rgb_uni[start:start + win]
        pulse = bandpass(detrend(method_fn(seg, fs)), fs, band)
        est = estimate_bpm(pulse, fs, band)
        ts.append(t_uni[start + win - 1])
        bpms.append(est.bpm)
        snrs.append(est.snr_db)
    return np.array(ts), np.array(bpms), np.array(snrs)


def load_reference(path: str, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """CSV (timestamp, bpm). Se os tempos parecerem relativos, soma t0."""
    ref = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    cols = ref.dtype.names
    rt, rbpm = ref[cols[0]], ref[cols[1]]
    if rt[0] < t0 - 1.0:          # heurística: referência em tempo relativo
        rt = rt + t0
    return rt, rbpm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", help="gravação .npz feita pelo main.py")
    p.add_argument("--rois", nargs="+", default=None,
                   help="ROIs a combinar (padrão: todas as gravadas)")
    p.add_argument("--ref", type=str, default=None,
                   help="CSV de referência: timestamp,bpm")
    p.add_argument("--window", type=float, default=DEFAULT.window_s)
    p.add_argument("--step", type=float, default=DEFAULT.update_every_s)
    p.add_argument("--fs", type=float, default=DEFAULT.resample_fs)
    args = p.parse_args()

    band = DEFAULT.band
    t, rgb, rois = load_recording(args.npz, args.rois)
    print(f"[analyze] {args.npz}: {len(t)} amostras, "
          f"{t[-1] - t[0]:.1f} s, ROIs: {', '.join(rois)}")

    ref = load_reference(args.ref, t[0]) if args.ref else None

    fig, axes = plt.subplots(len(METHODS), 3, figsize=(15, 9), constrained_layout=True)
    fig.canvas.manager.set_window_title(f"rPPG offline - {args.npz}")

    t_uni, rgb_uni = resample_uniform(t, rgb, args.fs)
    for row, (name, fn) in enumerate(METHODS.items()):
        pulse = bandpass(detrend(fn(rgb_uni, args.fs)), args.fs, band)
        est = estimate_bpm(pulse, args.fs, band)
        bt, bpms, snrs = bpm_over_time(t, rgb, fn, args.fs, args.window,
                                       args.step, band)

        ax = axes[row, 0]
        ax.plot(t_uni - t_uni[0], pulse, lw=0.7)
        ax.set_ylabel(name.upper())
        ax.set_title("pulso filtrado" if row == 0 else "")

        ax = axes[row, 1]
        ax.plot(est.freqs * 60, est.power / est.power.max(), lw=0.9)
        ax.axvline(est.bpm, color="r", ls="--", lw=0.8)
        ax.set_title("espectro (sinal inteiro)" if row == 0 else "")
        ax.set_xlabel("bpm" if row == len(METHODS) - 1 else "")

        ax = axes[row, 2]
        ax.plot(bt - t[0], bpms, "o-", ms=3, lw=0.9, label="estimado")
        ax.set_title("BPM na janela deslizante" if row == 0 else "")
        ax.set_xlabel("tempo (s)" if row == len(METHODS) - 1 else "")

        line = f"{name.upper():6s}: pico global {est.bpm:6.1f} bpm, SNR {est.snr_db:5.1f} dB"
        if ref is not None:
            rt, rbpm = ref
            ref_interp = np.interp(bt, rt, rbpm)
            err = bpms - ref_interp
            mae = np.abs(err).mean()
            rmse = np.sqrt((err ** 2).mean())
            line += f" | MAE {mae:5.2f} bpm, RMSE {rmse:5.2f} bpm"
            ax.plot(rt - t[0], rbpm, "k--", lw=1.0, label="referência")
            ax.legend(fontsize=8)
        print(line)

    plt.show()


if __name__ == "__main__":
    main()
