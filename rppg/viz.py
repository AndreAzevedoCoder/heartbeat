"""Plots matplotlib ao vivo (o HUD desenhado no vídeo mora em `rppg/ui.py`)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class LivePlotter:
    """Figura matplotlib atualizada dentro do loop do vídeo, sem bloquear.

    Usa modo interativo (plt.ion) e redesenha só quando `update` é chamado
    (uma vez por estimativa, ~1 Hz), então o custo é baixo.
    """

    def __init__(self, band: tuple[float, float]):
        plt.ion()
        self.band = band
        self.fig, (self.ax_rgb, self.ax_pulse, self.ax_spec) = plt.subplots(
            3, 1, figsize=(7, 7), constrained_layout=True
        )
        self.fig.canvas.manager.set_window_title("rPPG - sinais")

        self.lines_rgb = [
            self.ax_rgb.plot([], [], color=c, lw=0.8, label=l)[0]
            for c, l in zip("rgb", "RGB")
        ]
        self.ax_rgb.set_title("Sinal RGB bruto (media da ROI)")
        self.ax_rgb.legend(loc="upper right", fontsize=8)

        (self.line_pulse,) = self.ax_pulse.plot([], [], color="tab:green", lw=1.0)
        self.ax_pulse.set_title("Pulso filtrado")

        (self.line_spec,) = self.ax_spec.plot([], [], color="tab:blue", lw=1.0)
        (self.peak_marker,) = self.ax_spec.plot([], [], "rv", ms=8)
        self.ax_spec.set_title("Espectro")
        self.ax_spec.set_xlabel("bpm")
        self.ax_spec.set_xlim(band[0] * 60, band[1] * 60)

        self.fig.show()

    def update(
        self,
        t: np.ndarray,
        rgb: np.ndarray,
        t_pulse: np.ndarray,
        pulse: np.ndarray,
        freqs: np.ndarray,
        power: np.ndarray,
        peak_freq: float,
    ) -> None:
        t0 = t[0]
        for i, line in enumerate(self.lines_rgb):
            line.set_data(t - t0, rgb[:, i])
        self.ax_rgb.relim(); self.ax_rgb.autoscale_view()

        self.line_pulse.set_data(t_pulse - t0, pulse)
        self.ax_pulse.relim(); self.ax_pulse.autoscale_view()

        p = power / (power.max() + 1e-12)
        self.line_spec.set_data(freqs * 60.0, p)
        self.peak_marker.set_data([peak_freq * 60.0], [1.0])
        self.ax_spec.set_ylim(0, 1.1)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        plt.close(self.fig)
