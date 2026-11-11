"""Configuração central do pipeline de rPPG.

Tudo que vale a pena experimentar está aqui: janela de análise, banda de
frequência, ROIs, método padrão etc. `main.py` e `analyze.py` sobrescrevem
esses valores via argparse.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- captura ---
    camera_index: int = 0
    target_fps: float = 30.0          # fps pedido à câmera (os timestamps reais mandam)
    resample_fs: float = 30.0         # taxa uniforme usada antes da FFT
    capture_size: tuple[int, int] = (1280, 720)  # pedido à câmera; quanto maior, melhor o recorte 9:16

    # --- ROIs ---
    # subconjunto de {"forehead", "left_cheek", "right_cheek"}
    rois: tuple[str, ...] = ("forehead", "left_cheek", "right_cheek")

    # --- extração do pulso ---
    method: str = "pos"               # "green" | "chrom" | "pos"
    pos_window_s: float = 1.6         # janela do overlap-add do POS (Wang et al. 2017)

    # --- filtragem ---
    band: tuple[float, float] = (0.7, 4.0)   # Hz -> 42 a 240 bpm
    filter_order: int = 3

    # --- estimativa de BPM ---
    window_s: float = 10.0            # janela deslizante de análise
    update_every_s: float = 1.0       # intervalo entre estimativas
    min_window_s: float = 5.0         # mínimo de sinal antes da 1ª estimativa
    nfft_factor: int = 8              # zero-padding da FFT (x tamanho do sinal)
    bpm_median_n: int = 5             # mediana das últimas N estimativas exibida

    # --- visualização ---
    show_plot: bool = False           # figura matplotlib ao vivo (pesada; ligue com --plot)
    mirror: bool = True               # espelhar preview da webcam

    # --- enquadramento / HUD ---
    portrait: bool = True             # recorta a saída em 9:16 (vídeo vertical)
    output_size: tuple[int, int] = (540, 960)  # 9:16; use (1080, 1920) para exportar
    follow_face: bool = True          # recorte acompanha o rosto
    frame_smooth: float = 0.12        # suavização do enquadramento (0=travado, 1=sem suavizar)
    zoom: float = 1.0                 # >1 aproxima o recorte
    heart_sync: bool = True           # engancha o coração na fase do pulso medido
    heart_sync_gain: float = 0.12     # força do engate (ponderada pela qualidade do sinal)
    display_window_s: float = 6.0     # trecho do pulso mostrado na onda do HUD

    # --- saída ---
    show_window: bool = True          # abre a janela OpenCV
    save_path: str | None = None      # grava o vídeo renderizado (mp4)
    duration_s: float | None = None   # encerra sozinho depois de N segundos

    @property
    def bpm_band(self) -> tuple[float, float]:
        low, high = self.band
        return low * 60.0, high * 60.0


DEFAULT = Config()
