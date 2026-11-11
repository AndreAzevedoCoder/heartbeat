"""HUD de vídeo: enquadramento 9:16, coração batendo e painéis de informação.

Só depende de OpenCV + numpy (nada de matplotlib aqui), então pode ser
desenhado dentro do loop de captura sem custo relevante.

Nota: as fontes Hershey do OpenCV não têm acentos — os textos na tela são
escritos sem acento de propósito.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Tema
# --------------------------------------------------------------------------- #
# Limiares de qualidade do sinal, em dB de SNR. Ficam aqui porque são uma
# decisão de apresentação: valores típicos de rPPG são modestos, então o HUD
# informa a qualidade pela cor e pela barra, sem interromper com avisos.
SNR_FAIR_DB = 2.0
SNR_GOOD_DB = 5.0


def snr_quality(snr_db: float | None) -> float:
    """SNR em dB -> 0..1. Usado na barra e para pesar a sincronia do coração."""
    if snr_db is None or not np.isfinite(snr_db):
        return 0.0
    return float(np.clip((snr_db + 3.0) / 11.0, 0.0, 1.0))


@dataclass(frozen=True)
class Theme:
    """Paleta do HUD, em BGR (como o OpenCV espera)."""

    accent: tuple[int, int, int] = (109, 77, 255)      # #FF4D6D - vermelho do coracao
    accent_dim: tuple[int, int, int] = (150, 120, 255)
    wave: tuple[int, int, int] = (235, 220, 255)       # rosa claro da onda
    text: tuple[int, int, int] = (247, 245, 245)
    muted: tuple[int, int, int] = (155, 150, 160)
    panel: tuple[int, int, int] = (20, 16, 22)
    good: tuple[int, int, int] = (130, 220, 150)
    warn: tuple[int, int, int] = (70, 190, 255)
    bad: tuple[int, int, int] = (85, 85, 255)
    roi: tuple[int, int, int] = (220, 230, 235)

    def quality_color(self, snr_db: float | None) -> tuple[int, int, int]:
        if snr_db is None:
            return self.muted
        if snr_db >= SNR_GOOD_DB:
            return self.good
        if snr_db >= SNR_FAIR_DB:
            return self.warn
        return self.bad

    def wave_color(self, snr_db: float | None) -> tuple[int, int, int]:
        """A onda é auto-normalizada, então a cor é quem mostra a qualidade."""
        return self.accent_dim if (snr_db or 0.0) >= SNR_FAIR_DB else self.muted


THEME = Theme()
FONT = cv2.FONT_HERSHEY_DUPLEX


# --------------------------------------------------------------------------- #
# Primitivas de desenho
# --------------------------------------------------------------------------- #
def rounded_panel(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float = 0.55,
    radius: int = 24,
) -> None:
    """Retângulo arredondado translúcido, composto in-place na ROI."""
    x, y, w, h = rect
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return

    radius = int(min(radius, w // 2, h // 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius),
                   (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    mask = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1]
    a = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    patch = np.empty_like(roi, dtype=np.float32)
    patch[:] = np.array(color, dtype=np.float32)
    roi[:] = (roi.astype(np.float32) * (1.0 - a) + patch * a).astype(np.uint8)


def text(
    img: np.ndarray,
    s: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    """Texto com sombra leve, para legibilidade sobre o vídeo."""
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 2), FONT, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(s: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(s, FONT, scale, thickness)[0][0]


def _heart_unit(n: int = 256) -> np.ndarray:
    """Contorno de coração normalizado em [-0.5, 0.5], com y para baixo.

    Curva paramétrica clássica: x = 16 sin³t, y = 13cos t - 5cos 2t
    - 2cos 3t - cos 4t.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    y = (13.0 * np.cos(t) - 5.0 * np.cos(2 * t)
         - 2.0 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.column_stack([x, -y])          # -y: coordenadas de imagem
    return pts / (np.abs(pts).max() * 2.0)


_HEART = _heart_unit()


def draw_heart(
    img: np.ndarray,
    center: tuple[int, int],
    size: float,
    color: tuple[int, int, int],
    scale: float = 1.0,
    glow: float = 0.55,
    supersample: int = 4,
) -> None:
    """Coração preenchido com borda suave e brilho ao redor.

    Desenha num patch supersampleado e reduz com INTER_AREA — é o jeito mais
    simples de ter antialiasing num polígono preenchido no OpenCV.
    """
    side = int(max(size * scale * 2.2, 8))
    ss = supersample
    mask = np.zeros((side * ss, side * ss), dtype=np.uint8)
    pts = (_HEART * size * scale * ss + side * ss / 2.0).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)

    alpha = mask.astype(np.float32) / 255.0
    if glow > 0.0:
        halo = cv2.GaussianBlur(mask, (0, 0), max(size * 0.16, 1.0))
        alpha = np.clip(alpha + halo.astype(np.float32) / 255.0 * glow, 0.0, 1.0)

    x0 = int(center[0] - side / 2)
    y0 = int(center[1] - side / 2)
    px0, py0 = max(x0, 0), max(y0, 0)
    px1 = min(x0 + side, img.shape[1])
    py1 = min(y0 + side, img.shape[0])
    if px1 <= px0 or py1 <= py0:
        return

    a = alpha[py0 - y0:py1 - y0, px0 - x0:px1 - x0][..., None]
    roi = img[py0:py1, px0:px1]
    patch = np.empty_like(roi, dtype=np.float32)
    patch[:] = np.array(color, dtype=np.float32)
    roi[:] = (roi.astype(np.float32) * (1.0 - a) + patch * a).astype(np.uint8)


def draw_waveform(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    values: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Desenha o sinal `values` dentro de `rect`, com linha de base e brilho."""
    x, y, w, h = rect
    baseline = y + h // 2
    cv2.line(img, (x, baseline), (x + w, baseline), (70, 66, 74), 1, cv2.LINE_AA)

    v = np.asarray(values, dtype=np.float64)
    if v.size < 4 or not np.isfinite(v).all():
        return
    amp = np.abs(v).max()
    if amp < 1e-9:
        return

    # subamostra para no máximo ~1 ponto por pixel de largura
    if v.size > w:
        idx = np.linspace(0, v.size - 1, w).astype(int)
        v = v[idx]
    xs = np.linspace(x, x + w - 1, v.size)
    ys = baseline - v / amp * (h / 2 - 2)
    pts = np.column_stack([xs, ys]).astype(np.int32)

    cv2.polylines(img, [pts], False, color, thickness + 3, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, (255, 255, 255), max(thickness - 1, 1),
                  cv2.LINE_AA)


def draw_meter(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    frac: float,
    color: tuple[int, int, int],
) -> None:
    """Barra horizontal arredondada preenchida em `frac` (0..1)."""
    x, y, w, h = rect
    frac = float(np.clip(frac, 0.0, 1.0))
    rounded_panel(img, rect, (90, 86, 96), alpha=0.75, radius=h // 2)
    if frac > 0.02:
        rounded_panel(img, (x, y, max(int(w * frac), h), h), color,
                      alpha=0.95, radius=h // 2)


def draw_rois(
    img: np.ndarray,
    polygons: dict[str, np.ndarray],
    theme: Theme = THEME,
    alpha: float = 0.45,
) -> None:
    """Contorno das ROIs, discreto para não poluir o vídeo."""
    if not polygons:
        return
    overlay = img.copy()
    for poly in polygons.values():
        cv2.polylines(overlay, [poly.astype(np.int32)], True, theme.roi, 1,
                      cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


# --------------------------------------------------------------------------- #
# Batimento do coração
# --------------------------------------------------------------------------- #
def beat_envelope(phase: float) -> float:
    """Escala do coração em função da fase (0..1): dois "thumps" por ciclo.

    Imita sístole + fechamento das válvulas, que é o que faz a animação
    parecer um coração de verdade em vez de uma pulsação senoidal.
    """
    p = phase % 1.0
    d = min(p, 1.0 - p)                          # distância circular até 0
    main = np.exp(-(d / 0.075) ** 2)
    second = np.exp(-((p - 0.33) / 0.075) ** 2)
    return float(1.0 + 0.34 * main + 0.13 * second)


class HeartBeat:
    """Fase do batimento: avança com o BPM e se engancha na fase do sinal.

    O acumulador sozinho é estável mas "solto"; a fase instantânea medida no
    sinal de pulso é precisa mas ruidosa. Um PLL simples (nudge com ganho
    pequeno) dá o melhor dos dois: o coração bate junto com o pulso real.
    """

    def __init__(self, gain: float = 0.12, fallback_bpm: float = 60.0):
        self.phase = 0.0
        self.gain = gain
        self.fallback_bpm = fallback_bpm

    def advance(
        self,
        dt: float,
        bpm: float | None,
        measured_phase: float | None = None,
        gain: float | None = None,
    ) -> float:
        """Avança `dt` segundos e devolve a escala atual do coração.

        `gain` sobrescreve o ganho do engate neste passo — serve para confiar
        menos na fase medida quando o sinal está ruim.
        """
        dt = float(np.clip(dt, 0.0, 0.2))        # ignora travadas longas
        freq = (bpm if bpm else self.fallback_bpm) / 60.0
        self.phase = (self.phase + freq * dt) % 1.0

        if measured_phase is not None:
            k = self.gain if gain is None else float(np.clip(gain, 0.0, 1.0))
            err = (measured_phase - self.phase + 0.5) % 1.0 - 0.5
            self.phase = (self.phase + k * err) % 1.0

        return beat_envelope(self.phase)

    @property
    def scale(self) -> float:
        return beat_envelope(self.phase)


# --------------------------------------------------------------------------- #
# Enquadramento 9:16
# --------------------------------------------------------------------------- #
class Transform:
    """Mapeia pontos do frame original para o frame recortado/redimensionado."""

    def __init__(self, origin: tuple[float, float], scale: float):
        self.origin = origin
        self.scale = scale

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return (pts - np.array(self.origin)) * self.scale

    def apply_dict(self, polys: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: self(v) for k, v in polys.items()}


class PortraitFramer:
    """Recorta o frame em 9:16 seguindo o rosto e redimensiona para a saída.

    O recorte segue o centro do rosto com suavização exponencial, para o
    enquadramento não tremer a cada frame. `zoom` > 1 aproxima.
    """

    def __init__(
        self,
        out_size: tuple[int, int] = (540, 960),
        follow: bool = True,
        smooth: float = 0.12,
        zoom: float = 1.0,
    ):
        self.out_w, self.out_h = out_size
        self.follow = follow
        self.smooth = smooth
        self.zoom = max(zoom, 1.0)
        self._center: np.ndarray | None = None

    @property
    def aspect(self) -> float:
        return self.out_w / self.out_h

    def apply(
        self, frame: np.ndarray, face_center: np.ndarray | None = None
    ) -> tuple[np.ndarray, Transform]:
        h, w = frame.shape[:2]

        crop_h = min(float(h), w / self.aspect) / self.zoom
        crop_w = crop_h * self.aspect

        target = np.array([w / 2.0, h / 2.0])
        if self.follow and face_center is not None:
            # um pouco abaixo do centro do rosto: deixa espaço para o HUD
            target = np.asarray(face_center, dtype=np.float64) + np.array([0.0, crop_h * 0.06])
        if self._center is None:
            self._center = target
        else:
            self._center = self._center + self.smooth * (target - self._center)

        cx = float(np.clip(self._center[0], crop_w / 2, w - crop_w / 2))
        cy = float(np.clip(self._center[1], crop_h / 2, h - crop_h / 2))
        x0, y0 = cx - crop_w / 2, cy - crop_h / 2

        crop = frame[int(round(y0)):int(round(y0 + crop_h)),
                     int(round(x0)):int(round(x0 + crop_w))]
        interp = cv2.INTER_AREA if crop.shape[0] > self.out_h else cv2.INTER_LINEAR
        out = cv2.resize(crop, (self.out_w, self.out_h), interpolation=interp)

        scale = self.out_h / crop.shape[0]
        return out, Transform((round(x0), round(y0)), scale)


# --------------------------------------------------------------------------- #
# HUD completo
# --------------------------------------------------------------------------- #
@dataclass
class HudState:
    """Tudo que o HUD precisa saber para um frame."""

    method: str
    bpm: float | None = None
    snr_db: float | None = None
    heart_scale: float = 1.0
    pulse: np.ndarray | None = None
    face_found: bool = True
    recording: bool = False
    rec_seconds: float = 0.0
    calibrating: float | None = None   # 0..1 enquanto a janela enche
    warning: str | None = None
    fps: float | None = None


def draw_hud(
    frame: np.ndarray,
    polygons: dict[str, np.ndarray],
    state: HudState,
    theme: Theme = THEME,
) -> np.ndarray:
    """Desenha o HUD inteiro sobre `frame` (in-place) e devolve o frame."""
    h, w = frame.shape[:2]
    s = h / 960.0                       # fator de escala do layout
    m = int(24 * s)

    draw_rois(frame, polygons, theme)

    # --- chips do topo --------------------------------------------------- #
    chip_h = int(42 * s)
    label = f"metodo  {state.method.upper()}"
    chip_w = text_width(label, 0.55 * s, 1) + int(32 * s)
    rounded_panel(frame, (m, m, chip_w, chip_h), theme.panel, 0.55, int(chip_h / 2))
    text(frame, label, (m + int(16 * s), m + int(chip_h * 0.68)), 0.55 * s,
         theme.text, 1)

    if state.fps is not None:
        fps_txt = f"{state.fps:.0f} fps"
        fw = text_width(fps_txt, 0.5 * s, 1) + int(28 * s)
        rounded_panel(frame, (m + chip_w + int(8 * s), m, fw, chip_h),
                      theme.panel, 0.45, int(chip_h / 2))
        text(frame, fps_txt, (m + chip_w + int(22 * s), m + int(chip_h * 0.68)),
             0.5 * s, theme.muted, 1)

    if state.recording:
        rec_txt = f"REC  {int(state.rec_seconds // 60):01d}:{int(state.rec_seconds % 60):02d}"
        rw = text_width(rec_txt, 0.55 * s, 1) + int(54 * s)
        rx = w - m - rw
        rounded_panel(frame, (rx, m, rw, chip_h), theme.panel, 0.6, int(chip_h / 2))
        cv2.circle(frame, (rx + int(22 * s), m + chip_h // 2), int(7 * s),
                   theme.bad, -1, cv2.LINE_AA)
        text(frame, rec_txt, (rx + int(38 * s), m + int(chip_h * 0.68)), 0.55 * s,
             theme.text, 1)

    # --- aviso centralizado ---------------------------------------------- #
    if state.warning:
        tw = text_width(state.warning, 0.62 * s, 1)
        bw, bh = tw + int(44 * s), int(52 * s)
        bx, by = (w - bw) // 2, int(h * 0.16)
        rounded_panel(frame, (bx, by, bw, bh), theme.panel, 0.66, int(bh / 2))
        color = theme.bad if not state.face_found else theme.warn
        text(frame, state.warning, (bx + int(22 * s), by + int(bh * 0.66)),
             0.62 * s, color, 1)

    # --- cartão inferior -------------------------------------------------- #
    card_h = int(320 * s)
    card_w = w - 2 * m
    card_y = h - card_h - m
    rounded_panel(frame, (m, card_y, card_w, card_h), theme.panel, 0.62,
                  int(34 * s))

    # coração batendo: o brilho pulsa junto com a escala, senão a animação
    # quase não se nota depois que o vídeo é comprimido
    heart_size = 112 * s
    heart_cx = m + int(100 * s)
    heart_cy = card_y + int(100 * s)
    live = state.bpm is not None
    beat = state.heart_scale if live else 1.0
    glow = 0.30 + 0.60 * float(np.clip((beat - 1.0) / 0.34, 0.0, 1.0)) if live else 0.18
    draw_heart(frame, (heart_cx, heart_cy), heart_size, theme.accent,
               scale=beat, glow=glow)

    # número do BPM
    num_x = m + int(196 * s)
    num_baseline = card_y + int(118 * s)
    if state.bpm is not None:
        num = f"{state.bpm:.0f}"
        num_scale = 2.7 * s
        text(frame, num, (num_x, num_baseline), num_scale, theme.text, 2)
        nw = text_width(num, num_scale, 2)
        text(frame, "bpm", (num_x + nw + int(14 * s), num_baseline), 0.78 * s,
             theme.muted, 1)
    else:
        # placeholder discreto; o traço do Hershey fica na meia-altura, então
        # desce um pouco para ficar alinhado com o rótulo
        text(frame, "--", (num_x, num_baseline - int(16 * s)), 2.0 * s,
             theme.muted, 2)
        nw = text_width("--", 2.0 * s, 2)
        text(frame, "bpm", (num_x + nw + int(14 * s), num_baseline), 0.78 * s,
             theme.muted, 1)

    # qualidade do sinal
    qcolor = theme.quality_color(state.snr_db)
    meter_rect = (num_x, card_y + int(166 * s), int(230 * s), int(10 * s))
    if state.calibrating is not None:
        text(frame, f"calibrando  {int(state.calibrating * 100)}%",
             (num_x, card_y + int(152 * s)), 0.52 * s, theme.muted, 1)
        draw_meter(frame, meter_rect, state.calibrating, theme.accent_dim)
    else:
        snr_txt = f"SNR {state.snr_db:.1f} dB" if state.snr_db is not None else "SNR ---"
        text(frame, snr_txt, (num_x, card_y + int(152 * s)), 0.52 * s, qcolor, 1)
        draw_meter(frame, meter_rect, snr_quality(state.snr_db), qcolor)

    # onda do pulso
    wave_rect = (m + int(24 * s), card_y + int(206 * s),
                 card_w - int(48 * s), int(92 * s))
    if state.pulse is not None:
        draw_waveform(frame, wave_rect, state.pulse, theme.wave_color(state.snr_db),
                      max(int(2 * s), 2))
    else:
        text(frame, "aguardando sinal...",
             (wave_rect[0], wave_rect[1] + wave_rect[3] // 2), 0.5 * s,
             theme.muted, 1)

    return frame
