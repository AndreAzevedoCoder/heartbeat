"""rPPG ao vivo: estima batimentos cardíacos pela webcam (ou arquivo de vídeo).

A saída é enquadrada em 9:16 (vídeo vertical) com um HUD que mostra o BPM, a
qualidade do sinal, a onda do pulso e um coração que bate em sincronia com o
pulso detectado.

Uso:
    python main.py                       # webcam, 9:16
    python main.py --video clip.avi      # arquivo (ex.: UBFC-rPPG)
    python main.py --method chrom --rois forehead left_cheek
    python main.py --output-size 1080 1920 --zoom 1.2   # exportar bonito
    python main.py --no-portrait --plot  # quadro cheio + plots matplotlib

Teclas na janela de vídeo:
    1/2/3  troca o método (GREEN / CHROM / POS)
    r      começa/para a gravação (salva um .npz ao parar)
    q      sai
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from datetime import datetime

import cv2
import numpy as np

from config import DEFAULT, Config
from rppg.capture import FrameSource
from rppg.methods import METHODS
from rppg.roi import ROI_LANDMARKS, FaceObservation, FaceROIExtractor
from rppg.signal_processing import extract_pulse, process_window, pulse_phase
from rppg.ui import HeartBeat, HudState, PortraitFramer, draw_hud, snr_quality
from rppg.viz import LivePlotter

KEY_TO_METHOD = {ord("1"): "green", ord("2"): "chrom", ord("3"): "pos"}


def parse_args() -> tuple[Config, str | None]:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=str, default=None,
                   help="arquivo de vídeo em vez da webcam")
    p.add_argument("--camera", type=int, default=DEFAULT.camera_index)
    p.add_argument("--method", choices=list(METHODS), default=DEFAULT.method)
    p.add_argument("--rois", nargs="+", choices=list(ROI_LANDMARKS),
                   default=list(DEFAULT.rois))
    p.add_argument("--window", type=float, default=DEFAULT.window_s,
                   help="janela de análise em segundos")
    p.add_argument("--band", type=float, nargs=2, default=list(DEFAULT.band),
                   metavar=("LOW", "HIGH"), help="banda em Hz")
    p.add_argument("--fps", type=float, default=DEFAULT.target_fps)
    p.add_argument("--output-size", type=int, nargs=2,
                   default=list(DEFAULT.output_size), metavar=("W", "H"),
                   help="tamanho da janela/saída (padrão 9:16)")
    p.add_argument("--zoom", type=float, default=DEFAULT.zoom,
                   help=">1 aproxima o recorte no rosto")
    p.add_argument("--no-portrait", action="store_true",
                   help="não recortar em 9:16 (mostra o quadro inteiro)")
    p.add_argument("--no-follow", action="store_true",
                   help="recorte fixo no centro, sem seguir o rosto")
    p.add_argument("--plot", action="store_true",
                   help="abre também a figura matplotlib ao vivo")
    p.add_argument("--save", type=str, default=None, metavar="OUT.mp4",
                   help="grava o vídeo 9:16 com o HUD neste arquivo")
    p.add_argument("--duration", type=float, default=None, metavar="SEG",
                   help="encerra sozinho depois de N segundos")
    p.add_argument("--no-window", action="store_true",
                   help="não abre janela (para gravar sem GUI)")
    args = p.parse_args()

    cfg = replace(
        DEFAULT,
        camera_index=args.camera,
        method=args.method,
        rois=tuple(args.rois),
        window_s=args.window,
        band=tuple(args.band),
        target_fps=args.fps,
        resample_fs=args.fps,
        output_size=tuple(args.output_size),
        zoom=args.zoom,
        portrait=not args.no_portrait,
        follow_face=not args.no_follow,
        show_plot=args.plot,
        save_path=args.save,
        duration_s=args.duration,
        show_window=not args.no_window,
    )
    return cfg, args.video


class Recorder:
    """Acumula (t, RGB por ROI) enquanto grava e salva num .npz."""

    def __init__(self, rois: tuple[str, ...]):
        self.rois = rois
        self.active = False
        self.started_at = 0.0
        self._t: list[float] = []
        self._rgb: dict[str, list[np.ndarray]] = {}

    def toggle(self, cfg: Config, now: float) -> str | None:
        if not self.active:
            self.active = True
            self.started_at = now
            self._t, self._rgb = [], {r: [] for r in self.rois}
            print("[rec] gravando...")
            return None
        self.active = False
        if not self._t:
            print("[rec] nada gravado.")
            return None
        path = f"rec_{datetime.now():%Y%m%d_%H%M%S}.npz"
        np.savez(
            path,
            t=np.array(self._t),
            rois=np.array(self.rois),
            band=np.array(cfg.band),
            **{f"rgb_{r}": np.array(v) for r, v in self._rgb.items()},
        )
        print(f"[rec] salvo em {path} ({len(self._t)} amostras)")
        return path

    def append(self, t: float, obs: FaceObservation | None) -> None:
        if not self.active:
            return
        self._t.append(t)
        for r in self.rois:
            v = obs.means.get(r) if obs else None
            self._rgb[r].append(v if v is not None else np.full(3, np.nan))

    def seconds(self, now: float) -> float:
        return now - self.started_at if self.active else 0.0


def main() -> None:
    cfg, video = parse_args()
    source = FrameSource(video if video else cfg.camera_index, cfg.target_fps,
                         cfg.capture_size)
    print(f"[main] resolução de captura: {source.size[0]}x{source.size[1]}")

    extractor = FaceROIExtractor(cfg.rois)
    plotter = LivePlotter(cfg.band) if cfg.show_plot else None
    recorder = Recorder(cfg.rois)
    framer = PortraitFramer(cfg.output_size, cfg.follow_face, cfg.frame_smooth,
                            cfg.zoom) if cfg.portrait else None
    heart = HeartBeat()

    method = cfg.method
    buf_len = int(cfg.resample_fs * (cfg.window_s + 5))
    buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=buf_len)  # (t, RGB médio das ROIs)

    writer: cv2.VideoWriter | None = None
    bpm_history: deque[float] = deque(maxlen=cfg.bpm_median_n)
    frame_times: deque[float] = deque(maxlen=30)
    first_ts: float | None = None
    bpm_shown: float | None = None
    snr_db: float | None = None
    display_pulse: np.ndarray | None = None
    last_update = 0.0
    prev_ts: float | None = None
    mirror = cfg.mirror and video is None

    for frame, ts in source.frames():
        if mirror:
            # espelha antes de detectar: assim os polígonos já saem nas
            # coordenadas que vão para a tela, sem conversão depois
            frame = cv2.flip(frame, 1)

        dt = 0.0 if prev_ts is None else ts - prev_ts
        prev_ts = ts
        if first_ts is None:
            first_ts = ts
        if dt > 0:
            frame_times.append(dt)

        obs = extractor.process(frame, ts)

        warning = None
        if obs is None:
            warning = "rosto nao detectado"
        else:
            stacked = np.array([obs.means[r] for r in cfg.rois])
            combined = np.nanmean(stacked, axis=0)  # média das ROIs habilitadas
            if np.isfinite(combined).all():
                buffer.append((ts, combined))
        recorder.append(ts, obs)  # sem rosto vira NaN no .npz

        t_arr = np.array([b[0] for b in buffer]) if buffer else np.empty(0)
        span = (t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 0.0
        rgb_arr = np.array([b[1] for b in buffer]) if buffer else np.empty((0, 3))

        # Onda do HUD: recalculada a cada frame numa janela curta, para a
        # animação ficar contínua (o BPM é mais caro e roda a ~1 Hz).
        if span >= 2.0:
            sel = t_arr >= t_arr[-1] - cfg.display_window_s
            try:
                _, display_pulse = extract_pulse(
                    t_arr[sel], rgb_arr[sel], METHODS[method],
                    cfg.resample_fs, cfg.band, cfg.filter_order,
                )
            except ValueError:
                display_pulse = None

        # Estimativa periódica de BPM na janela longa
        if span >= cfg.min_window_s and (ts - last_update) >= cfg.update_every_s:
            last_update = ts
            sel = t_arr >= t_arr[-1] - cfg.window_s
            try:
                t_uni, pulse, est = process_window(
                    t_arr[sel], rgb_arr[sel], METHODS[method],
                    cfg.resample_fs, cfg.band, cfg.filter_order, cfg.nfft_factor,
                )
            except ValueError:
                est = None
            if est is not None:
                bpm_history.append(est.bpm)
                bpm_shown = float(np.median(bpm_history))
                snr_db = est.snr_db
                if plotter is not None:
                    plotter.update(t_arr[sel], rgb_arr[sel], t_uni, pulse,
                                   est.freqs, est.power, est.peak_freq)

        # Coração: avança com o BPM e se engancha na fase do pulso real.
        # O quanto ele confia na fase medida acompanha a qualidade do sinal —
        # com SNR baixo o engate fica suave em vez de desligar, senão a
        # sincronia quase nunca valeria na prática.
        measured, gain = None, None
        if cfg.heart_sync and display_pulse is not None and bpm_shown is not None:
            phase = pulse_phase(display_pulse, cfg.resample_fs, bpm_shown / 60.0)
            if np.isfinite(phase):
                measured = phase
                gain = cfg.heart_sync_gain * (0.2 + 0.8 * snr_quality(snr_db))
        heart_scale = heart.advance(dt, bpm_shown, measured, gain)

        # Enquadramento 9:16 (os polígonos acompanham o recorte)
        polygons = obs.polygons if obs else {}
        if framer is not None:
            display, transform = framer.apply(frame, obs.center if obs else None)
            polygons = transform.apply_dict(polygons)
        else:
            display = frame.copy()

        calibrating = None
        if bpm_shown is None:
            calibrating = float(np.clip(span / cfg.min_window_s, 0.0, 1.0))

        state = HudState(
            method=method,
            bpm=bpm_shown,
            snr_db=snr_db,
            heart_scale=heart_scale,
            pulse=display_pulse,
            face_found=obs is not None,
            recording=recorder.active,
            rec_seconds=recorder.seconds(ts),
            calibrating=calibrating,
            warning=warning,
            fps=(1.0 / np.mean(frame_times)) if frame_times else None,
        )
        draw_hud(display, polygons, state)

        if cfg.save_path:
            if writer is None:
                writer = cv2.VideoWriter(
                    cfg.save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    cfg.target_fps, (display.shape[1], display.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"não consegui gravar em {cfg.save_path}")
                print(f"[main] gravando vídeo em {cfg.save_path}")
            writer.write(display)

        if cfg.duration_s is not None and ts - first_ts >= cfg.duration_s:
            break

        if not cfg.show_window:
            continue

        cv2.imshow("rPPG", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEY_TO_METHOD:
            method = KEY_TO_METHOD[key]
            bpm_history.clear()
            display_pulse = None
            print(f"[main] método: {method.upper()}")
        if key == ord("r"):
            recorder.toggle(cfg, ts)

    if recorder.active:
        recorder.toggle(cfg, prev_ts or 0.0)  # salva o que ficou pendente
    if writer is not None:
        writer.release()
        print(f"[main] vídeo salvo em {cfg.save_path}")
    source.release()
    extractor.close()
    if plotter is not None:
        plotter.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
