"""Captura de frames da webcam ou de arquivo de vídeo, com timestamps reais."""
from __future__ import annotations

import time
from typing import Iterator

import cv2
import numpy as np


def _try_set(cap: cv2.VideoCapture, prop: int, value: float, name: str) -> None:
    """Tenta configurar uma propriedade da câmera e loga se pegou."""
    ok = cap.set(prop, value)
    actual = cap.get(prop)
    status = "ok" if ok and abs(actual - value) < 1e-3 else f"ignorado (lido: {actual})"
    print(f"[capture] {name} = {value}: {status}")


class FrameSource:
    """Itera (frame_bgr, timestamp_s) a partir da webcam ou de um vídeo.

    - Webcam: timestamps via time.monotonic() no momento da leitura, porque
      o intervalo real entre frames flutua (o sinal é reamostrado depois).
    - Vídeo: timestamps do próprio arquivo (CAP_PROP_POS_MSEC), com fallback
      para indice_do_frame / fps.
    """

    def __init__(
        self,
        source: int | str,
        target_fps: float = 30.0,
        capture_size: tuple[int, int] | None = None,
    ):
        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"não consegui abrir a fonte de vídeo: {source!r}")

        if not self.is_file:
            self._configure_webcam(target_fps, capture_size)
        self._frame_idx = 0
        self._file_fps = self.cap.get(cv2.CAP_PROP_FPS) or target_fps

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _configure_webcam(
        self, target_fps: float, capture_size: tuple[int, int] | None
    ) -> None:
        """Pede fps/resolução e tenta desligar auto-exposição e auto white balance.

        Exposição/WB automáticos modulam o brilho e contaminam o sinal de
        pulso. Nem todo driver aceita (no macOS, em geral, não) — por isso
        logamos o resultado em vez de confiar nele.
        """
        if capture_size is not None:
            _try_set(self.cap, cv2.CAP_PROP_FRAME_WIDTH, capture_size[0], "largura")
            _try_set(self.cap, cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1], "altura")
        _try_set(self.cap, cv2.CAP_PROP_FPS, target_fps, "fps")
        # 0.25 = manual no backend V4L2; 0 funciona em alguns outros backends.
        _try_set(self.cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure(manual)")
        _try_set(self.cap, cv2.CAP_PROP_AUTO_WB, 0, "auto_white_balance(off)")

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            if self.is_file:
                ts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = ts_ms / 1000.0 if ts_ms > 0 else self._frame_idx / self._file_fps
            else:
                ts = time.monotonic()
            self._frame_idx += 1
            yield frame, ts

    def release(self) -> None:
        self.cap.release()
