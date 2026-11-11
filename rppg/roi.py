"""Detecção de ROIs de pele (testa e bochechas) com o MediaPipe Face Landmarker.

Usa a Tasks API nova (`mediapipe.tasks.python.vision.FaceLandmarker`) em modo
VIDEO. O modelo `face_landmarker.task` é baixado automaticamente na primeira
execução.

Os índices abaixo referem-se à malha de 478 pontos do Face Mesh. Os polígonos
foram escolhidos para pegar pele contínua, evitando olhos, sobrancelhas e boca.
Para experimentar outras regiões, basta editar `ROI_LANDMARKS`.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Polígonos (sequências fechadas de índices de landmarks) por ROI.
ROI_LANDMARKS: dict[str, list[int]] = {
    # Testa: do topo da linha do cabelo até logo acima das sobrancelhas.
    "forehead": [
        10, 338, 297, 332, 284,          # linha superior (oval do rosto, lado direito)
        300, 293, 334, 296, 336,         # topo da sobrancelha direita
        9,                               # glabela (entre as sobrancelhas)
        107, 66, 105, 63, 70,            # topo da sobrancelha esquerda
        54, 103, 67, 109,                # linha superior, lado esquerdo, de volta ao 10
    ],
    # Bochecha esquerda (lado esquerdo da imagem sem espelhar).
    "left_cheek": [117, 118, 119, 100, 142, 203, 205, 187, 147, 123],
    # Bochecha direita (espelho da esquerda).
    "right_cheek": [346, 347, 348, 329, 371, 423, 425, 411, 376, 352],
}


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Baixa o face_landmarker.task se ainda não existir."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[roi] baixando modelo Face Landmarker para {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("[roi] modelo baixado.")
    return path


@dataclass
class FaceObservation:
    """Resultado da detecção num frame.

    - means: {roi: array (3,) com a média (R, G, B)} — NaN se a máscara ficou vazia;
    - polygons: {roi: array (K, 2) em pixels} para desenhar;
    - center: centro do rosto em pixels (usado para enquadrar em 9:16);
    - bbox: (x0, y0, x1, y1) do rosto.
    """

    means: dict[str, np.ndarray]
    polygons: dict[str, np.ndarray]
    center: np.ndarray
    bbox: tuple[float, float, float, float]


class FaceROIExtractor:
    """Detecta o rosto e extrai a média RGB de cada ROI por frame."""

    def __init__(self, rois: tuple[str, ...], model_path: Path = MODEL_PATH):
        unknown = set(rois) - set(ROI_LANDMARKS)
        if unknown:
            raise ValueError(f"ROIs desconhecidas: {unknown}")
        self.rois = tuple(rois)

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(
        self, frame_bgr: np.ndarray, timestamp_s: float
    ) -> FaceObservation | None:
        """Processa um frame e devolve a observação, ou None se não houver rosto."""
        ts_ms = int(timestamp_s * 1000)
        if ts_ms <= self._last_ts_ms:            # o modo VIDEO exige ts crescente
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        landmarks = result.face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float64)

        means: dict[str, np.ndarray] = {}
        polygons: dict[str, np.ndarray] = {}
        for roi in self.rois:
            poly = pts[ROI_LANDMARKS[roi]].astype(np.int32)
            polygons[roi] = poly
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            if cv2.countNonZero(mask) == 0:
                means[roi] = np.full(3, np.nan)
                continue
            b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
            means[roi] = np.array([r, g, b])

        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return FaceObservation(
            means=means,
            polygons=polygons,
            center=pts.mean(axis=0),
            bbox=(float(x0), float(y0), float(x1), float(y1)),
        )

    def close(self) -> None:
        self._landmarker.close()
