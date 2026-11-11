"""Testes do HUD: enquadramento 9:16, animação do coração e desenho completo."""
from __future__ import annotations

import numpy as np
import pytest

from rppg.signal_processing import bandpass, detrend, pulse_phase
from rppg.ui import (
    THEME,
    HeartBeat,
    HudState,
    PortraitFramer,
    beat_envelope,
    draw_heart,
    draw_hud,
    snr_quality,
)
from tests.synthetic import HR_HZ, synthetic_rgb

FS = 30.0


def _frame(w: int = 1280, h: int = 720) -> np.ndarray:
    """Frame cinza com um gradiente, para dar pra ver se algo foi desenhado."""
    g = np.linspace(40, 180, w, dtype=np.uint8)
    return np.repeat(g[None, :], h, axis=0)[:, :, None].repeat(3, axis=2).copy()


# --- enquadramento ---------------------------------------------------------- #
def test_framer_outputs_9x16():
    framer = PortraitFramer(out_size=(540, 960))
    out, _ = framer.apply(_frame())
    assert out.shape[:2] == (960, 540)
    assert abs(out.shape[1] / out.shape[0] - 9 / 16) < 1e-6


def test_framer_transform_maps_points_into_crop():
    framer = PortraitFramer(out_size=(540, 960), follow=False)
    out, tf = framer.apply(_frame())
    # o centro do frame original deve cair no centro da saída
    mapped = tf(np.array([[640.0, 360.0]]))[0]
    assert abs(mapped[0] - 270) < 2
    assert abs(mapped[1] - 480) < 2


def test_framer_snaps_on_first_frame():
    """O primeiro frame já sai enquadrado no rosto (sem pan de abertura)."""
    framer = PortraitFramer(out_size=(540, 960), follow=True, smooth=0.1)
    face = np.array([300.0, 360.0])
    tf = framer.apply(_frame(), face)[1]
    assert tf.origin[0] == pytest.approx(300 - 405 / 2, abs=2)  # crop_w = 720*9/16


def test_framer_follows_face_smoothly():
    framer = PortraitFramer(out_size=(540, 960), follow=True, smooth=0.2)
    centered = framer.apply(_frame())[1].origin[0]      # inicializa no centro
    face = np.array([300.0, 360.0])
    first_step = framer.apply(_frame(), face)[1].origin[0]
    for _ in range(80):
        last = framer.apply(_frame(), face)[1].origin[0]
    assert first_step < centered          # começou a andar na direção do rosto
    assert last < first_step              # mas não chegou de um salto
    assert last == pytest.approx(300 - 405 / 2, abs=2)  # e converge no rosto


def test_framer_crop_stays_inside_frame():
    framer = PortraitFramer(out_size=(540, 960), follow=True, smooth=1.0)
    for face in ([0.0, 0.0], [1279.0, 719.0], [640.0, 360.0]):
        out, tf = framer.apply(_frame(), np.array(face))
        assert out.shape[:2] == (960, 540)
        assert tf.origin[0] >= 0 and tf.origin[1] >= 0


def test_framer_zoom_tightens_crop():
    wide = PortraitFramer(out_size=(540, 960), zoom=1.0).apply(_frame())[1].scale
    tight = PortraitFramer(out_size=(540, 960), zoom=1.5).apply(_frame())[1].scale
    assert tight > wide  # mais zoom -> mais ampliação dos pixels


# --- coração ---------------------------------------------------------------- #
def test_beat_envelope_peaks_at_phase_zero():
    assert beat_envelope(0.0) > beat_envelope(0.5)
    assert beat_envelope(0.0) == pytest.approx(beat_envelope(1.0))
    # o segundo "thump" existe e é menor que o primeiro
    assert beat_envelope(0.5) < beat_envelope(0.33) < beat_envelope(0.0)


def _phase_dist(a: float, b: float) -> float:
    """Distância circular entre duas fases em [0, 1)."""
    return abs((a - b + 0.5) % 1.0 - 0.5)


def test_heartbeat_advances_at_bpm():
    heart = HeartBeat(gain=0.0)
    for _ in range(30):                      # 1 s a 30 fps, 120 bpm -> 2 ciclos
        heart.advance(1 / 30, bpm=120.0)
    assert _phase_dist(heart.phase, 0.0) < 0.02

    heart = HeartBeat(gain=0.0)
    for _ in range(15):                      # meio segundo a 60 bpm -> meio ciclo
        heart.advance(1 / 30, bpm=60.0)
    assert _phase_dist(heart.phase, 0.5) < 0.02


def test_heartbeat_ignores_long_frame_gaps():
    """Uma travada de 2 s não deve girar a fase várias vezes."""
    heart = HeartBeat(gain=0.0)
    heart.advance(2.0, bpm=180.0)
    assert heart.phase <= 0.2 * 3.0 + 1e-9   # dt foi limitado a 0.2 s


def test_heartbeat_locks_onto_measured_phase():
    heart = HeartBeat(gain=0.3)
    heart.phase = 0.5
    for _ in range(80):
        heart.advance(0.0, bpm=60.0, measured_phase=0.2)
    assert heart.phase == pytest.approx(0.2, abs=0.02)


def test_heartbeat_tracks_synthetic_pulse_phase():
    """O coração deve contrair junto com os picos do pulso sintético."""
    _, rgb = synthetic_rgb(fs=FS, duration_s=12.0)
    pulse = bandpass(detrend(rgb[:, 1]), FS)

    heart = HeartBeat(gain=0.25)
    # alimenta o PLL quadro a quadro com a fase medida na janela crescente
    for end in range(int(4 * FS), len(pulse)):
        measured = pulse_phase(pulse[:end], FS, HR_HZ)
        heart.advance(1 / FS, bpm=HR_HZ * 60.0, measured_phase=measured)

    final = pulse_phase(pulse, FS, HR_HZ)
    assert _phase_dist(heart.phase, final) < 0.05
    # o pulso termina perto de um pico, então o coração está contraído
    assert heart.scale > 1.0


def test_draw_heart_modifies_only_its_region():
    img = _frame(200, 200)
    before = img.copy()
    draw_heart(img, (100, 100), 40, (0, 0, 255), scale=1.2)
    assert not np.array_equal(img, before)
    assert np.array_equal(img[:20, :20], before[:20, :20])  # canto intacto


def test_draw_heart_near_edge_does_not_crash():
    img = _frame(120, 120)
    for center in ((0, 0), (119, 119), (-20, 60), (200, 60)):
        draw_heart(img, center, 40, (0, 0, 255))


# --- HUD completo ----------------------------------------------------------- #
def _hud_state(**kw) -> HudState:
    base = dict(method="pos", bpm=72.4, snr_db=6.1, heart_scale=1.2,
                pulse=np.sin(np.linspace(0, 20, 180)))
    base.update(kw)
    return HudState(**base)


def test_draw_hud_renders_without_error():
    img = _frame(540, 960)
    before = img.copy()
    polys = {"forehead": np.array([[200, 200], [340, 200], [340, 260], [200, 260]])}
    out = draw_hud(img, polys, _hud_state())
    assert out.shape == (960, 540, 3)
    assert not np.array_equal(out, before)


@pytest.mark.parametrize("state", [
    _hud_state(bpm=None, snr_db=None, pulse=None, calibrating=0.4),
    _hud_state(face_found=False, warning="rosto nao detectado"),
    _hud_state(snr_db=-1.0),                         # SNR baixo não vira aviso
    _hud_state(recording=True, rec_seconds=75.0, fps=29.7),
    _hud_state(pulse=np.zeros(180)),                 # sinal sem amplitude
    _hud_state(pulse=np.array([0.1, 0.2])),          # sinal curto demais
])
def test_draw_hud_handles_edge_states(state):
    draw_hud(_frame(540, 960), {}, state)


def test_draw_hud_scales_to_other_sizes():
    for size in ((1080, 1920), (360, 640), (720, 1280)):
        img = _frame(*size)
        draw_hud(img, {}, _hud_state())


# --- qualidade do sinal ----------------------------------------------------- #
def test_snr_quality_monotonic_and_clamped():
    assert snr_quality(None) == 0.0
    assert snr_quality(float("nan")) == 0.0
    assert snr_quality(-50.0) == 0.0
    assert snr_quality(50.0) == 1.0
    assert snr_quality(0.0) < snr_quality(4.0) < snr_quality(9.0)


def test_low_snr_still_colors_and_syncs():
    """SNR baixo muda cor e peso do engate, mas não desliga nada."""
    assert THEME.quality_color(-1.0) == THEME.bad
    assert THEME.quality_color(8.0) == THEME.good
    assert THEME.wave_color(-1.0) == THEME.muted
    assert THEME.wave_color(8.0) == THEME.accent_dim
    # mesmo com sinal ruim o coração ainda se engancha, só que devagar
    weak = HeartBeat(gain=0.12)
    weak.phase = 0.5
    for _ in range(200):
        weak.advance(0.0, bpm=72.0, measured_phase=0.2,
                     gain=0.12 * (0.2 + 0.8 * snr_quality(-1.0)))
    assert _phase_dist(weak.phase, 0.2) < 0.05


def test_heartbeat_gain_override():
    heart = HeartBeat(gain=0.5)
    heart.phase = 0.5
    heart.advance(0.0, bpm=60.0, measured_phase=0.0, gain=0.0)
    assert heart.phase == pytest.approx(0.5)      # ganho 0 ignora a medida
