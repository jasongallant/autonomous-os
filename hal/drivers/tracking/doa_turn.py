"""Turn toward whoever starts speaking, from the mic array's sound direction.

VAD confirms speech on the mic thread; this module hands the turn to a short worker so
the mic thread never waits on USB or servos. One small swivel plus a glance, then the
body goes back to idle. It never fights vision: a face already in frame, tracking, a
scene, or any other hold keeps the body.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from hal import config
from hal.drivers.voice import doa

logger = logging.getLogger(__name__)

SAMPLE_COUNT = 6
SAMPLE_PERIOD_S = 0.05
# Turn time scales with the angle, like a person's: quick for a small turn, never abrupt.
MIN_TURN_S = 0.6
MAX_TURN_S = 1.6
HOLD_AFTER_TURN_S = 6.0

_lock = threading.Lock()
_busy = False
_last_turn_t = 0.0
_reader: Optional[doa.DoaReader] = None


def set_reader(reader: Optional[doa.DoaReader]) -> None:
    """Sim and tests inject a reader here; None restores the USB one."""
    global _reader
    _reader = reader


def _get_reader() -> doa.DoaReader:
    global _reader
    if _reader is None:
        _reader = doa.UsbDoaReader()
    return _reader


def azimuth_to_yaw(azimuth_deg: float, offset_deg: float, sign: float) -> float:
    """Chip azimuth to a body yaw in [-180, 180): 0 is straight ahead, negative is left."""
    return (sign * (azimuth_deg - offset_deg) + 180.0) % 360.0 - 180.0


def turn_duration(delta_deg: float) -> float:
    return min(MAX_TURN_S, max(MIN_TURN_S, MIN_TURN_S + abs(delta_deg) / 90.0))


def plan_target(
    current: Dict[str, float], yaw_deg: float, yaw_min: float, yaw_max: float,
    glance_deg: float,
) -> Dict[str, float]:
    """Positions after turning to `yaw_deg` (clamped) and glancing by `glance_deg`."""
    target = dict(current)
    target["base_yaw.pos"] = min(yaw_max, max(yaw_min, yaw_deg))
    if glance_deg:
        target["base_pitch.pos"] = current.get("base_pitch.pos", 0.0) + glance_deg
    return target


def on_speech_start() -> bool:
    """Called from the mic thread when VAD confirms speech. Returns at once."""
    global _busy
    if not config.DOA_TURN_ENABLED:
        return False
    now = time.monotonic()
    with _lock:
        if _busy or (now - _last_turn_t) < config.DOA_TURN_COOLDOWN_S:
            return False
        _busy = True
    threading.Thread(target=_run, name="doa-turn", daemon=True).start()
    return True


def _skip_reason(svc: Any) -> Optional[str]:
    import hal.app_state as state
    from hal.drivers.motors import hold
    from hal.drivers.tracking import gaze

    if svc is None:
        return "no body"
    if getattr(state, "_sleeping", False):
        return "asleep"
    if getattr(svc, "_tracking_active", False):
        return "tracking has the body"
    if getattr(svc, "_music_playing", False):
        return "music has the body"
    held = hold.holder(svc)
    if held:
        return f"held by {held}"
    voice = getattr(state, "voice_service", None)
    if voice is not None and getattr(voice, "_tts_is_speaking", lambda: False)():
        return "he is speaking (echo)"
    if gaze.face_recently_seen(config.DOA_TURN_SKIP_IF_FACE_S):
        return "a face is already in frame"
    return None


def _collect_azimuth(reader: doa.DoaReader) -> Optional[float]:
    angles = []
    for _ in range(SAMPLE_COUNT):
        s = reader.read()
        if s is not None and s.speech:
            angles.append(s.azimuth_deg)
        time.sleep(SAMPLE_PERIOD_S)
    # Too few speech samples is a click or a door, not a person talking.
    if len(angles) < SAMPLE_COUNT // 2:
        return None
    return doa.circular_median_deg(angles)


def _run() -> None:
    global _busy, _last_turn_t
    try:
        import hal.app_state as state
        from hal.drivers.tracking import aim, body
        from hal.drivers.tracking.constants import YAW_MAX, YAW_MIN

        svc = getattr(state, "animation_service", None)
        why = _skip_reason(svc)
        if why:
            logger.info("[doa] no turn: %s", why)
            return
        azimuth = _collect_azimuth(_get_reader())
        if azimuth is None:
            logger.info("[doa] no turn: no steady speech direction")
            return
        yaw = azimuth_to_yaw(azimuth, config.DOA_OFFSET_DEG, config.DOA_SIGN)
        current = svc.get_positions()
        delta = yaw - current.get("base_yaw.pos", 0.0)
        if abs(delta) < config.DOA_MIN_DELTA_DEG:
            logger.info("[doa] already facing %+.0f deg (azimuth %.0f)", yaw, azimuth)
            return
        target = plan_target(current, yaw, YAW_MIN, YAW_MAX, config.DOA_GLANCE_PITCH_DEG)
        ceiling = getattr(getattr(state.safety_policy, "motion", None), "max_cog_offset_mm", None)
        if ceiling is not None:
            from hal.drivers.motors.recording_stability import limit_lean_back

            target, _ = limit_lean_back(current, target, ceiling)
        duration = aim.min_move_duration(
            state.safety_policy, target, current, turn_duration(delta)
        )
        # Re-check just before moving: speech took a moment to read, and tracking or a
        # scene may have claimed the body since.
        why = _skip_reason(svc)
        if why:
            logger.info("[doa] no turn: %s", why)
            return
        with aim.servo_ownership():
            svc.move_and_hold(target, duration=duration)
        _last_turn_t = time.monotonic()
        logger.info(
            "[doa] turned %+.0f deg to azimuth %.0f (yaw %+.0f) over %.1fs",
            delta, azimuth, target["base_yaw.pos"], duration,
        )
        body.release_to_idle_later(HOLD_AFTER_TURN_S, "doa turn done")
    except Exception as e:
        logger.warning("[doa] turn skipped: %s", e)
    finally:
        with _lock:
            _busy = False
