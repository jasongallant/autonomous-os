"""Turn toward whoever starts speaking, from the mic array's sound direction.

A background poller keeps the last few seconds of the chip's readings. VAD confirms
speech on the mic thread; this module hands the turn to a short worker so the mic
thread never waits on USB or servos. The worker takes the readings around the speech
onset, turns if they agree, and then looks for the talker's face at that bearing (one
search glance, which centres a face and lifts to a head above the frame). It never
fights vision: a face already in view where the sound is, tracking, a scene, or any
other hold keeps the body. When it can't tell where the voice came from, gaze's
remembered-pose reacquire gets the speech start instead.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from hal import config
from hal.drivers.voice import doa

logger = logging.getLogger(__name__)

SAMPLE_PERIOD_S = 0.05
# Turn time scales with the angle, like a person's: quick for a small turn, never abrupt.
MIN_TURN_S = 0.6
MAX_TURN_S = 1.6
HOLD_AFTER_TURN_S = 6.0
# Nobody was found at a bearing: don't glance there again on every utterance for this long.
FIND_FACE_RETRY_S = 30.0

_lock = threading.Lock()
_busy = False
_last_turn_t = 0.0
_last_empty_look: Tuple[float, float] = (0.0, 0.0)  # (monotonic, yaw) of a look that found nobody
_reader: Optional[doa.DoaReader] = None
# The running direction source: a DoaPoller on the chip, or doa_raw.RawMicDoa. Both answer
# bearing_between(t0, t1) -> (azimuth or None, readings used).
_source: Any = None


def set_reader(reader: Optional[doa.DoaReader]) -> None:
    """Sim and tests inject a chip reader here; None restores the configured source."""
    global _reader, _source
    if _source is not None:
        _source.stop()
    _reader = reader
    _source = None


def set_source(source: Any) -> None:
    """Tests inject any object with bearing_between(), running, start() and stop()."""
    global _source
    if _source is not None:
        _source.stop()
    _source = source


def _get_reader() -> doa.DoaReader:
    global _reader
    if _reader is None:
        _reader = doa.UsbDoaReader(register=config.DOA_REGISTER)
    return _reader


def _get_source() -> Any:
    global _source
    if _source is None:
        import hal.app_state as state

        def asleep() -> bool:
            return bool(getattr(state, "_sleeping", False))

        if config.DOA_SOURCE == "raw" and _reader is None:
            from hal.drivers.voice import doa_raw

            _source = doa_raw.RawMicDoa(
                config.DOA_RAW_DEVICE, doa_raw.parse_mics_mm(config.DOA_MIC_MM), paused=asleep,
            )
        else:
            _source = doa.DoaPoller(_get_reader(), SAMPLE_PERIOD_S, paused=asleep)
    return _source


def start() -> None:
    """Begin listening, so a speech start has readings from before VAD confirmed it."""
    if not config.DOA_TURN_ENABLED:
        return
    _get_source().start()
    what = (f"raw mics on {config.DOA_RAW_DEVICE}" if config.DOA_SOURCE == "raw"
            else f"chip register {config.DOA_REGISTER}")
    logger.info(
        "[doa] listening (%s); turns use %.1f s of readings ending %.1f s after speech start",
        what, config.DOA_WINDOW_S, config.DOA_LOCK_WAIT_S,
    )


def handles_speech_start() -> bool:
    """True when the sound direction decides a speech start's turn (gaze waits for it)."""
    return config.DOA_TURN_ENABLED


def azimuth_to_yaw(azimuth_deg: float, offset_deg: float, sign: float) -> float:
    """Chip azimuth to a body yaw in [-180, 180): 0 is straight ahead, negative is left."""
    return (sign * (azimuth_deg - offset_deg) + 180.0) % 360.0 - 180.0


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


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
        # A turn just made (or under way) is fresher than gaze's remembered pose, so
        # these return without handing the speech start back to gaze.
        if _busy or (now - _last_turn_t) < config.DOA_TURN_COOLDOWN_S:
            return False
        _busy = True
    threading.Thread(target=_run, args=(now,), name="doa-turn", daemon=True).start()
    return True


def _skip_reason(svc: Any) -> Optional[str]:
    import hal.app_state as state
    from hal.drivers.motors import hold

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
    return None


def _collect_azimuth(t_speech: float) -> Tuple[Optional[float], int]:
    """The azimuth from the readings around this speech onset, and how many there were."""
    src = _get_source()
    end = t_speech + config.DOA_LOCK_WAIT_S
    # A chip poller without its thread (sim, tests, start() not called) is read here.
    poll = None if src.running else getattr(src, "poll_once", None)
    while time.monotonic() < end:
        if poll is not None:
            poll()
        time.sleep(max(0.005, min(SAMPLE_PERIOD_S, end - time.monotonic())))
    if poll is not None:
        poll()
    return src.bearing_between(end - config.DOA_WINDOW_S, time.monotonic())


def _fall_back_to_gaze(why: str) -> None:
    try:
        from hal.drivers.tracking import gaze

        gaze.request_deferred_repoint(why)
    except Exception as e:
        logger.debug("[doa] gaze fallback skipped: %s", e)


def _find_face(yaw: float) -> bool:
    """One search glance for the talker at `yaw`. True if the search ran (it then owns
    the hand-back to idle); False if the caller must release the body itself."""
    global _last_empty_look
    if not config.DOA_FIND_FACE:
        return False
    now = time.monotonic()
    t, y = _last_empty_look
    if now - t < FIND_FACE_RETRY_S and abs(_wrap180(yaw - y)) < config.DOA_IN_VIEW_DEG:
        logger.info("[doa] not looking again: nobody was at %+.0f deg %.0f s ago", y, now - t)
        return False
    from hal.drivers.tracking import search

    res = search.search_for_subject(
        target="person", glance=True, seed_yaw=yaw, on_progress=lambda _v, _t: None,
    )
    logger.info("[doa] looked for the talker at %+.0f deg: %s after %d look(s)",
                yaw, res.reason, res.looks_visited)
    if not res.found:
        _last_empty_look = (now, yaw)
    # A search that never got going (no camera, no detector) has not scheduled a release.
    return bool(res.found or res.looks_visited)


def _run(t_speech: float) -> None:
    global _busy, _last_turn_t
    try:
        import hal.app_state as state
        from hal.drivers.tracking import aim, body, gaze
        from hal.drivers.tracking.constants import YAW_MAX, YAW_MIN

        svc = getattr(state, "animation_service", None)
        why = _skip_reason(svc)
        if why:
            logger.info("[doa] no turn: %s", why)
            _fall_back_to_gaze(f"no sound turn ({why})")
            return
        azimuth, n = _collect_azimuth(t_speech)
        if azimuth is None:
            logger.info("[doa] no turn: no steady speech direction (%d speech readings)", n)
            _fall_back_to_gaze("no steady sound direction")
            return
        yaw = azimuth_to_yaw(azimuth, config.DOA_OFFSET_DEG, config.DOA_SIGN)
        current = svc.get_positions()
        heading = current.get("base_yaw.pos", 0.0)
        delta = yaw - heading
        if (gaze.face_seen_within(config.DOA_TURN_SKIP_IF_FACE_S)
                and abs(delta) <= config.DOA_IN_VIEW_DEG):
            logger.info("[doa] no turn: the face in view is the talker (sound %+.0f deg off, "
                        "azimuth %.0f)", delta, azimuth)
            return
        if abs(delta) < config.DOA_MIN_DELTA_DEG:
            # Facing the voice with no face in view: the head is likely low (a seated
            # talker's face above the frame), so look without turning.
            logger.info("[doa] already facing %+.0f deg (azimuth %.0f), no face in view",
                        yaw, azimuth)
            if _find_face(heading):
                _last_turn_t = time.monotonic()
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
        t_move = time.monotonic()
        with aim.servo_ownership():
            svc.move_and_hold(target, duration=duration)
        _last_turn_t = time.monotonic()
        logger.info(
            "[doa] turned %+.0f deg to azimuth %.0f (yaw %+.0f) over %.1fs, starting %.2fs "
            "after speech start", delta, azimuth, target["base_yaw.pos"], duration,
            t_move - t_speech,
        )
        if not _find_face(target["base_yaw.pos"]):
            body.release_to_idle_later(HOLD_AFTER_TURN_S, "doa turn done")
    except Exception as e:
        logger.warning("[doa] turn skipped: %s", e)
    finally:
        with _lock:
            _busy = False
