"""DOA turn: azimuth mapping, readers, the speech-onset window, guards, and what follows the turn."""

import math
import struct
import time
from unittest import mock

import pytest

import hal.app_state as state
import hal.config as config
from hal.drivers.tracking import doa_turn
from hal.drivers.tracking.search import SearchResult
from hal.drivers.voice import doa


class _Svc:
    def __init__(self, yaw=0.0):
        self.pos = {"base_yaw.pos": yaw, "base_pitch.pos": 0.0}
        self.moves = []
        self._hold_mode = False
        self._tracking_active = False
        self._music_playing = False
        self._body_owners = 0

    def get_positions(self):
        return dict(self.pos)

    def move_and_hold(self, target, duration=0.0):
        self.moves.append((dict(target), duration))
        self.pos.update(target)

    def acquire_body(self):
        pass

    def release_body(self):
        pass


class _Calls:
    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return self.ret


@pytest.fixture
def body(monkeypatch):
    svc = _Svc()
    monkeypatch.setattr(state, "animation_service", svc, raising=False)
    monkeypatch.setattr(state, "_sleeping", False, raising=False)
    monkeypatch.setattr(state, "voice_service", None, raising=False)
    monkeypatch.setattr(state, "safety_policy", None, raising=False)
    monkeypatch.setattr(config, "DOA_TURN_ENABLED", True)
    monkeypatch.setattr(config, "DOA_OFFSET_DEG", 0.0)
    monkeypatch.setattr(config, "DOA_SIGN", 1.0)
    monkeypatch.setattr(config, "DOA_GLANCE_PITCH_DEG", 4.0)
    monkeypatch.setattr(config, "DOA_LOCK_WAIT_S", 0.06)
    monkeypatch.setattr(config, "DOA_WINDOW_S", 0.06)
    monkeypatch.setattr(config, "DOA_FIND_FACE", False)
    monkeypatch.setattr(doa_turn, "SAMPLE_PERIOD_S", 0.005)
    monkeypatch.setattr(doa_turn, "_last_turn_t", 0.0)
    monkeypatch.setattr(doa_turn, "_last_empty_look", (0.0, 0.0))
    monkeypatch.setattr(doa_turn, "_busy", False)
    monkeypatch.setattr("hal.drivers.tracking.gaze.face_seen_within", lambda s: False)
    svc.fallbacks = _Calls(True)
    monkeypatch.setattr("hal.drivers.tracking.gaze.request_deferred_repoint", svc.fallbacks)
    svc.releases = _Calls()
    monkeypatch.setattr("hal.drivers.tracking.body.release_to_idle_later", svc.releases)
    yield svc
    doa_turn.set_reader(None)


def test_azimuth_to_yaw_wraps_and_signs():
    assert doa_turn.azimuth_to_yaw(90, 0, 1) == 90
    assert doa_turn.azimuth_to_yaw(270, 0, 1) == -90
    assert doa_turn.azimuth_to_yaw(10, 350, 1) == 20
    assert doa_turn.azimuth_to_yaw(90, 0, -1) == -90
    assert doa_turn.azimuth_to_yaw(180, 0, 1) == -180
    # Bernard's predicted mapping (sign -1, offset 270): chip 270 is straight ahead,
    # chip 180 is his right.
    assert doa_turn.azimuth_to_yaw(270, 270, -1) == 0
    assert doa_turn.azimuth_to_yaw(180, 270, -1) == 90


def test_circular_median_across_the_seam():
    got = doa.circular_median_deg([358, 2, 0, 359, 1])
    assert min(got, 360 - got) < 1.5
    assert doa.circular_median_deg([]) is None


def test_parse_reply():
    ok = bytes([0]) + struct.pack("<HH", 725, 1)
    assert doa.parse_doa_reply(ok) == doa.DoaSample(5.0, True)
    assert doa.parse_doa_reply(bytes([1]) + ok[1:]) is None
    assert doa.parse_doa_reply(b"\x00") is None


def test_parse_selected_reply_nan_is_silence():
    talk = bytes([0]) + struct.pack("<ff", math.radians(-90.0), 0.0)
    got = doa.parse_selected_reply(talk)
    assert got.speech and abs(got.azimuth_deg - 270.0) < 1e-3
    quiet = bytes([0]) + struct.pack("<ff", float("nan"), 1.0)
    assert doa.parse_selected_reply(quiet) == doa.DoaSample(0.0, False)
    assert doa.parse_selected_reply(bytes([64]) + talk[1:]) is None


def test_usb_reader_retries_while_the_chip_is_busy():
    reader = doa.UsbDoaReader()
    busy = bytes([64, 0, 0, 0, 0])
    ok = bytes([0]) + struct.pack("<HH", 120, 1)
    reader._dev = mock.Mock(ctrl_transfer=mock.Mock(side_effect=[busy, ok]))
    assert reader.read() == doa.DoaSample(120.0, True)
    with pytest.raises(ValueError):
        doa.UsbDoaReader(register="nope")


def test_agreement_rule():
    assert doa.agreed_azimuth([88, 90, 92, 91], 4, 20, 0.75) == pytest.approx(90.5)
    # One stray reflection among agreeing readings is outvoted.
    assert doa.agreed_azimuth([88, 90, 92, 91, 250], 4, 20, 0.75) == pytest.approx(90.5)
    # Scattered readings are not a direction.
    assert doa.agreed_azimuth([10, 100, 190, 280, 95], 4, 20, 0.75) is None
    assert doa.agreed_azimuth([90, 90, 90], 4, 20, 0.75) is None


def test_poller_window_holds_only_its_own_speech():
    reader = doa.FakeDoaReader(270.0)
    poller = doa.DoaPoller(reader, 0.005)
    for _ in range(5):
        poller.poll_once()
    time.sleep(0.01)
    t_speech = time.monotonic()
    reader.azimuth_deg = 90.0
    for _ in range(5):
        poller.poll_once()
    reader.speech = False
    poller.poll_once()
    assert poller.speech_between(t_speech, time.monotonic()) == [90.0] * 5


def test_plan_target_clamps_and_glances():
    t = doa_turn.plan_target({"base_yaw.pos": 0, "base_pitch.pos": 1}, 170, -135, 135, 4)
    assert t["base_yaw.pos"] == 135 and t["base_pitch.pos"] == 5


def _run_turn(reader=None):
    if reader is not None:
        doa_turn.set_reader(reader)
    assert doa_turn.on_speech_start()
    for _ in range(300):
        if not doa_turn._busy:
            return
        time.sleep(0.01)
    raise AssertionError("turn never finished")


def test_turns_toward_speaker(body):
    _run_turn(doa.FakeDoaReader(90.0))
    assert len(body.moves) == 1
    target, duration = body.moves[0]
    assert target["base_yaw.pos"] == 90.0 and target["base_pitch.pos"] == 4.0
    assert 0.6 <= duration <= 1.6
    assert body.fallbacks.calls == []
    assert body.releases.calls and body.releases.calls[0][0][0] == doa_turn.HOLD_AFTER_TURN_S


class _LaggingChip:
    """Reads the previous talker for `lag_s` after `switch_t`, then the new one."""

    def __init__(self, old, new, lag_s):
        self.old, self.new, self.lag_s = old, new, lag_s
        self.switch_t = None

    def read(self):
        now = time.monotonic()
        late = self.switch_t is not None and now >= self.switch_t + self.lag_s
        return doa.DoaSample(self.new if late else self.old, True)


def test_the_window_waits_out_the_chips_lag(body, monkeypatch):
    """2026-10-18: the chip reported the previous talker for seconds after a new one began;
    reading it right at VAD would turn him to whoever spoke last."""
    monkeypatch.setattr(config, "DOA_LOCK_WAIT_S", 0.12)
    monkeypatch.setattr(config, "DOA_WINDOW_S", 0.06)
    chip = _LaggingChip(old=300.0, new=90.0, lag_s=0.04)
    doa_turn.set_reader(chip)
    chip.switch_t = time.monotonic()
    _run_turn()
    assert body.moves and body.moves[0][0]["base_yaw.pos"] == 90.0


def test_background_poller_feeds_the_turn(body, monkeypatch):
    monkeypatch.setattr(config, "DOA_WINDOW_S", 0.1)
    doa_turn.set_reader(doa.FakeDoaReader(-60.0 % 360))
    doa_turn.start()
    try:
        assert doa_turn._source.running
        _run_turn()
    finally:
        doa_turn._source.stop()
    assert body.moves and body.moves[0][0]["base_yaw.pos"] == -60.0


class _FixedSource:
    running = True

    def __init__(self, answer):
        self.answer = answer
        self.windows = []

    def bearing_between(self, t0, t1):
        self.windows.append((t0, t1))
        return self.answer

    def start(self):
        pass

    def stop(self):
        pass


def test_any_source_with_a_bearing_turns_him(body):
    """The raw-mic source answers the same question as the chip poller."""
    src = _FixedSource((90.0, 9))
    doa_turn.set_source(src)
    _run_turn()
    assert body.moves and body.moves[0][0]["base_yaw.pos"] == 90.0
    (t0, t1), = src.windows
    assert t1 - t0 >= config.DOA_WINDOW_S - 1e-3


def test_raw_source_is_built_from_config(body, monkeypatch):
    from hal.drivers.voice import doa_raw

    monkeypatch.setattr(config, "DOA_SOURCE", "raw")
    doa_turn.set_reader(None)
    src = doa_turn._get_source()
    assert isinstance(src, doa_raw.RawMicDoa) and not src.running


def test_small_angle_stays_put(body):
    _run_turn(doa.FakeDoaReader(5.0))
    assert body.moves == []
    assert body.fallbacks.calls == []


def test_no_speech_hands_the_speech_start_back_to_gaze(body):
    _run_turn(doa.FakeDoaReader(90.0, speech=False))
    assert body.moves == []
    assert len(body.fallbacks.calls) == 1


class _Scattered:
    def __init__(self):
        self.i = 0

    def read(self):
        self.i += 1
        return doa.DoaSample((self.i * 97.0) % 360, True)


def test_scattered_readings_do_not_turn_him(body):
    _run_turn(_Scattered())
    assert body.moves == []
    assert len(body.fallbacks.calls) == 1


def test_disabled_and_cooldown(body, monkeypatch):
    monkeypatch.setattr(config, "DOA_TURN_ENABLED", False)
    assert not doa_turn.on_speech_start()
    assert not doa_turn.handles_speech_start()
    monkeypatch.setattr(config, "DOA_TURN_ENABLED", True)
    _run_turn(doa.FakeDoaReader(90.0))
    assert not doa_turn.on_speech_start()  # inside the cooldown
    assert body.fallbacks.calls == []  # a fresh turn beats the remembered pose


@pytest.mark.parametrize("flag", ["_tracking_active", "_music_playing", "_hold_mode"])
def test_guards_keep_the_body(body, flag):
    setattr(body, flag, True)
    _run_turn(doa.FakeDoaReader(90.0))
    assert body.moves == []
    assert len(body.fallbacks.calls) == 1


def test_asleep(body, monkeypatch):
    monkeypatch.setattr(state, "_sleeping", True)
    _run_turn(doa.FakeDoaReader(90.0))
    assert body.moves == []


def test_a_face_in_view_where_the_sound_is_keeps_him_still(body, monkeypatch):
    monkeypatch.setattr("hal.drivers.tracking.gaze.face_seen_within", lambda s: True)
    _run_turn(doa.FakeDoaReader(20.0))
    assert body.moves == []
    assert body.fallbacks.calls == []


def test_a_voice_from_outside_the_view_turns_him_even_with_a_face_in_frame(body, monkeypatch):
    monkeypatch.setattr("hal.drivers.tracking.gaze.face_seen_within", lambda s: True)
    _run_turn(doa.FakeDoaReader(90.0))
    assert body.moves and body.moves[0][0]["base_yaw.pos"] == 90.0


def test_tts_speaking_skips(body, monkeypatch):
    monkeypatch.setattr(state, "voice_service", mock.Mock(_tts_is_speaking=lambda: True))
    _run_turn(doa.FakeDoaReader(90.0))
    assert body.moves == []


def test_after_turning_he_looks_for_the_face_there(body, monkeypatch):
    monkeypatch.setattr(config, "DOA_FIND_FACE", True)
    search = _Calls(SearchResult(True, "found", looks_visited=1))
    monkeypatch.setattr("hal.drivers.tracking.search.search_for_subject", search)
    _run_turn(doa.FakeDoaReader(90.0))
    assert len(search.calls) == 1
    kw = search.calls[0][1]
    assert kw["seed_yaw"] == 90.0 and kw["glance"] is True and kw["target"] == "person"
    assert body.releases.calls == []  # the search hands the body back itself


def test_a_search_that_cannot_run_still_releases_the_body(body, monkeypatch):
    monkeypatch.setattr(config, "DOA_FIND_FACE", True)
    search = _Calls(SearchResult(False, "no camera or animation service"))
    monkeypatch.setattr("hal.drivers.tracking.search.search_for_subject", search)
    _run_turn(doa.FakeDoaReader(90.0))
    assert len(body.releases.calls) == 1


def test_facing_the_voice_with_no_face_looks_up_once(body, monkeypatch):
    """The 2026-10-19 failure: talker in front, camera on the desk, head above the frame."""
    monkeypatch.setattr(config, "DOA_FIND_FACE", True)
    search = _Calls(SearchResult(False, "nothing", looks_visited=3))
    monkeypatch.setattr("hal.drivers.tracking.search.search_for_subject", search)
    _run_turn(doa.FakeDoaReader(5.0))
    assert body.moves == []
    assert len(search.calls) == 1 and search.calls[0][1]["seed_yaw"] == 0.0
    # Nobody there: the next utterance from the same place doesn't glance again.
    monkeypatch.setattr(doa_turn, "_last_turn_t", 0.0)
    _run_turn(doa.FakeDoaReader(5.0))
    assert len(search.calls) == 1


def test_a_new_utterance_in_an_open_session_is_a_speech_start(monkeypatch):
    """Inside a Gemini Live session the local VAD's speech start doesn't fire; the server's
    ACTIVITY_START hands the cue to doa_turn instead, and a failure there never reaches
    the receive loop."""
    from realtime.voice_agent import gemini_live

    calls = _Calls(True)
    monkeypatch.setattr(doa_turn, "on_speech_start", calls)
    gemini_live._turn_toward_voice()
    assert len(calls.calls) == 1

    def boom():
        raise RuntimeError("usb gone")

    monkeypatch.setattr(doa_turn, "on_speech_start", boom)
    gemini_live._turn_toward_voice()
