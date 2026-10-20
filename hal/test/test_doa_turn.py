"""DOA turn: azimuth mapping, guards, and the move it asks the body for."""

import struct
import time
from unittest import mock

import pytest

import hal.app_state as state
import hal.config as config
from hal.drivers.tracking import doa_turn
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
    monkeypatch.setattr(doa_turn, "SAMPLE_PERIOD_S", 0.0)
    monkeypatch.setattr(doa_turn, "_last_turn_t", 0.0)
    monkeypatch.setattr(doa_turn, "_busy", False)
    monkeypatch.setattr("hal.drivers.tracking.gaze.face_recently_seen", lambda s: False)
    monkeypatch.setattr("hal.drivers.tracking.body.release_to_idle_later", lambda *a: None)
    yield svc
    doa_turn.set_reader(None)


def test_azimuth_to_yaw_wraps_and_signs():
    assert doa_turn.azimuth_to_yaw(90, 0, 1) == 90
    assert doa_turn.azimuth_to_yaw(270, 0, 1) == -90
    assert doa_turn.azimuth_to_yaw(10, 350, 1) == 20
    assert doa_turn.azimuth_to_yaw(90, 0, -1) == -90
    assert doa_turn.azimuth_to_yaw(180, 0, 1) == -180


def test_circular_median_across_the_seam():
    got = doa.circular_median_deg([358, 2, 0, 359, 1])
    assert min(got, 360 - got) < 1.5
    assert doa.circular_median_deg([]) is None


def test_parse_reply():
    ok = bytes([0]) + struct.pack("<HH", 725, 1)
    assert doa.parse_doa_reply(ok) == doa.DoaSample(5.0, True)
    assert doa.parse_doa_reply(bytes([1]) + ok[1:]) is None
    assert doa.parse_doa_reply(b"\x00") is None


def test_plan_target_clamps_and_glances():
    t = doa_turn.plan_target({"base_yaw.pos": 0, "base_pitch.pos": 1}, 170, -135, 135, 4)
    assert t["base_yaw.pos"] == 135 and t["base_pitch.pos"] == 5


def _run_turn(reader):
    doa_turn.set_reader(reader)
    assert doa_turn.on_speech_start()
    for _ in range(200):
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


def test_small_angle_stays_put(body):
    _run_turn(doa.FakeDoaReader(5.0))
    assert body.moves == []


def test_no_speech_flag_stays_put(body):
    _run_turn(doa.FakeDoaReader(90.0, speech=False))
    assert body.moves == []


def test_disabled_and_cooldown(body, monkeypatch):
    monkeypatch.setattr(config, "DOA_TURN_ENABLED", False)
    assert not doa_turn.on_speech_start()
    monkeypatch.setattr(config, "DOA_TURN_ENABLED", True)
    _run_turn(doa.FakeDoaReader(90.0))
    assert not doa_turn.on_speech_start()  # inside the cooldown


@pytest.mark.parametrize("flag", ["_tracking_active", "_music_playing", "_hold_mode"])
def test_guards_keep_the_body(body, flag):
    setattr(body, flag, True)
    _run_turn(doa.FakeDoaReader(90.0))
    assert body.moves == []


def test_asleep_and_face_in_frame(body, monkeypatch):
    monkeypatch.setattr(state, "_sleeping", True)
    _run_turn(doa.FakeDoaReader(90.0))
    monkeypatch.setattr(state, "_sleeping", False)
    monkeypatch.setattr(doa_turn, "_last_turn_t", 0.0)
    monkeypatch.setattr("hal.drivers.tracking.gaze.face_recently_seen", lambda s: True)
    _run_turn(doa.FakeDoaReader(90.0))
    assert body.moves == []


def test_tts_speaking_skips(body, monkeypatch):
    monkeypatch.setattr(state, "voice_service", mock.Mock(_tts_is_speaking=lambda: True))
    _run_turn(doa.FakeDoaReader(90.0))
    assert body.moves == []
