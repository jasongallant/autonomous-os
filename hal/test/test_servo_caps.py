"""Tests for the optional servo hardware caps (hal/drivers/motors/servo_caps.py).

HAL_SERVO_ACCEL / HAL_SERVO_TORQUE_LIMIT: unset must leave every write exactly as
before; set, every Acceleration write is capped and Torque_Limit is written at
configure, with torque still off. No hardware: the bus is a recorder.
"""
import ast
import contextlib
import os
import threading

import pytest

from hal import config
from hal.drivers.motors import servo_caps
from hal.drivers.motors.servo_caps import capped_acceleration, capped_torque_limit, write_caps_raw

HAL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


@pytest.fixture
def caps(monkeypatch):
    def _set(accel=None, torque=None):
        monkeypatch.setattr(config, "SERVO_ACCEL", accel)
        monkeypatch.setattr(config, "SERVO_TORQUE_LIMIT", torque)
    _set()
    return _set


class _Packets:
    """scservo_sdk PacketHandler stand-in: every servo answers, writes are recorded."""

    def __init__(self):
        self.writes = []

    def ping(self, ph, sid):
        return 777, 0, 0  # COMM_SUCCESS == 0

    def write1ByteTxRx(self, ph, sid, addr, value):
        self.writes.append((sid, addr, value))
        return 0, 0

    def write2ByteTxRx(self, ph, sid, addr, value):
        self.writes.append((sid, addr, value))
        return 0, 0


# --- env parsing -----------------------------------------------------------------------------

def test_optional_int_unset_and_blank_are_none(monkeypatch):
    monkeypatch.delenv("HAL_TEST_CAP", raising=False)
    assert config._optional_int("HAL_TEST_CAP", 1, 254) is None
    monkeypatch.setenv("HAL_TEST_CAP", "  ")
    assert config._optional_int("HAL_TEST_CAP", 1, 254) is None


def test_optional_int_clamps_to_register_range(monkeypatch):
    monkeypatch.setenv("HAL_TEST_CAP", "40")
    assert config._optional_int("HAL_TEST_CAP", 1, 254) == 40
    monkeypatch.setenv("HAL_TEST_CAP", "999")
    assert config._optional_int("HAL_TEST_CAP", 1, 254) == 254
    monkeypatch.setenv("HAL_TEST_CAP", "0")
    assert config._optional_int("HAL_TEST_CAP", 1, 254) == 1


# --- capped_acceleration ---------------------------------------------------------------------

def test_uncapped_passes_every_value_through(caps):
    for v in (0, 30, 254):
        assert capped_acceleration(v) == v


def test_cap_limits_high_values_and_keeps_low_ones(caps):
    caps(accel=40)
    assert capped_acceleration(254) == 40
    assert capped_acceleration(30) == 30


def test_cap_turns_zero_into_the_cap(caps):
    # 0 is "no ramp" on the servo, the hardest start of all, not the gentlest.
    caps(accel=40)
    assert capped_acceleration(0) == 40


# --- capped_torque_limit ---------------------------------------------------------------------

def test_torque_limit_unset_cap_passes_value_through(caps):
    assert capped_torque_limit(None) is None
    assert capped_torque_limit(800) == 800


def test_torque_limit_cap_lowers_but_never_raises(caps):
    caps(torque=300)
    assert capped_torque_limit(800) == 300
    assert capped_torque_limit(200) == 200
    assert capped_torque_limit(None) is None


# --- write_caps_raw --------------------------------------------------------------------------

def test_raw_writes_nothing_when_unset(caps):
    pk = _Packets()
    assert write_caps_raw(pk, None, 3) == []
    assert pk.writes == []


def test_raw_writes_both_registers(caps):
    caps(accel=40, torque=600)
    pk = _Packets()
    assert write_caps_raw(pk, None, 3) == ["accel=40", "torque_limit=600"]
    assert pk.writes == [(3, servo_caps.ACCELERATION_REG, 40), (3, servo_caps.TORQUE_LIMIT_REG, 600)]


# --- AnimationService._configure_servos_raw --------------------------------------------------

def _raw_configure(energize=True):
    pytest.importorskip("lerobot")
    from types import SimpleNamespace
    from hal.drivers.motors.animation_service import AnimationService

    svc = object.__new__(AnimationService)
    pk = _Packets()
    motors = {name: SimpleNamespace(id=i) for i, name in enumerate(
        ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"), start=1)}
    svc.bus_lock = threading.Lock()
    svc.robot = SimpleNamespace(bus=SimpleNamespace(port_handler=None, packet_handler=pk, motors=motors))
    svc._configure_servos_raw(energize=energize)
    return pk.writes


def test_configure_raw_unchanged_without_caps(caps):
    regs = {addr for _, addr, _ in _raw_configure()}
    assert servo_caps.ACCELERATION_REG not in regs
    assert servo_caps.TORQUE_LIMIT_REG not in regs


def test_configure_raw_writes_caps_before_torque_on(caps):
    caps(accel=40, torque=600)
    writes = _raw_configure()
    for sid in range(1, 6):
        mine = [(addr, v) for s, addr, v in writes if s == sid]
        assert (servo_caps.ACCELERATION_REG, 40) in mine
        assert (servo_caps.TORQUE_LIMIT_REG, 600) in mine
        torque_on = mine.index((40, 1))  # Torque_Enable = 1
        assert mine.index((servo_caps.TORQUE_LIMIT_REG, 600)) < torque_on
        assert mine.index((servo_caps.ACCELERATION_REG, 40)) < torque_on


def test_configure_raw_writes_caps_when_asleep(caps):
    caps(accel=40, torque=600)
    writes = _raw_configure(energize=False)
    assert (1, servo_caps.TORQUE_LIMIT_REG, 600) in writes
    assert (1, 40, 1) not in writes


# --- LeLampFollower.configure ----------------------------------------------------------------

class _Bus:
    motors = ("base_yaw", "base_pitch")

    def __init__(self):
        self.configure_kwargs = None
        self.writes = []

    @contextlib.contextmanager
    def torque_disabled(self):
        yield

    def configure_motors(self, **kwargs):
        self.configure_kwargs = kwargs

    def write(self, name, motor, value):
        self.writes.append((name, motor, value))


def _follower_configure():
    pytest.importorskip("lerobot")
    from hal.follower.hal_follower import LeLampFollower

    follower = object.__new__(LeLampFollower)
    follower.bus = _Bus()
    follower.configure()
    return follower.bus


def test_follower_configure_unchanged_without_caps(caps):
    bus = _follower_configure()
    assert bus.configure_kwargs == {"acceleration": 254}  # LeRobot's own default
    assert not [w for w in bus.writes if w[0] == "Torque_Limit"]


def test_follower_configure_applies_caps(caps):
    caps(accel=40, torque=600)
    bus = _follower_configure()
    assert bus.configure_kwargs == {"acceleration": 40}
    assert [w for w in bus.writes if w[0] == "Torque_Limit"] == [
        ("Torque_Limit", "base_yaw", 600), ("Torque_Limit", "base_pitch", 600)]


# --- every named Acceleration write in HAL goes through the cap ------------------------------

def test_every_acceleration_write_is_capped():
    """A new `write("Acceleration", ...)` or `sync_write("Acceleration", ...)` must go
    through capped_acceleration(), or HAL_SERVO_ACCEL silently stops being a cap."""
    offenders, seen = [], 0
    for root, dirs, files in os.walk(HAL_DIR):
        dirs[:] = [d for d in dirs if d not in (".venv", "test", "__pycache__")]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("write", "sync_write") and len(node.args) >= 2):
                    continue
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value == "Acceleration":
                    seen += 1
                    if "capped_acceleration" not in ast.unparse(node.args[1]):
                        offenders.append(f"{os.path.relpath(path, HAL_DIR)}:{node.lineno}")
    assert seen >= 2  # the tracker's start and end writes; guards against a vacuous pass
    assert offenders == []
