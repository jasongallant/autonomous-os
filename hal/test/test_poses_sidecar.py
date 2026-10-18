"""Tests for the per-unit poses sidecar (hal/drivers/motors/poses.py).

<HAL_DEVICE_ID>.poses.json overrides the lamp's zero/startup/rest ticks; without it,
or where it is wrong, the lamp's numbers stand. No hardware, no serial port.
"""
import ast
import json
import os

from hal.drivers.motors.poses import load_poses, poses_path

HAL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DEFAULTS = {
    "zero":    {"base_yaw": 2025, "base_pitch": 2674},
    "startup": {"base_yaw": 2025, "base_pitch": 2674},
    "rest":    {"base_yaw": 2029, "base_pitch": 2030},
}


def _write(tmp_path, payload, lamp_id="bernard"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    poses_path(lamp_id, tmp_path).write_text(text)


def test_no_file_returns_the_defaults_as_copies(tmp_path):
    poses = load_poses("bernard", DEFAULTS, tmp_path)
    assert poses == DEFAULTS
    poses["rest"]["base_yaw"] = 1
    assert DEFAULTS["rest"]["base_yaw"] == 2029


def test_file_is_named_after_the_device_id(tmp_path):
    assert poses_path("bernard-fd7c", tmp_path) == tmp_path / "bernard-fd7c.poses.json"


def test_full_file_overrides_every_pose(tmp_path):
    _write(tmp_path, {
        "zero":    {"base_yaw": 2047, "base_pitch": 1469},
        "startup": {"base_yaw": 2046, "base_pitch": 1470},
        "rest":    {"base_yaw": 2045, "base_pitch": 1471},
    })
    poses = load_poses("bernard", DEFAULTS, tmp_path)
    assert poses["zero"] == {"base_yaw": 2047, "base_pitch": 1469}
    assert poses["startup"] == {"base_yaw": 2046, "base_pitch": 1470}
    assert poses["rest"] == {"base_yaw": 2045, "base_pitch": 1471}


def test_partial_file_keeps_defaults_for_the_rest(tmp_path):
    _write(tmp_path, {"rest": {"base_pitch": 1500}})
    poses = load_poses("bernard", DEFAULTS, tmp_path)
    assert poses["rest"] == {"base_yaw": 2029, "base_pitch": 1500}
    assert poses["zero"] == DEFAULTS["zero"]
    assert poses["startup"] == DEFAULTS["startup"]


def test_another_devices_file_is_not_read(tmp_path):
    _write(tmp_path, {"rest": {"base_yaw": 1}}, lamp_id="other")
    assert load_poses("bernard", DEFAULTS, tmp_path) == DEFAULTS


def test_unreadable_or_malformed_files_fall_back(tmp_path):
    for text in ("", "{not json", "[1, 2]", '"rest"', "null"):
        _write(tmp_path, text)
        assert load_poses("bernard", DEFAULTS, tmp_path) == DEFAULTS, text


def test_invalid_entries_are_skipped_and_valid_ones_kept(tmp_path):
    _write(tmp_path, {
        "rest": {
            "base_yaw": 4096,        # past the 12-bit register
            "base_pitch": -1,
            "nope": 100,             # not a joint
        },
        "zero": {"base_yaw": "2047", "base_pitch": 1469},   # string, then a good value
        "startup": {"base_yaw": True, "base_pitch": 12.5},  # bool and float are not ticks
        "elsewhere": {"base_yaw": 1},                       # not a pose
        "nested": 5,
    })
    poses = load_poses("bernard", DEFAULTS, tmp_path)
    assert poses["rest"] == DEFAULTS["rest"]
    assert poses["zero"] == {"base_yaw": 2025, "base_pitch": 1469}
    assert poses["startup"] == DEFAULTS["startup"]
    assert set(poses) == set(DEFAULTS)


def test_pose_that_is_not_an_object_is_ignored(tmp_path):
    _write(tmp_path, {"rest": [1, 2, 3], "zero": {"base_yaw": 2000}})
    poses = load_poses("bernard", DEFAULTS, tmp_path)
    assert poses["rest"] == DEFAULTS["rest"]
    assert poses["zero"]["base_yaw"] == 2000


def test_bounds_are_inclusive(tmp_path):
    _write(tmp_path, {"rest": {"base_yaw": 0, "base_pitch": 4095}})
    assert load_poses("bernard", DEFAULTS, tmp_path)["rest"] == {"base_yaw": 0, "base_pitch": 4095}


def test_animation_service_reads_poses_only_from_the_loaded_copies():
    """The module constants are defaults; every move must use the per-unit values."""
    path = os.path.join(HAL_DIR, "drivers", "motors", "animation_service.py")
    tree = ast.parse(open(path).read())
    constants = {"ZERO_RAW", "STARTUP_RAW", "REST_RAW"}
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in constants and isinstance(node.ctx, ast.Load):
            reads.append(node.lineno)
    # One read each, in AnimationService.__init__, as load_poses' defaults.
    assert len(reads) == 3, f"raw pose constants read at lines {reads}"
    init = next(
        n for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == "AnimationService"
        for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assert all(init.lineno <= line <= init.end_lineno for line in reads)
