"""Focus flags the rpicam driver passes to rpicam-vid."""
import pytest

pytest.importorskip("cv2")

from hal.drivers.camera.rpicam_capture_device import _focus_args

_VARS = ("HAL_CAMERA_AUTOFOCUS", "HAL_CAMERA_LENS_POSITION", "HAL_CAMERA_AUTOFOCUS_RANGE")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)


def test_unset_adds_no_focus_flags():
    assert _focus_args() == []


def test_continuous_autofocus(monkeypatch):
    monkeypatch.setenv("HAL_CAMERA_AUTOFOCUS", "Continuous")
    assert _focus_args() == ["--autofocus-mode", "continuous"]


def test_autofocus_range_follows_the_mode(monkeypatch):
    monkeypatch.setenv("HAL_CAMERA_AUTOFOCUS", "continuous")
    monkeypatch.setenv("HAL_CAMERA_AUTOFOCUS_RANGE", "normal")
    assert _focus_args() == ["--autofocus-mode", "continuous",
                             "--autofocus-range", "normal"]


def test_lens_position_wins_over_autofocus(monkeypatch):
    monkeypatch.setenv("HAL_CAMERA_AUTOFOCUS", "continuous")
    monkeypatch.setenv("HAL_CAMERA_LENS_POSITION", "2.0")
    assert _focus_args() == ["--lens-position", "2.0"]


@pytest.mark.parametrize("var,value", [
    ("HAL_CAMERA_AUTOFOCUS", "sharp"),
    ("HAL_CAMERA_LENS_POSITION", "near"),
    ("HAL_CAMERA_LENS_POSITION", "-1"),
    ("HAL_CAMERA_AUTOFOCUS_RANGE", "far"),
])
def test_bad_values_are_ignored(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    assert _focus_args() == []
