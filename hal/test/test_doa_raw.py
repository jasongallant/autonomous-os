"""Raw-mic direction of arrival: SRP-PHAT on synthetic arrivals over the Flex ring geometry."""

import numpy as np
import pytest

from hal import config
from hal.drivers.voice import doa_raw

MICS = doa_raw.parse_mics_mm(config.DOA_MIC_MM)


def _arrival(azimuth_deg, seconds=1.0, seed=0, level=2000.0, noise=0.05):
    """Four channels of speech-like bursts from a far source at `azimuth_deg` (mic frame)."""
    rng = np.random.default_rng(seed)
    n = int(doa_raw.FS * seconds)
    src = rng.standard_normal(n) * (1 + 3 * (rng.random(n // 2000 + 1).repeat(2000)[:n] > 0.5))
    spec = np.fft.rfft(src)
    f = np.fft.rfftfreq(n, 1.0 / doa_raw.FS)
    u = np.array([np.cos(np.radians(azimuth_deg)), np.sin(np.radians(azimuth_deg))])
    tau = -(MICS @ u) / 1000.0 / doa_raw.SPEED_OF_SOUND        # mics toward the source hear it first
    chans = [np.fft.irfft(spec * np.exp(-2j * np.pi * f * t), n=n) for t in tau]
    x = np.stack(chans, axis=1)
    x += noise * rng.standard_normal(x.shape)
    return (x / np.abs(x).max() * level).astype(np.int16)


def _feed(doa, x, t_end):
    """Feed in HOP blocks so the last sample lands at t_end."""
    n = len(x)
    for k in range(0, n, doa_raw.HOP):
        block = x[k:k + doa_raw.HOP]
        doa.feed(block, t=t_end - (n - (k + len(block))) / doa_raw.FS)


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 104.0, 180.0, 251.0, 330.0])
def test_finds_a_synthetic_talker(azimuth):
    doa = doa_raw.RawMicDoa("device_doa", MICS)
    quiet = (np.random.default_rng(9).standard_normal((doa_raw.FS, 4)) * 20).astype(np.int16)
    _feed(doa, np.concatenate([quiet, _arrival(azimuth)]), t_end=2.0)
    got, n = doa.bearing_between(1.0, 2.0)
    assert n >= doa_raw.MIN_FRAMES
    assert abs((got - azimuth + 180) % 360 - 180) <= 3


def test_a_quiet_window_is_not_a_direction():
    doa = doa_raw.RawMicDoa("device_doa", MICS)
    quiet = (np.random.default_rng(3).standard_normal((2 * doa_raw.FS, 4)) * 20).astype(np.int16)
    _feed(doa, quiet, t_end=2.0)
    assert doa.bearing_between(1.0, 2.0)[0] is None


def test_too_few_frames():
    doa = doa_raw.RawMicDoa("device_doa", MICS)
    _feed(doa, _arrival(90.0, seconds=0.1), t_end=1.0)
    assert doa.bearing_between(0.0, 1.0) == (None, 2)  # 0.1 s makes two 64 ms frames


def test_frames_are_stamped_at_their_end():
    doa = doa_raw.RawMicDoa("device_doa", MICS)
    _feed(doa, _arrival(90.0, seconds=0.5), t_end=10.0)
    stamps = [t for t, _rms, _c in doa._frames]
    assert stamps[-1] == pytest.approx(10.0 - (len(_arrival(90.0, 0.5)) - doa_raw.N
                                              - doa_raw.HOP * (len(stamps) - 1)) / doa_raw.FS)
    assert all(b - a == pytest.approx(doa_raw.HOP / doa_raw.FS) for a, b in zip(stamps, stamps[1:]))


def test_geometry_parsing():
    assert MICS.shape == (4, 2) and MICS[0].tolist() == [22.2, -22.2]
    with pytest.raises(ValueError):
        doa_raw.parse_mics_mm("1,2,3;4,5,6")
    with pytest.raises(ValueError):
        doa_raw.parse_mics_mm("1,2")


def test_resolve_input_by_alsa_name():
    class _Sd:
        @staticmethod
        def query_devices():
            return [{"name": "default", "max_input_channels": 2},
                    {"name": "device_doa", "max_input_channels": 4},
                    {"name": "device_speaker", "max_input_channels": 0}]

    assert doa_raw.resolve_input("device_doa", _Sd) == 1
    assert doa_raw.resolve_input("plug:device_doa", _Sd) == 1
    assert doa_raw.resolve_input("missing", _Sd) == "missing"
    assert doa_raw.resolve_input(3, _Sd) == 3
