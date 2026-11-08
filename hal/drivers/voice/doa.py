"""Sound direction of arrival from the reSpeaker XVF3800 over its USB control endpoint.

The read is a vendor control transfer on the device, separate from the ALSA capture
stream, so it never disturbs the voice pipeline. Azimuth is the chip's own: degrees
0..359 in the array's frame. Mapping it onto the body's yaw is `doa_turn`'s job.
"""

from __future__ import annotations

import logging
import math
import statistics
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

XVF_VENDOR_ID = 0x2886
# Ids from Seeed's reSpeaker_Flex python_control/xvf_host.py.
# DOA_VALUE: resource 20, command 18, two uint16 (azimuth deg, speech flag).
_DOA_RESID = 20
_DOA_CMD = 18
_DOA_BYTES = 4
# AUDIO_MGR_SELECTED_AZIMUTHS: resource 35, command 11, two float32 radians (the processed
# DoA, NaN when no fixed beam holds speech; the auto-select beam's azimuth).
_SEL_RESID = 35
_SEL_CMD = 11
_SEL_BYTES = 8
_READ_FLAG = 0x80
_CTRL_IN_VENDOR_DEVICE = 0xC0
_TIMEOUT_MS = 100
_STATUS_OK = 0
# The chip's servicer asks for the read again while it is busy (xvf_host retries on it).
_STATUS_RETRY = 64
_RETRIES = 3

REGISTERS = ("doa_value", "selected")
# Enough agreeing readings to trust a direction: Seeed's reBot-arm + Flex demo wants 4
# readings within 15 deg; a speech-onset window is longer, so 3 in 4 within 20 deg.
AGREE_MIN_SAMPLES = 4
AGREE_DEG = 20.0
AGREE_SHARE = 0.75


@dataclass(frozen=True)
class DoaSample:
    azimuth_deg: float
    speech: bool


class DoaReader(Protocol):
    def read(self) -> Optional[DoaSample]: ...


class UsbDoaReader:
    """Reads one direction register from the first XVF3800 on the bus. `read()` never raises."""

    def __init__(self, vendor_id: int = XVF_VENDOR_ID, register: str = "doa_value") -> None:
        if register not in REGISTERS:
            raise ValueError(f"unknown DOA register {register!r}; known: {REGISTERS}")
        self._vendor_id = vendor_id
        self._register = register
        self._dev = None
        self._lock = threading.Lock()

    def _open(self):
        import usb.core  # pyusb, imported late so the HAL runs without it

        dev = usb.core.find(idVendor=self._vendor_id)
        if dev is None:
            raise RuntimeError("no XVF3800 on the USB bus")
        return dev

    def read(self) -> Optional[DoaSample]:
        if self._register == "selected":
            resid, cmd, n, parse = _SEL_RESID, _SEL_CMD, _SEL_BYTES, parse_selected_reply
        else:
            resid, cmd, n, parse = _DOA_RESID, _DOA_CMD, _DOA_BYTES, parse_doa_reply
        with self._lock:
            try:
                if self._dev is None:
                    self._dev = self._open()
                for _ in range(_RETRIES):
                    raw = bytes(self._dev.ctrl_transfer(
                        _CTRL_IN_VENDOR_DEVICE, 0, cmd | _READ_FLAG, resid, n + 1, _TIMEOUT_MS,
                    ))
                    if not raw or raw[0] != _STATUS_RETRY:
                        break
            except Exception as e:
                logger.debug("[doa] read failed: %s", e)
                self._dev = None
                return None
        return parse(raw)


def parse_doa_reply(raw: bytes) -> Optional[DoaSample]:
    """First byte is the control status (0 = ok), then azimuth and speech as uint16 LE."""
    if len(raw) < _DOA_BYTES + 1 or raw[0] != _STATUS_OK:
        return None
    azimuth, speech = struct.unpack("<HH", raw[1:_DOA_BYTES + 1])
    return DoaSample(float(azimuth % 360), bool(speech))


def parse_selected_reply(raw: bytes) -> Optional[DoaSample]:
    """Status byte, then two float32 radians; the first is NaN while nobody speaks."""
    if len(raw) < _SEL_BYTES + 1 or raw[0] != _STATUS_OK:
        return None
    processed, _auto = struct.unpack("<ff", raw[1:_SEL_BYTES + 1])
    if math.isnan(processed):
        return DoaSample(0.0, False)
    return DoaSample(math.degrees(processed) % 360, True)


class FakeDoaReader:
    """Sim and tests: returns whatever azimuth was set last."""

    def __init__(self, azimuth_deg: Optional[float] = None, speech: bool = True) -> None:
        self.azimuth_deg = azimuth_deg
        self.speech = speech

    def read(self) -> Optional[DoaSample]:
        if self.azimuth_deg is None:
            return None
        return DoaSample(self.azimuth_deg % 360, self.speech)


class DoaPoller:
    """Keeps the last few seconds of readings, polled in the background.

    The chip lags a new talker and VAD confirms speech only after it began, so a speech
    start needs the readings from around its onset, not one read made after it.
    """

    def __init__(self, reader: DoaReader, period_s: float = 0.05, keep_s: float = 3.0,
                 paused: Callable[[], bool] = lambda: False) -> None:
        self._reader = reader
        self._period_s = period_s
        self._paused = paused
        self._buf: Deque[Tuple[float, DoaSample]] = deque(maxlen=int(keep_s / period_s) + 1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="doa-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> None:
        s = self._reader.read()
        if s is not None:
            with self._lock:
                self._buf.append((time.monotonic(), s))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._paused():
                    self.poll_once()
            except Exception as e:  # a background poller must never take HAL down
                logger.debug("[doa] poll skipped: %s", e)
            self._stop.wait(self._period_s)

    def speech_between(self, t0: float, t1: float) -> List[float]:
        """Azimuths of the speech-flagged readings taken between t0 and t1 (monotonic)."""
        with self._lock:
            return [s.azimuth_deg for t, s in self._buf if t0 <= t <= t1 and s.speech]

    def bearing_between(self, t0: float, t1: float) -> Tuple[Optional[float], int]:
        """The agreed azimuth of the window's speech readings, and how many there were."""
        angles = self.speech_between(t0, t1)
        return agreed_azimuth(angles, AGREE_MIN_SAMPLES, AGREE_DEG, AGREE_SHARE), len(angles)


def circular_median_deg(angles: List[float]) -> Optional[float]:
    """Median of angles on a circle: rotate so the data's gap sits at the seam first."""
    if not angles:
        return None
    pts = sorted(a % 360 for a in angles)
    # Seam goes in the widest gap between neighbours, so the median is taken on a line.
    gaps = [((pts[(i + 1) % len(pts)] - pts[i]) % 360 or 360, i) for i in range(len(pts))]
    _, i = max(gaps)
    start = pts[(i + 1) % len(pts)]
    return (start + statistics.median((p - start) % 360 for p in pts)) % 360


def agreed_azimuth(angles: List[float], min_count: int, tol_deg: float,
                   share: float) -> Optional[float]:
    """Median of the readings near the median, if enough of them agree; else None.

    Scattered readings are a room reflection, a door or the chip still hunting, not a
    talker to turn to.
    """
    if len(angles) < min_count:
        return None
    med = circular_median_deg(angles)
    near = [a for a in angles if abs((a - med + 180.0) % 360.0 - 180.0) <= tol_deg]
    if len(near) < share * len(angles):
        return None
    return circular_median_deg(near)
