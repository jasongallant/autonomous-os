"""Sound direction of arrival from the reSpeaker XVF3800 over its USB control endpoint.

The read is a vendor control transfer on the device, separate from the ALSA capture
stream, so it never disturbs the voice pipeline. Azimuth is the chip's own: degrees
0..359 in the array's frame. Mapping it onto the body's yaw is `doa_turn`'s job.
"""

from __future__ import annotations

import logging
import statistics
import struct
import threading
from dataclasses import dataclass
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)

XVF_VENDOR_ID = 0x2886
# xvf_host's DOA_VALUE: resource 20, command 18, two uint16 (azimuth deg, speech flag).
_DOA_RESID = 20
_DOA_CMD = 18
_DOA_BYTES = 4
_READ_FLAG = 0x80
_CTRL_IN_VENDOR_DEVICE = 0xC0
_TIMEOUT_MS = 100


@dataclass(frozen=True)
class DoaSample:
    azimuth_deg: float
    speech: bool


class DoaReader(Protocol):
    def read(self) -> Optional[DoaSample]: ...


class UsbDoaReader:
    """Reads DOA_VALUE from the first XVF3800 on the bus. `read()` never raises."""

    def __init__(self, vendor_id: int = XVF_VENDOR_ID) -> None:
        self._vendor_id = vendor_id
        self._dev = None
        self._lock = threading.Lock()

    def _open(self):
        import usb.core  # pyusb, imported late so the HAL runs without it

        dev = usb.core.find(idVendor=self._vendor_id)
        if dev is None:
            raise RuntimeError("no XVF3800 on the USB bus")
        return dev

    def read(self) -> Optional[DoaSample]:
        with self._lock:
            try:
                if self._dev is None:
                    self._dev = self._open()
                raw = bytes(self._dev.ctrl_transfer(
                    _CTRL_IN_VENDOR_DEVICE, 0, _DOA_CMD | _READ_FLAG, _DOA_RESID,
                    _DOA_BYTES + 1, _TIMEOUT_MS,
                ))
            except Exception as e:
                logger.debug("[doa] read failed: %s", e)
                self._dev = None
                return None
        return parse_doa_reply(raw)


def parse_doa_reply(raw: bytes) -> Optional[DoaSample]:
    """First byte is the control status (0 = ok), then azimuth and speech as uint16 LE."""
    if len(raw) < _DOA_BYTES + 1 or raw[0] != 0:
        return None
    azimuth, speech = struct.unpack("<HH", raw[1:_DOA_BYTES + 1])
    return DoaSample(float(azimuth % 360), bool(speech))


class FakeDoaReader:
    """Sim and tests: returns whatever azimuth was set last."""

    def __init__(self, azimuth_deg: Optional[float] = None, speech: bool = True) -> None:
        self.azimuth_deg = azimuth_deg
        self.speech = speech

    def read(self) -> Optional[DoaSample]:
        if self.azimuth_deg is None:
            return None
        return DoaSample(self.azimuth_deg % 360, self.speech)


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
