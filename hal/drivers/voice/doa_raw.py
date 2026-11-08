"""Sound direction from the mic array's raw channels: SRP-PHAT over a speech onset.

For arrays whose chip direction can't follow a new talker. On Bernard (reSpeaker Flex in
the base, bench 2026-10-24) the chip's processed DoA only picks between two focused beams
that barely move, while the four raw mics, steered over the firmware's own geometry, put
88-100 % of 0.5 s windows within 30 deg of the pose the yaw limits allow.

A capture thread reads the raw channels (an ALSA PCM that routes only them, e.g.
`device_doa`) and keeps, per 32 ms hop, the PHAT-weighted cross-spectrum of every mic pair.
Steering is linear in those spectra, so a speech start sums the window's loud frames and
steers once. The azimuth is the source direction in the mic coordinates' own frame,
degrees counter-clockwise from +x; doa_turn maps it onto the body.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FS = 16000
N = 1024            # 64 ms frames
HOP = 512           # 32 ms
BAND_HZ = (300.0, 4000.0)
SPEED_OF_SOUND = 343.0
GRID = np.arange(360.0)
MIN_FRAMES = 4
# The window's loudest frame must stand this far above the quiet floor of the last few
# seconds. Loose on purpose: VAD has already confirmed speech, and a talk that began
# earlier raises the floor (2026-10-24: a 3x gate refused a live speech start).
FLOOR_RATIO = 1.5


def parse_mics_mm(spec: str) -> np.ndarray:
    """"x,y;x,y;..." in mm, one pair per raw channel in capture order."""
    pts = [tuple(float(v) for v in p.split(",")) for p in spec.split(";") if p.strip()]
    if len(pts) < 2 or any(len(p) != 2 for p in pts):
        raise ValueError(f"mic geometry needs x,y pairs separated by ';', got {spec!r}")
    return np.array(pts, dtype=float)


def _pairs(m: int):
    return [(a, b) for a in range(m) for b in range(a + 1, m)]


def steering(mics_mm: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """(P, F, G) phase terms: a pair's cross-spectrum times these, summed over F, is its
    PHAT correlation at the delay a source at each grid azimuth would give."""
    phi = np.radians(GRID)
    u = np.stack([np.cos(phi), np.sin(phi)], axis=-1)                  # (G, 2)
    out = []
    for a, b in _pairs(len(mics_mm)):
        # Seconds by which mic a hears a far source later than mic b.
        tau = -(u @ (mics_mm[a] - mics_mm[b])) / 1000.0 / SPEED_OF_SOUND  # (G,)
        out.append(np.exp(2j * np.pi * freqs[:, None] * tau[None, :]))
    return np.stack(out).astype(np.complex64)


class RawMicDoa:
    """Background SRP-PHAT on the raw mics; `bearing_between()` answers for a time window."""

    def __init__(self, device: Any, mics_mm: np.ndarray, keep_s: float = 3.0,
                 paused: Callable[[], bool] = lambda: False) -> None:
        self._device = device
        self._mics = np.asarray(mics_mm, dtype=float)
        freqs = np.fft.rfftfreq(N, 1.0 / FS)
        self._band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
        self._steer = steering(self._mics, freqs[self._band])
        self._pairs = _pairs(len(self._mics))
        self._window = np.hanning(N)
        self._frames: Deque[Tuple[float, float, np.ndarray]] = deque(maxlen=int(keep_s * FS / HOP) + 1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._paused = paused
        self._tail = np.zeros((0, len(self._mics)), dtype=np.float32)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="doa-raw", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def feed(self, block: np.ndarray, t: Optional[float] = None) -> None:
        """Take (n, mics) samples; every full frame they complete is analysed and kept."""
        t = time.monotonic() if t is None else t
        buf = np.concatenate([self._tail, np.asarray(block, dtype=np.float32)])
        hops = (len(buf) - N) // HOP + 1 if len(buf) >= N else 0
        for k in range(hops):
            # A frame ends (len(buf) - (k*HOP + N)) samples before the block's last one.
            end_lag = (len(buf) - (k * HOP + N)) / FS
            self._analyse(buf[k * HOP:k * HOP + N], t - end_lag)
        self._tail = buf[hops * HOP:] if hops else buf

    def _analyse(self, frame: np.ndarray, t: float) -> None:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        spec = np.fft.rfft(frame * self._window[:, None], axis=0)[self._band]  # (F, M)
        cross = np.empty((len(self._pairs), spec.shape[0]), dtype=np.complex64)
        for p, (a, b) in enumerate(self._pairs):
            r = spec[:, a] * np.conj(spec[:, b])
            cross[p] = r / (np.abs(r) + 1e-9)
        with self._lock:
            self._frames.append((t, rms, cross))

    def bearing_between(self, t0: float, t1: float) -> Tuple[Optional[float], int]:
        """Azimuth of the loud frames between t0 and t1, and how many were used."""
        with self._lock:
            frames = list(self._frames)
        inside = [(rms, c) for t, rms, c in frames if t0 <= t <= t1]
        if len(inside) < MIN_FRAMES:
            return None, len(inside)
        floor = float(np.percentile([rms for _, rms, _ in frames], 10))
        level = np.array([rms for rms, _ in inside])
        if level.max() < FLOOR_RATIO * max(floor, 1.0):
            logger.info("[doa] raw: loudest frame %.0f rms is under %.1fx the %.0f floor",
                        level.max(), FLOOR_RATIO, floor)
            return None, 0
        loud = [c for rms, c in inside if rms >= np.median(level)]
        if len(loud) < MIN_FRAMES:
            loud = [c for _, c in sorted(inside, key=lambda rc: rc[0])[-MIN_FRAMES:]]
        total = np.sum(loud, axis=0)                                       # (P, F)
        srp = np.einsum("pf,pfg->g", total, self._steer).real
        return float(GRID[int(np.argmax(srp))]), len(loud)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._capture()
            except Exception as e:  # a background reader must never take HAL down
                logger.warning("[doa] raw-mic capture stopped (%s); retrying in 5 s", e)
            if self._stop.wait(5.0):
                return

    def _capture(self) -> None:
        import sounddevice as sd

        device = resolve_input(self._device, sd)
        with sd.InputStream(device=device, channels=len(self._mics), samplerate=FS,
                            dtype="int16", blocksize=HOP) as stream:
            logger.info("[doa] raw-mic capture on %s (%d ch)", self._device, len(self._mics))
            while not self._stop.is_set():
                block, _overflow = stream.read(HOP)
                if self._paused():
                    self._tail = self._tail[:0]
                    continue
                self.feed(block)


def resolve_input(name: Any, sd: Any) -> Any:
    """An ALSA PCM name (as in asound.conf) to its sounddevice index; anything else as is."""
    if not isinstance(name, str):
        return name
    want = name.split(":")[-1]
    for i, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) and (
            info["name"] == want or info["name"].startswith(want + ",")
        ):
            return i
    return name
