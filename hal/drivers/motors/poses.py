"""Per-unit rest, zero and startup poses (raw servo ticks) from a sidecar file.

ZERO_RAW, STARTUP_RAW and REST_RAW in animation_service.py are ticks tuned on the
lamp's STS3215 build. A body with other servos, horns or joint ranges needs its own,
and calibration does not cover them (they bypass it). The sidecar sits beside the
unit's calibration file:

    <calibration dir>/<HAL_DEVICE_ID>.poses.json

    {"zero":    {"base_yaw": 2025, "base_pitch": 2030, ...},
     "startup": {"base_yaw": 2025, ...},
     "rest":    {"base_yaw": 2029, ...}}

Every key is optional, and so is every joint inside a pose: whatever the file does not
name keeps the lamp's value. A missing file changes nothing. A file that cannot be
read, or an entry that is not a valid tick for a known joint, is logged and skipped;
HAL still starts, on the lamp's numbers for that entry.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Union

logger = logging.getLogger(__name__)

TICKS_MAX = 4095  # STS series position register, 12 bits

Pose = Dict[str, int]


def poses_path(lamp_id: str, directory: Union[str, Path]) -> Path:
    return Path(directory) / f"{lamp_id}.poses.json"


def load_poses(
    lamp_id: str,
    defaults: Dict[str, Pose],
    directory: Union[str, Path],
) -> Dict[str, Pose]:
    """`defaults` ({"zero": {...}, "startup": {...}, "rest": {...}}) with the sidecar laid over it.

    Always returns fresh dicts, so callers may keep them without aliasing `defaults`.
    """
    poses = {name: dict(pose) for name, pose in defaults.items()}
    path = poses_path(lamp_id, directory)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return poses
    except OSError as e:
        logger.warning("poses sidecar %s unreadable (%s); using defaults", path, e)
        return poses

    try:
        data = json.loads(raw)
    except ValueError as e:
        logger.warning("poses sidecar %s is not valid JSON (%s); using defaults", path, e)
        return poses
    if not isinstance(data, dict):
        logger.warning("poses sidecar %s must be a JSON object; using defaults", path)
        return poses

    applied = 0
    for name, pose in data.items():
        if name not in poses:
            logger.warning("poses sidecar %s: unknown pose %r ignored", path, name)
            continue
        if not isinstance(pose, dict):
            logger.warning("poses sidecar %s: pose %r must be an object; ignored", path, name)
            continue
        for joint, ticks in pose.items():
            if joint not in poses[name]:
                logger.warning("poses sidecar %s: %s has no joint %r; ignored", path, name, joint)
            elif not _is_tick(ticks):
                logger.warning(
                    "poses sidecar %s: %s.%s = %r is not an integer 0-%d; ignored",
                    path, name, joint, ticks, TICKS_MAX,
                )
            else:
                poses[name][joint] = ticks
                applied += 1
    logger.info("poses sidecar %s: %d value(s) override the defaults", path, applied)
    return poses


def _is_tick(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= TICKS_MAX
