"""Optional hardware caps on servo acceleration and torque.

HAL_SERVO_ACCEL and HAL_SERVO_TORQUE_LIMIT (hal/config.py) let a body whose
servos are stronger than the lamp's STS3215, or whose supply cannot feed every
joint stalling at once, limit what the servos may do. Unset, nothing here
changes a value HAL writes.

Every Acceleration write in HAL goes through capped_acceleration(), so a
tracking session or LeRobot's configure_motors() cannot lift the cap.
Torque_Limit is written once per configure, and a per-joint limit from the board's
servo_overload config goes through capped_torque_limit(), so it can only lower the cap.
Both are RAM registers on the STS series (no Lock, no EEPROM wear).
"""

from typing import Optional

from hal import config

ACCELERATION_REG = 41  # 1 byte
TORQUE_LIMIT_REG = 48  # 2 bytes


def capped_acceleration(requested: int) -> int:
    """`requested` limited to HAL_SERVO_ACCEL.

    The servo reads Acceleration 0 as no ramp at all, the hardest possible
    start, so under a cap 0 becomes the cap rather than passing through.
    """
    cap = config.SERVO_ACCEL
    if cap is None:
        return requested
    return cap if requested <= 0 else min(requested, cap)


def capped_torque_limit(requested: Optional[int]) -> Optional[int]:
    """`requested` limited to HAL_SERVO_TORQUE_LIMIT; None stays None.

    Used for the board's per-joint limit: it may lower a joint below the cap, never above it.
    """
    cap = config.SERVO_TORQUE_LIMIT
    if requested is None or cap is None:
        return requested
    return min(requested, cap)


def write_caps_raw(packet_handler, port_handler, servo_id: int) -> list[str]:
    """Write the configured caps to one servo over scservo_sdk.

    Returns a short description of what was written, for the configure log.
    """
    written = []
    if config.SERVO_ACCEL is not None:
        packet_handler.write1ByteTxRx(port_handler, servo_id, ACCELERATION_REG, config.SERVO_ACCEL)
        written.append(f"accel={config.SERVO_ACCEL}")
    if config.SERVO_TORQUE_LIMIT is not None:
        packet_handler.write2ByteTxRx(port_handler, servo_id, TORQUE_LIMIT_REG, config.SERVO_TORQUE_LIMIT)
        written.append(f"torque_limit={config.SERVO_TORQUE_LIMIT}")
    return written
