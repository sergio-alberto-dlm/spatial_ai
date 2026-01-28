"""Teleoperation interfaces for data collection."""

from spatial_ai.recording.adapters.base import TeleoperationInterface

__all__ = ["TeleoperationInterface"]

# Optional imports for concrete implementations
try:
    from spatial_ai.recording.teleop.keyboard import KeyboardTeleop, KeyboardTeleopConfig

    __all__.extend(["KeyboardTeleop", "KeyboardTeleopConfig"])
except ImportError:
    pass  # pynput not installed
