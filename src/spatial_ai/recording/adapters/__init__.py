"""Simulator adapters for unified data collection interface."""

from spatial_ai.recording.adapters.base import (
    SimulatorAdapter,
    SimulatorAdapterFactory,
    TeleoperationInterface,
)

__all__ = [
    "SimulatorAdapter",
    "SimulatorAdapterFactory",
    "TeleoperationInterface",
]

# Optional imports for concrete adapters
try:
    from spatial_ai.recording.adapters.maniskill import ManiSkillAdapter, ManiSkillConfig

    __all__.extend(["ManiSkillAdapter", "ManiSkillConfig"])
except ImportError:
    pass  # ManiSkill3 not installed
