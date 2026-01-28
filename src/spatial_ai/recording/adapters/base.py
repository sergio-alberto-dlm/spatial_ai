"""Abstract base classes for simulator adapters.

This module defines the interfaces that all simulator adapters must implement,
ensuring consistent data collection across different simulation backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from spatial_ai.recording.types import (
    Action,
    ControlMode,
    Observation,
    SensorConfig,
)


class SimulatorAdapter(ABC):
    """Abstract interface for simulator backends.

    This interface abstracts away simulator-specific APIs to provide a unified
    way to collect data from different simulation environments (ManiSkill3,
    PhysTwin, etc.).

    Implementations should handle:
    - Environment initialization and configuration
    - Observation extraction and normalization
    - Action execution
    - State save/restore for replay
    """

    @abstractmethod
    def get_sensor_configs(self) -> Dict[str, SensorConfig]:
        """Return configuration for all available sensors.

        Returns:
            Dictionary mapping sensor names to their configurations.
        """
        pass

    @abstractmethod
    def get_observations(self) -> Dict[str, Observation]:
        """Get current observations from all sensors.

        Returns:
            Dictionary mapping sensor names to current observations.
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Dict[str, Observation], Dict[str, Any]]:
        """Execute action and return new observations.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observations, info) where info contains:
                - reward: Optional reward signal
                - terminated: Whether episode ended due to success/failure
                - truncated: Whether episode ended due to time limit
                - success: Whether task was completed successfully
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Observation], Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (initial_observations, info).
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get full simulator state for replay.

        Returns:
            Dictionary containing complete simulator state that can be
            used to restore the environment to this exact configuration.
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore simulator to a saved state.

        Args:
            state: State dictionary from get_state().
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """Return action space specification.

        Returns:
            Dictionary describing the action space, including:
                - shape: Shape of action array
                - dtype: Data type
                - low: Lower bounds (if bounded)
                - high: Upper bounds (if bounded)
                - control_mode: Default control mode
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """Return observation space specification.

        Returns:
            Dictionary mapping sensor names to their space specifications.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up simulator resources."""
        pass

    # Optional methods for specific capabilities

    def get_cable_state(self) -> Optional[NDArray]:
        """Get cable keypoints/mesh for deformable tracking.

        Returns:
            Array of shape (N, 3) containing 3D positions of cable keypoints,
            or None if not available.
        """
        return None

    def get_contact_info(self) -> Optional[Dict[str, Any]]:
        """Get contact/collision information.

        Returns:
            Dictionary with contact information, or None if not available.
        """
        return None

    def render(self, mode: str = "rgb_array") -> Optional[NDArray]:
        """Render the environment.

        Args:
            mode: Rendering mode ("rgb_array", "human", etc.)

        Returns:
            Rendered image array if mode is "rgb_array", None otherwise.
        """
        return None

    @property
    def simulator_name(self) -> str:
        """Return the name of the simulator."""
        return self.__class__.__name__


class TeleoperationInterface(ABC):
    """Abstract interface for teleoperation input devices.

    This interface allows different input methods (keyboard, mouse, VR controllers,
    SpaceMouse, etc.) to be used interchangeably for data collection.
    """

    @abstractmethod
    def get_action(self) -> Optional[Action]:
        """Get action from teleop device.

        Returns:
            Action if user provided input, None if no input this frame.
        """
        pass

    @abstractmethod
    def is_episode_complete(self) -> bool:
        """Check if user signaled episode end.

        Returns:
            True if user wants to end the current episode.
        """
        pass

    @abstractmethod
    def is_episode_success(self) -> bool:
        """Check if user marked episode as success.

        Returns:
            True if user marked the episode as successful.
        """
        pass

    @abstractmethod
    def is_recording_paused(self) -> bool:
        """Check if recording is paused.

        Returns:
            True if user paused recording (for repositioning, etc.)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset teleop state for new episode."""
        pass

    def get_gripper_action(self) -> Optional[float]:
        """Get gripper action separately from arm action.

        Returns:
            Gripper command (0=open, 1=closed) or None if not applicable.
        """
        return None

    @property
    def control_mode(self) -> ControlMode:
        """Return the control mode used by this teleop interface."""
        return ControlMode.JOINT_POSITION

    def update(self) -> None:
        """Update internal state (poll devices, process events, etc.)."""
        pass


class SimulatorAdapterFactory:
    """Factory for creating simulator adapters."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type) -> None:
        """Register a simulator adapter class.

        Args:
            name: Name to register the adapter under.
            adapter_class: The adapter class to register.
        """
        cls._registry[name] = adapter_class

    @classmethod
    def create(cls, name: str, **kwargs) -> SimulatorAdapter:
        """Create a simulator adapter by name.

        Args:
            name: Name of the registered adapter.
            **kwargs: Arguments to pass to the adapter constructor.

        Returns:
            Instantiated simulator adapter.

        Raises:
            ValueError: If adapter name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Return list of available adapter names."""
        return list(cls._registry.keys())
