"""Core data types for the recording module.

This module defines the fundamental data structures used throughout the
data recording pipeline, including sensor observations, robot actions,
and episode metadata.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class SensorType(Enum):
    """Supported sensor modalities for data recording."""

    # Visual sensors
    RGB = auto()
    DEPTH = auto()
    SEGMENTATION = auto()
    POINT_CLOUD = auto()

    # Proprioceptive sensors
    JOINT_STATE = auto()
    JOINT_VELOCITY = auto()
    JOINT_TORQUE = auto()
    END_EFFECTOR_POSE = auto()

    # Force/tactile sensors
    FORCE_TORQUE = auto()
    TACTILE = auto()

    # Motion sensors
    IMU = auto()

    # Cable-specific (deformable object tracking)
    CABLE_KEYPOINTS = auto()
    CABLE_MESH = auto()


class ControlMode(Enum):
    """Robot control modes for action commands."""

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    END_EFFECTOR_POSE = "ee_pose"
    END_EFFECTOR_VELOCITY = "ee_velocity"
    DELTA_JOINT_POSITION = "delta_joint_position"
    DELTA_END_EFFECTOR_POSE = "delta_ee_pose"


@dataclass
class SensorConfig:
    """Configuration for a single sensor.

    Attributes:
        name: Unique identifier for this sensor.
        sensor_type: Type of sensor (RGB, DEPTH, JOINT_STATE, etc.).
        frequency_hz: Nominal sampling frequency in Hz.
        shape: Shape of the sensor data array.
        dtype: NumPy dtype of the sensor data.
        transform: Optional 4x4 extrinsic transformation matrix (sensor to world).
        intrinsics: Optional camera intrinsics for visual sensors.
        metadata: Additional sensor-specific configuration.
    """

    name: str
    sensor_type: SensorType
    frequency_hz: float
    shape: Tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    transform: Optional[NDArray] = None
    intrinsics: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {self.frequency_hz}")
        if not self.shape:
            raise ValueError("shape cannot be empty")
        if self.transform is not None and self.transform.shape != (4, 4):
            raise ValueError(f"transform must be 4x4, got {self.transform.shape}")


@dataclass
class Observation:
    """Single observation from a sensor.

    Attributes:
        sensor_name: Name of the sensor that produced this observation.
        timestamp_ns: Timestamp in nanoseconds (monotonic clock).
        data: Observation data as a NumPy array.
        metadata: Optional additional metadata (e.g., camera exposure, gain).
    """

    sensor_name: str
    timestamp_ns: int
    data: NDArray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_s(self) -> float:
        """Get timestamp in seconds."""
        return self.timestamp_ns / 1e9

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensor_name": self.sensor_name,
            "timestamp_ns": self.timestamp_ns,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Create from dictionary."""
        return cls(
            sensor_name=data["sensor_name"],
            timestamp_ns=data["timestamp_ns"],
            data=np.asarray(data["data"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Action:
    """Robot action command.

    Attributes:
        timestamp_ns: Timestamp when action was commanded (nanoseconds).
        values: Action values as a NumPy array.
        control_mode: Control mode (joint_position, ee_pose, etc.).
        metadata: Optional additional metadata.
    """

    timestamp_ns: int
    values: NDArray
    control_mode: ControlMode = ControlMode.JOINT_POSITION
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_s(self) -> float:
        """Get timestamp in seconds."""
        return self.timestamp_ns / 1e9

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "values": self.values,
            "control_mode": self.control_mode.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create from dictionary."""
        return cls(
            timestamp_ns=data["timestamp_ns"],
            values=np.asarray(data["values"]),
            control_mode=ControlMode(data["control_mode"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EpisodeMetadata:
    """Metadata for a complete episode.

    Attributes:
        episode_id: Unique identifier for this episode.
        task_description: Human-readable description of the task.
        start_timestamp_ns: Episode start time in nanoseconds.
        end_timestamp_ns: Episode end time in nanoseconds.
        success: Whether the episode completed successfully.
        simulator: Name of the simulator used (e.g., "maniskill3", "phystwin").
        robot_config: Robot configuration details.
        num_frames: Total number of frames in the episode.
        notes: Optional human-written notes.
        tags: Optional tags for categorization.
    """

    episode_id: str
    task_description: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    success: bool
    simulator: str
    robot_config: Dict[str, Any] = field(default_factory=dict)
    num_frames: int = 0
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """Get episode duration in seconds."""
        return (self.end_timestamp_ns - self.start_timestamp_ns) / 1e9

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "task_description": self.task_description,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "success": self.success,
            "simulator": self.simulator,
            "robot_config": self.robot_config,
            "num_frames": self.num_frames,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeMetadata":
        """Create from dictionary."""
        return cls(
            episode_id=data["episode_id"],
            task_description=data["task_description"],
            start_timestamp_ns=data["start_timestamp_ns"],
            end_timestamp_ns=data["end_timestamp_ns"],
            success=data["success"],
            simulator=data["simulator"],
            robot_config=data.get("robot_config", {}),
            num_frames=data.get("num_frames", 0),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


@dataclass
class Frame:
    """A single synchronized frame of data.

    Contains all observations and action for a single timestep,
    synchronized to a master timestamp.

    Attributes:
        timestamp_ns: Master timestamp in nanoseconds.
        frame_index: Index of this frame within the episode.
        observations: Dictionary of sensor observations.
        action: Optional action command for this frame.
        info: Additional frame information (e.g., reward, done flags).
    """

    timestamp_ns: int
    frame_index: int
    observations: Dict[str, Observation]
    action: Optional[Action] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_s(self) -> float:
        """Get timestamp in seconds."""
        return self.timestamp_ns / 1e9


@dataclass
class Episode:
    """A complete episode of recorded data.

    Attributes:
        metadata: Episode metadata.
        frames: List of synchronized frames.
        sensor_configs: Configuration for each sensor.
    """

    metadata: EpisodeMetadata
    frames: List[Frame]
    sensor_configs: Dict[str, SensorConfig]

    def __len__(self) -> int:
        """Return number of frames in the episode."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> Frame:
        """Get frame by index."""
        return self.frames[idx]

    def get_observations(self, sensor_name: str) -> List[Observation]:
        """Get all observations from a specific sensor."""
        return [f.observations.get(sensor_name) for f in self.frames if sensor_name in f.observations]

    def get_actions(self) -> List[Action]:
        """Get all actions in the episode."""
        return [f.action for f in self.frames if f.action is not None]
