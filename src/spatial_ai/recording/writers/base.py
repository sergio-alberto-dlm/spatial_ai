"""Abstract base classes for dataset writers.

This module defines the interface that all dataset writers must implement,
enabling consistent data output across different formats (LeRobot, HDF5, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from spatial_ai.recording.types import (
    Action,
    EpisodeMetadata,
    Frame,
    Observation,
    SensorConfig,
)


@dataclass
class WriterConfig:
    """Base configuration for dataset writers.

    Attributes:
        output_dir: Directory to write dataset to.
        overwrite: Whether to overwrite existing data.
        compression: Compression settings (format-specific).
    """

    output_dir: Path
    overwrite: bool = False
    compression: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Convert string path to Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


class DatasetWriter(ABC):
    """Abstract interface for dataset writers.

    Writers handle the conversion and storage of recorded data into
    specific dataset formats. They should handle:
    - Buffering and batching for efficient I/O
    - Format-specific encoding (video, parquet, etc.)
    - Metadata management
    - Incremental writing (streaming to disk)
    """

    def __init__(self, config: WriterConfig, sensor_configs: Dict[str, SensorConfig]):
        """Initialize the writer.

        Args:
            config: Writer configuration.
            sensor_configs: Configuration for each sensor being recorded.
        """
        self._config = config
        self._sensor_configs = sensor_configs
        self._is_open = False
        self._current_episode_index = 0

    @abstractmethod
    def open(self) -> None:
        """Open the writer and prepare for data.

        This should create necessary directories, initialize file handles,
        and prepare any buffers.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the writer and finalize all data.

        This should flush any remaining buffers, write metadata,
        and close file handles.
        """
        pass

    @abstractmethod
    def start_episode(self, task_description: str) -> str:
        """Start recording a new episode.

        Args:
            task_description: Human-readable description of the task.

        Returns:
            Episode ID for tracking.
        """
        pass

    @abstractmethod
    def add_frame(
        self,
        observations: Dict[str, Observation],
        action: Optional[Action] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single frame to the current episode.

        Args:
            observations: Dictionary of sensor observations.
            action: Optional action command.
            info: Optional additional information.
        """
        pass

    @abstractmethod
    def end_episode(self, success: bool, notes: str = "") -> EpisodeMetadata:
        """End the current episode.

        Args:
            success: Whether the episode was successful.
            notes: Optional notes about the episode.

        Returns:
            Metadata for the completed episode.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the dataset after all episodes are written.

        This should compute statistics, validate the dataset,
        and prepare it for use.
        """
        pass

    def __enter__(self) -> "DatasetWriter":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def is_open(self) -> bool:
        """Check if writer is open."""
        return self._is_open

    @property
    def output_path(self) -> Path:
        """Get the output directory path."""
        return self._config.output_dir

    @property
    def num_episodes(self) -> int:
        """Get number of episodes written so far."""
        return self._current_episode_index


class DatasetWriterFactory:
    """Factory for creating dataset writers."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, writer_class: type) -> None:
        """Register a dataset writer class.

        Args:
            name: Name to register the writer under.
            writer_class: The writer class to register.
        """
        cls._registry[name] = writer_class

    @classmethod
    def create(
        cls,
        name: str,
        config: WriterConfig,
        sensor_configs: Dict[str, SensorConfig],
        **kwargs,
    ) -> DatasetWriter:
        """Create a dataset writer by name.

        Args:
            name: Name of the registered writer.
            config: Writer configuration.
            sensor_configs: Sensor configurations.
            **kwargs: Additional arguments for the writer.

        Returns:
            Instantiated dataset writer.

        Raises:
            ValueError: If writer name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown writer '{name}'. Available: {available}")
        return cls._registry[name](config, sensor_configs, **kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Return list of available writer names."""
        return list(cls._registry.keys())


@dataclass
class DatasetStats:
    """Statistics for a recorded dataset.

    Attributes:
        num_episodes: Total number of episodes.
        total_frames: Total number of frames across all episodes.
        total_duration_s: Total duration in seconds.
        success_rate: Fraction of successful episodes.
        sensor_stats: Per-sensor statistics (min, max, mean, std).
        action_stats: Action statistics (min, max, mean, std).
    """

    num_episodes: int = 0
    total_frames: int = 0
    total_duration_s: float = 0.0
    success_rate: float = 0.0
    sensor_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    action_stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_episodes": self.num_episodes,
            "total_frames": self.total_frames,
            "total_duration_s": self.total_duration_s,
            "success_rate": self.success_rate,
            "sensor_stats": self.sensor_stats,
            "action_stats": self.action_stats,
        }
