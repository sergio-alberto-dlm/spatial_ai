"""HDF5 dataset writer for raw trajectory storage.

This writer stores recorded data in HDF5 format, compatible with ManiSkill3
trajectory conventions for easy replay and debugging. Data can later be
converted to LeRobotDataset format.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from spatial_ai.recording.types import (
    Action,
    EpisodeMetadata,
    Observation,
    SensorConfig,
    SensorType,
)
from spatial_ai.recording.writers.base import (
    DatasetStats,
    DatasetWriter,
    DatasetWriterFactory,
    WriterConfig,
)


@dataclass
class HDF5WriterConfig(WriterConfig):
    """Configuration for HDF5 dataset writer.

    Attributes:
        compression: Compression algorithm ("gzip", "lzf", or None).
        compression_opts: Compression level (1-9 for gzip).
        include_env_states: Whether to store full simulator state for replay.
        chunk_episodes: Number of episodes per HDF5 file before rotation.
    """

    compression: Optional[str] = "gzip"
    compression_opts: int = 4
    include_env_states: bool = True
    chunk_episodes: int = 1000

    def __post_init__(self):
        """Validate and convert config."""
        super().__post_init__()
        if self.compression not in (None, "gzip", "lzf"):
            raise ValueError(f"Unsupported compression: {self.compression}")
        if self.compression == "gzip" and not 1 <= self.compression_opts <= 9:
            raise ValueError(f"gzip compression_opts must be 1-9, got {self.compression_opts}")


class HDF5Writer(DatasetWriter):
    """HDF5 dataset writer following ManiSkill trajectory conventions.

    Stores trajectories in a format compatible with ManiSkill3's trajectory
    replay functionality, with additional metadata in a JSON sidecar file.

    File structure:
        dataset.h5
        ├── traj_0/
        │   ├── actions: [T, action_dim]
        │   ├── terminated: [T]
        │   ├── truncated: [T]
        │   ├── success: [T]
        │   ├── env_states: [T+1, state_dim] (optional)
        │   └── obs/
        │       ├── {sensor_name}: [T+1, ...]
        │       └── ...
        └── traj_1/ ...

        dataset.json
        {
            "num_episodes": int,
            "episodes": [...],
            "sensor_configs": {...},
            "recording_info": {...}
        }
    """

    def __init__(self, config: HDF5WriterConfig, sensor_configs: Dict[str, SensorConfig]):
        """Initialize the HDF5 writer.

        Args:
            config: Writer configuration.
            sensor_configs: Configuration for each sensor being recorded.
        """
        super().__init__(config, sensor_configs)
        self._config: HDF5WriterConfig = config
        self._h5_file: Optional[h5py.File] = None
        self._h5_path: Optional[Path] = None
        self._json_path: Optional[Path] = None

        # Current episode state
        self._current_traj_group: Optional[h5py.Group] = None
        self._current_episode_id: Optional[str] = None
        self._current_task_description: str = ""
        self._episode_start_time_ns: int = 0
        self._frame_count: int = 0

        # Buffers for current episode (stored in memory, flushed on end_episode)
        self._obs_buffers: Dict[str, List[np.ndarray]] = {}
        self._action_buffer: List[np.ndarray] = []
        self._terminated_buffer: List[bool] = []
        self._truncated_buffer: List[bool] = []
        self._success_buffer: List[bool] = []
        self._env_state_buffer: List[Dict[str, Any]] = []

        # Metadata tracking
        self._episodes_metadata: List[Dict[str, Any]] = []

    def open(self) -> None:
        """Open the HDF5 file and prepare for writing."""
        if self._is_open:
            return

        # Create output directory
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

        # Set file paths
        self._h5_path = self._config.output_dir / "dataset.h5"
        self._json_path = self._config.output_dir / "dataset.json"

        # Handle existing file
        if self._h5_path.exists():
            if self._config.overwrite:
                self._h5_path.unlink()
                if self._json_path.exists():
                    self._json_path.unlink()
            else:
                raise FileExistsError(
                    f"Dataset already exists at {self._h5_path}. "
                    "Set overwrite=True to replace."
                )

        # Open HDF5 file
        self._h5_file = h5py.File(self._h5_path, "w")

        # Write file-level attributes
        self._h5_file.attrs["format"] = "spatial_ai_hdf5_v1"
        self._h5_file.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        self._is_open = True

    def close(self) -> None:
        """Close the HDF5 file."""
        if not self._is_open:
            return

        # End any in-progress episode
        if self._current_traj_group is not None:
            self.end_episode(success=False, notes="Closed without explicit end")

        # Close HDF5 file
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

        self._is_open = False

    def start_episode(self, task_description: str) -> str:
        """Start recording a new episode.

        Args:
            task_description: Human-readable description of the task.

        Returns:
            Episode ID (format: "traj_{index}").
        """
        if not self._is_open:
            raise RuntimeError("Writer is not open")
        if self._current_traj_group is not None:
            raise RuntimeError("Previous episode not ended. Call end_episode() first.")

        # Generate episode ID
        episode_id = f"traj_{self._current_episode_index}"
        self._current_episode_id = episode_id
        self._current_task_description = task_description

        # Create trajectory group
        self._current_traj_group = self._h5_file.create_group(episode_id)
        self._current_traj_group.create_group("obs")

        # Reset buffers
        self._obs_buffers = {name: [] for name in self._sensor_configs}
        self._action_buffer = []
        self._terminated_buffer = []
        self._truncated_buffer = []
        self._success_buffer = []
        self._env_state_buffer = []

        # Reset counters
        self._frame_count = 0
        self._episode_start_time_ns = time.perf_counter_ns()

        return episode_id

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
            info: Optional additional information (terminated, truncated, success, env_state).
        """
        if self._current_traj_group is None:
            raise RuntimeError("No episode in progress. Call start_episode() first.")

        info = info or {}

        # Buffer observations
        for sensor_name, obs in observations.items():
            if sensor_name in self._obs_buffers:
                self._obs_buffers[sensor_name].append(obs.data)
            else:
                # New sensor not in config - add it
                self._obs_buffers[sensor_name] = [obs.data]

        # Buffer action (if provided)
        if action is not None:
            self._action_buffer.append(action.values)

        # Buffer info flags
        self._terminated_buffer.append(info.get("terminated", False))
        self._truncated_buffer.append(info.get("truncated", False))
        self._success_buffer.append(info.get("success", False))

        # Buffer environment state if requested
        if self._config.include_env_states and "env_state" in info:
            self._env_state_buffer.append(info["env_state"])

        self._frame_count += 1

    def end_episode(self, success: bool, notes: str = "") -> EpisodeMetadata:
        """End the current episode and flush data to disk.

        Args:
            success: Whether the episode was successful.
            notes: Optional notes about the episode.

        Returns:
            Metadata for the completed episode.
        """
        if self._current_traj_group is None:
            raise RuntimeError("No episode in progress.")

        end_time_ns = time.perf_counter_ns()

        # Get compression kwargs
        compression_kwargs = {}
        if self._config.compression:
            compression_kwargs["compression"] = self._config.compression
            if self._config.compression == "gzip":
                compression_kwargs["compression_opts"] = self._config.compression_opts

        # Write observations to HDF5
        obs_group = self._current_traj_group["obs"]
        for sensor_name, data_list in self._obs_buffers.items():
            if data_list:
                data_array = np.stack(data_list, axis=0)
                obs_group.create_dataset(
                    sensor_name,
                    data=data_array,
                    **compression_kwargs,
                )

        # Write actions
        if self._action_buffer:
            action_array = np.stack(self._action_buffer, axis=0)
            self._current_traj_group.create_dataset(
                "actions",
                data=action_array,
                **compression_kwargs,
            )

        # Write info flags
        if self._terminated_buffer:
            self._current_traj_group.create_dataset(
                "terminated",
                data=np.array(self._terminated_buffer, dtype=bool),
            )
        if self._truncated_buffer:
            self._current_traj_group.create_dataset(
                "truncated",
                data=np.array(self._truncated_buffer, dtype=bool),
            )
        if self._success_buffer:
            # Override with final success flag
            success_array = np.array(self._success_buffer, dtype=bool)
            success_array[-1] = success  # Final frame reflects episode success
            self._current_traj_group.create_dataset("success", data=success_array)

        # Write trajectory attributes
        self._current_traj_group.attrs["episode_id"] = self._current_episode_id
        self._current_traj_group.attrs["task_description"] = self._current_task_description
        self._current_traj_group.attrs["num_frames"] = self._frame_count
        self._current_traj_group.attrs["success"] = success
        self._current_traj_group.attrs["notes"] = notes

        # Create metadata
        metadata = EpisodeMetadata(
            episode_id=self._current_episode_id,
            task_description=self._current_task_description,
            start_timestamp_ns=self._episode_start_time_ns,
            end_timestamp_ns=end_time_ns,
            success=success,
            simulator="unknown",  # Will be set by collector
            num_frames=self._frame_count,
            notes=notes,
        )
        self._episodes_metadata.append(metadata.to_dict())

        # Flush to disk
        self._h5_file.flush()

        # Reset state
        self._current_traj_group = None
        self._current_episode_id = None
        self._current_episode_index += 1

        return metadata

    def finalize(self) -> None:
        """Finalize the dataset and write metadata JSON."""
        if not self._is_open:
            return

        # Update HDF5 file attributes
        self._h5_file.attrs["num_episodes"] = self._current_episode_index

        # Compute statistics
        stats = self._compute_stats()

        # Write JSON metadata
        metadata = {
            "format": "spatial_ai_hdf5_v1",
            "num_episodes": self._current_episode_index,
            "created_at": self._h5_file.attrs.get("created_at", ""),
            "episodes": self._episodes_metadata,
            "sensor_configs": {
                name: {
                    "sensor_type": cfg.sensor_type.name,
                    "frequency_hz": cfg.frequency_hz,
                    "shape": list(cfg.shape),
                    "dtype": str(cfg.dtype),
                }
                for name, cfg in self._sensor_configs.items()
            },
            "stats": stats.to_dict(),
        }

        with open(self._json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _compute_stats(self) -> DatasetStats:
        """Compute dataset statistics from recorded data."""
        if self._current_episode_index == 0:
            return DatasetStats()

        total_frames = 0
        total_duration = 0.0
        successful_episodes = 0

        for ep_meta in self._episodes_metadata:
            total_frames += ep_meta.get("num_frames", 0)
            duration = (ep_meta["end_timestamp_ns"] - ep_meta["start_timestamp_ns"]) / 1e9
            total_duration += duration
            if ep_meta.get("success", False):
                successful_episodes += 1

        success_rate = successful_episodes / self._current_episode_index if self._current_episode_index > 0 else 0.0

        return DatasetStats(
            num_episodes=self._current_episode_index,
            total_frames=total_frames,
            total_duration_s=total_duration,
            success_rate=success_rate,
        )


# Register with factory
DatasetWriterFactory.register("hdf5", HDF5Writer)
