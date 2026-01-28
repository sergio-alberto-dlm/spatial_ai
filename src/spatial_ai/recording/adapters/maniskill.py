"""ManiSkill3 simulator adapter.

This adapter provides a unified interface to ManiSkill3 robotics environments
for data collection and teleoperation.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from spatial_ai.recording.types import (
    Action,
    ControlMode,
    Observation,
    SensorConfig,
    SensorType,
)
from spatial_ai.recording.adapters.base import (
    SimulatorAdapter,
    SimulatorAdapterFactory,
)

# ManiSkill3 imports - optional dependency
try:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers environments
    import torch

    HAS_MANISKILL = True
except ImportError:
    HAS_MANISKILL = False
    gym = None
    torch = None


@dataclass
class ManiSkillConfig:
    """Configuration for ManiSkill3 adapter.

    Attributes:
        env_id: Environment ID (e.g., "PickCubeSO100-v1").
        obs_mode: Observation mode ("rgb", "depth", "rgb+depth", etc.).
        control_mode: Robot control mode ("pd_joint_pos", "pd_ee_pose", etc.).
        render_mode: Rendering mode ("rgb_array", "human", None).
        sim_backend: Simulation backend ("cpu", "gpu", "auto").
        num_envs: Number of parallel environments (use 1 for teleoperation).
        camera_width: Camera image width.
        camera_height: Camera image height.
        camera_names: List of camera names to capture.
        seed: Random seed for environment.
        env_kwargs: Additional environment keyword arguments.
    """

    env_id: str
    obs_mode: str = "rgb+depth"
    control_mode: str = "pd_joint_pos"
    render_mode: Optional[str] = "rgb_array"
    sim_backend: str = "auto"
    num_envs: int = 1
    camera_width: int = 640
    camera_height: int = 480
    camera_names: List[str] = field(default_factory=lambda: ["base_camera"])
    seed: Optional[int] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


class ManiSkillAdapter(SimulatorAdapter):
    """Adapter for ManiSkill3 robotics simulator.

    Provides unified interface for data collection from ManiSkill3 environments,
    handling observation extraction and GPU tensor conversion.
    """

    def __init__(self, config: ManiSkillConfig):
        """Initialize the ManiSkill adapter.

        Args:
            config: Adapter configuration.

        Raises:
            ImportError: If ManiSkill3 is not installed.
        """
        if not HAS_MANISKILL:
            raise ImportError(
                "ManiSkill3 is not installed. Install with: pip install mani_skill"
            )

        self._config = config
        self._env: Optional[gym.Env] = None
        self._sensor_configs: Dict[str, SensorConfig] = {}
        self._last_obs: Dict[str, Any] = {}
        self._last_info: Dict[str, Any] = {}
        self._episode_count: int = 0
        self._step_count: int = 0

        # Initialize environment
        self._init_environment()

    def _init_environment(self) -> None:
        """Initialize the ManiSkill3 environment."""
        env_kwargs = {
            "obs_mode": self._config.obs_mode,
            "control_mode": self._config.control_mode,
            "render_mode": self._config.render_mode,
            "num_envs": self._config.num_envs,
            "sim_backend": self._config.sim_backend,
            **self._config.env_kwargs,
        }

        # Set camera resolution if environment supports it
        if "sensor_configs" not in env_kwargs:
            sensor_configs = {}
            for cam_name in self._config.camera_names:
                sensor_configs[cam_name] = {
                    "width": self._config.camera_width,
                    "height": self._config.camera_height,
                }
            if sensor_configs:
                env_kwargs["sensor_configs"] = sensor_configs

        self._env = gym.make(self._config.env_id, **env_kwargs)

        # Build sensor configurations
        self._sensor_configs = self._build_sensor_configs()

    def _build_sensor_configs(self) -> Dict[str, SensorConfig]:
        """Build sensor configurations from environment observation space."""
        configs = {}
        obs_space = self._env.observation_space

        # Get a sample observation to determine actual shapes
        sample_obs, _ = self._env.reset(seed=self._config.seed)

        # Extract camera observations
        if "sensor_data" in sample_obs:
            for cam_name in sample_obs["sensor_data"]:
                cam_data = sample_obs["sensor_data"][cam_name]

                if "rgb" in cam_data:
                    rgb = self._tensor_to_numpy(cam_data["rgb"])
                    configs[f"{cam_name}_rgb"] = SensorConfig(
                        name=f"{cam_name}_rgb",
                        sensor_type=SensorType.RGB,
                        frequency_hz=30.0,  # Depends on sim FPS
                        shape=rgb.shape,
                        dtype=np.dtype(np.uint8),
                        metadata={"camera": cam_name},
                    )

                if "depth" in cam_data:
                    depth = self._tensor_to_numpy(cam_data["depth"])
                    configs[f"{cam_name}_depth"] = SensorConfig(
                        name=f"{cam_name}_depth",
                        sensor_type=SensorType.DEPTH,
                        frequency_hz=30.0,
                        shape=depth.shape,
                        dtype=np.dtype(np.float32),
                        metadata={"camera": cam_name, "unit": "meters"},
                    )

                if "segmentation" in cam_data:
                    seg = self._tensor_to_numpy(cam_data["segmentation"])
                    configs[f"{cam_name}_segmentation"] = SensorConfig(
                        name=f"{cam_name}_segmentation",
                        sensor_type=SensorType.SEGMENTATION,
                        frequency_hz=30.0,
                        shape=seg.shape,
                        dtype=np.dtype(np.int32),
                        metadata={"camera": cam_name},
                    )

        # Extract agent/proprioceptive observations
        if "agent" in sample_obs:
            agent = sample_obs["agent"]

            if "qpos" in agent:
                qpos = self._tensor_to_numpy(agent["qpos"])
                configs["joint_state"] = SensorConfig(
                    name="joint_state",
                    sensor_type=SensorType.JOINT_STATE,
                    frequency_hz=100.0,  # Higher frequency for state
                    shape=qpos.shape,
                    dtype=np.dtype(np.float32),
                    metadata={"dof": qpos.shape[-1]},
                )

            if "qvel" in agent:
                qvel = self._tensor_to_numpy(agent["qvel"])
                configs["joint_velocity"] = SensorConfig(
                    name="joint_velocity",
                    sensor_type=SensorType.JOINT_VELOCITY,
                    frequency_hz=100.0,
                    shape=qvel.shape,
                    dtype=np.dtype(np.float32),
                )

        # Extract extra observations (end-effector pose, etc.)
        if "extra" in sample_obs:
            extra = sample_obs["extra"]

            if "tcp_pose" in extra:
                tcp = self._tensor_to_numpy(extra["tcp_pose"])
                configs["ee_pose"] = SensorConfig(
                    name="ee_pose",
                    sensor_type=SensorType.END_EFFECTOR_POSE,
                    frequency_hz=100.0,
                    shape=tcp.shape,
                    dtype=np.dtype(np.float32),
                    metadata={"format": "pos_quat_wxyz"},
                )

        return configs

    def _tensor_to_numpy(self, data: Any) -> np.ndarray:
        """Convert PyTorch tensor (GPU or CPU) to numpy array.

        Handles batch dimension removal for num_envs=1.

        Args:
            data: Input data (tensor or array).

        Returns:
            Numpy array with batch dimension removed if present.
        """
        if data is None:
            return np.array([])

        if hasattr(data, "cpu"):  # PyTorch tensor
            arr = data.cpu().numpy()
        else:
            arr = np.asarray(data)

        # Remove batch dimension if num_envs=1
        if self._config.num_envs == 1 and arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        return arr

    def _extract_observations(
        self, raw_obs: Dict[str, Any], timestamp_ns: int
    ) -> Dict[str, Observation]:
        """Convert ManiSkill3 observation dict to Observation format.

        Args:
            raw_obs: Raw observation dictionary from environment.
            timestamp_ns: Timestamp in nanoseconds.

        Returns:
            Dictionary mapping sensor names to Observation objects.
        """
        observations = {}

        # Extract camera observations
        if "sensor_data" in raw_obs:
            for cam_name in raw_obs["sensor_data"]:
                cam_data = raw_obs["sensor_data"][cam_name]

                if "rgb" in cam_data:
                    rgb = self._tensor_to_numpy(cam_data["rgb"])
                    # Convert to uint8 if needed
                    if rgb.dtype != np.uint8:
                        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
                    observations[f"{cam_name}_rgb"] = Observation(
                        sensor_name=f"{cam_name}_rgb",
                        timestamp_ns=timestamp_ns,
                        data=rgb,
                        metadata={"camera": cam_name},
                    )

                if "depth" in cam_data:
                    depth = self._tensor_to_numpy(cam_data["depth"])
                    observations[f"{cam_name}_depth"] = Observation(
                        sensor_name=f"{cam_name}_depth",
                        timestamp_ns=timestamp_ns,
                        data=depth.astype(np.float32),
                        metadata={"camera": cam_name, "unit": "meters"},
                    )

                if "segmentation" in cam_data:
                    seg = self._tensor_to_numpy(cam_data["segmentation"])
                    observations[f"{cam_name}_segmentation"] = Observation(
                        sensor_name=f"{cam_name}_segmentation",
                        timestamp_ns=timestamp_ns,
                        data=seg.astype(np.int32),
                        metadata={"camera": cam_name},
                    )

        # Extract agent/proprioceptive observations
        if "agent" in raw_obs:
            agent = raw_obs["agent"]

            if "qpos" in agent:
                qpos = self._tensor_to_numpy(agent["qpos"])
                observations["joint_state"] = Observation(
                    sensor_name="joint_state",
                    timestamp_ns=timestamp_ns,
                    data=qpos.astype(np.float32),
                    metadata={"dof": qpos.shape[-1] if qpos.ndim > 0 else 0},
                )

            if "qvel" in agent:
                qvel = self._tensor_to_numpy(agent["qvel"])
                observations["joint_velocity"] = Observation(
                    sensor_name="joint_velocity",
                    timestamp_ns=timestamp_ns,
                    data=qvel.astype(np.float32),
                )

        # Extract extra observations
        if "extra" in raw_obs:
            extra = raw_obs["extra"]

            if "tcp_pose" in extra:
                tcp = self._tensor_to_numpy(extra["tcp_pose"])
                observations["ee_pose"] = Observation(
                    sensor_name="ee_pose",
                    timestamp_ns=timestamp_ns,
                    data=tcp.astype(np.float32),
                    metadata={"format": "pos_quat_wxyz"},
                )

        return observations

    def get_sensor_configs(self) -> Dict[str, SensorConfig]:
        """Return configuration for all available sensors."""
        return self._sensor_configs.copy()

    def get_observations(self) -> Dict[str, Observation]:
        """Get current observations from all sensors."""
        timestamp_ns = time.perf_counter_ns()
        return self._extract_observations(self._last_obs, timestamp_ns)

    def step(self, action: Action) -> Tuple[Dict[str, Observation], Dict[str, Any]]:
        """Execute action and return new observations.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observations, info).
        """
        # Convert Action to numpy array for environment
        action_array = action.values

        # Add batch dimension if needed
        if self._config.num_envs == 1 and action_array.ndim == 1:
            action_array = action_array[np.newaxis, :]

        # Convert to tensor if environment expects it
        if torch is not None and hasattr(self._env, "device"):
            action_array = torch.from_numpy(action_array).to(self._env.device)

        # Step environment
        obs, reward, terminated, truncated, info = self._env.step(action_array)

        # Store for later access
        self._last_obs = obs
        self._last_info = info
        self._step_count += 1

        # Extract observations
        timestamp_ns = time.perf_counter_ns()
        observations = self._extract_observations(obs, timestamp_ns)

        # Build info dict
        step_info = {
            "reward": float(self._tensor_to_numpy(reward)) if hasattr(reward, "cpu") else float(reward),
            "terminated": bool(self._tensor_to_numpy(terminated)) if hasattr(terminated, "cpu") else bool(terminated),
            "truncated": bool(self._tensor_to_numpy(truncated)) if hasattr(truncated, "cpu") else bool(truncated),
            "success": bool(info.get("success", False)),
        }

        # Include environment state if available
        if hasattr(self._env, "get_state"):
            step_info["env_state"] = self.get_state()

        return observations, step_info

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Observation], Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (initial_observations, info).
        """
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs, info = self._env.reset(**reset_kwargs)

        self._last_obs = obs
        self._last_info = info
        self._step_count = 0
        self._episode_count += 1

        timestamp_ns = time.perf_counter_ns()
        observations = self._extract_observations(obs, timestamp_ns)

        reset_info = {
            "episode": self._episode_count,
        }
        reset_info.update(info)

        return observations, reset_info

    def get_state(self) -> Dict[str, Any]:
        """Get full simulator state for replay."""
        if hasattr(self._env.unwrapped, "get_state"):
            state = self._env.unwrapped.get_state()
            if hasattr(state, "cpu"):
                state = self._tensor_to_numpy(state)
            return {"sim_state": state}
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore simulator to a saved state."""
        if "sim_state" in state and hasattr(self._env.unwrapped, "set_state"):
            sim_state = state["sim_state"]
            if torch is not None:
                sim_state = torch.from_numpy(sim_state)
                if hasattr(self._env, "device"):
                    sim_state = sim_state.to(self._env.device)
            self._env.unwrapped.set_state(sim_state)

    @property
    def action_space(self) -> Dict[str, Any]:
        """Return action space specification."""
        space = self._env.action_space

        # Get bounds
        low = self._tensor_to_numpy(space.low) if hasattr(space, "low") else None
        high = self._tensor_to_numpy(space.high) if hasattr(space, "high") else None

        return {
            "shape": space.shape,
            "dtype": str(space.dtype),
            "low": low,
            "high": high,
            "control_mode": self._config.control_mode,
        }

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Return observation space specification."""
        return {
            sensor_name: {
                "shape": config.shape,
                "dtype": str(config.dtype),
                "sensor_type": config.sensor_type.name,
            }
            for sensor_name, config in self._sensor_configs.items()
        }

    def close(self) -> None:
        """Clean up simulator resources."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self, mode: str = "rgb_array") -> Optional[NDArray]:
        """Render the environment.

        Args:
            mode: Rendering mode.

        Returns:
            Rendered image if mode is "rgb_array".
        """
        if self._env is not None and hasattr(self._env, "render"):
            frame = self._env.render()
            if frame is not None:
                return self._tensor_to_numpy(frame)
        return None

    @property
    def simulator_name(self) -> str:
        """Return the simulator name."""
        return "maniskill3"


# Register with factory
if HAS_MANISKILL:
    SimulatorAdapterFactory.register("maniskill", ManiSkillAdapter)
