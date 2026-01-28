#!/usr/bin/env python3
"""Record demonstrations in ManiSkill3 using keyboard teleoperation.

This script provides a CLI for recording robot demonstrations in ManiSkill3
environments. Data is saved in HDF5 format compatible with ManiSkill3
trajectory replay, and can later be converted to LeRobotDataset format.

Usage:
    # Basic usage with defaults
    python -m spatial_ai.scripts.record_maniskill

    # Specify environment and number of episodes
    python -m spatial_ai.scripts.record_maniskill env.env_id=PickCube-v1 recording.max_episodes=5

    # Use installed entry point
    record-maniskill env.env_id=PushCube-v1
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="../../../configs/recording",
    config_name="maniskill_cable",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for ManiSkill recording.

    Args:
        cfg: Hydra configuration.
    """
    # Print configuration
    print("=" * 60)
    print("ManiSkill3 Data Recording")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Import here to allow --help without loading heavy dependencies
    from spatial_ai.recording.adapters.maniskill import ManiSkillAdapter, ManiSkillConfig
    from spatial_ai.recording.teleop.keyboard import KeyboardTeleop, KeyboardTeleopConfig
    from spatial_ai.recording.writers.hdf5 import HDF5Writer, HDF5WriterConfig
    from spatial_ai.recording.collector import DataCollector, CollectorConfig

    # Build adapter configuration
    env_cfg = cfg.get("env", {})
    adapter_config = ManiSkillConfig(
        env_id=env_cfg.get("env_id", "PickCube-v1"),
        obs_mode=env_cfg.get("obs_mode", "rgb+depth"),
        control_mode=env_cfg.get("control_mode", "pd_joint_pos"),
        render_mode=env_cfg.get("render_mode", "rgb_array"),
        sim_backend=env_cfg.get("sim_backend", "auto"),
        num_envs=env_cfg.get("num_envs", 1),
        camera_width=env_cfg.get("camera_width", 640),
        camera_height=env_cfg.get("camera_height", 480),
        camera_names=list(env_cfg.get("camera_names", ["base_camera"])),
    )

    print(f"\nInitializing ManiSkill3 environment: {adapter_config.env_id}")
    adapter = ManiSkillAdapter(adapter_config)

    # Get action space info
    action_space = adapter.action_space
    action_dim = action_space["shape"][0]
    print(f"Action space: dim={action_dim}, control_mode={adapter_config.control_mode}")

    # Build teleop configuration
    teleop_cfg = cfg.get("teleop", {})
    teleop_config = KeyboardTeleopConfig(
        joint_delta=teleop_cfg.get("joint_delta", 0.05),
        gripper_delta=teleop_cfg.get("gripper_delta", 0.1),
        action_repeat=teleop_cfg.get("action_repeat", 1),
    )

    # Determine gripper dimension (usually 1 or 2 for parallel gripper)
    # For most ManiSkill robots, gripper is included in action space
    gripper_dim = 1 if action_dim > 6 else 0
    arm_dim = action_dim - gripper_dim

    teleop = KeyboardTeleop(
        config=teleop_config,
        action_dim=arm_dim,
        gripper_dim=gripper_dim,
    )

    # Build writer configuration
    output_cfg = cfg.get("output", {})
    output_dir = Path(output_cfg.get("dir", "data/recordings/maniskill"))

    writer_config = HDF5WriterConfig(
        output_dir=output_dir,
        overwrite=output_cfg.get("overwrite", False),
        compression=output_cfg.get("compression", "gzip"),
        compression_opts=output_cfg.get("compression_opts", 4),
        include_env_states=output_cfg.get("include_env_states", True),
    )

    print(f"Output directory: {output_dir}")
    writer = HDF5Writer(writer_config, adapter.get_sensor_configs())

    # Build collector configuration
    rec_cfg = cfg.get("recording", {})
    collector_config = CollectorConfig(
        fps=rec_cfg.get("fps", 30),
        max_episode_steps=rec_cfg.get("max_episode_steps", 500),
        max_episodes=rec_cfg.get("max_episodes", 10),
        timeout_seconds=rec_cfg.get("timeout_seconds", 120),
        auto_reset=rec_cfg.get("auto_reset", True),
        save_on_interrupt=rec_cfg.get("save_on_interrupt", True),
        task_description=rec_cfg.get("task_description", f"ManiSkill: {adapter_config.env_id}"),
        verbose=rec_cfg.get("verbose", True),
    )

    # Create collector
    collector = DataCollector(adapter, teleop, writer, collector_config)

    # Run collection
    print("\nStarting data collection...")
    print("Press Escape or Ctrl+C to stop.\n")

    try:
        stats = collector.run()

        # Print final statistics
        print("\n" + "=" * 60)
        print("Recording Complete!")
        print("=" * 60)
        print(f"Episodes recorded: {stats.num_episodes}")
        print(f"Total frames: {stats.total_frames}")
        print(f"Total duration: {stats.total_duration_s:.1f}s")
        print(f"Success rate: {stats.success_rate * 100:.1f}%")
        print(f"Output: {output_dir}")
        print("=" * 60)

    finally:
        # Clean up
        teleop.close()
        adapter.close()


if __name__ == "__main__":
    main()
