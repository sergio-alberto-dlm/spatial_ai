"""Data collection orchestrator.

This module provides the main DataCollector class that coordinates
simulator adapters, teleoperation interfaces, and dataset writers
for recording robot demonstrations.
"""

import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from spatial_ai.recording.types import Action, ControlMode, EpisodeMetadata
from spatial_ai.recording.adapters.base import SimulatorAdapter, TeleoperationInterface
from spatial_ai.recording.writers.base import DatasetStats, DatasetWriter


@dataclass
class CollectorConfig:
    """Configuration for the data collector.

    Attributes:
        fps: Target recording framerate.
        max_episode_steps: Maximum steps per episode.
        max_episodes: Maximum number of episodes to record.
        timeout_seconds: Episode timeout in seconds.
        auto_reset: Automatically reset after each episode.
        save_on_interrupt: Save partial data on keyboard interrupt.
        task_description: Default task description for episodes.
        seed: Base random seed (incremented per episode).
        verbose: Print status messages.
    """

    fps: float = 30.0
    max_episode_steps: int = 1000
    max_episodes: int = 100
    timeout_seconds: float = 300.0
    auto_reset: bool = True
    save_on_interrupt: bool = True
    task_description: str = "Robot manipulation task"
    seed: Optional[int] = None
    verbose: bool = True


class DataCollector:
    """Single-threaded data collection orchestrator.

    Coordinates simulator adapter, teleoperation interface, and dataset writer
    to record robot demonstrations. Handles episode management, frame rate
    control, and graceful shutdown.

    Example:
        ```python
        adapter = ManiSkillAdapter(config)
        teleop = KeyboardTeleop(teleop_config, action_dim=7)
        writer = HDF5Writer(writer_config, adapter.get_sensor_configs())

        collector = DataCollector(adapter, teleop, writer, CollectorConfig())
        stats = collector.run()
        ```
    """

    def __init__(
        self,
        adapter: SimulatorAdapter,
        teleop: TeleoperationInterface,
        writer: DatasetWriter,
        config: CollectorConfig,
    ):
        """Initialize the data collector.

        Args:
            adapter: Simulator adapter for environment interaction.
            teleop: Teleoperation interface for user input.
            writer: Dataset writer for data storage.
            config: Collector configuration.
        """
        self._adapter = adapter
        self._teleop = teleop
        self._writer = writer
        self._config = config

        # State
        self._running = False
        self._interrupted = False
        self._current_episode = 0
        self._current_step = 0
        self._episode_start_time = 0.0

        # Statistics
        self._total_frames = 0
        self._successful_episodes = 0

        # Setup signal handler for graceful shutdown
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal for graceful shutdown."""
        if self._config.verbose:
            print("\n[Collector] Interrupt received, finishing current episode...")
        self._interrupted = True

    def run(self) -> DatasetStats:
        """Run the collection loop.

        Collects episodes until max_episodes is reached, user quits,
        or interrupted.

        Returns:
            Statistics for the recorded dataset.
        """
        # Install signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

        self._running = True
        self._writer.open()

        try:
            while self._should_continue():
                if self._config.verbose:
                    print(f"\n[Episode {self._current_episode + 1}/{self._config.max_episodes}]")

                episode_metadata = self._run_episode()

                if episode_metadata.success:
                    self._successful_episodes += 1

                if self._config.verbose:
                    status = "SUCCESS" if episode_metadata.success else "FAILURE"
                    print(f"  {status} - {episode_metadata.num_frames} frames, "
                          f"{episode_metadata.duration_s:.1f}s")

                self._current_episode += 1

                # Check for quit or interrupt
                if self._interrupted or self._is_quit_requested():
                    break

        except Exception as e:
            if self._config.verbose:
                print(f"\n[Collector] Error: {e}")
            raise

        finally:
            # Finalize and close
            self._writer.finalize()
            self._writer.close()
            self._running = False

            # Restore original signal handler
            signal.signal(signal.SIGINT, self._original_sigint_handler)

        return self._compute_stats()

    def _should_continue(self) -> bool:
        """Check if collection should continue."""
        return (
            self._running
            and not self._interrupted
            and self._current_episode < self._config.max_episodes
            and not self._is_quit_requested()
        )

    def _is_quit_requested(self) -> bool:
        """Check if teleop requested quit."""
        if hasattr(self._teleop, "is_quit_requested"):
            return self._teleop.is_quit_requested()
        return False

    def _run_episode(self) -> EpisodeMetadata:
        """Run a single episode of data collection.

        Returns:
            Metadata for the completed episode.
        """
        # Get episode seed
        seed = None
        if self._config.seed is not None:
            seed = self._config.seed + self._current_episode

        # Reset environment and teleop
        obs, reset_info = self._adapter.reset(seed=seed)
        self._teleop.reset()

        # Start episode in writer
        episode_id = self._writer.start_episode(self._config.task_description)

        if self._config.verbose:
            print(f"  Recording: {episode_id}")
            if hasattr(self._teleop, "print_key_bindings"):
                self._teleop.print_key_bindings()

        # Timing
        frame_duration = 1.0 / self._config.fps
        self._episode_start_time = time.perf_counter()
        self._current_step = 0

        # Write initial observation (no action yet)
        self._writer.add_frame(
            observations=obs,
            action=None,
            info={"initial": True},
        )

        # Episode loop
        while self._current_step < self._config.max_episode_steps:
            frame_start = time.perf_counter()

            # Check timeout
            elapsed = frame_start - self._episode_start_time
            if elapsed > self._config.timeout_seconds:
                if self._config.verbose:
                    print("  Timeout reached")
                break

            # Update teleop
            self._teleop.update()

            # Check pause
            if self._teleop.is_recording_paused():
                self._sleep_until(frame_start + frame_duration)
                continue

            # Check episode complete
            if self._teleop.is_episode_complete():
                break

            # Get action
            action = self._teleop.get_action()
            if action is None:
                action = self._create_zero_action()

            # Step environment
            next_obs, step_info = self._adapter.step(action)

            # Write frame
            self._writer.add_frame(
                observations=next_obs,
                action=action,
                info=step_info,
            )

            # Check termination from environment
            if step_info.get("terminated", False) or step_info.get("truncated", False):
                if self._config.verbose:
                    if step_info.get("terminated"):
                        print("  Episode terminated")
                    else:
                        print("  Episode truncated")
                break

            # Check for interrupt
            if self._interrupted:
                break

            self._current_step += 1
            self._total_frames += 1

            # Frame rate control
            self._sleep_until(frame_start + frame_duration)

        # Determine success
        success = self._teleop.is_episode_success()
        if not success and "success" in step_info:
            success = step_info.get("success", False)

        # End episode
        notes = ""
        if self._interrupted:
            notes = "Interrupted by user"

        return self._writer.end_episode(success=success, notes=notes)

    def _create_zero_action(self) -> Action:
        """Create a zero action for the current control mode."""
        action_space = self._adapter.action_space
        shape = action_space.get("shape", (7,))

        return Action(
            timestamp_ns=time.perf_counter_ns(),
            values=np.zeros(shape, dtype=np.float32),
            control_mode=ControlMode(action_space.get("control_mode", "joint_position")),
        )

    def _sleep_until(self, target_time: float) -> None:
        """Sleep until the target time.

        Args:
            target_time: Target time in seconds (from perf_counter).
        """
        remaining = target_time - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    def _compute_stats(self) -> DatasetStats:
        """Compute final statistics."""
        total_duration = time.perf_counter() - self._episode_start_time if self._current_episode > 0 else 0.0

        success_rate = (
            self._successful_episodes / self._current_episode
            if self._current_episode > 0
            else 0.0
        )

        return DatasetStats(
            num_episodes=self._current_episode,
            total_frames=self._total_frames,
            total_duration_s=total_duration,
            success_rate=success_rate,
        )

    def stop(self) -> None:
        """Stop the collection loop."""
        self._running = False
        self._interrupted = True

    @property
    def is_running(self) -> bool:
        """Check if collector is currently running."""
        return self._running

    @property
    def current_episode(self) -> int:
        """Get current episode number."""
        return self._current_episode

    @property
    def current_step(self) -> int:
        """Get current step within episode."""
        return self._current_step
