"""Keyboard-based teleoperation interface.

This module provides keyboard control for robot teleoperation during
data collection. Uses pynput for cross-platform keyboard input.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from spatial_ai.recording.types import Action, ControlMode
from spatial_ai.recording.adapters.base import TeleoperationInterface

# pynput is an optional dependency
try:
    from pynput import keyboard

    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    keyboard = None


# Default key bindings for joint control
# Format: key -> (action_type, index_or_command, direction)
DEFAULT_KEY_BINDINGS: Dict[str, Tuple[str, Any, int]] = {
    # Joint control (QWERTY layout for 6-DOF)
    "q": ("joint", 0, +1),  # Joint 0 positive
    "a": ("joint", 0, -1),  # Joint 0 negative
    "w": ("joint", 1, +1),  # Joint 1 positive
    "s": ("joint", 1, -1),  # Joint 1 negative
    "e": ("joint", 2, +1),  # Joint 2 positive
    "d": ("joint", 2, -1),  # Joint 2 negative
    "r": ("joint", 3, +1),  # Joint 3 positive
    "f": ("joint", 3, -1),  # Joint 3 negative
    "t": ("joint", 4, +1),  # Joint 4 positive
    "g": ("joint", 4, -1),  # Joint 4 negative
    "y": ("joint", 5, +1),  # Joint 5 positive
    "h": ("joint", 5, -1),  # Joint 5 negative
    "u": ("joint", 6, +1),  # Joint 6 positive (if present)
    "j": ("joint", 6, -1),  # Joint 6 negative (if present)
}


@dataclass
class KeyboardTeleopConfig:
    """Configuration for keyboard teleoperation.

    Attributes:
        joint_delta: Position increment per keypress (radians).
        gripper_delta: Gripper change per keypress.
        action_repeat: Number of steps to repeat each action.
        key_bindings: Custom key mappings (overrides defaults).
        gripper_open_value: Value for open gripper.
        gripper_closed_value: Value for closed gripper.
    """

    joint_delta: float = 0.05
    gripper_delta: float = 0.1
    action_repeat: int = 1
    key_bindings: Dict[str, Tuple[str, Any, int]] = field(default_factory=dict)
    gripper_open_value: float = 0.04  # Common for Franka
    gripper_closed_value: float = 0.0


class KeyboardTeleop(TeleoperationInterface):
    """Keyboard-based teleoperation using pynput.

    Provides joint-space control via keyboard for recording robot demonstrations.
    Uses non-blocking keyboard listener for real-time input.

    Key bindings:
        - Q/A, W/S, E/D, R/F, T/G, Y/H, U/J: Joint 0-6 +/-
        - Space: Toggle gripper
        - Enter: Mark episode as success and complete
        - Backspace: Mark episode as failure and complete
        - P: Pause/resume recording
        - Escape: Quit recording
    """

    def __init__(
        self,
        config: KeyboardTeleopConfig,
        action_dim: int,
        gripper_dim: int = 1,
        on_quit: Optional[Callable[[], None]] = None,
    ):
        """Initialize keyboard teleoperation.

        Args:
            config: Teleoperation configuration.
            action_dim: Dimension of the action space (excluding gripper).
            gripper_dim: Dimension of gripper action (usually 1).
            on_quit: Optional callback when user presses Escape.

        Raises:
            ImportError: If pynput is not installed.
        """
        if not HAS_PYNPUT:
            raise ImportError(
                "pynput is not installed. Install with: pip install pynput"
            )

        self._config = config
        self._action_dim = action_dim
        self._gripper_dim = gripper_dim
        self._total_dim = action_dim + gripper_dim
        self._on_quit = on_quit

        # Merge default bindings with custom ones
        self._key_bindings = DEFAULT_KEY_BINDINGS.copy()
        self._key_bindings.update(config.key_bindings)

        # State
        self._current_action = np.zeros(action_dim, dtype=np.float32)
        self._gripper_state = config.gripper_open_value  # Start open
        self._key_states: Dict[str, bool] = {}  # Track which keys are pressed

        # Episode control flags
        self._episode_complete = False
        self._episode_success = False
        self._paused = False
        self._quit_requested = False

        # Keyboard listener
        self._listener: Optional[keyboard.Listener] = None
        self._setup_listener()

    def _setup_listener(self) -> None:
        """Set up the keyboard listener."""
        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()

    def _key_to_char(self, key) -> Optional[str]:
        """Convert pynput key to character string."""
        try:
            if hasattr(key, "char") and key.char is not None:
                return key.char.lower()
            elif hasattr(key, "name"):
                return key.name.lower()
        except AttributeError:
            pass
        return None

    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        char = self._key_to_char(key)
        if char is None:
            return

        self._key_states[char] = True

        # Handle special keys
        if char == "space":
            # Toggle gripper
            if self._gripper_state == self._config.gripper_open_value:
                self._gripper_state = self._config.gripper_closed_value
            else:
                self._gripper_state = self._config.gripper_open_value

        elif char == "enter" or char == "return":
            # Complete episode with success
            self._episode_complete = True
            self._episode_success = True

        elif char == "backspace":
            # Complete episode with failure
            self._episode_complete = True
            self._episode_success = False

        elif char == "p":
            # Toggle pause
            self._paused = not self._paused

        elif char == "escape" or char == "esc":
            # Quit
            self._quit_requested = True
            if self._on_quit:
                self._on_quit()

    def _on_key_release(self, key) -> None:
        """Handle key release events."""
        char = self._key_to_char(key)
        if char is not None:
            self._key_states[char] = False

    def update(self) -> None:
        """Update action based on currently pressed keys."""
        # Reset delta action
        delta_action = np.zeros(self._action_dim, dtype=np.float32)

        # Apply all pressed keys
        for char, is_pressed in self._key_states.items():
            if not is_pressed:
                continue

            if char in self._key_bindings:
                action_type, index, direction = self._key_bindings[char]

                if action_type == "joint" and index < self._action_dim:
                    delta_action[index] += direction * self._config.joint_delta

        # Accumulate into current action
        self._current_action += delta_action

    def get_action(self) -> Optional[Action]:
        """Get action from keyboard input.

        Returns:
            Action with current joint positions and gripper state,
            or None if paused.
        """
        if self._paused or self._quit_requested:
            return None

        # Combine arm action with gripper
        if self._gripper_dim > 0:
            full_action = np.concatenate([
                self._current_action,
                np.array([self._gripper_state] * self._gripper_dim, dtype=np.float32),
            ])
        else:
            full_action = self._current_action.copy()

        return Action(
            timestamp_ns=time.perf_counter_ns(),
            values=full_action,
            control_mode=ControlMode.JOINT_POSITION,
        )

    def is_episode_complete(self) -> bool:
        """Check if user signaled episode end."""
        return self._episode_complete or self._quit_requested

    def is_episode_success(self) -> bool:
        """Check if user marked episode as success."""
        return self._episode_success

    def is_recording_paused(self) -> bool:
        """Check if recording is paused."""
        return self._paused

    def is_quit_requested(self) -> bool:
        """Check if user requested to quit."""
        return self._quit_requested

    def reset(self) -> None:
        """Reset teleop state for new episode."""
        self._current_action = np.zeros(self._action_dim, dtype=np.float32)
        self._gripper_state = self._config.gripper_open_value
        self._key_states.clear()
        self._episode_complete = False
        self._episode_success = False
        # Note: Don't reset _paused or _quit_requested

    def get_gripper_action(self) -> Optional[float]:
        """Get gripper action separately."""
        return self._gripper_state

    @property
    def control_mode(self) -> ControlMode:
        """Return the control mode used by this teleop interface."""
        return ControlMode.JOINT_POSITION

    def close(self) -> None:
        """Stop the keyboard listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def __del__(self):
        """Clean up on deletion."""
        self.close()

    def print_key_bindings(self) -> None:
        """Print the key bindings to console."""
        print("\n=== Keyboard Teleoperation Controls ===")
        print("Joint Control (delta position):")
        print("  Q/A: Joint 0 +/-")
        print("  W/S: Joint 1 +/-")
        print("  E/D: Joint 2 +/-")
        print("  R/F: Joint 3 +/-")
        print("  T/G: Joint 4 +/-")
        print("  Y/H: Joint 5 +/-")
        print("  U/J: Joint 6 +/- (if present)")
        print("\nGripper:")
        print("  Space: Toggle gripper open/closed")
        print("\nEpisode Control:")
        print("  Enter: Complete episode (success)")
        print("  Backspace: Complete episode (failure)")
        print("  P: Pause/resume recording")
        print("  Escape: Quit recording")
        print("=" * 40 + "\n")
