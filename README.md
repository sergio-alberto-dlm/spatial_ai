# Spatial AI

Robotics platform for cable manipulation with RL, World Models, and VLA approaches.

## Installation

```bash
pip install -e ".[dev]"

# With ManiSkill3 support
pip install -e ".[maniskill,dev]"
```

## Usage

### Record demonstrations

```bash
record-maniskill env.env_id=PickCube-v1 recording.max_episodes=5
```

See `ACTION_PLAN.md` for detailed documentation.
