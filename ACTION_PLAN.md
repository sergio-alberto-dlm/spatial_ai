# Spatial AI - Cable Manipulation Robotics Platform

## Project Overview

Build a robotics experimentation platform for the **AI for Industry Challenge** targeting cable management and insertion in electronics assembly. The platform will support three main approaches:

1. **Reinforcement Learning from scratch**
2. **World Models (V-JEPA AC) + Planning**
3. **Vision-Language-Action model (pi 0)**

### Platform Modules

| Module | Status | Description |
|--------|--------|-------------|
| **Data Recording** | In Progress | Sensor sync, validation, LeRobotDataset output |
| Data Visualization | Planned | Rerun-based analysis and exploration |
| Model Training | Planned | RL, World Models, VLA training pipelines |
| Deployment | Planned | Sim-to-real transfer, ROS integration |

---

# Module 1: Data Recording

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Simulators    │     │   Data Pipeline  │     │     Outputs      │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ ManiSkill3       │────>│ Adapter          │     │ LeRobotDataset   │
│ PhysTwin         │────>│ SyncBuffer       │────>│ v3.0 (Parquet +  │
│ (Real Robot)     │     │ Validator        │     │ MP4)             │
└──────────────────┘     │ Preprocessor     │     │                  │
                         │ Writer           │     │ Rerun Dashboard  │
                         └────────┬─────────┘     │ (gRPC stream)    │
                                  │               │                  │
                                  └──────────────>│ HuggingFace Hub  │
                                                  └──────────────────┘
```

## Simulators

### ManiSkill3
- GPU-parallelized robotics simulator (30,000+ FPS on 4090)
- Built on SAPIEN physics engine
- Teleoperation via click+drag interface
- Sensors: RGB, Depth, Segmentation, Touch, Joint states
- Native format: HDF5 trajectories

### PhysTwin
- Gaussian Splatting-based simulator
- Spring-mass physics for deformable objects (ideal for cables)
- Creates digital twins from real video footage
- Keyboard-based teleoperation
- Differentiable physics via Nvidia Warp

## Output Format: LeRobotDataset v3.0

Target format for HuggingFace/PyTorch compatibility:
- **Parquet**: Tabular data (joint states, actions, timestamps)
- **MP4**: Video streams (H.264 codec, YUV420p)
- **Metadata**: info.json, stats.json, episodes/

```
dataset/
├── meta/
│   ├── info.json              # Schema, FPS, features
│   ├── stats.json             # Normalization statistics
│   └── episodes/chunk-000/    # Episode boundaries
├── data/chunk-000/
│   └── file-000.parquet       # States, actions
└── videos/
    └── camera_rgb/chunk-000/
        └── file-000.mp4       # Video frames
```

## Visualization: Rerun Dashboard

Layout:
- **Left**: RGB + Depth camera views
- **Center**: 3D scene with cable visualization
- **Right**: Joint state time series + validation logs

Features:
- kHz time-series streaming
- Remote gRPC streaming to server
- Custom blueprints for cable manipulation

---

## Core Components

### 1. Data Types (`types.py`)

```python
class SensorType(Enum):
    RGB, DEPTH, SEGMENTATION, JOINT_STATE, JOINT_VELOCITY,
    END_EFFECTOR_POSE, FORCE_TORQUE, IMU, POINT_CLOUD, TACTILE,
    CABLE_KEYPOINTS  # Cable-specific deformable tracking

@dataclass
class Observation:
    sensor_name: str
    timestamp_ns: int  # Nanosecond precision
    data: NDArray
    metadata: Dict[str, Any]

@dataclass
class Action:
    timestamp_ns: int
    values: NDArray
    control_mode: str  # "joint_position", "ee_pose", etc.
```

### 2. Simulator Adapter Interface (`adapters/base.py`)

```python
class SimulatorAdapter(ABC):
    def get_sensor_configs(self) -> Dict[str, SensorConfig]
    def get_observations(self) -> Dict[str, Observation]
    def step(self, action: Action) -> Dict[str, Observation]
    def reset(self, seed: Optional[int]) -> Dict[str, Observation]
    def get_cable_state(self) -> Optional[NDArray]  # Deformable tracking
```

### 3. Synchronization Buffer (`sync.py`)

Handles multi-rate sensors:
- Images @ 30-60 Hz
- Joint states @ 100-1000 Hz
- IMU @ 500-1000 Hz

Interpolation methods:
- `nearest`: For images (no interpolation)
- `linear`: For continuous states
- `cubic`: For smooth trajectories

### 4. Validation Rules (`validation.py`)

| Rule | Purpose |
|------|---------|
| `CableInFrameRule` | Ensure cable visibility in camera |
| `ActionContinuityRule` | Detect discontinuous action jumps |
| `ImageQualityRule` | Check blur, saturation |
| `TemporalConsistencyRule` | Detect timestamp issues |
| `RangeValidationRule` | Values within expected bounds |

### 5. LeRobotDataset Writer (`writers/lerobot.py`)

- Parquet schema matching v3.0 spec
- MP4 video encoding (ffmpeg subprocess)
- Chunking: ~1000 episodes per chunk
- Statistics computation on finalize

---

## Directory Structure

```
spatial_ai/
├── pyproject.toml
├── ACTION_PLAN.md               # This file
├── configs/
│   └── recording/
│       ├── default.yaml
│       ├── maniskill_cable.yaml
│       └── phystwin_cable.yaml
├── src/spatial_ai/
│   ├── recording/
│   │   ├── types.py             # Core data types
│   │   ├── collector.py         # Main orchestrator
│   │   ├── sync.py              # Multi-rate sync
│   │   ├── validation.py        # Quality validation
│   │   ├── preprocessing.py     # Filter/normalize
│   │   ├── adapters/
│   │   │   ├── base.py          # Abstract interface
│   │   │   ├── maniskill.py     # ManiSkill3 adapter
│   │   │   └── phystwin.py      # PhysTwin adapter
│   │   ├── teleop/
│   │   │   ├── base.py          # Teleop interface
│   │   │   ├── keyboard.py      # Keyboard teleop
│   │   │   └── mouse_drag.py    # ManiSkill click+drag
│   │   ├── writers/
│   │   │   ├── base.py          # Writer interface
│   │   │   ├── lerobot.py       # LeRobotDataset v3.0
│   │   │   └── hdf5.py          # Native HDF5 backup
│   │   ├── converters/
│   │   │   ├── maniskill_to_lerobot.py
│   │   │   └── phystwin_to_lerobot.py
│   │   └── dashboard/
│   │       ├── rerun_dashboard.py
│   │       ├── blueprints.py
│   │       └── streaming.py
│   ├── tasks/cable/
│   │   ├── cable_insertion.py
│   │   └── cable_routing.py
│   └── utils/
│       ├── transforms.py
│       ├── video.py
│       └── io.py
├── scripts/
│   ├── record_maniskill.py
│   ├── record_phystwin.py
│   ├── convert_to_lerobot.py
│   ├── validate_dataset.py
│   └── launch_dashboard.py
└── tests/
    ├── test_sync.py
    ├── test_validation.py
    └── test_adapters/
```

---

## Implementation Phases

### Phase 1: Core Foundation
- [x] Define data types (`types.py`)
- [x] Create abstract interfaces (`adapters/base.py`, `writers/base.py`)
- [x] Implement ManiSkill3 adapter with basic observation extraction
- [x] Simple collector with single-threaded loop
- [x] HDF5 writer for raw data storage
- [x] Keyboard teleoperation interface (`teleop/keyboard.py`)
- [x] Recording script with Hydra config (`scripts/record_maniskill.py`)

### Phase 2: Synchronization & Validation
- [ ] Multi-rate sync buffer with interpolation
- [ ] Validation framework with cable-specific rules
- [ ] Real-time validation during collection
- [ ] Latency monitoring and warnings

### Phase 3: LeRobotDataset Writer
- [ ] Parquet schema matching v3.0 spec
- [ ] MP4 video encoding pipeline (ffmpeg)
- [ ] Chunking logic (~1000 episodes/chunk)
- [ ] Statistics computation on finalize
- [ ] ManiSkill HDF5 → LeRobot converter

### Phase 4: Rerun Dashboard
- [ ] Blueprint-based layout configuration
- [ ] Multi-modal logging (images, 3D, time series)
- [ ] Cable visualization as 3D line strips
- [ ] Validation error overlay
- [ ] Remote gRPC streaming

### Phase 5: PhysTwin Integration
- [ ] PhysTwin adapter (Gaussian splat rendering)
- [ ] Spring-mass state extraction for cables
- [ ] Keyboard teleoperation interface
- [ ] PhysTwin → LeRobot converter

### Phase 6: Polish & CLI
- [ ] Recording scripts with Hydra config
- [ ] Dataset validation CLI
- [ ] Integration tests
- [ ] Documentation

---

## Dependencies

```toml
[project]
name = "spatial_ai"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
    "pyarrow>=14.0",
    "h5py>=3.9",
    "rerun-sdk>=0.18",
    "lerobot>=0.4.0",
    "opencv-python>=4.8",
    "imageio[ffmpeg]>=2.31",
    "hydra-core>=1.3",
    "aiofiles>=23.0",
]

[project.optional-dependencies]
maniskill = ["mani_skill>=3.0", "gymnasium>=0.29"]
phystwin = ["warp-lang>=1.0"]
dev = ["pytest>=7.0", "black>=23.0", "ruff>=0.1", "mypy>=1.5"]
```

---

## Verification Strategy

### Unit Tests
- Sync buffer interpolation with synthetic multi-rate data
- Validation rules with edge cases
- LeRobot writer parquet/video correctness

### Integration Tests

1. **ManiSkill3 Recording**:
   ```bash
   python scripts/record_maniskill.py --env CableInsertion-v1 --episodes 5
   ```

2. **LeRobot Compatibility**:
   ```python
   from lerobot.datasets import LeRobotDataset
   dataset = LeRobotDataset("path/to/recorded")
   sample = dataset[0]  # Should load without errors
   ```

3. **Rerun Dashboard**:
   ```bash
   python scripts/launch_dashboard.py
   ```

### End-to-End Test
1. Record 10 episodes in ManiSkill3 via teleoperation
2. Validate dataset with CLI tool
3. Convert to LeRobotDataset format
4. Load with LeRobot library
5. Train a simple policy to verify data usability

---

## Key Design Decisions

1. **Abstract Adapter Pattern**: Enables adding new simulators without changing core logic
2. **LeRobotDataset as Target Format**: Native PyTorch/HuggingFace compatibility
3. **Rerun for Visualization**: Superior kHz time-series support, better than RViz for multimodal
4. **Parquet + MP4**: Efficient storage, v3.0 format separates tabular from video data
5. **Cable-Specific Tracking**: CABLE_KEYPOINTS sensor type for deformable object state
6. **No ROS for Now**: Direct simulator APIs for recording phase; ROS bridge deferred to deployment phase

---

## References

- [LeRobotDataset v3.0 Documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [ManiSkill3 Documentation](https://maniskill.readthedocs.io/en/latest/)
- [PhysTwin GitHub](https://github.com/Jianghanxiao/PhysTwin)
- [Rerun Documentation](https://rerun.io/docs)
- [AI for Industry Challenge](https://www.hgf-ai4industry.de/)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-24 | Initial plan created - Data Recording module design |
| 2026-01-26 | Phase 1 complete - HDF5Writer, ManiSkillAdapter, KeyboardTeleop, DataCollector, recording script |
