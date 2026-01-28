# Development Setup

## Quick Start

### Linux

You can use either pip (with venv) or conda.

**Option A: pip + venv**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[recording,maniskill,dev]"
```

**Option B: conda**

```bash
conda create -n spatial_ai "python=3.11"
conda activate spatial_ai
pip install -e ".[recording,maniskill,dev]"
```

### macOS

Conda is recommended on macOS. ManiSkill supports macOS for CPU simulation and standard rendering (GPU simulation is not yet supported).

**Step 1: Create conda environment**

```bash
conda create -n spatial_ai "python=3.11"
conda activate spatial_ai
```

**Step 2: Install Vulkan SDK**

ManiSkill requires Vulkan on macOS via MoltenVK. Download the installer:

<https://sdk.lunarg.com/sdk/download/1.3.290.0/mac/vulkansdk-macos-1.3.290.0.dmg>

Open the `.dmg` and follow the installer. The default install path is fine — note the path (typically `~/VulkanSDK/1.3.290.0/macOS`). During component selection, you can skip "Development libraries" to save space.

**Step 3: Set Vulkan environment variables**

Add these to your `~/.zshrc` (or `~/.bashrc`):

```bash
export VULKAN_SDK=~/VulkanSDK/1.3.290.0/macOS
export PATH=$VULKAN_SDK/bin:$PATH
export VK_ICD_FILENAMES=$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json
export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH
```

Then reload your shell:

```bash
source ~/.zshrc
```

**Step 4: Install the project**

On macOS, use the nightly build of ManiSkill:

```bash
pip install -e ".[recording,maniskill-nightly,dev]"
```

**Step 5: Verify Vulkan + ManiSkill**

```bash
python -m sapien.example.hello_world
```

This should open a GUI window with a red cube on a plane. If it segfaults on the first run, simply run it again.

---

## Install Only the Recording Module

If you only need data collection capabilities (no visualization, no training):

```bash
pip install -e ".[recording]"

# Then add your simulator of choice:
pip install -e ".[maniskill]"        # Linux
pip install -e ".[maniskill-nightly]" # macOS
```

---

## Verify Installation

```bash
# Check package is installed
python -c "import spatial_ai; print(spatial_ai.__version__)"

# Run tests
pytest tests/
```

## Project Structure

```
spatial_ai/
├── src/spatial_ai/      # Main package
│   ├── recording/       # Data recording module
│   ├── tasks/           # Task definitions
│   └── utils/           # Utilities
├── configs/             # Hydra configuration files
├── scripts/             # CLI entry points
├── tests/               # Test suite
└── data/                # Data directory (gitignored)
```
