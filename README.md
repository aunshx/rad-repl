# RAD-SAC Implementation - EEC 256 Final Project

**Course:** EEC 256 Introduction to Reinforcement Learning  
**Author:** Aunsh Bandivadekar  
**Project:** Option 3 - Replicating "Reinforcement Learning with Augmented Data"

This project implements the RAD (Reinforcement Learning with Augmented Data) algorithm from Laskin et al. (NeurIPS 2020), demonstrating how simple data augmentations can dramatically improve sample efficiency in pixel-based reinforcement learning.

## Table of Contents

- [Code Repository](#code-repository)
- [Project Overview](#project-overview)
- [Requirements](#requirements)
  - [Core Dependencies](#core-dependencies)
- [Quick Start](#quick-start)
  - [Automated Setup (Recommended)](#automated-setup-recommended)
  - [Manual Setup (Alternative)](#manual-setup-alternative)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Quick Test (3 minutes)](#quick-test-3-minutes)
  - [Full Experiments (180+ hours)](#full-experiments-200-hours)
  - [Single Experiment](#single-experiment)
- [Understanding Results](#understanding-results)
  - [Log Files](#log-files)
  - [Success Criteria](#success-criteria)
  - [Example Analysis](#example-analysis)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
- [Technical Details](#technical-details)
  - [Algorithm Implementation](#algorithm-implementation)
  - [Experimental Setup](#experimental-setup)
  - [Expected Timeline](#expected-timeline)
- [Learning Objectives](#learning-objectives)
- [Key Papers](#key-papers)
- [Getting Help](#getting-help)

## Code Repository 

The code repository for this project can be found on [here](https://github.com/aunshx/rad-repl.git) on Github.

## Project Overview

RAD shows that applying simple data augmentations (like random crop and translate) directly to the RL objective can achieve similar performance to more complex methods like CURL, but with much simpler implementation. This project replicates key results from Table 1 of the original paper.

## Requirements

- Python 3.8+
- macOS (Apple Silicon) or Linux
- ~10GB free disk space for dependencies
- ~180+ hours for full experiments (or 3 minutes for quick test)

### Core Dependencies

The automated setup script will install these packages:

```bash
# Core ML packages
torch torchvision         # PyTorch for neural networks
numpy==1.23.5            # Numerical computing (pinned for compatibility)

# RL Environment packages  
dm-control==1.0.5        # DeepMind Control Suite physics environments
mujoco==2.3.6           # MuJoCo physics engine (required by dm-control)
dmc2gym                 # Wrapper to make DMControl compatible with Gym

# Utilities and visualization
matplotlib              # Plotting results
imageio imageio-ffmpeg  # Video recording capabilities
scikit-image           # Image processing
opencv-python          # Computer vision utilities
termcolor              # Colored terminal output
```

## Quick Start

### Automated Setup (Recommended)

```bash
# Download/clone the project files
# Ensure you have: train.py, rad_sac.py, run.py, setup.py

# Run the complete setup script
python setup.py

# Activate environment and test
conda activate eec256_rad
python run.py --test
```

### Manual Setup (Alternative)

If the automated setup fails, you can set up manually:

```bash
# 1. Remove any existing environment
conda env remove -n eec256_rad -y

# 2. Create fresh environment
conda create -n eec256_rad python=3.8 pip -y

# 3. Activate environment
conda activate eec256_rad

# 4. Install packages
pip install torch torchvision numpy==1.23.5 matplotlib imageio imageio-ffmpeg scikit-image opencv-python termcolor
pip install dm-control==1.0.5 mujoco==2.3.6
pip install git+https://github.com/1nadequacy/dmc2gym.git

# 5. Fix compatibility issues
python setup.py

# 6. Test
python run.py --test
```

## Project Structure

```
project/
├── train.py           # Main training script with RAD implementation
├── rad_sac.py         # RAD-SAC agent with data augmentations  
├── run.py             # Experiment runner and dependency checker
├── setup.py  # Automated setup script
├── README.md          # This file
└── results/           # Generated during experiments
    ├── walker_walk_rad_seed_1/
    ├── walker_walk_rad_seed_2/
    ├── walker_walk_pixel_sac_seed_1/
    └── ...
```

## Usage

### Quick Test (3 minutes)

```bash
conda activate eec256_rad
python run.py --test
```

Runs a short experiment to verify everything works.

### Full Experiments (180+ hours)

```bash
conda activate eec256_rad
python run.py --full
```

Runs the complete experimental suite with multiple seeds.

### Single Experiment

```bash
conda activate eec256_rad
python train.py --domain_name walker --task_name walk --data_augs crop --seed 1 --num_train_steps 100000
```

## Understanding Results

### Log Files

Each experiment creates two log files:

**train.log** - Training metrics logged during training:

```json
{"duration": 11.41, "episode_reward": 40.35, "episode": 3.0, "batch_reward": 0.078, "critic_loss": 0.961, "actor_loss": -0.829, "step": 1500}
```

**eval.log**: Evaluation metrics logged every 500 steps

```json
{"episode": 0.0, "episode_reward": 24.217133273796414, "eval_time": 9.727205991744995, "mean_episode_reward": 24.217133273796414, "best_episode_reward": 44.39462270492695, "step": 0}
{"episode": 1.0, "episode_reward": 30.111354787240714, "eval_time": 9.82511281967163, "mean_episode_reward": 30.111354787240714, "best_episode_reward": 58.14674976984445, "step": 500}
```

- `episode`: Evaluation episode number
- `episode_reward`: Reward from the current evaluation episode
- `mean_episode_reward`: Average reward across all evaluation episodes at this step
- `best_episode_reward`: Highest reward achieved in any evaluation episode so far
- `step`: Training step when this evaluation was performed
- `eval_time`: Time taken to run the evaluation (seconds)
- `duration`: Time taken to complete the episode (seconds)
- `batch_reward`: Average reward in the current training batch
- `critic_loss`: Loss value for the critic network (lower = better learning)
- `actor_loss`: Loss value for the actor network (can be negative)

### Success Criteria

- RAD should significantly outperform Pixel SAC**

### Example Analysis

```python
import json
import matplotlib.pyplot as plt

# Read evaluation results
with open('results/walker_walk_rad_seed_1/**/eval.log', 'r') as f:
    rad_data = [json.loads(line) for line in f]

# Plot learning curve
rewards = [d['episode_reward'] for d in rad_data]
steps = [d['step'] for d in rad_data]
plt.plot(steps, rewards, label='RAD')
plt.xlabel('Training Steps')
plt.ylabel('Episode Reward')
plt.legend()
plt.show()
```

## Troubleshooting

### Common Issues

**1. Setup Script Fails**

```bash
# Try manual installation
conda env remove -n eec256_rad -y
conda create -n eec256_rad python=3.8 pip -y
conda activate eec256_rad
# Then follow manual setup steps above
```

**2. Environment Import Errors**

```bash
# Make sure you're in the right environment
conda activate eec256_rad
# Check installed packages
pip list | grep -E "(torch|dm-control|dmc2gym)"
```

**3. DMControl/MuJoCo Issues**

```bash
# Set environment variables (macOS)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

**4. Memory Issues**

```bash
# Reduce batch size in run.py
# Edit run.py and change batch_size from 512 to 128 or 64
```

**5. File Not Found Errors**

```bash
# Make sure all required files are present:
ls train.py rad_sac.py run.py setup.py
```

## Technical Details

### Algorithm Implementation

- **Base Algorithm:** SAC (Soft Actor-Critic) for continuous control
- **Key Innovation:** Simple data augmentations (crop, translate) applied during training
- **Network Architecture:** 4-layer CNN encoder + 3-layer MLP actor/critic
- **Augmentations:** Random crop and translation for visual invariance

### Experimental Setup

- **Environment:** DMControl Suite (Walker Walk task)
- **Training Steps:** 100,000 per experiment
- **Evaluation:** Every 10,000 steps with 10 episodes
- **Seeds:** 5 random seeds for statistical significance

### Expected Timeline

| Task | Time | Description |
|------|------|-------------|
| Setup | 5-10 min | Install dependencies |
| Quick test | 3 min | Verify everything works |
| Full experiments | 180+ hours | Complete replication |
| Analysis | 1-2 hours | Plot results, compute statistics |

## Learning Objectives

This project demonstrates:

1. **Data Augmentation in RL:** How simple augmentations improve sample efficiency
2. **SAC Algorithm:** Implementation of Soft Actor-Critic for continuous control
3. **Pixel-based RL:** Learning from raw images vs state vectors
4. **Experimental Methodology:** Proper evaluation with multiple seeds
5. **Reproducibility:** Clear documentation and dependency management

## Key Papers

- **Main Paper:** Laskin et al. "Reinforcement Learning with Augmented Data" (NeurIPS 2020)
- **SAC:** Haarnoja et al. "Soft Actor-Critic" (ICML 2018)
- **DMControl:** Tassa et al. "DeepMind Control Suite" (2018)

## Getting Help

1. **First:** Try `python run.py --test` to identify issues
2. **Check:** Error messages in terminal output
3. **Verify:** All files (train.py, rad_sac.py, run.py, setup.py) are present
4. **Confirm:** Conda environment is activated (`conda activate eec256_rad`)

---

**Note:** The automated setup script (`setup.py`) handles all the complex dependency installation and compatibility fixes automatically. If you encounter any issues, the script provides detailed error messages and fallback options.
