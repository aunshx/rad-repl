# RAD-SAC Implementation - EEC 256 Final Project

**Course:** EEC 256 Introduction to Reinforcement Learning  
**Author:** Aunsh Bandivadekar  
**Project:** Option 3 - Replicating "Reinforcement Learning with Augmented Data"

This project implements the RAD (Reinforcement Learning with Augmented Data) algorithm from Laskin et al. (NeurIPS 2020), demonstrating how simple data augmentations can dramatically improve sample efficiency in pixel-based reinforcement learning.

## Project Overview

RAD shows that applying simple data augmentations (like random crop and translate) directly to the RL objective can achieve similar performance to more complex methods like CURL, but with much simpler implementation. This project replicates key results from Table 1 of the original paper.

## Requirements

- Python 3.8+
- macOS (Apple Silicon or Intel) or Linux
- ~5GB free disk space for dependencies
- ~10+ hours for full experiments (or 3 minutes for quick test)

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

### Full Experiments (200+ hours)

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

- `train.log`: Training metrics (episode rewards, losses)
- `eval.log`: Evaluation metrics (performance during training)

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
| Full experiments | 200+ hours | Complete replication |
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

## Submission Checklist

- [ ] Code runs without errors (`python run.py --test` passes)
- [ ] Full experiments completed (`python run.py --full`)
- [ ] Results show RAD outperforming baseline
- [ ] All code files properly commented
- [ ] README explains setup and usage
- [ ] Setup script allows easy reproduction

---

**Note:** The automated setup script (`setup.py`) handles all the complex dependency installation and compatibility fixes automatically. If you encounter any issues, the script provides detailed error messages and fallback options.
