#!/usr/bin/env python3
"""
Complete RAD-SAC Setup Script
EEC 256 Final Project - Aunsh Bandivadekar

This script sets up everything from scratch:
1. Removes old environment
2. Creates fresh conda environment
3. Installs all packages
4. Fixes dmc2gym compatibility issues
5. Fixes train.py FrameStack wrapper
6. Tests everything works

Usage:
    python complete_setup.py
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, description):
    """Run a command with error handling."""
    print(f"Running {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed: {e.stderr.strip()}")
        return False

def check_command_exists(command):
    """Check if a command exists."""
    return subprocess.run(f"which {command}", shell=True, capture_output=True).returncode == 0

def install_system_dependencies():
    """Install system dependencies."""
    system = platform.system().lower()
    
    if system == "darwin":
        print("macOS detected - Installing system dependencies")
        
        if not check_command_exists("brew"):
            print("Installing Homebrew")
            install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            if not run_command(install_cmd, "Installing Homebrew"):
                print("WARNING: Homebrew installation failed. Please install manually from https://brew.sh")
                return False
        
        # Install OpenGL libraries
        for package in ["glew", "glfw"]:
            run_command(f"brew install {package}", f"Installing {package}")
        
        return True
    else:
        print(f"Platform: {system} - Skipping macOS-specific dependencies")
        return True

def setup_conda_environment():
    """Set up the complete conda environment."""
    print("\n" + '-' * 60)
    print("SETTING UP CONDA ENVIRONMENT")
    print('-' * 60)
    
    # Check if conda exists
    if not check_command_exists("conda"):
        print("ERROR: Conda not found. Please install Anaconda or Miniconda")
        return False
    
    # Remove existing environment
    print("Removing any existing environment")
    run_command("conda env remove -n eec256_rad -y", "Removing existing environment")
    
    # Create fresh environment
    if not run_command("conda create -n eec256_rad python=3.8 pip -y", "Creating fresh conda environment"):
        return False
    
    # Install packages
    print("Installing core packages")
    packages = [
        "torch torchvision numpy==1.23.5 matplotlib imageio imageio-ffmpeg scikit-image opencv-python termcolor",
        "dm-control==1.0.5 mujoco==2.3.6",
        "git+https://github.com/1nadequacy/dmc2gym.git"
    ]
    
    for package_group in packages:
        if not run_command(f"conda run -n eec256_rad pip install {package_group}", f"Installing {package_group}"):
            print(f"WARNING: Failed to install {package_group}")
    
    return True

def fix_dmc2gym():
    """Fix dmc2gym compatibility issues."""
    print("\n" + '-' * 60)
    print("FIXING DMC2GYM COMPATIBILITY")
    print('-' * 60)
    
    fix_script = '''
import site, os
for path in site.getsitepackages():
    f = os.path.join(path, "dmc2gym", "__init__.py")
    if os.path.exists(f):
        new_content = """from dmc2gym.wrappers import DMCWrapper

def make(domain_name, task_name, seed=0, visualize_reward=True, from_pixels=False, height=84, width=84, frame_skip=1, episode_length=1000, environment_kwargs=None, setting_kwargs=None, task_kwargs=None, camera_id=0, render_kwargs=None, channels_first=True):
    # Fix the task_kwargs to include required random seed
    if task_kwargs is None:
        task_kwargs = {}
    task_kwargs['random'] = seed
    
    # Create the wrapper directly
    return DMCWrapper(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs=task_kwargs,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=height,
        width=width,
        camera_id=camera_id,
        frame_skip=frame_skip,
        environment_kwargs=environment_kwargs,
        channels_first=channels_first
    )
"""
        with open(f, "w") as file:
            file.write(new_content)
        print("Fixed dmc2gym compatibility")
        break
'''
    
    # Write and execute the fix
    with open("temp_fix_dmc2gym.py", "w") as f:
        f.write(fix_script)
    
    try:
        success = run_command("conda run -n eec256_rad python temp_fix_dmc2gym.py", "Fixing dmc2gym")
        return success
    finally:
        if os.path.exists("temp_fix_dmc2gym.py"):
            os.remove("temp_fix_dmc2gym.py")

def fix_train_py():
    """Fix the FrameStack wrapper in train.py."""
    print("\n" + '-' * 60)
    print("FIXING TRAIN.PY")
    print('-' * 60)
    
    if not os.path.exists("train.py"):
        print("ERROR: train.py not found in current directory")
        return False
    
    try:
        # Read train.py
        with open("train.py", "r") as f:
            content = f.read()
        
        # Fix the FrameStack wrapper
        old_line = "self._max_episode_steps = env._max_episode_steps"
        new_line = 'self._max_episode_steps = getattr(env, "_max_episode_steps", 1000)'
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            
            with open("train.py", "w") as f:
                f.write(content)
            
            print("SUCCESS: Fixed FrameStack wrapper in train.py")
            return True
        else:
            print("INFO: FrameStack wrapper already fixed or not found")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to fix train.py: {e}")
        return False

def test_installation():
    """Test the complete installation."""
    print("\n" + '-' * 60)
    print("TESTING INSTALLATION")
    print('-' * 60)
    
    test_script = '''
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"SUCCESS: PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("SUCCESS: CUDA available")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("SUCCESS: MPS (Apple Silicon) available") 
    else:
        print("INFO: Using CPU")
    
    import dm_control
    print("SUCCESS: dm-control imported")
    
    import dmc2gym
    print("SUCCESS: dmc2gym imported")
    
    # Test environment creation
    env = dmc2gym.make("cartpole", "balance", seed=1, visualize_reward=False, from_pixels=True, height=84, width=84)
    obs = env.reset()
    env.close()
    print(f"SUCCESS: Environment test passed (shape: {obs.shape})")
    
    print("\\nALL TESTS PASSED!")
    print("You can now run: python run.py --test")
    
except Exception as e:
    print(f"ERROR: Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    with open("temp_test.py", "w") as f:
        f.write(test_script)
    
    try:
        success = run_command("conda run -n eec256_rad python temp_test.py", "Testing complete installation")
        return success
    finally:
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")

def main():
    """Main setup function."""
    print("RAD-SAC Complete Setup Script")
    print("EEC 256 Final Project")
    print("This will set up everything from scratch")
    print('-' * 60)
    
    success = True
    
    # Step 1: Install system dependencies
    if not install_system_dependencies():
        print("WARNING: System dependency installation had issues, but continuing")
    
    # Step 2: Set up conda environment
    if not setup_conda_environment():
        print("ERROR: Failed to set up conda environment")
        return False
    
    # Step 3: Fix dmc2gym
    if not fix_dmc2gym():
        print("ERROR: Failed to fix dmc2gym")
        return False
    
    # Step 4: Fix train.py
    if not fix_train_py():
        print("ERROR: Failed to fix train.py")
        return False
    
    # Step 5: Test everything
    if test_installation():
        print("\n" + '-' * 60)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print('-' * 60)
        print("Next steps:")
        print("1. conda activate eec256_rad")
        print("2. python run.py --test")
        print("3. python run.py --full")
        print("\nYour RAD-SAC environment is ready!")
        return True
    else:
        print("\n" + '-' * 60)
        print("SETUP FAILED")
        print('-' * 60)
        print("Check error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)