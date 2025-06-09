"""
Experiment Runner for RAD-SAC Implementation
"""

import subprocess
import time
import os
import sys

def run_experiment(env_name, domain_name, task_name, action_repeat, augmentation, seed, debug_mode=False):
    """
    Run a single RAD-SAC experiment with specified parameters.
    
    This function sets up the command line arguments and runs the training script
    for one environment/method/seed combination. It handles both quick tests and
    full experiments based on the debug_mode flag.
    
    Args:
        env_name (str): Human-readable environment name (e.g., 'walker_walk')
        domain_name (str): DMControl domain (e.g., 'walker')
        task_name (str): DMControl task (e.g., 'walk')
        action_repeat (int): Number of times to repeat each action (temporal abstraction)
        augmentation (str): Data augmentation type ('crop', 'translate', 'no_aug')
        seed (int): Random seed for reproducibility
        debug_mode (bool): If True, run shorter experiment for testing
        
    Returns:
        bool: True if experiment completed successfully, False otherwise
    """
    
    # Determine method name for results directory
    method = 'rad' if augmentation != 'no_aug' else 'pixel_sac'
    work_dir = f"./results/{env_name}_{method}_seed_{seed}"
    os.makedirs(work_dir, exist_ok=True)

    # Configure experiment parameters based on mode
    if debug_mode:
        # Debug mode: Fast parameters for testing
        num_train_steps = '2500'     # Only 2k steps (vs 100k for full)
        eval_freq = '500'            # Evaluate every 500 steps
        batch_size = '32'            # Smaller batch size
        num_eval_episodes = '5'      # Fewer evaluation episodes
        print(f"DEBUG MODE: {num_train_steps} steps only")
    else:
        # Full mode: Paper's parameters for replication
        num_train_steps = '100500'   # Full 100k steps as in paper
        eval_freq = '10000'          # Evaluate every 10k steps  
        batch_size = '512'           # Full batch size for stable training
        num_eval_episodes = '10'     # More evaluation episodes for better statistics

    # Environment-specific image size configuration
    # This follows the paper's experimental setup
    if augmentation == 'no_aug':
        # No augmentation: use standard 84x84 images
        image_size = '84'
    elif domain_name == 'walker' and task_name == 'walk':
        # Special case: Walker Walk uses crop instead of translate
        image_size = '84'
        augmentation = 'crop'  # Override to crop for Walker Walk
    else:
        # Other environments with augmentation: use larger 108x108 for translate
        image_size = '108'

    # Build the command to run train.py with all necessary arguments
    cmd = [
        'python', 'train.py',
        # Environment configuration
        '--domain_name', domain_name,
        '--task_name', task_name,
        '--action_repeat', str(action_repeat),
        
        # Basic setup
        '--work_dir', work_dir,
        '--seed', str(seed),
        '--encoder_type', 'pixel',  # Always use pixel observations
        
        # Image processing configuration
        '--pre_trans_img_size', '100',  # Size before augmentation
        '--image_size', image_size,     # Final size after augmentation
        '--frame_stack', '3',           # Stack 3 frames for temporal info
        
        # The key RAD parameter!
        '--data_augs', augmentation,
        
        # Training configuration
        '--num_train_steps', num_train_steps,
        '--eval_freq', eval_freq,
        '--num_eval_epis', num_eval_episodes,
        '--log_int', '100',            # Log every 100 steps
        '--batch_size', batch_size,
        '--agent', 'rad_sac',          # Use our RAD-SAC implementation
        
        # Network architecture (from paper)
        '--enc_feat_dim', '50',        # Encoder feature dimension
        '--num_layers', '4',           # 4 conv layers
        '--num_filters', '32',         # 32 filters per layer
        '--hidden_dim', '1024',        # Hidden layer size
        
        # SAC hyperparameters (from paper)
        '--disc', '0.99',              # Discount factor
        '--init_temp', '0.1',          # Initial temperature
        '--crt_lr', '1e-3',            # Critic learning rate
        '--act_lr', '1e-3',            # Actor learning rate
    ]

    print(f"Running: {method} on {domain_name}_{task_name} seed {seed}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        # Run the training script
        if debug_mode:
            # Debug mode: show output in real-time for easier debugging
            result = subprocess.run(cmd, text=True)
            success = result.returncode == 0
        else:
            # Full mode: capture output to avoid cluttering terminal
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
        duration = time.time() - start_time

        if success:
            print(f"Success: {method} {env_name} s{seed} completed in {duration/60:.1f} minutes")
            return True
        else:
            # Handle failure cases with detailed error reporting
            print(f"Failure: {method} {env_name} s{seed} failed:")
            print(f"Return code: {result.returncode}")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error (last 500 chars): {result.stderr[-500:]}")
            if hasattr(result, 'stdout') and result.stdout:
                print(f"Output (last 500 chars): {result.stdout[-500:]}")
            return False

    except Exception as e:
        print(f"Failure: {method} {env_name} s{seed} error: {str(e)}")
        return False

def test_dependencies():
    """
    Test if all required packages and files are available.
    
    This performs a comprehensive check of the environment before running
    experiments to catch issues early and provide helpful error messages.
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    print("Testing dependencies")
    
    # Test PyTorch installation
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        
        # Test CUDA availability if present
        if torch.cuda.is_available():
            print(f"CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS (Apple Silicon GPU) available")
        else:
            print("No GPU acceleration available (will use CPU - slower but works)")
    except ImportError:
        print("PyTorch not found. Try:")
        print("  conda activate eec256_rad")
        print("  conda install pytorch torchvision -c pytorch")
        return False
        
    # Test NumPy
    try:
        import numpy
        print(f"NumPy {numpy.__version__}")
    except ImportError:
        print("NumPy not found. Install with: conda install numpy=1.23.5")
        return False
        
    # Test DMControl Suite
    try:
        import dm_control
        print(f"dm-control found")
        
        # Test creating a simple environment to catch deeper issues
        try:
            import dmc2gym
            env = dmc2gym.make('cartpole', 'balance', seed=1, visualize_reward=False, from_pixels=True, height=84, width=84)
            obs = env.reset()
            env.close()
            print(f"DMControl environment creation successful (obs shape: {obs.shape})")
        except Exception as e:
            print(f"DMControl environment creation failed: {e}")
            print("This might cause issues but let's continue")
            
    except ImportError:
        print("dm-control not found. Try:")
        print("  conda activate eec256_rad") 
        print("  pip install dm-control==1.0.5")
        return False
        
    # Test dmc2gym
    try:
        import dmc2gym
        print("dmc2gym found")
    except ImportError:
        print("dmc2gym not found. Try:")
        print("  conda activate eec256_rad")
        print("  pip install git+https://github.com/1nadequacy/dmc2gym.git")
        return False
        
    # Test for required files
    if not os.path.exists('train.py'):
        print("train.py not found in current directory")
        return False
    else:
        print("train.py found")
        
    if not os.path.exists('rad_sac.py'):
        print("rad_sac.py not found in current directory")
        return False
    else:
        print("rad_sac.py found")
        
    # Test termcolor for pretty console output
    try:
        import termcolor
        print("termcolor found")
    except ImportError:
        print("termcolor not found (optional). Install with: pip install termcolor")
        
    return True

def quick_test():
    """
    Run a quick test to verify the implementation works.
    
    This runs a single short experiment (1000 steps, ~2-3 minutes) to test:
    - All dependencies are working
    - The code runs without errors
    - Results are saved properly
    
    This is much faster than running full experiments and catches most issues.
    
    Returns:
        bool: True if test passed, False if there were issues
    """
    print('-' * 50)
    print("QUICK TEST MODE")
    print("Running 1 experiment with 2000 steps (~2-3 minutes)")
    print("This tests that everything is working before running full experiments")
    print('-' * 50)
    
    # Check dependencies first
    if not test_dependencies():
        print("\nDependency test failed. Fix issues above.")
        return False
    
    print("\nStarting quick test experiment")
    
    # Run one short experiment to test the full pipeline
    success = run_experiment(
        env_name='walker_walk',      # Simple environment
        domain_name='walker', 
        task_name='walk',
        action_repeat=2,             # Standard action repeat for walker
        augmentation='crop',         # RAD with crop augmentation
        seed=1,                      # Fixed seed for reproducibility
        debug_mode=True              # Use fast parameters
    )
    
    if success:
        print("\n" + '-' * 50)
        print("QUICK TEST PASSED! The code is working.")
        print("You can now run full experiments with: python run.py --full")
        
        # Check if results were created and try to read them
        result_dir = "./results/walker_walk_rad_seed_1"
        if os.path.exists(f"{result_dir}/eval.log"):
            print(f"Results saved in {result_dir}")
            
            # Try to read and display the final evaluation score
            try:
                with open(f"{result_dir}/eval.log", 'r') as f:
                    lines = f.readlines()
                    if lines:
                        import json
                        last_eval = json.loads(lines[-1])
                        score = last_eval.get('episode_reward', 'N/A')
                        print(f"Final evaluation score: {score}")
                        
                        # Give context about what this score means
                        if isinstance(score, (int, float)) and score > 0:
                            print("This shows the agent learned something!")
                        else:
                            print("Score seems low, but this is just a quick test.")
            except Exception as e:
                print("Results saved but couldn't read final score (that's okay)")
        else:
            print("Results directory created but eval.log not found")
            
    else:
        print("\n" + '-' * 50)
        print("QUICK TEST FAILED")
        print("Check error messages above. Common issues:")
        print("- Missing dependencies (run pip install commands above)")
        print("- CUDA out of memory (try running on CPU)")
        print("- File permissions (check write access to current directory)")
        
    return success

def main():
    """
    Main function that handles different run modes.
    
    This function parses command line arguments and either:
    1. Runs a quick test to verify everything works
    2. Runs the full experimental suite to replicate paper results
    3. Shows help information
    """
    
    # Parse command line arguments to determine run mode
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # Quick test mode
            return quick_test()
        elif sys.argv[1] == '--help':
            # Show usage information
            print("RAD-SAC Experiment Runner")
            print('-' * 40)
            print("Usage:")
            print("python run.py --test     # Quick test (2000 steps, ~3 minutes)")
            print("python run.py --full     # Full experiments (100k steps, ~10+ hours)")
            print("python run.py            # Same as --full")
            print("\nThe quick test is recommended before running full experiments!")
            return
    
    # Default to full experiment mode
    print("FULL EXPERIMENT MODE")
    print("This will replicate the results from Table 1 of the RAD paper")
    
    # Test dependencies before starting long experiments
    if not test_dependencies():
        print("\nPlease fix dependency issues before running experiments")
        print("Try running 'python run.py --test' first to debug issues")
        return
    
    # Define experimental setup based on the RAD paper
    # We focus on one environment initially but can easily add more
    environments = [
        # Format: (env_name, domain_name, task_name, action_repeat)
        # Locomotion tasks - bipedal walking and quadruped running
        ('walker_walk', 'walker', 'walk', 2),           # Bipedal humanoid walking
        ('cheetah_run', 'cheetah', 'run', 4),           # Quadruped running task
        # Manipulation tasks - reaching and catching objects  
        ('reacher_easy', 'reacher', 'easy', 4),         # Simple reaching task
        # Control tasks - classic control problems
        ('cartpole_swingup', 'cartpole', 'swingup', 8), # Inverted pendulum
        ('finger_spin', 'finger', 'spin', 2),           # Spinning object with finger
    ]

    # Compare RAD (with augmentation) vs baseline (without augmentation)
    methods = [
        ('crop', 'RAD'),           # RAD with crop augmentation
        ('no_aug', 'Pixel SAC'),   # Baseline without augmentation
    ]

    # Multiple seeds for statistical significance
    # The paper typically uses 5 seeds for error bars
    seeds = [1, 2, 3, 4, 5]

    # Calculate total experiments
    total = len(environments) * len(methods) * len(seeds)
    current = 0
    successful = 0

    print(f"\nEXPERIMENTAL PLAN")
    print(f"Environments: {len(environments)}")
    print(f"Methods: {len(methods)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Total experiments: {total}")
    print(f"Estimated time: {total * 2} - {total * 3} hours")
    print("\nStarting experiments")
    print('-' * 60)

    # Run all experiment combinations
    for env_name, domain_name, task_name, action_repeat in environments:
        for aug, method_name in methods:
            for seed in seeds:
                current += 1
                print(f"\n[{current}/{total}] {method_name} on {env_name} seed {seed}")
                print("-" * 40)

                success = run_experiment(
                    env_name, domain_name, task_name,
                    action_repeat, aug, seed,
                    debug_mode=False  # Full experiments
                )

                if success:
                    successful += 1
                    print(f"Experiment {current} completed successfully")
                else:
                    print(f"Experiment {current} failed - continuing with next one")

    # Report final results
    print("\n" + '-' * 60)
    print(f"EXPERIMENTS COMPLETED!")
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")

    if successful >= total * 0.8:
        print("\nGreat! Most experiments completed successfully.")
        print("\nNext steps:")
        print("1. Analyze results: Look at the .log files in results directories")
        print("2. Compare performance: RAD should outperform Pixel SAC")
        print("3. Plot learning curves: Create graphs showing training progress")
        print("4. Calculate statistics: Compute mean ± std across seeds")
        
        # Show where results are saved
        print(f"\nResults saved in:")
        for env_name, _, _, _ in environments:
            print(f"  ./results/{env_name}_rad_seed_*/")
            print(f"  ./results/{env_name}_pixel_sac_seed_*/")
            
        print("\nTo analyze results:")
        print("1. Check eval.log files for final performance")
        print("2. Plot learning curves from the log files")
        print("3. Compare RAD vs Pixel SAC performance")
        print("4. Calculate mean ± standard deviation across seeds")
        
    else:
        print(f"\nOnly {successful}/{total} experiments succeeded")
        print("This might indicate issues with:")
        print("- Hardware (insufficient memory/compute)")
        print("- Dependencies (missing packages)")
        print("- Environment setup")
        print("\nCheck error messages above and consider:")
        print("- Running fewer seeds initially")
        print("- Using smaller batch sizes")
        print("- Running on a machine with more resources")

    # Provide practical next steps regardless of success rate
    print(f"\nANALYSIS TIPS:")
    print("Each experiment creates two log files:")
    print("- train.log: Training metrics (losses, episode rewards)")
    print("- eval.log: Evaluation metrics (performance during training)")
    print("\nExample analysis (you can implement this):")
    print("import json")
    print("import matplotlib.pyplot as plt")
    print("# Read eval.log and plot learning curves")
    print("# Compare final performance between RAD and Pixel SAC")

if __name__ == "__main__":
    main()