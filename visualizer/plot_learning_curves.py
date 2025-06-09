import matplotlib.pyplot as plt
import numpy as np
import json
import os

def read_eval_log(base_dir):
    """Read evaluation data from a single log file"""
    steps = []
    rewards = []
    
    try:
        if os.path.exists(base_dir):
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if subdirs:
                log_path = os.path.join(base_dir, subdirs[0], 'eval.log')
                with open(log_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        steps.append(data['step'])
                        rewards.append(data['mean_episode_reward'])
            else:
                print(f"Warning: No subdirectory found in {base_dir}")
        else:
            print(f"Warning: Directory {base_dir} not found")
    except Exception as e:
        print(f"Warning: Could not read from {base_dir}: {e}")
    
    return np.array(steps), np.array(rewards)

def read_all_seeds(env_name, method_name, num_seeds=5):
    """Read all seeds for a given environment and method"""
    all_steps = None
    all_rewards = []
    
    for seed in range(1, num_seeds + 1):
        base_dir = f"../results/{env_name}_{method_name}_seed_{seed}"
        steps, rewards = read_eval_log(base_dir)
        
        if len(steps) > 0:
            if all_steps is None:
                all_steps = steps
            all_rewards.append(rewards)
    
    if all_rewards:
        all_rewards = np.array(all_rewards)
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        return all_steps, mean_rewards, std_rewards
    else: return None, None, None

def plot_individual_learning_curves():
    """Create individual learning curve plots for each environment."""
    
    environments = [
        ('finger_spin', 'Finger Spin'),
        ('cartpole_swingup', 'Cartpole Swingup'),
        ('reacher_easy', 'Reacher Easy'),
        ('cheetah_run', 'Cheetah Run'),
        ('walker_walk', 'Walker Walk'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (env_name, env_title) in enumerate(environments):
        ax = axes[idx]
        steps, mean_rad, std_rad = read_all_seeds(env_name, 'rad')
        if steps is not None:
            ax.plot(steps/1000, mean_rad, color='#2ecc71', linewidth=2.5, label='RAD')
            ax.fill_between(steps/1000, mean_rad - std_rad, mean_rad + std_rad,alpha=0.3, color='#2ecc71')
        
        steps, mean_pixel, std_pixel = read_all_seeds(env_name, 'pixel_sac')
        if steps is not None:
            ax.plot(steps/1000, mean_pixel, color='#e74c3c', linewidth=2.5, label='Pixel SAC')
            ax.fill_between(steps/1000, mean_pixel - std_pixel, mean_pixel + std_pixel,alpha=0.3, color='#e74c3c')
        
        ax.set_title(env_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Environment Steps (×1000)', fontsize=12)
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        if idx == 0: ax.legend(loc='upper left', fontsize=12, frameon=True)
    
    fig.suptitle('RAD vs Pixel SAC Learning Curves - DMControl100k',fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('./plots/learning_curves.pdf', bbox_inches='tight')
    print("Saved learning curves to ./plots/learning_curves.png")
    
    plt.show()

def plot_combined_learning_curves():
    """Create a single plot with all environments' learning curves."""
    
    environments = [
        ('finger_spin', 'Finger Spin'),
        ('cartpole_swingup', 'Cartpole Swingup'),
        ('reacher_easy', 'Reacher Easy'),
        ('cheetah_run', 'Cheetah Run'),
        ('walker_walk', 'Walker Walk'),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    env_colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12', '#e67e22']
    
    for idx, ((env_name, env_title), color) in enumerate(zip(environments, env_colors)):
        steps, mean_rad, std_rad = read_all_seeds(env_name, 'rad')
        if steps is not None:
            ax.plot(steps/1000, mean_rad, color=color, linewidth=2, 
                   label=f'{env_title} (RAD)', linestyle='-')
        
        steps, mean_pixel, std_pixel = read_all_seeds(env_name, 'pixel_sac')
        if steps is not None:
            ax.plot(steps/1000, mean_pixel, color=color, linewidth=2,
                   label=f'{env_title} (Pixel SAC)', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Environment Steps (×1000)', fontsize=14)
    ax.set_ylabel('Episode Reward', fontsize=14)
    ax.set_title('Learning Curves: RAD (solid) vs Pixel SAC (dashed)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    plt.savefig('./plots/learning_curves_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig('./plots/learning_curves_combined.pdf', bbox_inches='tight')
    print("Saved combined learning curves to ./plots/learning_curves_combined.png")
    
    plt.show()

def plot_learning_curves_with_final_values():
    """Create learning curves with final performance annotations."""
    
    environments = [
        ('finger_spin', 'Finger Spin'),
        ('cartpole_swingup', 'Cartpole Swingup'),
        ('reacher_easy', 'Reacher Easy'),
        ('cheetah_run', 'Cheetah Run'),
        ('walker_walk', 'Walker Walk')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (env_name, env_title) in enumerate(environments):
        ax = axes[idx]
        steps, mean_rad, std_rad = read_all_seeds(env_name, 'rad')
        if steps is not None:
            ax.plot(steps/1000, mean_rad, color='#2ecc71', linewidth=2.5, label='RAD')
            ax.fill_between(steps/1000, mean_rad - std_rad, mean_rad + std_rad,
                           alpha=0.3, color='#2ecc71')
            
            final_rad = mean_rad[-1]
            ax.annotate(f'{final_rad:.0f}', 
                       xy=(100, final_rad), xytext=(90, final_rad),
                       fontsize=10, fontweight='bold', color='#2ecc71',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        steps, mean_pixel, std_pixel = read_all_seeds(env_name, 'pixel_sac')
        if steps is not None:
            ax.plot(steps/1000, mean_pixel, color='#e74c3c', linewidth=2.5, label='Pixel SAC')
            ax.fill_between(steps/1000, mean_pixel - std_pixel, mean_pixel + std_pixel,
                           alpha=0.3, color='#e74c3c')
            
            final_pixel = mean_pixel[-1]
            ax.annotate(f'{final_pixel:.0f}', 
                       xy=(100, final_pixel), xytext=(90, final_pixel),
                       fontsize=10, fontweight='bold', color='#e74c3c',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            if steps is not None and final_rad > 0:
                improvement = final_rad / final_pixel
                ax.text(0.05, 0.95, f'{improvement:.1f}x improvement', 
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       verticalalignment='top', color='darkgreen',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.set_title(env_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Environment Steps (×1000)', fontsize=12)
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        if idx == 0: ax.legend(loc='upper left', fontsize=12, frameon=True)
    
    fig.suptitle('RAD vs Pixel SAC Learning Curves with Final Performance',fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig('./plots/learning_curves_annotated.png', dpi=300, bbox_inches='tight')
    plt.savefig('./plots/learning_curves_annotated.pdf', bbox_inches='tight')
    print("Saved annotated learning curves to ./plots/learning_curves_annotated.png")
    
    plt.show()

if __name__ == "__main__":
    print("Creating learning curve visualizations")
    print("-" * 50)
    
    print("Creating individual learning curves")
    plot_individual_learning_curves()
    
    print("\nCreating combined learning curves")
    plot_combined_learning_curves()
    
    print("\nCreating annotated learning curves")
    plot_learning_curves_with_final_values()
    
    print("\nAll learning curves saved to ./plots/")