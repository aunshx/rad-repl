import matplotlib.pyplot as plt
import numpy as np
import json
import os

# CURL and State SAC values from the paper
CURL_VALUES = {
    'finger_spin': (767, 56),
    'cartpole_swingup': (582, 146),
    'reacher_easy': (538, 233),
    'cheetah_run': (299, 48),
    'walker_walk': (403, 24),
}

STATE_SAC_VALUES = {
    'finger_spin': (811, 46),
    'cartpole_swingup': (835, 22),
    'reacher_easy': (746, 25),
    'cheetah_run': (616, 18),
    'walker_walk': (891, 82),
}

def read_performance_from_logs(env_name, method_name, num_seeds=5):
    """Read final performance from generated log files."""
    performances = []
    
    for seed in range(1, num_seeds + 1):
        base_dir = f"../results/{env_name}_{method_name}_seed_{seed}"
        if os.path.exists(base_dir):
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if subdirs:
                log_path = os.path.join(base_dir, subdirs[0], 'eval.log')
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Get last evaluation (final performance)
                            last_eval = json.loads(lines[-1])
                            performances.append(last_eval['mean_episode_reward'])
                except: print(f"Warning: Could not read {log_path}")
    
    if performances: return np.mean(performances), np.std(performances)
    else:
        return None, None

def create_bar_chart():
    """Create bar chart with all 4 methods."""
    
    environments = ['finger_spin', 'cartpole_swingup', 'reacher_easy', 'cheetah_run', 'walker_walk']
    env_labels = ['Finger\nSpin', 'Cartpole\nSwingup', 'Reacher\nEasy', 'Cheetah\nRun', 'Walker\nWalk']
    
    rad_means = []
    rad_stds = []
    pixel_means = []
    pixel_stds = []
    curl_means = []
    curl_stds = []
    state_means = []
    state_stds = []
    
    for env in environments:
        mean, std = read_performance_from_logs(env, 'rad')
        rad_means.append(mean if mean else 0)
        rad_stds.append(std if std else 0)
        
        mean, std = read_performance_from_logs(env, 'pixel_sac')
        pixel_means.append(mean if mean else 0)
        pixel_stds.append(std if std else 0)
        
        mean, std = CURL_VALUES[env]
        curl_means.append(mean)
        curl_stds.append(std)
        
        mean, std = STATE_SAC_VALUES[env]
        state_means.append(mean)
        state_stds.append(std)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(environments))
    width = 0.2
    bars1 = ax.bar(x - 1.5*width, rad_means, width, yerr=rad_stds,label='RAD', color='#2ecc71', capsize=5)
    bars2 = ax.bar(x - 0.5*width, curl_means, width, yerr=curl_stds,label='CURL', color='#9b59b6', capsize=5)
    bars3 = ax.bar(x + 0.5*width, pixel_means, width, yerr=pixel_stds,label='Pixel SAC', color='#e74c3c', capsize=5)
    bars4 = ax.bar(x + 1.5*width, state_means, width, yerr=state_stds,label='State SAC (Oracle)', color='#f39c12',capsize=5)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 300:
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for i, env in enumerate(environments):
        if rad_means[i] > 0 and pixel_means[i] > 0:
            improvement = rad_means[i] / pixel_means[i]
    
    ax.set_xlabel('Environment', fontsize=14)
    ax.set_ylabel('Episode Reward (100k steps)', fontsize=14)
    ax.set_title('RAD Performance Comparison - DMControl100k', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels)
    ax.set_ylim(0, 1000)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/all_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('./plots/all_methods_comparison.pdf', bbox_inches='tight')
    print("Saved to ./plots/all_methods_comparison.png")
    
    plt.show()

def print_summary_table():
    """Print a summary table of all results."""
    
    print("\n" + '-'*90)
    print("PERFORMANCE SUMMARY")
    print('-'*90)
    print(f"{'Environment':<20} {'RAD':<15} {'CURL':<15} {'Pixel SAC':<15} {'State SAC':<15} {'Improvement'}")
    print("-"*90)
    
    environments = ['finger_spin', 'cartpole_swingup', 'reacher_easy', 'cheetah_run', 'walker_walk', 
    ]
    improvements = []
    for env in environments:
        row = f"{env:<20}"
        rad_mean, rad_std = read_performance_from_logs(env, 'rad')
        if rad_mean:
            row += f"{rad_mean:.0f} ± {rad_std:.0f}".ljust(15)
        else:
            row += "N/A".ljust(15)
        curl_mean, curl_std = CURL_VALUES[env]
        row += f"{curl_mean} ± {curl_std}".ljust(15)
        pixel_mean, pixel_std = read_performance_from_logs(env, 'pixel_sac')
        if pixel_mean:
            row += f"{pixel_mean:.0f} ± {pixel_std:.0f}".ljust(15)
        else:
            row += "N/A".ljust(15)
        state_mean, state_std = STATE_SAC_VALUES[env]
        row += f"{state_mean} ± {state_std}".ljust(15)
        if rad_mean and pixel_mean:
            improvement = rad_mean / pixel_mean
            improvements.append(improvement)
            row += f"{improvement:.1f}x"
        
        print(row)
    
    print("-"*90)
    if improvements:
        print(f"Average improvement: {np.mean(improvements):.1f}x")
    print('-'*90)

if __name__ == "__main__":
    print("Creating visualization with:")
    print("RAD and Pixel SAC: From generated logs")
    print("CURL and State SAC: From the paper (hardcoded)")
    
    create_bar_chart()
    print_summary_table()