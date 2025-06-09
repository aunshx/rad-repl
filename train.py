"""
This is the main training script that implements the RAD (Reinforcement Learning with 
Augmented Data) algorithm. The script sets up the 
environment, replay buffer, agent, and runs the main training loop.
"""

from collections import defaultdict, deque
from termcolor import colored
import torch
import torch.nn as nn
import numpy as np
import gym
import sys
import random
import time
import torch
import shutil
import argparse
import os
import math
import json
import dmc2gym  # DeepMind Control Suite to Gym wrapper
import copy
from torch.utils.data import Dataset
from rad_sac import RadSacAgent

# ---------------------------------
# UTILITY CLASSES AND FUNCTIONS

class eval_mode(object):
    """
    Context manager for evaluation mode. This temporarily sets neural networks to evaluation mode (no dropout, fixed batch norm) and then restores the original training state. This is crucial for consistent evaluation
    during training since we want deterministic behavior when measuring performance.
    Usage:
        with eval_mode(agent):
            action = agent.select_action(obs)
    """
    def __init__(self, *models): self.models = models
    def __enter__(self):
        # Store current training states and switch to eval mode
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)  # Set to evaluation mode
    def __exit__(self, *args):
        # Restore original training states
        for model, state in zip(self.models, self.prev_states):model.train(state)
        return False

def set_seed_everywhere(seed):
    """
    Set the random seeds for reproducibility across all libraries. This is essential for scientific reproducibility as we need the same random
    initialization and sampling behavior across runs. We set seeds for PyTorch (both CPU and GPU), NumPy (used in data augmentations and environment) and Python's random module
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

def make_dir(dir_path):
    """    
    Simple utility function that attempts to create a directory and ignores the error if it already exists. Used for creating results directories.
    """
    try: os.mkdir(dir_path)
    except OSError: pass  # Directory already exists, which is fine
    return dir_path

# --------------------------
# REPLAY BUFFER IMPLEMENTATION

class ReplayBuffer(Dataset):
    """
    Experience replay buffer for off-policy reinforcement learning. This stores environment transitions (s, a, r, s', done) and provides methods to sample batches for training.The buffer is implemented as a circular buffer that overwrites old experiences when it reaches capacity. This is memory efficient for long training runs.
    """
    
    def __init__(self, obs_shape, act_shape, capacity, batch_size, device, image_size=84, 
                 pre_image_size=84, transform=None):
        """
        Initialize the replay buffer
        Args:
            obs_shape: Shape of observations (e.g., (9, 84, 84) for 3 stacked frames)
            act_shape: Shape of actions (e.g., (6,) for 6-dimensional continuous actions)
            capacity: Maximum number of transitions to store
            batch_size: Size of batches to sample
            device: PyTorch device (CPU or CUDA)
            image_size: Final image size after augmentation (84x84 is standard)
            pre_image_size: Image size before augmentation (may be larger for cropping)
            transform: Optional additional transforms (not used in RAD)
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.pre_image_size = pre_image_size
        self.transform = transform
        
        # Choose data type based on observation shape
        # Images use uint8 (0-255), low-dimensional states use float32
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        # Pre-allocate arrays for efficiency
        # This avoids dynamic memory allocation during training
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *act_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        # Buffer state tracking
        self.idx = 0        # Current insertion index
        self.last_save = 0  # For incremental saving (not used in our implementation)
        self.full = False   # Whether buffer has wrapped around

    def add(self, obs, action, reward, next_obs, done):
        """
        Add a transition to the replay buffer
        This stores a single (s, a, r, s', done) transition. We use np.copyto for efficiency rather than assignment, which avoids creating new arrays.
        
        Args: obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode terminated
        """
        # Copy data to avoid reference issues
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)  # Store NOT done for easier computation

        # Update buffer state with circular indexing
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0  # Mark as full when we wrap around

    def sample_rad(self, aug_funcs):
        """
        Sample a batch with RAD augmentations applied. This is the key method that implements RAD. It samples a batch of transitions and applies data augmentations to the observations. The augmentations are
        applied consistently to both current and next observations to maintain
        temporal consistency.
        
        Args: aug_funcs: Dictionary mapping augmentation names to functions e.g., {'crop': random_crop, 'translate': random_translate}
        Returns: Tuple of (obs, actions, rewards, next_obs, not_dones) as PyTorch tensors
        """
        # Sample random indices from the filled portion of the buffer
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)
      
        # Get observations for the sampled indices
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        
        # Apply augmentations if specified
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # Apply crop and cutout first (these change image size)
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug: 
                    # Translation requires special handling to ensure consistency
                    # First crop to pre_image_size, then translate to final size
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    # Use same random translation for current and next observations
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)                     

        # Convert to PyTorch tensors and move to device
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        # Normalize pixel values from [0, 255] to [0, 1]
        obses = obses / 255.
        next_obses = next_obses / 255.

        # Apply remaining augmentations (those that don't change image size)
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # Skip augmentations already applied above
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug: continue
                obses = func(obses)
                next_obses = func(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        """
        Save replay buffer to disk incrementally. This saves only the new experiences since the last save, which is more efficient than saving the entire buffer each time.
        Args:
            save_dir: Directory to save buffer chunks
        """
        if self.idx == self.last_save: return  # Nothing new to save
            
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        """Load replay buffer from disk."""
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start  # Verify consistent loading order
            # Restore buffer contents
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        """
        Getting a single item from the buffer. This implements the Dataset interface for PyTorch DataLoader compatibility, though we primarily use sample_rad() for RAD training.
        Args: idx: Index (ignored, we sample randomly)
        Returns: Single transition tuple
        """
        # Sample randomly rather than using provided index
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        idx = idx[0]
        # Get transition components
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]
        # Apply transforms if specified
        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        """Return buffer capacity for Dataset interface."""
        return self.capacity 

# ---------------------------------
# ENV WRAPPERS

class FrameStack(gym.Wrapper):
    """
    Frame stacking stacks the last k frames together to provide temporal context to the agent. For eg, with k=3, the observation becomes 9 channels (3 RGB frames) instead
    of 3 channels (1 RGB frame). This helps the agent understand motion and velocity.
    """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)  # Efficient circular buffer for frames
        
        # Update observation space to reflect stacked frames
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),  # k times more channels
            dtype=env.observation_space.dtype
        )
        # Preserve episode length limit
        self._max_episode_steps = getattr(env, "_max_episode_steps", 1000)

    def reset(self):
        """
        Reset environment and initialize frame stack.
        Returns: Stacked initial observation
        """
        obs = self.env.reset()
        # Fill frame stack with initial observation
        for _ in range(self._k): self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        """
        Step environment and update frame stack
        """
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)  # Add new frame (automatically removes oldest)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """
        Concatenates all frames in the stack along the channel dimension. For RGB images, this turns k frames of shape (3, H, W) into (3k, H, W).
        Returns: Stacked observation array
        """
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

# --------------------------------
# IMAGE PROCESSING UTILITIES

def center_crop_image(image, output_size):
    """
    Here we crop the image from the center, which is used during evaluation to ensure consistent cropping (unlike random crop during training). Center cropping preserves the most important visual information.
    
    Args: image: Input image with shape (C, H, W)
        output_size: Desired output size (square)
    Returns: Center-cropped image of shape (C, output_size, output_size)
    """
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    # Calculate crop coordinates from center
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    # Perform the crop
    image = image[:, top:top + new_h, left:left + new_w]
    return image

def center_translate(image, size):
    """
    Center an image within a larger frame of zeros. THis is as part of the translation augmentation pipeline where we first crop to a smaller size, then translate within a larger frame.
    
    Args: image: Input image with shape (C, H, W)
        size: Size of the larger frame
    Returns: Image centered in larger frame of shape (C, size, size)
    """
    c, h, w = image.shape
    assert size >= h and size >= w, "Output size must be at least as large as input"
    
    # Create larger frame filled with zeros
    outs = np.zeros((c, size, size), dtype=image.dtype)
    
    # Calculate center position
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    
    # Place image at center
    outs[:, h1:h1 + h, w1:w1 + w] = image
    return outs

# ---------------------
# LOGGING & METRICS

class AcgMet(object):
    """
    Accumulating metric for tracking running averages. This maintains a running sum and count to compute running averages of metrics like rewards, losses, etc. More memory efficient than storing all individual values
    """
    
    def __init__(self):
        self._sum = 0
        self._count = 0
        
    def update(self, value, n=1):
        """Add new value(s) to the accumulator."""
        self._sum += value
        self._count += n
        
    def value(self):
        """Get current average value."""
        return self._sum / max(1, self._count)

class MetricGrp(object):
    """
    Group of metrics that can be logged together. Each metric group (train/eval) has its own formatting specification.
    """
    
    def __init__(self, file_name, formating):
        """
        Initialize metrics group.
        """
        self._file_name = file_name
        # Remove existing log file to start fresh
        if os.path.exists(file_name): os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AcgMet)  # Auto-create meters as needed

    def log(self, key, value, n=1):
        """Log a value to the appropriate meter."""
        self._meters[key].update(value, n)

    def _prime_meters(self):
        """
        Prepare meter data for output. Here we extract values from all meters and cleans up key names
        by removing train/eval prefixes and replacing slashes with underscores.
        Returns: Dictionary of cleaned metric names to values
        """
        data = dict()
        for key, meter in self._meters.items():
            # Clean up key names for output
            if key.startswith('train'): 
                key = key[len('train') + 1:]
            else: 
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')  # Replace slashes for valid file names
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        """Write metrics to log file as JSON."""
        with open(self._file_name, 'a') as f: 
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        """
        Format a metric value for display
        Args:key: Metric name
            value: Metric value
            ty: Format type ('int', 'float', 'time')
        Returns:Formatted string
        """
        template = '%s: '
        if ty == 'int': template += '%d'
        elif ty == 'float': template += '%.04f'
        elif ty == 'time': template += '%.01f s'
        else: raise ValueError('invalid format type: %s' % ty)
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        """
        Print metrics to console with color coding.
        
        Args: data: Dictionary of metric values
            prefix: 'train' or 'eval' for color coding
        """
        # Color code: yellow for train, green for eval
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        
        # Format each configured metric
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        """
        Output all metrics and clear meters.
        Args: step: Training step number
            prefix: 'train' or 'eval'
        """
        if len(self._meters) == 0: return  # Nothing to dump
            
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()  # Reset for next period

class Logger(object):
    """
    Main logging class that manages train and eval metrics.
    It creates separate metric groups for training and evaluation each with their own formatting specifications and log files.
    """
    
    def __init__(self, log_dir, use_tb=False, config='rl'):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for log files
            use_tb: Whether to use TensorBoard (not implemented in our version)
            config: Configuration type (not used)
        """
        self._log_dir = log_dir
        
        # Training metrics: episode info, rewards, and loss values
        self._train_mg = MetricGrp(
            os.path.join(log_dir, 'train.log'),
            formating=[
                ('episode', 'E', 'int'), # Episode number
                ('step', 'S', 'int'), # Training step
                ('duration', 'D', 'time'), # Episode duration
                ('episode_reward', 'R', 'float'), # Episode reward
                ('batch_reward', 'BR', 'float'), # Average batch reward
                ('actor_loss', 'A_LOSS', 'float'), # Actor loss
                ('critic_loss', 'CR_LOSS', 'float') # Critic loss
            ]
        )
        
        # Evaluation metrics: simpler, just step and reward
        self._eval_mg = MetricGrp(
            os.path.join(log_dir, 'eval.log'),
            formating=[
                ('step', 'S', 'int'), # Training step
                ('episode_reward', 'ER', 'float') # Evaluation reward
            ]
        )

    def log(self, key, value, step, n=1):
        """
        Log a metric value
        Args: key: Metric key (must start with 'train' or 'eval')
            value: Metric value
            step: Training step (not used here, passed to dump())
            n: Number of samples (for averaging)
        """
        assert key.startswith('train') or key.startswith('eval')
        
        # Convert tensor values to scalars
        if type(value) == torch.Tensor: value = value.item()
            
        # Route to appropriate meter group
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def dump(self, step):
        """Dump both training and evaluation metrics"""
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')

# ---------------------------------
# ARGUMENT PARSING

def parse_args():
    """
    This defines all the hyperparameters and settings for the RAD-SAC algorithm. The defaults are chosen based on the original paper and work well for most
    DMControl tasks.
    
    Returns: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser()
    
    # Environment configuration
    parser.add_argument('--domain_name', default='cartpole',help='DMControl domain (e.g., cartpole, walker, finger)')
    parser.add_argument('--task_name', default='swingup',help='DMControl task (e.g., swingup, walk, spin)')
    parser.add_argument('--pre_trans_img_size', default=100, type=int,help='Image size before augmentation (larger for cropping)')
    parser.add_argument('--image_size', default=84, type=int, help='Final image size after augmentation')
    parser.add_argument('--action_repeat', default=1, type=int, help='Number of times to repeat each action')
    parser.add_argument('--frame_stack', default=3, type=int,help='Number of frames to stack for temporal info')
    
    # Replay buffer configuration
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int,    help='Maximum number of transitions in replay buffer')
    
    # Training configuration
    parser.add_argument('--agent', default='rad_sac', type=str,    help='Agent type (should be rad_sac for this project)')
    parser.add_argument('--init_steps', default=1000, type=int,
                       help='Number of random exploration steps before training')
    parser.add_argument('--num_train_steps', default=1000000, type=int,
                       help='Total number of training steps')
    parser.add_argument('--batch_size', default=32, type=int,
                       help='Batch size for training')
    parser.add_argument('--hidden_dim', default=1024, type=int,
                       help='Hidden dimension for actor/critic networks')
    
    # Evaluation configuration
    parser.add_argument('--eval_freq', default=1000, type=int,
                       help='How often to evaluate the agent (in steps)')
    parser.add_argument('--num_eval_epis', default=10, type=int,
                       help='Number of episodes to run during evaluation')
    
    # Critic hyperparameters
    parser.add_argument('--crt_lr', default=1e-3, type=float,
                       help='Critic learning rate')
    parser.add_argument('--crt_beta', default=0.9, type=float,
                       help='Beta1 for critic Adam optimizer')
    parser.add_argument('--crt_tau', default=0.01, type=float,
                       help='Soft update coefficient for critic target network')
    parser.add_argument('--crt_targ_upd_freq', default=2, type=int,
                       help='Frequency of critic target network updates')
    
    # Actor hyperparameters
    parser.add_argument('--act_lr', default=1e-3, type=float,
                       help='Actor learning rate')
    parser.add_argument('--act_beta', default=0.9, type=float,
                       help='Beta1 for actor Adam optimizer')
    parser.add_argument('--act_log_std_min', default=-10, type=float,
                       help='Minimum log standard deviation for actor policy')
    parser.add_argument('--act_log_std_max', default=2, type=float,
                       help='Maximum log standard deviation for actor policy')
    parser.add_argument('--act_upd_freq', default=2, type=int,
                       help='Frequency of actor network updates')
    
    # Encoder hyperparameters
    parser.add_argument('--encoder_type', default='pixel', type=str,
                       help='Encoder type (pixel for image observations)')
    parser.add_argument('--enc_feat_dim', default=50, type=int,
                       help='Encoder feature dimension')
    parser.add_argument('--enc_lr', default=1e-3, type=float,
                       help='Encoder learning rate')
    parser.add_argument('--enc_tau', default=0.05, type=float,
                       help='Soft update coefficient for encoder target network')
    parser.add_argument('--num_layers', default=4, type=int,
                       help='Number of convolutional layers in encoder')
    parser.add_argument('--num_filters', default=32, type=int,
                       help='Number of filters per convolutional layer')
    parser.add_argument('--latent_dim', default=128, type=int,
                       help='Latent dimension (not used in basic RAD)')
    
    # SAC hyperparameters
    parser.add_argument('--disc', default=0.99, type=float,
                       help='Discount factor gamma')
    parser.add_argument('--init_temp', default=0.1, type=float,
                       help='Initial temperature for SAC entropy regularization')
    parser.add_argument('--alpha_lr', default=1e-4, type=float,
                       help='Learning rate for temperature parameter')
    parser.add_argument('--alpha_beta', default=0.5, type=float,
                       help='Beta1 for temperature Adam optimizer')
    
    # Misc
    parser.add_argument('--seed', default=1, type=int,help='Random seed for reproducibility')
    parser.add_argument('--work_dir', default='.', type=str, help='Working directory for saving results')
    parser.add_argument('--save_tb', default=False, action='store_true',help='Save TensorBoard logs (not implemented)')
    parser.add_argument('--save_buffer', default=False, action='store_true',help='Save replay buffer to disk')
    parser.add_argument('--detach_enc', default=False, action='store_true',help='Detach encoder gradients during actor updates')
    
    # Data augmentation configuration
    parser.add_argument('--data_augs', default='crop', type=str, help='Data augmentations to use (crop, translate, or no_aug)')
    parser.add_argument('--log_int', default=100, type=int, help='Logging interval (in training steps)')
    
    args = parser.parse_args()
    return args

# ------------------------
# EVALUATION FUNC

def evaluate(env, agent, num_episodes, L, step, args):
    """
    This runs the agent in the environment for multiple episodes without training to measure its current performance. We use deterministic actions (mean of policy)
    rather than stochastic sampling for more consistent evaluation.
    
    Args:env: Environment to evaluate in
        agent: Agent to evaluate
        num_episodes: Number of episodes to run
        L: Logger for recording results
        step: Current training step
        args: Configuration arguments
    """
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        """Run the evaluation loop"""
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                # Apply same image preprocessing as during trainin this ensures consistent observation processing
                if args.encoder_type == 'pixel' and 'crop' in args.data_augs: obs = center_crop_image(obs, args.image_size)
                if args.encoder_type == 'pixel' and 'translate' in args.data_augs:
                    # For translation: first crop, then center translate
                    obs = center_crop_image(obs, args.pre_trans_img_size)
                    obs = center_translate(obs, args.image_size)
                
                # Select action in evaluation mode (no training)
                with eval_mode(agent):
                    if sample_stochastically: action = agent.sample_action(obs / 255.)  # Stochastic
                    else: action = agent.select_action(obs / 255.)  # Deterministic
                # Take action in environment
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            # Log individual episode reward
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        # Log evaluation timing and statistics
        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    # Run deterministic evaluation (more reliable for measuring progress)
    run_eval_loop(sample_stochastically=False)
    L.dump(step)  # Write all logged metrics to files

# --------------------
# AGENT FACTORY FUNC

def make_agent(obs_shape, act_shape, args, device):
    """
    This factory function creates the agent with all the hyperparameters parsed from command line arguments. It centralizes agent creation and makes it easy to experiment with different configurations.
    """
    return RadSacAgent(
        obs_shape=obs_shape,
        act_shape=act_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        disc=args.disc,
        init_temp=args.init_temp,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        act_lr=args.act_lr,
        act_beta=args.act_beta,
        act_log_std_min=args.act_log_std_min,
        act_log_std_max=args.act_log_std_max,
        act_upd_freq=args.act_upd_freq,
        crt_lr=args.crt_lr,
        crt_beta=args.crt_beta,
        crt_tau=args.crt_tau,
        crt_targ_upd_freq=args.crt_targ_upd_freq,
        enc_feat_dim=args.enc_feat_dim,
        enc_lr=args.enc_lr,
        enc_tau=args.enc_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_int=args.log_int,
        detach_enc=args.detach_enc,
        latent_dim=args.latent_dim,
        data_augs=args.data_augs
    )

# ------------------------
# MAIN TRAINING LOOP

def main():
    """
    Main training func that implements the RAD-SAC training loop
    """
    # Parse command line arguments
    args = parse_args()
    
    # Handle random seed generation if not specified
    if args.seed == -1: args.__dict__["seed"] = np.random.randint(1, 1000000)
    set_seed_everywhere(args.seed)

    # Determine image sizes based on augmentation strategy
    # For translation, we need a larger pre-transform size to crop from
    pre_trans_img_size = args.pre_trans_img_size if 'crop' in args.data_augs else args.image_size
    pre_image_size = args.pre_trans_img_size  # Store for replay buffer

    # Create DMControl environment
    # DMControl provides high-quality physics simulations for RL research
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,  # Don't show reward as visual overlay
        from_pixels=(args.encoder_type == 'pixel'),  # Use pixel observations
        height=pre_trans_img_size,
        width=pre_trans_img_size,
        frame_skip=args.action_repeat  # Action repeat for temporal abstraction
    )
    env.seed(args.seed)

    # Apply frame stacking for temporal information
    # This is crucial for pixel-based RL since single frames lack velocity info
    if args.encoder_type == 'pixel': env = FrameStack(env, k=args.frame_stack)
    
    # Create unique experiment directory with timestamp and configuration
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = 'im' + str(args.image_size) +'-b'  \
    + str(args.batch_size)
    args.work_dir = args.work_dir + '/'  + exp_name
    make_dir(args.work_dir)

    # Save configuration for reproducibility
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f: json.dump(vars(args), f, sort_keys=True, indent=4)

    # Set up compute device (prefer CUDA if available)
    # TODO: If you're using this on an Apple silicon MAc then change cuda to mps
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get action space shape
    act_shape = env.action_space.shape

    # Set up observation shapes for different encoder types
    if args.encoder_type == 'pixel':
        # For pixels: (channels * frame_stack, height, width)
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack, pre_trans_img_size, pre_trans_img_size)
    else:
        # For state-based observations
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    # Create replay buffer with appropriate observation shape
    # The buffer stores pre-augmentation observations and applies augmentations during sampling
    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        act_shape=act_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        pre_image_size=pre_image_size,
    )

    # Create the RAD-SAC agent
    agent = make_agent(obs_shape=obs_shape, act_shape=act_shape, args=args,device=device)
    L = Logger(args.work_dir, use_tb=False) # Set up logging

    # Initialize training variables
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    # --------------------
    # MAIN TRAINING LOOP
    print(f"Starting RAD-SAC training for {args.num_train_steps} steps")
    print(f"Environment: {args.domain_name}_{args.task_name}")
    print(f"Augmentations: {args.data_augs}")
    print(f"Device: {device}")
    
    for step in range(args.num_train_steps):
        # --------------------------
        # PERIODIC EVALUATION
        if step % args.eval_freq == 0:
            print(f"Step {step}: Running evaluation")
            L.log('eval/episode', episode, step)
            evaluate(env, agent, args.num_eval_epis, L, step, args)

        # --------------------
        # EPISODE RESET HANDLING
        if done:
            if step > 0:
                # Log episode timing
                if step % args.log_int == 0:
                    L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
            
            # Log episode reward
            if step % args.log_int == 0:
                L.log('train/episode_reward', episode_reward, step)
                print(f"Episode {episode}, Step {step}: Reward = {episode_reward:.2f}")

            # Reset for new episode
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            
            if step % args.log_int == 0:
                L.log('train/episode', episode, step)

        # -----------------------
        # ACTION SELECTION
        # Initial random exploration phase
        # This helps fill the replay buffer with diverse experiences
        if step < args.init_steps: action = env.action_space.sample()
        else:
            # Use learned policy for action selection
            with eval_mode(agent): action = agent.sample_action(obs / 255.)

        # --------------------------------
        # AGENT TRAINING
        if step >= args.init_steps:
            # Only start training after initial exploration
            num_updates = 1  # Number of gradient updates per environment step
            # The replay buffer applies augmentations during samplin
            for _ in range(num_updates): agent.update(replay_buffer, L, step)

        # -------------------------
        # ENV INTERACTION
        next_obs, reward, done, _ = env.step(action)

        # Handle infinite bootstrap for time limits
        # If episode ends due to time limit (not failure), we don't want to bootstrap from a terminal state
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        
        episode_reward += reward
        
        # Store transition in replay buffer
        # Note: observations are stored without augmentation
        # Augmentations are applied during sampling in sample_rad()
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        # Prepare for next step
        obs = next_obs
        episode_step += 1

    print("Training completed")
    print(f"Results saved in: {args.work_dir}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()