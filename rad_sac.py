"""
RAD-SAC Implementation for EEC 256 Final Project

The below code implements the RAD (Reinforcement Learning with Augmented Data) algorithm
from the NeurIPS 2020 paper by Laskin et al. The key insight is that simple data
augmentations can dramatically improve sample efficiency in pixel-based RL without
needing complex auxiliary losses like CURL.

Project Option 3: Replicating "Reinforcement Learning with Augmented Data"
Course: EEC 256 Introduction to Reinforcement Learning
Author: Aunsh Bandivadekar
"""

import numpy as np
import torch
import gym
import os
import copy
import math
import random
import time
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# DATA AUGMENTATION FUNCTIONS
# These are the core data augmentations from the RAD paper
# The paper shows that simple augmentations like crop and translate are surprisingly effective for improving sample efficiency

def random_crop(imgs, out=84):
    """
    Random crop augmentation is the most effective augmentation in the paper
    
    This function randomly crops images to introduce translation invariance
    According to Table 1, this is the default augmentation for most environments
    except Walker Walk which uses crop instead of translate.
    
    Args:
        imgs: Batch of images with shape (B, C, H, W)
        out: Output size (typically 84x84 for DMControl)
    Returns: Randomly cropped images of size (B, C, out, out)
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1  # Maximum valid crop position
    # Generate random crop positions for each image in the batch
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # Create output tensor and crop each image
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)): cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    
    return cropped

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    """
    Random translate augmentation is the novel contribution of the RAD paper. This places the original image randomly within a larger frame of zeros
    
    Args: imgs: Input images (B, C, H, W)
        size: Size of the larger frame to translate within
        return_random_idxs: Whether to return translation indices
        h1s, w1s: Predefined translation positions (for consistency across frames)
    Returns: Translated images with same positioning across the batch
    """
    n, c, h, w = imgs.shape
    assert size >= h and size >= w, "Target size must be larger than input"
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype) # Create larger frame filled with zeros
    
    # Generate random positions if not provided
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    
    # Place each image at its random position
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s): out[:, h1:h1 + h, w1:w1 + w] = img
    # Return indices for consistent augmentation across frame stacks
    if return_random_idxs: return outs, dict(h1s=h1s, w1s=w1s)

    return outs

def no_aug(x):
    """No augmentation - used for baseline comparisons"""
    return x

# ----------------------------------
# NEURAL NETWORK UTILITIES

def t_weights(src, trg):
    """
    Tie weights between two network layers
    Used to share convolutional weights between actor and critic encoders
    """
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# Output dimensions for different image sizes and network depths
# These constants are crucial for proper encoder architecture
OUT_DIM = {2: 39, 4: 35, 6: 31}        # For 84x84 images
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}     # For 64x64 images  
OUT_DIM_108 = {4: 47}                  # For 108x108 images (with translate)

class Encoder(nn.Module):
    """
    Convolutional encoder for pixel observations.
    
    This is the visual encoder used in SAC for processing pixel observations.
    The architecture follows the paper: 4 conv layers with 32 filters each,
    followed by a linear layer and layer normalization.
    
    Key design choices which were replicated from the paper are 4 convolutional layers (num_layers=4), 32 filters per layer (num_filters=32), feature dimension of 50 for the final representation and ReLU activations throughout
    """
    
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3, "Expected (C, H, W) observation shape"
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Build convolutional layers
        # First layer: input channels -> num_filters with stride 2 (downsampling)
        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        
        # Remaining layers: num_filters -> num_filters with stride 1
        for i in range(num_layers - 1): self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # Calculate output dimensions based on input size
        # This is critical for determining the linear layer input size
        if obs_shape[-1] == 108:
            assert num_layers in OUT_DIM_108
            out_dim = OUT_DIM_108[num_layers]
        elif obs_shape[-1] == 64: out_dim = OUT_DIM_64[num_layers]
        else: out_dim = OUT_DIM[num_layers]

        # Final linear layer to compress to feature dimension
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        # Layer normalization for stable training
        self.ln = nn.LayerNorm(self.feature_dim)
        self.output_logits = output_logits

    def forward_conv(self, obs):
        """Forward pass through convolutional layers only"""
        # Normalize pixel values to [0, 1] range
        if obs.max() > 1.: obs = obs / 255.
        conv = torch.relu(self.convs[0](obs)) # Apply first conv layer with ReLU
        # Apply remaining conv layers
        for i in range(1, self.num_layers): conv = torch.relu(self.convs[i](conv))
        # Flatten for linear layer
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        """
        Full forward pass through the encoder
        Args: obs: Input observations (pixel values)
            detach: Whether to detach gradients (used in actor updates)
        """
        h = self.forward_conv(obs)
        if detach:h = h.detach() # Detach gradients if requested (prevents encoder updates during actor training)
        # Apply linear layer and layer normalization
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)

        # Output logits or tanh-squashed values
        if self.output_logits: out = h_norm
        else: out = torch.tanh(h_norm)
        return out

    def copy_conv_weights_from(self, source):
        """
        Copy convolutional weights from another encoder
        This is used to tie weights between actor and critic encoders
        """
        for i in range(self.num_layers): t_weights(src=source.convs[i], trg=self.convs[i])

def make_encoder(obs_shape, feature_dim, num_layers, num_filters, output_logits=False):
    """
    A fac function for creating encoders uses only pixel encoding
    """
    return Encoder(obs_shape, feature_dim, num_layers, num_filters, output_logits)

# ----------------------------
# SAC-SPECIFIC UTILITIES  

def soft_update_params(net, target_net, tau):
    """
    Soft update of target network parameters
    This is a key component of SAC as we slowly update the target networks
    using: target = tau * online + (1 - tau) * target
    This provide more stable learning than hard updates
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()): target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def center_crop_image(image, output_size):
    """
    Center crop an image to the specified output size, this is used for evaluation when we want consistent cropping
    """
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size
    # Calculate center crop coordinates
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w] # Crop the image
    return image
        
def gaussian_logprob(noise, log_std):
    """
    Compute the Gaussian log probability for SAC policy
    This is used in the SAC objective for entropy regularization.
    """
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    """
    Apply tanh squashing function for bounded actions
    SAC uses this to map unbounded Gaussian actions to bounded action space.
    See Appendix C from the SAC paper (Haarnoja et al.) for details https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None: pi = torch.tanh(pi)
    if log_pi is not None: log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True) # Correct log probability after tanh transformation
    return mu, pi, log_pi

def weight_init(m):
    """
    Custom weight initialization for neural network layers
    This uses orthogonal initialization which often works well for RL
    The delta-orthogonal init for conv layers is from the paper: Fixup Initialization: Residual Learning Without Normalization https://openreview.net/pdf?id=H1gsz30cKX
    """
    if isinstance(m, nn.Linear):
        # Orthogonal initialization for linear layers
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # Delta-orthogonal initialization for conv layers
        assert m.weight.size(2) == m.weight.size(3), "Expected square kernels"
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

# ----------------------------
# SAC NETWORK ARCHITECTURES

class Actor(nn.Module):
    """
    SAC Actor Network for continuous control
    This network outputs a stochastic policy parameterized as a Gaussian. The actor learns to maximize expected return while maintaining high entropy for exploration (the core idea of SAC).
    The architecture has a shared encoder for processing observations, a 3-layer MLP for policy head and outputs mean and log_std for Gaussian policy
    """
    def __init__(self, obs_shape, act_shape, hidden_dim,
                 enc_feat_dim, log_std_min, log_std_max, num_layers, num_filters):
        super().__init__()

        # Create encoder for processing observations
        self.encoder = make_encoder(obs_shape, enc_feat_dim, num_layers, num_filters, output_logits=True)

        # Clamp log standard deviation to prevent numerical instability
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # Policy head: 3-layer MLP
        # Output dimension is 2 * action_dim for mean and log_std
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * act_shape[0])
        )
        self.outputs = dict() # For logging purposes (not used in our simplified version)
        self.apply(weight_init) # Apply custom weight initialization

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_enc=False):
        """
        Forward pass through the actor network
        Args: obs: Input observations
            compute_pi: Whether to sample actions
            compute_log_pi: Whether to compute log probabilities
            detach_enc: Whether to detach encoder gradients
            
        Returns: mu: Mean of the policy distribution
            pi: Sampled actions (if compute_pi=True)
            log_pi: Log probabilities (if compute_log_pi=True)
            log_std: Log standard deviations
        """
        obs = self.encoder(obs, detach=detach_enc) # Encode observations
        mu, log_std = self.trunk(obs).chunk(2, dim=-1) # Get mean and log_std from policy head

        # Constrain log_std to valid range for numerical stability
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        # Store outputs for potential logging
        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()
        # Sample actions if requested
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else: pi = None
        # Compute log probabilities if requested
        if compute_log_pi: log_pi = gaussian_logprob(noise, log_std)
        else: log_pi = None
        mu, pi, log_pi = squash(mu, pi, log_pi) # Apply tanh squashing to bound actions

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=10000): 
        """Simplified logging method (no-op in this this version)"""
        pass

class Qfunc(nn.Module):
    """
    Q-function for SAC critic
    Simple 3-layer MLP that takes state-action pairs and outputs Q-values. SAC uses two Q-functions to reduce overestimation bias
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        # 3-layer MLP: [obs, action] -> hidden -> hidden -> Q-value
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        """
        Forward pass through Q-function
        Args: obs: Encoded observations
            action: Actions taken
        Returns: Q-value for the state-action pair
        """
        assert obs.size(0) == action.size(0), "Batch sizes must match"
        # Concatenate observations and actions
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class Critic(nn.Module):
    """
    SAC Critic Network with twin Q-functions
    Uses two Q-functions to reduce overestimation bias (Double Q-learning. The critic provides value estimates for the SAC actor updates.
    The architecture has a shared encoder for processing observations, two separate Q-function heads and target networks are maintained separately
    """
    def __init__(self, obs_shape, act_shape, hidden_dim,enc_feat_dim, num_layers, num_filters):
        super().__init__()
        # Shared encoder for both Q-functions
        self.encoder = make_encoder(obs_shape, enc_feat_dim, num_layers, num_filters, output_logits=True)
        # Twin Q-functions for double Q-learning
        self.Q1 = Qfunc(self.encoder.feature_dim, act_shape[0], hidden_dim)
        self.Q2 = Qfunc(self.encoder.feature_dim, act_shape[0], hidden_dim)
        self.outputs = dict() # For logging purposes
        self.apply(weight_init) # Apply custom weight initialization

    def forward(self, obs, action, detach_enc=False):
        """
        Forward pass through both Q-functions
        Args: obs: Input observations
            action: Actions to evaluate
            detach_enc: Whether to stop gradients through encoder
        Returns: q1, q2: Q-values from both Q-functions
        """
        obs = self.encoder(obs, detach=detach_enc) # Encode observations (optionally detach gradients)
        # Compute Q-values from both functions
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        # Store for potential logging
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return q1, q2

    def log(self, L, step, log_freq=10000): 
        """Simplified logging method (no-op in our version)."""
        pass

# -----------------------
# MAIN RAD-SAC AGENT

class RadSacAgent(object):
    """
    RAD-SAC Agent is the main contribution of the paper. This agent combines SAC (Soft Actor-Critic) with simple data augmentations
    applied directly to the RL objective.
    """
    def __init__(self, obs_shape, act_shape, device, hidden_dim=256, disc=0.99,
                 init_temp=0.01, alpha_lr=1e-3, alpha_beta=0.9, act_lr=1e-3,
                 act_beta=0.9, act_log_std_min=-10, act_log_std_max=2,
                 act_upd_freq=2, crt_lr=1e-3, crt_beta=0.9, crt_tau=0.005,
                 crt_targ_upd_freq=2, encoder_type='pixel', enc_feat_dim=50,
                 enc_lr=1e-3, enc_tau=0.005, num_layers=4, num_filters=32, log_int=100, detach_enc=False,
                 latent_dim=128, data_augs=''):
        
        # Store hyperparameters
        self.device = device
        self.disc = disc
        self.crt_tau = crt_tau
        self.enc_tau = enc_tau
        self.act_upd_freq = act_upd_freq
        self.crt_targ_upd_freq = crt_targ_upd_freq
        self.log_int = log_int
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_enc = detach_enc
        self.data_augs = data_augs

        # Parse data augmentations string and create function mapping
        # Example: 'crop' -> {crop: random_crop}
        self.augs_funcs = {}
        aug_to_func = {
            'crop': random_crop,
            'translate': random_translate,
            'no_aug': no_aug,
        }
        # Build augmentation function dictionary
        for aug_name in self.data_augs.split('-'):
            assert aug_name in aug_to_func, f'Invalid data augmentation: {aug_name}'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]
        # Create actor network
        self.actor = Actor(obs_shape, act_shape, hidden_dim,enc_feat_dim, act_log_std_min, act_log_std_max,
            num_layers, num_filters).to(device)
        # Create critic networks (online and target)
        self.critic = Critic(obs_shape, act_shape, hidden_dim,enc_feat_dim, num_layers, num_filters).to(device)
        self.critic_target = Critic(obs_shape, act_shape, hidden_dim,enc_feat_dim, num_layers, num_filters).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Initialize target network with same weights as online network

        # Tie encoder weights between actor and critic
        # This is important for sample efficiency
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        # SAC temperature parameter (controls exploration vs exploitation)
        self.log_alpha = torch.tensor(np.log(init_temp), dtype=torch.float32).to(device)
        self.log_alpha.requires_grad = True
        # Target entropy for automatic temperature tuning
        # Set to -|A| as suggested in SAC paper
        self.target_entropy = -np.prod(act_shape)
        
        # Create optimizers for all components
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_lr, betas=(act_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=crt_lr, betas=(crt_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))
        # Loss function for contrastive learning (unused in RAD)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # Set networks to training mode
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        """Sets the training mode for all networks"""
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        """Current temp parameter value."""
        return self.log_alpha.exp()

    def select_action(self, obs):
        """Select deterministic action for evaluation. It uses the mean of the policy distribution (no sampling)"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        """
        Here we sample stochastic action for training, also uses the full stochastic policy for exploration
        """
        # Crop observation if needed for consistency
        if obs.shape[-1] != self.image_size: obs = center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        """
        Update critic networks using SAC loss. This is the standard SAC critic update with Bellman backup.
        The key is that obs and next_obs have already been augmented when sampled from the replay buffer
        """
        with torch.no_grad():
            # Compute target value using next observations
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi # Take minimum Q-value to reduce overestimation bias
            target_Q = reward + (not_done * self.disc * target_V) # Bellman backup

        # Get current Q estimates from both critics
        current_Q1, current_Q2 = self.critic(obs, action, detach_enc=self.detach_enc)
        # Compute critic loss (MSE between current and target Q-values)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_int == 0:L.log('train_critic/loss', critic_loss, step) # Log critic loss occasionally

        # Update critic networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Log additional critic information
        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        """
        Here we update the actor and temp parameter. The actor update maximizes the SAC objective:
        J(π) = E[Q(s,a) - α*log π(a|s)]
        Temperature α is tuned automatically to match target entropy
        """
        _, pi, log_pi, log_std = self.actor(obs, detach_enc=True) # Detach encoder to prevent updates during actor training
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_enc=True)  # Get Q-values for sampled actions
        actor_Q = torch.min(actor_Q1, actor_Q2) # Take minimum Q-value (conservative estimate)
        
        # Actor loss: maximize Q-values while maintaining entropy
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        # Log actor metrics
        if step % self.log_int == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
            
        # Compute entropy for logging
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_int == 0: L.log('train_actor/entropy', entropy.mean(), step)
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.log(L, step)

        # Update temperature parameter (automatic entropy tuning)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_int == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        """This is the main update step for RAD-SAC"""
        # Sample batch from replay buffer with augmentations applied
        obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(self.augs_funcs) # sample_rad applies augmentations
        if step % self.log_int == 0: L.log('train/batch_reward', reward.mean(), step) # Log batch reward for monitoring
        self.update_critic(obs, action, reward, next_obs, not_done, L, step) # Update critic networks

        # Update actor (less frequently for stability)
        if step % self.act_upd_freq == 0: self.update_actor_and_alpha(obs, L, step)

        # Update target networks (soft updates)
        if step % self.crt_targ_upd_freq == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1, self.crt_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2, self.crt_tau)
            soft_update_params(self.critic.encoder, self.critic_target.encoder,self.enc_tau)

    def save(self, model_dir, step):
        """Save actor and critic networks"""
        torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        """Load saved networks"""
        self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(torch.load('%s/critic_%s.pt' % (model_dir, step)))