# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train Diffusion Policy with online RL (PPO) for Isaac Lab environment."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Diffusion Policy with online RL for Isaac Lab environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--task", type=str, default="Template-So-100-FishRod-CubeLift-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations.")
parser.add_argument("--horizon", type=int, default=50, help="Number of steps per rollout before PPO update (not episode length).")
parser.add_argument("--max_episode_steps", type=int, default=None, help="Maximum steps per episode before forced reset (None = no limit, episode continues). Useful to prevent infinite episodes.")
parser.add_argument("--save_interval", type=int, default=2, help="Save checkpoint every N iterations.")
parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N iterations.")
parser.add_argument("--checkpoint_dir", type=str, default="outputs/diffusion_rl_checkpoints", help="Directory to save checkpoints.")
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path.")
parser.add_argument("--pretrained", type=str, default=None, help="Load pretrained Diffusion Policy weights (e.g., from LeRobot) before training.")
parser.add_argument("--dataset_root", type=str, default=None, help="Root directory of dataset (for LeRobot stats).")
parser.add_argument("--freeze_vision", action="store_true", default=False, help="Freeze vision encoder when using pretrained weights.")
# Diffusion Policy specific arguments
parser.add_argument("--diffusion_horizon", type=int, default=16, help="Diffusion policy prediction horizon (number of predicted actions).")
parser.add_argument("--n_action_steps", type=int, default=8, help="Number of action steps to execute before replanning.")
parser.add_argument("--n_obs_steps", type=int, default=2, help="Number of observation steps for temporal conditioning.")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for value network.")
parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers (for fallback policy).")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (for fallback policy).")
# RL hyperparameters
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for policy and value networks.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda for advantage estimation.")
parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clipping epsilon.")
parser.add_argument("--entropy_coef", type=float, default=0.001, help="Entropy coefficient for exploration.")
parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO update epochs per iteration.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for PPO updates.")
parser.add_argument("--use_vision", action="store_true", default=False, help="Use vision input (cameras).")
parser.add_argument("--image_size", type=int, default=512, help="Image size for vision input (512x512 per camera).")
parser.add_argument("--dump_input", action="store_true", default=False, help="Dump model inputs (images) for debugging.")
parser.add_argument("--dump_dir", type=str, default="outputs/train_debug", help="Directory to dump debug inputs.")
parser.add_argument("--dump_step", type=int, default=None, help="Global step index at which to dump model inputs (if not set, dumps at first step).")
parser.add_argument("--camera_source", type=str, choices=["sensor", "viewport"], default="sensor", help="Where to read RGB from: 'sensor' (default) or 'viewport' (GUI capture).")

# Action mapping arguments (from Act_RL.py)
parser.add_argument("--action_map", type=str, default="none", choices=["none","abs2norm","delta2norm","abs2default"], help="Map model actions to env normalized commands.")
parser.add_argument("--arm_scale", type=float, default=0.5, help="Env JointPositionActionCfg scale used to normalize arm actions.")
parser.add_argument("--arm_scales", type=str, default=None, help="Comma-separated per-joint scales for 5 arm joints.")
parser.add_argument("--arm_bias", type=str, default=None, help="Comma-separated per-joint bias (rad) for 5 arm joints; added after scaling.")
parser.add_argument("--min_cmd", type=float, default=0.0, help="Minimum normalized command magnitude per joint when using abs2norm.")
parser.add_argument("--pos_tol", type=float, default=0.0, help="Position error tolerance.")
parser.add_argument("--curr_from_env", action="store_true", default=False, help="Use env.scene['robot'].joint_pos (radians) as current joints.")
parser.add_argument("--axis_signs", type=str, default=None, help="Comma-separated signs for 5 arm axes.")
parser.add_argument("--gripper_gain", type=float, default=1.0, help="Multiply gripper action output by this gain.")
parser.add_argument("--anti_stall", action="store_true", default=False, help="Enable anti-stall.")
parser.add_argument("--anti_stall_window", type=int, default=15, help="Window size (steps) for anti-stall detection.")
parser.add_argument("--anti_stall_thresh", type=float, default=0.05, help="Mean absolute normalized arm magnitude threshold to trigger boost.")
parser.add_argument("--anti_stall_boost", type=float, default=1.5, help="Multiplicative boost applied to normalized arm when anti-stall triggers.")
parser.add_argument("--camera_format", type=str, choices=["rgb", "bgr"], default="rgb", help="Camera color format (rgb or bgr).")

# Exploration arguments
parser.add_argument("--exploration_noise", type=float, default=0.1, help="Standard deviation of Gaussian noise added to actions for exploration (0.0 = no exploration).")
parser.add_argument("--exploration_decay", type=float, default=0.995, help="Decay factor for exploration noise per iteration.")
parser.add_argument("--exploration_min", type=float, default=0.01, help="Minimum exploration noise (floor value after decay).")
# Reward-Weighted Finetuning arguments
parser.add_argument("--bc_lambda", type=float, default=0.1, help="Weight for reward-weighted BC loss (0.0 = no BC, higher = more BC). Recommended: 0.01-0.5")
parser.add_argument("--advantage_clip", type=float, default=10.0, help="Clip advantages for stability in reward-weighted BC loss.")
parser.add_argument("--positive_advantage_only", action="store_true", default=True, help="Only use positive advantage samples for BC loss (conservative update).")
parser.add_argument("--bc_batch_size", type=int, default=2, help="Batch size for BC loss computation (smaller than PPO batch to save memory).")

# Jaw-Payload contact reward arguments
parser.add_argument("--jaw_contact_reward", type=float, default=0.5, help="Extra reward when jaw touches payload (distance < threshold).")
parser.add_argument("--jaw_contact_threshold", type=float, default=0.05, help="Distance threshold (meters) for jaw-payload contact detection.")
parser.add_argument("--jaw_distance_reward_scale", type=float, default=2.0, help="Scale factor for distance-based reward (reward = scale * distance_delta).")
parser.add_argument("--jaw_distance_penalty_scale", type=float, default=1.0, help="Scale factor for distance-based penalty when moving away (penalty = scale * distance_delta).")
parser.add_argument("--jaw_prim", type=str, default="Moving_Jaw", help="Name of the jaw body/link in the robot articulation.")
parser.add_argument("--payload_prim", type=str, default="/World/FishRod/Rod/Payload", help="USD prim path of the payload object.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if using vision
if args_cli.use_vision:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
from typing import Dict, Tuple, Optional
import time
import cv2

from isaaclab_tasks.utils import parse_env_cfg
import SO_100.tasks  # noqa: F401


# Try LeRobot imports
try:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy as make_lerobot_policy
    HAVE_LEROBOT = True
except ImportError:
    HAVE_LEROBOT = False
    print("[WARNING] LeRobot not found. Pretrained loading might fail.")


def resize_with_padding_rgb(img_hwc_float01: np.ndarray, target_hw=(512, 512)) -> np.ndarray:
    """Copy of resize_with_padding_rgb from test_ACT.py (letterbox to target_hw)."""
    # img_hwc_float01: (H, W, 3), values in [0,1]
    h, w = img_hwc_float01.shape[:2]
    # Validate target size tuple and sanitize values
    try:
        if target_hw is None or not isinstance(target_hw, (tuple, list)) or len(target_hw) != 2:
            return img_hwc_float01
        th_raw, tw_raw = target_hw
        th_f = float(th_raw)
        tw_f = float(tw_raw)
        if not np.isfinite(th_f) or not np.isfinite(tw_f):
            return img_hwc_float01
        th = int(round(th_f))
        tw = int(round(tw_f))
    except Exception:
        return img_hwc_float01
    if th is None or tw is None or th <= 0 or tw <= 0:
        return img_hwc_float01
    if h is None or w is None or h <= 0 or w <= 0:
        return img_hwc_float01
    # Non-square: direct resize
    if th != tw:
        try:
            return cv2.resize(img_hwc_float01, (tw, th), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return img_hwc_float01
    # Square: letterbox pad to preserve aspect ratio
    scale = min(tw / w, th / h)
    new_w = int(w * scale) if w > 0 else 0
    new_h = int(h * scale) if h > 0 else 0
    if new_w <= 0 or new_h <= 0:
        return img_hwc_float01
    try:
        resized = cv2.resize(img_hwc_float01, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return img_hwc_float01
    padded = np.zeros((th, tw, 3), dtype=resized.dtype)
    off_y = (th - new_h) // 2
    off_x = (tw - new_w) // 2
    padded[off_y:off_y+new_h, off_x:off_x+new_w, :] = resized
    return padded


def _preprocess_single_cam_for_act(
    img,
    image_size: int = 512,
    camera_format: str = "rgb",
) -> np.ndarray:
    """Preprocess a single camera: Letterbox Resize (SmolVLA style).
    
    Matches 'LeRobotAdapter' logic in test_ACT.py (lines 869-925).
    - Float conversion
    - BGR->RGB if needed
    - Letterbox resize (padding)
    - NO HSV Boost / Gamma Correction (assuming Viewport or handled elsewhere)
    """
    if img is None:
        return None

    # Convert to numpy
    if isinstance(img, torch.Tensor):
        cam_np = img.detach().cpu().numpy()
    else:
        cam_np = np.array(img)

    # Remove batch dim if present; if multi-batch, take first
    if cam_np.ndim == 4:
        cam_np = cam_np[0]
    # If CHW, convert to HWC
    if cam_np.ndim == 3 and cam_np.shape[0] in (1, 3, 4) and cam_np.shape[-1] not in (1, 3, 4):
        cam_np = np.transpose(cam_np, (1, 2, 0))
    # Drop alpha if present
    if cam_np.ndim == 3 and cam_np.shape[-1] == 4:
        cam_np = cam_np[..., :3]
    # Grayscale to RGB
    if cam_np.ndim == 2:
        cam_np = np.stack([cam_np] * 3, axis=-1)
    if cam_np.ndim == 3 and cam_np.shape[-1] == 1:
        cam_np = np.repeat(cam_np, 3, axis=-1)

    th, tw = image_size, image_size
    # Final guard: if still invalid, fallback to zeros of target size
    if cam_np.ndim != 3 or cam_np.shape[-1] != 3:
        cam_np = np.zeros((int(th), int(tw), 3), dtype=np.float32)

    # Convert to float [0,1]
    if cam_np.dtype == np.uint8:
        cam_np = cam_np.astype(np.float32) / 255.0
    else:
        cam_np = np.clip(cam_np.astype(np.float32), 0.0, 1.0)

    # If incoming frames are BGR, convert to RGB
    if camera_format == "bgr":
        cam_np = cam_np[:, :, ::-1]

    # Letterbox Resize (Padding)
    h, w = cam_np.shape[:2]
    if (h == th) and (w == tw):
        img_resized = cam_np
    else:
        img_resized = resize_with_padding_rgb(cam_np, target_hw=(th, tw))

    # CHW
    cam_chw = np.transpose(img_resized, (2, 0, 1))
    return cam_chw

class DiffusionPolicy(nn.Module):
    """
    Wrapper for LeRobot Diffusion Policy with RL training support.
    If 'pretrained_path' is provided, it loads the LeRobot Diffusion model.
    Uses observation history and action queue mechanism similar to test_Diffision.py.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        diffusion_horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        use_vision: bool = False,
        image_size: int = 512,
        pretrained_path: str = None,
        dataset_root: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.diffusion_horizon = diffusion_horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_vision = use_vision
        self.image_size = image_size
        self.device_str = device
        self.lerobot_policy = None
        
        # Observation history buffer for temporal conditioning
        self.obs_history = []
        
        # Action queue for executing predicted actions
        self.action_queue = []
        
        # Step counter for tracking
        self.step_count = 0
        
        # Log std for action distribution (PPO requirement)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -5.0)
        self.min_log_std = -10.0
        self.max_log_std = -5.0
        
        # ACTION RESIDUAL ADAPTER - This is what we train with RL!
        # Larger MLP for better learning capacity
        # Still memory efficient: ~200KB parameters (vs 200MB diffusion policy)
        self.action_adapter = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        # Scale for residual (start small, let it learn)
        self.adapter_scale = nn.Parameter(torch.tensor(0.1))
        
        # Try loading LeRobot Diffusion policy if path provided
        if pretrained_path and HAVE_LEROBOT:
            try:
                print(f"[INFO] Loading LeRobot Diffusion Policy from {pretrained_path}")
                
                # Handle directory vs file
                if os.path.isfile(pretrained_path):
                    pretrained_dir = os.path.dirname(pretrained_path)
                else:
                    pretrained_dir = pretrained_path
                
                # Load config
                train_cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
                train_cfg.policy.pretrained_path = pretrained_dir
                train_cfg.policy.device = device
                
                # Update with CLI overrides
                if diffusion_horizon != 16:
                    train_cfg.policy.horizon = diffusion_horizon
                if n_action_steps != 8:
                    train_cfg.policy.n_action_steps = n_action_steps
                if n_obs_steps != 2:
                    train_cfg.policy.n_obs_steps = n_obs_steps
                
                # Cache input keys for forward pass
                self.lerobot_input_keys = list(train_cfg.policy.input_features.keys())
                print(f"[INFO] LeRobot Diffusion expects input keys: {self.lerobot_input_keys}")
                
                # Get actual config values
                self.diffusion_horizon = getattr(train_cfg.policy, 'horizon', diffusion_horizon)
                self.n_action_steps = getattr(train_cfg.policy, 'n_action_steps', n_action_steps)
                self.n_obs_steps = getattr(train_cfg.policy, 'n_obs_steps', n_obs_steps)
                
                print(f"[INFO] Diffusion config: horizon={self.diffusion_horizon}, n_action_steps={self.n_action_steps}, n_obs_steps={self.n_obs_steps}")
                
                # Load policy with dataset stats
                ds_meta = None
                if dataset_root:
                    try:
                        print(f"[INFO] Loading dataset meta from {dataset_root}")
                        ds_meta = LeRobotDatasetMetadata(train_cfg.dataset.repo_id, root=dataset_root, revision=train_cfg.dataset.revision)
                    except Exception as e:
                        print(f"[WARNING] Could not load dataset meta: {e}")
                
                # If ds_meta is still None, try to create a dummy one or fetch from HF
                if ds_meta is None:
                    print("[WARNING] No local dataset meta found. Trying to load from Hugging Face...")
                    try:
                        ds_meta = LeRobotDatasetMetadata(train_cfg.dataset.repo_id)
                    except Exception as e:
                        print(f"[WARNING] Could not load from HF: {e}")
                        print("[WARNING] Creating dummy dataset metadata...")
                        # Create a dummy metadata object
                        class DummyMeta:
                            stats = {}
                            task = "dummy_task"
                        
                        # Populate stats with expected keys
                        input_features = train_cfg.policy.input_features
                        output_features = train_cfg.policy.output_features
                        stats = {}
                        
                        # Input stats
                        for key, feature in input_features.items():
                            shape = feature.shape
                            if 'image' in key:
                                stats[key] = {
                                    'mean': torch.tensor([0.485, 0.456, 0.406]),  # ImageNet mean
                                    'std': torch.tensor([0.229, 0.224, 0.225]),   # ImageNet std
                                    'min': torch.tensor([0.0, 0.0, 0.0]),
                                    'max': torch.tensor([1.0, 1.0, 1.0]),
                                }
                            elif 'state' in key:
                                dim = shape[0]
                                stats[key] = {
                                    'mean': torch.zeros(dim),
                                    'std': torch.ones(dim),
                                    'min': torch.ones(dim) * -10.0,
                                    'max': torch.ones(dim) * 10.0,
                                }
                        
                        # Output stats (action)
                        for key, feature in output_features.items():
                            dim = feature.shape[0]
                            stats[key] = {
                                'mean': torch.zeros(dim),
                                'std': torch.ones(dim),
                                'min': torch.ones(dim) * -1.0,
                                'max': torch.ones(dim) * 1.0,
                            }
                            
                        ds_meta = DummyMeta()
                        ds_meta.stats = stats

                self.lerobot_policy = make_lerobot_policy(cfg=train_cfg.policy, ds_meta=ds_meta)
                self.lerobot_policy.train()  # Set to train mode for RL
                print("[INFO] Successfully loaded LeRobot Diffusion Policy!")
                return
            except Exception as e:
                print(f"[ERROR] Failed to load LeRobot Diffusion policy: {e}")
                import traceback
                traceback.print_exc()
                print("[INFO] Falling back to simple MLP policy...")
        
        # Fallback: Simple MLP policy (not a real diffusion model, just for structure)
        print("[WARNING] Using fallback MLP policy - not a true Diffusion model!")
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Vision encoder (if using cameras)
        if use_vision:
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),  # 6 channels for dual cameras
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            fusion_dim = hidden_dim + 128
        else:
            self.vision_encoder = None
            fusion_dim = hidden_dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action decoder - outputs diffusion_horizon actions
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, diffusion_horizon * action_dim),
        )
    
    def reset_episode(self):
        """Reset observation history and action queue for new episode."""
        self.obs_history = []
        self.action_queue = []
        self.step_count = 0
        if hasattr(self.lerobot_policy, 'reset'):
            self.lerobot_policy.reset()
    
    def _add_to_history(self, state: torch.Tensor, image: Optional[torch.Tensor] = None):
        """Add current observation to history buffer."""
        obs_entry = {
            "state": state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state,
            "image": image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else image,
        }
        self.obs_history.append(obs_entry)
        if len(self.obs_history) > self.n_obs_steps:
            self.obs_history.pop(0)
    
    def _pad_history(self):
        """Pad observation history to n_obs_steps if needed."""
        while len(self.obs_history) < self.n_obs_steps:
            if len(self.obs_history) > 0:
                self.obs_history.insert(0, self.obs_history[0].copy())
            else:
                # Empty history, create zeros
                self.obs_history.append({
                    "state": np.zeros((1, self.state_dim), dtype=np.float32),
                    "image": None,
                })
    
    def encode(self, state: torch.Tensor, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode state and optional image into latent representation (fallback)."""
        state_feat = self.state_encoder(state)
        
        if self.use_vision and image is not None and self.vision_encoder is not None:
            vision_feat = self.vision_encoder(image).flatten(1)
            feat = torch.cat([state_feat, vision_feat], dim=1)
        else:
            feat = state_feat
        
        feat = self.fusion(feat)
        return feat
    
    def forward(self, state: torch.Tensor, image: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate action chunk using Diffusion Policy.
        Uses observation history and generates actions for the full horizon.
        
        Returns: (action_chunk, log_std)
            action_chunk: (B, diffusion_horizon, action_dim)
            log_std: (action_dim,)
        """
        # Add current observation to history
        self._add_to_history(state, image)
        
        # If LeRobot Diffusion policy is loaded, use it
        if self.lerobot_policy is not None:
            # Pad history if needed
            self._pad_history()
            
            # Build batch with temporal stacking
            # Stack states from history: [B, n_obs_steps, state_dim]
            states_list = []
            main_imgs_list = []
            jaw_imgs_list = []
            
            for obs in self.obs_history[-self.n_obs_steps:]:
                states_list.append(obs["state"])
                if obs["image"] is not None:
                    # Image is (1, 6, H, W) - split into main and jaw
                    img = obs["image"]
                    if isinstance(img, np.ndarray):
                        main_imgs_list.append(img[:, :3])  # [1, 3, H, W]
                        if img.shape[1] > 3:
                            jaw_imgs_list.append(img[:, 3:])  # [1, 3, H, W]
            
            # Stack states: [B, n_obs_steps, state_dim]
            state_np = np.stack(states_list, axis=1)  # [1, n_obs_steps, 6]
            state_tensor = torch.from_numpy(state_np).to(self.device_str).float()
            
            # Build batch with separate image keys
            batch = {
                "observation.state": state_tensor,
            }
            
            if len(main_imgs_list) > 0:
                main_np = np.stack(main_imgs_list, axis=1)  # [1, n_obs_steps, 3, H, W]
                main_tensor = torch.from_numpy(main_np).to(self.device_str).float()
                batch["observation.images.front"] = main_tensor
                
            if len(jaw_imgs_list) > 0:
                jaw_np = np.stack(jaw_imgs_list, axis=1)  # [1, n_obs_steps, 3, H, W]
                jaw_tensor = torch.from_numpy(jaw_np).to(self.device_str).float()
                batch["observation.images.jaw"] = jaw_tensor
            
            # Normalize inputs
            batch = self.lerobot_policy.normalize_inputs(batch)
            
            # Stack images after normalization
            imgs = []
            for key in ["observation.images.front", "observation.images.jaw"]:
                if key in batch:
                    img = batch[key]
                    if img.dim() == 4:  # [B, C, H, W] -> [B, 1, C, H, W]
                        img = img.unsqueeze(1)
                    imgs.append(img)
            
            if len(imgs) > 0:
                batch["observation.images"] = torch.stack(imgs, dim=2)  # [B, n_obs_steps, num_cams, C, H, W]
            
            # Generate actions using diffusion (FROZEN - no gradients)
            with torch.no_grad():
                actions = self.lerobot_policy.diffusion.generate_actions(batch)
                actions = self.lerobot_policy.unnormalize_outputs({"action": actions})["action"]
            
            # actions shape: (1, horizon, action_dim)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)
            
            # APPLY ACTION ADAPTER (TRAINABLE!)
            # This learns residual corrections to improve diffusion policy actions
            B, H, A = actions.shape
            state_flat = state.view(B, -1)  # Flatten state
            
            # For each timestep in horizon, compute residual
            actions_adapted = []
            for t in range(H):
                action_t = actions[:, t, :]  # (B, A)
                adapter_input = torch.cat([state_flat, action_t], dim=-1)  # (B, state_dim + action_dim)
                residual = self.action_adapter(adapter_input)  # (B, A)
                adapted = action_t + self.adapter_scale * residual  # Add scaled residual
                actions_adapted.append(adapted)
            
            actions = torch.stack(actions_adapted, dim=1)  # (B, H, A)
            
            # Clamp log_std
            clamped_log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
            
            return actions, clamped_log_std

        # Fallback: Simple MLP policy
        feat = self.encode(state, image)
        feat = feat.unsqueeze(1)
        feat = self.transformer(feat)
        feat = feat.squeeze(1)
        
        actions_flat = self.action_decoder(feat)
        actions = actions_flat.view(-1, self.diffusion_horizon, self.action_dim)
        
        clamped_log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        return actions, clamped_log_std
    
    def get_action(self, state: torch.Tensor, image: Optional[torch.Tensor] = None, deterministic: bool = True) -> torch.Tensor:
        """Get single action from policy with action queue support."""
        # If we have queued actions, use them
        if len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            self.step_count += 1
            return action
        
        # Need to generate new actions
        actions, log_std = self.forward(state, image)
        
        # Queue up actions for n_action_steps
        actions_np = actions.detach().cpu().numpy()
        if actions_np.ndim == 3:
            actions_np = actions_np[0]  # Remove batch dim
        
        for i in range(min(self.n_action_steps, len(actions_np))):
            act = torch.from_numpy(actions_np[i:i+1]).to(state.device).float()
            self.action_queue.append(act)
        
        # Return first action
        action = self.action_queue.pop(0)
        self.step_count += 1
        return action


class ValueNetwork(nn.Module):
    """Value network for PPO critic."""
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int = 256,
        use_vision: bool = False,
    ):
        super().__init__()
        self.use_vision = use_vision
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Vision encoder (if using cameras)
        if use_vision:
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            fusion_dim = hidden_dim + 128
        else:
            self.vision_encoder = None
            fusion_dim = hidden_dim
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state: torch.Tensor, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute state value."""
        # Encode state
        state_feat = self.state_encoder(state)
        
        # Encode image if available
        if self.use_vision and image is not None:
            vision_feat = self.vision_encoder(image).flatten(1)
            feat = torch.cat([state_feat, vision_feat], dim=1)
        else:
            feat = state_feat
        
        # Compute value
        value = self.value_head(feat)
        return value.squeeze(-1)


class RolloutBuffer:
    """Buffer for collecting rollout data with vision support."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, num_envs: int, device: torch.device, 
                 use_vision: bool = False, image_channels: int = 6, image_size: int = 128):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.use_vision = use_vision
        self.ptr = 0
        self.size = 0
        
        # Allocate buffers
        self.states = torch.zeros((capacity, num_envs, state_dim), device=device)
        self.actions = torch.zeros((capacity, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((capacity, num_envs), device=device)
        self.dones = torch.zeros((capacity, num_envs), device=device)
        self.values = torch.zeros((capacity, num_envs), device=device)
        self.log_probs = torch.zeros((capacity, num_envs), device=device)
        
        # Vision buffer (only if vision is enabled)
        # Store on CPU to save GPU memory, will transfer to GPU during training
        if use_vision:
            self.images = torch.zeros((capacity, num_envs, image_channels, image_size, image_size), device='cpu', dtype=torch.float16)  # Use float16 to save memory
        else:
            self.images = None
    
    def add(self, state, action, reward, done, value, log_prob, image=None):
        """Add transition to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        if self.use_vision and image is not None:
            # Store on CPU with half precision to save memory
            self.images[self.ptr] = image.cpu().half() if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float16)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get(self):
        """Get all data from buffer."""
        if self.use_vision:
            return (
                self.states[:self.size],
                self.actions[:self.size],
                self.rewards[:self.size],
                self.dones[:self.size],
                self.values[:self.size],
                self.log_probs[:self.size],
                self.images[:self.size],
            )
        else:
            return (
                self.states[:self.size],
                self.actions[:self.size],
                self.rewards[:self.size],
                self.dones[:self.size],
                self.values[:self.size],
                self.log_probs[:self.size],
                None,
            )
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
    
    returns = advantages + values
    return advantages, returns


def compute_reward_weighted_bc_loss_with_accumulation(
    policy: "DiffusionPolicy",
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    images: Optional[torch.Tensor],
    batch_size: int = 1,  # Process ONE sample at a time for memory
    advantage_clip: float = 10.0,
    positive_only: bool = True,
    device: str = "cuda",
    max_grad_norm: float = 1.0,
    bc_lambda: float = 0.1,
) -> float:
    """
    Compute Reward-Weighted BC loss with gradient accumulation.
    
    This processes samples one at a time, accumulating gradients,
    then does a single optimizer step at the end.
    This is MUCH more memory efficient than storing all losses.
    
    Returns:
        bc_loss_value: The average BC loss value (float, not tensor)
    """
    if policy.lerobot_policy is None:
        return 0.0
    
    # Flatten dimensions
    num_steps, num_envs = states.shape[:2]
    total_samples = num_steps * num_envs
    
    states_flat = states.view(total_samples, -1)
    actions_flat = actions.view(total_samples, -1)
    advantages_flat = advantages.view(total_samples)
    
    if images is not None:
        images_flat = images.view(total_samples, *images.shape[2:])
    else:
        images_flat = None
    
    # Filter for positive advantages only
    if positive_only:
        positive_mask = advantages_flat > 0
        num_positive = positive_mask.sum().item()
        
        if num_positive == 0:
            adv_min = advantages_flat.min().item()
            adv_max = advantages_flat.max().item()
            adv_mean = advantages_flat.mean().item()
            print(f"[BC DEBUG] No positive advantages! min={adv_min:.4f}, max={adv_max:.4f}, mean={adv_mean:.4f}")
            positive_only = False
        else:
            positive_mask_cpu = positive_mask.cpu()
            states_flat = states_flat[positive_mask]
            actions_flat = actions_flat[positive_mask]
            advantages_flat = advantages_flat[positive_mask]
            if images_flat is not None:
                images_flat = images_flat[positive_mask_cpu]
            total_samples = states_flat.shape[0]
    
    if total_samples == 0:
        return 0.0
    
    # Clip and normalize advantages for weighting
    advantages_clipped = torch.clamp(advantages_flat, -advantage_clip, advantage_clip)
    weights = torch.softmax(advantages_clipped, dim=0) * total_samples
    
    # Zero gradients before accumulation
    optimizer.zero_grad()
    
    # Track loss values for logging
    total_loss = 0.0
    num_batches = 0
    
    # Limit number of samples to process (for memory)
    # CRITICAL: For 7.66 GiB GPU with vision, process only 1 sample at a time
    # Even 1 sample may OOM - will handle gracefully by skipping images
    max_samples = min(total_samples, 1)  # VERY conservative
    sample_indices = torch.randperm(total_samples)[:max_samples]
    
    # Global flag to skip images if we encounter OOM
    skip_images_due_to_oom = False
    
    # CRITICAL: MUST use exact same parameters as policy was trained with
    # Diffusion policy will fail with AssertionError if horizon doesn't match config
    n_obs_steps = policy.n_obs_steps  # MUST match policy config
    horizon = policy.diffusion_horizon  # MUST match policy config (e.g., 64)
    
    # CRITICAL: Freeze vision encoder to reduce backward memory by 80%!
    # Vision encoder is the biggest memory hog during backward pass
    if hasattr(policy.lerobot_policy, 'diffusion') and hasattr(policy.lerobot_policy.diffusion, 'rgb_encoder'):
        for param in policy.lerobot_policy.diffusion.rgb_encoder.parameters():
            param.requires_grad = False
        print("  [BC] Froze vision encoder for memory efficiency")
    
    for idx in sample_indices:
        try:
            idx = idx.item()
            
            # Get single sample
            batch_state = states_flat[idx:idx+1].to(device)
            batch_action = actions_flat[idx:idx+1].to(device)
            weight = weights[idx:idx+1].to(device)
            
            # Transfer images from CPU to GPU for BC loss
            if images_flat is not None:
                batch_image = images_flat[idx:idx+1].to(device=device, dtype=torch.float32)
            else:
                batch_image = None
            
            # Build batch for diffusion model
            batch_state_temporal = batch_state.unsqueeze(1).repeat(1, n_obs_steps, 1)
            
            batch = {"observation.state": batch_state_temporal}
            
            if batch_image is not None and batch_image.shape[1] == 6:
                # Use original image size - NO downsampling
                front = batch_image[:, :3].unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)
                jaw = batch_image[:, 3:].unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)
                batch["observation.images.front"] = front
                batch["observation.images.jaw"] = jaw
            
            # Normalize inputs
            batch = policy.lerobot_policy.normalize_inputs(batch)
            
            # Stack images
            if "observation.images.front" in batch:
                imgs = []
                for key in ["observation.images.front", "observation.images.jaw"]:
                    if key in batch:
                        img = batch[key]
                        if img.dim() == 4:
                            img = img.unsqueeze(1)
                        imgs.append(img)
                if len(imgs) > 0:
                    batch["observation.images"] = torch.stack(imgs, dim=2)
            
            # Prepare target actions
            target_actions = batch_action.unsqueeze(1).repeat(1, horizon, 1)
            target_actions_normalized = policy.lerobot_policy.normalize_targets({"action": target_actions})["action"]
            
            batch["action"] = target_actions_normalized
            batch["action_is_pad"] = torch.zeros(
                (1, horizon), dtype=torch.bool, device=device
            )
            
            # Forward pass with gradient checkpointing to save memory
            # This trades compute for memory by not storing intermediate activations
            # Saves 50-70% GPU memory at cost of 20-30% slower training
            try:
                # Try to use gradient checkpointing if available
                from torch.utils.checkpoint import checkpoint
                # Wrap the forward pass in checkpoint
                output = checkpoint(policy.lerobot_policy.forward, batch, use_reentrant=False)
            except:
                # Fallback to normal forward if checkpointing fails
                output = policy.lerobot_policy.forward(batch)
            
            # Extract loss
            if isinstance(output, tuple) and len(output) > 0:
                loss_value = output[0]
            elif isinstance(output, dict) and 'loss' in output:
                loss_value = output['loss']
            elif isinstance(output, torch.Tensor):
                loss_value = output
            else:
                continue
            
            # Weight the loss and scale for accumulation
            weighted_loss = loss_value * weight.mean() * bc_lambda / max_samples
            
            # Backward with accumulation (no optimizer step yet)
            weighted_loss.backward()
            
            total_loss += loss_value.item()
            num_batches += 1
            
            # Clear computation graph immediately
            del batch, output, loss_value, weighted_loss
            del batch_state, batch_action, weight
            if batch_image is not None:
                del batch_image
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"[WARNING] OOM in BC batch {idx}, skipping...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[ERROR] BC batch failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Unfreeze vision encoder after BC loss (for inference)
    if hasattr(policy.lerobot_policy, 'diffusion') and hasattr(policy.lerobot_policy.diffusion, 'rgb_encoder'):
        for param in policy.lerobot_policy.diffusion.rgb_encoder.parameters():
            param.requires_grad = True
    
    if num_batches == 0:
        optimizer.zero_grad()
        return 0.0
    
    # Clip gradients and step optimizer
    # NaN/Inf checking is DISABLED - it causes OOM on memory-limited GPUs
    # clip_grad_norm_ will handle any NaN/Inf gracefully
    try:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        avg_loss = total_loss / num_batches
        
        # Print status
        print(f"  📚 BC Loss: {avg_loss:.6f} (λ={bc_lambda}, samples={num_batches}, mode=vision-frozen)")
        return avg_loss
    except Exception as e:
        print(f"  [WARNING] BC optimizer step failed: {e}")
        optimizer.zero_grad()
        return 0.0


def train_ppo_epoch(
    policy: DiffusionPolicy,
    value_net: ValueNetwork,
    optimizer_policy: optim.Optimizer,
    optimizer_value: optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    batch_size: int,
    images: Optional[torch.Tensor] = None,
    # Reward-Weighted BC loss parameters
    bc_lambda: float = 0.0,
    bc_batch_size: int = 4,
    advantage_clip: float = 10.0,
    positive_advantage_only: bool = True,
) -> Dict[str, float]:
    """Perform one PPO update epoch with Reward-Weighted BC finetuning for Diffusion Policy."""
    device = states.device
    # Flatten batch dimensions
    num_steps, num_envs = states.shape[:2]
    total_samples = num_steps * num_envs
    
    # Save original shapes for BC loss computation later
    original_state_shape = states.shape
    original_action_shape = actions.shape
    original_image_shape = images.shape if images is not None else None
    
    states = states.view(total_samples, -1)
    actions = actions.view(total_samples, -1)
    old_log_probs = old_log_probs.view(total_samples)
    advantages_orig = advantages.clone()  # Keep original for BC loss
    advantages = advantages.view(total_samples)
    returns = returns.view(total_samples)
    
    # Keep images on CPU (float16) - will transfer to GPU per mini-batch
    if images is not None:
        # images: (num_steps, num_envs, C, H, W) -> (total_samples, C, H, W)
        # Keep on CPU as float16 to save memory
        images = images.view(total_samples, *images.shape[2:])
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Training metrics
    policy_losses = []
    value_losses = []
    entropy_losses = []
    
    # Mini-batch updates
    indices = torch.randperm(total_samples)
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        batch_indices = indices[start:end]
        
        # Skip incomplete batches (drop_last=True)
        current_batch_size = end - start
        if current_batch_size < batch_size:
            continue
        
        try:
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            # Transfer images to GPU only for this mini-batch (and convert to float32)
            if images is not None:
                batch_images = images[batch_indices].to(device=states.device, dtype=torch.float32)
            else:
                batch_images = None
        
            # ECONOMIC RL: Train only log_std (exploration control) + value network
            # Diffusion policy is FROZEN - we don't have enough GPU memory to train it
            # Instead, we optimize exploration to find good actions
            
            # Use stored actions as predictions (diffusion model output is deterministic)
            action_pred = batch_actions
            
            # Get log_std from policy (this is what we're training for exploration)
            log_std = policy.log_std
            log_std_clamped = torch.clamp(log_std, min=-20.0, max=2.0)
            std = torch.exp(log_std_clamped) + 1e-8
            
            dist = torch.distributions.Normal(action_pred, std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            new_log_probs = torch.clamp(new_log_probs, min=-100.0, max=100.0)
            batch_old_log_probs = torch.clamp(batch_old_log_probs, min=-100.0, max=100.0)
            
            # Compute entropy for exploration
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Simple policy gradient loss (REINFORCE-style)
            # This encourages actions that lead to higher advantages
            log_ratio = torch.clamp(new_log_probs - batch_old_log_probs, min=-20.0, max=20.0)
            ratio = torch.clamp(torch.exp(log_ratio), min=1e-8, max=100.0)
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            values_pred = value_net(batch_states, batch_images)
            value_loss = nn.functional.mse_loss(values_pred, batch_returns)
            
            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Check for NaN loss and skip update if found
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf loss detected, skipping this batch")
                continue
            
            # Update policy
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in policy.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"[WARNING] NaN/Inf gradients detected, skipping this batch")
                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                continue
            
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            optimizer_policy.step()
            optimizer_value.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
            
            # Clear mini-batch tensors immediately to free GPU memory
            try:
                del batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns
                if batch_images is not None:
                    del batch_images
                del new_log_probs, entropy, ratio, surr1, surr2, policy_loss, value_loss, loss
                del values_pred
            except Exception:
                pass  # Some variables might not exist, that's ok
        
        except torch.cuda.OutOfMemoryError:
            print(f"[WARNING] OOM in PPO mini-batch, skipping and cleaning up...")
            torch.cuda.empty_cache()
            continue
        
        # Force GPU memory cleanup every batch
        torch.cuda.empty_cache()
    
    # =====================================================
    # REWARD-WEIGHTED BC LOSS FOR DIFFUSION MODEL TRAINING  
    # =====================================================
    # This is the key learning signal for the diffusion model
    # PPO above only trains log_std, BC loss trains the actual model
    bc_loss_value = 0.0
    if bc_lambda > 0 and policy.lerobot_policy is not None:
        # CRITICAL: Aggressive memory cleanup before BC loss
        # Move value network to CPU temporarily to free GPU memory
        print("[BC] Preparing GPU memory...")
        value_net_device = next(value_net.parameters()).device
        value_net.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Print available memory
        mem_free = (torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**2
        print(f"[BC] Free GPU memory: {mem_free:.0f} MiB")
        
        try:
            # Reshape back to original format for BC loss
            states_bc = states.view(num_steps, num_envs, -1)
            actions_bc = actions.view(num_steps, num_envs, -1)
            # Use the original (non-normalized) advantages that we saved
            advantages_bc = advantages_orig  # Already in (num_steps, num_envs) shape
            
            images_bc = None
            if images is not None:
                images_bc = images.view(num_steps, num_envs, *images.shape[1:])
            
            # Use gradient accumulation version - handles backward internally
            bc_loss_value = compute_reward_weighted_bc_loss_with_accumulation(
                policy=policy,
                optimizer=optimizer_policy,
                states=states_bc,
                actions=actions_bc,
                advantages=advantages_bc,
                images=images_bc,
                batch_size=1,  # Process 1 sample at a time
                advantage_clip=advantage_clip,
                positive_only=positive_advantage_only,
                device=device,
                max_grad_norm=max_grad_norm,
                bc_lambda=bc_lambda,
            )
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [WARNING] BC loss OOM - skipping BC update this epoch")
            bc_loss_value = 0.0
        except Exception as e:
            print(f"  [WARNING] BC loss computation failed: {e}")
            bc_loss_value = 0.0
        finally:
            # Move value network back to GPU
            value_net.to(value_net_device)
            # Final cleanup after BC loss
            torch.cuda.empty_cache()
    
    return {
        'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
        'value_loss': np.mean(value_losses) if value_losses else 0.0,
        'entropy': np.mean(entropy_losses) if entropy_losses else 0.0,
        'bc_loss': bc_loss_value,
    }


def main():
    """Train Diffusion Policy with online RL (PPO)."""
    print(f"[INFO] Training task: {args_cli.task}")
    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Diffusion horizon: {args_cli.diffusion_horizon}, n_action_steps: {args_cli.n_action_steps}, n_obs_steps: {args_cli.n_obs_steps}")
    print(f"[INFO] Use vision: {args_cli.use_vision}")
    
    # Set seed before creating environment
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    
    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    
    # Set seed in environment config if possible
    if hasattr(env_cfg, 'seed'):
        env_cfg.seed = args_cli.seed
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Get actual number of environments from the created environment
    # This is important because Isaac Lab may override the num_envs from config
    actual_num_envs = env.unwrapped.num_envs if hasattr(env.unwrapped, 'num_envs') else args_cli.num_envs
    print(f"[INFO] Actual number of environments from Isaac Lab: {actual_num_envs}")
    
    # For training, we use only 1 env (slice data from first env)
    # This simplifies training and avoids dimension mismatches
    training_num_envs = 1
    print(f"[INFO] Training with num_envs: {training_num_envs}")
    
    # Get dimensions BEFORE reset to avoid CUDA errors
    obs_space = env.observation_space['policy']
    action_space = env.action_space
    
    if isinstance(obs_space, gym.spaces.Dict):
        # Use joint_pos dimension - handle multi-dimensional shapes
        joint_pos_space = obs_space['joint_pos']
        print(f"[DEBUG] joint_pos_space type: {type(joint_pos_space)}")
        print(f"[DEBUG] joint_pos_space shape: {joint_pos_space.shape if hasattr(joint_pos_space, 'shape') else 'NO SHAPE'}")
        
        if hasattr(joint_pos_space, 'shape'):
            # Isaac Lab observation spaces can be (num_envs, state_dim) or (state_dim,)
            # We always want the STATE dimension, not the environment dimension
            if len(joint_pos_space.shape) > 1:
                # Shape is (num_envs, state_dim) -> use FIRST dimension if num_envs=1, else LAST
                if joint_pos_space.shape[0] > joint_pos_space.shape[1]:
                    # More envs than state_dim -> (num_envs, state_dim)
                    state_dim = joint_pos_space.shape[-1]
                else:
                    # More state_dim than envs -> (state_dim, num_envs) OR (num_envs=1, state_dim)
                    # For SO-100: expect (1, 6) -> want 6
                    state_dim = max(joint_pos_space.shape)  # Take the larger dimension
                print(f"[DEBUG] Multi-dim shape detected, using state_dim={state_dim}")
            else:
                state_dim = joint_pos_space.shape[0]
                print(f"[DEBUG] Single-dim shape detected, using state_dim={state_dim}")
        else:
            state_dim = 6  # Default fallback
            print(f"[DEBUG] No shape attribute, using default state_dim=6")
    elif hasattr(obs_space, 'shape'):
        # Handle non-dict observation spaces
        if len(obs_space.shape) > 1:
            state_dim = obs_space.shape[-1]
        else:
            state_dim = obs_space.shape[0]
    else:
        # Fallback for complex spaces
        print(f"[WARNING] Could not determine state_dim from obs_space type: {type(obs_space)}")
        state_dim = 6  # Default for SO-100 robot
    
    if hasattr(action_space, 'shape'):
        # Handle multi-dimensional action spaces
        if len(action_space.shape) > 1:
            action_dim = action_space.shape[-1]
        else:
            action_dim = action_space.shape[0]
    elif hasattr(action_space, 'n'):
        action_dim = action_space.n
    else:
        print(f"[WARNING] Could not determine action_dim from action_space type: {type(action_space)}")
        action_dim = 6  # Default for SO-100 robot
    
    print(f"[INFO] Observation space: {obs_space}")
    print(f"[INFO] Action space: {action_space}")
    print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create device
    device = torch.device(args_cli.device)
    
    # Create Diffusion Policy network
    policy = DiffusionPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        diffusion_horizon=args_cli.diffusion_horizon,
        n_action_steps=args_cli.n_action_steps,
        n_obs_steps=args_cli.n_obs_steps,
        hidden_dim=args_cli.hidden_dim,
        num_layers=args_cli.num_layers,
        num_heads=args_cli.num_heads,
        use_vision=args_cli.use_vision,
        image_size=args_cli.image_size,
        pretrained_path=args_cli.pretrained,
        dataset_root=args_cli.dataset_root,
        device=args_cli.device,
    ).to(device)
    
    # CRITICAL: Aggressive memory cleanup before creating value network and optimizers
    print("[INFO] Cleaning GPU memory before optimizer initialization...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Print memory usage after loading diffusion policy
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        mem_free = (torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / 1024**3
        print(f"[MEMORY] After Diffusion Policy load:")
        print(f"  Allocated: {mem_allocated:.2f} GiB")
        print(f"  Reserved: {mem_reserved:.2f} GiB")
        print(f"  Free: {mem_free:.2f} GiB")
    
    # IMPORTANT: Value network uses STATE ONLY (no vision) to save GPU memory
    # The diffusion policy handles vision internally, value net only needs state for estimation
    value_net = ValueNetwork(
        state_dim=state_dim,
        hidden_dim=args_cli.hidden_dim,
        use_vision=False,  # Disable vision for value net to save memory
    ).to(device)
    
    torch.cuda.empty_cache()
    
    # Create optimizers - ONLY train action_adapter and log_std (memory efficient!)
    # Diffusion policy is frozen, we only train the small residual adapter
    trainable_params = [
        {'params': policy.action_adapter.parameters(), 'lr': args_cli.learning_rate},
        {'params': [policy.log_std], 'lr': args_cli.learning_rate * 0.1},  # Lower LR for log_std
        {'params': [policy.adapter_scale], 'lr': args_cli.learning_rate * 0.01},  # Even lower for scale
    ]
    optimizer_policy = optim.Adam(trainable_params)
    optimizer_value = optim.Adam(value_net.parameters(), lr=args_cli.learning_rate)
    
    # Count trainable parameters
    adapter_params = sum(p.numel() for p in policy.action_adapter.parameters())
    print(f"[INFO] Trainable parameters: {adapter_params + 1 + policy.action_dim} (adapter: {adapter_params}, log_std: {policy.action_dim}, scale: 1)")
    print(f"[INFO] This is ~{adapter_params * 4 / 1024:.1f} KB - very memory efficient!")
    
    # Skip optimizer state initialization to save memory on low-memory GPUs (< 8GB)
    # State will be initialized lazily during first optimizer.step() in BC loss
    # This saves ~80-100 MiB of GPU memory at startup
    print("[INFO] Optimizer state will be initialized lazily during first BC loss (memory-efficient)")
    torch.cuda.empty_cache()
    
    # Create buffer - ENABLE vision storage for Reward-Weighted BC finetuning
    # Images are needed to compute BC loss with the diffusion model
    buffer = RolloutBuffer(
        capacity=args_cli.horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=training_num_envs,  # Always 1 for single-env training
        device=device,
        use_vision=args_cli.use_vision and args_cli.bc_lambda > 0,  # Store images only if BC loss is enabled
        image_channels=6,  # 2 cameras x 3 channels
        image_size=args_cli.image_size,
    )
    
    # Create checkpoint directory
    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)
    
    # If pretrained loaded, save it immediately as "last_checkpoint.pt"
    # This ensures we start exactly from a saved state in the desired format
    if args_cli.pretrained:
        initial_ckpt_path = os.path.join(args_cli.checkpoint_dir, "last_checkpoint.pt")
        print(f"[INFO] Saving initial model state to: {initial_ckpt_path}")
        torch.save({
            "state_dict": policy.state_dict(),
            "optimizer": optimizer_policy.state_dict(),  # Use policy optimizer state
            "tag": "initial_pretrained",
        }, initial_ckpt_path)
        
        # NOW LOAD IT BACK TO START TRAINING (Double check)
        print(f"[INFO] Reloading from initial checkpoint to start training...")
        checkpoint = torch.load(initial_ckpt_path, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer_policy.load_state_dict(checkpoint["optimizer"])
        start_iter = 0
        print(f"[INFO] Ready to start training from iter {start_iter}")

    # Load checkpoint if resuming (overrides initial pretrained)
    if args_cli.resume:
        resume_path = args_cli.resume
        
        # Handle --resume auto: automatically find latest checkpoint
        if resume_path.lower() == "auto":
            latest_path = os.path.join(args_cli.checkpoint_dir, "latest.pt")
            if os.path.exists(latest_path):
                resume_path = latest_path
                print(f"[INFO] Auto-resume: Found checkpoint at {resume_path}")
            else:
                print(f"[WARNING] Auto-resume: No checkpoint found at {latest_path}, starting fresh")
                resume_path = None
        
        if resume_path and os.path.exists(resume_path):
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
            
            # Clear GPU memory before loading checkpoint
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Load checkpoint to CPU first to avoid OOM, then load state dicts
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            
            # Handle both formats (RL checkpoint vs Pretrained-style)
            if 'policy' in checkpoint: # Standard RL checkpoint
                policy.load_state_dict(checkpoint['policy'])
                if 'value_net' in checkpoint: value_net.load_state_dict(checkpoint['value_net'])
                if 'optimizer_policy' in checkpoint: optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
                if 'optimizer_value' in checkpoint: optimizer_value.load_state_dict(checkpoint['optimizer_value'])
                start_iter = checkpoint.get('iteration', 0) + 1
            elif 'state_dict' in checkpoint: # Pretrained-style checkpoint
                policy.load_state_dict(checkpoint['state_dict'])
                if 'optimizer' in checkpoint: optimizer_policy.load_state_dict(checkpoint['optimizer'])
                start_iter = 0 # Reset iteration count for new phase
            
            # Free checkpoint from memory
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
                
            print(f"[INFO] Resumed from iteration {start_iter}")
        elif resume_path:
            print(f"[WARNING] Resume checkpoint not found: {resume_path}, starting fresh")
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    success_count = 0
    total_episodes = 0
    
    # Print training configuration
    print("\n" + "="*80)
    print("🚀 STARTING PPO TRAINING WITH DIFFUSION POLICY")
    print("="*80)
    print(f"\n📋 CONFIGURATION:")
    print(f"   Task: {args_cli.task}")
    print(f"   Device: {device}")
    print(f"   Num Envs (requested): {args_cli.num_envs}")
    print(f"   Num Envs (actual): {actual_num_envs}")
    print(f"   Max Iterations: {args_cli.max_iterations}")
    print(f"   Horizon (PPO Update): {args_cli.horizon} steps")
    if args_cli.max_episode_steps:
        print(f"   Max Episode Steps: {args_cli.max_episode_steps} (forced reset)")
    else:
        print(f"   Max Episode Steps: No limit (continues until env termination)")
    print(f"   Batch Size: {args_cli.batch_size}")
    print(f"\n🎓 LEARNING:")
    print(f"   Learning Rate: {args_cli.learning_rate}")
    print(f"   Gamma: {args_cli.gamma}")
    print(f"   GAE Lambda: {args_cli.gae_lambda}")
    print(f"   Clip Epsilon: {args_cli.clip_epsilon}")
    print(f"   PPO Epochs: {args_cli.ppo_epochs}")
    print(f"   Entropy Coef: {args_cli.entropy_coef}")
    print(f"   Value Coef: {args_cli.value_coef}")
    print(f"\n🎬 DIFFUSION POLICY:")
    print(f"   Diffusion Horizon: {args_cli.diffusion_horizon}")
    print(f"   N Action Steps: {args_cli.n_action_steps}")
    print(f"   N Obs Steps: {args_cli.n_obs_steps}")
    print(f"   State Dim: {state_dim}")
    print(f"   Action Dim: {action_dim}")
    print(f"   Use Vision: {args_cli.use_vision}")
    if args_cli.use_vision:
        print(f"   Image Size: {args_cli.image_size}x{args_cli.image_size}")
        print(f"   Camera Source: {getattr(args_cli, 'camera_source', 'sensor')}")
        print(f"   Camera Format: {getattr(args_cli, 'camera_format', 'rgb')}")
    print(f"\n📚 REWARD-WEIGHTED BC FINETUNING:")
    print(f"   BC Lambda (λ): {args_cli.bc_lambda}")
    print(f"   BC Batch Size: {args_cli.bc_batch_size}")
    print(f"   Advantage Clip: {args_cli.advantage_clip}")
    print(f"   Positive Advantage Only: {args_cli.positive_advantage_only}")
    if args_cli.bc_lambda > 0:
        print(f"   Status: ✅ ENABLED - Diffusion model will be trained!")
    else:
        print(f"   Status: ❌ DISABLED - Only exploration parameters trained")
    if args_cli.pretrained:
        print(f"\n🔄 PRETRAINED MODEL:")
        print(f"   Path: {args_cli.pretrained}")
        print(f"   Dataset Root: {args_cli.dataset_root}")
    if args_cli.action_map != "none":
        print(f"\n🎯 ACTION MAPPING:")
        print(f"   Mode: {args_cli.action_map}")
        if args_cli.arm_scales:
            print(f"   Arm Scales: {args_cli.arm_scales}")
        if args_cli.axis_signs:
            print(f"   Axis Signs: {args_cli.axis_signs}")
    print(f"\n💾 CHECKPOINTS:")
    print(f"   Directory: {args_cli.checkpoint_dir}")
    print(f"   Save Interval: {args_cli.save_interval} iterations")
    print(f"   Log Interval: {args_cli.log_interval} iterations")
    print("="*80 + "\n")
    
    # Training loop
    print("[INFO] Initializing environment...")
    try:
        # Try reset with seed first
        obs_dict, _ = env.reset(seed=args_cli.seed)
        policy.reset_episode()  # Reset Diffusion Policy observation history and action queue
    except TypeError:
        # Fallback if seed parameter not supported
        obs_dict, _ = env.reset()
        policy.reset_episode()
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"[ERROR] CUDA error during reset: {e}")
            print("[FIX] Try reducing num_envs or checking GPU memory")
            print("[FIX] You can also try: CUDA_LAUNCH_BLOCKING=1 python ...")
            raise
        else:
            raise
    
    dump_done = False  # track whether we've already dumped debug inputs
    
    # Initialize exploration noise
    current_exploration_noise = args_cli.exploration_noise
    print(f"[INFO] Exploration noise: {current_exploration_noise} (decay: {args_cli.exploration_decay}, min: {args_cli.exploration_min})")
    
    # Initialize jaw-payload distance tracking (for distance-based reward)
    previous_jaw_payload_distance = None
    
    # Track learning progress
    prev_avg_reward = 0.0
    best_reward = -float('inf')
    reward_history = []
    
    for iteration in range(start_iter, args_cli.max_iterations):
        iter_start_time = time.time()
        
        # Collect rollout
        buffer.clear()
        episode_reward_sum = torch.zeros(training_num_envs, device=device)
        episode_length = torch.zeros(training_num_envs, dtype=torch.int32, device=device)
        
        for step in range(args_cli.horizon):
            # Periodic GPU memory cleanup (every 10 steps)
            if step % 10 == 0 and step > 0:
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            
            # Extract state and images
            obs = obs_dict["policy"]
            if isinstance(obs, dict):
                state = obs['joint_pos']  # Shape: [num_envs, state_dim]
                # For single-env training, ensure batch dimension is 1
                if state.shape[0] != 1:
                    state = state[:1]  # Take only first env
                # Extract images if vision is enabled
                if args_cli.use_vision:
                    main_cam = None
                    jaw_cam = None
                    
                    # VIEWPORT CAPTURE (exactly like test_ACT.py LeRobotAdapter)
                    # If camera_source is 'viewport', capture directly from GUI (sRGB, Tone Mapped)
                    if args_cli.camera_source == 'viewport':
                        try:
                            import omni.kit.viewport.utility as vp_utils
                            vp = vp_utils.get_active_viewport_window()
                            if vp:
                                tex = vp.viewport_api.get_texture()
                                if tex:
                                    img = tex.get_rgba_image()
                                    # img is HxWx4 uint8 RGBA; drop A
                                    # This is sRGB (Tone Mapped) - exactly like test_ACT.py
                                    main_cam = img[..., :3]
                        except Exception:
                            main_cam = None
                    
                    # Fallback to sensor if viewport failed or not requested
                    if main_cam is None:
                        try:
                            cam_t = env.unwrapped.scene["camera"].data.output["rgb"]
                            if isinstance(cam_t, torch.Tensor):
                                main_cam = cam_t.detach().cpu().numpy()
                            else:
                                main_cam = np.array(cam_t)
                            # Camera sensors return [num_envs, H, W, C] - take first env only
                            if main_cam.ndim == 4:
                                main_cam = main_cam[0]
                        except Exception:
                            main_cam = obs.get('camera_rgb', None) if isinstance(obs, dict) else None
                    
                    # Extract jaw camera (always from sensor - viewport only shows main view)
                    try:
                        jaw_cam_t = env.unwrapped.scene["jaw_camera"].data.output["rgb"]
                        if isinstance(jaw_cam_t, torch.Tensor):
                            jaw_cam = jaw_cam_t.detach().cpu().numpy()
                        else:
                            jaw_cam = np.array(jaw_cam_t)
                        # Camera sensors return [num_envs, H, W, C] - take first env only
                        if jaw_cam.ndim == 4:
                            jaw_cam = jaw_cam[0]
                    except Exception:
                        jaw_cam = obs.get('jaw_camera_rgb', None) if isinstance(obs, dict) else None
                    
                    # Process cameras using the same style as test_ACT.py (SmolVLA pipeline)
                    if main_cam is not None and jaw_cam is not None:

                        # --- RAW DUMP FOR DEBUGGING ---
                        if args_cli.dump_input and not dump_done:
                            # Compute global step
                            global_step = iteration * args_cli.horizon + step
                            should_dump = (
                                (args_cli.dump_step is None and iteration == 0 and step == 0)
                                or (args_cli.dump_step is not None and global_step == args_cli.dump_step)
                            )
                            if should_dump:
                                os.makedirs(args_cli.dump_dir, exist_ok=True)
                                print(f"[DEBUG] Dumping RAW cameras at step {global_step}")
                                
                                for name, raw_img in [("front", main_cam), ("jaw", jaw_cam)]:
                                    # Convert to numpy
                                    if isinstance(raw_img, torch.Tensor):
                                        raw_np = raw_img.detach().cpu().numpy()
                                    else:
                                        raw_np = np.array(raw_img)
                                    
                                    # Handle dimensions: (B, H, W, C) -> (H, W, C)
                                    if raw_np.ndim == 4:
                                        raw_np = raw_np[0]
                                    
                                    # Save NPY (Truth)
                                    np.save(os.path.join(args_cli.dump_dir, f"raw_{name}_step{global_step:05d}.npy"), raw_np)
                                    
                                    # Convert to uint8 for PNG
                                    if raw_np.dtype != np.uint8:
                                        raw_u8 = (np.clip(raw_np, 0, 1) * 255).astype(np.uint8)
                                    else:
                                        raw_u8 = raw_np
                                    
                                    # Save assuming Input was RGB (standard) -> Convert to BGR for OpenCV
                                    if raw_u8.shape[-1] == 3:
                                        bgr = cv2.cvtColor(raw_u8, cv2.COLOR_RGB2BGR)
                                        cv2.imwrite(os.path.join(args_cli.dump_dir, f"raw_{name}_step{global_step:05d}_assumeRGB.png"), bgr)
                                        # Save assuming Input was BGR -> Save directly
                                        cv2.imwrite(os.path.join(args_cli.dump_dir, f"raw_{name}_step{global_step:05d}_assumeBGR.png"), raw_u8)
                                    elif raw_u8.shape[-1] == 4: # RGBA
                                        bgr = cv2.cvtColor(raw_u8, cv2.COLOR_RGBA2BGR)
                                        cv2.imwrite(os.path.join(args_cli.dump_dir, f"raw_{name}_step{global_step:05d}_assumeRGBA.png"), bgr)

                        cam_chw = _preprocess_single_cam_for_act(
                            main_cam,
                            image_size=args_cli.image_size,
                            camera_format=getattr(args_cli, "camera_format", "rgb"),
                        )
                        jaw_chw = _preprocess_single_cam_for_act(
                            jaw_cam,
                            image_size=args_cli.image_size,
                            camera_format=getattr(args_cli, "camera_format", "rgb"),
                        )
                        # Concatenate: (3, H, W) + (3, H, W) -> (6, H, W)
                        combined_np = np.concatenate([cam_chw, jaw_chw], axis=0)
                        image = torch.from_numpy(combined_np).float().unsqueeze(0).to(device)  # (1, 6, H, W)

                        # Optionally dump the exact model inputs for debugging (once)
                        if args_cli.dump_input and not dump_done:
                            # Compute a global step index similar to eval script semantics
                            global_step = iteration * args_cli.horizon + step
                            should_dump = (
                                (args_cli.dump_step is None and iteration == 0 and step == 0)
                                or (args_cli.dump_step is not None and global_step == args_cli.dump_step)
                            )
                            if should_dump:
                                os.makedirs(args_cli.dump_dir, exist_ok=True)
                                # Save front camera (convert back to HWC, [0,255] uint8, RGB -> BGR for cv2)
                                front_hwc = np.transpose(cam_chw, (1, 2, 0))
                                front_img = np.clip(front_hwc * 255.0, 0, 255).astype(np.uint8)
                                front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                                front_path = os.path.join(args_cli.dump_dir, f"train_step{global_step:05d}_front.png")
                                cv2.imwrite(front_path, front_bgr)
                                
                                # Save jaw camera
                                jaw_hwc = np.transpose(jaw_chw, (1, 2, 0))
                                jaw_img = np.clip(jaw_hwc * 255.0, 0, 255).astype(np.uint8)
                                jaw_bgr = cv2.cvtColor(jaw_img, cv2.COLOR_RGB2BGR)
                                jaw_path = os.path.join(args_cli.dump_dir, f"train_step{global_step:05d}_jaw.png")
                                cv2.imwrite(jaw_path, jaw_bgr)

                                # Also save NPY tensors for exact model inputs
                                np.save(os.path.join(args_cli.dump_dir, f"train_step{global_step:05d}_front.npy"), cam_chw)
                                np.save(os.path.join(args_cli.dump_dir, f"train_step{global_step:05d}_jaw.npy"), jaw_chw)
                                np.save(os.path.join(args_cli.dump_dir, f"train_step{global_step:05d}_combined.npy"), combined_np)

                                print(f"[DUMP] Saved model input images to {args_cli.dump_dir} at global_step={global_step}")
                                dump_done = True
                    else:
                        image = None
                else:
                    image = None
            else:
                state = obs
                image = None
            
            # Get action from policy using action queue mechanism
            # This only runs diffusion model every n_action_steps, not every step
            with torch.no_grad():
                # get_action() uses action queue - runs diffusion only when queue is empty
                action = policy.get_action(state, image)
                
                # Ensure correct shape
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                
                # Add exploration noise if enabled
                if current_exploration_noise > 0:
                    # Add Gaussian noise for exploration
                    noise = torch.randn_like(action) * current_exploration_noise
                    action_sampled = action + noise
                    # Clip to reasonable range (assuming normalized actions in [-1, 1])
                    action_sampled = torch.clamp(action_sampled, -1.0, 1.0)
                else:
                    # Use deterministic action (no noise)
                    action_sampled = action
                
                # Compute log probability (still needed for PPO) with numerical stability
                log_std = policy.log_std
                log_std_clamped = torch.clamp(log_std, min=-20.0, max=2.0)
                std = torch.exp(log_std_clamped) + 1e-8
                dist = torch.distributions.Normal(action, std)
                log_prob = dist.log_prob(action_sampled).sum(dim=-1)
                # Clamp log_prob to prevent extreme values that cause NaN in PPO
                log_prob = torch.clamp(log_prob, min=-100.0, max=100.0)
                
                # Get value (state only, no image for memory efficiency)
                value = value_net(state, None)
            
            # --- ACTION MAPPING LOGIC (Matched to test_ACT.py) ---
            # Convert to numpy for mapping
            if isinstance(action_sampled, torch.Tensor):
                action_np = action_sampled.detach().cpu().numpy()
            else:
                action_np = np.array(action_sampled, dtype=np.float32)
            
            # Ensure shape (B, A)
            if action_np.ndim == 1:
                action_np = action_np.reshape(1, -1)
                
            # Map actions to environment normalized space if requested
            try:
                if args_cli.action_map != "none":
                    # Extract current joint positions
                    curr = None
                    if args_cli.curr_from_env:
                        try:
                            # Use env-provided absolute joint positions (from FIRST env)
                            # Note: This assumes homogenous environments or uses env 0 for ref
                            jp = env.unwrapped.scene["robot"].data.joint_pos
                            curr = np.array(jp[:action_np.shape[0], :5].cpu().numpy(), dtype=np.float32)
                        except Exception:
                            pass
                    else:
                        # From observation
                        # obs['joint_pos'] is (B, 6)
                        if isinstance(obs, dict) and 'joint_pos' in obs:
                            st = obs['joint_pos']
                            if isinstance(st, torch.Tensor):
                                curr = st.detach().cpu().numpy()
                            else:
                                curr = np.array(st)
                            curr = curr[:, :5] # Arm joints only

                    # Arm joints are first 5 dims
                    arm = action_np[:, :5]
                    
                    # Optional per-axis sign remapping
                    try:
                        if getattr(args_cli, 'axis_signs', None):
                            signs = [float(x.strip()) for x in str(args_cli.axis_signs).split(',')]
                            if len(signs) >= 5:
                                sign_arr = np.array(signs[:5], dtype=np.float32).reshape(1, 5)
                                arm = arm * sign_arr
                    except Exception:
                        pass
                        
                    # Prepare per-joint scales
                    try:
                        if getattr(args_cli, 'arm_scales', None):
                            scales_vec = np.array([float(x.strip()) for x in str(args_cli.arm_scales).split(',')][:5], dtype=np.float32)
                        else:
                            scales_vec = np.full((5,), float(args_cli.arm_scale or 0.5), dtype=np.float32)
                    except Exception:
                        scales_vec = np.full((5,), float(getattr(args_cli, 'arm_scale', 0.5) or 0.5), dtype=np.float32)
                    
                    # Expand scales to batch size
                    scales_vec = scales_vec.reshape(1, 5)

                    if args_cli.action_map == 'abs2norm' and curr is not None:
                        err = arm - curr
                        arm = err / np.maximum(scales_vec, 1e-6)
                        # Min cmd shaping
                        if args_cli.min_cmd > 0.0:
                            pos_tol = float(args_cli.pos_tol or 0.0)
                            mag = np.abs(arm)
                            need_boost = (np.abs(err) > pos_tol) & (mag < args_cli.min_cmd)
                            arm = np.where(need_boost, np.sign(arm) * args_cli.min_cmd, arm)
                            
                    elif args_cli.action_map == 'delta2norm':
                        arm = arm / np.maximum(scales_vec, 1e-6)
                        
                    elif args_cli.action_map == 'abs2default':
                        try:
                            # Get default joints from env
                            if not hasattr(env, '_default_joints_cache'):
                                try:
                                    dj = env.unwrapped.scene["robot"].data.default_joint_pos
                                    # Cache it
                                    env._default_joints_cache = np.array(dj[0, :5].cpu().numpy(), dtype=np.float32)
                                except Exception:
                                    # Fallback to current joints at step 0
                                    if curr is not None:
                                        env._default_joints_cache = curr[0]
                                    else:
                                        env._default_joints_cache = np.zeros(5, dtype=np.float32)
                            
                            defaults = env._default_joints_cache.reshape(1, 5)
                            arm = (arm - defaults) / np.maximum(scales_vec, 1e-6)
                            
                            # Min cmd shaping against absolute current
                            if args_cli.min_cmd > 0.0 and curr is not None:
                                err = (action_np[:, :5] - curr)
                                pos_tol = float(args_cli.pos_tol or 0.0)
                                mag = np.abs(arm)
                                need_boost = (np.abs(err) > pos_tol) & (mag < args_cli.min_cmd)
                                arm = np.where(need_boost, np.sign(arm) * args_cli.min_cmd, arm)
                        except Exception:
                            pass
                    
                    # Clip to [-1, 1]
                    arm = np.clip(arm, -1.0, 1.0)
                    
                    # Optional arm bias
                    try:
                        if getattr(args_cli, 'arm_bias', None):
                            bias_vals = [float(x) for x in str(args_cli.arm_bias).split(',') if len(str(args_cli.arm_bias)) > 0]
                            if len(bias_vals) >= 5:
                                bias_arr = np.array(bias_vals[:5], dtype=np.float32).reshape(1, 5)
                                arm = np.clip(arm + bias_arr, -1.0, 1.0)
                    except Exception:
                        pass
                    
                    # Anti-stall (simplified for batch - tracking history per env is complex here, skipping per-env hist for now or applying global)
                    # For simplicity in training loop, we might skip stateful anti-stall or apply simple boost
                    # Skipping complex stateful anti-stall to avoid batch mismanagement
                    
                    action_np[:, :5] = arm
                    
                    # Gripper processing
                    if action_np.shape[1] >= 6:
                        gripper_raw = action_np[:, 5].copy()
                        if args_cli.action_map == 'abs2default':
                            # Get default gripper
                            try:
                                if not hasattr(env, '_default_gripper_cache'):
                                    dj = env.unwrapped.scene["robot"].data.default_joint_pos
                                    env._default_gripper_cache = float(dj[0, 5].cpu().numpy())
                            except Exception:
                                env._default_gripper_cache = 0.0
                            
                            gripper_scale = 1.1
                            gripper_norm = (gripper_raw - env._default_gripper_cache) / gripper_scale
                            action_np[:, 5] = gripper_norm
                        
                        # Gripper gain
                        if args_cli.gripper_gain != 1.0:
                            action_np[:, 5] = action_np[:, 5] * args_cli.gripper_gain
                        
                        action_np[:, 5] = np.clip(action_np[:, 5], -1.0, 1.0)

            except Exception as e:
                print(f"[WARNING] Action mapping failed: {e}")
            
            # Convert back to tensor for env step
            action_env = torch.from_numpy(action_np).to(device)
            
            # Debug: Print shapes
            if step == 0 and iteration == 0:
                print(f"[DEBUG] action_np.shape: {action_np.shape}")
                print(f"[DEBUG] action_env.shape BEFORE broadcast: {action_env.shape}")
                print(f"[DEBUG] num_envs: {actual_num_envs}")
            
            # Ensure action matches num_envs (broadcast if needed)
            if action_env.shape[0] == 1 and actual_num_envs > 1:
                action_env = action_env.repeat(actual_num_envs, 1)
            elif action_env.shape[0] != actual_num_envs:
                # If mismatch, take only first action and broadcast
                action_env = action_env[:1].repeat(actual_num_envs, 1)
            
            if step == 0 and iteration == 0:
                print(f"[DEBUG] action_env.shape AFTER broadcast: {action_env.shape}")
            
            # Step environment with MAPPED action
            obs_dict, reward, terminated, truncated, info = env.step(action_env)

            # --- JAW-PAYLOAD DISTANCE-BASED REWARD ---
            # Calculate distance between gripper center (midpoint of Fixed_Jaw and Moving_Jaw) and payload center
            # Reward for getting closer, penalize for moving away
            gripper_center_distance = None  # Track for contact check
            try:
                if args_cli.jaw_contact_reward > 0 or args_cli.jaw_distance_reward_scale > 0:
                    robot = env.unwrapped.scene["robot"]
                    
                    # Get Fixed_Jaw and Moving_Jaw positions
                    fixed_jaw_idx = robot.find_bodies("Fixed_Jaw")[0][0]
                    moving_jaw_idx = robot.find_bodies("Moving_Jaw")[0][0]
                    fixed_jaw_pos = robot.data.body_pos_w[:, fixed_jaw_idx, :]  # World position
                    moving_jaw_pos = robot.data.body_pos_w[:, moving_jaw_idx, :]  # World position
                    
                    # Calculate gripper center (midpoint between Fixed_Jaw and Moving_Jaw)
                    gripper_center_pos = (fixed_jaw_pos[0] + moving_jaw_pos[0]) / 2.0
                    
                    # Get payload position from USD prim
                    from pxr import UsdGeom
                    stage = env.unwrapped.sim.stage
                    payload_prim = stage.GetPrimAtPath(args_cli.payload_prim)
                    if payload_prim.IsValid():
                        xformable = UsdGeom.Xformable(payload_prim)
                        world_transform = xformable.ComputeLocalToWorldTransform(0)
                        payload_pos = world_transform.ExtractTranslation()
                        payload_pos_tensor = torch.tensor([payload_pos[0], payload_pos[1], payload_pos[2]], 
                                                          device=device, dtype=torch.float32)
                        
                        # Calculate current distance from gripper center to payload center
                        current_distance = torch.norm(gripper_center_pos - payload_pos_tensor).item()
                        gripper_center_distance = current_distance  # Save for contact check
                        
                        # Distance-based reward (if previous distance exists)
                        if previous_jaw_payload_distance is not None:
                            distance_delta = previous_jaw_payload_distance - current_distance  # Positive if getting closer
                            
                            if distance_delta > 0:
                                # Getting closer -> reward (ONLY reward when approaching)
                                distance_reward = distance_delta * args_cli.jaw_distance_reward_scale
                                reward = reward + torch.tensor([distance_reward], device=device, dtype=torch.float32).expand_as(reward)
                                if distance_reward > 0.01:  # Only print significant rewards
                                    print(f"📍 [APPROACHING] Iter {iteration+1} Step {step}: Δdist={distance_delta:.4f}m (closer) -> +{distance_reward:.4f} reward")
                            # else: Moving away or same distance -> NO reward, NO penalty (just 0)
                        
                        # Update previous distance
                        previous_jaw_payload_distance = current_distance
                        
                        # Contact bonus reward (if gripper center is within threshold of payload center)
                        if current_distance < args_cli.jaw_contact_threshold:
                            contact_bonus = torch.tensor([args_cli.jaw_contact_reward], device=device, dtype=torch.float32)
                            reward = reward + contact_bonus.expand_as(reward)
                            print(f"✋ [GRIPPER-CONTACT] Iter {iteration+1} Step {step}: Gripper center to payload = {current_distance:.4f}m < {args_cli.jaw_contact_threshold}m -> +{args_cli.jaw_contact_reward} bonus!")
            except Exception as e:
                if step == 0 and iteration == 0:
                    print(f"[WARNING] Jaw-payload contact reward failed: {e}")
            
            # Convert to tensors if needed and ensure correct shape
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, device=device, dtype=torch.float32)
            if not isinstance(terminated, torch.Tensor):
                terminated = torch.tensor(terminated, device=device, dtype=torch.bool)
            if not isinstance(truncated, torch.Tensor):
                truncated = torch.tensor(truncated, device=device, dtype=torch.bool)
            
            # Ensure 1D tensors
            if reward.ndim == 0:
                reward = reward.unsqueeze(0)
            if terminated.ndim == 0:
                terminated = terminated.unsqueeze(0)
            if truncated.ndim == 0:
                truncated = truncated.unsqueeze(0)
            
            done = terminated | truncated
            
            # For single-env training (num_envs=1), slice all tensors to first env only
            # This handles the case where Isaac Lab environment returns data for multiple envs
            # but we only want to train on one
            state_to_store = state[:1] if state.shape[0] > 1 else state
            action_to_store = action_sampled[:1] if action_sampled.shape[0] > 1 else action_sampled
            reward_to_store = reward[:1] if reward.shape[0] > 1 else reward
            done_to_store = done[:1] if done.shape[0] > 1 else done
            value_to_store = value[:1] if value.shape[0] > 1 else value
            log_prob_to_store = log_prob[:1] if log_prob.shape[0] > 1 else log_prob
            
            # Store image for BC loss if enabled
            image_to_store = image if (args_cli.bc_lambda > 0 and args_cli.use_vision and image is not None) else None
            
            # Store transition (include image for Reward-Weighted BC finetuning)
            buffer.add(state_to_store, action_to_store, reward_to_store, done_to_store.float(), value_to_store, log_prob_to_store, image=image_to_store)
            
            # Delete image tensor after storing to free GPU memory
            if image is not None:
                del image
                image = None
            
            # Track metrics (only for first env since we train with single env)
            episode_reward_sum += reward_to_store
            episode_length += 1
            
            # Check if any episode reached max_episode_steps (forced truncation)
            if args_cli.max_episode_steps is not None:
                for env_idx in range(training_num_envs):
                    if episode_length[env_idx] >= args_cli.max_episode_steps:
                        truncated[env_idx] = True
                        done[env_idx] = True
                        print(f"[Env {env_idx}] Reached max_episode_steps={args_cli.max_episode_steps}, forcing reset")
            
            # Print step reward (every 10 steps to avoid spam)
            if step % 10 == 0:
                for env_idx in range(min(training_num_envs, 3)):  # Show max 3 envs
                    # Include gripper-payload distance info
                    dist_info = f" | Grip-Pay Dist: {gripper_center_distance:.4f}m" if gripper_center_distance is not None else ""
                    contact_info = " 🤏" if gripper_center_distance is not None and gripper_center_distance < args_cli.jaw_contact_threshold else ""
                    print(f"[Iter {iteration+1} Step {step} Env {env_idx}] Reward: {reward_to_store[env_idx].item():.4f} | Episode Total: {episode_reward_sum[env_idx].item():.4f} | Ep Length: {int(episode_length[env_idx].item())}{dist_info}{contact_info}")
            
            # Handle done episodes
            if done_to_store.any():
                for env_idx in range(training_num_envs):
                    if done_to_store[env_idx]:
                        ep_reward = episode_reward_sum[env_idx].item()
                        ep_length = episode_length[env_idx].item()
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)
                        total_episodes += 1
                        
                        # Print episode completion
                        status = "✅ SUCCESS" if terminated[env_idx] else "⏹️  TRUNCATED"
                        print(f"\n{'='*60}")
                        print(f"🏁 EPISODE COMPLETE [Env {env_idx}]")
                        print(f"{'='*60}")
                        print(f"Status: {status}")
                        print(f"Total Reward: {ep_reward:.4f}")
                        print(f"Episode Length: {int(ep_length)} steps")
                        print(f"Total Episodes: {total_episodes}")
                        print(f"Success Rate: {(success_count / total_episodes * 100):.1f}%")
                        print(f"{'='*60}\n")
                        
                        if terminated[env_idx]:
                            success_count += 1
                        
                        # Reset jaw-payload distance tracking on episode end
                        previous_jaw_payload_distance = None
                        
                        # Reset Diffusion Policy observation history and action queue for new episode
                        policy.reset_episode()
                        
                        # CRITICAL: Clear GPU memory after each episode
                        try:
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        
                        # Reset tracking
                        episode_reward_sum[env_idx] = 0
                        episode_length[env_idx] = 0
        
        # Compute advantages and returns
        states, actions, rewards, dones, values, log_probs, images = buffer.get()
        advantages, returns = compute_gae(
            rewards, values, dones, 
            gamma=args_cli.gamma, 
            lam=args_cli.gae_lambda
        )
        
        # Perform PPO updates with Reward-Weighted BC finetuning
        for epoch in range(args_cli.ppo_epochs):
            metrics = train_ppo_epoch(
                policy=policy,
                value_net=value_net,
                optimizer_policy=optimizer_policy,
                optimizer_value=optimizer_value,
                states=states,
                actions=actions,
                old_log_probs=log_probs,
                advantages=advantages,
                returns=returns,
                clip_epsilon=args_cli.clip_epsilon,
                entropy_coef=args_cli.entropy_coef,
                value_coef=args_cli.value_coef,
                max_grad_norm=args_cli.max_grad_norm,
                batch_size=args_cli.batch_size,
                images=images,  # Pass images for BC loss
                # Reward-Weighted BC loss parameters
                bc_lambda=args_cli.bc_lambda,
                bc_batch_size=args_cli.bc_batch_size,
                advantage_clip=args_cli.advantage_clip,
                positive_advantage_only=args_cli.positive_advantage_only,
            )
        
        # Clear training data after PPO updates to free memory
        del states, actions, rewards, dones, values, log_probs, advantages, returns
        if images is not None:
            del images
        
        # Force GPU memory cleanup after each iteration
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        
        iter_time = time.time() - iter_start_time
        
        # Decay exploration noise
        if current_exploration_noise > args_cli.exploration_min:
            current_exploration_noise = max(
                current_exploration_noise * args_cli.exploration_decay,
                args_cli.exploration_min
            )
        
        # Logging
        if (iteration + 1) % args_cli.log_interval == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
            success_rate = (success_count / total_episodes * 100) if total_episodes > 0 else 0.0
            
            # Compute additional statistics
            reward_std = np.std(episode_rewards) if len(episode_rewards) > 1 else 0.0
            reward_min = np.min(episode_rewards) if episode_rewards else 0.0
            reward_max = np.max(episode_rewards) if episode_rewards else 0.0
            
            # Compute FPS (frames per second)
            fps = (args_cli.horizon * training_num_envs) / iter_time if iter_time > 0 else 0.0
            
            # Estimate time remaining
            iters_remaining = args_cli.max_iterations - (iteration + 1)
            time_remaining_sec = iters_remaining * iter_time
            time_remaining_min = time_remaining_sec / 60
            time_remaining_hr = time_remaining_min / 60
            
            print(f"\n{'='*80}")
            print(f"[ITERATION {iteration + 1}/{args_cli.max_iterations}]")
            print(f"{'='*80}")
            print(f"⏱️  TIMING:")
            print(f"   Iteration Time: {iter_time:.2f}s")
            print(f"   FPS: {fps:.1f} (frames/sec)")
            print(f"   ETA: {time_remaining_hr:.1f}h ({time_remaining_min:.0f}min)")
            print(f"\n📊 EPISODES:")
            print(f"   Total Episodes: {total_episodes}")
            print(f"   Successful: {success_count} ({success_rate:.1f}%)")
            print(f"   Avg Length: {avg_length:.1f} steps")
            print(f"\n🎯 REWARDS:")
            print(f"   Mean: {avg_reward:.4f}")
            print(f"   Std:  {reward_std:.4f}")
            print(f"   Min:  {reward_min:.4f}")
            print(f"   Max:  {reward_max:.4f}")
            print(f"\n🔍 EXPLORATION:")
            print(f"   Noise: {current_exploration_noise:.4f}")
            print(f"\n🧠 LOSSES:")
            print(f"   Policy Loss: {metrics['policy_loss']:.6f}")
            print(f"   Value Loss:  {metrics['value_loss']:.6f}")
            print(f"   Entropy:     {metrics['entropy']:.6f}")
            if args_cli.bc_lambda > 0:
                print(f"   BC Loss:     {metrics['bc_loss']:.6f} (λ={args_cli.bc_lambda})")
            
            # LEARNING PROGRESS INDICATOR
            reward_history.append(avg_reward)
            if avg_reward > best_reward:
                best_reward = avg_reward
                new_best = True
            else:
                new_best = False
            
            # Calculate moving average (last 10 iterations)
            recent_rewards = reward_history[-10:] if len(reward_history) >= 10 else reward_history
            moving_avg = sum(recent_rewards) / len(recent_rewards)
            
            print(f"\n🎓 LEARNING STATUS:")
            if new_best:
                print(f"   🏆 NEW BEST REWARD! {best_reward:.4f}")
            
            if iteration > 0:
                reward_change = avg_reward - prev_avg_reward
                if reward_change > 0.001:
                    trend = f"📈 +{reward_change:.4f}"
                elif reward_change < -0.001:
                    trend = f"📉 {reward_change:.4f}"
                else:
                    trend = "➡️  stable"
                print(f"   Reward Trend:  {trend}")
            
            print(f"   Moving Avg:    {moving_avg:.4f} (last {len(recent_rewards)} iters)")
            print(f"   Best Reward:   {best_reward:.4f}")
            print(f"   log_std:       {policy.log_std.mean().item():.4f}")
            print(f"   adapter_scale: {policy.adapter_scale.item():.4f} (residual strength)")
            
            # Simple progress bar
            progress = (iteration + 1) / args_cli.max_iterations * 100
            bar_len = 20
            filled = int(bar_len * progress / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"   Progress:      [{bar}] {progress:.1f}%")
            
            prev_avg_reward = avg_reward
            
            # GPU memory stats
            try:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"\n💾 GPU MEMORY:")
                    print(f"   Allocated: {allocated:.2f} GB")
                    print(f"   Reserved:  {reserved:.2f} GB")
            except Exception:
                pass
            
            print(f"{'='*80}\n")
        
        # Save checkpoint (only latest.pt)
        if (iteration + 1) % args_cli.save_interval == 0:
            avg_reward_ckpt = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_length_ckpt = np.mean(episode_lengths) if episode_lengths else 0.0
            success_rate_ckpt = (success_count / total_episodes) if total_episodes > 0 else 0.0
            
            # Save only latest checkpoint
            latest_path = os.path.join(args_cli.checkpoint_dir, "latest.pt")
            torch.save({
                'iteration': iteration,
                'policy': policy.state_dict(),
                'value_net': value_net.state_dict(),
                'optimizer_policy': optimizer_policy.state_dict(),
                'optimizer_value': optimizer_value.state_dict(),
                'metrics': {
                    'avg_reward': avg_reward_ckpt,
                    'avg_length': avg_length_ckpt,
                    'success_rate': success_rate_ckpt,
                },
            }, latest_path)
            
            # Get file size
            try:
                file_size_mb = os.path.getsize(latest_path) / (1024**2)
                print(f"\n💾 CHECKPOINT SAVED:")
                print(f"   Path: {latest_path}")
                print(f"   Size: {file_size_mb:.2f} MB")
                print(f"   Iteration: {iteration + 1}")
                print(f"   Episodes: {total_episodes}")
                print(f"   Avg Reward: {avg_reward_ckpt:.4f}")
                print(f"   Success Rate: {success_rate_ckpt*100:.1f}%\n")
            except Exception:
                print(f"[INFO] Saved checkpoint to: {latest_path}")
    
    print("\n[INFO] Training completed!")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Final avg reward: {np.mean(episode_rewards):.4f}")
    print(f"  Final success rate: {(success_count / total_episodes * 100):.1f}%")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

