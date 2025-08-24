# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Template-So-100-CubeLift-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=2000, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--action_gain", type=float, default=1.0, help="Multiply model action outputs by this gain before clamping.")
parser.add_argument("--action_map", type=str, default="none", choices=["none","abs2norm","delta2norm","abs2default"], help="Map model actions to env normalized commands: none (as-is), abs2norm (absolute target to normalized via (target-current)/scale), delta2norm (delta target to normalized via delta/scale), abs2default (absolute target to normalized via (target-default)/scale)")
parser.add_argument("--arm_scale", type=float, default=0.5, help="Env JointPositionActionCfg scale used to normalize arm actions.")
parser.add_argument("--arm_scales", type=str, default=None, help="Comma-separated per-joint scales for 5 arm joints (e.g., '0.5,0.5,0.5,0.5,0.5'). Overrides --arm_scale if provided.")
parser.add_argument("--binarize_gripper", action="store_true", default=True, help="Threshold last action dim to {-1, +1} before sending to env.")
parser.add_argument("--invert_gripper", action="store_true", default=False, help="Flip gripper sign after binarization if needed (swap open/close).")
# Add a complementary flag to disable binarization
parser.add_argument("--no_binarize_gripper", action="store_false", dest="binarize_gripper", help="Disable gripper binarization.")
parser.add_argument("--dump_rgb", action="store_true", default=False, help="Save per-step RGB observations seen by the policy.")
parser.add_argument("--dump_rgb_dir", type=str, default="outputs/eval_rgb", help="Directory to save RGB frames.")
parser.add_argument("--dump_rgb_every", type=int, default=1, help="Save every N steps (default: 1 = every step).")
parser.add_argument("--dump_rgb_processed", action="store_true", default=True, help="Also save the processed (resized, normalized) image fed to the model.")
parser.add_argument("--camera_format", type=str, choices=["rgb", "bgr"], default="rgb", help="Interpret incoming camera frames as this format before feeding the model.")
parser.add_argument("--camera_source", type=str, choices=["sensor", "viewport"], default="sensor", help="Where to read RGB from: Isaac Lab sensor (env.scene['camera']) or UI viewport (screenshot).")
parser.add_argument("--dump_only_model_input", action="store_true", default=False, help="If set, only save the exact image fed to the model (proc_model_input.png)")
# New: dump the exact model input at a specific step index
parser.add_argument("--dump_model_input_step", type=int, default=None, help="If set, save exactly the model input RGB at the given step index (e.g., 50).")
parser.add_argument("--dump_model_input_dir", type=str, default="outputs/model_input_rgb", help="Directory to save the exact model input RGB (PNG + NPY).")
# New: dump the raw camera frame (as it first arrives from the env) at a specific step index
parser.add_argument("--dump_raw_cam_step", type=int, default=None, help="If set, save obs['camera_rgb'] exactly as received at the given step (e.g., 50).")
parser.add_argument("--dump_raw_cam_dir", type=str, default="outputs/raw_cam", help="Directory to save raw camera frames (PNG + NPY) without any processing.")
parser.add_argument("--squash", action="store_true", default=False, help="Apply tanh squashing to network outputs before inverse-scaling to reduce saturation.")
parser.add_argument("--rate_limit", type=float, default=0.0, help="Max per-step change in action (in env units) for arm joints; 0 disables.")
parser.add_argument("--rate_limit_gripper", type=float, default=0.0, help="Max per-step change for gripper channel; 0 disables.")
parser.add_argument("--saturation_window", type=int, default=10, help="If actions hit clip for this many consecutive steps, reduce gain adaptively.")
parser.add_argument("--gain_decay", type=float, default=0.5, help="Factor to multiply action_gain when sustained saturation is detected.")
parser.add_argument("--on_done", type=str, default="reset", choices=["stop","reset"], help="What to do when env signals done: stop (return) or reset (continue until horizon).")
# Min command shaping for abs2norm to overcome stiction near target
parser.add_argument("--min_cmd", type=float, default=0.0, help="Minimum normalized command magnitude per joint when using abs2norm if error > pos_tol (0 disables).")
parser.add_argument("--pos_tol", type=float, default=0.0, help="Position error tolerance (in env units, e.g., radians) under which min_cmd is not applied.")
# Anti-stall booster if normalized arm commands stay very small over a window
parser.add_argument("--anti_stall", action="store_true", default=False, help="Enable anti-stall: if mean |arm| over a window is below threshold, boost commands.")
parser.add_argument("--anti_stall_window", type=int, default=15, help="Window size (steps) for anti-stall detection.")
parser.add_argument("--anti_stall_thresh", type=float, default=0.05, help="Mean absolute normalized arm magnitude threshold to trigger boost.")
parser.add_argument("--anti_stall_boost", type=float, default=1.5, help="Multiplicative boost applied to normalized arm when anti-stall triggers (then clipped to [-1,1]).")
# Use absolute joint positions from env for abs2norm current state instead of normalized obs
parser.add_argument("--curr_from_env", action="store_true", default=False, help="Use env.scene['robot'].joint_pos (radians) as current joints for abs2norm mapping.")
parser.add_argument("--axis_signs", type=str, default=None, help="Comma-separated signs for 5 arm axes (e.g., '1,-1,1,1,1'). Applied after mapping, before clipping.")
# Add dataset root for policies that need dataset stats (e.g., SmolVLA)
parser.add_argument("--dataset_root", type=str, default="/home/beable/IsaacLab-SO_100/dataset1", help="Local dataset root providing stats (meta/info.json) for VLA models.")
# Force model input RGB size (H, W); default 512x512 for square inputs
parser.add_argument("--rgb_height", type=int, default=512, help="Model input RGB height after resize/padding.")
parser.add_argument("--rgb_width", type=int, default=512, help="Model input RGB width after resize/padding.")
parser.add_argument("--normalize_state_limits", action="store_true", default=False, help="Normalize observation.state to [-1,1] using joint limits (mid/half).")
parser.add_argument("--state_lower", type=str, default=None, help="Comma-separated lower limits (rad) for 6 joints (Rotation,Pitch,Elbow,Wrist_Pitch,Wrist_Roll,Jaw).")
parser.add_argument("--state_upper", type=str, default=None, help="Comma-separated upper limits (rad) for 6 joints (Rotation,Pitch,Elbow,Wrist_Pitch,Wrist_Roll,Jaw).")
# Replan frequency override (default: use training n_action_steps)
parser.add_argument("--replan_every", type=int, default=None, help="Force the VLA policy to reset/replan every N steps. If not set, uses training n_action_steps.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# also enable cameras if dumping rgb or training used vision
try:
    if getattr(args_cli, 'dump_rgb', False):
        args_cli.enable_cameras = True
    meta_path = 'outputs/bc_training_meta.txt'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as mf:
            for line in mf:
                if line.startswith('arch=') and 'vision' in line:
                    args_cli.enable_cameras = True
                    print('[INFO] Enabling cameras based on training meta (vision).')
                    break
except Exception:
    pass

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from isaaclab_tasks.utils import parse_env_cfg

import SO_100.tasks  # noqa: F401

import os
import numpy as np
import torch.nn as nn
import cv2

# SmolVLA support (model-only additions)
try:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy as make_lerobot_policy
    HAVE_SMOLVLA = True
except Exception:
    HAVE_SMOLVLA = False

# Helper: resize with padding to 512x512 (letterbox) in RGB float[0,1]
def resize_with_padding_rgb(img_hwc_float01: np.ndarray, target_hw=(512, 512)) -> np.ndarray:
	# img_hwc_float01: (H, W, 3), values in [0,1]
	# If non-square target, use exact resize; if square, letterbox-pad to preserve aspect
	h, w = img_hwc_float01.shape[:2]
	# Validate target size tuple and sanitize values
	try:
		if target_hw is None or not isinstance(target_hw, (tuple, list)) or len(target_hw) != 2:
			return img_hwc_float01
		th_raw, tw_raw = target_hw
		# Coerce to floats first to handle strings or numpy types, then to ints
		th_f = float(th_raw)
		tw_f = float(tw_raw)
		# Check finiteness and positivity
		if not np.isfinite(th_f) or not np.isfinite(tw_f):
			return img_hwc_float01
		th = int(round(th_f))
		tw = int(round(tw_f))
	except Exception:
		return img_hwc_float01
	# Guard against invalid sizes
	if th is None or tw is None or th <= 0 or tw <= 0:
		# Invalid target -> return input as-is
		return img_hwc_float01
	# Guard against invalid input image dimensions
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


class AdvancedBCNetwork(nn.Module):
    """Matches training MLP architecture (state-only)."""

    def __init__(self, input_dim=6, output_dim=6, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.05),
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LightVisionCNN(nn.Module):
    """Small and fast CNN backbone to match training script."""

    def __init__(self, img_channels=3, state_dim=6, action_dim=6, width_mul: float = 1.0, dropout: float = 0.05):
        super().__init__()
        c1 = max(8, int(16 * width_mul))
        c2 = max(16, int(32 * width_mul))
        c3 = max(32, int(64 * width_mul))
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, c1, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(c1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        fusion_dim = c3 + state_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, action_dim),
        )

    def forward(self, state, image):
        feat = self.cnn(image).flatten(1)
        x = torch.cat([feat, state], dim=1)
        return self.head(x)


def rollout(policy, env, horizon, device):
    """Run a single rollout with the policy."""
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])
    print(f"[DEBUG] Initial obs_dict keys: {list(obs_dict.keys())}")
    print(f"[DEBUG] Policy obs type: {type(obs_dict['policy'])}")
    # prepare dump dir if enabled
    if args_cli.dump_rgb:
        os.makedirs(args_cli.dump_rgb_dir, exist_ok=True)
    
    # Detailed observation debugging
    if isinstance(obs_dict['policy'], dict):
        print(f"[DEBUG] Policy obs dict keys: {list(obs_dict['policy'].keys())}")
        for key, value in obs_dict['policy'].items():
            if hasattr(value, 'shape'):
                print(f"[DEBUG] {key} shape: {value.shape}, dtype: {value.dtype}")
            else:
                print(f"[DEBUG] {key} type: {type(value)}, value: {value}")
    elif hasattr(obs_dict['policy'], 'shape'):
        print(f"[DEBUG] Policy obs shape: {obs_dict['policy'].shape}, dtype: {obs_dict['policy'].dtype}")
    else:
        print(f"[DEBUG] Policy obs: {obs_dict['policy']}")

    for i in range(horizon):
        # Prepare observations - pass the full obs_dict to policy
        obs = obs_dict["policy"]

        # Raw camera dump (exactly as received) at specified step
        try:
            if getattr(args_cli, 'dump_raw_cam_step', None) is not None and isinstance(obs, dict) and 'camera_rgb' in obs:
                target = int(args_cli.dump_raw_cam_step)
                if i == target:
                    import os, cv2
                    os.makedirs(args_cli.dump_raw_cam_dir, exist_ok=True)
                    cam = obs['camera_rgb']
                    if isinstance(cam, torch.Tensor):
                        cam_np = cam.detach().cpu().numpy()
                    else:
                        cam_np = np.array(cam)
                    # cam_np shape: (1, H, W, 3) -> remove batch dim
                    if cam_np.ndim == 4 and cam_np.shape[0] == 1:
                        cam_np = cam_np[0]
                    # Save .npy (exact raw array)
                    np.save(os.path.join(args_cli.dump_raw_cam_dir, f"raw_cam_obs_step_{i:05d}.npy"), cam_np)
                    # Prepare uint8 image
                    out = cam_np
                    if out.dtype != np.uint8:
                        if out.max() <= 1.0:
                            out = (out * 255.0).clip(0, 255)
                        out = out.astype(np.uint8)
                    # Save under different channel assumptions for diagnostics
                    # 1) Assume env is RGB (convert to BGR for cv2)
                    bgr_from_rgb = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(args_cli.dump_raw_cam_dir, f"raw_cam_obs_step_{i:05d}_assumeRGB.png"), bgr_from_rgb)
                    # 2) Assume env is already BGR (write directly)
                    cv2.imwrite(os.path.join(args_cli.dump_raw_cam_dir, f"raw_cam_obs_step_{i:05d}_assumeBGR.png"), out)
                    # 3) Channel-swapped visualization
                    cv2.imwrite(os.path.join(args_cli.dump_raw_cam_dir, f"raw_cam_obs_step_{i:05d}_channelswapped.png"), out[:, :, ::-1])
                    # 4) Simple gamma-corrected visualization (approx. sRGB)
                    try:
                        lin = out.astype(np.float32) / 255.0
                        gamma = 1/2.2
                        viz = np.power(np.clip(lin, 0.0, 1.0), gamma)
                        viz_uint8 = (viz * 255.0).clip(0, 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(args_cli.dump_raw_cam_dir, f"raw_cam_obs_step_{i:05d}_gamma.png"), viz_uint8)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARNING] Failed to dump raw camera in rollout at step {i}: {e}")
        
        if i == 0:  # Print first observation details
            if isinstance(obs, dict):
                print(f"[DEBUG] First obs - joint_pos: {obs['joint_pos'].shape}, camera_rgb: {obs['camera_rgb'].shape}")
            else:
                print(f"[DEBUG] First obs shape: {obs.shape}")
        
        traj["obs"].append(obs)

        # Compute actions - pass the full observation dict
        act = policy(obs)
        # Convert to numpy for logging / optional clipping
        if isinstance(act, torch.Tensor):
            action_np = act.detach().cpu().numpy()
        else:
            action_np = np.array(act, dtype=np.float32)
        # Ensure shape (1, A)
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)

        # Map actions to environment normalized space if requested
        try:
            if args_cli.action_map != "none":
                # Extract current joint positions
                if args_cli.curr_from_env:
                    try:
                        # Use env-provided absolute joint positions
                        jp = env.unwrapped.scene["robot"].data.joint_pos
                        curr = np.array(jp[:1, :5].cpu().numpy(), dtype=np.float32)
                    except Exception:
                        curr = None
                else:
                    if isinstance(obs, dict) and isinstance(obs.get('joint_pos', None), torch.Tensor):
                        curr = obs['joint_pos'].detach().cpu().numpy().reshape(1, -1)
                    elif isinstance(obs, dict):
                        curr = np.array(obs['joint_pos']).reshape(1, -1)
                    else:
                        curr = None
                # Arm joints are first 5 dims; gripper is last dim
                arm = action_np[:, :5]
                # Optional per-axis sign remapping (to match teleop key mapping)
                try:
                    if getattr(args_cli, 'axis_signs', None):
                        signs = [float(x.strip()) for x in str(args_cli.axis_signs).split(',')]
                        if len(signs) >= 5:
                            arm = arm * np.array(signs[:5], dtype=np.float32).reshape(1, 5)
                except Exception:
                    pass
                # Prepare per-joint scales
                try:
                    if getattr(args_cli, 'arm_scales', None):
                        scales_vec = np.array([float(x.strip()) for x in str(args_cli.arm_scales).split(',')][:5], dtype=np.float32)
                        if scales_vec.shape[0] < 5:
                            raise ValueError
                    else:
                        scales_vec = np.full((5,), float(args_cli.arm_scale or 0.5), dtype=np.float32)
                except Exception:
                    scales_vec = np.full((5,), float(getattr(args_cli, 'arm_scale', 0.5) or 0.5), dtype=np.float32)
                if args_cli.action_map == 'abs2norm' and curr is not None:
                    err = arm - curr[:, :5]
                    arm = err / np.maximum(scales_vec.reshape(1, 5), 1e-6)
                    # Apply min_cmd shaping if configured
                    if args_cli.min_cmd > 0.0:
                        pos_tol = float(args_cli.pos_tol or 0.0)
                        # For joints where |err| > pos_tol, enforce a minimum normalized magnitude
                        mag = np.abs(arm)
                        need_boost = (np.abs(err) > pos_tol) & (mag < args_cli.min_cmd)
                        arm = np.where(need_boost, np.sign(arm) * args_cli.min_cmd, arm)
                elif args_cli.action_map == 'delta2norm':
                    arm = arm / np.maximum(scales_vec.reshape(1, 5), 1e-6)
                elif args_cli.action_map == 'abs2default':
                    try:
                        # Cache default joints from env if available
                        if not hasattr(rollout, '_default_joints') or rollout._default_joints is None:
                            try:
                                dj = env.unwrapped.scene["robot"].data.default_joint_pos
                                rollout._default_joints = np.array(dj[:1, :5].cpu().numpy(), dtype=np.float32)
                            except Exception:
                                # Fallback to first-step absolute joints
                                jp0 = env.unwrapped.scene["robot"].data.joint_pos
                                rollout._default_joints = np.array(jp0[:1, :5].cpu().numpy(), dtype=np.float32)
                        # Normalize absolute targets against default joints
                        arm = (arm - rollout._default_joints) / np.maximum(scales_vec.reshape(1, 5), 1e-6)
                        # Optionally apply min_cmd shaping using instantaneous error vs current (absolute)
                        if args_cli.min_cmd > 0.0:
                            try:
                                jp_abs = env.unwrapped.scene["robot"].data.joint_pos
                                curr_abs = np.array(jp_abs[:1, :5].cpu().numpy(), dtype=np.float32)
                            except Exception:
                                curr_abs = curr[:, :5] if curr is not None else None
                            if curr_abs is not None:
                                err = (action_np[:, :5] - curr_abs)
                                pos_tol = float(args_cli.pos_tol or 0.0)
                                mag = np.abs(arm)
                                need_boost = (np.abs(err) > pos_tol) & (mag < args_cli.min_cmd)
                                arm = np.where(need_boost, np.sign(arm) * args_cli.min_cmd, arm)
                    except Exception:
                        pass
                # Clip to [-1, 1]
                arm = np.clip(arm, -1.0, 1.0)
                # Anti-stall: if arm magnitude stays tiny for a window, boost
                try:
                    if args_cli.anti_stall and args_cli.action_map in ['abs2norm', 'abs2default']:
                        if not hasattr(rollout, '_arm_hist'):
                            rollout._arm_hist = []
                        rollout._arm_hist.append(np.mean(np.abs(arm)))
                        if len(rollout._arm_hist) > max(1, args_cli.anti_stall_window):
                            rollout._arm_hist.pop(0)
                        if len(rollout._arm_hist) == max(1, args_cli.anti_stall_window):
                            mean_mag = float(np.mean(rollout._arm_hist))
                            if mean_mag < float(args_cli.anti_stall_thresh or 0.0):
                                arm = np.clip(arm * float(args_cli.anti_stall_boost or 1.0), -1.0, 1.0)
                                if i % 10 == 0:
                                    print(f"[INFO] Anti-stall boost applied at step {i}: mean|arm|={mean_mag:.3f}")
                except Exception:
                    pass
                action_np[:, :5] = arm
                # Gripper: binarize to {-1, +1} if enabled
                if args_cli.binarize_gripper and action_np.shape[1] >= 6:
                    action_np[:, 5] = np.where(action_np[:, 5] >= 0.0, 1.0, -1.0)
                    if args_cli.invert_gripper:
                        action_np[:, 5] = -action_np[:, 5]
        except Exception:
            pass
        
        # Print action values every 10 steps to track changes
        if i % 10 == 0:
            flat = action_np.flatten()
            action_magnitude = np.linalg.norm(flat)
            gripper_action = flat[5]  # Gripper action
            arm_magnitude = np.linalg.norm(flat[:5])  # Arm action magnitude
            print(f"[DEBUG] Step {i}: Total: {action_magnitude:.4f}, Arm: {arm_magnitude:.4f}, Gripper: {gripper_action:.4f}")
            print(f"[DEBUG] Step {i}: Action: {flat}")
            # Also log current joint observations (env and obs)
            try:
                # From env (absolute radians)
                jp_env = None
                try:
                    jp_t = env.unwrapped.scene["robot"].data.joint_pos
                    jp_env = np.array(jp_t[:1, :6].cpu().numpy(), dtype=np.float32)
                except Exception:
                    pass
                # From observation dict if present
                jp_obs = None
                if isinstance(obs, dict) and 'joint_pos' in obs:
                    if isinstance(obs['joint_pos'], torch.Tensor):
                        jp_obs = obs['joint_pos'].detach().cpu().numpy()
                    else:
                        jp_obs = np.array(obs['joint_pos'], dtype=np.float32)
                # Print both if available
                if jp_env is not None:
                    print(f"[DEBUG] Step {i}: joint_pos (env abs) = {jp_env}")
                if jp_obs is not None:
                    print(f"[DEBUG] Step {i}: joint_pos (obs) = {jp_obs}")
            except Exception as e:
                print(f"[DEBUG] Step {i}: failed to log joint observations: {e}")
            if i == 0 and args_cli.action_map in ['abs2norm','abs2default','delta2norm']:
                try:
                    dj = env.unwrapped.scene["robot"].data.default_joint_pos
                    dj_np = np.array(dj[:1, :5].cpu().numpy(), dtype=np.float32)
                    jp = env.unwrapped.scene["robot"].data.joint_pos
                    jp_np = np.array(jp[:1, :5].cpu().numpy(), dtype=np.float32)
                    abs_pred = action_np[:, :5] if args_cli.action_map in ['abs2default','abs2norm'] else None
                    print(f"[DEBUG] Step 0: default_joints(rad)={dj_np}, curr_abs(rad)={jp_np}")
                    if abs_pred is not None:
                        print(f"[DEBUG] Step 0: abs_target(pred)={abs_pred}")
                except Exception as e:
                    print(f"[DEBUG] Step 0: debug fetch failed: {e}")

        # Convert to torch tensor on the environment/device and step
        action_t = torch.from_numpy(action_np).to(device).float()
        obs_prev = obs
        obs_dict, _, terminated, truncated, _ = env.step(action_t)
        obs = obs_dict["policy"]
        # Track state change (joint_pos delta)
        try:
            jp_prev = obs_prev['joint_pos'] if isinstance(obs_prev, dict) else None
            jp_curr = obs['joint_pos'] if isinstance(obs, dict) else None
            if jp_prev is not None and jp_curr is not None:
                if isinstance(jp_prev, torch.Tensor): jp_prev = jp_prev.cpu().numpy()
                if isinstance(jp_curr, torch.Tensor): jp_curr = jp_curr.cpu().numpy()
                d = float(np.linalg.norm(jp_curr - jp_prev))
                if i % 10 == 0:
                    print(f"[DEBUG] Step {i}: joint_pos delta norm: {d:.6f}")
        except Exception:
            pass

        # Record trajectory
        traj["actions"].append(action_t.detach().cpu().tolist())
        traj["next_obs"].append(obs)

        if terminated or truncated:
            status = "terminated" if terminated else "truncated"
            print(f"[DEBUG] Episode {status} at step {i}")
            if args_cli.on_done == "reset":
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"]
                continue
            else:
                return bool(terminated), traj

    print(f"[DEBUG] Episode reached horizon limit {horizon}")
    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    print(f"[DEBUG] Task name: {args_cli.task}")
    
    # List available environments
    import gymnasium as gym
    env_ids = [env_spec.id for env_spec in gym.envs.registry.values()]
    so_100_envs = [env_id for env_id in env_ids if "So-100" in env_id]
    print(f"[DEBUG] Available SO-100 environments: {so_100_envs}")
    
    # parse configuration like in play.py
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    
    # Try to create environment with configuration
    try:
        render_mode = "rgb_array" if args_cli.video else None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
        print(f"[DEBUG] Successfully created environment: {args_cli.task}")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return
    
    # Set seed using modern gymnasium method
    torch.manual_seed(args_cli.seed)
    env.reset(seed=args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load policy
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
        policy = None  # Will set if SmolVLA branch succeeds

        # Try SmolVLA (LeRobot) pretrained layout first: model.safetensors + train_config.json
        try:
            if HAVE_SMOLVLA:
                import os
                ckpt_path = args_cli.checkpoint
                pretrained_dir = None
                if os.path.isdir(ckpt_path):
                    if os.path.exists(os.path.join(ckpt_path, "model.safetensors")) and os.path.exists(os.path.join(ckpt_path, "train_config.json")):
                        pretrained_dir = ckpt_path
                else:
                    parent = os.path.dirname(ckpt_path)
                    if ckpt_path.endswith(".safetensors") and os.path.exists(os.path.join(parent, "train_config.json")):
                        pretrained_dir = parent
                if pretrained_dir:
                    print(f"[INFO] Detected SmolVLA pretrained directory: {pretrained_dir}")
                    # Load training config and dataset stats
                    train_cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
                    train_cfg.policy.pretrained_path = pretrained_dir
                    # Enforce device
                    train_cfg.policy.device = device.type if hasattr(device, 'type') else str(device)
                    # Dataset stats from local dataset_root (no data loading)
                    ds_meta = LeRobotDatasetMetadata(train_cfg.dataset.repo_id, root=args_cli.dataset_root, revision=train_cfg.dataset.revision)
                    lerobot_policy = make_lerobot_policy(cfg=train_cfg.policy, ds_meta=ds_meta)

                    class LeRobotAdapter:
                        def __init__(self, policy_module, device_str: str = "cuda", resize_hw=(512, 512), use_imagenet_stats: bool = False, env=None, n_action_steps: int = 100):
                            self.policy = policy_module
                            self.device = device_str
                            self.step_count = 0
                            # From training config
                            self.resize_target = resize_hw
                            self.use_imagenet = bool(use_imagenet_stats)
                            self.env = env
                            self.n_action_steps = int(n_action_steps)
                            if self.use_imagenet:
                                self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                                self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                            else:
                                self.img_mean = None
                                self.img_std = None
                        def start_episode(self):
                            self.step_count = 0
                            if hasattr(self.policy, 'reset'):
                                self.policy.reset()
                        def __call__(self, obs):
                            import numpy as np
                            import torch
                            import cv2
                            # Extract camera tensor from env sensor per Isaac Lab docs (preferred), fallback to obs
                            cam = None
                            if getattr(args_cli, 'camera_source', 'sensor') == 'viewport':
                                try:
                                    import omni.kit.viewport.utility as vp_utils
                                    vp = vp_utils.get_active_viewport_window()
                                    tex = vp.viewport_api.get_texture()
                                    img = tex.get_rgba_image()
                                    # img is HxWx4 uint8 RGBA; drop A
                                    cam = img[..., :3]
                                except Exception:
                                    cam = None
                            if cam is None:
                                try:
                                    cam_t = self.env.unwrapped.scene["camera"].data.output["rgb"]  # torch.uint8, (B,H,W,3|4)
                                    if isinstance(cam_t, torch.Tensor):
                                        cam = cam_t.detach().cpu().numpy()
                                    else:
                                        cam = np.array(cam_t)
                                except Exception:
                                    cam = obs.get('camera_rgb', None) if isinstance(obs, dict) else None
                            # If we reached the end of a chunk, reset to force replan
                            if self.step_count > 0 and self.n_action_steps > 0 and (self.step_count % self.n_action_steps) == 0:
                                if hasattr(self.policy, 'reset'):
                                    self.policy.reset()
                                    print(f"[INFO] Replanning at step {self.step_count} (chunk boundary={self.n_action_steps})")
                            # Build state from env's absolute joint positions to match dataset (6 dims)
                            state_np = None
                            try:
                                jp = self.env.unwrapped.scene["robot"].data.joint_pos
                                jp_np = jp[:1, :6].detach().cpu().numpy() if isinstance(jp, torch.Tensor) else np.array(jp)[:1, :6]
                                state_np = jp_np.astype(np.float32)
                            except Exception:
                                # Fallback to obs
                                if isinstance(obs, dict):
                                    st = obs.get('joint_pos', obs)
                                else:
                                    st = obs
                                if isinstance(st, torch.Tensor):
                                    state_np = st.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
                                else:
                                    state_np = np.array(st, dtype=np.float32).reshape(1, -1)
                            # Optional normalization of state to [-1,1] using joint limits
                            if getattr(args_cli, 'normalize_state_limits', False):
                                # Try to get limits from env first
                                low = None; high = None
                                try:
                                    rob = self.env.unwrapped.scene["robot"]
                                    low_t = getattr(rob.data, 'joint_pos_limit_lower', None)
                                    high_t = getattr(rob.data, 'joint_pos_limit_upper', None)
                                    if low_t is not None and high_t is not None:
                                        low = (low_t[:1, :6].detach().cpu().numpy() if isinstance(low_t, torch.Tensor) else np.array(low_t)[:1, :6]).astype(np.float32)
                                        high = (high_t[:1, :6].detach().cpu().numpy() if isinstance(high_t, torch.Tensor) else np.array(high_t)[:1, :6]).astype(np.float32)
                                except Exception:
                                    pass
                                # Fallback to CLI-provided limits
                                if (low is None or high is None) and getattr(args_cli, 'state_lower', None) and getattr(args_cli, 'state_upper', None):
                                    try:
                                        low = np.array([float(x.strip()) for x in str(args_cli.state_lower).split(',')][:6], dtype=np.float32).reshape(1, 6)
                                        high = np.array([float(x.strip()) for x in str(args_cli.state_upper).split(',')][:6], dtype=np.float32).reshape(1, 6)
                                    except Exception:
                                        low = None; high = None
                                # If still None, use joystick defaults you provided
                                if low is None or high is None:
                                    low = np.array([ -2.1, -0.1, -0.2, -1.8, -3.14159, 0.0 ], dtype=np.float32).reshape(1, 6)
                                    high = np.array([  2.1,  3.45,  3.14159,  1.8,  3.14159, 0.8 ], dtype=np.float32).reshape(1, 6)
                                mid = 0.5 * (low + high)
                                half = 0.5 * (high - low)
                                half = np.maximum(half, 1e-6)
                                state_np = np.clip((state_np - mid) / half, -1.0, 1.0)
                            batch = {
                                "observation.state": torch.from_numpy(state_np).to(self.device).float(),
                                "task": "Follow waypoints and manipulate objects",
                            }
                            proc_img_for_dump = None
                            # Sanitize camera to HWC3 float[0,1]
                            if cam is not None:
                                # Convert to numpy
                                if isinstance(cam, torch.Tensor):
                                    cam_np = cam.detach().cpu().numpy()
                                else:
                                    cam_np = np.array(cam)
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
                                # Final guard: if still invalid, fallback to zeros of target size
                                th, tw = self.resize_target
                                if cam_np.ndim != 3 or cam_np.shape[-1] != 3:
                                    cam_np = np.zeros((int(th), int(tw), 3), dtype=np.float32)
                                # Convert to float [0,1]
                                if cam_np.dtype == np.uint8:
                                    cam_np = cam_np.astype(np.float32) / 255.0
                                else:
                                    cam_np = np.clip(cam_np.astype(np.float32), 0.0, 1.0)
                                # If incoming frames are BGR, convert to RGB
                                if getattr(args_cli, 'camera_format', 'rgb') == 'bgr':
                                    cam_np = cam_np[:, :, ::-1]
                                # If frame already matches target, skip resizing; otherwise resize/pad
                                h, w = cam_np.shape[:2]
                                if (h == th) and (w == tw):
                                    img_resized = cam_np
                                else:
                                    img_resized = resize_with_padding_rgb(cam_np, target_hw=self.resize_target)
                                # Optional ImageNet normalization if used in training
                                if self.img_mean is not None and self.img_std is not None:
                                    img_resized = (img_resized - self.img_mean[None, None, :]) / (self.img_std[None, None, :] + 1e-8)
                                cam_chw = np.transpose(img_resized[..., :3], (2, 0, 1))
                                img_bchw = cam_chw.reshape(1, *cam_chw.shape)
                                batch["observation.images.rgb"] = torch.from_numpy(img_bchw).to(self.device).float()
                                proc_img_for_dump = img_bchw.copy()
                            with torch.no_grad():
                                act = self.policy.select_action(batch)
                            # Optional dump of exact model input at specified step
                            try:
                                if getattr(args_cli, 'dump_model_input_step', None) is not None and proc_img_for_dump is not None:
                                    target = int(args_cli.dump_model_input_step)
                                    if self.step_count == target:
                                        import os, cv2
                                        os.makedirs(args_cli.dump_model_input_dir, exist_ok=True)
                                        chw = proc_img_for_dump[0]
                                        hwc = np.transpose(chw, (1, 2, 0))
                                        # De-normalize for visualization if ImageNet stats were applied
                                        if self.img_mean is not None and self.img_std is not None:
                                            vis = (hwc * self.img_std[None, None, :]) + self.img_mean[None, None, :]
                                            vis = np.clip(vis, 0.0, 1.0)
                                        else:
                                            vis = np.clip(hwc, 0.0, 1.0)
                                        uint8 = (vis * 255.0).clip(0, 255).astype(np.uint8)
                                        bgr = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
                                        cv2.imwrite(os.path.join(args_cli.dump_model_input_dir, f"lerobot_step_{self.step_count:05d}.png"), bgr)
                                        # also save raw normalized array for analysis
                                        np.save(os.path.join(args_cli.dump_model_input_dir, f"lerobot_step_{self.step_count:05d}.npy"), chw)
                            except Exception as e:
                                print(f"[WARNING] Failed to dump model input at step {self.step_count}: {e}")
                            # Apply optional squashing and action gain (on arm dims 0..4)
                            if isinstance(act, torch.Tensor):
                                act = act.reshape(1, -1).to(self.device).float()
                                if getattr(args_cli, 'squash', False):
                                    act = torch.tanh(act)
                                gain = float(getattr(args_cli, 'action_gain', 1.0) or 1.0)
                                if gain != 1.0:
                                    act[:, :5] = act[:, :5] * gain
                            self.step_count += 1
                            return act
                    # Derive resize target and normalization from train config
                    # Read expected image shape and normalization from train_config.json
                    try:
                        import json
                        cfg_json = os.path.join(pretrained_dir, "train_config.json")
                        with open(cfg_json, 'r') as f:
                            cfgd = json.load(f)
                        shape = cfgd.get('policy', {}).get('input_features', {}).get('observation.images.rgb', {}).get('shape', [3, 0, 0])
                        h_cfg = int(shape[1]) if len(shape) >= 2 else 0
                        w_cfg = int(shape[2]) if len(shape) >= 3 else 0
                        if h_cfg > 0 and w_cfg > 0:
                            resize_hw = (h_cfg, w_cfg)
                        else:
                            resize_hw = (int(getattr(args_cli, 'rgb_height', 512)), int(getattr(args_cli, 'rgb_width', 512)))
                        use_imnet = bool(cfgd.get('dataset', {}).get('use_imagenet_stats', False))
                    except Exception:
                        # Fallbacks
                        resize_hw = (int(getattr(args_cli, 'rgb_height', 512)), int(getattr(args_cli, 'rgb_width', 512)))
                    use_imnet = bool(getattr(train_cfg.dataset, 'use_imagenet_stats', False))
                    try:
                        n_actions = int(getattr(train_cfg.policy, 'n_action_steps', 100) or 100)
                    except Exception:
                        n_actions = 100
                    # CLI override for replan frequency
                    if getattr(args_cli, 'replan_every', None):
                        try:
                            over = int(args_cli.replan_every)
                            if over > 0:
                                n_actions = over
                        except Exception:
                            pass
                    policy = LeRobotAdapter(lerobot_policy, device_str=(device.type if hasattr(device, 'type') else str(device)), resize_hw=resize_hw, use_imagenet_stats=use_imnet, env=env, n_action_steps=n_actions)
                    print("[INFO] Loaded LeRobot policy successfully (type from config)")
        except Exception as e:
            print(f"[INFO] SmolVLA detection/load failed (will try other formats): {e}")

        if policy is None:
            try:
                # Try robomimic format first
                policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)
                print("[INFO] Loaded robomimic checkpoint successfully")
            except Exception as e:
                print(f"[INFO] Not a robomimic checkpoint, trying direct PyTorch loading: {e}")
                # Load as simple PyTorch checkpoint
                checkpoint = torch.load(args_cli.checkpoint, map_location=device)
                
                # Create a simple BC policy that uses the trained model
                class BCPolicy:
                    def __init__(self, checkpoint, device, action_gain: float = 1.0, action_low=None, action_high=None, img_center: float = 0.5, img_scale: float = 1.0):
                        self.device = device
                        self.checkpoint = checkpoint
                        self.last_action = None  # For action smoothing
                        self.prev_action = None  # For momentum smoothing
                        self.momentum = 0.9      # Momentum factor for smoothing
                        self.action_gain = action_gain
                        self.action_low = action_low
                        self.action_high = action_high
                        self.img_center = float(img_center)
                        self.img_scale = float(img_scale)
                        self.img_mean = None
                        self.img_std = None
                        print(f"[INFO] Loaded checkpoint with {len(checkpoint)} parameters")
                        # Determine architecture
                        meta_arch = 'mlp'
                        meta_path = 'outputs/bc_training_meta.txt'
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, 'r') as f:
                                    for line in f:
                                        if line.startswith('arch='):
                                            meta_arch = line.strip().split('=')[1]
                                            break
                                print(f"[INFO] Detected training architecture: {meta_arch}")
                            except Exception as e:
                                print(f"[WARNING] Failed to read meta file: {e}")
 
                    # Fallback: infer from checkpoint keys when meta is missing
                    # Infer architecture from checkpoint keys (takes precedence over meta)
                    inferred_arch = None
                    if 'model_state_dict' in checkpoint:
                        try:
                            ks = list(checkpoint['model_state_dict'].keys())
                            if any(k.startswith('cnn.') for k in ks):
                                inferred_arch = 'vision'
                            elif any(k.startswith('input_layer.') for k in ks):
                                inferred_arch = 'mlp'
                        except Exception:
                            inferred_arch = None
                    if inferred_arch:
                        print(f"[INFO] Inferred architecture from checkpoint: {inferred_arch}")
                        meta_arch = inferred_arch

                    # If meta says 'vision', or keys indicate CNN, use vision
                    self.is_vision = (meta_arch == 'vision')
                    # Attempt to read training meta for image size and normalization
                    self.image_size = 128
                    meta_center = 0.5
                    meta_scale = 1.0
                    meta_width_mul = 1.0
                    meta_dropout = 0.05
                    try:
                        if os.path.exists('outputs/bc_training_meta.txt'):
                            with open('outputs/bc_training_meta.txt', 'r') as mf:
                                for line in mf:
                                    if line.startswith('image_size='):
                                        self.image_size = int(line.strip().split('=')[1])
                                    elif line.startswith('img_center='):
                                        meta_center = float(line.strip().split('=')[1])
                                    elif line.startswith('img_scale='):
                                        meta_scale = float(line.strip().split('=')[1])
                                    elif line.startswith('cnn_width_mul='):
                                        try:
                                            meta_width_mul = float(line.strip().split('=')[1])
                                        except Exception:
                                            pass
                                    elif line.startswith('dropout='):
                                        try:
                                            meta_dropout = float(line.strip().split('=')[1])
                                        except Exception:
                                            pass
                                    elif line.startswith('img_mean='):
                                        try:
                                            parts = line.strip().split('=')[1].split(',')
                                            self.img_mean = np.array([float(x) for x in parts], dtype=np.float32)
                                        except Exception:
                                            self.img_mean = None
                                    elif line.startswith('img_std='):
                                        try:
                                            parts = line.strip().split('=')[1].split(',')
                                            self.img_std = np.array([float(x) for x in parts], dtype=np.float32)
                                        except Exception:
                                            self.img_std = None
                    except Exception:
                        pass
                    # Override from meta
                    self.img_center = meta_center
                    self.img_scale = meta_scale
                    self.width_mul = meta_width_mul
                    self.dropout = meta_dropout

                    # If per-channel stats files exist, prefer them
                    try:
                        p_mean = 'outputs/img_mean.npy'
                        p_std = 'outputs/img_std.npy'
                        if os.path.exists(p_mean) and os.path.exists(p_std):
                            self.img_mean = np.load(p_mean)
                            self.img_std = np.load(p_std)
                    except Exception:
                        pass

                    # Derive input/output dims from scalers if available
                    inferred_state_dim = 6
                    inferred_action_dim = 6
                    try:
                        if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
                            inferred_state_dim = int(self.scaler.n_features_in_)
                    except Exception:
                        pass
                    try:
                        if self.act_scaler is not None and hasattr(self.act_scaler, 'n_features_in_'):
                            inferred_action_dim = int(self.act_scaler.n_features_in_)
                    except Exception:
                        pass

                    if self.is_vision:
                        self.net = LightVisionCNN(img_channels=3, state_dim=inferred_state_dim, action_dim=inferred_action_dim, width_mul=getattr(self, 'width_mul', 1.0), dropout=getattr(self, 'dropout', 0.05)).to(device)
                    else:
                        self.net = AdvancedBCNetwork(input_dim=inferred_state_dim, output_dim=inferred_action_dim, hidden_dims=[256, 256, 128]).to(device)
                    print(f"[INFO] Policy arch: {'vision' if self.is_vision else 'mlp'}; image_size={self.image_size}; state_dim={inferred_state_dim}, action_dim={inferred_action_dim}")
                    
                    # Load model weights
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        try:
                            # Remove 'network.' prefix if present
                            new_state = {}
                            for k, v in state_dict.items():
                                new_state[k[8:]] = v if k.startswith('network.') else v
                            # Prefer strict load; if fails, try non-strict
                            try:
                                self.net.load_state_dict(state_dict, strict=True)
                                print("[INFO] Loaded model weights successfully (strict)")
                            except Exception as e_strict:
                                print(f"[WARNING] Strict load failed: {e_strict}; trying non-strict")
                                self.net.load_state_dict(state_dict, strict=False)
                                print("[INFO] Loaded model weights successfully (non-strict)")
                        except Exception as e:
                            print(f"[WARNING] Failed to load state dict strictly: {e}, trying non-strict mapping")
                            try:
                                self.net.load_state_dict(new_state, strict=False)
                                print("[INFO] Loaded model weights with adjusted keys")
                            except Exception as e2:
                                print(f"[ERROR] Could not load weights: {e2}")
                    else:
                        print("[WARNING] No model_state_dict found in checkpoint")
                    
                    # Load scaler if available
                    self.scaler = None
                    self.act_scaler = None
                    try:
                        import pickle
                        # Try advanced scaler first
                        with open('outputs/obs_scaler_optimized.pkl', 'rb') as f:
                            self.scaler = pickle.load(f)
                        print("[INFO] Loaded optimized observation scaler")
                    except:
                        try:
                            # Fallback to simple scaler
                            with open('outputs/obs_scaler.pkl', 'rb') as f:
                                self.scaler = pickle.load(f)
                            print("[INFO] Loaded simple observation scaler")
                        except:
                            print("[WARNING] No scaler found, using raw observations")

                    # Load action scaler
                    try:
                        import pickle
                        with open('outputs/act_scaler_optimized.pkl', 'rb') as f:
                            self.act_scaler = pickle.load(f)
                        print("[INFO] Loaded optimized action scaler")
                    except:
                        print("[WARNING] No action scaler found; using raw action outputs")
                    
                    # Set all layers to eval mode
                    self.net.eval()
                
                def __call__(self, obs):
                    # Extract joint positions and optional image
                    if isinstance(obs, dict):
                        joint_pos = obs['joint_pos']
                        cam = obs.get('camera_rgb', None)
                    else:
                        joint_pos = obs
                        cam = None

                    raw_cam_np = None
                    proc_cam_t = None

                    # To numpy
                    if isinstance(joint_pos, torch.Tensor):
                        joint_pos = joint_pos.cpu().numpy()
                    elif isinstance(joint_pos, list):
                        joint_pos = np.array(joint_pos)
                    if len(joint_pos.shape) == 1:
                        joint_pos = joint_pos.reshape(1, -1)
                    
                    # Scale obs
                    if self.scaler is not None:
                        joint_pos_scaled = self.scaler.transform(joint_pos)
                    else:
                        joint_pos_scaled = joint_pos
                    
                    # Capture raw camera if present (for dumping), and preprocess for vision model
                    if cam is not None:
                        if isinstance(cam, torch.Tensor):
                            cam_np = cam.squeeze(0) if cam.shape[0] == 1 else cam
                            cam_np = cam_np.detach().cpu().numpy()
                        raw_cam_np = cam_np.copy()
                    if self.is_vision and raw_cam_np is not None:
                        cam = raw_cam_np
                        raw_min, raw_max = float(cam.min()), float(cam.max())
                        # If frames are BGR but we expect RGB, swap channels
                        if args_cli.camera_format == 'bgr':
                            cam = cam[:, :, ::-1]
                        if cam.max() > 1.0:
                            cam = cam / 255.0
                        try:
                            cam = cv2.resize(cam, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                        except Exception:
                            pass
                        # Enhance color saturation and contrast
                        # Convert to HSV for saturation adjustment
                        hsv = cv2.cvtColor(cam, cv2.COLOR_RGB2HSV).astype(np.float32)
                        # Increase saturation by 50%
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 1)
                        # Slightly increase value (brightness)
                        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 1)
                        # Convert back to RGB
                        cam = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
                        # No normalization - use enhanced values as-is
                        cam_chw = np.transpose(cam, (2, 0, 1))
                        cam_chw = cam_chw.reshape(1, *cam_chw.shape)
                        proc_cam_t = torch.FloatTensor(cam_chw).to(next(self.net.parameters()).device)
                        if self.step_count == 0:
                            proc_min, proc_max = float(proc_cam_t.min().cpu().item()), float(proc_cam_t.max().cpu().item())
                            print(f"[INFO] Using vision input; format={args_cli.camera_format}, raw[min,max]=({raw_min:.3f},{raw_max:.3f}), processed[min,max]=({proc_min:.3f},{proc_max:.3f}), proc_shape={list(proc_cam_t.shape)}")
                    
                    # Convert observations to torch tensor
                    joint_pos_tensor = torch.FloatTensor(joint_pos_scaled).to(next(self.net.parameters()).device)
                    
                    # Forward pass
                    with torch.no_grad():
                        if self.is_vision and proc_cam_t is not None:
                            # Vision model
                            action = self.net(joint_pos_tensor, proc_cam_t)
                        else:
                            # State-only model
                            action = self.net(joint_pos_tensor)
                    
                    # Apply squashing if enabled
                    if args_cli.squash:
                        action = torch.tanh(action)

                    # Convert back to numpy
                    action = action.cpu().numpy().flatten()
                    
                    # Apply action scaler inverse transform if available
                    if self.act_scaler is not None:
                        action = self.act_scaler.inverse_transform(action.reshape(1, -1)).flatten()
                    
                    # Apply action gain
                    action = action * self.action_gain
                    
                    # Apply momentum smoothing
                    if self.prev_action is not None:
                        action = self.momentum * self.prev_action + (1 - self.momentum) * action
                    self.prev_action = action.copy()
                    
                    # Clip actions using environment bounds if provided
                    if self.action_low is not None and self.action_high is not None:
                        clipped = np.clip(action, self.action_low, self.action_high)
                        saturated = bool(np.any(clipped <= self.action_low) or np.any(clipped >= self.action_high))
                    if self.step_count % 10 == 0 and saturated:
                        print(f"[WARNING] Action saturated at step {self.step_count}: before_clip={action}")
                    action = clipped

                    # Track saturation without altering action_gain
                    if 'saturated' in locals() and saturated:
                        self.consecutive_small_actions += 1
                    else:
                        self.consecutive_small_actions = 0

                    # Apply rate limiting
                    if args_cli.rate_limit > 0:
                        arm_diff = np.abs(action[:5] - self.prev_action[:5])
                        if np.any(arm_diff > args_cli.rate_limit):
                            print(f"[WARNING] Arm action rate limit exceeded at step {self.step_count}. Current: {action[:5]}, Previous: {self.prev_action[:5]}, Diff: {arm_diff}")
                            action[:5] = self.prev_action[:5] + np.sign(action[:5] - self.prev_action[:5]) * args_cli.rate_limit
                    if args_cli.rate_limit_gripper > 0:
                        gripper_diff = np.abs(action[5] - self.prev_action[5])
                        if gripper_diff > args_cli.rate_limit_gripper:
                            print(f"[WARNING] Gripper action rate limit exceeded at step {self.step_count}. Current: {action[5]}, Previous: {self.prev_action[5]}, Diff: {gripper_diff}")
                            action[5] = self.prev_action[5] + np.sign(action[5] - self.prev_action[5]) * args_cli.rate_limit_gripper

                    # Optionally dump RGB frames
                    if args_cli.dump_rgb and raw_cam_np is not None and (self.step_count % max(1, args_cli.dump_rgb_every) == 0):
                        try:
                            os.makedirs(args_cli.dump_rgb_dir, exist_ok=True)
                            if not args_cli.dump_only_model_input:
                                # Normalize raw to uint8 safely
                                raw_arr = raw_cam_np
                                if raw_arr.dtype != np.uint8:
                                    if raw_arr.max() <= 1.0:
                                        raw_arr = (raw_arr * 255.0).clip(0, 255)
                                    raw_arr = raw_arr.astype(np.uint8)
                                # Save assuming raw is RGB (convert to BGR for cv2)
                                raw_bgr_from_rgb = cv2.cvtColor(raw_arr, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(args_cli.dump_rgb_dir, f"step_{self.step_count:05d}_raw_assumeRGB.png"), raw_bgr_from_rgb)
                                # Save assuming raw is already BGR (write directly)
                                cv2.imwrite(os.path.join(args_cli.dump_rgb_dir, f"step_{self.step_count:05d}_raw_assumeBGR.png"), raw_arr)
                                # Save channel-swapped (R<->B)
                                raw_swapped = raw_arr[:, :, ::-1]
                                cv2.imwrite(os.path.join(args_cli.dump_rgb_dir, f"step_{self.step_count:05d}_raw_channelswapped.png"), raw_swapped)

                            # Save processed (and exact model input visualization)
                            if args_cli.dump_rgb_processed and proc_cam_t is not None:
                                proc_np = proc_cam_t[0].detach().cpu().numpy()  # CHW in [0,1]
                                proc_hwc = np.transpose(proc_np, (1, 2, 0))
                                proc_uint8 = (proc_hwc * 255.0).clip(0, 255).astype(np.uint8)
                                proc_bgr_from_rgb = cv2.cvtColor(proc_uint8, cv2.COLOR_RGB2BGR)
                                if not args_cli.dump_only_model_input:
                                    cv2.imwrite(os.path.join(args_cli.dump_rgb_dir, f"step_{self.step_count:05d}_proc_assumeRGB.png"), proc_bgr_from_rgb)
                                    cv2.imwrite(os.path.join(args_cli.dump_rgb_dir, f"step_{self.step_count:05d}_proc_assumeBGR.png"), proc_uint8)
                                    cv2.imwrite(os.path.join(args_cli.dump_rgb_dir, f"step_{self.step_count:05d}_proc_channelswapped.png"), proc_uint8[:, :, ::-1])
                                # Save exactly what model sees (in training assumed RGB)
                                cv2.imwrite(os.path.join(args_cli.dump_rgb_dir, f"step_{self.step_count:05d}_proc_model_input.png"), proc_bgr_from_rgb)

                            # Print channel stats at first dump
                            if self.step_count == 0:
                                r_mean = float(raw_arr[:, :, 2].mean())
                                g_mean = float(raw_arr[:, :, 1].mean())
                                b_mean = float(raw_arr[:, :, 0].mean())
                                print(f"[INFO] Raw channel means (B,G,R): ({b_mean:.1f}, {g_mean:.1f}, {r_mean:.1f})")
                        except Exception as e:
                            print(f"[WARNING] Failed to dump RGB at step {self.step_count}: {e}")

                    # New: dump exactly the model input at a specified step
                    try:
                        if getattr(args_cli, 'dump_model_input_step', None) is not None and proc_cam_t is not None:
                            target = int(args_cli.dump_model_input_step)
                            if self.step_count == target:
                                os.makedirs(args_cli.dump_model_input_dir, exist_ok=True)
                                proc_np = proc_cam_t[0].detach().cpu().numpy()  # CHW
                                hwc = np.transpose(proc_np, (1, 2, 0))
                                uint8 = (hwc * 255.0).clip(0, 255).astype(np.uint8)
                                bgr = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(args_cli.dump_model_input_dir, f"bc_step_{self.step_count:05d}.png"), bgr)
                                # Save raw CHW array for analysis
                                np.save(os.path.join(args_cli.dump_model_input_dir, f"bc_step_{self.step_count:05d}.npy"), proc_np)
                    except Exception as e:
                        print(f"[WARNING] Failed to dump exact model input at step {self.step_count}: {e}")

                    # New: dump raw camera frame at a specified step
                    try:
                        if getattr(args_cli, 'dump_raw_cam_step', None) is not None and raw_cam_np is not None:
                            target = int(args_cli.dump_raw_cam_step)
                            if self.step_count == target:
                                os.makedirs(args_cli.dump_raw_cam_dir, exist_ok=True)
                                if raw_cam_np.dtype != np.uint8:
                                    if raw_cam_np.max() <= 1.0:
                                        raw_cam_np = (raw_cam_np * 255.0).clip(0, 255)
                                    raw_cam_np = raw_cam_np.astype(np.uint8)
                                hwc = np.transpose(raw_cam_np, (1, 2, 0))
                                uint8 = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(args_cli.dump_raw_cam_dir, f"raw_cam_step_{self.step_count:05d}.png"), uint8)
                                np.save(os.path.join(args_cli.dump_raw_cam_dir, f"raw_cam_step_{self.step_count:05d}.npy"), raw_cam_np)
                    except Exception as e:
                        print(f"[WARNING] Failed to dump raw camera at step {self.step_count}: {e}")

                    self.step_count += 1
                    
                    return action
                
                def start_episode(self):
                    self.last_action = None  # Reset action memory
                    self.step_count = 0  # Reset step counter
                    self.consecutive_small_actions = 0  # Reset BC drift counter
            
            # Obtain env action bounds if available
            try:
                low = getattr(env.action_space, 'low', None)
                high = getattr(env.action_space, 'high', None)
                if low is not None and high is not None:
                    low = np.array(low, dtype=np.float32)
                    high = np.array(high, dtype=np.float32)
            except Exception:
                low = None; high = None
            policy = BCPolicy(checkpoint, device, action_gain=args_cli.action_gain, action_low=low, action_high=high, img_center=0.5, img_scale=1.0)
            # If meta provided img_center/scale, BCPolicy overrides these in init
            print("[INFO] Created BC policy with trained model")
    else:
        print("[WARNING] No checkpoint provided, skipping policy loading")
        return

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        terminated, traj = rollout(policy, env, args_cli.horizon, device)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 