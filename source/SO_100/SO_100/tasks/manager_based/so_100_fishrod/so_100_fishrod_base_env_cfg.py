
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import yaml
from dataclasses import MISSING
from isaaclab.sensors.camera.camera_cfg import CameraCfg


from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg

from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg 
from . import mdp
from .so_100_fishrod_robot_cfg import SO100_CFG

# Just use standard mdp.image - cleanup will be done every 300 steps only
def image_with_memory_cleanup(env, sensor_cfg: SceneEntityCfg, data_type: str = "rgb"):
    """Wrapper around mdp.image (cleanup done at 300 step intervals instead)."""
    return mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type)

# Paths to local USD assets to mirror Isaac Sim scene
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(_THIS_DIR, "asset")
_CFG_DIR = os.path.join(_ASSET_DIR, "configuration")
LIVESTREAM_BASE_USD = os.path.join(_CFG_DIR, "Livestream_test_base.usd")
LIVESTREAM_PHYSICS_USD = os.path.join(_CFG_DIR, "Livestream_test_physics.usd")
LIVESTREAM_SENSOR_USD = os.path.join(_CFG_DIR, "Livestream_test_sensor.usd")
BASE_PLATE_USD = os.path.join(_CFG_DIR, "base_plate_layer1-v5.tmp.usd")
FISHROD_USDA = os.path.join(_ASSET_DIR, "fishrod.usda")
# Prefer cleaned/tuned USD if available (local asset first)
FISHROD_CLEAN_LOCAL = os.path.join(_ASSET_DIR, "fishrod_clean.usda")
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../../.."))
_FISHROD_CLEAN_CANDIDATE = os.path.join(_REPO_ROOT, "fishrod", "fishrod_clean.usda")
FISHROD_USD_SELECTED = (
    FISHROD_CLEAN_LOCAL if os.path.isfile(FISHROD_CLEAN_LOCAL)
    else (_FISHROD_CLEAN_CANDIDATE if os.path.isfile(_FISHROD_CLEAN_CANDIDATE) else FISHROD_USDA)
)

# Load fishrod spawn settings from fishrod/config.yaml (to mirror fish_rod_with_config.py)
_FISHROD_CFG_PATH = os.path.join(_REPO_ROOT, "fishrod", "config.yaml")

def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles (degrees) to quaternion (w, x, y, z)"""
    import math
    
    # Convert to radians
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    
    # Compute quaternion components
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (w, x, y, z)

def generate_randomized_pose(config):
    """Generate randomized position and orientation based on config"""
    import random
    import numpy as np
    
    fishrod_config = config.get('fishrod', {})
    randomization_config = fishrod_config.get('randomization', {})
    
    # Check if randomization is enabled
    if not randomization_config.get('enabled', False):
        # Return base values without randomization
        base_position = fishrod_config.get('base_position', [0.0, 0.0, 0.0])
        base_orientation = fishrod_config.get('base_orientation', [0.0, 0.0, 0.0])
        return base_position, base_orientation
    
    # Set random seed if specified
    seed = randomization_config.get('seed')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Get base values
    base_position = fishrod_config.get('base_position', [0.0, 0.0, 0.0])
    base_orientation = fishrod_config.get('base_orientation', [0.0, 0.0, 0.0])
    
    # Generate position offsets
    pos_offset_config = randomization_config.get('position_offset_range', {})
    x_offset = random.uniform(pos_offset_config.get('x', [0.0, 0.0])[0], pos_offset_config.get('x', [0.0, 0.0])[1])
    y_offset = random.uniform(pos_offset_config.get('y', [0.0, 0.0])[0], pos_offset_config.get('y', [0.0, 0.0])[1])
    z_offset = random.uniform(pos_offset_config.get('z', [0.0, 0.0])[0], pos_offset_config.get('z', [0.0, 0.0])[1])
    
    # Apply offsets to base position
    randomized_position = [
        base_position[0] + x_offset,
        base_position[1] + y_offset,
        base_position[2] + z_offset
    ]
    
    # Generate orientation offsets
    orient_offset_config = randomization_config.get('orientation_offset_range', {})
    roll_offset = random.uniform(orient_offset_config.get('roll', [0.0, 0.0])[0], orient_offset_config.get('roll', [0.0, 0.0])[1])
    pitch_offset = random.uniform(orient_offset_config.get('pitch', [0.0, 0.0])[0], orient_offset_config.get('pitch', [0.0, 0.0])[1])
    yaw_offset = random.uniform(orient_offset_config.get('yaw', [0.0, 0.0])[0], orient_offset_config.get('yaw', [0.0, 0.0])[1])
    
    # Apply offsets to base orientation
    randomized_orientation = [
        base_orientation[0] + roll_offset,
        base_orientation[1] + pitch_offset,
        base_orientation[2] + yaw_offset
    ]
    
    print(f"🎲 Randomization enabled:")
    print(f"   Base position: {base_position}")
    print(f"   Randomized position: {randomized_position}")
    print(f"   Position offsets: [{x_offset:.3f}, {y_offset:.3f}, {z_offset:.3f}]")
    print(f"   Base orientation: {base_orientation}")
    print(f"   Randomized orientation: {randomized_orientation}")
    print(f"   Orientation offsets: [{roll_offset:.1f}, {pitch_offset:.1f}, {yaw_offset:.1f}] degrees")
    
    return randomized_position, randomized_orientation

def _load_fishrod_spawn_cfg():
    try:
        with open(_FISHROD_CFG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    fishrod_cfg = cfg.get("fishrod", {})
    scale = fishrod_cfg.get("scale", 0.15)
    
    # Use randomized position if enabled
    randomized_position, randomized_orientation = generate_randomized_pose(cfg)
    
    return float(scale), tuple(randomized_position), tuple(randomized_orientation)

_FISHROD_SCALE, _FISHROD_BASE_POS, _FISHROD_BASE_ORIENT = _load_fishrod_spawn_cfg()

def _load_camera_cfg():
    try:
        with open(_FISHROD_CFG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    camera_cfg = cfg.get("camera", {})
    eye = tuple(camera_cfg.get("eye", [1.1, 0.0, 1.6]))
    target = tuple(camera_cfg.get("target", [0.0, 0.0, 0.0]))
    return eye, target

# Load external camera settings from robot_camera_config.yaml (if present)
_ROBOT_CAM_CFG_PATH = os.path.join(_REPO_ROOT, "fishrod", "robot_camera_config.yaml")
def _load_ext_camera_cfg():
    try:
        with open(_ROBOT_CAM_CFG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    cam_cfg = ((cfg.get("camera", {}) or {}).get("external", {}) or {})
    root_path = cam_cfg.get("root_path", "/World/ExtCamera")
    pos = tuple(cam_cfg.get("position", [1.1, 0.0, 1.6]))
    # Orientation as XYZW quaternion in config, convert to (w, x, y, z)
    q_xyzw = cam_cfg.get("orientation", [0.5, 0.5, 0.5, 0.5])
    if isinstance(q_xyzw, (list, tuple)) and len(q_xyzw) == 4:
        q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
    else:
        q_wxyz = (1.0, 0.0, 0.0, 0.0)
    settings = cam_cfg.get("settings", {}) or {}
    focal_length = float(settings.get("focal_length", 50.0))
    fov = float(settings.get("fov", 60.0))
    near_clip = float(settings.get("near_clip", 0.1))
    far_clip = float(settings.get("far_clip", 1000.0))
    return root_path, pos, q_wxyz, focal_length, fov, near_clip, far_clip

_EXT_CAM_ROOT, _EXT_CAM_POS, _EXT_CAM_ROT, _EXT_CAM_FOCAL, _EXT_CAM_FOV, _EXT_CAM_NEAR, _EXT_CAM_FAR = _load_ext_camera_cfg()

# Always use local SimpleRoom from asset folder for environment
_LOCAL_SIMPLE_ROOM = os.path.join(_ASSET_DIR, "SimpleRoom.usda")

def _pick_local_simple_room() -> str:
    # Use SimpleRoom.usda directly (has proper scene content)
    print(f"[DEBUG] Checking for SimpleRoom USD file:")
    print(f"[DEBUG]   SimpleRoom path: {_LOCAL_SIMPLE_ROOM}")
    print(f"[DEBUG]   SimpleRoom exists: {os.path.isfile(_LOCAL_SIMPLE_ROOM)}")
    
    if os.path.isfile(_LOCAL_SIMPLE_ROOM):
        print(f"[DEBUG] ✅ Using SimpleRoom.usda: {_LOCAL_SIMPLE_ROOM}")
        return _LOCAL_SIMPLE_ROOM
    
    error_msg = f"Local SimpleRoom.usda not found at: {_LOCAL_SIMPLE_ROOM}"
    print(f"[DEBUG] ❌ ERROR: {error_msg}")
    raise RuntimeError(error_msg)

# Load environment settings from robot_camera_config.yaml (if present)
_ENV_BG_PATH, _ENV_REF_PATH = (None, None)
try:
	with open(_ROBOT_CAM_CFG_PATH, "r") as f:
		_cfg_tmp = yaml.safe_load(f) or {}
	_env_cfg = (_cfg_tmp.get("environment", {}) or {})
	_ENV_BG_PATH = _env_cfg.get("background_path", "/Isaac/Environments/Grid/default_environment.usd")
	_ENV_REF_PATH = _env_cfg.get("reference_path", "/Environment")
except Exception:
	_ENV_BG_PATH = "/Isaac/Environments/Grid/default_environment.usd"
	_ENV_REF_PATH = "/Environment"

def _resolve_env_usd_path(bg_path: str) -> str:
    """Compose a valid Nucleus path from ISAAC_NUCLEUS_DIR and a background_path.
    Accepts values like "/Isaac/Environments/..." or "/Environments/..." and avoids duplicating "Isaac".
    """
    base = ISAAC_NUCLEUS_DIR.rstrip("/")
    try:
        # Prefer local simple_room.usd in asset folder when requested
        local_simple = os.path.join(_ASSET_DIR, "simple_room.usd")
        local_simple_wrapper = os.path.join(_ASSET_DIR, "simple_room_wrapper.usda")
        if isinstance(bg_path, str) and (
            "Simple_Room/simple_room.usd" in bg_path or bg_path.endswith("simple_room.usd")
        ):
            # Enforce local asset usage; raise if missing
            if os.path.isfile(local_simple_wrapper):
                return local_simple_wrapper
            if os.path.isfile(local_simple):
                return local_simple
            raise RuntimeError(
                f"Environment USD requested as Simple_Room but local file not found: {local_simple}"
            )
        if not isinstance(bg_path, str) or len(bg_path) == 0:
            return f"{base}/Environments/Grid/default_environment.usd"
        # Known missing content on some installs → fallback to Grid
        if "Simple_Room/simple_room.usd" in bg_path:
            return f"{base}/Environments/Grid/default_environment.usd"
        if bg_path.startswith("/Isaac/"):
            # ISAAC_NUCLEUS_DIR already ends with "/Isaac"
            return f"{base}{bg_path[len('/Isaac') :]}"
        if bg_path.startswith("/Environments/"):
            return f"{base}{bg_path}"
        # If given an absolute path under another root, return as-is
        if bg_path.startswith("/"):
            return bg_path
        # Otherwise join as relative under /Environments
        return f"{base}/Environments/{bg_path}"
    except Exception:
        return f"{base}/Environments/Grid/default_environment.usd"


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and an object."""
    robot = ArticulationCfg(
        prim_path="/SOARM100",
        spawn=SO100_CFG.spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.43, 0.0, 0.0),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # 90-degree rotation around Z-axis
            joint_pos={
                "Rotation": 0.0,
                "Pitch": 0.0,
                "Elbow": 0.0,
                "Wrist_Pitch": 0.0,
                "Wrist_Roll": 0.0,
                "Jaw": 0.0,
            },
            joint_vel={".*": 0.0}
        ),
        actuators=SO100_CFG.actuators
    )
    fishrod = AssetBaseCfg(
        prim_path="/World/FishRod",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_FISHROD_BASE_POS,
            rot=euler_to_quaternion(*_FISHROD_BASE_ORIENT),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=FISHROD_USD_SELECTED,
            scale=(_FISHROD_SCALE, _FISHROD_SCALE, _FISHROD_SCALE),
        ),
    )
    # Background environment with ground (simple_room includes floor)
    # NOTE: Must NOT use /World prefix - scene manager will replicate under /World/envs/env_N
    ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        spawn=sim_utils.UsdFileCfg(usd_path=_pick_local_simple_room())
    )
    camera = CameraCfg(
        prim_path="/World/camera",
        update_period=0.0,
        width=512,
        height=512,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(-0.1, -1.0, 0.9),
            rot=(0.89101,0.45399,0,0),
            convention="local"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=70.0,
            focus_distance=1.0,
            horizontal_aperture=36.0,
            clipping_range=(0.1, 10.0)
        ),
        history_length=1,
        debug_vis=False
    )
   
    jaw_camera = CameraCfg(
        prim_path="/SOARM100/Fixed_Jaw/jaw_camera",
        update_period=0.0,
        width=512,
        height=512,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.04, 0.07),
            rot=(0.57567, -0.41062, 0.45923, 0.53769),
            convention="local"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0,
            focus_distance=1.0,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 1000.0)
        ),
        history_length=1,
        debug_vis=False
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="Moving_Jaw",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        scale=0.1
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Jaw"],
        open_command_expr=1.0,
        close_command_expr=0.0
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # Include both joint_pos and camera_rgb to match the model architecture
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        camera_rgb = ObsTerm(
            func=image_with_memory_cleanup,  # Use our wrapper function
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
            },
        )
        jaw_camera_rgb = ObsTerm(
            func=image_with_memory_cleanup,  # Use our wrapper function
            params={
                "sensor_cfg": SceneEntityCfg("jaw_camera"),
                "data_type": "rgb",
            },
        )
        
        # Remove other observations that are not in the dataset
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # Randomize fishrod position and orientation on each reset
    randomize_fishrod = EventTerm(
        func=lambda env, env_ids: _randomize_fishrod_pose_event(env, env_ids),
        mode="reset"
    )
    
    # FULL GPU memory reset every ~400 steps (6.67 seconds at 60Hz)
    clear_gpu_cache = EventTerm(
        func=lambda env, env_ids: _clear_gpu_cache_event(env, env_ids),
        mode="interval",
        interval_range_s=(6.67, 6.67),  # ~400 steps at 60Hz
    )

def _randomize_fishrod_pose_event(env, env_ids):
    """Randomize fishrod position and orientation on reset."""
    try:
        import torch
        import yaml
        
        # Reload config for fresh randomization
        with open(_FISHROD_CFG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        
        # Generate new randomized pose
        randomized_position, randomized_orientation = generate_randomized_pose(cfg)
        
        # Convert to tensors
        pos_tensor = torch.tensor(randomized_position, dtype=torch.float32, device=env.device)
        quat_tensor = torch.tensor(euler_to_quaternion(*randomized_orientation), dtype=torch.float32, device=env.device)
        
        # Get fishrod asset
        fishrod = env.scene["fishrod"]
        
        # Set position and orientation for all reset environments
        for env_id in env_ids:
            fishrod.data.root_pos_w[env_id] = pos_tensor
            fishrod.data.root_quat_w[env_id] = quat_tensor
        
        # Write changes to simulation
        fishrod.write_root_pose_to_sim(fishrod.data.root_pos_w, fishrod.data.root_quat_w, env_ids=env_ids)
        
    except Exception as e:
        print(f"[WARNING] Fishrod randomization failed: {e}")

def _clear_gpu_cache_event(env, env_ids):
    """Aggressively clear GPU cache to prevent memory overflow with dual cameras."""
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            # CRITICAL: Force Python GC BEFORE CUDA empty_cache
            gc.collect()
            gc.collect()
            gc.collect()
            # Now empty CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Final GC pass
            gc.collect()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-4, "num_steps": 12000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 12000}
    )

@configclass
class SO100LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5)  # Tek ortam
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2  # 60Hz / 2 = 30Hz control (matches dataset frequency)
        self.episode_length_s = 20.0  # Increased from 5.0 to 20.0 seconds for longer episodes
        self.sim.dt = 1.0 / 60.0  # simulation.py ile aynı
        self.sim.render_interval = self.decimation
        self.sim.device = "cpu"  # AMD GPU için CPU kullan
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.01
        self.sim.physx.solver_type = "tgs"
        self.sim.physx.solver_position_iteration_count = 20
        self.sim.physx.solver_velocity_iteration_count = 1
        self.sim.physx.stabilization_threshold = 0.0001
        # Tight contact margins to mirror fish_rod_with_config.py
        self.sim.physx.contact_offset = 0.0001
        self.sim.physx.rest_offset = 0.0
        # Pull main camera settings from fishrod/config.yaml
        _eye, _lookat = _load_camera_cfg()
        self.viewer.eye = _eye
        self.viewer.lookat = _lookat
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        # Static USD-driven transforms; no config overrides

        # Apply renderer/viewport settings
        self._apply_renderer_settings()

    def _apply_renderer_settings(self) -> None:
        """Configure renderer settings (AA, AO, shadows, tone mapping) if available."""
        try:
            import carb.settings  # type: ignore
            settings = carb.settings.get_settings()
            
            # Anti-aliasing / sampling
            settings.set("/app/renderer/multiSamples", 8)
            settings.set("/rtx/post/aa/op", 3)
            
            # RTX Indirect Diffuse Lighting - Ambient Occlusion
            # Enable Indirect Diffuse first
            settings.set("/rtx/indirectDiffuse/enabled", True)
            # Then enable AO within Indirect Diffuse (this is the UI checkbox)
            settings.set("/rtx/indirectDiffuse/ambientOcclusion/enabled", True)
            settings.set("/rtx/indirectDiffuse/ambientOcclusion/radius", 0.5)
            settings.set("/rtx/indirectDiffuse/ambientOcclusion/intensity", 1.0)
            
            # Try alternative paths as well
            settings.set("/rtx/indirectDiffuse/ao/enabled", True)
            settings.set("/rtx/indirectDiffuse/aoEnabled", True)
            
            # Standard SSAO (Post-process)
            settings.set("/rtx/ambientOcclusion/enabled", True)
            settings.set("/rtx/ambientOcclusion/intensity", 1.0)
            settings.set("/rtx/ambientOcclusion/radius", 0.5)
            settings.set("/rtx/ambientOcclusion/falloff", 0.5)
            settings.set("/rtx/ambientOcclusion/quality", 2)
            
            # Shadows
            settings.set("/rtx/shadows/enabled", True)
            settings.set("/rtx/contactShadows/enabled", True)
            
            # Tone mapping / exposure
            settings.set("/rtx/post/toneMapping/enabled", True)
            settings.set("/rtx/post/toneMapping/exposure", 1.0)
            
            # Bloom and color correction
            settings.set("/rtx/post/bloom/enabled", False)
            settings.set("/rtx/post/colorCorrection/enabled", True)
            
            # Debug: Print current AO settings
            ao_enabled = settings.get("/rtx/indirectDiffuse/ambientOcclusion/enabled")
            print(f"[INFO] Indirect Diffuse AO enabled: {ao_enabled}")
            print("[INFO] Render settings applied: Indirect Diffuse AO + SSAO enabled")
        except Exception as exc:
            print(f"[WARNING] Renderer settings could not be applied: {exc}")
