
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
from .so_100_robot_cfg import SO100_CFG

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and an object."""
    robot = ArticulationCfg(
        prim_path="/SOARM100",
        spawn=SO100_CFG.spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
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
    ee_frame = FrameTransformerCfg(
        prim_path="/SOARM100/Moving_Jaw",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/task_cube",
                name="task_cube",
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0))
            )
        ]
    )
    object = RigidObjectCfg(
        prim_path="/World/task_cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.09, 0.1)),  # Increased z to 0.1
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        )
    )
    plate = AssetBaseCfg(
        prim_path="/World/plate",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.27, -0.14, 0.008), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.0075),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            )
        )
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd")
    )
    
    
  
    
    camera = CameraCfg(
        prim_path="/World/camera",
        update_period=0.0,
        width=640,
        height=360,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(1.1, 0.0, 1.6),
            rot=(0.68301, 0.18301, 0.18301, 0.68301),
            convention="local"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=50.0,
            focus_distance=1.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 10.0)
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
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
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

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=2)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=25.0)
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
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.5, "asset_cfg": SceneEntityCfg("object")}
    )

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
        self.decimation = 1
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
        self.sim.physx.contact_offset = 0.04
        self.sim.physx.rest_offset = 0.001
        self.viewer.eye = (1.1, 0.0, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        # Apply renderer/viewport settings
        self._apply_renderer_settings()

    def _apply_renderer_settings(self) -> None:
        """Configure renderer settings (AA, AO, shadows, tone mapping) if available."""
        try:
            import carb.settings  # type: ignore
            settings = carb.settings.get_settings()
            # Anti-aliasing / sampling
            settings.set("/app/renderer/multiSamples", 8)                 # MSAA 8x (RTX Real-Time)
            settings.set("/rtx/post/aa/op", 3)                            # 0: none, 1:FXAA, 2:TAA, 3:TAA+ (varsa)
            # Ambient occlusion and shadows
            settings.set("/rtx/ambientOcclusion/enabled", True)
            settings.set("/rtx/shadows/enabled", True)
            settings.set("/rtx/contactShadows/enabled", True)
            # Tone mapping / exposure
            settings.set("/rtx/post/toneMapping/enabled", True)
            settings.set("/rtx/post/toneMapping/exposure", 1.0)
            # Bloom and color correction
            settings.set("/rtx/post/bloom/enabled", False)
            settings.set("/rtx/post/colorCorrection/enabled", True)
            # Optional: switch to ray-traced GI/path tracing if desired (may be expensive)
            # settings.set("/rtx/hydra/renderer", "RayTracedLighting")    # or "PathTracing" if supported
        except Exception as exc:
            print(f"Renderer settings could not be applied: {exc}")
