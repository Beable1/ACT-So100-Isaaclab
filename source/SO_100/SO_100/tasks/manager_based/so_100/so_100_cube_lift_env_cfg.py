import copy
import dataclasses
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from .so_100_base_env_cfg import SO100LiftEnvCfg
from .so_100_robot_cfg import SO100_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG

@configclass
class SO100CubeLiftEnvCfg(SO100LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _robot_cfg = dataclasses.replace(SO100_CFG, prim_path="/SOARM100")
        if _robot_cfg.init_state is None:
            _robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
        _robot_cfg.init_state = dataclasses.replace(
            _robot_cfg.init_state,
            pos=(0.0, 0.0, 0.0),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # 90 derece Z ekseninde
            joint_pos=SO100_CFG.init_state.joint_pos,
            joint_vel={".*": 0.0}
        )
        self.scene.robot = _robot_cfg
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=2.0,
            use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            open_command_expr={"Jaw": 0.3},
            close_command_expr={"Jaw": 0.0}
        )
        self.commands.object_pose.body_name = "Moving_Jaw"
        self.commands.object_pose.debug_vis = False
        self.scene.object = RigidObjectCfg(
            prim_path="/World/task_cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.09, 0.05)),  # z=0.03 -> 0.05
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False
                ),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.1,
                    density=1000.0
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True  # Çarpışma etkinleştirildi
                )
            )
        )
        marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        marker_cfg.markers = {
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            )
        }
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.cube_marker = FrameTransformerCfg(
            prim_path="/World/task_cube",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/task_cube",
                    name="cube",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

@configclass
class SO100CubeLiftEnvCfg_PLAY(SO100CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1  # Tek ortam
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
