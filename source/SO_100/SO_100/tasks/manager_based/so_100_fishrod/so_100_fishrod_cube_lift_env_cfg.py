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
from .so_100_fishrod_base_env_cfg import SO100LiftEnvCfg
from .so_100_fishrod_robot_cfg import SO100_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG
import os
from .so_100_fishrod_base_env_cfg import _ASSET_DIR

FISHROD_USDA = os.path.join(_ASSET_DIR, "fishrod.usda")


@configclass
class SO100FishRodCubeLiftEnvCfg(SO100LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Ensure robot stage path/jaw names match fish_rod_with_config.py usage
        _robot_cfg = dataclasses.replace(SO100_CFG, prim_path="/SOARM100")
        if _robot_cfg.init_state is None:
            _robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
        _robot_cfg.init_state = dataclasses.replace(
            _robot_cfg.init_state,
            pos=[-0.43, 0.0, 0.0],
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
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            scale=0.8,
            use_default_offset=True
        )
        self.commands.object_pose.body_name = "Moving_Jaw"
        self.commands.object_pose.debug_vis = False


@configclass
class SO100FishRodCubeLiftEnvCfg_PLAY(SO100FishRodCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1  # Tek ortam
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
