# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def payload_world_position(
    env: ManagerBasedRLEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    payload_frame_cfg: SceneEntityCfg = SceneEntityCfg("payload_frame"),
) -> torch.Tensor:
    """World-frame position of the payload, preferring scene tensors (RigidObject) and
    falling back to a FrameTransformer target if needed.
    """
    # Use direct string names to avoid issues with SceneEntityCfg parameters
    payload_name = "payload"
    payload_frame_name = "payload_frame"
    
    # Prefer dynamic tensors from a wrapped RigidObject if present
    try:
        payload_obj: RigidObject = env.scene[payload_name]
        return payload_obj.data.root_pos_w[:, :3]
    except KeyError:
        pass
    
    # Fallback to frame transformer sensor if available
    try:
        frame: FrameTransformer = env.scene[payload_frame_name]
        return frame.data.target_pos_w[..., 0, :]
    except KeyError:
        pass
    
    # If neither available, return zeros to avoid crashes
    return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)


def fixed_jaw_world_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World-frame position of Fixed_Jaw.
    
    Gets the position directly from the robot's articulation body states.
    """
    # Use direct string name to avoid issues with SceneEntityCfg parameters
    robot_name = "robot"
    
    try:
        robot: Articulation = env.scene[robot_name]
    except KeyError:
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    
    # Find body index for Fixed_Jaw
    fixed_jaw_idx = None
    
    for i, body_name in enumerate(robot.body_names):
        if "Fixed_Jaw" in body_name or body_name == "Fixed_Jaw":
            fixed_jaw_idx = i
            break
    
    # Get position from body state
    if fixed_jaw_idx is not None:
        return robot.data.body_state_w[:, fixed_jaw_idx, :3]
    
    # If not found, return zeros to avoid crashes
    return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
