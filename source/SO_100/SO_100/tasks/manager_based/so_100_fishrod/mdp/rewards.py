# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward


def payload_position_command_error(env: ManagerBasedRLEnv, command_name: str, payload_cfg: SceneEntityCfg = SceneEntityCfg("payload")) -> torch.Tensor:
    """Penalize tracking of the payload position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the payload (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # Get payload position using the existing observation function
    from .observations import payload_world_position
    curr_pos_w = payload_world_position(env, payload_cfg)
    
    # Get command from command manager
    command = env.command_manager.get_command(command_name)
    
    # For payload, we assume the command is in world frame directly
    # If you need to transform from robot frame, uncomment the lines below:
    # robot: RigidObject = env.scene["robot"]
    # des_pos_b = command[:, :3]
    # des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    
    # For now, assuming command is already in world frame
    des_pos_w = command[:, :3]
    
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def payload_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, payload_cfg: SceneEntityCfg = SceneEntityCfg("payload")
) -> torch.Tensor:
    """Reward tracking of the payload position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the payload (in world frame) and maps it with a tanh kernel.
    """
    # Get payload position using the existing observation function
    from .observations import payload_world_position
    curr_pos_w = payload_world_position(env, payload_cfg)
    
    # Get command from command manager
    command = env.command_manager.get_command(command_name)
    
    # For payload, we assume the command is in world frame directly
    # If you need to transform from robot frame, uncomment the lines below:
    # robot: RigidObject = env.scene["robot"]
    # des_pos_b = command[:, :3]
    # des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    
    # For now, assuming command is already in world frame
    des_pos_w = command[:, :3]
    
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def payload_orientation_command_error(env: ManagerBasedRLEnv, command_name: str, payload_cfg: SceneEntityCfg = SceneEntityCfg("payload")) -> torch.Tensor:
    """Penalize tracking payload orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the payload (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # Get payload orientation from RigidObject if available
    if payload_cfg.name in env.scene:
        payload_obj: RigidObject = env.scene[payload_cfg.name]
        curr_quat_w = payload_obj.data.root_state_w[:, 3:7]
    else:
        # If payload is not a RigidObject, return zeros (no orientation tracking)
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    # Get command from command manager
    command = env.command_manager.get_command(command_name)
    
    # For payload, we assume the command orientation is in world frame directly
    # If you need to transform from robot frame, uncomment the lines below:
    # robot: RigidObject = env.scene["robot"]
    # des_quat_b = command[:, 3:7]
    # des_quat_w = quat_mul(robot.data.root_state_w[:, 3:7], des_quat_b)
    
    # For now, assuming command orientation is already in world frame
    des_quat_w = command[:, 3:7]
    
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def payload_is_lifted(env: ManagerBasedRLEnv, minimal_height: float, payload_cfg: SceneEntityCfg = SceneEntityCfg("payload")) -> torch.Tensor:
    """Reward the agent for lifting the payload above the minimal height."""
    from .observations import payload_world_position
    payload_pos_w = payload_world_position(env, payload_cfg)
    return torch.where(payload_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def payload_jaw_distance(
    env: ManagerBasedRLEnv,
    std: float,
) -> torch.Tensor:
    """Reward the agent for reducing distance between payload and Fixed_Jaw using tanh-kernel.
    
    The reward increases as the distance decreases. Uses tanh kernel for smooth reward signal.
    """
    from .observations import payload_world_position, fixed_jaw_world_position
    
    # Get positions with explicit SceneEntityCfg
    payload_cfg = SceneEntityCfg("payload")
    robot_cfg = SceneEntityCfg("robot")
    payload_pos_w = payload_world_position(env, payload_cfg)
    jaw_pos_w = fixed_jaw_world_position(env, robot_cfg)
    
    # Calculate L2 distance
    distance = torch.norm(payload_pos_w - jaw_pos_w, dim=1)
    
    # Tanh kernel: 1 when distance=0, approaches 0 as distance increases
    return 1 - torch.tanh(distance / std)


def payload_jaw_distance_error(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize distance between payload and Fixed_Jaw using L2-norm.
    
    Returns the L2 distance as a penalty (higher distance = higher penalty).
    """
    from .observations import payload_world_position, fixed_jaw_world_position
    
    # Get positions with explicit SceneEntityCfg
    payload_cfg = SceneEntityCfg("payload")
    robot_cfg = SceneEntityCfg("robot")
    payload_pos_w = payload_world_position(env, payload_cfg)
    jaw_pos_w = fixed_jaw_world_position(env, robot_cfg)
    
    # Calculate and return L2 distance as penalty
    return torch.norm(payload_pos_w - jaw_pos_w, dim=1)
