from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.mdp.rewards import *


# region -- from isaaclab --

# def track_lin_vel_xy_exp == gym: _reward_tracking_lin_vel, 惩罚当前机器人在X、Y方向速度与命令不一致
# params["std"] = math.sqrt(0.25)  # 0.25 等效于 std ** 2


# def track_ang_vel_z_exp == gym: _reward_tracking_ang_vel, 惩罚当前机器人在角度转向速度与命令不一致
# params["std"] = math.sqrt(0.25)  # 0.25 等效


# def lin_vel_z_l2 == gym: _reward_lin_vel_z, 惩罚机器人在Z轴上的速度 对应现象为机器人上下起伏很大


# def ang_vel_xy_l2 == gym: _reward_ang_vel_xy, 惩罚机器人在X轴和Y轴上的角速度 对应现象为遏制机器人左右晃动和前后晃动


# def flat_orientation_l2 == gym: _reward_orientation, 鼓励机器人与初始姿态的基座方向一致


# def joint_torques_l2 == gym: _reward_torques, 机器人运控各电机输出的力矩的平方和, 让模型找到最省力矩的方案
# params["asset_cfg"].joint_names = self.joint_names # or leg_joint_names + wheel_joint_names or ".*"


# def joint_vel_l2 == gym: _reward_dof_vel, 惩罚关节速度


# def joint_acc_l2 == gym: _reward_dof_acc, 惩罚关节加速度


# def base_height_l2 == gym: _reward_base_height, 惩罚基座高度不保持在期望的高度上
# params["target_height"] = 0.4
# params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
# params["asset_cfg"].body_names = ["base"]


# undesired_contacts == gym: _reward_collision, 惩罚碰撞
# params["asset_cfg"].body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]
# params["threshold"] = 0.1  # isaaclab 的默认是1.0


# def action_rate_l2 == gym: _reward_action_rate, 惩罚动作变化率


# def joint_pos_limits == gym: _reward_dof_pos_limits, 惩罚关节位置超出限制
# params["asset_cfg"].joint_names = leg_joint_names

# endregion


# region -- custom reward --

def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 需要修改 params["asset_cfg"].joint_names = leg_joint_names
    use_gravity_gating: bool = False,
    gating_max: float = 0.7,
) -> torch.Tensor:
    """ == gym: _reward_stand_still, 惩罚速度命令很小的时候基座应该维持默认姿态 """

    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    if use_gravity_gating:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, gating_max) / gating_max
    return reward


def hip_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = hip_joint_names
) -> torch.Tensor:
    """ == gym: _reward_hip_default, 惩罚髋关节不在默认位置 """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids  # 这里配置成 hip joints
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return torch.sum(torch.square(q - q0), dim=1)


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = leg_joint_names
) -> torch.Tensor:
    """ 惩罚各个关节不在默认位置 """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return torch.sum(torch.square(q - q0), dim=1)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    use_gravity_gating: bool = False,
    gating_max: float = 0.7,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    if use_gravity_gating:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, gating_max) / gating_max
    return reward


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ratio: float = 5.0, # robotlab中默认值为4.0
    use_gravity_gating: bool = False,
    gating_max: float = 0.7,
) -> torch.Tensor:
    """ == gym: _reward_feet_stumble """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > ratio * forces_z, dim=1).float()

    if use_gravity_gating:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, gating_max) / gating_max
    return reward


def hip_action_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # params["asset_cfg"].joint_names = hip_joint_names
) -> torch.Tensor:
    """Penalize hip joint actions (L2 squared)."""
    action = env.action_manager.action
    joint_ids = asset_cfg.joint_ids  # 配成 hip joints 对应的 action 维度

    reward = torch.sum(torch.square(action[:, joint_ids]), dim=1)
    return reward











# class RewardsGo2w:
#     only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
#     tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
#     soft_dof_pos_limit = 1.0  # percentage of urdf limits, values above this limit are penalized
#     soft_dof_vel_limit = 1.
#     soft_torque_limit = 1.
#     base_height_target = 0.4
#     max_contact_force = 100.  # forces above this value are penalized
#
#     class ScalesGo2W:
#         termination = 0  # 25/8/23 zsy说不用加
#         tracking_lin_vel = 1.5  # 惩罚当前机器人在X、Y方向速度与命令不一致
#         tracking_ang_vel = 0.75  # 惩罚当前机器人在角度转向速度与命令不一致
#         lin_vel_z = -1.0  # 惩罚机器人在Z轴上的速度 对应现象为机器人上下起伏很大
#         ang_vel_xy = -0.05  # 惩罚机器人在X轴和Y轴上的角速度 对应现象为遏制机器人左右晃动和前后晃动
#         orientation = -0.5  # 强烈鼓励机器人与初始姿态的基座方向一致
#         torques = -0.0003  # 机器人运控各电机输出的力矩的平方和 让模型找到最省力矩的方案
#         dof_vel = 0
#         dof_acc = 0
#         base_height = -10  # 惩罚基座高度不保持在期望的高度上
#         feet_air_time = 0
#         collision = -1.
#         feet_stumble = 0
#         action_rate = -0.01
#         stand_still = -0.5  # 惩罚速度命令很小的时候基座应该维持默认姿态
#
#         # 以下为csq的新奖励
#         dof_pos_limits = 0
#         hip_action_l2 = 0  # -0.5
#         hip_default = 0  # 惩罚髋关节不在默认位置



# # ------------ reward functions----------------
# def _reward_lin_vel_z(self):
#     # Penalize z axis base linear velocity
#     return torch.square(self.base_lin_vel[:, 2])
#
#
# def _reward_ang_vel_xy(self):
#     # Penalize xy axes base angular velocity
#     return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
#
#
# def _reward_orientation(self):
#     # Penalize non flat base orientation
#     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
#
#
# def _reward_base_height(self):
#     # Penalize base height away from target
#     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
#     return torch.square(base_height - self.cfg.rewards.base_height_target)
#
#
# def _reward_torques(self):
#     # Penalize torques
#     return torch.sum(torch.square(self.torques), dim=1)
#
#
# def _reward_dof_vel(self):
#     # Penalize dof velocities
#     self.dof_vel[:, self.wheel_indices] = 0
#     return torch.sum(torch.square(self.dof_vel), dim=1)
#
#
# def _reward_dof_acc(self):
#     # Penalize dof accelerations
#     return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
#
#
# def _reward_collision(self):
#     # Penalize collisions on selected bodies
#     return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
#
#
# def _reward_termination(self):
#     # Terminal reward / penalty
#     return self.reset_buf * ~self.time_out_buf
#
#
# def _reward_dof_pos_limits(self):
#     # Penalize dof positions too close to the limit
#     out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
#     out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
#     return torch.sum(out_of_limits, dim=1)
#
#
# def _reward_dof_vel_limits(self):
#     # Penalize dof velocities too close to the limit
#     # clip to max error = 1 rad/s per joint to avoid huge penalties
#     return torch.sum(
#         (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
#         dim=1)
#
#
# def _reward_torque_limits(self):
#     # penalize torques too close to the limit
#     return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.),
#                      dim=1)
#
#
# def _reward_tracking_lin_vel(self):
#     # Tracking of linear velocity commands (xy axes)
#     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#     return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
#
#
# def _reward_tracking_ang_vel(self):
#     # Tracking of angular velocity commands (yaw)
#     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
#     return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
#
#
# def _reward_feet_air_time(self):
#     # Reward long steps
#     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
#     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
#     contact_filt = torch.logical_or(contact, self.last_contacts)
#     self.last_contacts = contact
#     first_contact = (self.feet_air_time > 0.) * contact_filt
#     self.feet_air_time += self.dt
#     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
#                             dim=1)  # reward only on first contact with the ground
#     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
#     self.feet_air_time *= ~contact_filt
#     return rew_airTime
#
#
# def _reward_feet_stumble(self):
#     # Penalize feet hitting vertical surfaces
#     return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
#                      5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
#
#
# def _reward_stand_still(self):
#     # Penalize motion at zero commands
#     dof_err = self.dof_pos - self.default_dof_pos
#     dof_err[:, self.wheel_indices] = 0
#     return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
#
#
# def _reward_feet_contact_forces(self):
#     # penalize high contact forces
#     return torch.sum(
#         (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(
#             min=0.), dim=1)
#
#
# def _reward_orientation_quat(self):
#     # Penalize non flat base orientation
#     orientation_error = torch.sum(torch.square(self.root_states[:, :7] - self.base_init_state[0:7]), dim=1)
#     return torch.exp(-orientation_error / self.cfg.rewards.tracking_sigma)
#
#
# def _reward_hip_action_l2(self):
#     action_l2 = torch.sum(self.actions[:, [0, 4, 8, 12]] ** 2, dim=1)
#     # self.episode_metric_sums['leg_action_l2'] += action_l2
#     # print("action_l2", action_l2.shape)
#     return action_l2
#
#
# def _reward_hip_default(self):
#     hip_err = torch.sum((self.dof_pos[:, [0, 4, 8, 12]] - self.default_dof_pos[:, [0, 4, 8, 12]]) ** 2, dim=1)
#     # print("penalty",penalty.shape)
#     return hip_err



















