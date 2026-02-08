from __future__ import annotations
import math
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_apply_inverse, euler_xyz_from_quat, quat_inv, quat_mul
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




# ----------------- Object Transport -----------------
def object_relative_xy_position_ngt(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    work_only_when_cmd: bool = True,
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    rel_distance = torch.linalg.norm((obj.data.root_pos_w - robot.data.root_pos_w)[:, :2], dim=1)  # world frame
    if bool(work_only_when_cmd):
        cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
        rel_distance *= (cmd > 0.0)
    return rel_distance


def object_relative_xy_velocity_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    lin_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    return torch.sum(torch.square(lin_vel_in_robot_frame[:, :2]), dim=1)


def object_relative_z_velocity_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    lin_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    return torch.square(lin_vel_in_robot_frame[:, 2])


def object_relative_roll_pitch_angle_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    projected_gravity_w = quat_apply(obj.data.root_quat_w, obj.data.projected_gravity_b)
    projected_gravity_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, projected_gravity_w)
    return torch.sum(torch.square(projected_gravity_in_robot_frame[:, :2]), dim=1)


def object_relative_roll_pitch_velocity_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ang_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w - robot.data.root_ang_vel_w)
    return torch.sum(torch.abs(ang_vel_in_robot_frame[:, :2]), dim=1)


def object_dangerous_state_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    x_max: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    roll_pitch_max: float | None = None,
    vel_xy_max: float | None = None,
) -> torch.Tensor:
    robot: RigidObject | Articulation = env.scene[robot_cfg.name]
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    object_position_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)
    object_in_danger = torch.zeros_like(object_position_in_robot_frame[:, 0], dtype=torch.bool)
    if x_max is not None:
        object_in_danger |= torch.abs(object_position_in_robot_frame[:, 0]) > x_max
    if y_max is not None:
        object_in_danger |= torch.abs(object_position_in_robot_frame[:, 1]) > y_max
    if z_min is not None:
        object_in_danger |= object_position_in_robot_frame[:, 2] < z_min
    if roll_pitch_max is not None:
        object_in_danger |= torch.acos(-object.data.projected_gravity_b[:, 2]).abs() > (roll_pitch_max * math.pi / 180)
    if vel_xy_max is not None:
        object_lin_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, object.data.root_lin_vel_w - robot.data.root_lin_vel_w)
        object_in_danger |= torch.linalg.norm(object_lin_vel_in_robot_frame[:, :2], dim=1) > vel_xy_max
    return object_in_danger


def object_lose_contact_ngt(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_contact_sensor", body_names="Object"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] # type: ignore
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] # type: ignore
    return torch.logical_and(last_contact_time > 0.0, current_air_time > 0.0).reshape(-1)


# ----- Base Control -----

def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    # smoothstep: 3x^2 - 2x^3, x in [0,1]
    return x * x * (3.0 - 2.0 * x)

import isaaclab.utils.math as math_utils
def track_lin_vel_x_exp_acc_gated(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    acc_soft: float, acc_hard: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tracking reward gated by |base ax|. When |ax| is large, tracking reward is suppressed, and becomes 0 above acc_hard."""
    asset: Articulation = env.scene[asset_cfg.name]

    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    r_track = torch.exp(-lin_vel_error / (std ** 2))

    # --- acceleration gate (you said you can directly read it) ---
    # body_com_lin_acc_w: [N, num_bodies, 3] or [N, 3]? 你给的 slice 是 [:, 0]，这里按“x分量”写
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze()
    base_lin_acc_w = asset.data.body_com_lin_acc_w[:, asset_cfg.body_ids].squeeze()
    base_lin_acc_b = math_utils.quat_apply_inverse(body_quat, base_lin_acc_w)
    ax = base_lin_acc_b[:, 0]
    ax_abs = torch.abs(ax)

    # guard: ensure acc_soft < acc_hard
    # gate = 1                      if ax<=acc_soft
    #      = smooth decay 1->0       if acc_soft<ax<acc_hard
    #      = 0                      if ax>=acc_hard
    t = (ax_abs - acc_soft) / (acc_hard - acc_soft + 1e-6)
    t = torch.clamp(t, 0.0, 1.0)
    gate = 1.0 - _smoothstep01(t)
    gate = torch.where(ax_abs >= acc_hard, torch.zeros_like(gate), gate)

    return r_track * gate


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-lin_vel_error / std**2)

def joint_action_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    which_joint: str = None
) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    # TODO 没有使用asset_cfg.body_names
    if which_joint == "leg":
        return torch.sum(torch.square(env.action_manager.action[:, :12] - env.action_manager.prev_action[:, :12]), dim=1)
    elif which_joint == "wheel":
        return torch.sum(torch.square(env.action_manager.action[:, 12:16] - env.action_manager.prev_action[:, 12:16]), dim=1)
    elif which_joint is None:
        return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    else:
         raise ValueError(f"Unknown which_joint option: {which_joint}")