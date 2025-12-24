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


# def object_relative_xy_velocity_ngt(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ) -> torch.Tensor:
#     robot: RigidObject = env.scene[robot_cfg.name]
#     obj: RigidObject = env.scene[object_cfg.name]
#     lin_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
#     return torch.sum(torch.square(lin_vel_in_robot_frame[:, :2]), dim=1)


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

