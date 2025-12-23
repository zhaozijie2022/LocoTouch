import math
from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.utils import configclass
import locotouch.mdp.gym_dreamwaq_reward_funcs as reward_funcs  # 奖励项实现函数


LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]

BASE_LINK_NAME = "base"
FOOT_LINK_NAME = ".*_foot"

HIP_JOINT_NAMES = [
    "FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint",
]


@configclass
class GymDreamWaqRewardsCfg:
    """Configuration for DreamWaq robot rewards in Gym environment."""

    track_lin_vel_xy_exp = RewardTermCfg(
        func=reward_funcs.track_lin_vel_xy_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        }
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=reward_funcs.track_ang_vel_z_exp,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        }
    )
    lin_vel_z_l2 = RewardTermCfg(
        func=reward_funcs.lin_vel_z_l2,
        weight=-1.0
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=reward_funcs.ang_vel_xy_l2,
        weight=-0.05
    )
    flat_orientation_l2 = RewardTermCfg(
        func=reward_funcs.flat_orientation_l2,
        weight=-0.5
    )
    joint_torques_l2 = RewardTermCfg(
        func=reward_funcs.joint_torques_l2,
        weight=-3e-4,  # 0.0003
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES)
        }
    )

    base_height_l2 = RewardTermCfg(
        func=reward_funcs.base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.40,
        },
    )

    undesired_contacts = RewardTermCfg(
        func=reward_funcs.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]),
            "threshold": 0.1, # isaaclab的默认是1.0
        },
    )

    action_rate_l2 = RewardTermCfg(
        func=reward_funcs.action_rate_l2,
        weight=-0.01
    )

    stand_still_without_cmd = RewardTermCfg(
        func=reward_funcs.stand_still_without_cmd,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )

    hip_deviation_l2 = RewardTermCfg(
        func=reward_funcs.hip_deviation_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINT_NAMES),
        },
    )

    joint_deviation_l2 = RewardTermCfg(
        func=reward_funcs.joint_deviation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        }
    )




































