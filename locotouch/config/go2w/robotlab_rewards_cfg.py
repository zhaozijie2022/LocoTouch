import math
from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.utils import configclass
import locotouch.mdp.robotlab_reward_funcs as reward_funcs  # 奖励项实现函数


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

@configclass
class RobotLabRewardsCfg:

    # General
    is_terminated = RewardTermCfg(
        func=reward_funcs.is_terminated,
        weight=0.0
    )

    # Root penalties
    lin_vel_z_l2 = RewardTermCfg(
        func=reward_funcs.lin_vel_z_l2,
        weight=-2.0
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=reward_funcs.ang_vel_xy_l2,
        weight=-0.05
    )
    flat_orientation_l2 = RewardTermCfg(
        func=reward_funcs.flat_orientation_l2,
        weight=0.0
    )
    base_height_l2 = RewardTermCfg(
        func=reward_funcs.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.40,
        },
    )
    body_lin_acc_l2 = RewardTermCfg(
        func=reward_funcs.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME])},
    )


    # Joint penalties
    joint_torques_l2 = RewardTermCfg(
        func=reward_funcs.joint_torques_l2,
        weight=-2.5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        }
    )
    joint_torques_wheel_l2 = RewardTermCfg(
        func=reward_funcs.joint_torques_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)
        }
    )
    joint_vel_l2 = RewardTermCfg(
        func=reward_funcs.joint_vel_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        }
    )
    joint_vel_wheel_l2 = RewardTermCfg(
        func=reward_funcs.joint_vel_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)
        }
    )
    joint_acc_l2 = RewardTermCfg(
        func=reward_funcs.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        }
    )
    joint_acc_wheel_l2 = RewardTermCfg(
        func=reward_funcs.joint_acc_l2,
        weight=-2.5e-9,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)
        }
    )

    # def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
    #     rew_term = RewardTermCfg(
    #         func=reward_funcs.joint_deviation_l1,
    #         weight=weight,
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)
    #         },
    #     )
    #     setattr(self, attr_name, rew_term)

    joint_pos_limits = RewardTermCfg(
        func=reward_funcs.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        }
    )
    joint_vel_limits = RewardTermCfg(
        func=reward_funcs.joint_vel_limits,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "soft_ratio": 1.0
        },
    )
    joint_power = RewardTermCfg(
        func=reward_funcs.joint_power,
        weight=-2e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    stand_still_without_cmd = RewardTermCfg(
        func=reward_funcs.stand_still_without_cmd,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    joint_pos_penalty = RewardTermCfg(
        func=reward_funcs.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    wheel_vel_penalty = RewardTermCfg(
        func=reward_funcs.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    joint_mirror = RewardTermCfg(
        func=reward_funcs.joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
        },
    )

    # action_mirror = RewardTermCfg(
    #     func=reward_funcs.action_mirror,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
    #     },
    # )
    #
    # action_sync = RewardTermCfg(
    #     func=reward_funcs.action_sync,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "joint_groups": [
    #             ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
    #             ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
    #             ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
    #         ],
    #     },
    # )

    # Action penalties
    # applied_torque_limits = RewardTermCfg(
    #     func=reward_funcs.applied_torque_limits,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*")
    #     },
    # )
    action_rate_l2 = RewardTermCfg(
        func=reward_funcs.action_rate_l2,
        weight=-0.01
    )
    # smoothness_1 = RewardTermCfg(func=reward_funcs.smoothness_1, weight=0.0)  # Same as action_rate_l2
    # smoothness_2 = RewardTermCfg(func=reward_funcs.smoothness_2, weight=0.0)  # Unvaliable now

    # Contact sensor
    undesired_contacts = RewardTermCfg(
        func=reward_funcs.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]),
            "threshold": 1.0,
        },
    )
    contact_forces = RewardTermCfg(
        func=reward_funcs.contact_forces,
        weight=-1.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "threshold": 100.0
        },
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewardTermCfg(
        func=reward_funcs.track_lin_vel_xy_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25)
        }
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=reward_funcs.track_ang_vel_z_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25)
        }
    )

    # Others
    feet_air_time = RewardTermCfg(
        func=reward_funcs.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
        },
    )

    # feet_air_time_variance = RewardTermCfg(
    #     func=reward_funcs.feet_air_time_variance_penalty,
    #     weight=0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME])},
    # )

    feet_gait = RewardTermCfg(
        func=reward_funcs.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    feet_contact = RewardTermCfg(
        func=reward_funcs.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    feet_contact_without_cmd = RewardTermCfg(
        func=reward_funcs.feet_contact_without_cmd,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "command_name": "base_velocity",
        },
    )

    feet_stumble = RewardTermCfg(
        func=reward_funcs.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
        },
    )

    feet_slide = RewardTermCfg(
        func=reward_funcs.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_NAME]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_NAME]),
        },
    )

    feet_height = RewardTermCfg(
        func=reward_funcs.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_NAME]),
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewardTermCfg(
        func=reward_funcs.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[FOOT_LINK_NAME]),
            "tanh_mult": 2.0,
            "target_height": -0.2,
            "command_name": "base_velocity",
        },
    )

    # feet_distance_y_exp = RewardTermCfg(
    #     func=reward_funcs.feet_distance_y_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_width": float,
    #     },
    # )

    # feet_distance_xy_exp = RewardTermCfg(
    #     func=reward_funcs.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )

    upward = RewardTermCfg(
        func=reward_funcs.upward,
        weight=1.0
    )

