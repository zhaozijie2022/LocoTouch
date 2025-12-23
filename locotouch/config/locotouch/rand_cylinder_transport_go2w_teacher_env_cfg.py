"""
Random Cylinder Transport Task Configuration - Go2W Version
Go2W 轮腿机器人的随机圆柱体运输任务（无触觉传感器测试版本）
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg, EventTermCfg, TerminationTermCfg, RewardTermCfg, ObservationTermCfg, CurriculumTermCfg
from isaaclab.utils import configclass

import numpy as np
import locotouch.mdp as mdp
from isaaclab.envs.mdp import JointVelocityActionCfg  # 轮子速度控制
from locotouch.assets.go2w import Go2W_CFG  # 使用 Go2W 机器人
from locotouch.assets.go2w_transport import Go2W_TRANSPORT_CFG
from .object_transport_teacher_env_cfg import (
    ObjectTransportTeacherEnvCfg,
    locotouch_object_transport_play_env_post_init_func,
)
import locotouch.mdp.robotlab_reward_funcs as robotlab_rewards


# new-import
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns



@configclass
class RobotLabRewardsCfg:

    # General
    is_terminated = RewardTermCfg(func=robotlab_rewards.is_terminated, weight=0.0)

    # Root penalties
    lin_vel_z_l2 = RewardTermCfg(func=robotlab_rewards.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewardTermCfg(func=robotlab_rewards.ang_vel_xy_l2, weight=0.0)
    flat_orientation_l2 = RewardTermCfg(func=robotlab_rewards.flat_orientation_l2, weight=0.0)
    base_height_l2 = RewardTermCfg(
        func=robotlab_rewards.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    body_lin_acc_l2 = RewardTermCfg(
        func=robotlab_rewards.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )

    # Joint penalties
    joint_torques_l2 = RewardTermCfg(
        func=robotlab_rewards.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewardTermCfg(
        func=robotlab_rewards.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewardTermCfg(
        func=robotlab_rewards.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewardTermCfg(
            func=robotlab_rewards.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)

    joint_pos_limits = RewardTermCfg(
        func=robotlab_rewards.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewardTermCfg(
        func=robotlab_rewards.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    joint_power = RewardTermCfg(
        func=robotlab_rewards.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    stand_still_without_cmd = RewardTermCfg(
        func=robotlab_rewards.stand_still_without_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    joint_pos_penalty = RewardTermCfg(
        func=robotlab_rewards.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    wheel_vel_penalty = RewardTermCfg(
        func=robotlab_rewards.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    joint_mirror = RewardTermCfg(
        func=robotlab_rewards.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_mirror = RewardTermCfg(
        func=robotlab_rewards.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_sync = RewardTermCfg(
        func=robotlab_rewards.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )

    # Action penalties
    applied_torque_limits = RewardTermCfg(
        func=robotlab_rewards.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewardTermCfg(func=robotlab_rewards.action_rate_l2, weight=0.0)
    # smoothness_1 = RewardTermCfg(func=robotlab_rewards.smoothness_1, weight=0.0)  # Same as action_rate_l2
    # smoothness_2 = RewardTermCfg(func=robotlab_rewards.smoothness_2, weight=0.0)  # Unvaliable now

    # Contact sensor
    undesired_contacts = RewardTermCfg(
        func=robotlab_rewards.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    contact_forces = RewardTermCfg(
        func=robotlab_rewards.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewardTermCfg(
        func=robotlab_rewards.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=robotlab_rewards.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Others
    feet_air_time = RewardTermCfg(
        func=robotlab_rewards.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_air_time_variance = RewardTermCfg(
        func=robotlab_rewards.feet_air_time_variance_penalty,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )

    feet_gait = RewardTermCfg(
        func=robotlab_rewards.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    feet_contact = RewardTermCfg(
        func=robotlab_rewards.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    feet_contact_without_cmd = RewardTermCfg(
        func=robotlab_rewards.feet_contact_without_cmd,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
        },
    )

    feet_stumble = RewardTermCfg(
        func=robotlab_rewards.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_slide = RewardTermCfg(
        func=robotlab_rewards.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )

    feet_height = RewardTermCfg(
        func=robotlab_rewards.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewardTermCfg(
        func=robotlab_rewards.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            "command_name": "base_velocity",
        },
    )

    feet_distance_y_exp = RewardTermCfg(
        func=robotlab_rewards.feet_distance_y_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,
        },
    )

    # feet_distance_xy_exp = RewardTermCfg(
    #     func=robotlab_rewards.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )

    upward = RewardTermCfg(func=robotlab_rewards.upward, weight=0.0)

    joint_vel_wheel_l2 = RewardTermCfg(
        func=robotlab_rewards.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewardTermCfg(
        func=robotlab_rewards.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewardTermCfg(
        func=robotlab_rewards.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


# 重写 ObservationsCfg
class GymDreamWaQRewardsCfg:
    """ 全面继承自gym_dreamwaq"""
    pass



@configclass
class RandCylinderTransportGo2WTeacherEnvCfg(ObjectTransportTeacherEnvCfg):
    
    # Go2W 关节配置
    base_link_name = "base"
    foot_link_name = ".*_foot"
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    
    def __post_init__(self):
        super().__post_init__()
        
        # ========== 机器人配置 ==========
        self.scene.replicate_physics = False
        # self.scene.robot = Go2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot = Go2W_TRANSPORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # region ------------------------------Sence------------------------------
        # zz 增加地形传感器, 后续万一用到了
        # 扫描 base 周围 1.6 * 10 的高程图, 分辨率 0.1
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        # 只扫描 base 下方0.1 * 0.1, 但分辨率较高 0.05
        self.scene.height_scanner_base = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        # 补全传感器, 用于计算RobotLab的reward
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True
        )

        # 随机化圆柱体配置, from rand_cylinder_transport_teacher_env_cfg
        env_num = self.scene.num_envs
        radius_range = (0.03, 0.03)  # 半径
        height_range = (0.03, 0.03)  # 高度
        size_range = np.array([radius_range, height_range])
        size_samples = np.random.uniform(size_range[:, 0], size_range[:, 1], (env_num, 2))
        color_samples = np.random.uniform(0.0, 1.0, (env_num, 3)).astype(np.float32)

        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    sim_utils.CylinderCfg(
                        radius=float(size_samples[i, 0]),
                        height=float(size_samples[i, 1]),
                        axis="Z",  # 竖直放置
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=tuple(map(float, color_samples[i]))
                        ),
                    )
                    for i in range(env_num)
                ],
                random_choice=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                activate_contact_sensors=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),  # 物体固定为0.5kg csq 25/11/20
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=1.0e-9,
                    rest_offset=-0.002
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.events.reset_object_position.func = mdp.ResetObjectStateUniform
        # endregion

        # region ------------------------------Observations------------------------------
        # 参考 robot_lab 和 gym_dreamwaq , 轮子的位置是没必要加进来的
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )

        # 参考 gym_dreamwaq, 不添加 base_lin_vel, 基座线速度
        self.observations.policy.base_lin_vel = None # 本来 locomotion_base_env_cfg 里也没有
        self.observations.policy.height_scan = None

        # 为 critic 添加特权信息
        self.observations.critic.base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,  # gym_dreamwaq 中的scale
        )
        # self.observations.policy.height_scan = ObservationTermCfg(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )
        self.observations.policy.height_scan = None

        # TODO 不知道robot_lab这里是在干什么, 先注释掉吧
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # 参考 robot_lab 和 gym_dreamwaq, 移除历史帧
        self.observations.policy.history_length = 0
        self.observations.policy.velocity_commands.history_length = 0
        self.observations.policy.base_ang_vel.history_length = 0
        self.observations.policy.projected_gravity.history_length = 0
        self.observations.policy.joint_pos.history_length = 0
        self.observations.policy.joint_vel.history_length = 0
        self.observations.policy.last_action.history_length = 0
        self.observations.policy.object_state.history_length = 0

        self.observations.critic.history_length = 0
        self.observations.critic.velocity_commands.history_length = 0
        self.observations.critic.base_ang_vel.history_length = 0
        self.observations.critic.projected_gravity.history_length = 0
        self.observations.critic.joint_pos.history_length = 0
        self.observations.critic.joint_vel.history_length = 0
        self.observations.critic.last_action.history_length = 0
        self.observations.critic.object_state.history_length = 0

        # 对齐 gym_dreamwaq 中的观测 scale, 除了 commands_scale 外和 locomotion_base_env_cfg 一致

        # endregion

        # region ------------------------------Actions------------------------------
        # 腿部：位置控制（12D）- 与执行器 DCMotorCfg 对应
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=self.leg_joint_names,
            scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},  # gym_dreamwaq 统一为 0.25, 这里从robot_lab中继承
            use_default_offset=True,
            clip={".*": (-100.0, 100.0)},
            preserve_order=True
        )

        # 轮子：速度控制（4D）- 与执行器 ImplicitActuatorCfg 对应
        self.actions.joint_vel = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=self.wheel_joint_names,
            scale=10.0,  # 从 gym_dreamwaq 中 10.0, robot_lab中为 5.0
            use_default_offset=True,
            clip={".*": (-100.0, 100.0)},
        )
        # endregion

        # region ------------------------------Events------------------------------
        
        # startup:
        # 躯干质量随机, body_name trunk -> base
        self.events.randomize_trunk_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base")
        # 足端摩擦力在 object_transport_teacher_env_cfg 中已经修改, 不需要额外操作

        # reset:
        # 躯干摩擦力
        self.events.randomize_trunk_sensor_physics_material.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base")
        # TODO: 暂时移除物体相关的randomize
        # 物体摩擦力
        # self.events.randomize_object_physics_material = None
        # 物体质量
        # self.events.object_mass_randomization = None
        # 物体初始化位置
        # self.events.reset_object_position = None
        # reset_base 延续 object_transport_teacher_env_cfg 中的设定
        # self.events.reset_base = None

        self.events.randomize_object_physics_material.params["static_friction_range"] = (0.3, 1.0)

        # interval:
        # push_robot 延续 object_transport_teacher_env_cfg
        self.events.push_object = None

        # endregion

        # region ------------------------------Terminations------------------------------
        # 保留 locomotion_base_env_cfg 中的所有 termination

        self.terminations.object_below_robot = None  # TODO: 去除object掉落就reset的设定
        # TODO：如果圆柱体倾倒超过 30 度就终止
        # self.terminations.object_bad_orientation = TerminationTermCfg(
        #     func=mdp.bad_orientation,
        #     params={
        #         "asset_cfg": SceneEntityCfg("object"),
        #         "limit_angle": math.pi / 6,  # 30 度
        #     },
        # )
        # endregion

        # region ------------------------------Commands------------------------------
        # TODO: 先只允许 x 方向移动, 禁止侧向移动和旋转
        # self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.rel_standing_envs = 0.2  # 提高 0 指令环境的比例
        # self.commands.base_velocity.final_rel_standing_envs = 0.1  # 原来是 0.1 / 0.05
        # self.commands.base_velocity.initial_zero_command_steps = 0  # set to 0 to encourage exploration
        # self.commands.base_velocity.final_initial_zero_command_steps = 50  # 前50步无论如何采样也输出0指令

        self.commands.base_velocity = mdp.UniformVelocityCommandMultiSamplingCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.1,
            final_rel_standing_envs=0.05,
            initial_zero_command_steps=0,
            final_initial_zero_command_steps=50,
            rel_heading_envs=0.0,
            heading_command=False,
            # heading_control_stiffness=0.5,
            # debug_vis=True,
            ranges=mdp.UniformVelocityCommandMultiSamplingCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.0, 0.0),
            ),
        )


        # endregion

        # region ------------------------------Curriculums------------------------------

        self.curriculum.command_xy_levels = CurriculumTermCfg(
            func=mdp.command_xy_levels_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (0.1, 1.0),
            },
        )
        self.curriculum.command_z_levels = CurriculumTermCfg(
            func=mdp.command_z_levels_vel,
            params={
                "reward_term_name": "track_ang_vel_z_exp",
                "range_multiplier": (0.1, 1.0),
            },
        )

        self.curriculum.velocity_commands = None

        # endregion

        # region ------------------------------Rewards------------------------------
        self.rewards: RobotLabRewardsCfg = RobotLabRewardsCfg()

        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still_without_cmd.weight = -2.0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        self.disable_zero_weight_rewards()
        # endregion


    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)

@configclass
class RandCylinderTransportGo2WTeacherEnvCfg_PLAY(RandCylinderTransportGo2WTeacherEnvCfg):
    """测试/可视化版本"""
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()
        locotouch_object_transport_play_env_post_init_func(self)
        
        # ⚠️ 重要：locotouch_object_transport_play_env_post_init_func 会将机器人替换为 LocoTouch
        # 我们需要重新设置为 Go2W_CFG（使用 Play 版本以便可视化）
        # from locotouch.assets.go2w import Go2W_PLAY_CFG
        # self.scene.robot = Go2W_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        from locotouch.assets.go2w_transport import Go2W_TRANSPORT_PLAY_CFG
        self.scene.robot = Go2W_TRANSPORT_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 确保命令范围限制在 PLAY 版本中仍然生效（post_init 函数可能会覆盖）
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

