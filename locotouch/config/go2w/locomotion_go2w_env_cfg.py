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
import locotouch.mdp.robotlab_reward_funcs as robotlab_rewards  # 奖励项实现函数

from locotouch.assets.go2w import Go2W_CFG as Robot_CFG
from locotouch.config.base.locomotion_base_env_cfg import LocomotionBaseEnvCfg, smaller_scene_for_playing


# new-import
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from .gym_dreamwaq_rewards_cfg import GymDreamWaqRewardsCfg
from .robotlab_rewards_cfg import RobotLabRewardsCfg


@configclass
class LocomotionGo2WEnvCfg(LocomotionBaseEnvCfg):
    
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
        self.scene.robot = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # region ------------------------------Sence------------------------------
        # zz 增加地形传感器, 后续万一用到了
        self.scene.height_scanner = RayCasterCfg( # 扫描 base 周围 1.6 * 10 的高程图, 分辨率 0.1
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner_base = RayCasterCfg( # 只扫描 base 下方0.1 * 0.1, 但分辨率较高 0.05
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.contact_forces = ContactSensorCfg( # 补全传感器, 用于计算RobotLab的reward
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True
        )
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
        self.observations.policy.height_scan = None
        # self.observations.policy.height_scan = ObservationTermCfg(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )

        # 参考 robot_lab 和 gym_dreamwaq, 移除历史帧, 是否需要为每个项单独指定?
        self.observations.policy.history_length = 6
        self.observations.critic.history_length = 6

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
        # 足端摩擦力
        self.events.randomize_foot_physics_material.params["static_friction_range"] = (0.6, 1.5)
        self.events.randomize_foot_physics_material.params["dynamic_friction_range"] = (0.6, 1.5)
        self.events.randomize_foot_physics_material.params["restitution_range"] = (0.0, 0.3)

        # reset:
        # 躯干摩擦力
        # 物体摩擦力
        # self.events.randomize_object_physics_material = None
        # 物体质量
        # self.events.object_mass_randomization = None
        # 物体初始化位置
        # self.events.reset_object_position = None
        # reset_base 延续 object_transport_teacher_env_cfg 中的设定
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }}
        self.events.push_robot.interval_range_s = (6.0, 10.0)
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.3, 0.3),
                "z": (-0.1, 0.1),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
        }

        # self.events.randomize_object_physics_material.params["static_friction_range"] = (0.3, 1.0)

        # interval:
        # push_robot 延续 object_transport_teacher_env_cfg

        # endregion

        # region ------------------------------Terminations------------------------------
        self.terminations.base_contact = None
        # endregion

        # region ------------------------------Commands------------------------------
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
                ang_vel_z=(-math.pi / 4, math.pi / 4),
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
        # endregion

        # region ------------------------------Rewards------------------------------
        # self.rewards: RobotLabRewardsCfg = RobotLabRewardsCfg()
        self.rewards: GymDreamWaqRewardsCfg = GymDreamWaqRewardsCfg()
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
class LocomotionGo2WEnvCfg_PLAY(LocomotionGo2WEnvCfg):
    """测试/可视化版本"""
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()

        self.scene.robot = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        smaller_scene_for_playing(self)

        # ---------------------------------------------------------------------
        # Play 时固定 commands 范围（避免 curriculum 训练时的动态范围影响 play）
        # 方案：直接把 commands 的 ranges 设成“最终想要的范围”，并同步 curriculum 的 range_multiplier
        # ---------------------------------------------------------------------

        # 你想在 play 用的最大范围（建议与你训练时最终期望的一致）
        play_command_maximum_ranges = [
            self.commands.base_velocity.ranges.lin_vel_x[1],   # 1.0
            self.commands.base_velocity.ranges.lin_vel_y[1],   # 0.5
            self.commands.base_velocity.ranges.ang_vel_z[1],   # pi/4
        ]

        # 1) 覆盖 commands ranges
        self.commands.base_velocity.ranges.lin_vel_x = (-play_command_maximum_ranges[0], play_command_maximum_ranges[0])
        self.commands.base_velocity.ranges.lin_vel_y = (-play_command_maximum_ranges[1], play_command_maximum_ranges[1])
        self.commands.base_velocity.ranges.ang_vel_z = (-play_command_maximum_ranges[2], play_command_maximum_ranges[2])

        # 2) 固定“站立比例 / 初始零命令步数”为最终值（你参考代码里的那两行）
        self.commands.base_velocity.initial_zero_command_steps = self.commands.base_velocity.final_initial_zero_command_steps
        self.commands.base_velocity.rel_standing_envs = self.commands.base_velocity.final_rel_standing_envs

        # 3) 避免 play 时 curriculum 还在“缩放 range”
        #    你这里的 curriculum 是 command_xy_levels / command_z_levels（range_multiplier 从 0.1 -> 1.0）
        #    play 直接设成 (1.0, 1.0) 让它不再变化
        if getattr(self, "curriculum", None) is not None:
            if getattr(self.curriculum, "command_xy_levels", None) is not None:
                self.curriculum.command_xy_levels.params["range_multiplier"] = (1.0, 1.0)
            if getattr(self.curriculum, "command_z_levels", None) is not None:
                self.curriculum.command_z_levels.params["range_multiplier"] = (1.0, 1.0)