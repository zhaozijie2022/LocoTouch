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

from .legged_gym_rewards_cfg import LeggedGymRewardsCfg
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

        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        import locotouch.terrains as custom_terrain_gen
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

        MY_TERRAINS_CFG = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(
                    proportion=0.2,
                ),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.2, grid_width=0.45, grid_height_range=(0.00, 0.20), platform_width=2.0
                ),
                "perlin_rough": custom_terrain_gen.HfPerlinNoiseTerrainCfg(
                    proportion=0.4, noise_range=(0.00, 0.10), noise_step=0.005,
                    frequency=0.7, octaves=2, lacunarity=2.0, persistence=0.5, border_width=0.25
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(0.00, 0.10), noise_step=0.005, border_width=0.25
                ),
            },
            seed=1,
        )
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=MY_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
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

        # 移除 params={"action_name": "joint_pos"}, 包括轮子动作
        self.observations.policy.last_action = ObservationTermCfg(
            func=mdp.last_action,
            scale=1.0,
            history_length=6
        )

        self.observations.critic.last_action = ObservationTermCfg(
            func=mdp.last_action,
            scale=1.0,
            history_length=6
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
        # self.observations.critic.height_scan = None
        self.observations.critic.height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        self.observations.policy.history_length = 6
        self.observations.critic.history_length = 6

        # endregion

        # region ------------------------------Actions------------------------------
        # 腿部：位置控制（12D）- 与执行器 DCMotorCfg 对应
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=self.leg_joint_names,
            scale=0.25,
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
        self.events.randomize_trunk_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[self.base_link_name])

        # 足端摩擦力
        self.events.randomize_foot_physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.randomize_foot_physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.randomize_foot_physics_material.params["restitution_range"] = (0.0, 0.5)

        # 关节惯量随机
        self.events.randomize_rigid_body_inertia = EventTermCfg(
            func=mdp.randomize_rigid_body_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "inertia_distribution_params": (0.5, 1.5),
                "operation": "scale",
            },
        )

        # 基座质心
        self.events.randomize_com_positions = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )

        # reset:
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.2),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.35, 0.35),
            }
        }

        self.events.randomize_reset_joints = EventTermCfg(
            func=mdp.reset_joints_by_scale,
            # func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        self.events.randomize_apply_external_force_torque = EventTermCfg(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "force_range": (-10.0, 10.0),
                "torque_range": (-10.0, 10.0),
            },
        )

        # 电机的PID参数
        self.events.randomize_actuator_gains = EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )

        # interval:
        self.events.push_robot.interval_range_s = (6.0, 10.0)
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.3, 0.3),
            },
        }

        # endregion

        # region ------------------------------Terminations------------------------------
        self.terminations.base_contact = None
        # endregion

        # region ------------------------------Commands------------------------------
        self.commands.base_velocity = mdp.UniformVelocityCommandMultiSamplingCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.1,
            final_rel_standing_envs=0.1,
            initial_zero_command_steps=50,
            final_initial_zero_command_steps=50,
            rel_heading_envs=0.0,
            heading_command=False,
            # heading_control_stiffness=0.5,
            # debug_vis=True,
            ranges=mdp.UniformVelocityCommandMultiSamplingCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.3, 0.3),
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
        if self.scene.terrain.terrain_type == "generator":
            self.curriculum.terrain_levels = CurriculumTermCfg(
                func=mdp.terrain_levels_vel
            )
            # TODO: commands的课程开启后不生效, 原因暂时未知
            self.curriculum.command_xy_levels = None
            self.curriculum.command_z_levels = None
        # endregion

        # region ------------------------------Rewards------------------------------
        # self.rewards: RobotLabRewardsCfg = RobotLabRewardsCfg()
        self.rewards: LeggedGymRewardsCfg = LeggedGymRewardsCfg()


        import isaaclab.envs.mdp.rewards as reward_funcs
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.joint_torques_l2.weight = -2.0e-4
        self.rewards.joint_acc_l2 = RewardTermCfg(
            func=reward_funcs.joint_acc_l2,
            weight=-2.5e-7,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)
            }
        )
        self.rewards.joint_wheel_acc_l2 = RewardTermCfg(
            func=reward_funcs.joint_acc_l2,
            weight=-2.5e-9,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names)
            }
        )
        self.rewards.joint_deviation_l2.weight = -0.1
        self.rewards.hip_deviation_l2.weight = -0.3
        self.rewards.stand_still_without_cmd.weight = -0.25
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
        if self.scene.terrain.terrain_type != "generator":
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


        # # ---------------------------------------------------------------------
        # # Play：把 terrain 难度拉满（初始化就放到最难那一行），并关闭 terrain curriculum
        # # ---------------------------------------------------------------------
        # if self.scene.terrain.terrain_type ==  "generator":
        #     # 1) 计算最大 terrain level：一般对应 num_rows-1
        #     tg = self.scene.terrain.terrain_generator
        #     max_level = int(getattr(tg, "num_rows", 1)) - 1
        #     max_level = max(max_level, 0)
        #
        #     # 2) 初始化等级拉满：所有 env reset 时从最高 level 里采样
        #     self.scene.terrain.max_init_terrain_level = max_level
        #
        # # 3) 关掉 terrain 的 curriculum（否则它可能因为 move_down 又降回去）
        # if getattr(self, "curriculum", None) is not None:
        #     # 你这里的 term 名叫 terrain_levels（来自 terrain_levels = CurrTerm(...)）
        #     if getattr(self.curriculum, "terrain_levels", None) is not None:
        #         # 最稳妥：直接禁用这个 term（不同版本字段名可能略有差异）
        #         if hasattr(self.curriculum.terrain_levels, "enable"):
        #             self.curriculum.terrain_levels.enable = False
        #         elif hasattr(self.curriculum.terrain_levels, "enabled"):
        #             self.curriculum.terrain_levels.enabled = False
        #         else:
        #             # 实在没有开关字段，就把它置空（部分配置系统支持）
        #             self.curriculum.terrain_levels = None