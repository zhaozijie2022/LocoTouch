import math
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg, EventTermCfg, TerminationTermCfg, RewardTermCfg, ObservationTermCfg, \
    CurriculumTermCfg
from isaaclab.utils import configclass

import numpy as np
import locotouch.mdp as mdp
from locotouch.assets.go2w_transport import Go2W_TRANSPORT_CFG as Robot_CFG
from locotouch.config.go2w.locomotion_go2w_env_cfg import LocomotionGo2WEnvCfg

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns


@configclass
class TransportGo2WBaseControlEnvCfg(LocomotionGo2WEnvCfg):
    """期望不加入物体, 训练背部平台的倾角实现物体运载"""

    def __post_init__(self):
        super().__post_init__()

        # ========== 机器人配置 ==========
        # increase the rigid patch count for more objects
        self.scene.replicate_physics = False
        self.scene.robot = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.num_envs = 20
        # self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096

        # region Scene
        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
        import locotouch.terrains as custom_terrain_gen

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(
                    proportion=0.0
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(0.00, 0.05), noise_step=0.005, border_width=0.25
                ),
                "perlin_rough": custom_terrain_gen.HfPerlinNoiseTerrainCfg(
                    proportion=0.2, noise_range=(0.00, 0.10), noise_step=0.005,
                    frequency=0.7, octaves=2, lacunarity=2.0, persistence=0.5, border_width=0.25
                ),
                # "x_wave": custom_terrain_gen.HfXWaveTerrainCfg(
                #     proportion=0.0, amplitude_range=(0.04, 0.10), wave_length=(1.55, 1.65), border_width=0.25
                # ),
                "speed_bump": custom_terrain_gen.HfSpeedBumpTerrainCfg(
                    proportion=0.6, num_bumps=8, bump_height_range=(0.03, 0.07),
                    random_flat_ratio=(0.0, 0.40), random_bump_width=(0.30, 0.35),
                    num_gaps=2, random_gap_length=(0.5, 1.5), gap_margin=0.5,
                    platform_width=2.0, border_width=0.25,
                ),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.0, grid_width=0.45, grid_height_range=(0.00, 0.10), platform_width=2.0
                ),
            },
            seed=1,
        )
        # endregion

        # region Observations
        # 仅 last_action 用 history=1，其余 term 保持 6：将组 history_length 置为 None 以使用 per-term 配置
        self.observations.policy.history_length = None
        self.observations.critic.history_length = None
        self.observations.policy.last_action = ObservationTermCfg(
            func=mdp.last_action,
            scale=1.0,
            history_length=1,
        )
        self.observations.critic.last_action = ObservationTermCfg(
            func=mdp.last_action,
            scale=1.0,
            history_length=1,
        )
        # 加入背部平台加速度（可选）
        # from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
        # self.observations.policy.base_lin_acc = ObservationTermCfg(
        #     func=mdp.base_lin_acc,
        #     scale=0.25,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
        #     },
        #     noise=Unoise(n_min=-0.5, n_max=0.5),
        # )
        #
        # self.observations.critic.base_lin_acc = ObservationTermCfg(
        #     func=mdp.base_lin_acc,
        #     scale=0.25,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
        #     },
        # )
        # endregion

        # region Actions
        # TODO 在action的config中直接加入低通滤波, 避免action-rate的崩溃
        # self.actions.joint_pos = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=self.leg_joint_names,
        #     scale=0.25,
        #     use_default_offset=True,
        #     clip={".*": (-100.0, 100.0,)},
        #     # clip={".*": (-1.2, 1.2)},
        #     preserve_order=True,
        # )
        self.actions.joint_pos = mdp.JointPositionLowPassActionCfg(
            asset_name="robot",
            joint_names=self.leg_joint_names,
            scale=0.25,
            use_default_offset=True,
            clip={".*": (-100.0, 100.0,)},
            # clip={".*": (-1.2, 1.2)},
            preserve_order=True,
            control_frequency=50.0,
            cut_off_frequency=5.0,
            order=2,
        )

        from isaaclab.envs.mdp import JointVelocityActionCfg  # 轮子速度控制
        # 轮子：速度控制（4D）- 与执行器 ImplicitActuatorCfg 对应
        # self.actions.joint_vel = JointVelocityActionCfg(
        #     asset_name="robot",
        #     joint_names=self.wheel_joint_names,
        #     scale=10.0,
        #     use_default_offset=True,
        #     clip={".*": (-100.0, 100.0,)},
        #     # clip={".*": (-10.0, 10.0)},
        # )
        self.actions.joint_vel = mdp.JointVelocityLowPassActionCfg(
            asset_name="robot",
            joint_names=self.wheel_joint_names,
            scale=10.0,
            use_default_offset=True,
            clip={".*": (-100.0, 100.0,)},
            # clip={".*": (-10.0, 10.0)},
            control_frequency=50.0,
            cut_off_frequency=5.0,
            order=2,
        )
        # endregion

        # region Events
        # startup:

        # reset:
        # 机器人初始化位置（相对于地形原点）
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.1, 0.1),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (-0.2, 0.2),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            }
        }
        # interval:

        # endregion

        # region Terminations
        self.terminations.base_height_below_minimum = None
        self.terminations.base_orientation = None
        self.terminations.terrain_out_of_bounds = TerminationTermCfg(
            func=mdp.terrain_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
            time_out=True,
        )
        # endregion

        # region Commands
        # 仅保留 x 方向速度指令
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.final_rel_standing_envs = 0.1
        self.commands.base_velocity.initial_zero_command_steps = 50
        self.commands.base_velocity.final_initial_zero_command_steps = 50
        self.commands.base_velocity.resampling_time_range = (6.0, 8.0)

        # endregion

        # region Curriculums
        self.curriculum.command_z_levels = None
        # endregion

        # region Rewards
        import locotouch.mdp.custom_reward_funcs as custom_reward_funcs

        # 增加了reset屏蔽和clip, weight不变
        self.rewards.action_rate_l2 = RewardTermCfg(
            func=custom_reward_funcs.custom_action_rate_l2_with_clip,
            weight=-0.01,
            params={
                "threshold": 7.0,
            }
        )

        # 惩罚 roll & pitch -0.5 -> -15.0
        self.rewards.flat_orientation_l2.weight = -10.0
        # 抑制背部平台的上下速度 -1.0 -> -5.0
        self.rewards.lin_vel_z_l2.weight = -5.0
        # xy方向角速度就是roll和pitch的角速度惩罚 -0.05 -> -0.25
        self.rewards.ang_vel_xy_l2.weight = -0.25
        # 额外惩罚base的俯仰角pitch
        self.rewards.base_pitch_angle_l2 = RewardTermCfg(
            func=custom_reward_funcs.custom_base_pitch_angle_l2,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
            }
        )

        # 鼓励速度跟踪
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_lin_vel_x_exp = RewardTermCfg(
            func=custom_reward_funcs.custom_track_lin_vel_x_exp,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
            }
        )
        # cmd没有y方向速度, 这里的0.75是鼓励y方向不要漂移
        self.rewards.track_lin_vel_y_exp = RewardTermCfg(
            func=custom_reward_funcs.custom_track_lin_vel_y_exp,
            weight=0.75,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
            }
        )
        self.rewards.track_ang_vel_z_exp = RewardTermCfg(
            func=custom_reward_funcs.custom_track_ang_vel_z_exp,
            weight=0.75,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
            }
        )

        # self.rewards.track_lin_vel_x_exp = RewardTermCfg(
        #     func=custom_reward_funcs.track_lin_vel_x_exp_acc_gated,
        #     weight=1.0,
        #     params={
        #         "command_name": "base_velocity",
        #         "std": math.sqrt(0.25),
        #         # acc在1.5到5.0
        #         "acc_soft": 1.5,  # acc < gR / H; acc < \mu g
        #         "acc_hard": 5.0,
        #         "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
        #     }
        # )

        # 惩罚 base xyz加速度
        self.rewards.base_acc_l2 = RewardTermCfg(
            func=custom_reward_funcs.custom_base_acc_l2,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "threshold": (1.5, 10.0),
                "xyz": (1.0, 1.0, 1.0)
            }
        )

        # 移除joint_deviation, 保留stand_still_without_cmd和base_height
        # self.rewards.joint_deviation_l2 = None
        self.rewards.base_height_l2 = RewardTermCfg(
            func=custom_reward_funcs.custom_base_height_l2,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
                "target_height": 0.40,
                "terrain_height_threshold": (-0.2, 0.2),
            }
        )


        # endregion


@configclass
class TransportGo2WBaseControlEnvCfg_PLAY(TransportGo2WBaseControlEnvCfg):
    """测试/可视化版本"""

    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()

        from locotouch.assets.go2w_transport import Go2W_TRANSPORT_PLAY_CFG as Robot_PLAY_CFG
        self.scene.robot = Robot_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        from locotouch.config.base.locomotion_base_env_cfg import smaller_scene_for_playing
        smaller_scene_for_playing(self)


        env_num = self.scene.num_envs
        radius_range = (0.05, 0.05)  # (0.025, 0.075)

        # height_range = (0.15, 0.25)
        # size_range = np.array([radius_range, height_range])
        # size_samples = np.random.uniform(size_range[:, 0], size_range[:, 1], (env_num, 2))

        hr_ratio_range = (4.0, 4.0)  # (3.0, 6.0)
        radii = np.random.uniform(radius_range[0], radius_range[1], size=(env_num, 1))
        hr_ratios = np.random.uniform(hr_ratio_range[0], hr_ratio_range[1], size=(env_num, 1))
        heights = radii * hr_ratios
        size_samples = np.concatenate([radii, heights], axis=1)

        color_samples = np.random.uniform(0.0, 1.0, (env_num, 3)).astype(np.float32)
        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(  # 根据上述的采样生成多个object
                assets_cfg=[
                    sim_utils.CylinderCfg(
                        radius=float(size_samples[i, 0]),
                        height=float(size_samples[i, 1]),
                        axis="Z",
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(map(float, color_samples[i]))),
                    )  # type: ignore
                    for i in range(env_num)
                ],
                random_choice=False,  # 表示不是随机复用, 而是每个环境一个独立的object
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                activate_contact_sensors=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )

        self.events.reset_object_position = EventTermCfg(
            func=mdp.ResetObjectStateUniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.00, 0.00),
                    "y": (-0.00, 0.00),
                    "z": (0.05, 0.05),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.0, 0.0)
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "reference_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        if self.scene.terrain.terrain_type == "generator":
            self.scene.terrain.terrain_generator.border_width = 5.0
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 4
        else:
            play_command_maximum_ranges = [
                self.commands.base_velocity.ranges.lin_vel_x[1],  # 1.0
                self.commands.base_velocity.ranges.lin_vel_y[1],  # 0.5
                self.commands.base_velocity.ranges.ang_vel_z[1],  # pi/4
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













