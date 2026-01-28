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
import locotouch.mdp.transport_go2w_reward_funcs as object_reward_funcs


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

        # region ------------------------------Scene------------------------------
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
                    proportion=0.2
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.0, noise_range=(0.00, 0.10), noise_step=0.005, border_width=0.25
                ),
                "perlin_rough": custom_terrain_gen.HfPerlinNoiseTerrainCfg(
                    proportion=0.2, noise_range=(0.00, 0.10), noise_step=0.005,
                    frequency=0.7, octaves=2, lacunarity=2.0, persistence=0.5, border_width=0.25
                ),
                "x_wave": custom_terrain_gen.HfXWaveTerrainCfg(
                    proportion=0.0, amplitude_range=(0.04, 0.10), wave_length=(1.55, 1.65), border_width=0.25
                ),
                "trap_speed_bump": custom_terrain_gen.HfSpeedBumpTerrainCfg(
                    proportion=0.2, num_bumps=6, bump_height_range=(0.03, 0.07),
                    random_flat_ratio=(0.20, 0.50), random_bump_width=(0.30, 0.40),
                    num_gaps=2, random_gap_length=(0.5, 1.5), gap_margin=0.5,
                    platform_width=2.0, border_width=0.25,
                ),
                "tri_speed_bump": custom_terrain_gen.HfSpeedBumpTerrainCfg(
                    proportion=0.2, num_bumps=6, bump_height_range=(0.03, 0.07),
                    random_flat_ratio=(0.0, 0.0), random_bump_width=(0.30, 0.40),
                    num_gaps=2, random_gap_length=(0.5, 1.5), gap_margin=0.5,
                    platform_width=2.0, border_width=0.25
                ),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.2, grid_width=0.45, grid_height_range=(0.00, 0.20), platform_width=2.0
                ),
            },
            seed=1,
        )
        # endregion

        # region ------------------------------Observations------------------------------
        # 加入背部平台加速度
        self.observations.policy.base_lin_acc = ObservationTermCfg(
            func=mdp.base_lin_acc,
            scale=0.25,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
            },
        )

        self.observations.critic.base_lin_acc = ObservationTermCfg(
            func=mdp.base_lin_acc,
            scale=0.25,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
            },
        )
        # endregion

        # region ------------------------------Actions------------------------------
        pass
        # endregion

        # region ------------------------------Events------------------------------
        pass
        # startup:

        # reset:
        # 机器人初始化位置
        self.events.reset_base.params["pose_range"]["x"] = (-1.5, 1.5)  # 保证在中心平台
        self.events.reset_base.params["pose_range"]["y"] = (-1.5, 1.5)
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi, math.pi)

        # interval:

        # endregion

        # region ------------------------------Terminations------------------------------
        pass
        # endregion

        # region ------------------------------Commands------------------------------
        # 仅保留 x 方向
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5,1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.final_rel_standing_envs = 0.1
        self.commands.base_velocity.initial_zero_command_steps = 50
        self.commands.base_velocity.final_initial_zero_command_steps = 50
        self.commands.base_velocity.resampling_time_range = (3.0, 3.0)

        # endregion

        # region ------------------------------Curriculums------------------------------
        self.curriculum.command_z_levels = None
        # endregion

        # region ------------------------------Rewards------------------------------
        import locotouch.mdp.transport_go2w_reward_funcs as object_reward_funcs

        # 调大背部平台保持水平的奖励权重
        self.rewards.action_rate_l2.weight = 0.001 # 减少动作变化惩罚

        self.rewards.flat_orientation_l2.weight = -2.5

        self.rewards.lin_vel_z_l2.weight = -5.0
        # 抑制背部平台的晃动
        self.rewards.ang_vel_xy_l2.weight = -0.1

        # 惩罚base的线加速度
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_lin_vel_x_exp = RewardTermCfg(
            func=object_reward_funcs.track_lin_vel_x_exp_acc_gated,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
                "acc_soft": 1.5,  # acc < gR / H; acc < \mu g
                "acc_hard": 5.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
            }
        )
        self.rewards.track_lin_vel_y_exp = RewardTermCfg(
            func=object_reward_funcs.track_lin_vel_y_exp,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
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

        # ---------------------------------------------------------------------
        # Play 时固定 commands 范围（避免 curriculum 训练时的动态范围影响 play）
        # 方案：直接把 commands 的 ranges 设成“最终想要的范围”，并同步 curriculum 的 range_multiplier
        # ---------------------------------------------------------------------

        # 你想在 play 用的最大范围（建议与你训练时最终期望的一致）
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













