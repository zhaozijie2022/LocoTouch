"""
Random Cylinder Transport Task Configuration - No Tactile Test Version
专门用于测试不带触觉传感器的 Go1 机器人是否能完成负载运输任务
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg, EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

import numpy as np
import locotouch.mdp as mdp
from locotouch.assets.go1 import Go1_CFG  # 使用纯 Go1，完全没有触觉装置
from .object_transport_teacher_env_cfg import (
    ObjectTransportTeacherEnvCfg,
    locotouch_object_transport_play_env_post_init_func,
)


@configclass
class RandCylinderTransportNoTactileTestEnvCfg(ObjectTransportTeacherEnvCfg):
    """
    随机竖直圆柱体运输任务 - 无触觉传感器测试版本
    
    关键特性:
    - 使用纯 Go1 机器人（Go1_CFG），完全没有触觉传感器装置
    - 圆柱体竖直放置在机器人背上（axis="Z"）
    - 修复了所有 body 命名问题（Go1 使用 base 而非 trunk，足端使用双下划线）
    - 竖直圆柱体的稳定性更具挑战性，需要同时控制 roll 和 pitch
    - 仅依靠本体感觉（IMU、关节编码器）+ 物体状态估计
    """
    def __post_init__(self):
        super().__post_init__()
        
        # 关键修改：使用纯 Go1 机器人，完全没有触觉装置
        self.scene.replicate_physics = False
        self.scene.robot = Go1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 添加随机化圆柱体
        env_num = self.scene.num_envs
        radius_range = (0.07, 0.07)  # 固定半径: 7 cm
        height_range = (0.05, 0.05)  # 固定高度: 5 cm
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
                        axis="Z",  # 改为 Z 轴，让圆柱体竖着放
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
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=1.0e-9,
                    rest_offset=-0.002
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        
        # 使用统一的物体状态重置函数
        self.events.reset_object_position.func = mdp.ResetObjectStateUniform
        
        # ========== 修复 Go1 特定的 body 命名问题 ==========
        
        # 1. 修复躯干质量随机化：trunk -> base
        self.events.randomize_trunk_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),  # Go1 中躯干叫 base
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        
        # 2. 修复躯干摩擦力随机化：trunk -> base
        self.events.randomize_trunk_sensor_physics_material = EventTermCfg(
            func=mdp.randomize_friction_restitution,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),  # Go1 中躯干叫 base
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (1.0, 1.0),
                "make_consistent": True,
                "restitution_range": (0.0, 0.2),
            },
        )
        
        # 3. 修复足端摩擦力随机化：使用 Go1 的足端命名（双下划线）
        self.events.randomize_foot_physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,  # 使用这个函数才支持 num_buckets
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),  # 通配符匹配所有足端
                "static_friction_range": (0.6, 1.5),
                "dynamic_friction_range": (0.6, 1.5),
                "make_consistent": True,
                "restitution_range": (0.0, 0.3),
                "num_buckets": 8000,
            },
        )
        
        # 4. 修复步态奖励中的足端名称：使用 Go1 的命名（双下划线）
        # Go1 的足端：a__FL_foot, a__FR_foot, a__RL_foot, a__RR_foot
        self.rewards.gait.params["synced_feet_pair_names"] = (
            ("a__FR_foot", "a__RL_foot"),  # 对角线对：右前 + 左后
            ("a__FL_foot", "a__RR_foot"),  # 对角线对：左前 + 右后
        )
        
        # 物理属性随机化
        self.events.randomize_object_physics_material.params["static_friction_range"] = (0.3, 1.0)
        
        # 速度课程学习参数
        self.curriculum.velocity_commands.params["reset_envs_episode_length"] = 0.98
        self.curriculum.velocity_commands.params["error_threshold_lin"] = 0.08
        self.curriculum.velocity_commands.params["error_threshold_ang"] = 0.1
        
        # 竖直圆柱体特定的奖励和终止条件
        # 竖着放的圆柱体需要同时关心 roll 和 pitch（倾倒），而不是只关心 roll（滚动）
        self.rewards.object_roll_pitch_angle.func = mdp.object_relative_roll_pitch_angle_ngt  # 同时惩罚 roll 和 pitch
        self.rewards.object_roll_pitch_angle.weight = -0.1  # 增加权重，因为竖着放更容易倾倒
        self.rewards.object_roll_pitch_velocity.func = mdp.object_relative_roll_pitch_velocity_ngt  # 同时惩罚角速度
        self.rewards.object_roll_pitch_velocity.weight = -0.1  # 增加权重
        self.rewards.object_yaw_alignment.weight = -0.05  # 降低 yaw 对齐的重要性
        
        # 初始姿态：竖直放置，不需要随机 pitch
        self.events.reset_object_position.params["pose_range"]["pitch"] = (0.0, 0.0)  # 保持竖直
        self.events.reset_object_position.params["pose_range"]["roll"] = (0.0, 0.0)   # 保持竖直
        
        # 终止条件：如果圆柱体倾倒超过 30 度就终止
        self.terminations.object_bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation,  # 使用通用的 bad_orientation 而不是只检查 roll
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "limit_angle": math.pi / 6,  # 30 度，竖着放更严格
            },
        )


@configclass
class RandCylinderTransportNoTactileTestEnvCfg_PLAY(RandCylinderTransportNoTactileTestEnvCfg):
    """测试/可视化版本"""
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()
        locotouch_object_transport_play_env_post_init_func(self)
