"""
Random Cylinder Transport Task Configuration - Go2W Version
Go2W 轮腿机器人的随机圆柱体运输任务（无触觉传感器测试版本）
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg, EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

import numpy as np
import locotouch.mdp as mdp
import locotouch.mdp_go2w as mdp_go2w  # Go2W 特有的 MDP 函数
from isaaclab.envs.mdp import JointVelocityActionCfg  # 轮子速度控制
from locotouch.assets.go2w import Go2W_CFG  # 使用 Go2W 机器人
from .object_transport_teacher_env_cfg import (
    ObjectTransportTeacherEnvCfg,
    locotouch_object_transport_play_env_post_init_func,
)


@configclass
class RandCylinderTransportGo2WTestEnvCfg(ObjectTransportTeacherEnvCfg):
    """
    随机竖直圆柱体运输任务 - Go2W 轮腿机器人版本
    
    关键特性:
    - 使用 Go2W 轮腿机器人（Go2W_CFG）
    - 16 个关节：12 个腿部关节 + 4 个轮子关节
    - 腿部使用位置控制，轮子使用速度控制
    - 圆柱体竖直放置在机器人背上（axis="Z"）
    - 保持与 Go1 版本相同的奖励函数和终止条件
    - 仅依靠本体感觉（IMU、关节编码器）+ 物体状态估计
    """
    
    # Go2W 关节配置
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
        self.scene.robot = Go2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # ========== 随机化圆柱体配置 ==========
        env_num = self.scene.num_envs
        radius_range = (0.03, 0.07)  # 半径范围: 3-7 cm
        height_range = (0.1, 0.4)    # 高度范围: 10-40 cm
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
        
        # ========== Go2W 特有的观察空间调整 ==========
        # 关键：轮子位置不参与观察，使用 Go2W 专用的观察函数
        self.observations.policy.joint_pos.func = mdp_go2w.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        
        # Critic 也要使用相同的观察函数
        self.observations.critic.joint_pos.func = mdp_go2w.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        
        # 注意：观察空间的 asset_cfg 使用默认的 ".*" 会自动匹配所有 16 个关节
        # locomotion_go2w 显式设置是因为它的父类不同，这里不需要显式设置
        
        # ========== Go2W 双模式动作空间 ==========
        # 与 Go2W Locomotion 对齐：腿部位置控制 + 轮子速度控制
        
        # 腿部：位置控制（12D）- 与执行器 DCMotorCfg 对应
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        
        # 轮子：速度控制（4D）- 与执行器 ImplicitActuatorCfg 对应
        # 动态添加 joint_vel 动作项
        self.actions.joint_vel = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=self.wheel_joint_names,
            scale=5.0,  # 与 Go2W Locomotion 一致
            use_default_offset=True,
            clip={".*": (-100.0, 100.0)},
        )
        
        # 观察空间：保持 last_action 观察腿部位置即可（与父类一致）
        # 注意：last_action 默认观察 "joint_pos"，这对于 Go2W 依然适用
        # 如果需要同时观察轮子速度命令的历史，需要额外添加 last_action_vel 观察项
        
        # ========== Go2W 特有的 body 命名修正 ==========
        # Go2W 的 body 命名（注意：没有双下划线）
        # 躯干: "base"
        # 足端: "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        
        # 1. 修复躯干质量随机化
        self.events.randomize_trunk_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        
        # 2. 修复躯干摩擦力随机化
        self.events.randomize_trunk_sensor_physics_material = EventTermCfg(
            func=mdp.randomize_friction_restitution,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (1.0, 1.0),
                "make_consistent": True,
                "restitution_range": (0.0, 0.2),
            },
        )
        
        # 3. 修复足端摩擦力随机化（Go2W 的足端命名：无双下划线）
        self.events.randomize_foot_physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": (0.6, 1.5),
                "dynamic_friction_range": (0.6, 1.5),
                "make_consistent": True,
                "restitution_range": (0.0, 0.3),
                "num_buckets": 8000,
            },
        )
        
        # 4. 修复步态奖励中的足端名称（Go2W 的足端：FL_foot, FR_foot, RL_foot, RR_foot）
        # 注意：Go2W 可能不使用传统步态，但保持配置一致
        if hasattr(self.rewards, 'gait') and self.rewards.gait is not None:
            self.rewards.gait.params["synced_feet_pair_names"] = (
                ("FR_foot", "RL_foot"),  # 对角线对：右前 + 左后
                ("FL_foot", "RR_foot"),  # 对角线对：左前 + 右后
            )
        
        # ========== 物理属性随机化 ==========
        self.events.randomize_object_physics_material.params["static_friction_range"] = (0.3, 1.0)
        
        # ========== 速度课程学习参数 ==========
        self.curriculum.velocity_commands.params["reset_envs_episode_length"] = 0.98
        self.curriculum.velocity_commands.params["error_threshold_lin"] = 0.08
        self.curriculum.velocity_commands.params["error_threshold_ang"] = 0.1
        
        # ========== 竖直圆柱体特定的奖励和终止条件 ==========
        # 保持与 Go1 版本完全一致
        self.rewards.object_roll_pitch_angle.func = mdp.object_relative_roll_pitch_angle_ngt
        self.rewards.object_roll_pitch_angle.weight = -0.1
        self.rewards.object_roll_pitch_velocity.func = mdp.object_relative_roll_pitch_velocity_ngt
        self.rewards.object_roll_pitch_velocity.weight = -0.1
        self.rewards.object_yaw_alignment.weight = -0.05
        
        # 初始姿态：竖直放置
        self.events.reset_object_position.params["pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_object_position.params["pose_range"]["roll"] = (0.0, 0.0)
        
        # 终止条件：如果圆柱体倾倒超过 30 度就终止
        self.terminations.object_bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "limit_angle": math.pi / 6,  # 30 度
            },
        )


@configclass
class RandCylinderTransportGo2WTestEnvCfg_PLAY(RandCylinderTransportGo2WTestEnvCfg):
    """测试/可视化版本"""
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()
        locotouch_object_transport_play_env_post_init_func(self)
        
        # ⚠️ 重要：locotouch_object_transport_play_env_post_init_func 会将机器人替换为 LocoTouch
        # 我们需要重新设置为 Go2W_CFG（使用 Play 版本以便可视化）
        from locotouch.assets.go2w import Go2W_PLAY_CFG
        self.scene.robot = Go2W_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

