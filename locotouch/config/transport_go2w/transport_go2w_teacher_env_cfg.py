import math
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg, EventTermCfg, TerminationTermCfg, RewardTermCfg, ObservationTermCfg, CurriculumTermCfg
from isaaclab.utils import configclass

import numpy as np
import locotouch.mdp as mdp
from locotouch.assets.go2w_transport import Go2W_TRANSPORT_CFG as Robot_CFG
from locotouch.config.go2w.locomotion_go2w_env_cfg import LocomotionGo2WEnvCfg

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
import locotouch.mdp.transport_go2w_reward_funcs as object_reward_funcs


@configclass
class TransportGo2WTeacherEnvCfg(LocomotionGo2WEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        
        # ========== 机器人配置 ==========
        # increase the rigid patch count for more objects
        self.scene.replicate_physics = False
        # self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096

        # region ------------------------------Scene------------------------------
        # 增加物体和接触传感器
        env_num = self.scene.num_envs
        radius_range = (0.025, 0.075)

        # height_range = (0.15, 0.25)
        # size_range = np.array([radius_range, height_range])
        # size_samples = np.random.uniform(size_range[:, 0], size_range[:, 1], (env_num, 2))

        hr_ratio_range = (3.0, 5.0)
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
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(map(float, color_samples[i]))),) # type: ignore
                    for i in range(env_num) ],
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

        self.scene.object_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",  # net_history_forces: (N, 3, 1, 3)
            history_length=3,
            track_air_time=True,
        )
        # endregion

        # region ------------------------------Observations------------------------------
        # 添加物体相关的观测, 标准 13维状态, 未碰撞时为 non_contact_obs
        from locotouch.config.locotouch.object_transport_teacher_env_cfg import NoisyObjectStateCfg
        noisy_object_cfg = NoisyObjectStateCfg()
        self.observations.policy.object_state = noisy_object_cfg.object_state

        denoised_object_cfg = NoisyObjectStateCfg()
        denoised_object_cfg.object_state.params["add_uniform_noise"] = False
        self.observations.critic.object_state = denoised_object_cfg.object_state
        # endregion

        # region ------------------------------Actions------------------------------
        # pass
        # endregion

        # region ------------------------------Events------------------------------
        
        # startup:

        # reset:
        # 背部平台摩擦力
        # TODO body写 platform 会报错
        self.events.randomize_platform_physics_material = EventTermCfg(
            func=mdp.randomize_friction_restitution,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "static_friction_range": (0.3, 0.7),
                "dynamic_friction_range": (0.3, 0.7),
                "make_consistent": True,
                "restitution_range": (0.0, 0.2),
            },
        )
        # 物体摩擦力
        self.events.randomize_object_physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "static_friction_range": (0.3, 0.7),
                "dynamic_friction_range": (0.3, 0.7),
                "make_consistent": True,
                "restitution_range": (0.0, 0.2),
                "num_buckets": 8000,
            },
        )
        # 物体质量
        self.events.object_mass_randomization = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "mass_distribution_params": (0.5, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        # 物体初始化位置
        # TODO: 修改板子尺寸后, 同步修改物体初始化位置
        self.events.reset_object_position = EventTermCfg(
            func=mdp.ResetObjectStateUniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.02, 0.00),
                    "y": (-0.01, 0.01),
                    "z": (0.001, 0.002),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.0, 0.0)
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "reference_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # 机器人初始化位置
        self.events.reset_base.params["pose_range"]["x"] = (-0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (-0.0, 0.0)

        # interval:
        # TODO: 暂时关闭 push_object
        # self.events.push_object = EventTermCfg(
        #     func=mdp.push_by_setting_velocity,
        #     mode="interval",
        #     interval_range_s=(6.0, 8.0),
        #     params={
        #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        #         "velocity_range": {
        #             "x": (-0.3, 0.3),
        #             "y": (-0.3, 0.3),
        #             "z": (-0.2, 0.2),
        #             "roll": (-math.pi / 20, math.pi / 20),
        #             "pitch": (-math.pi / 20, math.pi / 20),
        #             "yaw": (-math.pi / 5, math.pi / 5), },
        #     },
        # )
        # endregion

        # region ------------------------------Terminations------------------------------
        # 保留 locomotion_base_env_cfg 中的所有 termination
        self.terminations.object_bad_orientation = TerminationTermCfg(
            func=mdp.bad_roll,
            params={"asset_cfg": SceneEntityCfg("object"),
                    "limit_angle": math.pi / 3,
                    },
        )
        self.terminations.object_below_robot = TerminationTermCfg(
            func=mdp.object_below_robot,
            params={"robot_cfg": SceneEntityCfg("robot"),
                    "object_cfg": SceneEntityCfg("object"),
                    },
            )

        # endregion

        # region ------------------------------Commands------------------------------
        # pass
        # endregion

        # region ------------------------------Curriculums------------------------------
        # pass
        # endregion

        # region ------------------------------Rewards------------------------------
        self.rewards.object_xy_position = RewardTermCfg(
            func=object_reward_funcs.object_relative_xy_position_ngt,
            weight=-0.5,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "command_name": "base_velocity",
                "work_only_when_cmd": True,
            }
        )
        self.rewards.object_z_velocity = RewardTermCfg(
            func=object_reward_funcs.object_relative_z_velocity_ngt,
            weight=-0.5,
            params = {
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
        self.rewards.object_roll_pitch_angle = RewardTermCfg(
            func=object_reward_funcs.object_relative_roll_pitch_angle_ngt,
            weight=-0.05,
            params = {
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
        self.rewards.object_roll_pitch_velocity = RewardTermCfg(
            func=object_reward_funcs.object_relative_roll_pitch_velocity_ngt,
            weight=-0.05,
            params = {
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
        self.rewards.object_dangerous_state = RewardTermCfg(
            func=object_reward_funcs.object_dangerous_state_ngt, weight=-50.0,
            params={
                # TODO: limit
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "x_max": 0.125,
                "y_max": 0.097,
                "z_min": 0.095,
                "roll_pitch_max": None,  # in degree
                "vel_xy_max": None,
            }
        )

        self.rewards.object_z_contact = RewardTermCfg(
            func=object_reward_funcs.object_lose_contact_ngt,
            weight=0.0,
            params = {
                "object_cfg": SceneEntityCfg("object"),
                "sensor_cfg": SceneEntityCfg("object_contact_sensor", body_names="Object"),
            }
        )


        # endregion


@configclass
class TransportGo2WTeacherEnvCfg_PLAY(TransportGo2WTeacherEnvCfg):
    """测试/可视化版本"""
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()

        from locotouch.assets.go2w_transport import Go2W_TRANSPORT_PLAY_CFG as Robot_PLAY_CFG
        self.scene.robot = Robot_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")













