"""Transport Go2W BaseControl —— oracc 化改造版 (平地 x 方向加速度追踪)。

目标: 用背部平台的倾角实现物体运载。改造 = transport 自己的 policy 观测/动作
  + oracc (LocoWM) 的奖励/课程/地形 + realacc (LocoWM) 的域随机化。

要点 (对齐 oracc, 见 LocoWM/locowm/config/go2w/oracc_env_cfg.py):
  * 只追踪 pitch (对齐由前向加速度导出的理想倾角), roll 单独惩罚; 去掉耦合的 flat_orientation。
  * jerk + 软死区幅值两条无上限的加速度限制, 堵住"顶着加速度上限猛冲"的 reward-hacking。
  * 拆开 ang_vel_xy -> 只罚 roll-rate、放松 pitch-rate, 解放"加速前倾"所需的俯仰角速度。
  * 平地单一地形、x-only 命令、关地形课程。
  * policy 观测保持 transport 原样 (真机可得, 带噪, 无 acc); critic 对齐 oracc —— 追加
    base_lin_acc + ideal_projected_gravity 两项特权观测 (经 LPF 正确滤波)。
  * 域随机化对齐 realacc (保留 base 全套 DR)。

自包含: oracc 的内联奖励函数与 AccTrackCfg 直接放本文件; 只复用 locotouch.mdp 既有符号
(custom_base_lin_acc / sanitize 为本次移植)。track_pitch_exp 相对 oracc 做了两处增强 (见其 docstring)。
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Tuple
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, Articulation
from isaaclab.managers import SceneEntityCfg, EventTermCfg, TerminationTermCfg, RewardTermCfg, ObservationTermCfg, \
    CurriculumTermCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import numpy as np
import locotouch.mdp as mdp
import locotouch.mdp.custom_reward_funcs as custom_reward_funcs
from locotouch.mdp.sanitize import sanitize, DEFAULT_CLIP
from locotouch.assets.go2w_transport import Go2W_TRANSPORT_CFG as Robot_CFG
from locotouch.config.go2w.locomotion_go2w_env_cfg import LocomotionGo2WEnvCfg

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# 内联 mdp 奖励 (自包含, 从 LocoWM oracc 移植)
# =============================================================================
def track_pitch_exp(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = math.sqrt(0.1),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """只追踪 **pitch**: base 的 pitch (projected_gravity 的 x 分量) 对齐由**前向加速度**导出的理想 pitch。

    相对 oracc 的**唯一改动** (对应用户意见): **增大区分度、避免过早饱和** —— 默认 std 由 √0.25
    减到 √0.1, exp 更 sharp, 小 pitch 误差也能显著拉低奖励 (不再"pitch 大致对了就拿满"), 逼策略把 pitch 追得更准。

    **刻意不做**"随加速度放大奖励": 那会激励策略主动追求更大的加速度 (reward-hacking)。这里完美追踪时
    奖励恒为 1 (与加速度大小无关) —— 加速度由速度命令追踪驱动, 并受 base_jerk_l2 / base_acc_soft_l2 限制,
    pitch 奖励只负责"给定当前加速度, 把倾角追准", 不额外奖励加速本身。

    oracc 已修的两处原 ideal_projected_gravity 作奖励目标的错 (保留):
      * **退化**: 目标参考 LEVEL 系 (不随实际俯仰旋转), 直接用前向加速度导出 target_gx; acc=0 -> 0
        -> err=g_b0² 有真实回正梯度 (原实现巡航时任意俯仰都拿满、无梯度)。
      * **符号**: 前向加速 (ax>0) 应前倾 (nose-down) -> projected_gravity_b[0]=sinφ>0 -> 目标取 +ax_eff/norm。

    用机身系 (frame='base') 前向加速度 (有 yaw 也对); 与 jerk/acc_soft 共享同一 LPF 值。
    roll (y 分量) 交给 roll_orientation_l2 单独惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b                            # (N, 3)
    acc_b = mdp.custom_base_lin_acc(env, asset_cfg, frame="base")   # (N, 3) 机身系 LPF 加速度 (前向 = x)
    cfg = env.cfg.acc_track
    ax = acc_b[:, 0]
    ax_eff = torch.where(torch.abs(ax) < cfg.zero_threshold, torch.zeros_like(ax), ax) * cfg.acc_gain
    g = 9.81
    target_gx = ax_eff / torch.sqrt(ax_eff * ax_eff + g * g)        # +号 = 前倾 (前向加速 -> nose-down -> g_b0>0)
    err = torch.square(g_b[:, 0] - target_gx)                       # 只比较 pitch (x 分量)
    return sanitize(torch.exp(-err / (std ** 2)), clip)


def roll_orientation_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """惩罚 **roll**: base projected_gravity 的 y 分量² (侧倾)。x-only 前进不应出现 roll。"""
    asset: Articulation = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    return sanitize(torch.square(g_b[:, 1]), clip)


def base_jerk_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xyz: Tuple[float, float, float] = (1.0, 1.0, 0.0),
    clip: float = 1.0e5,
) -> torch.Tensor:
    """惩罚 base 加速度的变化率 (jerk = Δacc/dt) 的水平 (x,y) 分量, 逐轴加权 (无上限)。

    用 jerk 而非加速度幅值: 幅值惩罚有饱和上限 -> 可"顶着上限猛冲"以尽快拿满速度追踪奖励
    (reward-hacking); jerk 无上限且不与追踪对抗 (平滑加速 jerk≈0 不罚, 只罚"猛冲"式突变)。
    用 LPF 后世界系加速度 (经 env.cfg.acc_track, 与俯仰无关); 每 episode 头两步置 prev=当前 -> jerk=0,
    避免跨 episode 加速度不连续被误罚。
    """
    acc_w = mdp.custom_base_lin_acc(env, asset_cfg, frame="world")  # (N, 3)
    if getattr(env, "_basecontrol_prev_base_acc_w", None) is None or env._basecontrol_prev_base_acc_w.shape != acc_w.shape:
        env._basecontrol_prev_base_acc_w = acc_w.clone()
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    prev = torch.where(fresh, acc_w, env._basecontrol_prev_base_acc_w)
    jerk = (acc_w - prev) / env.step_dt                            # (N, 3), m/s^3
    env._basecontrol_prev_base_acc_w = acc_w.clone()

    w = torch.tensor(xyz, device=jerk.device, dtype=jerk.dtype)
    pen = torch.sum(torch.square(jerk) * w, dim=1) / float(sum(xyz))
    return sanitize(pen, clip)


def base_acc_soft_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 2.0,
    xyz: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    clip: float = 1.0e5,
) -> torch.Tensor:
    """惩罚 base 水平加速度**幅值超过 threshold 的部分** (软死区 + 无上限二次): pen = Σ relu(|acc|−thr)²。

    反制"缩短加速时间/高 acc 冲刺"的时间套利 (一段 Δv 的总加速惩罚 ∝ acc), 并把 acc 压回 -> 理想 pitch 可达。
    无饱和上限 -> 不可 hack; 软死区 |acc|≤thr 免费 -> 不压制追踪所需的适度加速。用 LPF 世界系加速度
    (与 jerk / ideal_projected_gravity 同一信号)。
    """
    acc_w = mdp.custom_base_lin_acc(env, asset_cfg, frame="world")  # (N, 3), LPF world
    excess = torch.clamp(torch.abs(acc_w) - threshold, min=0.0)     # 每轴超出量
    w = torch.tensor(xyz, device=acc_w.device, dtype=acc_w.dtype)
    pen = torch.sum(torch.square(excess) * w, dim=1) / float(sum(xyz))
    return sanitize(pen, clip)


def ang_vel_roll_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy: Tuple[float, float] = (1.0, 0.0),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """惩罚 base 角速度的 roll/pitch 分量, 逐轴加权。默认 xy=(1,0): **只罚 roll-rate (ωx), 放松 pitch-rate (ωy)**。

    把基座 ang_vel_xy_l2 (同时罚 ωx²+ωy²) 拆开: "加速前倾"必然需要 pitch 角速度, 若还罚 ωy 就等于给想要的
    动作罚款。解放 ωy, 只留 ωx 抑制侧翻。pitch 稳定由 track_pitch_exp 设定值提供。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang = asset.data.root_ang_vel_b[:, :2]   # (N, 2) = (roll-rate ωx, pitch-rate ωy)
    w = torch.tensor(xy, device=ang.device, dtype=ang.dtype)
    return sanitize(torch.sum(torch.square(ang) * w, dim=1), clip)


# 左右配对的腿关节 (逐位对应: FL↔FR, RL↔RR; 不含轮子 —— 轮子自由旋转, 位置对称无意义)
_LEFT_LEG_JOINTS = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
_RIGHT_LEG_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]


def leg_symmetry_l2(
    env: "ManagerBasedRLEnv",
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """惩罚腿部关节的**左右不对称**: pen = Σ_pairs [(q_L − q_R) − (q0_L − q0_R)]²。

    姿态正则的"轴分离"版, 与 joint_deviation_l2 互补: 纯俯仰(前后腿差, 左右对称)该项=0 -> 不挡 pitch;
    只罚左右不对称 (侧向乱蹬)。相对默认 L−R 关系的偏差形式 -> 与关节符号约定无关。
    left_cfg/right_cfg 须**逐位配对**且 preserve_order=True (否则 joint_ids 会被排序、错配)。
    """
    asset: Articulation = env.scene[left_cfg.name]
    qL = asset.data.joint_pos[:, left_cfg.joint_ids]
    qR = asset.data.joint_pos[:, right_cfg.joint_ids]
    q0L = asset.data.default_joint_pos[:, left_cfg.joint_ids]
    q0R = asset.data.default_joint_pos[:, right_cfg.joint_ids]
    diff = (qL - qR) - (q0L - q0R)
    return sanitize(torch.sum(torch.square(diff), dim=1), clip)


def feet_off_ground_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """惩罚**轮子离地**: 逐轮取接触力 (历史窗口 max), 低于 threshold 记为离地 (软量, 有梯度)。

    go2w 的 `*_foot` 链**就是轮子** (URDF collision = cylinder 轮胎); 查 `contact_forces` 上 `.*_foot`
    体的接触力 = 轮-地接触力。平地 x-only 四轮应全程着地; 速度突变时若靠"翘后轮"实现前倾则后轮离地 ->
    罚之逼其四轮着地下用腿长重构俯仰。历史窗口 max 抗单步接触噪声, 只罚**持续离地**。
    """
    cs = env.scene.sensors[sensor_cfg.name]
    f = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]  # (N, 轮数)
    off = torch.clamp(1.0 - f / threshold, min=0.0)   # 着地(f>=thr)=0, 越接近离地(f->0)->1
    return sanitize(torch.sum(off, dim=1), clip)


# =============================================================================
# AccTrackCfg (内联): base 加速度低通 + 理想倾角跟踪 的全局超参。
# 供 track_pitch_exp / base_jerk_l2 / base_acc_soft_l2 / mdp.custom_base_lin_acc /
# mdp.ideal_projected_gravity 通过 env.cfg.acc_track 读取 (obs 与 reward 共用同一处真源)。
# =============================================================================
@configclass
class AccTrackCfg:
    use_lpf: bool = True            # 对 base-acc 做平滑
    cut_off_frequency: float = 1.0  # LPF 截止频率 (Hz)
    control_frequency: float = 50.0 # 控制频率 (Hz) = 1/step_dt (step_dt = 4 * 0.005 = 0.02s)
    acc_gain: float = 1.0           # 水平加速度增益 (=1 -> 期望倾角=物理 tanφ=a/g, 不放大、不激励更大加速度)
    zero_threshold: float = 0.3     # 死区, 避免噪声抖动


# 自定义 obs 项的物理边界 (移植自 LocoWM), 防加速度类项数值爆炸; 未列出走 DEFAULT_CLIP。
OBS_CLIP_BOUNDS = {
    "projected_gravity": 1.0,
    "ideal_projected_gravity": 1.0,
    "base_ang_vel": 100.0,
    "base_lin_vel": 100.0,
    "base_lin_acc": 1000.0,
    "joint_vel": 1000.0,
    "joint_pos": 10.0,
    "last_action": 100.0,
    "velocity_commands": 10.0,
    "height_scan": 5.0,
}


@configclass
class TransportGo2WBaseControlEnvCfg(LocomotionGo2WEnvCfg):
    """期望不加入物体, 训练背部平台的倾角实现物体运载 (oracc 化: 加速度追踪 + 加速前倾)。"""

    # base-acc 平滑 + 理想倾角跟踪 全局超参 (reward 与 critic ideal_projected_gravity 共用)
    acc_track: AccTrackCfg = AccTrackCfg()

    def __post_init__(self):
        super().__post_init__()

        # ========== 机器人配置 ==========
        self.scene.replicate_physics = False
        self.scene.robot = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # region ------------------------------Terrain (对齐 oracc: 只有平地)------------------------------
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
            curriculum=False,
            seed=1,
        )
        self.scene.terrain.max_init_terrain_level = 0
        # endregion

        # region ------------------------------Observations------------------------------
        # policy: 保持 transport 原样 (真机可得, 带噪, 无 base_lin_vel/height_scan/acc), 仅调 history。
        # 仅 last_action 用 history=1, 其余保持 6: 组 history_length 置 None 以使用 per-term 配置。
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
        # critic 对齐 oracc: 追加特权加速度观测 (policy 不加)。
        #   base_lin_acc: 原始机身系加速度 (同 oracc)。
        #   ideal_projected_gravity: 内部经 custom_base_lin_acc 的 LPF -> 承载"正确滤波后的加速度"信息。
        self.observations.critic.base_lin_acc = ObservationTermCfg(
            func=mdp.base_lin_acc,
            scale=0.25,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
            history_length=6,
        )
        self.observations.critic.ideal_projected_gravity = ObservationTermCfg(
            func=mdp.ideal_projected_gravity,
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
            history_length=6,
        )
        # endregion

        # region ------------------------------Actions (保持不变: LowPass 位置/速度)------------------------------
        self.actions.joint_pos = mdp.JointPositionLowPassActionCfg(
            asset_name="robot",
            joint_names=self.leg_joint_names,
            scale=0.25,
            use_default_offset=True,
            clip={".*": (-100.0, 100.0,)},
            preserve_order=True,
            control_frequency=50.0,
            cut_off_frequency=5.0,
            order=1,
        )
        self.actions.joint_vel = mdp.JointVelocityLowPassActionCfg(
            asset_name="robot",
            joint_names=self.wheel_joint_names,
            scale=10.0,
            use_default_offset=True,
            clip={".*": (-100.0, 100.0,)},
            control_frequency=50.0,
            cut_off_frequency=15.0,
            order=1,
        )
        # endregion

        # region ------------------------------Events / Domain Randomization (对齐 realacc)------------------------------
        # realacc = 保留 LocoWM base 全套 DR。transport 多数已一致, 仅 3 处对齐:
        # 1) foot 摩擦范围 (transport go2w 覆盖成了 0.5/0.8, 恢复到 realacc 的 0.4~2.0)
        self.events.randomize_foot_physics_material.params["static_friction_range"] = (0.4, 2.0)
        self.events.randomize_foot_physics_material.params["dynamic_friction_range"] = (0.4, 2.0)
        # 2) reset_base: 保留 base control 的初始位姿 (z=0.05), 速度 roll/pitch/yaw 恢复 realacc 的 ±0.35
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.05, 0.05),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.35, 0.35),
            },
        }
        # 3) reset 关节缩放随机 (transport 关成了 (1,1), 恢复 realacc 的 0.95~1.05)
        self.events.randomize_reset_joints.params["position_range"] = (0.95, 1.05)
        # 其余 (base 质量/惯性/com/外力力矩/执行器增益/push_robot) 已与 realacc 一致, 不动。
        # endregion

        # region ------------------------------Terminations------------------------------
        self.terminations.base_height_below_minimum = None
        self.terminations.base_orientation = None
        self.terminations.terrain_out_of_bounds = TerminationTermCfg(
            func=mdp.terrain_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
            time_out=True,
        )
        # endregion

        # region ------------------------------Commands (x-only; 意见 1: x 范围保持 1.5)------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.initial_zero_command_steps = 50
        self.commands.base_velocity.resampling_time_range = (6.0, 8.0)
        self.commands.base_velocity.bang_bang_envs = 0.05
        # endregion

        # region ------------------------------Curriculum (对齐 oracc: 仅 command_x)------------------------------
        self.curriculum.terrain_levels = None          # 平地无难度轴
        self.curriculum.command_y_levels = None
        self.curriculum.command_z_levels = None        # x-only, 去 z 命令课程
        self.curriculum.command_x_levels = CurriculumTermCfg(
            func=mdp.command_axis_levels_vel,
            params={
                "reward_term_name": "track_lin_vel_x_exp",
                "range_multiplier": (0.1, 1.0),
                "upper_threshold": 0.8,
                "lower_threshold": 0.5,
                "ema_alpha": 0.5,
            },
        )
        # endregion

        # region ------------------------------Rewards (重建为 oracc 加速度追踪奖励集)------------------------------
        # -- 速度追踪 (x 命令; y/z 命令恒 0 -> 惩罚横向/偏航漂移, 强制直线 x) --
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_lin_vel_x_exp = RewardTermCfg(
            func=custom_reward_funcs.custom_track_lin_vel_x_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.track_lin_vel_y_exp = RewardTermCfg(
            func=custom_reward_funcs.custom_track_lin_vel_y_exp,
            weight=0.75,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.track_ang_vel_z_exp = RewardTermCfg(
            func=custom_reward_funcs.custom_track_ang_vel_z_exp,
            weight=0.75,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        # -- action rate (沿用 transport 的带 reset 屏蔽 + clip 版本, 语义等价 oracc safe_action_rate) --
        self.rewards.action_rate_l2 = RewardTermCfg(
            func=custom_reward_funcs.custom_action_rate_l2_with_clip,
            weight=-0.01,
            params={"threshold": 7.0},
        )
        # -- 调权 / 去除 (对齐 oracc) --
        self.rewards.lin_vel_z_l2.weight = -2.0        # oracc 值 (transport base control 曾 -5.0)
        self.rewards.flat_orientation_l2 = None        # 与"加速前倾"互斥, 去掉 (改用 roll_orientation + track_pitch)
        self.rewards.ang_vel_xy_l2 = None              # 拆成 ang_vel_roll_l2 (解放 pitch-rate)
        self.rewards.base_height_l2.weight = -20             # oracc 去掉
        # (transport 旧的 base_pitch_angle_l2 / base_acc_l2 不再添加 -> 自然消失)

        # -- 姿态分离: 罚 roll + 只追踪 pitch (加速前倾; track_pitch 已增大区分度, 不随加速度放大奖励) --
        self.rewards.roll_orientation_l2 = RewardTermCfg(
            func=roll_orientation_l2,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
        )
        self.rewards.track_pitch_exp = RewardTermCfg(
            func=track_pitch_exp,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "std": math.sqrt(0.1),
            },
        )
        # -- 加速度限制: jerk (管突变) + 软死区幅值 (管峰值/堵时间套利), 均无上限 --
        self.rewards.base_jerk_l2 = RewardTermCfg(
            func=base_jerk_l2,
            weight=-1.0e-3,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "xyz": (1.0, 0.0, 0.0),
                "clip": 1.0e5,
            },
        )
        self.rewards.base_acc_soft_l2 = RewardTermCfg(
            func=base_acc_soft_l2,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "threshold": 1.0,
                "xyz": (1.0, 0.0, 0.0),
                "clip": 1.0e5,
            },
        )
        # -- 角速度: 只罚 roll-rate、放松 pitch-rate --
        self.rewards.ang_vel_roll_l2 = RewardTermCfg(
            func=ang_vel_roll_l2,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "xy": (1.0, 0.0),
                "clip": 100.0,
            },
        )
        # -- 姿态保险: 左右对称 (不挡 pitch, 只罚侧向乱蹬) --
        self.rewards.leg_symmetry_l2 = RewardTermCfg(
            func=leg_symmetry_l2,
            weight=-0.1,
            params={
                "left_cfg": SceneEntityCfg("robot", joint_names=_LEFT_LEG_JOINTS, preserve_order=True),
                "right_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_LEG_JOINTS, preserve_order=True),
                "clip": 1.0e6,
            },
        )
        # -- 惩罚轮子离地 (平地 x-only 四轮应全程着地; 逼其用腿长重构俯仰而非翘后轮) --
        self.rewards.feet_off_ground = RewardTermCfg(
            func=feet_off_ground_l2,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
                "threshold": 1.0,
                "clip": 1.0e6,
            },
        )
        # endregion

        # 幂等补注入 obs sanitizer (含新加的 critic 特权项; 在 noise/scale 前, 健康数据数值不变)
        mdp.add_obs_sanitizers(self.observations, overrides=OBS_CLIP_BOUNDS)


@configclass
class TransportGo2WBaseControlEnvCfg_PLAY(TransportGo2WBaseControlEnvCfg):
    """测试/可视化版本 (加物体目视平台托运 + "加速前倾"行为)。"""

    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()

        from locotouch.assets.go2w_transport import Go2W_TRANSPORT_PLAY_CFG as Robot_PLAY_CFG
        self.scene.robot = Robot_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        from locotouch.config.base.locomotion_base_env_cfg import smaller_scene_for_playing
        smaller_scene_for_playing(self)

        env_num = self.scene.num_envs
        radius_range = (0.05, 0.05)  # (0.025, 0.075)

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
                    "z": (0.01, 0.01),
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

        # 控制play时的bang-bang比例
        self.commands.base_velocity.bang_bang_envs = 1.00

        if getattr(self, "curriculum", None) is not None:
            if getattr(self.curriculum, "command_x_levels", None) is not None:
                self.curriculum.command_x_levels.params["range_multiplier"] = (1.0, 1.0)
            if getattr(self.curriculum, "command_y_levels", None) is not None:
                self.curriculum.command_y_levels.params["range_multiplier"] = (1.0, 1.0)
            if getattr(self.curriculum, "command_z_levels", None) is not None:
                self.curriculum.command_z_levels.params["range_multiplier"] = (1.0, 1.0)
