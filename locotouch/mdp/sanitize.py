"""传感器/物理数据的全局 sanitize 工具 (nan/inf + 离群值防护)。

从 LocoWM (locowm/mdp/sanitize.py) 移植, 供 oracc 化的加速度追踪奖励与 critic 特权
加速度观测使用。

动机: PhysX 的原始物理量 (body_com_lin_acc_w / joint_acc / 接触力等) 在硬接触、
穿模、reset 瞬态时会瞬间冲到 1e6+ 甚至 inf/nan, 直接进入 obs / reward 会导致 loss 爆炸
乃至 nan 权重死锁。

设计:
    - ``sanitize(x, clip)``            : nan/±inf -> 有限, 并夹到 [-clip, clip]。obs 与 reward 共用。
    - ``obs_sanitize_modifier(...)``   : 作为 ObservationTermCfg.modifiers 注入 (在 func 后、
                                         noise/scale 前最先执行), 逐 obs 项防护。
    - ``add_obs_sanitizers(...)``      : 遍历所有 group 的所有 obs 项, 幂等地前插 sanitize modifier。

宽度默认 ``DEFAULT_CLIP`` 取一个正常物理量永不可达的值, 只拦 inf/nan 与物理爆炸;
每一项的宽度都可在 cfg 里逐项覆盖 (reward 项传 ``clip=``, obs 项经 ``overrides``)。
"""

from __future__ import annotations

import torch
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.modifiers import ModifierCfg

# 正常物理量永不可达; 仅拦截 inf/nan 与 1e6 级物理爆炸
DEFAULT_CLIP: float = 1.0e6


def sanitize(x: torch.Tensor, clip: float = DEFAULT_CLIP) -> torch.Tensor:
    """nan/±inf -> 有限, 并夹到 ``[-clip, clip]``。健康数据几乎无影响 (clip 极宽)。"""
    return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)


def obs_sanitize_modifier(data: torch.Tensor, clip: float = DEFAULT_CLIP) -> torch.Tensor:
    """ObservationTermCfg.modifiers 用: 逐 obs 项 sanitize。

    注意签名必须是 ``(data, 具名参数...)`` 且不含 ``**kwargs`` —— isaaclab 的 modifier
    校验 (observation_manager._prepare_terms) 用 inspect.signature 做集合比对,
    ``**kwargs`` 会被判为缺失强制参数而报错。
    """
    return sanitize(data, clip)


def add_obs_sanitizers(
    obs_cfg,
    default_clip: float = DEFAULT_CLIP,
    overrides: dict[str, float] | None = None,
) -> None:
    """遍历 ``obs_cfg`` 每个 group 的每个 ObservationTermCfg, 幂等地前插 sanitize modifier。

    - 幂等: 已含本 sanitizer 的项跳过, 可安全多次调用 (子类在 __post_init__ 追加 obs 项后再调一次)。
    - modifier 在 noise/scale 之前执行, 因此夹的是**原始** (pre-scale) 值。
    - ``overrides``: ``{term_attr_name: clip}`` 对个别项收紧 (如加速度类项)。
    """
    overrides = overrides or {}
    for group in vars(obs_cfg).values():
        if group is None:
            continue
        for name, term in vars(group).items():
            if not isinstance(term, ObservationTermCfg):
                continue
            mods = list(term.modifiers or [])
            if any(getattr(m, "func", None) is obs_sanitize_modifier for m in mods):
                continue
            clip = overrides.get(name, default_clip)
            term.modifiers = [ModifierCfg(func=obs_sanitize_modifier, params={"clip": clip})] + mods
