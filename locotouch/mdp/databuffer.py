"""每步只推进一次的低通滤波 (StepwiseLPF)。

从 LocoWM (locowm/mdp/databuffer.py) 移植, 供 ``custom_base_lin_acc`` 对 base 加速度做
平滑, 用于加速度追踪奖励 (track_pitch / jerk / acc_soft) 与 critic 的
``ideal_projected_gravity`` 特权观测 —— 保证「加速度经过正确滤波」。
"""

from __future__ import annotations

import math
import torch


class StepwiseLPF:
    """每一步调用一次, 不会因为 policy / critic / reward 的多次调用而重复前进滤波

        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
        alpha = 1 - exp(-2*pi * f_cut / f_ctrl)

    对刚 reset 的环境 (reset_mask=True) 直接输出原始值, 避免上一 episode 的
    历史污染新 episode 的首帧。
    """

    def __init__(self, cut_off_frequency: float, control_frequency: float):
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * cut_off_frequency / control_frequency)
        self._y: torch.Tensor | None = None   # 上一次(即当前 step)的滤波输出
        self._last_step: int | None = None    # 上一次前进滤波所在的 step

    def __call__(
        self,
        x: torch.Tensor,
        step: int,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 首次调用或 batch 形状变化 -> 用原始值初始化
        if self._y is None or self._y.shape != x.shape:
            self._y = x.clone()
            self._last_step = None

        # 同一 step 的重复调用 -> 不再前进, 返回已平滑值
        if step == self._last_step:
            return self._y

        y = self.alpha * x + (1.0 - self.alpha) * self._y
        if reset_mask is not None:
            y = torch.where(reset_mask, x, y)   # 新 episode 首帧: 用原始值, 丢弃旧历史

        self._y = y
        self._last_step = step
        return y
