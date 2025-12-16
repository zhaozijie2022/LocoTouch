# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


import math
import torch


class ModifyVelCommandsRangeBasedOnTrackingError:
    """Curriculum: expand command ranges based on tracking errors (NOT rewards).

    Tracks per-episode mean squared errors:
      e_lin = ||cmd_xy - base_vel_xy||^2
      e_ang = (cmd_wz - base_wz)^2
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

        p = cfg.params
        self.command_name = p["command_name"]
        self.max_ranges = torch.tensor(p["command_maximum_ranges"], device=env.device, dtype=torch.float32)
        self.bins = p.get("curriculum_bins", [20, 20, 20])

        # thresholds are now interpreted as *MSE thresholds* (unit: (m/s)^2, (rad/s)^2)
        self.err_th_lin = float(p["error_threshold_lin"])
        self.err_th_ang = float(p["error_threshold_ang"])

        self.repeat_lin = int(p.get("repeat_times_lin", 1))
        self.repeat_ang = int(p.get("repeat_times_ang", 1))

        # runtime buffers
        n = env.num_envs
        self.sum_lin = torch.zeros(n, device=env.device)
        self.sum_ang = torch.zeros(n, device=env.device)
        self.count = torch.zeros(n, device=env.device)

        # per-dimension difficulty index (0..bins-1)
        self.idx = torch.zeros((n, 3), device=env.device, dtype=torch.long)

        # per-dimension success streak
        self.streak_lin = torch.zeros(n, device=env.device, dtype=torch.long)
        self.streak_ang = torch.zeros(n, device=env.device, dtype=torch.long)

        # apply initial ranges immediately
        self._apply_ranges_to_command()

    def __call__(self, env, env_ids=None):
        # 1) accumulate per-step tracking error
        cmd = env.command_manager.get_command(self.command_name)  # (N,3): vx, vy, wz

        # pick your actual field names here (w vs b frame, depending on your project)
        v = env.scene["robot"].data.root_lin_vel_w  # (N,3)
        w = env.scene["robot"].data.root_ang_vel_w  # (N,3)

        lin_err = ((cmd[:, :2] - v[:, :2]) ** 2).sum(dim=1)   # (N,)
        ang_err = (cmd[:, 2] - w[:, 2]) ** 2                  # (N,)

        self.sum_lin += lin_err
        self.sum_ang += ang_err
        self.count += 1.0

        # 2) on episode end, update curriculum for done envs
        done = env.termination_manager.dones  # (N,) bool
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            self._update_for_envs(ids)
            self._reset_stats(ids)

    def _reset_stats(self, ids):
        self.sum_lin[ids] = 0.0
        self.sum_ang[ids] = 0.0
        self.count[ids] = 0.0

    def _update_for_envs(self, ids):
        mean_lin = self.sum_lin[ids] / torch.clamp(self.count[ids], min=1.0)
        mean_ang = self.sum_ang[ids] / torch.clamp(self.count[ids], min=1.0)

        lin_ok = mean_lin < self.err_th_lin
        ang_ok = mean_ang < self.err_th_ang

        # update streaks
        self.streak_lin[ids] = torch.where(lin_ok, self.streak_lin[ids] + 1, torch.zeros_like(self.streak_lin[ids]))
        self.streak_ang[ids] = torch.where(ang_ok, self.streak_ang[ids] + 1, torch.zeros_like(self.streak_ang[ids]))

        # expand bins when streak reaches repeat_times
        expand_lin = self.streak_lin[ids] >= self.repeat_lin
        expand_ang = self.streak_ang[ids] >= self.repeat_ang

        # idx order: [vx, vy, wz]
        # for lin: expand vx & vy together (you can split if you want)
        if expand_lin.any():
            lin_ids = ids[expand_lin]
            self.idx[lin_ids, 0] = torch.clamp(self.idx[lin_ids, 0] + 1, max=self.bins[0] - 1)
            self.idx[lin_ids, 1] = torch.clamp(self.idx[lin_ids, 1] + 1, max=self.bins[1] - 1)
            self.streak_lin[lin_ids] = 0

        if expand_ang.any():
            ang_ids = ids[expand_ang]
            self.idx[ang_ids, 2] = torch.clamp(self.idx[ang_ids, 2] + 1, max=self.bins[2] - 1)
            self.streak_ang[ang_ids] = 0

        self._apply_ranges_to_command()

    def _apply_ranges_to_command(self):
        # map idx (0..bins-1) -> ranges (0..max)
        # linear ramp; can replace with nonlinear schedule
        frac = torch.stack([
            self.idx[:, 0].float() / max(self.bins[0] - 1, 1),
            self.idx[:, 1].float() / max(self.bins[1] - 1, 1),
            self.idx[:, 2].float() / max(self.bins[2] - 1, 1),
        ], dim=1)  # (N,3)

        cur_max = frac * self.max_ranges[None, :]  # (N,3)

        # apply *global* ranges using the max across envs (simple & stable)
        global_max = cur_max.max(dim=0).values

        term = self.env.command_manager.get_term(self.command_name)
        term.cfg.ranges.lin_vel_x = (-float(global_max[0]), float(global_max[0]))
        term.cfg.ranges.lin_vel_y = (-float(global_max[1]), float(global_max[1]))
        term.cfg.ranges.ang_vel_z = (-float(global_max[2]), float(global_max[2]))

