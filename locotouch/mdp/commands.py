from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg


# ----------------- Go2W -----------------
from isaaclab.managers import CommandTerm, CommandTermCfg

class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

# 增加multi-sampling + initial zero command steps 的 commands
class UniformVelocityCommandMultiSampling(UniformVelocityCommand):
    """UniformVelocityCommand + multi-sampling ranges + initial zero command steps + binary maximal command."""

    def __init__(self, cfg: UniformVelocityCommandMultiSamplingCfg, env: "ManagerBasedEnv | ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # --- snapshot initial ranges into previous_ranges ---
        self.cfg.previous_ranges.lin_vel_x = tuple(self.cfg.ranges.lin_vel_x)
        self.cfg.previous_ranges.lin_vel_y = tuple(self.cfg.ranges.lin_vel_y)
        self.cfg.previous_ranges.ang_vel_z = tuple(self.cfg.ranges.ang_vel_z)

        # flags
        self.lin_vel_x_equal_ranges = True
        self.lin_vel_y_equal_ranges = True
        self.ang_vel_z_equal_ranges = True
        self.first_time_set_ranges = True

        # build sampling bins (3 segments: new-low, old-mid, new-high)
        self.lin_vel_x_sampling_ranges = torch.tensor(
            [
                (self.cfg.ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[0]),
                (self.cfg.previous_ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[1]),
                (self.cfg.previous_ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_x[1]),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.lin_vel_y_sampling_ranges = torch.tensor(
            [
                (self.cfg.ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[0]),
                (self.cfg.previous_ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[1]),
                (self.cfg.previous_ranges.lin_vel_y[1], self.cfg.ranges.lin_vel_y[1]),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.ang_vel_z_sampling_ranges = torch.tensor(
            [
                (self.cfg.ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[0]),
                (self.cfg.previous_ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[1]),
                (self.cfg.previous_ranges.ang_vel_z[1], self.cfg.ranges.ang_vel_z[1]),
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # probabilities for [new, old, new]
        # 要求 new_command_probs <= 0.5，否则中间那项会负
        p = float(self.cfg.new_command_probs)
        mid = 1.0 - 2.0 * p
        if mid < 0:
            raise ValueError(f"new_command_probs too large: {p}, must satisfy 1-2p >= 0.")
        self.sampling_probs = torch.tensor([p, mid, p], device=self.device, dtype=torch.float32)

        # command buffer + init-zero logic
        self.vel_command_b_buffer = torch.zeros_like(self.vel_command_b)
        self.initial_zero_command_steps = int(self.cfg.initial_zero_command_steps)

        # binary maximal command
        self.binary_maximal_command = bool(self.cfg.binary_maximal_command)
        if self.binary_maximal_command:
            # 这里用 [-1, 1] 组合（你原本注释里也这样）
            scales = [-1, 1]
            combos = []
            for i in scales:
                for j in scales:
                    for k in scales:
                        combos.append([i, j, k])
            self.maximal_command_sampling = torch.tensor(combos, device=self.device, dtype=torch.float32)

        # (可选) 你想 log 一下当前 range / stand prob / init steps，就保留这些 metrics
        self.metrics["lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ang_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["initial_zero_command_steps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rel_standing_envs"] = torch.zeros(self.num_envs, device=self.device)

    def set_ranges(
        self,
        lin_vel_x: tuple[float, float] | None,
        lin_vel_y: tuple[float, float] | None,
        ang_vel_z: tuple[float, float] | None,
    ):
        """更新 range，同时维护 previous_ranges，并在 range 全相等时切到 final_* 配置。"""

        if lin_vel_x is not None:
            self.cfg.previous_ranges.lin_vel_x = tuple(self.cfg.ranges.lin_vel_x)
            self.cfg.ranges.lin_vel_x = tuple(lin_vel_x)
            self.lin_vel_x_equal_ranges = (self.cfg.previous_ranges.lin_vel_x == self.cfg.ranges.lin_vel_x)
            self.lin_vel_x_sampling_ranges[:] = torch.tensor(
                [
                    (self.cfg.ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[0]),
                    (self.cfg.previous_ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[1]),
                    (self.cfg.previous_ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_x[1]),
                ],
                device=self.device,
                dtype=torch.float32,
            )

        if lin_vel_y is not None:
            self.cfg.previous_ranges.lin_vel_y = tuple(self.cfg.ranges.lin_vel_y)
            self.cfg.ranges.lin_vel_y = tuple(lin_vel_y)
            self.lin_vel_y_equal_ranges = (self.cfg.previous_ranges.lin_vel_y == self.cfg.ranges.lin_vel_y)
            self.lin_vel_y_sampling_ranges[:] = torch.tensor(
                [
                    (self.cfg.ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[0]),
                    (self.cfg.previous_ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[1]),
                    (self.cfg.previous_ranges.lin_vel_y[1], self.cfg.ranges.lin_vel_y[1]),
                ],
                device=self.device,
                dtype=torch.float32,
            )

        if ang_vel_z is not None:
            self.cfg.previous_ranges.ang_vel_z = tuple(self.cfg.ranges.ang_vel_z)
            self.cfg.ranges.ang_vel_z = tuple(ang_vel_z)
            self.ang_vel_z_equal_ranges = (self.cfg.previous_ranges.ang_vel_z == self.cfg.ranges.ang_vel_z)
            self.ang_vel_z_sampling_ranges[:] = torch.tensor(
                [
                    (self.cfg.ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[0]),
                    (self.cfg.previous_ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[1]),
                    (self.cfg.previous_ranges.ang_vel_z[1], self.cfg.ranges.ang_vel_z[1]),
                ],
                device=self.device,
                dtype=torch.float32,
            )

        # 如果三维 range 都相等：切换到 final_*（保持你原逻辑）
        if self.lin_vel_x_equal_ranges and self.lin_vel_y_equal_ranges and self.ang_vel_z_equal_ranges:
            self.initial_zero_command_steps = int(self.cfg.final_initial_zero_command_steps)
            self.cfg.rel_standing_envs = float(self.cfg.final_rel_standing_envs)

        self.first_time_set_ranges = False

    def _update_metrics(self):
        # 只保留你想看的 metrics（完全去掉 gait/foot）
        self.metrics["lin_vel_x"][:] = float(self.cfg.ranges.lin_vel_x[1])
        self.metrics["lin_vel_y"][:] = float(self.cfg.ranges.lin_vel_y[1])
        self.metrics["ang_vel_z"][:] = float(self.cfg.ranges.ang_vel_z[1])
        self.metrics["initial_zero_command_steps"][:] = float(self.initial_zero_command_steps)
        self.metrics["rel_standing_envs"][:] = float(self.cfg.rel_standing_envs)

        # 你原本也在算 tracking error，如果还想保留就留：
        self.metrics["error_vel_xy"] = torch.linalg.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        self.metrics["error_vel_yaw"] = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if self.binary_maximal_command:
            idx = torch.randint(0, self.maximal_command_sampling.shape[0], (env_ids.numel(),), device=self.device)
            max_cmd = torch.tensor(
                [self.cfg.ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_y[1], self.cfg.ranges.ang_vel_z[1]],
                device=self.device,
                dtype=torch.float32,
            )
            self.vel_command_b[env_ids] = self.maximal_command_sampling[idx] * max_cmd
        else:
            # 若所有 range 相等：直接走父类采样（等价于普通 UniformVelocityCommand）
            if self.lin_vel_x_equal_ranges and self.lin_vel_y_equal_ranges and self.ang_vel_z_equal_ranges:
                super()._resample_command(env_ids.tolist())
            else:
                r = torch.empty(env_ids.numel(), device=self.device)

                # x
                if self.lin_vel_x_equal_ranges:
                    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
                else:
                    self._sample_dim_with_bins(env_ids, dim=0, bins=self.lin_vel_x_sampling_ranges)

                # y
                if self.lin_vel_y_equal_ranges:
                    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
                else:
                    self._sample_dim_with_bins(env_ids, dim=1, bins=self.lin_vel_y_sampling_ranges)

                # yaw
                if self.ang_vel_z_equal_ranges:
                    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
                else:
                    self._sample_dim_with_bins(env_ids, dim=2, bins=self.ang_vel_z_sampling_ranges)

                # standing envs
                self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= float(self.cfg.rel_standing_envs)

        # buffer 无论如何都更新（保持你原逻辑）
        self.vel_command_b_buffer[env_ids] = self.vel_command_b[env_ids].clone()

        # 初始置零（第一次 reset 也能生效）
        self._set_zero_command_for_beginning_steps()

    def _sample_dim_with_bins(self, env_ids: torch.Tensor, dim: int, bins: torch.Tensor):
        """按 sampling_probs 在三个区间采样并写入 vel_command_b[..., dim]."""
        bin_indices = torch.multinomial(self.sampling_probs, env_ids.numel(), replacement=True)
        for i in range(3):
            mask = (bin_indices == i)
            if not mask.any():
                continue
            selected_envs = env_ids[mask]
            low, high = float(bins[i, 0].item()), float(bins[i, 1].item())
            self.vel_command_b[selected_envs, dim] = torch.empty(selected_envs.numel(), device=self.device).uniform_(low, high)

    def _update_command(self):
        # 与你原来一致：先置零/恢复，再走父类更新（父类里会在到时 resample）
        self._set_zero_command_for_beginning_steps()
        self._recover_command_for_beginning_steps()
        super()._update_command()

    def _set_zero_command_for_beginning_steps(self):
        if self.initial_zero_command_steps <= 0:
            return
        mask = self._env.episode_length_buf < self.initial_zero_command_steps
        if mask.any():
            ids = mask.nonzero(as_tuple=True)[0]
            self.vel_command_b[ids] = 0.0  # 直接置零即可（不需要乘 buffer）

    def _recover_command_for_beginning_steps(self):
        if self.initial_zero_command_steps <= 0:
            return
        mask = self._env.episode_length_buf == self.initial_zero_command_steps
        if mask.any():
            ids = mask.nonzero(as_tuple=True)[0]
            self.vel_command_b[ids] = self.vel_command_b_buffer[ids].clone()


@configclass
class UniformVelocityCommandMultiSamplingCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandMultiSampling

    @configclass
    class PreviousRanges:
        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)

    previous_ranges: PreviousRanges = PreviousRanges()
    new_command_probs: float = 0.15
    final_rel_standing_envs: float = 0.0
    initial_zero_command_steps: int = 0
    final_initial_zero_command_steps: int = 0
    binary_maximal_command: bool = False


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """




# ----------------- LocoTouch -----------------
class UniformVelocityCommandGaitLogging(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandGaitLoggingCfg, env: ManagerBasedEnv|ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = self.cfg.sensor_cfg
        self.sensor_cfg.resolve(self._env.scene)
        self.contact_sensor: ContactSensor = self._env.scene.sensors[self.sensor_cfg.name]
        self.metrics["foot_air_time_variance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_step_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_1_step_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_2_step_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["step_air_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_1_air_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_2_air_time"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        self.metrics["error_vel_xy"] = torch.linalg.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        self.metrics["error_vel_yaw"] = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        last_air_time = self.contact_sensor.data.last_air_time[:, self.sensor_cfg.body_ids]
        self.metrics["foot_air_time_variance"] = torch.var(last_air_time, dim=1)

        gait_reward_name = "gait"
        gait_func = self._env.reward_manager.get_term_cfg(gait_reward_name).func
        if hasattr(gait_func, "valid_last_air_time") and gait_func.valid_last_air_time is not None:
            valid_last_air_time = gait_func.valid_last_air_time
            valid_last_air_time_env = torch.all(valid_last_air_time > 1.0e-6, dim=1)
            masked_valid_last_air_time = valid_last_air_time[valid_last_air_time_env]
            average_last_air_time = torch.mean(masked_valid_last_air_time)
            average_frequency = (1.0 / average_last_air_time / 2.0) if average_last_air_time > 0 else 0.0
            self.metrics["foot_step_frequency"][:] = average_frequency
            pair_1_average_last_air_time = torch.mean(masked_valid_last_air_time[:, [0, 1]])
            pair_1_average_frequency = (1.0 / pair_1_average_last_air_time / 2.0) if pair_1_average_last_air_time > 0 else 0.0
            self.metrics["pair_1_step_frequency"][:] = pair_1_average_frequency
            pair_2_average_last_air_time = torch.mean(masked_valid_last_air_time[:, [2, 3]])
            pair_2_average_frequency = (1.0 / pair_2_average_last_air_time / 2.0) if pair_2_average_last_air_time > 0 else 0.0
            self.metrics["pair_2_step_frequency"][:] = pair_2_average_frequency

            self.metrics["step_air_time"][:] = average_last_air_time if average_last_air_time > 0 else 0.0
            self.metrics["pair_1_air_time"][:] = pair_1_average_last_air_time if pair_1_average_last_air_time > 0 else 0.0
            self.metrics["pair_2_air_time"][:] = pair_2_average_last_air_time if pair_2_average_last_air_time > 0 else 0.0



@configclass
class UniformVelocityCommandGaitLoggingCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandGaitLogging
    sensor_cfg: SceneEntityCfg=SceneEntityCfg("robot_contact_senosr", body_names=".*foot")


class UniformVelocityCommandGaitLoggingMultiSampling(UniformVelocityCommandGaitLogging):
    def __init__(self, cfg: UniformVelocityCommandGaitLoggingMultiSamplingCfg, env: ManagerBasedEnv|ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.cfg.previous_ranges.lin_vel_x = tuple(self.cfg.ranges.lin_vel_x)
        self.cfg.previous_ranges.lin_vel_y = tuple(self.cfg.ranges.lin_vel_y)
        self.cfg.previous_ranges.ang_vel_z = tuple(self.cfg.ranges.ang_vel_z)
        self.lin_vel_x_equal_ranges = True
        self.lin_vel_y_equal_ranges = True
        self.ang_vel_z_equal_ranges = True
        self.first_time_set_ranges = True

        self.lin_vel_x_sampling_ranges = torch.tensor([(self.cfg.ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[0]),
                                                   (self.cfg.previous_ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[1]),
                                                   (self.cfg.previous_ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_x[1])], device=self.device)
        self.lin_vel_y_sampling_ranges = torch.tensor([(self.cfg.ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[0]),
                                                   (self.cfg.previous_ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[1]),
                                                   (self.cfg.previous_ranges.lin_vel_y[1], self.cfg.ranges.lin_vel_y[1])], device=self.device)
        self.ang_vel_z_sampling_ranges = torch.tensor([(self.cfg.ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[0]),
                                                   (self.cfg.previous_ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[1]),
                                                   (self.cfg.previous_ranges.ang_vel_z[1], self.cfg.ranges.ang_vel_z[1])], device=self.device)
        self.sampling_probs = torch.tensor([self.cfg.new_command_probs, 1.0 - 2 * self.cfg.new_command_probs, self.cfg.new_command_probs], device=self.device)
        
        self.vel_command_b_buffer = torch.zeros_like(self.vel_command_b)
        self.initial_zero_command_steps = self.cfg.initial_zero_command_steps
        self.binary_maximal_command = self.cfg.binary_maximal_command
        if self.binary_maximal_command:
            sample_scales = [-1, 0, 1]
            sample_scales = [-1, 1]
            sample_scales_len = len(sample_scales)
            self.maximal_command_sampling = torch.zeros((sample_scales_len**3, 3), device=self.device)
            index = 0
            for i in sample_scales:
                for j in sample_scales:
                    for k in sample_scales:
                        self.maximal_command_sampling[index] = torch.tensor([i, j, k], device=self.device)
                        index += 1

        self.metrics["lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ang_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["initial_zero_command_steps"] = torch.ones(self.num_envs, device=self.device) * self.initial_zero_command_steps
        self.metrics["rel_standing_envs"] = torch.ones(self.num_envs, device=self.device) * self.cfg.rel_standing_envs

    def set_ranges(self, lin_vel_x: tuple[float, float] | None, lin_vel_y: tuple[float, float] | None, ang_vel_z: tuple[float, float] | None):
        if lin_vel_x is not None:
            self.cfg.previous_ranges.lin_vel_x = tuple(self.cfg.ranges.lin_vel_x)
            self.cfg.ranges.lin_vel_x = tuple(lin_vel_x)
            self.lin_vel_x_equal_ranges = self.cfg.previous_ranges.lin_vel_x == self.cfg.ranges.lin_vel_x
            self.lin_vel_x_sampling_ranges[:] = torch.tensor([(self.cfg.ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[0]),
                                                    (self.cfg.previous_ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[1]),
                                                    (self.cfg.previous_ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_x[1])], device=self.device)
        if lin_vel_y is not None:
            self.cfg.previous_ranges.lin_vel_y = tuple(self.cfg.ranges.lin_vel_y)
            self.cfg.ranges.lin_vel_y = tuple(lin_vel_y)
            self.lin_vel_y_equal_ranges = self.cfg.previous_ranges.lin_vel_y == self.cfg.ranges.lin_vel_y
            self.lin_vel_y_sampling_ranges[:] = torch.tensor([(self.cfg.ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[0]),
                                                    (self.cfg.previous_ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[1]),
                                                    (self.cfg.previous_ranges.lin_vel_y[1], self.cfg.ranges.lin_vel_y[1])], device=self.device)
        if ang_vel_z is not None:
            self.cfg.previous_ranges.ang_vel_z = tuple(self.cfg.ranges.ang_vel_z)
            self.cfg.ranges.ang_vel_z = tuple(ang_vel_z)
            self.ang_vel_z_equal_ranges = self.cfg.previous_ranges.ang_vel_z == self.cfg.ranges.ang_vel_z
            self.ang_vel_z_sampling_ranges[:] = torch.tensor([(self.cfg.ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[0]),
                                                    (self.cfg.previous_ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[1]),
                                                    (self.cfg.previous_ranges.ang_vel_z[1], self.cfg.ranges.ang_vel_z[1])], device=self.device)
        print("^ " * 60)
        print("lin_vel_x_equal_ranges: ", self.lin_vel_x_equal_ranges)
        print("lin_vel_y_equal_ranges: ", self.lin_vel_y_equal_ranges)
        print("ang_vel_z_equal_ranges: ", self.ang_vel_z_equal_ranges)

        if self.lin_vel_x_equal_ranges and self.lin_vel_y_equal_ranges and self.ang_vel_z_equal_ranges:
            self.initial_zero_command_steps = self.cfg.final_initial_zero_command_steps
            self.cfg.rel_standing_envs = self.cfg.final_rel_standing_envs
            print("~ " * 60)
            print("Set initial_zero_command_steps to ", self.initial_zero_command_steps)
            print("Set rel_standing_envs to ", self.cfg.rel_standing_envs)

        self.first_time_set_ranges = False

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["lin_vel_x"][:] = self.cfg.ranges.lin_vel_x[1]
        self.metrics["lin_vel_y"][:] = self.cfg.ranges.lin_vel_y[1]
        self.metrics["ang_vel_z"][:] = self.cfg.ranges.ang_vel_z[1]
        self.metrics["initial_zero_command_steps"][:] = self.initial_zero_command_steps
        self.metrics["rel_standing_envs"][:] = self.cfg.rel_standing_envs
        last_air_time = self.contact_sensor.data.last_air_time[:, self.sensor_cfg.body_ids]
        self.metrics["foot_air_time_variance"] = torch.var(last_air_time, dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        if self.binary_maximal_command:
            sampling_indexed = torch.randint(0, self.maximal_command_sampling.shape[0], (len(env_ids),), device=self.device)
            maximal_commands = torch.tensor([self.cfg.ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_y[1], self.cfg.ranges.ang_vel_z[1]], device=self.device)
            self.vel_command_b[env_ids] = self.maximal_command_sampling[sampling_indexed] * maximal_commands
        else:
            if self.lin_vel_x_equal_ranges and self.lin_vel_y_equal_ranges and self.ang_vel_z_equal_ranges:
                super()._resample_command(env_ids)
            else:
                r = torch.empty(len(env_ids), device=self.device)
                if self.lin_vel_x_equal_ranges:
                    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
                else:
                    bin_indices = torch.multinomial(self.sampling_probs, len(env_ids), replacement=True)
                    for i, (low, high) in enumerate(self.lin_vel_x_sampling_ranges):
                        in_env_ids = bin_indices == i
                        if in_env_ids.any():
                            selected_envs = env_ids[torch.nonzero(in_env_ids, as_tuple=True)[0]]
                            self.vel_command_b[selected_envs, 0] = torch.empty(len(selected_envs), device=self.device).uniform_(low, high)
                if self.lin_vel_y_equal_ranges:
                    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
                else:
                    bin_indices = torch.multinomial(self.sampling_probs, len(env_ids), replacement=True)
                    for i, (low, high) in enumerate(self.lin_vel_y_sampling_ranges):
                        in_env_ids = bin_indices == i
                        if in_env_ids.any():
                            selected_envs = env_ids[torch.nonzero(in_env_ids, as_tuple=True)[0]]
                            self.vel_command_b[selected_envs, 1] = torch.empty(len(selected_envs), device=self.device).uniform_(low, high)
                if self.ang_vel_z_equal_ranges:
                    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
                else:
                    bin_indices = torch.multinomial(self.sampling_probs, len(env_ids), replacement=True)
                    for i, (low, high) in enumerate(self.ang_vel_z_sampling_ranges):
                        in_env_ids = bin_indices == i
                        if in_env_ids.any():
                            selected_envs = env_ids[torch.nonzero(in_env_ids, as_tuple=True)[0]]
                            self.vel_command_b[selected_envs, 2] = torch.empty(len(selected_envs), device=self.device).uniform_(low, high)
                # update standing envs
                self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # update buffer no matter what
        self.vel_command_b_buffer[env_ids] = self.vel_command_b[env_ids].clone()
        self._set_zero_command_for_beginning_steps()  # this is necessary for the first reset of the script

    def _update_command(self):
        self._set_zero_command_for_beginning_steps()
        self._recover_command_for_beginning_steps()
        super()._update_command()

    def _set_zero_command_for_beginning_steps(self):
        set_zero_command_envs = self._env.episode_length_buf < self.initial_zero_command_steps 
        if set_zero_command_envs.any():
            set_zero_command_envs = set_zero_command_envs.nonzero(as_tuple=True)[0]
            self.vel_command_b[set_zero_command_envs] = self.vel_command_b_buffer[set_zero_command_envs] * 0.0

    def _recover_command_for_beginning_steps(self):
        recover_command_envs = self._env.episode_length_buf == self.initial_zero_command_steps
        if recover_command_envs.any():
            recover_command_envs = recover_command_envs.nonzero(as_tuple=True)[0]
            self.vel_command_b[recover_command_envs] = self.vel_command_b_buffer[recover_command_envs].clone()


@configclass
class UniformVelocityCommandGaitLoggingMultiSamplingCfg(UniformVelocityCommandGaitLoggingCfg):
    class_type: type = UniformVelocityCommandGaitLoggingMultiSampling

    @configclass
    class PreviousRanges:
        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)

    previous_ranges: PreviousRanges = PreviousRanges()
    new_command_probs: float = 0.15
    final_rel_standing_envs: float = 0.0
    initial_zero_command_steps: int = 0
    final_initial_zero_command_steps: int = 0
    binary_maximal_command: bool = False

