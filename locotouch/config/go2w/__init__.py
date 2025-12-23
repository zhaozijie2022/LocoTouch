import gymnasium as gym
from .agents import rsl_rl_ppo_cfg
from . import (
    locomotion_go2w_env_cfg
)

# ----------------------------------- Locomotion Go2W  -----------------------------------
gym.register(
    id="Isaac-LocomotionGo2W-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_go2w_env_cfg.LocomotionGo2WEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionGo2WPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-LocomotionGo2W-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_go2w_env_cfg.LocomotionGo2WEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionGo2WPPORunnerCfg,
    },
)
"""
Go2W 轮腿机器人运载任务 - 16个关节（12腿+4轮）

训练命令:
python locotouch/scripts/train.py --task Isaac-LocomotionGo2W-v1 --num_envs=4096 --headless

测试命令:
python locotouch/scripts/play.py --task Isaac-LocomotionGo2W-Play-v1 --num_envs=20 --load_run=2025-12-23_00-02-59

"""


