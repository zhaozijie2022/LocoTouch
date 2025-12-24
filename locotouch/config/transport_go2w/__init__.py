import gymnasium as gym
from .agents import rsl_rl_ppo_cfg
from . import (
    transport_go2w_teacher_env_cfg
)

# ----------------------------------- Locomotion Go2W  -----------------------------------
gym.register(
    id="Isaac-TransportGo2WTeacher-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_go2w_teacher_env_cfg.TransportGo2WTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportGo2WTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-TransportGo2WTeacher-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_go2w_teacher_env_cfg.TransportGo2WTeacherEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportGo2WTeacherPPORunnerCfg,
    },
)
"""
Go2W 轮腿机器人运载任务

训练命令:
python locotouch/scripts/train.py --task Isaac-TransportGo2WTeacher-v1 --num_envs=4096 --headless

测试命令:
python locotouch/scripts/play.py --task Isaac-TransportGo2WTeacher-Play-v1 --num_envs=20 --load_run=2025-12-23_21-59-26

"""


