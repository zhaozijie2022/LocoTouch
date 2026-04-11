import gymnasium as gym
from .agents import rsl_rl_ppo_cfg
from . import (
    transport_go2w_teacher_env_cfg,
    transport_go2w_base_control_env_cfg
)

# region -- Transport Go2W  --
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
# endregion

# region -- Base Control Go2W  --
gym.register(
    id="Isaac-TransportGo2WBaseControl-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_go2w_base_control_env_cfg.TransportGo2WBaseControlEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportGo2WBaseControlPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-TransportGo2WBaseControl-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_go2w_base_control_env_cfg.TransportGo2WBaseControlEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportGo2WBaseControlPPORunnerCfg,
    },
)
"""
Go2W 轮腿机器人运载任务

训练命令:
python locotouch/scripts/train.py --task Isaac-TransportGo2WBaseControl-v1 --num_envs=4096 --headless

python locotouch/scripts/train.py --task Isaac-TransportGo2WBaseControl-v1 --num_envs=4096 --headless --resume --load_run=2026-04-09_23-26-21 --checkpoint=model_5000.pt

测试命令:
python locotouch/scripts/play.py --task Isaac-TransportGo2WBaseControl-Play-v1 --num_envs=20 --load_run=2026-04-09_23-26-21

"""
# endregion