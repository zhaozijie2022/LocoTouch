from isaaclab.utils import configclass
from locotouch.config.base.locomotion_vel_cur_base_env_cfg import LocomotionVelCurBaseEnvCfg, locomotion_vel_cur_play_env_post_init_func
from .locomotion_go2w_env_cfg import go2w_locomotion_train_env_post_init_func, go2w_locomotion_play_env_post_init_func


def go2w_locomotion_vel_cur_train_env_post_init_func(env_cfg: LocomotionVelCurBaseEnvCfg) -> None:
    go2w_locomotion_train_env_post_init_func(env_cfg)

def go2w_locomotion_vel_cur_play_env_post_init_func(env_cfg: LocomotionVelCurBaseEnvCfg) -> None:
    go2w_locomotion_play_env_post_init_func(env_cfg)
    locomotion_vel_cur_play_env_post_init_func(env_cfg)


@configclass
class LocomotionVelCurEnvCfg(LocomotionVelCurBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        go2w_locomotion_vel_cur_train_env_post_init_func(self)


class LocomotionVelCurEnvCfg_PLAY(LocomotionVelCurEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        go2w_locomotion_vel_cur_play_env_post_init_func(self)

