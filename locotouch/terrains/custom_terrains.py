
from __future__ import annotations  # 将来类型注解作为字符串处理, 防止循环导入

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import custom_terrains_cfg


import noise


@height_field_to_mesh
def perlin_noise_terrain(difficulty: float, cfg: custom_terrains_cfg.HfPerlinNoiseTerrainCfg) -> np.ndarray:
    """ 借用 hf_terrains.random_uniform_terrain, 只更改了核心的随机采样部分
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # sample heights randomly from the range along a grid
    height_field_downsampled = np.zeros((width_downsampled, length_downsampled))
    for i in range(width_downsampled):
        for j in range(length_downsampled):
            # 生成Perlin噪声值 (-1.0 ~ 1.0)
            n = noise.pnoise2(
                i * cfg.frequency,          # X坐标缩放（控制噪声频率）
                j * cfg.frequency,          # Y坐标缩放
                octaves=cfg.octaves,        # 噪声层数（推荐4~8）
                persistence=cfg.persistence,            # 每层幅度衰减
                lacunarity=cfg.lacunarity,             # 每层频率增长
                repeatx=width_downsampled,  # 周期性（避免边缘不连续）
                repeaty=length_downsampled,
                base=cfg.seed               # 随机种子
            )
            # 映射到 [height_min, height_max]
            h = height_min + (n + 1) * 0.5 * (height_max - height_min)
            height_field_downsampled[i, j] = h
    # create interpolation function for the sampled heights


    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)




@height_field_to_mesh
def x_wave_terrain(difficulty: float, cfg: custom_terrains_cfg.HfXWaveTerrainCfg) -> np.ndarray:
    r"""
        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right)
    """
    if isinstance(cfg.wave_length, tuple):
        wave_length = cfg.wave_length[0] + difficulty * (
            cfg.wave_length[1] - cfg.wave_length[0]
        )
    else:
        wave_length = cfg.wave_length

    if wave_length <= 0:
        raise ValueError(f"wave_length must be positive. Got: {wave_length}")

    # resolve terrain configuration
    # 一块地形只有一个振幅
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # compute the wave number: nu = 2 * pi / lambda
    wave_length_pixels = wave_length / cfg.horizontal_scale
    wave_number = 2 * np.pi / wave_length_pixels

    # create meshgrid for the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the waves
    hf_raw += amplitude_pixels * np.sin(xx * wave_number)
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)