
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

    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)

    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])

    x = np.linspace(0, cfg.size[0], width_px)
    y = np.linspace(0, cfg.size[1], length_px)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    h_meters = amplitude * np.sin(2.0 * np.pi * xv / wave_length)
    hf = np.round(h_meters / cfg.vertical_scale).astype(np.int16)

    return hf




@height_field_to_mesh
def speed_bump_terrain(difficulty: float, cfg: custom_terrains_cfg.HfSpeedBumpTerrainCfg) -> np.ndarray:
    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((width_px, length_px), dtype=np.int_)

    actual_height = cfg.bump_height_range[0] + difficulty * (cfg.bump_height_range[1] - cfg.bump_height_range[0])
    bump_height_px = int(actual_height / cfg.vertical_scale)  # 高度

    # 每条减速带的上坡起始位置
    x_starts_px = np.arange(width_px / cfg.num_bumps / 2, width_px, width_px / cfg.num_bumps).astype(dtype=np.int_)
    # 采样每条减速带的形状
    bump_width = np.random.uniform(cfg.random_bump_width[0], cfg.random_bump_width[1], size=cfg.num_bumps)
    flat_ratio = np.random.uniform(cfg.random_flat_ratio[0], cfg.random_flat_ratio[1], size=cfg.num_bumps)
    flat_width = bump_width * flat_ratio
    ramp_width = (bump_width - flat_width) / 2
    ramp_width_px = np.round(ramp_width / cfg.horizontal_scale).astype(np.int_)
    flat_width_px = np.round(flat_width / cfg.horizontal_scale).astype(np.int_)
    bump_width_px = ramp_width_px * 2 + flat_width_px

    # 生成周期的减速带
    for i in range(width_px):
        h = 0  # 当前高度
        for k in range(cfg.num_bumps):
            dx_px = i - x_starts_px[k]
            if 0 <= dx_px < bump_width_px[k]:
                if dx_px < ramp_width_px[k]:  # 上坡
                    # t = t * t * (3.0 - 2.0 * t)
                    h = bump_height_px / ramp_width_px[k] * dx_px
                elif dx_px < ramp_width_px[k] + flat_width_px[k]: # 平台
                    h = bump_height_px
                else: # 下坡
                    # t = t * t * (3.0 - 2.0 * t)
                    h = bump_height_px / ramp_width_px[k] * (bump_width_px[k] - dx_px)
                break
        hf[i, :] = int(round(h))

    # 在中心挖一个平坦的平台，方便放置机器人
    platform_radius_px = int(cfg.platform_width/ 2 / cfg.horizontal_scale)
    center_x_px = width_px // 2
    center_y_px = length_px // 2
    hf[
        center_x_px - platform_radius_px: center_x_px + platform_radius_px,
        center_y_px - platform_radius_px: center_y_px + platform_radius_px
    ] = 0

    # 在每条减速带内沿 y 随机挖空若干段（制造间断）
    y_periods = np.linspace(cfg.gap_margin, (length_px - cfg.gap_margin), cfg.num_gaps + 1)
    y_periods_px = np.round(y_periods).astype(np.int_)

    for k in range(cfg.num_bumps):
        gap_length = np.random.uniform(cfg.random_gap_length[0], cfg.random_gap_length[1], size=cfg.num_gaps)
        gap_length_px = np.round(gap_length / cfg.horizontal_scale).astype(np.int_)

        for p in range(cfg.num_gaps):
            y0 = int(np.random.uniform(y_periods_px[p], y_periods_px[p + 1] - gap_length_px[p]))
            y1 = y0 + gap_length_px[p]

            hf[x_starts_px[k]:x_starts_px[k] + bump_width_px[k], y0:y1] = 0

    return hf