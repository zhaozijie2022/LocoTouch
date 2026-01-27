from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from . import custom_terrains



@configclass
class HfPerlinNoiseTerrainCfg(HfTerrainBaseCfg):
    """柏林噪声"""

    function = custom_terrains.perlin_noise_terrain

    # perlin 噪声参数
    frequency: float = 0.1                 # 噪声频率（控制起伏密度）
    octaves: int = 4                       # 噪声层数
    lacunarity: float = 2.0                 # 每层频率的倍增因子
    persistence: float = 0.5                # 每层振幅的衰减
    seed: int = 42                         # 随机种子


    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


@configclass
class HfXWaveTerrainCfg(HfTerrainBaseCfg):
    """X方向波浪, 等距离波长"""

    function = custom_terrains.x_wave_terrain

    amplitude_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the wave (in m)."""

    wave_length: float | tuple[float, float] = MISSING


@configclass
class HfSpeedBumpTerrainCfg(HfTerrainBaseCfg):
    """周期出现的减速带地形/ 适配人字形和梯形减速带"""
    function = custom_terrains.speed_bump_terrain

    num_bumps: int = 8
    """减速带数量"""

    bump_height_range: tuple[float, float] = MISSING
    """减速带高度范围 (m), 与difficulty有关"""

    random_flat_ratio: tuple[float, float] = MISSING
    """平顶减速带比例范围 (0-1), 与difficulty无关, 域随机化参数"""

    random_bump_width: tuple[float, float] = MISSING
    """减速带宽度范围 (m), 与difficulty无关, 域随机化参数"""

    num_gaps: int = 4
    """减速带间隙数量"""

    random_gap_length: tuple[float, float] = MISSING
    """减速带间隙长度范围 (m), 与difficulty无关, 域随机化参数"""

    gap_margin: float = 0.5
    """减速带间隙边缘预留 (m)"""

    platform_width: float = 2.0
    """场地中心平地尺寸 (m)"""

