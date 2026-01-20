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

    # 直接指定波长
    wave_length: float | tuple[float, float] = 1.0



