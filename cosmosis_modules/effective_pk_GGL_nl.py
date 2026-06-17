from cosmosis.datablock import names, option_section as opt
import numpy as np
import matplotlib.pyplot as plt
import camb
import warnings
from scipy.interpolate import interp1d
    
def setup(options):
    return []

def execute(block, config):
    # 1. 获取 zstar 处的线性功率谱和 sigma_8
    Plin_star = block[names.matter_power_lin, "p_k"][-1, :] 
    sig8_star = block[names.growth_parameters, "sigma_8"][-1] 
    
    # 2. 获取功率谱对应的红移网格
    z = block[names.matter_power_nl, "z"]
    
    # ==================== 【关键修复区：获取 H(z) / H0 】 ====================
    # 在标准 CosmoSIS 中，背景 H(z)/c 被储存在 distances 区块的 'h' 中
    z_bg = block[names.distances, "z"]
    h_bg = block[names.distances, "h"]
    
    # 将背景的 h_bg 插值到功率谱的红移网格 z 上
    h_interp = interp1d(z_bg, h_bg, kind='cubic')
    
    # 因为 h = H(z)/c，所以 h_interp(z) / h_interp(0) 正好就是无量纲的 H(z) / H0！
    # 这样彻底避免了单位换算的错误
    E_z = h_interp(z) / h_interp(0.0)
    # ======================================================================
    
    # 3. 计算非线性 Boost 因子
    B = block[names.matter_power_nl, "p_k"] / block[names.matter_power_lin, "p_k"]
    Omega_m0 = block[names.cosmological_parameters, "omega_m"]
    
    # 4. 构造外尔势骨架乘子: E(z)^2 / ((1+z)^3 * Omega_m)
    # E_z 是一个数组，算出来的乘子 reshape 成二维以便广播乘法
    skeleton_multiplier = (E_z**2 / ((1.0 + z)**3 * Omega_m0)).reshape(len(z), 1)
    print(skeleton_multiplier)

    # 5. 暴力覆盖非线性谱和线性谱！(变为纯外尔骨架)
    block[names.matter_power_nl, "p_k"] = B * Plin_star / sig8_star**2 * skeleton_multiplier
    # block[names.matter_power_lin, "p_k"] = Plin_star / sig8_star**2 * skeleton_multiplier

    # 6. 把 B 存入，如果下游有需要可以读（虽然极简排法中下游没再读了）
    block[names.matter_power_nl, "B"] = B
    
    return 0