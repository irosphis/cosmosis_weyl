from builtins import range
from cosmosis.datablock import names, option_section
import numpy as np
from scipy.interpolate import interp1d

def setup(options):
    return options.get_bool(option_section, "apply_to_cl", True)

def execute(block, apply_to_cl):
    if not apply_to_cl: return 0

    n_z_bins_shear = block.get_int("galaxy_shear_cl", "nbin_b", default=0)
    n_z_bins_pos = block.get_int("galaxy_shear_cl", "nbin_a", default=0)
    if n_z_bins_pos == 0: return 0

    # ==================== 1. 读取 Weyl 振幅 (Jhat) ====================
    Jhats =[block.get_double("J_bin_bias", "Jhat%d" % i, 1.0) for i in range(1, n_z_bins_pos + 1)]

    # ==================== 2. 获取基础宇宙学参数 ====================
    Omega_m0 = block[names.cosmological_parameters, "omega_m"]
    
    # 获取 E(z) = H(z)/H0
    z_bg = block[names.distances, "z"]
    h_bg = block[names.distances, "h"]
    h_interp = interp1d(z_bg, h_bg, kind='cubic')
    
    def Omega_m_z(z_val):
        E_z = h_interp(z_val) / h_interp(0.0)
        return Omega_m0 * (1.0 + z_val)**3 / (E_z**2)

    # 获取 sigma_8(z)
    z_growth = block[names.growth_parameters, "z"]
    sig8_growth = block[names.growth_parameters, "sigma_8"]
    sig8_interp = interp1d(z_growth, sig8_growth, kind='linear', fill_value="extrapolate")

    # ==================== 3. 执行物理映射 ====================
    z_nz = block["nz_lens", "z"]
    
    for i in range(1, n_z_bins_pos + 1):
        # 提取当前透镜 Bin 的有效红移
        nz_i = block["nz_lens", f"bin_{i}"]
        z_eff = np.average(z_nz, weights=nz_i)
        
        # 计算该红移处 GR 对应的 Jhat 值
        Jhat_GR = Omega_m_z(z_eff) * sig8_interp(z_eff)
        
        # 目标核 / GR 核 的严格比例
        mapping_ratio = Jhats[i-1] / Jhat_GR

        # 映射到 GGL 信号上
        if block.has_section('galaxy_shear_cl'):
            for j in range(1, n_z_bins_shear + 1):
                name = f"bin_{i}_{j}"
                if block.has_value('galaxy_shear_cl', name):
                    cl_array = block["galaxy_shear_cl", name]
                    block["galaxy_shear_cl", name] = cl_array * mapping_ratio

    return 0