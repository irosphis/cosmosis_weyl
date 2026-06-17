from builtins import range
from cosmosis.datablock import names, option_section
import numpy as np

def setup(options):
    perbin = options.get_bool(option_section, "perbin", True)
    apply_to_cl = options.get_bool(option_section, "apply_to_cl", True)
    ell_split = options.get_double(option_section, "ell_split", 500.0)
    
    print(f"Setup J_bin_bias Independent Band-Power Mode.")
    print(f"Splitting at ell = {ell_split}")
    
    return perbin, apply_to_cl, ell_split

def bins_count_cl(block):
    if block.has_section('galaxy_shear_cl'):
        n_z_bins_shear = block["galaxy_shear_cl", "nbin_b"]
        n_z_bins_pos = block["galaxy_shear_cl", "nbin_a"]
    else:
        raise ValueError("No galaxy_shear_cl found.")
    return n_z_bins_pos, n_z_bins_shear

def execute(block, config):
    perbin, apply_to_cl, ell_split = config

    if not apply_to_cl:
        raise ValueError("Must apply to C_ell.")

    n_z_bins_pos, n_z_bins_shear = bins_count_cl(block)

    # 1. 获取 ell 数组
    if block.has_value("galaxy_shear_cl", "ell"):
        ell_vals = block["galaxy_shear_cl", "ell"]
    else:
        raise ValueError("Cannot find 'ell' in DataBlock.")

    # 2. 生成掩码 (Mask)
    # mask_low: 对应小 ell 的位置 (True)
    # mask_high: 对应大 ell 的位置 (True)
    mask_high = (ell_vals > ell_split)
    mask_low = ~mask_high  # 取反，即 ell <= ell_split

    # 3. 读取参数 (J_low 和 J_high)
    # 这一步把所有的参数读出来存到列表里
    J_low_list = []
    J_high_list = []

    for i in range(1, n_z_bins_pos + 1):
        if perbin:
            # 读取 J_low_1, J_low_2 ...
            val_low = block["J_bin_bias", f"J_low_{i}"]
            # 读取 J_high_1, J_high_2 ...
            val_high = block["J_bin_bias", f"J_high_{i}"]
        else:
            # 全局统一的情况
            val_low = block["J_bin_bias", "J_low_0"]
            val_high = block["J_bin_bias", "J_high_0"]
            
        J_low_list.append(val_low)
        J_high_list.append(val_high)

    # 4. 循环应用
    for i in range(n_z_bins_pos):
        # 当前 bin 的两个独立参数
        val_l = J_low_list[i]
        val_h = J_high_list[i]
        
        # 构造修正数组
        # 创建一个全 1 数组，或者空数组
        correction_vector = np.zeros_like(ell_vals)
        
        # 填入大尺度部分
        correction_vector[mask_low] = val_l
        # 填入小尺度部分
        correction_vector[mask_high] = val_h
        
        # 应用到 C_ell
        if block.has_section('galaxy_shear_cl'):
            for j in range(n_z_bins_shear):
                name = f"bin_{i+1}_{j+1}"
                old_cl = block["galaxy_shear_cl", name]
                
                # 数组乘法
                new_cl = old_cl * correction_vector
                
                block["galaxy_shear_cl", name] = new_cl

    return 0