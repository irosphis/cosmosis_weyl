from cosmosis.datablock import names, option_section
import numpy as np
import scipy.interpolate as interp

def setup(options):
    perbin = options.get_bool(option_section, "perbin", True)
    apply_to_cl = options.get_bool(option_section, "apply_to_cl", True)
    k_split = options.get_double(option_section, "k_split", 0.1)
    
    z_means = [0.295, 0.467, 0.626, 0.771] 
    
    return perbin, apply_to_cl, k_split, z_means

def execute(block, config):
    perbin, apply_to_cl, k_split, z_means = config

    z_arr = block[names.distances, 'z']
    chi_arr = block[names.distances, 'd_m']
    
    chi_func = interp.interp1d(z_arr, chi_arr)
    
    lens_chis = chi_func(z_means) 
    
    # ell_splits = k_split * lens_chis
    
    n_z_bins_pos = len(z_means)
    J_low_list = []
    J_high_list = []
    for i in range(1, n_z_bins_pos + 1):
        if perbin:
            J_low_list.append(block["J_bin_bias", f"J_low_{i}"])
            J_high_list.append(block["J_bin_bias", f"J_high_{i}"])
        else:
            J_low_list.append(block["J_bin_bias", "J_low_0"])
            J_high_list.append(block["J_bin_bias", "J_high_0"])

    # 获取 ell 坐标
    if block.has_section('galaxy_shear_cl'):
        ell_vals = block["galaxy_shear_cl", "ell"]
        n_z_bins_shear = block["galaxy_shear_cl", "nbin_b"]
    else:
        return 0

    # 循环应用
    # for i in range(n_z_bins_pos):
    #     val_l = J_low_list[i]
    #     val_h = J_high_list[i]
        
    #     mask_high = (ell_vals > ell_splits[i])
    #     mask_low = ~mask_high
        
    #     correction = np.ones_like(ell_vals)
    #     correction[mask_low] = val_l
    #     correction[mask_high] = val_h
        
    #     for j in range(n_z_bins_shear):
    #         name = f"bin_{i+1}_{j+1}"
    #         if block.has_value('galaxy_shear_cl', name):
    #             block["galaxy_shear_cl", name] *= correction


    delta_k = 0.02  # 这个值控制过渡的平滑程度（通常取 0.01 到 0.05 之间）

    for i in range(n_z_bins_pos):
        val_l = J_low_list[i]
        val_h = J_high_list[i]
        
        # 1. 物理映射：将当前的 ell_vals 转换为物理波数 k
        # 严格来说是 (ell_vals + 0.5) / chi，这比直接比对 ell 更符合物理定义
        k_vals = (ell_vals + 0.5) / lens_chis[i]
        
        # 2. 计算平滑窗口函数 W(k)
        # np.tanh(x) 在 x < 0 时趋近 -1，在 x > 0 时趋近 1
        # 所以 w_k 在 k << k_split 时趋近于 1.0 (低频区)
        # 在 k >> k_split 时趋近于 0.0 (高频区)
        w_k = 0.5 * (1.0 - np.tanh((k_vals - k_split) / delta_k))
        
        # 3. 组合出平滑、连续的修正项
        correction = val_l * w_k + val_h * (1.0 - w_k)
        
        # 4. 乘回角功率谱
        for j in range(n_z_bins_shear):
            name = f"bin_{i+1}_{j+1}"
            if block.has_value('galaxy_shear_cl', name):
                block["galaxy_shear_cl", name] *= correction
    return 0