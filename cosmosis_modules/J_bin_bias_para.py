from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np  # 必须引入 numpy 进行数组操作

# ---------------------------------------------------------
# 1. 定义你的尺度依赖模型
# 这是一个唯象模型示例，你可以根据物理需要随意修改
# ---------------------------------------------------------
def compute_scale_factor(ell, A, ell_0, n_index):
    """
    计算尺度依赖因子 f(l)。
    J(l) = J_amp * f(l)
    这里假设 f(l) = 1 + A * (ell / ell_0)^n
    当 A=0 时，退化为常数 J。
    """
    # 避免除以0
    ell_0 = max(ell_0, 1.0)
    
    # 计算形状修正
    shape_correction = 1.0 + A * (ell / ell_0)**n_index
    
    return shape_correction

# ---------------------------------------------------------
# 2. Setup 保持大部分不变，但可能需要读取新的配置项
# ---------------------------------------------------------
def setup(options):
    perbin = options.get_bool(option_section, "perbin", True)
    auto_only = options.get_bool(option_section, "auto_only", False)
    apply_to_cl = options.get_bool(option_section, "apply_to_cl", True)
    
    # 读取尺度依赖的参数配置（这里设为全局参数，也可以设为每个bin独立）
    # 默认值 A=0 代表无尺度依赖（回到原文章的情况）
    mg_A = options.get_double(option_section, "mg_A", 0.0) 
    mg_ell0 = options.get_double(option_section, "mg_ell0", 1000.0)
    mg_n = options.get_double(option_section, "mg_n", 2.0)

    if apply_to_cl:
        print(f"Applying Scale-Dependent J to C_ell values.")
    #     print(f"Model: J(l) = J_amp * [1 + {mg_A} * (l / {mg_ell0})^{mg_n}]")
    # else:
    #     # 警告：直接在 xi 上乘尺度因子通常是物理上错误的
    #     print("WARNING: Applying scale dependence directly to xi is mathematically incorrect!")
    #     print("You should apply to C_ell first, then transform to xi.")
        
    return perbin, auto_only, apply_to_cl, (mg_A, mg_ell0, mg_n)

# 辅助函数保持不变
def bins_count_cl(block):
    n_z_bins_shear = 0
    n_z_bins_pos = 0
    if block.has_section('galaxy_shear_cl'):
        if "galaxy_shear_xi" in block:
            raise ValueError("Found xi but asked for cl application.")
        n_z_bins_shear = block["galaxy_shear_cl", "nbin_b"]
        n_z_bins_pos = block["galaxy_shear_cl", "nbin_a"]
    if n_z_bins_pos==0:
        raise ValueError("No C_ell density tracers found.")
    return n_z_bins_pos, n_z_bins_shear

def bins_count_xi(block):
    n_z_bins_shear = 0
    n_z_bins_pos = 0
    if block.has_section('galaxy_shear_xi'):
        n_z_bins_shear = block["galaxy_shear_xi", "nbin_b"]
        n_z_bins_pos = block["galaxy_shear_xi", "nbin_a"]
    if n_z_bins_pos==0:
        raise ValueError("No xi density tracers found.")
    return n_z_bins_pos, n_z_bins_shear

# ---------------------------------------------------------
# 3. Execute 模块：引入 ell 数组运算
# ---------------------------------------------------------
def execute(block, config):
    # 解包配置参数，包括新的 MG 参数
    perbin, auto_only, apply_to_cl, mg_params = config
    fixed_A, ell0, n = mg_params

    if block.has_value("My_J_bin_bias", "mg_A"):
        A = block["My_J_bin_bias", "mg_A"]
        print(f"### Current mg_A: {A:.5f}")
    else:
        print("Using default mg_A from config:", fixed_A)
        A = fixed_A

    if apply_to_cl:
        n_z_bins_pos, n_z_bins_shear = bins_count_cl(block)
        
        # === 关键修改 1: 获取 ell 数组 ===
        # 通常 ell 存储在同一个 section 下
        if block.has_value("galaxy_shear_cl", "ell"):
            ell_vals = block["galaxy_shear_cl", "ell"]
        else:
            raise ValueError("Cannot find 'ell' array in galaxy_shear_cl section!")
            
        # === 关键修改 2: 计算形状因子向量 ===
        # 这是一个与 C_ell 长度相同的数组
        scale_factor = compute_scale_factor(ell_vals, A, ell0, n)
        
    else:
        n_z_bins_pos, n_z_bins_shear = bins_count_xi(block)
        # 如果是 xi，我们这里暂时不做尺度依赖，或者只做常数乘法
        # 因为 ell 在这里没有定义
        scale_factor = 1.0 

    # 读取 J_amp (原本的 Jhat，现在作为整体幅度)
    if perbin:
        biases_amp = [block["J_bin_bias", "Jhat%d" % pos_bin]
                      for pos_bin in range(1, n_z_bins_pos + 1)]
    else:
        biases_amp = [block["J_bin_bias", "Jhat0"] for pos_bin in range(n_z_bins_pos)]

    # 循环应用
    for pos_bin1 in range(n_z_bins_pos):
        J_amp = biases_amp[pos_bin1] # 这是 J 的整体幅度 (红移依赖)

        if apply_to_cl:
            if block.has_section('galaxy_shear_cl'):
                for shear_bin in range(n_z_bins_shear):
                    name = "bin_{}_{}".format(pos_bin1 + 1, shear_bin + 1)
                    
                    # 获取原始 Cl
                    old_cl = block["galaxy_shear_cl", name]
                    
                    # === 关键修改 3: 组合幅度和形状 ===
                    # J_total(l, z) = J_amp(z) * Shape(l)
                    # C_new = C_old * J_total
                    # 注意：如果 J 是乘在功率谱上的平方项 (P_psi ~ J^2)，这里要不要平方？
                    # 答：文章中 C_gl ~ <g, gamma> ~ J。是一次项。所以直接乘。
                    
                    J_total_array = J_amp * scale_factor
                    
                    # 执行数组乘法
                    new_cl = old_cl * J_total_array
                    
                    # 写回 block
                    block["galaxy_shear_cl", name] = new_cl

        else:
            # 对于 xi，只能做简单的幅度缩放 (J_amp)，不能做简单的尺度依赖
            if block.has_section('galaxy_shear_xi'):
                for shear_bin in range(n_z_bins_shear):
                    name = "bin_{}_{}".format(pos_bin1 + 1, shear_bin + 1)
                    block['galaxy_shear_xi', name] *= J_amp

    return 0