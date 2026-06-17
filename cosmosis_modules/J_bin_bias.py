from builtins import range
from cosmosis.datablock import names, option_section
import sys

#该模块从 DataBlock 中读取观测量（例如 C_ell 或 ξ），
#并对每个透镜红移 bin（foreground lens bin）乘以一个校正因子 Jhat_i

# 流程中只运行一次
def setup(options):
    perbin = options.get_bool(option_section, "perbin", True)
    auto_only = options.get_bool(option_section, "auto_only", False) # 貌似无作用
    apply_to_cl = options.get_bool(option_section, "apply_to_cl", True)
    if apply_to_cl:
        print("Applying J bin biases to C_ell values")
    else:
        print("Applying J bin biases to xi values")
    return perbin, auto_only, apply_to_cl

# 从block中读取有多少个红移bin，定义的辅助函数 这里是角功率谱
def bins_count_cl(block):
    n_z_bins_shear = 0
    n_z_bins_pos = 0
    
    if block.has_section('galaxy_shear_cl'):
        if "galaxy_shear_xi" in block:
            raise ValueError("""You've asked to apply binwise bias to the C(l)s, but we found xis
                in the block, so this almost definitely unintended""")
        n_z_bins_shear = block["galaxy_shear_cl", "nbin_b"] #b是背景
        n_z_bins_pos = block["galaxy_shear_cl", "nbin_a"] #a是前景

    if n_z_bins_pos==0:
        raise ValueError("Used bin bias module with apply_to_cl=T but did not find any C_ell density tracers")

    return n_z_bins_pos, n_z_bins_shear #前景的bin数量和背景的bin数量

# 从block中读取有多少个红移bin，定义的辅助函数 这里是ξ关联函数
def bins_count_xi(block):
    n_z_bins_shear = 0
    n_z_bins_pos = 0
    
    if block.has_section('galaxy_shear_xi'):
        n_z_bins_shear = block["galaxy_shear_xi", "nbin_b"]
        n_z_bins_pos = block["galaxy_shear_xi", "nbin_a"]

    if n_z_bins_pos==0:
        raise ValueError("Used bin bias module with apply_to_cl=F but did not find any xi density tracers")

    return n_z_bins_pos, n_z_bins_shear

# 这里是执行模块，会在每次迭代中运行
def execute(block, options):
    perbin, auto_only, apply_to_cl = options

    if apply_to_cl:
        n_z_bins_pos, n_z_bins_shear = bins_count_cl(block)
    else:
        n_z_bins_pos, n_z_bins_shear = bins_count_xi(block)


    # We may be doing per-bin biases or a single global value
    if perbin:
        # per-bin - use b1,b2,b3, ...
        biases = [block["J_bin_bias", "Jhat%d" % pos_bin]
                  for pos_bin in range(1, n_z_bins_pos + 1)]
    else:
        # all the same - just use b0
        biases = [block["J_bin_bias", "Jhat0"] for pos_bin in range(n_z_bins_pos)]

    # 外循环n_z_bins_pos是透镜星系分bin
    for pos_bin1 in range(n_z_bins_pos):
        bias1 = biases[pos_bin1]

        if apply_to_cl:
            if block.has_section('galaxy_shear_cl'):
                # 内循环n_z_bins_shear是背景星系分bin
                for shear_bin in range(n_z_bins_shear):
                    name = "bin_{}_{}".format(pos_bin1 + 1, shear_bin + 1)
                    block["galaxy_shear_cl", name] *= bias1 # 将偏置乘到对应的 C_ell 上

        else:
            if block.has_section('galaxy_shear_xi'):
                for shear_bin in range(n_z_bins_shear):
                    name = "bin_{}_{}".format(pos_bin1 + 1, shear_bin + 1)
                    block['galaxy_shear_xi', name] *= bias1

    return 0
