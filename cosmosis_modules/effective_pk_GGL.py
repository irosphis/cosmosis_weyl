from cosmosis.datablock import names, option_section as opt
import numpy as np
import matplotlib.pyplot as plt
import camb
import warnings
from scipy.interpolate import interp1d
    
def setup(options):
    return []
    
def execute(block, config):
    # Getting Plin at z=zstar 得到线性zstar处功率谱
    Plin_star=block[names.matter_power_lin,"P_K"][-1,:] #最后一个红移点处的线性功率谱，注意lin的是始终没有修改过的
    
    # Getting different quantities from the datablock
    H=block[names.growth_parameters, "H"]
    H0=block[names.cosmological_parameters, "h0"]*100.
    z=block[names.matter_power_nl,"z"]
    B=block[names.matter_power_nl,"P_K"]/block[names.matter_power_lin, "P_K"] #boost factor
    Omega_m0=block[names.cosmological_parameters, "Omega_m"]
    sig8_z=block[names.growth_parameters,"sigma_8"]
    
    # Getting sigma8 at z=zstar
    sig8_star=block[names.growth_parameters,"sigma_8"][-1] #最后一个红移点处的sigma8
    
    #Replacing the nonlinear matter power spectra by the effective one removing Jhat
    block[names.matter_power_nl,"P_K"] = B*Plin_star/sig8_star**2*np.array((H/H0)**2/(1.+z)**3/Omega_m0).reshape(len(z),1)

    #Storing the boost into the data block  用于计算并保存原始 Boost (B = P_nl / P_lin) 在后面的GC计算中，将用于计算非线性Pnl，这样可以确保初始的Pk保持不变
    block[names.matter_power_nl,"B"] = B
    
    #Printing Jhat fiducials
    # Jhat_z=(np.array(sig8_z*Omega_m0*(1.+z)**3*H0**2/H**2).reshape(len(z)))
    # Jhat_func=interp1d(z,Jhat_z)
    #print('Jhat fiducials:',Jhat_func([0.2946683204029144, 0.46670347176087024, 0.626278460689437, 0.771314100934901,0.9,1.0]))
    #print('Jhat fiducials:',Jhat_func([0.2796,0.4367,0.5795,0.7271,0.8486]))
    
    return 0
