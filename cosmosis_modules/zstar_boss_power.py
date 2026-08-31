"""Temporarily expose the explicit z-star linear template to BOSS/gRPT.

The prepare mode replaces ``matter_power_lin`` by

    Q_star(k) = P_lin(k, z_star) / sigma8(z_star)^2

on every redshift slice.  The BOSS wrapper's ``use_zstar`` mode supplies the
late-time reference evolution inside gRPT.  A second instance of this module
in restore mode puts the original CAMB spectrum back before the lensing part
of the pipeline runs.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from cosmosis.datablock import names, option_section


def setup(options):
    mode = options.get_string(option_section, "mode", default="prepare").lower()
    if mode not in ("prepare", "restore"):
        raise ValueError("zstar_boss_power: mode must be 'prepare' or 'restore'")
    return {
        "mode": mode,
        "z_star": options.get_double(option_section, "z_star", default=5.0),
        "backup_section": options.get_string(
            option_section, "backup_section", default="zstar_power_backup"
        ),
        "derived_section": options.get_string(
            option_section, "derived_section", default="zstar_derived"
        ),
    }


def _validate_increasing(name, values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError(f"zstar_boss_power: {name} must be a 1D increasing grid")
    return values


def execute(block, config):
    power_section = names.matter_power_lin
    backup_section = config["backup_section"]

    if config["mode"] == "restore":
        if not block.has_value(backup_section, "p_k"):
            raise ValueError("zstar_boss_power: no saved CAMB spectrum to restore")
        block[power_section, "p_k"] = np.array(
            block[backup_section, "p_k"], dtype=float, copy=True
        )
        block[config["derived_section"], "boss_power_restored"] = 1
        return 0

    z = _validate_increasing("matter-power redshift", block[power_section, "z"])
    k_h = _validate_increasing("matter-power wavenumber", block[power_section, "k_h"])
    p_lin = np.asarray(block[power_section, "p_k"], dtype=float)
    if p_lin.shape != (z.size, k_h.size) or np.any(p_lin <= 0.0):
        raise ValueError(
            "zstar_boss_power: P_lin must be positive with shape (n_z, n_k)"
        )

    z_growth = _validate_increasing(
        "growth redshift", block[names.growth_parameters, "z"]
    )
    sigma8 = np.asarray(
        block[names.growth_parameters, "sigma_8"], dtype=float
    )
    if sigma8.shape != z_growth.shape or np.any(sigma8 <= 0.0):
        raise ValueError("zstar_boss_power: sigma_8(z) must be positive")

    z_star = config["z_star"]
    if not (z[0] <= z_star <= z[-1] and z_growth[0] <= z_star <= z_growth[-1]):
        raise ValueError(
            f"zstar_boss_power: z_star={z_star} is outside the CAMB/growth grid"
        )

    p_star = CubicSpline(z, p_lin, axis=0)(z_star)
    sigma8_star = float(CubicSpline(z_growth, sigma8)(z_star))
    q_star = p_star / sigma8_star**2
    if np.any(~np.isfinite(q_star)) or np.any(q_star <= 0.0):
        raise ValueError("zstar_boss_power: constructed Q_star is not positive")

    # A private copy is essential: DataBlock arrays can otherwise share memory.
    block[backup_section, "p_k"] = np.array(p_lin, copy=True)
    block[power_section, "p_k"] = np.tile(q_star, (z.size, 1))

    derived = config["derived_section"]
    block[derived, "z_star"] = float(z_star)
    block[derived, "sigma8_star"] = sigma8_star
    block[derived, "boss_power_prepared"] = 1
    block[derived, "qstar_k_h"] = np.array(k_h, copy=True)
    block[derived, "qstar_p_k"] = np.array(q_star, copy=True)
    return 0


def cleanup(config):
    pass
