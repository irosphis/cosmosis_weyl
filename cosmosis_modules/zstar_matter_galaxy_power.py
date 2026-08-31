"""Build the explicit-z-star galaxy--matter template used by GGL.

The leading term is the Tutusaus et al. construction

    P_eff(k,z) = B(k,z) Q_star(k) / Omega_m(z),

where B=P_nl/P_lin and Q_star=P_lin(z_star)/sigma8(z_star)^2.  Projection is
followed by a separate multiplication by bhat*Jhat.  An optional residual
term keeps the existing KCAP nonlinear galaxy-bias correction; it is clearly
separated from the leading, explicitly normalized term.
"""

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from cosmosis.datablock import names, option_section


def setup(options):
    return {
        "z_star": options.get_double(option_section, "z_star", default=5.0),
        "z_sep": options.get_double(option_section, "z_sep", default=0.5),
        "include_bias_correction": options.get_bool(
            option_section, "include_nonlinear_bias_correction", default=True
        ),
        "bhat_section": options.get_string(
            option_section, "bhat_section", default="growth_amplitudes"
        ),
        "bias_section": options.get_string(
            option_section, "bias_section", default="bias_parameters"
        ),
        "derived_section": options.get_string(
            option_section, "derived_section", default="zstar_derived"
        ),
    }


def _increasing(name, values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError(
            f"zstar_matter_galaxy_power: {name} must be a 1D increasing grid"
        )
    return values


def _positive_power_on_grid(section_name, block, z_target, k_target):
    z = _increasing(f"{section_name} z", block[section_name, "z"])
    k = _increasing(f"{section_name} k_h", block[section_name, "k_h"])
    p = np.asarray(block[section_name, "p_k"], dtype=float)
    if p.shape != (z.size, k.size) or np.any(p <= 0.0):
        raise ValueError(
            f"zstar_matter_galaxy_power: {section_name} must be positive and "
            "have shape (n_z, n_k)"
        )
    if z_target[0] < z[0] or z_target[-1] > z[-1]:
        raise ValueError(
            f"zstar_matter_galaxy_power: target z is outside {section_name}"
        )
    if k_target[0] < k[0] * (1.0 - 1.0e-10) or k_target[-1] > k[-1] * (1.0 + 1.0e-10):
        raise ValueError(
            f"zstar_matter_galaxy_power: target k is outside {section_name}"
        )
    interpolator = RegularGridInterpolator(
        (z, np.log(k)), np.log(p), bounds_error=True
    )
    zz, logkk = np.meshgrid(z_target, np.log(k_target), indexing="ij")
    return np.exp(interpolator((zz, logkk)))


def execute(block, config):
    gm_section = "matter_galaxy_power"
    z = _increasing("matter_galaxy_power z", block[gm_section, "z"])
    k_h = _increasing("matter_galaxy_power k_h", block[gm_section, "k_h"])
    p_gm_reference = np.asarray(block[gm_section, "p_k"], dtype=float)
    if p_gm_reference.shape != (z.size, k_h.size):
        raise ValueError(
            "zstar_matter_galaxy_power: reference P_gm has the wrong shape"
        )

    p_lin = _positive_power_on_grid(names.matter_power_lin, block, z, k_h)
    p_nl = _positive_power_on_grid(names.matter_power_nl, block, z, k_h)
    boost = p_nl / p_lin

    z_growth = _increasing("growth z", block[names.growth_parameters, "z"])
    sigma8 = np.asarray(block[names.growth_parameters, "sigma_8"], dtype=float)
    if sigma8.shape != z_growth.shape or np.any(sigma8 <= 0.0):
        raise ValueError("zstar_matter_galaxy_power: sigma_8(z) must be positive")

    z_star = config["z_star"]
    if not (z[0] <= z_star <= z[-1] and z_growth[0] <= z_star <= z_growth[-1]):
        raise ValueError(
            f"zstar_matter_galaxy_power: z_star={z_star} is outside the grids"
        )
    p_star = CubicSpline(z, p_lin, axis=0)(z_star)
    sigma8_star = float(CubicSpline(z_growth, sigma8)(z_star))
    q_star = p_star / sigma8_star**2

    z_background = _increasing("distance z", block[names.distances, "z"])
    hubble = np.asarray(block[names.distances, "h"], dtype=float)
    if hubble.shape != z_background.shape or np.any(hubble <= 0.0):
        raise ValueError("zstar_matter_galaxy_power: H(z) must be positive")
    hubble_on_z = CubicSpline(z_background, hubble)(z)
    hubble_0 = float(CubicSpline(z_background, hubble)(0.0))
    omega_m0 = float(block[names.cosmological_parameters, "omega_m"])
    omega_m_z = omega_m0 * (1.0 + z) ** 3 * (hubble_0 / hubble_on_z) ** 2
    sigma8_z = CubicSpline(z_growth, sigma8)(z)
    jhat_reference = omega_m_z * sigma8_z

    # The leading spectrum contains neither bhat nor Jhat.  They are applied
    # only after the line-of-sight projection, just as in the explicit-z-star
    # angular construction.
    p_effective = boost * q_star[None, :] / omega_m_z[:, None]

    correction_fraction = 0.0
    if config["include_bias_correction"]:
        low = z < config["z_sep"]
        bhat = np.where(
            low,
            block[config["bhat_section"], "bhat_bin_1"],
            block[config["bhat_section"], "bhat_bin_2"],
        ).astype(float)
        b1_reference = np.where(
            low,
            block[config["bias_section"], "b1_bin_1"],
            block[config["bias_section"], "b1_bin_2"],
        ).astype(float)
        if np.any(bhat <= 0.0) or np.any(jhat_reference <= 0.0):
            raise ValueError(
                "zstar_matter_galaxy_power: bhat and reference Jhat must be positive"
            )
        nonlinear_bias = p_gm_reference - b1_reference[:, None] * p_nl
        correction = nonlinear_bias / (
            bhat[:, None] * jhat_reference[:, None]
        )
        p_effective = p_effective + correction
        denom = np.sqrt(np.mean(p_effective**2))
        if denom > 0.0:
            correction_fraction = float(np.sqrt(np.mean(correction**2)) / denom)

    if np.any(~np.isfinite(p_effective)):
        raise ValueError("zstar_matter_galaxy_power: effective P_gm is not finite")
    block[gm_section, "p_k"] = p_effective

    derived = config["derived_section"]
    block[derived, "z_star_ggl"] = float(z_star)
    block[derived, "sigma8_star_ggl"] = sigma8_star
    block[derived, "nonlinear_bias_correction_rms_fraction"] = correction_fraction
    return 0


def cleanup(config):
    pass
