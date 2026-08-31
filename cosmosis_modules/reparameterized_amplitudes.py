"""Map reparameterized late-time amplitudes onto the legacy bias interface.

The amplitude-reparameterized model samples bhat=b1*sigma8 and
fhat=f*sigma8 directly.  The existing KCAP BOSS and P_gm modules still
expect b1, so this module converts bhat to an effective b1 at each lens-bin
effective redshift using CAMB's reference sigma8(z).

The reference sigma8 is an internal normalization in this model.  At linear
order it cancels from observables after the corresponding fhat and Jhat
reparameterizations have been applied.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from cosmosis.datablock import names, option_section


def setup(options):
    z_eff = np.atleast_1d(options[option_section, "z_eff"]).astype(float)
    if z_eff.ndim != 1 or z_eff.size == 0:
        raise ValueError("reparameterized_amplitudes: z_eff must contain at least one redshift")

    return {
        "z_eff": z_eff,
        "input_section": options.get_string(
            option_section, "input_section", default="growth_amplitudes"
        ),
        "bias_section": options.get_string(
            option_section, "bias_section", default="bias_parameters"
        ),
        "derived_section": options.get_string(
            option_section, "derived_section", default="reparameterized_derived"
        ),
        "jhat_section": options.get_string(
            option_section, "jhat_section", default="J_bin_bias"
        ),
        "compute_eg": options.get_bool(
            option_section, "compute_eg", default=True
        ),
    }


def execute(block, config):
    z_growth = np.asarray(block[names.growth_parameters, "z"], dtype=float)
    sigma8_growth = np.asarray(
        block[names.growth_parameters, "sigma_8"], dtype=float
    )

    if z_growth.ndim != 1 or sigma8_growth.shape != z_growth.shape:
        raise ValueError(
            "reparameterized_amplitudes: growth z and sigma_8 arrays must be one-dimensional "
            "and have the same length"
        )
    if not np.all(np.isfinite(z_growth)) or not np.all(np.isfinite(sigma8_growth)):
        raise ValueError("reparameterized_amplitudes: growth arrays contain non-finite values")
    if not np.all(np.diff(z_growth) > 0.0):
        raise ValueError("reparameterized_amplitudes: growth redshift grid must be increasing")
    if np.any(sigma8_growth <= 0.0):
        raise ValueError("reparameterized_amplitudes: reference sigma_8(z) must be positive")

    sigma8_of_z = InterpolatedUnivariateSpline(z_growth, sigma8_growth, ext=2)

    if config["compute_eg"]:
        z_background = np.asarray(block[names.distances, "z"], dtype=float)
        hubble_background = np.asarray(block[names.distances, "h"], dtype=float)
        if (
            z_background.ndim != 1
            or hubble_background.shape != z_background.shape
            or not np.all(np.diff(z_background) > 0.0)
        ):
            raise ValueError(
                "reparameterized_amplitudes: distance z and h arrays must have matching "
                "one-dimensional increasing grids"
            )
        hubble_of_z = InterpolatedUnivariateSpline(
            z_background, hubble_background, ext=2
        )
        hubble_0 = float(hubble_of_z(0.0))
        if not np.isfinite(hubble_0) or hubble_0 <= 0.0:
            raise ValueError("reparameterized_amplitudes: reference H(0) must be positive")

    input_section = config["input_section"]
    bias_section = config["bias_section"]
    derived_section = config["derived_section"]

    for index, z_eff in enumerate(config["z_eff"], start=1):
        bhat = block[input_section, f"bhat_bin_{index}"]
        fhat = block[input_section, f"fhat_bin_{index}"]
        sigma8_eff = float(sigma8_of_z(z_eff))

        if not np.isfinite(bhat) or bhat <= 0.0:
            raise ValueError(
                f"reparameterized_amplitudes: bhat_bin_{index} must be positive and finite"
            )
        if not np.isfinite(fhat) or fhat <= 0.0:
            raise ValueError(
                f"reparameterized_amplitudes: fhat_bin_{index} must be positive and finite"
            )

        # Legacy modules consume b1.  Both the BOSS wedges and P_gm modules
        # read this same value, so bhat is shared by the two likelihoods.
        b1_eff = bhat / sigma8_eff
        f_eff = fhat / sigma8_eff
        block[bias_section, f"b1_bin_{index}"] = b1_eff

        # Save the normalization and converted parameters for diagnostics.
        block[derived_section, f"z_eff_bin_{index}"] = float(z_eff)
        block[derived_section, f"sigma8_ref_bin_{index}"] = sigma8_eff
        block[derived_section, f"b1_eff_bin_{index}"] = b1_eff
        block[derived_section, f"f_eff_bin_{index}"] = f_eff

        if config["compute_eg"]:
            jhat = block[config["jhat_section"], f"Jhat{index}"]
            if not np.isfinite(jhat) or jhat <= 0.0:
                raise ValueError(
                    f"reparameterized_amplitudes: Jhat{index} must be positive and finite"
                )
            # The E_G convention used by the paper contains the conformal
            # Hubble rate, H_conformal=a*H_physical.  The distances block
            # stores H_physical/c, so H_conformal/H0 is E(z)/(1+z).
            expansion_ratio = float(hubble_of_z(z_eff)) / hubble_0
            conformal_hubble_ratio = expansion_ratio / (1.0 + z_eff)
            eg = (
                conformal_hubble_ratio**2
                / (1.0 + z_eff)
                * jhat
                / fhat
            )
            block[derived_section, f"eg_bin_{index}"] = eg

    return 0


def cleanup(config):
    pass
