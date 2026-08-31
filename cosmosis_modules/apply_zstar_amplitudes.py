"""Apply the late-time bhat*Jhat amplitudes after GGL projection."""

import numpy as np

from cosmosis.datablock import option_section


def setup(options):
    return {
        "cl_section": options.get_string(
            option_section, "cl_section", default="galaxy_shear_cl"
        ),
        "bhat_section": options.get_string(
            option_section, "bhat_section", default="growth_amplitudes"
        ),
        "jhat_section": options.get_string(
            option_section, "jhat_section", default="J_bin_bias"
        ),
        "derived_section": options.get_string(
            option_section, "derived_section", default="zstar_derived"
        ),
    }


def execute(block, config):
    section = config["cl_section"]
    n_lens = block.get_int(section, "nbin_a", default=0)
    n_source = block.get_int(section, "nbin_b", default=0)
    if n_lens == 0 or n_source == 0:
        raise ValueError(
            "apply_zstar_amplitudes: projected galaxy-shear spectra were not found"
        )

    for lens_bin in range(1, n_lens + 1):
        bhat = float(
            block[config["bhat_section"], f"bhat_bin_{lens_bin}"]
        )
        jhat = float(block[config["jhat_section"], f"Jhat{lens_bin}"])
        amplitude = bhat * jhat
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError(
                f"apply_zstar_amplitudes: invalid bhat*Jhat in lens bin {lens_bin}"
            )
        for source_bin in range(1, n_source + 1):
            name = f"bin_{lens_bin}_{source_bin}"
            if block.has_value(section, name):
                block[section, name] = np.asarray(block[section, name]) * amplitude
        block[
            config["derived_section"], f"bhat_jhat_bin_{lens_bin}"
        ] = amplitude
    return 0


def cleanup(config):
    pass
