from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.special import hyp2f1, logsumexp


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/paper_figures_20260811/figures"
WITH_PRIOR_PATH = ROOT / "chain/chain_reparameterized_with_planck_nautilus.txt"
NO_PRIOR_PATH = ROOT / "chain/chain_reparameterized_no_planck_nautilus.txt"

Z_EFF = np.array([0.38, 0.61])
BLUE = "#0072B2"
ORANGE = "#D55E00"
VERMILLION = "#C7254E"
SKY = "#56B4E9"
GREY = "#777777"
LIGHT_GREY = "#B6B6B6"
BLACK = "#171717"


@dataclass
class Chain:
    label: str
    names: list[str]
    data: np.ndarray
    index: dict[str, int]
    weight: np.ndarray

    def get(self, name: str) -> np.ndarray:
        return self.data[:, self.index[name]]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_chain(path: Path, label: str) -> Chain:
    with path.open(encoding="utf-8") as handle:
        names = handle.readline().lstrip("#").strip().split("\t")
    data = np.loadtxt(path, comments="#")
    index = {name: position for position, name in enumerate(names)}
    valid = np.isfinite(data[:, index["log_weight"]]) & np.isfinite(
        data[:, index["post"]]
    )
    data = data[valid]
    log_weight = data[:, index["log_weight"]]
    weight = np.exp(log_weight - logsumexp(log_weight))
    return Chain(label, names, data, index, weight)


def weighted_quantile(
    values: np.ndarray, weight: np.ndarray, probabilities: list[float] | np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    cumulative = np.cumsum(weight[order])
    return np.interp(np.asarray(probabilities), cumulative, values[order])


def posterior_point(values: np.ndarray, weight: np.ndarray) -> tuple[float, float, float]:
    mean = float(weight @ values)
    q16, q84 = weighted_quantile(values, weight, [0.16, 0.84])
    return mean, mean - float(q16), float(q84) - mean


def weighted_curve_band(
    curves: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.empty((3, curves.shape[1]))
    for column in range(curves.shape[1]):
        result[:, column] = weighted_quantile(
            curves[:, column], weight, [0.16, 0.50, 0.84]
        )
    return result[0], result[1], result[2]


def omega_m_z(omega_m0: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    return omega_m0 * (1.0 + redshift) ** 3 / (
        omega_m0 * (1.0 + redshift) ** 3 + 1.0 - omega_m0
    )


def growth_factor_flat_lcdm(omega_m0: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    scale_factor = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    numerator = scale_factor * hyp2f1(
        1.0 / 3.0,
        1.0,
        11.0 / 6.0,
        -ratio * scale_factor**3,
    )
    denominator = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -ratio)
    return numerator / denominator


def growth_rate_flat_lcdm(omega_m0: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    scale_factor = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    argument = -ratio * scale_factor**3
    first = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, argument)
    derivative = hyp2f1(4.0 / 3.0, 2.0, 17.0 / 6.0, argument)
    return 1.0 - (6.0 / 11.0) * ratio * scale_factor**3 * derivative / first


def jhat_gr_curves(
    omega_m0: np.ndarray, sigma8_0: np.ndarray, redshift: np.ndarray
) -> np.ndarray:
    om = omega_m0[:, None]
    s8 = sigma8_0[:, None]
    z = redshift[None, :]
    return omega_m_z(om, z) * s8 * growth_factor_flat_lcdm(om, z)


def eg_gr_curves(omega_m0: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    om = omega_m0[:, None]
    z = redshift[None, :]
    return om / growth_rate_flat_lcdm(om, z)


def chain_observables(chain: Chain) -> dict[str, np.ndarray]:
    return {
        "f1": chain.get("growth_amplitudes--fhat_bin_1"),
        "J1": chain.get("J_bin_bias--jhat1"),
        "EG1": chain.get("HATTED_DERIVED--EG_BIN_1"),
        "f2": chain.get("growth_amplitudes--fhat_bin_2"),
        "J2": chain.get("J_bin_bias--jhat2"),
        "EG2": chain.get("HATTED_DERIVED--EG_BIN_2"),
    }


def make_getdist_samples(chain: Chain) -> MCSamples:
    observable = chain_observables(chain)
    names = ["f1", "J1", "EG1", "f2", "J2", "EG2"]
    labels = [
        r"\widehat f_1",
        r"\widehat J_1",
        r"E_{G,1}",
        r"\widehat f_2",
        r"\widehat J_2",
        r"E_{G,2}",
    ]
    sample_matrix = np.column_stack([observable[name] for name in names])
    return MCSamples(
        samples=sample_matrix,
        weights=chain.weight,
        names=names,
        labels=labels,
        label=chain.label,
        settings={
            "ignore_rows": 0,
            "smooth_scale_1D": 0.35,
            "smooth_scale_2D": 0.45,
            "contours": [0.68, 0.95],
        },
    )


def save_figure(fig: plt.Figure, stem: str, dpi: int = 240) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=dpi)


def plot_corner(with_prior: Chain, no_prior: Chain) -> None:
    with_samples = make_getdist_samples(with_prior)
    no_samples = make_getdist_samples(no_prior)

    plotter = plots.get_subplot_plotter(width_inch=10.2)
    plotter.settings.axes_fontsize = 9.5
    plotter.settings.lab_fontsize = 13
    plotter.settings.legend_fontsize = 11
    plotter.settings.linewidth = 1.25
    plotter.settings.linewidth_contour = 1.2
    plotter.settings.alpha_filled_add = 0.55
    plotter.settings.figure_legend_frame = False
    plotter.settings.legend_loc = "upper right"
    plotter.settings.num_plot_contours = 2
    plotter.triangle_plot(
        [no_samples, with_samples],
        params=["f1", "J1", "EG1", "f2", "J2", "EG2"],
        filled=True,
        contour_colors=[ORANGE, BLUE],
        contour_ls=["--", "-"],
        contour_lws=[1.1, 1.35],
        legend_labels=["No Prior", "With Prior"],
        legend_loc="upper right",
    )

    for row in range(6):
        for column in range(row + 1):
            axis = plotter.subplots[row, column]
            if axis is None:
                continue
            axis.tick_params(direction="in", top=True, right=True)
    plotter.fig.subplots_adjust(wspace=0.06, hspace=0.06)
    save_figure(plotter.fig, "fig1_late_time_corner", dpi=210)
    plt.close(plotter.fig)


def draw_asymmetric_points(
    axis: plt.Axes,
    redshift: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    **kwargs,
) -> None:
    axis.errorbar(redshift, values, yerr=np.vstack([lower, upper]), **kwargs)


def chain_points(
    chain: Chain, stems: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, lower, upper = [], [], []
    for stem in stems:
        point = posterior_point(chain.get(stem), chain.weight)
        mean.append(point[0])
        lower.append(point[1])
        upper.append(point[2])
    return np.asarray(mean), np.asarray(lower), np.asarray(upper)


def planck_samples(size: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260811)
    omega_m = rng.normal(0.315, 0.007, size=size)
    sigma8 = rng.normal(0.811, 0.006, size=size)
    return omega_m, sigma8


def plot_jhat_evolution(with_prior: Chain, no_prior: Chain) -> None:
    z_plot = np.linspace(0.18, 0.84, 180)

    planck_om, planck_s8 = planck_samples()
    planck_curves = jhat_gr_curves(planck_om, planck_s8, z_plot)
    planck_lo, planck_mid, planck_hi = np.percentile(
        planck_curves, [16.0, 50.0, 84.0], axis=0
    )

    with_curves = jhat_gr_curves(
        with_prior.get("COSMOLOGICAL_PARAMETERS--OMEGA_M"),
        with_prior.get("COSMOLOGICAL_PARAMETERS--SIGMA_8"),
        z_plot,
    )
    with_lo, with_mid, with_hi = weighted_curve_band(with_curves, with_prior.weight)

    des_z = np.array([0.295, 0.467, 0.626, 0.771])
    des_with = np.array([0.325, 0.333, 0.387, 0.354])
    des_with_low = np.array([0.015, 0.019, 0.029, 0.035])
    des_with_high = np.array([0.015, 0.017, 0.026, 0.035])
    des_no = np.array([0.327, 0.328, 0.388, 0.345])
    des_no_low = np.array([0.017, 0.023, 0.032, 0.040])
    des_no_high = np.array([0.017, 0.019, 0.032, 0.040])

    our_with, our_with_low, our_with_high = chain_points(
        with_prior, ("J_bin_bias--jhat1", "J_bin_bias--jhat2")
    )
    our_no, our_no_low, our_no_high = chain_points(
        no_prior, ("J_bin_bias--jhat1", "J_bin_bias--jhat2")
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.25), sharex=True, sharey=True)
    configurations = [
        (
            axes[0],
            "With Prior",
            des_with,
            des_with_low,
            des_with_high,
            our_with,
            our_with_low,
            our_with_high,
        ),
        (
            axes[1],
            "No Prior",
            des_no,
            des_no_low,
            des_no_high,
            our_no,
            our_no_low,
            our_no_high,
        ),
    ]

    for axis, title, des, des_low, des_high, ours, ours_low, ours_high in configurations:
        axis.fill_between(
            z_plot, planck_lo, planck_hi, color="#777777", alpha=0.14, linewidth=0
        )
        axis.plot(
            z_plot,
            planck_mid,
            color=BLACK,
            linestyle="--",
            linewidth=1.7,
            label="Planck 2018 GR reference",
        )
        if title == "With Prior":
            axis.fill_between(
                z_plot, with_lo, with_hi, color=BLUE, alpha=0.17, linewidth=0
            )
            axis.plot(
                z_plot,
                with_mid,
                color=BLUE,
                linewidth=1.8,
                label="GR from With-Prior posterior",
            )
        draw_asymmetric_points(
            axis,
            des_z,
            des,
            des_low,
            des_high,
            fmt="o",
            color=GREY,
            markerfacecolor="white",
            markeredgecolor=GREY,
            markeredgewidth=1.1,
            markersize=5.5,
            capsize=2.5,
            elinewidth=1.0,
            zorder=3,
            label="DES Y3 (Tutusaus et al. 2024)",
        )
        draw_asymmetric_points(
            axis,
            Z_EFF,
            ours,
            ours_low,
            ours_high,
            fmt="*",
            color=VERMILLION,
            markeredgecolor=VERMILLION,
            markersize=11,
            capsize=3.0,
            elinewidth=1.7,
            markeredgewidth=1.2,
            zorder=5,
            label="This work",
        )
        axis.set_title(title, fontsize=12.5, pad=7)
        axis.set_xlabel(r"Redshift $z$")
        axis.set_xlim(0.18, 0.84)
        axis.set_ylim(0.245, 0.485)
        axis.xaxis.set_minor_locator(MultipleLocator(0.05))
        axis.yaxis.set_minor_locator(MultipleLocator(0.01))

    axes[0].set_ylabel(r"Weyl evolution $\widehat J(z)$")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", frameon=False, fontsize=8.2)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="lower right", frameon=False, fontsize=8.2)
    fig.subplots_adjust(wspace=0.06)
    save_figure(fig, "fig2_jhat_evolution")
    plt.close(fig)


def plot_eg_compilation(with_prior: Chain, no_prior: Chain) -> None:
    z_plot = np.linspace(0.18, 0.84, 180)
    planck_om, _ = planck_samples()
    planck_curves = eg_gr_curves(planck_om, z_plot)
    gr_lo, gr_mid, gr_hi = np.percentile(
        planck_curves, [16.0, 50.0, 84.0], axis=0
    )

    galaxy_z = np.array([0.37, 0.32, 0.57, 0.27, 0.31, 0.55, 0.57])
    galaxy_eg = np.array([0.392, 0.48, 0.30, 0.43, 0.27, 0.26, 0.420])
    galaxy_low = np.array([0.065, 0.10, 0.07, 0.13, 0.08, 0.07, 0.056])
    galaxy_high = galaxy_low.copy()

    cmb_z = np.array([0.57, 0.316, 0.555])
    cmb_eg = np.array([0.24, 0.40, 0.36])
    cmb_low = np.array([0.06, 0.09, 0.05])
    cmb_high = np.array([0.06, 0.11, 0.06])

    grimm_z = np.array([0.295, 0.467, 0.626, 0.771])
    grimm_eg = np.array([0.447, 0.378, 0.396, 0.345])
    grimm_err = np.array([0.027, 0.026, 0.033, 0.039])

    our_with, our_with_low, our_with_high = chain_points(
        with_prior, ("HATTED_DERIVED--EG_BIN_1", "HATTED_DERIVED--EG_BIN_2")
    )
    our_no, our_no_low, our_no_high = chain_points(
        no_prior, ("HATTED_DERIVED--EG_BIN_1", "HATTED_DERIVED--EG_BIN_2")
    )

    fig, axis = plt.subplots(figsize=(8.0, 5.35))
    axis.fill_between(z_plot, gr_lo, gr_hi, color="#777777", alpha=0.16, linewidth=0)
    axis.plot(
        z_plot,
        gr_mid,
        color=BLACK,
        linestyle="--",
        linewidth=1.8,
        label="Planck 2018 GR prediction",
        zorder=2,
    )

    draw_asymmetric_points(
        axis,
        galaxy_z,
        galaxy_eg,
        galaxy_low,
        galaxy_high,
        fmt="o",
        color=LIGHT_GREY,
        markerfacecolor="white",
        markeredgecolor="#888888",
        markersize=4.5,
        capsize=2.0,
        elinewidth=0.9,
        zorder=1,
        label="Earlier galaxy-lensing measurements",
    )
    draw_asymmetric_points(
        axis,
        cmb_z,
        cmb_eg,
        cmb_low,
        cmb_high,
        fmt="s",
        color="#888888",
        markerfacecolor="white",
        markeredgecolor="#666666",
        markersize=4.8,
        capsize=2.0,
        elinewidth=0.9,
        zorder=1,
        label="Earlier CMB-lensing measurements",
    )
    axis.errorbar(
        grimm_z,
        grimm_eg,
        yerr=grimm_err,
        fmt="D",
        color=BLUE,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markersize=5.2,
        capsize=2.5,
        elinewidth=1.1,
        zorder=3,
        label="DES Y3 (Grimm et al. 2024)",
    )
    draw_asymmetric_points(
        axis,
        Z_EFF - 0.008,
        our_with,
        our_with_low,
        our_with_high,
        fmt="o",
        color=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=7.2,
        capsize=3.0,
        elinewidth=1.6,
        zorder=5,
        label="This work: With Prior",
    )
    draw_asymmetric_points(
        axis,
        Z_EFF + 0.008,
        our_no,
        our_no_low,
        our_no_high,
        fmt="*",
        color=ORANGE,
        markeredgecolor=ORANGE,
        markersize=10.5,
        capsize=3.0,
        elinewidth=1.6,
        zorder=6,
        label="This work: No Prior",
    )

    axis.set_xlabel(r"Redshift $z$")
    axis.set_ylabel(r"Gravity statistic $E_G(z)$")
    axis.set_xlim(0.18, 0.84)
    axis.set_ylim(0.15, 0.63)
    axis.xaxis.set_minor_locator(MultipleLocator(0.05))
    axis.yaxis.set_minor_locator(MultipleLocator(0.025))
    axis.legend(loc="upper right", frameon=False, fontsize=8.3, ncol=1)
    save_figure(fig, "fig3_eg_compilation")
    plt.close(fig)


def write_summary(with_prior: Chain, no_prior: Chain) -> None:
    rows: list[dict[str, object]] = []
    for chain in (with_prior, no_prior):
        for quantity, columns in {
            "Jhat": ("J_bin_bias--jhat1", "J_bin_bias--jhat2"),
            "fhat": (
                "growth_amplitudes--fhat_bin_1",
                "growth_amplitudes--fhat_bin_2",
            ),
            "EG": ("HATTED_DERIVED--EG_BIN_1", "HATTED_DERIVED--EG_BIN_2"),
        }.items():
            for bin_index, (redshift, column) in enumerate(zip(Z_EFF, columns), start=1):
                mean, lower, upper = posterior_point(chain.get(column), chain.weight)
                rows.append(
                    {
                        "configuration": chain.label,
                        "quantity": quantity,
                        "bin": bin_index,
                        "redshift": redshift,
                        "mean": mean,
                        "error_low_68": lower,
                        "error_high_68": upper,
                    }
                )
    with (OUT / "plotted_chain_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    with_prior = load_chain(WITH_PRIOR_PATH, "With Prior")
    no_prior = load_chain(NO_PRIOR_PATH, "No Prior")
    plot_corner(with_prior, no_prior)
    plot_jhat_evolution(with_prior, no_prior)
    plot_eg_compilation(with_prior, no_prior)
    write_summary(with_prior, no_prior)
    print(f"Wrote paper-figure previews to {OUT}")


if __name__ == "__main__":
    main()
