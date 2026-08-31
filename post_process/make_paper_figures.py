from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from matplotlib.ticker import MultipleLocator
from scipy.special import hyp2f1, logsumexp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "figures"
WITH_PATH = ROOT / "chain/chain_explicit_zstar_with_planck_nautilus.txt"
NO_PATH = ROOT / "chain/chain_explicit_zstar_no_planck_nautilus.txt"

Z_EFF = np.array([0.38, 0.61])
BLUE = "#006eb8"
CRIMSON = "crimson"
GREY = "gray"


@dataclass
class Chain:
    label: str
    names: list[str]
    data: np.ndarray
    index: dict[str, int]
    weight: np.ndarray

    def get(self, name: str) -> np.ndarray:
        return self.data[:, self.index[name]]


def load_chain(path: Path, label: str) -> Chain:
    with path.open(encoding="utf-8") as handle:
        names = handle.readline().lstrip("#").strip().split("\t")
    data = np.loadtxt(path, comments="#")
    index = {name: i for i, name in enumerate(names)}
    valid = np.isfinite(data[:, index["log_weight"]]) & np.isfinite(
        data[:, index["post"]]
    )
    data = data[valid]
    log_weight = data[:, index["log_weight"]]
    weight = np.exp(log_weight - logsumexp(log_weight))
    return Chain(label, names, data, index, weight)


def weighted_quantile(values: np.ndarray, weight: np.ndarray, q) -> np.ndarray:
    order = np.argsort(values)
    cumulative = np.cumsum(weight[order])
    return np.interp(np.asarray(q), cumulative, values[order])


def point(chain: Chain, name: str) -> tuple[float, float, float]:
    values = chain.get(name)
    mean = float(chain.weight @ values)
    low, high = weighted_quantile(values, chain.weight, [0.16, 0.84])
    return mean, mean - float(low), float(high) - mean


def two_points(chain: Chain, names: tuple[str, str]):
    values, low, high = zip(*(point(chain, name) for name in names))
    return np.asarray(values), np.asarray(low), np.asarray(high)


def omega_m_z(omega_m0, redshift):
    return omega_m0 * (1.0 + redshift) ** 3 / (
        omega_m0 * (1.0 + redshift) ** 3 + 1.0 - omega_m0
    )


def growth_factor(omega_m0, redshift):
    a = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    numerator = a * hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -ratio * a**3)
    denominator = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -ratio)
    return numerator / denominator


def growth_rate(omega_m0, redshift):
    a = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    argument = -ratio * a**3
    first = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, argument)
    derivative = hyp2f1(4.0 / 3.0, 2.0, 17.0 / 6.0, argument)
    return 1.0 - (6.0 / 11.0) * ratio * a**3 * derivative / first


def curve_band(curves: np.ndarray, weight: np.ndarray):
    band = np.empty((3, curves.shape[1]))
    for i in range(curves.shape[1]):
        band[:, i] = weighted_quantile(curves[:, i], weight, [0.16, 0.50, 0.84])
    return band


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )


def getdist_samples(chain: Chain) -> MCSamples:
    columns = [
        "growth_amplitudes--fhat_bin_1",
        "J_bin_bias--jhat1",
        "HATTED_DERIVED--EG_BIN_1",
        "growth_amplitudes--fhat_bin_2",
        "J_bin_bias--jhat2",
        "HATTED_DERIVED--EG_BIN_2",
    ]
    names = ["f1", "J1", "EG1", "f2", "J2", "EG2"]
    labels = [
        r"\hat{f}_1",
        r"\hat{J}_1",
        r"E_{G,1}",
        r"\hat{f}_2",
        r"\hat{J}_2",
        r"E_{G,2}",
    ]
    return MCSamples(
        samples=np.column_stack([chain.get(name) for name in columns]),
        weights=chain.weight,
        names=names,
        labels=labels,
        label=chain.label,
        settings={"ignore_rows": 0, "contours": [0.68, 0.95]},
    )


def plot_corner(with_prior: Chain, no_prior: Chain) -> None:
    no_samples = getdist_samples(no_prior)
    with_samples = getdist_samples(with_prior)
    parameters = ["f1", "J1", "EG1", "f2", "J2", "EG2"]
    plotter = plots.get_subplot_plotter(width_inch=10.0)
    plotter.settings.axes_fontsize = 14
    plotter.settings.lab_fontsize = 16
    plotter.settings.legend_fontsize = 16
    plotter.settings.alpha_filled_add = 0.4
    plotter.settings.figure_legend_frame = False
    plotter.triangle_plot(
        [no_samples, with_samples],
        params=parameters,
        filled=True,
        contour_colors=[GREY, BLUE],
        contour_ls=["--", "-"],
        legend_labels=["No Prior", "With Prior"],
    )

    for diagonal, parameter in enumerate(parameters):
        axis = plotter.subplots[diagonal, diagonal]
        no_value = no_samples.getInlineLatex(parameter, limit=1)
        with_value = with_samples.getInlineLatex(parameter, limit=1)
        text_x = 0.98 if diagonal == len(parameters) - 1 else 0.5
        text_alignment = "right" if diagonal == len(parameters) - 1 else "center"
        axis.text(
            text_x,
            1.12,
            f"${no_value}$",
            color=GREY,
            transform=axis.transAxes,
            ha=text_alignment,
            va="bottom",
            fontsize=11,
        )
        axis.text(
            text_x,
            1.02,
            f"${with_value}$",
            color=BLUE,
            transform=axis.transAxes,
            ha=text_alignment,
            va="bottom",
            fontsize=11,
        )
    plotter.fig.subplots_adjust(wspace=0.08, hspace=0.08)

    old_height = plotter.fig.get_figheight()
    plotter.fig.set_figheight(old_height + 0.45)
    height_scale = old_height / plotter.fig.get_figheight()
    for axis in plotter.fig.axes:
        position = axis.get_position()
        axis.set_position(
            [
                position.x0,
                position.y0 * height_scale,
                position.width,
                position.height * height_scale,
            ]
        )
    plotter.fig.savefig(OUT / "comparison_noprior_vs_withprior.pdf")
    plt.close(plotter.fig)


def plot_jhat(chain: Chain, with_prior: bool) -> None:
    z = np.linspace(0.15, 0.85, 180)
    ours, ours_low, ours_high = two_points(
        chain, ("J_bin_bias--jhat1", "J_bin_bias--jhat2")
    )

    if with_prior:
        des = np.array([0.325, 0.333, 0.387, 0.354])
        des_err = np.array([[0.015, 0.019, 0.029, 0.035], [0.015, 0.017, 0.026, 0.035]])
        om = chain.get("COSMOLOGICAL_PARAMETERS--OMEGA_M")[:, None]
        s8 = chain.get("COSMOLOGICAL_PARAMETERS--SIGMA_8")[:, None]
        zz = z[None, :]
        curves = omega_m_z(om, zz) * s8 * growth_factor(om, zz)
        low, middle, high = curve_band(curves, chain.weight)
        curve_label = r"$\Lambda$CDM, With-Prior posterior"
    else:
        des = np.array([0.327, 0.328, 0.388, 0.345])
        des_err = np.array([[0.017, 0.023, 0.032, 0.040], [0.017, 0.019, 0.032, 0.040]])
        rng = np.random.default_rng(20260824)
        om = rng.normal(0.315, 0.007, 16000)[:, None]
        s8 = rng.normal(0.811, 0.006, 16000)[:, None]
        zz = z[None, :]
        curves = omega_m_z(om, zz) * s8 * growth_factor(om, zz)
        low, middle, high = np.percentile(curves, [16, 50, 84], axis=0)
        curve_label = r"External Planck 2018 $\Lambda$CDM reference"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(z, middle, color="royalblue" if with_prior else "black", linestyle="-." if with_prior else "--", linewidth=2, label=curve_label)
    ax.fill_between(z, low, high, color="royalblue" if with_prior else "black", alpha=0.15 if with_prior else 0.10, edgecolor="none")
    ax.errorbar(
        [0.295, 0.467, 0.626, 0.771], des, yerr=des_err, fmt="o",
        color=GREY, alpha=0.7, markersize=8, capsize=4, elinewidth=1.5,
        markeredgewidth=1.5, label="DES Y3 (Tutusaus et al. 2024)",
    )
    ax.errorbar(
        Z_EFF, ours, yerr=np.vstack([ours_low, ours_high]), fmt="*",
        color=CRIMSON, markersize=15, capsize=4, elinewidth=2,
        markeredgewidth=2, zorder=5, label="KiDS-1000+BOSS+2dFLenS",
    )
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Weyl Potential Evolution $\hat{J}(z)$")
    ax.set_xlim(0.2, 0.85)
    ax.set_ylim(0.24, 0.49)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.legend(loc="best", frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    suffix = "withprior" if with_prior else "noprior"
    fig.savefig(OUT / f"J_hat_evolution_comparison_{suffix}.pdf")
    plt.close(fig)


def eg_gr(omega_m0, redshift):
    return omega_m0 / growth_rate(omega_m0, redshift)


def plot_eg(chain: Chain, with_prior: bool) -> None:
    z = np.linspace(0.15, 0.85, 180)
    ours, ours_low, ours_high = two_points(
        chain, ("HATTED_DERIVED--EG_BIN_1", "HATTED_DERIVED--EG_BIN_2")
    )
    om = chain.get("COSMOLOGICAL_PARAMETERS--OMEGA_M")[:, None]
    curves = eg_gr(om, z[None, :])
    low, middle, high = curve_band(curves, chain.weight)

    z_lit = np.array([0.37, 0.32, 0.57, 0.27, 0.31, 0.55, 0.57, 0.57, 0.316, 0.555])
    eg_lit = np.array([0.392, 0.48, 0.30, 0.43, 0.27, 0.26, 0.420, 0.24, 0.40, 0.36])
    err_low = np.array([0.065, 0.10, 0.07, 0.13, 0.08, 0.07, 0.056, 0.06, 0.09, 0.05])
    err_high = np.array([0.065, 0.10, 0.07, 0.13, 0.08, 0.07, 0.056, 0.06, 0.11, 0.06])
    z_des = np.array([0.295, 0.467, 0.626, 0.771])
    eg_des = np.array([0.447, 0.378, 0.396, 0.345])
    err_des = np.array([0.027, 0.026, 0.033, 0.039])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        z_lit, eg_lit, yerr=np.vstack([err_low, err_high]), fmt="s",
        color="silver", alpha=0.8, markersize=6, capsize=3, elinewidth=1.2,
        markeredgewidth=1.2, markerfacecolor="white", zorder=1, label="Earlier measurements",
    )
    ax.errorbar(
        z_des, eg_des, yerr=err_des, fmt="D", color="steelblue", alpha=0.85,
        markersize=7, capsize=4, elinewidth=1.5, markeredgewidth=1.5,
        zorder=2, label="DES Y3 (Grimm et al. 2024)",
    )
    ax.fill_between(z, low, high, color="black", alpha=0.10, edgecolor="none")
    ax.plot(z, middle, color="black", linestyle="--", linewidth=2, zorder=3, label="GR prediction")
    ax.errorbar(
        Z_EFF, ours, yerr=np.vstack([ours_low, ours_high]), fmt="*",
        color=CRIMSON, markersize=16, capsize=5, elinewidth=2.5,
        markeredgewidth=2, zorder=5, label="KiDS-1000+BOSS+2dFLenS",
    )
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Gravitational Slip Statistic $E_G(z)$")
    ax.set_xlim(0.18, 0.82)
    ax.set_ylim(0.15, 0.65)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.legend(loc="best", frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    suffix = "withprior" if with_prior else "noprior"
    fig.savefig(OUT / f"EG_compilation_{suffix}.pdf")
    plt.close(fig)


def plot_gravity_models(chain: Chain) -> None:
    z_obs = Z_EFF
    j_samples = np.column_stack(
        [chain.get("J_bin_bias--jhat1"), chain.get("J_bin_bias--jhat2")]
    )
    j_obs = np.average(j_samples, axis=0, weights=chain.weight)
    covariance = np.cov(j_samples.T, aweights=chain.weight)
    inverse_covariance = np.linalg.inv(covariance)

    omega_m0 = float(
        chain.weight @ chain.get("COSMOLOGICAL_PARAMETERS--OMEGA_M")
    )
    sigma8_0 = float(
        chain.weight @ chain.get("COSMOLOGICAL_PARAMETERS--SIGMA_8")
    )

    def j_gr(redshift):
        redshift = np.asarray(redshift)
        return (
            omega_m_z(omega_m0, redshift)
            * sigma8_0
            * growth_factor(omega_m0, redshift)
        )

    def omega_lambda_z(redshift):
        return (1.0 - omega_m0) / (
            omega_m0 * (1.0 + redshift) ** 3 + 1.0 - omega_m0
        )

    model_functions = {
        "Standard": omega_lambda_z,
        "Constant": lambda redshift: np.ones_like(np.asarray(redshift)),
        "Exponential": lambda redshift: np.exp(1.0 + np.asarray(redshift)),
    }
    j_reference = j_gr(z_obs)
    residual = j_obs - j_reference
    fits = {}
    for name, function in model_functions.items():
        design = function(z_obs) * j_reference
        fisher = float(design @ inverse_covariance @ design)
        best = float(design @ inverse_covariance @ residual / fisher)
        error = fisher**-0.5
        remaining = residual - best * design
        chi2 = float(remaining @ inverse_covariance @ remaining)
        fits[name] = (best, error, chi2)

    baseline_chi2 = float(residual @ inverse_covariance @ residual)
    print(f"Conditional GR chi2: {baseline_chi2:.3f}")
    for name, (best, error, chi2) in fits.items():
        print(
            f"{name}: Sigma0={best:.4f} +/- {error:.4f}; "
            f"significance={abs(best) / error:.2f}; chi2={chi2:.3f}"
        )

    z_plot = np.linspace(0.15, 0.85, 200)
    j_base = j_gr(z_plot)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        z_plot,
        j_base,
        color="black",
        linewidth=2.5,
        zorder=5,
        label=r"GR ($\Lambda$CDM, $\Sigma_0=0$)",
    )
    colors = ["#0072B2", "#D55E00", "#CC79A7"]
    linestyles = ["--", "-.", ":"]
    labels = [
        r"Standard: $g(z)=\Omega_\Lambda(z)$",
        r"Constant: $g(z)=1$",
        r"Exponential: $g(z)=e^{1+z}$",
    ]
    for color, linestyle, label, (name, function) in zip(
        colors, linestyles, labels, model_functions.items()
    ):
        best, error, _ = fits[name]
        response = function(z_plot)
        central = (1.0 + best * response) * j_base
        lower = (1.0 + (best - error) * response) * j_base
        upper = (1.0 + (best + error) * response) * j_base
        ax.plot(
            z_plot,
            central,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            zorder=4,
            label=label,
        )
        ax.fill_between(
            z_plot, lower, upper, color=color, alpha=0.11, edgecolor="none"
        )

    ax.errorbar(
        z_obs,
        j_obs,
        yerr=np.sqrt(np.diag(covariance)),
        fmt="s",
        color="#2F3E63",
        markersize=8,
        capsize=5,
        elinewidth=2.5,
        markeredgewidth=2,
        zorder=6,
        label="KiDS-1000+BOSS+2dFLenS",
    )
    ax.set_xlabel(r"Redshift $z$", fontsize=16)
    ax.set_ylabel(r"Weyl Potential Evolution $\hat{J}(z)$", fontsize=16)
    ax.set_xlim(0.18, 0.8)
    ax.set_ylim(0.23, 0.44)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.legend(loc="lower left", frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    fig.savefig(OUT / "J_hat_evolution_analytical_errors_withprior.pdf")
    plt.close(fig)


def main() -> None:
    set_style()
    with_prior = load_chain(WITH_PATH, "With Prior")
    no_prior = load_chain(NO_PATH, "No Prior")
    plot_corner(with_prior, no_prior)
    plot_jhat(with_prior, True)
    plot_jhat(no_prior, False)
    plot_eg(with_prior, True)
    plot_eg(no_prior, False)
    plot_gravity_models(with_prior)
    print("Updated six paper figures in", OUT)


if __name__ == "__main__":
    main()
