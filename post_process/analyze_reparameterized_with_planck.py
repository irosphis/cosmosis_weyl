from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.special import hyp2f1, logsumexp
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/reparameterized_with_planck_20260808"
NEW_CHAIN = ROOT / "chain/chain_reparameterized_with_planck_nautilus.txt"
OLD_CHAIN = ROOT / "MyTest/chains/chain_withprior.txt"
BESTFIT_DIR = ROOT / "kcap/output/analysis_reparameterized_with_planck_maxlike_20260809"


@dataclass
class Chain:
    names: list[str]
    data: np.ndarray
    index: dict[str, int]
    weight: np.ndarray

    def get(self, name: str) -> np.ndarray:
        return self.data[:, self.index[name]]


def load_chain(path: Path) -> Chain:
    with path.open(encoding="utf-8") as handle:
        names = handle.readline().lstrip("#").strip().split("\t")
    data = np.loadtxt(path, comments="#")
    index = {name: i for i, name in enumerate(names)}
    log_weight = data[:, index["log_weight"]]
    weight = np.exp(log_weight - logsumexp(log_weight))
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite chain entry in {path}")
    if not np.isclose(weight.sum(), 1.0):
        raise ValueError(f"Weights do not normalize in {path}")
    return Chain(names, data, index, weight)


def weighted_quantile(
    values: np.ndarray, weight: np.ndarray, probabilities: np.ndarray | list[float]
) -> np.ndarray:
    order = np.argsort(values)
    cumulative = np.cumsum(weight[order])
    return np.interp(np.asarray(probabilities), cumulative, values[order])


def weighted_stats(values: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    mean = float(weight @ values)
    sd = float(np.sqrt(weight @ ((values - mean) ** 2)))
    q = weighted_quantile(values, weight, [0.005, 0.025, 0.16, 0.5, 0.84, 0.975, 0.995])
    return {
        "mean": mean,
        "sd": sd,
        "q005": float(q[0]),
        "q025": float(q[1]),
        "q16": float(q[2]),
        "median": float(q[3]),
        "q84": float(q[4]),
        "q975": float(q[5]),
        "q995": float(q[6]),
    }


def weighted_covariance(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mean = weight @ matrix
    centered = matrix - mean
    return centered.T @ (weight[:, None] * centered)


def linear_growth_rate_flat_lcdm(omega_m0: np.ndarray, redshift: float) -> np.ndarray:
    scale_factor = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    argument = -ratio * scale_factor**3
    first = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, argument)
    derivative = hyp2f1(4.0 / 3.0, 2.0, 17.0 / 6.0, argument)
    return 1.0 - (6.0 / 11.0) * ratio * scale_factor**3 * derivative / first


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contour_levels(histogram: np.ndarray, probabilities=(0.95, 0.68)) -> list[float]:
    flat = np.sort(histogram.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    thresholds = []
    for probability in probabilities:
        thresholds.append(float(flat[min(np.searchsorted(cumulative, probability), len(flat) - 1)]))
    return sorted(thresholds)


def draw_weighted_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    color: str,
) -> None:
    xlim = weighted_quantile(x, weight, [0.001, 0.999])
    ylim = weighted_quantile(y, weight, [0.001, 0.999])
    hist, xedge, yedge = np.histogram2d(
        x, y, bins=65, range=[xlim, ylim], weights=weight
    )
    hist = gaussian_filter(hist, sigma=1.1)
    xcenter = 0.5 * (xedge[:-1] + xedge[1:])
    ycenter = 0.5 * (yedge[:-1] + yedge[1:])
    levels = contour_levels(hist)
    ax.contour(xcenter, ycenter, hist.T, levels=levels, colors=color, linewidths=[1.0, 1.7])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_corner(
    arrays: list[np.ndarray],
    labels: list[str],
    weight: np.ndarray,
    output: Path,
    title: str,
) -> None:
    count = len(arrays)
    covariance = weighted_covariance(np.column_stack(arrays), weight)
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigma, sigma)
    limits = [weighted_quantile(a, weight, [0.001, 0.999]) for a in arrays]
    fig, axes = plt.subplots(count, count, figsize=(9.2, 9.2))
    for row in range(count):
        for col in range(count):
            ax = axes[row, col]
            if row == col:
                ax.hist(
                    arrays[col], bins=60, range=limits[col], weights=weight,
                    density=True, histtype="stepfilled", color="#377eb8", alpha=0.6,
                )
                q = weighted_quantile(arrays[col], weight, [0.16, 0.5, 0.84])
                ax.axvline(q[1], color="#0b3c5d", lw=1.4)
                ax.axvspan(q[0], q[2], color="#377eb8", alpha=0.12)
                ax.set_xlim(limits[col])
                ax.set_yticks([])
            elif row > col:
                draw_weighted_contours(ax, arrays[col], arrays[row], weight, "#377eb8")
            else:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, rf"$\rho={correlation[row, col]:+.2f}$",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12,
                )
            if row == count - 1 and col <= row:
                ax.set_xlabel(labels[col], fontsize=11)
            elif row != col:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=11)
            elif row > col:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    new = load_chain(NEW_CHAIN)
    old = load_chain(OLD_CHAIN)

    key = {
        "bhat1": new.get("growth_amplitudes--bhat_bin_1"),
        "fhat1": new.get("growth_amplitudes--fhat_bin_1"),
        "bhat2": new.get("growth_amplitudes--bhat_bin_2"),
        "fhat2": new.get("growth_amplitudes--fhat_bin_2"),
        "Jhat1": new.get("J_bin_bias--jhat1"),
        "Jhat2": new.get("J_bin_bias--jhat2"),
        "EG1": new.get("HATTED_DERIVED--EG_BIN_1"),
        "EG2": new.get("HATTED_DERIVED--EG_BIN_2"),
        "b1eff1": new.get("HATTED_DERIVED--B1_EFF_BIN_1"),
        "b1eff2": new.get("HATTED_DERIVED--B1_EFF_BIN_2"),
        "feff1": new.get("HATTED_DERIVED--F_EFF_BIN_1"),
        "feff2": new.get("HATTED_DERIVED--F_EFF_BIN_2"),
        "Omega_m": new.get("COSMOLOGICAL_PARAMETERS--OMEGA_M"),
        "S8": new.get("COSMOLOGICAL_PARAMETERS--S_8"),
        "sigma8": new.get("COSMOLOGICAL_PARAMETERS--SIGMA_8"),
    }

    summary_rows: list[dict[str, object]] = []
    for name, values in key.items():
        summary_rows.append({"quantity": name, **weighted_stats(values, new.weight)})
    write_csv(OUT / "posterior_summary.csv", summary_rows)

    omega_m0 = key["Omega_m"]
    gr: dict[str, np.ndarray] = {}
    gr_rows: list[dict[str, object]] = []
    for bin_index, redshift in ((1, 0.38), (2, 0.61)):
        sigma8_z = new.get(f"HATTED_DERIVED--SIGMA8_REF_BIN_{bin_index}")
        expansion_squared = omega_m0 * (1.0 + redshift) ** 3 + 1.0 - omega_m0
        omega_m_z = omega_m0 * (1.0 + redshift) ** 3 / expansion_squared
        gr[f"Jhat{bin_index}"] = omega_m_z * sigma8_z
        gr[f"fhat{bin_index}"] = (
            linear_growth_rate_flat_lcdm(omega_m0, redshift) * sigma8_z
        )
        exact_geometry_factor = (
            key[f"EG{bin_index}"]
            * key[f"fhat{bin_index}"]
            / key[f"Jhat{bin_index}"]
        )
        gr[f"EG{bin_index}"] = (
            exact_geometry_factor
            * gr[f"Jhat{bin_index}"]
            / gr[f"fhat{bin_index}"]
        )
        for quantity in ("Jhat", "fhat", "EG"):
            measured = key[f"{quantity}{bin_index}"]
            prediction = gr[f"{quantity}{bin_index}"]
            difference = measured - prediction
            measured_stats = weighted_stats(measured, new.weight)
            prediction_stats = weighted_stats(prediction, new.weight)
            difference_stats = weighted_stats(difference, new.weight)
            gr_rows.append(
                {
                    "bin": bin_index,
                    "redshift": redshift,
                    "quantity": quantity,
                    "measured_mean": measured_stats["mean"],
                    "measured_sd": measured_stats["sd"],
                    "measured_q16": measured_stats["q16"],
                    "measured_q84": measured_stats["q84"],
                    "gr_mean": prediction_stats["mean"],
                    "gr_sd": prediction_stats["sd"],
                    "difference_mean": difference_stats["mean"],
                    "difference_sd": difference_stats["sd"],
                    "tension_sigma": difference_stats["mean"] / difference_stats["sd"],
                    "probability_below_gr": float(new.weight[difference < 0].sum()),
                }
            )
    write_csv(OUT / "gr_consistency.csv", gr_rows)

    old_values = {
        "Omega_m": old.get("COSMOLOGICAL_PARAMETERS--OMEGA_M"),
        "S8": old.get("COSMOLOGICAL_PARAMETERS--S_8"),
        "Jhat1": old.get("J_bin_bias--jhat1"),
        "Jhat2": old.get("J_bin_bias--jhat2"),
        "fhat1": old.get("LSS_PARAMETERS--FSIGMA_8_BIN_1"),
        "fhat2": old.get("LSS_PARAMETERS--FSIGMA_8_BIN_2"),
    }
    for bin_index, redshift in ((1, 0.38), (2, 0.61)):
        factor = old_values["Omega_m"] + (1.0 - old_values["Omega_m"]) / (1.0 + redshift) ** 3
        old_values[f"EG{bin_index}"] = (
            factor * old_values[f"Jhat{bin_index}"] / old_values[f"fhat{bin_index}"]
        )

    comparison_rows: list[dict[str, object]] = []
    for quantity in ("Omega_m", "S8", "Jhat1", "Jhat2", "fhat1", "fhat2", "EG1", "EG2"):
        new_stats = weighted_stats(key[quantity], new.weight)
        old_stats = weighted_stats(old_values[quantity], old.weight)
        comparison_rows.append(
            {
                "quantity": quantity,
                "old_mean": old_stats["mean"],
                "old_sd": old_stats["sd"],
                "old_q16": old_stats["q16"],
                "old_q84": old_stats["q84"],
                "new_mean": new_stats["mean"],
                "new_sd": new_stats["sd"],
                "new_q16": new_stats["q16"],
                "new_q84": new_stats["q84"],
                "mean_shift_over_independent_sigma": (
                    new_stats["mean"] - old_stats["mean"]
                ) / np.hypot(new_stats["sd"], old_stats["sd"]),
            }
        )
    write_csv(OUT / "old_new_comparison.csv", comparison_rows)

    correlation_names = ["bhat1", "fhat1", "Jhat1", "EG1", "bhat2", "fhat2", "Jhat2", "EG2"]
    correlation_matrix = weighted_covariance(
        np.column_stack([key[name] for name in correlation_names]), new.weight
    )
    correlation_sigma = np.sqrt(np.diag(correlation_matrix))
    correlation_matrix /= np.outer(correlation_sigma, correlation_sigma)
    with (OUT / "key_correlation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", *correlation_names])
        for name, row in zip(correlation_names, correlation_matrix):
            writer.writerow([name, *row])

    bounds = {
        "halo_model_parameters--a": (2.0, 3.13),
        "intrinsic_alignment_parameters--a": (-6.0, 6.0),
        "growth_amplitudes--bhat_bin_1": (0.1, 5.0),
        "growth_amplitudes--fhat_bin_1": (0.05, 1.2),
        "growth_amplitudes--bhat_bin_2": (0.1, 5.0),
        "growth_amplitudes--fhat_bin_2": (0.05, 1.2),
        "bias_parameters--b2_bin_1": (-4.0, 8.0),
        "bias_parameters--gamma3_bin_1": (-8.0, 8.0),
        "bias_parameters--a_vir_bin_1": (0.0, 12.0),
        "bias_parameters--b2_bin_2": (-4.0, 8.0),
        "bias_parameters--gamma3_bin_2": (-8.0, 8.0),
        "bias_parameters--a_vir_bin_2": (0.0, 12.0),
        "J_bin_bias--jhat1": (0.2, 0.5),
        "J_bin_bias--jhat2": (0.2, 0.5),
    }
    boundary_rows: list[dict[str, object]] = []
    for name, (lower, upper) in bounds.items():
        values = new.get(name)
        span = upper - lower
        stats = weighted_stats(values, new.weight)
        boundary_rows.append(
            {
                "quantity": name,
                "prior_lower": lower,
                "prior_upper": upper,
                "q005": stats["q005"],
                "q025": stats["q025"],
                "median": stats["median"],
                "q975": stats["q975"],
                "q995": stats["q995"],
                "mass_within_lower_2pct": float(new.weight[values < lower + 0.02 * span].sum()),
                "mass_within_upper_2pct": float(new.weight[values > upper - 0.02 * span].sum()),
            }
        )
    write_csv(OUT / "prior_boundary_check.csv", boundary_rows)

    fit_specs = [
        (
            "BOSS LOWZ", BESTFIT_DIR / "data_vector/xi_wedges_like_data_bin_1.txt",
            BESTFIT_DIR / "data_vector/xi_wedges_like_theory_bin_1.txt",
            BESTFIT_DIR / "data_vector/xi_wedges_like_inv_cov_bin_1.txt",
        ),
        (
            "BOSS CMASS", BESTFIT_DIR / "data_vector/xi_wedges_like_data_bin_2.txt",
            BESTFIT_DIR / "data_vector/xi_wedges_like_theory_bin_2.txt",
            BESTFIT_DIR / "data_vector/xi_wedges_like_inv_cov_bin_2.txt",
        ),
        (
            "GGL", BESTFIT_DIR / "theory_data_covariance/data.txt",
            BESTFIT_DIR / "theory_data_covariance/theory.txt",
            BESTFIT_DIR / "theory_data_covariance/inv_covariance.txt",
        ),
    ]
    residual_sets: list[tuple[str, np.ndarray]] = []
    goodness_rows: list[dict[str, object]] = []
    for label, data_path, theory_path, inverse_covariance_path in fit_specs:
        data = np.loadtxt(data_path).reshape(-1)
        theory = np.loadtxt(theory_path).reshape(-1)
        inverse_covariance = np.loadtxt(inverse_covariance_path)
        precision_cholesky = np.linalg.cholesky(inverse_covariance)
        whitened = precision_cholesky.T @ (data - theory)
        chi_squared = float(whitened @ whitened)
        residual_sets.append((label, whitened))
        goodness_rows.append(
            {
                "dataset": label,
                "n_data": len(data),
                "chi2": chi_squared,
                "chi2_per_data": chi_squared / len(data),
                "fixed_model_pte": chi2.sf(chi_squared, len(data)),
                "max_abs_whitened_residual": float(np.max(np.abs(whitened))),
            }
        )
    total_n = sum(int(row["n_data"]) for row in goodness_rows)
    total_chi2 = sum(float(row["chi2"]) for row in goodness_rows)
    goodness_rows.append(
        {
            "dataset": "Combined",
            "n_data": total_n,
            "chi2": total_chi2,
            "chi2_per_data": total_chi2 / total_n,
            "fixed_model_pte": chi2.sf(total_chi2, total_n),
            "max_abs_whitened_residual": max(
                float(row["max_abs_whitened_residual"]) for row in goodness_rows
            ),
        }
    )
    write_csv(OUT / "goodness_of_fit.csv", goodness_rows)

    plot_items = [
        ("bhat1", r"$\widehat b_1$", None, None),
        ("bhat2", r"$\widehat b_2$", None, None),
        ("fhat1", r"$\widehat f_1$", old_values["fhat1"], gr["fhat1"]),
        ("fhat2", r"$\widehat f_2$", old_values["fhat2"], gr["fhat2"]),
        ("Jhat1", r"$\widehat J_1$", old_values["Jhat1"], gr["Jhat1"]),
        ("Jhat2", r"$\widehat J_2$", old_values["Jhat2"], gr["Jhat2"]),
        ("EG1", r"$E_{G,1}$", old_values["EG1"], gr["EG1"]),
        ("EG2", r"$E_{G,2}$", old_values["EG2"], gr["EG2"]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for ax, (name, label, old_array, gr_array) in zip(axes.ravel(), plot_items):
        values = key[name]
        limits = weighted_quantile(values, new.weight, [0.001, 0.999])
        if old_array is not None:
            old_limits = weighted_quantile(old_array, old.weight, [0.001, 0.999])
            limits = [min(limits[0], old_limits[0]), max(limits[1], old_limits[1])]
            ax.hist(
                old_array, bins=65, range=limits, weights=old.weight, density=True,
                histtype="step", color="#777777", lw=1.6, label="Original",
            )
        ax.hist(
            values, bins=65, range=limits, weights=new.weight, density=True,
            histtype="stepfilled", color="#377eb8", alpha=0.45, label="Reparameterized",
        )
        if gr_array is not None:
            gr_stats = weighted_stats(gr_array, new.weight)
            ax.axvline(gr_stats["mean"], color="#e41a1c", ls="--", lw=1.5, label="GR")
        ax.set_xlabel(label, fontsize=12)
        ax.set_yticks([])
        ax.set_xlim(limits)
    handles = [
        Line2D([0], [0], color="#377eb8", lw=6, alpha=0.45, label="Reparameterized posterior"),
        Line2D([0], [0], color="#777777", lw=1.6, label="Original posterior"),
        Line2D([0], [0], color="#e41a1c", lw=1.5, ls="--", label="GR prediction"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Key weighted posterior distributions", y=1.03, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "key_marginals.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    for ax, quantity, xlabel in zip(
        axes, ("Jhat", "fhat", "EG"),
        (r"$\widehat J$", r"$\widehat f=f\sigma_8$", r"$E_G$"),
    ):
        for bin_index, redshift in ((1, 0.38), (2, 0.61)):
            old_stats = weighted_stats(old_values[f"{quantity}{bin_index}"], old.weight)
            new_stats = weighted_stats(key[f"{quantity}{bin_index}"], new.weight)
            gr_stats = weighted_stats(gr[f"{quantity}{bin_index}"], new.weight)
            ax.errorbar(
                old_stats["mean"], redshift - 0.012,
                xerr=[[old_stats["mean"] - old_stats["q16"]], [old_stats["q84"] - old_stats["mean"]]],
                fmt="o", color="#777777", capsize=3,
            )
            ax.errorbar(
                new_stats["mean"], redshift + 0.012,
                xerr=[[new_stats["mean"] - new_stats["q16"]], [new_stats["q84"] - new_stats["mean"]]],
                fmt="o", color="#377eb8", capsize=3,
            )
            ax.errorbar(
                gr_stats["mean"], redshift,
                xerr=gr_stats["sd"], fmt="D", color="#e41a1c", ms=5, capsize=2,
            )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Effective redshift")
    axes[0].set_yticks([0.38, 0.61], labels=["LOWZ  0.38", "CMASS  0.61"])
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="#777777", lw=0, label="Original (GR-fixed f)"),
            Line2D([0], [0], marker="o", color="#377eb8", lw=0, label="Reparameterized measurement"),
            Line2D([0], [0], marker="D", color="#e41a1c", lw=0, label="GR prediction"),
        ], loc="best", fontsize=9,
    )
    fig.suptitle("Original versus reparameterized-amplitude constraints", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "old_new_gr_comparison.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    order = [(1, "Jhat"), (1, "fhat"), (1, "EG"), (2, "Jhat"), (2, "fhat"), (2, "EG")]
    residual_labels = [r"$\widehat J_1$", r"$\widehat f_1$", r"$E_{G,1}$", r"$\widehat J_2$", r"$\widehat f_2$", r"$E_{G,2}$"]
    tensions = []
    for bin_index, quantity in order:
        row = next(r for r in gr_rows if r["bin"] == bin_index and r["quantity"] == quantity)
        tensions.append(float(row["tension_sigma"]))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(len(tensions))
    colors = ["#377eb8" if abs(value) < 1 else "#ff7f00" for value in tensions]
    ax.barh(positions, tensions, color=colors, alpha=0.82)
    ax.axvspan(-1, 1, color="#4daf4a", alpha=0.10, label=r"$|\Delta|<1\sigma$")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(-2, color="#999999", lw=0.8, ls="--")
    ax.axvline(2, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(positions, residual_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Posterior mean difference from GR / posterior $\sigma$")
    ax.set_xlim(-2.35, 2.35)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Consistency with the Planck-conditioned GR prediction")
    fig.tight_layout()
    fig.savefig(OUT / "gr_standardized_residuals.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    display_labels = [r"$\hat b_1$", r"$\hat f_1$", r"$\hat J_1$", r"$E_{G,1}$", r"$\hat b_2$", r"$\hat f_2$", r"$\hat J_2$", r"$E_{G,2}$"]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(display_labels)), display_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(display_labels)), display_labels)
    for row in range(len(display_labels)):
        for col in range(len(display_labels)):
            ax.text(col, row, f"{correlation_matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Weighted correlation")
    ax.set_title("Posterior correlation of measured late-time amplitudes")
    fig.tight_layout()
    fig.savefig(OUT / "key_correlation_heatmap.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    plot_corner(
        [key["bhat1"], key["fhat1"], key["Jhat1"], key["EG1"]],
        [r"$\widehat b_1$", r"$\widehat f_1$", r"$\widehat J_1$", r"$E_{G,1}$"],
        new.weight, OUT / "corner_lowz.png", "LOWZ reparameterized-amplitude posterior (z=0.38)",
    )
    plot_corner(
        [key["bhat2"], key["fhat2"], key["Jhat2"], key["EG2"]],
        [r"$\widehat b_2$", r"$\widehat f_2$", r"$\widehat J_2$", r"$E_{G,2}$"],
        new.weight, OUT / "corner_cmass.png", "CMASS reparameterized-amplitude posterior (z=0.61)",
    )

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=False)
    for ax, (label, residual) in zip(axes, residual_sets):
        ax.axhspan(-2, 2, color="#4daf4a", alpha=0.08)
        ax.axhline(0, color="black", lw=0.8)
        ax.scatter(np.arange(len(residual)), residual, s=12, color="#377eb8")
        ax.set_ylabel(label)
        ax.set_ylim(-4.2, 4.2)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Whitened data-vector index")
    fig.suptitle("Best-likelihood whitened residuals", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "bestfit_whitened_residuals.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    new_like = new.get("post") - new.get("prior")
    old_like = old.get("post") - old.get("prior")
    run_rows = [
        {"metric": "new_rows", "value": len(new.data)},
        {"metric": "new_effective_samples", "value": 1.0 / (new.weight @ new.weight)},
        {"metric": "new_max_loglike", "value": np.max(new_like)},
        {"metric": "old_max_loglike", "value": np.max(old_like)},
        {"metric": "delta_chi2_new_minus_old", "value": -2.0 * (np.max(new_like) - np.max(old_like))},
    ]
    write_csv(OUT / "run_diagnostics.csv", run_rows)

    print(f"Wrote diagnostics to {OUT}")


if __name__ == "__main__":
    main()
