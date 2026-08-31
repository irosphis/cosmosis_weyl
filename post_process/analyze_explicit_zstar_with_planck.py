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
OUT = ROOT / "analysis/explicit_zstar_with_planck"
ZSTAR_PATH = ROOT / "chain/chain_explicit_zstar_with_planck_nautilus.txt"
REPARAMETERIZED_PATH = ROOT / "chain/chain_reparameterized_with_planck_nautilus.txt"
BESTFIT = OUT / "bestfit"


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
    index = {name: position for position, name in enumerate(names)}
    valid = np.all(np.isfinite(data), axis=1)
    data = data[valid]
    log_weight = data[:, index["log_weight"]]
    weight = np.exp(log_weight - logsumexp(log_weight))
    return Chain(names, data, index, weight)


def weighted_quantile(values, weight, probabilities):
    order = np.argsort(values)
    cumulative = np.cumsum(weight[order])
    return np.interp(np.asarray(probabilities), cumulative, values[order])


def weighted_stats(values, weight):
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


def weighted_covariance(matrix, weight):
    mean = weight @ matrix
    centered = matrix - mean
    return centered.T @ (weight[:, None] * centered)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def growth_rate(omega_m0, redshift):
    a = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    argument = -ratio * a**3
    return 1.0 - (6.0 / 11.0) * ratio * a**3 * hyp2f1(
        4.0 / 3.0, 2.0, 17.0 / 6.0, argument
    ) / hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, argument)


def contour_levels(histogram, probabilities=(0.95, 0.68)):
    flat = np.sort(histogram.ravel())[::-1]
    cumulative = np.cumsum(flat) / np.sum(flat)
    values = [flat[min(np.searchsorted(cumulative, p), len(flat) - 1)] for p in probabilities]
    return sorted(values)


def draw_contours(ax, x, y, weight, color, linestyle="-", fill=False):
    xr = weighted_quantile(x, weight, [0.002, 0.998])
    yr = weighted_quantile(y, weight, [0.002, 0.998])
    hist, xe, ye = np.histogram2d(x, y, bins=60, range=[xr, yr], weights=weight)
    hist = gaussian_filter(hist, 1.1)
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    levels = contour_levels(hist)
    if fill:
        ax.contourf(xc, yc, hist.T, levels=[levels[0], levels[1], hist.max()], colors=[color, color], alpha=0.12)
    ax.contour(xc, yc, hist.T, levels=levels, colors=color, linestyles=linestyle, linewidths=[1.0, 1.7])


def comparison_corner(zstar, reparameterized, bin_index, output):
    specs = [
        (f"growth_amplitudes--bhat_bin_{bin_index}", rf"$\widehat b_{bin_index}$"),
        (f"growth_amplitudes--fhat_bin_{bin_index}", rf"$\widehat f_{bin_index}$"),
        (f"J_bin_bias--jhat{bin_index}", rf"$\widehat J_{bin_index}$"),
        (f"HATTED_DERIVED--EG_BIN_{bin_index}", rf"$E_{{G,{bin_index}}}$"),
    ]
    arrays_z = [zstar.get(name) for name, _ in specs]
    arrays_h = [reparameterized.get(name) for name, _ in specs]
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for row in range(4):
        for col in range(4):
            ax = axes[row, col]
            if row == col:
                limits = [
                    min(weighted_quantile(arrays_z[col], zstar.weight, [0.002])[0], weighted_quantile(arrays_h[col], reparameterized.weight, [0.002])[0]),
                    max(weighted_quantile(arrays_z[col], zstar.weight, [0.998])[0], weighted_quantile(arrays_h[col], reparameterized.weight, [0.998])[0]),
                ]
                ax.hist(arrays_h[col], 55, range=limits, weights=reparameterized.weight, density=True, histtype="step", color="#666666", lw=1.5)
                ax.hist(arrays_z[col], 55, range=limits, weights=zstar.weight, density=True, histtype="stepfilled", color="#0072B2", alpha=0.35)
                ax.set_yticks([])
            elif row > col:
                draw_contours(ax, arrays_h[col], arrays_h[row], reparameterized.weight, "#666666", "--")
                draw_contours(ax, arrays_z[col], arrays_z[row], zstar.weight, "#0072B2", "-", True)
            else:
                ax.axis("off")
            if row == 3:
                ax.set_xlabel(specs[col][1])
            elif row != col:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(specs[row][1])
            elif row > col:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], color="#0072B2", lw=3, label=r"Explicit $z_\star$"),
        Line2D([0], [0], color="#666666", lw=1.5, ls="--", label="Previous reparameterized"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"Redshift bin {bin_index}: weighted posterior comparison", y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 10, "mathtext.fontset": "dejavuserif"})
    zstar = load_chain(ZSTAR_PATH)
    reparameterized = load_chain(REPARAMETERIZED_PATH)

    labels = {
        "bhat1": "growth_amplitudes--bhat_bin_1",
        "fhat1": "growth_amplitudes--fhat_bin_1",
        "Jhat1": "J_bin_bias--jhat1",
        "EG1": "HATTED_DERIVED--EG_BIN_1",
        "bhat2": "growth_amplitudes--bhat_bin_2",
        "fhat2": "growth_amplitudes--fhat_bin_2",
        "Jhat2": "J_bin_bias--jhat2",
        "EG2": "HATTED_DERIVED--EG_BIN_2",
        "Omega_m": "COSMOLOGICAL_PARAMETERS--OMEGA_M",
        "S8": "COSMOLOGICAL_PARAMETERS--S_8",
        "sigma8": "COSMOLOGICAL_PARAMETERS--SIGMA_8",
        "nonlinear_correction_fraction": "ZSTAR_DERIVED--NONLINEAR_BIAS_CORRECTION_RMS_FRACTION",
    }

    summary = []
    comparison = []
    for short, name in labels.items():
        zs = weighted_stats(zstar.get(name), zstar.weight)
        summary.append({"quantity": short, **zs})
        if name in reparameterized.index:
            hs = weighted_stats(reparameterized.get(name), reparameterized.weight)
            comparison.append({
                "quantity": short,
                "reparameterized_mean": hs["mean"], "reparameterized_sd": hs["sd"],
                "reparameterized_q16": hs["q16"], "reparameterized_q84": hs["q84"],
                "zstar_mean": zs["mean"], "zstar_sd": zs["sd"],
                "zstar_q16": zs["q16"], "zstar_q84": zs["q84"],
                "shift_independent_sigma": (zs["mean"] - hs["mean"]) / np.hypot(zs["sd"], hs["sd"]),
            })
    write_csv(OUT / "posterior_summary.csv", summary)
    write_csv(OUT / "explicit_zstar_vs_reparameterized.csv", comparison)

    omega_m0 = zstar.get(labels["Omega_m"])
    gr_rows = []
    gr_values = {}
    for b, redshift in ((1, 0.38), (2, 0.61)):
        sigma8_z = zstar.get(f"HATTED_DERIVED--SIGMA8_REF_BIN_{b}")
        expansion2 = omega_m0 * (1.0 + redshift) ** 3 + 1.0 - omega_m0
        omega_m_z = omega_m0 * (1.0 + redshift) ** 3 / expansion2
        gr_values[f"Jhat{b}"] = omega_m_z * sigma8_z
        gr_values[f"fhat{b}"] = growth_rate(omega_m0, redshift) * sigma8_z
        eg = zstar.get(labels[f"EG{b}"])
        jhat = zstar.get(labels[f"Jhat{b}"])
        fhat = zstar.get(labels[f"fhat{b}"])
        geometry = eg * fhat / jhat
        gr_values[f"EG{b}"] = geometry * gr_values[f"Jhat{b}"] / gr_values[f"fhat{b}"]
        for quantity in ("Jhat", "fhat", "EG"):
            measured = zstar.get(labels[f"{quantity}{b}"])
            predicted = gr_values[f"{quantity}{b}"]
            difference = measured - predicted
            ms = weighted_stats(measured, zstar.weight)
            ps = weighted_stats(predicted, zstar.weight)
            ds = weighted_stats(difference, zstar.weight)
            gr_rows.append({
                "bin": b, "redshift": redshift, "quantity": quantity,
                "measured_mean": ms["mean"], "measured_sd": ms["sd"],
                "measured_q16": ms["q16"], "measured_q84": ms["q84"],
                "gr_mean": ps["mean"], "gr_sd": ps["sd"],
                "difference_mean": ds["mean"], "difference_sd": ds["sd"],
                "tension_sigma": ds["mean"] / ds["sd"],
                "probability_below_gr": float(zstar.weight[difference < 0].sum()),
            })
    write_csv(OUT / "gr_consistency.csv", gr_rows)

    corr_names = ["bhat1", "fhat1", "Jhat1", "EG1", "bhat2", "fhat2", "Jhat2", "EG2"]
    matrix = np.column_stack([zstar.get(labels[name]) for name in corr_names])
    covariance = weighted_covariance(matrix, zstar.weight)
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigma, sigma)
    with (OUT / "correlation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", *corr_names])
        for name, row in zip(corr_names, correlation):
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
    boundary_rows = []
    for name, (lower, upper) in bounds.items():
        values = zstar.get(name)
        stats = weighted_stats(values, zstar.weight)
        span = upper - lower
        boundary_rows.append({
            "quantity": name, "prior_lower": lower, "prior_upper": upper,
            "q005": stats["q005"], "q025": stats["q025"],
            "median": stats["median"], "q975": stats["q975"], "q995": stats["q995"],
            "mass_lower_2pct": float(zstar.weight[values < lower + 0.02 * span].sum()),
            "mass_upper_2pct": float(zstar.weight[values > upper - 0.02 * span].sum()),
        })
    write_csv(OUT / "prior_boundary_check.csv", boundary_rows)

    fit_specs = [
        ("BOSS LOWZ", "data_vector/xi_wedges_like_data_bin_1.txt", "data_vector/xi_wedges_like_theory_bin_1.txt", "data_vector/xi_wedges_like_inv_cov_bin_1.txt"),
        ("BOSS CMASS", "data_vector/xi_wedges_like_data_bin_2.txt", "data_vector/xi_wedges_like_theory_bin_2.txt", "data_vector/xi_wedges_like_inv_cov_bin_2.txt"),
        ("GGL", "theory_data_covariance/data.txt", "theory_data_covariance/theory.txt", "theory_data_covariance/inv_covariance.txt"),
    ]
    gof_rows = []
    residuals = []
    for label, data_file, theory_file, precision_file in fit_specs:
        data = np.loadtxt(BESTFIT / data_file).reshape(-1)
        theory = np.loadtxt(BESTFIT / theory_file).reshape(-1)
        precision = np.loadtxt(BESTFIT / precision_file)
        residual = data - theory
        whitened = np.linalg.cholesky(precision).T @ residual
        chi_squared = float(whitened @ whitened)
        residuals.append((label, whitened))
        gof_rows.append({
            "dataset": label, "n_data": len(data), "chi2": chi_squared,
            "chi2_per_data": chi_squared / len(data),
            "fixed_model_pte": chi2.sf(chi_squared, len(data)),
            "max_abs_whitened_residual": float(np.max(np.abs(whitened))),
        })
    total_n = sum(row["n_data"] for row in gof_rows)
    total_chi2 = sum(row["chi2"] for row in gof_rows)
    gof_rows.append({
        "dataset": "Combined", "n_data": total_n, "chi2": total_chi2,
        "chi2_per_data": total_chi2 / total_n,
        "fixed_model_pte": chi2.sf(total_chi2, total_n),
        "max_abs_whitened_residual": max(row["max_abs_whitened_residual"] for row in gof_rows),
    })
    write_csv(OUT / "goodness_of_fit.csv", gof_rows)

    zlike = zstar.get("post") - zstar.get("prior")
    hlike = reparameterized.get("post") - reparameterized.get("prior")
    diagnostics = [
        {"metric": "rows", "value": len(zstar.data)},
        {"metric": "effective_samples", "value": 1.0 / (zstar.weight @ zstar.weight)},
        {"metric": "max_loglike", "value": np.max(zlike)},
        {"metric": "reparameterized_max_loglike", "value": np.max(hlike)},
        {"metric": "delta_chi2_explicit_zstar_minus_reparameterized", "value": -2.0 * (np.max(zlike) - np.max(hlike))},
        {"metric": "log_z", "value": -139.500801663674},
        {"metric": "reparameterized_log_z", "value": -139.38282282772187},
        {"metric": "delta_log_z", "value": -139.500801663674 + 139.38282282772187},
    ]
    write_csv(OUT / "run_diagnostics.csv", diagnostics)

    plot_names = ["bhat1", "fhat1", "Jhat1", "EG1", "bhat2", "fhat2", "Jhat2", "EG2"]
    plot_labels = [r"$\widehat b_1$", r"$\widehat f_1$", r"$\widehat J_1$", r"$E_{G,1}$", r"$\widehat b_2$", r"$\widehat f_2$", r"$\widehat J_2$", r"$E_{G,2}$"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, short, label in zip(axes.ravel(), plot_names, plot_labels):
        zvalues = zstar.get(labels[short])
        hvalues = reparameterized.get(labels[short])
        lo = min(weighted_quantile(zvalues, zstar.weight, [0.001])[0], weighted_quantile(hvalues, reparameterized.weight, [0.001])[0])
        hi = max(weighted_quantile(zvalues, zstar.weight, [0.999])[0], weighted_quantile(hvalues, reparameterized.weight, [0.999])[0])
        ax.hist(hvalues, 65, range=(lo, hi), weights=reparameterized.weight, density=True, histtype="step", color="#666666", lw=1.6)
        ax.hist(zvalues, 65, range=(lo, hi), weights=zstar.weight, density=True, histtype="stepfilled", color="#0072B2", alpha=0.4)
        ax.set_xlabel(label, fontsize=12)
        ax.set_yticks([])
    fig.legend(handles=[Line2D([0], [0], color="#0072B2", lw=5, alpha=0.5, label=r"Explicit $z_\star$"), Line2D([0], [0], color="#666666", lw=1.6, label="Previous reparameterized")], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "explicit_zstar_vs_reparameterized_marginals.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "explicit_zstar_vs_reparameterized_marginals.pdf", bbox_inches="tight")
    plt.close(fig)

    comparison_corner(zstar, reparameterized, 1, OUT / "corner_lowz_comparison.png")
    comparison_corner(zstar, reparameterized, 2, OUT / "corner_cmass_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for ax, quantity, xlabel in zip(axes, ("Jhat", "fhat", "EG"), (r"$\widehat J$", r"$\widehat f$", r"$E_G$")):
        for b, redshift in ((1, 0.38), (2, 0.61)):
            ms = weighted_stats(zstar.get(labels[f"{quantity}{b}"]), zstar.weight)
            gs = weighted_stats(gr_values[f"{quantity}{b}"], zstar.weight)
            ax.errorbar(ms["mean"], redshift, xerr=[[ms["mean"] - ms["q16"]], [ms["q84"] - ms["mean"]]], fmt="o", color="#0072B2", capsize=3)
            ax.errorbar(gs["mean"], redshift, xerr=gs["sd"], fmt="s", color="#D55E00", capsize=3)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Effective redshift")
    axes[0].set_yticks([0.38, 0.61])
    fig.legend(handles=[Line2D([0], [0], marker="o", color="#0072B2", lw=0, label=r"Explicit $z_\star$"), Line2D([0], [0], marker="s", color="#D55E00", lw=0, label="Planck-conditioned GR")], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / "explicit_zstar_gr_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5))
    for ax, (label, residual) in zip(axes, residuals):
        ax.axhspan(-2, 2, color="#4daf4a", alpha=0.08)
        ax.axhline(0, color="black", lw=0.8)
        ax.scatter(np.arange(len(residual)), residual, s=11, color="#0072B2")
        ax.set_ylabel(label)
        ax.set_ylim(-4.1, 4.1)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Whitened data-vector index")
    fig.tight_layout()
    fig.savefig(OUT / "bestfit_whitened_residuals.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote explicit-zstar diagnostics to {OUT}")


if __name__ == "__main__":
    main()
