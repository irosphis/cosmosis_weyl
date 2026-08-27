from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import hyp2f1, logsumexp
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/reparameterized_no_planck_20260809"
NEW_NO_PATH = ROOT / "chain/chain_reparameterized_no_planck_nautilus.txt"
NEW_WITH_PATH = ROOT / "chain/chain_reparameterized_with_planck_nautilus.txt"
OLD_NO_PATH = ROOT / "MyTest/chains/chain_noprior.txt"
BESTFIT_DIR = ROOT / "kcap/output/analysis_reparameterized_no_planck_maxlike_20260811"

from analyze_reparameterized_with_planck import (
    plot_corner,
    weighted_covariance,
    weighted_quantile,
    weighted_stats,
)


@dataclass
class Chain:
    names: list[str]
    data: np.ndarray
    index: dict[str, int]
    weight: np.ndarray
    dropped_rows: int

    def get(self, name: str) -> np.ndarray:
        return self.data[:, self.index[name]]


def load_chain(path: Path) -> Chain:
    with path.open(encoding="utf-8") as handle:
        names = handle.readline().lstrip("#").strip().split("\t")
    data = np.loadtxt(path, comments="#")
    index = {name: i for i, name in enumerate(names)}
    valid = np.isfinite(data[:, index["log_weight"]]) & np.isfinite(data[:, index["post"]])
    dropped = int((~valid).sum())
    data = data[valid]
    log_weight = data[:, index["log_weight"]]
    weight = np.exp(log_weight - logsumexp(log_weight))
    return Chain(names, data, index, weight, dropped)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def linear_growth_rate_flat_lcdm(omega_m0: np.ndarray, redshift: float) -> np.ndarray:
    scale_factor = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    argument = -ratio * scale_factor**3
    first = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, argument)
    derivative = hyp2f1(4.0 / 3.0, 2.0, 17.0 / 6.0, argument)
    return 1.0 - (6.0 / 11.0) * ratio * scale_factor**3 * derivative / first


def chain_values(chain: Chain) -> dict[str, np.ndarray]:
    return {
        "bhat1": chain.get("growth_amplitudes--bhat_bin_1"),
        "fhat1": chain.get("growth_amplitudes--fhat_bin_1"),
        "bhat2": chain.get("growth_amplitudes--bhat_bin_2"),
        "fhat2": chain.get("growth_amplitudes--fhat_bin_2"),
        "Jhat1": chain.get("J_bin_bias--jhat1"),
        "Jhat2": chain.get("J_bin_bias--jhat2"),
        "EG1": chain.get("HATTED_DERIVED--EG_BIN_1"),
        "EG2": chain.get("HATTED_DERIVED--EG_BIN_2"),
        "b1eff1": chain.get("HATTED_DERIVED--B1_EFF_BIN_1"),
        "b1eff2": chain.get("HATTED_DERIVED--B1_EFF_BIN_2"),
        "feff1": chain.get("HATTED_DERIVED--F_EFF_BIN_1"),
        "feff2": chain.get("HATTED_DERIVED--F_EFF_BIN_2"),
        "Omega_m": chain.get("COSMOLOGICAL_PARAMETERS--OMEGA_M"),
        "S8": chain.get("COSMOLOGICAL_PARAMETERS--S_8"),
        "sigma8": chain.get("COSMOLOGICAL_PARAMETERS--SIGMA_8"),
    }


def original_values(chain: Chain) -> dict[str, np.ndarray]:
    omega_m = chain.get("COSMOLOGICAL_PARAMETERS--OMEGA_M")
    values = {
        "Omega_m": omega_m,
        "S8": chain.get("COSMOLOGICAL_PARAMETERS--S_8"),
        "sigma8": chain.get("COSMOLOGICAL_PARAMETERS--SIGMA_8"),
        "Jhat1": chain.get("J_bin_bias--jhat1"),
        "Jhat2": chain.get("J_bin_bias--jhat2"),
        "fhat1": chain.get("LSS_PARAMETERS--FSIGMA_8_BIN_1"),
        "fhat2": chain.get("LSS_PARAMETERS--FSIGMA_8_BIN_2"),
    }
    for bin_index, redshift in ((1, 0.38), (2, 0.61)):
        geometry = omega_m + (1.0 - omega_m) / (1.0 + redshift) ** 3
        values[f"EG{bin_index}"] = (
            geometry * values[f"Jhat{bin_index}"] / values[f"fhat{bin_index}"]
        )
    return values


def finite_stats(values: np.ndarray, chain: Chain) -> dict[str, float]:
    valid = np.isfinite(values)
    weight = chain.weight[valid]
    weight /= weight.sum()
    return weighted_stats(values[valid], weight)


def comparison_row(
    quantity: str,
    first_values: np.ndarray,
    first_chain: Chain,
    second_values: np.ndarray,
    second_chain: Chain,
    comparison: str,
) -> dict[str, object]:
    first = finite_stats(first_values, first_chain)
    second = finite_stats(second_values, second_chain)
    return {
        "comparison": comparison,
        "quantity": quantity,
        "first_mean": first["mean"],
        "first_sd": first["sd"],
        "first_q16": first["q16"],
        "first_q84": first["q84"],
        "second_mean": second["mean"],
        "second_sd": second["sd"],
        "second_q16": second["q16"],
        "second_q84": second["q84"],
        "shift_over_independent_sigma": (
            first["mean"] - second["mean"]
        ) / np.hypot(first["sd"], second["sd"]),
    }


def main() -> None:
    new_no = load_chain(NEW_NO_PATH)
    new_with = load_chain(NEW_WITH_PATH)
    old_no = load_chain(OLD_NO_PATH)
    no = chain_values(new_no)
    with_prior = chain_values(new_with)
    original = original_values(old_no)

    summary_rows = []
    for name, values in no.items():
        summary_rows.append({"quantity": name, **weighted_stats(values, new_no.weight)})
    write_csv(OUT / "posterior_summary.csv", summary_rows)

    comparisons = []
    common_with = (
        "Omega_m", "S8", "sigma8", "bhat1", "fhat1", "bhat2", "fhat2",
        "Jhat1", "Jhat2", "EG1", "EG2", "b1eff1", "b1eff2", "feff1", "feff2",
    )
    for quantity in common_with:
        comparisons.append(
            comparison_row(
                quantity, no[quantity], new_no, with_prior[quantity], new_with,
                "new_no_minus_new_with",
            )
        )
    for quantity in ("Omega_m", "S8", "sigma8", "fhat1", "fhat2", "Jhat1", "Jhat2", "EG1", "EG2"):
        comparisons.append(
            comparison_row(
                quantity, no[quantity], new_no, original[quantity], old_no,
                "new_no_minus_original_no",
            )
        )
    write_csv(OUT / "chain_comparisons.csv", comparisons)

    omega_m = no["Omega_m"]
    formal_gr: dict[str, np.ndarray] = {}
    gr_rows = []
    for bin_index, redshift in ((1, 0.38), (2, 0.61)):
        sigma8_z = new_no.get(f"HATTED_DERIVED--SIGMA8_REF_BIN_{bin_index}")
        expansion_squared = omega_m * (1.0 + redshift) ** 3 + 1.0 - omega_m
        omega_m_z = omega_m * (1.0 + redshift) ** 3 / expansion_squared
        formal_gr[f"Jhat{bin_index}"] = omega_m_z * sigma8_z
        formal_gr[f"fhat{bin_index}"] = (
            linear_growth_rate_flat_lcdm(omega_m, redshift) * sigma8_z
        )
        geometry = no[f"EG{bin_index}"] * no[f"fhat{bin_index}"] / no[f"Jhat{bin_index}"]
        formal_gr[f"EG{bin_index}"] = (
            geometry * formal_gr[f"Jhat{bin_index}"] / formal_gr[f"fhat{bin_index}"]
        )
        for stem in ("Jhat", "fhat", "EG"):
            measured = no[f"{stem}{bin_index}"]
            prediction = formal_gr[f"{stem}{bin_index}"]
            difference = measured - prediction
            ms = weighted_stats(measured, new_no.weight)
            gs = weighted_stats(prediction, new_no.weight)
            ds = weighted_stats(difference, new_no.weight)
            gr_rows.append(
                {
                    "bin": bin_index,
                    "redshift": redshift,
                    "quantity": stem,
                    "measured_mean": ms["mean"],
                    "measured_sd": ms["sd"],
                    "formal_gr_mean": gs["mean"],
                    "formal_gr_sd": gs["sd"],
                    "difference_mean": ds["mean"],
                    "difference_sd": ds["sd"],
                    "formal_tension_sigma": ds["mean"] / ds["sd"],
                    "probability_below_formal_gr": float(new_no.weight[difference < 0].sum()),
                    "physical_interpretation_valid": stem == "EG",
                }
            )
    write_csv(OUT / "formal_gr_diagnostic.csv", gr_rows)

    bounds = {
        "cosmological_parameters--omch2": (0.051, 0.2),
        "cosmological_parameters--ombh2": (0.019, 0.026),
        "cosmological_parameters--h0": (0.64, 0.82),
        "cosmological_parameters--n_s": (0.84, 1.1),
        "cosmological_parameters--s_8_input": (0.1, 1.3),
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
    edge_rows = []
    for name, (lower, upper) in bounds.items():
        values = new_no.get(name)
        stats = weighted_stats(values, new_no.weight)
        span = upper - lower
        edge_rows.append(
            {
                "quantity": name,
                "lower": lower,
                "upper": upper,
                "q005": stats["q005"],
                "q025": stats["q025"],
                "median": stats["median"],
                "q975": stats["q975"],
                "q995": stats["q995"],
                "mass_lower_2pct": float(new_no.weight[values < lower + 0.02 * span].sum()),
                "mass_upper_2pct": float(new_no.weight[values > upper - 0.02 * span].sum()),
            }
        )
    write_csv(OUT / "prior_boundary_check.csv", edge_rows)

    correlation_names = ["S8", "bhat1", "fhat1", "Jhat1", "EG1", "bhat2", "fhat2", "Jhat2", "EG2"]
    covariance = weighted_covariance(
        np.column_stack([no[name] for name in correlation_names]), new_no.weight
    )
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigma, sigma)
    with (OUT / "correlation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", *correlation_names])
        for name, row in zip(correlation_names, correlation):
            writer.writerow([name, *row])

    fit_specs = [
        ("BOSS LOWZ", "data_vector/xi_wedges_like_data_bin_1.txt", "data_vector/xi_wedges_like_theory_bin_1.txt", "data_vector/xi_wedges_like_inv_cov_bin_1.txt"),
        ("BOSS CMASS", "data_vector/xi_wedges_like_data_bin_2.txt", "data_vector/xi_wedges_like_theory_bin_2.txt", "data_vector/xi_wedges_like_inv_cov_bin_2.txt"),
        ("GGL", "theory_data_covariance/data.txt", "theory_data_covariance/theory.txt", "theory_data_covariance/inv_covariance.txt"),
    ]
    goodness = []
    residual_sets = []
    for label, data_rel, theory_rel, precision_rel in fit_specs:
        data = np.loadtxt(BESTFIT_DIR / data_rel).reshape(-1)
        theory = np.loadtxt(BESTFIT_DIR / theory_rel).reshape(-1)
        precision = np.loadtxt(BESTFIT_DIR / precision_rel)
        cholesky = np.linalg.cholesky(precision)
        residual = cholesky.T @ (data - theory)
        value = float(residual @ residual)
        residual_sets.append((label, residual))
        goodness.append(
            {
                "dataset": label,
                "n_data": len(data),
                "chi2": value,
                "chi2_per_data": value / len(data),
                "fixed_model_pte": chi2.sf(value, len(data)),
                "max_abs_whitened_residual": float(np.max(np.abs(residual))),
            }
        )
    total_n = sum(int(row["n_data"]) for row in goodness)
    total_chi2 = sum(float(row["chi2"]) for row in goodness)
    goodness.append(
        {
            "dataset": "Combined",
            "n_data": total_n,
            "chi2": total_chi2,
            "chi2_per_data": total_chi2 / total_n,
            "fixed_model_pte": chi2.sf(total_chi2, total_n),
            "max_abs_whitened_residual": max(float(row["max_abs_whitened_residual"]) for row in goodness),
        }
    )
    write_csv(OUT / "goodness_of_fit.csv", goodness)

    new_no_like = new_no.get("post") - new_no.get("prior")
    new_with_like = new_with.get("post") - new_with.get("prior")
    old_no_like = old_no.get("post") - old_no.get("prior")
    run_rows = [
        {"metric": "calls", "value": 41600},
        {"metric": "reported_neff", "value": 8266.427807125707},
        {"metric": "weighted_neff", "value": 1.0 / (new_no.weight @ new_no.weight)},
        {"metric": "log_z", "value": -135.60230129781584},
        {"metric": "log_z_error", "value": 0.01099869252750629},
        {"metric": "new_no_max_loglike", "value": np.max(new_no_like)},
        {"metric": "new_with_max_loglike", "value": np.max(new_with_like)},
        {"metric": "old_no_max_loglike", "value": np.max(old_no_like)},
        {"metric": "delta_chi2_new_no_minus_new_with", "value": -2 * (np.max(new_no_like) - np.max(new_with_like))},
        {"metric": "delta_chi2_new_no_minus_old_no", "value": -2 * (np.max(new_no_like) - np.max(old_no_like))},
        {"metric": "old_no_failed_rows_dropped", "value": old_no.dropped_rows},
    ]
    write_csv(OUT / "run_diagnostics.csv", run_rows)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.1))
    items = [
        ("bhat1", r"$\widehat b_1$"), ("bhat2", r"$\widehat b_2$"),
        ("fhat1", r"$\widehat f_1$"), ("fhat2", r"$\widehat f_2$"),
        ("Jhat1", r"$\widehat J_1$"), ("Jhat2", r"$\widehat J_2$"),
        ("EG1", r"$E_{G,1}$"), ("EG2", r"$E_{G,2}$"),
    ]
    for ax, (name, label) in zip(axes.ravel(), items):
        arrays = [(no[name], new_no.weight, "No Planck", "#377eb8")]
        if name in with_prior:
            arrays.append((with_prior[name], new_with.weight, "With Planck", "#e41a1c"))
        limits = [
            min(weighted_quantile(a, w, [0.001])[0] for a, w, _, _ in arrays),
            max(weighted_quantile(a, w, [0.999])[0] for a, w, _, _ in arrays),
        ]
        for array, weight, _, color in arrays:
            ax.hist(array, bins=65, range=limits, weights=weight, density=True, histtype="step", color=color, lw=1.8)
        ax.set_xlabel(label, fontsize=12)
        ax.set_yticks([])
    fig.legend(
        handles=[
            Line2D([0], [0], color="#377eb8", lw=2, label="Reparameterized No Planck prior"),
            Line2D([0], [0], color="#e41a1c", lw=2, label="Reparameterized With Planck prior"),
        ], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False,
    )
    fig.suptitle("Late-time reparameterized amplitudes: impact of Planck priors", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(OUT / "with_vs_without_planck_marginals.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].hist(no["S8"], bins=70, weights=new_no.weight, density=True, histtype="stepfilled", color="#377eb8", alpha=0.55)
    axes[0].hist(with_prior["S8"], bins=50, weights=new_with.weight, density=True, histtype="step", color="#e41a1c", lw=1.8)
    axes[0].set_xlabel(r"Reference $S_8$")
    axes[0].set_yticks([])
    for ax, yname, ylabel in zip(axes[1:], ("Jhat2", "EG2"), (r"$\widehat J_2$", r"$E_{G,2}$")):
        plot_range = [
            weighted_quantile(no["S8"], new_no.weight, [0.001, 0.999]),
            weighted_quantile(no[yname], new_no.weight, [0.001, 0.999]),
        ]
        h, xe, ye = np.histogram2d(
            no["S8"], no[yname], bins=65, range=plot_range, weights=new_no.weight
        )
        ax.pcolormesh(xe, ye, h.T, shading="auto", cmap="Blues")
        ax.set_xlabel(r"Reference $S_8$")
        ax.set_ylabel(ylabel)
    axes[0].legend(handles=[Line2D([0], [0], color="#377eb8", lw=5, alpha=.55, label="No Planck"), Line2D([0], [0], color="#e41a1c", lw=2, label="With Planck")], frameon=False)
    fig.suptitle("Reference-amplitude drift and residual template coupling")
    fig.tight_layout()
    fig.savefig(OUT / "reference_s8_diagnostic.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    plot_corner(
        [no["bhat1"], no["fhat1"], no["Jhat1"], no["EG1"]],
        [r"$\widehat b_1$", r"$\widehat f_1$", r"$\widehat J_1$", r"$E_{G,1}$"],
        new_no.weight, OUT / "corner_lowz.png", "LOWZ reparameterized No-Prior posterior",
    )
    plot_corner(
        [no["bhat2"], no["fhat2"], no["Jhat2"], no["EG2"]],
        [r"$\widehat b_2$", r"$\widehat f_2$", r"$\widehat J_2$", r"$E_{G,2}$"],
        new_no.weight, OUT / "corner_cmass.png", "CMASS reparameterized No-Prior posterior",
    )

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2))
    for ax, (label, residual) in zip(axes, residual_sets):
        ax.axhspan(-2, 2, color="#4daf4a", alpha=0.08)
        ax.axhline(0, color="black", lw=0.8)
        ax.scatter(np.arange(len(residual)), residual, s=12, color="#377eb8")
        ax.set_ylabel(label)
        ax.set_ylim(-4.2, 4.2)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Whitened data-vector index")
    fig.suptitle("No-Prior maximum-likelihood whitened residuals")
    fig.tight_layout()
    fig.savefig(OUT / "bestfit_whitened_residuals.png", dpi=210, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote No-Prior diagnostics to {OUT}")


if __name__ == "__main__":
    main()
