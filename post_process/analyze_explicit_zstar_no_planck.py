from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import hyp2f1, logsumexp
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/explicit_zstar_no_planck_20260818"
CHAIN_PATH = ROOT / "chain/chain_explicit_zstar_no_planck_nautilus.txt"
CHECKPOINT_PATH = ROOT / "chain/chain_explicit_zstar_no_planck_nautilus.sampler_status.hdf5"
REPARAMETERIZED_NO_PATH = ROOT / "chain/chain_reparameterized_no_planck_nautilus.txt"
ZSTAR_WITH_PATH = ROOT / "chain/chain_explicit_zstar_with_planck_nautilus.txt"
BESTFIT_DIR = ROOT / "kcap/output/analysis_explicit_zstar_no_planck_maxlike_20260818"


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
    valid = np.isfinite(data[:, index["log_weight"]]) & np.isfinite(data[:, index["post"]])
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


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata(path: Path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#") or "=" not in line:
                continue
            key, value = line[1:].strip().split("=", 1)
            result[key] = value
    return result


def linear_growth_rate_flat_lcdm(omega_m0, redshift):
    scale_factor = 1.0 / (1.0 + redshift)
    ratio = (1.0 - omega_m0) / omega_m0
    argument = -ratio * scale_factor**3
    numerator = hyp2f1(4.0 / 3.0, 2.0, 17.0 / 6.0, argument)
    denominator = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, argument)
    return 1.0 - (6.0 / 11.0) * ratio * scale_factor**3 * numerator / denominator


def chain_values(chain: Chain):
    return {
        "bhat1": chain.get("growth_amplitudes--bhat_bin_1"),
        "fhat1": chain.get("growth_amplitudes--fhat_bin_1"),
        "Jhat1": chain.get("J_bin_bias--jhat1"),
        "EG1": chain.get("HATTED_DERIVED--EG_BIN_1"),
        "bhat2": chain.get("growth_amplitudes--bhat_bin_2"),
        "fhat2": chain.get("growth_amplitudes--fhat_bin_2"),
        "Jhat2": chain.get("J_bin_bias--jhat2"),
        "EG2": chain.get("HATTED_DERIVED--EG_BIN_2"),
        "Omega_m": chain.get("COSMOLOGICAL_PARAMETERS--OMEGA_M"),
        "S8": chain.get("COSMOLOGICAL_PARAMETERS--S_8"),
        "sigma8": chain.get("COSMOLOGICAL_PARAMETERS--SIGMA_8"),
    }


def comparison_row(quantity, first_values, first_chain, second_values, second_chain, comparison):
    first = weighted_stats(first_values, first_chain.weight)
    second = weighted_stats(second_values, second_chain.weight)
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
            (first["mean"] - second["mean"]) / np.hypot(first["sd"], second["sd"])
        ),
    }


def load_timeout_data():
    stages = []
    points = []
    log_likelihood = []
    with h5py.File(CHECKPOINT_PATH, "r") as handle:
        sampler = handle["sampler"]
        indices = sorted(
            int(key.removeprefix("log_l_"))
            for key in sampler
            if key.startswith("log_l_") and key.removeprefix("log_l_").isdigit()
        )
        for stage in indices:
            ll = sampler[f"log_l_{stage}"][:]
            pp = sampler[f"points_{stage}"][:]
            failure = ~np.isfinite(ll)
            finite = ll[~failure]
            stages.append(
                {
                    "stage": stage,
                    "evaluations": len(ll),
                    "timeouts": int(failure.sum()),
                    "timeout_fraction": float(failure.mean()),
                    "max_finite_loglike": float(np.max(finite)),
                }
            )
            points.append(pp)
            log_likelihood.append(ll)
    return stages, np.vstack(points), np.concatenate(log_likelihood)


def main():
    plt.rcParams.update({"font.size": 10, "mathtext.fontset": "dejavuserif"})
    chain = load_chain(CHAIN_PATH)
    reparameterized_no = load_chain(REPARAMETERIZED_NO_PATH)
    zstar_with = load_chain(ZSTAR_WITH_PATH)
    values = chain_values(chain)
    reparameterized_values = chain_values(reparameterized_no)
    with_values = chain_values(zstar_with)

    labels = {
        "omch2": "cosmological_parameters--omch2",
        "ombh2": "cosmological_parameters--ombh2",
        "h0": "cosmological_parameters--h0",
        "n_s": "cosmological_parameters--n_s",
        "S8_input": "cosmological_parameters--s_8_input",
        "Omega_m": "COSMOLOGICAL_PARAMETERS--OMEGA_M",
        "S8": "COSMOLOGICAL_PARAMETERS--S_8",
        "sigma8": "COSMOLOGICAL_PARAMETERS--SIGMA_8",
        "A_s": "COSMOLOGICAL_PARAMETERS--A_S",
        "bhat1": "growth_amplitudes--bhat_bin_1",
        "fhat1": "growth_amplitudes--fhat_bin_1",
        "Jhat1": "J_bin_bias--jhat1",
        "EG1": "HATTED_DERIVED--EG_BIN_1",
        "bhat2": "growth_amplitudes--bhat_bin_2",
        "fhat2": "growth_amplitudes--fhat_bin_2",
        "Jhat2": "J_bin_bias--jhat2",
        "EG2": "HATTED_DERIVED--EG_BIN_2",
        "bhat_Jhat1": "ZSTAR_DERIVED--BHAT_JHAT_BIN_1",
        "bhat_Jhat2": "ZSTAR_DERIVED--BHAT_JHAT_BIN_2",
        "sigma8_star": "ZSTAR_DERIVED--SIGMA8_STAR",
        "sigma8_star_ggl": "ZSTAR_DERIVED--SIGMA8_STAR_GGL",
        "nonlinear_correction_fraction": "ZSTAR_DERIVED--NONLINEAR_BIAS_CORRECTION_RMS_FRACTION",
    }
    summary_rows = []
    for short, full in labels.items():
        summary_rows.append({"quantity": short, **weighted_stats(chain.get(full), chain.weight)})
    write_csv(OUT / "posterior_summary.csv", summary_rows)

    comparisons = []
    for quantity in values:
        comparisons.append(
            comparison_row(
                quantity,
                values[quantity],
                chain,
                reparameterized_values[quantity],
                reparameterized_no,
                "explicit_zstar_no_planck_minus_reparameterized_no_planck",
            )
        )
        comparisons.append(
            comparison_row(
                quantity,
                values[quantity],
                chain,
                with_values[quantity],
                zstar_with,
                "explicit_zstar_no_planck_minus_explicit_zstar_with_planck",
            )
        )
    write_csv(OUT / "chain_comparisons.csv", comparisons)

    gr_rows = []
    omega_m0 = values["Omega_m"]
    gr_eg = {}
    for bin_index, redshift in ((1, 0.38), (2, 0.61)):
        omega_m_z = (
            omega_m0 * (1.0 + redshift) ** 3
            / (omega_m0 * (1.0 + redshift) ** 3 + 1.0 - omega_m0)
        )
        growth = linear_growth_rate_flat_lcdm(omega_m0, redshift)
        geometry = values[f"EG{bin_index}"] * values[f"fhat{bin_index}"] / values[f"Jhat{bin_index}"]
        prediction = geometry * omega_m_z / growth
        gr_eg[bin_index] = prediction
        measured = values[f"EG{bin_index}"]
        difference = measured - prediction
        ms = weighted_stats(measured, chain.weight)
        gs = weighted_stats(prediction, chain.weight)
        ds = weighted_stats(difference, chain.weight)
        gr_rows.append(
            {
                "bin": bin_index,
                "redshift": redshift,
                "measured_mean": ms["mean"],
                "measured_sd": ms["sd"],
                "measured_q16": ms["q16"],
                "measured_q84": ms["q84"],
                "gr_mean": gs["mean"],
                "gr_sd": gs["sd"],
                "difference_mean": ds["mean"],
                "difference_sd": ds["sd"],
                "tension_sigma": ds["mean"] / ds["sd"],
                "probability_below_gr": float(chain.weight[difference < 0].sum()),
            }
        )
    write_csv(OUT / "eg_gr_consistency.csv", gr_rows)

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
    boundary_rows = []
    for name, (lower, upper) in bounds.items():
        array = chain.get(name)
        stats = weighted_stats(array, chain.weight)
        span = upper - lower
        boundary_rows.append(
            {
                "quantity": name,
                "lower": lower,
                "upper": upper,
                "q005": stats["q005"],
                "q025": stats["q025"],
                "median": stats["median"],
                "q975": stats["q975"],
                "q995": stats["q995"],
                "mass_lower_2pct": float(chain.weight[array < lower + 0.02 * span].sum()),
                "mass_upper_2pct": float(chain.weight[array > upper - 0.02 * span].sum()),
            }
        )
    write_csv(OUT / "prior_boundary_check.csv", boundary_rows)

    correlation_names = ["S8", "bhat1", "fhat1", "Jhat1", "EG1", "bhat2", "fhat2", "Jhat2", "EG2"]
    matrix = np.column_stack([values[name] for name in correlation_names])
    covariance = weighted_covariance(matrix, chain.weight)
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigma, sigma)
    with (OUT / "correlation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", *correlation_names])
        for name, row in zip(correlation_names, correlation):
            writer.writerow([name, *row])

    stages, unit_points, all_loglike = load_timeout_data()
    write_csv(OUT / "timeout_by_stage.csv", stages)
    failed = ~np.isfinite(all_loglike)
    early_count = sum(row["evaluations"] for row in stages if row["stage"] <= 12)
    parameter_positions = {
        "omch2": (0, 0.051, 0.2),
        "ombh2": (1, 0.019, 0.026),
        "h0": (2, 0.64, 0.82),
        "n_s": (3, 0.84, 1.1),
        "S8_input": (4, 0.1, 1.3),
        "bhat1": (12, 0.1, 5.0),
        "fhat1": (13, 0.05, 1.2),
        "bhat2": (14, 0.1, 5.0),
        "fhat2": (15, 0.05, 1.2),
    }
    timeout_parameter_rows = []
    early_success = np.arange(len(failed)) < early_count
    early_success &= ~failed
    for name, (position, lower, upper) in parameter_positions.items():
        physical = lower + (upper - lower) * unit_points[:, position]
        good_q = np.quantile(physical[early_success], [0.05, 0.5, 0.95])
        bad_q = np.quantile(physical[failed], [0.05, 0.5, 0.95])
        timeout_parameter_rows.append(
            {
                "quantity": name,
                "early_success_q05": good_q[0],
                "early_success_median": good_q[1],
                "early_success_q95": good_q[2],
                "timeout_q05": bad_q[0],
                "timeout_median": bad_q[1],
                "timeout_q95": bad_q[2],
            }
        )
    write_csv(OUT / "timeout_parameter_comparison.csv", timeout_parameter_rows)

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
        residual = np.linalg.cholesky(precision).T @ (data - theory)
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
    total_n = sum(row["n_data"] for row in goodness)
    total_chi2 = sum(row["chi2"] for row in goodness)
    goodness.append(
        {
            "dataset": "Combined",
            "n_data": total_n,
            "chi2": total_chi2,
            "chi2_per_data": total_chi2 / total_n,
            "fixed_model_pte": chi2.sf(total_chi2, total_n),
            "max_abs_whitened_residual": max(row["max_abs_whitened_residual"] for row in goodness),
        }
    )
    write_csv(OUT / "goodness_of_fit.csv", goodness)

    meta = metadata(CHAIN_PATH)
    loglike = chain.get("post") - chain.get("prior")
    reparameterized_loglike = reparameterized_no.get("post") - reparameterized_no.get("prior")
    with_loglike = zstar_with.get("post") - zstar_with.get("prior")
    run_rows = [
        {"metric": "calls", "value": int(meta["evaluations"])},
        {"metric": "successful_calls", "value": int(meta["successes"])},
        {"metric": "timeouts", "value": int(failed.sum())},
        {"metric": "timeout_fraction", "value": float(failed.sum() / int(meta["evaluations"]))},
        {"metric": "calls_before_timeouts_ceased", "value": early_count},
        {"metric": "later_calls_with_zero_timeouts", "value": int(meta["evaluations"]) - early_count},
        {"metric": "reported_neff", "value": float(meta["neff"])},
        {"metric": "weighted_neff", "value": float(1.0 / (chain.weight @ chain.weight))},
        {"metric": "log_z", "value": float(meta["log_z"])},
        {"metric": "log_z_error", "value": float(meta["log_z_error"])},
        {"metric": "max_loglike", "value": float(np.max(loglike))},
        {"metric": "reparameterized_no_max_loglike", "value": float(np.max(reparameterized_loglike))},
        {"metric": "delta_chi2_explicit_zstar_no_planck_minus_reparameterized_no_planck", "value": float(-2.0 * (np.max(loglike) - np.max(reparameterized_loglike)))},
        {"metric": "explicit_zstar_with_planck_max_loglike", "value": float(np.max(with_loglike))},
        {"metric": "delta_chi2_explicit_zstar_no_planck_minus_explicit_zstar_with_planck", "value": float(-2.0 * (np.max(loglike) - np.max(with_loglike)))},
        {"metric": "delta_log_z_explicit_zstar_no_planck_minus_reparameterized_no_planck", "value": float(meta["log_z"]) - (-135.60230129781584)},
        {"metric": "complete", "value": int(meta["complete"])},
    ]
    write_csv(OUT / "run_diagnostics.csv", run_rows)

    plot_items = [
        ("bhat1", r"$\widehat b_1$"), ("fhat1", r"$\widehat f_1$"),
        ("Jhat1", r"$\widehat J_1$"), ("EG1", r"$E_{G,1}$"),
        ("bhat2", r"$\widehat b_2$"), ("fhat2", r"$\widehat f_2$"),
        ("Jhat2", r"$\widehat J_2$"), ("EG2", r"$E_{G,2}$"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.4))
    for ax, (name, label) in zip(axes.ravel(), plot_items):
        arrays = [values[name], reparameterized_values[name], with_values[name]]
        weights = [chain.weight, reparameterized_no.weight, zstar_with.weight]
        lo = min(weighted_quantile(a, w, [0.001])[0] for a, w in zip(arrays, weights))
        hi = max(weighted_quantile(a, w, [0.999])[0] for a, w in zip(arrays, weights))
        ax.hist(arrays[2], 65, range=(lo, hi), weights=weights[2], density=True, histtype="step", color="#D55E00", lw=1.5)
        ax.hist(arrays[1], 65, range=(lo, hi), weights=weights[1], density=True, histtype="step", color="#555555", lw=1.4, ls="--")
        ax.hist(arrays[0], 65, range=(lo, hi), weights=weights[0], density=True, histtype="stepfilled", color="#0072B2", alpha=0.38)
        ax.set_xlabel(label, fontsize=12)
        ax.set_yticks([])
    fig.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", lw=5, alpha=0.45, label=r"Explicit $z_\star$, No Planck"),
            Line2D([0], [0], color="#555555", lw=1.5, ls="--", label="Reparameterized, No Planck"),
            Line2D([0], [0], color="#D55E00", lw=1.5, label=r"Explicit $z_\star$, With Planck"),
        ],
        loc="upper center", ncol=3, frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "late_time_marginal_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "late_time_marginal_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    axes[0].hist(values["S8"], 65, weights=chain.weight, density=True, histtype="stepfilled", color="#0072B2", alpha=0.45)
    axes[0].hist(reparameterized_values["S8"], 65, weights=reparameterized_no.weight, density=True, histtype="step", color="#555555", ls="--")
    axes[0].hist(with_values["S8"], 55, weights=zstar_with.weight, density=True, histtype="step", color="#D55E00")
    axes[0].set_xlabel(r"Reference $S_8$")
    axes[0].set_yticks([])
    for ax, key, ylabel in zip(axes[1:], ("Jhat2", "EG2"), (r"$\widehat J_2$", r"$E_{G,2}$")):
        xr = weighted_quantile(values["S8"], chain.weight, [0.001, 0.999])
        yr = weighted_quantile(values[key], chain.weight, [0.001, 0.999])
        hist, xe, ye = np.histogram2d(values["S8"], values[key], bins=65, range=[xr, yr], weights=chain.weight)
        ax.pcolormesh(xe, ye, hist.T, shading="auto", cmap="Blues")
        ax.set_xlabel(r"Reference $S_8$")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(OUT / "reference_s8_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axes[0].bar([row["stage"] for row in stages], [row["timeout_fraction"] for row in stages], color="#D55E00")
    axes[0].axvline(12.5, color="black", ls="--", lw=1)
    axes[0].set_xlabel("Nautilus bound/shell stage")
    axes[0].set_ylabel("Timeout fraction")
    axes[0].set_title("No timeout after stage 12")
    omch2 = 0.051 + (0.2 - 0.051) * unit_points[:, 0]
    ns = 0.84 + (1.1 - 0.84) * unit_points[:, 3]
    early_finite = early_success
    axes[1].scatter(omch2[early_finite], ns[early_finite], s=3, color="#999999", alpha=0.12, rasterized=True)
    axes[1].scatter(omch2[failed], ns[failed], s=9, color="#D55E00", alpha=0.75, label="Timed out")
    axes[1].set_xlabel(r"$\Omega_c h^2$")
    axes[1].set_ylabel(r"$n_s$")
    axes[1].legend(frameon=False)
    axes[1].set_title("Timeouts occupy extreme early proposals")
    fig.tight_layout()
    fig.savefig(OUT / "timeout_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for bin_index, redshift in ((1, 0.38), (2, 0.61)):
        ms = weighted_stats(values[f"EG{bin_index}"], chain.weight)
        gs = weighted_stats(gr_eg[bin_index], chain.weight)
        ax.errorbar(ms["mean"], redshift + 0.008, xerr=[[ms["mean"] - ms["q16"]], [ms["q84"] - ms["mean"]]], fmt="o", color="#0072B2", capsize=3)
        ax.errorbar(gs["mean"], redshift - 0.008, xerr=gs["sd"], fmt="s", color="#009E73", capsize=3)
    ax.set_xlabel(r"$E_G$")
    ax.set_ylabel("Effective redshift")
    ax.set_yticks([0.38, 0.61])
    ax.grid(alpha=0.2)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#0072B2", lw=0, label="No-Planck measurement"),
            Line2D([0], [0], marker="s", color="#009E73", lw=0, label="Amplitude-independent GR"),
        ], frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUT / "eg_gr_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5))
    for ax, (label, residual) in zip(axes, residual_sets):
        ax.axhspan(-2, 2, color="#009E73", alpha=0.08)
        ax.axhline(0, color="black", lw=0.8)
        ax.scatter(np.arange(len(residual)), residual, s=11, color="#0072B2")
        ax.set_ylabel(label)
        ax.set_ylim(-4.2, 4.2)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Whitened data-vector index")
    fig.tight_layout()
    fig.savefig(OUT / "bestfit_whitened_residuals.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote guarded explicit-zstar No-Prior diagnostics to {OUT}")


if __name__ == "__main__":
    main()
