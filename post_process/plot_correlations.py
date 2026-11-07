from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/prior_sensitivity"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp


RUNS = {
    "no_planck": ("No-Prior", "chain_explicit_zstar_no_planck_nautilus.txt", "pipeline2x2pt_weyl_explicit_zstar_no_planck.ini"),
    "with_planck": ("With-Prior", "chain_explicit_zstar_with_planck_nautilus.txt", "pipeline2x2pt_weyl_explicit_zstar_with_planck.ini"),
    "fixed_as": ("Fixed As", "chain_explicit_zstar_fixed_as_nautilus.txt", "pipeline2x2pt_weyl_explicit_zstar_fixed_as.ini"),
    "shape_priors": ("Free-S8", "chain_explicit_zstar_shape_priors_nautilus.txt", "pipeline2x2pt_weyl_explicit_zstar_free_s8.ini"),
}
COLS = {
    "S8": "cosmological_parameters--s_8",
    "As_1e9": "cosmological_parameters--a_s",
    "ns": "cosmological_parameters--n_s",
    "h": "cosmological_parameters--h0",
    "Omega_m": "cosmological_parameters--omega_m",
    "omch2": "cosmological_parameters--omch2",
    "ombh2": "cosmological_parameters--ombh2",
    "bhat1": "growth_amplitudes--bhat_bin_1",
    "bhat2": "growth_amplitudes--bhat_bin_2",
    "Jhat1": "j_bin_bias--jhat1",
    "Jhat2": "j_bin_bias--jhat2",
    "fhat1": "growth_amplitudes--fhat_bin_1",
    "fhat2": "growth_amplitudes--fhat_bin_2",
    "EG1": "reparameterized_derived--eg_bin_1",
    "EG2": "reparameterized_derived--eg_bin_2",
}
OBS = ["bhat1", "bhat2", "Jhat1", "Jhat2", "fhat1", "fhat2", "EG1", "EG2"]
COSMO = ["S8", "As_1e9", "ns", "h", "Omega_m", "omch2", "ombh2"]
LABELS = {"S8": r"$S_8$", "As_1e9": r"$10^9 A_s$", "ns": r"$n_s$", "h": r"$h$", "Omega_m": r"$\Omega_m$"}
LABELS.update({f"{p}{i}": rf"$\hat{{{p[:-3]}}}_{i}$" for p in ["bhat", "Jhat", "fhat"] for i in [1, 2]})
LABELS.update({f"EG{i}": rf"$E_G(z_{i})$" for i in [1, 2]})


def normalize(name):
    return name.lower().replace("hatted_derived--", "reparameterized_derived--")


def metadata(path):
    found = {}
    if not path.exists():
        return found
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read().decode("utf-8", errors="replace")
    for line in tail.splitlines():
        if line.startswith("#") and not line.startswith("##") and "=" in line:
            key, value = line[1:].split("=", 1)
            found[key.strip()] = value.strip()
    return found


@dataclass
class Chain:
    label: str
    names: list[str]
    data: np.ndarray
    weight: np.ndarray
    meta: dict
    nvar: int

    def raw(self, name):
        return self.data[:, self.names.index(normalize(name))]

    def get(self, key):
        values = self.raw(COLS.get(key, key))
        return values * 1e9 if key == "As_1e9" else values

    @property
    def ess(self):
        return float(1 / (self.weight @ self.weight))


def load(case):
    label, filename, _ = RUNS[case]
    path = ROOT / "chain" / filename
    with path.open() as stream:
        names = [normalize(n) for n in stream.readline().lstrip("#").split()]
        nvar = None
        for line in stream:
            if line.startswith("#n_varied="):
                nvar = int(line.split("=", 1)[1])
                break
    data = np.loadtxt(path, comments="#", ndmin=2)
    good = np.isfinite(data[:, names.index("log_weight")]) & np.isfinite(data[:, names.index("post")])
    data = data[good]
    weights = data[:, names.index("log_weight")]
    weights = np.exp(weights - logsumexp(weights))
    return Chain(label, names, data, weights, metadata(path), nvar)


def stats(x, w):
    mean = float(w @ x)
    sd = float(np.sqrt(w @ (x - mean) ** 2))
    order = np.argsort(x)
    quantiles = np.interp([0.025, 0.16, 0.5, 0.84, 0.975], np.cumsum(w[order]), x[order])
    return dict(zip(["mean", "sd", "q025", "q16", "median", "q84", "q975"], [mean, sd, *map(float, quantiles)]))


def correlation(x, y, w):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    x, y = x - w @ x, y - w @ y
    denominator = np.sqrt((w @ x**2) * (w @ y**2))
    return float(w @ (x * y) / denominator) if denominator > 1e-20 else None


def weighted_rank(x, w):
    _, inverse = np.unique(x, return_inverse=True)
    mass = np.bincount(inverse, weights=w)
    return (np.cumsum(mass) - 0.5 * mass)[inverse]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figures(chains):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11})
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.8))
    for row, key in enumerate(["S8", "As_1e9", "ns"]):
        for column, target in enumerate(OBS[:4]):
            ax = axes[row, column]
            xr = [stats(c.get(key), c.weight) for c in chains.values()]
            yr = [stats(c.get(target), c.weight) for c in chains.values()]
            xlim = (min(s["q025"] for s in xr), max(s["q975"] for s in xr))
            ylim = (min(s["q025"] for s in yr), max(s["q975"] for s in yr))
            xpad, ypad = 0.28 * np.ptp(xlim), 0.28 * np.ptp(ylim)
            bounds = [(xlim[0] - xpad, xlim[1] + xpad), (ylim[0] - ypad, ylim[1] + ypad)]
            for case, color in [("no_planck", "#777777"), ("with_planck", "#006eb8")]:
                c = chains[case]
                hist, xe, ye = np.histogram2d(c.get(key), c.get(target), bins=65, range=bounds, weights=c.weight)
                density = gaussian_filter(hist, 1.05)
                flattened = np.sort(density.ravel())[::-1]
                cumulative = np.cumsum(flattened) / flattened.sum()
                levels = np.unique(np.interp([0.95, 0.68], cumulative, flattened))
                ax.contour((xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2, density.T, levels=levels, colors=color, linewidths=1.3)
            ax.set_xlabel(LABELS[key])
            ax.set_ylabel(LABELS[target])
            ax.tick_params(direction="in", top=True, right=True)
    fig.legend(handles=[Line2D([0], [0], color="#777777", label="No-Prior"), Line2D([0], [0], color="#006eb8", label="With-Prior")], loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "parameter_correlations.png", dpi=180)
    fig.savefig(OUT / "parameter_correlations.pdf")
    plt.close(fig)
    labels = {"no_planck": "No-Prior", "with_planck": "With-Prior", "fixed_as": "Fixed As", "shape_priors": r"Free-$S_8$"}
    colors = {"no_planck": "#777777", "with_planck": "#006eb8", "fixed_as": "#c44e52", "shape_priors": "#55a868"}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, target in zip(axes.flat, OBS):
        for position, (case, chain) in enumerate(chains.items()):
            s = stats(chain.get(target), chain.weight)
            ax.errorbar(s["median"], position, xerr=[[s["median"] - s["q16"]], [s["q84"] - s["median"]]], fmt="o", color=colors[case], capsize=3)
        ax.set_yticks(range(len(chains)), [labels[c] for c in chains], fontsize=8)
        ax.set_xlabel(LABELS[target])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "constraint_comparison.png", dpi=180)
    fig.savefig(OUT / "constraint_comparison.pdf")
    plt.close(fig)


def heatmap():
    with (OUT / "correlations.csv").open() as stream:
        values = {
            (row["case"], row["x"], row["y"]): float(row["pearson"])
            for row in csv.DictReader(stream)
            if row["pearson"]
        }
    rows = ["As_1e9", "S8", "ns"]
    columns = ["bhat1", "bhat2", "Jhat1", "Jhat2"]
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), layout="constrained")
    for ax, case, title in zip(axes, ["no_planck", "with_planck"], ["No-Prior", "With-Prior"]):
        matrix = np.array([[values[case, x, y] for y in columns] for x in rows])
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4), [r"$\hat b_1$", r"$\hat b_2$", r"$\hat J_1$", r"$\hat J_2$"])
        ax.set_yticks(range(3), [r"$A_s$", r"$S_8$", r"$n_s$"])
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="both", length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    fontsize=13, color="white" if abs(value) > 0.55 else "#222222")
    colorbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.035)
    colorbar.set_label(r"Pearson correlation", labelpad=10)
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.outline.set_visible(False)
    for suffix in ["png", "pdf"]:
        path = OUT / f"correlation_heatmap.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chains = {case: load(case) for case in RUNS}
    rows, correlations = [], []
    for case, chain in chains.items():
        if chain.meta.get("complete") != "1":
            raise ValueError(f"{case}: chain is incomplete")
        keys = list(COLS) + [n for n in chain.names[:chain.nvar] if n not in COLS.values()]
        for key in keys:
            rows.append({"case": case, "parameter": key, **stats(chain.get(key), chain.weight)})
        nuisance = [n for n in chain.names[:chain.nvar] if n.startswith(("bias_parameters--", "halo_model_parameters--", "intrinsic_alignment_parameters--", "nofz_shifts--"))]
        for key in COSMO + nuisance:
            x = chain.get(key)
            ranks = weighted_rank(x, chain.weight)
            for target in OBS:
                y = chain.get(target)
                correlations.append({
                    "case": case, "x": key, "y": target,
                    "pearson": correlation(x, y, chain.weight),
                    "spearman": correlation(ranks, weighted_rank(y, chain.weight), chain.weight),
                })
        print(f"{chain.label}: {len(chain.data)} samples, effective sample size {chain.ess:.0f}")
    write_csv(OUT / "posterior_summary.csv", rows)
    write_csv(OUT / "correlations.csv", correlations)
    figures(chains)
    heatmap()
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
