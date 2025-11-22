#!/usr/bin/env python3
"""
Unified plotting script (styled, centralized colors/markers):
- Generates two independent plots per (dist, dtype):
    1) speedup bars vs matrix size (baseline configurable)
    2) polar error vs matrix size (selected (algo, iter) pairs)
- Generates Pareto plots (polar_error vs batch_time_ms) per available matrix size.

Directory layout (per CSV run):
  outroot/
    {dist}/
      {dtype}/
        runtime_filtered_annotated.png      # speedup bars
        polar_error_filtered_annotated.png  # polar error lines
        pareto/
          pareto_size{SIZE}.<ext>

Notes
- Pareto plots are generated per size (filter size == SIZE).
- If your CSV contains multiple distributions or dtypes, the script iterates over
  each (dist, dtype) pair and produces the corresponding folder structure.
- The two "dual" plots are treated as fully independent plots.
- Colors/markers/linestyles are now controlled centrally via ALG_STYLE. No fallbacks.

Typical usage:
  python unify_plots_clean_styled.py \
    --csv results_1/ \
    --paper --line-width 5.0 --marker-size 10.0 --font-size 15 \
    --legend-outside
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, Iterable as It

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, MaxNLocator

# ------------------ Configuration ------------------
# Selected (algo, iter) to show on the two per-size plots
WANTED: Tuple[Tuple[str, int], ...] = (
    ("Muon", 5),
    ("Dion", 5),
    ("AOLxDion", 5),
    ("AOLxDion", 4),
)

# ------------------ Centralized style registry (NO FALLBACKS) ------------------
# Edit this dict to change colors/markers/linestyles consistently across all plots.
# Keys must match the 'algo' values in your CSV exactly.
ALG_STYLE: Dict[str, Dict[str, str]] = {
    "Muon": {"color": "#009E73", "marker": "o", "linestyle": "solid"},  # green
    "Dion": {"color": "#0072B2", "marker": "s", "linestyle": "solid"},  # blue
    "AOLxDion": {"color": "#56B4E9", "marker": "D", "linestyle": "solid"},  # sky blue
}


def style_for_algo(
    name: str, *, iter_value: Optional[int] = None
) -> Tuple[str, str, str]:
    """Return (color, marker, linestyle) for an algorithm strictly from ALG_STYLE.
    Raises KeyError if the algorithm is not present (no fallbacks by design).
    """
    if name not in ALG_STYLE:
        raise KeyError(
            f"Algorithm '{name}' not found in ALG_STYLE. Add it there to set color/marker/linestyle."
        )
    spec = ALG_STYLE[name]
    return spec["color"], spec["marker"], spec["linestyle"]


# ------------------ Shared helpers ------------------


def parse_filters(kvs: Optional[It[str]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"Bad --filters entry '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        v = v.strip()
        try:
            v_cast = ast.literal_eval(v)
        except Exception:
            v_cast = v
        out[k.strip()] = v_cast
    return out


def apply_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    for k, v in filters.items():
        if k not in df.columns:
            raise KeyError(f"Filter key '{k}' not found in columns: {list(df.columns)}")
        df = df[df[k] == v]
    return df


def ensure_cols(df: pd.DataFrame, needed: Iterable[str]):
    missing = set(needed) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")


def setup_paper_style(font="DejaVu Sans", base=11, lw=2.0, ms=6, grid=True):
    """Reasonable defaults for print/PDF."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # keep text as text in PDF
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [font],
            "font.size": base,
            "axes.titlesize": base + 1,
            "axes.labelsize": base,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "axes.linewidth": 0.96,
            "xtick.major.width": 0.96,
            "ytick.major.width": 0.96,
            "xtick.minor.width": 0.72,
            "ytick.minor.width": 0.72,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": grid,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )


def marker_cycle():
    return ["o", "s", "D", "^", "v", "P", "X", "h", "*"]


def linestyle_cycle():
    # Helps in B/W printouts
    return [
        "solid",
        (0, (4, 2)),
        "dashed",
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        "dashdot",
        (0, (5, 2, 1, 2)),
    ]


def apply_axis_format(ax, logx=False, logy=False):
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(LogFormatter(10))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(fmt)

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatter(10))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(fmt)

    ax.tick_params(which="both", direction="out", length=4)
    ax.minorticks_on()


# ------------------ Dual plots (now independent) ------------------


def dual_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to mean/std per (algo, iter, size) and keep only WANTED pairs."""
    mask = False
    for algo, it in WANTED:
        mask |= (df["algo"] == algo) & (df["iter"] == it)
    df = df[mask].copy()
    agg = df.groupby(["algo", "iter", "size"], as_index=False).agg(
        mean_runtime_ms=("batch_time_ms", "mean"),
        std_runtime_ms=("batch_time_ms", "std"),
        mean_polar_error=("polar_error", "mean"),
        std_polar_error=("polar_error", "std"),
    )
    return agg


def plot_polar_error_lines(ax, agg: pd.DataFrame):
    """
    Polar error vs size (lines) with inline labels:
    - No legend
    - Each curve is labeled next to its rightmost point
    """
    rightmost_x = []
    for (algo, it), sub in agg.groupby(["algo", "iter"]):
        label = f"{algo}\n(iter={it})"
        sub = sub.sort_values("size")
        x = sub["size"].to_numpy(dtype=float)
        y = sub["mean_polar_error"].to_numpy(dtype=float)
        yerr = sub["std_polar_error"].fillna(0).to_numpy(dtype=float)

        c, m, linestyle = style_for_algo(algo, iter_value=it)
        # exception for AOLxDion iter=4 to distinguish from iter=5
        if algo == "AOLxDion" and it == 4:
            linestyle = "dotted"

        ax.plot(x, y, marker=m, label=label, color=c, linestyle=linestyle)
        if np.any(yerr > 0):
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=c)

        # inline label at rightmost point
        xr, yr = x[-1], y[-1]
        rightmost_x.append(xr)
        ax.text(
            xr * 1.25,  # small data-space padding to the right
            yr,
            label,
            va="center",
            ha="left",
            fontsize=11,
            color=c,
        )

    # axes & formatting
    ax.set_xlabel("matrix size")
    ax.set_ylabel("polar error (mean)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(LogFormatter(2))
    ax.set_ylim(bottom=0.0)
    # give a bit of room on the right for labels
    if rightmost_x:
        ax.set_xlim(right=max(rightmost_x) * 1.15)
    ax.grid(True, which="both", ls="--", alpha=0.25)
    # no legend


def plot_speedup_bars(
    ax,
    agg: pd.DataFrame,
    baseline_algo: str,
    baseline_iter: int,
    max_sizes: int = 4,
):
    """
    Horizontal grouped bars only:
    - Matrix size on Y axis (group per size)
    - Speedup on X axis
    - Each bar labeled with algorithm name and separate numeric value (e.g., 'AOLxDion (iter=4)' inside, '2.3x' outside)
    - No legend, no vertical bars
    """
    agg_display = agg.copy()
    sizes_all = sorted(agg_display["size"].unique().tolist(), reverse=True)
    if max_sizes > 0:
        sizes = sizes_all[:max_sizes]
        agg_display = agg_display[agg_display["size"].isin(sizes)]
    else:
        sizes = sizes_all

    base = (
        agg_display[
            (agg_display["algo"] == baseline_algo)
            & (agg_display["iter"] == baseline_iter)
        ]
        .set_index("size")["mean_runtime_ms"]
        .rename("baseline_runtime")
    )
    if base.empty:
        raise SystemExit(
            f"Baseline {baseline_algo} (iter={baseline_iter}) not present after filtering."
        )

    dfp = agg_display.set_index("size").join(base, how="inner").reset_index().copy()
    dfp["speedup"] = dfp["baseline_runtime"] / dfp["mean_runtime_ms"]

    groups = [p for p in WANTED[::-1]]
    n = len(groups)

    y = np.arange(len(sizes), dtype=float)
    height = min(0.8 / max(n, 1), 0.22)
    offsets = (np.arange(n) - (n - 1) / 2) * (height + 0.01)

    hatch_map = {("AOLxDion", 4): "///"}
    global_max = 0.0

    for i, (algo, it) in enumerate(groups):
        # ensure style exists (no fallbacks)
        _ = style_for_algo(algo, iter_value=it)
        sub = dfp[(dfp["algo"] == algo) & (dfp["iter"] == it)]
        vals = [
            (
                sub[sub["size"] == s]["speedup"].values[0]
                if s in sub["size"].values
                else np.nan
            )
            for s in sizes
        ]

        c, _m, _ls = style_for_algo(
            algo, iter_value=it
        )  # marker/linestyle unused in bars
        hatch = hatch_map.get((algo, it), None)

        bar_y = y + offsets[i]
        bars = ax.barh(
            bar_y,
            vals,
            height=height,
            color=c,
            linewidth=0.0,
            hatch=hatch,
            edgecolor="white",
        )

        label_text = f"{algo} (iter={it})"
        for j, (rect, val) in enumerate(zip(bars, vals)):
            if np.isnan(val):
                continue
            global_max = max(global_max, float(val))
            y_pos = rect.get_y() + rect.get_height() / 2
            x_pos = float(val)

            # algorithm label inside or left of the bar
            ax.text(
                0.01 * max(1.0, global_max),
                y_pos,
                label_text,
                va="center",
                ha="left",
                fontsize=11,
                color="black",
            )

            # numeric value to the right of the bar
            pad = 0.02 * max(1.0, global_max)
            ax.text(
                x_pos + pad,
                y_pos,
                f"{val:.1f}x",
                va="center",
                ha="left",
                fontsize=11,
                color="black",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [str(int(s)) if float(s).is_integer() else str(s) for s in sizes]
    )
    ax.set_yticklabels(ax.get_yticklabels(), va="center")
    ax.tick_params(axis="y", rotation=90)
    ax.set_xlabel(f"speedup vs {baseline_algo} (iter={baseline_iter})")
    ax.set_ylabel("matrix size")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(left=0.0, right=max(1.0, global_max * 1.25))


def make_dual_plots(
    df: pd.DataFrame, out_dir: Path, baseline_algo: str, baseline_iter: int
):
    out_dir.mkdir(parents=True, exist_ok=True)

    df2 = df.dropna(subset=["polar_error"], how="any")
    agg = dual_aggregate(df2)

    # (1) Speedup bars
    fig1, ax1 = plt.subplots()
    plot_speedup_bars(ax1, agg, baseline_algo, baseline_iter)
    fig1.tight_layout()
    p1 = out_dir / "runtime_filtered_annotated.png"
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)

    # (2) Polar error lines
    fig2, ax2 = plt.subplots()
    plot_polar_error_lines(ax2, agg)
    fig2.tight_layout()
    p2 = out_dir / "polar_error_filtered_annotated.png"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)


# ------------------ Pareto plots (error vs runtime per size) ------------------


def pareto_aggregate(
    df: pd.DataFrame,
    hue_col: str,
    runtime_col: str,
    error_col: str,
    runtime_std_col: Optional[str] = None,
) -> pd.DataFrame:
    agg_cols = [runtime_col, error_col]

    agg = (
        df.groupby([hue_col, "iter"], as_index=False)[agg_cols]
        .median()
        .sort_values([hue_col, "iter"])
    )

    ystd = (
        df.groupby([hue_col, "iter"], as_index=False)[error_col]
        .std()
        .rename(columns={error_col: f"{error_col}_std"})
    )
    agg = pd.merge(agg, ystd, on=[hue_col, "iter"], how="left")

    if runtime_std_col and runtime_std_col in df.columns:
        xstd = (
            df.groupby([hue_col, "iter"], as_index=False)[runtime_std_col]
            .median()
            .rename(columns={runtime_std_col: f"{runtime_col}_std"})
        )
        agg = pd.merge(agg, xstd, on=[hue_col, "iter"], how="left")

    return agg


def make_pareto_per_size(
    df: pd.DataFrame,
    out_dir: Path,
    hue_col: str = "algo",
    runtime_col: str = "batch_time_ms",
    error_col: str = "polar_error",
    logx: bool = False,
    logy: bool = False,
    fig_size=(9.0, 6.0),
    out_ext: str = "png",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(df["size"].dropna().unique().tolist())
    if not sizes:
        return

    runtime_std_col_guess = f"{runtime_col}_std"

    for s in sizes:
        sub = df[df["size"] == s].copy()
        if sub.empty:
            continue

        # Determine a representative batch_size: number of line for given group of
        # dist x dtype x size x algo x iter. Take the most common value as there are multiple groups
        # in each pareto plot.
        batch_size_val = (
            sub.groupby(["dist", "dtype", "size", hue_col, "iter"]).size().mode()
        ).values[0]

        agg = pareto_aggregate(
            sub,
            hue_col,
            runtime_col,
            error_col,
            runtime_std_col=(
                runtime_std_col_guess if runtime_std_col_guess in sub.columns else None
            ),
        )
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=fig_size)
        has_xerr = f"{runtime_col}_std" in agg.columns
        has_yerr = f"{error_col}_std" in agg.columns

        for h, g in agg.groupby(hue_col, sort=False):
            # enforce style presence
            _ = style_for_algo(str(h))
            g = g.sort_values(runtime_col)
            c, m, ls = style_for_algo(str(h))

            xs = g[runtime_col].values
            ys = g[error_col].values
            iters = g["iter"].values

            ax.plot(
                xs,
                ys,
                marker=m,
                label=str(h),
                color=c,
                linewidth=3.0,
                linestyle=ls,
            )
            ax.scatter(xs, ys, edgecolor="none", zorder=3, s=60, color=c)

            # Add small iteration labels next to each point
            for x_val, y_val, itv in zip(xs, ys, iters):
                ax.annotate(
                    str(int(itv)) + " it",
                    xy=(x_val, y_val),
                    xytext=(6, 6),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color=c,
                )

            if has_xerr or has_yerr:
                xerr = g[f"{runtime_col}_std"].values if has_xerr else None
                yerr = g[f"{error_col}_std"].values if has_yerr else None
                ax.errorbar(
                    xs,
                    ys,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="none",
                    ecolor=c,
                    elinewidth=1.2,
                    capsize=3,
                    alpha=0.8,
                )

        # X label with batch_size and current size if batch_size known
        if batch_size_val is not None:
            ax.set_xlabel(f"time (ms): {batch_size_val} matrices ({s} x {s})")
        else:
            ax.set_xlabel("time (ms)")
        ax.set_ylabel(error_col.replace("_", " "))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(title=hue_col)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        plt.tight_layout()
        out_path = out_dir / f"pareto_size{s}.{out_ext}"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


# ------------------ Orchestration ------------------


def infer_groups(df: pd.DataFrame) -> Iterable[Tuple[object, object]]:
    have_dist = "dist" in df.columns
    have_dtype = "dtype" in df.columns
    if not have_dist and not have_dtype:
        yield ("unknown_dist", "unknown_dtype")
        return
    if have_dist and have_dtype:
        groups = (
            df.groupby(["dist", "dtype"])
            .size()
            .reset_index()[["dist", "dtype"]]
            .itertuples(index=False, name=None)
        )
    elif have_dist:
        groups = [(d, "unknown_dtype") for d in df["dist"].unique()]
    else:
        groups = [("unknown_dist", t) for t in df["dtype"].unique()]
    for g in groups:
        yield g


def run_for_group(
    df: pd.DataFrame,
    outroot: Path,
    dist: object,
    dtype: object,
    paper: bool,
    font: str,
    font_size: int,
    line_width: float,
    marker_size: float,
    # pareto controls
    pareto_hue_col: str,
    pareto_runtime_col: str,
    pareto_error_col: str,
    pareto_logx: bool,
    pareto_logy: bool,
    pareto_fig_w: float,
    pareto_fig_h: float,
    pareto_out_ext: str,
    # speedup baseline
    baseline_algo: str,
    baseline_iter: int,
):
    if paper:
        setup_paper_style(font=font, base=font_size, lw=line_width, ms=marker_size)

    sub = df.copy()
    if "dist" in sub.columns:
        sub = sub[sub["dist"] == dist]
    if "dtype" in sub.columns:
        sub = sub[sub["dtype"] == dtype]

    base_dir = outroot / str(dist) / str(dtype)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Independent plots (speedup + polar error)
    make_dual_plots(
        sub, base_dir, baseline_algo=baseline_algo, baseline_iter=baseline_iter
    )

    # Pareto per size
    pareto_dir = base_dir / "pareto"
    make_pareto_per_size(
        df=sub,
        out_dir=pareto_dir,
        hue_col=pareto_hue_col,
        runtime_col=pareto_runtime_col,
        error_col=pareto_error_col,
        logx=pareto_logx,
        logy=pareto_logy,
        fig_size=(pareto_fig_w, pareto_fig_h),
        out_ext=pareto_out_ext,
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Make speedup bars + polar-error lines (independent) and per-size Pareto plots, organized by dist/dtype."
        )
    )
    p.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="CSV file(s) or directory(ies) containing CSV files",
    )
    p.add_argument("--outroot", default="plots_out", help="Root output directory")

    # Optional pre-filters applied before grouping
    p.add_argument(
        "--filters",
        nargs="*",
        default=[],
        help="Key=Value filters to preselect rows (e.g., device='cuda:0 (NVIDIA L40S)' dim='(128, 128)' iter=5)",
    )

    # Styling
    p.add_argument("--paper", action="store_true", help="Enable paper-ready styling")
    p.add_argument("--font", default="DejaVu Sans", help="Font family")
    p.add_argument("--font-size", type=int, default=11, help="Base font size")
    p.add_argument("--line-width", type=float, default=2.0, help="Line width")
    p.add_argument("--marker-size", type=float, default=6.0, help="Marker size")
    # Kept for compatibility (ignored)
    p.add_argument(
        "--legend-outside",
        action="store_true",
        help="(Accepted but unused; kept for compatibility)",
    )

    # Pareto-specific knobs
    p.add_argument(
        "--pareto-hue-col", default="algo", help="Hue column (curve per value)"
    )
    p.add_argument(
        "--pareto-runtime-col",
        default="batch_time_ms",
        help="Runtime column for X axis",
    )
    p.add_argument(
        "--pareto-error-col", default="polar_error", help="Error column for Y axis"
    )
    p.add_argument(
        "--pareto-logx", action="store_true", help="Log scale on X for Pareto"
    )
    p.add_argument(
        "--pareto-logy", action="store_true", help="Log scale on Y for Pareto"
    )
    p.add_argument(
        "--pareto-figwidth",
        type=float,
        default=9.0,
        help="Pareto figure width (inches)",
    )
    p.add_argument(
        "--pareto-figheight",
        type=float,
        default=6.0,
        help="Pareto figure height (inches)",
    )
    p.add_argument(
        "--pareto-out-ext",
        default="png",
        choices=["png", "pdf", "svg"],
        help="File extension/format for Pareto plots",
    )

    # Speedup baseline
    p.add_argument(
        "--baseline-algo", default="Muon", help="Baseline algo for speedup bars"
    )
    p.add_argument(
        "--baseline-iter", type=int, default=5, help="Baseline iter for speedup bars"
    )

    args = p.parse_args()

    # Ingest CSV(s)
    dfs = []
    for cf in args.csv:
        pth = Path(cf)
        if pth.is_file():
            try:
                dfi = pd.read_csv(pth)
            except Exception as e:
                raise SystemExit(f"Error reading CSV '{pth}': {e}")
            dfs.append(dfi)
        elif pth.is_dir():
            csvs = list(pth.glob("*.csv"))
            if not csvs:
                raise SystemExit(f"No CSV files found in directory: {pth}")
            for c2 in csvs:
                try:
                    dfi = pd.read_csv(c2)
                except Exception as e:
                    raise SystemExit(f"Error reading CSV '{c2}': {e}")
                dfs.append(dfi)
        else:
            raise SystemExit(f"CSV path not found: {pth}")

    df = pd.concat(dfs, ignore_index=True)

    # Levy rename: dist=="levy" -> dist==f"levy-{alpha}" if levy_alpha exists
    if "dist" in df.columns and "levy_alpha" in df.columns:
        mask = df["dist"] == "levy"
        df.loc[mask, "dist"] = df.loc[mask, "levy_alpha"].apply(lambda a: f"levy-{a}")

    # Sanity columns
    ensure_cols(df, ["algo", "iter", "size", "batch_time_ms", "polar_error"])

    # Optional pre-filters (e.g., single device, single dim, etc.)
    filters = parse_filters(args.filters)
    if filters:
        df = apply_filters(df, filters)
        if df.empty:
            raise SystemExit("No rows after applying --filters.")

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    for dist, dtype in infer_groups(df):
        run_for_group(
            df=df,
            outroot=outroot,
            dist=dist,
            dtype=dtype,
            paper=args.paper,
            font=args.font,
            font_size=args.font_size,
            line_width=args.line_width,
            marker_size=args.marker_size,
            pareto_hue_col=args.pareto_hue_col,
            pareto_runtime_col=args.pareto_runtime_col,
            pareto_error_col=args.pareto_error_col,
            pareto_logx=args.pareto_logx,
            pareto_logy=args.pareto_logy,
            pareto_fig_w=args.pareto_figwidth,
            pareto_fig_h=args.pareto_figheight,
            pareto_out_ext=args.pareto_out_ext,
            baseline_algo=args.baseline_algo,
            baseline_iter=args.baseline_iter,
        )


if __name__ == "__main__":
    main()
