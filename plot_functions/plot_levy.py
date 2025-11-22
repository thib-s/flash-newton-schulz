#!/usr/bin/env python3
"""
Pareto evolution across Lévy distributions, one curve per (algorithm × distribution).

- One figure per matrix size.
- X: batch_time_ms (lower better), Y: polar_error (lower better).
- For each (algo, dist), the curve is the Pareto-efficient frontier over 'iter'.
- Colors encode algorithms; line styles encode distributions.
- No iteration annotations (as requested).
- Optional faint background points for context.

Usage:
  python pareto_evolution_by_alg_dist.py \
    --csv results_1/ \
    --paper --font-size 15 --line-width 5.0 --marker-size 10.0 \
    --legend-outside --pareto-logx
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from unify_plots import ALG_STYLE


# ------------------ Styling ------------------


def setup_paper_style(font="DejaVu Sans", base=11, lw=2.0, ms=6, grid=True):
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
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


# Distribution line styles (distinct, print-friendly)
DIST_STYLE: Dict[str, Dict[str, object]] = {
    "levy-1.0": {"linestyle": "solid"},
    "levy-1.5": {"linestyle": (0, (5, 2))},  # dashed
    "levy-2.0": {"linestyle": (0, (3, 1, 1, 1))},  # dash-dot-ish
}


# ------------------ Helpers ------------------


def parse_filters(kvs: Optional[Iterable[str]]) -> Dict[str, object]:
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


def normalize_levy_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 'levy' + levy_alpha -> 'levy-{alpha}' for consistency."""
    if "dist" in df.columns and "levy_alpha" in df.columns:
        mask = df["dist"] == "levy"
        df.loc[mask, "dist"] = df.loc[mask, "levy_alpha"].apply(lambda a: f"levy-{a}")
    return df


def aggregate_for_size(
    df: pd.DataFrame, runtime_col: str, error_col: str
) -> pd.DataFrame:
    """
    At a fixed size: median per (dist, algo, iter) for runtime & error.
    Returns columns: dist, algo, iter, runtime_col, error_col
    """
    group_keys = ["dist", "algo", "iter"]

    # Only aggregate columns that are not group keys to avoid duplicate names
    value_cols = [c for c in [runtime_col, error_col] if c not in group_keys]

    # Keep only needed columns; group keys are always kept
    keep = list(dict.fromkeys(group_keys + value_cols))  # dedupe while preserving order

    g = (
        df[keep]
        .groupby(group_keys, as_index=False)[value_cols]
        .median()
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    # If runtime_col was a group key (e.g., 'iter'), make sure it exists explicitly
    # (it will, via group_keys), so nothing to add. If it wasn't, it already exists via value_cols.
    return g


def compute_pareto_front_minmin(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """
    Pareto frontier for minimizing both x and y.
    Sort by x asc, keep strictly decreasing sequence in y.
    """
    if df.empty:
        return df
    d = df.sort_values([x, y], ascending=[True, True]).reset_index(drop=True)
    best_y = np.inf
    rows: List[int] = []
    for i, v in d[y].items():
        if v < best_y:
            rows.append(i)
            best_y = v
    return d.loc[rows].reset_index(drop=True)


# ------------------ Plotting ------------------


def plot_for_size(
    agg_s: pd.DataFrame,
    runtime_col: str,
    error_col: str,
    dists: List[str],
    legend_outside: bool,
    logx: bool,
    logy: bool,
    fig_size: Tuple[float, float],
    title: Optional[str],
    show_all_points: bool = True,
):
    """
    Draw one curve per (algo × dist):
      - color/marker by algo
      - line style by dist
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # background points (faint) to show the cloud for each (algo, dist)
    if show_all_points:
        for (algo, dist), g in agg_s.groupby(["algo", "dist"]):
            if algo not in ALG_STYLE or dist not in DIST_STYLE:
                continue
            ax.scatter(
                g[runtime_col].values,
                g[error_col].values,
                s=18,
                alpha=0.20,
                color=ALG_STYLE[algo]["color"],
                edgecolor="none",
                zorder=1,
            )

    # pareto fronts per (algo, dist)
    for (algo, dist), g in agg_s.groupby(["algo", "dist"]):
        if algo not in ALG_STYLE or dist not in DIST_STYLE:
            # skip unknown styles (enforce explicit mapping)
            continue
        front = compute_pareto_front_minmin(g, runtime_col, error_col)
        if front.empty:
            continue

        style_alg = ALG_STYLE[algo]
        style_dist = DIST_STYLE[dist]

        ax.plot(
            front[runtime_col].values,
            front[error_col].values,
            color=style_alg["color"],
            marker=style_alg["marker"],
            linestyle=style_dist["linestyle"],
            linewidth=3.0,
            markersize=6.0,
            label=f"{algo} — {dist}",
            zorder=3,
        )

    # ax.set_xlabel("time (ms)")
    # in plot_for_size(...)
    xlab = (
        "time (ms)" if runtime_col == "batch_time_ms" else runtime_col.replace("_", " ")
    )
    ax.set_xlabel(xlab)
    # set tick size to 1 if runtime_col is 'iter' (discrete)
    if runtime_col == "iter":
        ax.xaxis.get_major_locator().set_params(integer=True)

    ax.set_ylabel(error_col.replace("_", " "))
    ax.set_ylim(bottom=0, top=1.0)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    # Dual-legend: colors for algos, linestyles for dists
    # 1) algo legend (colors/markers)
    algo_handles = [
        Line2D(
            [0],
            [0],
            color=ALG_STYLE[a]["color"],
            marker=ALG_STYLE[a]["marker"],
            linestyle="solid",
            linewidth=2.5,
            markersize=6,
            label=a,
        )
        for a in ALG_STYLE.keys()
    ]
    # 2) dist legend (linestyles)
    dist_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=DIST_STYLE[d]["linestyle"],
            linewidth=2.5,
            label=d,
        )
        for d in dists
        if d in DIST_STYLE
    ]

    if legend_outside:
        leg1 = ax.legend(
            handles=algo_handles,
            title="algorithm",
            bbox_to_anchor=(1.01, 1.0),
            loc="upper left",
            borderaxespad=0.0,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=dist_handles,
            title="distribution",
            bbox_to_anchor=(1.01, 0.50),
            loc="upper left",
            borderaxespad=0.0,
        )
    else:
        leg1 = ax.legend(handles=algo_handles, title="algorithm", loc="upper right")
        ax.add_artist(leg1)
        ax.legend(handles=dist_handles, title="distribution", loc="lower right")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


# ------------------ Main ------------------


def main():
    p = argparse.ArgumentParser(
        description="Pareto fronts per (algorithm × distribution) across Lévy distributions, one plot per size."
    )
    p.add_argument(
        "--csv", required=True, nargs="+", help="CSV file(s) or directory(ies)"
    )
    p.add_argument("--outroot", default="plots_out", help="Root output directory")
    p.add_argument(
        "--filters",
        nargs="*",
        default=[],
        help="Pre-filters key=value (e.g., dtype='float32')",
    )

    # styling
    p.add_argument("--paper", action="store_true")
    p.add_argument("--font", default="DejaVu Sans")
    p.add_argument("--font-size", type=int, default=11)
    p.add_argument("--line-width", type=float, default=2.0)
    p.add_argument("--marker-size", type=float, default=6.0)
    p.add_argument("--legend-outside", action="store_true")

    # axes/log & output
    p.add_argument("--pareto-logx", action="store_true")
    p.add_argument("--pareto-logy", action="store_true")
    p.add_argument("--figwidth", type=float, default=9.0)
    p.add_argument("--figheight", type=float, default=6.0)
    p.add_argument("--out-ext", default="png", choices=["png", "pdf", "svg"])

    # columns
    p.add_argument("--runtime-col", default="batch_time_ms")
    p.add_argument("--error-col", default="polar_error")

    # which distributions to compare
    p.add_argument(
        "--dists",
        nargs="+",
        default=["levy-1.0", "levy-1.5", "levy-2.0"],
        help="Must match values in 'dist' after normalization",
    )

    args = p.parse_args()

    # read CSVs/dirs
    dfs: List[pd.DataFrame] = []
    for cf in args.csv:
        pth = Path(cf)
        if pth.is_file():
            dfs.append(pd.read_csv(pth))
        elif pth.is_dir():
            cs = list(pth.glob("*.csv"))
            if not cs:
                raise SystemExit(f"No CSVs in directory: {pth}")
            for c in cs:
                dfs.append(pd.read_csv(c))
        else:
            raise SystemExit(f"CSV path not found: {pth}")
    df = pd.concat(dfs, ignore_index=True)

    df = normalize_levy_dist(df)
    ensure_cols(df, ["algo", "iter", "size", "dist", args.runtime_col, args.error_col])

    # filters
    filters = parse_filters(args.filters)
    if filters:
        df = apply_filters(df, filters)
        if df.empty:
            raise SystemExit("No rows after applying --filters.")

    if args.paper:
        setup_paper_style(
            font=args.font, base=args.font_size, lw=args.line_width, ms=args.marker_size
        )

    # check requested dists exist
    avail_dists = df["dist"].dropna().unique().tolist()
    dists = [d for d in args.dists if d in avail_dists]
    if not dists:
        raise SystemExit(
            f"No requested dists present. Requested={args.dists} Available={sorted(avail_dists)}"
        )

    outdir = Path(args.outroot) / "pareto_evolution_alg_dist"
    outdir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(df["size"].dropna().unique().tolist())
    if not sizes:
        raise SystemExit("No size values found.")

    for s in sizes:
        sub = df[df["size"] == s].copy()
        if sub.empty:
            continue

        # aggregate per (dist, algo, iter) for this size
        agg_s = aggregate_for_size(sub, args.runtime_col, args.error_col)
        agg_s = agg_s[agg_s["dist"].isin(dists)]
        if agg_s.empty:
            continue

        title = f"Pareto fronts per algorithm × distribution — size {s}"
        fig, ax = plot_for_size(
            agg_s=agg_s,
            runtime_col=args.runtime_col,
            error_col=args.error_col,
            dists=dists,
            legend_outside=args.legend_outside,
            logx=args.pareto_logx,
            logy=args.pareto_logy,
            fig_size=(args.figwidth, args.figheight),
            title=title,
            show_all_points=True,
        )

        out_path = outdir / f"pareto_alg_dist_size{s}.{args.out_ext}"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    print(f"[done] Wrote curves to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
