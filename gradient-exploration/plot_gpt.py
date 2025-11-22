import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from ns_variants.flash_ns import (
    newton_schulz_torch,
    newton_schulz_triton_muon_plus,  # available if you enable it below
    newton_schulz_triton_aol,
)
from ns_variants import polar_express, reference_ns

# --- config ---
ns_impls = [
    "aol",
    "muon",
    # "muon+",  # uncomment if you have grads/muon+ and want it included
    "std_pe",
    # "aol_pe",
]
ns_impl = {
    "aol": newton_schulz_triton_aol,
    # "muon+": newton_schulz_triton_muon_plus,
    "muon": newton_schulz_torch,
    "std_pe": partial(polar_express, aol=False),
    # "aol_pe": partial(polar_express, aol=True),
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- compute errors per step and impl ---
rows = []

for impl in ns_impls:
    dirpath = f"grads/{impl}"
    if not os.path.isdir(dirpath):
        print(f"Skip '{impl}': directory not found: {dirpath}")
        continue

    filenames_grads = sorted([f for f in os.listdir(dirpath) if f.endswith(".pth")])
    print(f"[{impl}] Found files: {filenames_grads}")

    # Load loss values if present
    loss_map = {}
    csv_path = os.path.join(dirpath, "loss_vals.csv")
    if os.path.isfile(csv_path):
        try:
            loss_vals = pd.read_csv(csv_path)
            if {"step", "val_loss"}.issubset(loss_vals.columns):
                loss_map = dict(zip(loss_vals["step"], loss_vals["val_loss"]))
        except Exception as e:
            print(f"[{impl}] Could not read {csv_path}: {e}")

    # group filenames by step
    by_step = defaultdict(list)
    for fn in filenames_grads:
        # grads/ns_impl/update_size0_size1_step.pth
        try:
            step = int(os.path.splitext(fn)[0].split("_")[-1])
        except Exception:
            # skip unparseable filenames
            continue
        by_step[step].append(fn)

    for step, fns in sorted(by_step.items()):
        errs = []
        for fn in fns:
            grad_path = os.path.join(dirpath, fn)
            try:
                grad = torch.load(grad_path, map_location=device)
            except Exception as e:
                print(f"[{impl}] Skip file {grad_path}: {e}")
                continue

            with torch.no_grad():
                ref = reference_ns(grad)
                approx = ns_impl[impl](grad)

                num = torch.linalg.norm(approx - ref, ord="fro", dim=(-2, -1))
                den = torch.linalg.norm(ref, ord="fro", dim=(-2, -1))
                eps = torch.finfo(ref.dtype).eps if ref.is_floating_point() else 1e-12
                ratio = num / den.clamp_min(eps)
                if ratio.ndim > 0:
                    ratio = ratio.mean()
                errs.append(float(ratio))

        rows.append(
            {
                "step": step,
                "val_loss": float(loss_map.get(step, float("nan"))),
                "impl": impl,
                "polar_error": float(sum(errs) / len(errs)) if errs else float("nan"),
            }
        )

df = pd.DataFrame(rows)

# drop step == 0 entirely
if not df.empty and "step" in df:
    df = df[df["step"] != 0].sort_values(["impl", "step"])

# --- plot: steps vs polar_error, color=val_loss (log), marker by impl ---
markers = {
    "aol": "o",
    # "muon+": "s",
    "muon": "s",
    "std_pe": "^",
    # "aol_pe": "D",
}

os.makedirs("figs", exist_ok=True)

plt.figure(figsize=(8, 5))

# set up normalization for colorbar; handle degenerate cases
norm = None
use_colorbar = False
if not df.empty and "val_loss" in df:
    vals = df["val_loss"].to_numpy(dtype=float)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size >= 2:
        vmin = float(pos.min())
        vmax = float(pos.max())
        # ensure vmin < vmax for LogNorm
        if vmax <= vmin:
            vmax = vmin * 1.000001
        if vmin > 0 and vmax > vmin:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            use_colorbar = True

last = None
plotted_impls = []
legend_handles = []

labels = {
    "aol":"p-Muon",
    "muon":"Muon",
    "std_pe":"PolarExpress",
    # "aol_pe":"p-PolarExpress",
}

for impl in ns_impls:
    d = df[df.impl == impl] if not df.empty else pd.DataFrame()
    if d.empty:
        continue

    sc = plt.scatter(
        d.step.to_numpy(),
        d.polar_error.to_numpy(),
        c=d.val_loss.to_numpy() if use_colorbar else None,
        marker=markers.get(impl, "o"),
        norm=norm,
    )
    last = sc
    plotted_impls.append(impl)
    # Legend with same color for all shapes (shape encodes impl)
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker=markers.get(impl, "o"),
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            # Use the descriptive label from the 'labels' dictionary
            label=labels.get(impl, impl),
        )
    )

if use_colorbar and last is not None:
    cbar = plt.colorbar(last)
    cbar.set_label("val_loss (log)")

if legend_handles:
    plt.legend(handles=legend_handles, title="ns_impl")

plt.xlabel("steps")
plt.ylabel("polar error")
plt.yscale("log")  # Added for better visualization of error magnitudes
plt.grid(True, which="both", linestyle='--', linewidth=0.5) # Added for readability
plt.tight_layout()
plt.savefig("figs/polar_error_training.png", dpi=300)
plt.close()

print("Plot saved to figs/polar_error_training.png")
