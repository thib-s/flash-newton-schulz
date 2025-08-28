
#!/usr/bin/env python3
"""
Benchmark for Newton-Schulz variants.

It measures throughput (ms), and basic orthogonality quality metrics for:
  - newton_schulz_torch           (PyTorch-only reference)
  - newton_schulz_triton_dion     (Triton, Dion-style)
  - newton_schulz_triton_aol      (Triton, AOL rescaling)

Usage examples:
  python benchmark_clean.py
  python benchmark_clean.py --dims 128 256 512 1024 --batch 32 --rep 10 --warmup 5
  python benchmark_clean.py --no-plot --csv out.csv

On first run, `torch.compile` and Triton autotuning may add overhead during warmup.


This code was insipired by the benchmark code from flash-muon: https://github.com/nil0x9/flash-muon/tree/main
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from newton_schulz_triton import (
    newton_schulz_torch,
    newton_schulz_triton_dion,
    newton_schulz_triton_aol,
)

pd.set_option("display.float_format", "{:,.3f}".format)


# ------------------------- Helpers -------------------------
def device_string():
    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        return f"cuda:{dev_id} ({torch.cuda.get_device_name(dev_id)})"
    return "cpu"


@torch.inference_mode()
def orthogonality_error(X: torch.Tensor) -> float:
    """Return ||G - I||_F / sqrt(dim) in fp32, where G is the Gram of X on the smaller side."""
    m, n = X.shape[-2], X.shape[-1]
    if m <= n:
        gram = (X @ X.mT).to(torch.float32)
        I = torch.eye(m, device=X.device, dtype=torch.float32).expand(gram.shape)
        dim = m
    else:
        gram = (X.mT @ X).to(torch.float32)
        I = torch.eye(n, device=X.device, dtype=torch.float32).expand(gram.shape)
        dim = n
    return (torch.linalg.norm(gram - I, ord="fro", dim=(-2, -1)).mean() / (dim ** 0.5)).item()


def sv_stats(X: torch.Tensor):
    """Return (p02, p50, p98) of singular values of X (fp32)."""
    S = torch.linalg.svdvals(X.to(torch.float32))
    p = torch.tensor([0.02, 0.50, 0.98], device=X.device, dtype=torch.float32)
    q = torch.quantile(S, p, dim=-1).mean(dim=1) if S.ndim == 2 else torch.quantile(S, p)
    return tuple(float(v) for v in q)


def make_batch(batch, m, n, dtype, device, seed=0, uniform=False):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if uniform:
        X = (torch.rand((batch, m, n), generator=gen, device=device, dtype=dtype) - 0.5)
    else:
        X = torch.randn((batch, m, n), generator=gen, device=device, dtype=dtype)
    return X


def time_fn(fn, x, warmup, rep, synchronize=True):
    # Warmup
    for _ in range(warmup):
        fn(x)
    if synchronize and x.is_cuda:
        torch.cuda.synchronize()
    # Timed
    start = time.perf_counter()
    for _ in range(rep):
        y = fn(x)
    if synchronize and x.is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0 / rep  # ms
    return y, elapsed


# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Newton-Schulz implementations")
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512, 1024],
                        help="Square sizes to benchmark (m=n)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per shape")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--rep", type=int, default=5, help="Number of timed repetitions")
    parser.add_argument("--warmup", type=int, default=4, help="Warmup runs before timing")
    parser.add_argument("--uniform", action="store_true", help="Use U(-0.5,0.5) instead of N(0,1)")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable SV plots")
    parser.add_argument("--csv", type=str, default="", help="Optional path to save CSV results")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for input generation")
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Note: we call the already compiled functions; they carry @torch.compile internally.
    impls = [
        ("newton_schulz_torch", newton_schulz_torch),
        ("newton_schulz_triton_dion", newton_schulz_triton_dion),
        ("newton_schulz_triton_aol", newton_schulz_triton_aol),
    ]

    # Results
    bench = defaultdict(list)
    svs_cache = {}  # {(dim, name): np.ndarray of singular values}
    print(f"\nDevice: {device_string()}  |  dtype={args.dtype}  |  batch={args.batch}\n")

    for dim in args.dims:
        m = n = dim
        X = make_batch(args.batch, m, n, dtype=dtype, device=device, seed=args.seed, uniform=args.uniform)
        bench["device"].append(device_string())
        bench["dim"].append((m, n))

        for name, fn in impls:
            try:
                Y, t_ms = time_fn(fn, X, warmup=args.warmup, rep=args.rep, synchronize=True)
                err = orthogonality_error(Y)
                p02, p50, p98 = sv_stats(Y)
            except Exception as e:
                t_ms, err, p02, p50, p98 = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
                print(f"[WARN] {name} failed on dim {m}x{n}: {e}")

            bench[f"{name}_ms"].append(t_ms)
            bench[f"{name}_err"].append(err)
            bench[f"{name}_sv(2%,50%,98%)"].append((round(p02,3), round(p50,3), round(p98,3)))

        # Pretty print per-dim row summary
        row = {k: bench[k][-1] for k in bench if len(bench[k]) == len(bench["dim"])}
        # (We rely on the final table print below for a nice view.)

    df = pd.DataFrame(bench)
    print(df)

    # Save CSV if requested
    if args.csv:
        out_path = args.csv
        df.to_csv(out_path, index=False)
        print(f"\nSaved CSV to: {out_path}")

    # Optional: simple plot of singular value medians per impl (one figure per dim)
    if args.plot:
        for dim in args.dims:
            m = n = dim
            # regenerate inputs to compute SVD bands consistently for this dim
            X = make_batch(args.batch, m, n, dtype=dtype, device=device, seed=args.seed, uniform=args.uniform)
            fig = plt.figure(figsize=(8, 5))
            k = None
            for name, fn in impls:
                try:
                    Y = fn(X)
                    S = torch.linalg.svdvals(Y.to(torch.float32)).cpu().numpy()
                    if S.ndim == 1:
                        S = S[None, :]
                    if k is None:
                        k = S.shape[1]
                    xs = np.arange(1, k + 1)
                    med = np.median(S, axis=0)
                    lo = np.quantile(S, 0.02, axis=0)
                    hi = np.quantile(S, 0.98, axis=0)
                    plt.plot(xs, med, label=name, alpha=0.7)
                    plt.fill_between(xs, lo, hi, alpha=0.15)
                except Exception as e:
                    print(f"[WARN] Plot skipped for {name} at {m}x{n}: {e}")
            plt.title(f"Singular values - dim {m}×{n}")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.ylim(0.8, 1.2)
            plt.grid(True, which="both", linestyle="--", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            fig.savefig(f"svs_{m}x{n}.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
