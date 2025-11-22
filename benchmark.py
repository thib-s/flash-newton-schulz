#!/usr/bin/env python3
"""
Unified benchmark for Newton–Schulz variants — multi-dist/multi-dtype + runtime std.

Adds:
- Per-repetition timing -> mean & std (batch_time_ms, batch_time_ms_std)
- Per-sample timing std (sample_time_ms_std)
- Accepts multiple dtypes and distributions in one run (--dtype ... --dist ...)
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from newton_schulz_triton import NS_muon, NS_muon_plus, NS_ours  # noqa: F401
from torch.linalg import LinAlgError

pd.set_option("display.float_format", "{:.3f}".format)


def device_string() -> str:
    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        return f"cuda:{dev_id} ({torch.cuda.get_device_name(dev_id)})"
    return "cpu"


# ------------------------- Distributions -------------------------
@torch.inference_mode()
def levy_stable(
    alpha: float, beta: float, size: Tuple[int, ...], device: str = "cpu"
) -> torch.Tensor:
    """Chambers–Mallows–Stuck sampler for Lévy α-stable."""
    alpha_t = torch.tensor(alpha, device=device)
    U = torch.rand(size, device=device) * torch.pi - (torch.pi / 2)
    W = -torch.log(torch.rand(size, device=device))
    if float(alpha) == 1.0:
        return (2 / torch.pi) * (
            (torch.pi / 2 + beta * U) * torch.tan(U)
            - beta
            * torch.log((torch.pi / 2 * W * torch.cos(U)) / (torch.pi / 2 + beta * U))
        )
    t1 = beta * torch.tan(torch.pi * alpha_t / 2)
    B = torch.atan(t1) / alpha_t
    S = (1 + t1**2) ** (1 / (2 * alpha_t))
    num = torch.sin(alpha_t * (U + B))
    den = (torch.cos(U)) ** (1 / alpha_t)
    frac = num / den
    t2 = (torch.cos(U - alpha_t * (U + B)) / W) ** ((1 - alpha_t) / alpha_t)
    return S * frac * t2


@torch.inference_mode()
def make_batch(
    batch: int,
    m: int,
    n: int,
    dtype: torch.dtype,
    device: str,
    dist: str,
    seed: int = 0,
    levy_alpha: float = 1.5,
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    if dist == "normal":
        X = torch.randn(batch, m, n, generator=gen, device=device, dtype=dtype)
    elif dist == "uniform":
        X = torch.rand(batch, m, n, generator=gen, device=device, dtype=dtype) - 0.5
    elif dist == "levy":
        X = levy_stable(
            alpha=levy_alpha, beta=0.0, size=(batch, m, n), device=device
        ).to(dtype)
    else:
        raise ValueError(f"Unknown dist: {dist}")
    return X


# ------------------------- Metrics (per-sample) -------------------------
@torch.inference_mode()
def orthogonality_error_per_sample(X: torch.Tensor) -> torch.Tensor:
    X32 = X.to(torch.float32)
    m, n = X32.shape[-2], X32.shape[-1]
    if m <= n:
        gram = X32 @ X32.mT
        I = torch.eye(m, device=X32.device, dtype=torch.float32).expand_as(gram)
        dim = m
    else:
        gram = X32.mT @ X32
        I = torch.eye(n, device=X32.device, dtype=torch.float32).expand_as(gram)
        dim = n
    dif = gram - I
    norms = torch.linalg.norm(dif, ord="fro", dim=(-2, -1))
    return norms / (dim**0.5)


@torch.inference_mode()
def polar_factor_batched(G: torch.Tensor) -> torch.Tensor:
    try:
        U, S, Vh = torch.linalg.svd(G.to(torch.float32), full_matrices=False)
        Q = (U @ Vh).to(torch.float32)
        return Q
    except LinAlgError:
        return torch.zeros_like(G).fill_(float("nan"))


@torch.inference_mode()
def polar_error_per_sample(
    X: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    X64 = X.to(torch.float64)
    Q64 = Q.to(torch.float64)
    num = torch.linalg.norm(X64 - Q64, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(Q64, ord="fro", dim=(-2, -1))
    return (num / (den + eps)).to(torch.float32)


@torch.inference_mode()
def sv_quantiles_per_sample(
    X: torch.Tensor,
):
    S = torch.linalg.svdvals(X.to(torch.float32))
    p = torch.tensor([0.02, 0.50, 0.98], device=X.device, dtype=torch.float32)
    q = torch.quantile(S, p, dim=-1).T
    return q[:, 0], q[:, 1], q[:, 2]


# ------------------------- Timing -------------------------
@dataclass
class TimingResult:
    y: torch.Tensor
    batch_ms_mean: float
    batch_ms_std: float
    per_rep_times_ms: List[float]


@torch.inference_mode()
def time_fn(
    fn,
    x: torch.Tensor,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
    iter_steps: int,
    synchronize: bool = True,
) -> TimingResult:
    # warmup
    for _ in range(max(warmup, 0)):
        _ = fn(x, iter=iter_steps)
    if synchronize and x.is_cuda:
        torch.cuda.synchronize()

    # timed (measure each repetition)
    times: List[float] = []
    y = None
    for _ in range(max(rep, 1)):
        start = time.perf_counter()
        y = fn(x, iter=iter_steps, dtype=dtype)
        if synchronize and x.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    times_arr = np.array(times, dtype=np.float64)
    return TimingResult(
        y=y,
        batch_ms_mean=float(times_arr.mean()),
        batch_ms_std=float(times_arr.std(ddof=1)) if len(times_arr) > 1 else 0.0,
        per_rep_times_ms=[float(t) for t in times_arr],
    )


# ------------------------- Main experiment -------------------------
ALGOS: Dict[str, callable] = {
    "Muon": NS_muon,
    "Muon+": NS_muon_plus,
    "AOLxMuon+": NS_ours,
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Newton–Schulz benchmark (batched, per-sample metrics)"
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Square sizes to benchmark (m=n)",
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size per shape")

    # Accept multiple dtypes and distributions in one run
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="+",
        default=["bfloat16"],
        choices=["float16", "bfloat16", "float32"],
        help="Input dtype(s); space-separated to test multiple",
    )
    parser.add_argument(
        "--rep", type=int, default=5, help="Number of timed repetitions (averaged/std)"
    )
    parser.add_argument(
        "--warmup", type=int, default=4, help="Warmup runs before timing"
    )
    parser.add_argument(
        "--dist",
        type=str,
        nargs="+",
        default=["normal"],
        choices=["normal", "uniform", "levy"],
        help="Input distribution(s); space-separated to test multiple",
    )
    parser.add_argument(
        "--levy-alpha", type=float, default=1.5, help="Alpha for Lévy distribution"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for batch generation"
    )
    parser.add_argument(
        "--csv", type=str, default="", help="Path to save CSV (optional)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Iteration counts to test for each algo",
    )
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        default=list(ALGOS.keys()),
        help="Algorithms to benchmark",
    )
    args = parser.parse_args()

    torch._dynamo.config.cache_size_limit = 1000

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"\nDevice: {device_string()}  |  dtypes={args.dtype}  |  batch={args.batch}  |  dists={args.dist}\n"
    )

    rows = []

    # select only the requested algos
    selected_algos = {k: v for k, v in ALGOS.items() if k in args.algos}

    for dim in tqdm(args.dims, position=0, desc="Sizes", leave=False):
        m = n = int(dim)

        for dtype_str in tqdm(args.dtype, position=1, desc="Dtypes", leave=False):
            dtype = dtype_map[dtype_str]

            for dist_str in tqdm(
                args.dist, position=2, desc="Distributions", leave=False
            ):
                # Generate a single batched input for this size/dtype/dist
                G = make_batch(
                    args.batch,
                    m,
                    n,
                    dtype=dtype,
                    device=device,
                    dist=dist_str,
                    seed=args.seed,
                    levy_alpha=args.levy_alpha,
                )

                # Reference polar factor for *original* G (used for polar_error)
                Q_ref = polar_factor_batched(G)

                for algo_name, algo_fn in tqdm(
                    selected_algos.items(), position=3, desc="Algorithms", leave=False
                ):
                    for k_iter in tqdm(
                        args.iters, position=4, desc="Iterations", leave=False
                    ):
                        # Time the batched call
                        t_res = time_fn(
                            algo_fn,
                            G,
                            dtype=dtype,
                            warmup=args.warmup,
                            rep=args.rep,
                            iter_steps=k_iter,
                            synchronize=True,
                        )
                        Y = t_res.y
                        batch_time_ms = float(t_res.batch_ms_mean)
                        batch_time_ms_std = float(t_res.batch_ms_std)
                        sample_time_ms = batch_time_ms / args.batch
                        sample_time_ms_std = batch_time_ms_std / args.batch

                        # Metrics per sample
                        ortho = orthogonality_error_per_sample(Y).detach().cpu().numpy()
                        perr = polar_error_per_sample(Y, Q_ref).detach().cpu().numpy()
                        # p02, p50, p98 = sv_quantiles_per_sample(Y)
                        # p02 = p02.detach().cpu().numpy()
                        # p50 = p50.detach().cpu().numpy()
                        # p98 = p98.detach().cpu().numpy()

                        B = int(Y.shape[0])
                        dev_str = device_string()
                        for i in range(B):
                            rows.append(
                                {
                                    "device": dev_str,
                                    "dtype": dtype_str,
                                    "dist": dist_str,
                                    "levy_alpha": (
                                        args.levy_alpha
                                        if dist_str == "levy"
                                        else np.nan
                                    ),
                                    "dim": (m, n),
                                    "size": m,
                                    "algo": algo_name,
                                    "iter": k_iter,
                                    "sample_id": i,
                                    "batch_time_ms": batch_time_ms,  # mean
                                    "batch_time_ms_std": batch_time_ms_std,  # std across reps
                                    "sample_time_ms": sample_time_ms,  # mean
                                    "sample_time_ms_std": sample_time_ms_std,
                                    "orthogonality_error": float(ortho[i]),
                                    "polar_error": float(perr[i]),
                                    # "sv_p02": float(p02[i]),
                                    # "sv_p50": float(p50[i]),
                                    # "sv_p98": float(p98[i]),
                                }
                            )

    df = pd.DataFrame(rows)
    print(df.head())

    if args.csv:
        out_path = args.csv
        df.to_csv(out_path, index=False)
        print(f"\nSaved CSV to: {out_path}")

    # Optional: quick summary by algo/iter/size (printed, not plotted)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        summary = (
            df.groupby(["algo", "iter", "size", "dtype", "dist"], as_index=False)
            .agg(
                polar_error_mean=("polar_error", "mean"),
                polar_error_std=("polar_error", "std"),
                ortho_err_mean=("orthogonality_error", "mean"),
                batch_time_ms_mean=("batch_time_ms", "mean"),
                batch_time_ms_std_mean=("batch_time_ms_std", "mean"),
            )
            .sort_values(["size", "algo", "iter", "dtype", "dist"])
        )
        print("\nSummary (means/stds by algo, iter, size, dtype, dist):\n", summary)

    return df


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")
    main()


# #!/usr/bin/env python3
# """
# Unified benchmark for Newton–Schulz variants.

# - Measures throughput (ms) like the original benchmark.
# - Computes per-sample metrics (no batch averaging):
#     * orthogonality_error  = ||G - I||_F / sqrt(min(m,n))
#     * polar_error         = ||X - Q||_F / ||Q||_F where Q is the polar factor of the *input* G
#     * sv_p02 / sv_p50 / sv_p98 of each output sample
# - Supports batched inputs (one random batch per size), avoiding outer loops over seeds.
# - Assumes 3 algorithms with an `iter` parameter in [1..5]: NS_muon, NS_muon_plus, NS_ours.

# Results are returned and optionally saved as a pandas DataFrame (one row per *sample*).

# Usage examples:
#   python ns_unified_benchmark.py \
#       --dims 128 256 512 1024 \
#       --batch 32 \
#       --rep 5 --warmup 4 \
#       --dtype bfloat16 \
#       --dist normal \
#       --csv out.csv

# Notes:
# - If your NS_* functions are Triton-backed, first run may include compile/autotune during warmup.
# - Adjust the import below to match where your NS_* implementations live.
# """

# import argparse
# import time
# from dataclasses import dataclass
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from newton_schulz_triton import NS_muon, NS_muon_plus, NS_ours  # noqa: F401
# from torch.linalg import LinAlgError

# pd.set_option("display.float_format", "{:.3f}".format)


# def device_string() -> str:
#     if torch.cuda.is_available():
#         dev_id = torch.cuda.current_device()
#         return f"cuda:{dev_id} ({torch.cuda.get_device_name(dev_id)})"
#     return "cpu"


# # ------------------------- Distributions -------------------------
# @torch.inference_mode()
# def levy_stable(
#     alpha: float, beta: float, size: Tuple[int, ...], device: str = "cpu"
# ) -> torch.Tensor:
#     """Chambers–Mallows–Stuck sampler for Lévy α-stable."""
#     alpha_t = torch.tensor(alpha, device=device)
#     U = torch.rand(size, device=device) * torch.pi - (torch.pi / 2)
#     W = -torch.log(torch.rand(size, device=device))
#     if float(alpha) == 1.0:
#         return (2 / torch.pi) * (
#             (torch.pi / 2 + beta * U) * torch.tan(U)
#             - beta
#             * torch.log((torch.pi / 2 * W * torch.cos(U)) / (torch.pi / 2 + beta * U))
#         )
#     t1 = beta * torch.tan(torch.pi * alpha_t / 2)
#     B = torch.atan(t1) / alpha_t
#     S = (1 + t1**2) ** (1 / (2 * alpha_t))
#     num = torch.sin(alpha_t * (U + B))
#     den = (torch.cos(U)) ** (1 / alpha_t)
#     frac = num / den
#     t2 = (torch.cos(U - alpha_t * (U + B)) / W) ** ((1 - alpha_t) / alpha_t)
#     return S * frac * t2


# @torch.inference_mode()
# def make_batch(
#     batch: int,
#     m: int,
#     n: int,
#     dtype: torch.dtype,
#     device: str,
#     dist: str,
#     seed: int = 0,
#     levy_alpha: float = 1.5,
# ) -> torch.Tensor:
#     gen = torch.Generator(device=device).manual_seed(seed)
#     if dist == "normal":
#         X = torch.randn(batch, m, n, generator=gen, device=device, dtype=dtype)
#     elif dist == "uniform":
#         X = torch.rand(batch, m, n, generator=gen, device=device, dtype=dtype) - 0.5
#     elif dist == "levy":
#         X = levy_stable(
#             alpha=levy_alpha, beta=0.0, size=(batch, m, n), device=device
#         ).to(dtype)
#     else:
#         raise ValueError(f"Unknown dist: {dist}")
#     return X


# # ------------------------- Metrics (per-sample) -------------------------
# @torch.inference_mode()
# def orthogonality_error_per_sample(X: torch.Tensor) -> torch.Tensor:
#     """Return a vector of length B with ||G - I||_F / sqrt(min(m,n)) computed per sample in fp32."""
#     X32 = X.to(torch.float32)
#     m, n = X32.shape[-2], X32.shape[-1]
#     if m <= n:
#         gram = X32 @ X32.mT  # (B, m, m)
#         I = torch.eye(m, device=X32.device, dtype=torch.float32).expand_as(gram)
#         dim = m
#     else:
#         gram = X32.mT @ X32
#         I = torch.eye(n, device=X32.device, dtype=torch.float32).expand_as(gram)
#         dim = n
#     dif = gram - I
#     norms = torch.linalg.norm(dif, ord="fro", dim=(-2, -1))  # (B,)
#     return norms / (dim**0.5)


# @torch.inference_mode()
# def polar_factor_batched(G: torch.Tensor) -> torch.Tensor:
#     """Polar factor Q = U @ V^T of G, computed in float64 for stability, returned as float32."""
#     try:
#         U, S, Vh = torch.linalg.svd(G.to(torch.float64), full_matrices=False)
#         Q = (U @ Vh).to(torch.float32)
#         return Q
#     except LinAlgError:
#         return torch.zeros_like(G).fill_(float("nan"))


# @torch.inference_mode()
# def polar_error_per_sample(
#     X: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12
# ) -> torch.Tensor:
#     X64 = X.to(torch.float64)
#     Q64 = Q.to(torch.float64)
#     num = torch.linalg.norm(X64 - Q64, ord="fro", dim=(-2, -1))
#     den = torch.linalg.norm(Q64, ord="fro", dim=(-2, -1))
#     return (num / (den + eps)).to(torch.float32)


# @torch.inference_mode()
# def sv_quantiles_per_sample(
#     X: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Return (p02, p50, p98) per sample as 1D tensors of length B."""
#     S = torch.linalg.svdvals(X.to(torch.float32))  # (B, k)
#     p = torch.tensor([0.02, 0.50, 0.98], device=X.device, dtype=torch.float32)
#     q = torch.quantile(S, p, dim=-1).T  # (B, 3)
#     return q[:, 0], q[:, 1], q[:, 2]


# # ------------------------- Timing -------------------------
# @dataclass
# class TimingResult:
#     y: torch.Tensor
#     batch_ms: float


# @torch.inference_mode()
# def time_fn(
#     fn,
#     x: torch.Tensor,
#     warmup: int,
#     rep: int,
#     iter_steps: int,
#     synchronize: bool = True,
# ) -> TimingResult:
#     # warmup
#     for _ in range(max(warmup, 0)):
#         _ = fn(x, iter=iter_steps)
#     if synchronize and x.is_cuda:
#         torch.cuda.synchronize()
#     # timed
#     start = time.perf_counter()
#     y = None
#     for _ in range(max(rep, 1)):
#         y = fn(x, iter=iter_steps)
#     if synchronize and x.is_cuda:
#         torch.cuda.synchronize()
#     elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(rep, 1)
#     return TimingResult(y=y, batch_ms=elapsed_ms)


# # ------------------------- Main experiment -------------------------
# ALGOS: Dict[str, callable] = {
#     "NS_muon": NS_muon,
#     "NS_muon_plus": NS_muon_plus,
#     "NS_ours": NS_ours,
# }


# def main():
#     parser = argparse.ArgumentParser(
#         description="Unified Newton–Schulz benchmark (batched, per-sample metrics)"
#     )
#     parser.add_argument(
#         "--dims",
#         type=int,
#         nargs="+",
#         default=[128, 256, 512, 1024],
#         help="Square sizes to benchmark (m=n)",
#     )
#     parser.add_argument("--batch", type=int, default=32, help="Batch size per shape")
#     parser.add_argument(
#         "--dtype",
#         type=str,
#         default="bfloat16",
#         choices=["float16", "bfloat16", "float32"],
#         help="Input dtype",
#     )
#     parser.add_argument(
#         "--rep", type=int, default=5, help="Number of timed repetitions (averaged)"
#     )
#     parser.add_argument(
#         "--warmup", type=int, default=4, help="Warmup runs before timing"
#     )
#     parser.add_argument(
#         "--dist",
#         type=str,
#         default="normal",
#         choices=["normal", "uniform", "levy"],
#         help="Input distribution",
#     )
#     parser.add_argument(
#         "--levy-alpha", type=float, default=1.5, help="Alpha for Lévy distribution"
#     )
#     parser.add_argument(
#         "--seed", type=int, default=0, help="RNG seed for batch generation"
#     )
#     parser.add_argument(
#         "--csv", type=str, default="", help="Path to save CSV (optional)"
#     )
#     parser.add_argument(
#         "--iters",
#         type=int,
#         nargs="+",
#         default=[1, 2, 3, 4, 5],
#         help="Iteration counts to test for each algo",
#     )
#     args = parser.parse_args()

#     torch._dynamo.config.cache_size_limit = 100

#     dtype_map = {
#         "float16": torch.float16,
#         "bfloat16": torch.bfloat16,
#         "float32": torch.float32,
#     }
#     dtype = dtype_map[args.dtype]
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     print(
#         f"\nDevice: {device_string()}  |  dtype={args.dtype}  |  batch={args.batch}  |  dist={args.dist}\n"
#     )

#     rows = []

#     for dim in tqdm(args.dims, position=0, desc="Sizes", leave=False):
#         m = n = int(dim)
#         # Generate a single batched input for this size
#         G = make_batch(
#             args.batch,
#             m,
#             n,
#             dtype=dtype,
#             device=device,
#             dist=args.dist,
#             seed=args.seed,
#             levy_alpha=args.levy_alpha,
#         )

#         # Reference polar factor for *original* G (used for polar_error)
#         Q_ref = polar_factor_batched(G)

#         for algo_name, algo_fn in tqdm(
#             ALGOS.items(), position=1, desc="Algorithms", leave=False
#         ):
#             for k_iter in tqdm(args.iters, position=2, desc="Iterations", leave=False):
#                 # Time the batched call
#                 t_res = time_fn(
#                     algo_fn,
#                     G,
#                     warmup=args.warmup,
#                     rep=args.rep,
#                     iter_steps=k_iter,
#                     synchronize=True,
#                 )
#                 Y = t_res.y
#                 batch_time_ms = float(t_res.batch_ms)
#                 sample_time_ms = batch_time_ms / args.batch

#                 # Metrics per sample
#                 ortho = orthogonality_error_per_sample(Y).detach().cpu().numpy()
#                 perr = polar_error_per_sample(Y, Q_ref).detach().cpu().numpy()
#                 p02, p50, p98 = sv_quantiles_per_sample(Y)
#                 p02 = p02.detach().cpu().numpy()
#                 p50 = p50.detach().cpu().numpy()
#                 p98 = p98.detach().cpu().numpy()

#                 B = int(Y.shape[0])
#                 for i in range(B):
#                     rows.append(
#                         {
#                             "device": device_string(),
#                             "dtype": args.dtype,
#                             "dist": args.dist,
#                             "dim": (m, n),
#                             "size": m,
#                             "algo": algo_name,
#                             "iter": k_iter,
#                             "sample_id": i,
#                             "batch_time_ms": batch_time_ms,
#                             "sample_time_ms": sample_time_ms,
#                             "orthogonality_error": float(ortho[i]),
#                             "polar_error": float(perr[i]),
#                             "sv_p02": float(p02[i]),
#                             "sv_p50": float(p50[i]),
#                             "sv_p98": float(p98[i]),
#                         }
#                     )

#     df = pd.DataFrame(rows)
#     print(df.head())

#     if args.csv:
#         out_path = args.csv
#         df.to_csv(out_path, index=False)
#         print(f"\nSaved CSV to: {out_path}")

#     # Optional: quick summary by algo/iter/size (printed, not plotted)
#     with pd.option_context("display.max_rows", 200, "display.width", 160):
#         summary = (
#             df.groupby(["algo", "iter", "size"], as_index=False)
#             .agg(
#                 polar_error_mean=("polar_error", "mean"),
#                 polar_error_std=("polar_error", "std"),
#                 ortho_err_mean=("orthogonality_error", "mean"),
#                 batch_time_ms_mean=("batch_time_ms", "mean"),
#             )
#             .sort_values(["size", "algo", "iter"])
#         )
#         print("\nSummary (means/stds by algo, iter, size):\n", summary)

#     return df


# if __name__ == "__main__":
#     # torch.set_float32_matmul_precision("high")
#     main()
