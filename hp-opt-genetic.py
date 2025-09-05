import math, torch
from torch import Tensor
from newton_schulz_triton import ns_line_1, ns_line_2, ns_line_3


# ------------------------------------------------------------
# Error metric: ||G - I||_F / sqrt(dim), computed in fp32
# ------------------------------------------------------------
@torch.inference_mode()
def orthogonality_error(X: Tensor) -> Tensor:
    m, n = X.shape[-2], X.shape[-1]
    if m <= n:
        gram = (X @ X.mT).to(torch.float32)
        I = torch.eye(m, device=X.device, dtype=torch.float32).expand(gram.shape)
        dim = m
    else:
        gram = (X.mT @ X).to(torch.float32)
        I = torch.eye(n, device=X.device, dtype=torch.float32).expand(gram.shape)
        dim = n
    return torch.linalg.norm(gram - I, ord="fro", dim=(-2, -1)).mean() / math.sqrt(dim)


def levy_stable(alpha, beta, size, device="cpu", generator=None):
    """
    Generate samples from a Levy alpha-stable distribution using
    the Chambers-Mallows-Stuck method.
    alpha: stability parameter (0 < alpha <= 2)
    beta: skewness parameter (-1 <= beta <= 1)
    """
    alpha = torch.tensor(alpha, device=device)
    U = torch.rand(size, device=device, generator=generator) * torch.pi - (torch.pi / 2)
    W = -torch.log(torch.rand(size, device=device, generator=generator))

    if alpha == 1:
        return (2 / torch.pi) * (
            (torch.pi / 2 + beta * U) * torch.tan(U)
            - beta
            * torch.log((torch.pi / 2 * W * torch.cos(U)) / (torch.pi / 2 + beta * U))
        )

    term1 = beta * torch.tan(torch.pi * alpha / 2)
    B = torch.atan(term1) / alpha
    S = (1 + term1**2) ** (1 / (2 * alpha))

    numerator = torch.sin(alpha * (U + B))
    denominator = (torch.cos(U)) ** (1 / alpha)
    frac = numerator / denominator

    term2 = (torch.cos(U - alpha * (U + B)) / W) ** ((1 - alpha) / alpha)

    return S * frac * term2


# ------------------------------------------------------------
# Newton–Schulz runner (bf16) using 4 steps, all used
# Expects your Triton kernels: ns_line_1/2/3
# this is a dirty hack to allow dynamic (a,b,c) constants
# todo: find a cleaner way to do this
# ------------------------------------------------------------
@torch.inference_mode()
def ns_run_with_consts(
    G: Tensor,
    ns_consts: Tensor,  # (4,3): (a,b,c) per step
    epsilon: float = 1e-7,
    blowup_threshold: float = 1e5,
):
    ns_consts = ns_consts.detach().cpu().numpy()
    # assert ns_consts.shape == (4, 3)
    X = G.to(dtype=torch.bfloat16)

    transposed = False
    if X.size(-2) > X.size(-1):  # ensure rows <= cols
        X = X.mT
        transposed = True

    X = X.contiguous()
    m = X.size(-2)
    A = torch.empty((*X.shape[:-2], m, m), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    a, b, c = ns_consts[0]
    a, b, c = float(a), float(b), float(c)
    ns_line_1(X, out=A)  # gram matrix A = X @ X.mT
    s = torch.clamp_min(
        A.abs().sum(dim=-1, keepdim=True), min=epsilon
    )  # AOL rescaling vector
    X = X * torch.rsqrt(s)  # rescale X using s making it closer to orthogonal
    # first NS iteration with reuse of A
    A = (
        s.transpose(-2, -1) * A * s
    )  # rescale A with s^2 as it is cheaper than computing ns_line_1 again
    ns_line_2(A, alpha=c, beta=b, out=B)
    ns_line_3(A, X, a, out=C)
    X, C = C, X

    # Perform the remaining NS iterations
    for a, b, c in ns_consts[1:-1]:
        a, b, c = float(a), float(b), float(c)
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(B, X, a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    a, b, c = ns_consts[-1]
    a, b, c = float(a), float(b), float(c)
    ns_line_1(X, out=A)  # A = X @ X.mT
    ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
    ns_line_3(B, X, a, out=C)  # C = a * X + B @ X
    X, C = C, X  # Swap references to avoid unnecessary copies

    # Divergence guard (fast path)
    if not torch.isfinite(X).all() or X.abs().max().item() > blowup_threshold:
        return None

    if transposed:
        X = X.mT
    return X


# ------------------------------------------------------------
# Fixed eval pool (tall/wide/square), bf16 iteration, fp32 scoring
# ------------------------------------------------------------
def make_eval_pool(
    device="cuda",
    shapes=((64, 64), (128, 128), (256, 256)),
    batch_per_shape=128,
    seeds=(1234, 5678),
):
    pool = []
    gen = torch.Generator(device=device)
    for seed in seeds:
        gen.manual_seed(seed)
        for m, n in shapes:
            for alpha in [1.5, 1.75]:  # Levy alpha-stable with varying alpha
                G = levy_stable(
                    alpha=alpha,
                    beta=0,
                    size=(batch_per_shape, m, n),
                    device=device,
                    generator=gen,
                )
                pool.append(G)
            G = torch.randn(
                (batch_per_shape, m, n),
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
            pool.append(G)
    return pool


@torch.inference_mode()
def objective(ns_consts_flat: Tensor, pool, epsilon: float = 1e-7):
    ns_consts = ns_consts_flat.view(4, 3)
    errs = []
    for G32 in pool:
        X = ns_run_with_consts(G32, ns_consts, epsilon=epsilon)
        if X is None:
            return torch.tensor(1e6, device=G32.device)  # divergence penalty
        e = orthogonality_error(X)
        if not torch.isfinite(e):
            return torch.tensor(1e6, device=G32.device)
        errs.append(e)
    return torch.stack(errs).mean()


# ------------------------------------------------------------
# Cross-Entropy Method (CEM) — robust, simple, gradient-free
# ------------------------------------------------------------
def cem_optimize(
    obj_fn,
    init_mean: Tensor,
    init_std: float = 0.6,
    iters: int = 50,
    pop: int = 96,
    elite_frac: float = 0.2,
    momentum: float = 0.85,
    clip_std=(1e-3, 3.0),
    verbose: bool = True,
):
    device = init_mean.device
    dim = init_mean.numel()
    mean = init_mean.clone()
    std = torch.full((dim,), init_std, device=device)
    num_elite = max(2, int(pop * elite_frac))
    best_score = float("inf")
    best_theta = mean.clone()

    for t in range(iters):
        theta = mean + std * torch.randn(pop, dim, device=device)
        scores = torch.empty(pop, device=device)
        for i in range(pop):
            scores[i] = obj_fn(theta[i])

        # pick smallest scores
        vals, idx = torch.topk(-scores, k=num_elite)
        elite = theta[idx]
        elite_scores = -vals

        new_mean = elite.mean(dim=0)
        new_std = elite.std(dim=0, unbiased=False).clamp_(*clip_std)

        mean = momentum * mean + (1 - momentum) * new_mean
        std = momentum * std + (1 - momentum) * new_std

        if elite_scores[0].item() < best_score:
            best_score = elite_scores[0].item()
            best_theta = elite[0].clone()

        if verbose:
            print(
                f"[CEM] {t+1:02d}/{iters}  elite_mean={elite_scores.mean().item():.3e}  "
                f"best={best_score:.3e}  std≈{std.mean().item():.3f}"
            )
    elite_mean_theta = elite.mean(dim=0)
    return best_theta, best_score, elite_mean_theta, mean, std


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
seed_consts = torch.tensor(
    [
        [4.6051846, -9.6552305, 5.676981],
        [4.750538, -6.086122, 2.1790226],
        [2.776319, -2.3190296, 0.55232877],
        [2.423169, -2.2861216, 0.81937317],
    ],
    device="cpu",
).flatten()

pool = make_eval_pool(device="cuda")


def obj_wrap(theta_flat):
    return objective(theta_flat, pool, epsilon=1e-7)


best_theta, best_score, elite_mean_theta, mean, std = cem_optimize(
    obj_fn=obj_wrap,
    init_mean=seed_consts,
    init_std=0.05,  # shrink if you see frequent divergence
    iters=250,
    pop=256,
    elite_frac=0.2,
    momentum=0.85,
    verbose=True,
)

print("\nBest (a,b,c) triples:")
print(best_theta.view(4, 3))
print("\nElite mean (a,b,c) triples:")
print(elite_mean_theta.view(4, 3))
print("Best score:", best_score)
