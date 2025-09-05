"""
Use the polynomial method to tune hyperparameters of the Newton-Schulz iteration for orthogonalization,
as in https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b.

We use a mixture of distributions to fit the singular value mapping of the AOL method approximately then
learn to optimize the NS coefficients on top of it.
"""

import jax
import optax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from functools import partial
import torch
import math


# -----------------------------
# Lévy α-stable distribution (heavy-tailed)
# -----------------------------
def levy_stable(alpha, beta, size, rng=None):
    """
    Generate samples from a Lévy alpha-stable distribution using CMS method.
    alpha: stability parameter (0 < alpha <= 2)
    beta: skewness parameter (-1 <= beta <= 1)
    """
    rng = np.random.default_rng() if rng is None else rng
    U = rng.random(size) * np.pi - (np.pi / 2)
    W = -np.log(rng.random(size))

    if alpha == 1:
        return (2 / np.pi) * (
            (np.pi / 2 + beta * U) * np.tan(U)
            - beta * np.log((np.pi / 2 * W * np.cos(U)) / (np.pi / 2 + beta * U))
        )

    term1 = beta * np.tan(np.pi * alpha / 2)
    B = np.arctan(term1) / alpha
    S = (1 + term1**2) ** (1 / (2 * alpha))

    numerator = np.sin(alpha * (U + B))
    denominator = (np.cos(U)) ** (1 / alpha)
    frac = numerator / denominator
    term2 = (np.cos(U - alpha * (U + B)) / W) ** ((1 - alpha) / alpha)

    return S * frac * term2


def aol_rescale_np(parameter_matrix: np.ndarray, epsilon: float = 1e-6):
    A = parameter_matrix @ parameter_matrix.T
    s = np.clip(np.abs(A).sum(axis=-1), a_min=1e-10, a_max=float("inf"))
    return parameter_matrix / np.sqrt(s)[:, None]


def sample_matrix(shape, generator, rng):
    if generator == "uniform":
        pre = rng.uniform(low=0.0, high=1.0, size=shape)
    elif generator == "normal":
        pre = rng.normal(loc=0.0, scale=1.0, size=shape)
    elif generator == "levy":
        pre = levy_stable(alpha=1.5, beta=0.0, size=shape, rng=rng)
    elif generator == "bernoulli":
        pre = rng.binomial(n=1, p=0.5, size=shape).astype(float)
    else:
        raise ValueError(f"Unknown distribution generator {generator}")
    return pre


# -----------------------------
# Multi-distribution fit
# -----------------------------
def fit_aol_sv_poly_logspace_multi(
    shape,
    distributions,  # dict: { "uniform": weight, "normal": weight, ... }
    deg=3,
    seed=0,
    eps=1e-12,
    n_samples=3,  # how many matrices per distribution
):
    rng = np.random.default_rng(seed)

    X, Y, W = [], [], []

    for generator, weight in distributions.items():
        for _ in range(n_samples):
            pre = sample_matrix(shape, generator, rng)
            pre_svs = np.linalg.svd(pre, compute_uv=False).astype(np.float64)
            pre_svs /= max(pre_svs.max(), eps)

            post = aol_rescale_np(pre)
            post_svs = np.linalg.svd(post, compute_uv=False).astype(np.float64)

            mask = (
                np.isfinite(pre_svs)
                & np.isfinite(post_svs)
                & (pre_svs > 0)
                & (post_svs > 0)
            )
            x = np.log(pre_svs[mask] + eps)
            y = np.log(post_svs[mask] + eps)

            X.append(x)
            Y.append(y)
            W.append(np.full_like(x, weight))

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    W = np.concatenate(W)

    coeffs = np.polyfit(X, Y, deg, w=W)  # weighted polynomial fit
    return coeffs, eps


# -----------------------------
# Execution & Visualization
# -----------------------------
shapes = [(256, 256), (512, 512), (1024, 1024), (768, 128)]
fit_shape = (512, 512)

distributions = {
    "uniform": 1.0,
    "normal": 1.0,
    "levy": 1.0,
    "bernoulli": 1.0,
}

coeffs, eps = fit_aol_sv_poly_logspace_multi(
    shape=fit_shape,
    distributions=distributions,
    deg=5,
    seed=0,
    n_samples=150,
)

aol_coeffs = jnp.array(coeffs)

plt.figure(figsize=(5 * len(shapes), 10))
i = 1
for gen in distributions.keys():
    for shape in shapes:
        plt.subplot(len(distributions), len(shapes), i)
        rng = np.random.default_rng(123 + i)
        pre = sample_matrix(shape, gen, rng)
        pre_svs_r = np.linalg.svd(pre, compute_uv=False)
        post = aol_rescale_np(pre)
        post_svs_r = np.linalg.svd(post, compute_uv=False)

        x_eval = jnp.log(pre_svs_r / max(pre_svs_r.max(), eps) + eps)
        sim = jnp.exp(jnp.polyval(aol_coeffs, x_eval))

        plt.plot(pre_svs_r, label="Input SVs", alpha=0.8)
        plt.plot(post_svs_r, label="Output SVs", alpha=0.8)
        plt.plot(sim, label="Fitted AOL mapping", alpha=0.8)
        plt.yscale("log")
        plt.title(f"{gen}, shape={shape}")
        if i == 1:
            plt.legend()
        i += 1

plt.tight_layout()
plt.savefig("assets/aol_polynomial_fit.png")


# -----------------------------
# Optimize NS coefficients with AOL mapping
# -----------------------------
def poly(x: jnp.ndarray, w: jnp.ndarray):
    assert w.shape == (3,)
    w = w.astype(jnp.float32)
    return w[0] * x + w[1] * x**3 + w[2] * x**5


def poly_chain(x: jnp.ndarray, w_seq: jnp.ndarray):
    y = [x]
    for w in w_seq:
        y.append(poly(y[-1], w))
    return y


def min_of_polys(x: jnp.ndarray, w_seq: jnp.ndarray):
    y = jnp.full_like(x, jnp.inf)
    for w in w_seq:
        y = jnp.minimum(y, poly(x, w))
    return y


def _aol_transform(xs: jnp.ndarray, aol_coeffs: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Map xs ∈ (0,1] using an AOL-style polynomial on log-domain, then renormalize to (eps,1].
    y = exp( poly(log(xs)) ), normalized to max 1.
    """
    xs = xs.astype(jnp.float32)
    eps32 = jnp.array(eps, dtype=jnp.float32)
    xs = jnp.clip(xs, a_min=eps32)  # keep strictly positive
    logx = jnp.log(xs)
    poly_logx = jnp.polyval(
        aol_coeffs.astype(jnp.float32), logx
    )  # coeffs high→low order
    mapped = jnp.exp(poly_logx)
    denom = jnp.maximum(jnp.max(mapped), eps32)
    mapped = mapped / denom
    mapped = jnp.clip(mapped, a_min=eps32, a_max=1.0)
    return mapped


@partial(jax.jit, static_argnames=("n", "debug"))
def optimize_w(
    w_seq: jnp.ndarray,
    lr: float,
    n: int,
    aol_coeffs: jnp.ndarray | None = None,
    eps: float = 1e-6,
    debug: bool = False,
):
    def loss(w_seq: jnp.ndarray):
        w_seq = w_seq.astype(jnp.bfloat16)

        xs = (jnp.arange(2048, dtype=jnp.float32) + 1.0) / 2048.0

        # --- AOL transform (enabled if aol_coeffs provided)
        if aol_coeffs is not None:
            xs = _aol_transform(xs, aol_coeffs, eps)

        *zs, ys = jax.vmap(poly_chain, in_axes=(0, None))(xs, w_seq)
        y_max = jnp.amax(ys)
        y_min = jnp.amin(jnp.where(xs > 1.0 / 128.0, ys, jnp.inf))
        diff_ratio = (y_max - y_min) / jnp.clip(y_max, min=1e-3)

        slope_xs = (jnp.arange(320, dtype=jnp.float32) + 1.0) / 256.0
        min_ps = jax.vmap(min_of_polys, in_axes=(0, None))(slope_xs, w_seq)
        min_slope = jnp.amin(min_ps / slope_xs)

        z_max_seq = [jnp.amax(z) for z in zs]
        max_next_excess = sum(
            jnp.clip(poly(z + 1.0 / 16.0, w) - z, min=0.0)
            for z, w in zip(z_max_seq, w_seq)
        )

        obj_0 = ys[0] / y_max  # larger is better
        obj_1 = y_max  # closer to 1 is better
        obj_2 = jnp.log2(diff_ratio)  # smaller is better
        obj_3 = min_slope  # larger is better
        obj_4 = max_next_excess  # smaller is better

        objectives = (obj_0, obj_1, obj_2, obj_3, obj_4)

        if debug:
            jax.debug.print("{x}", x=objectives)

        # Default coeffs as used by @YouJiacheng
        loss = -4.0 * obj_0
        loss += 16.0 * jnp.square(obj_1 - 1)
        loss += 2.0 * jnp.clip(obj_2, min=-10)
        loss += -4.0 * jnp.clip(obj_3, max=1 / 2)
        loss += 64.0 * obj_4
        return loss, objectives

    loss_and_grad_fn = jax.value_and_grad(loss, argnums=0, has_aux=True)
    optimizer = optax.chain(
        optax.adam(learning_rate=lr),
        optax.clip_by_global_norm(1.0),
    )
    opt_state = optimizer.init(w_seq)

    def body_fn(carry: tuple[jnp.ndarray, optax.OptState], _):
        w_seq, opt_state = carry
        (_, objectives), grad = loss_and_grad_fn(w_seq)
        updates, opt_state = optimizer.update(grad, opt_state)
        w_seq = optax.apply_updates(w_seq, updates)
        return (w_seq, opt_state), objectives

    (w_seq, _), objectives = jax.lax.scan(body_fn, (w_seq, opt_state), length=n)
    return w_seq, objectives


def main(num_iters, coefs, eps=1e-6):
    BASE = 128.0
    # Identity mapping by default: poly(t) = t  => exp(poly(log x)) = x
    eps = jnp.array(eps)

    w_seq = jnp.array(
        [[3.5, -6.04444444444, 2.84444444444]] * num_iters, dtype=jnp.float32
    )
    for i in range(5):
        w_seq, objectives = optimize_w(
            w_seq, lr=2e-3, n=100000, aol_coeffs=coefs, eps=eps
        )
        print(w_seq.astype(jnp.bfloat16) * BASE)
        print(i, [obj[-1].item() for obj in objectives])
    for i in range(5):
        w_seq, objectives = optimize_w(
            w_seq, lr=1e-3, n=100000, aol_coeffs=coefs, eps=eps
        )
        print(w_seq.astype(jnp.bfloat16) * BASE)
        print(i, [obj[-1].item() for obj in objectives])
    for i in range(5):
        w_seq, objectives = optimize_w(
            w_seq, lr=5e-4, n=100000, aol_coeffs=coefs, eps=eps
        )
        print(w_seq.astype(jnp.bfloat16) * BASE)
        print(i, [obj[-1].item() for obj in objectives])
    for i in range(20):
        w_seq, objectives = optimize_w(
            w_seq, lr=5e-4, n=100000, aol_coeffs=coefs, eps=eps
        )
        print(w_seq.astype(jnp.bfloat16) * BASE)
        print(i, [obj[-1].item() for obj in objectives])
    return w_seq


num_iters = 4
w_seq_aol = main(num_iters, coefs=aol_coeffs)
w_seq_ns = main(num_iters, coefs=None)


def orthogonalize(G, ns_consts, aol=True):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2

    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    if not aol:
        X = X / torch.linalg.norm(X, ord=2, dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    # Perform the NS iterations
    for i, (a, b, c) in enumerate(ns_consts):
        A = X @ X.mT

        if i == 0 and aol:
            rescalings = torch.clamp(A.abs().sum(dim=-1, keepdim=True), min=1e-12)
            inv_sqrt = torch.rsqrt(rescalings)
            X = X * inv_sqrt
            A = (A * inv_sqrt) * inv_sqrt.transpose(-2, -1)

        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


# -----------------------------
# Visualize the effect of orthogonalization methods on a random matrix
# -----------------------------
shape = (512, 512)
distribs = ["levy", "normal", "uniform", "bernoulli"]
rng = np.random.default_rng(1234)
n_runs = 8

fig = plt.figure(figsize=(len(distribs) * 5, 5))  # <- keep a handle to the figure
axes = []

for i, gen in enumerate(distribs):
    ax = plt.subplot(1, len(distribs), i + 1)
    axes.append(ax)

    pre_list, aol_list, sim_list, ns_list, ns_aol_list = [], [], [], [], []
    for _ in range(n_runs):
        G = sample_matrix(shape, gen, rng)
        G = torch.from_numpy(G)
        consts_ns = torch.from_numpy(np.array(w_seq_ns))
        consts_aol = torch.from_numpy(np.array(w_seq_aol))

        post_ns_aol = orthogonalize(G, consts_aol, aol=True)
        post_ns = orthogonalize(G, consts_ns, aol=False)

        svs_pre = torch.linalg.svdvals(G)
        svs_pre /= svs_pre.max()

        post_aol = torch.from_numpy(aol_rescale_np(G.cpu().numpy()))
        svs_post_aol = torch.linalg.svdvals(post_aol)

        svs_sim_aol = _aol_transform(svs_pre.cpu().numpy(), aol_coeffs, eps)

        svs_post_ns = torch.linalg.svdvals(post_ns)
        svs_post_ns_aol = torch.linalg.svdvals(post_ns_aol)

        pre_list.append(svs_pre)
        sim_list.append(svs_sim_aol)
        aol_list.append(svs_post_aol)
        ns_list.append(svs_post_ns)
        ns_aol_list.append(svs_post_ns_aol)

    svs_pre_mean = torch.stack(pre_list, dim=0).mean(0)
    svs_post_aol_mean = torch.stack(aol_list, dim=0).mean(0)
    svs_sim_aol_mean = torch.from_numpy(np.stack(sim_list, axis=0).mean(0))
    svs_post_ns_mean = torch.stack(ns_list, dim=0).mean(0)
    svs_post_ns_aol_mean = torch.stack(ns_aol_list, dim=0).mean(0)

    ax.set_title(f"{gen}, shape={shape}")
    ax.plot(svs_pre_mean, label="Original SVs (normalized)")
    ax.plot(svs_sim_aol_mean, label="Estimated post-AOL SVs")
    ax.plot(svs_post_aol_mean, label="Actual post-AOL SVs")
    ax.plot(svs_post_ns_mean, label="Post NS SVs")
    ax.plot(svs_post_ns_aol_mean, label="Post NS+AOL SVs")
    # no per-axes legend here

# one shared legend, outside the plots
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.98),
)

plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for the legend
plt.savefig("assets/aol_ns_comparison.png", dpi=200)
