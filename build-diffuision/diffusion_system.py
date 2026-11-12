#!/usr/bin/env python3
"""A working 2D denoising diffusion model built from the copied diffusion parts.

The model is deliberately small so it trains on CPU in seconds:

    x0 in R^2  ->  linear beta schedule  ->  MLP noise predictor
               ->  toy diffusion training step (SGD + grad clipping)
               ->  ancestral DDPM sampling

The reusable algorithms are not reimplemented here.  They are loaded from the
``diffusion-*.py`` files that sit next to this script:

    diffusion-l02-linear-beta-schedule.py
    diffusion-l07-toy-diffusion-training-step.py
    diffusion-l07-toy-ddpm-sampler.py
    diffusion-l07-frechet-distance.py

Run ``python diffusion_system.py demo`` for a complete train-and-sample run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Loading the copied solutions as modules (their file names contain hyphens)
# ---------------------------------------------------------------------------
def load_part(name: str):
    path = HERE / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing part: {path}")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schedule_part = load_part("diffusion-l02-linear-beta-schedule")
train_part = load_part("diffusion-l07-toy-diffusion-training-step")
sampler_part = load_part("diffusion-l07-toy-ddpm-sampler")
frechet_part = load_part("diffusion-l07-frechet-distance")


# ---------------------------------------------------------------------------
# Data: 2D distributions that a tiny diffusion model can learn
# ---------------------------------------------------------------------------
def make_two_moons(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    half = n // 2
    theta = rng.uniform(0.0, np.pi, half)
    outer = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    inner = np.stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)
    x = np.concatenate([outer, inner], axis=0)
    if len(x) < n:
        x = np.concatenate([x, x[: n - len(x)]], axis=0)
    x = x + rng.normal(0.0, noise, x.shape)
    return x.astype(np.float32)


def make_spiral(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    half = n // 2
    theta = np.sqrt(rng.uniform(0.0, 1.0, half)) * 3.0 * np.pi
    r = theta / (3.0 * np.pi)
    arm = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x = np.concatenate([arm, -arm], axis=0)
    if len(x) < n:
        x = np.concatenate([x, x[: n - len(x)]], axis=0)
    x = x + rng.normal(0.0, noise, x.shape)
    return x.astype(np.float32)


def make_checkerboard(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    cells = rng.integers(0, 4, size=(n, 2))
    centers = cells.astype(np.float32) - 1.5
    jitter = rng.uniform(-0.35, 0.35, size=(n, 2)).astype(np.float32)
    x = centers + jitter + rng.normal(0.0, noise, (n, 2)).astype(np.float32)
    return x.astype(np.float32)


def make_gaussians(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    k = 8
    angles = np.arange(k) * (2.0 * np.pi / k)
    centers = np.stack([1.5 * np.cos(angles), 1.5 * np.sin(angles)], axis=1).astype(np.float32)
    labels = rng.integers(0, k, size=n)
    x = centers[labels] + rng.normal(0.0, max(noise, 0.08), (n, 2)).astype(np.float32)
    return x.astype(np.float32)


DATASETS = {
    "two-moons": make_two_moons,
    "spiral": make_spiral,
    "checkerboard": make_checkerboard,
    "gaussians": make_gaussians,
}


def build_data(name: str, n: int, noise: float, seed: int):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    rng = np.random.default_rng(seed)
    x = DATASETS[name](n, noise, rng)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    x_norm = ((x - mean) / std).astype(np.float32)
    return x, x_norm, mean.astype(np.float32), std.astype(np.float32)


def sliced_wasserstein(
    a, b, n_projections: int = 128, n_samples: int = 1000, seed: int = 0
) -> float:
    """Shape-sensitive distance between two 2D point clouds.

    Frechet distance only compares means and covariances, so a single Gaussian
    blob can score very well against a two-moon dataset.  Sliced Wasserstein
    compares the full one-dimensional marginals along random directions, so it
    notices when the shape is wrong.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(a), len(b))
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    dirs = rng.normal(size=(n_projections, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)
    return float(np.mean(np.abs(pa - pb)))


# ---------------------------------------------------------------------------
# The model: the same MLP the copied training/sampling functions expect
# ---------------------------------------------------------------------------
def init_params(
    dim: int, embed_dim: int, hidden: int, depth: int, seed: int
) -> dict[str, torch.Tensor]:
    """Standard PyTorch Linear initialization (kaiming-uniform, a=sqrt(5)).

    A tiny output layer would keep the hidden layers in a nearly linear "lazy"
    regime, which makes the denoiser collapse to a single Gaussian.
    """
    g = torch.Generator().manual_seed(seed)

    def linear(fan_in: int, fan_out: int):
        bound = 1.0 / math.sqrt(fan_in)
        w = torch.empty(fan_in, fan_out).uniform_(-bound, bound, generator=g)
        b = torch.empty(fan_out).uniform_(-bound, bound, generator=g)
        return w, b

    params: dict[str, torch.Tensor] = {}
    fan_in = dim + embed_dim
    for layer in range(1, depth):
        w, b = linear(fan_in, hidden)
        params[f"W{layer}"] = w
        params[f"b{layer}"] = b
        fan_in = hidden
    w, b = linear(fan_in, dim)
    params[f"W{depth}"] = w
    params[f"b{depth}"] = b
    return params


def time_embedding(timesteps: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """The same sinusoidal embedding used by the copied toy functions."""
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    angles = timesteps.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embed_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1))], dim=-1)
    return emb


def model_forward(
    x: torch.Tensor,
    timesteps: torch.Tensor,
    params: dict[str, torch.Tensor],
    embed_dim: int,
    depth: int,
) -> torch.Tensor:
    h = torch.cat([x, time_embedding(timesteps, embed_dim)], dim=-1)
    for layer in range(1, depth):
        h = torch.nn.functional.silu(h @ params[f"W{layer}"] + params[f"b{layer}"])
    return h @ params[f"W{depth}"] + params[f"b{depth}"]


def compute_deep_grads(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    alpha_bars: torch.Tensor,
    params: dict[str, torch.Tensor],
    embed_dim: int,
    depth: int,
    max_grad_norm: float,
):
    """Forward-diffuse, predict noise, and return clipped gradients."""
    w = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    a = alpha_bars[timesteps].unsqueeze(-1)
    x_t = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise
    eps_hat = model_forward(x_t, timesteps, w, embed_dim, depth)
    loss = ((eps_hat - noise) ** 2).mean()
    keys = list(w.keys())
    grads = torch.autograd.grad(loss, [w[k] for k in keys])
    grad_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    clip = torch.clamp(max_grad_norm / (grad_norm + 1e-6), max=1.0)
    grads = {k: g * clip for k, g in zip(keys, grads)}
    return float(loss.detach()), grads, float(grad_norm.detach())


def adam_update(
    params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
    m: dict[str, torch.Tensor],
    v: dict[str, torch.Tensor],
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        for k in params:
            g = grads[k]
            m[k].mul_(beta1).add_(g, alpha=1.0 - beta1)
            v[k].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            m_hat = m[k] / (1.0 - beta1**step)
            v_hat = v[k] / (1.0 - beta2**step)
            params[k] = params[k] - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return params


def system_ddpm_sample(
    x_T: torch.Tensor,
    betas: torch.Tensor,
    noise: torch.Tensor,
    params: dict[str, torch.Tensor],
    embed_dim: int,
    depth: int,
    clip_value: float | None,
) -> dict:
    """The same DDPM reverse chain as the copied toy sampler, for depth > 2."""
    T = betas.shape[0]
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    x = x_T.clone()
    pred_x0_history = []
    for k in range(T):
        t = T - 1 - k
        eps_hat = model_forward(
            x,
            torch.full((x.shape[0],), t, dtype=torch.long, device=x.device),
            params,
            embed_dim,
            depth,
        )
        a = alpha_bars[t]
        alpha_t = alphas[t]
        alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.ones_like(a)
        x_hat_0 = (x - torch.sqrt(1.0 - a) * eps_hat) / torch.sqrt(a)
        if clip_value is not None:
            x_hat_0 = torch.clamp(x_hat_0, -clip_value, clip_value)
        pred_x0_history.append(x_hat_0)
        coeff_x0 = betas[t] * torch.sqrt(alpha_bar_prev) / (1.0 - a)
        coeff_xt = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - a)
        mu = coeff_x0 * x_hat_0 + coeff_xt * x
        if t > 0:
            beta_tilde = betas[t] * (1.0 - alpha_bar_prev) / (1.0 - a)
            x = mu + torch.sqrt(beta_tilde) * noise[k]
        else:
            x = mu
    return {"samples": x, "pred_x0": torch.stack(pred_x0_history, dim=0)}


def sample_from_params(
    params: dict[str, torch.Tensor],
    betas: torch.Tensor,
    num_samples: int,
    embed_dim: int,
    clip_value: float,
    seed: int,
    depth: int,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x_t = torch.randn(num_samples, 2, generator=g)
    noise = torch.randn(len(betas), num_samples, 2, generator=g)
    with torch.no_grad():
        if depth == 2:
            # The unmodified copied sampler.
            out = sampler_part.toy_ddpm_sample(
                x_t, betas, noise, params, embed_dim, clip_value
            )
        else:
            out = system_ddpm_sample(
                x_t, betas, noise, params, embed_dim, depth, clip_value
            )
    return out["samples"]


# ---------------------------------------------------------------------------
# Train / sample / evaluate
# ---------------------------------------------------------------------------
def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    real, real_norm, mean, std = build_data(
        args.dataset, args.num_data, args.data_noise, args.seed
    )
    x_data = torch.from_numpy(real_norm)

    schedule = schedule_part.linear_beta_schedule(
        args.timesteps, args.beta_start, args.beta_end
    )
    betas = torch.tensor(schedule["betas"], dtype=torch.float32)
    alpha_bars = torch.tensor(schedule["alpha_bars"], dtype=torch.float32)

    params = init_params(
        2, args.embed_dim, args.hidden, args.depth, args.seed
    )

    print(
        f"dataset={args.dataset}  points={len(real)}  steps={args.steps}  "
        f"T={args.timesteps}  hidden={args.hidden}  embed={args.embed_dim}  "
        f"depth={args.depth}  optimizer={args.optimizer}"
    )
    print(
        f"alpha_bar[0]={alpha_bars[0].item():.6f}  "
        f"alpha_bar[T-1]={alpha_bars[-1].item():.6f}"
    )

    # Baseline: sample from the untrained model so we can prove training helped.
    before = sample_from_params(
        params,
        betas,
        args.num_samples,
        args.embed_dim,
        args.clip,
        args.seed + 1,
        args.depth,
    ).numpy()
    fd_before = frechet_part.frechet_distance(real_norm, before)
    swd_before = sliced_wasserstein(real_norm, before, seed=args.seed)
    print(f"Frechet distance before training: {fd_before:.4f} (shape-blind)")
    print(f"Sliced Wasserstein before training: {swd_before:.4f}")

    losses: list[float] = []
    adam_m: dict[str, torch.Tensor] = {}
    adam_v: dict[str, torch.Tensor] = {}
    if args.optimizer == "adam":
        adam_m = {k: torch.zeros_like(v) for k, v in params.items()}
        adam_v = {k: torch.zeros_like(v) for k, v in params.items()}

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(x_data), (args.batch_size,))
        x0 = x_data[idx]
        timesteps = torch.randint(0, args.timesteps, (args.batch_size,))
        noise = torch.randn_like(x0)

        if args.lr_schedule == "cosine":
            lr_t = args.lr * 0.5 * (
                1.0 + math.cos(math.pi * (step - 1) / max(1, args.steps))
            )
        else:
            lr_t = args.lr

        if args.depth == 2:
            if args.optimizer == "sgd":
                # The unmodified copied training step: MSE + clipped SGD.
                result = train_part.toy_diffusion_training_step(
                    x0,
                    timesteps,
                    noise,
                    alpha_bars,
                    params,
                    args.embed_dim,
                    lr_t,
                    args.max_grad_norm,
                )
                params = result["params"]
            else:
                # Use the copied step for the forward pass, loss, backward pass
                # and gradient clipping.  With learning_rate=1.0 it returns
                # old_params - clipped_grad, so old - returned recovers the
                # gradient and we can feed it to Adam instead of plain SGD.
                result = train_part.toy_diffusion_training_step(
                    x0,
                    timesteps,
                    noise,
                    alpha_bars,
                    params,
                    args.embed_dim,
                    1.0,
                    args.max_grad_norm,
                )
                grads = {k: params[k] - result["params"][k] for k in params}
                params = adam_update(
                    params,
                    grads,
                    adam_m,
                    adam_v,
                    step,
                    lr_t,
                    args.adam_beta1,
                    args.adam_beta2,
                    args.adam_eps,
                )
        else:
            loss_value, grads, grad_norm = compute_deep_grads(
                x0,
                timesteps,
                noise,
                alpha_bars,
                params,
                args.embed_dim,
                args.depth,
                args.max_grad_norm,
            )
            result = {"loss": loss_value, "grad_norm": grad_norm}
            if args.optimizer == "sgd":
                with torch.no_grad():
                    params = {
                        k: params[k] - lr_t * grads[k] for k in params
                    }
            else:
                params = adam_update(
                    params,
                    grads,
                    adam_m,
                    adam_v,
                    step,
                    lr_t,
                    args.adam_beta1,
                    args.adam_beta2,
                    args.adam_eps,
                )

        losses.append(float(result["loss"]))

        if step % args.log_every == 0:
            window = losses[-args.log_every :]
            print(
                f"step {step:5d}/{args.steps}  loss {np.mean(window):.5f}  "
                f"grad_norm {result['grad_norm']:.3f}"
            )

    generated_norm = sample_from_params(
        params,
        betas,
        args.num_samples,
        args.embed_dim,
        args.clip,
        args.seed + 2,
        args.depth,
    ).numpy()
    fd_after = frechet_part.frechet_distance(real_norm, generated_norm)
    swd_after = sliced_wasserstein(real_norm, generated_norm, seed=args.seed)
    print(f"Frechet distance after training:  {fd_after:.4f} (shape-blind)")
    print(f"Sliced Wasserstein after training:  {swd_after:.4f}")

    # Back to the original data coordinates for saving and plotting.
    generated = generated_norm * std + mean
    real_denorm = real

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(params, out_dir / "model.pt")
    np.save(out_dir / "samples.npy", generated)
    np.save(out_dir / "real.npy", real_denorm)

    config = {
        "dataset": args.dataset,
        "num_data": args.num_data,
        "data_noise": args.data_noise,
        "seed": args.seed,
        "steps": args.steps,
        "optimizer": args.optimizer,
        "timesteps": args.timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "embed_dim": args.embed_dim,
        "hidden": args.hidden,
        "depth": args.depth,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "max_grad_norm": args.max_grad_norm,
        "clip_value": args.clip,
        "num_samples": args.num_samples,
        "data_mean": mean.reshape(-1).tolist(),
        "data_std": std.reshape(-1).tolist(),
        "fd_before": fd_before,
        "fd_after": fd_after,
        "swd_before": swd_before,
        "swd_after": swd_after,
        "final_loss": float(np.mean(losses[-min(100, len(losses)) :])),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    np.save(out_dir / "losses.npy", np.asarray(losses, dtype=np.float64))

    if not args.no_plot:
        plot_result(
            real_denorm,
            generated,
            losses,
            out_dir / "result.png",
            title=(
                f"{args.dataset}  FD {fd_before:.3f} -> {fd_after:.3f}  |  "
                f"SWD {swd_before:.3f} -> {swd_after:.3f}"
            ),
        )
        print(f"plot written to {out_dir / 'result.png'}")

    print(f"model written to {out_dir / 'model.pt'}")
    print(f"samples written to {out_dir / 'samples.npy'}")
    return config


def load_checkpoint(path: Path):
    try:
        params = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        params = torch.load(path, map_location="cpu")
    if not isinstance(params, dict):
        raise ValueError(f"{path} does not contain a parameter dict")
    return params


def run_sample(args) -> None:
    ckpt_dir = Path(args.checkpoint).resolve().parent
    params = load_checkpoint(Path(args.checkpoint))
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {
            "timesteps": 300,
            "beta_start": 1e-4,
            "beta_end": 0.06,
            "embed_dim": 32,
            "depth": 2,
            "clip_value": 3.0,
            "data_mean": [0.0, 0.0],
            "data_std": [1.0, 1.0],
        }

    schedule = schedule_part.linear_beta_schedule(
        config["timesteps"], config["beta_start"], config["beta_end"]
    )
    betas = torch.tensor(schedule["betas"], dtype=torch.float32)
    generated_norm = sample_from_params(
        params,
        betas,
        args.num_samples,
        config["embed_dim"],
        config["clip_value"],
        args.seed,
        config.get("depth", 2),
    ).numpy()
    mean = np.asarray(config["data_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(config["data_std"], dtype=np.float32).reshape(1, -1)
    generated = generated_norm * std + mean

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "samples.npy", generated)
    print(f"{len(generated)} samples written to {out_dir / 'samples.npy'}")
    print(
        f"sample mean {generated.mean(0)}  sample std {generated.std(0)}"
    )

    real_path = Path(args.real) if args.real else ckpt_dir / "real.npy"
    if not args.no_plot and real_path.exists():
        real = np.load(real_path)
        losses_path = ckpt_dir / "losses.npy"
        losses = np.load(losses_path) if losses_path.exists() else np.array([])
        plot_result(
            real,
            generated,
            losses,
            out_dir / "sample.png",
            title=f"generated samples from {Path(args.checkpoint).name}",
        )
        print(f"plot written to {out_dir / 'sample.png'}")


def plot_result(real, generated, losses, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    lo = np.quantile(real, 0.01, axis=0)
    hi = np.quantile(real, 0.99, axis=0)
    pad = 0.25 * np.maximum(hi - lo, 1e-3)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    # Random subsets: the generated data is stored moon-by-moon, so slicing the
    # first N points would show only one arm of the two moons.
    rng = np.random.default_rng(0)
    real_plot = real[rng.choice(len(real), min(2000, len(real)), replace=False)]
    gen_plot = generated[
        rng.choice(len(generated), min(2000, len(generated)), replace=False)
    ]

    axes[0].scatter(real_plot[:, 0], real_plot[:, 1], s=3, alpha=0.45, c="#2b6cb0")
    axes[0].set_title("real data")
    axes[1].scatter(gen_plot[:, 0], gen_plot[:, 1], s=3, alpha=0.45, c="#c05621")
    axes[1].set_title("generated samples")
    for ax in axes[:2]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if len(losses):
        axes[2].plot(losses, linewidth=0.8, c="#2f855a")
        axes[2].set_title("training loss")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("MSE")
        axes[2].grid(alpha=0.3)
    else:
        axes[2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test: every copied part loads, one train step works, sampling works
# ---------------------------------------------------------------------------
def selftest(args) -> None:
    parts = sorted(p.stem for p in HERE.glob("diffusion-*.py"))
    assert len(parts) == 26, f"expected 26 copied parts, found {len(parts)}"
    for name in parts:
        load_part(name)

    schedule = schedule_part.linear_beta_schedule(300, 1e-4, 0.06)
    betas = torch.tensor(schedule["betas"], dtype=torch.float32)
    alpha_bars = torch.tensor(schedule["alpha_bars"], dtype=torch.float32)
    assert betas.shape == (300,)
    assert 0.0 < alpha_bars[-1] < 1e-2

    params = init_params(2, 16, 32, 2, seed=0)
    x0 = torch.randn(8, 2)
    t = torch.randint(0, 300, (8,))
    noise = torch.randn_like(x0)
    result = train_part.toy_diffusion_training_step(
        x0, t, noise, alpha_bars, params, 16, 1e-3, 1.0
    )
    assert set(result["params"]) == {"W1", "b1", "W2", "b2"}
    assert result["loss"] >= 0.0
    assert all(torch.isfinite(p).all() for p in result["params"].values())

    out = sampler_part.toy_ddpm_sample(
        torch.randn(8, 2),
        betas,
        torch.randn(300, 8, 2),
        result["params"],
        16,
        3.0,
    )
    assert out["samples"].shape == (8, 2)
    assert out["pred_x0"].shape == (300, 8, 2)
    assert torch.isfinite(out["samples"]).all()

    # Deep path: same DDPM math, deeper MLP.
    params3 = init_params(2, 16, 32, 3, seed=0)
    loss3, grads3, grad_norm3 = compute_deep_grads(
        x0, t, noise, alpha_bars, params3, 16, 3, 1.0
    )
    assert set(grads3) == set(params3)
    assert loss3 >= 0.0
    assert torch.isfinite(torch.tensor(grad_norm3))
    out3 = system_ddpm_sample(
        torch.randn(8, 2), betas, torch.randn(300, 8, 2), params3, 16, 3, 3.0
    )
    assert out3["samples"].shape == (8, 2)
    assert out3["pred_x0"].shape == (300, 8, 2)
    assert torch.isfinite(out3["samples"]).all()

    fd = frechet_part.frechet_distance(np.random.randn(50, 2), np.random.randn(50, 2))
    assert isinstance(fd, float) and fd >= 0.0

    print(f"PASS: {len(parts)} parts loaded, train step + sampler + Frechet distance OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="two-moons", choices=sorted(DATASETS))
    parser.add_argument("--num-data", type=int, default=5000)
    parser.add_argument("--data-noise", type=float, default=0.07)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.06)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(HERE / "runs" / "demo"))
    parser.add_argument("--no-plot", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Working 2D diffusion model assembled from the copied parts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="train and sample with demo defaults")
    add_train_args(demo)
    demo.set_defaults(func=train)

    train_cmd = sub.add_parser("train", help="train and save a model")
    add_train_args(train_cmd)
    train_cmd.set_defaults(func=train)

    sample_cmd = sub.add_parser("sample", help="sample from a saved checkpoint")
    sample_cmd.add_argument("--checkpoint", required=True)
    sample_cmd.add_argument("--num-samples", type=int, default=2000)
    sample_cmd.add_argument("--seed", type=int, default=1)
    sample_cmd.add_argument("--real", default=None, help="optional real-data .npy for plotting")
    sample_cmd.add_argument("--out", default=str(HERE / "runs" / "sample"))
    sample_cmd.add_argument("--no-plot", action="store_true")
    sample_cmd.set_defaults(func=run_sample)

    test_cmd = sub.add_parser("selftest", help="check every copied part loads and runs")
    test_cmd.set_defaults(func=selftest)

    args = parser.parse_args(argv)
    if getattr(args, "embed_dim", 2) % 2 != 0:
        parser.error("--embed-dim must be even")
    if getattr(args, "depth", 2) < 2:
        parser.error("--depth must be at least 2")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
