# Working 2D Diffusion System

A complete, runnable denoising diffusion model assembled from the 26 Python
solutions copied out of `study-plans/diffusion-from-scratch/`. The copied
solution files are left unmodified. `diffusion_system.py` adds only the glue:
2D data generation, parameter initialization, the training loop, checkpointing,
metrics, plotting, and a CLI.

## Quick start

```bash
cd /home/d/Desktop/TensorTonic-Solutions/build_diffuision
python diffusion_system.py selftest      # loads all 26 parts, checks both paths
python diffusion_system.py demo          # train + sample + score + plot
```

The default demo trains a 3-layer MLP denoiser for 20,000 Adam steps on the
two-moons distribution. It runs on CPU in about 35–60 seconds and writes to
`runs/demo/`.

Typical result from the shipped run:

```
Frechet distance:       3.678 -> 0.001   (shape-blind)
Sliced Wasserstein:     1.135 -> 0.049   (real-vs-real noise floor ~0.06)
```

The generated samples follow both moons, including the gap between the arms.

## What is actually running

```
2D data -> linear beta schedule -> MLP noise predictor
        -> noise-prediction MSE (Adam, cosine LR, grad clipping)
        -> ancestral DDPM reverse chain -> samples
        -> Frechet distance + sliced Wasserstein score
```

`runs/demo/` contains:

| File | Contents |
|---|---|
| `model.pt` | trained parameter dict (`W1`, `b1`, ... for the chosen depth) |
| `config.json` / `metrics.json` | dataset, depth, hyperparameters, FD/SWD before and after |
| `samples.npy` | 2,000 generated points in original data coordinates |
| `real.npy` | the real training points |
| `losses.npy` | per-step training loss |
| `result.png` | real vs. generated scatters and the loss curve |

## Two paths through the copied code

`--depth 2` uses the copied functions exactly:

- `diffusion-l07-toy-diffusion-training-step.py` for training (MSE + clipped
  SGD). With `--optimizer adam`, the system calls the copied step with
  `learning_rate=1.0`, recovers its clipped gradient as
  `old_params - returned_params`, and feeds that gradient to Adam. The loss,
  forward pass, backward pass, and clipping all come from the copied file.
- `diffusion-l07-toy-ddpm-sampler.py` for sampling.

The 2-layer model is a little too small for the two-moon manifold: it learns
the overall distribution but smooths away the gap between the arms. That is why
the default is depth 3.

`--depth 3` (default) keeps the same schedule and DDPM equations but uses a
deeper MLP; the copied toy functions are hardwired to two layers.
`system_ddpm_sample` follows `toy_ddpm_sample` line for line, and
`compute_deep_grads` uses the same forward-diffusion and MSE loss. Depth 4 also
works. Use `--depth 2` to see the pure copied path.

## Commands

```bash
# Different 2D distributions
python diffusion_system.py train --dataset spiral --steps 20000 --out runs/spiral
python diffusion_system.py train --dataset checkerboard --out runs/checker
python diffusion_system.py train --dataset gaussians --out runs/gaussians

# Pure copied 2-layer path (SGD)
python diffusion_system.py train --depth 2 --optimizer sgd --steps 20000 --out runs/two_layer

# Longer training with a wider deep model
python diffusion_system.py train --depth 3 --steps 50000 --hidden 384 --out runs/long

# Sample again from a saved checkpoint
python diffusion_system.py sample \
    --checkpoint runs/demo/model.pt \
    --num-samples 4000 \
    --out runs/demo_resample
```

Use `--no-plot` to skip matplotlib, and `--seed` for reproducibility.

## Which copied part does what

| Copied file | Role in the system |
|---|---|
| `diffusion-l02-linear-beta-schedule.py` | builds `betas`, `alphas`, `alpha_bars` |
| `diffusion-l07-toy-diffusion-training-step.py` | depth-2 training and the gradient source for depth-2 Adam |
| `diffusion-l07-toy-ddpm-sampler.py` | depth-2 ancestral sampling |
| `diffusion-l07-frechet-distance.py` | Frechet-distance metric |

The other 22 files are present and loadable but not on the default path. They
are the parts needed to extend this into an image model (schedules,
`tiny_unet_forward`, posterior variance, DDPM/DDIM/Karras steps,
classifier-free guidance, FID, etc.).

## Metrics: why both FD and SWD

Fréchet distance compares only the mean and covariance of two point clouds, so
a single Gaussian blob can score very well against a two-moon dataset. Sliced
Wasserstein compares the full one-dimensional marginals along random
directions, so it notices a wrong shape. The system reports both. On the
standardized data, a real-vs-real split scores about 0.06 SWD; the shipped demo
reaches 0.049. A blob scores around 0.6–0.7.

## How to extend it to images

1. Replace the 2D MLP denoiser with `diffusion-l03-tiny-unet-forward.py`
   (its `.md` in the source track documents every parameter name and shape).
2. Use `diffusion-l04-posterior-mean-variance.py` and
   `diffusion-l05-ddpm-ancestral-step.py` for the reverse chain.
3. Train on image tensors of shape `(B, C, H, W)` with `noise_prediction_loss`
   or `min_snr_weighted_loss`.
4. Evaluate with `diffusion-l07-frechet-distance.py` on flattened features.

## Environment and limitations

- Runs on CPU by design. The installed system PyTorch (`2.15.0+cu132`) cannot
  target the P2000's `sm_61`, and this workload does not need a GPU.
- This is a real train/sample loop, not a production image generator. The
  default samples are slightly diffuse compared with the training data, which
  is expected at this toy scale.
- The model is 2D only; `samples.npy` is an array of shape `(N, 2)`.

## Provenance

All `diffusion-*.py` files were copied unmodified on 2026-11-11 from:

```
/home/d/Desktop/TensorTonic-Solutions/study-plans/diffusion-from-scratch/
```

Each source file is a single-problem TensorTonic solution; the original
problem write-ups are the matching `.md` files in that track.
