# Micrograd Autograd System

A working reverse-mode automatic differentiation engine and neural-network
training demo assembled from the 25 solutions in
`study-plans/autograd-from-scratch/`. All 25 solution files and their problem
write-ups are copied here unmodified. `autograd_system.py` adds the glue: a
`Value` engine, an MLP, toy datasets, training, plotting, gradient checks, and
a CLI.

The `Value` backward pass uses the copied topological sort; its local
derivatives mirror the copied add/mul/tanh/exp/power VJP rules. Training uses
the copied gradient-descent step, and a reference trainer calls the copied
torch MLP trainer on the same initial weights.

## Quick start

```bash
cd /home/d/Desktop/TensorTonic-Solutions/build-autograd
python autograd_system.py selftest     # 25 parts + engine + torch reference
python autograd_system.py demo         # train XOR, plot, compare to reference
python autograd_system.py gradcheck    # analytic vs finite-difference grads
python autograd_system.py graph        # copied dict-graph engine walkthrough
```

The default XOR demo trains in about 2 seconds (4.5 seconds including the
reference run and plot).

## Results

Default XOR run (`2 -> 8 -> 1`, 33 parameters, 1200 steps, lr 0.05):

| Engine | Loss | Accuracy |
|---|---|---|
| `Value` engine (our glue) | 1.158923 -> **0.003015** | **100%** |
| Copied torch MLP trainer | 1.158923 -> 0.003012 | 100% |

Final-loss gap between the independent engines: **3e-6**.

Other modes:

```bash
# 2D circle classification (80 points, 12 hidden, 800 steps)
python autograd_system.py demo --dataset circle --hidden 12 --steps 800 \
    --out runs/circle
# -> loss 0.932956 -> 0.254442, accuracy 96.25% (about 72 s)

# Regression y = sin(2x) (60 points, 12 hidden, 800 steps)
python autograd_system.py demo --dataset regression --hidden 12 --points 60 \
    --steps 800 --out runs/regression
# -> MSE 0.577326 -> 0.025263 (about 36 s)
```

`runs/demo/` (and the optional circle/regression directories) contain
`model.json`, `metrics.json`, `losses.npy`, `reference_losses.npy`, and
`training.png`. Generated runs are ignored by git.

## The Value engine

```python
from autograd_system import Value

a, b, c = Value(0.5), Value(-1.2), Value(0.3)
loss = (a * b + c).tanh() * a.exp() + a.pow(2) + b * c
loss.backward()
print(a.grad, b.grad, c.grad)
```

- `Value` stores `data`, `grad`, `_prev` (parents), and a local `_backward`
  closure.
- `__add__`, `__mul__`, `__sub__`, `__neg__`, `__truediv__`, `tanh`, `exp`,
  and `pow` build the graph.
- `backward()` traverses the reachable graph, calls the copied
  `autograd-l05-topological-sort.py`, seeds the output gradient at 1.0, and
  runs the local VJPs in reverse order.
- Shared subexpressions accumulate gradients with `+=`, matching the copied
  `shared-path-gradient-accumulation` behavior.
- `gradcheck` compares the analytic gradients of
  `tanh(a*b + c) * exp(a) + a^2 + b*c` with central finite differences.

## MLP and training

```python
from autograd_system import MLP, train_value_mlp, make_dataset

xs, ys, meta = make_dataset("xor", seed=0)
mlp = MLP(meta["nin"], [8, 1], seed=0)
losses, _ = train_value_mlp(mlp, xs, ys, steps=1200, lr=0.05)
```

- `Neuron` computes `tanh(sum(w_i x_i) + b)` with `Value` parameters.
- `Layer` and `MLP` stack neurons; `parameters()`, `zero_grad()`, and
  `state_dict()` / `load_state_dict()` are provided.
- Each training step builds the mean-squared-error graph over the dataset,
  calls `backward()`, then updates every parameter with the copied
  `autograd-l01-gradient-descent-step.py` function.
- The reference path converts the initial weights to torch tensors and calls
  `autograd-l08-train-tiny-micrograd-mlp.py`. Its loss is a sum rather than a
  mean, so the reference learning rate is `lr / N`; the resulting final losses
  agree closely with the `Value` engine.

## What the copied files contribute

| Lesson | Files | Role |
|---|---|---|
| l01 | finite-difference derivative, product-chain check, gradient-descent step, scalar partials | numeric checks and the SGD update |
| l02 | build expression graph, trace reachable graph, addition/multiplication nodes | dict graph representation |
| l04 | neuron forward/backward, neuron gradient check, tanh forward/backward | torch scalar neuron reference |
| l05 | local VJP rules, reverse-mode autodiff, topological sort, autodiff gradient check | core reverse-mode math and the `Value.backward()` order |
| l06 | exp/power backward, shared-path gradient accumulation | `exp`/`pow` rules and shared-subgraph accumulation |
| l07 | dense layer forward, MLP forward, recursive parameter collection, scalar neuron module | torch layer/MLP reference |
| l08 | SGD training step, squared-error gradients, train tiny micrograd MLP | copied torch training reference |

The `graph` command exercises the dict-graph path directly:
`build_expression_graph` -> `trace_reachable_graph` -> `topological_sort` ->
`reverse_mode_autodiff` / `autodiff_gradient_check`, and compares its gradients
with `gradient_check_product_chain`. It also runs the shared-path and
exp/power examples.

## Limitations

- The engine is scalar; every dataset point and parameter is a Python object.
  XOR is fast, the 80-point circle takes about a minute for 800 steps.
- The datasets and networks are deliberately tiny. This is a learning system,
  not a replacement for PyTorch autograd.
- Runs on CPU; no GPU path is needed or used.
- The copied l07/l08 torch functions are reference implementations, not
  `nn.Module` classes; the system uses them for cross-checks.

## Provenance

Copied unmodified on 2026-11-11 from:

```
/home/d/Desktop/TensorTonic-Solutions/study-plans/autograd-from-scratch/
```

Each source directory contains the original problem write-up (`*.md`) and its
solution (`*.py`); both are copied here.
