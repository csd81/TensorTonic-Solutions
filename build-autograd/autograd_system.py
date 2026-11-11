#!/usr/bin/env python3
"""A working micrograd-style autograd system built from the copied solutions.

The copied ``autograd-*.py`` files next to this script contain the graph
representation (l02), reverse-mode engine (l05), local VJPs (l05/l06),
finite-difference checks (l01), tanh/neuron math (l04), MLP pieces (l07), and
SGD training code (l08).  This file adds a small ``Value`` engine whose
backward pass uses the copied topological sort, a Value-based MLP, toy data,
plotting, a CLI, and a reference trainer that calls the copied torch MLP
trainer.

Quick start:

    python autograd_system.py selftest
    python autograd_system.py demo
    python autograd_system.py graph
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Load every copied solution as a module (file names contain hyphens)
# ---------------------------------------------------------------------------
def load_part(name: str):
    path = HERE / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing copied part: {path}")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARTS = {
    "finite": load_part("autograd-l01-finite-difference-derivative"),
    "product_check": load_part("autograd-l01-gradient-check-product-chain"),
    "gd": load_part("autograd-l01-gradient-descent-step"),
    "partials": load_part("autograd-l01-scalar-expression-partials"),
    "build_graph": load_part("autograd-l02-build-expression-graph"),
    "trace": load_part("autograd-l02-trace-reachable-graph"),
    "add_node": load_part("autograd-l02-value-addition-node"),
    "mul_node": load_part("autograd-l02-value-multiplication-node"),
    "neuron_backward": load_part("autograd-l04-neuron-backward"),
    "neuron_forward": load_part("autograd-l04-neuron-forward"),
    "neuron_check": load_part("autograd-l04-neuron-gradient-check"),
    "tanh": load_part("autograd-l04-tanh-forward-backward"),
    "autodiff_check": load_part("autograd-l05-autodiff-gradient-check"),
    "vjp": load_part("autograd-l05-local-vjp-rules"),
    "reverse": load_part("autograd-l05-reverse-mode-autodiff"),
    "topo": load_part("autograd-l05-topological-sort"),
    "exp_power": load_part("autograd-l06-exp-power-backward"),
    "shared": load_part("autograd-l06-shared-path-gradient-accumulation"),
    "dense": load_part("autograd-l07-dense-layer-forward"),
    "mlp": load_part("autograd-l07-mlp-forward"),
    "params": load_part("autograd-l07-recursive-parameter-collection"),
    "scalar_module": load_part("autograd-l07-scalar-neuron-module"),
    "sgd_step": load_part("autograd-l08-sgd-training-step"),
    "squared": load_part("autograd-l08-squared-error-loss-gradients"),
    "train_tiny": load_part("autograd-l08-train-tiny-micrograd-mlp"),
}


# ---------------------------------------------------------------------------
# The Value engine
# ---------------------------------------------------------------------------
class Value:
    """A scalar node in a reverse-mode autodiff graph."""

    _next_uid = 0

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None
        self._uid = Value._next_uid
        Value._next_uid += 1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op={self._op})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other.pow(-1.0)

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def pow(self, exponent: float):
        exponent = float(exponent)
        if exponent == 0.0:
            return Value(1.0, (self,), "pow0")
        out = Value(self.data**exponent, (self,), f"pow{exponent:g}")

        def _backward():
            self.grad += (
                exponent * (self.data ** (exponent - 1.0)) * out.grad
            )

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        return self.pow(exponent)

    def backward(self) -> None:
        """Reverse-mode backward using the copied topological sort."""
        nodes = []
        by_id = {}
        seen = set()

        def visit(value: "Value") -> None:
            if value._uid in seen:
                return
            seen.add(value._uid)
            for child in value._prev:
                visit(child)
            nodes.append(
                {"id": value._uid, "parents": [p._uid for p in value._prev]}
            )
            by_id[value._uid] = value

        visit(self)
        order = PARTS["topo"].topological_sort(nodes, self._uid)
        self.grad = 1.0
        for uid in reversed(order):
            by_id[uid]._backward()


# ---------------------------------------------------------------------------
# Value-based neural network
# ---------------------------------------------------------------------------
class Neuron:
    def __init__(self, nin: int, rng: random.Random):
        scale = 1.0 / math.sqrt(max(1, nin))
        self.w = [Value(rng.uniform(-1.0, 1.0) * scale) for _ in range(nin)]
        self.b = Value(rng.uniform(-1.0, 1.0))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, rng: random.Random):
        self.neurons = [Neuron(nin, rng) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    def __init__(self, nin: int, nouts: list[int], seed: int = 0):
        rng = random.Random(seed)
        sizes = [nin] + list(nouts)
        self.sizes = sizes
        self.layers = [
            Layer(sizes[i], sizes[i + 1], rng) for i in range(len(nouts))
        ]

    def __call__(self, x):
        current = x
        for layer in self.layers:
            current = layer(current)
        return current[0] if len(current) == 1 else current

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = 0.0

    def n_params(self) -> int:
        return len(self.parameters())

    def state_dict(self) -> list[list[dict]]:
        state = []
        for layer in self.layers:
            layer_state = []
            for neuron in layer.neurons:
                layer_state.append(
                    {"w": [p.data for p in neuron.w], "b": neuron.b.data}
                )
            state.append(layer_state)
        return state

    def load_state_dict(self, state: list[list[dict]]) -> None:
        for layer, layer_state in zip(self.layers, state):
            for neuron, neuron_state in zip(layer.neurons, layer_state):
                if len(neuron.w) != len(neuron_state["w"]):
                    raise ValueError("parameter shape mismatch")
                for param, value in zip(neuron.w, neuron_state["w"]):
                    param.data = float(value)
                neuron.b.data = float(neuron_state["b"])

    def predict(self, x) -> float:
        return self(x).data


# ---------------------------------------------------------------------------
# Toy datasets
# ---------------------------------------------------------------------------
def make_dataset(name: str, seed: int, points: int = 80):
    if name == "xor":
        xs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
        ys = [-1.0, 1.0, 1.0, -1.0]
        return xs, ys, {"kind": "classification", "nin": 2}
    if name == "circle":
        rng = np.random.default_rng(seed + 1)
        xs_np = rng.uniform(-1.5, 1.5, size=(points, 2))
        ys_np = np.where((xs_np**2).sum(axis=1) < 1.0, 1.0, -1.0)
        return xs_np.tolist(), ys_np.tolist(), {"kind": "classification", "nin": 2}
    if name == "regression":
        xs_np = np.linspace(-2.0, 2.0, points)
        ys_np = np.sin(2.0 * xs_np)
        return [[float(x)] for x in xs_np], ys_np.tolist(), {
            "kind": "regression",
            "nin": 1,
        }
    raise ValueError(f"unknown dataset {name!r}")


# ---------------------------------------------------------------------------
# Training with the copied gradient-descent step
# ---------------------------------------------------------------------------
def train_value_mlp(
    mlp: MLP,
    xs,
    ys,
    steps: int,
    lr: float,
    log_every: int = 0,
):
    losses = []
    last_predicted_change = 0.0
    for step in range(1, steps + 1):
        mlp.zero_grad()
        loss = Value(0.0)
        for x, y in zip(xs, ys):
            diff = mlp(x) - y
            loss = loss + diff * diff
        loss = loss * (1.0 / len(xs))
        loss.backward()

        params = mlp.parameters()
        updated, predicted_change = PARTS["gd"].gradient_descent_step(
            [p.data for p in params], [p.grad for p in params], lr
        )
        for param, value in zip(params, updated):
            param.data = value
        losses.append(loss.data)
        last_predicted_change = predicted_change

        if log_every and step % log_every == 0:
            print(f"  step {step:5d}  loss {loss.data:.6f}")
    return losses, last_predicted_change


def classification_accuracy(mlp: MLP, xs, ys) -> float:
    correct = 0
    for x, y in zip(xs, ys):
        pred = mlp.predict(x)
        correct += int((pred > 0.0) == (y > 0.0))
    return correct / len(xs) if xs else 0.0


def regression_mse(mlp: MLP, xs, ys) -> float:
    if not xs:
        return 0.0
    total = 0.0
    for x, y in zip(xs, ys):
        total += (mlp.predict(x) - y) ** 2
    return total / len(xs)


# ---------------------------------------------------------------------------
# Reference path: the copied torch MLP trainer
# ---------------------------------------------------------------------------
def torch_reference_train(
    initial_state: list[list[dict]],
    xs,
    ys,
    steps: int,
    lr: float,
):
    weights = []
    biases = []
    for layer_state in initial_state:
        nin = len(layer_state[0]["w"])
        w = torch.tensor(
            [[neuron["w"][j] for j in range(nin)] for neuron in layer_state],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [neuron["b"] for neuron in layer_state], dtype=torch.float32
        )
        weights.append(w)
        biases.append(b)

    inputs = torch.tensor(xs, dtype=torch.float32)
    targets = torch.tensor(ys, dtype=torch.float32)
    n = max(1, len(xs))
    # Our Value loss is a mean; the copied trainer uses a sum, so scale its LR.
    preds, loss, trained_w, trained_b, history = PARTS[
        "train_tiny"
    ].train_tiny_micrograd_mlp(
        inputs, targets, weights, biases, lr / n, steps
    )
    mean_history = [float(h) / n for h in history]
    return {
        "loss": float(loss) / n,
        "history": mean_history,
        "predictions": [float(p) for p in preds],
        "weights": [w.tolist() for w in trained_w],
        "biases": [b.tolist() for b in trained_b],
    }


# ---------------------------------------------------------------------------
# Copied graph-engine demonstrations and checks
# ---------------------------------------------------------------------------
def graph_checks(verbose: bool = True):
    a, b, c, f = 1.5, -2.0, 0.5, 3.0
    h = 1e-6

    # l01 product-chain finite-difference check.
    loss, analytic, numerical, max_err = PARTS[
        "product_check"
    ].gradient_check_product_chain(a, b, c, f, h)

    # l02 dict graph representation.
    leaves = [
        {"id": "a", "data": a},
        {"id": "b", "data": b},
        {"id": "c", "data": c},
        {"id": "f", "data": f},
    ]
    operations = [
        {"id": "ab", "op": "*", "left": "a", "right": "b"},
        {"id": "e", "op": "+", "left": "ab", "right": "c"},
        {"id": "out", "op": "*", "left": "e", "right": "f"},
    ]
    nodes, output_id = PARTS["build_graph"].build_expression_graph(
        leaves, operations
    )
    reachable, edges = PARTS["trace"].trace_reachable_graph(nodes, output_id)
    order = PARTS["topo"].topological_sort(nodes, output_id)

    # l05 reverse-mode engine on the same expression.
    rev_leaves = [
        {"id": "a", "value": a},
        {"id": "b", "value": b},
        {"id": "c", "value": c},
        {"id": "f", "value": f},
    ]
    rev_ops = [
        {"id": "ab", "op": "mul", "parents": ["a", "b"]},
        {"id": "e", "op": "add", "parents": ["ab", "c"]},
        {"id": "out", "op": "mul", "parents": ["e", "f"]},
    ]
    rev_value, rev_grads = PARTS["reverse"].reverse_mode_autodiff(
        rev_leaves, rev_ops, "out"
    )
    auto_analytic, auto_numeric, auto_err = PARTS[
        "autodiff_check"
    ].autodiff_gradient_check(rev_leaves, rev_ops, "out", h)

    # l06 shared path: (a*b + a) * c -> a appears on two paths.
    shared_leaves = [
        {"id": "a", "value": 2.0},
        {"id": "b", "value": 3.0},
        {"id": "c", "value": 4.0},
    ]
    shared_ops = [
        {"id": "ab", "op": "mul", "parents": ["a", "b"]},
        {"id": "e", "op": "add", "parents": ["ab", "a"]},
        {"id": "out", "op": "mul", "parents": ["e", "c"]},
    ]
    shared_grads = PARTS["shared"].shared_path_gradient_accumulation(
        shared_leaves, shared_ops, "out"
    )
    shared_value, shared_ref = PARTS["reverse"].reverse_mode_autodiff(
        shared_leaves, shared_ops, "out"
    )
    assert abs(shared_value - 32.0) < 1e-9
    assert abs(shared_grads["a"] - 16.0) < 1e-9   # c*(b+1)
    assert abs(shared_grads["b"] - 8.0) < 1e-9    # a*c
    assert abs(shared_grads["c"] - 8.0) < 1e-9    # a*(b+1)
    assert shared_grads == shared_ref

    exp_y, exp_g, pow_y, pow_g = PARTS["exp_power"].exp_power_backward(2.0, 3.0, 1.0)
    assert abs(exp_y - math.exp(2.0)) < 1e-9
    assert abs(exp_g - math.exp(2.0)) < 1e-9
    assert abs(pow_y - 8.0) < 1e-9
    assert abs(pow_g - 12.0) < 1e-9

    assert abs(rev_value - loss) < 1e-9
    assert max_err < 1e-4
    assert auto_err < 1e-4
    for name, idx in (("a", 0), ("b", 1), ("c", 2), ("f", 3)):
        assert abs(rev_grads[name] - analytic[idx]) < 1e-9

    if verbose:
        print("expression: (a*b + c) * f")
        print(f"  value {loss:.6f}")
        print(f"  analytic grads {analytic}")
        print(f"  reverse-mode grads {rev_grads}")
        print(f"  finite-difference max error {max_err:.2e}")
        print(f"  copied autodiff-check max error {auto_err:.2e}")
        print(f"  reachable order {order}")
        print(f"  shared-path grads (a*b + a)*c: {shared_grads}")
        print(
            f"  exp/power backward: exp={exp_y:.4f}, exp_grad={exp_g:.4f}, "
            f"pow={pow_y:.4f}, pow_grad={pow_g:.4f}"
        )
    return max(max_err, auto_err)


def value_gradcheck(verbose: bool = True) -> float:
    """Compare Value.backward() with central finite differences."""

    def expression(a, b, c):
        return (a * b + c).tanh() * a.exp() + a.pow(2.0) + b * c

    a = Value(0.5)
    b = Value(-1.2)
    c = Value(0.3)
    out = expression(a, b, c)
    out.backward()
    analytic = [a.grad, b.grad, c.grad]

    numeric = []
    h = 1e-6
    for var in (a, b, c):
        original = var.data
        var.data = original + h
        plus = expression(a, b, c).data
        var.data = original - h
        minus = expression(a, b, c).data
        var.data = original
        numeric.append((plus - minus) / (2.0 * h))

    max_err = max(abs(x - y) for x, y in zip(analytic, numeric))
    if verbose:
        print("Value gradcheck: tanh(a*b + c) * exp(a) + a^2 + b*c")
        print(f"  analytic { [round(x, 6) for x in analytic] }")
        print(f"  numeric  { [round(x, 6) for x in numeric] }")
        print(f"  max error {max_err:.2e}")
    return float(max_err)


# ---------------------------------------------------------------------------
# Plotting and output
# ---------------------------------------------------------------------------
def plot_result(
    dataset: str,
    xs,
    ys,
    mlp: MLP,
    losses: list[float],
    reference_history: list[float] | None,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses, label="Value engine", linewidth=1.4)
    if reference_history:
        axes[0].plot(
            reference_history,
            label="copied torch trainer",
            linewidth=1.0,
            alpha=0.8,
        )
    axes[0].set_title("training loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if dataset == "regression":
        x_line = np.linspace(-2.0, 2.0, 120)
        y_line = [mlp.predict([float(x)]) for x in x_line]
        axes[1].scatter([x[0] for x in xs], ys, s=18, label="data", zorder=3)
        axes[1].plot(x_line, y_line, color="C1", label="MLP fit")
        axes[1].set_title("regression fit")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend()
    else:
        xs_np = np.asarray(xs, dtype=float)
        ys_np = np.asarray(ys, dtype=float)
        if dataset == "xor":
            axes[1].scatter(xs_np[:, 0], xs_np[:, 1], c=ys_np, cmap="coolwarm", s=120)
            for x, y in zip(xs, ys):
                axes[1].annotate(
                    f"{mlp.predict(x):+.2f}",
                    (x[0], x[1]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=9,
                )
            axes[1].set_xlim(-1.5, 1.5)
            axes[1].set_ylim(-1.5, 1.5)
            axes[1].set_title("XOR points (label = prediction)")
        else:
            grid = np.linspace(-1.6, 1.6, 35)
            zz = np.array(
                [
                    [mlp.predict([float(gx), float(gy)]) for gx in grid]
                    for gy in grid
                ]
            )
            axes[1].contourf(
                grid,
                grid,
                zz,
                levels=np.linspace(-1.0, 1.0, 11),
                cmap="coolwarm",
                alpha=0.65,
            )
            axes[1].scatter(
                xs_np[:, 0],
                xs_np[:, 1],
                c=ys_np,
                cmap="coolwarm",
                edgecolors="k",
                s=28,
            )
            axes[1].set_title("decision surface and data")
            axes[1].set_xlabel("x0")
            axes[1].set_ylabel("x1")
        axes[1].set_aspect("equal", adjustable="box")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_train(args) -> None:
    hidden = [int(part) for part in args.hidden.split(",") if part.strip()]
    xs, ys, meta = make_dataset(args.dataset, args.seed, args.points)
    nin = meta["nin"]
    mlp = MLP(nin, hidden + [1], seed=args.seed)
    initial_state = mlp.state_dict()

    print(
        f"dataset={args.dataset}  points={len(xs)}  architecture="
        f"{[nin] + hidden + [1]}  parameters={mlp.n_params()}"
    )
    print(
        f"steps={args.steps}  lr={args.lr}  seed={args.seed}  "
        f"engine=Value (copied topological sort)"
    )

    start = time.time()
    losses, predicted_change = train_value_mlp(
        mlp, xs, ys, args.steps, args.lr, args.log_every
    )
    elapsed = time.time() - start

    initial_loss = losses[0]
    final_loss = losses[-1]
    if meta["kind"] == "classification":
        metric = classification_accuracy(mlp, xs, ys)
        metric_name = "accuracy"
    else:
        metric = regression_mse(mlp, xs, ys)
        metric_name = "mse"

    print(
        f"Value engine: loss {initial_loss:.6f} -> {final_loss:.6f}  "
        f"{metric_name} {metric:.4f}  time {elapsed:.2f}s"
    )
    print(f"last predicted objective change: {predicted_change:.6f}")

    reference = None
    if not args.no_reference:
        reference = torch_reference_train(
            initial_state, xs, ys, args.steps, args.lr
        )
        print(
            f"copied torch trainer: loss {reference['history'][0]:.6f} -> "
            f"{reference['loss']:.6f}"
        )
        print(
            f"Value vs reference final-loss gap: "
            f"{abs(final_loss - reference['loss']):.6f}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": args.dataset,
        "architecture": [nin] + hidden + [1],
        "parameters": mlp.n_params(),
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        metric_name: metric,
        "reference_final_loss": reference["loss"] if reference else None,
        "elapsed_seconds": elapsed,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "model.json").write_text(
        json.dumps(
            {
                "architecture": [nin] + hidden + [1],
                "state_dict": mlp.state_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(out_dir / "losses.npy", np.asarray(losses, dtype=np.float64))
    if reference:
        np.save(
            out_dir / "reference_losses.npy",
            np.asarray(reference["history"], dtype=np.float64),
        )

    if not args.no_plot:
        plot_result(
            args.dataset,
            xs,
            ys,
            mlp,
            losses,
            reference["history"] if reference else None,
            out_dir / "training.png",
        )
        print(f"plot written to {out_dir / 'training.png'}")
    print(f"model written to {out_dir / 'model.json'}")
    print(f"metrics written to {out_dir / 'metrics.json'}")


def cmd_gradcheck(args) -> None:
    print("=== copied graph engine ===")
    graph_checks(verbose=True)
    print()
    print("=== Value engine ===")
    value_gradcheck(verbose=True)


def cmd_graph(args) -> None:
    graph_checks(verbose=True)
    print()
    print("Reachable graph nodes:")
    a, b, c, f = 1.5, -2.0, 0.5, 3.0
    leaves = [
        {"id": "a", "data": a},
        {"id": "b", "data": b},
        {"id": "c", "data": c},
        {"id": "f", "data": f},
    ]
    operations = [
        {"id": "ab", "op": "*", "left": "a", "right": "b"},
        {"id": "e", "op": "+", "left": "ab", "right": "c"},
        {"id": "out", "op": "*", "left": "e", "right": "f"},
    ]
    nodes, output_id = PARTS["build_graph"].build_expression_graph(
        leaves, operations
    )
    for node in nodes:
        print(
            f"  {node['id']:>3s}  data={node['data']:+.4f}  "
            f"parents={node['parents']}  op={node['op'] or 'leaf'}"
        )
    order = PARTS["topo"].topological_sort(nodes, output_id)
    print(f"topological order: {order}")
    print(f"final value: {PARTS['reverse'].reverse_mode_autodiff(
        [{'id': 'a', 'value': a}, {'id': 'b', 'value': b},
         {'id': 'c', 'value': c}, {'id': 'f', 'value': f}],
        [{'id': 'ab', 'op': 'mul', 'parents': ['a', 'b']},
         {'id': 'e', 'op': 'add', 'parents': ['ab', 'c']},
         {'id': 'out', 'op': 'mul', 'parents': ['e', 'f']}],
        'out')[0]:.6f}")


def cmd_selftest(args) -> None:
    for name, module in PARTS.items():
        if module is None:
            raise AssertionError(f"part {name} failed to load")

    # l01 finite differences and gradient checks.
    fx, fxh, slope = PARTS["finite"].finite_difference_derivative(
        [1.0, 2.0, 3.0], 2.0, 1e-6
    )
    assert abs(fx - 17.0) < 1e-9
    assert abs(slope - 14.0) < 1e-4
    d, da, db, dc = PARTS["partials"].scalar_expression_partials(2.0, 3.0, 4.0, 1e-6)
    assert abs(d - 10.0) < 1e-9
    assert abs(da - 3.0) < 1e-4 and abs(db - 2.0) < 1e-4 and abs(dc - 1.0) < 1e-4
    updated, change = PARTS["gd"].gradient_descent_step([1.0, 2.0], [0.5, -0.5], 0.1)
    assert abs(updated[0] - 0.95) < 1e-9 and abs(updated[1] - 2.05) < 1e-9
    assert abs(change + 0.05) < 1e-9

    # l02/l05/l06 graph engine.
    graph_err = graph_checks(verbose=False)
    assert graph_err < 1e-4

    # l05 local VJP rules.
    assert PARTS["vjp"].local_vjp("add", [2.0, 3.0], 5.0, 7.0) == [7.0, 7.0]
    assert PARTS["vjp"].local_vjp("mul", [2.0, 3.0], 6.0, 7.0) == [21.0, 14.0]
    t = math.tanh(0.4)
    vjp_tanh = PARTS["vjp"].local_vjp("tanh", [0.4], t, 5.0)
    assert abs(vjp_tanh[0] - 5.0 * (1 - t * t)) < 1e-12

    # l04 torch neuron math.
    inputs = torch.tensor([0.5, -0.3, 0.2], dtype=torch.float64)
    weights = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    bias = torch.tensor(0.1, dtype=torch.float64)
    aw, nw, ab, nb, err = PARTS["neuron_check"].neuron_gradient_check(
        inputs, weights, bias, 1e-6
    )
    assert float(err) < 1e-4
    a_val, y_val = PARTS["neuron_forward"].neuron_forward(inputs, weights, bias)
    y2, dx, dw, db = PARTS["neuron_backward"].neuron_backward(
        inputs, weights, bias, torch.tensor(1.0, dtype=torch.float64)
    )
    assert torch.allclose(y_val, y2)
    assert torch.allclose(db, torch.tensor(1.0 - float(y_val) ** 2, dtype=torch.float64))
    tanh_y, tanh_g = PARTS["tanh"].tanh_forward_backward(
        torch.tensor(0.3), torch.tensor(1.0)
    )
    assert abs(float(tanh_y) - math.tanh(0.3)) < 1e-6

    # l07 layer/MLP utilities.
    dense_in = torch.tensor([0.2, -0.4, 0.7])
    W = torch.randn(2, 3)
    bvec = torch.randn(2)
    dense_out = PARTS["dense"].dense_layer_forward(dense_in, W, bvec, True)
    assert dense_out.shape == (2,)
    W2 = torch.randn(1, 2)
    b2 = torch.randn(1)
    W_mlp1 = torch.randn(3, 2)
    b_mlp1 = torch.randn(3)
    W_mlp2 = torch.randn(1, 3)
    b_mlp2 = torch.randn(1)
    mlp_out, layer_outs = PARTS["mlp"].mlp_forward(
        torch.tensor([0.2, -0.4]), [W_mlp1, W_mlp2], [b_mlp1, b_mlp2]
    )
    assert mlp_out.shape == (1,) and len(layer_outs) == 2
    params, count = PARTS["params"].recursive_parameter_collection(
        [W_mlp1, W_mlp2], [b_mlp1, b_mlp2]
    )
    assert count == (3 * 2 + 3) + (1 * 3 + 1) and len(params) == count
    scalar = PARTS["scalar_module"].scalar_neuron_module(
        torch.tensor([0.2, -0.4]), torch.tensor([1.0, 2.0]), torch.tensor(0.1), True
    )
    assert scalar.shape == ()

    # l08 copied training utilities.
    preds = torch.tensor([0.5, -0.5])
    targets = torch.tensor([1.0, -1.0])
    loss, grads = PARTS["squared"].squared_error_loss_gradients(preds, targets)
    assert abs(float(loss) - 0.5) < 1e-6
    assert torch.allclose(grads, torch.tensor([-1.0, 1.0]))
    x_sgd = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    y_sgd = torch.tensor([1.0, -1.0, 0.0, 0.0])
    w_sgd = torch.randn(2, 1)
    b_sgd = torch.randn(1)
    before, after, nw, nb, gw, gb = PARTS["sgd_step"].sgd_training_step(
        x_sgd, y_sgd, w_sgd, b_sgd, 0.01
    )
    assert float(after) <= float(before) + 1e-6

    # The copied torch MLP trainer runs and decreases the XOR loss.
    xs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    ys = [-1.0, 1.0, 1.0, -1.0]
    init = MLP(2, [4, 1], seed=0).state_dict()
    ref = torch_reference_train(init, xs, ys, 300, 0.1)
    assert ref["loss"] < ref["history"][0]

    # Value engine gradcheck and a short XOR training run.
    assert value_gradcheck(verbose=False) < 1e-4
    mlp = MLP(2, [8, 1], seed=0)
    losses, _ = train_value_mlp(mlp, xs, ys, 100, 0.05)
    assert losses[-1] < losses[0]

    print(
        "PASS: 25 parts loaded; finite differences, graph/reverse/shared-path "
        "autodiff, Value gradcheck, torch neuron/MLP reference, copied SGD "
        "trainer, and XOR training all OK"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Micrograd-style autograd system from copied solutions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    test = sub.add_parser("selftest", help="check all copied parts and the glue")
    test.set_defaults(func=cmd_selftest)

    demo = sub.add_parser("demo", help="train a Value-based MLP")
    demo.add_argument("--dataset", choices=["xor", "circle", "regression"], default="xor")
    demo.add_argument("--hidden", default="8", help="comma-separated hidden widths")
    demo.add_argument("--points", type=int, default=80)
    demo.add_argument("--steps", type=int, default=1200)
    demo.add_argument("--lr", type=float, default=0.05)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--log-every", type=int, default=200)
    demo.add_argument("--out", default=str(HERE / "runs" / "demo"))
    demo.add_argument("--no-plot", action="store_true")
    demo.add_argument("--no-reference", action="store_true")
    demo.set_defaults(func=cmd_train)

    train = sub.add_parser("train", help="alias for demo with the same options")
    train.add_argument("--dataset", choices=["xor", "circle", "regression"], default="xor")
    train.add_argument("--hidden", default="8")
    train.add_argument("--points", type=int, default=80)
    train.add_argument("--steps", type=int, default=1200)
    train.add_argument("--lr", type=float, default=0.05)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=200)
    train.add_argument("--out", default=str(HERE / "runs" / "demo"))
    train.add_argument("--no-plot", action="store_true")
    train.add_argument("--no-reference", action="store_true")
    train.set_defaults(func=cmd_train)

    check = sub.add_parser("gradcheck", help="compare analytic and numerical gradients")
    check.set_defaults(func=cmd_gradcheck)

    graph = sub.add_parser("graph", help="print the copied expression graph")
    graph.set_defaults(func=cmd_graph)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
