"""
Pruning utilities for Distill-then-Prune.

- Unstructured: L1-norm per-weight pruning via torch.nn.utils.prune.
- Structured: Channel pruning via torch-pruning (MagnitudePruner + DepGraph).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import prune
from typing import List, Optional, Tuple, Any

from models import count_parameters


def _get_prunable_modules(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """Return (module, param_name) for Conv2d weight and Linear weight."""
    out: List[Tuple[nn.Module, str]] = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if not hasattr(m, "weight") or m.weight is None:
                continue
            out.append((m, "weight"))
    return out


def apply_unstructured_prune(
    model: nn.Module,
    amount: float = 0.3,
    prune_bias: bool = False,
) -> int:
    """
    Apply L1-unstructured pruning to Conv2d and Linear layers.
    Pruning is made permanent (reparametrization removed) before return.

    Args:
        model: Model to prune (modified in-place).
        amount: Fraction of weights to prune (0.3 = 30%).
        prune_bias: Whether to prune bias too (default False).

    Returns:
        Number of parameters that were pruned (zeros).
    """
    device = next(model.parameters()).device
    total_pruned = 0
    to_remove: List[Tuple[nn.Module, str]] = []

    for m, param_name in _get_prunable_modules(model):
        param = getattr(m, param_name)
        n = param.numel()
        k = int(n * amount)
        if k <= 0:
            continue
        prune.l1_unstructured(m, param_name, amount=amount)
        total_pruned += k
        to_remove.append((m, param_name))
        if prune_bias and hasattr(m, "bias") and m.bias is not None:
            prune.l1_unstructured(m, "bias", amount=amount)
            to_remove.append((m, "bias"))

    for m, param_name in to_remove:
        prune.remove(m, param_name)

    return total_pruned


def apply_structured_prune(
    model: nn.Module,
    example_inputs: torch.Tensor,
    amount: float = 0.3,
    ignored_layers: Optional[List[nn.Module]] = None,
    iterative_steps: int = 1,
) -> Tuple[int, int]:
    """
    Apply structured (channel) pruning via torch-pruning.
    Model structure is modified in-place.

    Args:
        model: Model to prune (modified in-place).
        example_inputs: Example input tensor, e.g. (1, 3, 32, 32).
        amount: Target channel sparsity (fraction of channels to remove).
        ignored_layers: Modules not to prune (e.g. final classifier).
        iterative_steps: Number of pruning steps (1 = one-shot).

    Returns:
        (params_before - params_after, FLOPs_before - FLOPs_after) as proxy for reduction.
    """
    try:
        import torch_pruning as tp
    except ImportError:
        raise ImportError("torch-pruning is required for structured pruning. pip install torch-pruning")

    model.eval()
    example_inputs = example_inputs.detach().to(next(model.parameters()).device)
    ignored = list(ignored_layers) if ignored_layers else []

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=amount,
        ignored_layers=ignored,
    )

    macs_before, nparams_before = tp.utils.count_ops_and_params(model, example_inputs)
    for _ in range(iterative_steps):
        pruner.step()
    macs_after, nparams_after = tp.utils.count_ops_and_params(model, example_inputs)

    return (nparams_before - nparams_after, macs_before - macs_after)


def get_student_ignored_layers(student: nn.Module) -> List[nn.Module]:
    """Return modules to exclude from pruning (MobileNetV2 classifier)."""
    ignored: List[nn.Module] = []
    if hasattr(student, "classifier") and isinstance(student.classifier, nn.Sequential):
        for m in student.classifier:
            if isinstance(m, nn.Linear):
                ignored.append(m)
                break
    return ignored


def count_flops_params(model: nn.Module, example_input: torch.Tensor) -> Tuple[int, int]:
    """Return (FLOPs, params) using thop. example_input: (1, C, H, W)."""
    try:
        from thop import profile
    except ImportError:
        return 0, count_parameters(model)
    model.eval()
    with torch.no_grad():
        flops, nparams = profile(model, inputs=(example_input,), verbose=False)
    return int(flops), int(nparams)


def get_model_size_mb(model: nn.Module) -> float:
    """Model size in MB (params + buffers)."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


if __name__ == "__main__":
    from models import get_student

    model = get_student(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    n0 = count_parameters(model)
    print(f"Params before: {n0:,}")

    # Unstructured
    nz = apply_unstructured_prune(model, amount=0.3)
    n1 = count_parameters(model)
    print(f"Params after L1 unstructured 30%: {n1:,} (zeroed {nz:,} weights)")

    # Structured (need fresh model; requires torch-pruning)
    try:
        import torch_pruning  # noqa: F401
    except ImportError:
        print("Structured pruning skipped (torch-pruning not installed)")
    else:
        model2 = get_student(num_classes=10)
        ign = get_student_ignored_layers(model2)
        dparams, dmacs = apply_structured_prune(model2, x, amount=0.3, ignored_layers=ign)
        print(f"Structured 30%: Δparams={dparams}, ΔMACs={dmacs}")
