"""Mask budget strategies for T-DSO / TaCQ-style global outlier selection.

Reconstructed from cpython-310 bytecode after the 2026-06-10 scratch incident.
The `global` path (the production default) is verified line-for-line against the
decompiled blueprint; `layer_adaptive` is reconstructed from structure — VERIFY
before relying on it for new results.
"""
from __future__ import annotations
import re
from collections import defaultdict
import torch


def layer_index(weight_name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", weight_name)
    if m:
        return int(m.group(1))
    return -1


def normalize_layer_weights(density: dict[int, float], n_layers: int) -> list[float]:
    """Turn accumulated per-layer feature counts into a probability vector."""
    layers = [i for i in range(n_layers) if density.get(i, 0) > 0]
    if not layers:
        return [1.0 / n_layers for _ in range(n_layers)]
    total = sum(density[i] for i in layers)
    return [density.get(i, 0) / total for i in range(n_layers)]


def allocate_layer_quotas(layer_weights, k_total: int, layers_present) -> dict[int, int]:
    """Split k_total FP16 outliers across layers (largest-remainder method)."""
    layers = sorted(layers_present)
    if not layers:
        return {}
    raw = {l: k_total * float(layer_weights[l]) for l in layers}
    k_l = {l: int(raw[l]) for l in layers}
    remainder = k_total - sum(k_l.values())
    order = sorted(layers, key=lambda l: raw[l] - k_l[l], reverse=True)
    for i in range(max(0, remainder)):
        k_l[order[i % len(order)]] += 1
    return k_l


def apply_mask_budget(saliency, mask_fraction, mode="global", layer_weights=None):
    """Return bool masks keeping ``mask_fraction`` of parameters.

    Modes
    -----
    global:
        Single global top-k on flattened saliency (TaCQ / v2 default).
    layer_adaptive:
        Same global budget, but per-layer quotas proportional to ``layer_weights``
        (typically transcoder task-feature density), then top-k within each layer.
    """
    total = sum(s.numel() for s in saliency.values())
    k_total = max(1, int(round(mask_fraction * total)))
    if mode == "global":
        flat = torch.cat([s.flatten() for s in saliency.values()])
        thresh = torch.kthvalue(flat, (total - k_total) + 1).values if k_total < total else flat.min()
        masks = {name: (s >= thresh).to(torch.bool) for name, s in saliency.items()}
        kept = sum(int(m.sum()) for m in masks.values())
        # Degenerate-saliency guard: massive ties at the threshold (e.g. an
        # all-zero saliency field) silently keep far more than the budget and
        # produce near-FP16 engines. Fail loudly instead.
        if kept > 2 * k_total:
            raise ValueError(
                f"degenerate saliency: kept {kept} weights vs target {k_total} "
                f"({100.0 * kept / total:.4f}% vs {100.0 * mask_fraction:.4f}%); "
                "threshold ties suggest the saliency signal is (near-)constant."
            )
        return masks, kept
    if mode != "layer_adaptive":
        raise ValueError(f"Unknown mask budget mode: {mode}")
    by_layer = defaultdict(list)
    for name, s in saliency.items():
        by_layer[layer_index(name)].append((name, s))
    layers_present = [li for li in by_layer if li >= 0]
    if layer_weights is None:
        layer_weights = {li: 1.0 for li in layers_present}
    k_l = allocate_layer_quotas(layer_weights, k_total, layers_present)
    masks = {}
    kept = 0
    for li, items in by_layer.items():
        if li < 0:
            continue
        k_layer = k_l.get(li, 0)
        if k_layer <= 0:
            for name, s in items:
                masks[name] = torch.zeros_like(s, dtype=torch.bool)
            continue
        names = [n for n, _ in items]
        flat = torch.cat([saliency[n].flatten() for n in names])
        n_layer = flat.numel()
        k_use = min(k_layer, n_layer)
        if k_use >= n_layer:
            for name in names:
                masks[name] = torch.ones_like(saliency[name], dtype=torch.bool)
                kept += masks[name].numel()
            continue
        thresh = torch.kthvalue(flat, (n_layer - k_use) + 1).values
        for name in names:
            s = saliency[name]
            m = (s >= thresh).to(torch.bool)
            masks[name] = m
            kept += int(m.sum())
    return masks, kept
