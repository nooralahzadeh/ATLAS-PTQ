# Source Generated with Decompyle++
# File: mask_budget.cpython-310.pyc (Python 3.10)

'''Mask budget strategies for T-DSO / TaCQ-style global outlier selection.'''
from __future__ import annotations
import re
from collections import defaultdict
import torch

def layer_index(weight_name = None):
    m = re.search('layers\\.(\\d+)\\.', weight_name)
    if m:
        return int(m.group(1))


def normalize_layer_weights(density = None, n_layers = None):
    '''Turn accumulated per-layer feature counts into a probability vector.'''
    layers = (lambda .0 = None: [ i for i in .0 if density.get(i, 0) > 0 ])(range(n_layers))
    if not layers:
        return (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(range(n_layers))
    total = None((lambda .0 = None: for i in .0:
density[i])(layers))
    return (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(range(n_layers))


def allocate_layer_quotas(layer_weights = None, k_total = None, layers_present = None):
    '''Split k_total FP16 outliers across layers (largest-remainder method).'''
    layers = sorted(layers_present)
    if not layers:
        return { }
    raw = (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(layers)
    k_l = (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(layers)
    remainder = k_total - sum(k_l.values())
    order = None(None, (lambda l = None: raw[l] - k_l[l]), True, **('key', 'reverse'))
    for i in range(max(0, remainder)):
        k_l[order[i % len(order)]] += 1
    return k_l


def apply_mask_budget(saliency = None, mask_fraction = None, mode = None, layer_weights = ('global', None)):
    '''Return bool masks keeping ``mask_fraction`` of parameters.

    Modes
    -----
    global:
        Single global top-k on flattened saliency (TaCQ / v2 default).
    layer_adaptive:
        Same global budget, but per-layer quotas proportional to ``layer_weights``
        (typically transcoder task-feature density), then top-k within each layer.
    '''
    total = sum((lambda .0: for s in .0:
s.numel())(saliency.values()))
    k_total = max(1, int(round(mask_fraction * total)))
    if mode == 'global':
        flat = torch.cat((lambda .0: [ s.flatten() for s in .0 ])(saliency.values()))
        thresh = torch.kthvalue(flat, (total - k_total) + 1).values if k_total < total else flat.min()
        masks = (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(saliency.items())
        kept = sum((lambda .0: for m in .0:
int(m.sum()))(masks.values()))
        return (masks, kept)
    if None != 'layer_adaptive':
        raise ValueError(f'''Unknown mask budget mode: {mode}''')
    by_layer = defaultdict(list)
    for name, s in saliency.items():
        by_layer[layer_index(name)].append((name, s))
    layers_present = (lambda .0: pass# WARNING: Decompyle incomplete
)(by_layer)
    if layer_weights is None:
        layer_weights = (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(layers_present)
    k_l = allocate_layer_quotas(layer_weights, k_total, layers_present)
    masks = { }
    kept = 0
    for li, items in by_layer.items():
        if li < 0:
            continue
        k_layer = k_l.get(li, 0)
        if k_layer <= 0:
            for name, s in items:
                masks[name] = torch.zeros_like(s, torch.bool, **('dtype',))
            continue
        names = (lambda .0: [ n for n, _ in .0 ])(items)
        flat = None((lambda .0 = None: [ saliency[n].flatten() for n in .0 ])(names))
        n_layer = flat.numel()
        k_use = min(k_layer, n_layer)
        if k_use >= n_layer:
            for name in names:
                masks[name] = torch.ones_like(saliency[name], torch.bool, **('dtype',))
                kept += masks[name].numel()
            continue
        thresh = torch.kthvalue(flat, (n_layer - k_use) + 1).values
        offset = 0
        for name in names:
            s = saliency[name]
            m = (s >= thresh).to(torch.bool)
            masks[name] = m
            kept += int(m.sum())
            offset += s.numel()
    return (masks, kept)

