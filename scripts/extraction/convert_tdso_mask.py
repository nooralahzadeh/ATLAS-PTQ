#!/usr/bin/env python3
"""Convert a T-DSO / TaCQ saliency mask bundle into the flat dict TACQ gptq expects.

Our extractors (`extract_tdso_v2_h200.py`, `extract_dictfree_saliency.py`,
`build_baseline_mask.py`) save::

    {"masks": {config_key: bool_tensor}, "meta": {...}}

TACQ `gptq/llama.py --important_mask` does ``torch.load(path)`` and then indexes
``important_mask[config_key]`` directly, where
``config_key = "model.layers.{i}.{name}.weight"`` (e.g.
``model.layers.0.self_attn.q_proj.weight``) — exactly the keys we store. So this
just unwraps the bundle and guarantees bool dtype on CPU.

Reconstructed after the 2026-06-10 scratch incident; verified against
TACQ/gptq/llama.py mask-loading contract (lines 88-149).
"""
from __future__ import annotations
import argparse
import torch
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="saliency mask bundle (.pt)")
    ap.add_argument("--out", required=True, help="flat important_mask (.pt) for gptq")
    args = ap.parse_args()

    obj = torch.load(args.inp, map_location="cpu")
    masks = obj["masks"] if isinstance(obj, dict) and "masks" in obj else obj
    flat = {k: v.to(torch.bool) for k, v in masks.items()}

    kept = sum(int(m.sum()) for m in flat.values())
    total = sum(m.numel() for m in flat.values())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flat, out_path)
    print(f"[convert] {args.inp} -> {out_path}: {len(flat)} tensors, "
          f"kept {kept}/{total} = {100.0 * kept / max(total, 1):.4f}%", flush=True)


if __name__ == "__main__":
    main()
