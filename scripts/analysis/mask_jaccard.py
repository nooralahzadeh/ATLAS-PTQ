#!/usr/bin/env python3
"""Compare two FP16-outlier masks (Jaccard, overlap, per-layer agreement).

Accepts either our bundle format {"masks": {...}, "meta": {...}} or a flat
dict[str, bool Tensor] (TaCQ's important_mask_*.pt format). Keys must be the
HF parameter names (model.layers.{i}.{module}.weight).
"""
from __future__ import annotations
import argparse
import torch


def load_mask(path: str) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "masks" in obj:
        if "meta" in obj:
            print(f"[{path}] meta: {obj['meta']}")
        obj = obj["masks"]
    return {k: v.to(torch.bool) for k, v in obj.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="first mask .pt")
    ap.add_argument("--b", required=True, help="second mask .pt")
    ap.add_argument("--per-layer", action="store_true", help="print per-tensor Jaccard")
    args = ap.parse_args()

    A = load_mask(args.a)
    B = load_mask(args.b)
    common = sorted(set(A) & set(B))
    only_a = set(A) - set(B)
    only_b = set(B) - set(A)
    if only_a:
        print(f"[warn] {len(only_a)} keys only in A (e.g. {next(iter(only_a))})")
    if only_b:
        print(f"[warn] {len(only_b)} keys only in B (e.g. {next(iter(only_b))})")

    inter = union = na = nb = 0
    rows = []
    for k in common:
        a, b = A[k], B[k]
        if a.shape != b.shape:
            print(f"[warn] shape mismatch {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        i = int((a & b).sum())
        u = int((a | b).sum())
        inter += i
        union += u
        na += int(a.sum())
        nb += int(b.sum())
        if args.per_layer and u:
            rows.append((i / u, k))

    print(f"common tensors: {len(common)}")
    print(f"kept A: {na:,}   kept B: {nb:,}")
    if union:
        print(f"Jaccard = {inter / union:.4f}")
        print(f"|A∩B| / |A| = {inter / max(na, 1):.4f}   |A∩B| / |B| = {inter / max(nb, 1):.4f}")
    if args.per_layer:
        rows.sort()
        print("\nworst 10 tensors by Jaccard:")
        for j, k in rows[:10]:
            print(f"  {j:.3f}  {k}")


if __name__ == "__main__":
    main()
