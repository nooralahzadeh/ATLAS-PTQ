#!/usr/bin/env python3
"""Build non-circuit control masks at a matched outlier budget."""

from __future__ import annotations
import argparse, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path: sys.path.insert(0, str(_SCRIPTS))
from mask_budget import apply_mask_budget

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",)
def is_target(name: str) -> bool: return name.endswith(TARGET_SUFFIXES) and "lm_head" not in name

@torch.no_grad()
def rtn_dequantize(W: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1; qmin = -(2 ** (bits - 1))
    scale = W.abs().amax(dim=1, keepdim=True) / max(qmax, 1)
    scale = torch.clamp(scale, min=1e-8); q = torch.clamp(torch.round(W / scale), qmin, qmax)
    return q * scale

@torch.no_grad()
def load_corrupt_weights(path: str) -> dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu"); return sd.get("state_dict", sd)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--mode", required=True, choices=["weight", "magnitude", "random"])
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--mask-fraction", type=float, default=0.0035)
    ap.add_argument("--corrupt-model", default=None,
                    help="Faithful per-bit GPTQ corrupt checkpoint for |W-W_q|; "
                         "falls back to RTN dequant if omitted.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    print(f"[cfg] mode={args.mode} bits={args.bits} frac={args.mask_fraction} "
          f"seed={args.seed} corrupt={'yes' if args.corrupt_model else 'rtn'}", flush=True)

    print("[load] model weights (cpu) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map={"": "cpu"}); model.eval()

    corrupt_sd = (load_corrupt_weights(args.corrupt_model)
                  if args.corrupt_model and Path(args.corrupt_model).is_file() else None)
    if args.mode == "magnitude" and corrupt_sd is None and args.corrupt_model:
        print(f"[warn] corrupt model {args.corrupt_model} missing; using RTN dequant", flush=True)

    saliency: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if not (isinstance(module, torch.nn.Linear) and is_target(name)): continue
            key = name + ".weight"; W = module.weight.detach().float()
            if args.mode == "weight": s = W.abs()
            elif args.mode == "random":
                s = torch.rand(W.shape, generator=torch.Generator().manual_seed(args.seed + (hash(key) & 0xFFFF)))
            else: # magnitude = |W| * |W - W_q|
                if corrupt_sd is not None and key in corrupt_sd: W2 = corrupt_sd[key].float()
                else: W2 = rtn_dequantize(module.weight.detach(), args.bits).float()
                s = W.abs() * (W - W2).abs()
            saliency[key] = s.cpu()

    total = sum(s.numel() for s in saliency.values())
    print(f"[score] {len(saliency)} tensors, {total} params; applying global budget ...", flush=True)
    masks, kept = apply_mask_budget(saliency, args.mask_fraction, mode="global")
    print(f"[mask] kept {kept}/{total} = {100.0*kept/total:.4f}% "
          f"(target {100*args.mask_fraction:.4f}%)\n", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"masks": masks, "meta": {"model": args.model, "mode": args.mode, "bits": args.bits,
                "mask_fraction": args.mask_fraction, "seed": args.seed,
                "total_params": total, "kept_params": kept,
                "delta_w": "corrupt_model" if corrupt_sd is not None else f"rtn{args.bits}bit",}}, out_path)
    print(f"[done] saved mask -> {out_path}", flush=True)

if __name__ == "__main__": main()
