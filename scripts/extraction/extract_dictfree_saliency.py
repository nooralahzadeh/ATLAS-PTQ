#!/usr/bin/env python3
"""Dictionary-free task-circuit saliency — no transcoder / SAE required."""

from __future__ import annotations
import argparse, sys, time, torch
from pathlib import Path
import torch.nn.functional as F
_SCRIPTS = Path(__file__).resolve().parents[1]; _EXTRACT = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS), str(_EXTRACT)):
    if _p not in sys.path: sys.path.insert(0, _p)
from transformers import AutoModelForCausalLM, AutoTokenizer
from mask_budget import apply_mask_budget
from extract_tdso_v2_h200 import (TARGET_SUFFIXES, assert_pairs_differ, configure_h200_backend, is_target, load_corrupt_weights, load_pairs, rtn_dequantize, tokenize_side,)
class NativeNeuronController:
    def __init__(self, source: str, quantile: float) -> None:
        self.source = source; self.quantile = quantile; self.phase = "idle"; self.mask = None
        self.corr_summ: dict[int, torch.Tensor] = {}; self.inter: dict[int, torch.Tensor] = {}; self.loss = None
    def _masked_token_max(self, feats: torch.Tensor) -> torch.Tensor:
        assert self.mask is not None; neg_inf = torch.finfo(feats.dtype).min
        masked = feats.masked_fill(~self.mask.unsqueeze(-1), neg_inf); return masked.amax(dim=1)
    def _masked_token_mean(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask.unsqueeze(-1).to(x.dtype); return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    def make_down_pre_hook(self, layer_idx: int):
        def pre_hook(module, inputs): self.inter[layer_idx] = inputs[0]; return None
        return pre_hook
    def make_mlp_hook(self, layer_idx: int, down_module: torch.nn.Linear):
        def hook(module, inputs, output):
            inter = self.inter.pop(layer_idx, None)
            if self.phase == "corr":
                with torch.no_grad():
                    if self.source == "mlp_neurons": self.corr_summ[layer_idx] = self._masked_token_max(inter).float()
                    else: self.corr_summ[layer_idx] = self._masked_token_mean(output).float()
                return None
            if self.phase != "clean": return None
            if self.source == "actdiff":
                mean_clean = self._masked_token_mean(output); direction = (mean_clean.float() - self.corr_summ[layer_idx]).to(output.dtype)
                y_target = direction.unsqueeze(1).detach()
            else:
                a_clean = self._masked_token_max(inter).float(); a_corr = self.corr_summ[layer_idx]
                delta = torch.relu(a_clean - a_corr); B = delta.shape[0]; f_mask = torch.zeros_like(delta)
                for b in range(B):
                    pos = delta[b][delta[b] > 0]
                    if pos.numel() == 0: continue
                    tau = torch.quantile(pos, self.quantile)
                    f_mask[b] = (delta[b] > tau).to(f_mask.dtype)
                filtered = inter * f_mask.unsqueeze(1).to(inter.dtype)
                with torch.no_grad(): y_target = F.linear(filtered, down_module.weight, down_module.bias)
                y_target = y_target.detach()
            contrib = -(output * y_target); contrib = contrib * self.mask.unsqueeze(-1).to(contrib.dtype)
            layer_loss = contrib.sum(); self.loss = layer_loss if self.loss is None else self.loss + layer_loss
            return None
        return hook
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", default=["data/contrastive/spider_contrastive_train1360.jsonl"])
    ap.add_argument("--allow-legacy-eval-pairs", action="store_true")
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--feature-source", default="mlp_neurons", choices=["mlp_neurons", "actdiff"])
    ap.add_argument("--corrupt-model", default=None)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--mask-fraction", type=float, default=0.0035)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--combine", default="mult", choices=["ce", "align", "boost", "add", "mult"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="masks/dictfree.pt")
    return ap.parse_args()
def main() -> None:
    args = parse_args(); configure_h200_backend(args.seed)
    from calib_split_policy import assert_calib_pairs_path
    for p in args.pairs: assert_calib_pairs_path(p, allow_legacy_eval=args.allow_legacy_eval_pairs)
    device = torch.device("cuda"); dtype = torch.bfloat16
    need_ce = args.combine != "align"; need_align = args.combine != "ce"
    print(f"[cfg] DICTFREE source={args.feature_source} combine={args.combine} "
          f"lam={args.lam} bits={args.bits} frac={args.mask_fraction} q={args.quantile} "
          f"bs={args.batch_size} pairs={args.pairs}", flush=True)
    print("[load] tokenizer + model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                    attn_implementation="flash_attention_2", device_map={"": "cuda"})
    except (ImportError, ValueError) as e:
        print(f"[warn] flash_attention_2 unavailable ({e}); using sdpa", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                    attn_implementation="sdpa", device_map={"": "cuda"})
    model.eval(); n_layers = model.config.num_hidden_layers
    ctrl = None; handles = []
    if need_align:
        ctrl = NativeNeuronController(args.feature_source, quantile=args.quantile)
        for i in range(n_layers):
            mlp = model.model.layers[i].mlp
            handles.append(mlp.down_proj.register_forward_pre_hook(ctrl.make_down_pre_hook(i)))
            handles.append(mlp.register_forward_hook(ctrl.make_mlp_hook(i, mlp.down_proj)))
    target_weights: dict[str, torch.nn.Parameter] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and is_target(name):
            module.weight.requires_grad_(True); target_weights[name + ".weight"] = module.weight
    print(f"[grad] {len(target_weights)} target linear weights", flush=True)
    g_ce = {n: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for n, p in target_weights.items()} if need_ce else None
    g_al = {n: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for n, p in target_weights.items()} if need_align else None
    pairs = load_pairs(args.pairs, args.max_pairs); print(f"[data] {len(pairs)} contrastive pairs", flush=True)
    t0 = time.time(); n_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = pairs[bi * args.batch_size:(bi + 1) * args.batch_size]
        clean = tokenize_side(tokenizer, [r["clean"] for r in batch], args.max_len, device)
        if need_align:
            corr = tokenize_side(tokenizer, [r["corrupted"] for r in batch], args.max_len, device)
            assert_pairs_differ(clean, corr, context=f" (batch {bi})")
            ctrl.phase = "corr"; ctrl.corr_summ = {}; ctrl.inter = {}; ctrl.mask = corr["attention_mask"].bool()
            with torch.no_grad(): model(input_ids=corr["input_ids"], attention_mask=corr["attention_mask"])
        if need_align:
            ctrl.phase = "clean"; ctrl.mask = clean["attention_mask"].bool(); ctrl.inter = {}; ctrl.loss = None
        labels = clean["input_ids"].clone(); labels[clean["attention_mask"] == 0] = -100
        out = model(input_ids=clean["input_ids"], attention_mask=clean["attention_mask"], labels=labels if need_ce else None)
        if need_align:
            model.zero_grad(set_to_none=True); ctrl.loss.backward(retain_graph=need_ce)
            with torch.no_grad():
                for name, p in target_weights.items():
                    if p.grad is not None: g_al[name] += p.grad.detach().abs().float().cpu()
        if need_ce:
            model.zero_grad(set_to_none=True); out.loss.backward()
            with torch.no_grad():
                for name, p in target_weights.items():
                    if p.grad is not None: g_ce[name] += p.grad.detach().abs().float().cpu()
        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            msg = f"[run] batch {bi+1}/{n_batches} elapsed={time.time()-t0:.1f}s"
            if need_ce: msg += f" ce={float(out.loss):.3e}"
            if need_align: msg += f" align={float(ctrl.loss):.3e}"
            print(msg, flush=True)
    for h in handles: h.remove()
    model.zero_grad(set_to_none=True)
    for p in target_weights.values(): p.requires_grad_(False)
    total = sum(p.numel() for p in target_weights.values())
    mean_ce = (sum(float(g.sum()) for g in g_ce.values()) / total) if need_ce else 1.0
    mean_al = (sum(float(g.sum()) for g in g_al.values()) / total) if need_align else 1.0
    mean_ce = max(mean_ce, 1e-20); mean_al = max(mean_al, 1e-20)
    print(f"[norm] mean_ce={mean_ce:.3e} mean_align={mean_al:.3e}", flush=True)
    def combine(gc: torch.Tensor | None, ga: torch.Tensor | None) -> torch.Tensor:
        if args.combine == "ce": return gc
        if args.combine == "align": return ga
        gch = gc / mean_ce; gah = ga / mean_al
        if args.combine == "boost": return gc * (1.0 + args.lam * gah)
        if args.combine == "add": return gch + args.lam * gah
        if args.combine == "mult": return torch.sqrt(torch.clamp(gch * gah, min=0.0))
        raise ValueError(args.combine)
    print("[score] computing saliency (offloading to CPU) ...", flush=True)
    corrupt_sd = load_corrupt_weights(args.corrupt_model, torch.device("cpu")) if args.corrupt_model else None
    saliency: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in target_weights.items():
            W = p.detach().float()
            g = combine(g_ce[name].to(device) if need_ce else None, g_al[name].to(device) if need_align else None)
            if corrupt_sd is not None and name in corrupt_sd: W2 = corrupt_sd[name].to(device).float()
            else: W2 = rtn_dequantize(p.detach(), args.bits).float()
            delta = (W - W2).abs(); s = (W.abs() * g) * delta; saliency[name] = s.to("cpu")
            if need_ce: g_ce[name] = None
            if need_align: g_al[name] = None
            del s, W, W2, g
    print("[score] mask budget=global on CPU ...", flush=True)
    masks, kept = apply_mask_budget(saliency, args.mask_fraction, mode="global")
    print(f"[mask] kept {kept}/{total} = {100.0*kept/total:.4f}% "
          f"(target {100*args.mask_fraction:.4f}%)\n", flush=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"masks": masks, "meta": {"model": args.model, "feature_source": args.feature_source,
                "combine": args.combine, "lam": args.lam, "bits": args.bits,
                "mask_fraction": args.mask_fraction, "quantile": args.quantile,
                "pairs": args.pairs, "n_pairs": len(pairs),
                "total_params": total, "kept_params": kept,
                "delta_w": "corrupt_model" if corrupt_sd is not None else f"rtn{args.bits}bit",}}, out_path)
    print(f"[done] saved mask -> {out_path} in {time.time()-t0:.1f}s", flush=True)
if __name__ == "__main__": main()
