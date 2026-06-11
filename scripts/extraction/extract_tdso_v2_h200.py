"""T-DSO v2 — hybrid task-gradient + transcoder task-circuit saliency (H200).

v1 replaced TaCQ's task CE gradient with a transcoder-reconstruction alignment
gradient. Because the transcoders only reconstruct the MLP output at cosine ~0.5,
that proxy is noisy and v1 merely tied TaCQ. v2 instead *augments* the real task
signal with an interpretable circuit boost:

  For each contrastive pair we run ONE clean forward and obtain, per target weight:
    g_ce    = |d(LM cross-entropy)/dW|           # TaCQ's task-sensitivity signal
    g_align = |d(transcoder task-feature align)/dW|   # task-circuit membership

  Combined gradient term (configurable via --combine):
    ce     : g = g_ce                                   (≈ TaCQ MSG; sanity)
    align  : g = g_align                                (≈ v1)
    boost  : g = g_ce * (1 + lam * g_align_hat)         (DEFAULT; keeps task base,
                                                          boosts task-circuit weights)
    add    : g = g_ce_hat + lam * g_align_hat
    mult   : g = sqrt(g_ce_hat * g_align_hat)           (strict AND)
  where *_hat are globally mean-normalized so the two scales are comparable.

  Saliency: S = (|W| * g) * |W_qbit - W_fp16|  ; keep global top-p -> FP16 mask.

Task-discriminative features are found exactly as v1 (clean-vs-corrupt TopK feature
delta, per-example top-(1-q) quantile), but here they only *reweight* the proven
TaCQ saliency rather than define the whole objective.

----------------------------------------------------------------------------------
RECONSTRUCTED after the 2026-06-10 scratch incident. The module-level helpers
(configure_h200_backend, _record_to_messages, load_pairs, tokenize_side,
TDSOController, is_target, rtn_dequantize, load_corrupt_weights, parse_args) were
recovered line-for-line from cpython-310 bytecode. ``main()`` was truncated in the
decompilation and is reconstructed here by mirroring the known-good dict-free
extractor. The produced masks/quantized models from the original run survived; if
you re-run extraction, sanity-check a mask against an existing one (Jaccard) before
trusting new numbers.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from calib_split_policy import assert_calib_pairs_path
from mask_budget import apply_mask_budget, normalize_layer_weights
from transcoder_io import load_all_transcoders, load_config, resolve_transcoder_dir

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def configure_h200_backend(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _record_to_messages(text: str) -> list[dict]:
    if text.startswith("<<SYS>>"):
        body = text[len("<<SYS>>"):]
        sys_part, _, user_part = body.partition("<<USER>>")
        return [
            {"role": "system", "content": sys_part.strip()},
            {"role": "user", "content": user_part.strip()},
        ]
    return [{"role": "user", "content": text}]


def load_pairs(paths, max_pairs=None):
    pairs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                pairs.append(json.loads(line))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def tokenize_side(tokenizer, texts, max_len, device):
    rendered = [
        tokenizer.apply_chat_template(_record_to_messages(t), tokenize=False, add_generation_prompt=True)
        for t in texts
    ]
    # No truncation (TaCQ convention: full prompts, drop nothing). Right-truncation
    # silently cut the corruption out of long MMLU_humanities pairs (clean ==
    # corrupted after tokenize => zero contrast => degenerate 100% masks).
    enc = tokenizer(
        rendered, return_tensors="pt", padding="longest", truncation=False,
        add_special_tokens=False,
    )
    n_tok = enc["input_ids"].shape[1]
    if n_tok > max_len:
        raise ValueError(
            f"batch is {n_tok} tokens > max_len={max_len}; refusing to truncate "
            f"(would corrupt the clean/corrupted contrast). Raise --max-len."
        )
    return {k: v.to(device) for k, v in enc.items()}


def assert_pairs_differ(clean, corr, context=""):
    """Fail loudly if a whole batch tokenized identically on both sides."""
    if clean["input_ids"].shape == corr["input_ids"].shape and torch.equal(
        clean["input_ids"], corr["input_ids"]
    ):
        raise ValueError(
            f"clean and corrupted prompts tokenized identically{context}; "
            "contrastive circuit signal would be zero. Check pair construction."
        )


class TDSOController:

    def __init__(self, transcoders, quantile):
        self.transcoders = transcoders
        self.quantile = quantile
        self.phase = "idle"
        self.mask = None
        self.corr_summ = {}
        self.loss = None
        self.feature_density = {}

    def _masked_token_max(self, feats):
        assert self.mask is not None
        neg_inf = torch.finfo(feats.dtype).min
        masked = feats.masked_fill(~self.mask.unsqueeze(-1), neg_inf)
        return masked.amax(dim=1)

    def make_hook(self, layer_idx):
        tc = self.transcoders[layer_idx]

        def hook(module, inputs, output):
            x_in = inputs[0]
            if self.phase == "corr":
                with torch.no_grad():
                    feats = tc.encode(x_in)
                    self.corr_summ[layer_idx] = self._masked_token_max(feats).float()
                return None
            if self.phase == "clean":
                feats = tc.encode(x_in)
                a_clean = self._masked_token_max(feats).float()
                a_corr = self.corr_summ[layer_idx]
                delta = torch.relu(a_clean - a_corr)
                B = delta.shape[0]
                f_mask = torch.zeros_like(delta)
                for b in range(B):
                    pos = delta[b][delta[b] > 0]
                    if pos.numel() == 0:
                        continue
                    tau = torch.quantile(pos, self.quantile)
                    f_mask[b] = (delta[b] > tau).to(f_mask.dtype)
                self.feature_density[layer_idx] = self.feature_density.get(layer_idx, 0) + float(f_mask.sum().item())
                filtered = feats * f_mask.unsqueeze(1).to(feats.dtype)
                with torch.no_grad():
                    y_target = tc.decode(filtered)
                y_target = y_target.detach()
                contrib = -(output * y_target)
                contrib = contrib * self.mask.unsqueeze(-1).to(contrib.dtype)
                layer_loss = contrib.sum()
                self.loss = layer_loss if self.loss is None else self.loss + layer_loss
            return None

        return hook


def is_target(name):
    return name.endswith(TARGET_SUFFIXES) and "lm_head" not in name


def rtn_dequantize(W, bits):
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    scale = W.abs().amax(dim=1, keepdim=True) / max(qmax, 1)
    scale = torch.clamp(scale, min=1e-08)
    q = torch.clamp(torch.round(W / scale), qmin, qmax)
    return q * scale


def load_corrupt_weights(path, device):
    p = Path(path)
    if p.is_file() and p.suffix == ".pt":
        sd = torch.load(p, map_location=device)
        return sd.get("state_dict", sd)
    raise ValueError(f"Unsupported corrupt-model path {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", default=["data/contrastive/spider_contrastive_train1360.jsonl"])
    ap.add_argument("--allow-legacy-eval-pairs", action="store_true",
                    help="DEBUG ONLY: skip check for eval-split contrastive files.")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--transcoder-repo", default="facebook/crv-8b-instruct-transcoders")
    ap.add_argument("--transcoder-dir", default=None)
    ap.add_argument("--transcoder-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--corrupt-model", default=None)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--mask-fraction", type=float, default=0.0035)
    ap.add_argument("--mask-budget", default="global", choices=["global", "layer_adaptive"],
                    help="global=single top-k (TaCQ-style); layer_adaptive=0.35%% split by transcoder feature density.")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--combine", default="boost", choices=["ce", "align", "boost", "add", "mult"])
    ap.add_argument("--lam", type=float, default=1, help="weight of the align term")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="masks/tdso_v2.pt")
    ap.add_argument("--save-saliency-out", default=None,
                    help="Optional sidecar .pt with g_ce, g_align, combined saliency tensors")
    return ap.parse_args()


def main():
    # NOTE: reconstructed body — see module docstring. Verify before trusting new masks.
    args = parse_args()
    configure_h200_backend(args.seed)
    for p in args.pairs:
        assert_calib_pairs_path(p, allow_legacy_eval=args.allow_legacy_eval_pairs)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    need_ce = args.combine != "align"
    need_align = args.combine != "ce"
    print(f"[cfg] model={args.model} combine={args.combine} lam={args.lam} bits={args.bits} "
          f"frac={args.mask_fraction} budget={args.mask_budget} q={args.quantile} "
          f"bs={args.batch_size} pairs={args.pairs}", flush=True)

    print("[load] tokenizer + model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="flash_attention_2", device_map={"": "cuda"})
    except (ImportError, ValueError) as e:
        print(f"[warn] flash_attention_2 unavailable ({e}); using sdpa", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="sdpa", device_map={"": "cuda"})
    model.eval()
    n_layers = model.config.num_hidden_layers

    ctrl = None
    handles = []
    if need_align:
        print("[load] transcoders ...", flush=True)
        tdir = resolve_transcoder_dir(args.transcoder_repo, args.transcoder_dir)
        cfg = load_config(tdir)
        transcoders = load_all_transcoders(
            tdir, n_layers=n_layers, k=cfg.k, device=args.transcoder_device, dtype=dtype)
        ctrl = TDSOController(transcoders, quantile=args.quantile)
        for i in range(n_layers):
            handles.append(model.model.layers[i].mlp.register_forward_hook(ctrl.make_hook(i)))

    target_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and is_target(name):
            module.weight.requires_grad_(True)
            target_weights[name + ".weight"] = module.weight
    print(f"[grad] {len(target_weights)} target linear weights", flush=True)

    g_ce = {n: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for n, p in target_weights.items()} if need_ce else None
    g_al = {n: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for n, p in target_weights.items()} if need_align else None

    pairs = load_pairs(args.pairs, args.max_pairs)
    print(f"[data] {len(pairs)} contrastive pairs", flush=True)
    t0 = time.time()
    n_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = pairs[bi * args.batch_size:(bi + 1) * args.batch_size]
        clean = tokenize_side(tokenizer, [r["clean"] for r in batch], args.max_len, device)
        if need_align:
            corr = tokenize_side(tokenizer, [r["corrupted"] for r in batch], args.max_len, device)
            assert_pairs_differ(clean, corr, context=f" (batch {bi})")
            ctrl.phase = "corr"
            ctrl.corr_summ = {}
            ctrl.mask = corr["attention_mask"].bool()
            with torch.no_grad():
                model(input_ids=corr["input_ids"], attention_mask=corr["attention_mask"])
            ctrl.phase = "clean"
            ctrl.mask = clean["attention_mask"].bool()
            ctrl.loss = None
        labels = clean["input_ids"].clone()
        labels[clean["attention_mask"] == 0] = -100
        out = model(input_ids=clean["input_ids"], attention_mask=clean["attention_mask"],
                    labels=labels if need_ce else None)
        if need_align:
            model.zero_grad(set_to_none=True)
            ctrl.loss.backward(retain_graph=need_ce)
            with torch.no_grad():
                for name, p in target_weights.items():
                    if p.grad is not None:
                        g_al[name] += p.grad.detach().abs().float().cpu()
        if need_ce:
            model.zero_grad(set_to_none=True)
            out.loss.backward()
            with torch.no_grad():
                for name, p in target_weights.items():
                    if p.grad is not None:
                        g_ce[name] += p.grad.detach().abs().float().cpu()
        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            msg = f"[run] batch {bi+1}/{n_batches} elapsed={time.time()-t0:.1f}s"
            if need_ce:
                msg += f" ce={float(out.loss):.3e}"
            if need_align:
                msg += f" align={float(ctrl.loss):.3e}"
            print(msg, flush=True)

    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    for p in target_weights.values():
        p.requires_grad_(False)

    total = sum(p.numel() for p in target_weights.values())
    mean_ce = (sum(float(g.sum()) for g in g_ce.values()) / total) if need_ce else 1.0
    mean_al = (sum(float(g.sum()) for g in g_al.values()) / total) if need_align else 1.0
    mean_ce = max(mean_ce, 1e-20)
    mean_al = max(mean_al, 1e-20)
    print(f"[norm] mean_ce={mean_ce:.3e} mean_align={mean_al:.3e}", flush=True)

    def combine(gc, ga):
        if args.combine == "ce":
            return gc
        if args.combine == "align":
            return ga
        gch = gc / mean_ce
        gah = ga / mean_al
        if args.combine == "boost":
            return gc * (1.0 + args.lam * gah)
        if args.combine == "add":
            return gch + args.lam * gah
        if args.combine == "mult":
            return torch.sqrt(torch.clamp(gch * gah, min=0.0))
        raise ValueError(args.combine)

    print("[score] computing saliency (offloading to CPU) ...", flush=True)
    corrupt_sd = load_corrupt_weights(args.corrupt_model, torch.device("cpu")) if args.corrupt_model else None
    saliency = {}
    sidecar = {} if args.save_saliency_out else None
    with torch.no_grad():
        for name, p in target_weights.items():
            W = p.detach().float()
            g = combine(g_ce[name].to(device) if need_ce else None,
                        g_al[name].to(device) if need_align else None)
            if corrupt_sd is not None and name in corrupt_sd:
                W2 = corrupt_sd[name].to(device).float()
            else:
                W2 = rtn_dequantize(p.detach(), args.bits).float()
            delta = (W - W2).abs()
            s = (W.abs() * g) * delta
            saliency[name] = s.to("cpu")
            if sidecar is not None:
                sidecar[name] = {
                    "g_ce": g_ce[name].clone() if need_ce else None,
                    "g_align": g_al[name].clone() if need_align else None,
                    "saliency": saliency[name].clone(),
                }
            if need_ce:
                g_ce[name] = None
            if need_align:
                g_al[name] = None
            del s, W, W2, g

    layer_weights = None
    if args.mask_budget == "layer_adaptive" and ctrl is not None:
        layer_weights = normalize_layer_weights(ctrl.feature_density, n_layers)
    print(f"[score] mask budget={args.mask_budget} on CPU ...", flush=True)
    masks, kept = apply_mask_budget(saliency, args.mask_fraction, mode=args.mask_budget, layer_weights=layer_weights)
    print(f"[mask] kept {kept}/{total} = {100.0*kept/total:.4f}% "
          f"(target {100*args.mask_fraction:.4f}%)\n", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"masks": masks, "meta": {
        "model": args.model, "combine": args.combine, "lam": args.lam, "bits": args.bits,
        "mask_fraction": args.mask_fraction, "mask_budget": args.mask_budget, "quantile": args.quantile,
        "pairs": args.pairs, "n_pairs": len(pairs), "total_params": total, "kept_params": kept,
        "delta_w": "corrupt_model" if corrupt_sd is not None else f"rtn{args.bits}bit",
    }}, out_path)
    print(f"[done] saved mask -> {out_path} in {time.time()-t0:.1f}s", flush=True)
    if sidecar is not None:
        torch.save(sidecar, args.save_saliency_out)
        print(f"[done] saved saliency sidecar -> {args.save_saliency_out}", flush=True)


if __name__ == "__main__":
    main()
