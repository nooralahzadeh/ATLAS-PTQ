# Source Generated with Decompyle++
# File: extract_tdso_v2_h200.cpython-310.pyc (Python 3.10)

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
TARGET_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

def configure_h200_backend(seed = None):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def _record_to_messages(text = None):
    if text.startswith('<<SYS>>'):
        body = text[len('<<SYS>>'):]
        (sys_part, _, user_part) = body.partition('<<USER>>')
        return [
            {
                'role': 'system',
                'content': sys_part.strip() },
            {
                'role': 'user',
                'content': user_part.strip() }]
    return [
        {
            'role': None,
            'content': text }]


def load_pairs(paths = None, max_pairs = None):
    pairs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                pairs.append(json.loads(line))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    pass
                None(None, None, None)
                return None
            None(None, None, None)
        with None:
            if not None:
                pass
    return pairs


def tokenize_side(tokenizer, texts, max_len, device):
    rendered = (lambda .0 = None: [ tokenizer.apply_chat_template(_record_to_messages(t), False, True, **('tokenize', 'add_generation_prompt')) for t in .0 ])(texts)
    enc = tokenizer(rendered, 'pt', 'longest', True, max_len, False, **('return_tensors', 'padding', 'truncation', 'max_length', 'add_special_tokens'))
    return (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(enc.items())


class TDSOController:
    
    def __init__(self = None, transcoders = None, quantile = None):
        self.transcoders = transcoders
        self.quantile = quantile
        self.phase = 'idle'
        self.mask = None
        self.corr_summ = { }
        self.loss = None
        self.feature_density = { }

    
    def _masked_token_max(self = None, feats = None):
        pass
    # WARNING: Decompyle incomplete

    
    def make_hook(self = None, layer_idx = None):
        tc = self.transcoders[layer_idx]
        
        def hook(module = None, inputs = None, output = None):
            x_in = inputs[0]
            if self.phase == 'corr':
                with torch.no_grad():
                    feats = tc.encode(x_in)
                    self.corr_summ[layer_idx] = self._masked_token_max(feats).float()
                    None(None, None, None)
                    return None
                    with None:
                        if not None:
                            pass
                return None
            if self.phase == 'clean':
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
                    None(None, None, None)
                with None:
                    if not None:
                        pass
                y_target = y_target.detach()
                contrib = -(output * y_target)
                contrib = contrib * self.mask.unsqueeze(-1).to(contrib.dtype)
                layer_loss = contrib.sum()
                self.loss = layer_loss if self.loss is None else self.loss + layer_loss

        return hook



def is_target(name = None):
    if name.endswith(TARGET_SUFFIXES):
        pass
    return 'lm_head' not in name


def rtn_dequantize(W = None, bits = None):
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)
    scale = W.abs().amax(1, True, **('dim', 'keepdim')) / max(qmax, 1)
    scale = torch.clamp(scale, 1e-08, **('min',))
    q = torch.clamp(torch.round(W / scale), qmin, qmax)
    return q * scale

rtn_dequantize = None(rtn_dequantize)

def load_corrupt_weights(path = None, device = None):
    p = Path(path)
    if p.is_file() and p.suffix == '.pt':
        sd = torch.load(p, device, **('map_location',))
        return sd.get('state_dict', sd)
    raise None(f'''Unsupported corrupt-model path {path}''')

load_corrupt_weights = None(load_corrupt_weights)

def parse_args():
    ap = argparse.ArgumentParser(__doc__, **('description',))
    ap.add_argument('--pairs', '+', [
        'data/contrastive/spider_contrastive_train1360.jsonl'], **('nargs', 'default'))
    ap.add_argument('--allow-legacy-eval-pairs', 'store_true', 'DEBUG ONLY: skip check for eval-split contrastive files.', **('action', 'help'))
    ap.add_argument('--model', 'NousResearch/Meta-Llama-3.1-8B-Instruct', **('default',))
    ap.add_argument('--transcoder-repo', 'facebook/crv-8b-instruct-transcoders', **('default',))
    ap.add_argument('--transcoder-dir', None, **('default',))
    ap.add_argument('--transcoder-device', 'cpu', [
        'cpu',
        'cuda'], **('default', 'choices'))
    ap.add_argument('--corrupt-model', None, **('default',))
    ap.add_argument('--bits', int, 2, **('type', 'default'))
    ap.add_argument('--mask-fraction', float, 0.0035, **('type', 'default'))
    ap.add_argument('--mask-budget', 'global', [
        'global',
        'layer_adaptive'], 'global=single top-k (TaCQ-style); layer_adaptive=0.35%% split by transcoder feature density.', **('default', 'choices', 'help'))
    ap.add_argument('--quantile', float, 0.95, **('type', 'default'))
    ap.add_argument('--combine', 'boost', [
        'ce',
        'align',
        'boost',
        'add',
        'mult'], **('default', 'choices'))
    ap.add_argument('--lam', float, 1, 'weight of the align term', **('type', 'default', 'help'))
    ap.add_argument('--batch-size', int, 4, **('type', 'default'))
    ap.add_argument('--max-len', int, 512, **('type', 'default'))
    ap.add_argument('--max-pairs', int, None, **('type', 'default'))
    ap.add_argument('--seed', int, 0, **('type', 'default'))
    ap.add_argument('--out', 'masks/tdso_v2.pt', **('default',))
    ap.add_argument('--save-saliency-out', None, 'Optional sidecar .pt with g_ce, g_align, combined saliency tensors', **('default', 'help'))
    return ap.parse_args()


def main():
    args = parse_args()
    configure_h200_backend(args.seed)
    for p in args.pairs:
        assert_calib_pairs_path(p, args.allow_legacy_eval_pairs, **('allow_legacy_eval',))
    device = torch.device('cuda')
    dtype = torch.bfloat16
    need_ce = args.combine != 'align'
    need_align = args.combine != 'ce'
    print(f'''[cfg] model={args.model} combine={args.combine} lam={args.lam} bits={args.bits} frac={args.mask_fraction} budget={args.mask_budget} q={args.quantile} bs={args.batch_size} pairs={args.pairs}''', True, **('flush',))
    print('[load] tokenizer + model ...', True, **('flush',))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None
