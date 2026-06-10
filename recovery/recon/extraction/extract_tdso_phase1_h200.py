# Source Generated with Decompyle++
# File: extract_tdso_phase1_h200.cpython-311.pyc (Python 3.11)

'''Transcoder-Directed Saliency Optimization (T-DSO), Phase 1 — H200.

Computes a global 2-bit-quantization importance mask for Llama-3.1-8B-Instruct by
attributing through *task-relevant transcoder features* instead of plain CE loss.

Pipeline (single global backward per batch, no layer loops):

  For each contrastive pair (clean prompt, corrupted prompt):
    1. (no_grad) forward the CORRUPTED prompt; per layer cache a per-feature task
       score a_corr = max over tokens of TopK transcoder features.
    2. (grad)   forward the CLEAN prompt; in a forward hook on every layer.mlp:
         f_clean   = transcoder.encode(mlp_input)                  # (B,T,F)
         a_clean   = max over tokens of f_clean                     # (B,F)
         delta     = relu(a_clean - a_corr)                         # task features
         tau       = quantile(delta>0, q)            # per example, default q=0.95
         f_mask    = delta > tau                                    # F_task
         y_target  = transcoder.decode(f_clean * f_mask).detach()   # idealized MLP out
         loss     += -sum( mlp_output * y_target )   over valid tokens
    3. loss.backward() (one call) -> gradients on all physical weights at once.
    4. Accumulate |grad| (TaCQ sample_abs semantics) into fp32 buffers.

  Saliency per physical weight:  S = (|W| * |W_grad|) * |W_2bit - W_fp16|
  Keep the global top fraction (default 0.35%, matching TaCQ) -> binary mask.

The transcoder ``mlp.hook_in`` equals HF ``post_attention_layernorm`` output and
``mlp.hook_out`` equals the MLP output (verified against the upstream fork loaded
with fold_ln=False), so we apply the transcoder directly to HF activations via
forward hooks — no TransformerLens at runtime.
'''
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transcoder_io import load_all_transcoders, load_config, resolve_transcoder_dir
TARGET_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

def configure_h200_backend(seed = None):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def _record_to_messages(text = None):
    '''Convert a stored contrastive prompt into chat messages.

    Records from data_prep_contrastive.py optionally carry ``<<SYS>>...<<USER>>...``
    markers (Spider). Everything else is treated as a single user message.
    '''
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
# WARNING: Decompyle incomplete


def tokenize_side(tokenizer = None, texts = None, max_len = None, device = ('texts', 'list[str]', 'max_len', 'int', 'device', 'torch.device', 'return', 'dict[str, torch.Tensor]')):
    pass
# WARNING: Decompyle incomplete


class TDSOController:
    '''Holds per-batch state shared by the per-layer MLP forward hooks.'''
    
    def __init__(self = None, transcoders = None, quantile = None):
        self.transcoders = transcoders
        self.quantile = quantile
        self.phase = 'idle'
        self.mask = None
        self.corr_summ = { }
        self.loss = None

    
    def _masked_token_max(self = None, feats = None):
        '''feats (B,T,F) -> (B,F) max over valid tokens.'''
        pass
    # WARNING: Decompyle incomplete

    
    def make_hook(self = None, layer_idx = None):
        pass
    # WARNING: Decompyle incomplete



def is_target(name = None):
