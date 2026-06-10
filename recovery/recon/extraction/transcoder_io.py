# Source Generated with Decompyle++
# File: transcoder_io.cpython-310.pyc (Python 3.10)

'''Standalone loader + math for the facebook/crv-8b-instruct-transcoders TopK PLTs.

These per-layer MLP transcoders approximate, for each decoder layer ``l``::

    x_l   = post_attention_layernorm(hidden_states)      # == TransformerLens mlp.hook_in
    feats = TopK_k( x_l @ W_enc.T + b_enc )               # k = 128
    recon = feats @ W_dec + b_dec                         # ~= mlp.hook_out (MLP output)

Verified against the upstream fork (zsquaredz/circuit-tracer):
  * ReplacementMLP.forward hooks the MLP *input* and *output*; the fork loads the
    HookedTransformer with ``fold_ln=False``, so ``mlp.hook_in`` is exactly the
    RMSNorm output (including the learnable weight) that HF feeds to ``layer.mlp``.
  * TopK keeps the k largest pre-activations (no ReLU), zeroing the rest.

Because the transcoder is just linear maps + TopK, we apply it directly to HF
activations via forward hooks; TransformerLens is not needed at runtime.
'''
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import yaml
from safetensors import safe_open
TranscoderConfig = dataclass(<NODE:12>)

def load_config(transcoder_dir = None):
    with open(Path(transcoder_dir) / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
        None(None, None, None)
    with None:
        if not None:
            pass
    return TranscoderConfig(cfg['model_name'], cfg['feature_input_hook'], cfg['feature_output_hook'], cfg['activation'], int(cfg['k']), **('model_name', 'feature_input_hook', 'feature_output_hook', 'activation', 'k'))


class LayerTranscoder:
    """Holds one layer's transcoder weights and applies encode/decode.

    All tensors are kept resident on-device (H200 budget) and require no grad:
    the transcoder only defines the *target direction*, gradients flow through the
    base model's physical weights, not the transcoder.
    """
    
    def __init__(self, W_enc, W_dec = None, b_enc = None, b_dec = None, k = ('W_enc', 'torch.Tensor', 'W_dec', 'torch.Tensor', 'b_enc', 'torch.Tensor', 'b_dec', 'torch.Tensor', 'k', 'int', 'return', 'None')):
        self.W_enc = W_enc
        self.W_dec = W_dec
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.k = k
        (self.d_transcoder, self.d_model) = W_enc.shape

    
    def encode(self = None, x = None):
        '''x: (..., d_model) -> sparse TopK features (..., d_transcoder).

        Weights are moved to x.device on demand so transcoders can live on CPU
        and be streamed per-layer (GH200 C2C makes this cheap), avoiding the
        ~68 GB needed to keep all 32 transcoders resident on GPU.
        '''
        W_enc = self.W_enc.to(x.device, True, **('non_blocking',))
        b_enc = self.b_enc.to(x.device, True, **('non_blocking',))
        pre = torch.nn.functional.linear(x.to(W_enc.dtype), W_enc, b_enc)
        (_, idx) = torch.topk(pre, self.k, -1, **('k', 'dim'))
        gate = torch.zeros_like(pre)
        gate.scatter_(-1, idx, 1, **('dim', 'index', 'value'))
        return pre * gate

    encode = None(encode)
    
    def decode(self = None, feats = None):
        '''feats: (..., d_transcoder) -> reconstruction (..., d_model).'''
        W_dec = self.W_dec.to(feats.device, True, **('non_blocking',))
        b_dec = self.b_dec.to(feats.device, True, **('non_blocking',))
        return (feats @ W_dec) + b_dec

    decode = None(decode)


def load_layer_transcoder(path = None, k = None, device = None, dtype = ('cuda', torch.bfloat16)):
    tensors = { }
    with safe_open(str(path), 'pt', str(device), **('framework', 'device')) as f:
        keys = set(f.keys())
        for key in ('W_enc', 'W_dec', 'b_enc', 'b_dec'):
            if key not in keys:
                raise KeyError(f'''{path} missing transcoder key \'{key}\' (have {sorted(keys)})''')
            tensors[key] = f.get_tensor(key).to(dtype)
        None(None, None, None)
    with None:
        if not None:
            pass
    return LayerTranscoder(tensors['W_enc'], tensors['W_dec'], tensors['b_enc'], tensors['b_dec'], k, **('W_enc', 'W_dec', 'b_enc', 'b_dec', 'k'))


def resolve_transcoder_dir(repo_id = None, local_dir = None):
    '''Return a directory containing config.yaml + layer_*.safetensors.

    If ``local_dir`` is given and valid, use it; otherwise locate (or fetch) the
    HF snapshot from the local cache.
    '''
    if local_dir is not None:
        d = Path(local_dir)
        if (d / 'config.yaml').is_file():
            return d
        raise None(f'''{d} has no config.yaml''')
    snapshot_download = snapshot_download
    import huggingface_hub
    return Path(snapshot_download(repo_id))


def load_all_transcoders(transcoder_dir = None, n_layers = None, k = None, device = ('cuda', torch.bfloat16), dtype = ('transcoder_dir', 'str | Path', 'n_layers', 'int', 'k', 'int', 'device', 'torch.device | str', 'dtype', 'torch.dtype', 'return', 'list[LayerTranscoder]')):
    transcoder_dir = Path(transcoder_dir)
    transcoders = []
    for layer in range(n_layers):
        path = transcoder_dir / f'''layer_{layer}.safetensors'''
        transcoders.append(load_layer_transcoder(path, k, device, dtype, **('k', 'device', 'dtype')))
    return transcoders

