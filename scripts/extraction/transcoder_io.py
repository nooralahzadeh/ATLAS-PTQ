"""Standalone loader + math for the facebook/crv-8b-instruct-transcoders TopK PLTs.

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

Reconstructed from cpython-310 bytecode after the 2026-06-10 scratch incident.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import yaml
from safetensors import safe_open


@dataclass
class TranscoderConfig:
    model_name: str
    feature_input_hook: str
    feature_output_hook: str
    activation: str
    k: int


def load_config(transcoder_dir: str | Path) -> TranscoderConfig:
    with open(Path(transcoder_dir) / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return TranscoderConfig(
        model_name=cfg["model_name"],
        feature_input_hook=cfg["feature_input_hook"],
        feature_output_hook=cfg["feature_output_hook"],
        activation=cfg["activation"],
        k=int(cfg["k"]),
    )


class LayerTranscoder:
    """Holds one layer's transcoder weights and applies encode/decode.

    All tensors are kept resident on-device (H200 budget) and require no grad:
    the transcoder only defines the *target direction*, gradients flow through the
    base model's physical weights, not the transcoder.
    """

    def __init__(self, W_enc: torch.Tensor, W_dec: torch.Tensor,
                 b_enc: torch.Tensor, b_dec: torch.Tensor, k: int) -> None:
        self.W_enc = W_enc
        self.W_dec = W_dec
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.k = k
        self.d_transcoder, self.d_model = W_enc.shape

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> sparse TopK features (..., d_transcoder).

        Weights are moved to x.device on demand so transcoders can live on CPU
        and be streamed per-layer (GH200 C2C makes this cheap), avoiding the
        ~68 GB needed to keep all 32 transcoders resident on GPU.
        """
        W_enc = self.W_enc.to(x.device, non_blocking=True)
        b_enc = self.b_enc.to(x.device, non_blocking=True)
        pre = torch.nn.functional.linear(x.to(W_enc.dtype), W_enc, b_enc)
        _, idx = torch.topk(pre, k=self.k, dim=-1)
        gate = torch.zeros_like(pre)
        gate.scatter_(dim=-1, index=idx, value=1)
        return pre * gate

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (..., d_transcoder) -> reconstruction (..., d_model)."""
        W_dec = self.W_dec.to(feats.device, non_blocking=True)
        b_dec = self.b_dec.to(feats.device, non_blocking=True)
        return (feats @ W_dec) + b_dec


def load_layer_transcoder(path: str | Path, k: int, device: torch.device | str = "cuda",
                          dtype: torch.dtype = torch.bfloat16) -> LayerTranscoder:
    tensors = {}
    with safe_open(str(path), framework="pt", device=str(device)) as f:
        keys = set(f.keys())
        for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
            if key not in keys:
                raise KeyError(f"{path} missing transcoder key '{key}' (have {sorted(keys)})")
            tensors[key] = f.get_tensor(key).to(dtype)
    return LayerTranscoder(
        W_enc=tensors["W_enc"], W_dec=tensors["W_dec"],
        b_enc=tensors["b_enc"], b_dec=tensors["b_dec"], k=k,
    )


def resolve_transcoder_dir(repo_id: str, local_dir: str | None = None) -> Path:
    """Return a directory containing config.yaml + layer_*.safetensors.

    If ``local_dir`` is given and valid, use it; otherwise locate (or fetch) the
    HF snapshot from the local cache.
    """
    if local_dir is not None:
        d = Path(local_dir)
        if (d / "config.yaml").is_file():
            return d
        raise FileNotFoundError(f"{d} has no config.yaml")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id))


def load_all_transcoders(transcoder_dir: str | Path, n_layers: int, k: int,
                         device: torch.device | str = "cuda",
                         dtype: torch.dtype = torch.bfloat16) -> list[LayerTranscoder]:
    transcoder_dir = Path(transcoder_dir)
    transcoders = []
    for layer in range(n_layers):
        path = transcoder_dir / f"layer_{layer}.safetensors"
        transcoders.append(load_layer_transcoder(path, k=k, device=device, dtype=dtype))
    return transcoders
