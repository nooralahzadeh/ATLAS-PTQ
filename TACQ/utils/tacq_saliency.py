"""
Layer-wise TaCQ saliency extraction for memory-constrained GPUs (≈24GB).

Computes, for each decoder block in Llama-3:
    S(W) = (|W| · |∇_W L|) · |W_quant − W|

where W_quant is a uniform per-channel simulated quantisation at `wbits`.
Only one transformer layer has requires_grad=True at a time so peak VRAM
stays near a single-layer backward pass (~18GB on 8B FP16).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_gptq import BaseQuantizeConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from gptq.quant import Quantizer


@dataclass
class SaliencyConfig:
    """Hyperparameters for TaCQ saliency / outlier mask generation."""

    wbits: int = 2
    mask_fraction: float = 0.015  # top 1.5% kept in FP16
    max_length: int = 2048
    device: str = "cuda"


@dataclass
class SaliencyResult:
    """Outputs of extract_tacq_saliency."""

    saliency_scores: dict[str, torch.Tensor]  # CPU float tensors
    important_masks: dict[str, torch.Tensor]  # CPU bool tensors
    config: SaliencyConfig


# ---------------------------------------------------------------------------
# Uniform weight quantisation (matches TACQ corrupt-model GPTQ settings)
# ---------------------------------------------------------------------------


def uniformly_quantize_weight(
    weight: torch.Tensor,
    wbits: int,
    sym: bool = False,
) -> torch.Tensor:
    """
    Simulate uniform per-output-channel quantisation without running GPTQ.

    Uses the same Quantizer as TACQ gptq/llama.py (--wbits, perchannel=True,
    sym=False, mse=False).
    """
    quantizer = Quantizer()
    quantizer.configure(wbits, perchannel=True, sym=sym, mse=False)
    w = weight.detach().float()
    quantizer.find_params(w, weight=True)
    return quantizer.quantize(w).to(dtype=weight.dtype)


def build_quantizer_from_gptq_config(gptq_config: BaseQuantizeConfig) -> int:
    """Read bit-width from an auto-gptq BaseQuantizeConfig."""
    return int(gptq_config.bits)


# ---------------------------------------------------------------------------
# Calibration batch helpers
# ---------------------------------------------------------------------------


class TextCalibrationDataset(Dataset):
    """Tokenised calibration strings for causal LM loss."""

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            self.samples.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_single_example(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """DataLoader collate for batch_size=1 (already single sequence)."""
    return batch[0]


def compute_causal_lm_loss(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Standard shifted cross-entropy over all non-padding tokens.

    Gradients flow into whichever layer weights currently have requires_grad=True.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = input_ids[..., 1:].contiguous()

    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].bool()
        labels = labels.masked_fill(~shift_mask, -100)

    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Layer utilities
# ---------------------------------------------------------------------------


def freeze_model(model: nn.Module) -> None:
    """Disable gradients on every parameter."""
    model.requires_grad_(False)


def linear_modules_in_layer(
    layer: nn.Module,
    layer_idx: int,
) -> dict[str, nn.Linear]:
    """
    Return all nn.Linear modules inside one decoder block.

    Keys match HuggingFace parameter names, e.g.
    model.layers.0.self_attn.q_proj
    """
    modules: dict[str, nn.Linear] = {}
    prefix = f"model.layers.{layer_idx}"
    for name, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name:
            continue
        modules[f"{prefix}.{name}.weight"] = module
    return modules


def enable_grad_on_layer(linears: dict[str, nn.Linear]) -> None:
    for module in linears.values():
        module.weight.requires_grad_(True)


def disable_grad_on_layer(linears: dict[str, nn.Linear]) -> None:
    for module in linears.values():
        module.weight.requires_grad_(False)


def aggressive_cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def top_fraction_mask_within_layer(
    scores: dict[str, torch.Tensor],
    fraction: float,
) -> dict[str, torch.Tensor]:
    """
    Rank all weights in one decoder layer jointly; mark top `fraction` as True.

    Masks are returned on CPU as bool tensors matching each weight shape.
    """
    if not scores:
        return {}

    keys = list(scores.keys())
    shapes = {k: scores[k].shape for k in keys}
    flat_parts = [scores[k].reshape(-1).float() for k in keys]
    flat_scores = torch.cat(flat_parts)

    total = flat_scores.numel()
    k_select = max(1, int(fraction * total))
    k_select = min(k_select, total)

    _, top_indices = torch.topk(flat_scores, k_select)
    flat_mask = torch.zeros(total, dtype=torch.bool)
    flat_mask[top_indices] = True

    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for key in keys:
        numel = shapes[key].numel()
        masks[key] = flat_mask[offset : offset + numel].view(shapes[key]).clone()
        offset += numel
    return masks


def compute_tacq_saliency_scores(
    linears: dict[str, nn.Linear],
    accumulated_abs_grad: dict[str, torch.Tensor],
    wbits: int,
) -> dict[str, torch.Tensor]:
    """
    S(W) = (|W| · |∇_W L|) · |W_quant − W|

    Returns CPU float tensors keyed by module name (without '.weight' suffix).
    """
    scores: dict[str, torch.Tensor] = {}
    for name, module in linears.items():
        w = module.weight.data
        grad_abs = accumulated_abs_grad[name].to(device=w.device, dtype=torch.float32)
        w_quant = uniformly_quantize_weight(w, wbits)
        delta = (w_quant.float() - w.float()).abs()
        score = (w.float().abs() * grad_abs) * delta
        scores[name] = score.detach().cpu()
        del grad_abs, w_quant, delta, score
    return scores


# ---------------------------------------------------------------------------
# Core extraction loop
# ---------------------------------------------------------------------------


def save_layer_mask(
    layer_idx: int,
    layer_masks: dict[str, torch.Tensor],
    layer_scores: dict[str, torch.Tensor],
    output_dir: Path,
    config: SaliencyConfig,
) -> Path:
    """Write one layer's masks immediately so a long run can resume after crashes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"layer_{layer_idx}_saliency.pt"
    torch.save(
        {
            "layer_idx": layer_idx,
            "important_masks": layer_masks,
            "saliency_scores": layer_scores,
            "config": {
                "wbits": config.wbits,
                "mask_fraction": config.mask_fraction,
                "max_length": config.max_length,
            },
        },
        path,
    )
    kept = sum(m.sum().item() for m in layer_masks.values())
    total = sum(m.numel() for m in layer_masks.values())
    print(f"  → cached layer {layer_idx} to {path} ({kept:,}/{total:,} = {100*kept/total:.2f}%)")
    return path


def extract_tacq_saliency(
    model: AutoModelForCausalLM,
    calibration_texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    config: SaliencyConfig | None = None,
    gptq_config: BaseQuantizeConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    layer_output_dir: Path | None = None,
    skip_existing_layers: bool = True,
) -> SaliencyResult:
    """
    Layer-by-layer TaCQ saliency extraction on a single GPU.

    Parameters
    ----------
    model:
        FP16 Llama-3 loaded on GPU.
    calibration_texts:
        Chat-formatted Spider strings (128 recommended).
    tokenizer:
        Matching tokenizer for the model.
    config:
        Saliency / mask hyperparameters.
    gptq_config:
        Optional auto-gptq config; overrides config.wbits if provided.

    Returns
    -------
    SaliencyResult with CPU saliency tensors and bool outlier masks.
    """
    if config is None:
        config = SaliencyConfig()

    wbits = build_quantizer_from_gptq_config(gptq_config) if gptq_config else config.wbits
    config = SaliencyConfig(
        wbits=wbits,
        mask_fraction=config.mask_fraction,
        max_length=config.max_length,
        device=config.device,
    )

    device = torch.device(config.device)
    model.eval()
    freeze_model(model)

    dataset = TextCalibrationDataset(calibration_texts, tokenizer, config.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single_example,
    )

    all_scores: dict[str, torch.Tensor] = {}
    all_masks: dict[str, torch.Tensor] = {}
    decoder_layers = model.model.layers
    num_layers = len(decoder_layers)
    persist_layers_only = layer_output_dir is not None

    for layer_idx, layer in enumerate(decoder_layers):
        if progress_callback:
            progress_callback(layer_idx, num_layers)

        layer_file = None
        if layer_output_dir is not None:
            layer_file = layer_output_dir / f"layer_{layer_idx}_saliency.pt"
            if skip_existing_layers and layer_file.exists():
                print(f"  Layer {layer_idx}: skipping (already on disk)")
                if not persist_layers_only:
                    cached = torch.load(layer_file, map_location="cpu")
                    all_masks.update(cached.get("important_masks", cached))
                    all_scores.update(cached.get("saliency_scores", {}))
                continue

        linears = linear_modules_in_layer(layer, layer_idx)
        if not linears:
            continue

        enable_grad_on_layer(linears)
        accumulated_abs_grad = {
            name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu")
            for name, module in linears.items()
        }

        num_examples = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

            model.zero_grad(set_to_none=True)
            loss = compute_causal_lm_loss(model, input_ids, attention_mask)
            loss.backward()

            for name, module in linears.items():
                if module.weight.grad is None:
                    continue
                accumulated_abs_grad[name].add_(
                    module.weight.grad.detach().abs().float().cpu()
                )

            del loss, input_ids, attention_mask
            model.zero_grad(set_to_none=True)
            aggressive_cuda_cleanup()
            num_examples += 1

        if num_examples == 0:
            raise RuntimeError("Calibration dataloader produced zero examples.")

        # Average accumulated |grad| over the calibration batch.
        for name in accumulated_abs_grad:
            accumulated_abs_grad[name].div_(num_examples)

        layer_scores = compute_tacq_saliency_scores(linears, accumulated_abs_grad, config.wbits)
        layer_masks = top_fraction_mask_within_layer(layer_scores, config.mask_fraction)

        if layer_output_dir is not None:
            save_layer_mask(layer_idx, layer_masks, layer_scores, layer_output_dir, config)
        else:
            all_scores.update(layer_scores)
            all_masks.update(layer_masks)

        disable_grad_on_layer(linears)
        del accumulated_abs_grad, layer_scores, layer_masks, linears
        aggressive_cuda_cleanup()

    return SaliencyResult(
        saliency_scores=all_scores,
        important_masks=all_masks,
        config=config,
    )


def load_all_layer_masks_from_dir(output_dir: Path, num_layers: int = 32) -> tuple[dict, dict]:
    """Rebuild combined mask/score dicts from per-layer files on disk."""
    masks: dict[str, torch.Tensor] = {}
    scores: dict[str, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        path = output_dir / f"layer_{layer_idx}_saliency.pt"
        if not path.exists():
            continue
        cached = torch.load(path, map_location="cpu")
        masks.update(cached.get("important_masks", {}))
        scores.update(cached.get("saliency_scores", {}))
    return masks, scores


def save_saliency_result(
    result: SaliencyResult,
    output_path: str,
) -> None:
    """Persist masks (+ scores) to disk for the downstream GPTQ step."""
    payload = {
        "config": {
            "wbits": result.config.wbits,
            "mask_fraction": result.config.mask_fraction,
            "max_length": result.config.max_length,
        },
        "important_masks": result.important_masks,
        "saliency_scores": result.saliency_scores,
    }
    torch.save(payload, output_path)
