"""
Assemble TaCQ mixed-precision circuits: sparse FP16 outliers on a quantized base.

Loads per-layer binary masks, extracts FP16 circuit weights from a reference model,
applies uniform simulated quantization to the base weights, and registers forward
hooks that overlay sparse FP16 values at inference time.
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.hf_auth import require_huggingface_token
from utils.tacq_saliency import uniformly_quantize_weight


LINEAR_SUFFIX = ".weight"


@dataclass
class SparseCircuitEntry:
    """Memory-efficient sparse storage for one linear weight matrix."""

    shape: tuple[int, ...]
    flat_indices: torch.Tensor  # int64 on CPU
    values: torch.Tensor  # fp16 on CPU


@dataclass
class CircuitAssemblyConfig:
    model_id: str = "Meta-Llama-3-8B-Instruct"
    wbits: int = 2
    device: str = "cuda"
    num_layers: int = 32


@dataclass
class AssembledCircuitModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    sparse_circuits: dict[str, SparseCircuitEntry]
    masks: dict[str, torch.Tensor]
    hook_handles: list[Any] = field(default_factory=list)


def _normalize_mask_dict(raw: Any, layer_idx: int | None = None) -> dict[str, torch.Tensor]:
    """Accept several on-disk layouts and return {param_name: bool mask}."""
    if isinstance(raw, dict):
        if "important_masks" in raw:
            masks = raw["important_masks"]
        elif "masks" in raw:
            masks = raw["masks"]
        elif all(isinstance(v, torch.Tensor) for v in raw.values()):
            masks = raw
        else:
            raise ValueError(f"Unrecognized mask payload keys: {list(raw.keys())}")
    else:
        raise ValueError("Mask file must contain a dict.")

    normalized: dict[str, torch.Tensor] = {}
    prefix = f"model.layers.{layer_idx}." if layer_idx is not None else None

    for key, mask in masks.items():
        name = key if key.endswith(LINEAR_SUFFIX) else f"{key}{LINEAR_SUFFIX}"
        if prefix and not name.startswith(prefix):
            name = f"{prefix}{name.removeprefix('model.layers.')}"
        normalized[name] = mask.bool().cpu()
    return normalized


def load_layer_mask_file(path: Path, layer_idx: int) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "layer_idx" in raw:
        layer_idx = int(raw["layer_idx"])
    masks = _normalize_mask_dict(raw, layer_idx=layer_idx)
    del raw
    aggressive_cleanup()
    return masks


def summarize_masks_from_directory(
    mask_dir: Path,
    num_layers: int = 32,
) -> tuple[int, int, int]:
    """Return (module_count, kept_weights, total_weights) without keeping all masks in RAM."""
    search_dirs = _mask_search_dirs(mask_dir)
    module_count = 0
    kept = 0
    total = 0

    for layer_idx in range(num_layers):
        path = _find_layer_mask_path(search_dirs, layer_idx)
        if path is None:
            raise FileNotFoundError(f"Missing layer mask file for layer {layer_idx} in {mask_dir}")
        layer_masks = load_layer_mask_file(path, layer_idx)
        module_count += len(layer_masks)
        kept += sum(m.sum().item() for m in layer_masks.values())
        total += sum(m.numel() for m in layer_masks.values())
        del layer_masks
        aggressive_cleanup()

    return module_count, kept, total


def _mask_search_dirs(mask_dir: Path) -> list[Path]:
    search_dirs = [mask_dir]
    nested = mask_dir / "layer_masks"
    if nested.is_dir():
        search_dirs.insert(0, nested)
    return search_dirs


def _find_layer_mask_path(search_dirs: list[Path], layer_idx: int) -> Path | None:
    for directory in search_dirs:
        for name in (f"layer_{layer_idx}_saliency.pt", f"layer_{layer_idx}.pt"):
            path = directory / name
            if path.exists():
                return path
    return None


def load_masks_from_directory(
    mask_dir: Path,
    num_layers: int = 32,
    allow_missing: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Load layer_0_saliency.pt … layer_{N-1}_saliency.pt from a directory.

    Warning: loads every layer into RAM (~1 GB per file). Prefer passing
    ``mask_dir`` directly to ``assemble_circuit_model`` for streaming assembly.
    """
    search_dirs = _mask_search_dirs(mask_dir)

    all_masks: dict[str, torch.Tensor] = {}
    missing: list[int] = []

    for layer_idx in range(num_layers):
        path = _find_layer_mask_path(search_dirs, layer_idx)
        if path is None:
            missing.append(layer_idx)
            continue
        all_masks.update(load_layer_mask_file(path, layer_idx))

    if missing and not allow_missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} layer mask file(s): {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''}. "
            f"Searched: {[str(d) for d in search_dirs]}. "
            f"Run full extraction: python scripts/extract_tacq_saliency.py"
        )
    return all_masks


def verify_all_layers_present(mask_dir: Path, num_layers: int = 32) -> int:
    """Return count of layer_*_saliency.pt files found."""
    count = 0
    for layer_idx in range(num_layers):
        for directory in (mask_dir, mask_dir / "layer_masks"):
            if (directory / f"layer_{layer_idx}_saliency.pt").exists():
                count += 1
                break
    return count


def load_masks_combined(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    return _normalize_mask_dict(raw)


def iter_decoder_linears(
    model: AutoModelForCausalLM,
) -> dict[str, nn.Linear]:
    modules: dict[str, nn.Linear] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and name:
                modules[f"model.layers.{layer_idx}.{name}{LINEAR_SUFFIX}"] = module
    return modules


def extract_sparse_circuits(
    reference_state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> dict[str, SparseCircuitEntry]:
    """Carve FP16 circuit weights (mask==True) into compact CPU sparse tensors."""
    circuits: dict[str, SparseCircuitEntry] = {}
    for name, mask in masks.items():
        if name not in reference_state:
            raise KeyError(f"Mask {name} not found in reference state dict.")
        weight = reference_state[name].detach().cpu().float()
        mask_cpu = mask.cpu().bool()
        if mask_cpu.shape != weight.shape:
            raise ValueError(f"Shape mismatch for {name}: mask {mask_cpu.shape} vs weight {weight.shape}")
        flat_idx = mask_cpu.view(-1).nonzero(as_tuple=False).squeeze(-1)
        if flat_idx.numel() == 0:
            continue
        values = weight.view(-1)[flat_idx].to(torch.float16)
        circuits[name] = SparseCircuitEntry(
            shape=tuple(weight.shape),
            flat_indices=flat_idx.long(),
            values=values,
        )
    return circuits


def build_quantized_base_state(
    reference_state: dict[str, torch.Tensor],
    linear_names: list[str],
    wbits: int,
) -> dict[str, torch.Tensor]:
    """Simulate uniform per-channel GPTQ weights (W_quant) for the base model."""
    quant_state: dict[str, torch.Tensor] = {}
    for i, name in enumerate(linear_names):
        w = reference_state[name].detach().cpu()
        quant_state[name] = uniformly_quantize_weight(w, wbits).to(torch.float16)
        if (i + 1) % 32 == 0:
            aggressive_cleanup()
    return quant_state


def apply_layer_quant_and_circuits(
    ref_layer: nn.Module,
    infer_layer: nn.Module,
    layer_idx: int,
    layer_masks: dict[str, torch.Tensor],
    wbits: int,
) -> dict[str, SparseCircuitEntry]:
    """Process one decoder block: carve sparse FP16 circuits and write quant weights."""
    circuits: dict[str, SparseCircuitEntry] = {}
    prefix = f"model.layers.{layer_idx}."

    for name, ref_module in ref_layer.named_modules():
        if not isinstance(ref_module, nn.Linear) or not name:
            continue
        full_name = f"{prefix}{name}{LINEAR_SUFFIX}"
        infer_module = infer_layer.get_submodule(name)
        if not isinstance(infer_module, nn.Linear):
            raise TypeError(f"Expected Linear at {full_name}, got {type(infer_module)}")

        w_cpu = ref_module.weight.data.detach().cpu().float()

        if full_name in layer_masks:
            mask_cpu = layer_masks[full_name].cpu().bool()
            flat_idx = mask_cpu.view(-1).nonzero(as_tuple=False).squeeze(-1)
            if flat_idx.numel() > 0:
                circuits[full_name] = SparseCircuitEntry(
                    shape=tuple(w_cpu.shape),
                    flat_indices=flat_idx.long(),
                    values=w_cpu.view(-1)[flat_idx].to(torch.float16),
                )

        w_quant = uniformly_quantize_weight(w_cpu, wbits).to(
            device=infer_module.weight.device,
            dtype=infer_module.weight.dtype,
        )
        infer_module.weight.data.copy_(w_quant)
        del w_cpu, w_quant

    aggressive_cleanup()
    return circuits


def apply_quantized_weights_and_extract_circuits(
    ref_model: AutoModelForCausalLM,
    infer_model: AutoModelForCausalLM,
    masks: dict[str, torch.Tensor],
    wbits: int,
) -> dict[str, SparseCircuitEntry]:
    """
    Single pass over decoder linears: extract sparse FP16 circuits and write W_quant
    into the inference model without materialising a full duplicate state dict.
    """
    ref_linears = iter_decoder_linears(ref_model)
    infer_linears = iter_decoder_linears(infer_model)
    circuits: dict[str, SparseCircuitEntry] = {}

    for name in ref_linears:
        if name not in infer_linears:
            raise KeyError(f"{name} missing from inference model.")
        w_cpu = ref_linears[name].weight.data.detach().cpu().float()

        if name in masks:
            mask_cpu = masks[name].cpu().bool()
            flat_idx = mask_cpu.view(-1).nonzero(as_tuple=False).squeeze(-1)
            if flat_idx.numel() > 0:
                circuits[name] = SparseCircuitEntry(
                    shape=tuple(w_cpu.shape),
                    flat_indices=flat_idx.long(),
                    values=w_cpu.view(-1)[flat_idx].to(torch.float16),
                )

        w_quant = uniformly_quantize_weight(w_cpu, wbits).to(
            device=infer_linears[name].weight.device,
            dtype=infer_linears[name].weight.dtype,
        )
        infer_linears[name].weight.data.copy_(w_quant)
        del w_cpu, w_quant
        aggressive_cleanup()

    return circuits


def apply_quantized_weights_and_extract_circuits_streaming(
    ref_model: AutoModelForCausalLM,
    infer_model: AutoModelForCausalLM,
    mask_dir: Path,
    num_layers: int,
    wbits: int,
) -> dict[str, SparseCircuitEntry]:
    """Layer-wise mask loading to keep host RAM bounded on 24 GB machines."""
    search_dirs = _mask_search_dirs(mask_dir)
    circuits: dict[str, SparseCircuitEntry] = {}

    for layer_idx in range(num_layers):
        path = _find_layer_mask_path(search_dirs, layer_idx)
        if path is None:
            raise FileNotFoundError(f"Missing mask file for layer {layer_idx} in {mask_dir}")

        print(f"  Layer {layer_idx + 1}/{num_layers}: {path.name}")
        layer_masks = load_layer_mask_file(path, layer_idx)
        layer_circuits = apply_layer_quant_and_circuits(
            ref_model.model.layers[layer_idx],
            infer_model.model.layers[layer_idx],
            layer_idx,
            layer_masks,
            wbits,
        )
        circuits.update(layer_circuits)
        del layer_masks, layer_circuits
        aggressive_cleanup()

    return circuits


def extract_sparse_circuits_streaming(
    ref_model: AutoModelForCausalLM,
    mask_dir: Path,
    num_layers: int,
) -> dict[str, SparseCircuitEntry]:
    """Extract FP16 circuit weights layer-by-layer without a full state_dict copy."""
    search_dirs = _mask_search_dirs(mask_dir)
    circuits: dict[str, SparseCircuitEntry] = {}

    for layer_idx in range(num_layers):
        path = _find_layer_mask_path(search_dirs, layer_idx)
        if path is None:
            raise FileNotFoundError(f"Missing mask file for layer {layer_idx} in {mask_dir}")

        layer_masks = load_layer_mask_file(path, layer_idx)
        ref_layer = ref_model.model.layers[layer_idx]
        prefix = f"model.layers.{layer_idx}."

        for name, ref_module in ref_layer.named_modules():
            if not isinstance(ref_module, nn.Linear) or not name:
                continue
            full_name = f"{prefix}{name}{LINEAR_SUFFIX}"
            if full_name not in layer_masks:
                continue
            weight = ref_module.weight.data.detach().cpu().float()
            mask_cpu = layer_masks[full_name].cpu().bool()
            flat_idx = mask_cpu.view(-1).nonzero(as_tuple=False).squeeze(-1)
            if flat_idx.numel() == 0:
                continue
            circuits[full_name] = SparseCircuitEntry(
                shape=tuple(weight.shape),
                flat_indices=flat_idx.long(),
                values=weight.view(-1)[flat_idx].to(torch.float16),
            )
            del weight, mask_cpu

        del layer_masks
        aggressive_cleanup()

    return circuits


def load_gptq_checkpoint_model(
    config: CircuitAssemblyConfig,
    checkpoint: Path,
) -> AssembledCircuitModel:
    """Load a finished GPTQ+TaCQ checkpoint (mixed precision already baked in)."""
    require_huggingface_token()
    loadstring = f"meta-llama/{config.model_id}"

    tokenizer = AutoTokenizer.from_pretrained(loadstring)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading GPTQ+TaCQ checkpoint: {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        loadstring,
        torch_dtype=torch.float16,
        device_map=config.device,
        low_cpu_mem_usage=True,
    )
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    del state, state_dict
    aggressive_cleanup()
    model.eval()

    return AssembledCircuitModel(
        model=model,
        tokenizer=tokenizer,
        sparse_circuits={},
        masks={},
        hook_handles=[],
    )


def _merge_sparse_into_weight(
    base_weight: torch.Tensor,
    entry: SparseCircuitEntry,
) -> torch.Tensor:
    merged = base_weight.clone()
    idx = entry.flat_indices.to(merged.device)
    vals = entry.values.to(device=merged.device, dtype=merged.dtype)
    merged.view(-1)[idx] = vals
    return merged


def register_circuit_hooks(
    model: AutoModelForCausalLM,
    sparse_circuits: dict[str, SparseCircuitEntry],
) -> list[Any]:
    """
    Forward pre/post hooks: overlay sparse FP16 circuit weights on each linear matmul.

    Base weights remain the simulated GPTQ values in module.weight; hooks temporarily
    patch in FP16 outliers for the duration of each forward call.
    """
    handles: list[Any] = []
    linears = iter_decoder_linears(model)

    for name, module in linears.items():
        if name not in sparse_circuits:
            continue
        entry = sparse_circuits[name]

        def make_pre_hook(circuit: SparseCircuitEntry):
            def pre_hook(mod: nn.Linear, _inputs: Any) -> None:
                mod._circuit_backup = mod.weight.data  # noqa: SLF001
                mod.weight.data = _merge_sparse_into_weight(mod._circuit_backup, circuit)

            return pre_hook

        def post_hook(mod: nn.Linear, _inputs: Any, _output: Any) -> None:
            if hasattr(mod, "_circuit_backup"):
                mod.weight.data = mod._circuit_backup
                del mod._circuit_backup

        handles.append(module.register_forward_pre_hook(make_pre_hook(entry)))
        handles.append(module.register_forward_hook(post_hook))

    return handles


def aggressive_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assemble_circuit_model(
    config: CircuitAssemblyConfig,
    masks: dict[str, torch.Tensor] | None = None,
    base_checkpoint: Path | None = None,
    mask_dir: Path | None = None,
) -> AssembledCircuitModel:
    """
    Build mixed-precision model:
      1. Load FP16 reference → extract sparse circuits → drop reference from RAM.
      2. Load or build quantized base weights.
      3. Attach forward hooks for FP16 overlay.

    Pass ``mask_dir`` (recommended) to stream one layer mask file at a time.
    """
    if masks is None and mask_dir is None:
        raise ValueError("Provide either ``masks`` or ``mask_dir``.")
    if masks is not None and mask_dir is not None:
        raise ValueError("Provide only one of ``masks`` or ``mask_dir``.")

    require_huggingface_token()
    loadstring = f"meta-llama/{config.model_id}"

    tokenizer = AutoTokenizer.from_pretrained(loadstring)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading inference model on GPU …")
    model = AutoModelForCausalLM.from_pretrained(
        loadstring,
        torch_dtype=torch.float16,
        device_map=config.device,
        low_cpu_mem_usage=True,
    )

    print("Loading FP16 reference on CPU (temporary) …")
    ref_model = AutoModelForCausalLM.from_pretrained(
        loadstring,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    if base_checkpoint and base_checkpoint.exists():
        print(f"Loading quantized base checkpoint: {base_checkpoint}")
        quant_payload = torch.load(base_checkpoint, map_location="cpu")
        quant_state = quant_payload.get("state_dict", quant_payload)
        model.load_state_dict(quant_state, strict=False)
        del quant_payload, quant_state
        if mask_dir is not None:
            print("Extracting sparse FP16 circuit weights from reference (streaming) …")
            sparse_circuits = extract_sparse_circuits_streaming(
                ref_model, mask_dir, config.num_layers
            )
            masks = {}
        else:
            assert masks is not None
            print("Extracting sparse FP16 circuit weights from reference …")
            reference_state = {k: v.detach().cpu() for k, v in ref_model.state_dict().items()}
            sparse_circuits = extract_sparse_circuits(reference_state, masks)
            del reference_state
    elif mask_dir is not None:
        print(
            f"Simulating uniform {config.wbits}-bit base + carving circuits "
            f"(streaming {config.num_layers} layers) …"
        )
        sparse_circuits = apply_quantized_weights_and_extract_circuits_streaming(
            ref_model, model, mask_dir, config.num_layers, config.wbits
        )
        masks = {}
    else:
        assert masks is not None
        print(f"Simulating uniform {config.wbits}-bit base + carving circuits (streaming) …")
        sparse_circuits = apply_quantized_weights_and_extract_circuits(
            ref_model, model, masks, config.wbits
        )

    del ref_model
    aggressive_cleanup()

    total_outliers = sum(e.values.numel() for e in sparse_circuits.values())
    print(f"  {len(sparse_circuits)} modules, {total_outliers:,} FP16 outlier weights")

    print("Registering circuit forward hooks …")
    handles = register_circuit_hooks(model, sparse_circuits)
    model.eval()

    return AssembledCircuitModel(
        model=model,
        tokenizer=tokenizer,
        sparse_circuits=sparse_circuits,
        masks=masks or {},
        hook_handles=handles,
    )


def remove_circuit_hooks(assembled: AssembledCircuitModel) -> None:
    for handle in assembled.hook_handles:
        handle.remove()
    assembled.hook_handles.clear()
