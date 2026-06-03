# ATLAS-PTQ

TaCQ (Task-Circuit Quantization) replication for **Meta-Llama-3-8B-Instruct** on **Spider** text-to-SQL.

Based on the upstream [TACQ](https://github.com/The-Inscrutable-X/TACQ) codebase, extended with L4/cluster-safe scripts for saliency extraction, circuit assembly, and GPTQ.

## Quick start (cluster)

```bash
cd TACQ
python3.12 -m venv tacq_venv
source tacq_venv/bin/activate
pip install -r requirements.txt
pip install auto-gptq==0.7.1   # BUILD_CUDA_EXT=0 if CUDA ext build fails

# HuggingFace token (accept Llama-3 license first)
cp .env.example .env
# edit .env → HUGGINGFACE_TOKEN=hf_...
```

### Data layout (outside repo)

```text
/home/ubuntu/tacq_data/
  importances/     # layer_0_saliency.pt … layer_31_saliency.pt (~34 GB)
  checkpoints/     # GPTQ outputs
  results/         # Spider eval logs
```

Spider data: see `TACQ/datasets_directory/DATASETS.md`. Place under `TACQ/datasets_directory/Spider/`.

## Pipeline

| Step | Script | Notes |
|------|--------|-------|
| 1. Saliency (32 layers) | `bash scripts/run_full_extraction_screen.sh` | Resume-safe; ~2–3 min/layer |
| 2. GPTQ base | `bash scripts/run_gptq_tacq_base.sh` | Needs RAM for masks; use strong node |
| 3. Spider eval | `bash scripts/run_tacq_gptq_spider_eval.sh` | Exec accuracy via test-suite-sql-eval |
| All-in-one | `bash scripts/run_tacq_gptq_pipeline.sh` | Steps 2+3 |

Smoke tests: `TESTING=1` before each script.

### Paper targets (Spider exec, Spider-conditioned)

| Config | Target |
|--------|--------|
| Unquantized | 67.6% |
| TaCQ 2-bit | 21.92% |
| TaCQ 3-bit | 58.32% |

Unquantized baseline: `bash scripts/run_unquantized_baseline.sh`

## Key custom modules

- `utils/tacq_saliency.py` — layer-wise saliency + masks (memory-safe)
- `utils/circuit_assembly.py` — streaming assembly + GPTQ checkpoint load
- `scripts/extract_tacq_saliency.py` — full 32-layer extraction CLI
- `scripts/merge_layer_masks_for_gptq.py` — build GPTQ mask/yaml from layer files
- `scripts/run_gptq_tacq_base.sh` — GPTQ with saliency masks

## Mask fraction

Current replication uses **1.5%** outliers per layer (`--mask-fraction 0.015`). Paper uses 0.35% (`0.0035`); change via `FRACTION=0.0035` in extraction scripts.

## Status on L4 (24 GB)

- Saliency extraction: **complete** (32/32 layer files)
- Circuit assembly (simulated 2-bit): runs but useless without GPTQ base
- GPTQ checkpoint: blocked on RAM (~6.6 GB dense mask + model); run on cluster with ≥64 GB RAM recommended
