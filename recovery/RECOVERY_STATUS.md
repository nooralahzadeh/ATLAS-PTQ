# Scratch incident recovery — 2026-06-10

Capstor scratch lost most **untracked** pipeline scripts (and briefly showed
tracked files as deleted). Scratch is **not** snapshotted, so recovery relies on:

1. **`.pyc` bytecode** in `__pycache__/` → decompiled with `pycdc` (built in
   `recovery/pycdc/`). Blueprints in `recovery/recon/`.
2. **My verbatim memory** for the two files I authored this session.
3. **Result logs** (`tacq_data/results/*.log`) — every stage echoes its exact
   `[cfg]`/invocation, so the lost `.sh` wrappers can be rebuilt faithfully.

## What SURVIVED (no action needed)

- All produced artifacts: `masks/*.pt` (43) and `tacq_data/**/*.pt` (290 incl.
  quantized models, e.g. `ablsrc_mult_GSM8k_2bit_quantized_model.pt`).
- All result logs: `tacq_data/results/*.log` (134).
- Git-tracked code (restored via `git restore`): `TACQ/gptq/llama.py`, TACQ utils.
- Surviving scripts: `scripts/{merge_spider_shard_preds.py, agents/*, evaluation/*,
  setup_llama31_venv.sh, *dinsql*, *circuit_scale*, *evaluate_base_fp16*,
  *steer_weight_squeeze*, *spider*}`.
- Contrastive calib data: `data/contrastive/*.jsonl` (verify count separately).
- `recovery/pyc/` — 3315 `.pyc` copied out of harm's way (recovery seeds).

## RECONSTRUCTED this session (placed back in tree)

| File | Source | Confidence |
|------|--------|-----------|
| `scripts/extraction/build_baseline_mask.py` | verbatim memory | HIGH |
| `scripts/extraction/extract_dictfree_saliency.py` | verbatim memory | HIGH |
| `scripts/mask_budget.py` | bytecode (global path verbatim) | HIGH |
| `scripts/calib_split_policy.py` | bytecode (booleans restored to fail-closed) | HIGH |
| `scripts/extraction/transcoder_io.py` | bytecode + memory | HIGH |
| `scripts/extraction/extract_tdso_v2_h200.py` | bytecode helpers verbatim; `main()` mirrored from dict-free | MED — VERIFY |

**Verify `extract_tdso_v2_h200.py`** before trusting *new* masks: re-extract one
config and Jaccard-compare its mask to the surviving production mask.

## BLUEPRINT ONLY (in `recovery/recon/`, not yet cleaned into tree)

Decompiled, need hand-cleanup (botched `with`/comprehensions/decorators):
- `recovery/recon/data_prep_contrastive.py` (187 ln) — contrastive pair builder.
  Pairs themselves survived in `data/contrastive/`, so low urgency.
- `recovery/recon/extraction/extract_tdso_phase1_h200.py` (v1, transcoder-only).
- `recovery/recon/prepare_spider_data.py`, `recovery/recon/analysis/mask_jaccard_report.py`,
  `recovery/recon/extraction/{authors_recon_check,validate_ghost_teacher,validate_transcoder_recon}.py`.
- `extract_tacq_baseline` — pycdc **segfaulted**; `.pyc` preserved in
  `recovery/pyc/scripts/__pycache__/extract_tacq_baseline.cpython-311.pyc`
  (retry with `pylingual`/newer pycdc, or rebuild from `tacq_native_*` logs).

## LOST — no bytecode (must rebuild from logs + memory)

`.sh` wrappers are not compiled, so only logs + my partial memory remain. The
result logs contain the exact args, so these are reproducible with care:

- `scripts/run_tdso_v2_task_conditioned_llama31.sh`
- `scripts/run_tacq_task_conditioned_llama31.sh`
- `scripts/run_task_conditioned_parallel_llama31.sh`
- `scripts/run_tdso_task_sequential_llama31.sh`
- `scripts/run_downstream_seed_llama31.sh`
- `scripts/run_downstream_firm_resume.sh`
- `scripts/run_fixed_mask_bitwidth_ablation.sh`
- `scripts/run_saliency_source_ablation.sh`
- `scripts/pretrain_eval_helpers.sh`
- `scripts/sbatch_saliency_source_ablation.sh`
- `scripts/sbatch_saliency_dictfree.sh`
- `scripts/sbatch_downstream_firm_resume.sh`
- `scripts/sbatch_downstream_firm_3bit_resume.sh`
- `scripts/extraction/convert_tdso_mask.py` — **no `.pyc`**, ~35 ln; rebuild from memory.

Reproduction recipe per `.sh`: `grep '\[cfg\]\|gptq\|wbits\|eval' tacq_data/results/<stage>.log`
gives the exact model / combine / bits / frac / pairs / GPTQ flags to wire back up.

## Prevention

- `git add` the reconstructed `scripts/**` now (this commit).
- Going forward keep `scripts/` and `data/contrastive/*.jsonl` tracked; keep large
  `*.pt` out via `.gitignore` (mirror to `$STORE`/project space, not scratch).
