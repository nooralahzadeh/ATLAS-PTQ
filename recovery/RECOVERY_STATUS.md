# Scratch incident recovery — 2026-06-10

## Root cause (evidence-based)

This was a **capstor/Lustre scratch storage incident**, not a git or user error.
Independent signatures all point to lost/zeroed file *contents* while inodes
survived:

1. **30 core dumps written simultaneously at 2026-06-08 14:50**, all **0 bytes** —
   processes couldn't finish writing their cores (I/O failure signature), not a
   clean logical crash.
2. **`git fsck` found a ref zeroed to an all-zero SHA1** (`refs/remotes/origin/HEAD`
   = `000…0`) — a file whose data blocks were lost while metadata remained. Fixed
   via `git remote set-head origin -a`.
3. **Both tracked AND untracked files vanished** from the working tree. No git
   command produces this: `clean` never removes tracked files; `checkout`/`restore`
   *restore* tracked files; `reset --hard` doesn't touch untracked files.
4. **No destructive command in shell history** (only `rm -rf core_nid00`).
5. The `vlm` branch holds only junk commits ("delete me"), so nothing was hidden by
   the `vlm→main` checkout in the reflog.

Contributing stressor: the earlier parallel-extraction **CPU OOM (~416 GB RSS)**
hammered the node around the same window. Scratch has **no snapshot/backup**, so
untracked files with no git copy were only recoverable via leftover `.pyc`.

## Recovery strategy

Scratch is **not** snapshotted, so recovery relies on:

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
- ~~`scripts/extraction/convert_tdso_mask.py`~~ → **REBUILT** (verified against
  `TACQ/gptq/llama.py` `--important_mask` contract: flat `{config_key: bool}` dict,
  `config_key = "model.layers.{i}.{name}.weight"`). Just unwraps `{"masks": ...}`.

Reproduction recipe per `.sh`: `grep '\[cfg\]\|gptq\|wbits\|eval' tacq_data/results/<stage>.log`
gives the exact model / combine / bits / frac / pairs / GPTQ flags to wire back up.

## Orchestration recipe (recovered from `downstream_firm_w2.log`)

The lost runners follow this exact structure (markers verbatim from the log), so
they can be rebuilt deterministically:

```
run_downstream_firm_resume.sh   (sbatch_downstream_firm_resume.sh: a0125/normal/12h/4gpu)
  Phase 0: build contrastive pairs, all seeds  (data_prep_contrastive.py)
  Phase 1: audit (strict) — protocol parity + eval-row leakage check (calib_split_policy)
  Phase 2: per seed -> run_downstream_seed_llama31.sh
             Step 1/3: contrastive pairs (seed, mmlu=test75 gsm8k=tacq)
             Step 2/3: TaCQ task-conditioned, PARALLEL 4-GPU
                       -> run_task_conditioned_parallel_llama31.sh
                          gpu0..3 = run_tacq_task_conditioned_llama31.sh task={GSM8k,MMLU_STEM,
                          MMLU_humanities,MMLU_social_sciences}
                            per task: build corrupt -> measure_importances (TACQ) ->
                                      gptq --important_mask --true-sequential --wbits N -> eval
             Step 3/3: T-DSO v2 mult, SEQUENTIAL gpu0 -> run_tdso_task_sequential_llama31.sh
                            per task: extract_tdso_v2_h200.py --combine mult ->
                                      convert_tdso_mask.py -> gptq -> eval
  Phase 3: fixed-mask ablation (GSM8k) -> run_fixed_mask_bitwidth_ablation.sh
```

`pretrain_eval_helpers.sh` provides the eval functions (GSM8k CoT exec / MMLU
acc). It is **not** yet rebuilt — its CoT/answer parsing must match the surviving
result logs exactly, so reconstruct it against `tacq_data/results/*eval*` outputs
rather than from memory.

## Prevention

- `git add` the reconstructed `scripts/**` now (this commit).
- Going forward keep `scripts/` and `data/contrastive/*.jsonl` tracked; keep large
  `*.pt` out via `.gitignore` (mirror to `$STORE`/project space, not scratch).
