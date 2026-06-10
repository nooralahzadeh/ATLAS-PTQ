#!/bin/bash
# Downstream "firm numbers" driver (resumable): pairs -> audit -> per-seed
# task-conditioned (TaCQ parallel + T-DSO seq) -> fixed-mask ablation.
#
# RECONSTRUCTED after the 2026-06-10 scratch incident — VERIFY before production.
# Grounded on downstream_firm_w2.log phase markers.
# Env: SEEDS WBITS RUN_FIXED_MASK FIXED_MASK_TASKS FORCE_RECOMPUTE
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
cd "$ROOT"
SEEDS=(${SEEDS:-0 1 2})
WBITS="${WBITS:-2}"
TASKS=(${TASKS:-GSM8k MMLU_STEM MMLU_humanities MMLU_social_sciences})
export FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-1}"
RUN_FIXED_MASK="${RUN_FIXED_MASK:-1}"
FIXED_MASK_TASKS=(${FIXED_MASK_TASKS:-GSM8k})
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Downstream firm numbers w$WBITS seeds=${SEEDS[*]} ==="
log "Protocol: MMLU test75 calib, GSM8K tacq CoT, max_len=2048, n_calib=128"

log "Phase 0: contrastive pairs (all seeds)"
for s in "${SEEDS[@]}"; do
  python scripts/data_prep_contrastive.py --tasks "${TASKS[@]}" --seed "$s" \
    --mmlu-calib-mode test75 --gsm8k-format tacq --n-calib 128 \
    || log "WARN: pair build for seed=$s skipped/failed (may already exist)"
done

log "Phase 1: audit (strict)"
python - "$WBITS" "${SEEDS[@]}" <<'PY' || { log "AUDIT FAILED"; exit 1; }
import sys, glob
sys.path.insert(0, "scripts")
from calib_split_policy import assert_calib_pairs_path
seeds = sys.argv[2:]
ok = True
for s in seeds:
    suf = "" if s == "0" else f"_s{s}"
    for p in glob.glob(f"data/contrastive/*{suf}.jsonl"):
        if suf == "" and any(x in p for x in ("_s1", "_s2")):
            continue
        try:
            assert_calib_pairs_path(p, allow_legacy_eval=False)
        except Exception as e:
            print("LEAK?", p, e); ok = False
print("=== AUDIT OK ===" if ok else "=== AUDIT FAILED ===")
sys.exit(0 if ok else 1)
PY
log "No eval-row leakage detected."

log "Phase 2: task-conditioned runs (TaCQ parallel + T-DSO seq per seed)"
for s in "${SEEDS[@]}"; do
  log ">>> seed=$s"
  SEED="$s" WBITS="$WBITS" TASKS="${TASKS[*]}" FORCE_RECOMPUTE="$FORCE_RECOMPUTE" \
    bash "$ROOT/scripts/run_downstream_seed_llama31.sh"
done

if [[ "$RUN_FIXED_MASK" == "1" ]]; then
  log "Phase 3: fixed-mask ablation tasks=${FIXED_MASK_TASKS[*]} (parallel, seed 0)"
  pids=(); gpu=0
  for task in "${FIXED_MASK_TASKS[@]}"; do
    for method in tacq tdso; do
      TASK="$task" SEED=0 METHOD="$method" GPU="$gpu" WBITS="$WBITS" \
        FORCE_RECOMPUTE="$FORCE_RECOMPUTE" RESUME=1 \
        LOG="$ROOT/tacq_data/results/fixed_mask_${task,,}_${method}_w${WBITS}.log" \
        bash "$ROOT/scripts/run_fixed_mask_bitwidth_ablation.sh" \
          >> "$ROOT/tacq_data/results/fixed_mask_${task,,}_${method}_w${WBITS}.log" 2>&1 &
      pids+=($!)
      gpu=$(((gpu + 1) % 4))
    done
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" == "0" ]] || log "WARN: one or more fixed-mask workers failed"
fi

log "=== Downstream firm w$WBITS complete ==="
