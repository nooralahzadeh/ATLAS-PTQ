# Results Ledger — ATLAS-PTQ

Canonical record of firm experimental numbers. Only commit numbers that come
straight from a run log (cite `engine=` + log path). Recreated 2026-06-10 after
the scratch incident (the previous untracked ledger was lost).

Model: `Meta-Llama-3.1-8B-Instruct` · FP16 outlier budget: **0.35%** · GPTQ
true-sequential · seed 0 unless noted.

---

## Saliency-source ablation (2-bit) — job 2513827

**Question.** Does the gain come from the *task circuit* and specifically from the
*conjunction* (CE ∩ circuit-alignment), or is it just bit-placement / magnitude?
And can we drop the external transcoder?

All arms share the identical mask format, the identical 0.35% global top-k
selection, and the identical GPTQ + eval pipeline. Only the **saliency signal**
that ranks weights differs.

Source log: `tacq_data/results/saliency_source_ablation_w2.log`

| Arm | Saliency signal | Dict? | GSM8k 2b | MMLU_STEM 2b |
|---|---|---|---|---|
| random | random weights | – | 2.5% | 30.3% |
| weight | \|W\| | – | 14.4% | 33.6% |
| magnitude | \|W\|·\|ΔW_quant\| | – | 16.7% | 27.8% |
| align_free | circuit alignment, MLP-neuron basis | no | 22.9% | 42.6% |
| ce (≈TaCQ) | CE-gradient saliency | – | 25.4% | 41.3% |
| align | circuit alignment, transcoder basis | yes | 25.6% | 39.2% |
| mult_free | CE ∩ alignment, MLP-neuron basis | no | 26.5% | **44.2%** |
| **mult** | **CE ∩ alignment, transcoder basis** | yes | **28.7%** | 38.9% |

MMLU_STEM arms completed by job 2517685 (2026-06-11) after fixing the
post-restore tokenizer bug; align_free/mult_free engines from job 2513954
(identical calibration semantics). MCQA chance level is 25%, so
random/weight/magnitude sit near chance.

### Interpretation (decision gates)

1. **Circuit matters, not just magnitude.** Both tasks: circuit-informed arms
   (ce/align/mult/_free) clearly beat weight/magnitude/random (GSM8k 22.9–28.7
   vs 2.5–16.7; STEM 38.9–44.2 vs near-chance 27.8–33.6). ✔
2. **The conjunction is the active ingredient — with the right basis.**
   GSM8k: mult 28.7 > ce 25.4 ≈ align 25.6 (transcoder conjunction wins).
   MMLU_STEM: mult_free 44.2 > ce 41.3 > align 39.2 ≈ mult 38.9 — here the
   conjunction only wins in the **dict-free** basis; the transcoder conjunction
   *underperforms* ce. Conjunctive saliency helps on both tasks, but the best
   feature basis is task-dependent. ✔ (qualified)
3. **The transcoder is removable — and sometimes harmful.** mult_free recovers
   92% of mult on GSM8k (26.5 vs 28.7) and **beats** it by +5.3 on MMLU_STEM
   (44.2 vs 38.9). The dictionary is an optional booster on GSM8k and a
   liability on STEM → the dict-free method is the safer default and the
   stronger paper story. ✔

**Status.** COMPLETE — GSM8k 2b 8/8 arms (job 2513827+2513954), MMLU_STEM 2b
8/8 arms (job 2517685 after fixing post-restore code drift: `use_fast=False`
tokenizer breakage and GSM8k truncation semantics; see RECOVERY_STATUS.md).

---

## 3-bit downstream head-to-head — job 2512614 (`down_firm3_r`)

TaCQ vs T-DSO v2-mult, matched 0.35% budget, 3 seeds. Job FAILED at the final
eval (v2-mult GSM8k seed2), so v2-mult has seeds 0–1 only; TaCQ has 0–2.
Source log: `tacq_data/results/downstream_firm_w3.log`. Metric: GSM8k accuracy,
MMLU average accuracy (test 75–100%).

| Task | TaCQ 3b (mean) | v2-mult 3b (mean) | Δ |
|---|---|---|---|
| GSM8k | 63.7 (63.8/63.2/64.0) | 63.3 (63.3/63.3) | −0.4 |
| MMLU_STEM | 56.5 (57.1/55.3/57.0) | 54.8 (54.0/55.7) | −1.7 |
| MMLU_humanities | 63.0 (61.7/64.3/62.9) | 63.1 (62.6/63.7) | +0.1 |
| MMLU_social_sciences | 73.0 (72.4/73.7/73.0) | 74.1 (74.5/73.7) | +1.1 |

**Finding.** At 3-bit the two methods are tied: every gap ≤1.7 pts and within
seed spread. The conjunction advantage seen at 2-bit (ablation table above)
**vanishes at 3-bit** → the contribution is a *low-bit* effect: task-circuit
saliency matters most when the bit budget is harshest.

### ✅ RESOLVED — firm "2-bit" T-DSO engines were actually 3-bit (wbits bug)
Investigation of `downstream_firm_w2.log`:
- The `tdsoV2mult_GSM8k_2bit = 64.9%` line is **stale provenance** — a
  `RESULT task=` summary echoed at 17:12:13, *before* the engine was built
  (17:45), repeated 5× in the log. Ignore all `RESULT task=` lines; only
  `RESULT calib=` lines are real evals.
- The real firm eval is `calib= 62.5%` (s0) / 60.7% (s1) — still anomalous.
- **Root cause:** the v2-mult GSM8k build block (17:33:43, the one that saved
  `tdsoV2mult_GSM8k_2bit_quantized_model.pt`) shows **224 layers all `bits = 3`**.
  `WBITS` was not propagated into the v2-mult GPTQ call, so every firm "2-bit"
  T-DSO engine is physically a **3-bit** model (file is byte-identical in size to
  the 3-bit engine; TaCQ 2-bit, built correctly at `bits = 2`, is a different
  size and scores 28.3%).

**Consequence.** Discard ALL firm 2-bit T-DSO v2-mult numbers. The trustworthy
genuine-2-bit numbers are:
- T-DSO mult GSM8k 2b = **28.7%** (saliency-source ablation, matched config)
- TaCQ GSM8k 2b = **28.3%** (firm, genuine `bits=2`) ≈ ablation ce 25.4%
The bug is the same one patched in `scripts/run_downstream_seed_llama31.sh`
(explicit `--wbits "$WBITS"`); firm 2-bit T-DSO must be rebuilt with the fix.

---

## Method naming (2026-06-11)

Family name: **ATLAS** = AND-gated Task-circuit Loss-Aligned Saliency.
- **ATLAS-T** = transcoder feature basis (internal: T-DSO v2 `mult`,
  engines `tdsoV2mult_*`).
- **ATLAS-N** = native MLP-neuron basis, dictionary-free (internal:
  `mult_free`).
- Circuit-only arms (`align` / `align_free`) stay descriptive, no branding.
- Internal pipeline names (tdsoV2mult, mult_free, ...) unchanged in code/logs;
  the mapping above is the paper-facing vocabulary.

### ⚠️ MMLU_humanities truncation bug (2026-06-11) — circuit arms contaminated

**Symptom.** Dict-free humanities masks (ALL bit-widths/seeds) kept **100%** of
weights instead of 0.35% → near-FP16 engines (mult_free = align_free = 65.96%,
FP16 = 66.0). Align loss was exactly 0 on every batch.

**Root cause.** Our extractors tokenized with right-truncation at
`max_len=2048`. Humanities 5-shot prompts run ~13k chars and the corruption
edits the LAST question (first clean/corrupt divergence at 9.5k–14k chars ≈
2.4k–3.5k tokens). After truncation clean == corrupted →
`relu(a_clean − a_corr) = 0` → all-zero saliency → global threshold kept
everything. Other tasks diverge at ~2k chars — inside the window — and are
unaffected (all masks exactly 0.35%).

**Why TaCQ doesn't hit this.** TaCQ has no input-corruption pathway at all —
its "contrastive" is weight-space (`grad × (W − W_quant) × W`) and its
dataloaders use `truncation=False` (drops whole samples, never cuts one). The
clean/corrupt input contrast is OUR addition; the truncation choice was ours.

**Transcoder arms also contaminated (silently).** Same tokenization; their
align loss ≠ 0 only because transcoder decoders have a bias (target degenerates
to a constant). Humanities transcoder masks were budget-correct but the circuit
term was noise — likely why firm mult hum (47.5) trailed TaCQ (51.5).

**Valid:** all GSM8k / MMLU_STEM / MMLU_social_sciences / Spider results; the
dumb arms (weight/magnitude/random) and TaCQ firm humanities (TACQ repo path,
untruncated). **Invalid:** every circuit-arm MMLU_humanities number (ablation
ce/align/mult/align_free/mult_free, firm tdsoV2mult, dict-free).

**Fix (applied, commit pending).** `tokenize_side`: `truncation=False` +
loud error if batch exceeds `--max-len` (now 4096 in runners; TaCQ
convention — calibration sees exactly the eval prompts). New
`assert_pairs_differ` guard in both extractors. `apply_mask_budget` now raises
if kept > 2× target (degenerate-tie guard).

**Reruns:** humfix_A = 2519273 (abl w2 s0 heavy arms + firm w2 s0-2, hum only);
humfix_B = 2519274 (gated: abl w3 s0 heavy, dict-free w2/w3 s1-2, firm w3
s0-2).

---

## Dict-free multi-seed harvest (2026-06-11 afternoon, jobs 2517952–58)

ATLAS-N (mult_free) / align_free, ablation protocol, humanities EXCLUDED
(truncation bug, reruns in flight). Acc %:

**2-bit** (s0 / s1 / s2):
- GSM8k    align_free 22.9 / 15.8 / 14.6 ; mult_free 26.5 / 17.2 / (gap job 2519430 — MTB token race killed GPTQ; mask intact)
- STEM     align_free 42.6 / 39.3 / 39.2 ; mult_free 44.2 / 40.8 / 40.8
- soc_sci  align_free 52.6 / 50.7 / 54.2 ; mult_free 55.9 / 54.1 / 50.8

**3-bit** (s0 / s1 / s2):
- GSM8k    align_free 59.7 / 57.7 / 57.9 ; mult_free 57.9 / 59.6 / 56.3
- STEM     align_free 53.8 / 54.4 / 53.3 ; mult_free 54.1 / 53.0 / 53.7
- soc_sci  align_free 71.7 / 72.0 / 72.8 ; mult_free 72.3 / 73.0 / 72.4

**Firm tdsoV2mult 3-bit seed 2 (2518045):** GSM8k 57.6, STEM 54.0,
soc_sci 73.2 (hum 61.1 INVALID → humfix_B). Table downstream3 now 3 seeds.

⚠️ **Heads-up for paper claims:** dict-free 2-bit seed variance is large
(GSM8k mult_free 26.5 vs 17.2; STEM s0 44.2 vs s1/s2 40.8). The s0-only
"ATLAS-N surpasses all transcoder arms on STEM" claim needs multi-seed
qualification once the grid completes. 3-bit convergence story is clean.

**humfix_A health check:** new no-truncation path verified — ce hum mask
exactly 0.3500%, transcoder align loss now varies per batch (1e5-range, was
constant 3.523e5), dict-free align nonzero (was 0.0).

---

## Transcoder-variant coverage audit (2026-06-11)

Decision: the paper KEEPS the transcoder version (T-DSO v2 mult) and tells the
discovery story: T-DSO wins → ablation dissects it → dict-free distills it.
Intro/abstract/results restructured accordingly (Step 1/2/3 narrative).

**Complete ✓**
- Spider 2b: mult 4 seeds (29.4±2.7), TaCQ 4 seeds (26.5±2.0), v1 align s0.
- TaCQ firm 2b + 3b: all 4 tasks × seeds 0–2 (genuine 2-bit).
- mult firm 3b: seeds 0–1 × 4 tasks.
- Ablation 2b s0 ce/align/mult: GSM8k + MMLU_STEM.

**In flight (jobs)**
- 2517684 `tdso2b_fix` — rebuild firm mult 2b (GSM8k s0–2, MMLU s0–1) at
  genuine `--wbits 2`. First numbers: GSM8k s0 raw 29.8%, MMLU_hum s0 47.5%.
- 2517959 `tdso_w2s2` — mult 2b seed-2 MMLU gap.
- 2518045 `tdso_w3s2` — mult 3b seed-2, all 4 tasks (fills Table downstream3
  to 3 seeds).
- 2518046 `abl_w2_humsoc` — ablation 2b s0 arms {ce,align,mult,weight,
  magnitude,random} on MMLU_humanities + social_sciences (completes the
  8-arm × 4-task 2-bit grid together with dfree jobs).
- 2518047 `abl_w3_s0` — ablation 3b s0, same 6 arms × all 4 tasks (direct
  controlled evidence for the "low-bit phenomenon" claim).
- 2517952–2517958 `dfree_*` — dict-free arms, 2b+3b, seeds 0–2, 4 tasks.

**Still missing after these land**
- Spider 3b multi-seed (s0 only: TaCQ 64.8 / v1 65.0 / mult 66.2).
- align (v1) multi-seed beyond Spider — secondary, ablation covers s0.

---

## TODO / pending firm numbers

- MMLU_STEM 2b: rerun arms {random, weight, magnitude, ce, align, mult}.
- 3-bit saliency-source ablation (mirror of the 2-bit table).
- Downstream firm seeds 1,2 (GSM8k, MMLU) + fixed-mask bitwidth ablation.
- Spider multi-seed for TaCQ + v2 mult (and π_nom 3b).
- Second model replication (Llama-3.2-1B).
- Paired-bootstrap significance on the headline conjunction gaps.
