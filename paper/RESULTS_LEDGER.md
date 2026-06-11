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

## TODO / pending firm numbers

- MMLU_STEM 2b: rerun arms {random, weight, magnitude, ce, align, mult}.
- 3-bit saliency-source ablation (mirror of the 2-bit table).
- Downstream firm seeds 1,2 (GSM8k, MMLU) + fixed-mask bitwidth ablation.
- Spider multi-seed for TaCQ + v2 mult (and π_nom 3b).
- Second model replication (Llama-3.2-1B).
- Paired-bootstrap significance on the headline conjunction gaps.
