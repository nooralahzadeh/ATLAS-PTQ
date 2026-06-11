#!/usr/bin/env python3
"""Paired bootstrap significance for ATLAS head-to-heads.

Compares two engines on the SAME evaluation examples and reports the mean
accuracy gap with a bootstrap CI and two-sided p-value for H0: gap == 0.

Per-example sources
-------------------
GSM8k : results.txt -- one True/False per line, deterministic order.
MMLU  : 0_evaluation_results.json -- per-subject accuracy; cluster-bootstrap
        over subjects (coarser; flagged in output).

Usage
-----
  python paired_bootstrap.py gsm8k --a <engineA> --b <engineB>
  python paired_bootstrap.py mmlu --a <engineA> --b <engineB> --task MMLU_STEM
  python paired_bootstrap.py file --a-file a.txt --b-file b.txt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/capstor/scratch/cscs/fnoorala/ATLAS-PTQ")


def _read_bool_lines(path: Path) -> np.ndarray:
    vals = []
    for ln in path.read_text().splitlines():
        s = ln.strip().lower()
        if s in ("true", "1"):
            vals.append(1)
        elif s in ("false", "0"):
            vals.append(0)
        elif s:
            raise ValueError(f"{path}: unparseable line {s!r}")
    return np.array(vals, dtype=np.float64)


def _gsm8k_vec(engine: str, results_root: Path) -> np.ndarray:
    p = results_root / "GSM8k" / engine / "results.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return _read_bool_lines(p)


def _mmlu_subject_acc(engine, task, results_root):
    p = results_root / task / f"results_{engine}" / "0_evaluation_results.json"
    if not p.exists():
        raise FileNotFoundError(p)
    d = json.load(open(p))
    return {k.replace(" accuracy", ""): v for k, v in d.items()
            if k.endswith(" accuracy") and k != "Average accuracy"}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int, rng: np.random.Generator):
    assert a.shape == b.shape, f"length mismatch: {a.shape} vs {b.shape}"
    n = a.shape[0]
    diff = a - b
    obs = float(diff.mean())
    idx = rng.integers(0, n, size=(iters, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    p_val = float(np.mean(np.sign(boot) != np.sign(obs))) if obs != 0 else 1.0
    return {"obs_diff": obs, "ci_lo": lo, "ci_hi": hi, "p_value": p_val,
            "n": int(n), "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    sub.required = True

    g = sub.add_parser("gsm8k")
    g.add_argument("--a", required=True, help="engine A name")
    g.add_argument("--b", required=True, help="engine B name")
    g.add_argument("--results-root", default=str(ROOT / "tacq_data" / "results"))
    g.add_argument("--iters", type=int, default=10000)
    g.add_argument("--seed", type=int, default=42)

    m = sub.add_parser("mmlu")
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    m.add_argument("--task", required=True, help="e.g. MMLU_STEM")
    m.add_argument("--results-root", default=str(ROOT / "tacq_data" / "results"))
    m.add_argument("--iters", type=int, default=10000)
    m.add_argument("--seed", type=int, default=42)

    f = sub.add_parser("file")
    f.add_argument("--a-file", required=True)
    f.add_argument("--b-file", required=True)
    f.add_argument("--iters", type=int, default=10000)
    f.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rr = Path(getattr(args, "results_root", ""))

    if args.mode == "gsm8k":
        a_vec = _gsm8k_vec(args.a, rr)
        b_vec = _gsm8k_vec(args.b, rr)
        unit = "example"
    elif args.mode == "mmlu":
        a_sub = _mmlu_subject_acc(args.a, args.task, rr)
        b_sub = _mmlu_subject_acc(args.b, args.task, rr)
        shared = sorted(set(a_sub) & set(b_sub))
        if not shared:
            print("ERROR: no shared subjects", file=sys.stderr); sys.exit(1)
        a_vec = np.array([a_sub[s] for s in shared])
        b_vec = np.array([b_sub[s] for s in shared])
        unit = "subject (cluster)"
    elif args.mode == "file":
        a_vec = _read_bool_lines(Path(args.a_file))
        b_vec = _read_bool_lines(Path(args.b_file))
        unit = "example"
    else:
        ap.print_help(); sys.exit(1)

    r = paired_bootstrap(a_vec, b_vec, args.iters, rng)
    print(f"A mean: {r['mean_a']:.4f}  B mean: {r['mean_b']:.4f}")
    print(f"Diff (A-B): {r['obs_diff']:+.4f}  95% CI: [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    print(f"p-value: {r['p_value']:.4f}  (n={r['n']} {unit}s, {args.iters} iterations)")
    if args.mode == "mmlu":
        print("NOTE: cluster-bootstrap over subjects; add per-example dumps for tighter test.")


if __name__ == "__main__":
    main()
