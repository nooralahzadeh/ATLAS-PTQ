#!/usr/bin/env python3
"""Collect Spider exec accuracy results and compare to paper targets."""
import glob
import json
import os
import re

RESULTS_ROOT = "/home/ubuntu/tacq_data/results/Spider"
SUMMARY_PATH = "/home/ubuntu/tacq_data/results/REPLICATION_SUMMARY.json"

PAPER_TARGETS = {
    "unquantized": 67.6,
    "tacq_2bit_spider_conditioned": 21.92,
    "tacq_3bit_spider_conditioned": 58.32,
}


def parse_exec_accuracy(log_path):
    with open(log_path) as f:
        text = f.read()
    # test-suite-sql-eval prints: execution ... 0.000 ... (last column is 'all')
    for line in text.splitlines():
        if line.strip().startswith("execution"):
            parts = line.split()
            if parts:
                try:
                    return float(parts[-1]) * 100
                except ValueError:
                    pass
    return None


def main():
    results = {}
    for log in glob.glob(f"{RESULTS_ROOT}/**/eval_exec.log", recursive=True):
        run_dir = os.path.dirname(log)
        name = os.path.basename(run_dir)
        acc = parse_exec_accuracy(log)
        if acc is not None:
            results[name] = {"exec_accuracy_percent": acc, "log": log}

    unquantized_log = f"{RESULTS_ROOT}/unquantized_Meta-Llama-3-8B-Instruct/eval_exec.log"
    if os.path.exists(unquantized_log):
        acc = parse_exec_accuracy(unquantized_log)
        if acc is not None:
            results["unquantized"] = {
                "exec_accuracy_percent": acc,
                "paper_target_percent": PAPER_TARGETS["unquantized"],
                "delta_vs_paper": acc - PAPER_TARGETS["unquantized"],
            }

    for key, pattern in [
        ("tacq_2bit", r"\+2bit\+.*quantized_model$"),
        ("tacq_3bit", r"\+3bit\+.*quantized_model$"),
    ]:
        for name, data in list(results.items()):
            if name == "unquantized":
                continue
            if re.search(pattern, name):
                target_key = f"{key}_spider_conditioned".replace("tacq_", "tacq_")
                paper_key = f"tacq_{'2bit' if '2bit' in key else '3bit'}_spider_conditioned"
                data["paper_target_percent"] = PAPER_TARGETS[paper_key]
                data["delta_vs_paper"] = data["exec_accuracy_percent"] - PAPER_TARGETS[paper_key]

    summary = {}
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)
    summary["results"] = results
    summary["paper_targets_exec_accuracy"] = PAPER_TARGETS
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
