#!/usr/bin/env python3
"""
Create a single-example RLVR parquet for debugging/testing.
Picks a hard 9-step GSM8K problem (index 1708).

Usage:
    python prepare_single_rlvr.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SRC = "/mnt/data8tb/Documents/project/my_bench_harness/data/gsm8k/socratic/train.jsonl"
DST = "/mnt/data8tb/Documents/project/rlvr_winter/verl-my-rlvr/data/single_rlvr.parquet"
EXAMPLE_IDX = 1708


def main():
    with open(SRC) as f:
        rows = [json.loads(line) for line in f]

    row = rows[EXAMPLE_IDX]
    question = row["question"]
    answer_num = row["answer"].split("####")[-1].strip()

    # Same prompt format as existing GRPO data: message dict with boxed format instruction
    prompt = [
        {
            "role": "user",
            "content": (
                "Output format: end your response with "
                "\\boxed{<answer>} where <answer> is the final answer.\n\n"
                f"Question: {question}\n"
                "Answer:"
            ),
        }
    ]

    record = {
        "prompt": [prompt],
        "reward_model": [{"ground_truth": answer_num}],
        "data_source": ["gsm8k"],
    }

    df = pd.DataFrame(record)
    Path(DST).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)

    print(f"Saved single example to {DST}")
    print(f"  Question: {question[:80]}...")
    print(f"  Gold answer: {answer_num}")


if __name__ == "__main__":
    main()
