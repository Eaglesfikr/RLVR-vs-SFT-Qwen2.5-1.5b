#!/usr/bin/env python3
"""
Prepare GSM8K data for veRL GRPO training.

Converts data to veRL's expected parquet format with:
- prompt: The formatted prompt with few-shot examples
- ground_truth: The gold answer (for reward computation)

Usage:
    python prepare_data.py
    python prepare_data.py --fewshot_k 0  # zero-shot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_fewshot_block(fewshot_path: str, k: int) -> str:
    """Build few-shot prompt block from examples file.

    Args:
        fewshot_path: Path to JSONL file with few-shot examples
        k: Number of few-shot examples to include

    Returns:
        Formatted few-shot block string
    """
    if k <= 0 or not fewshot_path:
        return ""

    if not Path(fewshot_path).exists():
        print(f"[WARN] Few-shot path not found: {fewshot_path}")
        return ""

    examples = load_jsonl(fewshot_path)[:k]
    blocks = []

    # Add format instruction at the start
    format_instruction = r"Output format: end your response with \boxed{<answer>} where <answer> is the final answer."
    blocks.append(format_instruction)

    for ex in examples:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()

        if not question or not answer:
            continue

        # GSM8K format: answer contains "solution text\n#### number"
        if "####" in answer:
            parts = answer.split("####")
            solution_part = parts[0].strip()
            final_answer = parts[-1].strip()
            blocks.append(f"Question: {question}\nAnswer:\n{solution_part}\n\n\\boxed{{{final_answer}}}")
        else:
            # Plain answer without solution
            blocks.append(f"Question: {question}\nAnswer: \\boxed{{{answer}}}")

    return "\n\n".join(blocks)


def prepare_gsm8k_data(
    data_path: str,
    gold_path: str,
    output_dir: str,
    fewshot_path: str = "",
    fewshot_k: int = 0,
) -> None:
    """Prepare GSM8K data for veRL.

    Creates train.parquet with columns:
    - prompt: Formatted prompt with few-shot examples
    - ground_truth: Gold answer for reward computation

    Args:
        data_path: Path to JSONL file with questions
        gold_path: Path to JSON file with gold answers
        output_dir: Directory to save parquet files
        fewshot_path: Path to JSONL file with few-shot examples
        fewshot_k: Number of few-shot examples (0 = zero-shot)
    """
    # Load gold answers from JSON
    with open(gold_path) as f:
        gold_answers = json.load(f)  # {"0": "72", "1": "10", ...}

    # Load questions from JSONL
    questions = load_jsonl(data_path)

    # Build few-shot block
    fewshot_block = ""
    if fewshot_k > 0 and fewshot_path:
        print(f"[INFO] Building {fewshot_k}-shot prompt from {fewshot_path}")
        fewshot_block = build_fewshot_block(fewshot_path, fewshot_k)
        if fewshot_block:
            print(f"[INFO] Few-shot block built ({len(fewshot_block)} chars)")
        else:
            print(f"[WARN] Failed to build few-shot block, using zero-shot")

    # Prepare records
    records = []
    for idx, item in enumerate(questions):
        question = item["question"].strip()
        gold = gold_answers.get(str(idx), "")

        # Build prompt
        if fewshot_block:
            prompt = f"{fewshot_block}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = (
                r"Output format: end your response with \boxed{<answer>} "
                "where <answer> is the final numerical answer.\n\n"
                f"Question: {question}\nAnswer:"
            )

        # veRL expects prompt as list of message dicts for chat template
        # Format: [{"role": "user", "content": "..."}]
        prompt_messages = [{"role": "user", "content": prompt}]

        records.append({
            "prompt": prompt_messages,
            # veRL expects reward_model dict containing ground_truth
            "reward_model": {"ground_truth": gold},
            "data_source": "gsm8k",  # Required by veRL's naive reward manager
        })

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df = pd.DataFrame(records)
    train_path = output_path / "train.parquet"
    df.to_parquet(train_path, index=False)

    print(f"[INFO] Saved {len(records)} examples to {train_path}")

    # Print example
    print("\n" + "=" * 60)
    print("EXAMPLE PROMPT (message format):")
    print("=" * 60)
    example_content = records[0]["prompt"][0]["content"]
    print(f"Role: {records[0]['prompt'][0]['role']}")
    print(f"Content (first 500 chars):\n{example_content[:500]}")
    print("...")
    print(f"\nGold answer: {records[0]['reward_model']['ground_truth']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K data for veRL")

    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/data8tb/Documents/project/my_bench_harness/data/gsm8k/socratic/train.jsonl",
        help="Path to JSONL file with questions",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        default="/mnt/data8tb/Documents/project/my_bench_harness/data/gsm8k/socratic/train_gold.json",
        help="Path to JSON file with gold answers",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data8tb/Documents/project/rlvr_winter/verl-my-rlvr/data",
        help="Directory to save parquet files",
    )
    parser.add_argument(
        "--fewshot_path",
        type=str,
        default="/mnt/data8tb/Documents/project/my_bench_harness/data/gsm8k/main/qwen-paper-few-shot.jsonl",
        help="Path to JSONL file with few-shot examples",
    )
    parser.add_argument(
        "--fewshot_k",
        type=int,
        default=0,
        help="Number of few-shot examples (0 = zero-shot, recommended for RLVR)",
    )

    args = parser.parse_args()

    prepare_gsm8k_data(
        data_path=args.data_path,
        gold_path=args.gold_path,
        output_dir=args.output_dir,
        fewshot_path=args.fewshot_path,
        fewshot_k=args.fewshot_k,
    )


if __name__ == "__main__":
    main()
