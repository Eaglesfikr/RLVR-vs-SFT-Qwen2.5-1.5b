#!/usr/bin/env python3
"""
Evaluate SFT/GRPO model on GSM8K test set.
Loads model, generates CoT responses, extracts \\boxed{} answers, and scores.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use the same reward function from GRPO training
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_fn import compute_score


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model-path", type=str, required=True,
                        help="HF model path or local checkpoint")
    parser.add_argument("--data-path", type=str,
                        default="data/gsm8k_test.parquet",
                        help="Path to GSM8K test parquet")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Generation batch size (1 for now)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    # Resolve paths
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = str(script_dir / model_path)

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = str(script_dir / data_path)

    print(f"Loading model from: {model_path}")
    print(f"Loading data from: {data_path}")

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {device}, Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

    # Load data
    df = pd.read_parquet(data_path)
    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"Evaluating on {len(df)} samples")

    results = []
    correct = 0
    total = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        prompt_data = row.get("prompt", "")
        reward_data = row.get("reward_model", {})
        data_source = row.get("data_source", "gsm8k")

        # Handle prompt: could be list of messages or plain string
        if isinstance(prompt_data, list):
            # Chat format: extract user message content
            user_msg = None
            for msg in prompt_data:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if user_msg:
                prompt = user_msg
            else:
                prompt = str(prompt_data)
        else:
            prompt = str(prompt_data)

        # Handle reward_model: get ground truth
        if isinstance(reward_data, dict):
            ground_truth = str(reward_data.get("ground_truth", ""))
        else:
            ground_truth = str(reward_data)

        if not prompt:
            continue

        # Generate response
        try:
            response = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"Error at sample {idx}: {e}")
            response = ""

        # Score
        score = compute_score(
            data_source="gsm8k",
            solution_str=response,
            ground_truth=str(ground_truth),
        )

        is_correct = score >= 0.5
        correct += 1 if is_correct else 0
        total += 1

        results.append({
            "idx": idx,
            "prompt": prompt[:100],
            "response": response[:200],
            "ground_truth": str(ground_truth),
            "score": score,
            "correct": is_correct,
        })

        acc = correct / total * 100
        elapsed = time.time() - start_time
        print(f"[{total}/{len(df)}] acc: {acc:.1f}% | score: {score}"
              f" | gt: {ground_truth} | {elapsed:.0f}s")

    # Summary
    final_acc = correct / total * 100
    print(f"\n{'='*50}")
    print(f"Model: {model_path}")
    print(f"Samples: {total}")
    print(f"Accuracy: {final_acc:.2f}%")
    print(f"Time: {time.time() - start_time:.0f}s")

    # Save results
    out_dir = script_dir / "eval_results"
    out_dir.mkdir(exist_ok=True)
    model_name = Path(model_path).name if Path(model_path).exists() else model_path.replace("/", "_")
    out_file = out_dir / f"eval_{model_name}_gsm8k.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_path,
            "accuracy": final_acc,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()