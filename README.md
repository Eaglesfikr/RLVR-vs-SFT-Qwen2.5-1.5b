# RLVR vs SFT: Empirical Analysis on Qwen 2.5 1.5B

This repository presents the results of training Qwen 2.5 1.5B Instruct with SFT and RLVR (GRPO) on the GSM8K dataset, analyzing performance across GSM8K and MATH benchmarks.

👉 **[View the full prompt and response from benchmark during this project on Hugging Face Spaces](https://huggingface.co/spaces/jayminban/RLVR-vs-SFT-Qwen2.5-1.5b)**

👉 **[View the top scoring model weights on Hugging Face Models](https://huggingface.co/jayminban/RLVR-vs-SFT-Qwen2.5-1.5b-checkpoints)**

### Experiments

1. **RLVR vs SFT on GSM8K train split** — standard comparison of both training methods on the training data
2. **Cheating analysis** — training directly on GSM8K test data to examine overfitting behavior differences between SFT and RLVR
3. **One-example RLVR** — inspired by [paper name](link), testing whether RLVR can improve performance from a single training example


### Why Qwen 2.5 1.5B?

Qwen 2.5 1.5B Instruct was chosen because it predates the introduction of RLVR by the DeepSeek team (DeepSeek-R1, January 2025). Released in September 2024 without any RLVR-aware training, it serves as a clean baseline for observing the effects of RLVR.

### Benchmark

All checkpoints were evaluated using my custom benchmark harness, **lm-eval-ledger**, with 0-shot prompting. The model generates chain-of-thought responses and the final answer is extracted from the first `\boxed{}` format and compared against ground truth.

0-shot was chosen because it produced the highest scores for the instruct model compared to few-shot settings. This was cross-validated against [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness), where 0-shot with my prompt template consistently outperformed all few-shot configurations evaluated with lm-eval-harness.

#### GSM8K Accuracy Across Evaluation Configurations (Qwen2.5-1.5B-Instruct)

| n-shot | lm-eval-harness | lm-eval-ledger (Chat Template) | lm-eval-ledger (No Chat Template, Qwen-Math2.5 few-shot) | lm-eval-ledger (No Chat Template, GSM8K train few-shot) |
|--------|-----------------|-------------------------------|-----------------------------------------------------|-------------------------------------------------------|
| 0      | 45.49           | **69.75**                     | 65.28                                               | 64.90                                                 |
| 1      | 52.24           | 64.29                         | 53.75                                               | 60.20                                                 |
| 2      | 55.34           | 61.94                         | 69.37                                               | 64.22                                                 |
| 3      | 55.34           | 63.31                         | 68.54                                               | 68.16                                                 |
| 4      | 56.63           | 63.99                         | 68.16                                               | 66.72                                                 |
| 5      | 56.48           | 61.18                         | 67.85                                               | 67.02                                                 |
| 6      | 57.24           | 63.00                         | 70.28                                               | 67.10                                                 |
| 7      | 56.25           | 56.63                         | **70.96**                                           | 65.13                                                 |
| 8      | 55.12               | 62.32                            | 69.14                                               | 67.70                                                 |

> **Reference:** The [Qwen2.5 technical report](https://arxiv.org/abs/2409.12122) reports **73.2%** on GSM8K for Qwen2.5-1.5B-Instruct (4-shot), though the exact evaluation setup (prompt template, answer extraction method) is not fully specified.

Every prompt, model response, and extracted answer from all benchmarks in this project are logged to a SQLite database and can be explored on [Hugging Face Spaces](link).

**lm-eval-ledger** with full SQLite logging will be released as a standalone project soon!


## Setup

All training was done using [verl](link). The training environment was built with [verlai/verl:vllm012.latest](https://hub.docker.com/layers/verlai/verl/vllm012.latest/images/sha256-9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d) Dockerfile.

### RLVR (GRPO)

Loss consists of three terms:

$$\mathcal{L}_{\text{GRPO}} = \mathcal{L}_{\text{policy}} + \lambda_1 \mathcal{L}_{\text{KL}} + \lambda_2 \mathcal{L}_{\text{entropy}}$$

- $\lambda_1$ (low variance KL): 0.001
- $\lambda_2$ (entropy coefficient): 0.001
- Learning rate: 5e-7
- Weight decay: 0.01, gradient clipping: 1.0

Entropy loss was added to encourage exploration, taken from [one example paper](link). 12 rollouts per prompt were generated with binary correctness reward only — assigns 1 if the extracted answer matches ground truth, 0 otherwise. No format reward.

### SFT

Standard cross-entropy loss on GSM8K Socratic chain-of-thought responses.

- Learning rate: 5e-7, cosine schedule with 5% warmup
- Weight decay: 0.01, gradient clipping: 1.0

All reward functions, training scripts, and training data are available in this repository.


### Compute Resources
Compute used for all training runs and benchmarks presented in this project.
#### Training

| Experiment | GPUs | Duration | Steps | Epochs |
|---|---|---|---|---|
| GRPO GSM8K Train | 6× RTX 4090 | 32h 12m | 4043 | 13 |
| GRPO GSM8K Test | 8× RTX 3090 | 20h 09m | 1620 | 30 |
| GRPO GSM8K 1-Example | 8× RTX 3090 | 11h 16m | 1000 | — |
| GRPO DRS 1-Example | 8× RTX 3090 | 12h 43m | 1000 | — |
| SFT GSM8K Train | 1× RTX 5090 | 2h 46m | 13076 | 7 |
| SFT GSM8K Test | 1× RTX 5090 | 1h 06m | 4935 | 15 |

#### Benchmarking (1× RTX 5090)

| Experiment | Checkpoints | Tasks | Duration |
|---|---|---|---|
| GRPO GSM8K Train | 82 | GSM8K + MATH | 4h 13m |
| GRPO GSM8K Test | 34 | GSM8K + MATH | 1h 45m |
| SFT GSM8K Test + Train | 231 | GSM8K + MATH | 9h 37m |
| GRPO 1-Example (both) | 41 | GSM8K + MATH | 2h 06m |
| **Total** | **388** | — | **17h 41m** |

## Combined Results

All GRPO and SFT experiments plotted together. GRPO consistently improves or maintains performance, while SFT degrades on both in-distribution and out-of-distribution benchmarks over time.

![All Benchmark Scores](plots/all-grpo-sft-benchmark-scores-overlay.png)


All evaluations are 0-shot unless otherwise noted. Best checkpoint is selected based on primary benchmark (GSM8K) performance.

| Method | Training Data | Steps | GSM8K (0-shot) | MATH (0-shot) |
|--------|--------------|-------|:--------------:|:-------------:|
| Qwen2.5-1.5B-Instruct (reproduced) | — | — | 69.7 | 49.2 |
| Qwen2.5-1.5B-Instruct (paper, 4-shot) | — | — | 73.2 | 55.2 |
| **GRPO**  | GSM8K train | 3,900 | **81.6** | 52.3 |
| **GRPO** | GSM8K test | 1,620 | 95.1* | 51.7 |
| **GRPO** | DSR 1 example | 1,000 | 74.2 | 49.4 |
| **GRPO**  | GSM8K 1 example | 1,000 | 74.2 | 49.4 |
| SFT | GSM8K train | 13,076 | 54.5 | 25.1 |
| SFT  | GSM8K test | 4,935 | 60.7* | 26.2 |

\* Trained on test data, not directly comparable.

## RLVR vs SFT on GSM8K Train Split

Both methods trained on the GSM8K training set. GRPO shows steady improvement without overfitting, while SFT drives the loss down but benchmark scores collapse — a textbook case of memorization without generalization.

**GRPO on GSM8K Train split:**

![GRPO On Train dataset](plots/grpo-gsm8k-train-reward-and-scores.png)

**SFT on GSM8K Train split:**

![SFT GSM8K Train Loss and Scores](plots/sft-gsm8k-train-loss-and-scores.png)

## Cheating Analysis — RLVR vs SFT on GSM8K Test Split
    
To test whether models can "cheat" by memorizing answers, I trained directly on the GSM8K test set. SFT overfits to the test answers — the loss drops but GSM8K accuracy still degrades after an initial spike, and MATH performance collapses. GRPO, by contrast, improves on GSM8K without harming MATH, suggesting it learns generalizable reasoning rather than memorizing answers.

**GRPO on GSM8K Test split:**

![GRPO GSM8K Test Reward and Scores](plots/grpo-train-vs-test-reward-and-scores.png)

**SFT on GSM8K Test split:**

![SFT GSM8K Test Loss and Scores](plots/sft-all-benchmark-scores-overlay.png)

## One-Example RLVR Analysis

Inspired by the [one-example paper](link), I tested whether GRPO can learn from a single training example. Two experiments were conducted: one using the π1 problem suggested by the paper, and one using a single GSM8K question.

**GRPO with single GSM8k and DSR π1 example:**

![GRPO Pi1 One Example Reward and Scores](plots/grpo-one-example-overlay-reward-and-scores.png)

## Training Dynamics

Entropy and KL divergence across all GRPO experiments. Entropy trends show how the model's output distribution evolves during training, while KL divergence tracks how far the policy drifts from the reference model.

![GRPO Entropy](plots/grpo-entropy-overlay.png)
![GRPO KL Loss](plots/grpo-kl-loss-overlay.png)


## 