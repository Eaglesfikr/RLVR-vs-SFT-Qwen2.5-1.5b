"""
Reward function for veRL GRPO training on GSM8K.

Uses binary reward based on answer correctness.
Extracts answers from \\boxed{answer} format and compares to gold.

veRL expects a compute_score function with signature:
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float
"""
from __future__ import annotations

import re

# ============================================================
# Answer Extraction
# ============================================================

_SOLUTION_CLIP_CHARS = 500  # Only check last N chars for efficiency
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

# Regex to extract from \boxed{answer} format
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{} format.

    Returns the FIRST match found, ignoring any later occurrences.
    """
    match = _BOXED_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_plain_number(text: str) -> str | None:
    """Extract last number from text (fallback for flexible mode)."""
    if len(text) > _SOLUTION_CLIP_CHARS:
        text = text[-_SOLUTION_CLIP_CHARS:]

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_number(text: str) -> str:
    """Extract first number from text, stripping commas."""
    match = _NUMBER_RE.search(text)
    if match:
        return match.group(0).replace(",", "")
    return text.strip()


def numeric_match(gold: str, pred: str) -> bool:
    """Match numbers, handling floats and integers."""
    try:
        g = float(gold.replace(",", ""))
        p = float(pred.replace(",", ""))
        if g == int(g) and p == int(p):
            return int(g) == int(p)
        return abs(g - p) < 1e-6
    except (ValueError, TypeError):
        return gold.strip() == pred.strip()


# ============================================================
# veRL Reward Function Interface
# ============================================================

def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: dict | None = None,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    **kwargs,
) -> float:
    """Compute reward for GSM8K completion.

    This is the veRL-compatible reward function interface.

    Args:
        data_source: Dataset name (ignored, for veRL compatibility)
        solution_str: Model's generated response
        ground_truth: Gold answer (just the number)
        extra_info: Extra info dict (ignored, for veRL compatibility)
        method: "strict" requires boxed format, "flexible" finds any number
        format_score: Score for wrong answer but correct format (default 0.0)
        score: Score for correct answer (default 1.0)
        **kwargs: Additional args ignored

    Returns:
        Reward score (0.0, format_score, or score)
    """
    # Extract answer from \boxed{} format
    extracted = extract_boxed(solution_str)

    if extracted is None and method == "flexible":
        # Last resort: find any number in response
        extracted = extract_plain_number(solution_str)

    # Determine reward and status
    if extracted is None:
        reward = 0.0
        status = "no_format"
    else:
        pred_num = extract_number(extracted)
        if numeric_match(ground_truth, pred_num):
            reward = score
            status = "correct"
        else:
            reward = format_score
            status = "wrong"

    return reward
