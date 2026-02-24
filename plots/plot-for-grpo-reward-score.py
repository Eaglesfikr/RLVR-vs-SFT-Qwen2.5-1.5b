import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
import numpy as np
from datetime import datetime

# --- Global style ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": True,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
})

# --- Paths ---
REWARD_DRS_CSV = "plot-data/rewards-grpo-drs-one-example.csv"
REWARD_GSM8K_ONE_CSV = "plot-data/rewards-grpo-gsm8k-one-example.csv"
REWARD_GSM8K_TEST_CSV = "plot-data/rewards-grpo-gsm8k-test.csv"
REWARD_GSM8K_TRAIN_CSV = "plot-data/rewards-grpo-train.csv"
SQLITE_ONE_EXAMPLES = "both-one-examples-success.sqlite3"
SQLITE_GSM8K_TEST = "gsm8k-test-grpo-30ep-success.sqlite3"
SQLITE_GSM8K_TRAIN = "grpo-gsm8k-train.sqlite3"
SAVE_DIR = "plots"
TIMESTAMP = datetime.now().strftime("%H%M")

# --- Base model scores (Qwen2.5-1.5B-Instruct) ---
BASE_GSM8K = 0.6975
BASE_MATH = 0.4942

# --- Colors ---
C_REWARD = "#E67E22"
C_REWARD_RAW = "#F5CBA7"
C_GSM8K = "#D94A6B"
C_MATH = "#3AAA5B"

# --- Load reward CSVs ---
reward_drs_df = pd.read_csv(REWARD_DRS_CSV)
reward_drs_df.columns = ["Step", "reward", "reward_min", "reward_max"]

reward_gsm8k_one_df = pd.read_csv(REWARD_GSM8K_ONE_CSV)
reward_gsm8k_one_df.columns = ["Step", "reward", "reward_min", "reward_max"]

reward_gsm8k_test_df = pd.read_csv(REWARD_GSM8K_TEST_CSV)
reward_gsm8k_test_df.columns = ["Step", "reward", "reward_min", "reward_max"]

reward_gsm8k_train_df = pd.read_csv(REWARD_GSM8K_TRAIN_CSV)
reward_gsm8k_train_df.columns = ["Step", "reward", "reward_min", "reward_max"]

# --- Load benchmark scores from SQLite ---
conn1 = sqlite3.connect(SQLITE_ONE_EXAMPLES)
summaries_one = pd.read_sql_query("SELECT model_tag, task_name, accuracy FROM summaries", conn1)
conn1.close()

conn2 = sqlite3.connect(SQLITE_GSM8K_TEST)
summaries_gsm8k_test = pd.read_sql_query("SELECT model_tag, task_name, accuracy FROM summaries", conn2)
conn2.close()

conn3 = sqlite3.connect(SQLITE_GSM8K_TRAIN)
summaries_gsm8k_train = pd.read_sql_query("SELECT model_tag, task_name, accuracy FROM summaries", conn3)
conn3.close()


def extract_scores(summaries, prefix):
    """Extract step and accuracy for a given model prefix."""
    subset = summaries[summaries["model_tag"].str.startswith(prefix)].copy()
    subset["step"] = subset["model_tag"].apply(
        lambda x: int(re.search(r"global_step_(\d+)", x).group(1))
    )
    gsm8k = subset[subset["task_name"] == "gsm8k_main(0)"][["step", "accuracy"]].rename(
        columns={"accuracy": "gsm8k_acc"}
    )
    math = subset[subset["task_name"] == "hendrycks_math(0)"][["step", "accuracy"]].rename(
        columns={"accuracy": "math_acc"}
    )
    merged = gsm8k.merge(math, on="step").sort_values("step")
    return merged


scores_drs = extract_scores(summaries_one, "dsr-one-example-grpo")
scores_gsm8k_one = extract_scores(summaries_one, "grpo-one-example-gsm8k")
scores_gsm8k_test = extract_scores(summaries_gsm8k_test, "grpo-gsm8k-test-30ep-success")
scores_gsm8k_train = extract_scores(summaries_gsm8k_train, "grpo-gsm8k-train-13ep-success")


def smooth(values, weight=0.9):
    """Exponential moving average for smoothing."""
    smoothed = []
    last = values.iloc[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def make_plot(reward_df, scores_df, title, save_name, total_epochs=None,
              raw_alpha=0.35, acc_ylim=(15, 85), smooth_weight=0.9,
              raw_only=False):
    fig, ax1 = plt.subplots(figsize=(13, 5.5))

    max_step = reward_df["Step"].max()

    # --- Left axis: Reward ---
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Mean Reward", color=C_REWARD)
    if raw_only:
        ax1.plot(
            reward_df["Step"], reward_df["reward"],
            color=C_REWARD, alpha=0.85, linewidth=1.0, label="Reward",
        )
    else:
        ax1.plot(
            reward_df["Step"], reward_df["reward"],
            color=C_REWARD_RAW, alpha=raw_alpha, linewidth=0.4, rasterized=True,
        )
        ax1.plot(
            reward_df["Step"], smooth(reward_df["reward"], weight=smooth_weight),
            color=C_REWARD, linewidth=2, label="Reward (smoothed)",
        )
    ax1.tick_params(axis="y", labelcolor=C_REWARD)
    ax1.set_xlim(0, max_step)
    ax1.grid(axis="y", color=C_REWARD, alpha=0.10, linewidth=0.5)

    # --- Right axis: Benchmark scores ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)")

    # Prepend step 0 with base model scores so curves connect to baseline
    base_row = pd.DataFrame({"step": [0], "gsm8k_acc": [BASE_GSM8K], "math_acc": [BASE_MATH]})
    scores_with_base = pd.concat([base_row, scores_df], ignore_index=True)

    ax2.plot(
        scores_with_base["step"], scores_with_base["gsm8k_acc"] * 100,
        color=C_GSM8K, marker="o", markersize=2.5, linewidth=1.3,
        label="GSM8K",
    )
    ax2.plot(
        scores_with_base["step"], scores_with_base["math_acc"] * 100,
        color=C_MATH, marker="s", markersize=2.5, linewidth=1.3,
        label="MATH",
    )

    # --- Horizontal lines for base model ---
    ax2.axhline(
        y=BASE_GSM8K * 100, color=C_GSM8K, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base GSM8K ({BASE_GSM8K:.1%})",
    )
    ax2.axhline(
        y=BASE_MATH * 100, color=C_MATH, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base MATH ({BASE_MATH:.1%})",
    )

    ax2.set_ylim(acc_ylim)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax2.grid(axis="y", color="#888888", alpha=0.10, linewidth=0.5)

    # --- Combined legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="lower right", fontsize=9, framealpha=0.85,
        edgecolor="#cccccc", borderpad=0.8, handlelength=2,
    )
    legend.get_frame().set_linewidth(0.6)

    # --- Secondary x-axis for epochs (if provided) ---
    if total_epochs is not None:
        steps_per_epoch = max_step / total_epochs
        ax_epoch = ax1.secondary_xaxis("top")
        # Show every 5th epoch if > 15 epochs to avoid cramping
        step_size = 5 if total_epochs > 15 else 1
        epoch_nums = list(range(step_size, total_epochs + 1, step_size))
        epoch_ticks = [i * steps_per_epoch for i in epoch_nums]
        ax_epoch.set_xticks(epoch_ticks)
        ax_epoch.set_xticklabels([str(i) for i in epoch_nums], fontsize=8)
        ax_epoch.set_xlabel("Epoch", fontsize=11, labelpad=6)
        ax_epoch.tick_params(length=3, width=0.6)
        ax_epoch.spines["top"].set_visible(True)
        ax_epoch.spines["top"].set_linewidth(0.6)
        ax_epoch.spines["top"].set_color("#aaaaaa")

    title_pad = 30 if total_epochs is not None else 12
    ax1.set_title(title, fontsize=15, fontweight="bold", pad=title_pad)

    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


# --- Plot 1: GRPO π1 One Example ---
make_plot(
    reward_drs_df, scores_drs,
    "GRPO on \u03C01 One Example: Mean Reward & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-pi1-one-example-reward-and-scores.png",
    raw_only=True,
)

# --- Plot 2: GRPO GSM8K One Example ---
make_plot(
    reward_gsm8k_one_df, scores_gsm8k_one,
    "GRPO on GSM8K One Example: Mean Reward & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-gsm8k-one-example-reward-and-scores.png",
    raw_only=True,
)

# --- Plot 3: GRPO GSM8K Test (30 epochs) ---
make_plot(
    reward_gsm8k_test_df, scores_gsm8k_test,
    "GRPO on GSM8K Test Set: Mean Reward & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-gsm8k-test-reward-and-scores.png",
    total_epochs=30,
    acc_ylim=(0, 100),
    raw_alpha=0.55,
)


# --- Plot 4: GRPO GSM8K Train (13 epochs) ---
make_plot(
    reward_gsm8k_train_df, scores_gsm8k_train,
    "GRPO on GSM8K Train Set: Mean Reward & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-gsm8k-train-reward-and-scores.png",
    total_epochs=13,
    acc_ylim=(0, 100),
    raw_alpha=0.55,
)


# --- Plot 5: Overlay benchmark scores from all four GRPO experiments ---
def make_overlay_plot(scores_list, labels, title, save_name):
    """Plot benchmark scores from multiple experiments on a normalized x-axis."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Distinct colors per experiment
    exp_colors = ["#D94A6B", "#4A90D9", "#E67E22", "#8E44AD"]
    gsm8k_styles = [("o", "-"), ("D", "-"), ("^", "-"), ("P", "-")]
    math_styles = [("s", "--"), ("d", "--"), ("v", "--"), ("X", "--")]

    for i, (scores_df, label) in enumerate(zip(scores_list, labels)):
        # Prepend base model at x=0, append last point at x=1
        base_row = pd.DataFrame({"step": [0], "gsm8k_acc": [BASE_GSM8K], "math_acc": [BASE_MATH]})
        s = pd.concat([base_row, scores_df], ignore_index=True)
        max_step = s["step"].max()
        x_norm = s["step"] / max_step  # normalize to 0-1

        mk_g, ls_g = gsm8k_styles[i]
        mk_m, ls_m = math_styles[i]

        ax.plot(
            x_norm, s["gsm8k_acc"] * 100,
            color=exp_colors[i], marker=mk_g, markersize=3, linewidth=1.5,
            linestyle=ls_g, label=f"{label} — GSM8K",
        )
        ax.plot(
            x_norm, s["math_acc"] * 100,
            color=exp_colors[i], marker=mk_m, markersize=3, linewidth=1.5,
            linestyle=ls_m, alpha=0.7, label=f"{label} — MATH",
        )

    # Base model lines
    ax.axhline(
        y=BASE_GSM8K * 100, color=C_GSM8K, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base GSM8K ({BASE_GSM8K:.1%})",
    )
    ax.axhline(
        y=BASE_MATH * 100, color=C_MATH, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base MATH ({BASE_MATH:.1%})",
    )

    ax.set_xlabel("Training Progress (normalized)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.grid(axis="y", color="#888888", alpha=0.12, linewidth=0.5)
    ax.grid(axis="x", color="#888888", alpha=0.08, linewidth=0.5)

    legend = ax.legend(
        fontsize=9, framealpha=0.85, edgecolor="#cccccc",
        borderpad=0.8, handlelength=2, ncol=2,
        loc="lower left",
    )
    legend.get_frame().set_linewidth(0.6)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


make_overlay_plot(
    [scores_drs, scores_gsm8k_one, scores_gsm8k_test, scores_gsm8k_train],
    ["\u03C01 1-Example", "GSM8K 1-Example", "GSM8K Test", "GSM8K Train"],
    "GRPO Benchmark Scores Comparison Across Experiments",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-all-benchmark-scores-overlay.png",
)


# --- Plot 5: Combined GRPO + SFT benchmark scores ---
SQLITE_SFT = "sft-gsm8k-test-train-success.sqlite3"
conn_sft = sqlite3.connect(SQLITE_SFT)
summaries_sft = pd.read_sql_query("SELECT model_tag, task_name, accuracy FROM summaries", conn_sft)
conn_sft.close()

scores_sft_test = extract_scores(summaries_sft, "sft-gsm8k-test-docker")
scores_sft_train = extract_scores(summaries_sft, "sft-gsm8k-train-docker")

all_scores = [scores_drs, scores_gsm8k_one, scores_gsm8k_test, scores_gsm8k_train, scores_sft_test, scores_sft_train]
all_labels = [
    "GRPO \u03C01 1-Example", "GRPO GSM8K 1-Example", "GRPO GSM8K Test", "GRPO GSM8K Train",
    "SFT GSM8K Test", "SFT GSM8K Train",
]


def make_combined_overlay(scores_list, labels, title, save_name):
    """Plot benchmark scores from all GRPO + SFT experiments on normalized x-axis."""
    fig, ax = plt.subplots(figsize=(14, 6))

    exp_colors = ["#D94A6B", "#4A90D9", "#E67E22", "#8E44AD", "#2ECC71", "#1ABC9C"]
    # Markers only for GRPO (first 4), none for SFT (last 2)
    gsm8k_markers = ["o", "D", "^", "P", None, None]
    math_markers = ["s", "d", "v", "X", None, None]

    for i, (scores_df, label) in enumerate(zip(scores_list, labels)):
        base_row = pd.DataFrame({"step": [0], "gsm8k_acc": [BASE_GSM8K], "math_acc": [BASE_MATH]})
        s = pd.concat([base_row, scores_df], ignore_index=True)
        max_step = s["step"].max()
        x_norm = s["step"] / max_step

        mk_g = gsm8k_markers[i]
        mk_m = math_markers[i]
        mk_kwargs_g = dict(marker=mk_g, markersize=3) if mk_g else {}
        mk_kwargs_m = dict(marker=mk_m, markersize=3) if mk_m else {}

        ax.plot(
            x_norm, s["gsm8k_acc"] * 100,
            color=exp_colors[i], linewidth=1.5,
            linestyle="-", label=f"{label} — GSM8K", **mk_kwargs_g,
        )
        ax.plot(
            x_norm, s["math_acc"] * 100,
            color=exp_colors[i], linewidth=1.5,
            linestyle="--", alpha=0.7, label=f"{label} — MATH", **mk_kwargs_m,
        )

    # Base model lines
    ax.axhline(
        y=BASE_GSM8K * 100, color=C_GSM8K, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base GSM8K ({BASE_GSM8K:.1%})",
    )
    ax.axhline(
        y=BASE_MATH * 100, color=C_MATH, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base MATH ({BASE_MATH:.1%})",
    )

    ax.set_xlabel("Training Progress (normalized)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.grid(axis="y", color="#888888", alpha=0.12, linewidth=0.5)
    ax.grid(axis="x", color="#888888", alpha=0.08, linewidth=0.5)

    legend = ax.legend(
        fontsize=8, framealpha=0.85, edgecolor="#cccccc",
        borderpad=0.8, handlelength=2, ncol=2,
        loc="lower left",
    )
    legend.get_frame().set_linewidth(0.6)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


make_combined_overlay(
    all_scores, all_labels,
    "All Benchmark Scores: GRPO & SFT Experiments",
    f"{SAVE_DIR}/{TIMESTAMP}-all-grpo-sft-benchmark-scores-overlay.png",
)


# --- Plot 7: GRPO Train vs Test — Reward & Benchmark Scores overlay ---
def make_train_test_overlay(reward_test, reward_train, scores_test_df, scores_train_df,
                            title, save_name):
    """Overlay GRPO train and test: reward on left y-axis, benchmark scores on right."""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    C_TEST = "#E67E22"
    C_TRAIN = "#8E44AD"

    # Normalize steps to 0-1 for both
    max_step_test = reward_test["Step"].max()
    max_step_train = reward_train["Step"].max()

    x_test = reward_test["Step"] / max_step_test
    x_train = reward_train["Step"] / max_step_train

    # --- Left axis: Reward (smoothed) ---
    ax1.set_xlabel("Training Progress (normalized)")
    ax1.set_ylabel("Mean Reward")
    ax1.plot(
        x_test, smooth(reward_test["reward"]),
        color=C_TEST, linewidth=1.8, alpha=0.35, label="Test — Reward",
    )
    ax1.plot(
        x_train, smooth(reward_train["reward"]),
        color=C_TRAIN, linewidth=1.8, alpha=0.35, label="Train — Reward",
    )
    ax1.set_xlim(0, 1)
    ax1.grid(axis="y", color="#888888", alpha=0.10, linewidth=0.5)

    # --- Right axis: Benchmark scores ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)")

    base_row = pd.DataFrame({"step": [0], "gsm8k_acc": [BASE_GSM8K], "math_acc": [BASE_MATH]})

    s_test = pd.concat([base_row, scores_test_df], ignore_index=True)
    s_test_norm = s_test["step"] / s_test["step"].max()

    s_train = pd.concat([base_row.copy(), scores_train_df], ignore_index=True)
    s_train_norm = s_train["step"] / s_train["step"].max()

    ax2.plot(s_test_norm, s_test["gsm8k_acc"] * 100,
             color=C_TEST, linestyle="--", linewidth=1.3, marker="o", markersize=2.5,
             label="Test — GSM8K")
    ax2.plot(s_test_norm, s_test["math_acc"] * 100,
             color=C_TEST, linestyle=":", linewidth=1.3, marker="s", markersize=2.5,
             alpha=0.7, label="Test — MATH")

    ax2.plot(s_train_norm, s_train["gsm8k_acc"] * 100,
             color=C_TRAIN, linestyle="--", linewidth=1.3, marker="D", markersize=2.5,
             label="Train — GSM8K")
    ax2.plot(s_train_norm, s_train["math_acc"] * 100,
             color=C_TRAIN, linestyle=":", linewidth=1.3, marker="d", markersize=2.5,
             alpha=0.7, label="Train — MATH")

    # Base model lines
    ax2.axhline(y=BASE_GSM8K * 100, color=C_GSM8K, linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Base GSM8K ({BASE_GSM8K:.1%})")
    ax2.axhline(y=BASE_MATH * 100, color=C_MATH, linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Base MATH ({BASE_MATH:.1%})")

    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=8, framealpha=0.85, edgecolor="#cccccc",
        borderpad=0.8, handlelength=2, ncol=2, loc="lower left",
    )
    legend.get_frame().set_linewidth(0.6)

    ax1.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


make_train_test_overlay(
    reward_gsm8k_test_df, reward_gsm8k_train_df,
    scores_gsm8k_test, scores_gsm8k_train,
    "GRPO Train vs Test: Reward & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-train-vs-test-reward-and-scores.png",
)


# --- Plot 8: GRPO One-Example overlay — π1 vs GSM8K ---
def make_one_example_overlay(reward_a, reward_b, scores_a, scores_b,
                             label_a, label_b, title, save_name):
    """Overlay two one-example GRPO experiments: reward + benchmark scores."""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    C_A = "#D94A6B"   # π1
    C_B = "#4A90D9"   # GSM8K one-example

    max_step_a = reward_a["Step"].max()
    max_step_b = reward_b["Step"].max()
    x_a = reward_a["Step"] / max_step_a
    x_b = reward_b["Step"] / max_step_b

    # --- Left axis: Reward ---
    ax1.set_xlabel("Training Progress (normalized)")
    ax1.set_ylabel("Mean Reward")
    ax1.plot(x_a, smooth(reward_a["reward"], weight=0.8),
             color=C_A, linewidth=1.8, alpha=0.35, label=f"{label_a} — Reward")
    ax1.plot(x_b, smooth(reward_b["reward"], weight=0.8),
             color=C_B, linewidth=1.8, alpha=0.35, label=f"{label_b} — Reward")
    ax1.set_xlim(0, 1)
    ax1.grid(axis="y", color="#888888", alpha=0.10, linewidth=0.5)

    # --- Right axis: Benchmark scores ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)")

    base_row = pd.DataFrame({"step": [0], "gsm8k_acc": [BASE_GSM8K], "math_acc": [BASE_MATH]})

    s_a = pd.concat([base_row, scores_a], ignore_index=True)
    s_a_norm = s_a["step"] / s_a["step"].max()
    s_b = pd.concat([base_row.copy(), scores_b], ignore_index=True)
    s_b_norm = s_b["step"] / s_b["step"].max()

    ax2.plot(s_a_norm, s_a["gsm8k_acc"] * 100,
             color=C_A, linestyle="--", linewidth=1.3, marker="o", markersize=2.5,
             label=f"{label_a} — GSM8K")
    ax2.plot(s_a_norm, s_a["math_acc"] * 100,
             color=C_A, linestyle=":", linewidth=1.3, marker="s", markersize=2.5,
             alpha=0.7, label=f"{label_a} — MATH")

    ax2.plot(s_b_norm, s_b["gsm8k_acc"] * 100,
             color=C_B, linestyle="--", linewidth=1.3, marker="D", markersize=2.5,
             label=f"{label_b} — GSM8K")
    ax2.plot(s_b_norm, s_b["math_acc"] * 100,
             color=C_B, linestyle=":", linewidth=1.3, marker="d", markersize=2.5,
             alpha=0.7, label=f"{label_b} — MATH")

    # Base model lines
    ax2.axhline(y=BASE_GSM8K * 100, color=C_GSM8K, linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Base GSM8K ({BASE_GSM8K:.1%})")
    ax2.axhline(y=BASE_MATH * 100, color=C_MATH, linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Base MATH ({BASE_MATH:.1%})")

    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=8, framealpha=0.85, edgecolor="#cccccc",
        borderpad=0.8, handlelength=2, ncol=2, loc="lower left",
    )
    legend.get_frame().set_linewidth(0.6)

    ax1.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


make_one_example_overlay(
    reward_drs_df, reward_gsm8k_one_df,
    scores_drs, scores_gsm8k_one,
    "\u03C01 1-Example", "GSM8K 1-Example",
    "GRPO One-Example: Reward & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-one-example-overlay-reward-and-scores.png",
)
