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
LOSS_TEST_CSV = "plot-data/sft-loss-gsm8k-test-wandb.csv"
LOSS_TRAIN_CSV = "plot-data/sft-loss-gsm8k-train-wandb.csv"
SQLITE_DB = "sft-gsm8k-test-train-success.sqlite3"
SAVE_DIR = "plots"
TIMESTAMP = datetime.now().strftime("%H%M")

# --- Base model scores (Qwen2.5-1.5B-Instruct) ---
BASE_GSM8K = 0.6975
BASE_MATH = 0.4942

# --- Colors ---
C_LOSS = "#4A90D9"
C_LOSS_RAW = "#A8CCEE"
C_GSM8K = "#D94A6B"
C_MATH = "#3AAA5B"
C_BASE_GSM8K = "#D94A6B"
C_BASE_MATH = "#3AAA5B"

# --- Load loss CSVs ---
loss_test_df = pd.read_csv(LOSS_TEST_CSV)
loss_test_df.columns = ["Step", "loss", "loss_min", "loss_max"]

loss_train_df = pd.read_csv(LOSS_TRAIN_CSV)
loss_train_df.columns = ["Step", "loss", "loss_min", "loss_max"]

# --- Load benchmark scores from SQLite ---
conn = sqlite3.connect(SQLITE_DB)
summaries = pd.read_sql_query("SELECT model_tag, task_name, accuracy FROM summaries", conn)
conn.close()


def extract_scores(summaries, prefix):
    """Extract step and accuracy for a given model prefix (test or train)."""
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


scores_test = extract_scores(summaries, "sft-gsm8k-test-docker")
scores_train = extract_scores(summaries, "sft-gsm8k-train-docker")


def smooth(values, weight=0.9):
    """Exponential moving average for smoothing."""
    smoothed = []
    last = values.iloc[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def make_plot(loss_df, scores_df, title, save_name, total_epochs):
    fig, ax1 = plt.subplots(figsize=(13, 5.5))

    # --- Epoch info ---
    max_step = loss_df["Step"].max()
    steps_per_epoch = max_step / total_epochs

    # --- Left axis: Loss ---
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Training Loss", color=C_LOSS)
    ax1.plot(
        loss_df["Step"], loss_df["loss"],
        color=C_LOSS_RAW, alpha=0.35, linewidth=0.4, rasterized=True,
    )
    ax1.plot(
        loss_df["Step"], smooth(loss_df["loss"]),
        color=C_LOSS, linewidth=2, label="Loss (smoothed)",
    )
    ax1.tick_params(axis="y", labelcolor=C_LOSS)
    ax1.set_xlim(0, max_step)
    ax1.grid(axis="y", color=C_LOSS, alpha=0.10, linewidth=0.5)

    # --- Right axis: Benchmark scores ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)")

    ax2.plot(
        scores_df["step"], scores_df["gsm8k_acc"] * 100,
        color=C_GSM8K, linewidth=1.3,
        label="GSM8K",
    )
    ax2.plot(
        scores_df["step"], scores_df["math_acc"] * 100,
        color=C_MATH, linewidth=1.3,
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

    ax2.set_ylim(15, 85)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax2.grid(axis="y", color="#888888", alpha=0.10, linewidth=0.5)

    # --- Combined legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="center right", fontsize=9, framealpha=0.85,
        edgecolor="#cccccc", borderpad=0.8, handlelength=2,
    )
    legend.get_frame().set_linewidth(0.6)

    # --- Secondary x-axis for epochs ---
    ax_epoch = ax1.secondary_xaxis("top")
    epoch_ticks = [i * steps_per_epoch for i in range(1, total_epochs + 1)]
    ax_epoch.set_xticks(epoch_ticks)
    ax_epoch.set_xticklabels([str(i) for i in range(1, total_epochs + 1)], fontsize=8)
    ax_epoch.set_xlabel("Epoch", fontsize=11, labelpad=6)
    ax_epoch.tick_params(length=3, width=0.6)
    # Re-enable top spine for the epoch axis
    ax_epoch.spines["top"].set_visible(True)
    ax_epoch.spines["top"].set_linewidth(0.6)
    ax_epoch.spines["top"].set_color("#aaaaaa")

    ax1.set_title(title, fontsize=15, fontweight="bold", pad=30)

    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


# --- Plot 1: SFT on GSM8K Test (15 epochs) ---
make_plot(
    loss_test_df, scores_test,
    "SFT on GSM8K Test Set: Training Loss & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-sft-gsm8k-test-loss-and-scores.png",
    total_epochs=15,
)

# --- Plot 2: SFT on GSM8K Train (7 epochs) ---
make_plot(
    loss_train_df, scores_train,
    "SFT on GSM8K Train Set: Training Loss & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-sft-gsm8k-train-loss-and-scores.png",
    total_epochs=7,
)


# --- Plot 3: Overlay benchmark scores + loss from both SFT experiments ---
def make_overlay_plot(loss_list, scores_list, labels, title, save_name):
    """Plot loss (left y) and benchmark scores (right y) on normalized x-axis."""
    fig, ax1 = plt.subplots(figsize=(13, 5.5))

    C_TEST_LOSS = "#4A90D9"
    C_TRAIN_LOSS = "#8E44AD"
    loss_colors = [C_TEST_LOSS, C_TRAIN_LOSS]
    line_styles = ["--", "-"]  # test = dashed, train = solid

    # --- Left axis: Loss (smoothed) ---
    ax1.set_xlabel("Training Progress (normalized)")
    ax1.set_ylabel("Training Loss")
    for i, (loss_df, label) in enumerate(zip(loss_list, labels)):
        max_step = loss_df["Step"].max()
        x_norm = loss_df["Step"] / max_step
        ax1.plot(
            x_norm, smooth(loss_df["loss"]),
            color=loss_colors[i], linewidth=1.8, alpha=0.35,
            linestyle=line_styles[i], label=f"{label} — Loss",
        )
    ax1.set_xlim(0, 1)
    ax1.grid(axis="y", color="#888888", alpha=0.10, linewidth=0.5)

    # --- Right axis: Benchmark scores ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)")

    for i, (scores_df, label) in enumerate(zip(scores_list, labels)):
        base_row = pd.DataFrame({"step": [0], "gsm8k_acc": [BASE_GSM8K], "math_acc": [BASE_MATH]})
        s = pd.concat([base_row, scores_df], ignore_index=True)
        max_step = s["step"].max()
        x_norm = s["step"] / max_step

        ax2.plot(
            x_norm, s["gsm8k_acc"] * 100,
            color=C_GSM8K, linewidth=1.5,
            linestyle=line_styles[i], label=f"{label} — GSM8K",
        )
        ax2.plot(
            x_norm, s["math_acc"] * 100,
            color=C_MATH, linewidth=1.5,
            linestyle=line_styles[i], alpha=0.7, label=f"{label} — MATH",
        )

    # Base model lines
    ax2.axhline(
        y=BASE_GSM8K * 100, color=C_GSM8K, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base GSM8K ({BASE_GSM8K:.1%})",
    )
    ax2.axhline(
        y=BASE_MATH * 100, color=C_MATH, linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Base MATH ({BASE_MATH:.1%})",
    )

    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=8, framealpha=0.85, edgecolor="#cccccc",
        borderpad=0.8, handlelength=2, ncol=2,
        loc="lower left",
    )
    legend.get_frame().set_linewidth(0.6)

    ax1.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


make_overlay_plot(
    [loss_test_df, loss_train_df],
    [scores_test, scores_train],
    ["GSM8K Test (15 ep)", "GSM8K Train (7 ep)"],
    "SFT Train vs Test: Loss & Benchmark Scores",
    f"{SAVE_DIR}/{TIMESTAMP}-sft-all-benchmark-scores-overlay.png",
)
