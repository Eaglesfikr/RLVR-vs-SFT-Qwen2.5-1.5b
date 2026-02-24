import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- Global style ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
})

# --- Paths ---
ENTROPY_PI1_CSV = "plot-data/entropy-grpo-dsr-one-example.csv"
ENTROPY_GSM8K_ONE_CSV = "plot-data/entropy-grpo-gsm8k-one-example.csv"
ENTROPY_GSM8K_TEST_CSV = "plot-data/entropy-grpo-gsm8k-test.csv"

ENTROPY_GSM8K_TRAIN_CSV = "plot-data/entropy-grpo-train.csv"

KL_PI1_CSV = "plot-data/kl_loss-grpo-drs-one-example.csv"
KL_GSM8K_ONE_CSV = "plot-data/kl_loss-grpo-gsm8k-one-example.csv"
KL_GSM8K_TEST_CSV = "plot-data/kl_loss-grpo-gsm8k-test.csv"
KL_GSM8K_TRAIN_CSV = "plot-data/kl_loss-grpo-train.csv"

SAVE_DIR = "plots"
TIMESTAMP = datetime.now().strftime("%H%M")

# --- Colors (one per experiment) ---
C_PI1 = "#D94A6B"
C_GSM8K_ONE = "#4A90D9"
C_GSM8K_TEST = "#E67E22"
C_GSM8K_TRAIN = "#8E44AD"

# --- Load CSVs ---
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = ["Step", "value", "value_min", "value_max"]
    return df

entropy_pi1 = load_csv(ENTROPY_PI1_CSV)
entropy_gsm8k_one = load_csv(ENTROPY_GSM8K_ONE_CSV)
entropy_gsm8k_test = load_csv(ENTROPY_GSM8K_TEST_CSV)
entropy_gsm8k_train = load_csv(ENTROPY_GSM8K_TRAIN_CSV)

kl_pi1 = load_csv(KL_PI1_CSV)
kl_gsm8k_one = load_csv(KL_GSM8K_ONE_CSV)
kl_gsm8k_test = load_csv(KL_GSM8K_TEST_CSV)
kl_gsm8k_train = load_csv(KL_GSM8K_TRAIN_CSV)


def smooth(values, weight=0.9):
    """Exponential moving average for smoothing."""
    smoothed = []
    last = values.iloc[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def make_overlay_plot(data_list, labels, colors, title, ylabel, save_name,
                      smooth_weight=0.9, raw_alphas=None):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    if raw_alphas is None:
        raw_alphas = [0.45] * len(data_list)

    for df, label, color, ra in zip(data_list, labels, colors, raw_alphas):
        # Raw
        ax.plot(
            df["Step"], df["value"],
            color=color, alpha=ra, linewidth=0.5, rasterized=True,
        )
        # Smoothed
        ax.plot(
            df["Step"], smooth(df["value"], weight=smooth_weight),
            color=color, linewidth=1.8, label=label,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#888888", alpha=0.12, linewidth=0.5)
    ax.grid(axis="x", color="#888888", alpha=0.08, linewidth=0.5)

    legend = ax.legend(
        fontsize=9, framealpha=0.85, edgecolor="#cccccc",
        borderpad=0.8, handlelength=2,
    )
    legend.get_frame().set_linewidth(0.6)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_name}")
    plt.close(fig)


LABELS = ["\u03C01 1-Example", "GSM8K 1-Example", "GSM8K Test", "GSM8K Train"]
COLORS = [C_PI1, C_GSM8K_ONE, C_GSM8K_TEST, C_GSM8K_TRAIN]
RAW_ALPHAS = [0.45, 0.45, 0.2, 0.2]

# --- Plot 1: Entropy ---
make_overlay_plot(
    [entropy_pi1, entropy_gsm8k_one, entropy_gsm8k_test, entropy_gsm8k_train],
    LABELS, COLORS,
    "GRPO Actor Entropy Across Experiments",
    "Entropy",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-entropy-overlay.png",
    raw_alphas=RAW_ALPHAS,
)

# --- Plot 2: KL Loss ---
make_overlay_plot(
    [kl_pi1, kl_gsm8k_one, kl_gsm8k_test, kl_gsm8k_train],
    LABELS, COLORS,
    "GRPO Actor KL Loss Across Experiments",
    "KL Loss",
    f"{SAVE_DIR}/{TIMESTAMP}-grpo-kl-loss-overlay.png",
    raw_alphas=RAW_ALPHAS,
)
