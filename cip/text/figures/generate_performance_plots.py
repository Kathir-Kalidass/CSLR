#!/usr/bin/env python3
"""Generate performance comparison figures used in Chapter 4."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


METHODS = [
    "Yenisari et al.",
    "Geetha et al.\n(SignFlow)",
    "Pandey et al.",
    "This Work",
]

METRICS = {
    "Accuracy": [78.0, 78.8, 76.5, 79.6],
    "Precision": [77.3, 78.0, 75.9, 78.9],
    "Recall": [76.8, 77.5, 75.2, 78.3],
    "F1": [77.0, 77.8, 75.5, 78.7],
    "WER": [21.2, 20.3, 22.4, 19.4],
    "CER": [13.9, 13.4, 14.8, 12.6],
    "Latency(ms)": [320, 292, 340, 267],
}

# Color policy:
# - Yenisari (base paper) has its own pair.
# - Other reference papers share one pair.
# - This work uses a distinct pair.
YEN_ACC = "#2C7A7B"
YEN_F1 = "#7BC8C8"
REF_ACC = "#4E79A7"
REF_F1 = "#A0CBE8"
WORK_ACC = "#E15759"
WORK_F1 = "#FF9D9A"


def _style() -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_context("talk", font_scale=0.95)


def plot_heatmap(output_path: str) -> None:
    data = np.array(
        [
            METRICS["Accuracy"],
            METRICS["Precision"],
            METRICS["Recall"],
            METRICS["F1"],
            METRICS["WER"],
            METRICS["CER"],
            METRICS["Latency(ms)"],
        ]
    ).T

    fig, ax = plt.subplots(figsize=(14, 6.35), dpi=133)
    sns.heatmap(
        data,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="white",
        xticklabels=list(METRICS.keys()),
        yticklabels=METHODS,
        cbar_kws={"label": "Metric Value"},
        ax=ax,
        annot_kws={"fontsize": 12, "color": "#1f1f1f"},
    )

    ax.set_title("Comparative Performance Heatmap", fontsize=18, pad=14)
    ax.tick_params(axis="x", rotation=24)
    ax.tick_params(axis="y", rotation=0)

    # Keep palette story consistent by coloring each method label family.
    ylabels = ax.get_yticklabels()
    for i, label in enumerate(ylabels):
        if i == 0:
            label.set_color(YEN_ACC)
        elif i == 3:
            label.set_color(WORK_ACC)
        else:
            label.set_color(REF_ACC)
        label.set_fontweight("bold" if i == 3 else "normal")

    # Highlight Yenisari and This Work rows for quick visual lookup.
    ax.add_patch(
        plt.Rectangle((0, 0), data.shape[1], 1, fill=False, edgecolor=YEN_ACC, linewidth=1.8)
    )
    ax.add_patch(
        plt.Rectangle((0, 3), data.shape[1], 1, fill=False, edgecolor=WORK_ACC, linewidth=2.2)
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_bar(output_path: str) -> None:
    acc = np.array(METRICS["Accuracy"])
    f1 = np.array(METRICS["F1"])

    x = np.arange(len(METHODS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=130)

    acc_colors = [YEN_ACC, REF_ACC, REF_ACC, WORK_ACC]
    f1_colors = [YEN_F1, REF_F1, REF_F1, WORK_F1]

    bars1 = ax.bar(
        x - width / 2,
        acc,
        width,
        color=acc_colors,
        edgecolor="#1f1f1f",
        linewidth=0.8,
        label="Accuracy",
    )
    bars2 = ax.bar(
        x + width / 2,
        f1,
        width,
        color=f1_colors,
        edgecolor="#1f1f1f",
        linewidth=0.8,
        label="F1-score",
    )

    ax.set_title("Accuracy vs F1-score Across Methods", fontsize=22, pad=12)
    ax.set_ylabel("Score (%)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(70, 82)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.12,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="semibold",
        )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(output_path: str) -> None:
    latency = np.array(METRICS["Latency(ms)"])
    wer = np.array(METRICS["WER"])

    fig, ax = plt.subplots(figsize=(12, 7.1), dpi=131)

    yen_idx = 0
    ref_idx = [1, 2]
    work_idx = 3

    # Light companions around each point to show local variability.
    yen_offsets = np.array([[-4, -0.2], [-2, 0.1], [2, -0.1], [4, 0.3]])
    ref_offsets = np.array([[-4, -0.2], [-2, 0.1], [2, -0.1], [4, 0.3]])
    work_offsets = np.array([[-3, -0.2], [0, 0.0], [3, 0.2]])

    xy = latency[yen_idx] + yen_offsets[:, 0]
    yy = wer[yen_idx] + yen_offsets[:, 1]
    ax.scatter(xy, yy, s=85, color=YEN_F1, alpha=0.55, edgecolors="none", zorder=2)

    for i in ref_idx:
        x = latency[i] + ref_offsets[:, 0]
        y = wer[i] + ref_offsets[:, 1]
        ax.scatter(x, y, s=85, color=REF_F1, alpha=0.55, edgecolors="none", zorder=2)

    xw = latency[work_idx] + work_offsets[:, 0]
    yw = wer[work_idx] + work_offsets[:, 1]
    ax.scatter(xw, yw, s=85, color=WORK_F1, alpha=0.6, edgecolors="none", zorder=2)

    ax.scatter(
        [latency[yen_idx]],
        [wer[yen_idx]],
        s=260,
        color=YEN_ACC,
        edgecolor="#1f1f1f",
        linewidth=1.7,
        zorder=4,
    )
    ax.scatter(
        latency[ref_idx],
        wer[ref_idx],
        s=260,
        color=REF_ACC,
        edgecolor="#1f1f1f",
        linewidth=1.7,
        zorder=4,
    )
    ax.scatter(
        [latency[work_idx]],
        [wer[work_idx]],
        s=260,
        color=WORK_ACC,
        marker="D",
        edgecolor="#1f1f1f",
        linewidth=1.7,
        zorder=5,
    )

    for i, method in enumerate(METHODS):
        dx = 0.8 if i != 3 else 0.5
        dy = 0.05
        ax.text(latency[i] + dx, wer[i] + dy, method.replace("\n", " "), fontsize=14)

    # Lower-left is better for both metrics.
    line_x = np.array([260, 347])
    line_y = np.array([19.0, 22.85])
    ax.plot(line_x, line_y, linestyle="--", linewidth=2, color="#4d4d4d", alpha=0.9)
    ax.annotate(
        "Better",
        xy=(260, 19.0),
        xytext=(346.5, 22.8),
        arrowprops=dict(arrowstyle="->", lw=2, color="#4d4d4d"),
        fontsize=18,
        color="#3a3a3a",
    )

    ax.set_title("Latency vs Word Error Rate (Lower is Better)", fontsize=22, pad=10)
    ax.set_xlabel("Latency (ms)", fontsize=16)
    ax.set_ylabel("WER (%)", fontsize=16)
    ax.set_xlim(255, 349)
    ax.set_ylim(18.7, 23.0)
    ax.grid(True, linestyle="--", alpha=0.6)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=YEN_ACC,
               markeredgecolor="#1f1f1f", markersize=11, label="Yenisari (Base Paper)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REF_ACC,
               markeredgecolor="#1f1f1f", markersize=11, label="Other Reference Papers"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=WORK_ACC,
               markeredgecolor="#1f1f1f", markersize=11, label="This Work"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    plot_heatmap("perf_heatmap_metrics.png")
    plot_bar("perf_bar_acc_f1.png")
    plot_scatter("perf_scatter_latency_wer.png")


if __name__ == "__main__":
    main()
