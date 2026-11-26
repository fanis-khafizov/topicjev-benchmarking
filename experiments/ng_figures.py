"""Visualize fitted centers, representative training texts and a held-out decision."""

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from benchmark import ROOT
from ng_discovery import OUTDIR, documents

DEST = ROOT / "tex"
BLUE = "#234d86"
GREEN = "#007168"
ORANGE = "#b3551f"
INK = "#172936"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name, prefix):
    fig.savefig(DEST / f"{prefix}-{name}.pdf", bbox_inches="tight", pad_inches=0.09)
    plt.close(fig)


def panel():
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.009,rounding_size=0.015",
            facecolor="#f7f9fa",
            edgecolor=color,
            linewidth=1.3,
        )
    )


def label(ax, x, y, s, **kw):
    ax.text(x, y, s, va="top", color=kw.pop("color", INK), **kw)


def short(s, n):
    words = s.split()
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def figures(source=OUTDIR, texts=None, prefix="ng", example_id=None, neighbor=18):
    if texts is None:
        texts = documents().set_index("id").text
    topics = json.loads((source / "topics.json").read_text())
    info = json.loads((source / "projection-info.json").read_text())
    data = pd.read_csv(source / "projection.csv").set_index("id")
    docs = pd.read_csv(source / "assignments.csv")
    if example_id is None:
        candidates = docs[docs.kmeans.eq(8)].sort_values("id")
        row = candidates[candidates.id.map(texts).str.len().gt(0)].iloc[0]
    else:
        row = docs[docs.id.eq(example_id)].iloc[0]
    example = int(row.kmeans)
    t = topics[example]
    reps = data.loc[t["representatives"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for k, col, name, size in [
        (-1, "#bbc4cf", "Другие кластеры", 12),
        (neighbor, ORANGE, f"Кластер {neighbor}", 26),
        (example, GREEN, f"Кластер {example}", 27),
    ]:
        sub = (
            data[~data.cluster.isin([example, neighbor])]
            if k == -1
            else data[data.cluster == k]
        )
        ax.scatter(
            sub.x, sub.y, s=size, color=col, alpha=0.65, label=name, linewidths=0
        )
    ax.scatter(
        reps.x,
        reps.y,
        s=90,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.25,
        label=f"5 представителей № {example}",
    )
    for k in [example, neighbor]:
        x, y = info["centers"][k]
        ax.scatter(x, y, marker="X", s=120, c=INK, edgecolor="white", linewidth=1)
        ax.annotate(
            f"центр {k}",
            (x, y),
            xytext=(16, 10 if k == example else -25),
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5),
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set(
        xlabel=f"Первая главная компонента ({info['variance'][0]:.1%})",
        ylabel=f"Вторая компонента ({info['variance'][1]:.1%})",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, "clusters", prefix)
    fig, ax = panel()
    box(ax, 0.012, 0.04, 0.51, 0.89, BLUE)
    box(ax, 0.585, 0.04, 0.395, 0.89, GREEN)
    label(
        ax, 0.035, 0.90, "Пять ближайших писем", weight="bold", color=BLUE, fontsize=13
    )
    for i, did in enumerate(t["representatives"]):
        snippet = textwrap.fill(
            short(texts[did], 18), 55, max_lines=2, placeholder=" …"
        )
        label(ax, 0.035, 0.765 - i * 0.142, f"{i + 1}. {snippet}", fontsize=9)
    ax.annotate(
        "",
        xy=(0.582, 0.5),
        xytext=(0.528, 0.5),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2},
    )
    label(ax, 0.607, 0.90, "Qwen3.5-4B", weight="bold", color=GREEN, fontsize=13)
    label(ax, 0.607, 0.76, "1. Обобщение и ключевые слова", weight="bold", fontsize=10)
    label(ax, 0.607, 0.67, textwrap.fill(", ".join(t["keywords"]), 41), fontsize=10)
    label(ax, 0.607, 0.41, "2. Название темы", weight="bold", fontsize=10)
    label(ax, 0.607, 0.32, textwrap.fill(t["title"], 34), fontsize=13, color=GREEN)
    save(fig, "naming", prefix)
    scores = (
        pd.read_csv(source / "choice-scores.csv").set_index("id").loc[row.id].to_numpy()
    )
    probs = scores
    order = np.argsort(-probs)[:3]
    fig, ax = panel()
    box(ax, 0.012, 0.05, 0.42, 0.88, BLUE)
    box(ax, 0.47, 0.05, 0.515, 0.88, GREEN)
    label(ax, 0.032, 0.89, "Новое письмо", color=BLUE, weight="bold", fontsize=13)
    label(ax, 0.032, 0.76, textwrap.fill(short(texts[row.id], 24), 42), fontsize=10)
    label(ax, 0.032, 0.36, f"Ближайший центр: {row.kmeans}", weight="bold", color=BLUE)
    if "luna" in row:
        refusal = " (отказ)" if row.get("api_refusal", False) else ""
        label(
            ax, 0.032, 0.245, f"GPT-6-Luna: кластер {row.luna}{refusal}", weight="bold"
        )
    if "laya_known" in row:
        label(ax, 0.032, 0.135, f"Laya по рубрикам: {row.laya_known}", fontsize=9)
    label(
        ax,
        0.49,
        0.89,
        "Выбор Laya среди 20 тем",
        weight="bold",
        color=GREEN,
        fontsize=12,
    )
    label(ax, 0.49, 0.77, "Три варианта с наибольшей оценкой", fontsize=9)
    for i, k in enumerate(order):
        name = topics[k]["title"]
        label(ax, 0.49, 0.65 - i * 0.14, textwrap.fill(name, 44), fontsize=9)
        label(ax, 0.91, 0.65 - i * 0.14, f"{probs[k]:.1%}", color=GREEN, weight="bold")
    label(ax, 0.49, 0.20, f"Выбран кластер {order[0]}.", weight="bold")
    save(fig, "decision", prefix)


if __name__ == "__main__":
    figures()
