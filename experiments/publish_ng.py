"""Generate report tables and a topic-by-newsgroup matrix from saved predictions."""

import json

import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score as ami

from benchmark import ROOT, OUT

dest = ROOT / "tex"
src = OUT / "ng-discovery"
docs = pd.read_csv(src / "assignments.csv")
topics = json.loads((src / "topics.json").read_text())
test = docs.set_index("id", drop=False)
assert len(test) == 160 and test.id.is_unique


def fmt(x):
    return f"{x:.3f}".replace(".", ",")


def table(path, header, rows):
    (dest / path).write_text(
        "\n".join(
            [
                r"\begin{tabular}{lrr}\toprule",
                header + r"\\\midrule",
                *[" & ".join(row) + r"\\" for row in rows],
                r"\bottomrule\end{tabular}",
                "",
            ]
        )
    )


rows = []
records = []
for key, name in [
    ("plsa", "PLSA"),
    ("bigartm", "Сглаженная ARTM"),
    ("kmeans", "Nomic + $k$-means"),
    ("laya", "Qwen + Laya"),
    ("luna", "Qwen + GPT-6-Luna"),
]:
    labels = (
        test[key]
        if key in test
        else pd.read_csv(src / f"{key}-forced.csv")
        .set_index("id")
        .loc[test.index, "closed"]
    )
    score = ami(test.label, labels)
    changes = int((test.kmeans != labels).sum()) if key in ["laya", "luna"] else None
    rows.append([name, fmt(score), str(changes) if changes is not None else "---"])
    records.append(dict(method=key, ami=score, changes=changes))
pd.DataFrame(records).to_csv(src / "comparison.csv", index=False)
table("ng-test.tex", "Метод & AMI & Смены из 160", rows)
counts = pd.crosstab(test.label, test.kmeans).reindex(columns=range(20), fill_value=0)
rr, cc = linear_sum_assignment(-counts.to_numpy())
coords = ["x y n"]
for i in rr:
    for j, k in enumerate(cc):
        n = counts.iloc[i, k]
        if n:
            coords.append(f"{j} {i} {n}")
(dest / "ng-heatmap.dat").write_text("\n".join(coords) + "\n")
ticks = ",".join(str(i) for i in range(20))
fig = r"""\begin{center}
\begin{tikzpicture}\begin{axis}[width=.77\linewidth,height=8.2cm,xmin=-.5,xmax=19.5,ymin=-.5,ymax=19.5,y dir=reverse,xtick={TICKS},ytick={TICKS},xticklabels={COLS},yticklabels={LABELS},tick label style={font=\scriptsize},xlabel={Кластер $k$-means},colorbar,point meta min=0,point meta max=8,colormap={density}{color=(white) color=(pine)},colorbar style={ytick={0,2,4,6,8}}]
\addplot[only marks,scatter,scatter src=explicit,mark=square*,mark size=3.6pt] table[x=x,y=y,meta=n]{ng-heatmap.dat};
\end{axis}\end{tikzpicture}
\par\small Число тестовых писем каждой рубрики в каждом кластере. В строке восемь писем. Для чтения рисунка столбцы переставлены по максимальному диагональному совпадению. Перестановка используется только при визуализации.
\end{center}
"""
(dest / "ng-figure.tex").write_text(
    fig.replace("TICKS", ticks)
    .replace("COLS", ",".join(map(str, cc)))
    .replace("LABELS", ",".join(counts.index))
)


def escape(text):
    return "".join(
        {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}.get(c, c)
        for c in text
    )


# Select illustrative clusters by a fixed rule: most test letters, then cluster ID.
examples = []
for k in sorted(range(20), key=lambda k: (-counts[k].sum(), k))[:2]:
    top = topics[k]
    assert top["cluster"] == k
    examples.append(
        f"кластер {k} «{escape(top['title'])}» (группа с наибольшим числом писем --- \\texttt{{{counts[k].idxmax()}}}, {counts[k].max()} из {counts[k].sum()})"
    )
(dest / "ng-figure.tex").open("a").write(
    "Два наиболее многочисленных тестовых кластера иллюстрируют связь названия с рубрикацией: "
    + " и ".join(examples)
    + ".\n"
)


# Preserve literal class names when inserting generated English text in LaTeX.
criteria = json.loads((src / "choices.json").read_text())["topic"]["criteria"]
topic_name = list(criteria)[8]
topic_lines = [
    r"\paragraph{Полный вариант для кластера 8.} \textit{"
    + escape(topic_name)
    + "}. "
    + escape(criteria[topic_name]),
    r"\paragraph{Все варианты в порядке предъявления.}",
    r"\begin{enumerate}",
]
for title in criteria:
    topic_lines.append(r"\item " + escape(title))
topic_lines.append(r"\end{enumerate}")
(dest / "ng-choices.tex").write_text("\n".join(topic_lines) + "\n")
