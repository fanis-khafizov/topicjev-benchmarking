"""Generate report tables and a topic-by-newsgroup matrix from saved predictions."""

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score as ami

from benchmark import ROOT, OUT

dest = ROOT / "tex"
src = OUT / "ng-discovery"
docs = pd.read_csv(src / "assignments.csv")
grid = pd.read_csv(src / "tuning.csv")
results = pd.read_csv(src / "comparison.csv")
topics = json.loads((src / "topics.json").read_text())
val = docs.query("split=='validation'")
test = docs.query("split=='test'")
assert len(val) == 200 and len(test) == 160 and docs.id.is_unique
best = grid.sort_values(["ami", "threshold"], ascending=False).iloc[0]


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


for r in grid.itertuples():
    pred = np.where(
        (val.decision < 20) & (val.confidence > r.threshold), val.decision, val.kmeans
    )
    assert np.isclose(ami(val.label, pred), r.ami)
table(
    "ng-tuning.tex",
    r"Порог $\tau$ & AMI & Смены из 200",
    [
        [str(r.threshold).replace(".", ","), fmt(r.ami), str(r.changed)]
        for r in grid.itertuples()
    ],
)
names = [
    "BigARTM",
    "Nomic + $k$-means",
    "Qwen + Laya, штатный порог",
    "Qwen + Laya, выбранный порог",
]
rows = []
for r, name in zip(results.itertuples(), names, strict=True):
    assert np.isclose(ami(test.label, test[r.method]), r.ami)
    rows.append(
        [name, fmt(r.ami), str(int(r.changes)) if pd.notna(r.changes) else "---"]
    )
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
# Select illustrative clusters by a fixed rule: most test letters, then cluster ID.
examples = []
for k in sorted(range(20), key=lambda k: (-counts[k].sum(), k))[:2]:
    top = topics[k]
    assert top["cluster"] == k
    examples.append(
        f"кластер {k} «{top['title']}» (группа с наибольшим числом писем --- \\texttt{{{counts[k].idxmax()}}}, {counts[k].max()} из {counts[k].sum()})"
    )
(dest / "ng-figure.tex").open("a").write(
    "Два наиболее многочисленных тестовых кластера иллюстрируют связь названия с рубрикацией: "
    + " и ".join(examples)
    + ".\n"
)
slide = []
for k in (8, 18):
    top = topics[k]
    slide.append(r"\point{" + top["title"] + "} " + ", ".join(top["keywords"]) + ".")
slide.append(
    r"\par\medskip Два описания обучающих кластеров. В рубрикаторе 20NG хоккею соответствует одна группа \texttt{rec.sport.hockey}."
)
(dest / "ng-example.tex").write_text("\n".join(slide) + "\n")
