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
names = [
    "PLSA (BigARTM)",
    "Сглаженная ARTM",
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
openai = pd.read_csv(src / "openai-comparison.csv")
api_docs = pd.read_csv(src / "openai-assignments.csv").set_index("id").loc[test.id]
assert np.array_equal(api_docs.kmeans, test.kmeans) and np.array_equal(
    api_docs.label, test.label
)
for r in openai.itertuples():
    assert np.isclose(ami(api_docs.label, api_docs[r.method]), r.ami)
    threshold_name = "штатный" if r.method == "default" else "выбранный"
    rows.append([f"Qwen + Luna, {threshold_name} порог", fmt(r.ami), str(r.changes)])
api_grid = pd.read_csv(src / "openai-tuning.csv")
assert np.allclose(api_grid.threshold, grid.threshold)
table(
    "ng-tuning.tex",
    r"Порог $\tau$ & AMI Laya & AMI Luna",
    [
        [f"{r.threshold:.1f}".replace(".", ","), fmt(r.ami), fmt(a)]
        for r, a in zip(grid.itertuples(), api_grid.ami, strict=True)
    ],
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
old = pd.read_csv(src / "qwen3-1.7b/comparison.csv").set_index("method")
new = results.set_index("method")
api = openai.set_index("method")
seeds = pd.read_csv(src / "classical-seeds.csv")
assert np.isclose(
    seeds.query("method=='PLSA' and seed==11").ami.item(), new.at["plsa", "ami"]
)
assert np.isclose(
    seeds.query("method=='smooth' and seed==11").ami.item(), new.at["bigartm", "ami"]
)


def valnum(n):
    return f"{n:g}".replace(".", ",")


text = (
    f"При пороге 0,1 Laya с описаниями Qwen3-1.7B дала AMI {fmt(old.at['default', 'ami'])}, "
    f"с описаниями Qwen3.5-4B --- {fmt(new.at['default', 'ami'])}. "
    f"Центры, пять представителей каждого кластера и тестовые письма совпадают. "
    f"После отдельного подбора порога для Laya ($\\tau={valnum(best.threshold)}$) и GPT-6-Luna "
    f"($\\tau={valnum(api.at['selected', 'threshold'])}$) тестовая AMI составила "
    f"{fmt(new.at['selected', 'ami'])} и {fmt(api.at['selected', 'ami'])} соответственно.\n\n"
    "В таблице для BigARTM приведена заранее заданная инициализация 11. Для проверки устойчивости "
    "повторили обучение с инициализациями 23 и 47. У PLSA диапазон тестовой AMI по трём запускам "
    f"составил {fmt(seeds.query('method=="PLSA"').ami.min())}--{fmt(seeds.query('method=="PLSA"').ami.max())}, "
    f"у сглаженной модели --- {fmt(seeds.query('method=="smooth"').ami.min())}--{fmt(seeds.query('method=="smooth"').ami.max())}. "
    "Во всех шести случаях на тестовых документах представлены все 20 тем.\n"
)
refusals = int((api_docs.decision == 20).sum())
agreed = int(((api_docs.decision < 20) & (api_docs.decision == api_docs.kmeans)).sum())
text += (
    f"\nПри штатном пороге GPT-6-Luna вернула Other для {refusals} тестовых писем, "
    f"подтвердила исходную тему для {agreed} и сменила её для {int(api.at['default', 'changes'])}. "
    "Большая доля отказов объясняет близость результата к исходному разбиению.\n"
)
(dest / "ng-results.tex").write_text(text)
summary = (
    f"На 160 тестовых письмах PLSA получила AMI {fmt(new.at['plsa', 'ami'])}, ближайший центр Nomic --- "
    f"{fmt(new.at['kmeans', 'ami'])}. При одинаковых описаниях Qwen3.5-4B и штатном пороге "
    f"Laya дала {fmt(new.at['default', 'ami'])}, GPT-6-Luna --- {fmt(api.at['default', 'ami'])}. "
    f"При порогах, выбранных по проверочной выборке, получено {fmt(new.at['selected', 'ami'])} и "
    f"{fmt(api.at['selected', 'ami'])} соответственно."
)
(dest / "ng-conclusion.tex").write_text(summary + "\n")
