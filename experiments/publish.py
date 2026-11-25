"""Create TeX tables and plotting coordinates directly from measured predictions."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
from benchmark import ROOT, OUT

TEX = ROOT / "tex"
names = {"ag": "AG News", "ng": "20 NG"}
methods = {"PLSA": "PLSA", "smooth": "Сглаживание", "decor": "Сглаж. + декор."}
data = pd.read_csv(OUT / "metrics.csv")
assert len(data) == 18 and not data.duplicated(["dataset", "method", "seed"]).any()
rows = []
graph = []
for tag, label in names.items():
    for i, (method, title) in enumerate(methods.items()):
        x = data[(data.dataset == tag) & (data.method == method)]
        rows.append(
            f"{label} & {title} & {x.ppl.mean():.0f} & {x.npmi.mean():.3f} & {x.diversity.mean():.3f} & {x.ami.mean():.3f} $\\pm$ {x.ami.std():.3f}"
        )
        graph.append(
            dict(
                dataset=tag,
                method=i + 1,
                percent=100 * x.ppl.mean() / x.unigram_ppl.mean(),
            )
        )
(TEX / "baseline-rows.tex").write_text(" \\\\\n".join(rows) + " \\\\\n")
pd.DataFrame(graph).pivot(index="method", columns="dataset", values="percent").to_csv(
    TEX / "ppl.dat", sep=" "
)
rows = []
sensitivity = []
routing = []
for engine, title in [("laya", "Laya"), ("openai", "OpenAI")]:
    summary = OUT / f"summary-{engine}-100.csv"
    if not summary.exists():
        continue
    for s in pd.read_csv(summary).itertuples():
        x = pd.read_csv(OUT / f"{s.dataset}-{engine}-100.csv")
        probs = np.array(x.probs.map(json.loads).tolist())
        scores = np.column_stack([probs, np.maximum(0, 1 - probs.sum(1))])
        raw = scores.argmax(1)
        other = probs.shape[1]
        keep = x.decision != other
        rows.append(
            f"{names[s.dataset]} & {title} & {s.ami_base:.3f} & {s.ami_all:.3f} & {sum(keep)}/{len(x)} & {s.agreement:.0%}".replace(
                "%", r"\%"
            )
        )
        if engine == "laya":
            routing.append(
                dict(
                    dataset=s.dataset,
                    accepted=int(keep.sum()),
                    direct=int(sum(raw == other)),
                    threshold=int(sum((raw != other) & ~keep)),
                )
            )
        sensitivity.append(
            f"{names[s.dataset]} & {title} & {sum(raw == other)} & {sum((raw != other) & ~keep)} & {ami(x.label, raw):.3f} & {ami(x.loc[keep, 'label'], x.loc[keep, 'topic']):.3f} & {s.ami_kept:.3f}"
        )
(TEX / "decision-rows.tex").write_text(" \\\\\n".join(rows) + " \\\\\n")
(TEX / "threshold-rows.tex").write_text(" \\\\\n".join(sensitivity) + " \\\\\n")

pd.DataFrame(routing).to_csv(TEX / "routing.dat", index=False, sep=" ")
