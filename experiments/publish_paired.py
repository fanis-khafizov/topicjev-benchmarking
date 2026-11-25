"""Build the paired benchmark tables used by the talk and the paper."""

import numpy as np
import pandas as pd
from benchmark import OUT, ROOT

tex = ROOT / "tex"
source = OUT / "paired"
comp = pd.read_csv(source / "comparison.csv")
base = pd.read_csv(source / "baselines.csv")
assert list(comp.dataset) == ["20NG", "nips", "kos"]
assert len(base) == 9 and not base.duplicated(["dataset", "method"]).any()


def table(file, spec, header, rows):
    text = [r"\begin{tabular}{" + spec + r"}\toprule", header + r"\\\midrule"]
    text.extend(" & ".join(map(str, row)) + r"\\" for row in rows)
    text.extend([r"\bottomrule\end{tabular}", ""])
    (tex / file).write_text("\n".join(text))


pairs, controls, plot = [], [], ["i delta minus plus"]
for i, row in enumerate(comp.itertuples(), 1):
    tag = row.dataset
    records = pd.read_csv(source / f"{tag}-documents.csv")
    assert records.id.is_unique and len(records) == row.n
    assert np.array_equal(
        records.routed,
        np.where(records.decision == 20, records.bigartm, records.decision),
    )
    n = records.tokens.sum()
    assert np.isclose(np.exp(records.baseline_nll.sum() / n), row.base)
    assert np.isclose(np.exp(records.hybrid_nll.sum() / n), row.hybrid)
    local = base[base.dataset == tag].set_index("method")
    assert np.isclose(local.at["smooth", "hard"], row.base)
    pairs.append(
        [
            tag.upper(),
            f"{row.base:.0f}",
            f"{row.hybrid:.0f}",
            f"{row.change_pct:+.1f}",
            row.accepted,
            row.changed,
        ]
    )
    controls.append(
        [
            tag.upper(),
            f"{local.at['smooth', 'unigram']:.0f}",
            *[f"{local.at[m, 'mixture']:.0f}" for m in ["PLSA", "smooth", "decor"]],
        ]
    )
    plot.append(
        f"{i} {row.change_pct} {row.change_pct - row.ci_low} {row.ci_high - row.change_pct}"
    )
table(
    "paired-table.tex",
    "lrrrrr",
    r"Корпус & BigARTM & + Laya & $\Delta$, \% & Принято & Смены",
    pairs,
)
table(
    "paired-controls.tex",
    "lrrrr",
    r"Корпус & Униграмма & PLSA & Сглаж. & Сглаж. + декор.",
    controls,
)
(tex / "paired-plot.dat").write_text("\n".join(plot) + "\n")
