"""Check seed sensitivity of the two classical topic models on the same letters."""

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from ng_discovery import OUTDIR, WORK, baseline, documents


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    df = documents()
    fit = df.query("split=='train'")
    test = df.query("split=='test'")
    rows = []
    for seed in (11, 23, 47):
        for name, smooth in [("PLSA", False), ("smooth", True)]:
            assignment = baseline(fit.text, test.text, smooth, seed)
            rows.append(
                dict(
                    method=name,
                    seed=seed,
                    ami=adjusted_mutual_info_score(test.label, assignment),
                    used_topics=len(set(assignment)),
                )
            )
    pd.DataFrame(rows).to_csv(OUTDIR / "classical-seeds.csv", index=False)


if __name__ == "__main__":
    main()
