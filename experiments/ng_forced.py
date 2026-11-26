"""Paired closed-choice ablation on the fixed 20NG test split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json

import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.metrics import adjusted_mutual_info_score as ami
from topicjev.entail import LayaEntail

from ng_discovery import OUTDIR, documents
from verify_topics import DecisionRefusal, OpenAIDecisions, REV, RATE, request_body


def run(backend, budget):
    test = pd.read_csv(OUTDIR / "assignments.csv").set_index("id")
    docs = documents().set_index("id").loc[test.index, "text"].tolist()
    schema = json.loads((OUTDIR / "choices.json").read_text())["topic"]
    labels = [json.dumps({k: v}) for k, v in schema["criteria"].items()]
    assert len(labels) == 20 and len(test) == 160

    def force(model):
        model.decoys = []
        model.threshold = 0.0

    if backend == "laya":
        with LayaEntail(
            snapshot_download("convaiinnovations/laya-typed-decisions", revision=REV),
            batch_size=4,
        ) as m:
            force(m)
            pred = m.entail(docs, labels)
            assert len(m.questions["topic"]["criteria"]) == 20
        pd.DataFrame([p["probs"] for p in pred], index=test.index).to_csv(
            OUTDIR / "choice-scores.csv"
        )
    else:
        closed = schema
        limits = [
            RATE
            * (
                len(json.dumps(request_body(t, closed), ensure_ascii=False).encode())
                + 2048
            )
            for t in docs
        ]
        if sum(limits) > budget:
            raise ValueError(f"Required budget reservation: {sum(limits)}")

        def score(pair):
            text, limit = pair
            with OpenAIDecisions(limit) as m:
                force(m)
                try:
                    result = m.entail([text], labels)[0]
                except DecisionRefusal:
                    result = {"class": None}
                assert m.questions["topic"] == closed
            return result

        pairs = list(dict.fromkeys(zip(docs, limits)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = dict(zip(pairs, pool.map(score, pairs), strict=True))
        pred = [answers[key] for key in zip(docs, limits)]
    test["closed"] = [p["class"] for p in pred]
    test["api_refusal"] = test.closed.isna()
    test["closed"] = test.closed.fillna(test.kmeans).astype(int)
    assert test.closed.between(0, 19).all()
    test[["label", "kmeans", "closed", "api_refusal"]].to_csv(
        OUTDIR / f"{backend}-forced.csv"
    )
    print(backend, ami(test.label, test.closed))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["laya", "luna"])
    p.add_argument("--budget-usd", type=Decimal, default=Decimal(0))
    args = p.parse_args()
    run(args.backend, args.budget_usd)
