"""Paired closed-choice ablation on the fixed 20NG test split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.metrics import adjusted_mutual_info_score as ami
from topicjev.entail import LayaEntail

from ng_discovery import OUTDIR, documents
from verify_topics import DecisionRefusal, OpenAIDecisions, REV, RATE, request_body


def run(backend, budget):
    source = "openai-assignments.csv" if backend == "luna" else "assignments.csv"
    test = pd.read_csv(OUTDIR / source).query("split=='test'").set_index("id")
    docs = documents().set_index("id").loc[test.index, "text"].tolist()
    schema = json.loads((OUTDIR / "choices.json").read_text())["topic"]
    labels = [json.dumps({k: v}) for k, v in schema["criteria"].items() if k != "Other"]
    assert len(labels) == 20 and len(test) == 160

    def force(model):
        model.decoys = []
        model.threshold = 0.0

    if backend == "laya":
        scores = (
            pd.read_csv(OUTDIR / "choice-scores.csv")
            .set_index("id")
            .loc[test.index]
            .to_numpy()
        )
        with LayaEntail(
            snapshot_download("convaiinnovations/laya-typed-decisions", revision=REV),
            batch_size=4,
        ) as m:
            force(m)
            pred = m.entail(docs, labels)
            assert len(m.questions["topic"]["criteria"]) == 20
    else:
        closed = dict(
            schema,
            criteria={k: v for k, v in schema["criteria"].items() if k != "Other"},
        )
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
            # Old schema must be served from the previous experiment's cache.
            with OpenAIDecisions(0) as old:
                scores = old.entail([text], labels)[0]["probs"]
            with OpenAIDecisions(limit) as m:
                force(m)
                try:
                    result = m.entail([text], labels)[0]
                except DecisionRefusal:
                    result = {"class": None}
                assert m.questions["topic"] == closed
            return scores, result

        pairs = list(dict.fromkeys(zip(docs, limits)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = dict(zip(pairs, pool.map(score, pairs), strict=True))
        scores, pred = zip(*(answers[key] for key in zip(docs, limits)), strict=True)
        scores = np.asarray(scores)
    test["masked"] = scores.argmax(axis=1)
    test["masked_tie"] = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) > 1
    test["closed"] = [p["class"] for p in pred]
    test["api_refusal"] = test.closed.isna()
    test["closed"] = test.closed.fillna(test.kmeans).astype(int)
    assert test.closed.between(0, 19).all()
    test = test[
        ["label", "kmeans", "default", "masked", "masked_tie", "closed", "api_refusal"]
    ]
    test.to_csv(OUTDIR / f"{backend}-forced.csv")
    for col in ["kmeans", "default", "masked", "closed"]:
        print(
            backend,
            col,
            ami(test.label, test[col]),
            int((test[col] != test.kmeans).sum()),
            flush=True,
        )
    print("masked ties", int(test.masked_tie.sum()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["laya", "luna"])
    p.add_argument("--budget-usd", type=Decimal, default=Decimal(0))
    args = p.parse_args()
    run(args.backend, args.budget_usd)
