"""Evaluate Decisions on the saved 20NG split and exact Qwen choice schema."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import fcntl
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from benchmark import CACHE
from ng_discovery import documents, OUTDIR, THRESHOLDS
from verify_topics import OpenAIDecisions, RATE, request_body


def run(budget):
    schema = json.loads((OUTDIR / "choices.json").read_text())["topic"]
    labels = [json.dumps({k: v}) for k, v in schema["criteria"].items() if k != "Other"]
    rows = pd.read_csv(OUTDIR / "assignments.csv")[
        ["id", "split", "label", "kmeans"]
    ].merge(documents()[["id", "text"]], on="id", validate="one_to_one")
    assert len(rows) == 360 and len(labels) == 20
    store = CACHE / "decisions"

    def packet(text):
        raw = json.dumps(request_body(text, schema), ensure_ascii=False).encode()
        path = store / (hashlib.sha256(raw).hexdigest() + ".json")
        reserve = RATE * (len(raw) + 2048)
        return text, path, reserve

    jobs = [packet(text) for text in rows.text]
    total = sum(r for _, p, r in jobs if not p.exists())
    print(f"New request reservation: USD {total:.5f}", flush=True)
    if total > budget:
        raise ValueError("Requested run exceeds budget.")

    def score(job):
        text, path, reserve = job
        with OpenAIDecisions(reserve) as model:
            pred = model.entail([text], labels)[0]
            assert model.questions["topic"] == schema
        usage = json.loads(path.read_text()).get("usage", {})
        return (
            pred["class"],
            pred["prob"],
            usage.get("input_tokens"),
            usage.get("output_tokens"),
        )

    # Identical requests share a cache key; dispatch each unique input once.
    distinct = {j[0]: j for j in jobs}
    with ThreadPoolExecutor(max_workers=4) as pool:
        scored = dict(zip(distinct, pool.map(score, distinct.values()), strict=True))
    rows[["decision", "confidence", "input_tokens", "output_tokens"]] = [
        scored[t] for t in rows.text
    ]
    rows = rows.drop(columns="text")
    val = rows.split.eq("validation")

    def route(t):
        return np.where(
            (rows.decision < 20) & (rows.confidence > t), rows.decision, rows.kmeans
        )

    grid = pd.DataFrame(
        [
            dict(
                threshold=t,
                ami=adjusted_mutual_info_score(rows.label[val], route(t)[val]),
                changed=sum(route(t)[val] != rows.kmeans[val]),
            )
            for t in THRESHOLDS
        ]
    )
    best = grid.sort_values(["ami", "threshold"], ascending=False).iloc[0].threshold
    for name, t in [("default", 0.1), ("selected", best)]:
        rows[name] = route(t)
    test = rows[~val]
    comparison = pd.DataFrame(
        [
            dict(
                method=name,
                threshold=t,
                ami=adjusted_mutual_info_score(test.label, test[name]),
                changes=sum(test[name] != test.kmeans),
            )
            for name, t in [("default", 0.1), ("selected", best)]
        ]
    )
    for name, df in [
        ("assignments", rows),
        ("tuning", grid),
        ("comparison", comparison),
    ]:
        df.to_csv(OUTDIR / f"openai-{name}.csv", index=False)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budget-usd", type=Decimal, required=True)
    with (CACHE / "verification.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(p.parse_args().budget_usd)
