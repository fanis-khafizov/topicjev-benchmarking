"""Paired evaluation of TopicJev backends on a fixed, nested test sample."""

import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
import os
import urllib.error
import urllib.request
import time

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from topicjev.entail import JevEntail, LayaEntail

from benchmark import ROOT, CACHE, OUT

MODEL = "gpt-6-luna"
REV = "57d0994ab292b9dc1737f64ce31addae25a3c127"
RATE = Decimal("0.10") / 1_000_000


def request_body(text, question):
    return dict(
        model=MODEL,
        input=text,
        questions=[
            dict(
                type="choice",
                name="topic",
                instructions=question["instructions"],
                choices=[
                    dict(value=k, description=v)
                    for k, v in question["criteria"].items()
                ],
            )
        ],
    )


class DecisionRefusal(ValueError):
    """The API returned an explicit refusal instead of a choice."""


class OpenAIDecisions(JevEntail):
    """Keep upstream topic descriptions, Other option, normalization and threshold."""

    def __init__(self, budget):
        super().__init__()
        self.budget = Decimal(str(budget))
        self.store = CACHE / "decisions"
        self.store.mkdir(exist_ok=True)
        self.spent_est = Decimal(0)

    def load_model(self):
        load_dotenv(ROOT / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY locally before a paid run.")

    def batch_score(self, pairs):
        result = []
        for pair in pairs:
            obj = request_body(pair.inpt, self.questions["topic"])
            raw = json.dumps(obj, ensure_ascii=False).encode()
            dest = self.store / (hashlib.sha256(raw).hexdigest() + ".json")
            pending = dest.with_suffix(".pending")
            if not dest.exists():
                # Conservative reservation at the documented standard short-context rate.
                reserve = RATE * (len(raw) + 2048)
                if self.spent_est + reserve > self.budget:
                    raise RuntimeError(
                        "Per-run budget reached; cached responses are resumable."
                    )
                if pending.exists():
                    raise RuntimeError(
                        f"Uncertain previous request: inspect usage before manually clearing {pending}"
                    )
                pending.touch(exist_ok=False)
                self.spent_est += reserve
                req = urllib.request.Request(
                    "https://api.openai.com/v1/decisions",
                    raw,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                    },
                )
                for retry in range(5):
                    try:
                        with urllib.request.urlopen(req, timeout=90) as response:
                            data = response.read()
                        break
                    except urllib.error.HTTPError as err:
                        if err.code == 429 and retry < 4:
                            time.sleep(
                                min(60, float(err.headers.get("Retry-After", 2**retry)))
                            )
                            continue
                        if 400 <= err.code < 500:
                            pending.unlink(missing_ok=True)
                        raise
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(dest)
                pending.unlink()
            resp = json.loads(dest.read_bytes())
            (answer,) = resp["answers"]
            if answer.get("type") == "refusal" and answer.get("name") == "topic":
                raise DecisionRefusal("OpenAI Decisions returned a refusal")
            if answer["type"] != "choice" or answer["name"] != "topic":
                raise ValueError(
                    "Refused or incomplete decision: inspect cached response"
                )
            distribution = {
                item["value"]: item["probability"] for item in answer["probabilities"]
            }
            if set(distribution) != set(self.questions["topic"]["criteria"]):
                raise ValueError("Unexpected choices")
            probs = np.array(
                [distribution[k] for k in self.questions["topic"]["criteria"]]
            )
            if not np.all(np.isfinite(probs) & (probs >= 0)) or not np.isclose(
                probs.sum(), 1, atol=1e-4
            ):
                raise ValueError("Malformed probability distribution")
            result.append(np.log(probs.clip(1e-12)).tolist())
        return result


def sample(tag, size):
    test = pd.read_json(
        CACHE / tag / "test.jsonl", lines=True, dtype={"id": str, "label": str}
    )
    base = pd.read_csv(OUT / f"{tag}-smooth-11.csv", dtype={"id": str, "label": str})
    test = test.merge(base, on=["id", "label"], validate="one_to_one")
    # Round-robin shuffled within each stratum: a prefix is also the pilot.
    test = test.sample(frac=1, random_state=392)
    test["rank"] = test.groupby("label").cumcount()
    return test.sort_values(["rank", "label"]).head(size)


def evaluate(backend, size, budget):
    topics = json.loads((OUT / "topics.json").read_text())
    if backend == "laya":
        from huggingface_hub import snapshot_download

        path = snapshot_download("convaiinnovations/laya-typed-decisions", revision=REV)
        model = LayaEntail(path, batch_size=4)
    else:
        model = OpenAIDecisions(budget)
    if backend != "estimate":
        model.load_model()
    summary = []
    try:
        for tag in ("ag", "ng"):
            rows = sample(tag, size)
            words = [
                t["words"]
                for t in topics
                if t["dataset"] == tag and t["seed"] == 11 and t["method"] == "smooth"
            ]
            labels = [
                json.dumps({" / ".join(w[:2]) + f" ({i})": ", ".join(w[:10])})
                for i, w in enumerate(words)
            ]
            docs = [t[:2400] for t in rows.text]
            if backend == "estimate":
                model.prep_pairs([], labels + model.decoys)
                bodies = [
                    json.dumps(
                        request_body(t, model.questions["topic"]), ensure_ascii=False
                    ).encode()
                    for t in docs
                ]
                print(
                    tag,
                    len(bodies),
                    "requests; conservative USD reservation:",
                    RATE * sum(len(b) + 2048 for b in bodies),
                )
                continue
            result = model.entail(docs, labels)
            pred = np.array([r["class"] for r in result])
            accepted = pred < len(labels)
            table = rows[["id", "label", "topic"]].copy()
            table["decision"] = pred
            table["max_prob"] = [r["prob"] for r in result]
            table["probs"] = [json.dumps(r["probs"]) for r in result]
            table.to_csv(OUT / f"{tag}-{backend}-{size}.csv", index=False)
            summary.append(
                dict(
                    dataset=tag,
                    n=len(rows),
                    backend=backend,
                    coverage=accepted.mean(),
                    ami_base=adjusted_mutual_info_score(rows.label, rows.topic),
                    ami_all=adjusted_mutual_info_score(rows.label, pred),
                    agreement=(pred == rows.topic).mean(),
                    ami_kept=adjusted_mutual_info_score(
                        rows.label[accepted], pred[accepted]
                    )
                    if accepted.sum() > 1
                    else None,
                )
            )
        if summary:
            pd.DataFrame(summary).to_csv(
                OUT / f"summary-{backend}-{size}.csv", index=False
            )
            print(pd.DataFrame(summary).to_string(index=False))
    finally:
        model.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["estimate", "openai", "laya"])
    p.add_argument("--size", type=int, default=100, help="documents per benchmark")
    p.add_argument("--budget-usd", type=Decimal, default=Decimal(0))
    a = p.parse_args()
    if not 1 <= a.size <= 800 or a.budget_usd < 0:
        p.error("invalid size or budget")
    with (CACHE / "verification.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluate(a.backend, a.size, a.budget_usd)
