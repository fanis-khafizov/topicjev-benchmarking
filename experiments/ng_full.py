"""Run independent stages on the official 20 Newsgroups bydate split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from laya.common import build_sequence
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from topicjev.entail import LayaEntail

from benchmark import CACHE, OUT, corpus
from ng_discovery import baseline, descriptions, embeddings
from verify_topics import DecisionRefusal, OpenAIDecisions, REV, RATE, request_body

DEST = OUT / "ng-full"
WORK = CACHE / "ng-full"
EXAMPLE_ID = "20news-bydate-test/rec.sport.hockey/53890"


def letters(split):
    df = corpus("20 Newsgroups", split, full=True)
    df["text"] = df.text.map(lambda s: " ".join(s.split()[:80]))
    assert len(df) == {"train": 11314, "test": 7532}[split] and df.id.is_unique
    return df


def save(frame, name):
    tmp = DEST / (name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(DEST / name)


def classical(train, test):
    df = test[["id", "label"]].copy()
    df["empty"] = test.text.eq("")
    for key, smooth in [("plsa", False), ("artm", True)]:
        df[key] = baseline(train.text, test.text, smooth=smooth, work_root=WORK)
        print(key, adjusted_mutual_info_score(df.label, df[key]), flush=True)
    save(df, "bigartm.csv")


def check_seeds(train, test):
    rows = []
    for seed in (11, 23, 47):
        for method, smooth in [("plsa", False), ("artm", True)]:
            if seed == 11:
                values = pd.read_csv(DEST / "bigartm.csv")[method]
            else:
                values = baseline(
                    train.text, test.text, smooth, seed=seed, work_root=WORK
                )
            rows.append(
                dict(
                    method=method,
                    seed=seed,
                    ami=adjusted_mutual_info_score(test.label, values),
                )
            )
    save(pd.DataFrame(rows), "seeds.csv")


def vectors(df):
    # Cache only complete batches, keyed by the ordered input (including the encoder revision).
    from ng_discovery import NOMIC

    parts = []
    for start in range(0, len(df), 1024):
        batch = df.text.iloc[start : start + 1024].tolist()
        key = hashlib.sha256((NOMIC + json.dumps(batch)).encode()).hexdigest()
        path = WORK / f"{key}.npy"
        if not path.exists():
            result = embeddings(batch)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, result)
            tmp.replace(path)
        parts.append(np.load(path))
        print("embeddings", min(start + 1024, len(df)), "/", len(df), flush=True)
    return np.vstack(parts)


def save_projection(fit, x, km):
    pca = PCA(n_components=2, svd_solver="full").fit(x)
    xy = pca.transform(x)
    save(
        pd.DataFrame(dict(id=fit.id, cluster=km.labels_, x=xy[:, 0], y=xy[:, 1])),
        "projection.csv",
    )
    info = dict(
        variance=pca.explained_variance_ratio_.tolist(),
        centers=pca.transform(km.cluster_centers_).tolist(),
    )
    (DEST / "projection-info.json").write_text(json.dumps(info, indent=2) + "\n")


def clusters(train, test):
    fit = train[train.text.ne("")].reset_index(drop=True)
    x = vectors(fit)
    km = KMeans(n_clusters=20, n_init=10, random_state=42).fit(x)
    default = int(np.bincount(km.labels_).argmax())
    df = test[["id", "label"]].copy()
    df["kmeans"] = default
    use = test.text.ne("")
    df.loc[use, "kmeans"] = km.predict(vectors(test[use]))
    save(df, "clusters.csv")
    save_projection(fit, x, km)
    centers = km.cluster_centers_
    reps = np.argsort(
        -(centers / np.linalg.norm(centers, axis=1, keepdims=True)) @ x.T, axis=1
    )[:, :5]
    topics = descriptions(fit, reps)
    tmp = DEST / "topics.json.tmp"
    tmp.write_text(json.dumps(topics, indent=2) + "\n")
    tmp.replace(DEST / "topics.json")


def topic_choices(topics):
    return {
        f"{t['title']} ({t['cluster']})": t["description"]
        + " (Keywords: "
        + ", ".join(t["keywords"])
        + ")"
        for t in topics
    }


def laya_predictions(model, labels, inputs):
    pred = {}
    for start in range(0, len(inputs), 128):
        batch = inputs[start : start + 128]
        key = hashlib.sha256(json.dumps([REV, labels, batch]).encode()).hexdigest()
        file = WORK / f"laya-{key}.json"
        if not file.exists():
            values = [int(p["class"]) for p in model.entail(batch, labels)]
            assert all(0 <= v < 20 for v in values)
            tmp = file.with_suffix(".tmp")
            tmp.write_text(json.dumps(values))
            tmp.replace(file)
        pred.update(zip(batch, json.loads(file.read_text()), strict=True))
        print("Laya", len(pred), "/", len(inputs), flush=True)
    return pred


def hockey_example(model, sample):
    choices = {
        "Hockey": "Ice hockey, players and teams.",
        "Baseball": "Baseball, players and teams.",
    }
    labels = [json.dumps({name: desc}) for name, desc in choices.items()]
    result = model.entail([sample], labels)[0]
    record = dict(
        id=EXAMPLE_ID,
        question=model.questions["topic"],
        probabilities=dict(zip(choices, result["probs"], strict=True)),
    )
    (DEST / "laya-example.json").write_text(json.dumps(record, indent=2) + "\n")


def known_categories(train, test):
    # Use the corpus taxonomy only; the model receives no example-to-category pairs.
    names = sorted(train.label.unique().tolist())
    assert len(names) == 20
    labels = [json.dumps({name: None}) for name in names]
    (DEST / "known-choices.json").write_text(json.dumps(names, indent=2) + "\n")
    inputs = test.text.drop_duplicates().tolist()
    path = snapshot_download("convaiinnovations/laya-typed-decisions", revision=REV)
    with LayaEntail(path, batch_size=4, threshold=0) as model:
        model.decoys = []
        pred = laya_predictions(model, labels, inputs)
    frame = test[["id", "label"]].copy()
    frame["empty"] = test.text.eq("")
    frame["laya_known"] = [names[pred[t]] for t in test.text]
    save(frame, "laya_known.csv")
    print(
        "Laya, dataset categories",
        adjusted_mutual_info_score(frame.label, frame.laya_known),
        flush=True,
    )


def decisions(test, backend, budget):
    topics = json.loads((DEST / "topics.json").read_text())
    df = pd.read_csv(DEST / "clusters.csv")
    assert df.id.equals(test.id) and len(topics) == 20
    choices = topic_choices(topics)
    labels = [json.dumps({k: v}) for k, v in choices.items()]
    use = test.text.ne("")
    inputs = test.loc[use, "text"].drop_duplicates().tolist()
    pred = {}
    if backend == "laya":
        path = snapshot_download("convaiinnovations/laya-typed-decisions", revision=REV)
        with LayaEntail(path, batch_size=4, threshold=0) as m:
            m.decoys = []
            m.prep_pairs([], labels)
            q = m.model._to_internal(m.questions["topic"])
            tok = m.model.tok
            ids, markers = build_sequence(
                tok,
                {"text": ""},
                q,
                m.model.cfg["max_len"],
                m.model.cfg["head_max_len"],
            )
            ends = markers[1:] + [ids.index(tok.sep_token_id, markers[-1])]
            encoded_options = [
                ids[start + 1 : end] for start, end in zip(markers, ends, strict=True)
            ]
            assert len(encoded_options) == len(choices)
            save(
                pd.DataFrame(
                    {
                        "option": list(choices),
                        "retained_tokens": [len(opt) for opt in encoded_options],
                        "encoded_prefix": [tok.decode(opt) for opt in encoded_options],
                    }
                ),
                "laya-options.csv",
            )
            pred = laya_predictions(m, labels, inputs)
            sample = test.set_index("id").at[EXAMPLE_ID, "text"]
            scores = m.entail([sample], labels)[0]["probs"]
            assert int(np.argmax(scores)) == pred[sample]
            save(
                pd.DataFrame(
                    [
                        {
                            "id": EXAMPLE_ID,
                            **{str(i): p for i, p in enumerate(scores)},
                        }
                    ]
                ),
                "choice-scores.csv",
            )
            hockey_example(m, sample)
    else:
        m = OpenAIDecisions(0)
        m.decoys = []
        m.prep_pairs([], labels)
        question = m.questions["topic"]
        jobs = []
        total = Decimal(0)
        for text in inputs:
            raw = json.dumps(request_body(text, question), ensure_ascii=False).encode()
            path = m.store / (hashlib.sha256(raw).hexdigest() + ".json")
            reserve = RATE * (len(raw) + 2048)
            if not path.exists():
                total += reserve
            jobs.append((text, reserve, path))
        print("Uncached USD reservation", total, "unique texts", len(jobs), flush=True)
        if total > budget:
            raise ValueError("Increase budget to cover reservation")

        def score(job):
            text, reserve, path = job
            with OpenAIDecisions(reserve) as model:
                model.decoys = []
                model.threshold = 0
                try:
                    value = int(model.entail([text], labels)[0]["class"])
                except DecisionRefusal:
                    value = None
            usage = json.loads(path.read_text())["usage"]
            return text, value, usage["input_tokens"]

        tokens = 0
        with ThreadPoolExecutor(max_workers=12) as pool:
            for start in range(0, len(jobs), 120):
                for text, value, n in pool.map(score, jobs[start : start + 120]):
                    pred[text] = value
                    tokens += n
                print("Luna", len(pred), "/", len(inputs), flush=True)
        (DEST / "usage.json").write_text(
            json.dumps(dict(unique_requests=len(inputs), input_tokens=tokens), indent=2)
            + "\n"
        )
    df["empty"] = ~use
    df["refusal"] = [bool(t and pred[t] is None) for t in test.text]
    df[backend] = [
        pred[t] if t and pred[t] is not None else base
        for t, base in zip(test.text, df.kmeans, strict=True)
    ]
    assert df[backend].between(0, 19).all()
    save(df, f"{backend}.csv")
    print(backend, adjusted_mutual_info_score(df.label, df[backend]), flush=True)


def summarize():
    df = pd.read_csv(DEST / "bigartm.csv")
    for key in ["kmeans", "laya", "luna", "laya_known"]:
        src = pd.read_csv(DEST / ("clusters.csv" if key == "kmeans" else key + ".csv"))
        assert df.id.equals(src.id) and df.label.equals(src.label)
        df[key] = src[key]
        if key == "luna":
            df["api_refusal"] = src.refusal
    assert len(df) == 7532 and df.id.is_unique
    save(df, "assignments.csv")
    rows = [
        dict(method=m, n=len(df), ami=adjusted_mutual_info_score(df.label, df[m]))
        for m in ["plsa", "artm", "kmeans", "laya", "luna", "laya_known"]
    ]
    save(pd.DataFrame(rows), "comparison.csv")
    print(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["classical", "seeds", "clusters", "laya", "luna", "known", "summary"],
    )
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal(0))
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    if args.stage == "summary":
        summarize()
    elif args.stage in ["laya", "luna"]:
        decisions(letters("test"), args.stage, args.budget_usd)
    else:
        {
            "classical": classical,
            "seeds": check_seeds,
            "clusters": clusters,
            "known": known_categories,
        }[args.stage](letters("train"), letters("test"))
