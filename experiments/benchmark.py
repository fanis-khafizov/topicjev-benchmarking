"""Fixed-split comparison using the official BigARTM command-line program."""

import argparse
import csv
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.datasets._twenty_newsgroups import (
    strip_newsgroup_footer,
    strip_newsgroup_header,
    strip_newsgroup_quoting,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
OUT = ROOT / "results"
AG = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
)
FILES = {
    "ag-train.csv": (
        AG + "train.csv",
        "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e",
    ),
    "ag-test.csv": (
        AG + "test.csv",
        "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699",
    ),
    "20news-bydate.tar.gz": (
        "https://ndownloader.figshare.com/files/5975967",
        "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610",
    ),
}
SEEDS = [11, 23, 47]
NAMES = ["AG News", "20 Newsgroups"]


def download(name):
    url, digest = FILES[name]
    path = CACHE / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".download")
        urllib.request.urlretrieve(url, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != digest:
            raise ValueError(name)
        tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(name)
    return path


def corpus(name, split, *, full=False):
    rows = []
    if name == "AG News":
        with download(f"ag-{split}.csv").open() as file:
            for i, (label, title, body) in enumerate(csv.reader(file)):
                text = html.unescape(
                    re.sub(r"(?<!&)#(\d+);", r"&#\1;", title + " " + body)
                )
                rows.append((str(i), label, text))
    else:
        with tarfile.open(download("20news-bydate.tar.gz"), mode="r|gz") as tar:
            for item in tar:
                if item.isfile() and item.name.startswith(
                    "20news-bydate-" + split + "/"
                ):
                    text = tar.extractfile(item).read().decode("latin1")
                    text = strip_newsgroup_footer(
                        strip_newsgroup_quoting(strip_newsgroup_header(text))
                    )
                    # Labels only supply sampling strata and external evaluation.
                    rows.append((item.name, item.name.split("/")[1], text))
    rows.sort(key=lambda r: r[0])
    df = pd.DataFrame(rows, columns=["id", "label", "text"])
    if full:
        return df
    df = df[df.text.str.count(r"\b[a-zA-Z]{3,}\b") >= 5]
    n = 4000 if split == "train" else 800
    rng = np.random.default_rng(2026 if split == "train" else 2027)
    groups = [
        g.iloc[rng.choice(len(g), n // df.label.nunique(), replace=False)]
        for _, g in df.groupby("label")
    ]
    return pd.concat(groups).sample(frac=1, random_state=2026).reset_index(drop=True)


def vw(path, x, vocab):
    with path.open("w") as f:
        for i in range(x.shape[0]):
            row = x.getrow(i)
            terms = " ".join(
                f"{vocab[j]}:{v}" for j, v in zip(row.indices, row.data) if v
            )
            f.write(f"d{i} | {terms}\n")


def cli(exe, wd, args):
    with (wd / "run.log").open("a") as log:
        subprocess.run(
            [exe, "--force", "--threads", "1", "--score-level", "0", *map(str, args)],
            cwd=wd,
            check=True,
            stdout=log,
            stderr=log,
        )


def theta(wd, file, cols, n):
    table = (
        pd.read_csv(wd / file, sep=";")
        .set_index("title")
        .loc[[f"d{i}" for i in range(n)], cols]
    )
    a = table.to_numpy()
    assert (
        np.isfinite(a).all() and (a >= 0).all() and np.allclose(a.sum(1), 1, atol=2e-5)
    )
    return a


def completion(theta, phi, held, unigram):
    coo = held.tocoo()
    p = np.einsum("ij,ij->i", theta[coo.row], phi[coo.col])
    return np.exp(
        -np.average(np.log(0.99 * p + 0.01 * unigram[coo.col]), weights=coo.data)
    )


def coherence(x, top):
    binary = (x > 0).astype(float)
    words = np.unique(top)
    probs = (binary[:, words].T @ binary[:, words]).toarray() / x.shape[0]
    scores = []
    for topic in top:
        for a, b in itertools.combinations(np.searchsorted(words, topic), 2):
            joint = probs[a, b]
            scores.append(
                np.log(joint / (probs[a, a] * probs[b, b])) / -np.log(joint)
                if joint
                else -1.0
            )
    return np.mean(scores)


def run(exe, name):
    tag = "ag" if name == NAMES[0] else "ng"
    wd = CACHE / tag
    wd.mkdir(parents=True, exist_ok=True)
    train, test = [corpus(name, split) for split in ("train", "test")]
    vec = CountVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-z]{3,}\b",
        min_df=4,
        max_df=0.6,
        max_features=10000,
    )
    x = vec.fit_transform(train.text)
    good = np.asarray(x.sum(1)).ravel() > 0
    x = x[good].tocsr()
    v = vec.transform(test.text)
    good = np.asarray(v.sum(1)).ravel() >= 2
    test, v = test[good].reset_index(drop=True), v[good].tocsr()
    rng = np.random.default_rng(932)
    obs = v.copy()
    obs.data = rng.binomial(obs.data, 0.5)
    held = v - obs
    eligible = (np.asarray(obs.sum(1)).ravel() > 0) & (
        np.asarray(held.sum(1)).ravel() > 0
    )
    obs, held = obs[eligible].tocsr(), held[eligible].tocsr()
    vocab = vec.get_feature_names_out()
    for file, matrix in [("train", x), ("test", v), ("obs", obs)]:
        vw(wd / (file + ".vw"), matrix, vocab)
    test.to_json(wd / "test.jsonl", orient="records", lines=True)
    unigram = np.asarray(x.sum(0)).ravel() / x.sum()
    hc = held.tocoo()
    null_pp = np.exp(-np.average(np.log(unigram[hc.col]), weights=hc.data))
    k = test.label.nunique()
    records, topics = [], []
    for method, seed in itertools.product(["PLSA", "smooth", "decor"], SEEDS):
        print(name, method, seed, flush=True)
        folder = wd / f"{method}-{seed}"
        folder.mkdir(exist_ok=True)
        common = ["-t", k, "--num-document-passes", 10]
        regs = (
            ["--regularizer", "0.1 SmoothTheta", "0.01 SmoothPhi"]
            if method in ("smooth", "decor")
            else []
        )
        if method == "decor":
            regs.append("5000 Decorrelation")
        cli(
            exe,
            folder,
            [
                *common,
                "-c",
                wd / "train.vw",
                "-p",
                25,
                "--rand-seed",
                seed,
                "--save-model",
                "model",
                "--write-model-readable",
                "phi.csv",
                *regs,
            ],
        )
        for part in ("test", "obs"):
            cli(
                exe,
                folder,
                [
                    *common,
                    "-c",
                    wd / (part + ".vw"),
                    "--load-model",
                    "model",
                    "--write-predictions",
                    part + ".csv",
                    *regs,
                ],
            )
        frame = (
            pd.read_csv(folder / "phi.csv", sep=";", keep_default_na=False)
            .set_index("token")
            .loc[vocab]
        )
        cols = list(frame.columns[1:])
        phi = frame[cols].to_numpy()
        assert (phi >= 0).all() and np.allclose(phi.sum(0), 1, atol=1e-4)
        top = np.argsort(-phi.T, axis=1)[:, :10]
        full = theta(folder, "test.csv", cols, len(test))
        pred = full.argmax(1)
        predfile = OUT / f"{tag}-{method}-{seed}.csv"
        pd.DataFrame({"id": test.id, "label": test.label, "topic": pred}).to_csv(
            predfile, index=False
        )
        words = [vocab[idx].tolist() for idx in top]
        topics.extend(
            dict(dataset=tag, method=method, seed=seed, topic=i, words=w)
            for i, w in enumerate(words)
        )
        ami = adjusted_mutual_info_score(test.label, pred)
        shuffled = adjusted_mutual_info_score(
            test.label, np.random.default_rng(123).permutation(pred)
        )
        records.append(
            dict(
                dataset=tag,
                method=method,
                seed=seed,
                k=k,
                train=x.shape[0],
                test=len(test),
                vocab=len(vocab),
                completion_docs=len(obs.indptr) - 1,
                completion_tokens=int(held.sum()),
                ppl=completion(
                    theta(folder, "obs.csv", cols, obs.shape[0]), phi, held, unigram
                ),
                unigram_ppl=null_pp,
                ami=ami,
                permuted_ami=shuffled,
                npmi=coherence(x, top),
                diversity=len(np.unique(top)) / top.size,
                active=len(set(pred)),
            )
        )
    return records, topics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigartm", default=os.getenv("BIGARTM_BIN", "bigartm"))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    records, topics = [], []
    for name in NAMES:
        r, t = run(args.bigartm, name)
        records.extend(r)
        topics.extend(t)
        pd.DataFrame(records).to_csv(OUT / "metrics.csv", index=False)
        (OUT / "topics.json").write_text(json.dumps(topics, indent=2))
