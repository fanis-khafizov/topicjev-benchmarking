"""Compare single-topic assignments by predictive likelihood on unseen word counts."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score
from huggingface_hub import snapshot_download
from topicjev.entail import LayaEntail

from benchmark import CACHE, OUT, cli, corpus, theta, vw
from verify_topics import REV

DEST = OUT / "paired"
K = 20
N = 160
HASHES = {
    "docword.kos.txt.gz": "6439c39f2fccb4876563e4b4c9628974d8ccbeada43d661292a5024911389e30",
    "docword.nips.txt.gz": "d45f9f7ceeadf118117e14346f018273007b765593a4dbe02915d37a1faa58b7",
    "vocab.kos.txt": "623b7b969793b32c4ea7d1ae8cd2e9ed99038bf51628f33f52349f9485aeab8e",
    "vocab.nips.txt": "be3f6337e3406ddc24837118e203da57c4c898ad5f6c9280908cd08a2609cf91",
}
METHODS = {
    "PLSA": [],
    "smooth": ["0.1 SmoothTheta", "0.01 SmoothPhi"],
    "decor": ["0.1 SmoothTheta", "0.01 SmoothPhi", "5000 Decorrelation"],
}


def uci_file(name):
    path = CACHE / "uci" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".download")
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"
            + name,
            tmp,
        )
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != HASHES[name]:
            raise ValueError(name)
        tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError(name)
    return path


def matrices(tag):
    if tag == "20NG":
        train, test = [corpus("20 Newsgroups", s) for s in ("train", "test")]
        vec = CountVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-z]{3,}\b",
            min_df=4,
            max_df=0.6,
            max_features=10000,
        )
        return (
            vec.fit_transform(train.text),
            vec.transform(test.text),
            vec.get_feature_names_out(),
            test.id.to_numpy(),
            test.label.to_numpy(),
        )
    vocab = np.array(uci_file(f"vocab.{tag}.txt").read_text().splitlines())
    with gzip.open(uci_file(f"docword.{tag}.txt.gz"), "rt") as f:
        d, v, nnz = [int(next(f)) for _ in range(3)]
        doc, word, count = np.loadtxt(f, dtype=int).T
    assert len(doc) == nnz and len(vocab) == v
    all_docs = csr_matrix((count, (doc - 1, word - 1)), shape=(d, v))
    ids = np.random.default_rng(2026).permutation(d)
    test, train = np.split(ids, [d // 5])
    x = all_docs[train]
    keep = np.asarray(x.sum(0)).ravel() > 0
    return x[:, keep], all_docs[test][:, keep], vocab[keep], test + 1, None


def prepare(tag, folder):
    x, v, vocab, ids, labels = matrices(tag)
    x = x[np.asarray(x.sum(1)).ravel() > 0].tocsr()
    v = v.tocsr()
    seen = v.copy()
    seen.data = np.random.default_rng(932).binomial(seen.data, 0.5)
    seen.eliminate_zeros()
    hidden = (v - seen).tocsr()
    eligible = np.flatnonzero(
        (np.asarray(seen.sum(1)).ravel() > 0) & (np.asarray(hidden.sum(1)).ravel() > 0)
    )
    rng = np.random.default_rng(392)
    if labels is None:
        pick = rng.choice(eligible, N, replace=False)
    else:
        pick = np.concatenate(
            [
                rng.choice(eligible[labels[eligible] == label], N // K, replace=False)
                for label in np.unique(labels)
            ]
        )
    seen = seen[pick].tocsr()
    texts = []
    for i in range(N):
        start, end = seen.indptr[i : i + 2]
        indices = seen.indices[start:end]
        counts = seen.data[start:end]
        # Same retained term counts for both methods. Lexical tie-breaking is fixed.
        order = np.lexsort((vocab[indices], -counts))
        texts.append(
            ", ".join(f"{vocab[indices[j]]} ({counts[j]})" for j in order[:24])
        )
        counts[order[24:]] = 0
    seen.eliminate_zeros()
    vw(folder / "train.vw", x, vocab)
    vw(folder / "seen.vw", seen, vocab)
    return (
        x,
        vocab,
        hidden[pick],
        ids[pick],
        None if labels is None else labels[pick],
        texts,
    )


def losses(hidden, distributions):
    coo = hidden.tocoo()
    return np.bincount(
        coo.row,
        weights=-coo.data * np.log(distributions[coo.row, coo.col]),
        minlength=N,
    )


def evaluate(tag, exe, model):
    folder = CACHE / "paired" / tag
    folder.mkdir(parents=True, exist_ok=True)
    x, vocab, hidden, ids, labels, texts = prepare(tag, folder)
    prior = np.asarray(x.sum(0)).ravel() / x.sum()
    n = np.asarray(hidden.sum(1)).ravel()
    rows = []
    for method, reg in METHODS.items():
        work = folder / method
        work.mkdir(exist_ok=True)
        params = ["-t", K, "--num-document-passes", 10]
        if reg:
            params += ["--regularizer", *reg]
        cli(
            exe,
            work,
            [
                *params,
                "-c",
                folder / "train.vw",
                "-p",
                25,
                "--rand-seed",
                11,
                "--save-model",
                "model",
                "--write-model-readable",
                "phi.csv",
            ],
        )
        cli(
            exe,
            work,
            [
                *params,
                "-c",
                folder / "seen.vw",
                "--load-model",
                "model",
                "--write-predictions",
                "theta.csv",
            ],
        )
        frame = (
            pd.read_csv(work / "phi.csv", sep=";", keep_default_na=False)
            .set_index("token")
            .loc[vocab]
        )
        phi = frame.iloc[:, 1:].to_numpy()
        assert np.allclose(phi.sum(0), 1, atol=0.0001) and (phi >= 0).all()
        th = theta(work, "theta.csv", list(frame.columns[1:]), N)

        def score(p):
            return losses(hidden, 0.99 * p + 0.01 * prior)

        def ppl(v):
            return np.exp(v.sum() / n.sum())

        base = th.argmax(1)
        base_loss = score(phi[:, base].T)
        rows.append(
            dict(
                dataset=tag,
                method=method,
                train=x.shape[0],
                vocab=len(vocab),
                n=N,
                hidden_tokens=n.sum(),
                unigram=ppl(losses(hidden, np.broadcast_to(prior, (N, len(vocab))))),
                mixture=ppl(score(th @ phi.T)),
                hard=ppl(base_loss),
            )
        )
        if method != "smooth":
            continue
        words = [vocab[t].tolist() for t in np.argsort(-phi.T, axis=1)[:, :10]]
        choices = [
            json.dumps({" / ".join(w[:2]) + f" ({j})": ", ".join(w)})
            for j, w in enumerate(words)
        ]
        pred = model.entail(texts, choices)
        decision = np.array([a["class"] for a in pred])
        accepted = decision < K
        routed = np.where(accepted, decision, base)
        new_loss = score(phi[:, routed].T)
        delta = new_loss - base_loss
        # Resample paired documents, retaining the token-weighted definition of PPL.
        ix = np.random.default_rng(1405).integers(0, N, size=(5000, N))
        pct = 100 * np.expm1(delta[ix].sum(1) / n[ix].sum(1))
        before = ppl(base_loss)
        after = ppl(new_loss)
        detail = pd.DataFrame(
            dict(
                id=ids,
                tokens=n,
                bigartm=base,
                decision=decision,
                routed=routed,
                baseline_nll=base_loss,
                hybrid_nll=new_loss,
            )
        )
        if labels is not None:
            detail["label"] = labels
        detail.to_csv(DEST / f"{tag}-documents.csv", index=False)
        (DEST / f"{tag}-topics.json").write_text(json.dumps(words, indent=2))
        row = dict(
            dataset=tag,
            n=N,
            accepted=accepted.sum(),
            changed=(routed != base).sum(),
            base=before,
            hybrid=after,
            change_pct=100 * (after / before - 1),
            ci_low=np.quantile(pct, 0.025),
            ci_high=np.quantile(pct, 0.975),
        )
        if labels is not None:
            row.update(
                ami_before=adjusted_mutual_info_score(labels, base),
                ami_after=adjusted_mutual_info_score(labels, routed),
            )
    return rows, row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigartm", type=Path, default=CACHE / "bin" / "bigartm")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    ckpt = snapshot_download("convaiinnovations/laya-typed-decisions", revision=REV)
    baseline, paired = [], []
    with LayaEntail(ckpt, batch_size=4) as model:
        for tag in ("20NG", "nips", "kos"):
            print(tag, flush=True)
            runs, comparison = evaluate(tag, str(args.bigartm.resolve()), model)
            baseline.extend(runs)
            paired.append(comparison)
            pd.DataFrame(baseline).to_csv(DEST / "baselines.csv", index=False)
            pd.DataFrame(paired).to_csv(DEST / "comparison.csv", index=False)
            print(comparison, flush=True)
