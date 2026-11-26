"""Inductive topic discovery and routing on cleaned 20 Newsgroups letters."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import Tokenizer
from topicjev.entail import LayaEntail
from topicjev.prompts import load_prompt

from benchmark import CACHE, OUT, cli, corpus, theta, vw

OUTDIR = OUT / "ng-discovery"
WORK = CACHE / "ng-discovery"
NOMIC = "e9b6763023c676ca8431644204f50c2b100d9aab"
QWEN = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"


def documents():
    train = corpus("20 Newsgroups", "train")
    groups = [g.sample(n=60, random_state=228) for _, g in train.groupby("label")]
    fit = pd.concat([g.iloc[:50] for g in groups])
    val = pd.concat([g.iloc[50:] for g in groups])
    ids = pd.read_csv(OUT / "paired/20NG-documents.csv").id
    test = corpus("20 Newsgroups", "test").set_index("id").loc[ids].reset_index()
    df = pd.concat(
        [
            fit.assign(split="train"),
            val.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    assert df.id.is_unique
    df["text"] = df.text.map(lambda s: " ".join(s.split()[:80]))
    return df


def embeddings(texts):
    weights = Path(
        snapshot_download(
            "nomic-ai/nomic-embed-text-v1.5",
            revision=NOMIC,
            allow_patterns=["tokenizer.json", "onnx/model_int8.onnx"],
        )
    )
    tok = Tokenizer.from_file(str(weights / "tokenizer.json"))
    tok.enable_truncation(256)
    tok.enable_padding()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 6
    sess = ort.InferenceSession(
        str(weights / "onnx/model_int8.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    vectors = []
    for start in range(0, len(texts), 8):
        batch = tok.encode_batch(["clustering: " + t for t in texts[start : start + 8]])
        args = {
            key: np.array([getattr(e, attr) for e in batch], dtype=np.int64)
            for key, attr in [
                ("input_ids", "ids"),
                ("attention_mask", "attention_mask"),
                ("token_type_ids", "type_ids"),
            ]
        }
        hidden = sess.run(None, {i.name: args[i.name] for i in sess.get_inputs()})[0]
        mask = args["attention_mask"][:, :, None]
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        vectors.extend(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
    return np.asarray(vectors, dtype=np.float32)


def descriptions(df, reps):
    from mlx_lm import load, generate

    weights = snapshot_download("mlx-community/Qwen3.5-4B-4bit", revision=QWEN)
    model, tok = load(weights)

    def ask(template, variables, keys):
        prompt = load_prompt(template).format(**variables)
        for attempt in range(3):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            answer = generate(model, tok, prompt=chat, max_tokens=512, verbose=False)
            try:
                obj = json.JSONDecoder().raw_decode(answer[answer.index("{") :])[0]
                if all(
                    isinstance(obj.get(k), str) and obj[k].strip()
                    for k in keys
                    if k != "keywords"
                ) and (
                    "keywords" not in keys
                    or (
                        isinstance(obj.get("keywords"), list)
                        and len(obj["keywords"]) == 5
                        and all(isinstance(w, str) for w in obj["keywords"])
                    )
                ):
                    return obj
            except (ValueError, TypeError):
                pass
            prompt += "\nReturn a valid JSON object with keys " + ", ".join(keys) + "."
        raise ValueError(answer)

    topics = []
    for i, indices in enumerate(reps):
        group = df.iloc[indices]
        reduced = ask(
            "reduce",
            {"text": json.dumps({"facts": group.text.tolist()})},
            ["title", "summary", "keywords"],
        )
        named = ask(
            "classes",
            {
                "text": json.dumps(reduced),
                "used_titles": json.dumps([t["title"] for t in topics]),
            },
            ["title", "desc"],
        )
        topics.append(
            dict(
                cluster=i,
                representatives=group.id.tolist(),
                keywords=reduced["keywords"],
                summary=reduced["summary"],
                title=named["title"],
                description=named["desc"],
            )
        )
        print(i, named["title"], flush=True)
    return topics


def baseline(fit, docs, smooth, seed=11):
    work = WORK / f"{'smooth' if smooth else 'plsa'}-{seed}"
    work.mkdir(exist_ok=True)
    vec = CountVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-z]{3,}\b",
        min_df=2,
        max_df=0.6,
        max_features=10000,
    )
    x = vec.fit_transform(fit)
    y = vec.transform(docs)
    # Empty rows receive the most frequent training topic (see below).
    supported = np.asarray(y.sum(1)).ravel() > 0
    x = x[np.asarray(x.sum(1)).ravel() > 0]
    vocab = vec.get_feature_names_out()
    vw(work / "fit.vw", x, vocab)
    vw(work / "eval.vw", y[supported], vocab)
    params = [
        "-t",
        20,
        "--num-document-passes",
        10,
    ]
    if smooth:
        params.extend(["--regularizer", "0.1 SmoothTheta", "0.01 SmoothPhi"])
    exe = CACHE / "bin/bigartm"
    cli(
        exe,
        work,
        [
            *params,
            "-c",
            work / "fit.vw",
            "-p",
            25,
            "--rand-seed",
            seed,
            "--save-model",
            "model",
            "--write-predictions",
            "fit.csv",
        ],
    )
    cli(
        exe,
        work,
        [
            *params,
            "-c",
            work / "eval.vw",
            "--load-model",
            "model",
            "--write-predictions",
            "eval.csv",
        ],
    )
    cols = (
        pd.read_csv(work / "eval.csv", sep=";", nrows=0)
        .columns.drop(["id", "title"])
        .tolist()
    )
    prior = theta(work, "fit.csv", cols, x.shape[0]).mean(0)
    result = np.full(len(docs), prior.argmax())
    result[supported] = theta(work, "eval.csv", cols, supported.sum()).argmax(1)
    return result


def main():
    OUTDIR.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    df = documents()
    df[["id", "split"]].to_csv(OUTDIR / "split.csv", index=False)
    fit = df.split.eq("train").to_numpy()
    vec = embeddings(df.text.tolist())
    km = KMeans(n_clusters=20, random_state=42, n_init=10).fit(vec[fit])
    docs = df.loc[~fit].copy()
    docs["kmeans"] = km.predict(vec[~fit])
    docs["plsa"] = baseline(df.text[fit], docs.text, smooth=False)
    docs["bigartm"] = baseline(df.text[fit], docs.text, smooth=True)
    centers = km.cluster_centers_
    reps = np.argsort(
        -(centers / np.linalg.norm(centers, axis=1, keepdims=True)) @ vec[fit].T, axis=1
    )[:, :5]
    pca = PCA(n_components=2, svd_solver="full").fit(vec[fit])
    xy = pca.transform(vec[fit])
    view = df.loc[fit, ["id"]].copy()
    view["cluster"] = km.labels_
    view["x"], view["y"] = xy.T
    view.to_csv(OUTDIR / "projection.csv", index=False)
    (OUTDIR / "projection-info.json").write_text(
        json.dumps(
            {
                "variance": pca.explained_variance_ratio_.tolist(),
                "centers": pca.transform(centers).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    path = OUTDIR / "topics.json"
    topics = descriptions(df[fit], reps)
    path.write_text(json.dumps(topics, indent=2) + "\n")
    choices = [
        json.dumps(
            {
                f"{t['title']} ({t['cluster']})": t["description"]
                + " (Keywords: "
                + ", ".join(t["keywords"])
                + ")"
            }
        )
        for t in topics
    ]
    model = LayaEntail("", threshold=0)
    model.decoys = []
    model.prep_pairs([], choices)
    (OUTDIR / "choices.json").write_text(json.dumps(model.questions, indent=2) + "\n")
    docs.loc[docs.split == "test"].drop(columns="text").to_csv(
        OUTDIR / "assignments.csv", index=False
    )


if __name__ == "__main__":
    main()
