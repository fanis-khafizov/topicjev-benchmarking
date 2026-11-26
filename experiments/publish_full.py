"""Publish full-split AMI, topic descriptions and figures."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from sklearn.metrics.cluster import expected_mutual_information

from benchmark import ROOT
from ng_full import DEST, EXAMPLE_ID, letters, topic_choices
from ng_figures import figures

TEX = ROOT / "tex"
METHODS = {
    "plsa": r"BigARTM, PLSA: максимум $\theta_{td}$",
    "artm": r"BigARTM, сглаживание $\Phi$ и $\Theta$: максимум $\theta_{td}$",
    "kmeans": "Nomic: ближайший центр $k$-means",
    "laya": "Темы кластеров Qwen: выбор Laya",
    "luna": "Темы кластеров Qwen: выбор GPT-6-Luna",
    "laya_known": "Известные рубрики Usenet: выбор Laya",
}


def fmt(x):
    return f"{x:.3f}".replace(".", ",")


def escape(text):
    special = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(special.get(c, c) for c in text)


def publish():
    train, test = letters("train"), letters("test")
    data = pd.read_csv(DEST / "assignments.csv")
    assert test.id.equals(data.id) and test.label.equals(data.label)
    scores = pd.read_csv(DEST / "comparison.csv").set_index("method").ami
    lines = [r"\begin{tabular}{lrr}\toprule", r"Метод & AMI & Смены из 7532\\\midrule"]
    for key, name in METHODS.items():
        assert np.isclose(
            scores[key], adjusted_mutual_info_score(data.label, data[key])
        )
        changes = (
            str(sum(data[key] != data.kmeans)) if key in ["laya", "luna"] else "---"
        )
        if key == "laya_known":
            lines.append(r"\midrule")
        lines.append(f"{name} & {fmt(scores[key])} & {changes}" + r"\\")
    (TEX / "full-test.tex").write_text(
        "\n".join([*lines, r"\bottomrule\end{tabular}", ""])
    )
    counts = pd.crosstab(data.label, data.kmeans).reindex(
        columns=range(20), fill_value=0
    )
    mi = mutual_info_score(data.label, data.kmeans)
    emi = expected_mutual_information(counts.to_numpy(), len(data))
    h = (entropy(counts.sum(0)) + entropy(counts.sum(1))) / 2
    assert np.isclose((mi - emi) / (h - emi), scores.kmeans)
    (TEX / "full-ami.tex").write_text(
        f"Для ближайшего центра Nomic на полном тесте взаимная информация равна {fmt(mi)}, её случайный уровень --- {fmt(emi)}, средняя энтропия --- {fmt(h)}. Получаем AMI {fmt(scores.kmeans)}.\n"
    )
    empty = int(data["empty"].sum())
    refused = int(data.api_refusal.sum())
    (TEX / "full-coverage.tex").write_text(
        f"После очистки {empty} тестовых писем оказались пустыми. GPT-6-Luna вернула отказ для {refused} непустых писем. Все 7532 документа включены в AMI.\n"
    )
    accepted = data[~data["empty"] & ~data.api_refusal]
    estimates = {
        m: fmt(adjusted_mutual_info_score(accepted.label, accepted[m]))
        for m in ("kmeans", "laya", "luna")
    }
    (TEX / "full-answered.tex").write_text(
        f"Для {len(accepted)} непустых писем сервис вернул распределение по темам. "
        f"На этой общей подвыборке AMI ближайшего центра составляет {estimates['kmeans']}, "
        f"Laya --- {estimates['laya']}, GPT-6-Luna --- {estimates['luna']}. "
        "Состав подвыборки зависит от ответов сервиса. Основное сравнение в таблице охватывает полный тест.\n"
    )
    known = scores["laya_known"]
    nonempty = data[~data["empty"]]
    same_input = {
        m: fmt(adjusted_mutual_info_score(nonempty.label, nonempty[m]))
        for m in ("laya", "laya_known")
    }
    correct = int(data.laya_known.eq(data.label).sum())
    accuracy_percent = f"{100 * correct / len(data):.1f}".replace(".", ",")
    (TEX / "full-known.tex").write_text(
        f"При известных именах рубрик AMI равна {fmt(known)}. Laya вернула имя группы публикации для "
        f"{correct} из {len(data)} писем ({accuracy_percent}\\%). "
        f"На {len(nonempty)} непустых письмах AMI Laya с описаниями Qwen равна {same_input['laya']}, "
        f"с названиями рубрик --- {same_input['laya_known']}. "
        "Здесь меняются и границы предлагаемых тем, и формулировки вариантов. "
        "Для оценки только названий нужно сравнить несколько описаний одних и тех же кластеров.\n"
    )
    (TEX / "full-conclusion.tex").write_text(
        f"Среди методов, извлекающих темы из корпуса, ближайший центр получил AMI {fmt(scores.kmeans)}. "
        f"Повторный выбор по описаниям Qwen дал {fmt(scores.laya)} для Laya и {fmt(scores.luna)} для GPT-6-Luna.\n\n"
        f"При известных рубриках Laya получила {fmt(known)}. Результат зависит от того, "
        "какие темы предложены модели и как они сформулированы.\n"
    )
    topics = json.loads((DEST / "topics.json").read_text())
    criteria = topic_choices(topics)
    (DEST / "choices.json").write_text(
        json.dumps(
            {
                "topic": {
                    "type": "choice",
                    "instructions": "Which topic does the text belong to?",
                    "criteria": criteria,
                }
            },
            indent=2,
        )
        + "\n"
    )
    (TEX / "full-choices.tex").write_text(
        "\n".join(
            [
                r"\begin{enumerate}",
                *[r"\item " + escape(t) for t in criteria],
                r"\end{enumerate}",
                "",
            ]
        )
    )
    figures(
        DEST,
        pd.concat([train, test]).set_index("id").text,
        prefix="full",
        example_id=EXAMPLE_ID,
        neighbor=5,
    )
    fractions = counts.div(counts.sum(axis=1), axis=0)
    _, order = linear_sum_assignment(-fractions.to_numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(
        100 * fractions.iloc[:, order], vmin=0, vmax=100, cmap="Blues", aspect="auto"
    )
    ax.set_xticks(range(20), order, fontsize=8)
    ax.set_yticks(range(20), counts.index, fontsize=8)
    ax.set_xlabel("Кластер Nomic + k-means")
    fig.colorbar(img, ax=ax, label="Доля писем данной рубрики, %")
    fig.savefig(TEX / "full-heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    publish()
