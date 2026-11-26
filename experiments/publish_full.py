"""Publish full-split AMI, topic descriptions and figures."""

import argparse
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
    "luna_known": "Известные рубрики Usenet: выбор GPT-6-Luna",
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


def publish(*, tables_only=False):
    test = letters("test")
    data = pd.read_csv(DEST / "assignments.csv")
    assert test.id.equals(data.id) and test.label.equals(data.label)
    comparison = pd.read_csv(DEST / "comparison.csv").set_index("method")
    scores = comparison.ami
    lines = [r"\begin{tabular}{lrrr}\toprule", r"Метод & Писем & AMI & Смены\\\midrule"]
    for key, name in METHODS.items():
        valid = data.dropna(subset=[key])
        assert comparison.at[key, "n"] == len(valid)
        assert np.isclose(
            scores[key], adjusted_mutual_info_score(valid.label, valid[key])
        )
        changes = (
            str(sum(data[key] != data.kmeans)) if key in ["laya", "luna"] else "---"
        )
        if key == "laya_known":
            lines.append(r"\midrule")
        lines.append(f"{name} & {len(valid)} & {fmt(scores[key])} & {changes}" + r"\\")
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
        f"После очистки {empty} тестовых писем оказались пустыми. GPT-6-Luna вернула отказ для {refused} непустых писем. В этом опыте с темами Qwen в AMI включены все 7532 документа.\n"
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
        "Состав подвыборки зависит от ответов сервиса. В основной строке метода с темами Qwen оценка охватывает полный тест.\n"
    )
    known = scores["laya_known"]
    nonempty = data[~data["empty"]]
    same_input = {
        m: fmt(adjusted_mutual_info_score(nonempty.label, nonempty[m]))
        for m in ("laya", "laya_known")
    }
    correct = int(data.laya_known.eq(data.label).sum())
    accuracy_percent = f"{100 * correct / len(data):.1f}".replace(".", ",")
    direct = data.dropna(subset=["luna_known"])
    paired = direct[~direct["empty"] & ~direct.api_refusal]
    both_ami = {
        m: fmt(adjusted_mutual_info_score(paired.label, paired[m]))
        for m in ("luna", "luna_known")
    }
    correct_luna = int(direct.luna_known.eq(direct.label).sum())

    def percent(count, total):
        return f"{100 * count / total:.1f}".replace(".", ",")

    known_laya_paired = fmt(adjusted_mutual_info_score(direct.label, direct.laya_known))
    combined = data.luna_known.fillna(data.laya_known)
    combined_ami = fmt(adjusted_mutual_info_score(data.label, combined))
    (TEX / "full-known.tex").write_text(
        f"При известных именах рубрик AMI Laya равна {fmt(known)}. Модель вернула имя группы публикации для "
        f"{correct} из {len(data)} писем ({accuracy_percent}\\%). "
        f"На {len(nonempty)} непустых письмах AMI Laya с описаниями Qwen равна {same_input['laya']}, "
        f"с названиями рубрик --- {same_input['laya_known']}. "
        f"GPT-6-Luna выбрала рубрику для {len(direct)} из {len(data)} писем. "
        f"На этих же письмах AMI равна {fmt(scores.luna_known)} для GPT-6-Luna и {known_laya_paired} для Laya. "
        f"Правильную рубрику GPT-6-Luna назвала для {correct_luna} писем "
        f"({percent(correct_luna, len(direct))}\\% среди ответов, {percent(correct_luna, len(data))}\\% от всего теста).\n\n"
        f"Если при отсутствии назначения GPT-6-Luna брать ответ Laya с теми же рубриками, "
        f"связка получает AMI {combined_ami} на всех {len(data)} письмах. "
        f"На {len(paired)} непустых письмах с прямым ответом GPT-6-Luna в обоих режимах "
        f"AMI составила {both_ami['luna']} для описаний Qwen и {both_ami['luna_known']} для названий групп. "
        "Здесь меняются и границы предлагаемых тем, и формулировки вариантов. "
        "Для оценки только названий нужно сравнить несколько описаний одних и тех же кластеров.\n"
    )
    (TEX / "full-conclusion.tex").write_text(
        f"Среди методов, извлекающих темы из корпуса, ближайший центр получил AMI {fmt(scores.kmeans)}. "
        f"Повторный выбор по описаниям Qwen дал {fmt(scores.laya)} для Laya и {fmt(scores.luna)} для GPT-6-Luna.\n\n"
        f"На {len(direct)} письмах, для которых GPT-6-Luna выбрала известную рубрику, "
        f"её AMI составляет {fmt(scores.luna_known)}. У Laya на тех же письмах AMI равна {known_laya_paired}.\n"
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
    if tables_only:
        return
    train = letters("train")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-only", action="store_true")
    publish(tables_only=parser.parse_args().tables_only)
