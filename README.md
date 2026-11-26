# Применение модели Jev в задачах тематического моделирования

[Статья](tex/report.pdf) · [Слайды](tex/talk.pdf) · [Исходники статьи для arXiv](tex/arxiv-source.tar.gz)

20 Newsgroups — коллекция сообщений дискуссионных групп Usenet. В архиве bydate 18 846 документов, из них 11 314 ранних сообщений для обучения и 7532 поздних для тестирования. У каждого документа известна группа публикации. В нашем сравнении на каждую группу приходится 50 обучающих и восемь тестовых писем, ещё десять отложены для проверки. Модели получают первые 80 слов после очистки. [Описание корпуса](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) · [Разбиение](results/ng-discovery/split.csv)

Qwen3.5-4B описывает кластеры Nomic по пяти ближайшим обучающим письмам. Laya и GPT-6-Luna выбирают из этих 20 тем. Для 25 тестовых писем API вернул отказ, и система сохранила ближайший центр. [Варианты ответа](results/ng-discovery/choices.json) · [Итоговые AMI](results/ng-discovery/comparison.csv) · [Проверка инициализаций BigARTM](results/ng-discovery/classical-seeds.csv)

## Воспроизведение

Команды рассчитаны на Apple Silicon с Xcode Command Line Tools, uv и TeX Live. Платный шаг использует ключ из локального файла .env. Для сборки материалов из сохранённых данных достаточно двух последних Python-команд и LaTeX.

~~~sh
uv sync --locked
./experiments/build_bigartm_mac.sh
uv run --locked python experiments/ng_discovery.py
uv run --locked python experiments/ng_baseline_check.py
uv run --locked python experiments/ng_forced.py laya
uv run --locked python experiments/ng_forced.py luna --budget-usd 0.18
uv run --locked python experiments/publish_ng.py
uv run --locked python experiments/ng_figures.py
(cd tex && latexmk -pdf report.tex && latexmk -xelatex talk.tex)
~~~
