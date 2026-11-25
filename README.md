# Применение модели Jev в задачах тематического моделирования

[Статья](tex/report.pdf) · [Слайды](tex/talk.pdf) · [LaTeX для arXiv](tex/arxiv-source.tar.gz) · [Названия, описания и пять представителей каждой темы](results/ng-discovery/topics.json)

Проверяем, помогает ли автоматическое описание кластера и повторный выбор темы восстановить рубрикацию англоязычных писем. Один и тот же очищенный фрагмент письма получают все методы. Сравниваем PLSA без регуляризаторов и сглаженную модель в BigARTM, ближайший центр k-means в пространстве Nomic и выбор между описаниями Qwen3.5-4B. Локальная Laya и OpenAI Decisions (GPT-6-Luna) получают один список из 20 тем и Other. [Точный вопрос и все варианты ответа](results/ng-discovery/choices.json) · [Результаты OpenAI](results/ng-discovery/openai-comparison.csv).

Из 20 Newsgroups bydate отобраны 1000 писем для обучения, 200 для выбора порога и 160 для итогового теста. Разбиения стратифицированы. Метки не участвуют в оценивании центров и создании описаний. Основной критерий — AMI с внешними рубриками, больше означает лучшее совпадение. По каждому письму сохранены назначения всех методов. [Разбиение](results/ng-discovery/split.csv) · [Подбор порога](results/ng-discovery/tuning.csv) · [Тест](results/ng-discovery/comparison.csv) · [Индивидуальные назначения](results/ng-discovery/assignments.csv)

На 160 тестовых письмах AMI составила **0,042** для PLSA, **0,105** для сглаженной ARTM и **0,548** для Nomic + k-means. При одинаковых описаниях Qwen3.5-4B и штатном пороге 0,1 Laya дала **0,367** (67 смен кластера), GPT-6-Luna — **0,547** (6 смен). На валидации выбран порог 1 для Laya и 0,3 для Luna. На тесте Luna дала Other на 140 письмах. Для 360 писем API вернул 548 499 входных токенов, около $0,055 по опубликованному стандартному тарифу. [Тариф Decisions](https://developers.openai.com/api/docs/guides/decisions)

[Проекция кластеров](tex/ng-clusters.pdf) · [От писем к названию темы](tex/ng-naming.pdf) · [Письмо и оценки вариантов](tex/ng-decision.pdf)

## Воспроизведение

Команды ниже рассчитаны на macOS с Apple Silicon, Python 3.12, Xcode Command Line Tools и установленным uv. Для генерации названий используется MLX-версия Qwen3.5-4B, для эмбеддингов — ONNX INT8 Nomic Embed Text v1.5. Скрипт загружает закреплённые версии весов и публичный архив корпуса. Для повторения платной части требуется OPENAI_API_KEY в окружении или в локальном файле .env. Сохранённых назначений достаточно для пересборки отчёта без обращения к API.

~~~sh
uv sync --locked
./experiments/build_bigartm_mac.sh
uv run --locked python experiments/ng_discovery.py
uv run --locked python experiments/ng_baseline_check.py
# Отдельный платный запуск. Перед отправкой проверяется резерв стоимости.
uv run --locked python experiments/ng_openai.py --budget-usd 0.55
uv run --locked python experiments/publish_ng.py
uv run --locked python experiments/ng_figures.py
(cd tex && latexmk -pdf report.tex && latexmk -xelatex talk.tex)
~~~

Готовые таблицы и визуализацию можно пересобрать из опубликованных CSV и JSON командами с `publish_ng.py` и `ng_figures.py`. Для сборки PDF нужен TeX Live с pdfLaTeX, XeLaTeX, PGFPlots и шрифтами Fira Sans и Libertinus. [Авторский пример TopicJev](https://github.com/anatems1/TopicJev/blob/e271339027555160e1a9dc2efa8fc6e71b1a5475/examples/kmeans_genlbs_laya.py)

[Предыдущий генератор Qwen3-1.7B на тех же кластерах](results/ng-discovery/qwen3-1.7b) · [Проверка трёх инициализаций BigARTM](results/ng-discovery/classical-seeds.csv)

## Предыдущие эксперименты

В дополнительном протоколе проверялось, насколько выбранная по десяти ключевым словам тема BigARTM предсказывает скрытые слова. На 20NG PPL выросла с 5864 до 7549 после проверки Laya. Этот протокол использует другой объём обучения и другой вход. Результаты на NIPS и KOS доступны в архиве. [Измерения](results/paired) · [Код](experiments/paired_completion.py) · [Описание протокола и прежние материалы](https://github.com/fanis-khafizov/topicjev-benchmarking/tree/d90aad0)
