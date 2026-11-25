# Применение модели Jev в задачах тематического моделирования

[Статья](tex/report.pdf) · [Слайды](tex/talk.pdf) · [LaTeX для arXiv](tex/arxiv-source.tar.gz) · [Названия, описания и пять представителей каждой темы](results/ng-discovery/topics.json)

Проверяем, помогает ли автоматическое описание кластера и повторный выбор темы восстановить рубрикацию англоязычных писем. Один и тот же очищенный фрагмент письма получают все методы. Сравниваем ведущую тему сглаженной модели BigARTM, ближайший центр k-means в пространстве Nomic и проверку темы моделью Laya по описаниям Qwen. Платный API не используется.

Из 20 Newsgroups bydate отобраны 1000 писем для обучения, 200 для выбора порога и 160 для итогового теста. Разбиения стратифицированы. Метки не участвуют в оценивании центров и создании описаний. Основной критерий — AMI с внешними рубриками, больше означает лучшее совпадение. По каждому письму сохранены назначения всех методов. [Разбиение](results/ng-discovery/split.csv) · [Подбор порога](results/ng-discovery/tuning.csv) · [Тест](results/ng-discovery/comparison.csv) · [Индивидуальные назначения](results/ng-discovery/assignments.csv)

На тесте AMI равна **0,105** для BigARTM, **0,548** для Nomic + k-means и **0,374** после проверки Laya со штатным порогом 0,1. Валидация выбрала порог 1, при котором сохраняются все исходные назначения k-means.

## Воспроизведение

Команды ниже рассчитаны на macOS с Apple Silicon, Python 3.12, Xcode Command Line Tools и установленным uv. Для генерации названий используется MLX-версия Qwen3-1.7B, для эмбеддингов — ONNX INT8 Nomic Embed Text v1.5. Скрипт загружает закреплённые версии весов и публичный архив корпуса. Все модели работают локально.

~~~sh
uv sync --locked
./experiments/build_bigartm_mac.sh
uv run --locked python experiments/ng_discovery.py
uv run --locked python experiments/publish_ng.py
(cd tex && latexmk -pdf report.tex && latexmk -xelatex talk.tex)
~~~

Готовые таблицы и визуализацию можно пересобрать из опубликованных CSV и JSON командой с `publish_ng.py`. Для сборки PDF нужен TeX Live с pdfLaTeX, XeLaTeX, PGFPlots и шрифтами Fira Sans и Libertinus. [Авторский пример TopicJev](https://github.com/anatems1/TopicJev/blob/e271339027555160e1a9dc2efa8fc6e71b1a5475/examples/kmeans_genlbs_laya.py)

## Предыдущие эксперименты

В дополнительном протоколе проверялось, насколько выбранная по десяти ключевым словам тема BigARTM предсказывает скрытые слова. На 20NG PPL выросла с 5864 до 7549 после проверки Laya. Этот протокол использует другой объём обучения и другой вход. Результаты на NIPS и KOS доступны в архиве. [Измерения](results/paired) · [Код](experiments/paired_completion.py) · [Описание протокола и прежние материалы](https://github.com/fanis-khafizov/topicjev-benchmarking/tree/d90aad0)
