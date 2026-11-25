# Jev и тематическое моделирование

[Слайды на 30 минут](tex/talk.pdf) · [Статья в формате препринта](tex/report.pdf) · [Исходник доклада](tex/talk.tex) · [Исходник статьи](tex/report.tex) · [Архив LaTeX для arXiv](tex/arxiv-source.tar.gz)

## Какой вопрос проверяется

Улучшает ли TopicJev с локальной моделью Laya выбор одной темы для документа? Для обеих систем фиксируем 20 тем сглаженной модели BigARTM и показываем одинаковые наблюдаемые слова. Бейзлайн выбирает наиболее вероятную тему по θ. TopicJev может заменить её темой, выбранной Laya по десяти топ-словам каждой темы. При отказе сохраняется исходная тема. Качество оцениваем на другой, скрытой части каждого документа. Чем меньше перплексия (PPL), тем лучше.

| Корпус | Документов | PPL BigARTM, одна тема | PPL после TopicJev | Изменение PPL (95% парный бутстрэп) |
|---|---:|---:|---:|---:|
| 20 Newsgroups | 160 | 5864 | 7549 | +28,7% [+4,3; +72,5] |
| NIPS | 160 | 3370 | 5261 | +56,1% [+42,1; +72,0] |
| KOS | 160 | 3906 | 6821 | +74,6% [+51,5; +105,2] |

На 20 Newsgroups AMI с известными классами снизилась с 0,270 до 0,216. Корпуса NIPS и KOS взяты из [UCI Bag of Words](https://archive.ics.uci.edu/dataset/164/bag%2Bof%2Bwords), в этой версии доступны счётчики слов. Во всех трёх случаях Laya видит список слов с частотами. Вывод относится к этой форме входа и одной инициализации. Для проверки семантической уместности темы нужна отдельная оценка.

[Результаты по документам и списки тем](results/paired) · [Контрольные значения PPL для трёх конфигураций BigARTM](results/paired/baselines.csv) · [Прежний пилот на AG News и 20 Newsgroups](https://github.com/fanis-khafizov/topicjev-benchmarking/tree/d14951c)

## Воспроизведение

Python 3.12, зависимости и версия TopicJev зафиксированы в uv.lock. Все команды выполняются из корня проекта. Нативный CLI BigARTM собирается отдельно. На Apple Silicon с Xcode Command Line Tools используйте скрипт ниже. Для Linux можно передать путь к CLI из сборки коммита 47e37f982de87aa67bfd475ff1f39da696b181b3.

~~~sh
uv sync --locked
./experiments/build_bigartm_mac.sh
uv run --locked python experiments/paired_completion.py
uv run --locked python experiments/publish_paired.py
(cd tex && latexmk -xelatex talk.tex && latexmk -pdf report.tex)
~~~

Для PDF нужен TeX Live с pdfLaTeX (статья), XeLaTeX (слайды), Beamer, PGFPlots и шрифтами Fira Sans/Libertinus. Слайды содержат заметки и интервалы для одного докладчика, скрытые в проекционной версии. Подготовка данных проверяет SHA-256 исходных файлов. Учебные PDF, исходные корпуса, секреты и веса исключены из Git.

## OpenAI Decisions

Все результаты выше получены с локальной Laya. Адаптер для OpenAI Decisions в `experiments/verify_topics.py` реализует протокол прежнего пилота на AG News и 20 Newsgroups. Для сравнения OpenAI с текущими результатами потребуется применить адаптер к тем же 480 видимым частям документов и тем же описаниям тем. Подробности настройки пилота доступны в [README пилотной версии](https://github.com/fanis-khafizov/topicjev-benchmarking/blob/d14951c/README.md).
