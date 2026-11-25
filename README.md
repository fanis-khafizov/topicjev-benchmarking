# Jev и тематическое моделирование

[Слайды на 30 минут](tex/talk.pdf) · [Подробный отчёт](tex/report.pdf) · [Исходник доклада](tex/talk.tex) · [Исходник отчёта](tex/report.tex)

Сравнение трёх конфигураций официального BigARTM на AG News и 20 Newsgroups. Локальная Laya через TopicJev проверяет темы по десяти топ-словам. Для OpenAI Decisions реализован адаптер к тому же протоколу.

## Результаты

18 запусков BigARTM, 3 инициализации на метод и корпус. PPL оценена по скрытой половине документа. Средние для PLSA и сглаженной модели составили 1866 и 1679 на AG News, 2662 и 2546 на 20 Newsgroups. Конфигурация со сглаживанием и seed 11 задаёт фиксированную систему тем для следующего этапа.

| 100 тестовых примеров каждого корпуса | AMI BigARTM | AMI TopicJev + Laya | Принято |
|---|---:|---:|---:|
| AG News | 0,362 | 0,215 | 23 |
| 20 Newsgroups | 0,301 | 0,154 | 90 |

Другие инициализации, метрики и контроль с перестановкой меток доступны в [metrics.csv](results/metrics.csv). В каталоге [results](results) лежат тестовые ID, назначения и топ-слова. Исходные тексты восстанавливаются из публичных архивов по ID. Платный эксперимент ожидает запуска.

## Воспроизведение

Python 3.12, зависимости и версия TopicJev зафиксированы в uv.lock. Все команды выполняются из корня проекта. Нативный CLI BigARTM собирается отдельно. На Apple Silicon с Xcode Command Line Tools используйте скрипт ниже. Для Linux можно передать путь к CLI из сборки коммита 47e37f982de87aa67bfd475ff1f39da696b181b3.

~~~sh
uv sync --locked
./experiments/build_bigartm_mac.sh
uv run --locked python experiments/benchmark.py --bigartm "$PWD/.cache/bin/bigartm"
uv run --locked python experiments/verify_topics.py laya --size 100
uv run --locked python experiments/publish.py
(cd tex && latexmk -xelatex talk.tex report.tex)
~~~

Для PDF нужен TeX Live с XeLaTeX, Beamer, PGFPlots и шрифтами Fira Sans/Libertinus. Слайды содержат заметки и интервалы для одного докладчика, скрытые в проекционной версии. Подготовка данных проверяет SHA-256 исходных файлов. Учебные PDF, исходные корпуса, секреты и веса исключены из Git.

## OpenAI Decisions

Требуется действующий ключ API-проекта с кредитами и правом вызова POST /v1/decisions. Для локального запуска задайте OPENAI_API_KEY в окружении либо в файле .env корня проекта. Модель gpt-6-luna, один вопрос choice на документ. [Справочник и тариф Decisions](https://developers.openai.com/api/docs/guides/decisions).

~~~sh
uv run --locked python experiments/verify_topics.py estimate --size 100
# После настройки ключа и выбора бюджета:
uv run --locked python experiments/verify_topics.py openai --size 100 --budget-usd 0.10
uv run --locked python experiments/publish.py
(cd tex && latexmk -xelatex talk.tex report.tex)
~~~

Здесь 100 означает число документов **на корпус**. Для пробного запуска используйте --size 2 (4 запроса в сумме). Документы пилота входят в полный набор, ответы кэшируются по хешу запроса. Локальный резерв на полный набор около $0,0862 при стандартной цене $0,10 за миллион входных токенов. Это консервативная оценка по длине запроса в UTF-8 с запасом на оформление. Фактический счёт определяется тарифами проекта и данными usage. Параметр --budget-usd ограничивает резерв новых запросов за один запуск. После сетевого сбоя с неясным биллингом скрипт сохраняет маркер .pending; сверьте использование в проекте перед повторной отправкой.

Адаптер наследует JevEntail, передаёт параметры choice и переводит вероятности из ответа OpenAI в логарифмы для штатной нормировки и порога TopicJev. Режим estimate использует тот же сериализатор и работает локально. После измерения на 100 документах publish.py включит строки OpenAI в обе таблицы. Интерпретацию новых результатов в тексте следует обновить после измерения.
