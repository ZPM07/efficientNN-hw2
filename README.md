# HW2 · Efficient Models 2026 (ITMO) — EL2N data diet: NLP adaptation

Перенос (адаптация) основного протокола **EL2N** из статьи *Deep Learning on a
Data Diet: Finding Important Examples Early in Training* (Paul, Ganguli,
Dziugaite, NeurIPS 2021, arXiv:2107.07075) на NLP-задачу классификации текста.
Это адаптация алгоритмического протокола, а не буквальное воспроизведение
оригинального эксперимента (CIFAR/ResNet).

По заданию HW2 сдаются: **презентация** (объяснение статьи) — `Презентация.pdf`,
**отчёт** (воспроизведение) — `report/report.pdf`, **код** — `code/`.

## Что переносится из статьи

Оригинал считает счёта на CIFAR/ResNet. Здесь — адаптация основного протокола
EL2N к задаче текстовой классификации (не буквальное воспроизведение
эксперимента: модель, режим обучения и точка scoring отличаются):

| | Статья (vision) | Адаптация (NLP) |
|---|---|---|
| Данные | CIFAR-10/100, CINIC-10 | **AG News** (120k train / 7.6k test, 4 класса) |
| Модель | ResNet-18/50 с нуля | **DistilBERT** (finetune, голова заново) |
| Счёт | EL2N = ‖p(x) − onehot(y)‖₂ | то же (формула идентична) |
| Усреднение | 10 инициализаций → mean | 10 сидов скоринга → mean |
| Отбор | keep-top-k глобально | то же |
| Переобучение | фикс. число шагов и LR-расписания | 2811 шагов (3 эпохи) фикс. для всех размеров |
| Бейзлайны | full dataset + random subsets | то же |
| Прогонов на точку | 4 | 4 |

Размеры подвыборок: **10 / 25 / 50 / 75%** + полный датасет. Всего 46 прогонов
(10 скоринг + 36 переобучений) на 2× RTX 4090.

## Структура

```
code/                код воспроизведения (пакет el2n_nlp + оркестратор)
  src/el2n_nlp/        конфигурация, данные, скоринг, отбор, обучение, отчёты
  tests/               юнит-тесты корректности протокола
  orchestrate.py       планировщик всех 46 прогонов по 2 GPU
  make_slides.py       генерация презентации из результатов
  make_report.py       генерация отчёта (md + html + pdf)
report/              отчёт: report.pdf (+ md/html, регенерируются make_report.py, gitignore)
Презентация.pdf      презентация (сборка: code/make_slides.py, исходники в build/)
results/             figures/, summary.json, table.md, top_examples.json
data/                кэш токенизированных данных (gitignore)
```

В git не входят регенерируемые артефакты: сырые счётчики по сидам
(`results/scores/`), per-run JSON (`results/runs/`) и индексы подвыборок со
средним рейтингом (`results/subsets/`) — всё это воспроизводится заново
`code/orchestrate.py` (~2 GPU·часа на 2×RTX 4090, сиды зафиксированы).
В репозитории — все итоговые числа и материалы, на которые ссылается отчёт:
`summary.json`, `table.md`, `top_examples.json`, графики, оба PDF.

## Как запустить

```bash
# 1. данные
python -m el2n_nlp.cli tokenize

# 2. все эксперименты (скоринг -> отбор -> переобучение)
python code/orchestrate.py

# 3. отчётные материалы
python -m el2n_nlp.cli collect      # сводка + графики
python -m el2n_nlp.cli examples     # top/bottom тексты по EL2N
python code/make_report.py          # report.md / report.pdf
python code/make_slides.py          # Презентация.pdf (промежуточное — в build/)
```

Требования окружения — в `code/requirements.txt` (PyTorch с CUDA
ставится отдельно, см. заголовок файла).
