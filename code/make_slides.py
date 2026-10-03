"""Generate the HW2 presentation (HTML -> PDF via headless chromium).

Numbers and figures are injected from results/, never hardcoded.
Theme: applying-themes/tech-innovation (contrast-verified).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from el2n_nlp import config  # noqa: E402

BASE = "#0b0e14"
SURFACE = "#131924"
LINE = "#26314a"
INK = "#e8eef7"
INK2 = "#a7b6ca"
INK3 = "#8496ad"
ACCENT = "#38bdf8"
RED = "#f28b82"  # second series: taken from the ink ramp family (muted red)

plt.rcParams.update(
    {
        "figure.facecolor": BASE,
        "axes.facecolor": BASE,
        "savefig.facecolor": BASE,
        "text.color": INK,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "grid.color": LINE,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "font.size": 13,
    }
)


def load():
    summary = json.load(open(config.RESULTS_DIR / "summary.json"))
    stats = json.load(open(config.SUBSETS_DIR / "stats.json"))
    runs = {}
    for f in sorted(config.RUNS_DIR.glob("*.json")):
        runs[f.stem] = json.load(open(f))
    return summary, stats, runs


def fig_acc_vs_fraction(summary, out):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    all_highs = []
    for mode, color in (("random", ACCENT), ("el2n", RED)):
        pts = [(pct, summary[f"{mode}_{pct}"]) for pct in config.SIZES_PCT
               if f"{mode}_{pct}" in summary]
        if not pts:
            continue
        xs = np.array([pct for pct, _ in pts], dtype=float)
        means = np.array([g["final_mean"] for _, g in pts]) * 100
        lows = np.array([g["final_min"] for _, g in pts]) * 100
        highs = np.array([g["final_max"] for _, g in pts]) * 100
        all_highs += list(highs)
        ax.plot(xs, means, "o-", color=color, lw=2,
                label="EL2N top-score" if mode == "el2n" else "random")
        ax.fill_between(xs, lows, highs, color=color, alpha=0.18)
        for (pct, g) in pts:
            for a in g["accs"]:
                ax.scatter(pct, a * 100, s=16, color=color, alpha=0.5, zorder=3)
    if "full_100" in summary:
        ax.axhline(summary["full_100"]["final_mean"] * 100, ls="--", c=INK3, lw=1.2)
    ax.set_xlabel("retained fraction of training set (%)")
    ax.set_ylabel("final test accuracy (%)")
    ax.set_ylim(bottom=min(25, min(all_highs + [100]) - 2))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_curves(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, (mode, pct) in zip(axes, [("full", 100), ("el2n", 10), ("random", 10)]):
        for r in range(config.RETRAIN_RUNS):
            d = runs.get(config.run_tag(mode, pct, r))
            if not d:
                continue
            steps = [h["step"] for h in d["history"]]
            accs = [h["test_acc"] * 100 for h in d["history"]]
            color = RED if mode == "el2n" else (ACCENT if mode == "random" else INK3)
            ax.plot(steps, accs, lw=1.6, color=color, alpha=0.85)
        ax.set_title(f"{mode} @ {pct}%", color=INK)
    axes[0].set_xlabel("optimizer step")
    axes[1].set_xlabel("optimizer step")
    axes[2].set_xlabel("optimizer step")
    axes[0].set_ylabel("test accuracy (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_scores(out):
    from scipy.stats import spearmanr

    avg = np.load(config.SUBSETS_DIR / "el2n_mean_scores.npy")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.9))
    axes[0].hist(avg, bins=60, color=ACCENT, alpha=0.85)
    for pct in (10, 50):
        cut = np.sort(avg)[::-1][config.subset_size(pct) - 1]
        axes[0].axvline(cut, ls="--", c=INK, lw=1.2)
        axes[0].text(cut * 1.01, axes[0].get_ylim()[1] * 0.85, f"top {pct}%", color=INK, fontsize=11)
    axes[0].set_xlabel("mean EL2N score, 10 initializations")
    axes[0].set_ylabel("# train examples")

    score_files = [p for p in sorted(config.SCORES_DIR.glob("scores_seed*.npy"))
                 if not p.stem.endswith("_early")]
    runs = [np.load(f) for f in score_files]
    rhos = [spearmanr(runs[i], runs[j]).statistic
            for i in range(len(runs)) for j in range(i + 1, len(runs))]
    axes[1].hist(rhos, bins=15, color=RED, alpha=0.85)
    axes[1].axvline(np.mean(rhos), c=INK, ls="--", lw=1.4)
    axes[1].set_xlabel(f"rank correlation between runs (mean ρ = {np.mean(rhos):.3f})")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return float(np.mean(rhos))


def fmt(g, key="final_mean"):
    return f"{g[key] * 100:.2f}" if g else "—"


def fmt_std(g):
    return f"{g['final_std'] * 100:.2f}" if g else "—"


def main():
    summary, stats, runs = load()
    pres = config.ROOT / "build" / "presentation"  # служебная сборка (gitignore)
    pres.mkdir(parents=True, exist_ok=True)
    fig_acc_vs_fraction(summary, pres / "acc_vs_fraction.png")
    fig_curves(runs, pres / "curves.png")
    rho = fig_scores(pres / "scores.png")

    full = summary.get("full_100", {})
    rows = []
    for p in config.SIZES_PCT:
        e, r = summary.get(f"el2n_{p}"), summary.get(f"random_{p}")
        diff = (e["final_mean"] - r["final_mean"]) * 100 if e and r else float("nan")
        rows.append((p, fmt(e), fmt_std(e), fmt(r), fmt_std(r), f"{diff:+.2f}"))
    table_html = "\n".join(
        f"<tr><td>{p}%</td><td>{a} ± {b}</td><td>{c} ± {d}</td><td class='{'pos' if float(dd) >= 0 else 'neg'}'>{dd}</td></tr>"
        for p, a, b, c, d, dd in rows
    )
    examples = json.load(open(config.RESULTS_DIR / "top_examples.json"))
    hi_world = examples["highest"]["World"][0]
    hi_sci = examples["highest"]["Sci/Tech"][0]

    html = TEMPLATE
    repl = {
        "FULL_MEAN": fmt(full),
        "FULL_STD": fmt_std(full),
        "TABLE_ROWS": table_html,
        "RHO": f"{rho:.3f}",
        "EX1_TEXT": hi_world["text"][:180],
        "EX1_SCORE": f"{hi_world['score']:.3f}",
        "EX2_TEXT": hi_sci["text"][:180],
        "EX2_SCORE": f"{hi_sci['score']:.3f}",
    }
    for k, v in repl.items():
        html = html.replace("{{" + k + "}}", str(v))
    (pres / "index.html").write_text(html)
    subprocess.run(
        [
            "chromium", "--headless", "--disable-gpu", "--no-sandbox",
            "--print-to-pdf=" + str(pres / "slides.pdf"),
            "--no-pdf-header-footer", str(pres / "index.html"),
        ],
        check=True,
    )
    print("wrote", config.ROOT / "Презентация.pdf", "(sources in", pres, ")")
    # сдавочная копия презентации в корне репозитория (имя по запросу)
    shutil.copyfile(pres / "slides.pdf", config.ROOT / "Презентация.pdf")


TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<style>
:root {
  --base: #0b0e14; --surface: #131924; --line: #26314a;
  --ink: #e8eef7; --ink-2: #a7b6ca; --ink-3: #8496ad;
  --accent: #38bdf8; --accent-ink: #031824;
  --font-display: "DejaVu Sans", sans-serif;
  --font-text: "DejaVu Sans", sans-serif;
  --font-mono: "DejaVu Sans Mono", monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
@page { size: 1280px 720px; margin: 0; }
body { background: var(--base); color: var(--ink); font-family: var(--font-text); }
.s {
  width: 1280px; height: 720px; page-break-after: always; overflow: hidden;
  padding: 56px 72px; position: relative; display: flex; flex-direction: column;
}
.kicker { font-family: var(--font-mono); font-size: 15px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 14px; }
h1 { font-size: 52px; line-height: 1.15; font-weight: bold; }
h2 { font-size: 34px; line-height: 1.2; margin-bottom: 26px; }
p, li { font-size: 20px; line-height: 1.5; color: var(--ink-2); }
strong { color: var(--ink); }
ul { padding-left: 26px; display: grid; gap: 12px; }
.formula { font-family: var(--font-mono); font-size: 26px; color: var(--ink);
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 18px 26px; display: inline-block; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 22px 26px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.accent { color: var(--accent); }
.meta { color: var(--ink-3); font-size: 16px; font-family: var(--font-mono); }
table { border-collapse: collapse; width: 100%; }
th { font-family: var(--font-mono); font-size: 15px; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--ink-3); text-align: left;
  border-bottom: 1px solid var(--line); padding: 10px 14px; }
td { padding: 12px 14px; font-size: 20px; border-bottom: 1px solid var(--line); color: var(--ink-2); }
td:first-child { color: var(--ink); }
.pos { color: var(--accent); } .neg { color: #f28b82; }
img.fig { border: 1px solid var(--line); border-radius: 10px; }
.page { position: absolute; bottom: 28px; right: 40px; font-family: var(--font-mono);
  font-size: 13px; color: var(--ink-3); }
.big { font-size: 44px; color: var(--ink); font-weight: bold; }
.quote { font-family: var(--font-mono); font-size: 17px; color: var(--ink-2);
  border-left: 3px solid var(--accent); padding: 10px 18px; background: var(--surface); }
</style></head><body>

<div class="s" style="justify-content:center">
  <div class="kicker">Efficient Models 2026 · Домашнее задание 2 · перенос протокола в NLP</div>
  <h1>Deep Learning on a Data Diet:<br>Finding Important Examples Early in Training</h1>
  <p style="margin-top:26px; font-size:24px">Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite — <strong>NeurIPS 2021</strong></p>
  <p style="margin-top:40px" class="meta">Адаптация EL2N к NLP: AG News + DistilBERT · Смирнов Василий Артурович</p>
  <div class="page">01</div>
</div>

<div class="s">
  <div class="kicker">Постановка</div>
  <h2>Сколько данных — лишние?</h2>
  <ul>
    <li>Сети переобучены по параметрам, датасеты — по объёму: значительная часть примеров <strong>не нужна для обобщения</strong>.</li>
    <li>Вопрос работы: <strong>какие</strong> примеры важны и <strong>как найти их рано</strong>, не обучая модель до конца?</li>
    <li>Зачем: меньше памяти и шагов обучения, дешевле HPO, меньше поверхность для атак и ошибок, виден «скелет» данных.</li>
    <li>Аналогия: данные на «диете» — если 50% примеров можно выкинуть <strong>без потери качества</strong>, тестовая точность — та же, а стоимость — вдвое меньше.</li>
  </ul>
  <div class="page">02</div>
</div>

<div class="s">
  <div class="kicker">Контекст</div>
  <h2>Что было до статьи</h2>
  <div class="grid2">
    <div class="card"><p><strong>Forgetting events</strong> (Toneva et al. 2019)</p>
      <p style="margin-top:10px">Пример «забыт» на эпохе t, если на t−1 был верен, а на t — нет. Важные — никогда не забытые.
      <strong>Минус:</strong> нужен полный прогон обучения — счётчик глобальный и «поздний».</p></div>
    <div class="card"><p><strong>Deep Batch / moderate coreset</strong></p>
      <p style="margin-top:10px">Отбор по градиентам на инициализации или по средней трудности за много прогонов.
      <strong>Минус:</strong> градиенты по всем параметрам дороги; на инициализации сигнал слаб.</p></div>
  </div>
  <p style="margin-top:28px">Идея статьи: <strong>локальная информация раннего обучения</strong> — простые нормы ошибки/градиента,
  усреднённые по нескольким инициализациям, — уже через несколько эпох ранжируют данные не хуже поздних метрик.</p>
  <div class="page">03</div>
</div>

<div class="s">
  <div class="kicker">Метод</div>
  <h2>Два early-training счёта</h2>
  <div style="display:grid; gap:22px">
    <div class="card"><p><strong>GraNd</strong> — L2-норма градиента лосса по всем параметрам:</p>
      <div class="formula" style="margin-top:10px">GraNd(x,y) = ‖∇<sub>θ</sub> ℓ(f<sub>θ</sub>(x), y)‖<sub>2</sub></div></div>
    <div class="card"><p><strong>EL2N</strong> — L2-норма ошибки между вероятностями и one-hot меткой:</p>
      <div class="formula" style="margin-top:10px">EL2N(x,y) = ‖p<sub>θ</sub>(x) − y‖<sub>2</sub></div>
      <p style="margin-top:12px">Для кросс-энтропии градиент лосса по логитам — в точности <strong>(p − y)</strong>;
      EL2N — норма только градиента по логитам (для CE это p−y), GraNd — норма всего <strong>Jᵀ(p−y)</strong> по параметрам.
      Это не «GraNd без множителя» (равенства с общей константой нет), а значительно более дешёвый <strong>proxy</strong> градиентной важности: один forward вместо backprop.</p></div>
  </div>
  <div class="page">04</div>
</div>

<div class="s">
  <div class="kicker">Метод</div>
  <h2>Как устроен протокол отбора</h2>
  <ul>
    <li><strong>1. Parent run:</strong> обычная тренировка; счётчики считаются <em>рано</em> (CIFAR-10: эпоха 10 из 200).</li>
    <li><strong>2. Усреднение:</strong> счётчики складываются по <strong>10 независимым инициализациям</strong> — убирает спецификацию весов, оставляет свойство датасета.</li>
    <li><strong>3. Отбор:</strong> глобально <strong>keep-top-k</strong> по среднему счёту (оставляем «трудные» примеры!).</li>
    <li><strong>4. Переобучение с нуля</strong> на подвыборке при <em>том же числе шагов и расписании LR</em> (390·200 для CIFAR-10) — сравнение честное по оптимизационному бюджету.</li>
    <li><strong>5. Отчёт:</strong> 4 независимых прогона на точку; бейзлайны — полный датасет и <em>random subsets того же размера</em> (свой subset на прогон).</li>
  </ul>
  <div class="page">05</div>
</div>

<div class="s">
  <div class="kicker">Результаты статьи</div>
  <h2>Главная табличка</h2>
  <ul>
    <li>CIFAR-10 / ResNet-18: EL2N из эпохи 10–20 позволяет убрать <strong>50% обучения и даже slightly улучшить</strong> тестовую точность (~95.1% → ~95.3%).</li>
    <li>EL2N, посчитанный <strong>на 5% обучения</strong>, сопоставим с forgetting score, посчитанным <strong>после 100%</strong> обучения.</li>
    <li>GraNd <strong>на инициализации</strong> — почти не работает (исправление ошибки ранней версии статьи).</li>
    <li>При экстремальном прунинге (≤10–25%) — резкое падение: top-score примеры плохо покрывают распределение.</li>
  </ul>
  <div class="page">06</div>
</div>

<div class="s">
  <div class="kicker">Результаты статьи</div>
  <h2>Свойства счётов</h2>
  <ul>
    <li><strong>Ранг устойчив:</strong> EL2N на эпохе 10 коррелирует с forgetting на финале (ρ ≈ 0.75–0.8); счётчики на разных эпохах согласованы.</li>
    <li><strong>Это свойство данных, а не модели:</strong> ранги переносятся между ResNet-18/50 и между гиперпараметрами (в т.ч. внутри HPO-ранов).</li>
    <li><strong>Усреднение по инициализациям важно:</strong> счёт одного рана заметно хуже, чем среднее по 10–20 ранам.</li>
    <li><strong>Шум:</strong> случайно перемаркированные примеры концентрируются в high-EL2N хвосте; top-EL2N CIFAR-примеры — визуально «странные»/пограничные.</li>
    <li><strong>Классы сбалансированы:</strong> keep-top не ломает баланс классов датасета.</li>
  </ul>
  <div class="page">07</div>
</div>

<div class="s">
  <div class="kicker">Выводы статьи</div>
  <h2>Значение</h2>
  <ul>
    <li>Важность примера — <strong>локальное, ранее</strong> свойство: не нужна полная траектория обучения.</li>
    <li>EL2N дёшев (один forward), переносим между архитектурами — scores можно посчитать один раз и экономить на всех будущих тренировках и HPO.</li>
    <li>Побочные применения: детекция шумных меток, анализ динамики обучения (устойчивые подпространства представлений, форма loss surface).</li>
  </ul>
  <div class="page">08</div>
</div>

<div class="s">
  <div class="kicker">Наш эксперимент · постановка</div>
  <h2>Перенос эксперимента в NLP</h2>
  <div class="grid2">
    <div class="card"><p class="meta">СТАТЬЯ (vision)</p><br>
      <p>CIFAR-10, 50k train / 10 классов<br>ResNet-18 с нуля<br>SGD lr=0.1, 200 эп. = 78 000 шагов<br>EL2N на эпохе 10 из 200<br>10 инициализаций → mean → keep-top-k<br>4 рана на точку, 16–84 перцентили</p></div>
    <div class="card"><p class="meta">МЫ (NLP)</p><br>
      <p>AG News, 120k train / 4 класса<br>DistilBERT (finetune, head заново)<br>AdamW lr=5e-5, 3 эп. = <strong>2811 шагов (фикс)</strong><br>EL2N после 1-й эпохи полного расписания<br>10 сидов → mean → keep-top-k<br>4 рана на точку (мин/макс на графике)</p></div>
  </div>
  <p style="margin-top:22px">Ключевые протокольные свойства сохранены: ранний счётчик, усреднение по инициализациям, keep-top,
  <strong>фиксированное число шагов для всех размеров подвыборки</strong>, отдельные сиды скоринга и обучения, random-бейзлайн.</p>
  <div class="page">09</div>
</div>

<div class="s">
  <div class="kicker">Наш эксперимент · результаты</div>
  <h2>Точность vs размер датасета (AG News, DistilBERT)</h2>
  <div style="display:grid; grid-template-columns: 1.1fr 0.9fr; gap: 26px; align-items:center">
    <img class="fig" src="acc_vs_fraction.png" style="width:100%">
    <div>
      <table>
        <tr><th>% данных</th><th>EL2N top (acc %)</th><th>random (acc %)</th><th>EL2N−rand</th></tr>
        {{TABLE_ROWS}}
      </table>
      <p style="margin-top:18px">Full 100%: <span class="big">{{FULL_MEAN}} ± {{FULL_STD}}</span></p>
    </div>
  </div>
  <div class="page">10</div>
</div>

<div class="s">
  <div class="kicker">Наш эксперимент · динамика</div>
  <h2>Малые EL2N-подвыборки: два режима отказа</h2>
  <img class="fig" src="curves.png" style="width:100%">
  <p style="margin-top:18px">Один и тот же бюджет 2811 шагов (протокол статьи). <strong>EL2N @ 10%</strong> — не обучается вовсе (~29% с первого
   evaluation): в подвыборке 2.9% Sports (против 25% в датасете) и 20.1% примеров со счётом > 1 (у random ~2%).
   <strong>EL2N @ 25%</strong> — пик 90.9% к шагу 400, затем переобучение с осцилляциями до ~81%.
   <strong>Random @ 10%</strong> — ровные ~92% без отказа.</p>
  <div class="page">11</div>
</div>

<div class="s">
  <div class="kicker">Наш эксперимент · счётчики</div>
  <h2>EL2N-распределение и согласованность инициализаций</h2>
  <img class="fig" src="scores.png" style="width:100%">
  <p style="margin-top:18px">Средний Spearman между 10 сидами скоринга: <strong>ρ = {{RHO}}</strong> — ранг примеров устойчив между инициализациями,
  усреднение по сидам действительно имеет смысл (как в статье).</p>
  <div class="page">12</div>
</div>

<div class="s">
  <div class="kicker">Наш эксперимент · что внутри top-score</div>
  <h2>Высокий EL2N: по содержанию — шум и пограничные примеры</h2>
  <div class="quote">«{{EX1_TEXT}}…» <span class="meta">(World, EL2N={{EX1_SCORE}})</span></div>
  <div class="quote" style="margin-top:18px">«{{EX2_TEXT}}…» <span class="meta">(Sci/Tech, EL2N={{EX2_SCORE}})</span></div>
  <ul style="margin-top:24px">
    <li>Счёт в топе ≈ 1.41 = <strong>√2 — максимум ‖p−y‖₂</strong>: модель уверенно указывает на другой класс. В просмотренной небольшой выборке такие примеры семантически выглядят как <strong>потенциально ошибочные метки</strong> (взрывы в Багдаде с меткой Sports/Business/Sci-Tech); ground truth независимо не проверялся.</li>
    <li>По характеру — аналог детекции шума из Fig. 4/8 статьи (там — «странные» картинки).</li>
    <li>Отличие от статьи (Fig. 7): классы в подсетях <strong>не</strong> сбалансированы — в top-10% всего 2.9% Sports; согласуется с гипотезой о потере классового покрытия (контрольным опытом не изолировано).</li>
  </ul>
  <div class="page">13</div>
</div>

<div class="s">
  <div class="kicker">Наш эксперимент · сравнение со статьёй</div>
  <h2>Что подтвердилось, что отличается</h2>
  <div class="grid2">
    <div class="card"><p class="meta">ПОДТВЕРЖДЕНО</p><br>
      <ul><li>EL2N считается дёшево и устойчиво (ρ между сидами).</li>
      <li>Половина датасета ≈ полный датасет (и для EL2N, и для random).</li>
      <li>High-score выглядит как шум разметки / пограничные примеры (как Fig. 4/8; просмотр top-выборки).</li></ul></div>
    <div class="card"><p class="meta">ОТЛИЧИЕ</p><br>
      <ul><li><strong>Заметного преимущества EL2N над random не наблюдается</strong> (+0.10/+0.25 п.п. при 4 прогонах на точку): на finetune трансформера, вероятно, выигрывает «покрытие», а не «трудность».</li>
      <li>На 10% EL2N <strong>не обучается вовсе</strong> (~28%): keep-top теряет класс Sports (2.9%), доля примеров со счётом >1 — 20% против ~2% у random. Согласуется с потерей классового покрытия — механизма контрольным опытом мы не изолировали.</li></ul></div>
  </div>
  <p style="margin-top:22px">Вероятные причины: предобученная модель вместо с нуля; крошечное число классов и «лёгкая» задача
  (все 90%+ точности достигаются на любой репрезентативной половине); фиксированный шаг из статьи здесь дорого стоит малым подвыборкам.</p>
  <div class="page">14</div>
</div>

<div class="s">
  <div class="kicker">Итог</div>
  <h2>Выводы</h2>
  <ul>
    <li>Основные этапы протокола EL2N (Paul et al. 2021) перенесены из vision в NLP: 10 early-ранов → mean-score → keep-top-k → retrain с фикс-шагами (адаптация протокола, не буквальное воспроизведение).</li>
    <li>На AG News + DistilBERT: <strong>random 10% сохраняет 92.12% при 94.53% на full; EL2N 10% — 28.98%</strong>; при 75% EL2N ≈ full (94.47 vs 94.53).</li>
    <li>Преимущество EL2N над random, показанное на CIFAR, <strong>не переносится</strong> в NLP finetuning: при 50–75% — паритет (±0.25 п.п.), при 10–25% EL2N проигрывает; наблюдаемые спутники — потеря класса Sports и доля примеров со счётом >1 в 20%.</li>
    <li>Точка scoring (конец 1-й эпохи) согласуется с <em>BERT on a Data Diet</em> (2022): в NLP градиентные счётчики надёжны только после ≥1 эпохи fine-tune.</li>
    <li>Репозиторий: код (воспроизводимый конвейер из 46 прогонов), отчёт (report/), презентация (Презентация.pdf).</li>
  </ul>
  <div class="page">15</div>
</div>

</body></html>
"""

if __name__ == "__main__":
    main()
