"""Recipe constants and seed bookkeeping for the reproduction.

Protocol mirrors Paul et al. 2021 (data_diet) with the vision-specific parts
swapped for the NLP analogue:

- scores are averaged over several independent early-training runs;
- subsets keep the examples with the *largest* mean EL2N score (global top-k);
- all retraining runs use the same fixed number of optimizer steps and the
  same LR schedule, independent of subset size (paper, Appendix B.2);
- retraining seeds are disjoint from the seeds used to compute scores.
"""

from pathlib import Path

# ---------------------------------------------------------------- dataset

DATASET = "fancyzhx/ag_news"
MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_TRAIN = 120_000
NUM_TEST = 7_600
NUM_CLASSES = 4
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# ---------------------------------------------------------------- training

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
MAX_LEN = 128  # covers 99% of AG News examples (p99 = 124 tokens)
EPOCHS_FULL = 3  # epochs over the FULL training set; total steps are fixed
STEPS_PER_EPOCH = NUM_TRAIN // BATCH_SIZE  # 937, tail batch dropped
TOTAL_STEPS = EPOCHS_FULL * STEPS_PER_EPOCH  # 2811 — same for every run
WARMUP_STEPS = int(0.05 * TOTAL_STEPS)
LR = 5e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EVAL_EVERY = 200

# ---------------------------------------------------------------- scoring

SCORE_RUNS = 10  # independent initializations averaged, as in the paper
# Score checkpoint: one full epoch into the (full-length) schedule, i.e. the
# LR trajectory seen at this point matches a run that would train all 3.
SCORING_STOP_STEP = STEPS_PER_EPOCH

# ------------------------------------------------------------------ design

SIZES_PCT = [10, 25, 50, 75]  # retained fraction of the training set, percent
RETRAIN_RUNS = 4  # independent runs per point (model init + SGD/dropout noise)


def subset_size(pct: int) -> int:
    return round(NUM_TRAIN * pct / 100)


# ------------------------------------------------------------------- seeds


def scoring_seeds() -> list[int]:
    """Seeds for the parent runs used to compute EL2N scores."""
    return [1000 + i for i in range(SCORE_RUNS)]


def retrain_model_seed(mode: str, pct: int, run: int) -> int:
    base = {"full": 2000, "el2n": 2100, "random": 2500}[mode]
    return base + pct * 10 + run


def retrain_model_seeds() -> list[int]:
    seeds = [retrain_model_seed("full", 100, r) for r in range(RETRAIN_RUNS)]
    for mode in ("el2n", "random"):
        for pct in SIZES_PCT:
            for r in range(RETRAIN_RUNS):
                seeds.append(retrain_model_seed(mode, pct, r))
    return seeds


def random_subset_seed(pct: int, run: int) -> int:
    # each random baseline run draws its own subset (official protocol)
    return 5000 + pct * 10 + run


def random_subset_seeds() -> list[int]:
    return [random_subset_seed(p, r) for p in SIZES_PCT for r in range(RETRAIN_RUNS)]


# ------------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SCORES_DIR = RESULTS_DIR / "scores"
SUBSETS_DIR = RESULTS_DIR / "subsets"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"


def score_path(seed: int) -> Path:
    return SCORES_DIR / f"scores_seed{seed}.npy"


def run_tag(mode: str, pct: int, run: int) -> str:
    return f"{mode}_{pct}pct_run{run}"


def run_path(mode: str, pct: int, run: int) -> Path:
    return RUNS_DIR / f"{run_tag(mode, pct, run)}.json"


def el2n_subset_path(pct: int) -> Path:
    return SUBSETS_DIR / f"el2n_{pct}pct.npy"


def random_subset_path(pct: int, run: int) -> Path:
    return SUBSETS_DIR / f"random_{pct}pct_run{run}.npy"
