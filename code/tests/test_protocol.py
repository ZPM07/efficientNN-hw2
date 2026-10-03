"""Protocol unit tests: EL2N formula, subset selection, fixed-step loop.

These guard the parts of the reproduction that must match the paper/official
code exactly; they run on CPU without model downloads.
"""

import numpy as np
import pytest
import torch

from el2n_nlp import config
from el2n_nlp.scores import el2n_from_probs, mean_scores
from el2n_nlp.subsets import keep_top_scores, random_subset


# ---------------------------------------------------------------- EL2N score


def test_el2n_formula_matches_definition():
    # EL2N(x,y) = || softmax(logits) - onehot(y) ||_2
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
    labels = torch.tensor([0, 0])  # second example is misclassified
    p = torch.softmax(logits, dim=-1)
    out = el2n_from_probs(p, labels)
    y = np.eye(4)[[0, 0]]
    expected = np.linalg.norm(p.numpy() - y, axis=1)
    assert np.allclose(out, expected, atol=1e-6)


def test_el2n_perfect_prediction_is_near_zero():
    logits = torch.full((3, 4), -20.0)
    logits[torch.arange(3), torch.tensor([1, 2, 3])] = 20.0
    p = torch.softmax(logits, dim=-1)
    out = el2n_from_probs(p, torch.tensor([1, 2, 3]))
    assert np.all(out < 1e-6)


def test_el2n_uniform_prediction_is_max():
    # uniform p over C classes: error = onehot - 1/C, norm = sqrt((C-1)/C^2)*sqrt(2)≈...
    C = 4
    p = torch.full((2, C), 1.0 / C)
    out = el2n_from_probs(p, labels=torch.tensor([0, 1]))
    single = np.linalg.norm(np.eye(C)[0] - np.full(C, 1.0 / C))
    assert np.allclose(out, single, atol=1e-6)


def test_mean_scores_averages_runs():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 1.0])
    m = mean_scores([a, b])
    assert np.allclose(m, [2.0, 3.0, 2.0])


# ---------------------------------------------------------- subset selection


def test_keep_top_scores_selects_global_max():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    idx = keep_top_scores(scores, 2)
    assert set(idx.tolist()) == {1, 3}
    assert idx.tolist() == sorted(idx.tolist())  # sorted indices, like official code


def test_keep_top_scores_full_size_keeps_all():
    scores = np.random.RandomState(0).rand(100)
    idx = keep_top_scores(scores, 100)
    assert set(idx.tolist()) == set(range(100))


def test_random_subset_deterministic_and_disjoint_runs():
    s1 = random_subset(1000, 100, seed=7)
    s2 = random_subset(1000, 100, seed=7)
    s3 = random_subset(1000, 100, seed=8)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    assert len(set(s1.tolist())) == 100
    assert s1.min() >= 0 and s1.max() < 1000


# ------------------------------------------------------------ training loop


def test_infinite_epoch_cycling_fixed_steps():
    from el2n_nlp.trainer import shuffled_batches

    n, bs = 500, 100
    idxs = np.arange(n)
    batches = list(shuffled_batches(idxs, bs, total_steps=25, seed=0))
    assert len(batches) == 25  # fixed step count regardless of subset size
    assert all(b.shape == (bs,) for b in batches)
    # tail dropping within the first epoch: epoch 1 covers exactly 5 full batches
    first_epoch = np.concatenate(batches[:5])
    assert len(set(first_epoch.tolist())) == 500


def test_cycler_reshuffles_every_epoch():
    from el2n_nlp.trainer import shuffled_batches

    n, bs = 40, 10
    batches = list(shuffled_batches(np.arange(n), bs, total_steps=8, seed=3))
    ep1 = set(batches[0].tolist()) | set(batches[1].tolist())
    ep1 |= set(batches[2].tolist()) | set(batches[3].tolist())
    ep2 = set(batches[4].tolist()) | set(batches[5].tolist())
    ep2 |= set(batches[6].tolist()) | set(batches[7].tolist())
    assert ep1 == ep2 == set(range(40))
    assert batches[0].tolist() != batches[4].tolist()


# ---------------------------------------------------------------- schedule


def test_lr_schedule_warmup_then_linear_decay():
    from el2n_nlp.trainer import lr_at

    total, warmup, base = 1000, 100, 5e-5
    assert lr_at(0, total, warmup, base) == pytest.approx(0.0)
    assert lr_at(warmup, total, warmup, base) == pytest.approx(base)
    assert lr_at(total, total, warmup, base) == pytest.approx(0.0)
    mid = lr_at((warmup + total) // 2, total, warmup, base)
    assert 0 < mid < base
    # monotone decrease after warmup
    vals = [lr_at(s, total, warmup, base) for s in range(warmup, total + 1, 50)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


# ----------------------------------------------------------------- recipe


def test_recipe_seed_domains_disjoint():
    scoring = set(config.scoring_seeds())
    model = set(config.retrain_model_seeds())
    subset = set(config.random_subset_seeds())
    assert not scoring & model
    assert not (scoring | model) & subset


def test_total_steps_is_three_full_epochs():
    assert config.STEPS_PER_EPOCH == config.NUM_TRAIN // config.BATCH_SIZE
    assert config.TOTAL_STEPS == 3 * config.STEPS_PER_EPOCH


def test_score_mode_stops_early_by_default():
    from el2n_nlp.trainer import resolve_score_stop

    assert resolve_score_stop(None) == config.SCORING_STOP_STEP == config.STEPS_PER_EPOCH
    assert resolve_score_stop(None) < config.TOTAL_STEPS
    assert resolve_score_stop(10) == 10
