"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, Nullen, kleinstes Gitter, Grenzen der Regler."""

import numpy as np
import pytest

import skw_evaluation as ev
import skw_losses as L
import skw_runs as R
import skw_training as T
from skw_grid import Grid, PathSet
from skw_world import WorldSpec, make_dataset


def test_exactly_two_nodes_per_side_is_the_smallest_grid(grid2):
    assert grid2.d == 4 and grid2.paths().shape == (2, 4)
    W, z = grid2.solve(np.array([[1.0, 1.0, 1.0, 1.0]]))
    assert W.tolist() == [[1, 0, 1, 0]] and z.tolist() == [2.0]       # Gleichstand: rechts vor unten


def test_world_run_on_the_smallest_grid_does_not_break():
    run = R.run_world(R.RunSpec(h=2, deg=2, noise=0.0, n=20, seed=1, steps=10, n_test=30), snapshot_steps=(0, 10))
    assert run.grid.d == 4 and run.data.C_test.shape == (30, 4)
    assert run.regret["orakel"] == pytest.approx(0.0, abs=1e-12)


def test_training_with_a_single_case_runs_and_never_gets_worse_on_the_training_loss(grid5):
    data = make_dataset(WorldSpec(deg=4), grid5.d, 1, 5, seed=3)
    kq = T.fit_least_squares(data.X_train, data.C_train)
    res = T.train_spo_plus(grid5, data.X_train, data.C_train, steps=20, init=kq)
    assert T.spo_plus_loss_mean(grid5, data.X_train, data.C_train, res.coef) <= T.spo_plus_loss_mean(grid5, data.X_train, data.C_train, kq) + 1e-12


def test_training_with_two_cases_runs(grid5):
    data = make_dataset(WorldSpec(deg=4), grid5.d, 2, 5, seed=3)
    res = T.train_spo_plus(grid5, data.X_train, data.C_train, steps=10)
    assert res.coef.shape == (6, 40) and np.all(np.isfinite(res.coef))


def test_all_zero_costs_and_all_zero_forecast_are_ties_with_regret_zero(grid5):
    C = np.zeros((3, 40))
    assert np.all(L.spo_loss(grid5, C, C) == 0)
    assert np.all(L.spo_plus_loss(grid5, C, C) == 0)
    assert np.all(L.spo_plus_subgradient(grid5, C, C) == 0)      # w*(0) = w(0): derselbe Weg, die Differenz ist 0
    W, z = grid5.solve(C)
    assert np.all(z == 0) and np.array_equal(W[0], grid5.paths()[0])


def test_negative_forecasts_are_allowed_and_give_a_valid_path(grid5):
    rng = np.random.default_rng(0)
    C = rng.uniform(1, 3, size=(20, 40))
    Chat = rng.uniform(-5, -1, size=(20, 40))
    assert np.all(L.spo_loss(grid5, C, Chat) >= -1e-9)
    assert np.all(grid5.solve(Chat)[0].sum(axis=1) == 8)


def test_hand_example_at_the_slider_corners():
    for c1 in (0.0, 8.0):
        for c2 in (0.0, 8.0):
            h = ev.hand_numbers(c1, c2)
            assert h["spo"] >= 0 and h["spo_plus"] >= h["spo"] - 1e-12 and h["chosen"] in (1, 2)
    assert ev.hand_numbers(0.0, 0.0)["chosen"] == 1                  # Gleichstand: Weg 1
    assert ev.hand_numbers(8.0, 8.0)["spo"] == 0.0
    tie = ev.hand_numbers(3.5, 3.5)                                  # Gleichstand, Weg 1 ist der richtige
    assert tie["chosen"] == 1 and tie["spo"] == 0.0 and tie["spo_plus"] == 2.0   # max(3 - 7, 5 - 7) + 7 - 3 = 2


def test_one_path_only_set_has_no_decision_to_make():
    one = PathSet(np.array([[1.0, 0.0, 1.0]]))
    W, z = one.solve(np.array([[2.0, 9.0, 3.0], [0.0, 0.0, 0.0]]))
    assert W.tolist() == [[1, 0, 1], [1, 0, 1]] and z.tolist() == [5.0, 0.0]


def test_extreme_seeds_run():
    for seed in (0, 999999):
        run = R.run_world(R.RunSpec(deg=6, noise=0.5, n=100, seed=seed, steps=5), snapshot_steps=(), keep_cases=False)
        assert np.isfinite(run.regret["kq"]) and np.isfinite(run.regret["spo"])


def test_case_numbers_first_and_last_selectable_case():
    run = R.run_world(R.RunSpec(deg=6, noise=0.0, n=100, seed=0, steps=10))
    for case in (1, 100):
        cn = ev.case_numbers(run, case)
        assert cn["best"] > 0 and cn["kq"]["extra_pct"] >= -1e-9 and cn["spo"]["extra_pct"] >= -1e-9
        assert cn["w_best"].sum() == 8 and cn["kq"]["w"].sum() == 8 and cn["spo"]["w"].sum() == 8


def test_grid_rows_and_band_rows_have_the_expected_shape():
    rows = ev.grid_rows(6)
    assert [(r["n"], r["noise"]) for r in rows] == [(100, 0.0), (1000, 0.0), (5000, 0.0), (100, 0.5), (1000, 0.5), (5000, 0.5)]
    br = ev.band_rows(6, 1000, 0.0)
    assert [r["Verfahren"].split(" (")[0] for r in br] == ["Orakel", "KQ", "SPO+", "Zufallswald"]
    assert br[0]["mse"] is None and br[3]["quelle"].startswith("Vormessung")


def test_grid_class_rejects_a_wrong_cost_width():
    with pytest.raises(ValueError):
        Grid(5).solve(np.ones((1, 41)))


def test_differing_cases_lists_exactly_the_cases_with_different_paths_and_next_case_wraps():
    run = R.run_world(R.RunSpec(deg=6, noise=0.0, n=1000, seed=0))
    cases = ev.differing_cases(run)
    assert cases == sorted(cases) and all(1 <= c <= 100 for c in cases) and 0 < len(cases) < 100
    Wk, _ = run.grid.solve(T.predict(run.coef_kq, run.data.X_test[:100]))
    Ws, _ = run.grid.solve(T.predict(run.coef_spo, run.data.X_test[:100]))
    assert cases == [i + 1 for i in range(100) if not np.array_equal(Wk[i], Ws[i])]
    assert ev.next_case(cases, 0) == cases[0] and ev.next_case(cases, cases[-1]) == cases[0]       # am Ende von vorn
    assert ev.next_case(cases, cases[0]) == cases[1] > cases[0]
    assert ev.next_case([], 5) is None and ev.next_case([7], 7) == 7 and ev.next_case([7], 3) == 7
