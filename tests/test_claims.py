"""Claims: jede Zahl der README steht hier als wörtliche Tabellenzeile, Band oder Ungleichung (DEMO-PLAYBOOK §11a).

Zahlen-Tabelle der README -> Test:

| README | Test |
|---|---|
| Raster ohne und mit Rauschen, Lernkurve, Strafe, Schritte, Stabilitätsprobe, Zufallswald-Tabelle | `test_readme_contains_every_generated_table_line` (wörtlich aus den JSON-Dateien erzeugt) |
| Bestes Regret, schlechteste Prognose (Grad 6, n = 5000) | `test_best_regret_with_the_worst_forecast` |
| Grad 1: nichts zu gewinnen | `test_linear_truth_leaves_nothing_to_win` |
| Wo gewinnt SPO+ (ohne und mit Rauschen) | `test_where_spo_plus_wins_without_noise`, `test_where_spo_plus_wins_with_noise` |
| Quadratfehler in den 13 Zellen mit besserem Regret (12 auf der angezeigten Stelle) | `test_forecast_error_is_not_lower_where_the_regret_is_lower` |
| Lücke zum Orakel (31 %, 42 %) | `test_share_of_the_gap_to_the_oracle` |
| Wenig Daten, Differenzen, Stabilitätsprobe, Schritte | `test_small_data_with_noise` |
| Strafe und früher Abbruch | `test_penalty_and_early_stopping_do_not_help_with_noise` |
| Lernkurve (99 %, Spanne) | `test_learning_curve_is_flat_after_25_steps_without_noise` |
| Korrekturen gegenüber dem Plan (Vormessungszahlen) | `test_plan_numbers_versus_this_measurement` |
| Vormessung: Zufallswald, Kern-Ridge | `test_frozen_forest_cells_hold_the_numbers_of_the_pre_measurement`, `test_kernel_ridge_numbers_of_the_pre_measurement` |
| Messung dieser Demo (Seeds, Schritte, Lernrate) | `test_band_file_describes_the_documented_setup`, `test_live_run_reproduces_the_bands` |
| Löser- und SPO+-Prüfungen der README (Scratch-Läufe) | nicht im Repo: Scratch-Läufe mit 300 bzw. 2000 Instanzen; im Repo die kleine Fassung in `test_oracle_spo_kuerzester_weg.py` |

Trainierte Zahlen hängen von der Plattform ab: gegen die Bänder wird in Ungleichungen und mit Toleranz geprüft, exakt nur das Handbeispiel (`test_losses.py`).
"""

from pathlib import Path

import numpy as np
import pytest

import readme_tables as T
import skw_bands as B
import skw_constants as C
from skw_evaluation import de
from skw_runs import RunSpec, run_world

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
GRID = [(d, n, e) for e in C.NOISE_CHOICES for d in C.DEG_CHOICES for n in C.N_CHOICES]


def _med(deg, n, noise, key):
    return B.cell(deg, n, noise)["summary"][key]["median"]


def _diff_median(deg, n, noise):
    c = B.cell(deg, n, noise)
    return float(np.median(np.array(c["regret_spo"]) - np.array(c["regret_kq"])))


def test_readme_contains_every_generated_table_line():
    blocks = [T.main_table(0.0), T.main_table(0.5), T.forest_table(), T.steps_table(), T.penalty_table(), T.stability_table(), T.curve_table()]
    missing = [line for lines in blocks for line in lines if line not in README]
    assert not missing, missing[:3]


def test_band_file_describes_the_documented_setup():
    meta = B.own()["meta"]
    assert meta["seeds"] == list(C.BAND_SEEDS) and meta["steps"] == C.STEPS and meta["lr"] == C.LR and meta["n_test"] == C.N_TEST
    assert len(B.own()["cells"]) == 24 and all(len(c["regret_kq"]) == 20 for c in B.own()["cells"])
    assert "Zufalls-Seeds 0 bis 19" in README and "300 Adam-Schritte" in README and "Lernrate 0,05" in README


def test_best_regret_with_the_worst_forecast():
    s = B.cell(6, 5000, 0.0)["summary"]
    assert s["regret_spo"]["median"] < 0.3 * s["regret_kq"]["median"]                   # "ein Viertel"
    assert s["regret_spo"]["median"] / s["regret_kq"]["median"] == pytest.approx(0.25, abs=0.02)
    assert s["mse_spo"]["median"] / s["mse_kq"]["median"] == pytest.approx(1.8, abs=0.1)  # "1,8-fachen Quadratfehler"
    assert (de(s["regret_spo"]["median"]), de(s["regret_kq"]["median"]), de(s["mse_spo"]["median"], 2), de(s["mse_kq"]["median"], 2)) == ("0,5", "2,0", "1,39", "0,76")
    assert B.cell(6, 5000, 0.0)["spo_besser_als_kq"] == 20
    w = B.wald_cell(6, 5000, 0.0)
    assert w["mse_wald"] == 0.29 and w["regret_wald"]["median"] == 0.8 and w["mse_wald"] < s["mse_kq"]["median"]   # der Wald hat den kleinsten Quadratfehler, aber mehr Regret als SPO+
    assert w["regret_wald"]["median"] > s["regret_spo"]["median"]
    assert "Median 0,5 gegen 2,0 %" in README and "(1,39 gegen 0,76)" in README and "0,29" in README


def test_linear_truth_leaves_nothing_to_win():
    for n in C.N_CHOICES:
        c = B.cell(1, n, 0.0)
        assert max(c["regret_kq"]) < 0.05 and c["spo_besser_als_kq"] == 0            # KQ: 0,0 % in jedem der 20 Seeds; SPO+ nie besser
        assert c["summary"]["regret_kq"]["median"] == pytest.approx(0.0, abs=0.05)
    assert "Bei Grad 1 ist SPO+ in keiner Zelle ohne Rauschen besser (0 von 20 Seeds)" in README


def test_where_spo_plus_wins_without_noise():
    counts = {(d, n): B.cell(d, n, 0.0)["spo_besser_als_kq"] for d in C.DEG_CHOICES for n in C.N_CHOICES}
    assert counts[(4, 100)] == 18 and counts[(6, 100)] == 20
    assert all(counts[(d, n)] == 20 for d in (4, 6) for n in (1000, 5000))
    assert all(counts[(1, n)] == 0 for n in C.N_CHOICES)
    assert counts[(2, 100)] == 0 and counts[(2, 1000)] == 20 and counts[(2, 5000)] == 20
    assert "n = 100: Grad 4 in 18, Grad 6 in 20 von 20 Seeds; n = 1000 und 5000: 20 von 20" in README


def test_where_spo_plus_wins_with_noise():
    counts = {(d, n): B.cell(d, n, 0.5)["spo_besser_als_kq"] for d in C.DEG_CHOICES for n in C.N_CHOICES}
    assert counts[(6, 1000)] == 20 and counts[(6, 5000)] == 20
    assert counts[(4, 1000)] == 13 and counts[(4, 5000)] == 16
    assert max(counts[(1, n)] for n in C.N_CHOICES) == 13 and max(counts[(2, n)] for n in C.N_CHOICES) == 9      # "bei Grad 1 und 2 höchstens 13 von 20"
    assert all(counts[(d, n)] < 20 for d in (1, 2, 4) for n in C.N_CHOICES)
    assert "bei Grad 4 sind es 13 und 16 von 20, bei Grad 1 und 2 höchstens 13 von 20" in README


def test_forecast_error_is_not_lower_where_the_regret_is_lower():
    cells = [(d, n, e) for (d, n, e) in GRID if _med(d, n, e, "regret_spo") < _med(d, n, e, "regret_kq")]
    assert len([c for c in cells if c[2] == 0.0]) == 8 and len([c for c in cells if c[2] == 0.5]) == 5
    assert all(_med(d, n, e, "mse_spo") >= _med(d, n, e, "mse_kq") for d, n, e in cells)
    assert all(_med(d, n, 0.0, "mse_spo") > _med(d, n, 0.0, "mse_kq") for d, n, e in cells if e == 0.0)             # ohne Rauschen in allen 8 höher
    assert de(_med(2, 1000, 0.0, "mse_kq"), 2) == "0,00" and de(_med(2, 1000, 0.0, "mse_spo"), 2) == "0,13"
    shown = [c for c in cells if round(_med(*c, "regret_spo"), 1) < round(_med(*c, "regret_kq"), 1)]                   # auf der angezeigten Stelle: 12 Zellen
    assert len([c for c in shown if c[2] == 0.0]) == 8 and len([c for c in shown if c[2] == 0.5]) == 4
    assert [c for c in cells if c not in shown] == [(1, 5000, 0.5)]                                                   # dort gleichauf (17,8 %)
    assert "In 12 Zellen der beiden Raster (8 ohne, 4 mit Rauschen)" in README
    assert "In einer weiteren Zelle (Grad 1, n = 5000, mit Rauschen) liegen die Median-Regrets auf der angezeigten Stelle gleichauf (17,8 %)" in README


def test_share_of_the_gap_to_the_oracle():
    for n, share in ((1000, 31), (5000, 42)):
        o, k, p = (_med(6, n, 0.5, key) for key in ("regret_orakel", "regret_kq", "regret_spo"))
        assert 100 * (k - p) / (k - o) == pytest.approx(share, abs=0.5)
    assert "schließt 31 % der Lücke" in README and "sind es 42 %" in README
    assert (de(_med(6, 1000, 0.5, "regret_orakel")), de(_med(6, 1000, 0.5, "regret_kq")), de(_med(6, 1000, 0.5, "regret_spo"))) == ("10,5", "13,1", "12,3")


def test_small_data_with_noise():
    c = {d: B.cell(d, 100, 0.5) for d in C.DEG_CHOICES}
    assert [c[d]["spo_besser_als_kq"] for d in C.DEG_CHOICES] == [1, 0, 0, 10]
    assert all(c[d]["summary"]["regret_spo"]["median"] > c[d]["summary"]["regret_kq"]["median"] for d in (1, 2, 4))
    assert [de(_diff_median(d, 100, 0.5)) for d in (1, 2, 4)] == ["1,1", "1,0", "0,9"]
    assert "Differenz 1,1, 1,0 und 0,9 Punkte im Median der Paare" in README
    stab = {e["deg"]: e for e in B.own()["stabil"]}
    kq6, sp6 = np.array(stab[6]["regret_kq"]), np.array(stab[6]["regret_spo"])
    assert len(kq6) == 100 and stab[6]["seeds"] == list(range(20, 120))
    assert int((sp6 < kq6).sum()) == 29 and float(np.median(sp6 - kq6)) == pytest.approx(0.5, abs=0.05) and float(np.median(sp6 - kq6)) > 0
    kq4, sp4 = np.array(stab[4]["regret_kq"]), np.array(stab[4]["regret_spo"])
    assert int((sp4 < kq4).sum()) == 6 and float(np.median(sp4 - kq4)) > 0.5
    assert "SPO+ besser in 29 von 100 und im Median 0,5 Punkte schlechter" in README
    steps = {(e["deg"], int(k)): v for e in B.own()["schritte"] for k, v in e["varianten"].items()}
    for deg in (4, 6):
        for k in (300, 600, 1000):
            v = steps[(deg, k)]
            assert v["summary_spo"]["median"] > v["summary_kq"]["median"]                # in keiner Schrittzahl unter KQ (Median)
    assert [sum(a < b for a, b in zip(steps[(6, k)]["regret_spo"], steps[(6, k)]["regret_kq"])) for k in (300, 600, 1000)] == [10, 10, 9]
    assert [sum(a < b for a, b in zip(steps[(4, k)]["regret_spo"], steps[(4, k)]["regret_kq"])) for k in (300, 600, 1000)] == [0, 0, 0]
    assert "(10 und 9 von 20)" in README


def test_penalty_and_early_stopping_do_not_help_with_noise():
    p5, p0 = B.penalty(0.5)["varianten"], B.penalty(0.0)["varianten"]
    kq5 = p5["ohne"]["summary_kq"]["median"]
    assert all(p5[k]["summary_spo"]["median"] > kq5 for k in ("ohne", "l2=0,1", "l2=1", "50 Schritte"))                    # kein Fall unter KQ
    assert 14.0 < kq5 < 14.1 and all(14.1 < p5[k]["summary_spo"]["median"] < 14.4 for k in ("ohne", "l2=0,1", "l2=1", "50 Schritte"))
    kq0 = p0["ohne"]["summary_kq"]["median"]
    assert p0["l2=0,1"]["summary_spo"]["median"] < p0["ohne"]["summary_spo"]["median"] < kq0              # λ = 0,1 hilft ohne Rauschen etwas
    assert p0["l2=1"]["summary_spo"]["median"] > p0["ohne"]["summary_spo"]["median"]                      # λ = 1 schadet ohne Rauschen (gegen SPO+ ohne Strafe)
    assert p0["l2=1"]["summary_spo"]["median"] < kq0
    assert (de(p0["l2=0,1"]["summary_spo"]["median"], 2), de(p0["ohne"]["summary_spo"]["median"], 2)) == ("1,04", "1,15")
    assert "λ = 0,1 mit 1,04 knapp unter 1,15 % (Unterschied klein, nicht auf Signifikanz geprüft) und damit weniger deutlich als in der Vormessung (1,02 gegen 1,34 %)" in README


def test_learning_curve_is_flat_after_25_steps_without_noise():
    c0, c5 = B.curve(0.0), B.curve(0.5)
    share = np.asarray(c0["anteil_gewinn_bei_25"], dtype=float)
    assert len(share) == 20 and 0.97 < np.median(share) < 1.01 and share.min() > 0.9 and share.max() < 1.1
    assert T.share_line() == "99 % (Minimum 93 %, Maximum 104 %)"
    sh5 = np.asarray(c5["anteil_gewinn_bei_25"], dtype=float)
    assert (de(100 * sh5.min(), 0), de(100 * sh5.max(), 0)) == ("74", "197")                      # mit Rauschen streut der Anteil stark
    s0, s5 = c0["summary"], c5["summary"]
    assert s0["0"]["median"] > 3 * s0["25"]["median"] and s0["25"]["median"] == pytest.approx(s0["300"]["median"], abs=0.1)
    assert (de(s5["0"]["median"]), de(s5["300"]["median"])) == ("13,1", "12,2")
    assert "Median 99 % des Gewinns, Spanne 93 bis 104 %" in README


def test_plan_numbers_versus_this_measurement():
    """Korrekturen gegenüber dem Plan: die neuen Werte (Grad 6, n = 1000) und die Differenz zu den Plan-Werten (Vormessung: 1,9 / 0,6 und 12,5 / 11,8)."""
    assert (de(_med(6, 1000, 0.0, "regret_kq")), de(_med(6, 1000, 0.0, "regret_spo"))) == ("2,0", "0,5")
    assert (de(_med(6, 1000, 0.5, "regret_kq")), de(_med(6, 1000, 0.5, "regret_spo"))) == ("13,1", "12,3")
    assert B.cell(6, 1000, 0.5)["spo_besser_als_kq"] == 20
    w = B.wald_cell(6, 1000, 0.0)
    assert (w["regret_kq"]["median"], w["regret_spo600"]["median"]) == (1.9, 0.6)                # Zahlen des Plans stehen in der Vormessung
    w5 = B.wald_cell(6, 1000, 0.5)
    assert (w5["regret_kq"]["median"], w5["regret_spo600"]["median"], w5["spo600_besser_als_kq"]) == (12.5, 11.8, 18)
    assert "KQ 1,9 → 2,0 %, SPO+ 0,6 → 0,5 %" in README and "KQ 12,5 → 13,1 %, SPO+ 11,8 → 12,3 %" in README
    w100 = B.wald_cell(6, 100, 0.5)
    assert (w100["spo600_besser_als_kq"], w100["regret_spo600"]["median"], w100["regret_kq"]["median"]) == (2, 14.5, 13.4)    # Plan: "2 von 20", 14,5 gegen 13,4


def test_frozen_forest_cells_hold_the_numbers_of_the_pre_measurement():
    meta = B.wald()
    assert "out_eo_v1_sp.txt" in meta["quelle"] and "NICHT live" in meta["hinweis"] and len(meta["cells"]) == 24
    # Spot-Check gegen Zahlen der Rohausgabe (Regret Median; Quadratfehler Median KQ / SPO+ / Wald)
    cells = {(c["deg"], c["n"], c["noise"]): c for c in meta["cells"]}
    c = cells[(6, 5000, 0.0)]
    assert (c["regret_kq"]["median"], c["regret_spo600"]["median"], c["regret_wald"]["median"]) == (1.8, 0.5, 0.8)
    assert (c["mse_kq"], c["mse_spo600"], c["mse_wald"]) == (0.65, 1.31, 0.29)
    c = cells[(6, 5000, 0.5)]
    assert (c["regret_kq"]["median"], c["regret_spo600"]["median"], c["regret_wald"]["median"]) == (12.2, 11.6, 11.5)
    c = cells[(1, 100, 0.0)]
    assert (c["regret_wald"]["median"], c["regret_wald"]["min"], c["regret_wald"]["max"]) == (0.7, 0.5, 1.1)
    assert "Wald (11,5 %) und SPO+ (11,6 %) in der Vormessung gleichauf, KQ liegt bei 12,2 %" in README


def test_kernel_ridge_numbers_of_the_pre_measurement():
    k = C.VORMESSUNG_KERN
    assert (k["kern_eto"], k["spo"], k["kq"], k["seeds"]) == (0.04, 0.55, 1.87, 20)
    assert "Kern-Ridge 0,04 %, SPO+ 0,55 %, KQ 1,87 % im Median" in README
    assert "11,33 % gegen 11,84 % (SPO+) und 12,47 % (KQ)" in README


def test_live_run_reproduces_the_bands():
    """Drei Seeds der Zelle Grad 6, ohne Rauschen, n = 1000 neu gerechnet: jeder Wert liegt (mit Toleranz für die Plattform) im Band der Messung."""
    cell = B.cell(6, 1000, 0.0)
    for seed in (0, 1, 2):
        run = run_world(RunSpec(deg=6, noise=0.0, n=1000, seed=seed), snapshot_steps=(), keep_cases=False)
        kq, spo = 100 * run.regret["kq"], 100 * run.regret["spo"]
        assert kq == pytest.approx(cell["regret_kq"][seed], abs=0.25)
        assert spo == pytest.approx(cell["regret_spo"][seed], abs=0.25)
        assert spo < kq                                                                      # Ungleichung dieser Zelle (20 von 20 Seeds)
        assert run.mse["spo"] > run.mse["kq"]                                                # Quadratfehler steigt (Median 0,77 → 1,49)
        assert abs(run.regret["orakel"]) < 1e-12
