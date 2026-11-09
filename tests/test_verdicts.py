"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text). Die Schwellen vergleichen die ANGEZEIGTEN Werte (eine Stelle bei Prozent, zwei beim Quadratfehler), damit der Satz nie
etwas anderes sagt als die Zahlen daneben."""

import skw_evaluation as ev


def _S(median, lo=None, hi=None):
    return {"median": median, "min": median if lo is None else lo, "max": median if hi is None else hi}


def test_de_uses_a_decimal_comma_and_hides_negative_zero():
    assert ev.de(2.0) == "2,0" and ev.de(0.04, 2) == "0,04" and ev.de(-1.26, 1) == "-1,3"
    assert ev.de(-0.0001, 1) == "0,0" and ev.de(-0.0, 1) == "0,0"
    assert ev.de(12.346, 2) == "12,35"


def test_hand_numbers_match_the_hand_calculation():
    h = ev.hand_numbers(4.0, 3.0)
    assert h["chosen"] == 2 and h["best"] == 1
    assert (h["chosen_cost"], h["best_cost"]) == (5.0, 3.0)
    assert (h["spo"], h["spo_plus"], h["sq"]) == (2.0, 4.0, 5.0)
    assert h["subgradient"] == (2.0, -2.0)


def test_hand_line_hits_t0_and_t1_and_uses_a_fallback_direction_for_a_perfect_forecast():
    line = ev.hand_line(4.0, 3.0)
    assert 0.0 in line["t"] and 1.0 in line["t"] and not line["degenerate"] and line["direction"] == (1.0, -2.0)
    i1 = list(line["t"]).index(1.0)
    assert (line["spo"][i1], line["spo_plus"][i1], line["sq"][i1]) == (2.0, 4.0, 5.0)
    perfect = ev.hand_line(3.0, 5.0)
    assert perfect["degenerate"] and perfect["direction"] == (1.0, -1.0)


def test_verdict_hand_wrong_decision_names_costs_regret_and_both_surrogates():
    kind, text = ev.verdict_hand(ev.hand_numbers(4.0, 3.0))
    assert kind == "warning"
    assert "Weg 2" in text and "in Wahrheit 5,0 statt 3,0" in text and "SPO-Verlust 2,0" in text
    assert "Quadratfehler ist 5,0" in text and "SPO+ liegt bei 4,0" in text


def test_verdict_hand_right_decision_with_wrong_forecast_is_a_success():
    kind, text = ev.verdict_hand(ev.hand_numbers(6.0, 7.0))     # Quadratfehler 9 + 4 = 13, Weg 1 gewinnt wie unter c
    assert kind == "success"
    assert "Weg 1" in text and "SPO-Verlust 0,0" in text and "Quadratfehler 13,0" in text and "SPO+ liegt bei 0,0" in text


def test_verdict_hand_exact_forecast_is_neutral():
    kind, text = ev.verdict_hand(ev.hand_numbers(3.0, 5.0))
    assert kind == "info" and "Quadratfehler 0,0" in text


def test_verdict_training_improvement_is_a_success_with_all_numbers():
    kind, text = ev.verdict_training(2.04, 0.52, 0.48, 140)
    assert kind == "success"
    assert "von 2,0 % (KQ-Start)" in text and "auf 0,5 % nach 25 Schritten" in text and "(Schritt 140) hat 0,5 %" in text


def test_verdict_training_compares_the_displayed_values():
    assert ev.verdict_training(2.04, 2.0, 2.01, 7)[0] == "info"       # beide zeigen 2,0
    kind, text = ev.verdict_training(2.04, 2.0, 2.06, 7)               # 2,1 gegen 2,0
    assert kind == "warning" and "2,1 % gegen 2,0 % bei KQ" in text and "von Schritt 7" in text


def test_verdict_forecast_better_decision_with_worse_forecast():
    kind, text = ev.verdict_forecast(2.04, 0.51, 0.77, 1.49, 20)
    assert kind == "success"
    assert "von 2,0 % auf 0,5 %" in text and "von 0,77 auf 1,49" in text and "Median über 20 Zufalls-Seeds" in text and "schlechteren Prognose" in text


def test_verdict_forecast_regret_down_forecast_not_worse_is_neutral():
    kind, text = ev.verdict_forecast(13.1, 12.3, 0.288, 0.284, 20)
    assert kind == "info" and "steigt dabei nicht (0,29 bei KQ, 0,28 bei SPO+" in text


def test_verdict_forecast_equal_and_worse_regret():
    kind, text = ev.verdict_forecast(2.04, 2.01, 0.1, 0.1, 20)         # beide 2,0
    assert kind == "info" and "denselben Wert (2,0 %" in text
    kind, text = ev.verdict_forecast(13.8, 14.6, 0.41, 0.55, 20)
    assert kind == "warning" and "14,6 % gegen 13,8 %" in text and "0,41 bei KQ, 0,55 bei SPO+" in text


def _row(n, noise, kq, spo, better):
    return {"n": n, "noise": noise, "kq": _S(kq), "spo": _S(spo), "besser": better, "seeds": 20, "orakel": 0.0, "wald": 0.0}


def test_verdict_grid_counts_cells_and_names_the_exceptions():
    rows = [_row(100, 0.0, 2.6, 1.2, 20), _row(1000, 0.0, 2.0, 0.5, 20), _row(5000, 0.0, 2.0, 0.5, 20),
            _row(100, 0.5, 13.8, 14.6, 0), _row(1000, 0.5, 13.1, 12.3, 20), _row(5000, 0.5, 13.0, 11.9, 20)]
    kind, text = ev.verdict_grid(4, rows)
    assert kind == "info"
    assert "Bei Grad 4 liegt der Median-Regret von SPO+ in 5 von 6 Zellen unter dem von KQ" in text
    assert "Nicht darunter: n = 100 mit Rauschen: 14,6 % gegen 13,8 % (0 von 20 Seeds mit SPO+ besser)." in text


def test_verdict_grid_without_exception_and_with_displayed_equality():
    rows = [_row(100, 0.0, 2.0, 1.0, 20)] * 6
    assert ev.verdict_grid(6, rows) == ("info", "Bei Grad 6 liegt der Median-Regret von SPO+ in 6 von 6 Zellen unter dem von KQ.")
    gleich = [_row(100, 0.0, 0.04, 0.02, 3)] * 6                       # beide zeigen 0,0: nicht darunter
    kind, text = ev.verdict_grid(1, gleich)
    assert "in 0 von 6 Zellen" in text and "ohne Rauschen: 0,0 % gegen 0,0 %" in text


def test_verdict_case_three_branches():
    assert ev.verdict_case(7, 10.0, 0.0, 0.04) == ("info", "Testfall 7: der beste Weg kostet 10,00. KQ und SPO+ wählen beide den besten Weg (Regret 0,0 %).")
    kind, text = ev.verdict_case(3, 12.5, 4.0, 1.0)
    assert kind == "success" and "kostet 12,50" in text and "Der Weg von KQ ist 4,0 % teurer, der von SPO+ 1,0 %." in text
    kind, text = ev.verdict_case(3, 12.5, 1.0, 4.0)
    assert kind == "warning" and "sein Weg ist 4,0 % teurer, der von KQ 1,0 %" in text
    assert ev.verdict_case(3, 12.5, 2.0, 2.0)[0] == "info"


def test_verdict_ranking_names_different_winners():
    rows = [
        {"Verfahren": "Orakel (kennt die mittleren Kosten)", "regret": _S(0.0), "mse": None},
        {"Verfahren": "KQ (linear)", "regret": _S(1.9), "mse": 0.65},
        {"Verfahren": "SPO+ (linear, 300 Schritte)", "regret": _S(0.5), "mse": 1.31},
        {"Verfahren": "Zufallswald (nicht live)", "regret": _S(0.8), "mse": 0.29},
    ]
    kind, text = ev.verdict_ranking(rows)
    assert kind == "success"
    assert "Das kleinste Regret hat SPO+ (0,5 %, Quadratfehler je Bogen 1,31)" in text
    assert "den kleinsten Quadratfehler hat Zufallswald (0,29, Regret 0,8 %)" in text


def test_verdict_ranking_same_winner_is_neutral():
    rows = [
        {"Verfahren": "Orakel (kennt die mittleren Kosten)", "regret": _S(0.0), "mse": None},
        {"Verfahren": "KQ (linear)", "regret": _S(0.0), "mse": 0.0},
        {"Verfahren": "SPO+ (linear, 300 Schritte)", "regret": _S(0.1), "mse": 0.01},
        {"Verfahren": "Zufallswald (nicht live)", "regret": _S(0.7), "mse": 0.02},
    ]
    kind, text = ev.verdict_ranking(rows)
    assert kind == "info" and text.startswith("KQ hat zugleich das kleinste Regret (0,0 %)")
