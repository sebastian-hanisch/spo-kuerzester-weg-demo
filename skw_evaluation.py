"""Kennzahlen, Verdikt-Logik und Tabellen für die App (keine Streamlit-Abhängigkeit): Handbeispiel, Lauf einer Welt, vorgerechnete Bänder.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.
Zahlen erscheinen mit Dezimalkomma (`de`)."""

from dataclasses import dataclass

import numpy as np

import skw_bands as B
import skw_constants as C
from skw_grid import two_paths
from skw_losses import spo_loss, spo_plus_loss, spo_plus_subgradient, squared_error
from skw_runs import RunSpec
from skw_training import predict


def _shown(value: float, digits: int = 1) -> float:
    """Der Wert so, wie er mit `digits` Stellen angezeigt wird (Vergleiche stimmen mit dem überein, was die Leser sehen)."""
    return float(f"{float(value):.{digits}f}")


def de(value: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; negative Nullen (-0,0) erscheinen als 0,0."""
    text = f"{float(value):.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",")


@dataclass(frozen=True)
class Settings:
    deg: int = C.DEFAULT_DEG
    noise: float = C.DEFAULT_NOISE
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    case: int = C.DEFAULT_CASE
    c1: float = C.DEFAULT_C1HAT
    c2: float = C.DEFAULT_C2HAT

    def run_spec(self) -> RunSpec:
        return RunSpec(deg=self.deg, noise=self.noise, n=self.n, seed=self.seed, p=C.N_FEATURES, h=C.GRID_H, steps=C.STEPS, lr=C.LR, n_test=C.N_TEST)


# --- Experiment 1: das Handbeispiel ------------------------------------------------------------------------------------------------------

def hand_numbers(c1hat: float, c2hat: float) -> dict:
    """Handbeispiel: zwei Wege mit je einem Bogen, wahre Kosten c = (3, 5), Vorhersage ch = (c1hat, c2hat)."""
    solver = two_paths()
    c = np.array([C.HAND_C])
    chat = np.array([[c1hat, c2hat]])
    W, _ = solver.solve(chat)
    return {
        "chosen": int(np.argmax(W[0])) + 1,
        "best": int(np.argmax(solver.solve(c)[0][0])) + 1,
        "chosen_cost": float(c[0] @ W[0]),
        "best_cost": float(solver.solve(c)[1][0]),
        "spo": float(spo_loss(solver, c, chat)[0]),
        "spo_plus": float(spo_plus_loss(solver, c, chat)[0]),
        "subgradient": tuple(float(v) for v in spo_plus_subgradient(solver, c, chat)[0]),
        "sq": float(squared_error(c, chat)[0]),
    }


def hand_line(c1hat: float, c2hat: float, points: int = C.T_POINTS) -> dict:
    """Die drei Verluste entlang der Geraden ch(t) = c + t d durch c (t = 0) und die gewählte Vorhersage (t = 1), d = ch - c.
    Ist ch = c, ist d nicht festgelegt: dann d = (1, -1) (die Gerade ist nur eine Richtung durch c)."""
    solver = two_paths()
    c = np.array(C.HAND_C)
    d = np.array([c1hat, c2hat]) - c
    degenerate = bool(np.all(d == 0))
    if degenerate:
        d = np.array([1.0, -1.0])
    t = np.linspace(C.T_LINE[0], C.T_LINE[1], points)
    t = np.unique(np.concatenate([t, [0.0, 1.0]]))   # t = 0 und t = 1 liegen sicher auf dem Raster
    chat = c[None, :] + t[:, None] * d[None, :]
    C_rep = np.repeat(c[None, :], len(t), axis=0)
    return {
        "t": t, "degenerate": degenerate, "direction": tuple(float(v) for v in d),
        "spo": spo_loss(solver, C_rep, chat), "spo_plus": spo_plus_loss(solver, C_rep, chat), "sq": squared_error(C_rep, chat),
    }


def verdict_hand(h: dict) -> tuple[str, str]:
    """Verdikt des Handbeispiels aus den Zahlen von `hand_numbers`."""
    if h["sq"] == 0:
        return "info", "Die Vorhersage ist exakt (Quadratfehler 0,0): beide Verluste sind 0. Verschieben Sie ĉ, um zu sehen, wie sich SPO, SPO+ und Quadratfehler trennen."
    if h["spo"] == 0:
        return "success", (
            f"Es wird Weg {h['chosen']} gewählt, und das ist der richtige: SPO-Verlust 0,0, obwohl die Vorhersage falsch ist "
            f"(Quadratfehler {de(h['sq'])}). SPO+ liegt bei {de(h['spo_plus'])}."
        )
    return "warning", (
        f"Es wird Weg {h['chosen']} gewählt; er kostet in Wahrheit {de(h['chosen_cost'])} statt {de(h['best_cost'])}: SPO-Verlust {de(h['spo'])}. "
        f"Der Quadratfehler ist {de(h['sq'])}, SPO+ liegt bei {de(h['spo_plus'])} (nie unter dem SPO-Verlust)."
    )


# --- Experiment 3: Training -----------------------------------------------------------------------------------------------------------------

def verdict_training(r_start: float, r25: float, r_chosen: float, best_step: int) -> tuple[str, str]:
    """Verdikt der Lernkurve (alle Werte in Prozent Regret auf den Testfällen): KQ-Start, nach 25 Schritten, gewählter Parameter (kleinster Trainingsverlust, Schritt `best_step`)."""
    a, b = de(r_start), de(r_chosen)
    head = f"Der Regret fällt von {a} % (KQ-Start) auf {de(r25)} % nach 25 Schritten; der gewählte Parameter (Schritt {best_step}) hat {b} %."
    if _shown(r_chosen) < _shown(r_start):
        return "success", head
    if _shown(r_chosen) == _shown(r_start):
        return "info", f"SPO+ ändert den Regret in dieser Instanz nicht ({a} % bei KQ und beim gewählten Parameter, Schritt {best_step}); nach 25 Schritten sind es {de(r25)} %."
    return "warning", f"SPO+ ist in dieser Instanz schlechter als sein Start: {b} % gegen {a} % bei KQ (nach 25 Schritten {de(r25)} %, gewählter Parameter von Schritt {best_step})."


# --- Experiment 4: Entscheidung gegen Prognose -----------------------------------------------------------------------------------------------

def verdict_forecast(regret_kq: float, regret_spo: float, mse_kq: float, mse_spo: float, seeds: int) -> tuple[str, str]:
    """Verdikt über Regret und mittleren Quadratfehler je Bogen (Medianwerte über `seeds` feste Zufalls-Seeds; Regret in Prozent)."""
    scope = f"Median über {seeds} Zufalls-Seeds"
    a, b = de(regret_kq), de(regret_spo)
    qa, qb = de(mse_kq, 2), de(mse_spo, 2)
    ra, rb = _shown(regret_kq), _shown(regret_spo)
    if rb < ra:
        if _shown(mse_spo, 2) > _shown(mse_kq, 2):
            return "success", (
                f"SPO+ senkt den Regret von {a} % auf {b} %, der mittlere Quadratfehler je Bogen steigt von {qa} auf {qb} ({scope}): "
                "die bessere Entscheidung kommt mit der schlechteren Prognose."
            )
        return "info", f"SPO+ senkt den Regret von {a} % auf {b} %; der mittlere Quadratfehler je Bogen steigt dabei nicht ({qa} bei KQ, {qb} bei SPO+; {scope})."
    if rb == ra:
        return "info", f"KQ und SPO+ haben im Regret denselben Wert ({a} %; {scope}); mittlerer Quadratfehler je Bogen {qa} gegen {qb}."
    return "warning", f"SPO+ liegt hier im Regret nicht unter KQ: {b} % gegen {a} % ({scope}); mittlerer Quadratfehler je Bogen {qa} bei KQ, {qb} bei SPO+."


def band_rows(deg: int, n: int, noise: float) -> list[dict]:
    """Zeilen der Tabelle für eine Zelle: Orakel, KQ, SPO+ (eigene Messung) und Zufallswald (Vormessung, nicht live)."""
    c, w = B.cell(deg, n, noise), B.wald_cell(deg, n, noise)
    s = c["summary"]
    return [
        {"Verfahren": "Orakel (kennt die mittleren Kosten)", "quelle": "eigene Messung", "regret": s["regret_orakel"], "mse": None},
        {"Verfahren": "KQ (linear)", "quelle": "eigene Messung", "regret": s["regret_kq"], "mse": s["mse_kq"]["median"]},
        {"Verfahren": "SPO+ (linear, 300 Schritte)", "quelle": "eigene Messung", "regret": s["regret_spo"], "mse": s["mse_spo"]["median"]},
        {"Verfahren": "Zufallswald (nicht live)", "quelle": "Vormessung mit scikit-learn", "regret": w["regret_wald"], "mse": w["mse_wald"]},
    ]


# --- Experiment 5: Datenmenge und Rauschen -----------------------------------------------------------------------------------------------

def grid_rows(deg: int) -> list[dict]:
    """Die sechs Zellen (n x Rauschen) zu einem Grad aus der eigenen Messung, dazu der Zufallswald der Vormessung."""
    rows = []
    for noise in C.NOISE_CHOICES:
        for n in C.N_CHOICES:
            c, w = B.cell(deg, n, noise), B.wald_cell(deg, n, noise)
            s = c["summary"]
            rows.append({
                "n": n, "noise": noise, "orakel": s["regret_orakel"]["median"], "kq": s["regret_kq"], "spo": s["regret_spo"],
                "wald": w["regret_wald"]["median"], "besser": c["spo_besser_als_kq"], "seeds": len(c["regret_kq"]),
            })
    return rows


def verdict_grid(deg: int, rows: list[dict]) -> tuple[str, str]:
    """Verdikt über die sechs Zellen: in wie vielen liegt der Median von SPO+ unter dem von KQ (auf eine Stelle gerundet), und welche Zellen sind die Ausnahmen."""
    def lab(r):
        return f"n = {r['n']} {'mit Rauschen' if r['noise'] > 0 else 'ohne Rauschen'}"

    lower = [r for r in rows if _shown(r["spo"]["median"]) < _shown(r["kq"]["median"])]
    rest = [r for r in rows if r not in lower]
    head = f"Bei Grad {deg} liegt der Median-Regret von SPO+ in {len(lower)} von {len(rows)} Zellen unter dem von KQ"
    if not rest:
        return "info", head + "."
    details = "; ".join(
        f"{lab(r)}: {de(r['spo']['median'])} % gegen {de(r['kq']['median'])} % ({r['besser']} von {r['seeds']} Seeds mit SPO+ besser)" for r in rest
    )
    return "info", f"{head}. Nicht darunter: {details}."


# --- Experiment 2: ein Testfall --------------------------------------------------------------------------------------------------------------

def case_numbers(run, case: int) -> dict:
    """Zahlen eines Testfalls (1-basiert): beste Kosten und die wahren Kosten der Wege, die KQ und SPO+ wählen, samt Aufschlag in Prozent der besten Kosten."""
    i = case - 1
    Ct = run.data.C_test[i:i + 1]
    best_w, best_z = run.grid.solve(Ct)
    out = {"best": float(best_z[0]), "w_best": best_w[0]}
    for key, coef in (("kq", run.coef_kq), ("spo", run.coef_spo)):
        W, _ = run.grid.solve(predict(coef, run.data.X_test[i:i + 1]))
        cost = float(Ct[0] @ W[0])
        out[key] = {"w": W[0], "cost": cost, "extra_pct": 100.0 * (cost - out["best"]) / out["best"]}
    return out


def differing_cases(run, limit: int = C.CASE_MAX) -> list[int]:
    """Nummern (1-basiert, aufsteigend) der ersten `limit` Testfälle, in denen KQ und SPO+ verschiedene Wege wählen."""
    X = run.data.X_test[:limit]
    w_kq, _ = run.grid.solve(predict(run.coef_kq, X))
    w_spo, _ = run.grid.solve(predict(run.coef_spo, X))
    return [int(i) + 1 for i in np.flatnonzero(np.any(w_kq != w_spo, axis=1))]


def next_case(cases: list[int], current: int):
    """Der nächste Fall der Liste nach `current` (am Ende von vorn); None bei leerer Liste."""
    if not cases:
        return None
    later = [c for c in cases if c > current]
    return later[0] if later else cases[0]


def verdict_case(case: int, best: float, kq_pct: float, spo_pct: float) -> tuple[str, str]:
    """Verdikt eines Testfalls: um wie viel Prozent teurer als der beste Weg sind die Wege, die KQ und SPO+ wählen."""
    a, b = de(kq_pct), de(spo_pct)
    base = f"Testfall {case}: der beste Weg kostet {de(best, 2)}. "
    if _shown(kq_pct) == 0 and _shown(spo_pct) == 0:
        return "info", base + "KQ und SPO+ wählen beide den besten Weg (Regret 0,0 %)."
    if _shown(spo_pct) < _shown(kq_pct):
        return "success", base + f"Der Weg von KQ ist {a} % teurer, der von SPO+ {b} %."
    if _shown(spo_pct) == _shown(kq_pct):
        return "info", base + f"KQ und SPO+ wählen gleich gute Wege (je {a} % über dem besten)."
    return "warning", base + f"Hier ist SPO+ schlechter: sein Weg ist {b} % teurer, der von KQ {a} %."


# --- Experiment 4: wer hat das kleinste Regret, wer den kleinsten Quadratfehler ------------------------------------------------------------------

def verdict_ranking(rows: list[dict]) -> tuple[str, str]:
    """Aus den Zeilen von `band_rows`: das Verfahren mit dem kleinsten Median-Regret und das mit dem kleinsten Quadratfehler (ohne Orakel); sind es verschiedene, ist das der Fall "gute Entscheidung, schlechte Prognose"."""
    cand = [r for r in rows if r["mse"] is not None]
    best_r = min(cand, key=lambda r: (_shown(r["regret"]["median"]), r["mse"]))
    best_q = min(cand, key=lambda r: (_shown(r["mse"], 2), r["regret"]["median"]))
    short = lambda r: r["Verfahren"].split(" (")[0]
    if best_r is best_q:
        return "info", f"{short(best_r)} hat zugleich das kleinste Regret ({de(best_r['regret']['median'])} %) und den kleinsten Quadratfehler je Bogen ({de(best_r['mse'], 2)}) unter KQ, SPO+ und Zufallswald."
    return "success", (
        f"Das kleinste Regret hat {short(best_r)} ({de(best_r['regret']['median'])} %, Quadratfehler je Bogen {de(best_r['mse'], 2)}); "
        f"den kleinsten Quadratfehler hat {short(best_q)} ({de(best_q['mse'], 2)}, Regret {de(best_q['regret']['median'])} %): "
        "die beste Prognose ist hier nicht die beste Entscheidung."
    )
