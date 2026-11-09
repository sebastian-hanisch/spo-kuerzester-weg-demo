"""Erzeugt die Zahlentabellen der README aus den vorgerechneten Bändern (`skw_bands.json`, `skw_wald_vormessung.json`).

`tests/test_claims.py` prüft, dass jede dieser Zeilen wörtlich in der README steht: die Zahlen der README sind damit nie von Hand abgeschrieben, und ändert sich eine Messung,
schlägt der Test an. Ausgabe zum Einfügen: `python tests/readme_tables.py`."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import skw_bands as B  # noqa: E402
import skw_constants as C  # noqa: E402
from skw_evaluation import de  # noqa: E402


def _band(s: dict) -> str:
    return f"{de(s['median'])} [{de(s['min'])}; {de(s['max'])}]"


def main_table(noise: float) -> list[str]:
    """Eigene Messung: je Grad und Datenmenge Median [Minimum; Maximum] des Regrets in Prozent, SPO+ besser als KQ, Quadratfehler je Bogen (Median)."""
    lines = ["| Grad | n | Orakel | KQ | SPO+ | SPO+ besser als KQ | Quadratfehler KQ / SPO+ |", "|---|---|---|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        for n in C.N_CHOICES:
            c = B.cell(deg, n, noise)
            s = c["summary"]
            lines.append(
                f"| {deg} | {n} | {de(s['regret_orakel']['median'])} | {_band(s['regret_kq'])} | {_band(s['regret_spo'])} | {c['spo_besser_als_kq']} von {len(c['regret_kq'])} | "
                f"{de(s['mse_kq']['median'], 2)} / {de(s['mse_spo']['median'], 2)} |"
            )
    return lines


def forest_table() -> list[str]:
    """Vormessung (scikit-learn, nicht live): Zufallswald, Regret (Median) und Quadratfehler je Bogen (Median) je Zelle zu Grad 6, dazu SPO+ der Vormessung (600 Schritte)."""
    lines = ["| n | Rauschstufe | Wald Regret | Wald Quadratfehler | KQ Quadratfehler | SPO+ (600 Schritte, Vormessung) Regret | SPO+ besser als KQ (Vormessung) |", "|---|---|---|---|---|---|---|"]
    for noise in C.NOISE_CHOICES:
        for n in C.N_CHOICES:
            w = B.wald_cell(6, n, noise)
            lines.append(
                f"| {n} | {de(noise)} | {de(w['regret_wald']['median'])} % | {de(w['mse_wald'], 2)} | {de(w['mse_kq'], 2)} | {de(w['regret_spo600']['median'])} % | {w['spo600_besser_als_kq']} von {w['seeds']} |"
            )
    return lines


def steps_table() -> list[str]:
    lines = ["| Grad | Adam-Schritte | KQ | SPO+ | SPO+ besser als KQ |", "|---|---|---|---|---|"]
    for e in B.own()["schritte"]:
        for steps, v in e["varianten"].items():
            better = int(sum(a < b for a, b in zip(v["regret_spo"], v["regret_kq"])))
            lines.append(f"| {e['deg']} | {steps} | {de(v['summary_kq']['median'])} % | {de(v['summary_spo']['median'])} % | {better} von {len(v['regret_kq'])} |")
    return lines


def penalty_table() -> list[str]:
    p0, p5 = B.penalty(0.0), B.penalty(0.5)
    names = {"ohne": "SPO+ ohne Strafe (300 Schritte)", "l2=0,1": "SPO+ mit Strafe λ = 0,1", "l2=1": "SPO+ mit Strafe λ = 1", "50 Schritte": "SPO+ nach 50 Schritten"}
    lines = ["| Variante | Rauschstufe 0 | Rauschstufe 0,5 |", "|---|---|---|",
             f"| KQ | {de(p0['varianten']['ohne']['summary_kq']['median'], 2)} % | {de(p5['varianten']['ohne']['summary_kq']['median'], 2)} % |"]
    for key, label in names.items():
        lines.append(f"| {label} | {de(p0['varianten'][key]['summary_spo']['median'], 2)} % | {de(p5['varianten'][key]['summary_spo']['median'], 2)} % |")
    return lines


def stability_table() -> list[str]:
    lines = ["| Grad | KQ (Median) | SPO+ (Median) | SPO+ besser als KQ | Median der Differenz SPO+ minus KQ |", "|---|---|---|---|---|"]
    for e in B.own()["stabil"]:
        kq, sp = np.array(e["regret_kq"]), np.array(e["regret_spo"])
        lines.append(f"| {e['deg']} | {de(np.median(kq))} % | {de(np.median(sp))} % | {int((sp < kq).sum())} von {len(kq)} | {de(np.median(sp - kq), 2)} Punkte |")
    return lines


def curve_table() -> list[str]:
    lines = ["| Schritte | Regret ohne Rauschen (Median [Minimum; Maximum]) | Regret mit Rauschen 0,5 (Median [Minimum; Maximum]) |", "|---|---|---|"]
    c0, c5 = B.curve(0.0), B.curve(0.5)
    for st in ("0", "5", "10", "25", "50", "100", "300"):
        lines.append(f"| {st} | {_band(c0['summary'][st])} % | {_band(c5['summary'][st])} % |")
    return lines


def share_line() -> str:
    s = np.asarray(B.curve(0.0)["anteil_gewinn_bei_25"], dtype=float)
    return f"{de(100 * np.median(s), 0)} % (Minimum {de(100 * s.min(), 0)} %, Maximum {de(100 * s.max(), 0)} %)"


if __name__ == "__main__":
    for title, lines in (("Raster ohne Rauschen", main_table(0.0)), ("Raster mit Rauschen", main_table(0.5)), ("Wald", forest_table()), ("Schritte", steps_table()),
                         ("Strafe", penalty_table()), ("Stabilität", stability_table()), ("Lernkurve", curve_table())):
        print(f"### {title}")
        print("\n".join(lines))
        print()
    print("Anteil:", share_line())
