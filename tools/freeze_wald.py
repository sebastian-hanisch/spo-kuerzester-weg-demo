"""Friert die Zufallswald-Zahlen der Vormessung in `skw_wald_vormessung.json` ein (Vormessung mit scikit-learn, in der App nicht live gerechnet).

    python tools/freeze_wald.py <pfad zu out_eo_v1_sp.txt> [--out skw_wald_vormessung.json]

Liest die Rohausgabe von `vormessung/eo_v1_sp.py` (Zufallswald mit 100 Bäumen und mindestens 5 Fällen je Blatt, nicht abgestimmt, 20 Seeds, 1000 Testfälle,
SPO+ dort 600 Adam-Schritte) und übernimmt je Zelle (Grad, n, Rauschen) den Regret des Waldes (Median, Minimum, Maximum, in Prozent) und den mittleren
quadratischen Fehler je Bogen (Median). Die KQ- und SPO+-Spalten der Vormessung werden ebenfalls festgehalten, damit die README beide Messungen nebeneinander zeigen kann.
Die Zufalls-Seeds und die Welt der Vormessung sind NICHT dieselben wie in dieser Demo (andere Zufallsquelle): die Zahlen sind vergleichbar, nicht gleich."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = r"\s*(-?\d+\.\d)\s*\[\s*(-?\d+\.\d),\s*(-?\d+\.\d)\]"
LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s*\|" + r"\s*\|".join([CELL] * 4) + r"\s*\|\s*(\d+)/(\d+)\s*\|\s*([\d.]+) / ([\d.]+) / ([\d.]+)\s*$")


def parse(text: str) -> list[dict]:
    noise = None
    cells = []
    for line in text.splitlines():
        m = re.match(r"--- Rauschen e = ([\d.]+)", line)
        if m:
            noise = float(m.group(1))
            continue
        m = LINE.match(line)
        if not m or noise is None:
            continue
        g = [float(x) for x in m.groups()[2:14]]
        better, seeds = int(m.group(15)), int(m.group(16))
        mse = [float(m.group(17 + i)) for i in range(3)]

        def triple(i):
            return {"median": g[3 * i], "min": g[3 * i + 1], "max": g[3 * i + 2]}

        cells.append({
            "deg": int(m.group(1)), "n": int(m.group(2)), "noise": noise,
            "regret_orakel": triple(0), "regret_kq": triple(1), "regret_spo600": triple(2), "regret_wald": triple(3),
            "spo600_besser_als_kq": better, "seeds": seeds,
            "mse_kq": mse[0], "mse_spo600": mse[1], "mse_wald": mse[2],
        })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", default=str(ROOT / "skw_wald_vormessung.json"))
    args = ap.parse_args()
    cells = parse(Path(args.source).read_text(encoding="utf-8"))
    if len(cells) != 24:
        sys.exit(f"erwartet 24 Zellen (4 Grade x 3 Datenmengen x 2 Rauschstufen), gefunden {len(cells)}")
    out = {
        "quelle": "Vormessung entscheidungsorientiert-planung/vormessung/eo_v1_sp.py, Rohausgabe out_eo_v1_sp.txt (2026-11-09)",
        "hinweis": "Zufallswald aus scikit-learn (100 Bäume, mindestens 5 Fälle je Blatt, nicht abgestimmt), NICHT live gerechnet; SPO+ dort mit 600 Adam-Schritten, 20 Seeds, 1000 Testfälle; "
                   "Regret in Prozent der besten Wegkosten, mse = mittlerer quadratischer Fehler je Bogen (Median)",
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cells)} Zellen nach {args.out}")


if __name__ == "__main__":
    main()
