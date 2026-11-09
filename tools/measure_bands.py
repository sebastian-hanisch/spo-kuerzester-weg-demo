"""Misst die Bänder, die die App als "vorgerechnet" zeigt, und schreibt `skw_bands.json` (feste Zufalls-Seeds 0 bis 19, SPO+ 300 Schritte).

    python tools/measure_bands.py [--jobs 2] [--out skw_bands.json]

Inhalt: (1) Raster Grad x Datenmenge x Rauschen: normierter Regret (Orakel, KQ, SPO+) und mittlerer Quadratfehler je Bogen je Seed;
(2) Lernkurve (Welt Grad 6, ohne Rauschen und mit Rauschen 0,5, n = 1000): Regret nach 0, 5, 10, 25, 50, 100, 300 Schritten und der Anteil des Gewinns bei Schritt 25;
(3) Strafe und früher Abbruch bei n = 100 (Grad 6); (4) Zahl der Adam-Schritte (300, 600, 1000) bei n = 100 mit Rauschen 0,5 (Grad 4 und 6); (5) Stabilitätsprobe: dieselbe Zelle (n = 100, Rauschen 0,5, Grad 4 und 6) mit 100 weiteren Zufalls-Seeds 20 bis 119. Die Zahlen sind trainiert und plattformabhängig; Tests prüfen sie nur als Bänder und Ungleichungen.
Dauer: einige Minuten (n = 5000 mit 300 Schritten ist der teuerste Teil). Nie mehr als zwei Prozesse (`--jobs`)."""

import argparse
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from skw_runs import RunSpec, cell_result, run_world  # noqa: E402

SEEDS = tuple(range(20))
DEGS, NS, NOISES = (1, 2, 4, 6), (100, 1000, 5000), (0.0, 0.5)
CURVE_STEPS = (0, 5, 10, 25, 50, 100, 300)


def _grid_cell(args):
    deg, n, noise, seed = args
    run = run_world(RunSpec(deg=deg, noise=noise, n=n, seed=seed), snapshot_steps=(), keep_cases=False)
    return (deg, n, noise, seed, cell_result(run))


def _curve_cell(args):
    noise, seed = args
    run = run_world(RunSpec(deg=6, noise=noise, n=1000, seed=seed), snapshot_steps=CURVE_STEPS, keep_cases=False)
    return (noise, seed, {s: 100.0 * run.curve[s] for s in CURVE_STEPS})


def _reg_cell(args):
    noise, seed, variant = args
    base = RunSpec(deg=6, noise=noise, n=100, seed=seed)
    spec = {"ohne": base, "l2=0,1": replace(base, l2=0.1), "l2=1": replace(base, l2=1.0), "50 Schritte": replace(base, steps=50)}[variant]
    run = run_world(spec, snapshot_steps=(), keep_cases=False)
    return (noise, seed, variant, 100.0 * run.regret["kq"], 100.0 * run.regret["spo"])


def _steps_cell(args):
    deg, noise, seed, steps = args
    run = run_world(RunSpec(deg=deg, noise=noise, n=100, seed=seed, steps=steps), snapshot_steps=(), keep_cases=False)
    return (deg, noise, seed, steps, 100.0 * run.regret["kq"], 100.0 * run.regret["spo"])


def _stab_cell(args):
    deg, seed = args
    run = run_world(RunSpec(deg=deg, noise=0.5, n=100, seed=seed), snapshot_steps=(), keep_cases=False)
    return (deg, seed, 100.0 * run.regret["kq"], 100.0 * run.regret["spo"])


def _summary(values):
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "skw_bands.json"))
    args = ap.parse_args()
    t0 = time.time()
    with Pool(max(1, min(args.jobs, 2))) as pool:
        # teuerste Zellen (n = 5000) zuerst, damit die Last gleichmäßig bleibt
        cells = sorted([(deg, n, e, s) for deg in DEGS for n in NS for e in NOISES for s in SEEDS], key=lambda c: -c[1])
        grid_res = pool.map(_grid_cell, cells, chunksize=4)
        curve_res = pool.map(_curve_cell, [(e, s) for e in NOISES for s in SEEDS], chunksize=2)
        reg_res = pool.map(_reg_cell, [(e, s, v) for e in NOISES for s in SEEDS for v in ("ohne", "l2=0,1", "l2=1", "50 Schritte")], chunksize=4)
        stab_res = pool.map(_stab_cell, [(deg, s) for deg in (4, 6) for s in range(20, 120)], chunksize=10)
        steps_res = pool.map(_steps_cell, [(deg, 0.5, s, st) for deg in (4, 6) for s in SEEDS for st in (300, 600, 1000)], chunksize=4)

    out = {
        "meta": {
            "beschreibung": "Eigene Messung dieser Demo (tools/measure_bands.py): lineares Modell, KQ gegen SPO+ (Adam, Lernrate 0,05, Start bei KQ), 5 x 5 Gitter, 5 Merkmale, 1000 Testfälle, Zufalls-Seeds 0 bis 19.",
            "seeds": list(SEEDS), "steps": 300, "lr": 0.05, "n_test": 1000,
            "einheiten": "Regret in Prozent der besten Wegkosten (Summe über die Testfälle); mse = mittlerer quadratischer Fehler je Bogen",
            "laufzeit_s": round(time.time() - t0),
        },
        "cells": [],
        "curve": [],
        "strafe": [],
        "schritte": [],
        "stabil": [],
    }
    for deg in DEGS:
        for n in NS:
            for e in NOISES:
                rows = sorted([r for (d, nn, ee, s, r) in grid_res if (d, nn, ee) == (deg, n, e)], key=lambda r: r["seed"])
                assert len(rows) == len(SEEDS)
                out["cells"].append({
                    "deg": deg, "n": n, "noise": e,
                    **{k: [r[k] for r in rows] for k in ("regret_orakel", "regret_kq", "regret_spo", "mse_kq", "mse_spo")},
                    "summary": {k: _summary([r[k] for r in rows]) for k in ("regret_orakel", "regret_kq", "regret_spo", "mse_kq", "mse_spo")},
                    "spo_besser_als_kq": int(sum(r["regret_spo"] < r["regret_kq"] for r in rows)),
                })
    for e in NOISES:
        rows = sorted([(s, c) for (ee, s, c) in curve_res if ee == e])
        by_step = {str(st): [c[st] for _, c in rows] for st in CURVE_STEPS}
        share = [(c[0] - c[25]) / (c[0] - c[300]) if c[0] - c[300] > 1e-9 else float("nan") for _, c in rows]
        out["curve"].append({
            "deg": 6, "n": 1000, "noise": e, "by_step": by_step,
            "summary": {k: _summary(v) for k, v in by_step.items()},
            "anteil_gewinn_bei_25": share,
        })
    for e in NOISES:
        entry = {"deg": 6, "n": 100, "noise": e, "varianten": {}}
        for v in ("ohne", "l2=0,1", "l2=1", "50 Schritte"):
            rows = sorted([(s, kq, sp) for (ee, s, vv, kq, sp) in reg_res if ee == e and vv == v])
            entry["varianten"][v] = {"regret_kq": [r[1] for r in rows], "regret_spo": [r[2] for r in rows], "summary_spo": _summary([r[2] for r in rows]), "summary_kq": _summary([r[1] for r in rows])}
        out["strafe"].append(entry)
    for deg in (4, 6):
        entry = {"deg": deg, "n": 100, "noise": 0.5, "varianten": {}}
        for st in (300, 600, 1000):
            rows = sorted([(s, kq, sp) for (d, ee, s, ss, kq, sp) in steps_res if d == deg and ss == st])
            entry["varianten"][str(st)] = {"regret_kq": [r[1] for r in rows], "regret_spo": [r[2] for r in rows], "summary_kq": _summary([r[1] for r in rows]), "summary_spo": _summary([r[2] for r in rows])}
        out["schritte"].append(entry)
    for deg in (4, 6):
        rows = sorted([(s, kq, sp) for (d, s, kq, sp) in stab_res if d == deg])
        out["stabil"].append({"deg": deg, "n": 100, "noise": 0.5, "seeds": [r[0] for r in rows], "regret_kq": [r[1] for r in rows], "regret_spo": [r[2] for r in rows]})
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
