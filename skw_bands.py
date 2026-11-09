"""Vorgerechnete Bänder (keine Streamlit-Abhängigkeit): `skw_bands.json` (eigene Messung, `tools/measure_bands.py`) und `skw_wald_vormessung.json`
(Vormessung mit scikit-learn, in dieser Demo NICHT live gerechnet; `tools/freeze_wald.py`)."""

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def own() -> dict:
    return json.loads((HERE / "skw_bands.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def wald() -> dict:
    return json.loads((HERE / "skw_wald_vormessung.json").read_text(encoding="utf-8"))


def cell(deg: int, n: int, noise: float) -> dict:
    """Eigene Messung einer Zelle (Grad, n, Rauschen): Listen je Seed und Zusammenfassungen."""
    for c in own()["cells"]:
        if (c["deg"], c["n"], c["noise"]) == (deg, n, noise):
            return c
    raise KeyError((deg, n, noise))


def wald_cell(deg: int, n: int, noise: float) -> dict:
    """Zelle der Vormessung (Zufallswald, KQ und SPO+ mit 600 Schritten)."""
    for c in wald()["cells"]:
        if (c["deg"], c["n"], c["noise"]) == (deg, n, noise):
            return c
    raise KeyError((deg, n, noise))


def curve(noise: float) -> dict:
    for c in own()["curve"]:
        if c["noise"] == noise:
            return c
    raise KeyError(noise)


def penalty(noise: float) -> dict:
    for c in own()["strafe"]:
        if c["noise"] == noise:
            return c
    raise KeyError(noise)
