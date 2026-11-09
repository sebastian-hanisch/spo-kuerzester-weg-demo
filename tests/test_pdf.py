"""PDF-Export: erzeugt ein gültiges PDF für jedes Preset (Umlaute und Sonderzeichen brechen die Kernschrift nicht)."""

import pytest

import skw_constants as C
import skw_evaluation as ev
from skw_pdf_export import _s, build_pdf


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_pdf_is_built_for_every_preset(name):
    from skw_runs import run_world

    p = C.PRESETS[name]
    s = ev.Settings(deg=p["deg"], noise=p["noise"], n=p["n"], seed=p["seed"])
    pdf = build_pdf(s, run_world(s.run_spec(), snapshot_steps=(), keep_cases=False))
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500


def test_special_characters_are_replaced_by_latin1_safe_text():
    assert _s("ĉ → 3 – λ") == "c^ -> 3 - ?"
    assert _s("Löser für Gitter ß") == "Löser für Gitter ß"
