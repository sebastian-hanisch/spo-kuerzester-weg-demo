"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusatztests dieser Demo (unter der Vorlage) ---------------------------------------------------------------------------------------------
# Die Vorlage prüft nur `st.slider`; hier zusätzlich Auswahlregler (`select_slider`) und Zahlenfeld, Permalink, Verdikte je Preset, größte Welt.

import pytest  # noqa: E402

import skw_constants as C  # noqa: E402
import skw_presets as P  # noqa: E402


def test_every_slider_select_slider_and_number_input_at_min_and_max_does_not_raise():
    for state_key, spec in P.SETTING_SPECS.items():
        for edge, value in (("min", spec.lo), ("max", spec.hi)):
            at = _fresh_app()
            found = False
            for art in ("slider", "select_slider", "number_input"):
                try:
                    widget = getattr(at, art)(key=state_key)
                except KeyError:
                    continue
                widget.set_value(value).run(timeout=TIMEOUT)
                found = True
                break
            assert found, state_key
            assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


def test_worst_case_at_the_slider_maximum_finishes_without_error():
    """Größte Welt (n = 5000, Grad 6, Rauschstufe 0,5): Timeout statt Zeit-Assertion."""
    at = _fresh_app()
    at.select_slider(key="n_slider").set_value(5000)
    at.select_slider(key="noise_slider").set_value(0.5)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_snaps_values_to_the_allowed_choices():
    at = AppTest.from_file(APP_PATH)
    at.query_params["deg"] = "3"
    at.query_params["noise"] = "0.3"
    at.query_params["n"] = "10"
    at.query_params["seed"] = "-4"
    at.query_params["case"] = "500"
    at.query_params["c1"] = "4.3"
    at.query_params["c2"] = "abc"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = at.session_state
    assert (s["deg_slider"], s["noise_slider"], s["n_slider"], s["seed_input"], s["case_slider"], s["c1_slider"], s["c2_slider"]) == (2, 0.5, 100, 0, C.CASE_MAX, 4.5, C.DEFAULT_C2HAT)


def test_permalink_roundtrip_writes_the_settings_to_the_address():
    at = _fresh_app()
    at.select_slider(key="deg_slider").set_value(4).run(timeout=TIMEOUT)
    assert dict(at.query_params) == {"deg": "4", "noise": "0.0", "n": "1000", "seed": "0", "case": "1", "c1": "4.0", "c2": "3.0"}


@pytest.mark.parametrize("name,expected_kind,fragment", [
    ("Nichtlineare Wahrheit, ohne Rauschen (Voreinstellung)", "success", "senkt den Regret"),
    ("Lineare Wahrheit: nichts zu gewinnen", "info", "denselben Wert"),
    ("Nichtlinear mit Rauschen", "success", "senkt den Regret"),
    ("Wenig Daten mit Rauschen", "warning", "liegt hier im Regret nicht unter KQ"),
])
def test_every_preset_shows_the_expected_core_question_verdict(name, expected_kind, fragment):
    at = _fresh_app()
    at.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    shown = {"success": at.success, "info": at.info, "warning": at.warning}[expected_kind]
    assert any(fragment in e.value for e in shown), (name, [e.value for e in list(at.success) + list(at.info) + list(at.warning)])


def test_all_five_experiments_the_boundaries_and_the_formulas_are_there():
    at = _fresh_app()
    headings = [s.value for s in at.subheader]
    for i in range(1, 6):
        assert any(h.startswith(f"🔬 Experiment {i}:") for h in headings), i
    assert "🚧 Wo die Annahmen enden" in headings
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any(e.label.startswith("So funktioniert") for e in at.expander)


def test_next_case_button_jumps_to_a_case_with_different_paths():
    from skw_evaluation import Settings, differing_cases
    from skw_runs import run_world

    at = _fresh_app()
    st = Settings()
    cases = differing_cases(run_world(st.run_spec(), snapshot_steps=(), keep_cases=False))
    assert cases, "Standardwelt: mindestens ein Testfall mit verschiedenen Wegen"
    at.button(key="next_case").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["case_slider"] == cases[0]            # von Fall 1 aus der erste der Liste
    assert any(f"Testfall {cases[0]}:" in e.value for e in list(at.success) + list(at.info) + list(at.warning))
