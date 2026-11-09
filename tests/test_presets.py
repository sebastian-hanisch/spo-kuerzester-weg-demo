"""Presets und Permalink: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellung, Einrasten der Adresszeile, Bänder zu jedem Preset vorhanden."""

import pytest

import skw_bands as B
import skw_constants as C
import skw_evaluation as ev
import skw_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_allowed_choices_and_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert p["deg"] in C.DEG_CHOICES and p["noise"] in C.NOISE_CHOICES and p["n"] in C.N_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_is_the_first_and_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    d = ev.Settings()
    assert C.PRESETS[first] == {"deg": d.deg, "noise": d.noise, "n": d.n, "seed": d.seed}
    assert {k: spec.default for k, spec in P.SETTING_SPECS.items()} == {
        "deg_slider": d.deg, "noise_slider": d.noise, "n_slider": d.n, "seed_input": d.seed, "case_slider": d.case, "c1_slider": d.c1, "c2_slider": d.c2,
    }


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_every_preset_has_a_precomputed_band_and_a_frozen_forest_cell():
    for p in C.PRESETS.values():
        assert B.cell(p["deg"], p["n"], p["noise"])["regret_kq"]
        assert B.wald_cell(p["deg"], p["n"], p["noise"])["regret_wald"]


def test_every_cell_of_the_choice_grid_has_a_band_and_a_forest_cell():
    for deg in C.DEG_CHOICES:
        for n in C.N_CHOICES:
            for noise in C.NOISE_CHOICES:
                assert len(B.cell(deg, n, noise)["regret_kq"]) == len(C.BAND_SEEDS)
                assert B.wald_cell(deg, n, noise)["seeds"] == 20


def test_snap_rounds_to_the_nearest_allowed_value_and_limits_the_rest():
    spec = P.SETTING_SPECS
    assert P.snap(spec["deg_slider"], 3) == 2          # genau in der Mitte von 2 und 4: der kleinere
    assert P.snap(spec["deg_slider"], 5) == 4 and P.snap(spec["deg_slider"], 99) == 6 and P.snap(spec["deg_slider"], -3) == 1
    assert P.snap(spec["noise_slider"], 0.3) == 0.5 and P.snap(spec["noise_slider"], 0.2) == 0.0 and P.snap(spec["noise_slider"], 7.0) == 0.5
    assert P.snap(spec["n_slider"], 10) == 100 and P.snap(spec["n_slider"], 2000) == 1000 and P.snap(spec["n_slider"], 4000) == 5000
    assert P.snap(spec["seed_input"], -5) == 0 and P.snap(spec["seed_input"], 10**9) == C.SEED_MAX
    assert P.snap(spec["case_slider"], 0) == C.CASE_MIN and P.snap(spec["case_slider"], 500) == C.CASE_MAX
    assert P.snap(spec["c1_slider"], 4.3) == 4.5 and P.snap(spec["c1_slider"], 4.2) == 4.0 and P.snap(spec["c1_slider"], 99.0) == C.CHAT_MAX and P.snap(spec["c2_slider"], -1.0) == C.CHAT_MIN


@pytest.mark.parametrize("deg,noise,n", [(d, e, n) for d in C.DEG_CHOICES for e in C.NOISE_CHOICES for n in C.N_CHOICES])
def test_settings_run_spec_matches_the_constants(deg, noise, n):
    spec = ev.Settings(deg=deg, noise=noise, n=n).run_spec()
    assert (spec.p, spec.h, spec.steps, spec.lr, spec.n_test) == (C.N_FEATURES, C.GRID_H, C.STEPS, C.LR, C.N_TEST)
    assert (spec.deg, spec.noise, spec.n) == (deg, noise, n)
