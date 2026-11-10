"""Der kürzeste Weg und der SPO-Verlust: der Löser in der Schleife - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Vorhersagemodell wird im 5 x 5 Gitter nicht auf den Fehler der Kantenzeiten, sondern auf den Regret des gewählten Weges (SPO-Verlust, SPO+) trainiert.
Aufbau nach DEMO-PLAYBOOK §2. Kern ohne Streamlit: skw_rng, skw_grid, skw_world, skw_losses, skw_training, skw_runs.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import skw_bands as B
import skw_constants as C
from skw_evaluation import (
    Settings,
    band_rows,
    case_numbers,
    de,
    differing_cases,
    grid_rows,
    hand_line,
    hand_numbers,
    next_case,
    verdict_case,
    verdict_forecast,
    verdict_grid,
    verdict_hand,
    verdict_ranking,
    verdict_training,
)
from skw_layout import dunkles_design, kopf_css
from skw_pdf_export import build_pdf
from skw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    randomize_seed,
    sync_query_params,
)
from skw_runs import run_world
from skw_visualization import (
    build_curve_figure,
    build_forecast_figure,
    build_grid_figure,
    build_grid_scan_figure,
    build_hand_figure,
)

st.set_page_config(page_title="Kürzester Weg und SPO-Verlust – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Training läuft: KQ, dann SPO+ mit 300 Adam-Schritten (bei 5000 Trainingsfällen dauert es spürbar länger) …")
def _world_run(deg: int, noise: float, n: int, seed: int):
    return run_world(Settings(deg=deg, noise=noise, n=n, seed=seed).run_spec())


def _pct(x: float) -> str:
    return f"{de(100 * x)} %"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in skw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Der kürzeste Weg und der SPO-Verlust: der Löser in der Schleife")
st.markdown(
    """
Wenn die Entscheidung ein **Weg im Netz** ist, zählt nicht, wie genau jede Kantenzeit vorhergesagt wird, sondern ob der **gewählte Weg gut** ist. Der **SPO-Verlust** misst genau das
(den Regret des gewählten Weges); weil er stückweise konstant ist, trainiert man seinen konvexen Ersatz **SPO+**, bei dem der **Löser in jeder Trainingsrunde läuft**.
Hier im 5 × 5 Gitter mit 40 Bögen und 70 Wegen: erst ein Fall mit zwei Wegen von Hand, dann das Training gegen Kleinste Quadrate (KQ), bei dem SPO+ ein falsches lineares Modell rettet, aber nicht in jeder Lage.

Weiter unten: **🔬 fünf Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Löser in der Schleife", expanded=True):
    st.markdown(
        """
1. **Kosten vorhersagen:** ein lineares Modell macht aus den Merkmalen $x$ einen Kostenvektor $\\hat c$ für alle 40 Bögen. Das Merkmal-Kosten-Muster der Welt ist nichtlinear (Grad 1 bis 6): ein lineares Modell kann es nur annähern.
2. **Den Löser fragen:** das Gitter ist azyklisch, der kürzeste Weg unter $\\hat c$ ist eine Rückwärtsrechnung über die Knoten. Der gewählte Weg wird mit den **wahren** Kosten bewertet: die Differenz zum besten Weg ist der **SPO-Verlust** (Regret).
3. **Den Regret trainieren:** der SPO-Verlust ist stückweise konstant, sein Gradient fast überall null. **SPO+** ist eine konvexe obere Schranke mit einem Teilgradienten, der zwei Löseraufrufe je Fall braucht (auf $c$ und auf $2\\hat c - c$): $2\\,(w^*(c) - w(2\\hat c - c))$.
4. **Vergleichen:** KQ (Quadratfehler der Kosten) gegen SPO+ (Regret des Weges) mit demselben linearen Modell, 240 Parametern und denselben Daten; der Zufallswald kommt als eingefrorene Vormessung dazu (nicht live).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste: Schnellstart und Kostenwelt -------------------------------------------------------------------------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kostenwelt**")
    deg = st.select_slider("Grad der Wahrheit", options=list(C.DEG_CHOICES), key="deg_slider",
                           help="1 = die Kosten hängen linear von den Merkmalen ab (ein lineares Modell kann sie exakt abbilden); je größer, desto stärker weicht die Wahrheit von jedem linearen Modell ab.")
    noise = st.select_slider("Rauschstufe", options=list(C.NOISE_CHOICES), key="noise_slider", format_func=lambda v: de(v),
                             help="Jeder Bogen kostet das Mittel mal einen Faktor aus [1 − e, 1 + e]. Bei 0 sind die Kosten ohne Rauschen, bei 0,5 liegt auch das Orakel beim Regret deutlich über null.")
    n = st.select_slider("Trainingsfälle n", options=list(C.N_CHOICES), key="n_slider",
                         help="Zahl der Fälle (Merkmale mit beobachteten Kosten); das lineare Modell hat 6 × 40 = 240 Parameter.")
    seed = st.number_input("Zufalls-Seed der Welt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welt generieren", width="stretch", on_click=randomize_seed)
    st.caption("Der Fall mit zwei Wegen (Experiment 1) und der gezeigte Testfall (Experiment 2) haben ihre Regler beim Experiment.")

run = _world_run(int(deg), float(noise), int(n), int(seed))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis dieser Welt")
st.caption(f"Eine Instanz (Zufalls-Seed {int(seed)}): {int(n)} Trainingsfälle, {C.N_TEST} Testfälle, normierter Regret = Summe der SPO-Verluste geteilt durch die Summe der besten Wegkosten.")
m1, m2, m3, m4, m5 = st.columns(5)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Regret Orakel", _pct(run.regret["orakel"]), help="Das Orakel kennt die mittleren Kosten der Testfälle; sein Regret ist der unvermeidbare Teil (Rauschen).")
m2.metric("Regret KQ", _pct(run.regret["kq"]), delta=f"{de(100 * (run.regret['kq'] - run.regret['orakel']))} Punkte über dem Orakel", delta_color="off")
m3.metric("Regret SPO+", _pct(run.regret["spo"]), delta=f"{de(100 * (run.regret['spo'] - run.regret['kq']))} Punkte gegen KQ", delta_color="inverse")
m4.metric("Quadratfehler KQ", de(run.mse["kq"], 2), help="Mittlerer quadratischer Fehler je Bogen auf den Testkosten.")
m5.metric("Quadratfehler SPO+", de(run.mse["spo"], 2), delta=f"{de(run.mse['spo'] - run.mse['kq'], 2)} gegen KQ", delta_color="inverse")

settings = Settings(int(deg), float(noise), int(n), int(seed), int(st.session_state.get("case_slider", C.DEFAULT_CASE)),
                    float(st.session_state.get("c1_slider", C.DEFAULT_C1HAT)), float(st.session_state.get("c2_slider", C.DEFAULT_C2HAT)))
if st.button("📄 PDF dieser Welt erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, run))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="spo-kuerzester-weg.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Rettet SPO+ das falsche lineare Modell?")
band = B.cell(int(deg), int(n), float(noise))
bs = band["summary"]
kind, text = verdict_forecast(bs["regret_kq"]["median"], bs["regret_spo"]["median"], bs["mse_kq"]["median"], bs["mse_spo"]["median"], len(band["regret_kq"]))
getattr(st, kind)(text)
st.caption(
    f"Das Verdikt gilt für diese Zelle (Grad {int(deg)}, Rauschstufe {de(noise)}, n = {int(n)}) und das lineare Modell: Median über {len(band['regret_kq'])} Zufalls-Seeds, vorgerechnet "
    f"mit `tools/measure_bands.py`; SPO+ war in {band['spo_besser_als_kq']} von {len(band['regret_kq'])} Seeds besser als KQ. Gegen eine flexible Modellklasse sagt es nichts."
)

st.markdown("---")

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: ein Fall von Hand, zwei Wege")
st.markdown(
    "Zwei Wege mit je einem Bogen. Die **wahren Kosten** sind $c = (3, 5)$: Weg 1 ist der beste. Stellen Sie eine **Vorhersage** $\\hat c$ ein; "
    "das Diagramm zeigt die drei Verluste entlang der Geraden durch $c$ und $\\hat c$ (bei $t = 0$ ist die Vorhersage exakt, bei $t = 1$ ist es Ihre)."
)
h1, h2 = st.columns(2)
with h1:
    c1 = st.slider("Vorhersage ĉ₁ (Weg 1)", C.CHAT_MIN, C.CHAT_MAX, key="c1_slider", step=C.CHAT_STEP, format="%.1f", help="Vorhergesagte Kosten von Weg 1 (wahr: 3).")
with h2:
    c2 = st.slider("Vorhersage ĉ₂ (Weg 2)", C.CHAT_MIN, C.CHAT_MAX, key="c2_slider", step=C.CHAT_STEP, format="%.1f", help="Vorhergesagte Kosten von Weg 2 (wahr: 5).")
hn = hand_numbers(float(c1), float(c2))
line = hand_line(float(c1), float(c2))
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Gewählter Weg", f"Weg {hn['chosen']}", help="Der Weg mit den kleineren vorhergesagten Kosten; bei Gleichstand Weg 1.")
k2.metric("SPO-Verlust", de(hn["spo"]), help="Wahre Kosten des gewählten Weges minus die besten wahren Kosten (3).")
k3.metric("SPO+", de(hn["spo_plus"]), help="Konvexe obere Schranke: max über die Wege von (c − 2ĉ)·w, plus 2ĉ·w*(c), minus z*(c).")
k4.metric("Teilgradient nach ĉ", f"({de(hn['subgradient'][0])}; {de(hn['subgradient'][1])})", help="2 (w*(c) − w(2ĉ − c)).")
k5.metric("Quadratfehler", de(hn["sq"]), help="Summe der quadrierten Abweichungen (c − ĉ)².")
st.plotly_chart(build_hand_figure(line, hn), width="stretch", key="hand_chart")
if line["degenerate"]:
    st.caption("ĉ = c legt keine Gerade fest; gezeigt ist die Richtung (1, −1) durch c.")
else:
    st.caption(f"Gerade ĉ(t) = c + t · (ĉ − c) mit Richtung ({de(line['direction'][0])}; {de(line['direction'][1])}). Schwarze Raute: Ihre Vorhersage (t = 1), grauer Punkt: die wahren Kosten (t = 0).")
kind, text = verdict_hand(hn)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: das Gitter, ein Testfall")
st.markdown(
    "Das 5 × 5 Gitter mit den **wahren Kosten** eines Testfalls (Farbe und Zahl: violett billig, gelb teuer). Dick: der Weg, den das jeweilige Modell wählt; gestrichelt: der beste Weg. "
    "Grad, Rauschstufe und Trainingsfälle stehen in der Seitenleiste."
)


def _goto_case(number):
    st.session_state["case_slider"] = number


differ = differing_cases(run)
cs1, cs2 = st.columns([3, 1])
with cs1:
    case = st.slider("Testfall Nr.", C.CASE_MIN, C.CASE_MAX, key="case_slider", help=f"Einer der ersten {C.CASE_MAX} von {C.N_TEST} Testfällen dieser Welt.")
with cs2:
    st.write("")
    nxt = next_case(differ, int(case))
    st.button("➡️ Nächster Fall mit verschiedenen Wegen", key="next_case", width="stretch", on_click=_goto_case, args=(nxt,), disabled=nxt is None,
              help="Springt zum nächsten der ersten 100 Testfälle, in dem KQ und SPO+ verschiedene Wege wählen (am Ende von vorn).")
st.caption(f"KQ und SPO+ wählen in {len(differ)} der ersten {C.CASE_MAX} Testfälle verschiedene Wege.")
cn = case_numbers(run, int(case))
Ct = run.data.C_test[int(case) - 1]
vmin, vmax = float(Ct.min()), float(Ct.max())
g1, g2 = st.columns(2)
with g1:
    st.markdown(f"**KQ wählt** (wahre Kosten des Weges {de(cn['kq']['cost'], 2)}, bester Weg {de(cn['best'], 2)})")
    st.plotly_chart(build_grid_figure(run.grid, Ct, cn["kq"]["w"], cn["w_best"], vmin, vmax, C.COLOR_KQ, "Weg von KQ"), width="stretch", key="grid_kq")
with g2:
    st.markdown(f"**SPO+ wählt** (wahre Kosten des Weges {de(cn['spo']['cost'], 2)}, bester Weg {de(cn['best'], 2)})")
    st.plotly_chart(build_grid_figure(run.grid, Ct, cn["spo"]["w"], cn["w_best"], vmin, vmax, C.COLOR_SPO, "Weg von SPO+"), width="stretch", key="grid_spo")
kind, text = verdict_case(int(case), cn["best"], cn["kq"]["extra_pct"], cn["spo"]["extra_pct"])
getattr(st, kind)(text)
st.caption("Ein einzelner Testfall sagt wenig über die Modelle; die Mittelwerte über alle Testfälle stehen oben im Ergebnis, die Mediane über Seeds in Experiment 4 und 5.")

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Training, KQ gegen SPO+")
st.markdown(
    "Adam mit fester Lernrate 0,05, Vollbatch, **300 Schritte**, Start bei der KQ-Lösung; in jeder Runde läuft der Löser auf allen Trainingsfällen. "
    "Links der Regret auf den Testfällen nach t Schritten, rechts der Trainingsverlust von SPO+. Zurückgegeben wird der Parameter mit dem kleinsten Trainingsverlust (Raute)."
)
st.plotly_chart(build_curve_figure(run.curve, run.train_losses, run.best_step, run.regret["spo"], run.regret["orakel"]), width="stretch", key="curve_chart")
kind, text = verdict_training(100 * run.curve[0], 100 * run.curve[25], 100 * run.regret["spo"], run.best_step)
getattr(st, kind)(text)
cv = B.curve(0.0)
share = np.asarray(cv["anteil_gewinn_bei_25"], dtype=float)
st.caption(
    f"Vorgerechnet (Grad {cv['deg']}, ohne Rauschen, n = {cv['n']}, {len(share)} Zufalls-Seeds): nach 25 Schritten sind im Median {de(100 * np.median(share), 0)} % des Regret-Gewinns zwischen KQ und Schritt 300 erreicht "
    f"(Spanne {de(100 * share.min(), 0)} bis {de(100 * share.max(), 0)} %). Das gilt für diese Welt; mit Rauschen streut der Anteil stark (siehe README)."
)

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Entscheidung gegen Prognose")
st.markdown(
    f"Regret und mittlerer quadratischer Fehler je Bogen für die gewählte Zelle (Grad {int(deg)}, Rauschstufe {de(noise)}, n = {int(n)}), **Median über {len(band['regret_kq'])} Zufalls-Seeds**. "
    "KQ, SPO+ und Orakel sind eigene Messungen dieser Demo; der **Zufallswald** ist eine Vormessung mit scikit-learn (100 Bäume, nicht abgestimmt) und wird hier **nicht live gerechnet**."
)
rows4 = band_rows(int(deg), int(n), float(noise))
st.plotly_chart(build_forecast_figure(rows4), width="stretch", key="forecast_chart")
table4 = pd.DataFrame([
    {"Verfahren": r["Verfahren"], "Regret (Median)": f"{de(r['regret']['median'])} %", "Spanne über Seeds": f"{de(r['regret']['min'])} bis {de(r['regret']['max'])} %",
     "Quadratfehler je Bogen": "–" if r["mse"] is None else de(r["mse"], 2), "Quelle": r["quelle"]} for r in rows4
])
st.dataframe(table4, hide_index=True, width="stretch")
kind, text = verdict_ranking(rows4)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Datenmenge und Rauschen")
st.markdown(
    f"Alle sechs Zellen aus Datenmenge (n = 100 / 1000 / 5000) und Rauschen (0 / 0,5) zum gewählten **Grad {int(deg)}**, Median über 20 Zufalls-Seeds (vorgerechnet); Balken: Spanne von Minimum bis Maximum. "
    "Wann verliert SPO+ gegen KQ?"
)
rows5 = grid_rows(int(deg))
st.plotly_chart(build_grid_scan_figure(rows5), width="stretch", key="scan_chart")
table5 = pd.DataFrame([
    {"n": r["n"], "Rauschstufe": de(r["noise"]), "Orakel": f"{de(r['orakel'])} %", "KQ": f"{de(r['kq']['median'])} %", "SPO+": f"{de(r['spo']['median'])} %",
     "SPO+ besser als KQ": f"{r['besser']} von {r['seeds']} Seeds", "Zufallswald (Vormessung)": f"{de(r['wald'])} %"} for r in rows5
])
st.dataframe(table5, hide_index=True, width="stretch")
kind, text = verdict_grid(int(deg), rows5)
getattr(st, kind)(text)
with st.expander("Helfen Strafe, früher Abbruch oder weniger Schritte bei n = 100?"):
    pen0, pen5 = B.penalty(0.0), B.penalty(0.5)
    names = {"ohne": "SPO+ ohne Strafe (300 Schritte)", "l2=0,1": "SPO+ mit Strafe λ = 0,1", "l2=1": "SPO+ mit Strafe λ = 1", "50 Schritte": "SPO+ nach 50 Schritten"}
    rows = [{"Variante": "KQ", "Rauschstufe 0": f"{de(pen0['varianten']['ohne']['summary_kq']['median'], 2)} %", "Rauschstufe 0,5": f"{de(pen5['varianten']['ohne']['summary_kq']['median'], 2)} %"}]
    for key, label in names.items():
        rows.append({"Variante": label, "Rauschstufe 0": f"{de(pen0['varianten'][key]['summary_spo']['median'], 2)} %", "Rauschstufe 0,5": f"{de(pen5['varianten'][key]['summary_spo']['median'], 2)} %"})
    st.markdown("Grad 6, n = 100, Median des Regrets über 20 Zufalls-Seeds (vorgerechnet):")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("Zahl der Adam-Schritte bei n = 100 und Rauschstufe 0,5 (Median des Regrets, 20 Zufalls-Seeds, SPO+ in wie vielen Seeds besser als KQ):")
    srows = []
    for entry in B.own()["schritte"]:
        for steps, v in entry["varianten"].items():
            better = int(sum(a < b for a, b in zip(v["regret_spo"], v["regret_kq"])))
            srows.append({"Grad": entry["deg"], "Schritte": int(steps), "KQ": f"{de(v['summary_kq']['median'])} %", "SPO+": f"{de(v['summary_spo']['median'])} %", "SPO+ besser": f"{better} von {len(v['regret_kq'])}"})
    st.dataframe(pd.DataFrame(srows), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
_lin0, _lin5 = B.cell(1, 1000, 0.0), B.cell(1, 100, 0.5)
_small = [(d, B.cell(d, 100, 0.5)) for d in C.DEG_CHOICES]
_small_text = ", ".join(f"Grad {d}: {c['spo_besser_als_kq']} von {len(c['regret_kq'])}" for d, c in _small)
_oracle5 = [B.cell(d, 100, 0.5)["summary"]["regret_orakel"]["median"] for d in C.DEG_CHOICES]
_stab6 = next(e for e in B.own()["stabil"] if e["deg"] == 6)
_stab6_n = len(_stab6["seeds"])
_stab6_better = int(sum(a < b for a, b in zip(_stab6["regret_spo"], _stab6["regret_kq"])))
_stab6_gap = de(float(np.median(np.array(_stab6["regret_spo"]) - np.array(_stab6["regret_kq"]))))
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Die Wahrheit ist nicht linear** (sonst gibt es nichts zu retten) | Grad 1, Rauschstufe 0, n = 1000: KQ hat einen Regret von {de(_lin0['summary']['regret_kq']['median'])} %, SPO+ ist in {_lin0['spo_besser_als_kq']} von {len(_lin0['regret_kq'])} Seeds besser. Mit Rauschen und n = 100 ist SPO+ bei Grad 1 nicht besser: Median {de(_lin5['summary']['regret_spo']['median'])} % gegen {de(_lin5['summary']['regret_kq']['median'])} % für KQ. |
| **Genug Daten** für 240 Parameter | n = 100 mit Rauschstufe 0,5, Zahl der Seeds mit SPO+ besser als KQ (von 20): {_small_text}. Bei den Graden ohne Gewinn ist SPO+ im Median schlechter als KQ. Bei Grad 6 ist es in diesen 20 Seeds ein Münzwurf; eine Stabilitätsprobe mit {_stab6_n} weiteren Seeds ergab SPO+ besser in {_stab6_better} Seeds, im Median {_stab6_gap} Punkte schlechter (siehe README). |
| **Das Modell ist die lineare Klasse** | Gegen eine flexible Klasse liegt SPO+ ohne Rauschen zurück: in einer Vormessung mit Gauß-Kern-Ridge (scikit-learn, nicht in dieser Demo) lag der Regret bei Grad 6 ohne Rauschen und n = 1000 bei {de(C.VORMESSUNG_KERN['kern_eto'], 2)} % gegen {de(C.VORMESSUNG_KERN['spo'], 2)} % für SPO+ ({C.VORMESSUNG_KERN['seeds']} Zufalls-Seeds); mit Rauschen war der Abstand klein und nicht in jeder Zelle zugunsten der flexiblen Klasse (README). SPO+ rettet das falsche lineare Modell, nicht jedes Modell. |
| **Rauschen setzt eine Untergrenze** | Bei Rauschstufe 0,5 hat schon das Orakel einen Regret von {de(min(_oracle5))} bis {de(max(_oracle5))} % (je nach Grad, Median über 20 Seeds); SPO+ kann diesen Teil nicht abbauen, höchstens einen Teil der Lücke zwischen KQ und Orakel. |
| **Eine Welt, ein Gitter** | Nur 5 × 5, nur diese Kostenformel, nur ein Rauschmodell, SPO+ mit Adam und fester Lernrate. Aussagen über andere Gitter, andere Wahrheiten oder andere Trainer sind nicht gemessen. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind Fälle $(x_i, c_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und beobachteten Bogenkosten $c_i \in \mathbb{R}^d$ und ein Löser, der zu Kosten $c$ einen kürzesten Weg $w(c)$ liefert. Gesucht ist ein Modell $\hat c(x)$, dessen gewählter Weg $w(\hat c(x))$ unter den wahren Kosten möglichst gut ist.

| Symbol | Bedeutung |
|---|---|
| $p = 5$, $d = 40$ | Zahl der Merkmale, Zahl der Bögen im 5 × 5 Gitter |
| $c$, $\hat c$ | wahre und vorhergesagte Bogenkosten |
| $w(c)$ | der Weg (0/1-Vektor über die Bögen), den der Löser bei Kosten $c$ wählt; bei Gleichstand rechts vor unten |
| $w^*(c)$, $z^*(c)$ | der beste Weg unter den wahren Kosten, also $w(c)$, und sein Wert $c^\top w^*(c)$ |
| $W$ | die Menge der 70 Wege |
| $B$, $\deg$, $e$ | Gewichtsmatrix der Welt, Grad der Wahrheit, Rauschstufe |

**SPO-Verlust (Regret).**

$$\ell_{\mathrm{SPO}}(\hat c, c) = c^\top w(\hat c) - z^*(c) \ge 0$$

Er ist in $\hat c$ stückweise konstant (er ändert sich nur, wenn sich der gewählte Weg ändert), sein Gradient ist fast überall null.

**SPO+-Verlust.**

$$\ell_{\mathrm{SPO+}}(\hat c, c) = \max_{w \in W}\; (c - 2\hat c)^\top w + 2\,\hat c^\top w^*(c) - z^*(c)$$

Das Maximum über $w$ ist ein Löseraufruf auf den Kosten $2\hat c - c$ (das Maximum von $(c - 2\hat c)^\top w$ ist das Negative des Minimums von $(2\hat c - c)^\top w$). Als Maximum linearer Funktionen von $\hat c$ plus ein linearer Term ist $\ell_{\mathrm{SPO+}}$ **konvex** in $\hat c$.

**SPO+ ist eine obere Schranke:** setzt man im Maximum $w = w(\hat c)$ ein, folgt

$$\ell_{\mathrm{SPO+}} \ge c^\top w(\hat c) - 2\hat c^\top w(\hat c) + 2\hat c^\top w^*(c) - z^*(c) = \ell_{\mathrm{SPO}} + 2\,\big(\hat c^\top w^*(c) - \hat c^\top w(\hat c)\big) \ge \ell_{\mathrm{SPO}},$$

weil $w(\hat c)$ unter $\hat c$ optimal ist, also $\hat c^\top w(\hat c) \le \hat c^\top w^*(c)$.

**Teilgradient.**

$$s(\hat c, c) = 2\,\big(w^*(c) - w(2\hat c - c)\big) \in \partial_{\hat c}\,\ell_{\mathrm{SPO+}}(\hat c, c)$$

Für das lineare Modell $\hat c = \theta^\top (1, x)$ folgt der Teilgradient nach $\theta$ aus der Kettenregel: $(1, x)\, s^\top$, gemittelt über die Fälle. Training: Adam (Lernrate 0,05, 300 Schritte, Start bei der KQ-Lösung).

**Handbeispiel.** Zwei Wege mit je einem Bogen, $c = (3, 5)$, $\hat c = (4, 3)$: gewählt wird Weg 2 (3 < 4), er kostet wirklich 5, der beste 3, also $\ell_{\mathrm{SPO}} = 2$; $\ell_{\mathrm{SPO+}} = \max(-5, -1) + 8 - 3 = 4$; Teilgradient $2\,((1,0) - (0,1)) = (2, -2)$; Quadratfehler $5$. Entlang $\hat c(t) = c + t\,(1, -2)$ ist $\ell_{\mathrm{SPO+}}(t) = \max(0,\, 6t - 2)$.

**Welt.** $x \sim \mathcal N(0, I_p)$, $B \in \{0,1\}^{d \times p}$ mit unabhängigen fairen Münzwürfen, mittlere Kosten $m_j(x) = \big((Bx)_j / \sqrt{p} + 3\big)^{\deg} / 3{,}5^{\deg} + 1$, beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$. Der normierte Regret ist $\sum_i \ell_{\mathrm{SPO}}(\hat c_i, c_i) / \sum_i z^*(c_i)$.

**Literatur.** Elmachtoub und Grigas, *Smart „Predict, then Optimize“*, Management Science 68(1), 2022: SPO-Verlust und SPO+ (nur Abstract gelesen; Satzbedingungen der Konsistenz nicht im Primärtext geprüft). Hu, Kallus und Mao, *Fast Rates for Contextual Linear Optimization*, Management Science 68(6), 2022 (arXiv 2011.03030, im Volltext gelesen, Nummern nach arXiv v3): nennen SPO+ unter Regularitätsbedingungen Fisher-konsistent, wenn die wahre Bedingungserwartung in der Modellklasse liegt. Das Gitter und die Kostenformel sind aus dem Gedächtnis nach dem Schema dieser Literatur nachgebaut, nicht gegen sie geprüft.
        """
    )

sync_query_params({
    "deg_slider": int(deg), "noise_slider": float(noise), "n_slider": int(n), "seed_input": int(seed),
    "case_slider": int(case), "c1_slider": float(c1), "c2_slider": float(c2),
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html)."
)
