"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Das Gitter und die Welt (Entscheidung Z2) ---------------------------------------------------------------------------------------------
GRID_H = 5                       # 5 x 5 Knoten: 40 Bögen, 70 Wege, je 8 Bögen
N_FEATURES = 5                   # Merkmale je Fall
N_TEST = 1000                    # Testfälle je Welt
STEPS, LR = 300, 0.05            # Adam: Schritte und Lernrate (Entscheidung Z3)
BAND_SEEDS = tuple(range(20))    # Zufalls-Seeds der vorgerechneten Bänder (skw_bands.json)

DEG_CHOICES = (1, 2, 4, 6)       # Grad der Wahrheit: 1 = linear
NOISE_CHOICES = (0.0, 0.5)       # Rauschstufe e
N_CHOICES = (100, 1000, 5000)    # Trainingsfälle
DEFAULT_DEG, DEFAULT_NOISE, DEFAULT_N, DEFAULT_SEED = 6, 0.0, 1000, 0
SEED_MAX = 999999
CASE_MIN, CASE_MAX, DEFAULT_CASE = 1, 100, 1    # Nummer des gezeigten Testfalls (1 bis 100 von 1000)

# --- Das Handbeispiel (zwei Wege mit je einem Bogen) ---------------------------------------------------------------------------------------
HAND_C = (3.0, 5.0)              # wahre Kosten
CHAT_MIN, CHAT_MAX, CHAT_STEP = 0.0, 8.0, 0.5
DEFAULT_C1HAT, DEFAULT_C2HAT = 4.0, 3.0
T_LINE = (-1.0, 3.0)             # Bereich des Parameters t auf der Geraden ch(t) = c + t (ch - c)
T_POINTS = 241

# --- Anzeige ------------------------------------------------------------------------------------------------------------------------------
COLOR_KQ = "#4c78a8"
COLOR_SPO = "#2F6B65"
COLOR_ORACLE = "#7f7f7f"
COLOR_WALD = "#b8860b"
COLOR_BEST = "#222222"
COLOR_LOSS_SPO = "#d62728"
COLOR_LOSS_PLUS = "#2F6B65"
COLOR_LOSS_SQ = "#4c78a8"
EDGE_COLORSCALE = "Viridis"

PRESETS = {
    "Nichtlineare Wahrheit, ohne Rauschen (Voreinstellung)": {"deg": 6, "noise": 0.0, "n": 1000, "seed": 0},
    "Lineare Wahrheit: nichts zu gewinnen": {"deg": 1, "noise": 0.0, "n": 1000, "seed": 0},
    "Nichtlinear mit Rauschen": {"deg": 6, "noise": 0.5, "n": 1000, "seed": 0},
    "Wenig Daten mit Rauschen": {"deg": 4, "noise": 0.5, "n": 100, "seed": 0},
}
PRESET_HELP = {
    "Nichtlineare Wahrheit, ohne Rauschen (Voreinstellung)": "Grad 6, Rauschstufe 0, 1000 Trainingsfälle, Zufalls-Seed 0: ein lineares Modell passt nicht zur Wahrheit; die Kosten sind trotzdem exakt bekannt.",
    "Lineare Wahrheit: nichts zu gewinnen": "Grad 1, Rauschstufe 0: das lineare Modell kann die Wahrheit exakt darstellen, KQ trifft jede Entscheidung.",
    "Nichtlinear mit Rauschen": "Grad 6, Rauschstufe 0,5, 1000 Trainingsfälle: ein Teil des Regrets ist unvermeidbar, auch das Orakel hat Regret.",
    "Wenig Daten mit Rauschen": "Grad 4, Rauschstufe 0,5, nur 100 Trainingsfälle bei 240 Parametern: in den 20 Zufalls-Seeds der Messung dieser Demo ist SPO+ hier in keinem Seed besser als KQ.",
}

# Messung der Vormessung (nicht in dieser Demo gerechnet): Gauß-Kern-Ridge mit Validierung gegen KQ und SPO+, Grad 6, ohne Rauschen, n = 1000, 20 Zufalls-Seeds, Median des Regrets in Prozent.
# Quelle: entscheidungsorientiert-planung/vormessung/out_eo_v1d_kernel.txt (eo_v1d_kernel.py, scikit-learn; dort SPO+ mit 300 Schritten).
VORMESSUNG_KERN = {"kern_eto": 0.04, "spo": 0.55, "kq": 1.87, "seeds": 20}
