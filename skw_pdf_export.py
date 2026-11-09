"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from skw_evaluation import Settings, band_rows, de, hand_numbers, verdict_forecast

_REPLACE = {"→": "->", "–": "-", "—": "-", "ĉ": "c^", "−": "-", "≥": ">=", "·": "*", "×": "x", "σ": "sigma", "€": "EUR", "²": "^2", "ˆ": "^"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, run) -> bytes:
    """`run`: das `WorldRun` zu den Einstellungen (Live-Lauf einer Instanz)."""
    h = hand_numbers(settings.c1, settings.c2)
    rows = band_rows(settings.deg, settings.n, settings.noise)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Der kürzeste Weg und der SPO-Verlust"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Welt: 5 x 5 Gitter, 5 Merkmale, Grad {settings.deg}, Rauschstufe {de(settings.noise)}, {settings.n} Trainingsfälle, Zufalls-Seed {settings.seed}, "
        f"{run.spec.n_test} Testfälle. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Diese Instanz (ein Zufalls-Seed)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lines = [
        f"Normierter Regret: Orakel {de(100 * run.regret['orakel'])} %, KQ {de(100 * run.regret['kq'])} %, SPO+ {de(100 * run.regret['spo'])} %.",
        f"Mittlerer quadratischer Fehler je Bogen: KQ {de(run.mse['kq'], 2)}, SPO+ {de(run.mse['spo'], 2)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Median über 20 Zufalls-Seeds (vorgerechnet, 300 Adam-Schritte)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in rows:
        mse = "-" if r["mse"] is None else de(r["mse"], 2)
        pdf.multi_cell(0, 5, _s(f"{r['Verfahren']}: Regret {de(r['regret']['median'])} % (Spanne {de(r['regret']['min'])} bis {de(r['regret']['max'])} %), Quadratfehler je Bogen {mse}  [{r['quelle']}]"),
                       new_x="LMARGIN", new_y="NEXT")
    kq, spo = rows[1], rows[2]
    kind, text = verdict_forecast(kq["regret"]["median"], spo["regret"]["median"], kq["mse"], spo["mse"], 20)
    pdf.ln(1)
    pdf.multi_cell(0, 5, _s(text.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Handbeispiel: zwei Wege"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Wahre Kosten c = (3, 5), Vorhersage ĉ = ({de(settings.c1)}, {de(settings.c2)}): gewählt Weg {h['chosen']}, SPO-Verlust {de(h['spo'])}, "
        f"SPO+ {de(h['spo_plus'])}, Teilgradient ({de(h['subgradient'][0])}; {de(h['subgradient'][1])}), Quadratfehler {de(h['sq'])}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der kürzeste Weg und der SPO-Verlust: SPO+ gegen Kleinste Quadrate im Gitter. Zufallswald: Vormessung mit scikit-learn, nicht live. Quelle: spo-kuerzester-weg-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
