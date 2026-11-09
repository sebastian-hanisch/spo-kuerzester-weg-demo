"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import skw_constants as C
from skw_evaluation import de

_LOCK = dict(fixedrange=True)


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(**_LOCK)
    fig.update_yaxes(**_LOCK)
    return fig


# --- Experiment 1: drei Verluste entlang einer Geraden --------------------------------------------------------------------------------------

def build_hand_figure(line: dict, h: dict) -> go.Figure:
    """SPO-Verlust (Treppe), SPO+ (konvex) und Quadratfehler entlang ch(t) = c + t (ch - c); t = 0 sind die wahren Kosten, t = 1 die gewählte Vorhersage."""
    t = line["t"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("SPO-Verlust (Treppe)", "SPO+ (konvex)", "Quadratfehler"), horizontal_spacing=0.08)
    series = [(line["spo"], C.COLOR_LOSS_SPO, h["spo"]), (line["spo_plus"], C.COLOR_LOSS_PLUS, h["spo_plus"]), (line["sq"], C.COLOR_LOSS_SQ, h["sq"])]
    for col, (y, color, here) in enumerate(series, start=1):
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=3, shape="hv" if col == 1 else "linear"), showlegend=False,
                                 hovertemplate="t = %{x:.2f}: %{y:.2f}<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=[1.0], y=[here], mode="markers", marker=dict(color="black", size=10, symbol="diamond"), showlegend=False,
                                 hovertemplate="Ihre Vorhersage (t = 1): %{y:.2f}<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers", marker=dict(color="gray", size=7), showlegend=False,
                                 hovertemplate="wahre Kosten (t = 0): %{y:.2f}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="t (0 = wahre Kosten, 1 = Ihre Vorhersage)" if col == 2 else None, row=1, col=col)
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=50))
    return _lock(fig)


# --- Experiment 2: das Gitter ---------------------------------------------------------------------------------------------------------------

def _edge_colors(costs: np.ndarray, vmin: float, vmax: float) -> list[str]:
    span = max(vmax - vmin, 1e-9)
    return sample_colorscale(C.EDGE_COLORSCALE, [float(np.clip((c - vmin) / span, 0, 1)) for c in costs])


def build_grid_figure(grid, c_true: np.ndarray, w_chosen: np.ndarray, w_best: np.ndarray, vmin: float, vmax: float, chosen_color: str, chosen_name: str) -> go.Figure:
    """5 x 5 Gitter: Bögen nach den WAHREN Kosten gefärbt und beschriftet, der gewählte Weg dick, der beste Weg gestrichelt (liegen beide übereinander, sieht man beide Linien)."""
    h = grid.h
    fig = go.Figure()
    colors = _edge_colors(c_true, vmin, vmax)
    mids_x, mids_y, hover = [], [], []
    for k, (a, b) in enumerate(grid.edges):
        (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=colors[k], width=3), hoverinfo="skip", showlegend=False))
        mids_x.append((x0 + x1) / 2)
        mids_y.append((y0 + y1) / 2)
        hover.append(f"Bogen {k + 1}: wahre Kosten {de(c_true[k], 2)}")
    for w, color, width, dash, name in ((w_chosen, chosen_color, 8, "solid", chosen_name), (w_best, C.COLOR_BEST, 3, "dot", "bester Weg (wahre Kosten)")):
        xs, ys = [], []
        for k in np.flatnonzero(w):
            a, b = grid.edges[k]
            (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mids_x, y=mids_y, mode="markers", marker=dict(size=14, opacity=0), hovertext=hover, hoverinfo="text", showlegend=False))
    nx = [grid.node_xy(v)[0] for v in range(grid.nodes)]
    ny = [grid.node_xy(v)[1] for v in range(grid.nodes)]
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="markers", marker=dict(size=9, color="#333", line=dict(color="white", width=1)), hoverinfo="skip", showlegend=False))
    for k in range(len(mids_x)):
        fig.add_annotation(x=mids_x[k], y=mids_y[k], text=de(c_true[k], 1), showarrow=False, font=dict(size=9), bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(0,0,0,0)")
    fig.add_annotation(x=0, y=0, text="Start", showarrow=False, xshift=-22, yshift=-12, font=dict(size=11))
    fig.add_annotation(x=h - 1, y=h - 1, text="Ziel", showarrow=False, xshift=20, yshift=12, font=dict(size=11))
    fig.update_xaxes(visible=False, range=[-0.35, h - 0.65])
    fig.update_yaxes(visible=False, range=[h - 0.45, -0.45])
    fig.update_layout(height=380, margin=dict(l=5, r=5, t=5, b=5), legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top"))
    return _lock(fig)


# --- Experiment 3: Lernkurve -----------------------------------------------------------------------------------------------------------------

def build_curve_figure(curve: dict, train_losses: np.ndarray, best_step: int, r_chosen: float, r_oracle: float) -> go.Figure:
    """Links: Regret auf den Testfällen nach Schritten (KQ-Start bei Schritt 0, gewählter Parameter als Raute, Orakel als Linie); rechts: SPO+-Trainingsverlust vor jedem Schritt."""
    steps = sorted(curve)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Regret auf den Testfällen (%)", "SPO+-Trainingsverlust (mittlere Kosteneinheiten)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=steps, y=[100 * curve[s] for s in steps], mode="lines+markers", line=dict(color=C.COLOR_SPO, width=3), name="SPO+ nach t Schritten",
                             hovertemplate="Schritt %{x}: %{y:.2f} %<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, max(steps)], y=[100 * curve[0]] * 2, mode="lines", line=dict(color=C.COLOR_KQ, dash="dash"), name="KQ (Start)", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, max(steps)], y=[100 * r_oracle] * 2, mode="lines", line=dict(color=C.COLOR_ORACLE, dash="dot"), name="Orakel", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[best_step], y=[100 * r_chosen], mode="markers", marker=dict(color="black", size=11, symbol="diamond"), name="gewählter Parameter",
                             hovertemplate="Schritt %{x}: %{y:.2f} %<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(train_losses)), y=train_losses, mode="lines", line=dict(color=C.COLOR_LOSS_PLUS, width=2), showlegend=False,
                             hovertemplate="vor Schritt %{x}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Adam-Schritte", row=1, col=1)
    fig.update_xaxes(title_text="Adam-Schritte", row=1, col=2)
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.28, x=0.0))
    return _lock(fig)


# --- Experiment 4: Entscheidung gegen Prognose -------------------------------------------------------------------------------------------------

_COLORS = {"Orakel": C.COLOR_ORACLE, "KQ": C.COLOR_KQ, "SPO+": C.COLOR_SPO, "Zufallswald": C.COLOR_WALD}


def _short(name: str) -> str:
    for key in _COLORS:
        if name.startswith(key):
            return key
    return name


def build_forecast_figure(rows: list[dict]) -> go.Figure:
    """Links: Regret je Verfahren (Median, Balken von Minimum bis Maximum über die Seeds); rechts: mittlerer quadratischer Fehler je Bogen (Median)."""
    names = [_short(r["Verfahren"]) for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Regret (%, Median und Spanne)", "Quadratfehler je Bogen (Median)"), horizontal_spacing=0.12)
    med = [r["regret"]["median"] for r in rows]
    fig.add_trace(go.Bar(x=names, y=med, marker_color=[_COLORS.get(n, "#999") for n in names],
                         error_y=dict(type="data", symmetric=False, array=[r["regret"]["max"] - r["regret"]["median"] for r in rows], arrayminus=[r["regret"]["median"] - r["regret"]["min"] for r in rows]),
                         showlegend=False, hovertemplate="%{x}: %{y:.2f} %<extra></extra>"), row=1, col=1)
    mse_names = [n for n, r in zip(names, rows) if r["mse"] is not None]
    mse_vals = [r["mse"] for r in rows if r["mse"] is not None]
    fig.add_trace(go.Bar(x=mse_names, y=mse_vals, marker_color=[_COLORS.get(n, "#999") for n in mse_names], showlegend=False, hovertemplate="%{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=40))
    return _lock(fig)


# --- Experiment 5: Datenmenge und Rauschen -----------------------------------------------------------------------------------------------------

def build_grid_scan_figure(rows: list[dict]) -> go.Figure:
    """Zwei Teilbilder (ohne und mit Rauschen), je n = 100 / 1000 / 5000: Median-Regret von Orakel, KQ, SPO+ (mit Spanne über die Seeds) und Zufallswald (Vormessung)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ohne Rauschen", "mit Rauschen (0,5)"), horizontal_spacing=0.1)
    for col, noise in enumerate(C.NOISE_CHOICES, start=1):
        sel = [r for r in rows if r["noise"] == noise]
        xs = [f"n = {r['n']}" for r in sel]
        for name, key, color in (("Orakel", "orakel", C.COLOR_ORACLE), ("KQ", "kq", C.COLOR_KQ), ("SPO+", "spo", C.COLOR_SPO), ("Zufallswald", "wald", C.COLOR_WALD)):
            ys = [r[key]["median"] if isinstance(r[key], dict) else r[key] for r in sel]
            err = None
            if key in ("kq", "spo"):
                err = dict(type="data", symmetric=False, array=[r[key]["max"] - r[key]["median"] for r in sel], arrayminus=[r[key]["median"] - r[key]["min"] for r in sel])
            fig.add_trace(go.Bar(x=xs, y=ys, name=name, marker_color=color, error_y=err, showlegend=(col == 1), legendgroup=name,
                                 hovertemplate=name + ", %{x}: %{y:.2f} %<extra></extra>"), row=1, col=col)
        fig.update_yaxes(title_text="Regret (%)" if col == 1 else None, row=1, col=col)
    fig.update_layout(barmode="group", height=360, margin=dict(l=40, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.18, x=0.0))
    return _lock(fig)
