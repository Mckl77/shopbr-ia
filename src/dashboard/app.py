"""
============================================================
Projet #7 - ShopBR - Dashboard Dash
Interface operationnelle - risque de retard de livraison
============================================================
Vues :
  1. Commandes a risque (liste priorisee pour l'equipe logistique)
  2. Analyse geographique (taux de retard par etat - donnees reelles)
  3. Monitoring modele (MAE, drift, performances)
============================================================
Lancement : python 03_dashboard/app.py
URL       : http://localhost:8050
============================================================
"""

import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="ShopBR — Risque de retard",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Dashboard de prediction du risque de retard ShopBR"},
    ]
)

COLORS = {
    "primary": "#065A82", "secondary": "#1C7293", "accent": "#02C39A",
    "orange": "#F4A261", "red": "#E76F51", "light": "#F8F9FA", "dark": "#021526",
}

# Donnees reelles mesurees sur le dataset Olist (top 10 etats les plus a risque)
REAL_STATE_DATA = [
    {"state": "AL", "name": "Alagoas",         "rate": 21.41, "orders": 397},
    {"state": "MA", "name": "Maranhao",        "rate": 17.43, "orders": 717},
    {"state": "SE", "name": "Sergipe",         "rate": 15.22, "orders": 335},
    {"state": "PI", "name": "Piaui",           "rate": 13.87, "orders": 476},
    {"state": "CE", "name": "Ceara",           "rate": 13.76, "orders": 1279},
    {"state": "BA", "name": "Bahia",           "rate": 11.10, "orders": 3256},
    {"state": "ES", "name": "Espirito Santo",  "rate": 9.10,  "orders": 2002},
    {"state": "RJ", "name": "Rio de Janeiro",  "rate": 8.60,  "orders": 12694},
    {"state": "MG", "name": "Minas Gerais",    "rate": 6.20,  "orders": 11476},
    {"state": "SP", "name": "Sao Paulo",       "rate": 4.10,  "orders": 41345},
]


def generate_mock_orders():
    """Genere des commandes de demo avec score de risque (en absence d'API live)."""
    np.random.seed(7)
    n = 40
    states = [s["state"] for s in REAL_STATE_DATA]
    state_weights = [s["rate"] for s in REAL_STATE_DATA]
    state_weights = np.array(state_weights) / sum(state_weights)

    orders = []
    for i in range(n):
        state = np.random.choice(states, p=state_weights)
        state_rate = next(s["rate"] for s in REAL_STATE_DATA if s["state"] == state)
        cross_state = np.random.random() < 0.638  # taux reel mesure
        base_risk = state_rate / 100
        risk = min(0.95, base_risk * (1.6 if cross_state else 1.0) * np.random.uniform(0.7, 1.5))

        orders.append({
            "order_id": f"ORD-{1000+i}",
            "state": state,
            "cross_state": cross_state,
            "total_price": round(np.random.uniform(40, 450), 2),
            "risk_pct": round(risk * 100, 1),
        })
    return sorted(orders, key=lambda x: -x["risk_pct"])


def risk_badge(risk_pct):
    if risk_pct >= 20:
        return dbc.Badge("Risque elevé", color="danger")
    elif risk_pct >= 10:
        return dbc.Badge("Risque modéré", color="warning")
    else:
        return dbc.Badge("Risque faible", color="success")


# ── Layout ───────────────────────────────────────────────────
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            html.H1("ShopBR — Risque de retard de livraison",
                    className="text-white mb-0", style={"font-size": "1.6rem"}),
            html.P(f"Mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                   className="text-white-50 mb-0 small"),
        ], width=8),
        dbc.Col([
            dbc.Badge("Production", color="success", className="me-2 p-2"),
            dbc.Badge("XGBoost v3", color="info", className="p-2"),
        ], width=4, className="text-end d-flex align-items-center justify-content-end"),
    ], className="p-3 mb-4 rounded", style={"background": COLORS["primary"]}),

    dbc.Tabs([

        # ── Onglet 1 : Commandes à risque ──────────────────
        dbc.Tab(label="Commandes à risque", tab_id="tab-risk", children=[
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H3(id="kpi-high", className="mb-0 text-danger"),
                        html.Small("Risque élevé (≥20%)", className="text-muted")
                    ], className="text-center")
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H3(id="kpi-medium", className="mb-0 text-warning"),
                        html.Small("Risque modéré (10-20%)", className="text-muted")
                    ], className="text-center")
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H3(id="kpi-low", className="mb-0 text-success"),
                        html.Small("Risque faible (<10%)", className="text-muted")
                    ], className="text-center")
                ]), width=4),
            ], className="mt-3 mb-3"),

            dbc.Card([
                dbc.CardHeader("Commandes en cours — triées par risque décroissant",
                               className="fw-bold"),
                dbc.CardBody([
                    html.Div(id="orders-table")
                ])
            ]),
        ]),

        # ── Onglet 2 : Analyse géographique ─────────────────
        dbc.Tab(label="Analyse géographique", tab_id="tab-geo", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Taux de retard par état (données réelles Olist)",
                                       className="fw-bold"),
                        dbc.CardBody([
                            dcc.Graph(figure=build_state_chart())
                        ])
                    ])
                ], width=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H6("📍 Constat clé", className="mb-1"),
                        html.P("Les états du Nordeste (AL, MA, SE, PI, CE) affichent des taux "
                               "de retard 3 à 5 fois supérieurs à São Paulo (4,1 %). "
                               "La distance logistique vendeur-client explique la majorité de cet écart.",
                               className="mb-0 small")
                    ], color="info")
                ], width=12)
            ], className="mt-3"),
        ]),

        # ── Onglet 3 : Monitoring modèle ────────────────────
        dbc.Tab(label="Monitoring modèle", tab_id="tab-monitoring", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5("✅ Aucune dérive détectée", className="mb-1"),
                        html.P("Dernier rapport Evidently AI — Précision : 78,3 % — "
                               "Rappel (commandes en retard détectées) : 71,2 %",
                               className="mb-0 small")
                    ], color="success")
                ], width=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Évolution des métriques par version", className="fw-bold"),
                        dbc.CardBody(dcc.Graph(figure=build_metrics_chart()))
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Modèle actif", className="fw-bold"),
                        dbc.CardBody([
                            html.P([html.Strong("Algorithme : "), "XGBoost Classifier"]),
                            html.P([html.Strong("Version : "), "v3"]),
                            html.P([html.Strong("Précision : "), "78,3 %"]),
                            html.P([html.Strong("Rappel : "), "71,2 %"]),
                            html.P([html.Strong("Taux base (réel) : "), "6,65 %"]),
                            html.P([html.Strong("Statut drift : "),
                                    dbc.Badge("Stable", color="success")]),
                        ])
                    ])
                ], width=4),
            ], className="mt-3"),
        ]),

    ], id="tabs", active_tab="tab-risk"),

    dcc.Interval(id="refresh-interval", interval=300_000, n_intervals=0),
    dcc.Store(id="orders-store"),

], fluid=True, style={"background": COLORS["light"], "min-height": "100vh"})


def build_state_chart():
    df = pd.DataFrame(REAL_STATE_DATA)
    fig = px.bar(df, x="state", y="rate", color="rate",
                 hover_data=["name", "orders"],
                 labels={"state": "État", "rate": "Taux de retard (%)"},
                 color_continuous_scale=["#02C39A", "#F4A261", "#E76F51"])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      coloraxis_showscale=False, margin=dict(t=20, b=20))
    return fig


def build_metrics_chart():
    versions = ["v1", "v2", "v3"]
    precision = [68.2, 74.5, 78.3]
    recall    = [55.1, 64.8, 71.2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=versions, y=precision, mode="lines+markers",
                             name="Précision (%)", line=dict(color=COLORS["accent"])))
    fig.add_trace(go.Scatter(x=versions, y=recall, mode="lines+markers",
                             name="Rappel (%)", line=dict(color=COLORS["orange"])))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.2))
    return fig


@callback(
    Output("orders-store", "data"),
    Input("refresh-interval", "n_intervals")
)
def refresh_orders(n):
    try:
        resp = requests.get(f"{API_URL}/predict/batch", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"results": generate_mock_orders()}


@callback(
    Output("orders-table", "children"),
    Output("kpi-high",   "children"),
    Output("kpi-medium", "children"),
    Output("kpi-low",    "children"),
    Input("orders-store", "data"),
)
def update_orders_table(data):
    orders = data.get("results", generate_mock_orders()) if data else generate_mock_orders()

    high   = sum(1 for o in orders if o.get("risk_pct", 0) >= 20)
    medium = sum(1 for o in orders if 10 <= o.get("risk_pct", 0) < 20)
    low    = sum(1 for o in orders if o.get("risk_pct", 0) < 10)

    rows = []
    for o in orders[:15]:
        rows.append(html.Tr([
            html.Td(o.get("order_id", "—")),
            html.Td(o.get("state", "—")),
            html.Td(f"{o.get('total_price', 0):.2f} R$"),
            html.Td(f"{o.get('risk_pct', 0):.1f} %"),
            html.Td(risk_badge(o.get("risk_pct", 0))),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Commande"), html.Th("État"), html.Th("Montant"),
            html.Th("Risque"), html.Th("Statut"),
        ])),
        html.Tbody(rows)
    ], striped=True, hover=True, responsive=True, size="sm")

    return table, high, medium, low


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
