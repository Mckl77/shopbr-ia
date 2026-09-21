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



# ── Commandes reelles Olist ─────────────────────────────────────────
# Le dashboard affiche les commandes les plus recentes du dataset,
# scorees par le modele via l'API. Le chargement des CSV (plusieurs
# dizaines de Mo) ne se fait qu'UNE fois, au premier affichage : les
# commandes preparees sont ensuite gardees en memoire.
DATA_DIR = os.getenv("DATA_DIR", "data")
_REAL_ORDERS_CACHE = None


def load_real_orders(n=40):
    """
    Construit les n commandes les plus recentes au format attendu par
    l'API (memes variables, calculees exactement comme a l'entrainement
    dans train_local.py), plus deux champs d'affichage : l'Etat du
    client et le montant.
    """
    global _REAL_ORDERS_CACHE
    if _REAL_ORDERS_CACHE is not None:
        return _REAL_ORDERS_CACHE

    orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
    products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv",
                           usecols=["product_id", "product_weight_g"])
    sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv",
                          usecols=["seller_id", "seller_state"])
    customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv",
                            usecols=["customer_id", "customer_state"])
    payments = pd.read_csv(f"{DATA_DIR}/olist_order_payments_dataset.csv",
                           usecols=["order_id", "payment_installments"])

    # On ne garde que les n commandes les plus recentes : c'est la
    # "file du jour" que l'equipe logistique consulterait.
    recent = orders.sort_values("order_purchase_timestamp", ascending=False)
    recent = recent[recent["order_id"].isin(items["order_id"])].head(n)

    it = items[items["order_id"].isin(recent["order_id"])].merge(products, on="product_id", how="left")
    agg = it.groupby("order_id").agg(
        nb_items=("order_item_id", "count"),
        nb_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"),
        main_seller=("seller_id", "first"),
    ).reset_index()
    pay = payments.groupby("order_id")["payment_installments"].max().reset_index()

    df = (recent.merge(agg, on="order_id")
                .merge(pay, on="order_id", how="left")
                .merge(customers, on="customer_id", how="left")
                .merge(sellers, left_on="main_seller", right_on="seller_id", how="left"))

    payload = []
    for _, r in df.iterrows():
        payload.append({
            "order_id": r["order_id"][:10],   # identifiant raccourci, plus lisible
            "nb_items": int(r["nb_items"]),
            "nb_sellers": int(r["nb_sellers"]),
            "total_price": round(float(r["total_price"]), 2),
            "total_freight": round(float(r["total_freight"]), 2),
            "cross_state_delivery": bool(r["customer_state"] != r["seller_state"]),
            "purchase_weekday": int(r["order_purchase_timestamp"].weekday()),
            "purchase_hour": int(r["order_purchase_timestamp"].hour),
            "estimated_weight_kg": round(float(r["weight_g"] or 0) / 1000, 2) if pd.notna(r["weight_g"]) else 0.0,
            "max_installments": int(r["payment_installments"]) if pd.notna(r["payment_installments"]) else 1,
            "state": r["customer_state"],
        })

    _REAL_ORDERS_CACHE = payload
    return payload


def risk_badge(risk_pct):
    if risk_pct >= 20:
        return dbc.Badge("Risque elevé", color="danger")
    elif risk_pct >= 10:
        return dbc.Badge("Risque modéré", color="warning")
    else:
        return dbc.Badge("Risque faible", color="success")


# ── Graphiques (definis AVANT le layout qui les utilise) ──────

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


@callback(
    Output("orders-store", "data"),
    Input("refresh-interval", "n_intervals")
)
def refresh_orders(n):
    """
    Envoie les commandes reelles a l'API (POST /predict/batch) et
    recupere un score de risque pour chacune.

    Correctif : l'ancienne version appelait cette adresse en GET, que
    l'API refuse (elle n'accepte que POST). Le dashboard retombait donc
    TOUJOURS sur ses donnees de demonstration, meme avec un modele charge.
    """
    try:
        orders = load_real_orders()
        resp = requests.post(f"{API_URL}/predict/batch",
                             json={"orders": orders}, timeout=30)
        if resp.status_code == 200:
            body = resp.json()
            # L'API renvoie les scores ; on y rattache l'Etat et le
            # montant de chaque commande pour l'affichage.
            infos = {o["order_id"]: o for o in orders}
            results = []
            for r in body.get("results", []):
                o = infos.get(r["order_id"], {})
                results.append({
                    "order_id": r["order_id"],
                    "state": o.get("state", "—"),
                    "total_price": o.get("total_price", 0),
                    "risk_pct": r["risk_pct"],
                })
            results.sort(key=lambda x: -x["risk_pct"])
            return {"results": results, "source": "api",
                    "model_version": body.get("model_version")}
    except Exception as e:
        print(f"Dashboard : API indisponible, bascule sur la demo ({e})")
    return {"results": generate_mock_orders(), "source": "demo"}


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

    # Bandeau de transparence : on affiche toujours d'ou viennent les
    # scores, pour ne jamais confondre donnees de demo et predictions.
    if data and data.get("source") == "api":
        banner = dbc.Alert(
            f"Scores calculés par le modèle ({data.get('model_version')}) "
            f"sur les {len(orders)} commandes les plus récentes du dataset.",
            color="success", className="py-2 small")
    else:
        banner = dbc.Alert(
            "API indisponible : données de démonstration simulées, "
            "ce ne sont pas des prédictions du modèle.",
            color="warning", className="py-2 small")

    return html.Div([banner, table]), high, medium, low


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
