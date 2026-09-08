"""
====================================================================
Projet certification - ShopBR - Logique metier du scoring
====================================================================

POURQUOI CE MODULE SEPARE ?
La logique metier (seuils de risque, construction des features) est
isolee ici, SANS dependance a Flask ni MLflow. Deux benefices :
1. Testabilite : les tests unitaires (tests/test_risk_logic.py) ne
   necessitent que pandas — la CI GitHub Actions est rapide et fiable.
2. Reutilisabilite : l'API, le dashboard et les scripts batch
   partagent exactement les memes regles, sans duplication.

Les seuils sont calibres sur le taux de retard REEL mesure au
Projet #6 (6,65 % en moyenne) : 20 % = 3x le taux de base.
Conformement au cadrage valide, la sortie exploitee est toujours
une PROBABILITE en pourcentage, jamais une decision binaire.
====================================================================
"""

import pandas as pd

# ── SEUILS DE RISQUE (calibres sur le taux de base 6,65 %) ──────────
RISK_HIGH_THRESHOLD   = 0.20  # > 20 % -> action immediate (3x le taux de base)
RISK_MEDIUM_THRESHOLD = 0.10  # 10-20 % -> surveillance

# Colonnes attendues par le modele, dans l'ordre de l'entrainement
FEATURE_COLUMNS = [
    "nb_items", "nb_sellers", "total_price", "total_freight",
    "cross_state_delivery", "purchase_weekday", "purchase_hour",
    "estimated_weight_kg", "max_installments",
]


def risk_to_action(risk_proba: float) -> dict:
    """
    Convertit une probabilite (ex: 0.23) en information actionnable :
    niveau, libelle, couleur (dashboard) et action recommandee.
    """
    if risk_proba >= RISK_HIGH_THRESHOLD:
        return {"level": "high", "label": "Risque eleve", "color": "red",
                "action": "Relancer le vendeur immediatement + prevenir le client"}
    elif risk_proba >= RISK_MEDIUM_THRESHOLD:
        return {"level": "medium", "label": "Risque modere", "color": "orange",
                "action": "Surveiller l'expedition"}
    else:
        return {"level": "low", "label": "Risque faible", "color": "green",
                "action": "Aucune action requise"}


def build_features(order_data: dict) -> pd.DataFrame:
    """
    Transforme le JSON d'une commande en DataFrame avec EXACTEMENT
    les colonnes de l'entrainement (regle absolue en ML : meme
    structure en inference qu'en entrainement).
    Valeurs par defaut si un champ est absent (robustesse API).
    """
    return pd.DataFrame([{
        "nb_items":             order_data.get("nb_items", 1),
        "nb_sellers":           order_data.get("nb_sellers", 1),
        "total_price":          order_data.get("total_price", 0.0),
        "total_freight":        order_data.get("total_freight", 0.0),
        "cross_state_delivery": int(order_data.get("cross_state_delivery", False)),
        "purchase_weekday":     order_data.get("purchase_weekday", 1),
        "purchase_hour":        order_data.get("purchase_hour", 12),
        "estimated_weight_kg":  order_data.get("estimated_weight_kg", 1.0),
        "max_installments":     order_data.get("max_installments", 1),
    }])[FEATURE_COLUMNS]
