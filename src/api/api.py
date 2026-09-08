"""
====================================================================
Projet #7 - ShopBR - API Flask
Sert les predictions de risque de retard de livraison
Competence C4.3 : API pour accueillir la solution IA en production
====================================================================

A QUOI SERT CE SCRIPT ?
C'est le "pont" entre le modele de Machine Learning (entraine dans
mlflow_train.py) et les utilisateurs finaux (le dashboard de l'equipe
logistique, ou tout autre systeme de ShopBR). Sans API, le modele
resterait un simple fichier inutilisable au quotidien.

QU'EST-CE QU'UNE API ?
API = Application Programming Interface. C'est une porte d'entree qui
permet a d'autres programmes (ici, le dashboard Dash) de communiquer
avec notre code, sans avoir besoin de connaitre les details internes.
Concretement, on expose des URL (qu'on appelle des "endpoints") que
n'importe quel programme peut appeler pour obtenir un resultat.

Par exemple, on envoie les details d'une commande, et l'API repond
"cette commande a 23% de risque d'etre en retard".

POURQUOI FLASK ?
Flask est un "micro-framework" Python : une bibliotheque legere pour
construire des API web rapidement, sans la complexite d'un framework
plus lourd. Il est tres repandu pour exposer des modeles de Machine
Learning en production.

ENDPOINTS DISPONIBLES DANS CE FICHIER :
  GET  /health             -> Verifie que l'API fonctionne (utilise par Kubernetes)
  POST /predict             -> Prediction pour UNE commande
  POST /predict/batch       -> Prediction pour PLUSIEURS commandes a la fois
  GET  /model/info          -> Informations sur le modele actuellement utilise
  GET  /metrics             -> Metriques techniques (pour le monitoring)
====================================================================
"""

# Flask : le framework web qui gere les requetes HTTP entrantes
# request : permet de lire les donnees envoyees par l'utilisateur de l'API
# jsonify : transforme un dictionnaire Python en reponse JSON (le format
#           standard d'echange de donnees entre applications web)
from flask import Flask, request, jsonify

# CORS (Cross-Origin Resource Sharing) : sans cette extension, le
# navigateur bloquerait par securite les appels venant du dashboard
# (qui tourne sur un autre port/domaine) vers cette API. On l'autorise
# explicitement ici.
from flask_cors import CORS

# MLflow : la bibliotheque qui gere le cycle de vie du modele (on l'a
# utilisee pour ENREGISTRER le modele dans mlflow_train.py, on l'utilise
# ici pour le RECHARGER au demarrage de l'API)
import mlflow
import mlflow.xgboost

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Flask(__name__) cree l'application web. C'est l'objet central autour
# duquel tout le reste du script s'organise.
app = Flask(__name__)
CORS(app)  # autorise les appels cross-origin (cf explication ci-dessus)

# os.getenv("NOM_VARIABLE", "valeur_par_defaut") lit une variable
# d'environnement (definie dans docker-compose.yml ou dans Kubernetes).
# Utiliser des variables d'environnement plutot que des valeurs codees
# en dur permet de changer la configuration SANS modifier le code,
# par exemple pour passer de l'environnement de test a la production.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "shopbr_delay_risk")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")  # on charge uniquement la version validee en Production, jamais une version d'essai

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Variables globales qui contiendront le modele charge en memoire et
# son numero de version. Elles sont vides (None) tant que load_model()
# n'a pas ete appelee.
model = None
model_version = None

# ── LOGIQUE METIER (seuils + features) ──────────────────────────────
# Les seuils de risque et la construction des features sont isoles
# dans risk_logic.py : module sans dependance Flask/MLflow, couvert
# par les tests unitaires de la CI (tests/test_risk_logic.py).
from risk_logic import risk_to_action, build_features  # noqa: E402


def load_model():
    """
    Charge le modele XGBoost depuis le "Model Registry" de MLflow
    (un peu comme un catalogue versionne de tous les modeles entraines).

    On charge specifiquement la version marquee "Production" : c'est
    le DAG Airflow (script airflow_dag.py) qui decide, apres chaque
    reentrainement mensuel, si la nouvelle version merite cette
    etiquette "Production" (uniquement si precision et rappel sont
    suffisants).

    Cette fonction est appelee UNE SEULE FOIS, au demarrage de l'API
    (voir tout en bas du fichier), pas a chaque requete : recharger le
    modele a chaque appel serait beaucoup trop lent.
    """
    global model, model_version
    try:
        # "models:/nom_du_modele/Production" est une syntaxe speciale
        # de MLflow qui dit : "donne-moi la derniere version marquee
        # Production de ce modele", sans avoir a connaitre son numero
        # exact.
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.xgboost.load_model(model_uri)

        # On recupere aussi le numero de version exact, utile pour
        # le traçage (savoir quelle version a produit quelle prediction)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        model_version = versions[0].version if versions else "unknown"
        logger.info(f"Modele charge : {MODEL_NAME} v{model_version} ({MODEL_STAGE})")
    except Exception as e:
        # Si le chargement echoue (par exemple MLflow indisponible au
        # demarrage), on ne fait pas planter toute l'API : elle demarre
        # quand meme, mais repondra une erreur explicite sur /predict
        # tant que le modele n'est pas disponible. C'est mieux qu'un
        # crash complet du service.
        logger.error(f"Erreur chargement modele : {e}")
        logger.warning("API demarre sans modele")


# ====================================================================
# ENDPOINTS DE L'API
# ====================================================================
# Chaque fonction ci-dessous est associee a une URL et une methode
# HTTP via le decorateur @app.route(...). Un decorateur (la ligne qui
# commence par @) "enrichit" la fonction qui suit : ici, il dit a
# Flask "quand quelqu'un appelle cette URL avec cette methode, execute
# cette fonction et renvoie son resultat".
#
# GET  = on demande de l'information (sans rien modifier)
# POST = on envoie des donnees pour obtenir un resultat (ici, une prediction)
# ====================================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint de "sante" de l'API. Kubernetes l'appelle automatiquement
    et regulierement (cf livenessProbe et readinessProbe dans
    deployment.yaml) pour verifier que le conteneur fonctionne
    correctement. Si cet endpoint ne repond pas, Kubernetes redemarre
    automatiquement le conteneur defaillant.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat()
    }), 200  # 200 = code HTTP signifiant "succes"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principal : predit le risque de retard pour UNE commande.

    Exemple de body JSON envoye par l'utilisateur de l'API :
    {
        "order_id": "abc123",
        "nb_items": 2,
        "nb_sellers": 1,
        "total_price": 250.0,
        "total_freight": 35.0,
        "cross_state_delivery": true,
        "purchase_weekday": 5,
        "purchase_hour": 14,
        "estimated_weight_kg": 2.5,
        "max_installments": 3
    }
    """
    # On verifie d'abord que le modele est bien charge. Si load_model()
    # a echoue au demarrage, on renvoie une erreur claire plutot que
    # de planter de maniere incomprehensible plus loin dans le code.
    if model is None:
        return jsonify({"error": "Modele non disponible"}), 503  # 503 = "Service indisponible"

    try:
        # request.get_json() lit le corps de la requete HTTP et le
        # transforme en dictionnaire Python.
        data = request.get_json()
        if not data:
            return jsonify({"error": "Body JSON manquant"}), 400  # 400 = "requete invalide"

        order_id = data.get("order_id", "unknown")
        features = build_features(data)

        # model.predict_proba(features) renvoie, pour chaque ligne, un
        # tableau [probabilite_classe_0, probabilite_classe_1], c'est-a-dire
        # [probabilite "pas en retard", probabilite "en retard"].
        # [0][1] = on prend la 1ere (et unique) ligne, puis la 2eme valeur
        # (la probabilite de retard, qui est ce qui nous interesse).
        risk_proba = float(model.predict_proba(features)[0][1])
        risk_info  = risk_to_action(risk_proba)

        logger.info(f"Prediction order={order_id} -> risque={risk_proba:.2%} ({risk_info['level']})")

        # jsonify transforme ce dictionnaire en reponse JSON, que le
        # dashboard (ou tout autre client de l'API) pourra facilement
        # interpreter.
        return jsonify({
            "order_id":       order_id,
            "risk_proba":     round(risk_proba, 4),
            "risk_pct":       round(risk_proba * 100, 1),
            "risk":           risk_info,
            "model_version":  model_version,
            "timestamp":      datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        # On capture toute erreur imprevue pour eviter que l'API ne
        # plante completement : on renvoie un message d'erreur clair
        # au lieu de laisser Flask afficher une erreur technique brute.
        logger.error(f"Erreur prediction : {e}")
        return jsonify({"error": str(e)}), 500  # 500 = "erreur serveur"


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Variante de /predict qui traite PLUSIEURS commandes en une seule
    requete, au lieu d'obliger le dashboard a faire un appel separe
    pour chaque commande (ce qui serait beaucoup plus lent en pratique).

    Body JSON attendu : {"orders": [ {...commande1...}, {...commande2...}, ... ]}
    """
    if model is None:
        return jsonify({"error": "Modele non disponible"}), 503

    try:
        data = request.get_json()
        orders = data.get("orders", [])
        if not orders:
            return jsonify({"error": "Liste 'orders' vide ou manquante"}), 400

        results = []
        # On boucle sur chaque commande recue et on applique exactement
        # la meme logique que dans /predict.
        for order_data in orders:
            order_id = order_data.get("order_id", "unknown")
            features = build_features(order_data)
            risk_proba = float(model.predict_proba(features)[0][1])
            risk_info  = risk_to_action(risk_proba)
            results.append({
                "order_id":   order_id,
                "risk_proba": round(risk_proba, 4),
                "risk_pct":   round(risk_proba * 100, 1),
                "risk":       risk_info,
            })

        # On compte combien de commandes sont a "risque eleve", une
        # information resumee utile pour un coup d'oeil rapide dans le dashboard.
        high_risk_count = sum(1 for r in results if r["risk"]["level"] == "high")
        logger.info(f"Batch de {len(results)} commandes - {high_risk_count} a risque eleve")

        return jsonify({
            "count":           len(results),
            "high_risk_count": high_risk_count,
            "results":         results,
            "model_version":   model_version,
            "timestamp":       datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Erreur prediction batch : {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """
    Renvoie des informations techniques sur le modele actuellement
    en production : sa version, ses metriques de performance
    (precision, rappel...), et ses parametres d'entrainement.

    Utile pour la transparence et la tracabilite : on peut toujours
    savoir exactement quel modele a produit quelle prediction, et
    quelles etaient ses performances mesurees au moment de son
    deploiement (exigence d'explicabilite du Bloc 4 de la certification).
    """
    if model is None:
        return jsonify({"error": "Modele non disponible"}), 503
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            v = versions[0]
            # On recupere le "run" MLflow associe a cette version, qui
            # contient toutes les metriques et parametres enregistres
            # pendant l'entrainement (cf mlflow_train.py).
            run = client.get_run(v.run_id)
            return jsonify({
                "name":       v.name,
                "version":    v.version,
                "stage":      v.current_stage,
                "run_id":     v.run_id,
                "metrics":    run.data.metrics,
                "params":     run.data.params,
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Endpoint au format "Prometheus" (un standard tres repandu pour le
    monitoring d'applications en production). Permet a un outil de
    supervision externe de recuperer automatiquement des indicateurs
    techniques simples sur l'etat de l'API.
    """
    return (
        f"# HELP model_version Numero de version du modele actif\n"
        f"# TYPE model_version gauge\n"
        f"model_version {model_version or 0}\n"
        f"# HELP model_loaded Modele charge (1=oui, 0=non)\n"
        f"# TYPE model_loaded gauge\n"
        f"model_loaded {1 if model is not None else 0}\n"
    ), 200, {"Content-Type": "text/plain"}


# ====================================================================
# DEMARRAGE DE L'APPLICATION
# ====================================================================
# Ce bloc ne s'execute que si on lance ce fichier directement
# (python api.py), pas si on l'importe depuis un autre script.
if __name__ == "__main__":
    # On charge le modele UNE SEULE FOIS, juste avant de demarrer le
    # serveur web, pour qu'il soit deja en memoire et pret a repondre
    # des la premiere requete.
    load_model()

    # app.run(...) demarre le serveur web Flask.
    # host="0.0.0.0" = accepte les connexions venant de n'importe quelle
    #   adresse (necessaire pour fonctionner dans un conteneur Docker,
    #   ou l'API doit etre joignable depuis l'exterieur du conteneur).
    # port=8000 = le port d'ecoute (doit correspondre a celui declare
    #   dans le Dockerfile et dans docker-compose.yml).
    # debug=False = mode production : en mode debug, Flask affiche des
    #   informations techniques detaillees en cas d'erreur, ce qui est
    #   pratique en developpement mais constitue une faille de securite
    #   en production (on ne veut pas exposer le code interne aux utilisateurs).
    app.run(host="0.0.0.0", port=8000, debug=False)
