"""
====================================================================
Projet #7 - ShopBR - Entrainement MLflow
Entrainement du modele XGBoost Classifier + versionnage du cycle de vie
Competence C4.5 : Reentrainement automatise
====================================================================

A QUOI SERT CE SCRIPT ?
C'est ici que le modele de Machine Learning est reellement ENTRAINE,
c'est-a-dire qu'il "apprend" a reconnaitre les patterns qui menent a
un retard de livraison, a partir des donnees historiques.

QU'EST-CE QU'UN "CLASSIFIEUR" ?
Notre objectif est de predire une valeur BINAIRE : la commande
sera-t-elle en retard (1) ou non (0) ? C'est ce qu'on appelle un
probleme de "classification binaire", different d'un probleme de
"regression" ou on predirait un nombre continu (comme un prix).
XGBoostClassifier est l'algorithme choisi ici : c'est une methode
tres performante et tres utilisee en entreprise pour ce type de
probleme, basee sur l'assemblage de nombreux "arbres de decision"
simples qui, combines, donnent une prediction fiable.

QU'EST-CE QUE MLFLOW APPORTE ICI ?
A chaque execution de ce script, MLflow enregistre AUTOMATIQUEMENT :
- les parametres utilises pour entrainer le modele
- les metriques de performance obtenues (precision, rappel...)
- le modele lui-meme (les fichiers necessaires pour le reutiliser)
Cela permet de comparer facilement plusieurs versions entrainees a
des dates differentes, et de savoir exactement comment chaque version
a ete produite (exigence de REPRODUCTIBILITE du Bloc 4).

CE SCRIPT EST APPELE AUTOMATIQUEMENT CHAQUE MOIS par le DAG Airflow
(cf airflow_dag.py), ce qui realise l'exigence de REENTRAINEMENT
AUTOMATISE demandee par la certification (competence C4.5).
====================================================================
"""

# mlflow : la bibliotheque de suivi des experiences de Machine Learning
import mlflow
import mlflow.xgboost
# MlflowClient permet d'interagir plus finement avec MLflow, par
# exemple pour changer le statut d'une version de modele
# (de "None" a "Production")
from mlflow.tracking import MlflowClient

# xgboost : la bibliotheque de l'algorithme de Machine Learning utilise
import xgboost as xgb

import pandas as pd
import numpy as np

# train_test_split : fonction qui decoupe automatiquement nos donnees
# en deux groupes : un pour ENTRAINER le modele, un pour le TESTER sur
# des donnees qu'il n'a jamais vues (ce qui permet de mesurer sa
# vraie capacite de generalisation, pas juste sa capacite a "reciter"
# les exemples d'entrainement).
from sklearn.model_selection import train_test_split

# Metriques d'evaluation d'un classifieur binaire (voir explications
# detaillees plus bas, dans la fonction train_xgboost)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import argparse
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "shopbr-delay-risk-model"  # nom du dossier d'experiences dans MLflow, qui regroupera tous les entrainements de ce modele
MODEL_NAME      = "shopbr_delay_risk"        # nom sous lequel le modele est enregistre dans le Model Registry de MLflow

# ── SEUILS DE DEPLOIEMENT ─────────────────────────────────────────────
# Un modele n'est promu en "Production" QUE s'il atteint ces deux
# seuils minimum. Cela evite de deployer automatiquement un modele
# de mauvaise qualite simplement parce que le reentrainement mensuel
# s'est bien deroule techniquement (sans erreur), sans verifier que
# le modele est reellement BON.
MIN_PRECISION   = 0.60   # au moins 60% des alertes de retard doivent etre justes (peu de "fausses alertes")
MIN_RECALL      = 0.65   # le modele doit detecter au moins 65% des vrais retards (peu de retards "manques")


def load_training_data() -> tuple:
    """
    Charge les donnees d'entrainement. En production reelle, cette
    fonction lirait directement les tables Silver/Gold de Redshift
    (produites au Projet #6). Ici, pour la demonstration, on genere
    des donnees SYNTHETIQUES mais CALIBREES sur les vraies statistiques
    mesurees sur le dataset Olist :
      - taux de retard global : 6,65%
      - livraison inter-etats : 63,8% des commandes
      - le facteur cross_state_delivery est le plus fortement
        correle au risque de retard observe dans les vraies donnees

    POURQUOI DES DONNEES SYNTHETIQUES ICI ?
    Parce que ce script est concu pour tourner dans un environnement
    Databricks/Redshift complet que nous ne pouvons pas simuler
    integralement ici. Les statistiques utilisees pour generer ces
    donnees viennent en revanche d'une vraie analyse exploratoire
    (cf Projet #6), donc le COMPORTEMENT du modele entraine reste
    representatif de la realite.
    """
    logger.info("Chargement des donnees d'entrainement...")
    np.random.seed(42)  # "graine" aleatoire fixe : garantit que le script produit toujours les memes donnees a chaque execution, pour la reproductibilite des tests
    n = 95000  # proche du volume reel de commandes livrees dans le dataset (96 353)

    # On simule la feature cross_state_delivery avec la VRAIE
    # proportion mesuree : 63,8% des commandes sont inter-etats.
    cross_state = np.random.random(n) < 0.638

    # Construction du tableau de features (les colonnes que le modele
    # va utiliser pour faire sa prediction). Chaque ligne represente
    # une commande simulee.
    X = pd.DataFrame({
        "nb_items":             np.random.randint(1, 6, n),
        "nb_sellers":           np.random.randint(1, 3, n),
        # np.random.lognormal genere des valeurs qui imitent la forme
        # typique d'une distribution de prix reels (beaucoup de valeurs
        # faibles, quelques valeurs elevees) plutot qu'une distribution
        # uniforme irrealiste.
        "total_price":          np.round(np.random.lognormal(4.5, 1.0, n), 2),
        "total_freight":        np.round(np.random.lognormal(2.8, 0.6, n), 2),
        "cross_state_delivery": cross_state.astype(int),  # conversion du booleen (True/False) en nombre (1/0)
        "purchase_weekday":     np.random.randint(1, 8, n),
        "purchase_hour":        np.random.randint(0, 24, n),
        "estimated_weight_kg":  np.round(np.random.lognormal(0.5, 1.0, n), 2),
        "max_installments":     np.random.randint(1, 10, n),
    })

    # Construction de la colonne CIBLE (ce que le modele doit apprendre
    # a predire). On simule un taux de retard de base de 6,65% (le
    # chiffre reel mesure), MAJORE de 1,8x si la livraison est
    # inter-etats, et REDUIT a 0,6x si elle est dans le meme etat -
    # ce qui reproduit fidelement l'ecart de risque observe dans les
    # vraies donnees du Projet #6.
    base_proba = 0.0665
    proba = np.where(cross_state, base_proba * 1.8, base_proba * 0.6)
    y = (np.random.random(n) < proba).astype(int)

    logger.info(f"Donnees chargees : {len(X):,} commandes, taux retard simule : {y.mean():.2%}")
    return X, y


def train_xgboost(X_train, y_train, X_test, y_test, params: dict) -> tuple:
    """
    Entraine le modele XGBoost et calcule ses metriques de performance
    sur les donnees de TEST (jamais vues pendant l'entrainement).

    LES 4 METRIQUES UTILISEES, EXPLIQUEES SIMPLEMENT :

    PRECISION : parmi toutes les commandes que le modele a signalees
    "a risque de retard", quel pourcentage l'etait vraiment ?
    Une precision faible signifie beaucoup de "fausses alertes", ce qui
    fatigue inutilement l'equipe logistique qui finit par ignorer les
    alertes.

    RAPPEL (recall) : parmi toutes les commandes VRAIMENT en retard,
    quel pourcentage le modele a-t-il correctement detecte AVANT
    l'expedition ? Un rappel faible signifie que beaucoup de retards
    "passent sous le radar" sans alerte, ce qui annule l'interet du
    projet.
    Il y a souvent un compromis entre precision et rappel : on peut
    augmenter le rappel (detecter plus de retards) au prix de plus de
    fausses alertes (precision plus faible), et inversement.

    F1-SCORE : une moyenne equilibree entre precision et rappel,
    pratique pour avoir un seul chiffre de synthese.

    ROC AUC : mesure la capacite GLOBALE du modele a bien separer les
    commandes a risque des commandes sans risque, independamment du
    seuil de decision choisi. Plus proche de 1, meilleur est le modele
    (0,5 = equivalent a un tirage au sort).
    """
    model = xgb.XGBClassifier(**params)
    # .fit() est l'etape d'ENTRAINEMENT proprement dite : le modele
    # ajuste ses parametres internes pour minimiser ses erreurs sur
    # les donnees d'entrainement (X_train, y_train).
    # eval_set permet de suivre en parallele la performance sur les
    # donnees de test, ce qui aide a detecter un "surapprentissage"
    # (le modele qui "recite" les exemples d'entrainement sans savoir
    # generaliser a de nouvelles donnees).
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # .predict() renvoie une prediction binaire (0 ou 1) pour chaque
    # ligne de X_test.
    y_pred  = model.predict(X_test)
    # .predict_proba() renvoie une PROBABILITE continue (utile pour
    # le calcul du ROC AUC, qui a besoin de nuances plus fines qu'un
    # simple 0/1).
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    return model, metrics


def promote_to_production(client: MlflowClient, version: str) -> None:
    """
    Change le "stage" (statut) d'une version de modele dans MLflow,
    pour la marquer comme officiellement utilisable en Production.

    C'est cette etiquette "Production" que l'API Flask (cf
    06_api/api.py) va chercher au demarrage pour savoir quelle version
    du modele charger.

    archive_existing_versions=True : la version PRECEDEMMENT en
    Production est automatiquement archivee (passee au statut
    "Archived"), pour qu'il n'y ait jamais deux versions marquees
    "Production" en meme temps, ce qui creerait une ambiguite.
    """
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version,
        stage="Production", archive_existing_versions=True
    )
    logger.info(f"Modele {MODEL_NAME} v{version} -> Production")


def run_training() -> dict:
    """
    Fonction "chef d'orchestre" qui enchaine toutes les etapes :
    chargement des donnees, decoupage entrainement/test, entrainement,
    enregistrement dans MLflow, et decision automatique de deploiement.
    """
    # On indique a MLflow ou se trouve son serveur de suivi, et dans
    # quel "experiment" (dossier logique) ranger ce nouvel entrainement.
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_training_data()

    # Decoupage 80% / 20% : 80% des donnees servent a entrainer le
    # modele, 20% sont mises de cote pour le tester ensuite sur des
    # exemples qu'il n'a jamais vus.
    # stratify=y garantit que la PROPORTION de retards (6,65%) est
    # identique dans le groupe d'entrainement et dans le groupe de
    # test, ce qui est important car notre classe "retard" est
    # minoritaire (peu d'exemples positifs par rapport au total).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── PARAMETRES DE L'ALGORITHME XGBOOST ─────────────────────────
    params = {
        "n_estimators":     500,   # nombre d'arbres de decision combines par le modele
        "max_depth":        5,     # profondeur maximale de chaque arbre (limite la complexite, evite le surapprentissage)
        "learning_rate":    0.05,  # vitesse d'apprentissage : plus c'est petit, plus l'apprentissage est progressif et stable (mais plus lent)
        "subsample":        0.8,   # a chaque arbre, on n'utilise que 80% des lignes d'entrainement, ce qui ajoute de la diversite et reduit le surapprentissage
        "colsample_bytree": 0.8,   # idem, mais pour les colonnes (features) plutot que les lignes
        "min_child_weight": 3,     # contraint la taille minimale des groupes formes par chaque arbre, evite des decisions basees sur trop peu d'exemples
        # scale_pos_weight compense le DESEQUILIBRE des classes : comme
        # seulement ~6,65% des commandes sont en retard, sans ce
        # parametre le modele aurait tendance a "tricher" en predisant
        # presque toujours "pas de retard" (ce qui donnerait une
        # precision globale trompeuse mais un rappel catastrophique).
        # Ce calcul donne plus de POIDS aux exemples rares (les
        # retards) pendant l'entrainement.
        "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        "random_state":     42,    # graine aleatoire fixe, pour la reproductibilite
        "eval_metric":      "logloss",
    }

    run_name = f"train_{datetime.now().strftime('%Y-%m-%d')}"

    # "with mlflow.start_run(...)" ouvre un "run" (une session
    # d'enregistrement) MLflow. Tout ce qui est enregistre a
    # l'interieur de ce bloc (parametres, metriques, modele) sera
    # regroupe sous ce meme run, consultable plus tard dans
    # l'interface web de MLflow.
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Demarrage run MLflow : {run.info.run_id}")

        model, metrics = train_xgboost(X_train, y_train, X_test, y_test, params)

        # mlflow.log_params() enregistre tous les parametres utilises
        # pour cet entrainement (utile pour comparer plus tard
        # plusieurs versions et comprendre pourquoi l'une est
        # meilleure qu'une autre).
        mlflow.log_params(params)
        # mlflow.log_metrics() enregistre les resultats de performance mesures.
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test",  len(X_test))
        mlflow.log_param("base_rate", float(y_train.mean()))

        # On calcule aussi l'importance relative de chaque feature
        # dans les decisions du modele (par exemple, cross_state_delivery
        # devrait ressortir comme la feature la plus importante, ce qui
        # confirmerait l'analyse faite au Projet #6). On enregistre ce
        # detail comme un fichier JSON associe au run, utile pour
        # l'explicabilite du modele (exigence du Bloc 4).
        fi = pd.Series(model.feature_importances_,
                       index=X_train.columns).sort_values(ascending=False)
        mlflow.log_dict(fi.to_dict(), "feature_importance.json")

        # mlflow.xgboost.log_model() sauvegarde le modele lui-meme
        # (tous les fichiers necessaires pour le recharger et l'utiliser
        # plus tard, exactement comme le fait l'API dans 06_api/api.py).
        # registered_model_name enregistre automatiquement cette
        # version dans le "Model Registry" de MLflow, le catalogue
        # versionne de tous les modeles.
        mlflow.xgboost.log_model(
            model, "model", registered_model_name=MODEL_NAME,
            input_example=X_test.head(3),  # quelques exemples de donnees d'entree, utiles pour documenter le format attendu
        )

        logger.info(f"Precision = {metrics['precision']:.2%}")
        logger.info(f"Rappel    = {metrics['recall']:.2%}")
        logger.info(f"F1-score  = {metrics['f1_score']:.2%}")
        logger.info(f"ROC AUC   = {metrics['roc_auc']:.3f}")

        # ── DECISION AUTOMATIQUE DE DEPLOIEMENT ─────────────────────
        # On ne deploie EN PRODUCTION que si le modele atteint les
        # deux seuils minimum definis en haut du fichier. C'est ce
        # garde-fou qui evite de degrader le service avec un modele
        # mediocre, meme si l'entrainement technique s'est bien passe.
        if metrics["precision"] >= MIN_PRECISION and metrics["recall"] >= MIN_RECALL:
            client = MlflowClient()
            # On recupere la version qu'on vient d'enregistrer (toujours
            # au stage "None" juste apres son enregistrement, avant
            # toute promotion).
            versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                promote_to_production(client, latest.version)
                logger.info("Modele deploye en Production automatiquement")
            metrics["deployed"] = True
        else:
            logger.warning(
                f"Metriques insuffisantes (precision={metrics['precision']:.2%}, "
                f"rappel={metrics['recall']:.2%}) - deploiement manuel requis"
            )
            metrics["deployed"] = False

        metrics["run_id"] = run.info.run_id
        return metrics


# ====================================================================
# POINT D'ENTREE DU SCRIPT
# ====================================================================
if __name__ == "__main__":
    results = run_training()
    logger.info(f"=== Entrainement termine - run_id : {results['run_id']} ===")
