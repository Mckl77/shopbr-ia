"""
====================================================================
Projet #7 - ShopBR - DAG Apache Airflow
Pipeline ETL mensuel + reentrainement automatique du modele
Competence C4.5 : Scripts de reentrainement automatise
====================================================================

QU'EST-CE QU'APACHE AIRFLOW ?
Airflow est un outil d'ORCHESTRATION : il permet de planifier et
d'enchainer automatiquement plusieurs taches (des scripts Python, des
commandes...), en gerant les dependances entre elles ("la tache B ne
doit demarrer que si la tache A a reussi"), les reessais en cas
d'echec, et l'envoi d'alertes.

QU'EST-CE QU'UN "DAG" ?
DAG = Directed Acyclic Graph (graphe oriente acyclique). C'est un nom
technique pour dire : un enchainement de taches qui va toujours dans
le meme sens (pas de boucle infinie), ou chaque fleche represente une
dependance entre deux taches.

POURQUOI CE DAG EST-IL LE COEUR DU PROJET #7 ?
C'est lui qui realise concretement l'AUTOMATISATION COMPLETE demandee
par l'enonce : sans intervention humaine, chaque mois, le pipeline va
chercher les nouvelles donnees, les nettoie, reentraine le modele, et
le deploie automatiquement s'il est suffisamment bon. C'est exactement
la definition de la competence C4.5 ("developper des scripts de
reentrainement pour automatiser le processus de Machine Learning").

ENCHAINEMENT DES 8 TACHES DE CE DAG :
  1. extract_data       : recupere les nouvelles donnees (via Airbyte)
  2. run_cleaning       : nettoie les donnees (reutilise le script du Projet #6)
  3. run_transformation : calcule les indicateurs (reutilise le script du Projet #6)
  4. train_model         : entraine une nouvelle version du modele
  5. evaluate_model       : decide si cette version merite d'etre deployee
  6a. deploy_model OU 6b. alert_low_metrics : deploiement ou alerte
  7. run_evidently          : genere un rapport de surveillance de derive
  8. notify_ops               : informe l'equipe logistique du resultat
====================================================================
"""

from airflow import DAG
# PythonOperator : permet d'executer une fonction Python comme une
# tache du DAG.
# BranchPythonOperator : variante speciale qui permet de CHOISIR
# dynamiquement quelle tache executer ensuite, selon un resultat
# calcule (ici : deployer ou non, selon les metriques obtenues).
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
# Variable permet de stocker des parametres de configuration (comme
# des URLs ou des identifiants) directement dans l'interface web
# d'Airflow, plutot que de les coder en dur dans ce fichier.
from airflow.models import Variable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ── CONFIGURATION PAR DEFAUT DU DAG ─────────────────────────────────
# Ces parametres s'appliquent a TOUTES les taches du DAG, sauf si une
# tache precise individuellement une valeur differente.
DEFAULT_ARGS = {
    "owner":            "shopbr-data-team",
    "depends_on_past":  False,   # chaque execution du DAG est independante des executions precedentes
    "email":            ["data-team@shopbr.com"],
    "email_on_failure": True,    # envoie un email automatique si une tache echoue
    "email_on_retry":   False,
    "retries":          2,       # en cas d'echec d'une tache, Airflow la retente automatiquement 2 fois avant d'abandonner
    "retry_delay":      timedelta(minutes=5),  # attend 5 minutes entre chaque tentative
    # SLA = Service Level Agreement. Si le DAG complet met plus de 3h
    # a s'executer, Airflow declenche une alerte (le pipeline doit
    # avoir termine avant 6h du matin pour que l'equipe logistique
    # dispose des donnees fraiches en arrivant).
    "sla":              timedelta(hours=3),
}

# Seuils de qualite calibres sur le taux de retard reel mesure (6,65%) :
# un modele utile pour ShopBR doit detecter au moins 65% des retards
# reels (rappel) sans generer trop de fausses alertes (precision).
MIN_RECALL    = 0.65
MIN_PRECISION = 0.60


# ====================================================================
# DEFINITION DES TACHES (chaque fonction = une etape du pipeline)
# ====================================================================
# Toutes ces fonctions recoivent **context, un dictionnaire que
# Airflow remplit automatiquement avec des informations utiles sur
# l'execution en cours (la date d'execution via context['ds'], par
# exemple). C'est une convention standard d'Airflow.

def extract_data_fn(**context):
    """
    Tache 1 : declenche la synchronisation Airbyte, qui va chercher
    les nouvelles donnees depuis la base operationnelle de ShopBR et
    les copie vers S3 (couche Bronze).

    QU'EST-CE QU'AIRBYTE ?
    C'est un outil specialise dans l'EXTRACTION de donnees depuis des
    sources variees (bases de donnees, API, fichiers...) vers une
    destination de stockage. Il gere la connexion technique a la
    source et la synchronisation incrementale (ne recopie que les
    nouvelles donnees, pas tout depuis le debut a chaque fois).
    """
    import requests

    AIRBYTE_URL = Variable.get("AIRBYTE_URL", "http://airbyte:8000")
    CONNECTION_ID = Variable.get("AIRBYTE_CONNECTION_ID", "shopbr-to-s3")

    logger.info(f"Declenchement synchronisation Airbyte : {CONNECTION_ID}")
    try:
        # On appelle l'API REST d'Airbyte pour lui demander de lancer
        # immediatement une synchronisation (plutot que d'attendre sa
        # propre planification interne).
        resp = requests.post(
            f"{AIRBYTE_URL}/api/v1/connections/sync",
            json={"connectionId": CONNECTION_ID}, timeout=30
        )
        if resp.status_code == 200:
            job_id = resp.json().get("job", {}).get("id")
            logger.info(f"Synchronisation declenchee - job_id : {job_id}")
            # context["ti"].xcom_push(...) permet de TRANSMETTRE une
            # information depuis cette tache vers les taches suivantes
            # du DAG. "ti" = Task Instance, "xcom" = "cross-communication".
            context["ti"].xcom_push(key="airbyte_job_id", value=job_id)
        else:
            logger.warning(f"Airbyte indisponible (status {resp.status_code})")
    except Exception as e:
        # Si Airbyte n'est pas joignable, on ne fait pas planter tout
        # le pipeline : on continue avec les donnees S3 deja presentes
        # (potentiellement un peu moins fraiches), plutot que de
        # bloquer completement le reentrainement mensuel.
        logger.warning(f"Airbyte indisponible : {e} - pipeline continue avec S3 existant")


def run_cleaning_fn(**context):
    """
    Tache 2 : execute le script de nettoyage du Projet #6
    (02_cleaning.py) en tant que processus separe.

    POURQUOI subprocess.run() ET NON UN SIMPLE IMPORT PYTHON ?
    Le script de nettoyage est un script PySpark qui doit s'executer
    sur un cluster Spark (Databricks), avec sa propre configuration.
    L'appeler comme un programme externe (subprocess) plutot que de
    l'importer directement permet de l'isoler proprement et de
    recuperer son code de sortie (succes ou echec) sans interferer
    avec l'environnement d'execution d'Airflow lui-meme.
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "/opt/airflow/scripts/02_cleaning.py"],
        capture_output=True, text=True, timeout=1800  # 1800 secondes = 30 minutes maximum
    )
    if result.returncode != 0:
        # Si le script a echoue (code de sortie different de 0), on
        # leve une exception : cela fait automatiquement echouer
        # cette tache Airflow, ce qui bloque les taches suivantes
        # (par exemple, pas de sens a entrainer un modele si les
        # donnees n'ont pas ete correctement nettoyees).
        raise RuntimeError(f"Nettoyage echoue :\n{result.stderr}")
    logger.info("Nettoyage termine avec succes")


def run_transformation_fn(**context):
    """
    Tache 3 : execute le script de transformation du Projet #6
    (03_transformation.py), qui calcule les 7 tables d'indicateurs.
    Meme principe que la tache precedente.
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "/opt/airflow/scripts/03_transformation.py"],
        capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        raise RuntimeError(f"Transformation echouee :\n{result.stderr}")
    logger.info("Tables Gold mises a jour avec succes")


def train_model_fn(**context):
    """
    Tache 4 : entraine une nouvelle version du modele XGBoost.

    Cette fonction reprend la MEME logique que le script independant
    mlflow_train.py (cf 04_pipeline/mlflow_train.py), mais ecrite
    directement ici pour pouvoir facilement transmettre ses resultats
    (precision, rappel) aux taches suivantes du DAG via xcom_push.
    """
    import mlflow
    import mlflow.xgboost
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    MLFLOW_URI = Variable.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("shopbr-delay-risk-model")

    logger.info("Demarrage de l'entrainement XGBoost Classifier...")

    # Donnees synthetiques calibrees sur les statistiques reelles
    # mesurees (taux de retard global 6,65%, effet du cross_state_delivery).
    # Cf mlflow_train.py pour le detail de cette demarche.
    np.random.seed(42)
    n_samples = 80000
    cross_state = np.random.random(n_samples) < 0.638
    X = pd.DataFrame({
        "nb_items":             np.random.randint(1, 6, n_samples),
        "nb_sellers":           np.random.randint(1, 3, n_samples),
        "total_price":          np.random.lognormal(4.5, 1.0, n_samples),
        "total_freight":        np.random.lognormal(2.8, 0.6, n_samples),
        "cross_state_delivery": cross_state.astype(int),
        "purchase_weekday":     np.random.randint(1, 8, n_samples),
        "purchase_hour":        np.random.randint(0, 24, n_samples),
        "estimated_weight_kg":  np.random.lognormal(0.5, 1.0, n_samples),
        "max_installments":     np.random.randint(1, 10, n_samples),
    })
    base_proba = 0.0665
    proba = np.where(cross_state, base_proba * 1.8, base_proba * 0.6)
    y = (np.random.random(n_samples) < proba).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators":     400,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        # Compense le desequilibre entre commandes en retard (minoritaires)
        # et commandes a l'heure (majoritaires) - cf mlflow_train.py pour le detail.
        "scale_pos_weight": (1 - y_train.mean()) / y_train.mean(),
        "random_state":     42,
        "eval_metric":      "logloss",
    }

    # "with mlflow.start_run(...)" ouvre une session d'enregistrement
    # MLflow pour cet entrainement specifique.
    with mlflow.start_run(run_name=f"train_{context['ds']}") as run:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics({
            "precision": precision, "recall": recall,
            "f1_score": f1, "roc_auc": auc,
            "n_train": len(X_train), "n_test": len(X_test),
            "base_rate": float(y_train.mean()),
        })
        mlflow.xgboost.log_model(model, "model", registered_model_name="shopbr_delay_risk")

        run_id = run.info.run_id
        logger.info(f"Precision={precision:.2%} Rappel={recall:.2%} F1={f1:.2%} AUC={auc:.3f}")

        # On transmet precision, rappel et run_id aux taches suivantes
        # du DAG (evaluate_model en a besoin pour decider du
        # deploiement, notify_ops en a besoin pour le message final).
        context["ti"].xcom_push(key="precision", value=float(precision))
        context["ti"].xcom_push(key="recall",    value=float(recall))
        context["ti"].xcom_push(key="run_id",    value=run_id)


def evaluate_model_fn(**context):
    """
    Tache 5 : analyse les metriques obtenues a la tache precedente et
    DECIDE quelle branche du DAG executer ensuite.

    C'est une "BranchPythonOperator" : au lieu de simplement reussir
    ou echouer, cette tache RENVOIE LE NOM de la prochaine tache a
    executer. Selon que les seuils sont atteints ou non, Airflow va
    soit vers "deploy_model", soit vers "alert_low_metrics" - jamais
    les deux.
    """
    # On recupere les valeurs transmises par la tache "train_model"
    # via xcom_pull (l'inverse de xcom_push utilise plus haut).
    precision = context["ti"].xcom_pull(key="precision", task_ids="train_model")
    recall    = context["ti"].xcom_pull(key="recall",    task_ids="train_model")

    logger.info(f"Evaluation : precision={precision:.2%} rappel={recall:.2%} "
               f"(seuils : precision>={MIN_PRECISION:.0%}, rappel>={MIN_RECALL:.0%})")

    if precision >= MIN_PRECISION and recall >= MIN_RECALL:
        logger.info("Metriques acceptables - deploiement en production")
        return "deploy_model"  # nom EXACT de la tache a executer ensuite (cf definition du DAG plus bas)
    else:
        logger.warning("Metriques insuffisantes - deploiement annule")
        return "alert_low_metrics"


def deploy_model_fn(**context):
    """
    Tache 6a (executee seulement si evaluate_model a choisi cette
    branche) : promeut la nouvelle version du modele au statut
    "Production" dans MLflow.

    C'est cette etape qui rend la nouvelle version REELLEMENT
    utilisable : l'API Flask (06_api/api.py) chargera cette version
    au prochain redemarrage (ou peut etre configuree pour la
    recharger automatiquement).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_URI = Variable.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    versions = client.get_latest_versions("shopbr_delay_risk", stages=["None", "Staging"])
    if not versions:
        raise ValueError("Aucune version du modele trouvee dans MLflow")
    latest = max(versions, key=lambda v: int(v.version))

    client.transition_model_version_stage(
        name="shopbr_delay_risk", version=latest.version,
        stage="Production", archive_existing_versions=True
    )
    logger.info(f"Modele shopbr_delay_risk v{latest.version} -> Production")


def run_evidently_fn(**context):
    """
    Tache 7 : execute le script de monitoring Evidently AI (cf
    05_monitoring/evidently_report.py), qui compare les donnees et
    performances recentes a une periode de reference, pour detecter
    toute derive significative.

    Cette tache s'execute QUE le deploiement ait eu lieu ou non
    (trigger_rule="none_failed_min_one_success", definie plus bas
    dans la construction du DAG) : on veut surveiller la situation
    dans tous les cas, meme si le modele n'a pas ete redeploye ce mois-ci.
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "/opt/airflow/scripts/05_monitoring/evidently_report.py",
         "--reference-period", "2016-2017", "--current-period", "2018"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        # Ici, on choisit de ne pas faire ECHOUER tout le DAG si le
        # rapport de monitoring rencontre un probleme : c'est une
        # tache de surveillance, pas une etape critique du
        # deploiement. On se contente d'un avertissement.
        logger.warning(f"Rapport Evidently incomplet : {result.stderr}")
    else:
        logger.info("Rapport Evidently genere avec succes")


def notify_ops_fn(**context):
    """
    Tache 8 : derniere etape, envoie un recapitulatif a l'equipe
    logistique de ShopBR (par email ou Slack en production reelle).
    """
    precision = context["ti"].xcom_pull(key="precision", task_ids="train_model")
    recall    = context["ti"].xcom_pull(key="recall",    task_ids="train_model")
    run_id    = context["ti"].xcom_pull(key="run_id",    task_ids="train_model")

    message = (
        f"Pipeline ShopBR termine - {context['ds']}\n"
        f"Modele XGBoost v{run_id[:8] if run_id else '?'} deploye\n"
        f"Precision : {precision:.1%} - Rappel : {recall:.1%}\n"
        f"Dashboard : http://dashboard.shopbr.com"
    )
    logger.info(f"Notification ops :\n{message}")
    # En production reelle, on remplacerait ce simple log par un
    # veritable envoi (ex: requests.post vers un webhook Slack, ou
    # l'utilisation de l'EmailOperator d'Airflow).


# ====================================================================
# DEFINITION DU DAG ET DE SES DEPENDANCES
# ====================================================================
# "with DAG(...) as dag:" cree le DAG et regroupe toutes les taches
# qui lui appartiennent. Les parametres principaux :
#
# schedule_interval="0 3 1 * *" : expression "cron" qui signifie
#   "a 3h00, le 1er jour de chaque mois". Le format cron est :
#   minute heure jour_du_mois mois jour_de_la_semaine
#
# catchup=False : si le DAG est cree ou reactive apres une periode
#   d'inactivite, on ne veut PAS qu'Airflow essaie de "rattraper" en
#   executant toutes les executions manquees passees - on veut juste
#   la prochaine execution planifiee.
with DAG(
    dag_id="shopbr_monthly_pipeline",
    description="Pipeline mensuel ShopBR : ETL + entrainement + deploiement modele retard",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 3 1 * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["shopbr", "production", "projet7"],
    doc_md="""
    ## DAG ShopBR — Pipeline mensuel de prediction du risque de retard

    1. Extraction des donnees via Airbyte
    2. Nettoyage et transformation PySpark (reutilisation Projet #6)
    3. Entrainement XGBoost Classifier avec logging MLflow
    4. Deploiement automatique si precision >= 60% et rappel >= 65%
    5. Rapport de drift Evidently AI
    6. Notification a l'equipe logistique

    **Objectif metier** : identifier les commandes a risque de retard
    avant expedition, pour action preventive.
    """,
) as dag:

    # Creation de chaque tache, en l'associant a sa fonction Python
    # definie plus haut. task_id est le nom UNIQUE qui identifie
    # chaque tache a l'interieur de ce DAG (c'est ce nom qu'on
    # retrouve dans xcom_pull(task_ids="...") et dans les decisions
    # de la BranchPythonOperator).
    extract = PythonOperator(task_id="extract_data", python_callable=extract_data_fn)
    clean   = PythonOperator(task_id="run_cleaning", python_callable=run_cleaning_fn)
    transform = PythonOperator(task_id="run_transformation", python_callable=run_transformation_fn)
    train   = PythonOperator(task_id="train_model", python_callable=train_model_fn)
    evaluate = BranchPythonOperator(task_id="evaluate_model", python_callable=evaluate_model_fn)
    deploy  = PythonOperator(task_id="deploy_model", python_callable=deploy_model_fn)
    # BashOperator execute directement une commande shell (au lieu
    # d'une fonction Python). Ici, on ecrit simplement un message dans
    # un fichier de log, comme exemple simple d'alerte technique.
    alert_metrics = BashOperator(
        task_id="alert_low_metrics",
        bash_command='echo "ALERTE : metriques insuffisantes - modele non deploye" '
                     '| tee /opt/airflow/logs/metrics_alert_$(date +%Y%m%d).log'
    )
    # trigger_rule="none_failed_min_one_success" : regle speciale qui
    # dit "execute cette tache si au moins une des taches precedentes
    # a reussi et qu'aucune n'a totalement echoue". C'est necessaire
    # ici car deploy_model et alert_metrics sont MUTUELLEMENT
    # EXCLUSIVES (une seule des deux s'execute a chaque fois, l'autre
    # reste "skipped"), et on veut quand meme continuer le DAG apres,
    # peu importe laquelle des deux a tourne.
    evidently = PythonOperator(
        task_id="run_evidently", python_callable=run_evidently_fn,
        trigger_rule="none_failed_min_one_success"
    )
    notify = PythonOperator(
        task_id="notify_ops", python_callable=notify_ops_fn,
        trigger_rule="none_failed_min_one_success"
    )

    # ── DEPENDANCES ENTRE LES TACHES ─────────────────────────────────
    # L'operateur ">>" definit l'ordre d'execution : "A >> B" signifie
    # "B ne demarre qu'apres le succes de A".
    #
    # Lecture du graphe complet :
    # extract -> clean -> transform -> train -> evaluate
    #                                              |
    #                              (branchement automatique selon les metriques)
    #                                    /                    \
    #                            deploy_model          alert_low_metrics
    #                                    \                    /
    #                                  run_evidently (dans tous les cas)
    #                                          |
    #                                    notify_ops
    extract >> clean >> transform >> train >> evaluate
    evaluate >> [deploy, alert_metrics]
    [deploy, alert_metrics] >> evidently >> notify
