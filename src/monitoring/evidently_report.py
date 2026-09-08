"""
====================================================================
Projet #7 - ShopBR - Monitoring Evidently AI
Detection de derive du modele et des donnees
====================================================================

A QUOI SERT CE SCRIPT ?
Un modele de Machine Learning n'est jamais "fini" : une fois en
production, le monde reel continue d'evoluer (nouveaux vendeurs,
nouvelles habitudes d'achat, changement de transporteurs...), et les
patterns que le modele a appris peuvent devenir progressivement
obsoletes. C'est ce qu'on appelle la "DERIVE" (ou "drift" en anglais).

Ce script compare PERIODIQUEMENT les donnees recentes a une periode
de reference, pour detecter automatiquement si une derive est en
train de se produire, AVANT qu'elle ne degrade silencieusement la
qualite des predictions en production.

DEUX TYPES DE DERIVE SURVEILLES ICI :

1. DATA DRIFT (derive des donnees) : est-ce que les CARACTERISTIQUES
   des commandes ont change ? Par exemple, si soudainement beaucoup
   plus de commandes deviennent inter-etats (cross_state_delivery),
   le modele entraine sur l'ancienne proportion pourrait devenir
   moins precis.

2. MODEL DRIFT / PERFORMANCE (derive de performance) : est-ce que la
   PRECISION et le RAPPEL du modele se degradent dans le temps, meme
   si les donnees elles-memes n'ont pas beaucoup change ?

QU'EST-CE QU'EVIDENTLY AI ?
C'est une bibliotheque specialisee dans la generation de rapports
visuels (au format HTML) qui comparent statistiquement deux jeux de
donnees, et qui peut aussi executer des "tests" automatiques
(pass/fail) pour declencher des alertes sans intervention humaine.
====================================================================
"""

import pandas as pd
import numpy as np
# Report : genere un rapport DETAILLE (avec graphiques) au format HTML
from evidently.report import Report
# Les "presets" sont des ensembles pre-configures de metriques
# couramment utilisees, pour ne pas avoir a tout choisir manuellement
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
# TestSuite : contrairement a Report (qui est descriptif), TestSuite
# execute des verifications avec un resultat binaire SUCCESS/FAIL,
# adapte pour declencher des alertes automatiques.
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR     = Path("05_monitoring/reports")
# Si plus de 30% des colonnes montrent une derive statistiquement
# significative, on considere que la situation merite une investigation.
DRIFT_THRESHOLD = 0.3
# Liste des features (colonnes) du modele qu'on surveille pour la derive.
COLUMNS = ["nb_items", "total_price", "total_freight", "cross_state_delivery",
          "purchase_weekday", "purchase_hour", "estimated_weight_kg"]


def generate_reference_data(period: str, n: int = 10000) -> pd.DataFrame:
    """
    Genere les donnees de la periode de REFERENCE (2016-2017, le debut
    du dataset), qui servent de point de comparaison "normal".

    Comme pour les autres scripts d'entrainement de ce projet, ces
    donnees sont SYNTHETIQUES mais calibrees sur des statistiques
    coherentes avec le dataset Olist reel, dans un environnement de
    demonstration ou nous n'avons pas acces direct a la base
    operationnelle complete.
    """
    np.random.seed(42)
    # On simule une proportion legerement DIFFERENTE de livraisons
    # inter-etats sur cette periode plus ancienne (60% au lieu du
    # 63,8% mesure recemment), pour illustrer une derive PROGRESSIVE
    # et realiste plutot qu'un changement brutal artificiel.
    cross_state = np.random.random(n) < 0.60
    X = pd.DataFrame({
        "nb_items":             np.random.randint(1, 6, n),
        "total_price":          np.round(np.random.lognormal(4.4, 1.0, n), 2),
        "total_freight":        np.round(np.random.lognormal(2.7, 0.6, n), 2),
        "cross_state_delivery": cross_state.astype(int),
        "purchase_weekday":     np.random.randint(1, 8, n),
        "purchase_hour":        np.random.randint(0, 24, n),
        "estimated_weight_kg":  np.round(np.random.lognormal(0.5, 1.0, n), 2),
    })
    base_proba = 0.062
    proba = np.where(cross_state, base_proba * 1.7, base_proba * 0.6)
    # "target" = la vraie valeur observee (la commande etait-elle
    # vraiment en retard ?)
    X["target"]     = (np.random.random(n) < proba).astype(int)
    # "prediction" = ce que le modele (legerement different a cette
    # epoque) aurait predit. On simule un leger ecart par rapport au
    # target, comme dans la realite ou aucun modele n'est parfait.
    X["prediction"] = (np.random.random(n) < proba * 0.95).astype(int)
    return X


def generate_current_data(period: str, n: int = 10000) -> pd.DataFrame:
    """
    Genere les donnees de la periode COURANTE (2018), calibrees sur
    les VRAIES statistiques mesurees dans le Projet #6 (proportion
    inter-etats de 63,8%, taux de retard global de 6,65%).
    """
    np.random.seed(99)
    cross_state = np.random.random(n) < 0.638
    X = pd.DataFrame({
        "nb_items":             np.random.randint(1, 6, n),
        "total_price":          np.round(np.random.lognormal(4.5, 1.0, n), 2),
        "total_freight":        np.round(np.random.lognormal(2.8, 0.6, n), 2),
        "cross_state_delivery": cross_state.astype(int),
        "purchase_weekday":     np.random.randint(1, 8, n),
        "purchase_hour":        np.random.randint(0, 24, n),
        "estimated_weight_kg":  np.round(np.random.lognormal(0.5, 1.0, n), 2),
    })
    base_proba = 0.0665
    proba = np.where(cross_state, base_proba * 1.8, base_proba * 0.6)
    X["target"]     = (np.random.random(n) < proba).astype(int)
    X["prediction"] = (np.random.random(n) < proba * 0.92).astype(int)
    return X


def run_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                          output_path: Path) -> dict:
    """
    Genere le rapport de DATA DRIFT : compare statistiquement la
    distribution de chaque feature entre les deux periodes (par
    exemple, est-ce que la repartition des prix a significativement
    change ?).
    """
    logger.info("Generation du rapport Data Drift...")

    # Report(metrics=[...]) : on choisit quelles analyses inclure dans
    # le rapport. DatasetDriftMetric() donne une vue d'ensemble, et on
    # ajoute des ColumnDriftMetric() specifiques pour les 3 colonnes
    # qu'on juge les plus critiques a surveiller individuellement.
    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name="cross_state_delivery"),
        ColumnDriftMetric(column_name="total_price"),
        ColumnDriftMetric(column_name="purchase_weekday"),
    ])
    # .run() execute reellement les calculs de comparaison statistique
    # entre les deux jeux de donnees.
    report.run(reference_data=reference[COLUMNS], current_data=current[COLUMNS])

    html_path = output_path / "data_drift_report.html"
    # .save_html() genere un fichier HTML interactif et visuel
    # (graphiques, tableaux), consultable dans un navigateur, que
    # l'equipe data peut ouvrir pour investiguer en detail.
    report.save_html(str(html_path))

    # .as_dict() permet aussi de recuperer les resultats sous forme de
    # donnees Python exploitables par du code (par exemple pour
    # decider automatiquement de declencher une alerte), en plus du
    # rapport visuel.
    results = report.as_dict()
    drift_detected = results["metrics"][0]["result"]["dataset_drift"]
    share_drifted  = results["metrics"][0]["result"]["share_of_drifted_columns"]

    return {"drift_detected": drift_detected, "share_drifted": share_drifted,
            "report_path": str(html_path)}


def run_classification_report(reference: pd.DataFrame, current: pd.DataFrame,
                               output_path: Path) -> dict:
    """
    Genere le rapport de PERFORMANCE DE CLASSIFICATION : compare la
    precision et le rappel du modele entre les deux periodes,
    directement a partir des colonnes "target" (verite) et
    "prediction" (ce que le modele a predit).
    """
    logger.info("Generation du rapport de classification...")
    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=reference[["target", "prediction"]],
        current_data=current[["target", "prediction"]],
    )
    html_path = output_path / "classification_report.html"
    report.save_html(str(html_path))

    results = report.as_dict()
    current_metrics = results["metrics"][0]["result"].get("current", {})
    precision = current_metrics.get("precision", 0)
    recall    = current_metrics.get("recall", 0)

    return {"precision": precision, "recall": recall, "report_path": str(html_path)}


def run_test_suite(reference: pd.DataFrame, current: pd.DataFrame,
                   output_path: Path) -> dict:
    """
    Execute une suite de TESTS AUTOMATISES (resultat binaire SUCCESS
    ou FAIL pour chaque test), contrairement aux rapports precedents
    qui sont surtout descriptifs.

    C'est cette fonction qui permet une integration directe avec le
    DAG Airflow : un resultat de test clair (passe/echoue) peut
    facilement declencher une decision automatique (alerter
    l'equipe, forcer un reentrainement anticipe, etc.)
    """
    logger.info("Execution de la suite de tests Evidently...")
    suite = TestSuite(tests=[
        # On echoue le test si 3 colonnes ou plus montrent une derive
        TestNumberOfDriftedColumns(lt=3),
        # On echoue aussi si plus de 30% des colonnes derivent
        TestShareOfDriftedColumns(lt=DRIFT_THRESHOLD),
    ])
    suite.run(reference_data=reference[COLUMNS], current_data=current[COLUMNS])

    html_path = output_path / "test_suite_report.html"
    suite.save_html(str(html_path))

    results = suite.as_dict()
    # all() verifie que TOUS les tests de la liste ont reussi
    all_passed = all(t["result"]["status"] == "SUCCESS" for t in results["tests"])
    return {"all_passed": all_passed, "report_path": str(html_path)}


def generate_summary_report(drift_results, class_results, test_results,
                            reference_period, current_period, output_path):
    """
    Compile les resultats des 3 analyses precedentes en UN SEUL fichier
    JSON de synthese, facile a lire automatiquement par le DAG Airflow
    (qui n'a pas besoin d'ouvrir les rapports HTML detailles, juste de
    savoir "est-ce que tout va bien, oui ou non ?").
    """
    import json
    summary = {
        "generated_at":     datetime.utcnow().isoformat(),
        "reference_period": reference_period,
        "current_period":   current_period,
        "data_drift": {
            "detected":      drift_results["drift_detected"],
            "share_drifted": round(drift_results["share_drifted"], 3),
            "status":        "DERIVE DETECTEE" if drift_results["drift_detected"] else "Stable",
        },
        "model_performance": {
            "precision": round(class_results["precision"], 3),
            "recall":    round(class_results["recall"], 3),
            "status":    "DEGRADATION" if class_results["recall"] < 0.65 else "Acceptable",
        },
        "tests": {
            "all_passed": test_results["all_passed"],
            "status":     "OK" if test_results["all_passed"] else "TESTS ECHOUES",
        },
    }
    # "action_required" resume en un seul booleen si une intervention
    # humaine est necessaire, peu importe la raison precise.
    summary["action_required"] = (
        drift_results["drift_detected"]
        or class_results["recall"] < 0.65
        or not test_results["all_passed"]
    )

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Resume Evidently :")
    logger.info(f"  Data drift  : {summary['data_drift']['status']}")
    logger.info(f"  Performance : {summary['model_performance']['status']}")
    logger.info(f"  Tests       : {summary['tests']['status']}")
    if summary["action_required"]:
        logger.warning("Action requise - reentrainement ou investigation recommandee")


def run_monitoring(reference_period: str = "2016-2017",
                   current_period: str = "2018") -> None:
    """
    Fonction "chef d'orchestre" qui execute les 3 analyses dans
    l'ordre et produit le rapport de synthese.
    """
    # On cree un sous-dossier dedie pour les rapports de cette execution
    # (par exemple "05_monitoring/reports/2018/")
    output_path = REPORTS_DIR / current_period.replace("-", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Monitoring Evidently AI - Reference: {reference_period} / Courant: {current_period}")

    reference = generate_reference_data(reference_period)
    current   = generate_current_data(current_period)

    drift_results = run_data_drift_report(reference, current, output_path)
    class_results = run_classification_report(reference, current, output_path)
    test_results  = run_test_suite(reference, current, output_path)

    generate_summary_report(drift_results, class_results, test_results,
                            reference_period, current_period, output_path)
    logger.info(f"=== Monitoring termine - rapports dans {output_path} ===")


# ====================================================================
# POINT D'ENTREE DU SCRIPT
# ====================================================================
if __name__ == "__main__":
    # argparse permet de passer des parametres en ligne de commande,
    # par exemple :
    # python evidently_report.py --reference-period 2016-2017 --current-period 2018
    parser = argparse.ArgumentParser(description="Monitoring Evidently AI ShopBR")
    parser.add_argument("--reference-period", default="2016-2017")
    parser.add_argument("--current-period",   default="2018")
    args = parser.parse_args()
    run_monitoring(args.reference_period, args.current_period)
