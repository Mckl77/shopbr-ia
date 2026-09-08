"""
====================================================================
Projet #6 - Big Data Certification | Architecte en Intelligence Artificielle
Script 04 - EXPORT des tables Gold vers Amazon Redshift
ShopBR - dataset Olist e-commerce Bresil
====================================================================

A QUOI SERT CE SCRIPT ?
C'est la toute derniere etape du pipeline Big Data. On prend les 7
tables d'indicateurs calculees au script 03 (couche Gold, stockees
sur S3 au format Delta Lake) et on les transfere dans Amazon Redshift,
qui est un ENTREPOT DE DONNEES (data warehouse).

POURQUOI UN ENTREPOT DE DONNEES SEPARE, PUISQU'ON A DEJA LES DONNEES
SUR S3 ?
S3 est excellent pour STOCKER de gros volumes de donnees a faible
cout, mais il n'est pas optimise pour faire des requetes SQL rapides
et complexes (comme "donne-moi le taux de retard moyen par etat,
trie du plus eleve au plus faible"). Redshift, lui, est specialement
concu pour ce type d'analyse rapide : c'est une base de donnees
"orientee colonnes", optimisee pour les calculs d'agregation sur de
grands volumes.

En resume : S3 = entrepot de stockage, Redshift = outil d'analyse
rapide branche sur cet entrepot.

COMMENT SE FAIT LE TRANSFERT TECHNIQUEMENT ?
On utilise le "connecteur Spark-Redshift", qui fonctionne en 2 etapes
automatiques :
  1. Spark ecrit d'abord les donnees dans un dossier temporaire sur S3
     (le "staging")
  2. Spark demande ensuite a Redshift d'executer une commande COPY,
     qui charge tres rapidement les donnees depuis ce dossier S3
     directement dans la table Redshift.
Cette methode est beaucoup plus rapide qu'un chargement ligne par
ligne classique.
====================================================================
"""

from pyspark.sql import SparkSession, DataFrame
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Chemin de lecture : les 7 tables Gold produites par le script 03
GOLD_PATH   = "dbfs:/mnt/shopbr/gold/"
# Dossier temporaire S3 utilise par le connecteur Spark-Redshift pour
# le transfert (cf explication ci-dessus)
S3_TEMP_DIR = "s3a://votre-bucket/redshift-staging/"

# ── PARAMETRES DE CONNEXION A REDSHIFT ───────────────────────────────
# sa-east-1 = la region AWS de Sao Paulo, choisie car elle est la plus
# proche geographiquement des donnees et des utilisateurs bresiliens
# (latence reduite) et car elle simplifie la conformite LGPD (donnees
# qui restent sur le territoire bresilien/sud-americain).
REDSHIFT_CONFIG = {
    "host":     "your-cluster.xxxx.sa-east-1.redshift.amazonaws.com",
    "port":     "5439",                # port standard de Redshift
    "database": "shopbr_dw",           # "dw" = data warehouse
    "user":     "admin",
    "schema":   "analytics",           # le schema (sorte de "dossier" dans la base) ou vivent nos tables
}

# Liste des 7 tables a transferer (memes noms que les dossiers produits
# par le script 03_transformation.py)
TABLES = ["late_rate_by_state", "late_rate_by_category", "seller_performance",
          "cross_state_impact", "delay_review_impact", "monthly_trend",
          "weekday_pattern"]


def get_redshift_url():
    """
    Construit l'URL de connexion JDBC vers Redshift.

    QU'EST-CE QUE JDBC ?
    JDBC (Java Database Connectivity) est un standard qui permet a une
    application (ici Spark, qui tourne sur la machine virtuelle Java)
    de se connecter a une base de donnees relationnelle comme Redshift,
    quel que soit le langage de programmation utilise au depart.

    L'URL ressemble a une adresse web : elle contient l'adresse du
    serveur, le port, le nom de la base, et des options de securite
    (ici, on force le chiffrement SSL pour que les donnees ne
    transitent jamais en clair sur le reseau).
    """
    c = REDSHIFT_CONFIG
    return (f"jdbc:redshift://{c['host']}:{c['port']}/{c['database']}"
            f"?user={c['user']}&ssl=true"
            f"&sslfactory=com.amazon.redshift.ssl.NonValidatingFactory")


def get_password():
    """
    Recupere le mot de passe Redshift de maniere SECURISEE, c'est-a-dire
    sans jamais l'ecrire en clair dans le code source.

    dbutils.secrets.get(...) est une fonctionnalite propre a Databricks :
    elle va chercher le mot de passe dans un coffre-fort numerique
    (appele "Secret Scope"), configure separement par l'administrateur.
    Le code lui-meme ne contient jamais le mot de passe reel.

    Si ce script est execute en dehors de Databricks (par exemple pour
    un test local), on essaie de recuperer le mot de passe depuis une
    variable d'environnement a la place.
    """
    try:
        return dbutils.secrets.get(scope="shopbr-scope", key="redshift-password")
    except Exception:
        import os
        pwd = os.environ.get("REDSHIFT_PASSWORD", "")
        if not pwd:
            raise ValueError("Mot de passe Redshift introuvable.")
        return pwd


def write_to_redshift(df: DataFrame, table: str, url: str, pwd: str):
    """
    Ecrit UN DataFrame Spark dans UNE table Redshift.

    Detail des options utilisees :
    - "url" / "password" : les identifiants de connexion
    - "dbtable" : le nom complet de la table cible (schema.nom_table)
    - "tempdir" : le dossier S3 temporaire utilise pour le transfert
    - "extracopyoptions" : des reglages avances pour la commande COPY
      de Redshift : ACCEPTINVCHARS (n'echoue pas sur des caracteres
      invalides, les remplace), TRUNCATECOLUMNS (coupe les valeurs
      trop longues plutot que d'echouer), STATUPDATE ON (met a jour
      automatiquement les statistiques de la table apres le chargement,
      ce qui aide Redshift a optimiser les futures requetes)
    - "preactions" : une commande SQL executee AVANT le chargement.
      Ici, on supprime la table existante pour la remplacer entierement
      a chaque execution (on dit que le script est "idempotent" : on
      peut le relancer plusieurs fois sans creer de doublons).
    - "postactions" : une commande executee APRES le chargement.
      ANALYZE recalcule les statistiques internes de la table, ce qui
      permet a Redshift de choisir le meilleur plan d'execution pour
      les requetes futures sur cette table.
    - .mode("overwrite") : indique a Spark qu'on remplace les donnees
      existantes plutot que de les ajouter a la suite (qui donnerait
      "append").
    """
    full = f"{REDSHIFT_CONFIG['schema']}.{table}"
    logger.info(f"Export -> {full} ({df.count():,} lignes)")
    (df.repartition(5)  # on limite a 5 "morceaux" pour l'ecriture : ces tables sont petites (quelques dizaines a 50 lignes), pas besoin de plus
       .write
       .format("io.github.spark_redshift_community.spark.redshift")  # le connecteur Spark-Redshift
       .option("url", url)
       .option("password", pwd)
       .option("dbtable", full)
       .option("tempdir", S3_TEMP_DIR)
       .option("extracopyoptions", "ACCEPTINVCHARS TRUNCATECOLUMNS STATUPDATE ON")
       .option("preactions",  f"DROP TABLE IF EXISTS {full};")
       .option("postactions", f"ANALYZE {full};")
       .mode("overwrite")
       .save())
    logger.info(f"Table '{full}' chargee.")


def run_export_pipeline(spark: SparkSession):
    """
    Fonction "chef d'orchestre" : boucle sur les 7 tables Gold, les lit
    depuis S3, et les ecrit une par une dans Redshift.
    """
    url = get_redshift_url()
    pwd = get_password()
    logger.info(f"Redshift : {REDSHIFT_CONFIG['host']} / {REDSHIFT_CONFIG['database']}")

    for table in TABLES:
        path = f"{GOLD_PATH}{table}/"
        # On lit la table Gold depuis S3 (format Delta Lake)
        df = spark.read.format("delta").load(path)
        # On l'ecrit dans Redshift
        write_to_redshift(df, table, url, pwd)

    logger.info("=== Export Redshift termine ===")


# ====================================================================
# POINT D'ENTREE DU SCRIPT
# ====================================================================
if __name__ == "__main__":
    try:
        spark
    except NameError:
        # spark.jars.packages indique a Spark de telecharger automatiquement
        # la bibliotheque (le "connecteur") necessaire pour parler a
        # Redshift, si elle n'est pas deja installee.
        spark = (SparkSession.builder
            .appName("ShopBR_Export_Redshift")
            .config("spark.jars.packages",
                    "io.github.spark-redshift-community:spark-redshift_2.12:6.2.0-spark_3.5")
            .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")
    run_export_pipeline(spark)
