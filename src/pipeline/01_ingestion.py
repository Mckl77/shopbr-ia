"""
====================================================================
Projet #6 - Big Data Certification | Architecte en Intelligence Artificielle
Script 01 - INGESTION des donnees ShopBR (dataset Olist e-commerce Bresil)
====================================================================

A QUOI SERT CE SCRIPT ?
Ce script est la toute premiere etape du pipeline de donnees. Il va
chercher les fichiers CSV bruts (tels qu'exportes par la base de
l'entreprise) et les copie dans notre "zone de stockage brute" (on
appelle ca la couche "Bronze" dans une architecture Big Data).

On ne transforme RIEN ici. On copie juste les donnees telles quelles,
en ajoutant seulement quelques informations de tracabilite (qui a
ingere quoi, et quand). Le nettoyage et les calculs viendront plus
tard, dans le script 02_cleaning.py.

POURQUOI UNE ETAPE SEPAREE JUSTE POUR "COPIER" LES DONNEES ?
Parce que si on a un bug dans le nettoyage, on ne veut pas avoir a
re-telecharger les fichiers sources. On garde toujours une copie
brute et intacte des donnees d'origine, comme un brouillon qu'on ne
modifie jamais.

ENVIRONNEMENT D'EXECUTION : AWS Glue (PySpark serverless, architecture v2)
Compatible Databricks pour la demonstration locale (chemins dbfs:/)
PySpark est une bibliotheque qui permet de traiter de tres gros
volumes de donnees en les repartissant automatiquement sur plusieurs
machines (un "cluster"). Contrairement a un script Python classique
qui tourne sur un seul ordinateur, PySpark peut traiter des millions
de lignes en parallele.

VOLUMETRIE REELLE DU DATASET :
99 441 commandes, 112 650 items vendus, 3 095 vendeurs,
99 441 clients, periode septembre 2016 - octobre 2018
====================================================================
"""

# ── IMPORTS ─────────────────────────────────────────────────────────
# SparkSession : c'est le "point d'entree" pour utiliser PySpark.
# Tout code PySpark commence par creer ou recuperer une SparkSession.
from pyspark.sql import SparkSession

# F est un raccourci tres courant en PySpark pour acceder a toutes les
# fonctions de transformation de donnees (compter, filtrer, renommer...)
from pyspark.sql import functions as F

# Ces classes servent a definir explicitement le "schema" d'un fichier,
# c'est-a-dire le nom et le type de chaque colonne (texte, nombre entier,
# nombre decimal, date...). C'est l'equivalent d'un plan qu'on donne a
# Spark avant de lire le fichier, pour qu'il sache a quoi s'attendre.
from pyspark.sql.types import (StructType, StructField, StringType,
                                IntegerType, FloatType, TimestampType)

# Le module logging sert a afficher des messages d'information pendant
# l'execution du script (un peu comme des "print", mais en plus
# professionnel : on peut classer les messages par niveau d'importance
# - INFO, WARNING, ERROR - et les retrouver facilement dans des logs).
import logging

# Configuration du logger : on definit le format des messages affiches
# (date + niveau d'importance + message).
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── CHEMINS DE STOCKAGE ──────────────────────────────────────────────
# Ces variables indiquent OU se trouvent les fichiers source (RAW_S3_PATH)
# et OU on va ecrire le resultat de cette etape (BRONZE_PATH).
# "s3a://" est le protocole pour acceder a un bucket Amazon S3 (un espace
# de stockage cloud) depuis Spark.
# "dbfs:/" est le systeme de fichiers propre a Databricks.
RAW_S3_PATH = "s3a://votre-bucket/shopbr/raw/"
BRONZE_PATH = "dbfs:/mnt/shopbr/bronze/"

# Ce dictionnaire liste les 8 fichiers CSV source que l'entreprise nous
# fournit chaque mois (export depuis sa base de donnees operationnelle).
# Cle = nom qu'on donnera a la table une fois en Bronze
# Valeur = nom du fichier CSV correspondant
SOURCE_FILES = {
    "orders":       "olist_orders_dataset.csv",        # les commandes
    "order_items":  "olist_order_items_dataset.csv",   # les produits achetes
    "payments":     "olist_order_payments_dataset.csv",# les paiements
    "reviews":      "olist_order_reviews_dataset.csv", # les avis clients
    "customers":    "olist_customers_dataset.csv",     # les clients
    "sellers":      "olist_sellers_dataset.csv",       # les vendeurs
    "products":     "olist_products_dataset.csv",      # le catalogue produit
    "category_translation": "product_category_name_translation.csv",  # traduction des categories
}


# ====================================================================
# DEFINITION DES SCHEMAS
# ====================================================================
# Pourquoi definir un schema a la main au lieu de laisser Spark le
# deviner automatiquement (ce qu'on appelle "l'inference de schema") ?
#
# 1. C'est BEAUCOUP plus rapide : sans schema explicite, Spark doit
#    d'abord lire tout le fichier une premiere fois juste pour deviner
#    les types de colonnes, avant de le relire une seconde fois pour
#    le traiter vraiment. Avec un schema fourni, il ne le lit qu'une fois.
#
# 2. C'est plus FIABLE : Spark peut se tromper en devinant (par exemple
#    prendre une colonne de codes postaux pour des nombres au lieu de
#    texte, ce qui ferait perdre les zeros au debut). En fixant le
#    schema nous-memes, on evite ce genre de surprise.
#
# StructType([...]) = la liste de toutes les colonnes du fichier
# StructField("nom_colonne", TypeDeLaColonne(), True/False) =
#   - "nom_colonne" : le nom exact de la colonne dans le CSV
#   - TypeDeLaColonne() : StringType (texte), IntegerType (nombre entier),
#     FloatType (nombre a virgule), TimestampType (date + heure)
#   - True/False : est-ce que cette colonne peut etre vide (nullable) ?
#     On met True presque partout car les donnees reelles contiennent
#     souvent des valeurs manquantes.
# ====================================================================

# Schema de la table des commandes : la table centrale du projet.
# C'est dans cette table qu'on trouve les dates qui nous permettront
# de calculer si une commande a ete livree en retard.
ORDERS_SCHEMA = StructType([
    StructField("order_id",                       StringType(),    True),  # identifiant unique de la commande
    StructField("customer_id",                     StringType(),    True),  # identifiant du client
    StructField("order_status",                    StringType(),    True),  # statut : delivered, shipped, canceled...
    StructField("order_purchase_timestamp",        TimestampType(), True),  # date/heure d'achat
    StructField("order_approved_at",               TimestampType(), True),  # date/heure de validation du paiement
    StructField("order_delivered_carrier_date",    TimestampType(), True),  # date de remise au transporteur
    StructField("order_delivered_customer_date",   TimestampType(), True),  # date de livraison reelle au client
    StructField("order_estimated_delivery_date",   TimestampType(), True),  # date de livraison promise au client
])

# Schema des items (produits) de chaque commande.
# Une commande peut contenir plusieurs items (plusieurs produits achetes
# en une seule fois), donc cette table peut avoir plusieurs lignes pour
# une seule commande.
ORDER_ITEMS_SCHEMA = StructType([
    StructField("order_id",            StringType(),    True),
    StructField("order_item_id",       IntegerType(),   True),  # numero de l'item dans la commande (1, 2, 3...)
    StructField("product_id",          StringType(),    True),
    StructField("seller_id",           StringType(),    True),  # quel vendeur a vendu ce produit
    StructField("shipping_limit_date", TimestampType(), True),  # date limite d'expedition par le vendeur
    StructField("price",               FloatType(),     True),  # prix du produit
    StructField("freight_value",       FloatType(),     True),  # frais de port pour ce produit
])

# Schema des paiements (une commande peut etre payee en plusieurs fois,
# par exemple en plusieurs mensualites - tres courant au Bresil)
PAYMENTS_SCHEMA = StructType([
    StructField("order_id",             StringType(),  True),
    StructField("payment_sequential",   IntegerType(), True),  # ordre du paiement si plusieurs
    StructField("payment_type",         StringType(),  True),  # carte de credit, boleto (virement local), etc.
    StructField("payment_installments", IntegerType(), True),  # nombre de mensualites choisies
    StructField("payment_value",        FloatType(),   True),  # montant de ce paiement
])

# Schema des avis clients (notes de 1 a 5 etoiles + commentaire textuel)
REVIEWS_SCHEMA = StructType([
    StructField("review_id",               StringType(),    True),
    StructField("order_id",                StringType(),    True),
    StructField("review_score",            IntegerType(),   True),  # note de 1 (tres mauvais) a 5 (excellent)
    StructField("review_comment_title",    StringType(),    True),
    StructField("review_comment_message",  StringType(),    True),
    StructField("review_creation_date",    TimestampType(), True),
    StructField("review_answer_timestamp", TimestampType(), True),
])

# Schema des clients : on garde uniquement leur localisation (ville/etat),
# jamais leur nom complet ou adresse exacte, par souci de minimisation
# des donnees personnelles (principe cle de la LGPD/RGPD).
CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id",              StringType(), True),
    StructField("customer_unique_id",       StringType(), True),  # identifiant pseudonymise du client
    StructField("customer_zip_code_prefix", StringType(), True),  # debut du code postal seulement
    StructField("customer_city",            StringType(), True),
    StructField("customer_state",           StringType(), True),  # ex: "SP" pour Sao Paulo
])

# Schema des vendeurs (meme logique que les clients : localisation seulement)
SELLERS_SCHEMA = StructType([
    StructField("seller_id",              StringType(), True),
    StructField("seller_zip_code_prefix", StringType(), True),
    StructField("seller_city",            StringType(), True),
    StructField("seller_state",           StringType(), True),
])

# Schema du catalogue produit (categorie, poids, dimensions du colis -
# des informations utiles pour comprendre pourquoi certains colis sont
# plus longs a livrer que d'autres)
PRODUCTS_SCHEMA = StructType([
    StructField("product_id",                  StringType(),  True),
    StructField("product_category_name",       StringType(),  True),  # categorie en portugais
    StructField("product_name_lenght",         IntegerType(), True),  # longueur du nom (orthographe Olist d'origine)
    StructField("product_description_lenght",  IntegerType(), True),
    StructField("product_photos_qty",          IntegerType(), True),  # nombre de photos du produit
    StructField("product_weight_g",             FloatType(),  True),  # poids en grammes
    StructField("product_length_cm",            FloatType(),  True),
    StructField("product_height_cm",             FloatType(), True),
    StructField("product_width_cm",              FloatType(), True),
])


# ====================================================================
# FONCTIONS DU SCRIPT
# ====================================================================

def create_spark_session() -> SparkSession:
    """
    Cree (ou recupere si elle existe deja) la SparkSession, c'est-a-dire
    la "connexion" qui nous permet d'utiliser PySpark.

    Sur Databricks, une SparkSession nommee `spark` est deja disponible
    automatiquement des qu'on ouvre un notebook : on n'a normalement pas
    besoin d'appeler cette fonction. Elle sert surtout si on veut tester
    ce script en dehors de Databricks, sur sa propre machine.
    """
    spark = (
        SparkSession.builder
        .appName("ShopBR_Ingestion")  # nom du job, visible dans l'interface de monitoring Spark
        # Active "l'execution adaptative" : Spark ajuste automatiquement
        # le nombre de "morceaux" de travail (partitions) selon le volume
        # reel de donnees, au lieu d'utiliser une valeur fixe qui serait
        # parfois trop grande, parfois trop petite.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Nombre de partitions utilisees lors d'un "shuffle" (operation
        # qui redistribue les donnees entre les machines, par exemple
        # lors d'un groupBy). 50 est raisonnable pour ce volume de donnees
        # (quelques centaines de milliers de lignes au maximum).
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")  # reduit le bruit des logs techniques de Spark
    logger.info(f"Session Spark creee - version : {spark.version}")
    return spark


def ingest_csv(spark: SparkSession, table_name: str, filename: str,
               schema: StructType = None) -> "DataFrame":
    """
    Lit UN fichier CSV source et le transforme en DataFrame Spark
    (l'equivalent d'un grand tableau de donnees, distribue sur le cluster).

    Parametres
    ----------
    spark      : la SparkSession active
    table_name : le nom qu'on donne a cette table (ex: "orders")
    filename   : le nom du fichier CSV a lire
    schema     : le schema explicite a utiliser (cf section au-dessus).
                 Si None, on laisse Spark deviner (uniquement pour la
                 petite table de traduction des categories, sans enjeu
                 de performance).

    Retourne
    --------
    Un DataFrame Spark contenant les donnees du CSV, enrichi de 3
    colonnes de tracabilite (voir plus bas).
    """
    path = f"{RAW_S3_PATH}{filename}"
    logger.info(f"Ingestion : {table_name} depuis {path}")

    # .option("header", "true")    -> la 1ere ligne du CSV contient les noms de colonnes
    # .option("multiLine", "true") -> certains champs texte (comme les
    #   commentaires d'avis) peuvent contenir des sauts de ligne ; cette
    #   option dit a Spark de bien gerer ce cas plutot que de couper la
    #   ligne au mauvais endroit.
    reader = spark.read.option("header", "true").option("multiLine", "true")

    if schema is not None:
        reader = reader.schema(schema)
    else:
        # "inferSchema" = Spark devine lui-meme les types de colonnes.
        # On ne l'utilise que pour la petite table de traduction (71 lignes),
        # ou la perte de performance est negligeable.
        reader = reader.option("inferSchema", "true")

    df = (
        reader.csv(path)
        # On ajoute 3 colonnes qui n'existent pas dans le fichier source,
        # mais qui sont essentielles pour la TRACABILITE : pouvoir dire,
        # pour chaque ligne de donnee, QUAND elle a ete integree dans
        # notre systeme et DE QUEL fichier elle vient. C'est une exigence
        # de bonnes pratiques en gouvernance des donnees (Bloc 1 du projet).
        .withColumn("ingested_at",   F.current_timestamp())  # horodatage de l'ingestion
        .withColumn("source_table",  F.lit(table_name))      # F.lit() = insere une valeur fixe (constante) dans toutes les lignes
        .withColumn("source_file",   F.lit(filename))
    )

    # .count() declenche reellement la lecture du fichier (en PySpark,
    # beaucoup d'operations sont "paresseuses" : elles ne s'executent
    # qu'au moment ou on demande un resultat concret, comme un compte
    # de lignes ou un affichage).
    count = df.count()
    logger.info(f"  -> {count:,} lignes ingerees pour {table_name}")
    return df


def ingest_all_tables(spark: SparkSession) -> dict:
    """
    Boucle sur les 8 tables sources et les ingere une par une, en
    appliquant a chacune le schema explicite qui lui correspond.

    Retourne un dictionnaire {nom_de_la_table: DataFrame correspondant},
    ce qui nous permettra ensuite de manipuler facilement chaque table
    individuellement (tables["orders"], tables["customers"], etc.)
    """
    # On associe chaque nom de table a son schema. La table de traduction
    # des categories n'a pas de schema dedie (None), car elle est petite
    # et sans enjeu de performance.
    schemas = {
        "orders":      ORDERS_SCHEMA,
        "order_items": ORDER_ITEMS_SCHEMA,
        "payments":    PAYMENTS_SCHEMA,
        "reviews":     REVIEWS_SCHEMA,
        "customers":   CUSTOMERS_SCHEMA,
        "sellers":     SELLERS_SCHEMA,
        "products":    PRODUCTS_SCHEMA,
        "category_translation": None,
    }

    dataframes = {}
    for table_name, filename in SOURCE_FILES.items():
        try:
            df = ingest_csv(spark, table_name, filename, schemas.get(table_name))
            dataframes[table_name] = df
        except Exception as e:
            # Si UNE table echoue a l'ingestion, on prefere arreter tout
            # le script plutot que de continuer avec des donnees
            # incompletes : mieux vaut un echec visible qu'une erreur
            # silencieuse decouverte bien plus tard.
            logger.error(f"Erreur ingestion {table_name} : {e}")
            raise

    total = sum(df.count() for df in dataframes.values())
    logger.info(f"Total toutes tables : {total:,} lignes")
    return dataframes


def save_to_bronze(dataframes: dict, base_path: str = BRONZE_PATH) -> None:
    """
    Sauvegarde chaque DataFrame (encore en memoire/dans le cluster Spark)
    sur le stockage permanent, au format Delta Lake.

    QU'EST-CE QUE DELTA LAKE ?
    C'est un format de fichier ameliore par rapport au simple CSV ou
    Parquet : il ajoute des fonctionnalites comme les transactions
    fiables (on ne se retrouve jamais avec un fichier a moitie ecrit
    si le job plante en cours de route), et le "time travel" (on peut
    revenir voir l'etat des donnees a une date passee).

    POURQUOI PARTITIONNER LA TABLE "orders" PAR MOIS ?
    Le partitionnement, c'est decouper physiquement les donnees en
    sous-dossiers (un dossier par mois ici). Cela rend les lectures
    futures beaucoup plus rapides quand on ne s'interesse qu'a certains
    mois : Spark peut directement ignorer les dossiers qui ne nous
    interessent pas, au lieu de scanner tout le fichier.
    """
    for table_name, df in dataframes.items():
        path = f"{base_path}{table_name}/"
        logger.info(f"Sauvegarde Bronze : {table_name} -> {path}")

        if table_name == "orders":
            # On cree une colonne "purchase_month" au format "2018-03"
            # par exemple, qui servira de cle de partitionnement.
            df_part = df.withColumn(
                "purchase_month",
                F.date_format("order_purchase_timestamp", "yyyy-MM")
            )
            (df_part.write.format("delta")
                .mode("overwrite")           # remplace les donnees existantes a chaque execution
                .option("overwriteSchema", "true")  # autorise a changer le schema si besoin
                .partitionBy("purchase_month")       # decoupage physique par mois
                .save(path))
        else:
            # Les autres tables sont plus petites et n'ont pas besoin
            # d'etre partitionnees.
            (df.write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .save(path))

        logger.info(f"  -> {table_name} sauvegarde avec succes")


# ====================================================================
# POINT D'ENTREE DU SCRIPT
# ====================================================================
# Ce bloc ne s'execute que si on lance ce fichier directement
# (par exemple via `python 01_ingestion.py`), pas si on l'importe
# depuis un autre script.
if __name__ == "__main__":
    try:
        # Sur Databricks, la variable `spark` existe deja automatiquement
        # des l'ouverture d'un notebook. On essaie de l'utiliser directement.
        spark
    except NameError:
        # Si on n'est pas sur Databricks (donc `spark` n'existe pas encore),
        # on cree notre propre SparkSession.
        spark = create_spark_session()

    # Etape 1 : on ingere les 8 tables sources
    tables = ingest_all_tables(spark)

    # Etape 2 : on affiche un petit apercu de chaque table, pour verifier
    # visuellement que tout s'est bien passe avant de sauvegarder
    for name, df in tables.items():
        logger.info(f"Apercu de {name} :")
        df.show(3, truncate=True)  # affiche les 3 premieres lignes

    # Etape 3 : on sauvegarde tout en couche Bronze
    save_to_bronze(tables)

    logger.info("=== Ingestion terminee avec succes ===")
