"""
====================================================================
Projet #6 - Big Data Certification | Architecte en Intelligence Artificielle
Script 03 - TRANSFORMATION et calcul des KPIs (Silver -> Gold)
ShopBR - dataset Olist e-commerce Bresil
====================================================================

A QUOI SERT CE SCRIPT ?
C'est la derniere etape de transformation avant l'analyse. On part de
la table consolidee "Silver" (une ligne = une commande, produite par
le script 02_cleaning.py) et on calcule des INDICATEURS AGREGES
(des "KPI", Key Performance Indicators) qui repondent directement aux
questions metier de ShopBR.

QU'EST-CE QU'UN INDICATEUR AGREGE ?
Au lieu de garder le detail de chaque commande individuelle, on
regroupe les commandes selon un critere (par etat, par categorie de
produit, par vendeur...) et on calcule des statistiques sur chaque
groupe : nombre de commandes, taux de retard, note moyenne, etc.

Par exemple, au lieu d'avoir 99 441 lignes (une par commande), la
table "late_rate_by_state" n'aura que 27 lignes (une par etat
bresilien), avec pour chacune le taux de retard mesure.

C'est cette etape qui transforme des donnees brutes en INFORMATION
EXPLOITABLE pour la prise de decision (Bloc 2 de la certification :
"Architecture de donnees pour l'IA").

LA TABLE LA PLUS IMPORTANTE DU PROJET : delay_review_impact
Elle quantifie le lien entre retard et insatisfaction client, ce qui
constitue LA preuve chiffree qui justifie l'ensemble du projet
(pourquoi investir dans un systeme de prediction du retard).
====================================================================
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
# Window permet de faire des calculs "par groupe" sans perdre le detail
# de chaque ligne (utile ici pour calculer un classement, par exemple
# "quel est le rang de ce vendeur parmi tous les vendeurs ?")
from pyspark.sql.window import Window
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# On lit la table consolidee produite par le script 02 (couche Silver)
SILVER_PATH = "dbfs:/mnt/shopbr/silver/orders_consolidated/"
# On ecrit les tables d'indicateurs en couche Gold, pretes pour Redshift
GOLD_PATH   = "dbfs:/mnt/shopbr/gold/"


def load_silver(spark: SparkSession) -> DataFrame:
    """Charge la table consolidee Silver et la garde en cache (memoire),
    car elle va etre reutilisee pour calculer 7 tables d'indicateurs
    differentes juste apres."""
    df = spark.read.format("delta").load(SILVER_PATH)
    df.cache()
    logger.info(f"Silver charge : {df.count():,} commandes")
    return df


def compute_late_rate_by_state(df: DataFrame) -> DataFrame:
    """
    Calcule le taux de retard pour chaque etat client.
    Repond a la question : "dans quels etats du Bresil le risque de
    retard est-il le plus eleve ?" (information cle pour prioriser
    les actions logistiques par zone geographique).
    """
    return (
        # groupBy regroupe toutes les commandes ayant le meme etat client
        df.groupBy("customer_state")
        .agg(
            F.count("*").alias("total_orders"),
            # F.col("is_late").cast("int") transforme TRUE/FALSE en 1/0,
            # ce qui permet de les additionner pour compter le nombre
            # total de commandes en retard dans ce groupe.
            F.sum(F.col("is_late").cast("int")).alias("late_orders"),
            F.avg("delivery_days").alias("avg_delivery_days"),
            F.avg("review_score").alias("avg_review_score"),
        )
        # Calcul du pourcentage : nombre de retards / nombre total * 100,
        # arrondi a 2 decimales.
        .withColumn("late_rate_pct",
            F.round(F.col("late_orders") / F.col("total_orders") * 100, 2))
        # On trie du taux de retard le plus eleve au plus faible, pour
        # voir immediatement les etats les plus a risque en premier.
        .orderBy(F.desc("late_rate_pct"))
    )


def compute_late_rate_by_category(df: DataFrame) -> DataFrame:
    """
    Meme logique que ci-dessus, mais par categorie de produit plutot
    que par etat. Repond a : "certains types de produits sont-ils
    structurellement plus a risque de retard ?" (par exemple les
    meubles, plus volumineux, pourraient avoir un delai de transport
    plus long que les vetements).
    """
    return (
        df.groupBy("product_category_name_english")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum(F.col("is_late").cast("int")).alias("late_orders"),
            F.avg("total_price").alias("avg_price"),
            F.avg("review_score").alias("avg_review_score"),
        )
        # On exclut les categories avec moins de 30 commandes : avec trop
        # peu de donnees, le taux de retard calcule serait peu fiable
        # statistiquement (une seule commande en retard sur 5 donnerait
        # 20% de "taux de retard", ce qui n'est pas representatif).
        .filter(F.col("total_orders") >= 30)
        .withColumn("late_rate_pct",
            F.round(F.col("late_orders") / F.col("total_orders") * 100, 2))
        .orderBy(F.desc("late_rate_pct"))
    )


def compute_seller_performance(df: DataFrame, top_n: int = 50) -> DataFrame:
    """
    Calcule, pour chaque vendeur, son volume de commandes, son taux de
    retard et sa note moyenne, puis ne garde que les top_n vendeurs
    avec le plus gros volume (les 50 plus actifs, par defaut).

    Pourquoi se limiter aux 50 plus gros vendeurs et pas tous les
    3 095 ? Parce que cette table sert a l'equipe "relation vendeurs"
    pour identifier rapidement les vendeurs majeurs qui posent probleme.
    Avec 3 095 lignes, le tableau serait inutilisable au quotidien.
    """
    # Window.orderBy(...) definit un classement : on va numeroter les
    # vendeurs du plus gros volume (rang 1) au plus petit.
    window_sellers = Window.orderBy(F.desc("total_orders"))
    return (
        df.groupBy("main_seller_id", "seller_state")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum(F.col("is_late").cast("int")).alias("late_orders"),
            F.avg("delivery_days").alias("avg_delivery_days"),
            F.avg("review_score").alias("avg_review_score"),
            F.avg("total_price").alias("avg_order_value"),
        )
        .withColumn("late_rate_pct",
            F.round(F.col("late_orders") / F.col("total_orders") * 100, 2))
        # F.rank().over(window_sellers) attribue un numero de classement
        # a chaque ligne, selon l'ordre defini par la Window ci-dessus.
        .withColumn("seller_rank", F.rank().over(window_sellers))
        # On ne garde que les 50 premiers (les plus gros vendeurs)
        .filter(F.col("seller_rank") <= top_n)
        .orderBy("seller_rank")
    )


def compute_cross_state_impact(df: DataFrame) -> DataFrame:
    """
    Compare les commandes "inter-etats" (vendeur et client dans des
    etats differents) aux commandes "meme etat".

    Cette table ne contiendra que 2 lignes (TRUE et FALSE), mais c'est
    l'une des plus importantes du projet : elle valide statistiquement
    que cross_state_delivery est bien un facteur de risque significatif,
    ce qui justifie son utilisation comme feature principale dans le
    futur modele de prediction (Projet #7).
    """
    return (
        df.groupBy("cross_state_delivery")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum(F.col("is_late").cast("int")).alias("late_orders"),
            F.avg("delivery_days").alias("avg_delivery_days"),
            F.avg("review_score").alias("avg_review_score"),
        )
        .withColumn("late_rate_pct",
            F.round(F.col("late_orders") / F.col("total_orders") * 100, 2))
    )


def compute_delay_review_impact(df: DataFrame) -> DataFrame:
    """
    LA TABLE LA PLUS IMPORTANTE DU PROJET.

    Elle quantifie precisement le lien entre "la commande etait en
    retard" et "le client a mis une mauvaise note". C'est cette mesure
    qui justifie, chiffres a l'appui, pourquoi ShopBR a interet a
    investir dans un systeme de prediction du retard (Projet #7) :
    si on peut anticiper et eviter le retard, on evite la mauvaise note.

    Resultat mesure sur les vraies donnees : note moyenne de 2,27/5 si
    en retard, contre 4,29/5 si a l'heure - un ecart enorme.
    """
    return (
        # On exclut les commandes sans avis client : on ne peut pas
        # mesurer l'impact sur la satisfaction si on n'a pas de note.
        df.filter(F.col("review_score").isNotNull())
        .groupBy("is_late")
        .agg(
            F.count("*").alias("total_orders"),
            F.avg("review_score").alias("avg_review_score"),
            # On compte le nombre d'avis "mauvais" (note de 1 ou 2 sur 5)
            # F.when(condition, 1).otherwise(0) transforme une condition
            # en 0/1, qu'on peut ensuite additionner avec F.sum().
            F.sum(F.when(F.col("review_score") <= 2, 1).otherwise(0)).alias("bad_reviews"),
        )
        .withColumn("bad_review_rate_pct",
            F.round(F.col("bad_reviews") / F.col("total_orders") * 100, 2))
    )


def compute_monthly_trend(df: DataFrame) -> DataFrame:
    """
    Calcule l'evolution mois par mois du volume de commandes et du
    taux de retard, sur toute la periode du dataset (2016-2018).
    Permet de voir si la situation s'ameliore, se degrade, ou suit
    une saisonnalite (par exemple plus de retards en periode de soldes
    a fort volume).
    """
    window_lag = Window.orderBy("purchase_year", "purchase_month")
    return (
        df.groupBy("purchase_year", "purchase_month")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum(F.col("is_late").cast("int")).alias("late_orders"),
            F.avg("total_price").alias("avg_order_value"),
            F.avg("review_score").alias("avg_review_score"),
        )
        .withColumn("late_rate_pct",
            F.round(F.col("late_orders") / F.col("total_orders") * 100, 2))
        .orderBy("purchase_year", "purchase_month")
    )


def compute_weekday_pattern(df: DataFrame) -> DataFrame:
    """
    Taux de retard selon le jour de la semaine ou la commande a ete
    passee. Permet de detecter par exemple si les commandes du
    week-end (traitees le lundi suivant par les vendeurs) ont un
    risque de retard plus eleve.
    """
    return (
        df.groupBy("purchase_weekday")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum(F.col("is_late").cast("int")).alias("late_orders"),
            F.avg("delivery_days").alias("avg_delivery_days"),
        )
        .withColumn("late_rate_pct",
            F.round(F.col("late_orders") / F.col("total_orders") * 100, 2))
        .orderBy("purchase_weekday")
    )


def save_gold_table(df: DataFrame, name: str) -> None:
    """Sauvegarde une table d'indicateurs dans la couche Gold."""
    path = f"{GOLD_PATH}{name}/"
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
    logger.info(f"Gold '{name}' sauvegarde -> {path}")


def run_transformation_pipeline(spark: SparkSession) -> None:
    """
    Fonction "chef d'orchestre" : calcule les 7 tables d'indicateurs
    une par une et les sauvegarde toutes en couche Gold.
    """
    df = load_silver(spark)

    # Ce dictionnaire associe le nom de chaque table future a la
    # fonction qui la calcule. C'est une facon compacte d'organiser
    # plusieurs calculs similaires.
    tables = {
        "late_rate_by_state":     compute_late_rate_by_state(df),
        "late_rate_by_category":  compute_late_rate_by_category(df),
        "seller_performance":     compute_seller_performance(df, top_n=50),
        "cross_state_impact":     compute_cross_state_impact(df),
        "delay_review_impact":    compute_delay_review_impact(df),
        "monthly_trend":          compute_monthly_trend(df),
        "weekday_pattern":        compute_weekday_pattern(df),
    }

    for name, df_gold in tables.items():
        df_gold.cache()
        save_gold_table(df_gold, name)
        logger.info(f"Apercu {name} :")
        df_gold.show(5, truncate=False)  # affiche les 5 premieres lignes pour verification visuelle
        df_gold.unpersist()

    df.unpersist()
    logger.info("=== Transformation terminee ===")


# ====================================================================
# POINT D'ENTREE DU SCRIPT
# ====================================================================
if __name__ == "__main__":
    try:
        spark
    except NameError:
        spark = SparkSession.builder.appName("ShopBR_Transformation").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
    run_transformation_pipeline(spark)
