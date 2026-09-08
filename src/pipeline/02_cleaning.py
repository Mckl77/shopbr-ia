"""
====================================================================
Projet #6 - Big Data Certification | Architecte en Intelligence Artificielle
Script 02 - NETTOYAGE et CONSOLIDATION (Bronze -> Silver)
ShopBR - dataset Olist e-commerce Bresil
====================================================================

A QUOI SERT CE SCRIPT ?
C'est ici qu'on transforme les donnees brutes (8 tables separees,
copiees telles quelles au script 01) en UNE SEULE table propre et
exploitable, ou chaque ligne represente une commande complete avec
toutes les informations necessaires pour predire si elle sera livree
en retard.

Cette etape s'appelle la couche "Silver" dans une architecture Big
Data en 3 niveaux (Bronze = brut, Silver = nettoye, Gold = pret pour
l'analyse).

OBJECTIF CONCRET :
Construire une table ou chaque ligne = une commande, avec :
  - une colonne "is_late" qui dit si la commande etait en retard (TRUE/FALSE)
  - toutes les informations utiles pour PREDIRE ce retard a l'avance
    (poids du colis, distance vendeur-client, nombre de produits...)

POURQUOI NETTOYER LES DONNEES ?
Les donnees du monde reel contiennent presque toujours des erreurs :
valeurs manquantes, doublons, valeurs aberrantes (un prix de 999999R$
qui est probablement une erreur de saisie). Si on entraine un modele
de prediction sur des donnees sales, le modele apprend de mauvais
patterns et devient peu fiable. Le nettoyage est donc une etape
absolument critique, pas une simple formalite.
====================================================================
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Chemins de lecture (Bronze, donnees brutes du script 01)
# et d'ecriture (Silver, donnees nettoyees produites par ce script)
BRONZE_PATH = "dbfs:/mnt/shopbr/bronze/"
SILVER_PATH = "dbfs:/mnt/shopbr/silver/"

# ── SEUILS METIER DE NETTOYAGE ───────────────────────────────────────
# Ces valeurs ne sont pas choisies au hasard : elles viennent d'une
# exploration prealable du dataset reel, pour identifier ce qui est
# une vraie valeur "extreme mais possible" et ce qui est clairement
# une erreur de saisie.
MAX_DELIVERY_DAYS = 200     # Au-dela de 200 jours de livraison : tres probablement une erreur de saisie
MAX_PRICE         = 7000.0  # Le 3eme quartile reel des prix est ~135 R$, donc 7000 R$ est un seuil large
                             # qui filtre uniquement les valeurs vraiment aberrantes, pas les produits chers legitimes
VALID_ORDER_STATUS = ["delivered"]  # On entraine le modele UNIQUEMENT sur les commandes livrees :
                                     # on ne peut pas savoir si une commande "en cours" sera en retard ou non


def load_bronze_table(spark: SparkSession, table_name: str) -> DataFrame:
    """
    Petite fonction utilitaire qui lit une table depuis la couche Bronze.
    On l'utilise pour chacune des 8 tables sources.
    """
    path = f"{BRONZE_PATH}{table_name}/"
    df = spark.read.format("delta").load(path)
    logger.info(f"Bronze charge : {table_name} ({df.count():,} lignes)")
    return df


def clean_orders(orders: DataFrame) -> DataFrame:
    """
    Nettoie la table des commandes et calcule les colonnes CIBLES
    (celles qu'on cherche a predire ou qui servent a verifier la qualite) :

    - is_late       : VRAI si la commande a ete livree en retard
                       (c'est la colonne que le modele de Machine Learning
                       du Projet #7 va apprendre a predire)
    - delivery_days  : nombre de jours entre l'achat et la livraison reelle
    - delay_days     : ecart entre date livree et date promise.
                        Positif = retard, negatif ou zero = livre a temps ou en avance
    """
    # ETAPE 1 : on ne garde que les commandes effectivement "delivered".
    # Si on gardait les commandes annulees ou en cours, on n'aurait pas
    # de date de livraison reelle pour savoir si elles etaient en retard.
    df = orders.filter(F.col("order_status").isin(VALID_ORDER_STATUS))

    # ETAPE 2 : on retire les lignes ou il manque une des 3 dates
    # indispensables a nos calculs. Sans ces dates, impossible de
    # savoir si la commande etait en retard.
    df = df.dropna(subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])

    # ETAPE 3 : calcul des colonnes cibles.
    # F.datediff(date_fin, date_debut) calcule le nombre de jours entre
    # deux dates. C'est une fonction PySpark, pas du Python classique :
    # elle s'execute de maniere distribuee sur toutes les lignes en
    # parallele, pas ligne par ligne comme le ferait une boucle Python.
    df = (
        df
        .withColumn(
            "delivery_days",
            # Nombre de jours reellement pris pour livrer
            F.datediff("order_delivered_customer_date", "order_purchase_timestamp")
        )
        .withColumn(
            "delay_days",
            # Ecart entre la date reelle de livraison et la date promise.
            # Si positif : on a livre APRES la date promise -> retard.
            # Si negatif ou zero : on a livre AVANT ou PILE a la date promise.
            F.datediff("order_delivered_customer_date", "order_estimated_delivery_date")
        )
        .withColumn(
            "is_late",
            # is_late est VRAI uniquement si delay_days est strictement positif.
            # C'est exactement cette colonne que le modele XGBoost du
            # Projet #7 va apprendre a predire AVANT que la livraison
            # n'ait lieu, a partir des autres informations de la commande.
            F.col("delay_days") > 0
        )
        # On extrait aussi des informations temporelles utiles pour
        # l'analyse et la prediction : annee, mois, jour de la semaine
        # et heure d'achat peuvent influencer le risque de retard
        # (exemple : commander un vendredi soir avant un week-end ferie
        # peut rallonger le delai de traitement).
        .withColumn("purchase_year",    F.year("order_purchase_timestamp"))
        .withColumn("purchase_month",   F.month("order_purchase_timestamp"))
        .withColumn("purchase_weekday", F.dayofweek("order_purchase_timestamp"))  # 1=dimanche, 7=samedi
        .withColumn("purchase_hour",    F.hour("order_purchase_timestamp"))
    )

    # ETAPE 4 : filtrage des delais de livraison aberrants.
    # Un delai negatif (livre AVANT meme d'avoir ete achete : impossible)
    # ou un delai superieur a 200 jours sont des erreurs de saisie connues
    # dans ce type de dataset reel. On les retire pour ne pas fausser
    # l'entrainement du futur modele.
    df = df.filter(
        (F.col("delivery_days") >= 0) &
        (F.col("delivery_days") <= MAX_DELIVERY_DAYS)
    )

    logger.info(f"Orders nettoyees : {df.count():,} lignes (commandes livrees uniquement)")
    return df


def clean_order_items(items: DataFrame) -> DataFrame:
    """
    Nettoie la table des items (produits achetes) et l'agrege au niveau
    "commande" : une commande peut contenir plusieurs items (par exemple
    3 produits achetes en une fois), mais pour notre prediction de
    retard, on veut UNE seule ligne par commande, pas une ligne par produit.

    C'est pourquoi on utilise groupBy("order_id") : on regroupe toutes
    les lignes ayant le meme order_id et on calcule des agregats
    (somme du prix total, nombre d'items, etc.)
    """
    # On retire les lignes incompletes (donnees indispensables manquantes)
    df = items.dropna(subset=["price", "freight_value", "seller_id", "product_id"])

    # Filtrage des prix aberrants : un prix negatif ou nul n'a pas de
    # sens commercial, et un prix superieur a MAX_PRICE est probablement
    # une erreur de saisie (cf seuils definis en haut du fichier)
    df = df.filter(
        (F.col("price") > 0) & (F.col("price") <= MAX_PRICE) &
        (F.col("freight_value") >= 0)
    )

    # groupBy("order_id") + agg(...) = pour chaque commande, on calcule :
    df_agg = (
        df.groupBy("order_id")
        .agg(
            F.count("*").alias("nb_items"),                        # combien de produits dans cette commande
            F.sum("price").alias("total_price"),                   # prix total de la commande
            F.sum("freight_value").alias("total_freight"),         # frais de port totaux
            F.countDistinct("seller_id").alias("nb_sellers"),      # combien de vendeurs differents (une commande peut regrouper plusieurs vendeurs)
            F.first("seller_id").alias("main_seller_id"),          # on garde le 1er vendeur comme "vendeur principal" pour simplifier les jointures
            F.first("product_id").alias("main_product_id"),        # idem pour le produit principal
        )
    )
    logger.info(f"Order items agreges : {df_agg.count():,} commandes")
    return df_agg


def clean_payments(payments: DataFrame) -> DataFrame:
    """
    Agrege les paiements au niveau commande (une commande peut etre
    payee en plusieurs fois, surtout au Bresil ou le paiement en
    plusieurs mensualites est tres courant).
    """
    df = payments.dropna(subset=["order_id", "payment_type", "payment_value"])
    df_agg = (
        df.groupBy("order_id")
        .agg(
            F.sum("payment_value").alias("total_payment_value"),
            F.max("payment_installments").alias("max_installments"),  # nombre maximum de mensualites utilisees
            F.first("payment_type").alias("payment_type"),            # type de paiement principal (carte, boleto...)
        )
    )
    return df_agg


def clean_reviews(reviews: DataFrame) -> DataFrame:
    """
    Nettoie les avis clients. On garde uniquement la note (review_score),
    pas le texte du commentaire, car notre objectif ici n'est pas
    d'analyser le texte mais simplement de VERIFIER, a posteriori,
    le lien entre retard et insatisfaction (ce qui justifie tout le projet).
    """
    df = reviews.dropna(subset=["order_id", "review_score"])
    # On verifie que la note est bien dans la plage attendue (1 a 5)
    df = df.filter(F.col("review_score").between(1, 5))
    # dropDuplicates : si jamais une commande a plusieurs avis (rare mais
    # possible dans les donnees reelles), on n'en garde qu'un seul, pour
    # eviter de dupliquer des lignes lors de la jointure finale.
    df_dedup = df.dropDuplicates(["order_id"])
    return df_dedup.select("order_id", "review_score")


def build_features(orders_clean: DataFrame, items_agg: DataFrame,
                   payments_agg: DataFrame, reviews_clean: DataFrame,
                   customers: DataFrame, sellers: DataFrame,
                   products: DataFrame, category_translation: DataFrame) -> DataFrame:
    """
    C'est LA fonction centrale du script : elle assemble (on dit "joint"
    en langage base de donnees) les 8 tables sources en une seule table
    finale, ou chaque ligne represente une commande complete.

    QU'EST-CE QU'UNE JOINTURE (JOIN) ?
    Une jointure permet de combiner deux tables qui partagent une colonne
    commune (par exemple "order_id" present a la fois dans orders et
    dans items_agg). Le resultat est une seule table contenant les
    colonnes des deux tables d'origine, alignees ligne par ligne selon
    cette colonne commune.

    "how='inner'" = on ne garde QUE les lignes presentes dans les DEUX
    tables. Si une commande n'a pas d'item associe (improbable mais
    possible en cas de donnees incompletes), elle disparait.

    "how='left'" = on garde TOUTES les lignes de la table de gauche,
    meme si elles n'ont pas de correspondance dans la table de droite
    (dans ce cas, les colonnes de la table de droite seront vides/null).
    On utilise "left" pour les reviews et les paiements, car une
    commande peut tres bien ne pas avoir encore recu d'avis client,
    sans que ca remette en cause son utilisation pour predire le retard.
    """
    df = (
        orders_clean
        .join(items_agg,    on="order_id", how="inner")  # chaque commande DOIT avoir au moins un item
        .join(payments_agg, on="order_id", how="left")   # un avis ou paiement manquant ne doit pas faire disparaitre la commande
        .join(reviews_clean, on="order_id", how="left")
        .join(customers,    on="customer_id", how="left")
        # Pour joindre la table des vendeurs, on doit d'abord renommer sa
        # colonne "seller_id" en "main_seller_id" pour qu'elle corresponde
        # au nom de colonne cree dans clean_order_items() ci-dessus.
        .join(sellers.withColumnRenamed("seller_id", "main_seller_id"),
              on="main_seller_id", how="left")
        .join(products.withColumnRenamed("product_id", "main_product_id"),
              on="main_product_id", how="left")
        .join(category_translation, on="product_category_name", how="left")
    )

    # ── FEATURE CLE DU PROJET ─────────────────────────────────────────
    # cross_state_delivery = est-ce que le vendeur et le client sont
    # dans des ETATS DIFFERENTS du Bresil (par exemple vendeur a Sao
    # Paulo, client a Bahia) ?
    #
    # C'est la feature la plus importante du modele, car l'exploration
    # des donnees reelles a montre que les livraisons inter-etats
    # representent 63,8% des commandes ET sont nettement plus a risque
    # de retard (plus de distance a parcourir, plus d'intermediaires
    # logistiques).
    df = df.withColumn(
        "cross_state_delivery",
        F.when(F.col("customer_state") != F.col("seller_state"), True).otherwise(False)
        # F.when(condition, valeur_si_vrai).otherwise(valeur_si_faux)
        # C'est l'equivalent PySpark d'un "if / else", mais qui s'applique
        # automatiquement a toutes les lignes du tableau en parallele.
    )

    # On convertit le poids du produit de grammes en kilogrammes, plus
    # lisible et plus proche de l'unite utilisee par les transporteurs.
    df = df.withColumn(
        "estimated_weight_kg",
        F.round(F.col("product_weight_g") / 1000.0, 2)  # round(valeur, 2) = arrondi a 2 decimales
    )

    # ── SELECTION FINALE DES COLONNES ─────────────────────────────────
    # Apres toutes ces jointures, le DataFrame contient beaucoup de
    # colonnes (parfois redondantes ou inutiles pour la suite). On
    # selectionne explicitement uniquement celles dont on a besoin pour
    # l'analyse (Gold) et pour entrainer le futur modele (Projet #7).
    # Cela allege aussi le volume de donnees stocke.
    df_final = df.select(
        "order_id",
        "customer_id",
        "main_seller_id",
        # Colonnes cibles (ce qu'on cherche a expliquer / predire)
        "is_late",
        "delay_days",
        "delivery_days",
        "review_score",
        # Features temporelles
        "order_purchase_timestamp",
        "purchase_year", "purchase_month", "purchase_weekday", "purchase_hour",
        # Features geographiques
        "customer_state", "customer_city",
        "seller_state",   "seller_city",
        "cross_state_delivery",
        # Features liees a la commande elle-meme
        "nb_items", "nb_sellers",
        "total_price", "total_freight", "total_payment_value",
        "payment_type", "max_installments",
        # Features produit
        "product_category_name_english",
        "estimated_weight_kg",
        "product_photos_qty",
    )

    return df_final


def save_to_silver(df: DataFrame, path: str = SILVER_PATH) -> None:
    """
    Sauvegarde la table finale consolidee dans la couche Silver.

    .repartition(20, "purchase_month") : redistribue physiquement les
    donnees en 20 "morceaux" (partitions), regroupes par mois d'achat.
    Cela equilibre la charge de travail entre les machines du cluster
    et accelere les lectures futures filtrees par periode.
    """
    (
        df.repartition(20, "purchase_month")
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("purchase_year", "purchase_month")  # meme logique de partitionnement physique qu'en Bronze
        .save(f"{path}orders_consolidated/")
    )
    logger.info(f"Silver sauvegarde : {path}orders_consolidated/")


def quality_report(df: DataFrame) -> None:
    """
    Affiche un petit rapport de qualite apres nettoyage, pour pouvoir
    verifier rapidement (a l'oeil, dans les logs) que les chiffres
    obtenus sont coherents avec ce qu'on attend.

    C'est une bonne pratique de toujours produire ce genre de rapport :
    cela permet de detecter immediatement un probleme (par exemple un
    taux de retard de 60% indiquerait clairement un bug quelque part,
    puisque le taux reel mesure sur ce dataset est d'environ 6,65%).
    """
    total = df.count()
    late_rate = df.filter(F.col("is_late")).count() / total
    null_review = df.filter(F.col("review_score").isNull()).count() / total

    logger.info("=" * 60)
    logger.info("  RAPPORT DE QUALITE")
    logger.info("=" * 60)
    logger.info(f"  Commandes consolidees : {total:,}")
    logger.info(f"  Taux de retard        : {late_rate:.2%}")
    logger.info(f"  Reviews manquantes    : {null_review:.2%}")
    logger.info("=" * 60)


def run_cleaning_pipeline(spark: SparkSession) -> DataFrame:
    """
    Fonction "chef d'orchestre" qui enchaine toutes les etapes de
    nettoyage dans le bon ordre : chargement des 8 tables, nettoyage
    individuel de chacune, puis jointure finale.
    """
    # Chargement des 8 tables depuis la couche Bronze
    orders    = load_bronze_table(spark, "orders")
    items     = load_bronze_table(spark, "order_items")
    payments  = load_bronze_table(spark, "payments")
    reviews   = load_bronze_table(spark, "reviews")
    customers = load_bronze_table(spark, "customers")
    sellers   = load_bronze_table(spark, "sellers")
    products  = load_bronze_table(spark, "products")
    cat_trans = load_bronze_table(spark, "category_translation")

    # Nettoyage individuel de chaque table
    orders_clean = clean_orders(orders)
    items_agg    = clean_order_items(items)
    payments_agg = clean_payments(payments)
    reviews_clean = clean_reviews(reviews)

    # Jointure finale : assemblage de tout en une seule table
    df_silver = build_features(
        orders_clean, items_agg, payments_agg, reviews_clean,
        customers, sellers, products, cat_trans
    )

    # .cache() garde le resultat en memoire sur le cluster, car on va
    # l'utiliser deux fois juste apres (une fois pour le rapport qualite,
    # une fois pour la sauvegarde). Sans cache, Spark recalculerait tout
    # depuis le debut a chaque utilisation, ce qui serait deux fois plus lent.
    df_silver.cache()

    quality_report(df_silver)
    save_to_silver(df_silver)

    # .unpersist() libere la memoire utilisee par le cache, maintenant
    # qu'on n'en a plus besoin. Bonne pratique pour ne pas saturer le
    # cluster inutilement.
    df_silver.unpersist()
    return df_silver


# ====================================================================
# POINT D'ENTREE DU SCRIPT
# ====================================================================
if __name__ == "__main__":
    try:
        spark
    except NameError:
        spark = SparkSession.builder.appName("ShopBR_Cleaning").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

    run_cleaning_pipeline(spark)
    logger.info("=== Nettoyage termine ===")
