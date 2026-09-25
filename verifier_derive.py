"""
====================================================================
ShopBR - Verification de la derive du taux de retard entre 2017 et 2018
====================================================================

A QUOI SERT CE SCRIPT ?
Il recalcule, devant temoin, les deux chiffres cites dans le dossier :
le taux de retard de 2017 et celui de 2018. Tout part des fichiers
CSV d'origine, sans etape intermediaire : n'importe qui peut relancer
la commande et retrouver les memes valeurs.

C'est la meme mesure que celle du rapport Evidently, mais affichee
en clair plutot que dans une interface.

COMMENT L'UTILISER ?
    docker compose exec api python verifier_derive.py
====================================================================
"""

import os

import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "data")

# On lit uniquement les colonnes utiles : la date d'achat, la date de
# livraison reelle et la date estimee annoncee au client.
orders = pd.read_csv(
    f"{DATA_DIR}/olist_orders_dataset.csv",
    usecols=["order_id", "order_status", "order_purchase_timestamp",
             "order_delivered_customer_date", "order_estimated_delivery_date"],
    parse_dates=["order_purchase_timestamp",
                 "order_delivered_customer_date",
                 "order_estimated_delivery_date"])

# On ne garde que les commandes reellement livrees : une commande
# annulee ou en cours n'a pas de verite connue.
livrees = orders[orders["order_status"] == "delivered"].dropna(
    subset=["order_delivered_customer_date"])

# Une commande est en retard si elle arrive apres la date estimee.
livrees = livrees.assign(
    en_retard=(livrees["order_delivered_customer_date"]
               > livrees["order_estimated_delivery_date"]),
    annee=livrees["order_purchase_timestamp"].dt.year)

print("\nTaux de retard par annee (commandes livrees)")
print("-" * 46)
for annee in sorted(livrees["annee"].unique()):
    lot = livrees[livrees["annee"] == annee]
    taux = lot["en_retard"].mean() * 100
    print(f"  {annee} : {taux:5.2f} %   ({len(lot):>6,} commandes)".replace(",", " "))

ref = livrees[livrees["annee"] == 2017]["en_retard"].mean() * 100
cur = livrees[livrees["annee"] == 2018]["en_retard"].mean() * 100
print("-" * 46)
print(f"  Derive 2017 -> 2018 : +{cur - ref:.2f} point de pourcentage")
print(f"  Soit une hausse de {(cur / ref - 1) * 100:.0f} % du taux de retard.\n")
print("  C'est cette derive qui justifie le reentrainement mensuel :")
print("  un modele appris sur 2017 aurait sous-estime le risque en 2018.\n")
