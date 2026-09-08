"""
====================================================================
Projet certification - ShopBR - Tests unitaires de la logique metier
====================================================================

QU'EST-CE QU'UN TEST UNITAIRE ?
C'est un petit programme qui verifie AUTOMATIQUEMENT qu'une fonction
de notre code fait bien ce qu'elle est censee faire. Par exemple :
"si je donne une probabilite de 0.25 (25 %) a risk_to_action(), elle
DOIT repondre 'risque eleve'". Si un jour quelqu'un modifie le code et
casse cette regle, le test echoue immediatement et on le sait AVANT
que le bug n'arrive en production.

QU'EST-CE QUE PYTEST ?
pytest est l'outil standard en Python pour executer les tests. Il
cherche tous les fichiers qui commencent par "test_", execute chaque
fonction qui commence par "test_", et affiche un rapport : combien de
tests passent (verts) ou echouent (rouges).
Pour lancer les tests a la main :  python -m pytest tests/ -v

QU'EST-CE QU'UN "assert" ?
C'est le coeur d'un test : "assert condition" signifie "verifie que
cette condition est vraie ; si elle est fausse, le test echoue".
Exemple : assert 2 + 2 == 4  -> passe.  assert 2 + 2 == 5  -> echoue.

POURQUOI CES TESTS NE TESTENT-ILS PAS LE MODELE XGBOOST LUI-MEME ?
Parce qu'un test unitaire doit etre RAPIDE et FIABLE : pas de
telechargement de modele, pas de serveur MLflow, pas de reseau. On
teste ici la LOGIQUE METIER (les seuils, la preparation des donnees),
qui est isolee dans src/api/risk_logic.py precisement pour cela.
Ces tests tournent en moins d'une seconde avec seulement pandas
installe - c'est ce que la CI GitHub Actions execute a chaque push
(.github/workflows/ci.yml) : aucun code casse ne peut etre deploye.
====================================================================
"""

import sys
from pathlib import Path

# ── Rendre risk_logic.py importable depuis ce fichier ───────────────
# Les tests sont dans tests/ et le module a tester dans src/api/.
# Python ne sait pas tout seul ou chercher : on ajoute donc le dossier
# src/api au "chemin de recherche" (sys.path) de Python.
# Path(__file__) = ce fichier ; .parents[1] = la racine du depot.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))

import pandas as pd  # noqa: E402  (import apres sys.path : voulu)
import pytest  # noqa: E402

from risk_logic import (  # noqa: E402
    FEATURE_COLUMNS,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    build_features,
    risk_to_action,
)


# ====================================================================
# PARTIE 1 - TESTS DES SEUILS DE RISQUE
# On verifie que les seuils du code correspondent EXACTEMENT a ce qui
# est ecrit dans le document de cadrage valide par l'ecole. Si
# quelqu'un change un seuil dans le code sans mettre a jour le
# cadrage (ou l'inverse), ce test le detecte.
# ====================================================================

def test_seuils_calibres_sur_taux_de_base():
    """Cadrage valide : > 20 % = action prioritaire, 10-20 % = surveillance.

    pytest.approx() compare des nombres decimaux "a peu pres" : en
    informatique, 0.20 peut etre stocke en memoire comme 0.20000000001,
    et une comparaison stricte (==) echouerait a tort.
    """
    assert RISK_HIGH_THRESHOLD == pytest.approx(0.20)
    assert RISK_MEDIUM_THRESHOLD == pytest.approx(0.10)
    # Coherence logique : le seuil "surveillance" est forcement plus
    # bas que le seuil "action immediate".
    assert RISK_MEDIUM_THRESHOLD < RISK_HIGH_THRESHOLD


# @pytest.mark.parametrize = executer LE MEME test plusieurs fois avec
# des valeurs differentes. Ici, 7 couples (probabilite, niveau attendu)
# -> pytest genere 7 tests independants. C'est plus lisible que 7
# fonctions quasi identiques copiees-collees.
@pytest.mark.parametrize("proba,level", [
    (0.00,   "low"),     # 0 %     -> risque faible
    (0.0999, "low"),     # 9,99 %  -> encore faible (juste sous le seuil)
    (0.10,   "medium"),  # 10 %    -> pile sur le seuil : surveillance
    (0.1999, "medium"),  # 19,99 % -> toujours surveillance
    (0.20,   "high"),    # 20 %    -> pile sur le seuil : action immediate
    (0.85,   "high"),    # 85 %    -> action immediate
    (1.0,    "high"),    # 100 %   -> action immediate
])
def test_risk_to_action_niveaux(proba, level):
    """La conversion probabilite -> niveau respecte les seuils du cadrage."""
    assert risk_to_action(proba)["level"] == level


def test_risk_to_action_structure():
    """Chaque reponse contient TOUJOURS les 4 memes cles.

    Pourquoi c'est important : le dashboard et l'API comptent sur ces
    cles (level, label, color, action). Si l'une disparait, l'interface
    plante. Et "action" ne doit jamais etre vide : c'est l'exigence
    d'EXPLICABILITE du cadrage - chaque score s'accompagne d'une
    consigne claire pour l'equipe logistique.
    """
    for proba in (0.05, 0.15, 0.5):  # un exemple par niveau
        info = risk_to_action(proba)
        assert set(info) == {"level", "label", "color", "action"}
        assert info["action"]  # chaine non vide = consigne presente


def test_frontieres_exactes_inclusives():
    """Cas limite : que se passe-t-il PILE sur un seuil ?

    Regle choisie (et documentee ici par le test) : la borne appartient
    au niveau SUPERIEUR, car le code utilise >= (superieur ou egal).
    Exactement 20 % -> "high". Exactement 10 % -> "medium".
    Tester les cas limites est un reflexe classique : c'est la que se
    cachent les bugs.
    """
    assert risk_to_action(RISK_MEDIUM_THRESHOLD)["level"] == "medium"
    assert risk_to_action(RISK_HIGH_THRESHOLD)["level"] == "high"


# ====================================================================
# PARTIE 2 - TESTS DE LA PREPARATION DES DONNEES (build_features)
# build_features() transforme le JSON recu par l'API en tableau
# (DataFrame pandas) au format EXACT attendu par le modele XGBoost.
# Regle d'or en Machine Learning : les donnees envoyees au modele en
# production doivent avoir exactement la meme structure (memes
# colonnes, meme ordre) que celles utilisees a l'entrainement.
# ====================================================================

def test_build_features_colonnes_et_ordre():
    """Les colonnes produites = celles de l'entrainement, dans l'ordre.

    XGBoost identifie les variables par leur POSITION : si les colonnes
    sont dans le desordre, le modele prend le prix pour le poids et
    fait des predictions absurdes... sans afficher la moindre erreur.
    Ce test est notre garde-fou contre ce bug silencieux.
    """
    df = build_features({})            # {} = commande vide (JSON minimal)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1                # 1 commande en entree -> 1 ligne


def test_build_features_valeurs_par_defaut():
    """Une commande incomplete ne fait pas planter l'API (robustesse).

    Si un champ manque dans le JSON, build_features() applique une
    valeur par defaut raisonnable au lieu de lever une erreur :
    l'API repond toujours, meme a un appel minimal.
    """
    row = build_features({}).iloc[0]   # .iloc[0] = premiere ligne du tableau
    assert row["nb_items"] == 1        # defaut : 1 article
    assert row["total_price"] == 0.0   # defaut : prix 0
    assert row["cross_state_delivery"] == 0  # defaut : vendeur et client dans le meme etat


def test_build_features_booleen_converti_en_entier():
    """XGBoost ne comprend que des NOMBRES : True/False -> 1/0.

    Le JSON envoie cross_state_delivery comme booleen (true/false) ;
    la fonction doit le convertir en entier avant le modele.
    """
    assert build_features({"cross_state_delivery": True}).iloc[0]["cross_state_delivery"] == 1
    assert build_features({"cross_state_delivery": False}).iloc[0]["cross_state_delivery"] == 0


def test_build_features_transmet_les_valeurs():
    """Les valeurs fournies arrivent intactes au modele (pas d'ecrasement).

    On envoie 5 valeurs precises et on verifie qu'on retrouve
    exactement les memes dans le tableau final : la fonction ne
    remplace les champs par des defauts QUE s'ils sont absents.
    """
    order = {"nb_items": 3, "total_price": 250.0, "purchase_hour": 22,
             "estimated_weight_kg": 4.5, "max_installments": 6}
    row = build_features(order).iloc[0]
    for k, v in order.items():
        assert row[k] == v


def test_build_features_ignore_champs_inconnus():
    """Un champ superflu dans le JSON n'atteint jamais le modele.

    Securite et stabilite : si un client de l'API envoie un champ
    inattendu (faute de frappe, tentative d'injection), il est
    simplement ignore - seules les 9 colonnes officielles passent.
    """
    df = build_features({"champ_inconnu": 42, "nb_items": 2})
    assert "champ_inconnu" not in df.columns
