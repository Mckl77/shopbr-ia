# ====================================================================
# Projet #7 - Mise en production ShopBR
# Dockerfile - Recette de construction de l'image Docker
# ====================================================================
#
# QU'EST-CE QU'UN DOCKERFILE ?
# C'est une recette de cuisine, ligne par ligne, qui explique a Docker
# comment construire une "image" : un paquet autonome contenant tout
# ce qu'il faut pour faire tourner notre application (le code Python,
# les bibliotheques necessaires, le systeme d'exploitation minimal).
#
# POURQUOI DOCKER ?
# Sans Docker, faire tourner cette application sur une nouvelle machine
# demanderait d'installer manuellement Python, les bonnes versions des
# bibliotheques, configurer l'environnement... Avec Docker, on construit
# l'image UNE FOIS et elle fonctionne IDENTIQUEMENT partout (sur
# l'ordinateur d'un developpeur, sur un serveur de test, en production
# sur Kubernetes). C'est ce qu'on appelle la "standardisation de
# l'environnement", demandee explicitement par l'enonce du Bloc 7.
#
# CHAQUE LIGNE COMMENÇANT PAR UN MOT EN MAJUSCULES (FROM, RUN, COPY...)
# EST UNE "INSTRUCTION" DOCKER. Docker les execute dans l'ordre, et
# chaque instruction cree une nouvelle "couche" (layer) dans l'image.
# ====================================================================

# FROM indique l'image de DEPART sur laquelle on construit la notre.
# Plutot que de partir de zero (installer un systeme Linux complet),
# on part d'une image officielle Python deja prete a l'emploi.
# "3.11-slim" = version 3.11 de Python, en version "allegee" (moins de
# fichiers inutiles, donc image finale plus petite et plus rapide a transferer).
FROM python:3.11-slim

# LABEL ajoute des metadonnees informatives a l'image (visible avec la
# commande "docker inspect"). Ce ne sont que des informations
# descriptives, sans impact technique sur le fonctionnement.
LABEL maintainer="ShopBR Data Team"
LABEL description="Image de production - prediction du risque de retard de livraison"
LABEL version="1.0.0"

# ENV definit des variables d'environnement DANS l'image, disponibles
# pour tous les programmes qui s'executeront a l'interieur du conteneur.
ENV PYTHONDONTWRITEBYTECODE=1 \
    # Empeche Python de creer des fichiers .pyc (cache de compilation)
    # inutiles dans un conteneur qui ne vit que le temps de son execution.
    PYTHONUNBUFFERED=1 \
    # Force Python a afficher ses logs immediatement (sans mise en
    # tampon), ce qui est indispensable pour bien voir les logs en
    # temps reel avec "docker logs".
    PIP_NO_CACHE_DIR=1 \
    # Empeche pip de garder un cache des paquets installes, ce qui
    # reduirait inutilement la taille de l'image finale.
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=shopbr_delay_risk \
    MODEL_STAGE=Production

# WORKDIR definit le dossier de travail DANS le conteneur. Toutes les
# commandes suivantes (COPY, RUN, CMD) s'executeront depuis ce dossier,
# un peu comme un "cd /app" qui resterait actif pour la suite du fichier.
WORKDIR /app

# RUN execute une commande PENDANT la construction de l'image (pas au
# moment ou le conteneur tourne, mais au moment ou on construit
# l'image elle-meme, avec "docker build").
# Ici, on installe quelques outils systeme indispensables :
# - build-essential : necessaire pour compiler certaines bibliotheques
#   Python qui contiennent du code C (comme certaines parties de XGBoost)
# - curl : utilise plus bas par le HEALTHCHECK pour tester que l'API repond
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    # On nettoie ensuite les fichiers temporaires d'installation pour
    # garder une image aussi legere que possible.
    && rm -rf /var/lib/apt/lists/*

# COPY copie un fichier depuis notre ordinateur (ou le depot de code)
# VERS l'interieur de l'image en construction.
# On copie d'abord UNIQUEMENT requirements.txt (la liste des
# dependances), avant de copier tout le reste du code. C'est une
# optimisation importante : Docker met en cache chaque etape. Si on ne
# modifie que le code Python (pas les dependances), Docker reutilisera
# le cache de l'installation des paquets au lieu de tout reinstaller,
# ce qui rend les reconstructions futures beaucoup plus rapides.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Maintenant on copie tout le reste du code de l'application.
COPY . .

# ── SECURITE : UTILISATEUR NON-ROOT ─────────────────────────────────
# Par defaut, un conteneur Docker s'execute avec les droits "root"
# (administrateur), ce qui est une mauvaise pratique de securite : si
# quelqu'un parvenait a exploiter une faille dans notre application,
# il aurait les droits maximaux a l'interieur du conteneur.
# On cree donc un utilisateur dedie ("shopbr") avec des droits limites,
# et on bascule dessus pour l'execution de l'application.
RUN useradd -m -u 1000 shopbr && chown -R shopbr:shopbr /app
USER shopbr

# EXPOSE documente (de maniere informative, sans effet technique direct)
# le port sur lequel l'application a l'interieur du conteneur va ecouter.
# Le port reel publie vers l'exterieur sera configure dans
# docker-compose.yml ou dans la configuration Kubernetes.
EXPOSE 8000

# HEALTHCHECK definit comment Docker (et plus tard Kubernetes) peut
# verifier automatiquement que l'application a l'interieur du
# conteneur fonctionne correctement, et pas seulement que le
# conteneur est "demarre" (un conteneur peut etre demarre mais avec
# une application plantee a l'interieur).
# --interval=30s  : verifie toutes les 30 secondes
# --timeout=10s   : abandonne si pas de reponse en 10 secondes
# --start-period=5s : laisse 5 secondes de battement au demarrage avant de commencer a tester
# --retries=3     : si 3 verifications echouent d'affilee, le conteneur est marque "unhealthy"
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# CMD definit la commande qui s'execute quand le conteneur DEMARRE
# (contrairement a RUN qui s'execute lors de la CONSTRUCTION de l'image).
# Ici, on lance simplement notre API Flask.
CMD ["python", "src/api/api.py"]
