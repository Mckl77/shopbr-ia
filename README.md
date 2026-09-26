# ShopBR — Solution IA (dépôt n°1 : développement)

Prédiction du risque de retard de livraison sur une marketplace
e-commerce brésilienne fictive (dataset réel Olist, 99 441 commandes).

> Projet de certification **Architecte en Intelligence Artificielle**
> (Mastère 2, Fonderie de l'Image). Ce dépôt contient le **code de la
> solution**. Le pipeline de déploiement vit dans le dépôt
> **shopbr-deploy** : deux dépôts distincts, pour que seul le second
> détienne les accès à la production.

---

## Le problème métier en trois chiffres

| Mesure (données réelles Olist) | Valeur |
|---|---|
| Note client moyenne si commande en retard | **2,27 / 5** (contre 4,29 à l'heure) |
| Taux de retard | **6,65 %** des commandes (retard d'au moins un jour) |
| Vélocité | 160 commandes/jour, pics ×7 (Black Friday : 1 176/jour) |

**Objectif** : attribuer à chaque commande, avant expédition, une
**probabilité de retard en pourcentage**, jamais une décision binaire.
Seuils calibrés sur le taux de base : au-delà de **20 %** action
immédiate, entre **10 et 20 %** surveillance, en dessous rien.

---

## Arborescence

```
shopbr-ia/
├── src/
│   ├── pipeline/        PySpark serverless (AWS Glue) : ingestion → Bronze/Silver/Gold
│   ├── training/        mlflow_train.py (XGBoost + registre MLflow), ray_train.py
│   ├── api/             api.py (Flask) + risk_logic.py (logique métier testée)
│   ├── dashboard/       app.py (Dash, 3 onglets)
│   └── monitoring/      evidently_report.py (dérive des données)
├── dags/                airflow_dag.py (réentraînement mensuel), airbyte_config.json
├── sql/                 redshift_setup.sql, redshift_optimize.sql
├── tests/               tests unitaires de la logique métier (exécutés par la CI)
├── docs/                politique de sécurité, diagramme d'architecture
├── train_local.py       entraînement local depuis les CSV (voir plus bas)
├── Dockerfile           image unique API + dashboard
├── docker-compose.yml   stack locale (API, dashboard, MLflow, Airflow, PostgreSQL)
└── .github/workflows/ci.yml   CI : lint → tests → déclenche le déploiement
```

## Démarrage rapide (local)

```bash
cp .env.example .env      # renseigner les valeurs (ce fichier n'est jamais versionné)
docker compose up -d
```

| Service | URL |
|---|---|
| API de prédiction | http://localhost:8000 (`/health`, `/predict`, `/predict/batch`) |
| Dashboard logistique | http://localhost:8050 |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

## Entraîner le modèle

Placer les 9 fichiers CSV Olist dans un dossier `data/` (non versionné),
puis :

```bash
docker compose exec api pip install -q xgboost scikit-learn
docker compose exec api python train_local.py
docker compose cp api:/app/models ./models
docker compose restart api
```

Le modèle est enregistré dans `models/` et chargé au démarrage de l'API.
En production, il vient du registre MLflow (`src/training/mlflow_train.py`).

## Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"order_id":"demo1","nb_items":2,"total_price":250,"total_freight":35,"cross_state_delivery":true}'
```

```json
{"order_id":"demo1","risk_pct":7.8,"risk_proba":0.078,
 "risk":{"level":"low","label":"Risque faible","action":"Aucune action requise"}}
```

## Tests

```bash
pip install pandas pytest
python -m pytest tests/ -v
```

15 tests, quelques secondes, sans modèle ni accès réseau. La logique
métier (seuils, variables du modèle) est isolée dans
`src/api/risk_logic.py` précisément pour rester testable sans
dépendances lourdes. C'est ce que la CI exécute à chaque envoi de code.

## Chaîne CI/CD

```
push sur main (ce dépôt)
  └─ CI : lint → 15 tests unitaires           [.github/workflows/ci.yml]
       └─ si vert : signal automatique ─────→ dépôt shopbr-deploy
                                                └─ build de l'image, publication ECR, déploiement Kubernetes
```

La construction de l'image est faite par le dépôt de déploiement, pas
par la CI : celle-ci valide le code, le second déploie.

Secret à configurer dans ce dépôt : `DEPLOY_REPO_TOKEN`, un jeton
d'accès personnel limité au dépôt `shopbr-deploy`.

## Résultats et limites

Le modèle atteint une AUC de 0,58 : il distingue mal les commandes à
risque. La cause est identifiée : le jeu de données Olist ne contient
ni transporteur, ni distance réelle, ni météo, qui sont les vrais
moteurs du retard. Les probabilités sont en revanche bien calibrées
(8,0 % prédit en moyenne pour 8,1 % observé).

Pistes d'amélioration : calculer la distance vendeur-client à partir du
fichier de géolocalisation, puis ajouter le transporteur et les jours
fériés brésiliens.

## Compétences de certification couvertes

| Code | Où |
|---|---|
| C2.4 | `sql/` (DISTKEY/SORTKEY), couches Bronze/Silver/Gold |
| C2.7 | `src/monitoring/`, contrôles de santé, journalisation |
| C4.3 | `src/api/api.py` (API REST intégrée à l'infrastructure) |
| C4.5 | `dags/airflow_dag.py` + `src/training/mlflow_train.py` |
