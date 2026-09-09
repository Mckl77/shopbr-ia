# ShopBR — Solution IA (dépôt n°1 : développement)

Prédiction du risque de retard de livraison sur une marketplace
e-commerce brésilienne fictive (dataset réel Olist, 99 441 commandes).

> Projet de certification **Architecte en Intelligence Artificielle**
> (Mastère 2 — Fonderie de l'Image). Ce dépôt contient le **code de la
> solution** (Bloc 4, exigence "deux dépôts GitHub distincts").
> Le pipeline de déploiement CI/CD vit dans le dépôt **shopbr-deploy**.

---

## Le problème métier en trois chiffres

| Mesure (données réelles Olist) | Valeur |
|---|---|
| Note client moyenne si commande en retard | **2,27 / 5** (vs 4,29 à l'heure) |
| Taux de retard global | 6,65 % (2017) → **9,37 % (2018)** |
| Vélocité | 160 commandes/jour, pics ×7 (Black Friday : 1 176/jour) |

**Objectif (cadrage validé)** : attribuer à chaque commande, avant
expédition, une **probabilité de retard en pourcentage** — jamais une
décision binaire. Seuils calibrés sur le taux de base : **> 20 %**
action immédiate, **10-20 %** surveillance, **< 10 %** rien.

---

## Arborescence

```
shopbr-ia/
├── src/
│   ├── pipeline/        PySpark serverless (AWS Glue) : ingestion → Bronze/Silver/Gold
│   ├── training/        mlflow_train.py (XGBoost + registry), ray_train.py (distribué)
│   ├── api/             api.py (Flask) + risk_logic.py (logique métier testée)
│   ├── dashboard/       app.py (Dash, 3 onglets)
│   └── monitoring/      evidently_report.py (dérive des données)
├── dags/                airflow_dag.py (réentraînement mensuel), airbyte_config.json
├── sql/                 redshift_setup.sql, redshift_optimize.sql
├── tests/               tests unitaires de la logique métier (CI)
├── docs/                politique de sécurité, diagramme d'architecture
├── Dockerfile           image unique API + dashboard
├── docker-compose.yml   stack locale complète (API, dashboard, MLflow, Airflow, PostgreSQL)
└── .github/workflows/ci.yml   CI : lint → tests → build → déclenche le CD
```

## Démarrage rapide (local)

```bash
cp .env.example .env      # renseigner les valeurs (jamais committées)
docker-compose up -d
```

| Service | URL |
|---|---|
| API de prédiction | http://localhost:8000 (`/health`, `/predict`, `/predict/batch`) |
| Dashboard logistique | http://localhost:8050 |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"order_id":"demo1","nb_items":2,"total_price":250,"total_freight":35,"cross_state_delivery":true}'
# → {"risk_pct": 23.4, "risk": {"level":"high","action":"Relancer le vendeur..."}}
```

## Tests

```bash
pip install pandas pytest
python -m pytest tests/ -v      # 15 tests, < 10 s, sans modèle ni réseau
```

La logique métier (seuils, features) est isolée dans
`src/api/risk_logic.py` précisément pour être testable sans
dépendances lourdes — c'est ce que la CI exécute à chaque push.

## Chaîne CI/CD (vue d'ensemble)

```
push sur main (ce dépôt)
  └─ CI : lint → tests → build Docker          [.github/workflows/ci.yml]
       └─ si vert : repository_dispatch ─────→ dépôt shopbr-deploy
                                                └─ CD : build+push ECR → deploy EKS
```

Secret à configurer dans **ce** dépôt : `DEPLOY_REPO_TOKEN`
(PAT fine-grained avec accès au dépôt shopbr-deploy) — et remplacer
`VOTRE_COMPTE` dans `ci.yml`.

## Compétences de certification couvertes

| Code | Où |
|---|---|
| C2.4 | `sql/` (DISTKEY/SORTKEY), couches Bronze/Silver/Gold |
| C2.7 | `src/monitoring/`, healthchecks, logs structurés |
| C4.3 | `src/api/api.py` (API REST intégrée à l'infra) |
<!-- test de la chaine CI/CD -->
| C4.5 | `dags/airflow_dag.py` + `src/training/mlflow_train.py` |


