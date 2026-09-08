"""
====================================================================
Projet #7 - ShopBR - ML Distribue avec Ray
Ray Train  : entrainement XGBoost distribue multi-workers
RLlib      : optimisation de la politique de priorisation logistique
====================================================================
Competence C2.6 : Clusters de calcul pour l'entrainement IA

A QUOI SERT CE SCRIPT ?
Ce script contient TROIS demonstrations differentes, toutes basees
sur la bibliotheque Ray :

1. RAY TRAIN : entrainer le MEME modele XGBoost que dans
   mlflow_train.py, mais en repartissant le calcul sur PLUSIEURS
   machines/workers en parallele, plutot que sur une seule.

2. RAY TUNE : tester automatiquement plusieurs combinaisons de
   parametres du modele (ce qu'on appelle des "hyperparametres") pour
   trouver la meilleure configuration, en parallelisant ces tests.

3. RLLIB : une approche DIFFERENTE du Machine Learning, l'apprentissage
   par renforcement (Reinforcement Learning), utilisee ici pour
   resoudre un probleme complementaire : non plus "predire le
   retard", mais "decider comment prioriser les commandes a risque
   dans l'entrepot quand les ressources logistiques sont limitees".

POURQUOI DISTRIBUER L'ENTRAINEMENT (RAY TRAIN) ALORS QUE
mlflow_train.py FONCTIONNE DEJA SUR UNE SEULE MACHINE ?
Avec le volume actuel (~95 000 commandes), une seule machine suffit
largement. Mais ShopBR prevoit d'integrer de nouveaux vendeurs : si le
volume de donnees double ou triple, un entrainement sur une seule
machine deviendrait beaucoup trop lent. Ray permet de PREPARER cette
montee en charge des maintenant, sans avoir a reecrire le code plus
tard quand le probleme deviendra reellement critique.
====================================================================
"""

# ray : la bibliotheque principale qui gere la distribution des
# calculs sur plusieurs machines (ou plusieurs coeurs/processus sur
# une seule machine, en mode local).
import ray
from ray import train
# XGBoostTrainer : composant specialise de Ray Train qui sait
# entrainer un modele XGBoost de maniere distribuee.
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
# Ray Tune : sous-bibliotheque dediee a l'optimisation d'hyperparametres
from ray.tune import TuneConfig, Tuner, grid_search
# RLlib : sous-bibliotheque dediee a l'apprentissage par renforcement
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
# gymnasium : bibliotheque standard pour definir des "environnements"
# d'apprentissage par renforcement (la "simulation" dans laquelle
# l'agent va s'entrainer)
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import mlflow
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


# ====================================================================
# PARTIE 0 : L'ENVIRONNEMENT POUR RLLIB
# ====================================================================
#
# QU'EST-CE QUE L'APPRENTISSAGE PAR RENFORCEMENT (REINFORCEMENT LEARNING) ?
# C'est une approche TRES DIFFERENTE de XGBoost. Au lieu d'apprendre a
# partir d'exemples deja etiquetes (donnees historiques avec leur
# bonne reponse connue), un "agent" apprend par ESSAI-ERREUR dans un
# environnement simule : il prend des decisions, recoit une
# "recompense" (reward) qui mesure la qualite de cette decision, et
# ajuste progressivement son comportement pour maximiser ses
# recompenses cumulees dans le temps.
#
# POURQUOI L'UTILISER ICI ?
# XGBoost (dans mlflow_train.py) repond a la question "cette commande
# va-t-elle etre en retard ?". Mais une fois qu'on sait qu'une
# commande est a risque, il reste une AUTRE decision a prendre :
# parmi toutes les commandes a risque, LAQUELLE traiter en priorite
# dans l'entrepot quand on n'a pas assez d'employes pour toutes les
# emballer immediatement ? C'est un probleme de DECISION SEQUENTIELLE
# sous contrainte de ressources, exactement le type de probleme pour
# lequel le Reinforcement Learning est concu.
class LogisticsPriorityEnv(gym.Env):
    """
    Environnement Gymnasium simule pour optimiser la priorisation des
    commandes a risque dans l'entrepot ShopBR.

    COMMENT FONCTIONNE CET ENVIRONNEMENT ?
    A chaque "etape" (step), l'environnement presente une commande
    avec certaines caracteristiques (son "etat" ou state), l'agent
    choisit une action (priorite basse/normale/haute), et
    l'environnement renvoie une recompense (reward) qui depend de la
    qualite de cette decision.

    STATE (l'etat observe par l'agent), 4 valeurs entre 0 et 1 :
      [risque_de_retard, valeur_de_la_commande, urgence, cross_state_delivery]

    ACTION (la decision que l'agent peut prendre) :
      0 = priorite basse
      1 = priorite normale
      2 = priorite haute (traitee immediatement)

    REWARD (recompense) :
      positive si la decision evite une mauvaise note client a moindre
      cout de ressources, negative si elle gaspille des ressources sur
      une commande peu risquee, ou si elle neglige une commande
      vraiment urgente et a risque.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: EnvContext = None):
        super().__init__()
        # action_space definit l'ensemble des actions possibles pour
        # l'agent : ici, un choix parmi 3 options (Discrete(3) = 0, 1 ou 2).
        self.action_space      = spaces.Discrete(3)
        # observation_space definit la forme et les limites de l'etat
        # observable par l'agent : 4 valeurs continues, chacune entre 0 et 1.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.max_steps  = 20  # un "episode" d'entrainement = 20 commandes traitees consecutivement (un batch d'entrepot)
        self.step_count = 0
        self.state      = None

    def reset(self, seed=None, options=None):
        """
        Reinitialise l'environnement au debut d'un nouvel "episode"
        (une nouvelle serie de 20 commandes a traiter). C'est une
        methode standard requise par l'interface Gymnasium.
        """
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._random_order_state()
        return self.state, {}

    def _random_order_state(self):
        """
        Genere une commande aleatoire realiste, en utilisant des
        distributions qui imitent la realite observee dans le Projet #6 :
        la plupart des commandes ont un risque faible (la loi "beta(2,8)"
        genere majoritairement des petites valeurs, comme le taux de
        retard reel de 6,65% qui reste minoritaire).
        """
        return np.array([
            np.random.beta(2, 8),       # risque de retard simule (biais realiste vers les valeurs faibles)
            np.random.uniform(0, 1),    # valeur normalisee de la commande
            np.random.uniform(0, 1),    # urgence (0 = tres urgent, 1 = pas urgent)
            np.random.randint(0, 2),    # cross_state_delivery (0 ou 1)
        ], dtype=np.float32)

    def step(self, action):
        """
        Methode centrale : execute UNE decision de l'agent et calcule
        la recompense correspondante.

        C'est ici que se trouve la "logique metier" qui traduit une
        decision de priorisation en consequence chiffree.
        """
        risk, value, urgency, cross_state = self.state

        # Cout de la priorisation : traiter une commande en priorite
        # haute mobilise plus de ressources (emballeurs, transport
        # express) que de la traiter normalement.
        priority_cost = [0.0, 0.05, 0.15][action]

        # "risk_mitigation" represente a quel point la priorisation
        # REDUIT le risque effectif de retard pour cette commande.
        # Une priorite haute reduit le risque de 70%, une priorite
        # normale de seulement 30%, une priorite basse n'a aucun effet.
        if action == 2:
            risk_mitigation = risk * 0.7
        elif action == 1:
            risk_mitigation = risk * 0.3
        else:
            risk_mitigation = 0.0

        # La recompense principale : la "valeur preservee" en evitant
        # une mauvaise note client, moins le cout des ressources utilisees.
        avoided_bad_review = risk_mitigation * value * 2.0
        reward = avoided_bad_review - priority_cost

        # Bonus specifique : si la commande est a la fois TRES urgente
        # (urgency < 0.2) ET vraiment a risque (risk > 0.15), et que
        # l'agent a correctement choisi la priorite haute, on ajoute
        # un bonus pour bien renforcer ce comportement souhaitable
        # pendant l'apprentissage.
        if urgency < 0.2 and risk > 0.15 and action == 2:
            reward += 0.3

        # On passe a la commande suivante (nouvel etat aleatoire) et
        # on verifie si l'episode (la serie de 20 commandes) est terminee.
        self.step_count += 1
        self.state = self._random_order_state()
        terminated = self.step_count >= self.max_steps

        # Une methode step() de Gymnasium doit toujours renvoyer ces 5
        # elements : le nouvel etat, la recompense obtenue, si
        # l'episode est termine, si l'episode a ete tronque
        # (interrompu pour une autre raison, non utilise ici), et des
        # informations additionnelles optionnelles (dictionnaire vide ici).
        return self.state, reward, terminated, False, {}


# ====================================================================
# PARTIE 1 : RAY TRAIN - ENTRAINEMENT XGBOOST DISTRIBUE
# ====================================================================

def run_distributed_training(num_workers: int = 2) -> dict:
    """
    Entraine le modele XGBoost en repartissant le calcul sur plusieurs
    "workers" (processus de calcul, potentiellement sur des machines
    differentes dans un vrai cluster).

    DIFFERENCE AVEC mlflow_train.py :
    Dans mlflow_train.py, xgb.XGBClassifier() entraine le modele sur
    UNE SEULE machine, du debut a la fin. Ici, XGBoostTrainer de Ray
    decoupe le jeu de donnees et le travail de calcul entre plusieurs
    workers qui collaborent pour produire UN SEUL modele final,
    exactement comme s'ils etaient une seule grosse machine plus
    puissante.
    """
    logger.info(f"Demarrage entrainement distribue Ray - {num_workers} workers")

    # ray.init() demarre (ou rejoint) le "cluster" Ray. En local (sur
    # un seul ordinateur), Ray simule plusieurs workers en utilisant
    # les differents coeurs du processeur. Sur un vrai cluster cloud,
    # chaque worker peut etre une machine physiquement separee.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    logger.info(f"Ray initialise - ressources : {ray.cluster_resources()}")

    # Generation des donnees d'entrainement (memes statistiques
    # calibrees que dans mlflow_train.py).
    np.random.seed(42)
    n = 95000
    cross_state = np.random.random(n) < 0.638
    df = pd.DataFrame({
        "nb_items":             np.random.randint(1, 6, n),
        "nb_sellers":           np.random.randint(1, 3, n),
        "total_price":          np.round(np.random.lognormal(4.5, 1.0, n), 2),
        "total_freight":        np.round(np.random.lognormal(2.8, 0.6, n), 2),
        "cross_state_delivery": cross_state.astype(int),
        "purchase_weekday":     np.random.randint(1, 8, n),
        "purchase_hour":        np.random.randint(0, 24, n),
        "estimated_weight_kg":  np.round(np.random.lognormal(0.5, 1.0, n), 2),
        "max_installments":     np.random.randint(1, 10, n),
    })
    base_proba = 0.0665
    proba = np.where(cross_state, base_proba * 1.8, base_proba * 0.6)
    df["is_late"] = (np.random.random(n) < proba).astype(int)

    # ray.data.from_pandas() convertit notre tableau pandas classique
    # en "Ray Dataset", une structure de donnees DISTRIBUEE que Ray
    # peut repartir entre plusieurs workers.
    dataset = ray.data.from_pandas(df)
    train_ds, val_ds = dataset.train_test_split(test_size=0.2)

    # Configuration de l'entrainement distribue
    trainer = XGBoostTrainer(
        # ScalingConfig definit COMMENT on distribue le calcul.
        scaling_config=ScalingConfig(
            num_workers=num_workers,    # nombre de workers paralleles
            use_gpu=False,               # True si des GPU sont disponibles sur le cluster (accelere encore l'entrainement)
            resources_per_worker={"CPU": 2, "memory": 2 * 1024**3},  # ressources allouees a CHAQUE worker
        ),
        # RunConfig configure la sauvegarde des resultats intermediaires
        run_config=RunConfig(
            name="shopbr_distributed_training",
            checkpoint_config=CheckpointConfig(
                # On garde uniquement les 3 meilleurs "checkpoints"
                # (sauvegardes intermediaires du modele pendant
                # l'entrainement), classes par leur score sur les
                # donnees d'entrainement.
                num_to_keep=3, checkpoint_score_attribute="train-logloss",
                checkpoint_score_order="min",  # "min" car logloss : plus c'est bas, meilleur c'est
            ),
        ),
        label_column="is_late",  # la colonne cible a predire
        params={
            "objective":        "binary:logistic",  # type de probleme : classification binaire
            "n_estimators":     500,
            "max_depth":        5,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "tree_method":      "hist",  # methode de construction des arbres optimisee pour la vitesse
            "eval_metric":      ["logloss", "error"],
        },
        datasets={"train": train_ds, "validation": val_ds},
    )

    # .fit() lance reellement l'entrainement distribue
    result = trainer.fit()
    logger.info(f"Entrainement distribue termine - metriques : {result.metrics}")

    # On enregistre aussi le resultat dans MLflow, exactement comme
    # pour l'entrainement classique, pour garder une trace comparable
    # de toutes les approches testees.
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="ray_distributed_train"):
        mlflow.log_params({"num_workers": num_workers, "n_samples": n})
        mlflow.log_metrics(result.metrics)

    return result.metrics


# ====================================================================
# PARTIE 2 : RAY TUNE - OPTIMISATION DES HYPERPARAMETRES
# ====================================================================

def run_hyperparameter_tuning(num_samples: int = 9) -> dict:
    """
    QU'EST-CE QU'UN HYPERPARAMETRE ?
    Ce sont les "reglages" d'un modele qu'on choisit AVANT
    l'entrainement (par opposition aux parametres internes que le
    modele apprend lui-meme a partir des donnees). Par exemple,
    n_estimators, max_depth, learning_rate sont des hyperparametres
    de XGBoost.

    Trouver la MEILLEURE combinaison d'hyperparametres "a la main"
    serait tres long (il faudrait tester chaque combinaison une par
    une, sequentiellement). Ray Tune AUTOMATISE et PARALLELISE cette
    recherche : plusieurs combinaisons sont testees EN MEME TEMPS sur
    differents workers, ce qui reduit considerablement le temps total
    necessaire.
    """
    logger.info(f"Demarrage Ray Tune - {num_samples} configurations")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    def train_fn(config):
        """
        Cette fonction interne represente UN ESSAI : elle entraine un
        modele avec UNE combinaison precise d'hyperparametres (donnee
        par "config"), et rapporte son score. Ray Tune appelle cette
        fonction plusieurs fois, avec des valeurs differentes de config.
        """
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        np.random.seed(42)
        n = 20000  # echantillon plus petit que l'entrainement complet, pour que chaque essai reste rapide
        cross_state = np.random.random(n) < 0.638
        X = pd.DataFrame({
            "nb_items":             np.random.randint(1, 6, n),
            "total_price":          np.round(np.random.lognormal(4.5, 1.0, n), 2),
            "cross_state_delivery": cross_state.astype(int),
            "purchase_weekday":     np.random.randint(1, 8, n),
        })
        proba = np.where(cross_state, 0.0665 * 1.8, 0.0665 * 0.6)
        y = (np.random.random(n) < proba).astype(int)

        model = xgb.XGBClassifier(**config)
        # cross_val_score() entraine et evalue le modele plusieurs
        # fois sur des sous-ensembles differents des donnees (3 fois
        # ici, cv=3), pour obtenir un score plus fiable qu'un simple
        # test unique.
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        # train.report() est la facon dont Ray Tune recupere le
        # resultat de cet essai pour le comparer aux autres.
        train.report({"recall": scores.mean()})

    # Tuner orchestre l'ensemble du processus de recherche.
    tuner = Tuner(
        train_fn,
        param_space={
            # grid_search([...]) dit a Ray Tune de tester TOUTES les
            # valeurs listees pour ce parametre, en les combinant avec
            # toutes les valeurs des autres parametres en grid_search
            # (recherche exhaustive sur une "grille" de combinaisons).
            "n_estimators":  grid_search([200, 400, 600]),
            "max_depth":     grid_search([4, 5, 6]),
            "learning_rate": grid_search([0.03, 0.05, 0.1]),
            "random_state":  42,
        },
        tune_config=TuneConfig(
            metric="recall",   # on cherche a OPTIMISER cette metrique...
            mode="max",        # ...en la MAXIMISANT (on veut le plus haut rappel possible)
            num_samples=num_samples,
        ),
    )
    results = tuner.fit()
    # get_best_result() recupere automatiquement la MEILLEURE
    # combinaison testee, selon le critere defini ci-dessus.
    best = results.get_best_result(metric="recall", mode="max")
    logger.info(f"Meilleure config : {best.config}")
    logger.info(f"Meilleur rappel  : {best.metrics['recall']:.2%}")
    return best.config


# ====================================================================
# PARTIE 3 : RLLIB - PRIORISATION LOGISTIQUE
# ====================================================================

def run_rllib_priority(num_iterations: int = 30) -> None:
    """
    Entraine un agent d'apprentissage par renforcement (algorithme
    PPO) sur l'environnement LogisticsPriorityEnv defini plus haut.

    QU'EST-CE QUE PPO ?
    PPO (Proximal Policy Optimization) est l'un des algorithmes de
    Reinforcement Learning les plus utilises en pratique, reconnu
    pour sa stabilite d'entrainement. Il ajuste progressivement la
    "politique" de l'agent (sa strategie de decision) pour maximiser
    les recompenses obtenues, sans faire de changements trop brusques
    qui destabiliseraient l'apprentissage.
    """
    logger.info(f"Entrainement RLlib PPO - {num_iterations} iterations")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # On doit d'abord "enregistrer" notre environnement personnalise
    # aupres de Ray, sous un nom (ici "ShopBRPriority-v0"), pour que
    # RLlib puisse le creer et l'utiliser pendant l'entrainement.
    from ray.tune.registry import register_env
    register_env("ShopBRPriority-v0", lambda config: LogisticsPriorityEnv(config))

    # PPOConfig configure l'algorithme d'apprentissage.
    config = (
        PPOConfig()
        .environment("ShopBRPriority-v0")  # quel environnement utiliser
        .rollouts(num_rollout_workers=2)   # combien de workers paralleles collectent de l'experience en simulant des episodes
        .training(
            gamma=0.95,           # "facteur d'actualisation" : a quel point l'agent valorise les recompenses futures par rapport aux recompenses immediates
            lr=0.0003,            # taux d'apprentissage de l'algorithme
            clip_param=0.2,       # parametre technique de PPO qui limite l'ampleur des changements de strategie a chaque mise a jour (stabilite)
            train_batch_size=3000 # nombre d'experiences collectees avant chaque mise a jour de la strategie
        )
        .evaluation(evaluation_num_workers=1, evaluation_interval=10)  # evalue periodiquement la performance de l'agent independamment de l'entrainement
    )

    # .build() construit l'algorithme pret a etre entraine
    algo = config.build()
    best_reward = -np.inf

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="rllib_logistics_priority"):
        # Boucle d'entrainement : chaque appel a algo.train() fait
        # progresser l'agent d'une "iteration" (l'agent simule de
        # nombreux episodes, observe ses recompenses, et ajuste sa strategie).
        for i in range(num_iterations):
            result = algo.train()
            reward = result["episode_reward_mean"]  # recompense moyenne obtenue sur les episodes recents

            if reward > best_reward:
                best_reward = reward
                algo.save()  # sauvegarde un "checkpoint" de l'agent a chaque nouveau record
                logger.info(f"  Iter {i+1:3d} - reward={reward:.3f} nouveau meilleur")
            elif (i + 1) % 10 == 0:
                logger.info(f"  Iter {i+1:3d} - reward={reward:.3f}")

            mlflow.log_metric("episode_reward_mean", reward, step=i)

        mlflow.log_metric("best_reward", best_reward)
        logger.info(f"RLlib termine - meilleur reward : {best_reward:.3f}")

    algo.stop()


# ====================================================================
# POINT D'ENTREE DU SCRIPT
# ====================================================================
if __name__ == "__main__":
    # argparse permet de choisir, en ligne de commande, quelle(s)
    # demonstration(s) lancer, par exemple :
    # python ray_train.py --mode all --num-workers 4
    parser = argparse.ArgumentParser(description="Entrainement distribue ShopBR avec Ray")
    parser.add_argument("--mode", choices=["distributed", "tune", "rllib", "all"], default="distributed")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument("--num-iterations", type=int, default=20)
    args = parser.parse_args()

    if args.mode in ("distributed", "all"):
        logger.info("=== Ray Train - Entrainement distribue ===")
        run_distributed_training(num_workers=args.num_workers)

    if args.mode in ("tune", "all"):
        logger.info("=== Ray Tune - Optimisation hyperparametres ===")
        run_hyperparameter_tuning(num_samples=args.num_samples)

    if args.mode in ("rllib", "all"):
        logger.info("=== RLlib - Priorisation logistique ===")
        run_rllib_priority(num_iterations=args.num_iterations)

    # ray.shutdown() arrete proprement le cluster Ray et libere les
    # ressources, une fois toutes les demonstrations terminees.
    if ray.is_initialized():
        ray.shutdown()
    logger.info("=== Ray termine ===")
