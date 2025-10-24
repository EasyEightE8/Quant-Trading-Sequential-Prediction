# Quant-Trading-Sequential-Prediction

API REST (Python/FastAPI) pour la prédiction séquentielle et la génération de signaux de trading quantitatif.

## Objectif du Projet

Ce projet consiste à développer un service web complet (API REST) capable d'analyser les données de marché en temps réel pour générer des signaux de trading. Le service remplit quatre fonctions principales :

* **Analyser** les données de marché (OHLCV, volumes) en tant que séries temporelles.
* **Prédire** la direction probable des prix (Hausse/Baisse) à l'aide d'un modèle de Deep Learning séquentiel (LSTM/GRU).
* **Valider** la performance de la stratégie via un module de backtesting robuste.
* **Servir** ces signaux (Achat/Vente/Neutre) via un endpoint sécurisé et performant.

*(Ce projet fait suite à mon travail sur le Credit-Risk-Anomaly-Detection, passant de la modélisation du risque de bilan à celle de l'opportunité de marché.)*

---

## Phase 1 : Data Engineering & Ingestion de Données de Marché

**Objectif :** Mettre en place l'infrastructure d'ingestion et de stockage des données de séries temporelles (OHLCV) servant de "source de vérité" pour l'entraînement, la validation et le backtesting.

**Compétences et Technologies :**
* `[ ] Data Sourcing (yfinance / API Bloomberg)`: Extraction de données historiques (OHLCV, volumes) à différentes granularités (Jour, Heure, Minute) pour un panier d'actifs (ex: S&P 500, Cryptos).
* `[ ] SGBD Time-Series (InfluxDB / PostgreSQL)`: (Optionnel) Conception d'un schéma optimisé pour le stockage et l'interrogation rapide de données temporelles à haute fréquence.
* `[ ] Python (Pandas, NumPy, TA-Lib)`: Développement de scripts pour le nettoyage, le rééchantillonnage (resampling) et le feature engineering (Moyennes Mobiles, RSI, MACD, Bandes de Bollinger, Volatilité).
* `[ ] SQL & Optimisation` : Développement de requêtes performantes (ex: window functions) pour l'extraction de fenêtres de données (rolling windows) nécessaires à l'entraînement.

---

## Phase 2 : Modélisation (Deep Learning) et Backtesting

**Objectif :** Développer, entraîner et évaluer le cœur algorithmique : un modèle de prédiction séquentielle (supervisé) et un système de validation de stratégie (backtester).

**Compétences et Technologies :**
* `[ ] R&D (Jupyter Notebooks)` : Utilisation de notebooks pour l'analyse exploratoire (EDA) des séries temporelles, la visualisation et le prototypage des modèles.
* `[ ] Data Visualization (Matplotlib/Plotly)` : Création de graphiques (chandeliers, indicateurs) pour analyser les comportements du marché et la distribution des *features*.
* `[ ] Deep Learning (TensorFlow/Keras)` :
    * `[ ] Pré-traitement` : Création de séquences (fenêtres glissantes 3D) et normalisation des données (`MinMaxScaler`) pour l'input des réseaux récurrents.
    * `[ ] Modèle 1 (Prédiction)` : Entraînement d'un modèle séquentiel (ex: **LSTM**, **GRU**) pour prédire la direction du prix (Classification : Hausse/Baisse) ou sa valeur (Régression) à l'horizon $N+1$.
* `[ ] Backtesting (Backtrader / VectorBT)` : Développement d'un script rigoureux pour simuler l'application de la stratégie sur des données historiques ("out-of-sample") et évaluer sa performance financière (Ratio de Sharpe, Max Drawdown, P&L).

---

## Phase 3 : Industrialisation (API REST & DevOps)

**Objectif :** "Industrialiser" le modèle de Deep Learning en l'exposant via une API REST robuste, performante et scalable. Mise en place d'un pipeline d'intégration et de déploiement continus (CI/CD).

**Compétences et Technologies :**
* `[ ] Back-end (FastAPI / Flask)` : Développement d'une API RESTful en Python. **FastAPI** est privilégié pour ses performances asynchrones, cruciales pour la finance en temps réel.
* `[ ] API Design` : Création d'un endpoint principal `/signal` acceptant un ticker (ex: `BTC-USD`) et retournant un signal de trading (Achat/Vente/Neutre) et un score de confiance en JSON.
* `[ ] Conteneurisation (Docker)` : Création d'un `Dockerfile` pour encapsuler l'application FastAPI et ses lourdes dépendances (TensorFlow, TA-Lib), assurant une portabilité parfaite.
* `[ ] CI/CD (Git / GitHub Actions)` : Hébergement du code source sur GitHub (Easy Eight E8) et configuration d'un pipeline CI/CD pour automatiser les tests et le build de l'image Docker.

---

## Phase 4 : Interface de Consommation (Dashboarding)

**Objectif :** Développer un tableau de bord analytique (client web) pour consommer l'API. Cette interface sert de démonstrateur pour le backtesting et la visualisation des signaux.

**Compétences et Technologies :**
* `[ ] Dashboarding (Streamlit / Plotly Dash)` : Création d'une application web légère en Python (alternative au front-end JS) pour interagir de manière dynamique avec le back-end.
* `[ ] Interaction API (Python Requests)` : Appel de l'endpoint `/signal` depuis l'application Streamlit pour afficher la prédiction la plus récente.
* `[ ] Visualisation Interactive (Plotly)` : Affichage des chandeliers de prix, des indicateurs techniques et superposition des signaux d'Achat/Vente générés par l'API pour validation visuelle.

---

## Phase 5 : Analyse de Robustesse et Données Alternatives

**Objectif :** Démontrer une compréhension approfondie des risques et des limites d'un modèle purement "technique" (basé sur OHLCV) et proposer une feuille de route d'amélioration.

**Compétences et Technologies :**
* `[ ] Analyse Financière (Quant)` : Rédaction d'une analyse critique dans ce `README` (ou un Notebook dédié) sur les risques du modèle (surapprentissage/overfitting, dérive de concept, sensibilité aux régimes de marché).
* `[ ] Données Alternatives (NLP)` : Proposition conceptuelle d'amélioration du modèle par l'intégration de *features* non structurées (ex: analyse de sentiment des news, tweets) via des modèles NLP (ex: FinBERT).
* `[ ] Connaissances Marché (Bloomberg)` : Proposition d'intégration de *features* macro-économiques (ex: taux directeurs, VIX) pour capturer le risque systémique, en lien avec les certifications **Bloomberg (BMC, BQL, BFF)**.

## Phase 5 : Analyse Critique et Feuille de Route Future

Le modèle Stacked LSTM a validé le pipeline MLOps, mais a montré un biais significatif :

Ratio de Sharpe Final : 0.88 après ajustement du seuil et du taux sans risque.

Max Drawdown : 18.66%.

Problème Majeur : 

Le modèle manque de capacité discriminante (il est fortement biaisé vers la tendance de fond haussière), rendant sa performance trop proche du "Buy-and-Hold". Ce biais est la principale limite à corriger.

Architecture Future : 

Étude du Temporal Convolutional Network (TCN) comme alternative plus stable au LSTM.