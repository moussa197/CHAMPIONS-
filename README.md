# Prédiction du Vainqueur de la Ligue des Champions

## 📌 Description

Ce projet utilise le **machine learning** pour analyser les performances des clubs européens et estimer quelle équipe a le plus de chances de remporter la Ligue des Champions. L'objectif est de transformer des données historiques, sportives et statistiques en probabilités de victoire compréhensibles et exploitables.

Les prédictions constituent une estimation statistique : elles ne remplacent ni le résultat réel d'un match, ni l'analyse d'un spécialiste du football.

## 🎯 Objectif principal

Le projet vise à :

- Analyser les données historiques de la Ligue des Champions ;
- Construire un modèle de machine learning basé sur **XGBoost** et/ou **Random Forest** ;
- Prédire l'équipe gagnante de la saison **2025-2026** ;
- Comparer les performances des équipes à partir d'indicateurs objectifs ;
- Produire une probabilité de victoire pour chaque équipe étudiée.

## 🔬 Démarche et méthodologie

### 1. Collecte des données

Les données sont issues principalement de jeux de données disponibles sur **Kaggle**, complétés si nécessaire par des informations publiques sur les clubs et les compétitions européennes.

### 2. Nettoyage et préparation

Cette étape comprend notamment :

- La suppression ou l'imputation des valeurs manquantes ;
- L'harmonisation des noms d'équipes et des compétitions ;
- La conversion des dates et des variables numériques ;
- La suppression des doublons et des données incohérentes ;
- La séparation des données d'entraînement et de test.

### 3. Feature engineering

De nouvelles variables sont créées pour mieux représenter le niveau et la dynamique des équipes, par exemple :

- Le coefficient UEFA ;
- La forme récente et le nombre de victoires consécutives ;
- Les buts marqués et encaissés ;
- Les expected goals (**xG**) et expected goals against (**xGA**) ;
- Les performances à domicile et à l'extérieur ;
- L'expérience en Ligue des Champions ;
- La qualité et la disponibilité des joueurs.

### 4. Entraînement du modèle

Les modèles **XGBoost** et **Random Forest** sont entraînés sur les rencontres et performances historiques. Une validation croisée et une recherche d'hyperparamètres peuvent être utilisées afin d'améliorer la robustesse des résultats et de limiter le surapprentissage.

### 5. Évaluation et prédictions

Les modèles sont évalués à l'aide de métriques adaptées à la classification et aux probabilités, notamment :

- L'accuracy ;
- La précision, le rappel et le score F1 ;
- La matrice de confusion ;
- Le log loss et la calibration des probabilités.

Le modèle retenu est ensuite appliqué aux équipes de la saison 2025-2026 afin de générer les probabilités de victoire.

## 📊 Datasets utilisés

- **Club Football Match Data (2000-2025)** : résultats, performances et statistiques historiques des clubs ;
- **UEFA Champions League 2025 Players Data** : informations et indicateurs relatifs aux joueurs ;
- **Football Matches Dataset 2025** : résultats et statistiques récentes des rencontres.

> Les fichiers de données ne sont pas nécessairement inclus dans le dépôt. Ils doivent être téléchargés depuis Kaggle et placés dans le dossier `data/` en respectant les noms attendus par les notebooks ou scripts du projet.

## 🛠️ Technologies

- **Python** : langage principal ;
- **Pandas** : manipulation et analyse des données ;
- **Scikit-learn** : prétraitement, entraînement et évaluation ;
- **XGBoost** : modèle de gradient boosting ;
- **Kaggle** : source principale des datasets.

## 📈 Résultats attendus

Le projet doit produire un classement probabiliste des équipes candidates, comprenant notamment :

- Une probabilité de victoire pour chaque équipe ;
- L'équipe présentant la probabilité estimée la plus élevée ;
- Un tableau ou une visualisation permettant de comparer les candidats ;
- Des indicateurs d'évaluation permettant de mesurer la fiabilité du modèle.

Ces résultats doivent être interprétés avec prudence, car une compétition sportive reste influencée par les blessures, les tirages, la forme du moment et de nombreux facteurs imprévisibles.

## 👥 Auteur et date

**Auteurs :** Moussa, Loan, Maxime , Mouhamedou et  Ousmane

**Date :** septembre 2026