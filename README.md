# 🏆 Prédiction du vainqueur de la Ligue des Champions

Projet de data science qui analyse l'historique de la Ligue des Champions (UEFA Champions League) et utilise le machine learning pour estimer la probabilité de victoire finale de chaque club engagé sur la saison **2026-2027**.

> Les prédictions sont des estimations statistiques. Elles ne remplacent ni le résultat réel des matchs, ni l'analyse d'un spécialiste : blessures, tirages au sort et forme du moment restent imprévisibles.

---

## ❓ La question du projet

- **Question :** quels clubs ont le plus de chances de remporter la Ligue des Champions 2026-2027 ?
- **Pourquoi :** passer d'un avis de supporter (« le Real gagne toujours ») à une estimation chiffrée et défendable, appuyée sur les données.
- **Livrable :** un classement des 36 clubs engagés, avec une probabilité de titre pour chacun.
- **Critère de réussite :** le modèle prédit les matchs passés mieux qu'une règle simple (par exemple « l'équipe à domicile gagne » ou « l'équipe avec le meilleur Elo gagne »), sur des saisons qu'il n'a pas vues pendant l'entraînement.

---

## 🎯 Objectifs

- Explorer et nettoyer les données historiques de la Ligue des Champions (2012-2026) ;
- Construire des indicateurs de niveau et de forme des clubs (Elo, forme récente, buts, expérience européenne, valeur de l'effectif) ;
- Entraîner et comparer des modèles **Random Forest** et **XGBoost** ;
- Produire un classement probabiliste des équipes candidates au titre 2026-2027.

---

## 📁 Structure du projet

```
CHAMPIONS-/
├── analyse/
│   └── EDA.ipynb                    # Analyse exploratoire (EDA)
├── notebooks/                       # Notebooks de modélisation (à venir)
├── data/                            # Datasets CSV (non versionnés, voir ci-dessous)
├── .gitignore
└── README.md
```

---

## 📊 Données

Les fichiers CSV sont **ignorés par Git** (`*.csv` dans `.gitignore`) car trop volumineux. Chaque membre doit les télécharger depuis Kaggle et les placer dans le dossier `data/`.

### 1. Football Data from Transfermarkt (source principale)

Lien : https://www.kaggle.com/datasets/davidcariboo/player-scores

| Fichier | Contenu |
|---|---|
| `games.csv` | Tous les matchs (score, date, phase, stade, entraîneurs, arbitre…) — **une ligne = un match** |
| `competitions.csv` | Liste des compétitions et leurs identifiants |
| `clubs.csv` | Informations sur les clubs |
| `club_games.csv` | Un match vu du côté de chaque club |
| `players.csv` | Informations sur les joueurs |
| `player_valuations.csv` | Historique des valeurs marchandes des joueurs |
| `appearances.csv` | Statistiques individuelles par match |
| `game_events.csv` | Événements des matchs (buts, cartons, remplacements) |
| `game_lineups.csv` | Compositions d'équipe |

`competitions.csv`, `player_valuations.csv` et `game_lineups.csv` ne sont pas nécessaires pour l'EDA ; `player_valuations.csv` servira pour la valeur des effectifs lors de la modélisation.

### 2. Club Football Match Data (2000-2025)

Lien : https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025

| Fichier | Contenu |
|---|---|
| `EloRatings.csv` | Classement Elo des clubs dans le temps (jusqu'au 1er septembre 2026) |
| `Matches.csv` | Matchs de championnats nationaux avec Elo, forme, tirs et cotes (jusqu'au 3 septembre 2026) |

⚠️ `Matches.csv` ne contient **que des championnats nationaux**, aucun match de Ligue des Champions. Il sert à mesurer la forme des clubs en championnat.

### Filtrer la Ligue des Champions

Dans `games.csv`, la colonne `competition_id` identifie la compétition :

- `CL` : Ligue des Champions, 1 871 matchs (saisons 2012-13 à 2025-26) ;
- `CLQ` : qualifications, 1 232 matchs.

```python
import pandas as pd

games = pd.read_csv("../data/games.csv")
ucl = games[games["competition_id"] == "CL"].sort_values("date")
```

### Points d'attention

- `season` correspond à l'année de début de saison : `2025` = saison 2025-2026 ;
- La saison 2025-2026 s'arrête aux demi-finales (dernier match : 6 mai 2026) : la finale 2026 n'est pas dans les données ;
- **Aucun match de la saison 2026-2027 n'est dans les données** : la liste des 36 clubs qualifiés doit venir d'une source externe (site de l'UEFA). Les Elo (1er septembre 2026) donnent le niveau des clubs juste avant la phase de ligue ;
- Les noms de phase ne sont pas uniformes (`last 16 1st leg` / `Last 16 1st Leg`, `Group A` / `Group Stage`…) et doivent être harmonisés ;
- Certains scores incluent les tirs au but (ex. finale 2016 notée 6-4 au lieu de 1-1, Atlético–PSV 2016 noté 8-7 au lieu de 0-0) : ils faussent les statistiques de buts ;
- Les colonnes `home_club_position` / `away_club_position` sont vides pour tous les matchs de Ligue des Champions : un `dropna()` sur toute la table supprime donc **tous** les matchs de C1 ;
- Les noms de clubs diffèrent entre les datasets (`FC Barcelona` / `Barcelona`, `Paris Saint-Germain` / `Paris SG`) : une table de correspondance est nécessaire pour fusionner avec les Elo.

---

## ⚙️ Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/moussa197/CHAMPIONS-.git
   cd CHAMPIONS-
   ```
2. Installer les dépendances (Python 3.10+ ou Anaconda) :
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
   ```
3. Télécharger les datasets Kaggle et placer les CSV dans `data/`.
4. Lancer Jupyter et ouvrir `analyse/EDA.ipynb` :
   ```bash
   jupyter notebook
   ```

Les notebooks lisent les données avec des chemins relatifs (`../data/games.csv`) : ils doivent être lancés depuis leur propre dossier.

---

## 🔬 Méthodologie

1. **Analyse exploratoire** : structure des données, qualité (valeurs manquantes, doublons, anomalies), distribution des buts et des résultats, avantage du terrain, meilleurs clubs, lien entre Elo et résultat.
2. **Nettoyage** : harmonisation des phases et des noms de clubs, conversion des dates, correction des scores avec tirs au but, gestion des valeurs manquantes.
3. **Feature engineering** : Elo avant chaque match, forme récente (championnat et Europe), buts marqués/encaissés, expérience en Ligue des Champions, valeur marchande de l'effectif, confrontations directes.
4. **Modélisation** : Random Forest et XGBoost pour prédire l'issue des matchs (victoire / nul / défaite), avec séparation chronologique entraînement / test.
5. **Évaluation** : accuracy, précision / rappel / F1, matrice de confusion, log loss et calibration des probabilités, comparaison avec une règle simple (Elo ou domicile).
6. **Simulation** : simulation de la saison 2026-2027 (phase de ligue puis phase finale) à partir des probabilités de match pour obtenir la probabilité de titre de chaque club.

---

## 📈 Premiers résultats (EDA)

Sur les 1 871 matchs de Ligue des Champions (2012-2026) :

- 106 clubs différents ;
- 1,74 but en moyenne pour l'équipe à domicile contre 1,36 à l'extérieur (+0,38) ;
- Résultats : 47,7 % de victoires à domicile, 19,8 % de nuls, 32,5 % de victoires à l'extérieur ;
- Clubs les plus victorieux : Real Madrid (112 victoires), Bayern Munich (111), FC Barcelone (85), Paris Saint-Germain (84), Manchester City (77).

---

## 🚧 Avancement

- [x] Collecte des données
- [ ] Analyse exploratoire (en cours)
- [ ] Nettoyage et harmonisation
- [ ] Feature engineering
- [ ] Entraînement des modèles
- [ ] Prédiction 2026-2027

---

## 🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Jupyter · Kaggle

---

## 👥 Auteurs

Maxime, Loan, Moussa, Mouhamedou et Ousmane — septembre 2026