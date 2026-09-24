# 🏆 Prédiction du vainqueur de la Ligue des Champions

Projet de data science qui analyse l'historique de la Ligue des Champions (UEFA Champions League) et utilise le machine learning pour estimer la probabilité de victoire finale de chaque club engagé sur la saison **2025-2026**.

> Les prédictions sont des estimations statistiques. Elles ne remplacent ni le résultat réel des matchs, ni l'analyse d'un spécialiste : blessures, tirages au sort et forme du moment restent imprévisibles.

---

## 🎯 Objectifs

- Explorer et nettoyer les données historiques de la Ligue des Champions (2012-2026) ;
- Construire des indicateurs de niveau et de forme des clubs (Elo, forme récente, buts, expérience européenne, valeur de l'effectif) ;
- Entraîner et comparer des modèles **Random Forest** et **XGBoost** ;
- Produire un classement probabiliste des équipes candidates au titre 2025-2026.

---

## 📁 Structure du projet

```
CHAMPIONS-/
├── analyse/
│   └── eda_champions_league.ipynb   # Analyse exploratoire (EDA)
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
| `games.csv` | Tous les matchs (score, date, phase, stade, entraîneurs, arbitre…) |
| `competitions.csv` | Liste des compétitions et leurs identifiants |
| `clubs.csv` | Informations sur les clubs |
| `club_games.csv` | Un match vu du côté de chaque club |
| `players.csv` | Informations sur les joueurs |
| `player_valuations.csv` | Historique des valeurs marchandes des joueurs |
| `appearances.csv` | Statistiques individuelles par match |
| `game_events.csv` | Événements des matchs (buts, cartons, remplacements) |
| `game_lineups.csv` | Compositions d'équipe |

### 2. Club Football Match Data (2000-2025)

Lien : https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025

| Fichier | Contenu |
|---|---|
| `EloRatings.csv` | Classement Elo des clubs dans le temps |
| `Matches.csv` | Matchs de championnats nationaux avec Elo, forme, tirs et cotes |

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
- La saison 2025-2026 s'arrête aux demi-finales (dernier match : 6 mai 2026) ;
- Les noms de phase ne sont pas uniformes (`last 16 1st leg` / `Last 16 1st Leg`, `Group A` / `Group Stage`…) et doivent être harmonisés ;
- Certaines finales incluent les tirs au but dans le score (ex. finale 2016 notée 6-4 au lieu de 1-1) ;
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
4. Lancer Jupyter et ouvrir `analyse/eda_champions_league.ipynb` :
   ```bash
   jupyter notebook
   ```

Les notebooks lisent les données avec des chemins relatifs (`../data/games.csv`) : ils doivent être lancés depuis leur propre dossier.

---

## 🔬 Méthodologie

1. **Analyse exploratoire** : structure des données, valeurs manquantes, distribution des buts, avantage du terrain, meilleurs clubs.
2. **Nettoyage** : harmonisation des phases et des noms de clubs, conversion des dates, gestion des valeurs manquantes.
3. **Feature engineering** : Elo avant chaque match, forme récente (championnat et Europe), buts marqués/encaissés, expérience en Ligue des Champions, valeur marchande de l'effectif, confrontations directes.
4. **Modélisation** : Random Forest et XGBoost pour prédire l'issue des matchs (victoire / nul / défaite), avec séparation chronologique entraînement / test.
5. **Évaluation** : accuracy, précision / rappel / F1, matrice de confusion, log loss et calibration des probabilités.
6. **Simulation** : simulation de la phase finale 2025-2026 à partir des probabilités de match pour obtenir la probabilité de titre de chaque club.

---

## 📈 Premiers résultats (EDA)

Sur les 1 871 matchs de Ligue des Champions (2012-2026) :

- 106 clubs différents ;
- 1,74 but en moyenne pour l'équipe à domicile contre 1,36 à l'extérieur (+0,38) ;
- 47,7 % de victoires à domicile ;
- Clubs les plus victorieux : Real Madrid (112 victoires), Bayern Munich (111), FC Barcelone (85), Paris Saint-Germain (84), Manchester City (77).

---

## 🚧 Avancement

- [x] Collecte des données
- [x] Analyse exploratoire
- [ ] Nettoyage et harmonisation
- [ ] Feature engineering
- [ ] Entraînement des modèles
- [ ] Prédiction 2025-2026

---

## 🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Jupyter · Kaggle

---

## 👥 Auteurs

Maxime, Loan, Moussa, Mouhamedou et Ousmane — septembre 2026
