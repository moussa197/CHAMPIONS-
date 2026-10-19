# 🏆 Prédiction du vainqueur de la Ligue des Champions

Projet de data science qui analyse l'historique de la Ligue des Champions (UEFA Champions League) et utilise le machine learning pour estimer la probabilité de victoire finale de chaque club engagé sur la saison **2026-2027**.

> Les prédictions sont des estimations statistiques. Elles ne remplacent ni le résultat réel des matchs, ni l'analyse d'un spécialiste : blessures, tirages au sort et forme du moment restent imprévisibles.

---

## ❓ La question du projet

- **Question :** quels clubs ont le plus de chances de remporter la Ligue des Champions 2026-2027 ?
- **Pourquoi :** passer d'un avis de supporter (« le Real gagne toujours ») à une estimation chiffrée et défendable, appuyée sur les données.
- **Livrable :** un classement des 36 clubs engagés, avec une probabilité de titre pour chacun.
- **Critère de réussite :** le modèle doit prédire les matchs mieux qu'une règle simple, sur des saisons qu'il n'a pas vues pendant l'entraînement. Référence à battre : la règle « l'équipe avec le meilleur Elo gagne » a raison dans **56 %** des matchs selon l'EDA v1 (Elo officiel, `EloRatings.csv`, matchs où les deux Elo sont connus) (à recalculer sur la base de modélisation).

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
│   ├── EDA.ipynb                    # Analyse exploratoire (EDA v1)
│   ├── v2_elo.ipynb                 # Elo avant chaque match de C1 (Moussa)
│   ├── v2_joueurs.ipynb             # Variables joueurs par équipe et par match (Mouhamedou)
│   ├── v2_fatigue.ipynb             # Repos et calendrier par club et par match (Ousmane)
│   ├── v2_participants.ipynb        # Clubs engagés en 2026-2027
│   └── EDA_v2.ipynb                 # EDA v2 : assemblage des tableaux, corrélations, conclusion (Loan)
├── notebooks/                       # Notebooks de modélisation (à venir)
├── data/                            # Datasets CSV (non versionnés, voir ci-dessous)
├── output/                          # CSV produits par les notebooks (non versionnés)
├── .gitignore
└── README.md
```

---

## 📊 Données

Les fichiers CSV sont **ignorés par Git** (`*.csv` dans `.gitignore`) car trop volumineux. Chaque membre doit les télécharger depuis Kaggle et les placer dans le dossier `data/`.

L'EDA utilise **3 fichiers** : `games.csv`, `game_events.csv` et `EloRatings.csv`.

### 1. Football Data from Transfermarkt (source principale)

Lien : https://www.kaggle.com/datasets/davidcariboo/player-scores

| Fichier | Contenu |
|---|---|
| `games.csv` | Tous les matchs (score, date, phase, stade, entraîneurs, arbitre…) — **une ligne = un match** |
| `game_events.csv` | Événements des matchs (buts, cartons, remplacements, tirs au but) — **une ligne = un événement** |
| `competitions.csv` | Liste des compétitions et leurs identifiants |
| `clubs.csv` | Informations sur les clubs |
| `club_games.csv` | Un match vu du côté de chaque club |
| `players.csv` | Informations sur les joueurs |
| `player_valuations.csv` | Historique des valeurs marchandes des joueurs |
| `appearances.csv` | Statistiques individuelles par match |
| `game_lineups.csv` | Compositions d'équipe |

`player_valuations.csv` servira pour la valeur des effectifs lors de la modélisation.

### 2. Club Football Match Data (2000-2025)

Lien : https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025

| Fichier | Contenu |
|---|---|
| `EloRatings.csv` | Classement Elo des clubs dans le temps (deux relevés par mois, jusqu'au 1er septembre 2026) — **une ligne = l'Elo d'un club à une date** |
| `Matches.csv` | Matchs de championnats nationaux avec Elo, forme, tirs et cotes (jusqu'au 3 septembre 2026) |

⚠️ `Matches.csv` ne contient **que des championnats nationaux**, aucun match de Ligue des Champions. Il servira à mesurer la forme des clubs en championnat.

### Filtrer la Ligue des Champions

Dans `games.csv`, la colonne `competition_id` identifie la compétition :

- `CL` : Ligue des Champions, 1 871 matchs (saisons 2012-13 à 2025-26) ;
- `CLQ` : qualifications, 1 232 matchs.

```python
import pandas as pd

games = pd.read_csv("../data/games.csv")
ucl = games[games["competition_id"] == "CL"].sort_values("date")
```

### Points d'attention (détaillés et traités dans l'EDA)

- `season` correspond à l'année de début de saison : `2025` = saison 2025-2026 ;
- La saison 2025-2026 s'arrête aux demi-finales (dernier match : 6 mai 2026) : la finale 2026 n'est pas dans les données ;
- **Aucun match de la saison 2026-2027 n'est dans les données** : la liste des 36 clubs qualifiés doit venir d'une source externe (site de l'UEFA) ;
- **8 scores incluent les tirs au but** (ex. Atlético–PSV 2016 noté 8-7 au lieu de 0-0) : ils sont détectés et corrigés dans l'EDA grâce à `game_events.csv` ;
- Les noms de phase ne sont pas uniformes (24 noms pour 7 phases) et `Group Stage` désigne la nouvelle phase de ligue depuis 2024 : une colonne `phase` harmonisée est créée dans l'EDA ;
- Les colonnes `home_club_position` / `away_club_position` sont vides pour tous les matchs de Ligue des Champions : un `dropna()` sur toute la table supprime donc **tous** les matchs de C1 ;
- Les noms de clubs diffèrent entre les datasets (`Paris Saint-Germain` / `Paris SG`) : une table de correspondance relie **84 clubs sur 106** aux Elo. Les 22 autres viennent de pays absents de la table Elo (Ukraine, Croatie, Suisse, Tchéquie…).

---

## ⚙️ Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/moussa197/CHAMPIONS-.git
   cd CHAMPIONS-
   ```
2. Créer (ou réutiliser) l'environnement virtuel `.venv` du projet, l'activer, puis y installer les dépendances (Python 3.10+). Utiliser `python -m pip` et `python -m jupyter` garantit que les paquets et le notebook passent par le Python du `.venv` et non par une autre installation (Anaconda, Python système) :
   ```bash
   python -m venv .venv
   # Windows : .venv\Scripts\activate    —    macOS / Linux : source .venv/bin/activate
   python -m pip install pandas numpy matplotlib jupyter scikit-learn xgboost
   ```
   Dans VS Code, choisir le noyau `.venv` en haut à droite du notebook.
3. Télécharger les datasets Kaggle et placer les CSV dans `data/`.
4. Lancer Jupyter et ouvrir `analyse/EDA_v2.ipynb` (après avoir exécuté les notebooks `v2_*`, voir « Ordre d'exécution ») :
   ```bash
   python -m jupyter notebook
   ```

Les notebooks lisent les données avec des chemins relatifs (`../data/games.csv`) : ils doivent être lancés depuis leur propre dossier.

---

## ▶️ Ordre d'exécution

Les CSV de `output/` ne sont pas commités : chacun les régénère en local, dans cet ordre.

1. `analyse/v2_elo.ipynb` → `output/elo_par_match.csv` (une ligne par match de C1) ;
2. `analyse/v2_joueurs.ipynb` → `output/joueurs_par_match.csv` (une ligne par équipe et par match) ;
3. `analyse/v2_fatigue.ipynb` → `output/fatigue_par_match.csv` (une ligne par club et par match) ;
4. `analyse/EDA_v2.ipynb` → `output/eda_matchs_c1.csv` (une ligne par match de C1, toutes les variables assemblées).

---

## 🔬 Méthodologie

1. **Analyse exploratoire** : structure des données, qualité (valeurs manquantes, doublons, scores faux, phases), répartition des résultats, avantage du terrain, taux de victoire, lien entre Elo et résultat.
2. **Nettoyage** : harmonisation des phases et des noms de clubs, conversion des dates, correction des scores avec tirs au but, gestion des valeurs manquantes.
3. **Feature engineering** : Elo avant chaque match, forme récente (championnat et Europe), buts marqués/encaissés, expérience en Ligue des Champions, valeur marchande de l'effectif, confrontations directes.
4. **Modélisation** : Random Forest et XGBoost pour prédire l'issue des matchs (victoire / nul / défaite), avec séparation chronologique entraînement / test (par exemple entraînement 2012-2022, test 2023-2025).
5. **Évaluation** : accuracy, précision / rappel / F1, matrice de confusion, log loss et calibration des probabilités, comparaison avec la règle « le meilleur Elo gagne » (56 % selon l'EDA v1 (Elo officiel), à recalculer sur la base de modélisation).
6. **Simulation** : simulation de la saison 2026-2027 (phase de ligue puis phase finale) à partir des probabilités de match pour obtenir la probabilité de titre de chaque club.

---

## 📈 Résultats de l'EDA

### EDA v1 (`analyse/EDA.ipynb`) — 1 871 matchs de C1 (2012-2026), après correction des 8 scores avec tirs au but

- **Résultats :** 47,6 % de victoires à domicile, 20,0 % de nuls, 32,4 % de victoires à l'extérieur ;
- **Avantage du terrain et public :** l'écart de buts domicile − extérieur tombe à **0,00** lors de la saison à huis clos (2020-2021, Covid), contre 0,2 à 0,6 les autres saisons ;
- **Elo :** la règle « le meilleur Elo gagne » a raison dans **56 %** des matchs, contre 48 % pour « l'équipe à domicile gagne ».

### EDA v2 (`analyse/v2_*.ipynb` et `analyse/EDA_v2.ipynb`)

- **Elo maison** (`v2_elo.ipynb`) : couvre **100 %** des matchs (contre 80 % pour l'Elo officiel) ; la règle « le meilleur Elo gagne » y atteint **56,0 %**. L'Elo championnat ne fait que **48,2 %** : il compare mal des clubs de pays différents ;
- **Lien avec le résultat (Spearman) :** écart de valeur d'effectif **0,446**, écart d'Elo global **0,416**, écart d'Elo C1 **0,404**, écart d'Elo championnat **0,22** ;
- **Elo par quintile :** le score moyen à domicile passe de **0,32** (écart d'Elo le plus défavorable) à **0,84** (écart le plus favorable) ;
- **Repos :** aucun lien net avec le résultat (**-0,06**) une fois retirés les clubs dont le championnat manque dans les données ; le lien de départ (-0,22) venait de ce biais ;
- **Variables redondantes :** Elo global / Elo championnat **0,79**, Elo global / Elo C1 **0,70**, Elo global / valeur d'effectif **0,67** : à ne pas utiliser ensemble sans précaution.

**Principales limites :** aucun match de 2026-2027, la rotation C1 n'est connue que pour 710 matchs sur 1 871, aucune donnée météo, et les corrélations ne contrôlent pas le niveau des équipes (corrélation ≠ causalité).

**Question à poursuivre :** un modèle fait-il mieux que la règle Elo (56 %), et la valeur d'effectif ajoute-t-elle quelque chose une fois l'Elo pris en compte ?
---

## 🚧 Avancement

- [x] Collecte des données
- [x] Analyse exploratoire (EDA v2 : assemblage des tableaux Elo, joueurs et fatigue fait, `output/eda_matchs_c1.csv`)
- [ ] Nettoyage et harmonisation (scores et phases déjà corrigés dans l'EDA)
- [ ] Feature engineering
- [ ] Entraînement des modèles
- [ ] Prédiction 2026-2027

---

## 🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Scikit-learn · XGBoost · Jupyter · Kaggle

---

## 👥 Auteurs

Maxime, Loan, Moussa, Mouhamedou et Ousmane — septembre 2026