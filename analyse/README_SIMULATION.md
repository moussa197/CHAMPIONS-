# Simulation 2026-2027 — Maxime

Branche : `feature/simulation`, créée depuis `develop` à jour puis enrichie de
`feature/model-baseline`, qui contient `MODEL_POISSON.ipynb` v2 (commit `65741bd`).
Les notebooks des autres membres ne sont pas modifiés.

## Lancer

1. Générer les entrées en exécutant, depuis `analyse/`, les notebooks
   `v2_elo`, `v2_joueurs`, `v2_fatigue`, `v2_participants`, `EDA_v2`, puis `MODEL_POISSON`.
2. Ouvrir `SIMULATION_2026.ipynb`, puis **Kernel → Restart & Run All**.

La simulation utilise `numpy`, `pandas`, `matplotlib`, `joblib` et `scikit-learn`.
Le notebook du modèle nécessite aussi `statsmodels`.
La graine est 2026 : 10 000 saisons cibles, 10 000 par backtest, puis une comparaison
avec l'Elo figé. Les sorties ne sont pas versionnées ; chacun les régénère localement.

## Livrables dans output/

- `resultats_simulation_2026.csv` : 36 clubs, points moyens, probabilités de chaque tour
  et marges de Monte-Carlo. `p_…` est entre 0 et 1 ; `pct_…` entre 0 et 100.
- `finales_probables_2026.csv` : les paires de finalistes et leurs fréquences.
- `seuils_simulation_2026.csv` : distribution des points du 8e et du 24e.
- `backtest_simulation_2026.csv` : Brier par tour et nombre de quart-de-finalistes
  parmi les huit favoris pour 2024-2025 et 2025-2026.
- `simulation_2026_metadata.json` : graine, paramètres, provenance et empreintes des entrées.

L'application peut joindre les résultats avec ses clubs via `club_id`.
`pct_barrages` signifie finir entre la 9e et la 24e place ; `pct_huitiemes` inclut
les qualifiés directs et les vainqueurs des barrages.

## Résultats de l'exécution de référence

Avec les fichiers locaux utilisés le 20 octobre 2026 : Arsenal 23,30 % de titre,
Manchester City 14,96 %, Bayern 12,40 %. La marge normale de Monte-Carlo du favori
est d'environ ±0,83 point de pourcentage. Paris obtient 37,37 % de top 8.
Ces chiffres sont calculés, pas recopiés depuis les repères du document.

Le backtest retrouve 6 des 8 quart-de-finalistes sur chacune des deux saisons.
Le Brier des quarts vaut 0,1158 puis 0,0993, contre 0,1728 pour l'uniforme.
Le Brier du titre 2024-2025 est moins bon que la baseline : 0,0301 contre 0,0270.
Le titre 2025-2026 n'est pas évalué car la finale manque ; les finalistes sont
retrouvés à partir des demi-finales, sans inventer la finale.

## Limites à expliquer

Les règles suivent les consignes du groupe : dernier départage aléatoire et retour
chez le club au meilleur rang de ligue à chaque tour. Le notebook cite les différences
avec les articles 18 et 19 UEFA ; ce n'est pas une reproduction exacte du règlement.

Les modèles des backtests sont réentraînés sur le passé. Les 81 lignes club-match
recalées avec toute leur saison sont corrigées dans une copie de travail, avec les
seules observations de dates antérieures. Les constantes Elo du groupe et les choix
de modèle restent rétrospectifs : deux saisons ne constituent pas un test aveugle
indépendant. L'erreur de Monte-Carlo ne mesure pas l'erreur du modèle.

Les blessures, transferts et matchs nationaux futurs ne sont pas simulés. Les buts
des équipes sont indépendants ; la prolongation conserve une cadence de buts
proportionnelle au temps. Les tests du notebook tentent de casser les règles de
classement, le tableau, les prolongations, les tirs au but et les mises à jour Elo.
