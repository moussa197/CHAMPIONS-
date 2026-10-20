# Simulation 2026-2027 — Maxime

Branche : `feature/simulation`, créée depuis `develop` à jour puis enrichie de
`feature/model-baseline`, qui contient `MODEL_POISSON.ipynb` (première récupération : `65741bd`).
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
- `finale_c1_2026_uefa.csv` : complément UEFA versionné dans le notebook, utilisé seulement
  pour l'évaluation historique ; score de 1-1 séparé de la victoire parisienne aux tirs au but.

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
La finale du 30 mai 2026 est maintenant ajoutée depuis le
[compte rendu UEFA](https://fr.uefa.com/uefachampionsleague/news/02a5-20c0308e844d-63210800ee1f-1000--paris-conserve-la-champions-league-aux-tirs-au-but-face-a-ar/)
: Paris–Arsenal 1-1, puis 4-3 aux tirs au but. Le Brier du titre 2025-2026 vaut
0,0239 contre 0,0270 pour l'uniforme. Ce complément ne modifie ni les CSV historiques
du groupe ni l'apprentissage des backtests. Le modèle de production n'est pas
réentraîné avec la finale ; ses probabilités 2026-2027 restent donc inchangées.

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

## Mac, Windows et versions de Python

Les lectures de notebooks et l'écriture JSON précisent `encoding='utf-8'` : elles
ne dépendent plus de l'encodage cp1252 par défaut de certains Windows.
L'empreinte figée de `ast.dump` a été retirée : le même code n'a pas la même
représentation sous Python 3.12 et 3.13. Le nouveau contrôle lit les fonctions
dans le même interpréteur et compare leurs résultats sur 282 cas, y compris
les scores tirés avec des graines identiques. Une vraie divergence numérique est refusée.

Depuis la racine, lancer les régressions avec `python3 -m unittest discover -s tests -v`
sur Mac, ou `py -m unittest discover -s tests -v` sous Windows. Le test de lecture
émule cp1252 et les fins de ligne Windows. Les tests de structure ont été exécutés
sous Python 3.12 et 3.13 ; les tests numériques et le notebook complet sous Python 3.12.
Une exécution native Windows et des versions identiques de NumPy/scikit-learn restent
utiles pour comparer les résultats exacts entre les postes.

## Étude météo

`ETUDE_METEO.ipynb` vérifie les fichiers locaux et produit `output/faisabilite_meteo.csv`.
Le constat actuel est : 1 871 matchs sans heure de coup d'envoi exploitable,
130 noms de stades distincts, aucune coordonnée ni météo dans les tables inspectées.
Une intégration sérieuse exige une collecte des lieux réels, heures UTC et prévisions
archivées dont la disponibilité est prouvée avant le match. Le notebook prépare le
schéma de collecte et refuse réanalyses, prévisions publiées trop tard, fuseaux absents
et doublons ; les tests utilisent des cas artificiels sans les exporter.

Aucun gain météo n'est encore mesuré. Les prévisions peuvent enrichir un match à
courte échéance, mais ne donnent pas la météo de toute une saison prévue en septembre.
Il faudrait alors des scénarios ou une climatologie connus à la date de prévision.
Le modèle actuel conserve son contrat et ses coefficients tant qu'une validation
chronologique sur les mêmes matchs n'a pas établi un gain.
