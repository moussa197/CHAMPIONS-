# Prompts réellement utilisés — Maxime, simulation

## 1. Demande du 20 octobre 2026

Résumé de la demande initiale : réaliser la partie simulation attribuée à Maxime
dans le projet CHAMPIONS-, récupérer le modèle de scores du groupe et créer
la branche `feature/simulation`.

Pièces fournies : `consignes_equipe.md` et deux captures précisant la répartition du
travail, la branche `feature/simulation` et le livrable `SIMULATION_2026.ipynb`.
Les consignes techniques du bloc Maxime servent de cahier des charges ; elles ne
constituent pas des résultats à recopier.

L'IA a proposé un moteur de scores de Poisson, un classement et un tableau,
des mises à jour Elo, 10 000 répétitions et deux backtests chronologiques.
Elle a récupéré les notebooks du modèle depuis `feature/model-baseline`, branche
réellement disponible, sans modifier leurs sources.

Vérifications exécutées : identité des quatre fonctions avec celles du modèle,
144 matchs et 36 clubs, 16/8/4/2/1 qualifiés, somme des titres à 100 %, remise à zéro
des Elo, neutralité de la finale, absence de règle des buts à l'extérieur, une seule
mise à jour Elo pour un match avec prolongation et reproductibilité par graine.
Les résultats ont été recalculés à partir des fichiers locaux.

Corrections retenues : recalage historique des Elo uniquement avec des observations
antérieures, aucune vérité de titre ajoutée pour la finale 2026 manquante, bornes de
Wilson pour ne pas présenter zéro titre observé comme une impossibilité.
La formule et les contrôles de durée ont aussi été relus et corrigés.

## 2. Choix explicite des règles

Après le signalement de différences entre les consignes et le règlement UEFA,
réponse réellement donnée par Maxime :

> Suivre les consignes du groupe, avec les simplifications expliquées

L'IA a conservé le départage final au hasard et le retour chez le meilleur rang
de ligue à tous les tours, en documentant les écarts aux articles 18 et 19 UEFA.
Elle n'a pas ajusté les probabilités pour retrouver les ordres de grandeur fournis.

Le bilan conserve les contre-exemples : le Brier du titre 2024-2025 est moins bon
que celui de la baseline uniforme. Il reste à faire relire ces choix et leurs limites
par un autre membre du groupe ; ces contrôles automatiques ne remplacent pas cette relecture.

## 3. Complément finale, météo et reproductibilité

Résumé de la demande suivante du 20 octobre 2026 : compléter la finale de l'année,
examiner l'intégration de la météo et résoudre les deux blocages remontés par un collègue
sous Windows (décodage cp1252 et comparaison des fonctions du modèle).

Vérifications : la finale Paris–Arsenal du 30 mai 2026 est confirmée par l'UEFA ; le score
1-1 et la victoire 4-3 aux tirs au but sont séparés. La vérité de titre complète le backtest,
sans ajouter la finale à son apprentissage. L'empreinte du même code a été comparée sous
Python 3.12 et 3.13 : elle diffère à cause de la représentation de l'AST. Le contrôle
numérique est désormais effectué dans l'environnement courant, sur 282 cas.

Les lectures/écritures de texte précisent UTF-8. Des régressions vérifient cp1252 simulé,
les fins de ligne Windows et les divergences réelles de calcul. La météo fait l'objet
d'un diagnostic local et d'une recherche dans les documentations Open-Meteo. Les tables
ne fournissent pas les heures et coordonnées nécessaires ; aucun coefficient météo ni
gain de performance n'est inventé. Les prévisions réellement disponibles avant le match
devront être collectées et évaluées avant de changer le modèle du groupe.
