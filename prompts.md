# Prompts utilisés par le groupe

Livrable demandé par le prof. Pour chaque prompt : **ce que l'IA a proposé**, **ce qu'on a vérifié nous-mêmes**, **ce qu'on a corrigé ou refusé**.

Règle du groupe : l'IA propose, nous vérifions. Aucun chiffre de ce dépôt n'est repris d'une réponse de l'IA sans avoir été recalculé dans un notebook qui tourne de bout en bout.

| # | Auteur / contribution | Partie | Prompt |
|---|---|---|---|
| 1 | Moussa | modèle | Diagnostic avant de coder |
| 2 | Moussa | modèle | Un pipeline vérifiable |
| 3 | Moussa | modèle | L'IA critique son propre résultat |
| 4 | Loan | forme et audit | Le cadre du projet et le diagnostic avant de coder |
| 5 | Loan | forme et audit | Construire la forme récente et auditer le modèle |
| 6 | Loan | forme et audit | L'IA critique son propre audit |
| 7 | Maxime | simulation | La demande du 20 octobre 2026 |
| 8 | Maxime | simulation | Le choix explicite des règles |
| 9 | Maxime | simulation | Complément : finale, météo et reproductibilité |
| 10 | Barryflex88 — analyse | analyse des joueurs | Exemple de formulation ajouté après l'analyse |
| 11 | Barryflex88 — analyse | analyse de la fatigue | Exemple de formulation ajouté après l'analyse |
| 12 | Barryflex88 — analyse | modèle v3 | Exemple de formulation ajouté après l'analyse : forces et joueurs récents |
| 13 | Barryflex88 — analyse | modèle v3 | Exemple de formulation ajouté après l'analyse : comparaison aux bookmakers |
| 14 | Barryflex88 — analyse | simulation et application | Exemple de formulation ajouté après l'analyse : intégration du modèle v3 |

---

## Partie modèle (Moussa)

*Reprend intégralement `prompts_moussa.md`, supprimé de la racine une fois intégré ici.*

### Prompt 1 — Le diagnostic avant de coder (diapo 24 du cours)
> « Tu es mon reviewer ML. Voici : question, grain, cible, colonnes, EDA, date de prédiction. 1. La cible est-elle définie ? 2. Quelles features sont disponibles à temps ? 3. Quelles fuites ou biais possibles ? 4. Quel split et quelle baseline ? 5. Quelle métrique et pourquoi ? Pose 3 questions bloquantes avant de coder. »

**Ce que l'IA a relevé :**
- la cible (les buts) mélange 90 et 120 minutes, à cause des prolongations ;
- une petite fuite possible dans l'Elo « maison recalé » (droite ajustée sur toute la saison) ;
- le risque le plus sérieux : le test 2023-2025 avait déjà servi à décider d'ajouter la variable « nouveau format ».

**Ses 3 questions bloquantes, et nos réponses :**
1. Forme récente : attendre Loan ? → Non : une section qui se branche quand son fichier arrive, et un pré-test rapide en attendant.
2. Le test déjà « vu » ? → On le garde pour le modèle choisi avant (Elo + terrain), on justifie le format uniquement par le test vers l'avant (2025-2026), et on l'écrit comme limite.
3. Forcer des nuls dans la matrice de confusion ? → Non : on garde la règle « issue la plus probable », car la simulation n'utilise que les probabilités.

**Ce qu'on a vérifié nous-mêmes :**
- les prolongations : **21 matchs sur 1 871 (1,1 %)**, 22 buts sur 5 735. L'effet est négligeable, et il est écrit comme limite ;
- l'Elo recalé : **81 lignes club-match sur 3 742 (2,2 %)**. Chiffre retrouvé de façon indépendante dans `AUDIT_MODELE.ipynb` (Loan).

### Prompt 2 — Un pipeline vérifiable (diapo 25)
> « Génère un notebook scikit-learn minimal : split adapté, pas de fuite, preprocessing en Pipeline ; baseline Dummy, modèle simple, random_state fixé ; métriques adaptées et tableau comparatif ; cellules Markdown qui expliquent chaque étape ; aucun téléchargement externe ni donnée inventée. »

**Ce que l'IA a produit :** `MODEL_POISSON.ipynb` v2, avec un `Pipeline` (`SimpleImputer` + `PoissonRegressor`), un split chronologique, `DummyClassifier` et `DummyRegressor`, et comme métriques la log loss, le RPS, l'accuracy, la balanced accuracy, la matrice de confusion et la MAE des buts.

**Ce qu'on a vérifié :**
- **Le Pipeline donne le même modèle que statsmodels.** Un `assert` compare les coefficients. Il a d'abord **échoué** : avec ses réglages par défaut, `PoissonRegressor` s'arrête trop tôt (écart de 0,0002). Corrigé avec `solver='newton-cholesky'` et `tol=1e-10` : l'écart tombe à 10⁻¹¹ ;
- **les fonctions exportées** (`probas_match`, `tirer_score`) donnent les mêmes probabilités que le Pipeline (à 10⁻⁹ près), et `tirer_score` retrouve les bons buts moyens sur 100 000 tirages ;
- **tous les chiffres de la v1 sont retrouvés à l'identique** avec le Pipeline.

**Ce qu'on a corrigé :** des phrases causales (« le nouveau format marque plus de buts ») remplacées par des observations (« depuis le nouveau format, on observe plus de buts »).

### Prompt 3 — L'IA critique son propre résultat (diapo 26)
> « Agis comme un auditeur hostile. Cherche : data leakage et cible mal définie, train/test contaminés, métrique trompeuse, classes déséquilibrées, faible taille d'échantillon, features non disponibles en production. Propose 3 tests concrets pour réfuter le modèle. »

**Les tests, et ce qu'ils ont donné (test 2023-2025, modèle Elo + terrain) :**

| Test | Résultat | Conclusion |
|---|---|---|
| **Cible mélangée** : on mélange les scores de l'apprentissage | log loss 1,026 ± 0,011, contre 1,020 pour le Dummy (0,902 pour le vrai modèle) | ✅ sans le vrai lien entre l'Elo et les scores, le modèle ne vaut plus rien : pas de fuite par la structure des données |
| **Retirer l'Elo « maison recalé »** (la petite fuite) | log loss 0,9050 dans les deux cas, avec ou sans ces matchs à l'apprentissage | ✅ la fuite n'a aucun effet mesurable |
| **Sous-groupes** | gain sur le Dummy : +0,14 en phase de groupes, seulement +0,03 en élimination directe | ⚠️ voir ci-dessous |
| **Calibration des favoris** | annoncé à 65 %, il gagne 75 % ; à 74 %, 77 % ; à 87 %, 91 % | ⚠️ sur 2023-2025, le modèle est un peu **trop prudent** avec les favoris |

**Ce que les tests ont révélé :**
1. **Le modèle paraît moins bon en élimination directe.** On a vérifié pourquoi : à écart d'Elo égal (moins de 150 points), son gain est le même en phase de groupes (+0,029) et en élimination directe (+0,027). Les matchs à élimination directe sont simplement plus **serrés** (113 points d'écart moyen, contre 149) : ils sont plus durs à prévoir pour n'importe quel modèle. Ce n'est pas un défaut propre au modèle, mais c'est important, car ce sont ces matchs qui décident du titre.
2. **Les favoris gagnent plus que prévu sur 2023-2025.** C'est cohérent avec le nouveau format (plus de buts, moins de nuls, donc des résultats plus tranchés), que la variable « format » corrige en partie. Attention : à l'échelle du tournoi, le backtest montrait l'inverse (un peu trop confiant sur les grands favoris). Les deux ne se contredisent pas : un match isolé et un parcours de 4 tours ne posent pas la même question.
3. **Les matchs avec un Elo estimé sont les mieux prédits** (gain de +0,34), parce que ce sont souvent des matchs très déséquilibrés contre de petits clubs. L'estimation ne dégrade donc pas les prédictions.

**Relecture indépendante (Loan, `AUDIT_MODELE.ipynb`) :**
- **Confirmé :** la cible mélangée (1,026), l'Elo recalé sans effet (0,905), et le gain identique à moins de 150 points d'écart (+0,029 contre +0,027) ;
- **Nuancé :** au-delà de 150 points, le gain en élimination directe (+0,048) ne repose que sur 40 matchs, avec un intervalle de −0,168 à +0,240 : on ne peut rien conclure ;
- **Contesté :** le point 3. À écart d'Elo égal, les matchs à Elo estimé restent bien mieux prédits (0,380 contre 0,840 au-delà de 300 points). Le déséquilibre n'explique donc qu'une partie de l'écart, et le point reste ouvert.

---

## Partie forme récente et audit (Loan)

Outil : Claude Code (Claude Opus 5.5), dans le dépôt, sur la branche `feature/forme-audit`.

### Prompt 4 — Le cadre du projet et le diagnostic avant de coder
> « Tu m'aides sur un projet de data science en groupe (ESEO, cours « Python for Data Science ») […] LES RÈGLES À RESPECTER : 1. AUCUNE FUITE […] 2. Split CHRONOLOGIQUE […] 3. Toujours une BASELINE chiffrée […] 11. Je ne modifie QUE mes fichiers […] 12. Toi, l'IA : avant de coder, fais-moi un diagnostic (cible ? variables disponibles à temps ? fuites possibles ? split ? baseline ? métrique ?) et pose-moi 3 questions bloquantes. Après le code, critique ton propre travail comme un auditeur hostile […] et propose 3 tests pour le réfuter. »

(Prompt complet : le contexte du projet, les fichiers utiles, le modèle, les règles et mes trois tâches.)

**Ce que l'IA a proposé (diagnostic) :**
- nos fichiers `output/` locaux étaient **périmés** (pas de colonne `ecart_elo_complet`, pas de calendrier) : il fallait relancer toute la chaîne ;
- les **tirs au but** ne faussent pas seulement la C1 : **1 213 autres scores** de `games.csv` (coupes, supercoupes, qualifications) incluent les tirs au but. Sans correction, la forme compterait des nuls comme des victoires ;
- un **biais de couverture** : pour les clubs dont le championnat manque dans `games.csv`, les « 5 derniers matchs » ne sont que des matchs de coupe ou d'Europe ;
- trois points suspects dans le modèle, repérés à la lecture du code : le recalage de l'Elo (fuite de principe), la réutilisation du test 2023-2025 pour la variable « nouveau format », et les réglages K et H de l'Elo maison choisis sur toutes les saisons.

**Ses 3 questions bloquantes, et nos réponses :**
1. Régénérer les sorties en exécutant les notebooks des autres ? → Oui, **en local seulement** : les copies exécutées vont dans un dossier temporaire, les `.ipynb` des autres ne sont pas modifiés.
2. Que faire des clubs sans championnat dans les données ? → On ajoute une **fenêtre de 60 jours** : seuls les matchs des 60 jours avant le match comptent.
3. L'audit doit-il refaire le test de la forme ? → **Non** : le test est déjà dans `MODEL_POISSON` (section 8), avec une règle fixée à l'avance. L'audit le **recalcule** et vérifie que la règle a été respectée. La décision est prise à un seul endroit et vérifiée par quelqu'un d'autre.

**Ce qu'on a vérifié nous-mêmes :**
- [x] les 3 exemples à la main de `v2_forme.ipynb`, section 3 : Benfica 2,20, Porto 2,50 sur 4 matchs, PSG 2,60 (la Supercoupe d'Europe contre Tottenham, notée 6-5 dans `games.csv`, compte bien 1 point). Les matchs utilisés et leurs dates sont affichés, et un `assert` recalcule chaque moyenne ;
- [x] qu'aucun match utilisé n'est joué le jour même ou après : `assert` sur les 3 742 lignes club-match ;
- [x] que la régénération ne touche à aucun fichier versionné des autres : `git status` vide après l'exécution des 6 notebooks.

**Ce qu'on a refusé :** l'IA ne trouvait pas la section 8 de `MODEL_POISSON` dans la version poussée. Plutôt que de deviner son contenu, on a attendu que Moussa la pousse (commit `65741bd`), puis on l'a auditée telle quelle.

### Prompt 5 — Construire la forme récente et auditer le modèle
> « TÂCHE 1 : analyse/v2_forme.ipynb → output/forme_par_match.csv […] points moyens par match (victoire 3, nul 1, défaite 0) sur ses 5 derniers matchs officiels, toutes compétitions, joués STRICTEMENT avant la date du match […] TÂCHE 2 : audit « équipe rouge » de MODEL_POISSON.ipynb → analyse/AUDIT_MODELE.ipynb. Le but n'est pas d'être gentil : c'est d'essayer de casser le modèle, avec au moins 3 tests chiffrés. »

**Ce que l'IA a produit :** `v2_forme.ipynb` (forme, 3 exemples, `assert` anti-fuite, corrélations) et `AUDIT_MODELE.ipynb` (5 tests et l'audit de la forme).

**Ce qu'on a vérifié :**
- l'audit **reproduit à l'identique** les chiffres de Moussa (`assert`) : glissante 0,9148, test 0,902, Dummy 1,020, et pour la forme 0,9148 / 0,9156 sur 4 saisons sur 8, d'où le rejet ;
- [x] chaque chiffre écrit dans une « Lecture » apparaît dans une sortie de cellule. Contrôle automatique sur les deux notebooks : les seuls nombres absents des sorties sont des années, un numéro de commit, et deux valeurs calculées à partir des sorties (9,5 % = 5,8 % + 3,7 % ; 99 % = 100 % − 1,0 %).

**Ce qu'on a corrigé (erreurs de l'IA détectées pendant le travail) :**
1. **Test de la cible mélangée mal posé.** Le premier `assert` exigeait que le modèle mélangé ne batte jamais le Dummy, et il a échoué (1,008 < 1,020). Ce n'était pas une fuite : mélanger les scores garde l'avantage du terrain, donc la bonne référence est le modèle « terrain seul » (1,028). La référence a été corrigée, et l'erreur est expliquée dans le notebook.
2. **Comparaison faussée sur l'Elo recalé.** La première version comparait des log loss calculées sur des matchs différents (0,9148 contre 0,9210). Corrigé : les deux modèles sont jugés sur les mêmes matchs (0,9207 contre 0,9210).
3. **Un sous-groupe trop beau.** Le modèle est meilleur sur les matchs à Elo estimé. Plutôt que d'accepter l'explication la plus simple (des matchs déséquilibrés), on a comparé à écart d'Elo égal : l'écart demeure, et le point est classé « suspect, non résolu ».

### Prompt 6 — L'IA critique son propre audit
> « Après le code, critique ton propre travail comme un auditeur hostile (fuite, métrique trompeuse, petit échantillon) et propose 3 tests pour le réfuter. »

**Les critiques de l'IA sur son propre travail :**
- **Petits échantillons :** les conclusions les plus fortes de l'audit (élimination directe : 118 matchs ; Elo estimé : 15 à 32 matchs par case) reposent sur peu de matchs. Les intervalles sont donnés, mais un seul découpage différent pourrait changer le verdict.
- **Test réutilisé :** l'audit utilise lui-même le test 2023-2025 plusieurs fois. C'est acceptable pour un diagnostic, à condition qu'**aucun choix de modèle** n'en découle.
- **Test de l'Elo officiel imparfait :** les relevés sont retrouvés par leur valeur, pas par le nom du club.
- **Une seule définition de la forme** (5 matchs, 60 jours) : le rejet vaut pour cette définition, pas pour toute idée de « forme ».

**Les 3 tests proposés pour réfuter l'audit (à faire) :**
1. **Elo estimé :** recalculer l'Elo complet avec une droite de recalage ajustée sur les **saisons précédentes** seulement, puis refaire le contrôle à écart égal. Si l'écart disparaît, c'était la fuite ; sinon, c'est une qualité de l'estimation.
2. **Élimination directe :** refaire le découpage par phase sur la **validation glissante 2015-2022** (226 matchs à élimination directe au lieu de 118). Si le gain y devient net, notre conclusion « non démontré » venait du petit échantillon.
3. **Elo officiel :** refaire la vérification **par club**, avec la table de correspondance des noms de `v2_elo.ipynb`, au lieu de chercher par valeur.

---

## Partie simulation (Maxime)

*Reprend intégralement `prompts_maxime.md`, supprimé de la racine une fois intégré ici.*

### Prompt 7 — La demande du 20 octobre 2026
**La demande (résumée par Maxime) :** réaliser la partie simulation attribuée à Maxime dans le projet CHAMPIONS-, récupérer le modèle de scores du groupe et créer la branche `feature/simulation`.

Pièces fournies : `consignes_equipe.md` et deux captures précisant la répartition du travail, la branche `feature/simulation` et le livrable `SIMULATION_2026.ipynb`. Les consignes techniques du bloc Maxime servent de cahier des charges ; ce ne sont pas des résultats à recopier.

**Ce que l'IA a proposé :** un moteur de scores de Poisson, un classement et un tableau, des mises à jour de l'Elo, 10 000 répétitions et deux backtests chronologiques. Elle a récupéré les notebooks du modèle depuis `feature/model-baseline`, la branche réellement disponible, sans modifier leurs sources.

**Ce qu'on a vérifié (contrôles exécutés) :**
- les quatre fonctions sont identiques à celles du modèle ;
- 144 matchs et 36 clubs ; 16, 8, 4, 2 puis 1 qualifiés ; la somme des titres fait 100 % ;
- la remise à zéro des Elo, la neutralité de la finale, l'absence de règle des buts à l'extérieur ;
- une seule mise à jour de l'Elo pour un match avec prolongation ;
- la reproductibilité par graine.

Les résultats ont été recalculés à partir des fichiers locaux.

**Ce qu'on a corrigé :**
- le recalage historique des Elo n'utilise que des observations **antérieures** ;
- aucune « vérité » de titre n'est ajoutée pour la finale 2026, qui manque dans les données ;
- des bornes de Wilson, pour ne pas présenter « zéro titre observé » comme une impossibilité ;
- la formule et les contrôles de durée ont aussi été relus et corrigés.

### Prompt 8 — Le choix explicite des règles
L'IA a signalé des différences entre les consignes du groupe et le règlement de l'UEFA. Réponse réellement donnée par Maxime :

> « Suivre les consignes du groupe, avec les simplifications expliquées »

**Ce que l'IA a fait :** elle a conservé le départage final au hasard et le retour chez le meilleur rang de la phase de ligue à tous les tours, en documentant les écarts aux articles 18 et 19 du règlement UEFA. Elle n'a pas ajusté les probabilités pour retrouver les ordres de grandeur fournis.

**Ce qui reste à faire :** le bilan garde les contre-exemples : pour le titre 2024-2025, le score de Brier est moins bon que celui de la baseline uniforme. Ces choix et leurs limites doivent encore être **relus par un autre membre du groupe** : les contrôles automatiques ne remplacent pas cette relecture.

### Prompt 9 — Complément : finale, météo et reproductibilité
**La demande (résumée par Maxime), le 20 octobre 2026 :** compléter la finale de l'année, examiner l'intégration de la météo, et résoudre les deux blocages remontés par un membre du groupe sous Windows (décodage cp1252 et comparaison des fonctions du modèle).

**Ce qu'on a vérifié :**
- la finale Paris–Arsenal du 30 mai 2026 est confirmée par l'UEFA. Le score de 1-1 et la victoire 4-3 aux tirs au but sont séparés. Cette finale complète la vérité de titre du backtest, sans être ajoutée à son apprentissage ;
- l'empreinte du même code a été comparée sous Python 3.12 et 3.13 : elle diffère à cause de la représentation de l'AST. Le contrôle numérique est désormais fait dans l'environnement courant, sur 282 cas ;
- les lectures et écritures de texte précisent l'UTF-8. Des tests de régression vérifient l'encodage cp1252 simulé, les fins de ligne Windows et les vraies divergences de calcul ;
- **relecture indépendante (Loan) :** `SIMULATION_2026.ipynb` tourne de bout en bout sous Windows avec Python 3.13, sans modification, et retrouve les chiffres publiés (Arsenal 23,30 %, Man City 14,96 %, Bayern 12,40 %).

**Ce qu'on a refusé d'inventer : la météo.** Elle a fait l'objet d'un diagnostic local et d'une recherche dans les documentations Open-Meteo. Nos tables ne donnent pas les heures ni les coordonnées des matchs : aucun coefficient météo ni gain de performance n'est inventé. Les prévisions réellement disponibles avant le match devront être collectées et évaluées avant de changer le modèle du groupe.

---

## Analyse et documentation rétrospectives

Cette partie présente le travail de documentation réalisé après analyse des notebooks
et de l'application du projet. Barryflex88 a précisé ne pas avoir rédigé de prompts
avant sa demande d'aide. Les formulations ci-dessous sont donc des exemples préparés
après l'analyse pour décrire les sujets et les consignes utiles ; elles ne sont pas des
messages réellement envoyés à une IA pendant le développement.

La contribution de Barryflex88 dans cette partie est cette analyse documentaire et sa
synthèse rétrospective ; elle porte sur le travail déjà présent dans le dépôt.

Les prompts 1 à 9 figuraient déjà dans ce document, mais leur historique n'a pas été
vérifié dans le cadre de cette contribution. On ne tire aucune conclusion ici sur les
prompts éventuellement utilisés par les autres membres. Les faits et résultats
résumés plus bas proviennent des notebooks, du README et de l'application versionnés ;
ils décrivent le contenu analysé, sans attribuer à Barryflex88 les développements
initiaux réalisés par le groupe.

### Exemple 10 — Analyse des joueurs et de l'effectif

**Exemple de formulation rédigé après l'analyse de `analyse/v2_joueurs.ipynb` :**

> « Analyse les joueurs et les effectifs des clubs de Ligue des champions à partir des
> CSV présents dans `data/`. Avant de coder, vérifie le grain des tables, les colonnes,
> la couverture des données et les risques de fuite temporelle. Construis un notebook
> reproductible qui étudie la valeur pré-match des joueurs, l'âge, la rotation et la
> dépendance au meilleur buteur. Recalcule les buts à partir des événements `Goals`
> afin d'exclure les tirs au but. Pour toute variable utilisée dans une prédiction,
> n'utilise que des informations connues avant le match : valeurs de joueurs datées
> avant le coup d'envoi et statistiques de buts de la saison précédente. Présente
> question, résultats calculés, interprétation et limites ; exporte un CSV au grain
> équipe-match. N'invente ni données ni résultats et indique les fichiers manquants. »

**Ce que le notebook documente (sans historique des propositions de l'IA) :**
`v2_joueurs.ipynb` recalcule le score à partir des
événements pour ne pas compter les tirs au but comme des buts du match, puis étudie
valeur pré-match, âge, rotation et dépendance au meilleur buteur. Le livrable est
`output/joueurs_par_match.csv`, une ligne par équipe et par match.

**Précisions visibles dans le notebook :**
- les événements `Goals` permettent de vérifier le score sans tirs au but ; la sortie
  du notebook indique 1 871 matchs de C1 vérifiables, dont 8 avec séance de tirs au but,
  et 8 scores corrigés ;
- les variables de rotation et de dépendance au meilleur buteur sont calculées avec
  la saison précédente, pas avec la fin de la saison en cours ;
- `player_valuations.csv` est une entrée nécessaire et non versionnée : le notebook
  explique comment la fournir localement au lieu de fabriquer des valeurs manquantes ;
- les associations observées ne sont pas présentées comme des effets causaux.

### Exemple 11 — Analyse de la fatigue et de la couverture des calendriers

**Exemple de formulation rédigé après l'analyse de `analyse/v2_fatigue.ipynb` :**

> « À partir de `games.csv` et des Elo publiés avant chaque match, étudie si le repos
> entre deux matchs est associé aux résultats des clubs en Ligue des champions, une
> fois le niveau des adversaires pris en compte. Définis précisément les jours de repos
> et la charge de matchs sur les 30 jours précédents ; exclue le match étudié et ne
> considère pas comme antérieur un match le même jour si l'ordre n'est pas connu.
> Détecte les saisons où la couverture des matchs de championnat manque et n'interprète
> pas une absence de match dans un calendrier incomplet comme du repos. Distingue les
> indicateurs pré-match des descriptions de saison, calcule les résultats dans un
> notebook reproductible et expose les limites des données, notamment le fait que les
> minutes jouées par les joueurs ne sont pas observées. »

**Ce que le notebook documente (sans historique des propositions de l'IA) :** les jours de repos sont calculés depuis le dernier
match antérieur visible dans `games.csv` ; la charge compte les matchs des 30 jours
avant la rencontre. L'analyse de repos est limitée aux clubs dont la saison comporte
au moins un match `domestic_league` dans les données, avec l'Elo connu avant le match.

**Précisions visibles dans le notebook :**
- un match disputé le même jour n'est pas supposé antérieur, faute d'heure ou d'ordre
  fiable ;
- une compétition par saison est décrite comme une information de saison complète,
  donc non disponible au début de cette saison ;
- la présence d'un match de championnat indique une couverture détectée, mais ne
  prouve pas que tout le calendrier est complet ; les clubs sans couverture détectée
  sont exclus de l'analyse de repos ;
- les indicateurs concernent le club, et ne mesurent ni les minutes jouées ni la
  fatigue individuelle.

## Exemples de formulation après analyse des travaux v3

### Exemple 12 — Forces d'attaque/défense et valeur des joueurs récents

**Exemple de formulation rédigé après l'analyse de `analyse/v2_forces.ipynb` et
`analyse/MODEL_POISSON.ipynb` :**

> « Fais évoluer le modèle de buts au-delà de l'Elo sans fuite temporelle. Estime une
> force d'attaque et une force de défense par club à partir des matchs de clubs récents
> de toutes compétitions, avec une fenêtre de trois ans, et ne conserve pour une
> rencontre que les matchs antérieurs au premier jour de son mois. Construis aussi une
> mesure de valeur de l'effectif à partir des 11 joueurs les plus utilisés sur les
> 5 derniers matchs du club et de leur dernière valorisation connue avant la rencontre.
> Décris le sens des variables, les données manquantes et les limites ; compare les
> ajouts à une baseline selon une validation chronologique fixée avant le test. Exporte
> les features historiques au grain match et celles des participants au format de
> saison. N'affirme aucun gain avant exécution et comparaison reproductible. »

**Ce que les livrables documentent (sans historique des propositions de l'IA) :**
`v2_forces.ipynb` estime l'attaque et la défense
sur les matchs de clubs des trois années précédentes ; le terme de match combine
l'attaque de l'équipe et la défense adverse. La valeur récente est fondée sur les 11
joueurs les plus utilisés lors des 5 derniers matchs, avec des valorisations antérieures
au match. Le modèle de scores v3 ajoute ces informations à l'Elo, au terrain et au
format.

**Précisions visibles dans les notebooks :**
- les dates de coupure empêchent l'utilisation de matchs ou de valeurs postérieurs à
  la date de prédiction ;
- le notebook signale que les données s'arrêtent au début de juillet 2026 : les
  participants de 2026-2027 n'intègrent donc pas les transferts de l'été ;
- les clubs ayant peu de matchs ou des matchs récents trop anciens sont explicitement
  signalés comme ayant des informations fragiles ;
- les features retenues et les métriques de modèle sont contrôlées par validation
  chronologique, plutôt que déduites d'une simple corrélation.

### Exemple 13 — Comparaison aux bookmakers

**Exemple de formulation rédigé après l'analyse de la section 10 bis de
`analyse/MODEL_POISSON.ipynb` :**

> « Compare les probabilités 1X2 du modèle aux cotes de clôture de bookmakers pour les
> matchs de Ligue des champions communs. Relie les matchs avec une méthode explicite,
> retire la marge des cotes, et compare les modèles sur exactement les mêmes rencontres.
> Choisis tout poids d'un éventuel mélange sur les saisons d'apprentissage seulement,
> puis évalue-le sur le test chronologique 2023-2025 sans le régler sur ce test. Rapporte
> l'accuracy et la log loss, l'incertitude de la différence, le nombre de matchs reliés,
> les limites de couverture et l'absence de cotes futures pour une simulation de saison.
> Si les fichiers de cotes facultatifs sont absents, explique que l'analyse est ignorée ;
> ne fabrique ni cote ni résultat. »

**Ce que le notebook documente (sans historique des propositions de l'IA) :** les cotes de clôture disponibles pour quatre
bookmakers sont reliées aux matchs, converties en probabilités après retrait de la
marge, puis comparées sur les observations communes. Le poids du mélange avec Pinnacle
est choisi sur 2018-2022 avant l'évaluation 2023-2025.

**Précisions visibles dans le notebook :**
- les données de cotes sont facultatives ; l'analyse est sautée si le dossier local
  n'existe pas ;
- le lien porte sur date et noms d'équipes, avec un contrôle des scores et des
  rencontres effectivement appariées ;
- le test du mélange n'est pas utilisé pour choisir son poids ;
- les résultats publiés indiquent un niveau comparable au bookmaker sur 2023-2025,
  sans prétendre que le modèle le bat significativement ; les cotes de clôture ne sont
  pas présentées comme disponibles pour simuler tous les matchs futurs d'une saison.

### Exemple 14 — Adapter la simulation et l'application au modèle v3

**Exemple de formulation rédigé après l'analyse de `analyse/SIMULATION_2026.ipynb`,
`analyse/README_SIMULATION.md` et `app/app.py` :**

> « Fais évoluer la simulation de la saison 2026-2027 pour utiliser le modèle de scores
> v3, qui combine Elo, terrain, format, forces d'attaque/défense et valeur récente des
> joueurs. Utilise uniquement les features connues avant le début de la saison ; garde
> fixes les forces et valeurs pendant la simulation si aucune mise à jour future valide
> n'est disponible, et fais évoluer l'Elo seulement selon la règle définie. Mets à jour
> les backtests avec les mêmes entrées pré-match et un entraînement strictement antérieur
> à chaque saison testée. Adapte aussi l'application pour lire les CSV et paramètres
> v3, calculer les probabilités d'un match, afficher les résultats de simulation et
> vérifier qu'ils correspondent aux prédictions exportées. Garde des messages explicites
> si une entrée manque, documente les règles simplifiées et les limites, puis teste
> reproductibilité, classement, tableau et calcul des probabilités. Ne réentraîne pas
> implicitement le modèle dans l'application. »

**Ce que les livrables documentent (sans historique des propositions de l'IA) :** la simulation de 10 000 saisons prend en compte
les forces et valeurs récentes au départ, les garde fixes pendant la saison simulée,
et fait évoluer l'Elo. `app/app.py` charge les paramètres et fichiers exportés,
reconstruit les variables v3 pour un match et expose aussi les résultats de simulation.

**Précisions visibles dans les livrables :**
- les entrées v3 de l'application sont calculées à partir des `club_id`, attaques,
  défenses et valeurs récentes ; les probabilités sont comparées aux prédictions
  exportées au démarrage ;
- fichiers ou colonnes manquants déclenchent des messages d'erreur explicites ; le
  modèle n'est pas réentraîné silencieusement par l'application ;
- la simulation conserve les simplifications décidées par le groupe et les documente
  comme telles, plutôt que de les présenter comme le règlement UEFA exact ;
- les backtests et les prévisions de saison sont distingués ; la finale ajoutée pour
  l'évaluation ne sert pas à entraîner le modèle de production ;
- les cotes des bookmakers sont écartées des matchs futurs de la simulation : elles
  ne sont pas disponibles à l'horizon de prévision de septembre.
