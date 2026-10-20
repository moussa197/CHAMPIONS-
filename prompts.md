# Prompts utilisés par le groupe

Livrable demandé par le prof. Pour chaque prompt : **ce que l'IA a proposé**, **ce qu'on a vérifié nous-mêmes**, **ce qu'on a corrigé ou refusé**.

Règle du groupe : l'IA propose, nous vérifions. Aucun chiffre de ce dépôt n'est repris d'une réponse de l'IA sans avoir été recalculé dans un notebook qui tourne de bout en bout.

| # | Auteur | Partie | Prompt |
|---|---|---|---|
| 1 | Moussa | modèle | Diagnostic avant de coder |
| 2 | Moussa | modèle | Un pipeline vérifiable |
| 3 | Moussa | modèle | L'IA critique son propre résultat |
| 4 | Loan | forme et audit | Le cadre du projet et le diagnostic avant de coder |
| 5 | Loan | forme et audit | Construire la forme récente et auditer le modèle |
| 6 | Loan | forme et audit | L'IA critique son propre audit |
| … | Maxime, Mouhamedou, Ousmane | à compléter | voir la dernière section |

---

## Partie modèle (Moussa)

Le détail complet est dans `prompts_moussa.md` (branche `feature/model-baseline`). En résumé :

### Prompt 1 — Le diagnostic avant de coder
> « Tu es mon reviewer ML. Voici : question, grain, cible, colonnes, EDA, date de prédiction. 1. La cible est-elle définie ? 2. Quelles features sont disponibles à temps ? 3. Quelles fuites ou biais possibles ? 4. Quel split et quelle baseline ? 5. Quelle métrique et pourquoi ? Pose 3 questions bloquantes avant de coder. »

- **Proposé :** la cible mélange 90 et 120 minutes (prolongations) ; petite fuite possible de l'Elo « maison recalé » ; le test 2023-2025 a déjà servi à choisir la variable « nouveau format ».
- **Vérifié :** prolongations = 21 matchs sur 1 871 (1,1 %), effet négligeable ; Elo recalé = 81 lignes club-match sur 3 742 (2,2 %), chiffre retrouvé de façon indépendante dans `AUDIT_MODELE.ipynb`.
- **Décidé :** garder le test pour le modèle choisi avant (Elo + terrain), et écrire la réutilisation du test comme une limite.

### Prompt 2 — Un pipeline vérifiable
> « Génère un notebook scikit-learn minimal : split adapté, pas de fuite, preprocessing en Pipeline ; baseline Dummy, modèle simple, random_state fixé ; métriques adaptées et tableau comparatif ; cellules Markdown qui expliquent chaque étape ; aucun téléchargement externe ni donnée inventée. »

- **Proposé :** `MODEL_POISSON.ipynb` v2 (`SimpleImputer` + `PoissonRegressor` dans un `Pipeline`, split chronologique, Dummy, log loss, RPS, accuracy, balanced accuracy, MAE).
- **Vérifié :** un `assert` compare le Pipeline à statsmodels. Il a d'abord **échoué** (écart de 0,0002, solveur trop peu précis) : corrigé avec `solver='newton-cholesky'`, `tol=1e-10`.
- **Corrigé :** les phrases causales (« le nouveau format marque plus de buts ») sont devenues des observations.

### Prompt 3 — L'IA critique son propre résultat
> « Agis comme un auditeur hostile. Cherche : data leakage et cible mal définie, train/test contaminés, métrique trompeuse, classes déséquilibrées, faible taille d'échantillon, features non disponibles en production. Propose 3 tests concrets pour réfuter le modèle. »

- **Proposé :** cible mélangée, retrait de l'Elo recalé, sous-groupes, calibration des favoris.
- **Vérifié :** les quatre tests ont été lancés. Les trois premiers ont été **refaits de façon indépendante** dans `AUDIT_MODELE.ipynb` (Loan), avec les mêmes chiffres : cible mélangée 1,026, Elo recalé sans effet (0,905), gain de +0,14 en groupes et de +0,03 en élimination directe.
- **Contesté par l'audit :** l'explication « les matchs à Elo estimé sont mieux prédits parce qu'ils sont déséquilibrés » ne tient qu'en partie. À écart d'Elo égal, ils restent bien mieux prédits (voir `AUDIT_MODELE.ipynb`, section 5). Le point reste ouvert.

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

## Prompts des autres membres (à compléter)

Pour chacun : le prompt **exact**, ce que l'IA a proposé, ce qu'on a vérifié, ce qu'on a corrigé ou refusé. **Ne rien reconstituer de mémoire** : coller le vrai prompt.

### Maxime — participants et calendrier 2026-2027
*À compléter par Maxime.*

### Mouhamedou — joueurs et effectif
*À compléter par Mouhamedou.*

### Ousmane — fatigue et calendrier
*À compléter par Ousmane.*
